use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{bail, Context, Result};
use memmap2::Mmap;

const MAGIC: &[u8; 8] = b"DNWGTV0\0";
const DETWGT_VERSION: u32 = 3;
const DET_NUM_SPEC_VERSION: u32 = 1;
const PAYLOAD_ALIGNMENT: usize = 64;

/// The `element_width` of a tensor stored as `Q8_0` blocks (detwgt v3): rows
/// of 34-byte blocks, each a half-precision scale and 32 signed bytes, as
/// GGUF lays them out. Such a tensor holds no Q16.16 values; its products and
/// lookups go through `det_num::q8`, never through a dequantized copy.
pub const Q8_0_WIDTH: u32 = 8;

#[derive(Debug)]
pub struct MmapDetwgt {
    mmap: Mmap,
    tensors: BTreeMap<String, TensorEntry>,
    /// Unique to this mapping, and never zero.
    mapping: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorEntry {
    pub dims: Vec<u64>,
    pub element_count: u64,
    pub element_width: u32,
    pub payload_offset: usize,
    pub payload_len: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct DetwgtSlice<'a> {
    bytes: &'a [u8],
    element_width: u32,
    count: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct DetwgtMatrixView<'a> {
    slice: DetwgtSlice<'a>,
    rows: usize,
    cols: usize,
}

impl MmapDetwgt {
    pub fn open(path: &Path) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        // SAFETY: The mapping is read-only and the file handle is not used for mutation here.
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("failed to mmap {}", path.display()))?;
        let tensors = parse_directory(&mmap, path)?;
        static NEXT_MAPPING: AtomicU64 = AtomicU64::new(1);
        Ok(Self {
            mmap,
            tensors,
            mapping: NEXT_MAPPING.fetch_add(1, Ordering::Relaxed),
        })
    }

    /// Every tensor name in the bundle, in name order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorEntry> {
        self.tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("model artifact has no tensor '{name}'"))
    }

    /// Whether the tensor is stored as `Q8_0` blocks.
    pub fn is_q8(&self, name: &str) -> Result<bool> {
        Ok(self.tensor(name)?.element_width == Q8_0_WIDTH)
    }

    /// A `Q8_0` tensor as a matrix of block rows.
    pub fn q8_matrix(&self, name: &str) -> Result<Q8MatrixView<'_>> {
        let tensor = self.q8_tensor(name)?;
        let cols = *tensor.dims.last().unwrap_or(&0) as usize;
        let rows = tensor.element_count as usize / cols;
        let bytes = self
            .mmap
            .get(tensor.payload_offset..tensor.payload_offset + tensor.payload_len)
            .ok_or_else(|| anyhow::anyhow!("tensor payload range is outside mmap"))?;
        let mut view = Q8MatrixView::new(bytes, rows, cols)
            .with_context(|| format!("invalid Q8_0 matrix for tensor '{name}'"))?;
        view.place = Some((self.mapping, tensor.payload_offset as u64));
        Ok(view)
    }

    /// Rows `start..end` of a `Q8_0` tensor's blocks, contiguous.
    pub fn q8_rows(&self, name: &str, start: usize, end: usize) -> Result<&[u8]> {
        let matrix = self.q8_matrix(name)?;
        if start > end || end > matrix.rows() {
            bail!("rows {start}..{end} out of range for tensor '{name}'");
        }
        Ok(&matrix.bytes[start * matrix.row_bytes..end * matrix.row_bytes])
    }

    /// Columns `start..end` of every row of a `Q8_0` tensor, which must fall
    /// on block boundaries: how one layer's slice of the packed per-layer
    /// embedding table is taken without requantizing it.
    pub fn q8_column_blocks(&self, name: &str, start: usize, end: usize) -> Result<Vec<u8>> {
        let matrix = self.q8_matrix(name)?;
        let block = det_num::q8::BLOCK;
        if start > end || end > matrix.cols() || start % block != 0 || end % block != 0 {
            bail!(
                "columns {start}..{end} of tensor '{name}' (width {}) do not fall on {block}-value blocks",
                matrix.cols()
            );
        }
        let (from, to) = (
            start / block * det_num::q8::BLOCK_BYTES,
            end / block * det_num::q8::BLOCK_BYTES,
        );
        let mut out = Vec::with_capacity(matrix.rows() * (to - from));
        for row in 0..matrix.rows() {
            out.extend_from_slice(&matrix.row(row)?[from..to]);
        }
        Ok(out)
    }

    fn q8_tensor(&self, name: &str) -> Result<&TensorEntry> {
        let tensor = self.tensor(name)?;
        if tensor.element_width != Q8_0_WIDTH {
            bail!("tensor '{name}' holds Q16.16 values, not Q8_0 blocks");
        }
        Ok(tensor)
    }

    pub fn slice(&self, name: &str) -> Result<DetwgtSlice<'_>> {
        let tensor = self.tensor(name)?;
        self.slice_for(tensor, 0, tensor.element_count as usize)
    }

    pub fn matrix(&self, name: &str, rows: usize, cols: usize) -> Result<DetwgtMatrixView<'_>> {
        let slice = self.slice(name)?;
        DetwgtMatrixView::new(slice, rows, cols)
            .with_context(|| format!("invalid matrix view for tensor '{name}'"))
    }

    pub fn matrix_rows(
        &self,
        name: &str,
        start: usize,
        end: usize,
        cols: usize,
    ) -> Result<DetwgtMatrixView<'_>> {
        let tensor = self.tensor(name)?;
        let rows = end
            .checked_sub(start)
            .ok_or_else(|| anyhow::anyhow!("matrix row range {start}..{end} underflows"))?;
        let value_start = start
            .checked_mul(cols)
            .ok_or_else(|| anyhow::anyhow!("matrix row start overflows for tensor '{name}'"))?;
        let count = rows
            .checked_mul(cols)
            .ok_or_else(|| anyhow::anyhow!("matrix row count overflows for tensor '{name}'"))?;
        let slice = self.slice_for(tensor, value_start, count)?;
        DetwgtMatrixView::new(slice, rows, cols)
            .with_context(|| format!("invalid row matrix view for tensor '{name}'"))
    }

    pub fn row_slice(&self, name: &str, idx: usize) -> Result<DetwgtSlice<'_>> {
        let tensor = self.tensor(name)?;
        let cols = *tensor
            .dims
            .last()
            .ok_or_else(|| anyhow::anyhow!("tensor '{name}' has no dimensions"))?
            as usize;
        let start = idx
            .checked_mul(cols)
            .ok_or_else(|| anyhow::anyhow!("row {idx} offset overflows for tensor '{name}'"))?;
        self.slice_for(tensor, start, cols)
            .with_context(|| format!("row {idx} out of range for tensor '{name}'"))
    }

    pub fn values(&self, name: &str) -> Result<Vec<i32>> {
        let tensor = self.tensor(name)?;
        self.read_values(tensor, 0, tensor.element_count as usize)
    }

    pub fn row(&self, name: &str, idx: usize) -> Result<Vec<i32>> {
        let tensor = self.tensor(name)?;
        let cols = *tensor
            .dims
            .last()
            .ok_or_else(|| anyhow::anyhow!("tensor '{name}' has no dimensions"))?
            as usize;
        let start = idx
            .checked_mul(cols)
            .ok_or_else(|| anyhow::anyhow!("row {idx} offset overflows for tensor '{name}'"))?;
        self.read_values(tensor, start, cols)
            .with_context(|| format!("row {idx} out of range for tensor '{name}'"))
    }

    pub fn rows(&self, name: &str, start: usize, end: usize) -> Result<Vec<i32>> {
        let tensor = self.tensor(name)?;
        let cols = *tensor
            .dims
            .last()
            .ok_or_else(|| anyhow::anyhow!("tensor '{name}' has no dimensions"))?
            as usize;
        let count = end
            .checked_sub(start)
            .and_then(|rows| rows.checked_mul(cols))
            .ok_or_else(|| anyhow::anyhow!("rows {start}..{end} overflow for tensor '{name}'"))?;
        self.read_values(tensor, start * cols, count)
            .with_context(|| format!("rows {start}..{end} out of range for tensor '{name}'"))
    }

    pub fn column_slice(&self, name: &str, start: usize, end: usize) -> Result<Vec<i32>> {
        let tensor = self.tensor(name)?;
        let cols = *tensor
            .dims
            .last()
            .ok_or_else(|| anyhow::anyhow!("tensor '{name}' has no dimensions"))?
            as usize;
        if end > cols || start > end {
            bail!("columns {start}..{end} out of range for tensor '{name}' width {cols}");
        }
        let rows = tensor.element_count as usize / cols;
        let mut out = Vec::with_capacity(rows * (end - start));
        for row in 0..rows {
            let offset = row * cols + start;
            out.extend(self.read_values(tensor, offset, end - start)?);
        }
        Ok(out)
    }

    fn read_values(&self, tensor: &TensorEntry, start: usize, count: usize) -> Result<Vec<i32>> {
        Ok(self.slice_for(tensor, start, count)?.values())
    }

    fn slice_for(
        &self,
        tensor: &TensorEntry,
        start: usize,
        count: usize,
    ) -> Result<DetwgtSlice<'_>> {
        if tensor.element_width == Q8_0_WIDTH {
            bail!("tensor is stored as Q8_0 blocks; read it with the q8_* methods");
        }
        let end = start
            .checked_add(count)
            .ok_or_else(|| anyhow::anyhow!("tensor value range overflows usize"))?;
        if end > tensor.element_count as usize {
            bail!(
                "tensor range {start}..{end} exceeds {} values",
                tensor.element_count
            );
        }
        let bytes_per_value = (tensor.element_width / 8) as usize;
        let byte_start = tensor.payload_offset + start * bytes_per_value;
        let byte_end = byte_start + count * bytes_per_value;
        let payload = self
            .mmap
            .get(byte_start..byte_end)
            .ok_or_else(|| anyhow::anyhow!("tensor payload range is outside mmap"))?;
        DetwgtSlice::new(payload, tensor.element_width, count)
    }
}

impl<'a> DetwgtSlice<'a> {
    pub fn new(bytes: &'a [u8], element_width: u32, count: usize) -> Result<Self> {
        if !matches!(element_width, 16 | 32) {
            bail!("unsupported detwgt element width {element_width}");
        }
        let bytes_per_value = (element_width / 8) as usize;
        if bytes.len() != count * bytes_per_value {
            bail!(
                "detwgt slice byte length {} does not match {count} values at {element_width} bits",
                bytes.len()
            );
        }
        Ok(Self {
            bytes,
            element_width,
            count,
        })
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn value_at(&self, idx: usize) -> Result<i32> {
        if idx >= self.count {
            bail!(
                "detwgt slice index {idx} out of range for {} values",
                self.count
            );
        }
        let bytes_per_value = (self.element_width / 8) as usize;
        let start = idx * bytes_per_value;
        let bytes = &self.bytes[start..start + bytes_per_value];
        Ok(match self.element_width {
            16 => i16::from_le_bytes([bytes[0], bytes[1]]) as i32,
            32 => i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            other => bail!("unsupported detwgt element width {other}"),
        })
    }

    pub fn values(&self) -> Vec<i32> {
        match self.element_width {
            16 => self
                .bytes
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as i32)
                .collect(),
            32 => self
                .bytes
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            _ => unreachable!("constructor validates width"),
        }
    }

    pub fn values_into(&self, out: &mut Vec<i32>) {
        out.clear();
        out.reserve(self.count);
        match self.element_width {
            16 => out.extend(
                self.bytes
                    .chunks_exact(2)
                    .map(|b| i16::from_le_bytes([b[0], b[1]]) as i32),
            ),
            32 => out.extend(
                self.bytes
                    .chunks_exact(4)
                    .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            ),
            _ => unreachable!("constructor validates width"),
        }
    }

    pub fn subslice(&self, start: usize, count: usize) -> Result<Self> {
        let end = start
            .checked_add(count)
            .ok_or_else(|| anyhow::anyhow!("detwgt slice range overflows usize"))?;
        if end > self.count {
            bail!("detwgt slice range {start}..{end} exceeds {}", self.count);
        }
        let bytes_per_value = (self.element_width / 8) as usize;
        let byte_start = start * bytes_per_value;
        let byte_end = byte_start + count * bytes_per_value;
        Self::new(&self.bytes[byte_start..byte_end], self.element_width, count)
    }
}

impl<'a> DetwgtMatrixView<'a> {
    pub fn new(slice: DetwgtSlice<'a>, rows: usize, cols: usize) -> Result<Self> {
        let expected = rows
            .checked_mul(cols)
            .ok_or_else(|| anyhow::anyhow!("matrix view shape overflows usize"))?;
        if slice.len() != expected {
            bail!(
                "matrix view has {} values, expected {expected} ({rows} x {cols})",
                slice.len()
            );
        }
        Ok(Self { slice, rows, cols })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn row(&self, row_idx: usize) -> Result<DetwgtSlice<'a>> {
        if row_idx >= self.rows {
            bail!("matrix row {row_idx} out of range for {} rows", self.rows);
        }
        self.slice.subslice(row_idx * self.cols, self.cols)
    }

    pub fn row_values(&self, row_idx: usize) -> Result<Vec<i32>> {
        Ok(self.row(row_idx)?.values())
    }
}

/// A `Q8_0` matrix: `rows` rows of `cols / 32` blocks each.
#[derive(Debug, Clone, Copy)]
pub struct Q8MatrixView<'a> {
    bytes: &'a [u8],
    rows: usize,
    cols: usize,
    row_bytes: usize,
    place: Option<(u64, u64)>,
}

impl<'a> Q8MatrixView<'a> {
    pub fn new(bytes: &'a [u8], rows: usize, cols: usize) -> Result<Self> {
        let row_bytes = det_num::q8::row_bytes(cols).ok_or_else(|| {
            anyhow::anyhow!("Q8_0 rows need a positive multiple of 32 columns, got {cols}")
        })?;
        if Some(bytes.len()) != rows.checked_mul(row_bytes) {
            bail!(
                "Q8_0 matrix has {} bytes, expected {rows} rows of {row_bytes}",
                bytes.len()
            );
        }
        Ok(Self {
            bytes,
            rows,
            cols,
            row_bytes,
            place: None,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Every row's blocks, contiguous.
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// For a whole tensor of a mapped file: a number unique to the mapping
    /// (never zero) and the tensor's offset in it. Together they name these
    /// bytes for as long as the mapping lives.
    pub fn place(&self) -> Option<(u64, u64)> {
        self.place
    }

    pub fn row(&self, row_idx: usize) -> Result<&'a [u8]> {
        if row_idx >= self.rows {
            bail!("matrix row {row_idx} out of range for {} rows", self.rows);
        }
        Ok(&self.bytes[row_idx * self.row_bytes..(row_idx + 1) * self.row_bytes])
    }

    /// One row's Q16.16 values, for lookups. Products use `det_num::q8::dot_row`.
    pub fn row_dequantized(&self, row_idx: usize) -> Result<Vec<i32>> {
        let mut out = vec![0; self.cols];
        det_num::q8::dequantize_row(self.row(row_idx)?, &mut out).map_err(anyhow::Error::msg)?;
        Ok(out)
    }
}

fn parse_directory(bytes: &[u8], path: &Path) -> Result<BTreeMap<String, TensorEntry>> {
    let mut cursor = Cursor::new(bytes);
    if cursor.take(8)? != MAGIC {
        bail!("{} is not a detwgt artifact", path.display());
    }
    let version = cursor.u32()?;
    if !(2..=DETWGT_VERSION).contains(&version) {
        bail!(
            "{} is detwgt v{version}; this reader expects v2 or v{DETWGT_VERSION}",
            path.display()
        );
    }
    let spec = cursor.u32()?;
    if spec != DET_NUM_SPEC_VERSION {
        bail!("unsupported det_num spec version {spec}");
    }

    let tensor_count = cursor.u64()?;
    let mut tensors = BTreeMap::new();
    for _ in 0..tensor_count {
        let name_len = cursor.u32()? as usize;
        let name = String::from_utf8(cursor.take(name_len)?.to_vec())?;
        let rank = cursor.u32()? as usize;
        let dims = (0..rank)
            .map(|_| cursor.u64())
            .collect::<Result<Vec<_>>>()?;
        let element_count = cursor.u64()?;
        if element_count != dims.iter().product::<u64>() {
            bail!("tensor '{name}' element count does not match its shape");
        }
        let element_width = cursor.u32()?;
        let expected_len = match element_width {
            16 | 32 => element_count as usize * (element_width / 8) as usize,
            Q8_0_WIDTH if version >= 3 => {
                let cols = *dims.last().unwrap_or(&0) as usize;
                det_num::q8::row_bytes(cols)
                    .map(|row| element_count as usize / cols * row)
                    .ok_or_else(|| {
                        anyhow::anyhow!("Q8_0 tensor '{name}' rows are not whole blocks")
                    })?
            }
            _ => bail!("tensor '{name}' has element width {element_width}"),
        };
        let payload_len = cursor.u64()? as usize;
        if payload_len != expected_len {
            bail!("tensor '{name}' payload length does not match its width");
        }
        let _max_row_mass = cursor.u64()?;
        cursor.align_to(PAYLOAD_ALIGNMENT, &name)?;
        let payload_offset = cursor.offset;
        cursor.take(payload_len)?;

        let entry = TensorEntry {
            dims,
            element_count,
            element_width,
            payload_offset,
            payload_len,
        };
        if tensors.insert(name.clone(), entry).is_some() {
            bail!("duplicate tensor '{name}' in artifact");
        }
    }
    if !cursor.at_end() {
        bail!("trailing bytes after the last tensor payload");
    }
    Ok(tensors)
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| anyhow::anyhow!("artifact offset overflows usize"))?;
        let slice = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| anyhow::anyhow!("artifact ended mid-record"))?;
        self.offset = end;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn align_to(&mut self, alignment: usize, name: &str) -> Result<()> {
        let padding = (alignment - (self.offset % alignment)) % alignment;
        if self.take(padding)?.iter().any(|byte| *byte != 0) {
            bail!("tensor '{name}' has non-zero padding");
        }
        Ok(())
    }

    fn at_end(&self) -> bool {
        self.offset == self.bytes.len()
    }
}
