//! Reader for `.detwgt` v2 — the deterministic weight artifact.
//!
//! Layout per `docs/plans/DETWGT_V2_FORMAT.md` in `raster-inference`:
//!
//! ```text
//!   header:  magic "DNWGTV0\0" | detwgt_version u32 | spec_version u32 | tensor_count u64
//!   tensor:  name_len u32 | name | rank u32 | dims u64×rank | element_count u64
//!            | element_width u32 | payload_len u64 | max_row_mass u64
//!            | zero padding to a 64-byte file offset | payload
//! ```
//!
//! The values are the reason this chain can consume the artifact at all: the
//! canonical value of every weight is its **sign-extended Q16.16 bit pattern
//! in i32**, which is exactly the representation the stages' kernels use. A
//! tensor stored as i16 is widened on read; the format guarantees that changes
//! no value.
//!
//! Loading is fail-closed, as the spec requires: any deviation is an error, not
//! a best-effort read.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::Path;

const MAGIC: &[u8; 8] = b"DNWGTV0\0";
const DETWGT_VERSION: u32 = 2;
const DET_NUM_SPEC_VERSION: u32 = 1;
const PAYLOAD_ALIGNMENT: usize = 64;

/// One tensor: its shape and its canonical Q16.16 values, row-major.
pub struct Tensor {
    pub dims: Vec<u64>,
    pub values: Vec<i32>,
}

impl Tensor {
    /// Row `idx` of a rank-2 tensor.
    pub fn row(&self, idx: usize) -> Result<&[i32], Box<dyn Error>> {
        let cols = *self.dims.last().ok_or("tensor has no dimensions")? as usize;
        let start = idx * cols;
        self.values
            .get(start..start + cols)
            .ok_or_else(|| format!("row {idx} out of range").into())
    }

    /// Rows `start..end` of a rank-2 tensor, flattened.
    pub fn rows(&self, start: usize, end: usize) -> Result<Vec<i32>, Box<dyn Error>> {
        let cols = *self.dims.last().ok_or("tensor has no dimensions")? as usize;
        self.values
            .get(start * cols..end * cols)
            .map(<[i32]>::to_vec)
            .ok_or_else(|| format!("rows {start}..{end} out of range").into())
    }

    /// Columns `start..end` of every row, flattened — how a per-layer slice of
    /// a packed `[vocab, layers · width]` tensor is extracted.
    pub fn column_slice(&self, start: usize, end: usize) -> Result<Vec<i32>, Box<dyn Error>> {
        let cols = *self.dims.last().ok_or("tensor has no dimensions")? as usize;
        if end > cols {
            return Err(format!("columns {start}..{end} out of range for width {cols}").into());
        }
        let mut out = Vec::with_capacity(self.values.len() / cols * (end - start));
        for row in self.values.chunks_exact(cols) {
            out.extend_from_slice(&row[start..end]);
        }
        Ok(out)
    }
}

/// Every tensor in the artifact, by name.
pub struct Artifact {
    tensors: BTreeMap<String, Tensor>,
}

impl Artifact {
    pub fn get(&self, name: &str) -> Result<&Tensor, Box<dyn Error>> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("model artifact has no tensor '{name}'").into())
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }
}

pub fn load(path: &Path) -> Result<Artifact, Box<dyn Error>> {
    let bytes = std::fs::read(path)?;
    let mut cursor = Cursor::new(&bytes);

    if cursor.take(8)? != MAGIC {
        return Err(format!("{} is not a detwgt artifact", path.display()).into());
    }
    let version = cursor.u32()?;
    if version != DETWGT_VERSION {
        return Err(format!(
            "{} is detwgt v{version}; this importer reads v{DETWGT_VERSION} (re-run gemma-det-num-wgt-converter)",
            path.display()
        )
        .into());
    }
    let spec = cursor.u32()?;
    if spec != DET_NUM_SPEC_VERSION {
        return Err(format!("unsupported det_num spec version {spec}").into());
    }

    let tensor_count = cursor.u64()?;
    let mut tensors = BTreeMap::new();
    for _ in 0..tensor_count {
        let name_len = cursor.u32()? as usize;
        let name = String::from_utf8(cursor.take(name_len)?.to_vec())?;
        let rank = cursor.u32()? as usize;
        let dims: Vec<u64> = (0..rank).map(|_| cursor.u64()).collect::<Result<_, _>>()?;
        let element_count = cursor.u64()? as usize;
        if element_count != dims.iter().product::<u64>() as usize {
            return Err(format!("tensor '{name}' element count does not match its shape").into());
        }
        let element_width = cursor.u32()?;
        let payload_len = cursor.u64()? as usize;
        let _max_row_mass = cursor.u64()?;
        if payload_len != element_count * (element_width as usize / 8) {
            return Err(format!("tensor '{name}' payload length does not match its width").into());
        }
        cursor.align_to(PAYLOAD_ALIGNMENT, &name)?;

        // Widening i16 storage to i32 is sign-extension, which the format
        // guarantees preserves every canonical value.
        let payload = cursor.take(payload_len)?;
        let values: Vec<i32> = match element_width {
            16 => payload
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as i32)
                .collect(),
            32 => payload
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            other => return Err(format!("tensor '{name}' has element width {other}").into()),
        };

        if tensors
            .insert(name.clone(), Tensor { dims, values })
            .is_some()
        {
            return Err(format!("duplicate tensor '{name}' in artifact").into());
        }
    }

    if !cursor.at_end() {
        return Err("trailing bytes after the last tensor payload".into());
    }
    Ok(Artifact { tensors })
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], Box<dyn Error>> {
        let slice = self
            .bytes
            .get(self.offset..self.offset + len)
            .ok_or("artifact ended mid-record")?;
        self.offset += len;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32, Box<dyn Error>> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, Box<dyn Error>> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// Skips to the next payload boundary, verifying the padding is zero — the
    /// spec makes non-zero padding a rejection, not a curiosity.
    fn align_to(&mut self, alignment: usize, name: &str) -> Result<(), Box<dyn Error>> {
        let padding = (alignment - (self.offset % alignment)) % alignment;
        if self.take(padding)?.iter().any(|byte| *byte != 0) {
            return Err(format!("tensor '{name}' has non-zero padding").into());
        }
        Ok(())
    }

    fn at_end(&self) -> bool {
        self.offset == self.bytes.len()
    }
}
