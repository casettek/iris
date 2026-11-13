use anyhow::{anyhow, bail, Result};
use det_num::ops::{mac_bits, requantize};
use det_num::Acc;
use prefill_range::input::PAGE_SIZE;
use raster::{Bytes, BytesPage};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

const DET_GEMV_MIN_PAR_ROWS: usize = 128;
const DET_GEMV_MIN_ROWS_PER_CHUNK: usize = 64;

fn parallelism_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("RASTER_PARALLELISM")
            .map(|value| !value.eq_ignore_ascii_case("off"))
            .unwrap_or(true)
    })
}

pub(crate) fn use_parallel() -> bool {
    parallelism_enabled() && rayon::current_num_threads() > 1
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Slab {
    data: Vec<i32>,
    rows: usize,
    cols: usize,
}

impl Slab {
    pub fn zeroed(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0; rows * cols],
            rows,
            cols,
        }
    }

    pub fn from_rows(rows: Vec<Vec<i32>>) -> Result<Self> {
        let cols = rows.first().map(Vec::len).unwrap_or(0);
        let mut data = Vec::with_capacity(rows.len() * cols);
        for (row_idx, row) in rows.into_iter().enumerate() {
            if row.len() != cols {
                bail!(
                    "activation row {row_idx} has width {}, expected {cols}",
                    row.len()
                );
            }
            data.extend(row);
        }
        Ok(Self {
            rows: if cols == 0 { 0 } else { data.len() / cols },
            cols,
            data,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn row(&self, row_idx: usize) -> &[i32] {
        &self.data[row_idx * self.cols..(row_idx + 1) * self.cols]
    }

    pub fn row_mut(&mut self, row_idx: usize) -> &mut [i32] {
        &mut self.data[row_idx * self.cols..(row_idx + 1) * self.cols]
    }

    pub fn iter_rows(&self) -> std::slice::Chunks<'_, i32> {
        self.data.chunks(self.cols.max(1))
    }

    pub fn as_flat_mut(&mut self) -> &mut [i32] {
        &mut self.data
    }

    /// Every value, row after row.
    pub fn as_flat(&self) -> &[i32] {
        &self.data
    }
}

/// Names the bytes of one `Q8_0` matrix for as long as they exist. Keys are
/// never reused, so weights cached under a key cannot be mistaken for another
/// matrix's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Q8Key(u64, u64);

impl Q8Key {
    /// A key no other matrix in this process has.
    pub fn unique() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(0, NEXT.fetch_add(1, Ordering::Relaxed))
    }

    /// The key of a tensor at `offset` in a mapped model file. `mapping` is
    /// non-zero and unique to that mapping.
    pub fn mapped(mapping: u64, offset: u64) -> Option<Self> {
        (mapping != 0).then_some(Self(mapping, offset))
    }
}

/// A matrix's committed `Q8_0` rows: `rows` rows of 34-byte blocks.
pub struct Q8Source<'a> {
    pub bytes: &'a [u8],
    pub key: Q8Key,
}

pub trait MatrixSource: Sync {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn row_values_into(&self, row_idx: usize, out: &mut Vec<i32>) -> Result<()>;

    /// The matrix as committed `Q8_0` rows, when it is one. A product over it
    /// may then run on a GPU; `row_dot` stays the definition of every value.
    fn q8_source(&self) -> Option<Q8Source<'_>> {
        None
    }

    fn row_dot(&self, row_idx: usize, input: &[i32]) -> Result<i32> {
        if input.len() != self.cols() {
            bail!(
                "deterministic linear input width mismatch: {} vs {}",
                input.len(),
                self.cols()
            );
        }
        if row_idx >= self.rows() {
            bail!(
                "deterministic linear row index {row_idx} out of range for {} rows",
                self.rows()
            );
        }
        let mut row = Vec::with_capacity(self.cols());
        self.row_values_into(row_idx, &mut row)?;
        Ok(dot_bits(input, &row))
    }

    fn matvec(&self, input: &[i32]) -> Result<Vec<i32>> {
        if input.len() != self.cols() {
            bail!(
                "deterministic linear input width mismatch: {} vs {}",
                input.len(),
                self.cols()
            );
        }
        let mut output = vec![0; self.rows()];
        self.matvec_into(input, &mut output)?;
        Ok(output)
    }

    fn matvec_into(&self, input: &[i32], output: &mut [i32]) -> Result<()> {
        if input.len() != self.cols() {
            bail!(
                "deterministic linear input width mismatch: {} vs {}",
                input.len(),
                self.cols()
            );
        }
        if output.len() != self.rows() {
            bail!(
                "deterministic linear output width mismatch: {} vs {}",
                output.len(),
                self.rows()
            );
        }
        if !use_parallel() || self.rows() < DET_GEMV_MIN_PAR_ROWS {
            rows_into(self, input, 0, output)?;
            return Ok(());
        }
        let chunk_rows =
            DET_GEMV_MIN_ROWS_PER_CHUNK.max(self.rows().div_ceil(rayon::current_num_threads()));
        output
            .par_chunks_mut(chunk_rows)
            .enumerate()
            .try_for_each(|(chunk_idx, out)| rows_into(self, input, chunk_idx * chunk_rows, out))?;
        Ok(())
    }
}

fn rows_into<M: MatrixSource + ?Sized>(
    matrix: &M,
    input: &[i32],
    row_offset: usize,
    output: &mut [i32],
) -> Result<()> {
    for (local_idx, out) in output.iter_mut().enumerate() {
        *out = matrix.row_dot(row_offset + local_idx, input)?;
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct Matrix {
    values: Vec<i32>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    pub(crate) fn allocated_bytes(&self) -> usize {
        self.values.capacity() * std::mem::size_of::<i32>()
    }
    pub fn from_region<const P: u64>(
        name: &str,
        region: &Bytes<P>,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        let values = unpack_region_i32s(name, region)?;
        let expected = rows
            .checked_mul(cols)
            .ok_or_else(|| anyhow!("{name} matrix shape overflows usize"))?;
        if values.len() != expected {
            bail!(
                "{name} has {} values, expected {expected} ({rows} x {cols})",
                values.len()
            );
        }
        Ok(Self { values, rows, cols })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn row(&self, row_idx: usize) -> &[i32] {
        &self.values[row_idx * self.cols..(row_idx + 1) * self.cols]
    }

    pub(crate) fn matvec(&self, input: &[i32]) -> Result<Vec<i32>> {
        if input.len() != self.cols {
            bail!(
                "deterministic linear input width mismatch: {} vs {}",
                input.len(),
                self.cols
            );
        }
        let mut output = vec![0; self.rows];
        self.matvec_into(input, &mut output)?;
        Ok(output)
    }

    fn matvec_into(&self, input: &[i32], output: &mut [i32]) -> Result<()> {
        if input.len() != self.cols {
            bail!(
                "deterministic linear input width mismatch: {} vs {}",
                input.len(),
                self.cols
            );
        }
        if output.len() != self.rows {
            bail!(
                "deterministic linear output width mismatch: {} vs {}",
                output.len(),
                self.rows
            );
        }
        if !use_parallel() || self.rows < DET_GEMV_MIN_PAR_ROWS {
            self.rows_into(input, 0, output);
            return Ok(());
        }
        let chunk_rows =
            DET_GEMV_MIN_ROWS_PER_CHUNK.max(self.rows.div_ceil(rayon::current_num_threads()));
        output
            .par_chunks_mut(chunk_rows)
            .enumerate()
            .for_each(|(chunk_idx, out)| self.rows_into(input, chunk_idx * chunk_rows, out));
        Ok(())
    }

    fn rows_into(&self, input: &[i32], row_offset: usize, output: &mut [i32]) {
        for (local_idx, out) in output.iter_mut().enumerate() {
            let row = self.row(row_offset + local_idx);
            *out = dot_bits(input, row);
        }
    }
}

impl MatrixSource for Matrix {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn row_values_into(&self, row_idx: usize, out: &mut Vec<i32>) -> Result<()> {
        out.clear();
        out.extend_from_slice(self.row(row_idx));
        Ok(())
    }

    fn row_dot(&self, row_idx: usize, input: &[i32]) -> Result<i32> {
        if input.len() != self.cols {
            bail!(
                "deterministic linear input width mismatch: {} vs {}",
                input.len(),
                self.cols
            );
        }
        if row_idx >= self.rows {
            bail!(
                "deterministic linear row index {row_idx} out of range for {} rows",
                self.rows
            );
        }
        Ok(dot_bits(input, self.row(row_idx)))
    }
}

/// A `Q8_0` matrix: rows of 34-byte blocks, as committed. Products go
/// through `det_num::q8::dot_row`; there is deliberately no Q16.16 view,
/// because a dequantized product rounds differently.
#[derive(Clone, Debug)]
pub struct Q8Matrix {
    bytes: Vec<u8>,
    rows: usize,
    cols: usize,
    row_bytes: usize,
    /// Names these bytes. A clone shares it: its bytes are the same.
    key: Q8Key,
}

impl Q8Matrix {
    pub(crate) fn allocated_bytes(&self) -> usize {
        self.bytes.capacity()
    }

    pub fn from_region<const P: u64>(
        name: &str,
        region: &Bytes<P>,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        Self::from_bytes(name, region_bytes(name, region, P)?, rows, cols)
    }

    pub fn from_bytes(name: &str, bytes: Vec<u8>, rows: usize, cols: usize) -> Result<Self> {
        let row_bytes = det_num::q8::row_bytes(cols)
            .ok_or_else(|| anyhow!("{name} rows of {cols} values are not whole Q8_0 blocks"))?;
        let expected = rows
            .checked_mul(row_bytes)
            .ok_or_else(|| anyhow!("{name} matrix shape overflows usize"))?;
        if bytes.len() != expected {
            bail!(
                "{name} has {} bytes, expected {expected} ({rows} x {cols} Q8_0)",
                bytes.len()
            );
        }
        Ok(Self {
            bytes,
            rows,
            cols,
            row_bytes,
            key: Q8Key::unique(),
        })
    }

    pub fn row(&self, row_idx: usize) -> &[u8] {
        &self.bytes[row_idx * self.row_bytes..(row_idx + 1) * self.row_bytes]
    }

    /// One row's Q16.16 values, for lookups (embeddings), not products.
    pub fn row_dequantized(&self, row_idx: usize) -> Result<Vec<i32>> {
        if row_idx >= self.rows {
            bail!("Q8_0 row {row_idx} out of range for {} rows", self.rows);
        }
        let mut out = vec![0; self.cols];
        det_num::q8::dequantize_row(self.row(row_idx), &mut out).map_err(anyhow::Error::msg)?;
        Ok(out)
    }
}

impl MatrixSource for Q8Matrix {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn row_values_into(&self, _row_idx: usize, _out: &mut Vec<i32>) -> Result<()> {
        bail!("Q8_0 rows have no Q16.16 values; products use row_dot")
    }

    fn q8_source(&self) -> Option<Q8Source<'_>> {
        Some(Q8Source {
            bytes: &self.bytes,
            key: self.key,
        })
    }

    fn row_dot(&self, row_idx: usize, input: &[i32]) -> Result<i32> {
        if row_idx >= self.rows {
            bail!(
                "deterministic linear row index {row_idx} out of range for {} rows",
                self.rows
            );
        }
        det_num::q8::dot_row(input, self.row(row_idx)).map_err(anyhow::Error::msg)
    }
}

pub fn matvec_from_source(matrix: &impl MatrixSource, input: &[i32]) -> Result<Vec<i32>> {
    matrix.matvec(input)
}

/// Activation values one block of tokens may hold while a matrix is swept:
/// about 256 KiB, so the block stays in cache beside a thread's share of rows.
const SLAB_BLOCK_VALUES: usize = 65_536;

/// A matrix applied to every row of `input`: output row `t` is the matrix
/// times input row `t`.
///
/// Each output value is `matrix.row_dot(row, input.row(t))`, exactly as a
/// per-token product computes it, so the result does not depend on the order
/// used here. The matrix is swept once per block of tokens rather than once
/// per token, and its rows are shared out among the threads. This is the one
/// place the layer kernel multiplies by a weight matrix. With the
/// `gpu-kernels` feature, a large enough product over a `Q8_0` matrix runs on
/// the GPU instead, with the same values.
pub fn linear_slab_from_source(input: &Slab, matrix: &impl MatrixSource) -> Result<Slab> {
    if input.cols() != matrix.cols() {
        bail!(
            "deterministic linear input row 0 has width {}, expected {}",
            input.cols(),
            matrix.cols()
        );
    }
    #[cfg(feature = "gpu-kernels")]
    if let Some(output) = crate::gpu::slab_product(input, matrix)? {
        return Ok(output);
    }
    let out_cols = matrix.rows();
    let mut output = Slab::zeroed(input.rows(), out_cols);
    if input.rows() == 0 || out_cols == 0 {
        return Ok(output);
    }
    let block_tokens = (SLAB_BLOCK_VALUES / input.cols().max(1)).clamp(1, input.rows());
    let block_len = block_tokens * out_cols;
    let parallel = use_parallel();
    let share_rows = parallel && out_cols >= DET_GEMV_MIN_PAR_ROWS;
    let rows_per_task = if share_rows {
        DET_GEMV_MIN_ROWS_PER_CHUNK.max(out_cols.div_ceil(rayon::current_num_threads()))
    } else {
        out_cols
    };

    // Each block's products, matrix row by matrix row: within a block,
    // `by_row[row * tokens + token]`. A task is a run of matrix rows against
    // one block of tokens, so its weights and activations both stay in cache.
    let mut by_row = vec![0_i32; input.rows() * out_cols];
    let sweep_block = |(block_idx, block): (usize, &mut [i32])| -> Result<()> {
        let first_token = block_idx * block_tokens;
        let tokens = block.len() / out_cols;
        let sweep_rows = |(task_idx, task): (usize, &mut [i32])| -> Result<()> {
            let first_row = task_idx * rows_per_task;
            for (local_row, products) in task.chunks_mut(tokens).enumerate() {
                for (token, product) in products.iter_mut().enumerate() {
                    *product =
                        matrix.row_dot(first_row + local_row, input.row(first_token + token))?;
                }
            }
            Ok(())
        };
        if share_rows {
            block
                .par_chunks_mut(rows_per_task * tokens)
                .enumerate()
                .try_for_each(sweep_rows)
        } else {
            sweep_rows((0, block))
        }
    };
    let by_token = |(out_block, block): (&mut [i32], &[i32])| {
        let tokens = block.len() / out_cols;
        for (token, out_row) in out_block.chunks_mut(out_cols).enumerate() {
            for (row, value) in out_row.iter_mut().enumerate() {
                *value = block[row * tokens + token];
            }
        }
    };
    if parallel {
        by_row
            .par_chunks_mut(block_len)
            .enumerate()
            .try_for_each(sweep_block)?;
        output
            .as_flat_mut()
            .par_chunks_mut(block_len)
            .zip(by_row.par_chunks(block_len))
            .for_each(by_token);
    } else {
        by_row
            .chunks_mut(block_len)
            .enumerate()
            .try_for_each(sweep_block)?;
        output
            .as_flat_mut()
            .chunks_mut(block_len)
            .zip(by_row.chunks(block_len))
            .for_each(by_token);
    }
    Ok(output)
}

pub fn dot_bits(a: &[i32], b: &[i32]) -> i32 {
    let acc = a
        .iter()
        .zip(b)
        .fold(0_i64, |acc, (left, right)| mac_bits(acc, *left, *right));
    requantize(Acc::from_bits(acc)).to_bits()
}

pub fn mac_weighted_value(acc: &mut i64, weight: i32, value: i32) {
    *acc = mac_bits(*acc, weight, value);
}

pub fn requantize_acc(acc: i64) -> i32 {
    requantize(Acc::from_bits(acc)).to_bits()
}

pub(crate) fn unpack_page_i32s(page: &BytesPage) -> Result<Vec<i32>> {
    let bytes = page.as_slice();
    if bytes.len() % 4 != 0 {
        bail!("page length {} is not i32-aligned", bytes.len());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("chunks_exact(4)")))
        .collect())
}

pub(crate) fn unpack_region_i32s<const P: u64>(name: &str, region: &Bytes<P>) -> Result<Vec<i32>> {
    let bytes = region_bytes(name, region, PAGE_SIZE)?;
    if bytes.len() % 4 != 0 {
        bail!("{name} byte length {} is not i32-aligned", bytes.len());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("chunks_exact(4)")))
        .collect())
}

/// A region's pages, checked for order and size, joined into one buffer.
pub(crate) fn region_bytes<const P: u64>(
    name: &str,
    region: &Bytes<P>,
    page_size: u64,
) -> Result<Vec<u8>> {
    if region.page_size() != P || P != page_size {
        bail!(
            "{name} page size mismatch: declared {P}, artifact {}",
            region.page_size()
        );
    }
    let mut bytes = Vec::with_capacity(region.byte_len() as usize);
    let mut expected_offset = 0_u64;
    for (expected_index, page) in region.pages().iter().enumerate() {
        if page.index() != expected_index as u64 {
            bail!(
                "{name} page index mismatch: got {}, expected {expected_index}",
                page.index()
            );
        }
        if page.offset() != expected_offset {
            bail!(
                "{name} page offset mismatch: got {}, expected {expected_offset}",
                page.offset()
            );
        }
        bytes.extend_from_slice(page.as_slice());
        expected_offset = expected_offset.saturating_add(page.len() as u64);
    }
    if expected_offset != region.byte_len() {
        bail!(
            "{name} byte length mismatch: pages total {expected_offset}, region {}",
            region.byte_len()
        );
    }
    Ok(bytes)
}

pub(crate) fn pack_i32_page(values: &[i32]) -> BytesPage {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    BytesPage::__from_parts(0, 0, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_random(seed: &mut u64) -> i32 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*seed >> 32) as i32
    }

    /// A `Q8_0` matrix with varied finite scales of either sign and varied weights.
    fn q8_matrix(rows: usize, cols: usize) -> Q8Matrix {
        let mut bytes = Vec::new();
        for row in 0..rows {
            for block in 0..cols / det_num::q8::BLOCK {
                let sign = ((row * 7 + block) % 2) as u16;
                let exponent = 6 + ((row + block) % 4) as u16;
                let mantissa = ((row * 131 + block * 17) % 1024) as u16;
                bytes.extend_from_slice(&(sign << 15 | exponent << 10 | mantissa).to_le_bytes());
                bytes.extend(
                    (0..det_num::q8::BLOCK)
                        .map(|col| ((row * 37 + block * 11 + col * 5) % 256) as u8),
                );
            }
        }
        Q8Matrix::from_bytes("test", bytes, rows, cols).unwrap()
    }

    fn slab(rows: usize, cols: usize, seed: &mut u64) -> Slab {
        let mut slab = Slab::zeroed(rows, cols);
        for value in slab.as_flat_mut() {
            *value = pseudo_random(seed);
        }
        slab
    }

    /// Shapes chosen to cross every boundary of the slab product: several
    /// token blocks with a short last one, rows shared among threads with a
    /// short last share, a matrix too small to share, and a single token.
    const SHAPES: [(usize, usize, usize); 5] = [
        (200, 64, 1100),
        (130, 96, 700),
        (64, 64, 3),
        (300, 32, 1),
        (1, 32, 5),
    ];

    #[test]
    fn slab_product_equals_one_q8_dot_per_value() {
        let mut seed = 7;
        for (rows, cols, tokens) in SHAPES {
            let matrix = q8_matrix(rows, cols);
            let input = slab(tokens, cols, &mut seed);
            let output = linear_slab_from_source(&input, &matrix).unwrap();
            assert_eq!((output.rows(), output.cols()), (tokens, rows));
            for token in 0..tokens {
                for row in 0..rows {
                    assert_eq!(
                        output.row(token)[row],
                        det_num::q8::dot_row(input.row(token), matrix.row(row)).unwrap(),
                        "{rows}x{cols}, token {token}, row {row}"
                    );
                }
            }
        }
    }

    #[test]
    fn slab_product_equals_one_fixed_point_dot_per_value() {
        let mut seed = 11;
        for (rows, cols, tokens) in SHAPES {
            let weights = slab(rows, cols, &mut seed);
            let matrix = Matrix {
                values: weights.data.clone(),
                rows,
                cols,
            };
            let input = slab(tokens, cols, &mut seed);
            let output = linear_slab_from_source(&input, &matrix).unwrap();
            for token in 0..tokens {
                for row in 0..rows {
                    assert_eq!(
                        output.row(token)[row],
                        dot_bits(input.row(token), weights.row(row)),
                        "{rows}x{cols}, token {token}, row {row}"
                    );
                }
            }
        }
    }

    #[test]
    fn slab_product_of_no_tokens_is_empty() {
        let matrix = q8_matrix(4, 32);
        let output = linear_slab_from_source(&Slab::zeroed(0, 32), &matrix).unwrap();
        assert_eq!(output.rows(), 0);
        assert!(linear_slab_from_source(&Slab::zeroed(2, 64), &matrix).is_err());
    }
}
