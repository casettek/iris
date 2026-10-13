use anyhow::{anyhow, bail, Result};
use det_num::ops::{mac_bits, requantize};
use det_num::Acc;
use prefill_range::input::PAGE_SIZE;
use raster::{Bytes, BytesPage};
use rayon::prelude::*;

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
}

pub trait MatrixSource: Sync {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn row_values_into(&self, row_idx: usize, out: &mut Vec<i32>) -> Result<()>;

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

pub fn matvec_from_source(matrix: &impl MatrixSource, input: &[i32]) -> Result<Vec<i32>> {
    matrix.matvec(input)
}

pub fn linear_slab_from_source(input: &Slab, matrix: &impl MatrixSource) -> Result<Slab> {
    if input.cols() != matrix.cols() {
        bail!(
            "deterministic linear input row 0 has width {}, expected {}",
            input.cols(),
            matrix.cols()
        );
    }
    let mut output = Slab::zeroed(input.rows(), matrix.rows());
    if use_parallel() {
        output
            .as_flat_mut()
            .par_chunks_mut(matrix.rows().max(1))
            .enumerate()
            .try_for_each(|(row_idx, out_row)| matrix.matvec_into(input.row(row_idx), out_row))?;
    } else {
        output
            .as_flat_mut()
            .chunks_mut(matrix.rows().max(1))
            .zip(input.iter_rows())
            .try_for_each(|(out_row, in_row)| matrix.matvec_into(in_row, out_row))?;
    }
    Ok(output)
}

pub(crate) fn linear_slab(input: &Slab, matrix: &Matrix) -> Result<Slab> {
    linear_slab_from_source(input, matrix)
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
    if region.page_size() != P || P != PAGE_SIZE {
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
    if bytes.len() % 4 != 0 {
        bail!("{name} byte length {} is not i32-aligned", bytes.len());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("chunks_exact(4)")))
        .collect())
}

pub(crate) fn pack_i32_page(values: &[i32]) -> BytesPage {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    BytesPage::__from_parts(0, 0, bytes)
}
