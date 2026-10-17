//! Convert mapped weights directly to Raster pages. Only the current external
//! is owned in memory; signed i16 values are widened exactly as by the eager
//! reader, using at most one page of temporary decoded values.

use std::error::Error;

use detwgt::{DetwgtSlice, MmapDetwgt};
use raster::{Bytes, BytesPage, List};

use super::externals::PAGE_SIZE;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Default)]
struct Pages {
    pages: Vec<BytesPage>,
    pending: Vec<u8>,
    byte_len: u64,
}

impl Pages {
    fn append(&mut self, slice: DetwgtSlice<'_>) -> Result<()> {
        let mut offset = 0;
        while offset < slice.len() {
            let count = (slice.len() - offset).min((PAGE_SIZE as usize - self.pending.len()) / 4);
            self.pending.reserve(count * 4);
            for value in slice.subslice(offset, count)?.values() {
                self.pending.extend_from_slice(&value.to_le_bytes());
            }
            self.byte_len += (count * 4) as u64;
            offset += count;
            if self.pending.len() == PAGE_SIZE as usize {
                self.flush();
            }
        }
        Ok(())
    }

    fn flush(&mut self) {
        if !self.pending.is_empty() {
            let index = self.pages.len() as u64;
            self.pages.push(BytesPage::__from_parts(
                index,
                index * PAGE_SIZE,
                std::mem::take(&mut self.pending),
            ));
        }
    }

    fn finish(mut self) -> Bytes<PAGE_SIZE> {
        self.flush();
        // Same layout as Bytes::paged, without first building and then copying
        // a contiguous buffer for the entire tensor.
        Bytes {
            byte_len: self.byte_len,
            page_size: PAGE_SIZE,
            pages: List::from(self.pages),
        }
    }
}

pub(super) fn tensor(weights: &MmapDetwgt, name: &str) -> Result<Bytes<PAGE_SIZE>> {
    let mut pages = Pages::default();
    pages.append(weights.slice(name)?)?;
    Ok(pages.finish())
}

pub(super) fn rows(
    weights: &MmapDetwgt,
    name: &str,
    start: usize,
    end: usize,
) -> Result<Bytes<PAGE_SIZE>> {
    let cols = *weights
        .tensor(name)?
        .dims
        .last()
        .ok_or("tensor has no dimensions")? as usize;
    let offset = start.checked_mul(cols).ok_or("row offset overflow")?;
    let count = end
        .checked_sub(start)
        .and_then(|rows| rows.checked_mul(cols))
        .ok_or("invalid row range")?;
    let mut pages = Pages::default();
    pages.append(weights.slice(name)?.subslice(offset, count)?)?;
    Ok(pages.finish())
}

pub(super) fn columns(
    weights: &MmapDetwgt,
    name: &str,
    start: usize,
    end: usize,
) -> Result<Bytes<PAGE_SIZE>> {
    let cols = *weights
        .tensor(name)?
        .dims
        .last()
        .ok_or("tensor has no dimensions")? as usize;
    if cols == 0 || start > end || end > cols {
        return Err(format!(
            "columns {start}..{end} out of range for tensor '{name}' width {cols}"
        )
        .into());
    }
    let slice = weights.slice(name)?;
    let mut pages = Pages::default();
    for row in 0..slice.len() / cols {
        pages.append(slice.subslice(row * cols + start, end - start)?)?;
    }
    Ok(pages.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapped_tensor_and_matrix_slices_match_eager_reader() {
        let dir = std::env::temp_dir().join(format!(
            "mapped-import-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        std::fs::create_dir_all(&dir).unwrap();
        for width in [16_u32, 32] {
            let path = dir.join(format!("weights-{width}.detwgt"));
            let name = "weights";
            let values: Vec<i32> = (0..7 * 8001)
                .map(|i| {
                    if width == 16 {
                        [i16::MIN as i32, -1, 0, i16::MAX as i32][i % 4]
                    } else {
                        [i32::MIN, -1, 0, i32::MAX][i % 4]
                    }
                })
                .collect();
            let mut file = b"DNWGTV0\0".to_vec();
            file.extend_from_slice(&2_u32.to_le_bytes());
            file.extend_from_slice(&1_u32.to_le_bytes());
            file.extend_from_slice(&1_u64.to_le_bytes());
            file.extend_from_slice(&(name.len() as u32).to_le_bytes());
            file.extend_from_slice(name.as_bytes());
            file.extend_from_slice(&2_u32.to_le_bytes());
            file.extend_from_slice(&7_u64.to_le_bytes());
            file.extend_from_slice(&8001_u64.to_le_bytes());
            file.extend_from_slice(&(values.len() as u64).to_le_bytes());
            file.extend_from_slice(&width.to_le_bytes());
            file.extend_from_slice(&(values.len() as u64 * (width / 8) as u64).to_le_bytes());
            file.extend_from_slice(&0_u64.to_le_bytes());
            file.resize(file.len().div_ceil(64) * 64, 0);
            for value in values {
                if width == 16 {
                    file.extend_from_slice(&(value as i16).to_le_bytes());
                } else {
                    file.extend_from_slice(&value.to_le_bytes());
                }
            }
            std::fs::write(&path, file).unwrap();
            let eager = detwgt::load(&path).unwrap();
            let expected = eager.get(name).unwrap();
            let mapped = MmapDetwgt::open(&path).unwrap();
            let cases = [
                (tensor(&mapped, name).unwrap(), expected.values.clone()),
                (
                    rows(&mapped, name, 1, 7).unwrap(),
                    expected.rows(1, 7).unwrap(),
                ),
                (
                    columns(&mapped, name, 1, 8000).unwrap(),
                    expected.column_slice(1, 8000).unwrap(),
                ),
                (columns(&mapped, name, 4, 4).unwrap(), Vec::new()),
            ];
            for (actual, values) in cases {
                let expected = super::super::externals::paged_i32s(&values).unwrap();
                assert_eq!(
                    raster::encode_raster_value(&actual).unwrap(),
                    raster::encode_raster_value(&expected).unwrap(),
                );
            }
            assert!(rows(&mapped, name, 0, 8).is_err());
            assert!(rows(&mapped, name, 3, 2).is_err());
            assert!(columns(&mapped, name, 0, 8002).is_err());
            assert!(columns(&mapped, name, 3, 2).is_err());
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    // Compare both the data and structural index/commitment, including signed
    // widening, page boundaries, and appends that split a page across rows.
    #[test]
    fn mapped_pages_match_eager_encoding() {
        for width in [16, 32] {
            for len in [0, 1, PAGE_SIZE as usize / 4, PAGE_SIZE as usize / 4 + 7] {
                let values: Vec<i32> = (0..len)
                    .map(|i| match width {
                        16 => [i16::MIN as i32, -1, 0, i16::MAX as i32][i % 4],
                        _ => [i32::MIN, -1, 0, i32::MAX][i % 4],
                    })
                    .collect();
                let raw: Vec<u8> = if width == 16 {
                    values
                        .iter()
                        .flat_map(|v| (*v as i16).to_le_bytes())
                        .collect()
                } else {
                    values.iter().flat_map(|v| v.to_le_bytes()).collect()
                };
                let slice = DetwgtSlice::new(&raw, width, len).unwrap();
                let mut pages = Pages::default();
                for offset in (0..len).step_by(13) {
                    pages
                        .append(slice.subslice(offset, (len - offset).min(13)).unwrap())
                        .unwrap();
                }
                let expected = super::super::externals::paged_i32s(&values).unwrap();
                assert_eq!(
                    raster::encode_raster_value(&pages.finish()).unwrap(),
                    raster::encode_raster_value(&expected).unwrap(),
                    "width={width}, len={len}",
                );
            }
        }
    }
}
