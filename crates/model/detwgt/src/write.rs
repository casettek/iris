//! Writer for `.detwgt` v3: Q16.16 tensors as before, plus `Q8_0` block
//! tensors (see [`crate::Q8_0_WIDTH`]).
//!
//! Records are streamed: each tensor's payload is written as it is added, so
//! a multi-gigabyte block tensor can come straight from a memory map.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};

use crate::Q8_0_WIDTH;

const MAGIC: &[u8; 8] = b"DNWGTV0\0";
const DETWGT_VERSION: u32 = 3;
const DET_NUM_SPEC_VERSION: u32 = 1;
const PAYLOAD_ALIGNMENT: u64 = 64;

pub struct DetwgtWriter {
    out: BufWriter<File>,
    offset: u64,
    remaining: u64,
}

impl DetwgtWriter {
    /// Starts a file that will hold exactly `tensor_count` tensors.
    pub fn create(path: &Path, tensor_count: u64) -> Result<Self> {
        let file =
            File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
        let mut writer = Self {
            out: BufWriter::with_capacity(1 << 22, file),
            offset: 0,
            remaining: tensor_count,
        };
        writer.bytes(MAGIC)?;
        writer.bytes(&DETWGT_VERSION.to_le_bytes())?;
        writer.bytes(&DET_NUM_SPEC_VERSION.to_le_bytes())?;
        writer.bytes(&tensor_count.to_le_bytes())?;
        Ok(writer)
    }

    /// A Q16.16 tensor, stored as `i16` when every value fits.
    pub fn q16(&mut self, name: &str, dims: &[u64], values: &[i32]) -> Result<()> {
        ensure!(
            dims.iter().product::<u64>() == values.len() as u64,
            "tensor '{name}' shape does not match its {} values",
            values.len()
        );
        let row = *dims.last().unwrap_or(&1) as usize;
        let mass = values
            .chunks(row.max(1))
            .map(|row| row.iter().map(|v| v.unsigned_abs() as u64).sum::<u64>())
            .max()
            .unwrap_or(0);
        let narrow = values.iter().all(|v| i16::try_from(*v).is_ok());
        let width = if narrow { 16 } else { 32 };
        self.header(
            name,
            dims,
            width,
            values.len() as u64 * u64::from(width / 8),
            mass,
        )?;
        for value in values {
            if narrow {
                self.bytes(&(*value as i16).to_le_bytes())?;
            } else {
                self.bytes(&value.to_le_bytes())?;
            }
        }
        Ok(())
    }

    /// A `Q8_0` tensor: its rows of blocks, exactly as GGUF stores them.
    pub fn q8_0(&mut self, name: &str, dims: &[u64], blocks: &[u8]) -> Result<()> {
        let cols = *dims.last().unwrap_or(&0) as usize;
        let row_bytes = det_num::q8::row_bytes(cols)
            .with_context(|| format!("Q8_0 tensor '{name}' rows are not whole blocks"))?;
        let rows = dims.iter().product::<u64>() as usize / cols;
        ensure!(
            blocks.len() == rows * row_bytes,
            "Q8_0 tensor '{name}' has {} bytes, expected {}",
            blocks.len(),
            rows * row_bytes
        );
        let mut values = vec![0_i32; cols];
        let mut mass = 0_u64;
        for row in blocks.chunks_exact(row_bytes) {
            det_num::q8::dequantize_row(row, &mut values)
                .map_err(|error| anyhow::anyhow!("Q8_0 tensor '{name}': {error}"))?;
            mass = mass.max(values.iter().map(|v| v.unsigned_abs() as u64).sum());
        }
        self.header(name, dims, Q8_0_WIDTH, blocks.len() as u64, mass)?;
        self.bytes(blocks)
    }

    pub fn finish(mut self) -> Result<()> {
        if self.remaining != 0 {
            bail!("detwgt file declared {} more tensors", self.remaining);
        }
        self.out.flush()?;
        Ok(())
    }

    fn header(&mut self, name: &str, dims: &[u64], width: u32, len: u64, mass: u64) -> Result<()> {
        ensure!(
            self.remaining > 0,
            "detwgt file already holds its declared tensors"
        );
        self.remaining -= 1;
        self.bytes(&(name.len() as u32).to_le_bytes())?;
        self.bytes(name.as_bytes())?;
        self.bytes(&(dims.len() as u32).to_le_bytes())?;
        for dim in dims {
            self.bytes(&dim.to_le_bytes())?;
        }
        self.bytes(&dims.iter().product::<u64>().to_le_bytes())?;
        self.bytes(&width.to_le_bytes())?;
        self.bytes(&len.to_le_bytes())?;
        self.bytes(&mass.to_le_bytes())?;
        let padding = (PAYLOAD_ALIGNMENT - self.offset % PAYLOAD_ALIGNMENT) % PAYLOAD_ALIGNMENT;
        self.bytes(&vec![0; padding as usize])
    }

    fn bytes(&mut self, bytes: &[u8]) -> Result<()> {
        self.out.write_all(bytes)?;
        self.offset += bytes.len() as u64;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MmapDetwgt;

    #[test]
    fn written_tensors_read_back() {
        let dir = std::env::temp_dir().join(format!("detwgt-write-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.detwgt");
        let mut block = vec![0_u8; det_num::q8::BLOCK_BYTES * 2];
        block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes()); // scale 1.0
        block[2] = 3;
        block[34..36].copy_from_slice(&0xbc00_u16.to_le_bytes()); // scale -1.0
        block[36] = 2;
        let mut writer = DetwgtWriter::create(&path, 3).unwrap();
        writer.q16("narrow", &[2, 2], &[1, -2, 3, -4]).unwrap();
        writer.q16("wide", &[3], &[1 << 20, 0, -1]).unwrap();
        writer.q8_0("blocks", &[2, 32], &block).unwrap();
        writer.finish().unwrap();

        let weights = MmapDetwgt::open(&path).unwrap();
        assert_eq!(weights.values("narrow").unwrap(), vec![1, -2, 3, -4]);
        assert_eq!(weights.tensor("narrow").unwrap().element_width, 16);
        assert_eq!(weights.values("wide").unwrap(), vec![1 << 20, 0, -1]);
        assert!(weights.is_q8("blocks").unwrap());
        assert!(weights.values("blocks").is_err());
        let matrix = weights.q8_matrix("blocks").unwrap();
        assert_eq!((matrix.rows(), matrix.cols()), (2, 32));
        assert_eq!(matrix.row_dequantized(0).unwrap()[0], 3 << 16);
        assert_eq!(matrix.row_dequantized(1).unwrap()[0], -2 << 16);
        assert_eq!(weights.q8_rows("blocks", 1, 2).unwrap(), &block[34..]);
        assert_eq!(weights.q8_column_blocks("blocks", 0, 32).unwrap(), block);
        assert!(weights.q8_column_blocks("blocks", 0, 16).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
