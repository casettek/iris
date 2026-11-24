//! Minimal GGUF v3 reader: metadata, tensor directory, and memory-mapped
//! tensor data. Only what conversion needs: no quantization formats beyond
//! `Q8_0`, and no tokenizer reconstruction.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use memmap2::Mmap;

pub const F32: u32 = 0;
pub const F16: u32 = 1;
pub const Q8_0: u32 = 8;
pub const BF16: u32 = 30;

/// A metadata value; arrays keep only their length.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Array(u64),
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// GGUF order: innermost (column) dimension first.
    pub dims: Vec<u64>,
    pub kind: u32,
    offset: u64,
}

impl TensorInfo {
    pub fn element_count(&self) -> u64 {
        self.dims.iter().product()
    }

    fn byte_len(&self) -> Result<u64> {
        let count = self.element_count();
        Ok(match self.kind {
            F32 => count * 4,
            F16 | BF16 => count * 2,
            Q8_0 => {
                ensure!(
                    self.dims.first().is_some_and(|cols| cols % 32 == 0),
                    "Q8_0 tensor rows are not whole blocks"
                );
                count / 32 * 34
            }
            other => bail!("unsupported GGUF tensor type {other}"),
        })
    }
}

pub struct Gguf {
    mmap: Mmap,
    pub metadata: BTreeMap<String, Value>,
    pub tensors: BTreeMap<String, TensorInfo>,
    data_start: usize,
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        // SAFETY: read-only mapping; the file is not mutated while mapped.
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("failed to mmap {}", path.display()))?;
        let mut cursor = Cursor {
            bytes: &mmap,
            offset: 0,
        };
        ensure!(
            cursor.take(4)? == b"GGUF",
            "{} is not a GGUF file",
            path.display()
        );
        let version = cursor.u32()?;
        ensure!(version == 3, "GGUF v{version} is unsupported; expected v3");
        let tensor_count = cursor.u64()?;
        let kv_count = cursor.u64()?;
        let mut metadata = BTreeMap::new();
        for _ in 0..kv_count {
            let key = cursor.string()?;
            let kind = cursor.u32()?;
            metadata.insert(key, cursor.value(kind)?);
        }
        let mut tensors = BTreeMap::new();
        for _ in 0..tensor_count {
            let name = cursor.string()?;
            let rank = cursor.u32()?;
            let dims = (0..rank)
                .map(|_| cursor.u64())
                .collect::<Result<Vec<_>>>()?;
            let kind = cursor.u32()?;
            let offset = cursor.u64()?;
            tensors.insert(name, TensorInfo { dims, kind, offset });
        }
        let alignment = match metadata.get("general.alignment") {
            Some(Value::Int(alignment)) => *alignment as usize,
            _ => 32,
        };
        let data_start = cursor.offset.div_ceil(alignment) * alignment;
        let gguf = Self {
            mmap,
            metadata,
            tensors,
            data_start,
        };
        for name in gguf.tensors.keys() {
            gguf.data(name)
                .with_context(|| format!("GGUF tensor '{name}' is malformed"))?;
        }
        Ok(gguf)
    }

    pub fn data(&self, name: &str) -> Result<&[u8]> {
        let info = self
            .tensors
            .get(name)
            .with_context(|| format!("GGUF has no tensor '{name}'"))?;
        let start = self.data_start + info.offset as usize;
        let end = start + info.byte_len()? as usize;
        self.mmap
            .get(start..end)
            .with_context(|| format!("tensor '{name}' data lies outside the file"))
    }

    pub fn int(&self, key: &str) -> Result<i64> {
        match self.metadata.get(key) {
            Some(Value::Int(value)) => Ok(*value),
            other => bail!("GGUF metadata '{key}' is not an integer: {other:?}"),
        }
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl Cursor<'_> {
    fn take(&mut self, len: usize) -> Result<&[u8]> {
        let slice = self
            .bytes
            .get(self.offset..self.offset + len)
            .context("GGUF header ended mid-record")?;
        self.offset += len;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(len)?).into_owned())
    }

    fn value(&mut self, kind: u32) -> Result<Value> {
        let bytes = |cursor: &mut Self, len| -> Result<[u8; 8]> {
            let mut out = [0; 8];
            out[..len].copy_from_slice(cursor.take(len)?);
            Ok(out)
        };
        Ok(match kind {
            0 => Value::Int(i64::from(self.take(1)?[0])),
            1 => Value::Int(i64::from(self.take(1)?[0] as i8)),
            2 => Value::Int(i64::from(u16::from_le_bytes(self.take(2)?.try_into()?))),
            3 => Value::Int(i64::from(i16::from_le_bytes(self.take(2)?.try_into()?))),
            4 => Value::Int(i64::from(self.u32()?)),
            5 => Value::Int(i64::from(self.u32()? as i32)),
            6 => Value::Float(f64::from(f32::from_le_bytes(self.take(4)?.try_into()?))),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let inner = self.u32()?;
                let count = self.u64()?;
                for _ in 0..count {
                    self.value(inner)?;
                }
                Value::Array(count)
            }
            10 => Value::Int(i64::try_from(self.u64()?)?),
            11 => Value::Int(i64::from_le_bytes(bytes(self, 8)?)),
            12 => Value::Float(f64::from_le_bytes(bytes(self, 8)?)),
            other => bail!("unknown GGUF metadata type {other}"),
        })
    }
}
