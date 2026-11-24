pub mod convert;
pub mod eager;
pub mod gguf;
pub mod mmap;
pub mod write;

pub use eager::{load, Artifact, Tensor};
pub use mmap::{DetwgtMatrixView, DetwgtSlice, MmapDetwgt, Q8MatrixView, TensorEntry, Q8_0_WIDTH};
pub use write::DetwgtWriter;

/// What every weight file begins with.
pub(crate) const MAGIC: &[u8; 8] = b"DNWGTV0\0";
/// The weight file format this crate writes. It reads this and version 2.
pub const DETWGT_VERSION: u32 = 3;
/// The deterministic-number specification a weight file's values follow.
pub const DET_NUM_SPEC_VERSION: u32 = 1;
