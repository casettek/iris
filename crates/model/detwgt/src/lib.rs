pub mod eager;
pub mod mmap;

pub use eager::{load, Artifact, Tensor};
pub use mmap::{DetwgtMatrixView, DetwgtSlice, MmapDetwgt, TensorEntry};
