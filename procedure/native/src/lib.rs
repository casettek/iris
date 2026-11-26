//! The native run of a procedure: every part of it run for its result, as
//! fast as possible, with inference generating the tokens a pipeline run then
//! verifies.

use std::path::{Path, PathBuf};

pub mod config;
pub mod executor;
pub mod infer;

pub use config::NativeInferenceConfig;
pub use executor::{NativeDiagnosticRunReport, NativeInferenceExecutor};
pub use infer::{run_infer, InferRun};

/// `path` as it is, or under `base` when it is relative.
pub fn absolute(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}
