#[allow(dead_code)]
#[path = "main.rs"]
mod main_impl;

pub mod externals;

pub use main_impl::{import_model, run_from_args, run_from_env, ImportConfig, ImportResult};
