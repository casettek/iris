//! Compiles `cuda/q8_product.cu` when the `gpu-kernels` feature is on. Without
//! the feature this does nothing and the crate needs no CUDA toolkit.
//!
//! `CUDA_HOME` names the toolkit (default `/usr/local/cuda`) and `NVCC_CCBIN`
//! the host compiler `nvcc` drives (default `g++`).

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn run(command: &mut Command) {
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("failed to start {command:?}: {error}"));
    assert!(status.success(), "{command:?} failed");
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if env::var_os("CARGO_FEATURE_GPU_KERNELS").is_none() {
        return;
    }
    println!("cargo:rerun-if-changed=cuda/q8_product.cu");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=NVCC_CCBIN");

    let cuda = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let host_compiler = env::var("NVCC_CCBIN").unwrap_or_else(|_| "g++".into());
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let object = out.join("q8_product.o");
    let archive = out.join("libiris_q8gpu.a");

    run(Command::new(format!("{cuda}/bin/nvcc"))
        .args(["-O3", "-std=c++17", "-ccbin", &host_compiler])
        .args(["-Xcompiler", "-fPIC", "-c", "cuda/q8_product.cu", "-o"])
        .arg(&object));
    let _ = std::fs::remove_file(&archive);
    run(Command::new("ar").arg("crs").arg(&archive).arg(&object));

    // The CUDA runtime is linked statically so the binary needs no library
    // path at run time, only the NVIDIA driver.
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=iris_q8gpu");
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    println!("cargo:rustc-link-lib=static=cudart_static");
    for library in ["stdc++", "dl", "rt", "pthread"] {
        println!("cargo:rustc-link-lib=dylib={library}");
    }
}
