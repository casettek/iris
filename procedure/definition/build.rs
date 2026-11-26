//! Compiles in the procedure kinds: every `.toml` file of `procedures/`, by
//! name. Adding a kind is adding its file.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::{env, fs};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let procedures = manifest_dir.join("../../procedures");
    println!("cargo:rerun-if-changed={}", procedures.display());
    let mut files: Vec<PathBuf> = fs::read_dir(&procedures)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", procedures.display()))
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "toml")
        })
        .collect();
    files.sort();

    let mut catalog =
        String::from("/// Each procedure kind's name and its shape's file, in name order.\n");
    catalog.push_str("pub(crate) const CATALOG: &[(&str, &str)] = &[\n");
    for file in &files {
        println!("cargo:rerun-if-changed={}", file.display());
        let name = file.file_stem().unwrap().to_str().unwrap();
        let path = file.canonicalize().unwrap();
        let path = path.to_str().unwrap();
        writeln!(catalog, "    ({name:?}, include_str!({path:?})),").unwrap();
    }
    catalog.push_str("];\n");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("catalog.rs");
    fs::write(&out, catalog).unwrap();
}
