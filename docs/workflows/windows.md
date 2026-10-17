# Native Windows inference

Use PowerShell with Visual Studio 2022 C++ build tools and the Rust 1.91.1
MSVC toolchain installed. The sibling Raster checkout must include its
`vendor/risc0-zkvm-platform` compatibility patch.

```powershell
rustup toolchain install 1.91.1 --profile minimal
.\scripts\build-windows.ps1 -Test
```

The script initializes the x64 Visual Studio environment, defines `NOMINMAX`
and selects C++20 with exception unwinding for Windows C++ compilation, and builds the release executable at
`target/windows/release/iris.exe`. Its separate target directory preserves the
WSL build under `target/release`. `-Test` runs host linkage and prover selection
regressions, direct executor tests, and the staged parity test.

The RISC Zero package versions remain pinned in `Cargo.lock`. Local patches fix
an MSVC guest allocator reference (`risc0-zkvm-platform 2.2.3`), Windows paths in
generated Rust source (`risc0-circuit-keccak 4.0.6`), and unconditional Unix actor
transport imports (`risc0-zkvm 3.0.6`). See each crate's `PATCH.md` in the sibling
Raster checkout's `vendor` directory. The `actor` prover remains Unix-only;
Windows requests for it fail explicitly. The local prover is retained.

The script sets `RISC0_SKIP_BUILD=1` while building. This skips generating proof
guest programs; it still compiles the host prover dependencies required by the
current dependency graph. This build is for native import, direct inference,
and staged claim generation. It does not establish native Windows proof
generation support.

## Model paths

WSL manifests contain `/mnt/c/...` paths. Import into a separate Windows
artifact directory rather than replacing a manifest used by an active WSL run:

```powershell
.\target\windows\release\iris.exe model import `
  --model ..\models\gemma-4-E2B-detwgt `
  --artifact-root runtime/windows-model-artifacts `
  --no-stage-fixtures
```

Set `model_manifest` in a separate run spec to
`runtime/windows-model-artifacts/gemma-4-E2B-detwgt/manifest.json`, then run:

```powershell
.\target\windows\release\iris.exe infer --run inference-windows.toml
.\target\windows\release\iris.exe claim build --run inference-windows.toml
```

For a backend running with Windows Node, set `IRIS_BINARY_PATH` to the absolute
Windows path of `target/windows/release/iris.exe` and `IRIS_MODEL_MANIFEST` to
the Windows import's manifest. Keep Windows and WSL dependency installations
and execution storage separate.
