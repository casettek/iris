param(
    [ValidateRange(1, 64)]
    [int]$Jobs = 4,
    [switch]$Test
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$targetDir = Join-Path $repoRoot 'target/windows'
$toolchain = '1.91.1' # Matches the sibling Raster checkout.

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw 'Install Visual Studio Build Tools with Desktop development with C++.'
}
$vsPath = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) {
    throw 'Visual Studio C++ build tools were not found.'
}
& (Join-Path $vsPath 'Common7/Tools/Launch-VsDevShell.ps1') -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null

$previousSkip = $env:RISC0_SKIP_BUILD
$previousCxxFlags = $env:CXXFLAGS_x86_64_pc_windows_msvc
Push-Location $repoRoot
try {
    # Iris native inference needs the host libraries, but does not generate proofs.
    $env:RISC0_SKIP_BUILD = '1'
    # Windows headers must not replace std::min/std::max. The pinned generated
    # kernels also use designated initializers, which MSVC requires C++20 for.
    $env:CXXFLAGS_x86_64_pc_windows_msvc = "$previousCxxFlags /DNOMINMAX /std:c++20 /EHsc".Trim()
    & cargo "+$toolchain" build --release --locked -p iris-cli --bin iris --target-dir $targetDir -j $Jobs
    if ($LASTEXITCODE -ne 0) { throw 'Windows Iris build failed.' }

    if ($Test) {
        & cargo "+$toolchain" test --release --locked -p risc0-zkvm-platform --test host_link --target-dir $targetDir -j $Jobs
        if ($LASTEXITCODE -ne 0) { throw 'Windows host linkage regression failed.' }
        & cargo "+$toolchain" test --release --locked -p raster-prover --test windows_host --target-dir $targetDir -j $Jobs
        if ($LASTEXITCODE -ne 0) { throw 'Windows prover selection regression failed.' }
        & cargo "+$toolchain" test --release --locked -p direct-infer --test executor --target-dir $targetDir -j $Jobs
        if ($LASTEXITCODE -ne 0) { throw 'Direct inference tests failed.' }
        & cargo "+$toolchain" test --release --locked -p staged-infer --test infer_parity --test worker_parity --target-dir $targetDir -j $Jobs
        if ($LASTEXITCODE -ne 0) { throw 'Staged inference parity test failed.' }
    }
    Write-Host "Windows executable: $(Join-Path $targetDir 'release/iris.exe')"
}
finally {
    Pop-Location
    $env:RISC0_SKIP_BUILD = $previousSkip
    $env:CXXFLAGS_x86_64_pc_windows_msvc = $previousCxxFlags
}
