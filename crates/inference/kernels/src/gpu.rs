//! The layer's `Q8_0` products on an NVIDIA GPU.
//!
//! `cuda/q8_product.cu` computes each output with the same integers as
//! `det_num::q8::dot_row`: exact block sums, a 128-bit sum of the scaled
//! blocks, one rounding and a saturation. There is no floating point, so a
//! value cannot depend on the device. `row_dot` stays the definition of every
//! value, and a sample of each product is checked against it before the
//! product is used.
//!
//! The GPU is an optimization and never required. A product stays on the CPU
//! when it is small, when the matrix has a scale the CPU would reject, or when
//! the device cannot be used; a device failure is reported once and the rest
//! of the run stays on the CPU. A product that disagrees with the CPU is an
//! error, not a fallback: it would mean wrong activations had been produced.
//!
//! Environment:
//! - `IRIS_GPU_KERNELS=off` keeps every product on the CPU.
//! - `IRIS_GPU_MIN_WORK` is the smallest product, in weight-activation
//!   multiplications, sent to the device (default 2^28).
//! - `IRIS_GPU_WEIGHT_BYTES` bounds the weights kept on the device (default 4 GiB).

use std::collections::HashMap;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use anyhow::{bail, Result};
use det_num::q8;

use crate::tensor::{MatrixSource, Q8Key, Q8Source, Slab};

/// Below this many multiplications the CPU finishes before a device round
/// trip pays off: a decode step's single token stays on the CPU.
const DEFAULT_MIN_WORK: u64 = 1 << 28;
const DEFAULT_WEIGHT_BYTES: u64 = 4 << 30;
/// Outputs of each product recomputed on the CPU, besides its first and last.
const SPOT_CHECKS: usize = 64;

#[repr(C)]
struct DeviceMatrix {
    _private: [u8; 0],
}

extern "C" {
    fn q8gpu_error(code: c_int) -> *const c_char;
    fn q8gpu_init() -> c_int;
    fn q8gpu_upload(
        weights: *const i8,
        scales: *const i64,
        rows: u32,
        cols: u32,
        out: *mut *mut DeviceMatrix,
    ) -> c_int;
    fn q8gpu_free(matrix: *mut DeviceMatrix);
    fn q8gpu_product(
        matrix: *const DeviceMatrix,
        activations: *const i32,
        tokens: u32,
        out: *mut i32,
    ) -> c_int;
}

fn device_error(code: c_int) -> String {
    // SAFETY: `q8gpu_error` returns a static NUL-terminated string for any code.
    unsafe { CStr::from_ptr(q8gpu_error(code)) }
        .to_string_lossy()
        .into_owned()
}

struct Settings {
    enabled: bool,
    min_work: u64,
    weight_bytes: u64,
}

fn settings() -> &'static Settings {
    static SETTINGS: OnceLock<Settings> = OnceLock::new();
    SETTINGS.get_or_init(|| {
        let number = |name: &str, default: u64| {
            std::env::var(name)
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(default)
        };
        Settings {
            enabled: !std::env::var("IRIS_GPU_KERNELS")
                .is_ok_and(|value| value.eq_ignore_ascii_case("off")),
            min_work: number("IRIS_GPU_MIN_WORK", DEFAULT_MIN_WORK),
            weight_bytes: number("IRIS_GPU_WEIGHT_BYTES", DEFAULT_WEIGHT_BYTES),
        }
    })
}

struct Held {
    matrix: *mut DeviceMatrix,
    bytes: u64,
    last_used: u64,
}

/// The device's state. Everything that touches the device holds this lock, so
/// one product runs at a time.
#[derive(Default)]
struct Device {
    ready: bool,
    failed: bool,
    held: HashMap<Q8Key, Held>,
    held_bytes: u64,
    clock: u64,
    sample: u64,
}

// SAFETY: the device pointers are only used while the `DEVICE` lock is held.
unsafe impl Send for Device {}

static DEVICE: Mutex<Option<Device>> = Mutex::new(None);
static PRODUCTS: AtomicU64 = AtomicU64::new(0);

/// How many products this process has run on the device.
pub fn products_on_device() -> u64 {
    PRODUCTS.load(Ordering::Relaxed)
}

impl Device {
    /// Stops using the device for the rest of the process.
    fn fail(&mut self, what: &str, code: c_int) {
        if !self.failed {
            eprintln!(
                "iris: GPU kernels disabled, continuing on the CPU: {what} failed: {}",
                device_error(code)
            );
        }
        self.failed = true;
    }

    fn start(&mut self) -> bool {
        if self.failed {
            return false;
        }
        if !self.ready {
            // SAFETY: plain FFI call with no arguments.
            let code = unsafe { q8gpu_init() };
            if code != 0 {
                self.fail("starting the device", code);
                return false;
            }
            self.ready = true;
        }
        true
    }

    /// The matrix on the device, uploading it on first use. `None` keeps the
    /// product on the CPU.
    fn matrix(
        &mut self,
        source: &Q8Source<'_>,
        rows: usize,
        cols: usize,
    ) -> Option<*const DeviceMatrix> {
        self.clock += 1;
        if let Some(held) = self.held.get_mut(&source.key) {
            held.last_used = self.clock;
            return Some(held.matrix);
        }
        let blocks = rows * cols / q8::BLOCK;
        if source.bytes.len() != blocks * q8::BLOCK_BYTES {
            return None;
        }
        let bytes = (rows * cols + blocks * std::mem::size_of::<i64>()) as u64;
        if bytes > settings().weight_bytes {
            return None;
        }
        // The device takes weights and decoded scales apart. A scale that is
        // not finite is the CPU's error to report.
        let mut weights = Vec::with_capacity(rows * cols);
        let mut scales = Vec::with_capacity(blocks);
        for block in source.bytes.chunks_exact(q8::BLOCK_BYTES) {
            scales.push(q8::scale_q24(u16::from_le_bytes([block[0], block[1]]))?);
            weights.extend(block[2..].iter().map(|byte| *byte as i8));
        }
        while self.held_bytes + bytes > settings().weight_bytes {
            let oldest = *self.held.iter().min_by_key(|(_, held)| held.last_used)?.0;
            let evicted = self.held.remove(&oldest)?;
            // SAFETY: the pointer came from `q8gpu_upload` and leaves the map here.
            unsafe { q8gpu_free(evicted.matrix) };
            self.held_bytes -= evicted.bytes;
        }
        let mut matrix = std::ptr::null_mut();
        // SAFETY: both buffers hold exactly the sizes the device reads.
        let code = unsafe {
            q8gpu_upload(
                weights.as_ptr(),
                scales.as_ptr(),
                rows as u32,
                cols as u32,
                &mut matrix,
            )
        };
        if code != 0 {
            self.fail("uploading weights", code);
            return None;
        }
        self.held.insert(
            source.key,
            Held {
                matrix,
                bytes,
                last_used: self.clock,
            },
        );
        self.held_bytes += bytes;
        Some(matrix)
    }

    /// The next of a fixed pseudo-random sequence, for choosing outputs to check.
    fn next_sample(&mut self) -> u64 {
        self.sample = self.sample.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.sample;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }
}

/// `matrix` applied to every row of `input` on the device, or `None` when the
/// product should run on the CPU.
pub(crate) fn slab_product<M: MatrixSource + ?Sized>(
    input: &Slab,
    matrix: &M,
) -> Result<Option<Slab>> {
    let settings = settings();
    if !settings.enabled {
        return Ok(None);
    }
    let work = input.rows() as u64 * matrix.rows() as u64 * matrix.cols() as u64;
    if work < settings.min_work {
        return Ok(None);
    }
    device_product(input, matrix)
}

/// [`slab_product`] without the size threshold.
fn device_product<M: MatrixSource + ?Sized>(input: &Slab, matrix: &M) -> Result<Option<Slab>> {
    let (rows, cols, tokens) = (matrix.rows(), matrix.cols(), input.rows());
    let Some(source) = matrix.q8_source() else {
        return Ok(None);
    };
    if tokens == 0
        || rows == 0
        || input.cols() != cols
        || [rows, cols, tokens]
            .iter()
            .any(|count| *count > i32::MAX as usize)
    {
        return Ok(None);
    }

    let mut guard = DEVICE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let device = guard.get_or_insert_with(Device::default);
    if !device.start() {
        return Ok(None);
    }
    let Some(on_device) = device.matrix(&source, rows, cols) else {
        return Ok(None);
    };
    let mut output = Slab::zeroed(tokens, rows);
    // SAFETY: `input` holds tokens * cols values and `output` tokens * rows,
    // the sizes the device reads and writes for this matrix.
    let code = unsafe {
        q8gpu_product(
            on_device,
            input.as_flat().as_ptr(),
            tokens as u32,
            output.as_flat_mut().as_mut_ptr(),
        )
    };
    if code != 0 {
        device.fail("a product", code);
        return Ok(None);
    }

    let outputs = tokens * rows;
    let checks = (0..SPOT_CHECKS)
        .map(|_| (device.next_sample() % outputs as u64) as usize)
        .chain([0, outputs - 1]);
    for index in checks {
        let (token, row) = (index / rows, index % rows);
        let expected = matrix.row_dot(row, input.row(token))?;
        let actual = output.row(token)[row];
        if actual != expected {
            bail!(
                "GPU Q8_0 product disagrees with the CPU at token {token}, row {row} of a \
                 {rows}x{cols} matrix: {actual} against {expected}"
            );
        }
    }
    if PRODUCTS.fetch_add(1, Ordering::Relaxed) == 0 {
        eprintln!("iris: running Q8_0 layer products on the GPU");
    }
    Ok(Some(output))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Q8Matrix;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// Every weight byte, and finite scales of either sign up to `max_exponent`.
    fn matrix(rows: usize, cols: usize, max_exponent: u64, rng: &mut Rng) -> Q8Matrix {
        let mut bytes = Vec::new();
        for _ in 0..rows * cols / q8::BLOCK {
            let value = rng.next();
            let exponent = (value >> 20) % (max_exponent + 1);
            let bits = ((value & 1) << 15) | (exponent << 10) | ((value >> 4) & 0x3ff);
            bytes.extend_from_slice(&(bits as u16).to_le_bytes());
            bytes.extend((0..q8::BLOCK).map(|_| (rng.next() >> 24) as u8));
        }
        Q8Matrix::from_bytes("test", bytes, rows, cols).unwrap()
    }

    fn activations(tokens: usize, cols: usize, magnitude_bits: u32, rng: &mut Rng) -> Slab {
        let mut slab = Slab::zeroed(tokens, cols);
        for value in slab.as_flat_mut() {
            let random = rng.next();
            *value = match random % 61 {
                0 => i32::MAX,
                1 => i32::MIN,
                2 => 0,
                _ => ((random >> 16) as i32) >> (32 - magnitude_bits),
            };
        }
        slab
    }

    /// Runs on the device and compares every value with `dot_row`. Returns
    /// false when this machine has no usable device.
    fn agrees(matrix: &Q8Matrix, input: &Slab) -> bool {
        let Some(output) = device_product(input, matrix).unwrap() else {
            return false;
        };
        for token in 0..input.rows() {
            for row in 0..matrix.rows() {
                assert_eq!(
                    output.row(token)[row],
                    q8::dot_row(input.row(token), matrix.row(row)).unwrap(),
                    "{}x{}, token {token}, row {row}",
                    matrix.rows(),
                    matrix.cols()
                );
            }
        }
        true
    }

    #[test]
    fn device_products_equal_one_q8_dot_per_value() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        // Shapes on and across the kernel's 16-row, 16-token tiles.
        for (rows, cols, tokens) in [
            (1, 32, 1),
            (16, 32, 16),
            (17, 64, 15),
            (130, 96, 33),
            (300, 1536, 70),
        ] {
            // Saturating, mid-range and small products.
            for (max_exponent, magnitude_bits) in [(30, 32), (18, 20), (5, 12)] {
                let matrix = matrix(rows, cols, max_exponent, &mut rng);
                let input = activations(tokens, cols, magnitude_bits, &mut rng);
                if !agrees(&matrix, &input) {
                    eprintln!("no usable GPU: skipped");
                    return;
                }
            }
        }
        assert!(products_on_device() > 0);
    }

    #[test]
    fn device_products_round_ties_to_even() {
        // Scale 2^-15 and activations in steps of 2^-2: every product is a
        // multiple of half a Q16.16 step, so half the outputs are exact ties.
        let (rows, cols, tokens) = (64, 32, 96);
        let mut bytes = Vec::new();
        for row in 0..rows {
            bytes.extend_from_slice(&0x0200_u16.to_le_bytes());
            let mut weights = [0_u8; q8::BLOCK];
            weights[row % q8::BLOCK] = [1_i8, -1, 3, -3, 5, -7, 127, -128][row % 8] as u8;
            bytes.extend_from_slice(&weights);
        }
        let matrix = Q8Matrix::from_bytes("ties", bytes, rows, cols).unwrap();
        let mut input = Slab::zeroed(tokens, cols);
        for (index, value) in input.as_flat_mut().iter_mut().enumerate() {
            *value = ((index / cols) as i32 - 48 + (index % cols) as i32) << 14;
        }
        if !agrees(&matrix, &input) {
            eprintln!("no usable GPU: skipped");
        }
    }

    #[test]
    fn a_matrix_with_a_scale_that_is_not_finite_stays_on_the_cpu() {
        let mut bytes = vec![0_u8; q8::BLOCK_BYTES];
        bytes[..2].copy_from_slice(&0x7c00_u16.to_le_bytes());
        let matrix = Q8Matrix::from_bytes("infinite", bytes, 1, 32).unwrap();
        assert!(device_product(&Slab::zeroed(4, 32), &matrix)
            .unwrap()
            .is_none());
    }
}
