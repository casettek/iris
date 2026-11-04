//! Fixture and trace capture shared by the three page-iteration regressions.
use raster::core::trace::TraceEvent;
use raster::prelude::*;
use std::sync::Mutex;

pub const WIDTH: u32 = 128;
pub const ONE: i32 = 1 << 16;
pub const ROWS_PER_PAGE: usize = 196_608 / (WIDTH as usize * 4);
/// Bytes of one `Q8_0` row of `WIDTH` values.
pub const Q8_ROW_BYTES: usize = WIDTH as usize / 32 * 34;
static ITERATIONS: Mutex<usize> = Mutex::new(0);

pub struct Counter;
impl raster_runtime::Publisher for Counter {
    fn publish(&self, event: TraceEvent) {
        if matches!(event, TraceEvent::RecurTileIterationExec(_)) {
            *ITERATIONS.lock().unwrap() += 1;
        }
    }
    fn finish(&self) {}
}

pub fn reset_count() {
    *ITERATIONS.lock().unwrap() = 0;
}
pub fn iteration_count() -> usize {
    *ITERATIONS.lock().unwrap()
}

/// Sparse, signed Q16 rows with an independently known dot product against ones.
/// Cases cover an empty input, a short page, two full pages and a short last page.
pub fn cases() -> [usize; 4] {
    [0, 1, ROWS_PER_PAGE * 2, ROWS_PER_PAGE * 2 + 1]
}

/// The same cases for `Q8_0` pages of `P` bytes.
pub fn q8_cases<const P: u64>() -> [usize; 4] {
    let per_page = P as usize / Q8_ROW_BYTES;
    [0, 1, per_page * 2, per_page * 2 + 1]
}

/// The `fixture` rows as exact `Q8_0` blocks: scale 0.25 holds both the
/// integer and the quarter-step column, so the dot products are unchanged.
pub fn q8_fixture<const P: u64>(rows: usize) -> (AuthRef<Bytes<P>>, Vec<i32>) {
    let mut bytes = Vec::new();
    let mut expected = Vec::new();
    for row in 0..rows {
        let a = row as i32 % 17 - 8;
        let b = row as i32 % 5 - 2;
        expected.push(a * ONE + b * (ONE / 4));
        for block in 0..WIDTH / 32 {
            bytes.extend_from_slice(&0x3400_u16.to_le_bytes());
            for col in 0..32 {
                let q = match (block, col) {
                    (0, 0) => 4 * a,
                    (0, 1) => b,
                    _ => 0,
                };
                bytes.push(q as i8 as u8);
            }
        }
    }
    let weights = Bytes::<P>::paged(bytes).unwrap();
    let reference = raster::store_value(&weights).unwrap();
    (
        raster::into_auth_ref(storage!(Bytes<P>, reference)),
        expected,
    )
}
pub fn fixture(rows: usize) -> (AuthRef<Bytes<196_608>>, Vec<i32>) {
    let mut bytes = Vec::new();
    let mut expected = Vec::new();
    for row in 0..rows {
        let a = (row as i32 % 17 - 8) * ONE;
        let b = (row as i32 % 5 - 2) * (ONE / 4);
        expected.push(a + b);
        for col in 0..WIDTH {
            let value = match col {
                0 => a,
                1 => b,
                _ => 0,
            };
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    let weights = Bytes::<196_608>::paged(bytes).unwrap();
    let reference = raster::store_value(&weights).unwrap();
    (
        raster::into_auth_ref(storage!(Bytes<196_608>, reference)),
        expected,
    )
}
