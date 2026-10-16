//! Fixture and trace capture shared by the three page-iteration regressions.
use raster::core::trace::TraceEvent;
use raster::prelude::*;
use std::sync::Mutex;

pub const WIDTH: u32 = 128;
pub const ONE: i32 = 1 << 16;
pub const ROWS_PER_PAGE: usize = 196_608 / (WIDTH as usize * 4);
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
