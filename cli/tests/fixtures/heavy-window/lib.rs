#![no_std]
extern crate alloc;
use raster::prelude::*;
use serde::{Deserialize, Serialize};
#[derive(Clone, Serialize, Deserialize, raster::Selectable)]
pub struct Values {
    pub values: List<u64>,
}
#[tile(kind = recur)]
pub fn heavy(input: RecurInput<u64>, state: RecurState<u64>) -> RecurState<u64> {
    let mut state = state;
    let mut value = input.into_value();
    let iterations = if value == 1 { 20000 } else { 200 };
    for _ in 0..iterations {
        value = value.wrapping_mul(6364136223846793005).wrapping_add(1);
    }
    *state = state.wrapping_add(value);
    state
}
#[tile]
pub fn finish(value: u64) -> u64 {
    value
}
