//! Deterministic fixed-point kernels — vendored verbatim from
//! `casettek/raster-inference`'s `src/shared/numerics/det_num/`.
//!
//! `types.rs` and `ops.rs` are byte-for-byte copies apart from the module's
//! `use` header, which is rewritten for `alloc`. Keep them that way: the value
//! of this crate is that it is *the same arithmetic*, not merely equivalent
//! arithmetic. `convert.rs` carries the local float-to-fixed conversions that
//! host import code needs before deterministic execution begins.
//!
//! `#![no_std]` so these compile into RISC0 replay guests alongside the tiles
//! that call them.

#![no_std]

extern crate alloc;

pub mod convert;
pub mod ops;
pub mod types;

pub use convert::{f32_to_acc, f32_to_act, f32_to_wgt};
pub use types::{Acc, Act, Wgt};
