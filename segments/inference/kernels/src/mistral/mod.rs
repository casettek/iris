//! `mistral-dense`: the family's native layer.
//!
//! Direct inference runs `layer`. `stage` is the `layer-mistral` stage
//! program's native twin, built on it. The family's integer functions are
//! `det-num`'s `mistral` module, which the stage's tiles compile too.

pub mod layer;
pub mod stage;
pub use det_num::mistral as num;

pub use layer::{layer_rows_and_keys, LayerMatrices, LayerParams};
