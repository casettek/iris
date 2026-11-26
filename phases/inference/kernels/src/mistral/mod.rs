//! `mistral-dense`: the family's native layer.
//!
//! The native run runs `layer`. `stage` is the `layer-mistral` stage
//! program's native twin, built on it. The family's integer functions are
//! `inference-numerics`'s `mistral` module, which the stage's tiles compile too.

pub mod layer;
pub mod stage;
pub use inference_numerics::mistral as num;

pub use layer::{layer_rows_and_keys, LayerMatrices, LayerParams};
