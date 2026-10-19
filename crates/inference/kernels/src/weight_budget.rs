//! Run-local accounting for immutable weights. Active-stage scratch is not retained here.
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct WeightCacheStats {
    pub budget_bytes: usize,
    pub retained_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub bypasses: u64,
    pub preparation_duration_ns: u128,
}

pub struct WeightBudget(Mutex<WeightCacheStats>);

impl WeightBudget {
    pub fn new(bytes: usize) -> Self {
        Self(Mutex::new(WeightCacheStats {
            budget_bytes: bytes,
            ..Default::default()
        }))
    }
    pub fn stats(&self) -> WeightCacheStats {
        *self.0.lock().unwrap()
    }
    pub fn hit(&self) {
        self.0.lock().unwrap().hits += 1;
    }
    pub fn miss(&self) {
        self.0.lock().unwrap().misses += 1;
    }
    pub fn prepared(&self, ns: u128) {
        self.0.lock().unwrap().preparation_duration_ns += ns;
    }
    /// Stable admission: a scan cannot evict weights that will be reused next token.
    pub fn reserve(&self, bytes: usize) -> bool {
        let mut state = self.0.lock().unwrap();
        if bytes > state.budget_bytes.saturating_sub(state.retained_bytes)
            || state.budget_bytes == 0
        {
            state.bypasses += 1;
            return false;
        }
        state.retained_bytes += bytes;
        true
    }
}

/// Struct storage plus all owned heap capacities. Allocator bookkeeping is excluded.
pub trait RetainedWeightBytes {
    fn retained_weight_bytes(&self) -> usize;
}

fn region<const P: u64>(value: &raster::Bytes<P>) -> usize {
    value.pages().capacity() * std::mem::size_of::<raster::BytesPage>()
        + value
            .pages()
            .iter()
            .map(|page| page.allocated_bytes())
            .sum::<usize>()
}

macro_rules! embedding {
    ($ty:ty) => {
        impl RetainedWeightBytes for $ty {
            fn retained_weight_bytes(&self) -> usize {
                std::mem::size_of::<Self>() + region(&self.values)
            }
        }
    };
}
embedding!(input_embedding::input::EmbeddingTable);
embedding!(decode_embed::input::EmbeddingTable);

impl RetainedWeightBytes for prefill_prepare_aux::input::PleLayer {
    fn retained_weight_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + region(&self.embeddings)
            + region(&self.projection)
            + self.params.norm_weights.allocated_bytes()
    }
}

impl RetainedWeightBytes for prefill_finalize::input::FinalHead {
    fn retained_weight_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + region(&self.projection)
            + self.params.norm_weights.allocated_bytes()
    }
}

impl RetainedWeightBytes for prefill_range::input::TransformerLayer {
    fn retained_weight_bytes(&self) -> usize {
        let p = &self.params;
        std::mem::size_of::<Self>()
            + [
                &self.w_q,
                &self.w_k,
                &self.w_v,
                &self.w_o,
                &self.w_gate,
                &self.w_up,
                &self.w_down,
                &self.ple_input_gate,
                &self.ple_layer_projection,
            ]
            .iter()
            .map(|r| region(r))
            .sum::<usize>()
            + [
                &p.norm_input,
                &p.norm_post_attn,
                &p.norm_pre_ffw,
                &p.norm_post_ffw,
                &p.q_norm,
                &p.k_norm,
                &p.ple_post_norm,
            ]
            .iter()
            .map(|p| p.allocated_bytes())
            .sum::<usize>()
    }
}

impl RetainedWeightBytes for String {
    fn retained_weight_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.capacity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn admission_is_bounded_and_does_not_evict() {
        let budget = WeightBudget::new(100);
        assert!(budget.reserve(60));
        assert!(!budget.reserve(41));
        assert!(budget.reserve(40));
        assert_eq!(budget.stats().retained_bytes, 100);
        assert_eq!(budget.stats().bypasses, 1);
        assert!(!WeightBudget::new(0).reserve(0));
    }
}
