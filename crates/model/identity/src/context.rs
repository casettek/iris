//! Execution context shared by all authored stage boundaries.
//! Stage schema discovery sees this file through a schema-only symlink.
use crate::ModelDescriptor;
use alloc::string::String;
use serde::{Deserialize, Serialize};

pub const ORIGIN: u32 = 0;
pub const PROMPT: u32 = 1;
pub const EMBEDDING: u32 = 2;
pub const AUXILIARY: u32 = 3;
pub const TRANSFORMER: u32 = 4;
pub const DECODE: u32 = 5;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, raster::Selectable)]
pub struct ModelContext {
    pub model_id: String,
    pub execution: raster::BytesPage,
    pub execution_fault: u32,
    pub descriptor: raster::BytesPage,
    pub fault: u32,
    pub layers: u32,
    pub completed_layers: u32,
    pub producer: u32,
}
impl ModelContext {
    pub fn bind(bytes: &[u8], expected: &str) -> Self {
        let mut out = Self {
            model_id: expected.into(),
            descriptor: raster::BytesPage::__from_parts(0, 0, bytes.into()),
            producer: PROMPT,
            ..Self::default()
        };
        let valid = (|| {
            let descriptor = ModelDescriptor::decode(bytes).ok()?;
            let root = parse_id(expected)?;
            descriptor.bind(&root).ok()?;
            out.layers = descriptor.transformer.len() as u32;
            Some(())
        })();
        if valid.is_none() {
            out.fault = 1;
        }
        out
    }
    pub fn is_origin(&self) -> bool {
        self == &Self::default()
    }
    pub fn require(&mut self, condition: bool) {
        if !condition {
            self.fault = 1;
        }
    }
    pub fn same_model(&mut self, other: &Self) {
        self.fault |= other.fault;
        self.execution_fault |= other.execution_fault;
        self.require(
            self.model_id == other.model_id
                && self.descriptor == other.descriptor
                && self.layers == other.layers,
        );
    }
    pub fn valid(&self) -> bool {
        parse_id(&self.model_id).is_some()
            && self.layers > 0
            && self.layers as usize <= crate::MAX_LAYERS
            && self.completed_layers <= self.layers
            && (PROMPT..=DECODE).contains(&self.producer)
    }
    pub fn embedding(prompt: &Self, previous: &Self) -> Self {
        let mut out = if prompt.is_origin() {
            previous.clone()
        } else {
            prompt.clone()
        };
        out.require(out.valid());
        if !prompt.is_origin() {
            out.require(prompt.producer == PROMPT && prompt.completed_layers == 0);
        }
        if !previous.is_origin() {
            out.same_model(previous);
            out.require(previous.producer == EMBEDDING && previous.completed_layers == 0);
        }
        out.begin_embedding_execution(prompt, previous);
        out.producer = EMBEDDING;
        out.completed_layers = 0;
        out
    }
    pub fn auxiliary(&self, layer: u32) -> Self {
        let mut out = self.clone();
        out.require(
            self.valid()
                && self.producer == EMBEDDING
                && self.completed_layers == 0
                && layer < self.layers,
        );
        out.auxiliary_execution(layer);
        out.producer = AUXILIARY;
        out.completed_layers = layer;
        out
    }
    pub fn layer(
        &self,
        layer: u32,
        aux: &Self,
        prior: &Self,
        donor_a: &Self,
        donor_b: &Self,
        donor: i32,
        donor_a_layer: i32,
        donor_b_layer: i32,
    ) -> Self {
        let mut out = self.clone();
        out.require(
            self.valid()
                && layer < self.layers
                && self.completed_layers == layer
                && self.producer == if layer == 0 { EMBEDDING } else { TRANSFORMER },
        );
        out.same_model(aux);
        out.require(aux.producer == AUXILIARY && aux.completed_layers == layer);
        if !prior.is_origin() {
            out.same_model(prior);
            out.require(
                (prior.producer == TRANSFORMER
                    && prior.completed_layers == layer.saturating_add(1))
                    || (prior.producer == EMBEDDING && prior.completed_layers == 0),
            );
        }
        for (context, candidate) in [(donor_a, donor_a_layer), (donor_b, donor_b_layer)] {
            out.fault |= context.fault;
            if donor >= 0 && candidate == donor {
                out.same_model(context);
                out.require(
                    context.producer == TRANSFORMER
                        && context.completed_layers == (donor as u32).saturating_add(1)
                        && (donor as u32) < layer,
                );
            }
        }
        out.require(donor < 0 || donor == donor_a_layer || donor == donor_b_layer);
        out.layer_execution(
            aux,
            prior,
            donor_a,
            donor_b,
            donor,
            donor_a_layer,
            donor_b_layer,
        );
        out.layer_step(
            layer,
            aux,
            prior,
            donor_a,
            donor_b,
            donor,
            donor_a_layer,
            donor_b_layer,
        );
        out.producer = TRANSFORMER;
        out.completed_layers = layer.saturating_add(1);
        out
    }
    pub fn head(&self, previous: &Self, prompt: &Self, prior: &Self) -> Self {
        let mut out = self.clone();
        out.require(
            self.valid() && self.producer == TRANSFORMER && self.completed_layers == self.layers,
        );
        out.same_model(previous);
        out.require(previous.producer == TRANSFORMER && previous.completed_layers == self.layers);
        out.same_model(prompt);
        out.require(prompt.producer == PROMPT && prompt.completed_layers == 0);
        if !prior.is_origin() {
            out.same_model(prior);
            out.require(prior.producer == DECODE && prior.completed_layers == self.layers);
        }
        out.head_execution(previous, prompt, prior);
        out.producer = DECODE;
        out
    }
    pub fn final_fault(&self, prompt: &Self, expected: &Self) -> u32 {
        let mut out = self.clone();
        out.same_model(prompt);
        out.same_model(expected);
        out.require(
            self.valid() && self.producer == DECODE && self.completed_layers == self.layers,
        );
        out.require(prompt.producer == PROMPT && prompt.completed_layers == 0);
        out.final_execution(prompt);
        out.fault
    }
}
pub fn parse_id(text: &str) -> Option<[u8; 32]> {
    if text.len() != 64 {
        return None;
    }
    let digit = |b| match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        _ => None,
    };
    let mut out = [0; 32];
    for (i, pair) in text.as_bytes().chunks_exact(2).enumerate() {
        out[i] = (digit(pair[0])? << 4) | digit(pair[1])?;
    }
    Some(out)
}

impl Default for ModelContext {
    fn default() -> Self {
        Self {
            model_id: String::new(),
            execution: raster::BytesPage::__from_parts(0, 0, alloc::vec![]),
            execution_fault: 0,
            descriptor: raster::BytesPage::__from_parts(0, 0, alloc::vec![]),
            fault: 0,
            layers: 0,
            completed_layers: 0,
            producer: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{format, vec};
    fn prompt(seed: u8) -> ModelContext {
        let d = ModelDescriptor {
            version: 1,
            tokenizer: [seed; 32],
            decoder: [2; 32],
            embedding: [3; 32],
            head: [4; 32],
            auxiliary: vec![[5; 32]; 2],
            transformer: vec![[6; 32]; 2],
        };
        let id = d
            .model_id()
            .unwrap()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        let mut p = ModelContext::bind(&d.encode().unwrap(), &id);
        p.bind_statement(&"11".repeat(32), false, 4, 4);
        p.bind_schedule(pipeline_core::StageSchedule {
            prompt_ranges: 1,
            claimed_ranges: 1,
        });
        p.begin_prompt_range();
        p.append_consumed_token(7);
        p.append_consumed_token(8);
        p.finish_prompt_range(true, 2, 0);
        p
    }
    fn embedded(p: &ModelContext) -> ModelContext {
        let mut e = ModelContext::embedding(p, &ModelContext::default());
        e.append_consumed_token(7);
        e.append_consumed_token(8);
        e.finish_embedding_execution(0, 2, 0);
        e
    }
    fn layer(input: &ModelContext, embedding: &ModelContext, index: u32) -> ModelContext {
        input.layer(
            index,
            &embedding.auxiliary(index),
            &ModelContext::default(),
            &ModelContext::default(),
            &ModelContext::default(),
            -1,
            -1,
            -1,
        )
    }
    #[test]
    fn honest_context_progresses_and_faults_are_sticky() {
        let p = prompt(1);
        let e = embedded(&p);
        let a = layer(&e, &e, 0);
        let b = layer(&a, &e, 1);
        let mut c = ModelContext::embedding(&ModelContext::default(), &e);
        c.append_consumed_token(9);
        c.finish_embedding_execution(2, 0, 1);
        let ca = c.layer(
            0,
            &c.auxiliary(0),
            &a,
            &ModelContext::default(),
            &ModelContext::default(),
            -1,
            -1,
            -1,
        );
        let cb = ca.layer(
            1,
            &c.auxiliary(1),
            &b,
            &ModelContext::default(),
            &ModelContext::default(),
            -1,
            -1,
            -1,
        );
        let h = cb.head(&b, &p, &ModelContext::default());
        assert_eq!(h.final_fault(&p, &p), 0);
        let mut bad = e.clone();
        bad.fault = 8;
        assert_ne!(
            layer(&layer(&bad, &bad, 0), &bad, 1)
                .head(&b, &p, &ModelContext::default())
                .final_fault(&p, &p),
            0
        );
    }
    #[test]
    fn mixed_models_skipped_repeated_swapped_layers_and_short_head_fail() {
        let p = prompt(1);
        let other = prompt(9);
        let e = embedded(&p);
        let a = layer(&e, &e, 0);
        let b = layer(&a, &e, 1);
        assert_ne!(ModelContext::embedding(&other, &e).fault, 0);
        assert_ne!(layer(&e, &e, 1).fault, 0);
        assert_ne!(layer(&a, &e, 0).fault, 0);
        assert_ne!(a.head(&a, &p, &ModelContext::default()).fault, 0);
        assert_ne!(b.head(&b, &other, &ModelContext::default()).fault, 0);
        assert_ne!(
            b.head(&b, &p, &ModelContext::default())
                .final_fault(&p, &other),
            0
        );
        assert_ne!(e.layer(0, &e.auxiliary(1), &e, &e, &e, -1, -1, -1).fault, 0);
        assert_ne!(
            ModelContext::default().final_fault(&ModelContext::default(), &ModelContext::default()),
            0
        );
    }
    #[test]
    fn donor_and_prior_cache_roles_are_checked() {
        let p = prompt(1);
        let e = embedded(&p);
        let a = layer(&e, &e, 0);
        let b = layer(&a, &e, 1);
        assert_eq!(
            a.layer(
                1,
                &e.auxiliary(1),
                &ModelContext::default(),
                &a,
                &e,
                0,
                0,
                -1
            )
            .fault,
            0
        );
        assert_ne!(a.layer(1, &e.auxiliary(1), &a, &a, &e, 0, 0, -1).fault, 0);
        assert_ne!(a.layer(1, &e.auxiliary(1), &b, &b, &e, 0, 0, -1).fault, 0);
        assert_ne!(a.layer(1, &e.auxiliary(1), &b, &e, &e, 0, 0, -1).fault, 0);
        let mut foreign = a.clone();
        foreign.model_id = prompt(9).model_id;
        assert_ne!(
            a.layer(1, &e.auxiliary(1), &b, &foreign, &e, 0, 0, -1)
                .fault,
            0
        );
    }
    #[test]
    fn descriptor_failure_preserves_declared_identity() {
        let p = prompt(1);
        let q = prompt(8);
        let wrong = ModelContext::bind(p.descriptor.as_slice(), &q.model_id);
        assert_eq!(wrong.model_id, q.model_id);
        assert_ne!(wrong.fault, 0);
        assert_ne!(ModelContext::bind(&[], &p.model_id).fault, 0);
    }
}
