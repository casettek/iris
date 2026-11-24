//! What only autoregressive generation needs: the whole vocabulary's logits and
//! the greedy selection over them. The chain's `sequence-check` stage folds the
//! same scores into a running argmax instead of building this list.
use anyhow::{bail, Result};
use raster::List;
use sequence_check::input::DecodeEdge;
use serde::{Deserialize, Serialize};

/// One scored token.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct LogitEntry {
    pub token_id: u32,
    pub value: i32,
}

/// The scores for the position after a pass's last row.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct PrefillLogits {
    pub decode_position: u32,
    pub logits: List<LogitEntry>,
    pub errors: List<String>,
}

/// Appends the highest-scoring token to the transcript. The first maximum wins.
pub fn select_next_token(logits: &PrefillLogits, prior: &DecodeEdge) -> Result<DecodeEdge> {
    let mut best: Option<&LogitEntry> = None;
    for entry in logits.logits.iter() {
        if best.is_none_or(|incumbent| entry.value > incumbent.value) {
            best = Some(entry);
        }
    }
    let Some(best) = best else {
        bail!("output decode requires at least one logit to select the next token");
    };
    let mut generated_token_ids = prior
        .generated_token_ids
        .iter()
        .copied()
        .collect::<Vec<_>>();
    generated_token_ids.push(best.token_id);
    Ok(DecodeEdge {
        model: prior.model.clone(),
        has_selected: true,
        decode_position: logits.decode_position,
        token_id: best.token_id,
        value: best.value,
        // Direct generation is given no claimed tokens.
        claim_mismatches: 0,
        first_claim_mismatch_position: 0,
        check_fault: 0,
        check_fault_position: 0,
        // Stated when the run's output is finalized.
        record: Default::default(),
        claim: Default::default(),
        generated_token_ids: List::from(generated_token_ids),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_keeps_the_first_token_on_a_tie_and_extends_the_transcript() {
        let logits = PrefillLogits {
            decode_position: 9,
            logits: List::from(vec![
                LogitEntry {
                    token_id: 5,
                    value: 10,
                },
                LogitEntry {
                    token_id: 3,
                    value: 10,
                },
            ]),
            errors: List::new(),
        };
        let prior = DecodeEdge {
            has_selected: true,
            decode_position: 8,
            token_id: 11,
            value: 1,
            generated_token_ids: List::from(vec![11, 12]),
            ..Default::default()
        };
        let edge = select_next_token(&logits, &prior).unwrap();
        assert!(edge.has_selected);
        assert_eq!(edge.decode_position, 9);
        assert_eq!(edge.token_id, 5);
        assert_eq!(edge.value, 10);
        assert_eq!(edge.generated_token_ids.as_slice(), &[11, 12, 5]);
        assert!(select_next_token(&PrefillLogits::default(), &prior).is_err());
    }
}
