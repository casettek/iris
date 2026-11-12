use model_identity::ModelDescriptor;
use raster::{BytesPage, List};
use sequence_check::finalize_model::decoder_membership;
use sequence_check::input::{DecoderTable, DecoderToken};

fn page(bytes: Vec<u8>) -> BytesPage {
    BytesPage::__from_parts(0, 0, bytes)
}

#[test]
fn decoder_text_and_flags_are_bound_to_model() {
    let token = DecoderToken {
        token: "hello".into(),
        special: false,
        terminal: false,
    };
    let table = DecoderTable {
        tokens: List::from(vec![token.clone()]),
    };
    let root = raster::core::tree::value_root(&table).unwrap();
    let descriptor = ModelDescriptor {
        version: 2,
        tokenizer: [1; 32],
        decoder: root,
        embedding: [2; 32],
        head: [3; 32],
        auxiliary: vec![[4; 32]],
        transformer: vec![[5; 32]],
    };
    let id = page(descriptor.model_id().unwrap().to_vec());
    let bytes = page(descriptor.encode().unwrap());
    let proof = page(vec![]);
    assert!(decoder_membership(&bytes, &id, 0, 1, &proof, &token).is_ok());
    let mut changed = token.clone();
    changed.token = "goodbye".into();
    assert!(decoder_membership(&bytes, &id, 0, 1, &proof, &changed).is_err());
    changed = token.clone();
    changed.special = true;
    assert!(decoder_membership(&bytes, &id, 0, 1, &proof, &changed).is_err());
    changed = token.clone();
    changed.terminal = true;
    assert!(decoder_membership(&bytes, &id, 0, 1, &proof, &changed).is_err());
    assert!(decoder_membership(&bytes, &id, 1, 1, &proof, &token).is_err());
    assert!(decoder_membership(&bytes, &id, 0, 2, &proof, &token).is_err());
    assert!(decoder_membership(&bytes, &page(vec![0; 32]), 0, 1, &proof, &token).is_err());
}

#[test]
fn decoder_proofs_bind_positions_and_handle_odd_tree_widths() {
    use sha2::{Digest, Sha256};
    let tokens: Vec<_> = ["zero", "one", "two"]
        .into_iter()
        .map(|text| DecoderToken {
            token: text.into(),
            special: false,
            terminal: false,
        })
        .collect();
    let table = DecoderTable {
        tokens: List::from(tokens.clone()),
    };
    let root = raster::core::tree::value_root(&table).unwrap();
    let d = ModelDescriptor {
        version: 2,
        tokenizer: [1; 32],
        decoder: root,
        embedding: [2; 32],
        head: [3; 32],
        auxiliary: vec![[4; 32]],
        transformer: vec![[5; 32]],
    };
    let id = page(d.model_id().unwrap().to_vec());
    let descriptor = page(d.encode().unwrap());
    for index in 0..tokens.len() {
        let mut at = index;
        let mut level: Vec<_> = tokens
            .iter()
            .map(|t| raster::core::tree::value_root(t).unwrap())
            .collect();
        let mut siblings = Vec::new();
        while level.len() > 1 {
            if level.len() % 2 == 1 {
                level.push(*level.last().unwrap());
            }
            siblings.extend_from_slice(&level[at ^ 1]);
            level = level
                .chunks_exact(2)
                .map(|pair| {
                    let mut h = Sha256::new();
                    h.update(b"list-node");
                    h.update(pair[0]);
                    h.update(pair[1]);
                    <[u8; 32]>::from(h.finalize())
                })
                .collect();
            at /= 2;
        }
        let proof = page(siblings);
        assert!(
            decoder_membership(&descriptor, &id, index as u32, 3, &proof, &tokens[index]).is_ok()
        );
        assert!(decoder_membership(
            &descriptor,
            &id,
            ((index + 1) % 3) as u32,
            3,
            &proof,
            &tokens[index]
        )
        .is_err());
        assert!(
            decoder_membership(&descriptor, &id, index as u32, 4, &proof, &tokens[index]).is_err()
        );
        let mut truncated = proof.as_slice().to_vec();
        truncated.pop();
        assert!(decoder_membership(
            &descriptor,
            &id,
            index as u32,
            3,
            &page(truncated),
            &tokens[index]
        )
        .is_err());
    }
}

#[test]
fn faults_survive_later_valid_tokens_and_verification_continues_after_eos() {
    use sequence_check::{
        advance_authenticated_finalize, authenticated_finalize_state, decoder_witness,
    };
    let token = DecoderToken {
        token: "hello".into(),
        special: false,
        terminal: false,
    };
    let table = DecoderTable {
        tokens: vec![token.clone()].into(),
    };
    let descriptor = ModelDescriptor {
        version: 2,
        tokenizer: [1; 32],
        decoder: raster::core::tree::value_root(&table).unwrap(),
        embedding: [2; 32],
        head: [3; 32],
        auxiliary: vec![[4; 32]],
        transformer: vec![[5; 32]],
    };
    let id = descriptor
        .model_id()
        .unwrap()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    let descriptor = page(descriptor.encode().unwrap());
    let witness = decoder_witness(&table).unwrap();
    let proof = &witness.paths[0].siblings;
    let state = authenticated_finalize_state(&descriptor, &id);
    assert_eq!(state.model_fault, 0);
    let bad = DecoderToken {
        terminal: true,
        ..token.clone()
    };
    let state = advance_authenticated_finalize(state, &descriptor, &id, 0, 1, proof, bad.clone());
    assert_eq!(state.model_fault, 1);
    let mut state = advance_authenticated_finalize(state, &descriptor, &id, 0, 1, proof, token);
    assert_eq!(state.model_fault, 1);
    state.model_fault = 0;
    state.stopped = true;
    let state = advance_authenticated_finalize(state, &descriptor, &id, 0, 1, proof, bad);
    assert_eq!(state.model_fault, 1);
    assert!(state.past_terminal);
    assert_eq!(
        authenticated_finalize_state(&page(vec![]), &id).model_fault,
        1
    );
    assert_eq!(
        authenticated_finalize_state(&descriptor, "not-an-id").model_fault,
        1
    );
}
