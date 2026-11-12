//! Model membership for the exact decoder entry consumed by finalization.
use crate::input::DecoderToken;
use alloc::{string::String, vec, vec::Vec};
use model_identity::{ModelDescriptor, ModelFault, Role};
use raster::core::input::{
    ListProofDirection, ListProofSibling, SelectionProof, SelectionProofStep, SelectorPath,
    SelectorSegment,
};
use raster::prelude::*;

/// Pure implementation shared by tile execution and native tests.
pub fn decoder_membership(
    descriptor: &BytesPage,
    expected_model_id: &BytesPage,
    token_id: u32,
    token_count: u64,
    siblings: &BytesPage,
    token: &DecoderToken,
) -> core::result::Result<(), ModelFault> {
    let descriptor = ModelDescriptor::decode(descriptor.as_slice())?;
    let id: [u8; 32] = expected_model_id
        .as_slice()
        .try_into()
        .map_err(|_| ModelFault::ModelId)?;
    let model = descriptor.bind(&id)?;
    if token_count == 0 || u64::from(token_id) >= token_count {
        return Err(ModelFault::SelectionPath);
    }
    let (mut index, mut width) = (u64::from(token_id), token_count);
    let bytes = siblings.as_slice();
    if bytes.len() > 64 * 32 || bytes.len() % 32 != 0 {
        return Err(ModelFault::Membership);
    }
    let mut chunks = bytes.chunks_exact(32);
    let mut nodes = Vec::new();
    while width > 1 {
        let hash = chunks
            .next()
            .ok_or(ModelFault::Membership)?
            .try_into()
            .unwrap();
        nodes.push(ListProofSibling {
            direction: if index % 2 == 0 {
                ListProofDirection::Right
            } else {
                ListProofDirection::Left
            },
            hash,
        });
        index /= 2;
        width = width / 2 + width % 2;
    }
    if chunks.next().is_some() {
        return Err(ModelFault::Membership);
    }
    let path = SelectorPath::new(vec![
        SelectorSegment::Field("tokens".into()),
        SelectorSegment::Index(token_id as u64),
    ]);
    let proof = SelectionProof {
        root_hash: *model.root(Role::Decoder)?,
        path: path.clone(),
        steps: vec![
            SelectionProofStep::Struct {
                field_index: 0,
                field_names: vec!["tokens".into()],
                siblings: vec![],
            },
            SelectionProofStep::List {
                index: token_id as u64,
                len: token_count,
                siblings: nodes,
            },
        ],
    };
    model.verify_value(Role::Decoder, &path, token, &proof)?;
    Ok(())
}

/// Canonical statement IDs only; no replacement with the witness's identity.
pub fn expected_id(text: &str) -> core::result::Result<BytesPage, ModelFault> {
    if text.len() != 64 {
        return Err(ModelFault::ModelId);
    }
    let digit = |b| match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        _ => Err(ModelFault::ModelId),
    };
    let mut bytes = Vec::with_capacity(32);
    for pair in text.as_bytes().chunks_exact(2) {
        bytes.push((digit(pair[0])? << 4) | digit(pair[1])?);
    }
    Ok(BytesPage::__from_parts(0, 0, bytes))
}

pub fn authenticated_finalize_state(
    descriptor: &BytesPage,
    model_id: &str,
) -> crate::input::FinalizeState {
    let mut state = crate::finalize::initial_finalize_state();
    let valid = (|| {
        let id = expected_id(model_id)?;
        let d = ModelDescriptor::decode(descriptor.as_slice())?;
        d.bind(&id.as_slice().try_into().map_err(|_| ModelFault::ModelId)?)?;
        Ok::<_, ModelFault>(())
    })();
    state.model_fault = u32::from(valid.is_err());
    state
}

#[tile(kind = iter, description = "Bind finalization to the statement model")]
pub fn begin_authenticated_finalize(
    descriptor: BytesPage,
    model_id: String,
) -> crate::input::FinalizeState {
    authenticated_finalize_state(&descriptor, &model_id)
}

pub fn advance_authenticated_finalize(
    mut state: crate::input::FinalizeState,
    descriptor: &BytesPage,
    model_id: &str,
    token_id: u32,
    token_count: u64,
    siblings: &BytesPage,
    token: DecoderToken,
) -> crate::input::FinalizeState {
    let verified = expected_id(model_id).and_then(|id| {
        decoder_membership(descriptor, &id, token_id, token_count, siblings, &token)
    });
    if verified.is_err() {
        state.model_fault = 1;
    }
    let safe_token = if verified.is_ok() {
        token
    } else {
        DecoderToken {
            token: String::new(),
            special: true,
            terminal: false,
        }
    };
    crate::finalize::advance_finalize_state(state, token_id, safe_token)
}

#[tile(kind = iter, description = "Authenticate and consume a generated decoder entry")]
pub fn consume_authenticated_decoder(
    state: crate::input::FinalizeState,
    descriptor: BytesPage,
    model_id: String,
    token_id: u32,
    token_count: u64,
    proof: crate::input::DecoderProof,
    token: DecoderToken,
) -> crate::input::FinalizeState {
    advance_authenticated_finalize(
        state,
        &descriptor,
        &model_id,
        token_id,
        token_count,
        &proof.siblings,
        token,
    )
}

/// Import/native helper. Build each reusable path from the actual token bytes.
/// This is not an execution authorization; the consuming tile checks it again.
pub fn decoder_witness(
    table: &crate::input::DecoderTable,
) -> core::result::Result<crate::input::DecoderWitness, ModelFault> {
    use sha2::{Digest, Sha256};
    let mut levels = Vec::new();
    let mut current = table
        .tokens
        .iter()
        .map(|token| raster::core::tree::value_root(token).map_err(|_| ModelFault::ValueEncoding))
        .collect::<core::result::Result<Vec<_>, _>>()?;
    while current.len() > 1 {
        if current.len() % 2 != 0 {
            current.push(*current.last().unwrap());
        }
        let next = current
            .chunks_exact(2)
            .map(|pair| {
                let mut h = Sha256::new();
                h.update(b"list-node");
                h.update(pair[0]);
                h.update(pair[1]);
                <[u8; 32]>::from(h.finalize())
            })
            .collect::<Vec<_>>();
        levels.push(current);
        current = next;
    }
    let paths = (0..table.tokens.len())
        .map(|mut index| {
            let mut bytes = Vec::with_capacity(levels.len() * 32);
            for level in &levels {
                bytes.extend_from_slice(&level[index ^ 1]);
                index /= 2;
            }
            crate::input::DecoderProof {
                siblings: BytesPage::__from_parts(0, 0, bytes),
            }
        })
        .collect::<Vec<_>>();
    Ok(crate::input::DecoderWitness {
        token_count: table.tokens.len() as u64,
        paths: paths.into(),
    })
}

pub fn final_context_fault(
    edge: &model_identity::ModelContext,
    prompt: &crate::prompt::PromptContext,
    descriptor: &BytesPage,
) -> u32 {
    let expected = model_identity::ModelContext::bind(descriptor.as_slice(), &prompt.model_id);
    let mut checked = edge.clone();
    let mut statement = expected.clone();
    statement.bind_statement(
        &prompt.input_text_sha256,
        prompt.raw_prompt,
        prompt.max_new_tokens,
        prompt.range_size,
    );
    checked.require_execution(
        statement
            .execution_context()
            .zip(edge.execution_context())
            .is_some_and(|(s, e)| {
                s.same_statement(&e)
                    && e.prompt.is_some_and(|p| p.position == prompt.prompt_tokens)
                    && prompt.complete
                    && prompt.fault == 0
            }),
    );
    checked.final_fault(&prompt.model, &expected)
}

#[tile(kind = iter, description = "Retain model faults and check final model lineage")]
pub fn verify_final_model_context(
    mut state: crate::input::FinalizeState,
    edge: model_identity::ModelContext,
    prompt: crate::prompt::PromptContext,
    descriptor: BytesPage,
    has_selected: bool,
    position: u32,
    token: u32,
) -> crate::input::FinalizeState {
    state.model_fault |= final_context_fault(&edge, &prompt, &descriptor);
    begin_final_history(&mut state, &edge, has_selected, position, token);
    state
}

/// Fold the emitted transcript from the checked prompt anchor, independently of
/// the teacher-forced tokens that produced the checked execution endpoint.
pub fn begin_final_history(
    state: &mut crate::input::FinalizeState,
    edge: &model_identity::ModelContext,
    selected: bool,
    position: u32,
    token: u32,
) {
    state.history = edge.clone();
    state.expected_token = token;
    if let Some(mut e) = edge.execution_context() {
        state
            .history
            .require_execution(selected && position.checked_add(1) == Some(e.end.position));
        e.expected = Some(e.end);
        e.end = e.before;
        state.history.set_execution(&e);
    } else {
        state.history.require_execution(false);
    }
    state.model_fault |= state.history.fault;
}
pub fn finish_final_history(state: &mut crate::input::FinalizeState) {
    if state.history.is_origin() {
        return;
    }
    let valid = state.history.execution_context().is_some_and(|e| {
        Some(e.end) == e.expected
            && e.end.position.checked_sub(e.before.position) == Some(state.count)
            && state.count > 0
            && state.count <= e.statement.max_new_tokens
            && state.last_token == state.expected_token
    });
    state.history.require_execution(valid);
    state.model_fault |= state.history.fault;
}
