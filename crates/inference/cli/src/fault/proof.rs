//! Iris witness orchestration adapted from raster-cli commands/prove.rs.
//! Cryptographic replay and proving remain in unchanged Raster libraries.
use sha2::{Digest, Sha256};

use std::collections::{BTreeMap, HashMap};

use raster_backend::ExecutionMode;

use raster_core::cfs::{CfsCoordinates, CfsCursor, ControlFlowSchema};
use raster_core::coordinate_index::IncrementalCoordinateIndex;
use raster_core::input::{SelectionWitness, StorageRef};
use raster_core::trace::{ExecStep, ExecTarget, FnInput, StepKind, StepRecord, Trace};
use raster_core::transition::{
    StorageEntry, StorageIndexValue, StorageLogWitness, StorageReadWitness, StorageWitness,
    StorageWriteWitness,
};

use raster_core::merkle::{to_node_hash, EMPTY_LEAF};
use raster_prover::authorization::authorize_external_inputs;
use raster_prover::replay::{ReplayResult, Replayer};
use raster_prover::trace::{
    Bytes, FraudEvidence, SerializableFrontier, TraceCommitment, TraceCommitmentExt, TraceTree,
};
use raster_prover::transition::{step_transitions, StepIo};
use raster_runtime::TraceRecorder;

#[derive(Debug, Clone)]
struct ProofStorageState {
    frontier: SerializableFrontier,
    append_entries: Vec<StorageEntry>,
    coordinate_index: IncrementalCoordinateIndex,
}

fn empty_storage_frontier() -> SerializableFrontier {
    SerializableFrontier {
        position: 0,
        leaf: EMPTY_LEAF.to_vec(),
        ommers: Vec::new(),
    }
}

fn storage_leaf_hash(entry: &StorageEntry) -> Vec<u8> {
    Sha256::digest(entry.to_bytes()).to_vec()
}

fn build_storage_log_witness(
    append_entries: &[StorageEntry],
    log_position: u64,
) -> StorageLogWitness {
    let mut tree = TraceTree::new(1);
    tree.append(Bytes(EMPTY_LEAF.to_vec()));
    let mut marked_position = None;

    for (index, entry) in append_entries.iter().enumerate() {
        tree.append(Bytes(storage_leaf_hash(entry)));
        if u64::try_from(index).expect("append entry index overflow") + 1 == log_position {
            marked_position = tree.mark();
        }
    }

    let marked_position = marked_position.unwrap_or_else(|| {
        panic!(
            "Missing append-log position {} while building storage log witness",
            log_position
        )
    });
    let auth_path = tree
        .witness(marked_position, 0)
        .expect("Failed to build storage log witness");

    StorageLogWitness {
        position: u64::from(marked_position),
        path_elems: auth_path.iter().map(|elem| to_node_hash(&elem.0)).collect(),
    }
}

fn apply_storage_write_to_state(
    state: &mut ProofStorageState,
    storage_write: &raster_runtime::StorageWriteRecord,
) {
    state.frontier = storage_write.frontier_after.clone();
    state.append_entries.push(storage_write.entry.clone());
    state.coordinate_index.insert(
        storage_write.entry.coordinates.clone(),
        StorageIndexValue {
            log_position: storage_write.log_position,
            object_commitment: storage_write.entry.object_commitment.clone(),
        },
    );
}

fn storage_state_from_prefix(
    trace: &[StepRecord],
    trace_recorder: &TraceRecorder,
) -> ProofStorageState {
    let mut state = ProofStorageState {
        frontier: empty_storage_frontier(),
        append_entries: Vec::new(),
        coordinate_index: IncrementalCoordinateIndex::new(),
    };
    for step_record in trace {
        // Only a storage-appending step owns the write recorded at its
        // coordinate. Gating on this keeps the entry-object write from being
        // applied twice: `ProgramStart` (append) shares coordinates `[]` with
        // the read-only `ProgramEnd` and `main`'s `SequenceEnd`.
        if !step_record.appends_to_storage() {
            continue;
        }
        if let Some(storage_write) = trace_recorder
            .step_witness_at(step_record.coordinates())
            .and_then(|witness| witness.storage_write())
        {
            apply_storage_write_to_state(&mut state, &storage_write);
        }
    }

    state
}

/// Prove that `main`'s entry-argument binding is already present at
/// coordinate `[]` (the sequence root) of the window's *initial* storage
/// state, so a window that opens after the program start can still tie its
/// execution to the public manifest (see `checks::entrypoint` in the
/// transition guest).
///
/// `None` when the binding is not in the initial state — which is the normal
/// case for a window covering the start of the trace, where the
/// `ProgramStart` step is replayed inside the window and authorizes itself.
/// The guest decides which of those two it is; this only supplies the witness
/// when one exists.
fn build_entrypoint_membership_witness(state: &ProofStorageState) -> Option<StorageReadWitness> {
    let coordinates = CfsCoordinates(vec![]);
    let index_witness = state.coordinate_index.membership_proof(&coordinates)?;
    let log_witness =
        build_storage_log_witness(&state.append_entries, index_witness.value.log_position);
    Some(StorageReadWitness {
        entry: StorageEntry {
            coordinates,
            object_commitment: index_witness.value.object_commitment.clone(),
        },
        log_witness,
        index_witness,
    })
}

fn build_storage_selection_witnesses(
    step_record: &StepRecord,
    input_source_witness: Option<&FnInput>,
    trace_recorder: &TraceRecorder,
) -> BTreeMap<String, SelectionWitness> {
    let Some(input_source_witness) = input_source_witness else {
        return BTreeMap::new();
    };

    input_source_witness
        .storage()
        .iter()
        .filter_map(|(binding_name, storage)| {
            if storage.selection.selected_len == 0 {
                return None;
            }
            let reference =
                StorageRef::new(storage.coordinates.clone(), storage.commitment.clone());
            let witness = trace_recorder
                // The recorded commitment says which view of the node it
                // committed to. This process did not produce the trace, so
                // it cannot infer that from the payload — it has none yet.
                .storage_selection_witness(
                    &reference,
                    &storage.selector,
                    storage.selection.payload_kind,
                )
                .unwrap_or_else(|error| {
                    panic!(
                        "Failed to build storage selection witness for '{}': {}",
                        binding_name, error
                    )
                });

            // A binding the step only forwards as a reference ships its root
            // instead of its value. The payload is read once here, on the
            // host, and never reaches the guest — which is the whole point:
            // `prompt-prepare` forwards a 106 MB `List<MergeBucket>` into a
            // sequence whose recorded input is 145 bytes, and rebuilding that
            // list's Merkle tree exhausted the guest heap.
            //
            // The same `binding_requires_payload` the guest enforces, so the
            // host cannot ship a shape the guest will reject — and cannot
            // choose a weaker one either.
            let witness = if raster_core::trace::binding_requires_payload(
                &step_record.kind,
                binding_name.as_str(),
                input_source_witness.storage(),
            ) {
                witness
            } else {
                witness.clone().into_reference().unwrap_or_else(|| {
                    panic!(
                        "Failed to reduce forwarded binding '{}' to a selection reference",
                        binding_name
                    )
                })
            };

            Some((binding_name.clone(), witness))
        })
        .collect()
}

/// Assemble the canonical `program.bin` frame for the fraud-proof guest by
/// reassembling the `ProgramDefinition` from source (CFS + compiled tile
/// registry + `Raster.toml`/synthesized manifest) and verifying it against
/// `Raster.lock` if present (the stale-lock drift check). Every window step
/// uses this same frame, so `program_commitment` continuity holds by
/// construction. See `docs/proposals/program-identity.md`.
fn build_program_frame(cfs: &ControlFlowSchema, replayer: &Replayer) -> Vec<u8> {
    super::program::reassemble_and_verify(replayer.project(), cfs, replayer)
        .unwrap_or_else(|e| panic!("Failed to assemble program definition: {e}"))
        .canonical_bytes()
}

pub fn prove(
    fraud_evidence: FraudEvidence,
    trace: &Trace,
    cfs: &ControlFlowSchema,
    trace_recorder: &TraceRecorder,
    replayer: &Replayer,
    input_manifest: Option<&str>,
    trace_commitment: &TraceCommitment,
) -> risc0_zkvm::Receipt {
    let mode = ExecutionMode::prove_and_verify();
    let FraudEvidence {
        window: fraud_window,
        input_sources_witnesses,
    } = fraud_evidence;
    let mut replayed_results: HashMap<StepRecord, ReplayResult> = HashMap::new();
    let mut recorded_step_io: HashMap<StepRecord, StepIo> = HashMap::new();
    let window_start_index = fraud_window
        .items
        .first()
        .and_then(|first_item| {
            trace
                .iter()
                .position(|step_record| step_record == first_item)
        })
        .unwrap_or(trace.len());
    let mut current_storage_state =
        storage_state_from_prefix(&trace[..window_start_index], trace_recorder);
    let initial_storage_state = current_storage_state.clone();

    // The recur-progress stack the window opens with, taken from the last step
    // *before* it. A window opening at index 0 has no predecessor and starts
    // from the canonical empty stack, which `step_transitions` reads as `None`.
    //
    // Reconstructed here, on the same prefix the store is rebuilt from, and
    // read off the recorder rather than re-derived: a second implementation of
    // the advance rules would fail as a commitment mismatch, indistinguishable
    // from the missing-seed bug this replaces. See
    // `window-seed-reconstruction.md` §2.
    let window_start_recur_progress = window_start_index
        .checked_sub(1)
        .and_then(|previous| trace_recorder.recur_progress_after(trace[previous].exec_index));

    for step_record in &fraud_window.items {
        let step_witness = trace_recorder
            .step_witness_at(step_record.coordinates())
            .unwrap_or_else(|| {
                panic!(
                    "Missing recorded I/O for fraud window step at coordinates {:?}",
                    step_record.coordinates()
                )
            });
        // The witness store is keyed by coordinates alone, and a sequence's
        // `SequenceStart` and `SequenceEnd` share coordinates — the End does
        // `get_mut` on the Start's entry and fills in `output_data` only
        // (`recorder.rs`, which already guards the same sharing for
        // `storage_write`). So both input fields still hold the *Start's*
        // values when this step is the End. Take each only when this step's
        // record actually declares the matching commitment: the guest refuses
        // an input source witness on a step that commits to none, and never
        // reads an input witness there.
        let input_witness = step_record
            .input_commitment()
            .and_then(|_| step_witness.input_data());
        let output_witness = step_witness.output_data();
        let input_source_witness = step_record
            .input_source_commitment()
            .and_then(|_| step_witness.input_source_witness());
        let sequence_scope_witness =
            step_record
                .coordinates()
                .try_parent()
                .and_then(|(parent_coordinates, _)| {
                    trace_recorder
                        .step_witness_at(&parent_coordinates)
                        .and_then(|witness| witness.input_source_witness())
                });
        let storage_selection_witnesses = build_storage_selection_witnesses(
            step_record,
            input_source_witness.as_ref(),
            trace_recorder,
        );
        let draft_transition_witness = step_witness.draft_transition_witness();
        let before_state = current_storage_state.clone();
        let mut storage_read_witnesses = Vec::new();
        // Only a step that declares storage roots can carry a storage witness:
        // the guest has nothing to verify membership *against* otherwise, and
        // refuses one outright ("Only execution steps may carry storage
        // witnesses"). A `SequenceStart` legitimately has storage bindings —
        // that is how a sequence receives its arguments — so without this gate
        // the reads get built for it and the proof is rejected.
        //
        // The write side already guards the same way one block below; this is
        // the read half of it. A sequence step is transparent to the storage
        // chain: the guest forwards its carried frontier unchanged, so there is
        // no root here for a membership proof to be relative to. The bindings
        // are proved where they are consumed, against that step's own root.
        // The exception is a recur site `Start`: it claims read-only roots
        // (`StepKind::SequenceStart::storage`), so the gate below admits it.
        //
        // Gated here rather than on the `StorageWitness` that wraps the loop so
        // the proofs are never built: each binding costs a coordinate-index
        // membership proof and a log witness whose only consumer is that
        // discarded witness.
        let step_reads_storage = step_record.storage_roots().is_some();
        if let Some(input_source_witness_ref) =
            input_source_witness.as_ref().filter(|_| step_reads_storage)
        {
            for storage_meta in input_source_witness_ref.storage().values() {
                let index_witness = before_state
                    .coordinate_index
                    .membership_proof(&storage_meta.coordinates)
                    .unwrap_or_else(|| {
                        panic!(
                            "Missing coordinate-index witness for storage input at {:?}",
                            storage_meta.coordinates
                        )
                    });
                let entry = StorageEntry {
                    coordinates: storage_meta.coordinates.clone(),
                    object_commitment: storage_meta.commitment.clone(),
                };
                let log_witness = build_storage_log_witness(
                    &before_state.append_entries,
                    index_witness.value.log_position,
                );
                storage_read_witnesses.push(StorageReadWitness {
                    entry,
                    log_witness,
                    index_witness,
                });
            }
        }
        let mut storage_write_witness = None;
        // Only a storage-appending step owns the write recorded at its
        // coordinate. `ProgramEnd` shares coordinates `[]` (and thus a
        // witness-store entry) with `ProgramStart`, so without this gate it
        // would re-apply `ProgramStart`'s entry-object write.
        if step_record.appends_to_storage() {
            if let Some(storage_write) = step_witness.storage_write() {
                let entry = storage_write.entry.clone();
                let index_non_membership_witness = before_state
                    .coordinate_index
                    .non_membership_proof(&entry.coordinates);
                apply_storage_write_to_state(&mut current_storage_state, &storage_write);
                let index_membership_witness = current_storage_state
                    .coordinate_index
                    .membership_proof(&entry.coordinates)
                    .expect("Missing coordinate-index membership proof after write");
                storage_write_witness = Some(StorageWriteWitness {
                    entry,
                    index_non_membership_witness,
                    index_membership_witness,
                });
            }
        }
        let storage_witness =
            if storage_read_witnesses.is_empty() && storage_write_witness.is_none() {
                None
            } else {
                Some(StorageWitness {
                    reads: storage_read_witnesses,
                    write: storage_write_witness,
                })
            };

        // A `ProgramEnd` step reads its output object from the current storage
        // state (which already contains it) and proves the selection that
        // narrows to the returned value — the same read + selection machinery
        // tile inputs use. The witnesses come straight from the recorded
        // output binding, not the shared `[]` witness-store entry.
        let (program_output_read_witness, program_output_selection_witness) = match &step_record
            .kind
        {
            StepKind::ProgramEnd(program_end) => match &program_end.output {
                Some(output) => {
                    let index_witness = before_state
                        .coordinate_index
                        .membership_proof(&output.coordinates)
                        .unwrap_or_else(|| {
                            panic!(
                                "Missing coordinate-index witness for program output at {:?}",
                                output.coordinates
                            )
                        });
                    let entry = StorageEntry {
                        coordinates: output.coordinates.clone(),
                        object_commitment: output.commitment.clone(),
                    };
                    let log_witness = build_storage_log_witness(
                        &before_state.append_entries,
                        index_witness.value.log_position,
                    );
                    let read = StorageReadWitness {
                        entry,
                        log_witness,
                        index_witness,
                    };
                    let selection = if output.selection.selected_len > 0 {
                        let reference =
                            StorageRef::new(output.coordinates.clone(), output.commitment.clone());
                        Some(
                            trace_recorder
                                .storage_selection_witness(
                                    &reference,
                                    &output.selector,
                                    output.selection.payload_kind,
                                )
                                .unwrap_or_else(|error| {
                                    panic!(
                                        "Failed to build program output selection witness: {}",
                                        error
                                    )
                                }),
                        )
                    } else {
                        None
                    };
                    (Some(read), selection)
                }
                None => (None, None),
            },
            _ => (None, None),
        };

        // A recur-sequence iteration's `End` that carries state reads the
        // object it returned as that state, against the current storage state
        // (D5b). Its output witness is that binding.
        let returned_state_read_witness = match (&step_record.kind, &step_record.recur_state) {
            (StepKind::SequenceEnd { .. }, Some(_)) => output_witness
                .as_ref()
                .and_then(|bytes| {
                    raster_core::postcard::from_bytes::<Option<raster_core::trace::StorageData>>(
                        bytes,
                    )
                    .ok()
                    .flatten()
                })
                .map(|returned| {
                    let index_witness = before_state
                        .coordinate_index
                        .membership_proof(&returned.coordinates)
                        .unwrap_or_else(|| {
                            panic!(
                                "Missing coordinate-index witness for returned state at {:?}",
                                returned.coordinates
                            )
                        });
                    let log_witness = build_storage_log_witness(
                        &before_state.append_entries,
                        index_witness.value.log_position,
                    );
                    StorageReadWitness {
                        entry: StorageEntry {
                            coordinates: returned.coordinates,
                            object_commitment: returned.commitment,
                        },
                        log_witness,
                        index_witness,
                    }
                }),
            _ => None,
        };

        recorded_step_io.insert(
            step_record.clone(),
            StepIo {
                input_witness: input_witness.clone(),
                output_witness,
                input_source_witness,
                sequence_scope_witness,
                storage_selection_witnesses,
                storage_witness,
                draft_transition_witness,
                program_output_read_witness,
                program_output_selection_witness,
                returned_state_read_witness,
            },
        );

        // Only a tile is replayed: it is the only step whose output is
        // verified by re-running it (see `StepRecord::requires_replay_proof`).
        if let StepKind::Exec(ExecStep {
            target: ExecTarget::Tile(tile_id),
            ..
        }) = &step_record.kind
        {
            let replay_input = input_witness.unwrap_or_default();
            match replayer.replay(tile_id, replay_input.as_slice(), mode) {
                Ok(replay_result) => {
                    replayed_results.insert(step_record.clone(), replay_result);
                }
                Err(e) => {
                    panic!("Failed to replay proof tile: {}", e);
                }
            }
        }
    }

    let manifested_inputs = raster_core::authorization::ManifestedInputs {
        manifest_bytes: std::fs::read(input_manifest.expect("manifest required"))
            .expect("read authorization manifest"),
    };

    let (authorization_receipt, authorization_journal) =
        authorize_external_inputs(&manifested_inputs);

    // Only meaningful when `main` declares entry arguments at all; without a
    // declaration the guest requires no witness (and rejects one: with no
    // entry arguments, `ProgramStart` writes nothing at the root `[]`).
    let entrypoint_membership_witness = CfsCursor::new(cfs.clone())
        .main_entrypoint_names()
        .is_some()
        .then(|| build_entrypoint_membership_witness(&initial_storage_state))
        .flatten();

    if let Some(frontier) = SerializableFrontier::from_bytes(&fraud_window.frontier) {
        println!();
        println!("Replaying transition frontier with transition guest...");

        let program_frame = build_program_frame(cfs, replayer);

        // Bind the window to the commitment it refutes: the window's start is
        // the initial frontier's position (seed + one leaf per pre-window
        // step), and the slice witness proves the window fingerprint occurs
        // in the commitment there. The guest re-derives and checks both.
        let commitment_header = trace_commitment.header();
        let window_start =
            usize::try_from(frontier.position).expect("window start position overflows usize");
        let fingerprint_slice = trace_commitment
            .fingerprint_slice_witness(window_start, fraud_window.fingerprint.len());

        let Some(receipt) = step_transitions(
            &frontier,
            &initial_storage_state.frontier,
            &initial_storage_state.coordinate_index.root(),
            &fraud_window.items,
            fraud_window.fingerprint,
            &commitment_header,
            &fingerprint_slice,
            &program_frame,
            &input_sources_witnesses,
            &recorded_step_io,
            &replayed_results,
            &authorization_journal,
            &authorization_receipt,
            entrypoint_membership_witness.as_ref(),
            window_start_recur_progress,
            // Lets a divergence the packed fingerprint is blind to still be
            // proven — the tail's roots are revealed in full.
            &trace_commitment.revealed_tail_roots,
        ) else {
            panic!("Failed to generate fraud proof");
        };

        return receipt;
    }

    panic!("Failed to generate fraud proof");
}
