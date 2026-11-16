//! `prompt-source`: sequences, one stage per run that has a source.
//!
//! ```text
//!   documents.chunks ──recur──▶ each document's hash and length, chained
//!   documents.attestations ──recur──▶ signatures checked; facts, names and bodies chained
//!   recipe.base ──recur──▶ text hash
//!   recipe.documents ──recur──▶ names chained, held to the attestations'
//!   recipe.fields ──recur seq──▶ [ documents.chunks ──recur──▶ value ] ──▶ text hash
//!   recipe.tail ──▶ SourceRecord
//! ```

use prompt_source::input::*;
use prompt_source::*;
use raster::prelude::*;

/// One field: scan the documents' chunks for its value, stopping where it
/// ends, and add its lead and value to the text.
#[sequence(kind = recur)]
fn take_field(
    input: RecurSequenceInput<Field>,
    state: RecurSequenceState<SourceFold>,
    chunks: List<DocumentChunk>,
) -> RecurSequenceState<SourceFold> {
    let field = into_ref!(input);
    let seed = call!(begin_scan, field.clone());
    let document = select!(u32, field.clone().document);
    let path = select!(String, field.clone().path);
    let scan = call_recur!(
        tile = scan_chunk,
        input = chunks,
        state = seed,
        args = (document, path)
    );
    let value = call!(finish_scan, scan);
    call!(fold_one_field, state, field, value)
}

/// Stage entrypoint. `signer` is the key attestations are checked against,
/// `recipe` how the prompt is made, and `documents` the run's documents with
/// the attestation over each.
#[sequence]
fn main(signer: SignerKey, recipe: Recipe, documents: SourceDocuments) -> SourceRecord {
    let chunks = select!(List<DocumentChunk>, documents.clone().chunks);
    let attestations = select!(List<Attestation>, documents.attestations);

    let no_bodies = call!(bodies_seed);
    let bodies = call_recur!(
        tile = fold_body,
        input = chunks.clone(),
        state = no_bodies,
        args = ()
    );
    let none_attested = call!(attested_seed);
    let signed = call_recur!(
        tile = fold_attested,
        input = attestations,
        state = none_attested,
        args = (signer.clone(),)
    );
    let attested = call!(hold_bodies, signed, bodies);

    let base = select!(List<String>, recipe.clone().base);
    let no_text = call!(text_seed);
    let based = call_recur!(tile = fold_base, input = base, state = no_text, args = ());
    let based = call!(end_base, based);
    let names = select!(List<DocumentName>, recipe.clone().documents);
    let named = call_recur!(tile = fold_name, input = names, state = based, args = ());
    let held = call!(hold_documents, named, attested.clone());

    let fields = select!(List<Field>, recipe.clone().fields);
    let text = call_recur_seq!(
        sequence = take_field,
        input = fields,
        state = held,
        args = (chunks,)
    );
    let tail = select!(String, recipe.tail);
    let record = call!(close, text, attested, signer, tail);
    raster::println!("prompt source → {:?}", record);
    record
}
