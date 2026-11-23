//! What a submission put on the data-availability layer, measured there.

use std::path::Path;

use anyhow::{ensure, Context, Result};
use inference_artifacts::availability::{Blob, BlobEntry};
use inference_artifacts::read_json;

use crate::posting::open_layer;

/// What a submission put on the layer, measured there: each blob it names is
/// read back, and its size is split among what it holds.
///
/// `bytes` is everything posted: the whole of every blob. Each blob's parts
/// are sized as the blob encodes them, and `framing_bytes` is what is left:
/// the magic, the submitter and the lengths. A run's inputs are split into
/// the text, the token marks, the claimed ids, the documents' bytes and the
/// rest (the run's settings, the forms, and a source's attestations, recipe
/// and signer's key), and a claim's checkpoints into the list, the
/// checkpoints and the input manifests.
pub fn measure_posting(submission_path: &Path, da_dir: Option<&Path>) -> Result<serde_json::Value> {
    #[derive(serde::Deserialize)]
    struct Posted {
        #[serde(default)]
        availability: Vec<inference_artifacts::availability::PostedBlob>,
    }
    let submission: Posted = read_json(submission_path)?;
    ensure!(
        !submission.availability.is_empty(),
        "{} names no posted data: nothing was posted",
        submission_path.display()
    );
    let layer = open_layer(da_dir)?;
    let encoded =
        |entry: &BlobEntry| -> Result<u64> { Ok(postcard::to_allocvec(entry)?.len() as u64) };
    let mut blobs = Vec::new();
    let mut total = 0u64;
    for (index, posted) in submission.availability.iter().enumerate() {
        let bytes = layer
            .read(&posted.pointer)
            .with_context(|| format!("blob {index} could not be read from the layer"))?;
        ensure!(
            bytes.len() as u64 == posted.length,
            "blob {index} is {} bytes on the layer, and the submission says {}",
            bytes.len(),
            posted.length
        );
        let blob =
            Blob::decode(&bytes).map_err(|fault| anyhow::anyhow!("blob {index}: {fault}"))?;
        let mut parts = Vec::new();
        let mut held = 0u64;
        for entry in &blob.entries {
            let size = encoded(entry)?;
            held += size;
            parts.push(match entry {
                BlobEntry::Run(run) => {
                    let sized = |value: &dyn Fn() -> postcard::Result<Vec<u8>>| -> Result<u64> {
                        Ok(value()?.len() as u64)
                    };
                    let (mut text, mut marks, mut claimed) = (0u64, 0u64, 0u64);
                    for inference in &run.inferences {
                        text += sized(&|| postcard::to_allocvec(&inference.text))?;
                        marks += sized(&|| postcard::to_allocvec(&inference.marks))?;
                        claimed += sized(&|| postcard::to_allocvec(&inference.claimed))?;
                    }
                    let (documents, bodies) = match &run.source {
                        Some(source) => (
                            source.documents.attestations.len(),
                            source
                                .documents
                                .chunks
                                .iter()
                                .map(|chunk| chunk.bytes.as_slice().len() as u64)
                                .sum(),
                        ),
                        None => (0, 0u64),
                    };
                    serde_json::json!({
                        "kind": "run_inputs",
                        "bytes": size,
                        "inferences": run.inferences.len(),
                        "values": run.values().len(),
                        "text_bytes": text,
                        "marks_bytes": marks,
                        "claimed_bytes": claimed,
                        "documents": documents,
                        "document_bytes": bodies,
                        "other_bytes": size.saturating_sub(text + marks + claimed + bodies),
                    })
                }
                BlobEntry::Checkpoints(package) => serde_json::json!({
                    "kind": "checkpoints",
                    "bytes": size,
                    "list_bytes": package.list.len(),
                    "trace_bytes": package.trace.len(),
                    "input_manifests": package.input_manifests.len(),
                    "input_manifest_bytes":
                        package.input_manifests.iter().map(Vec::len).sum::<usize>(),
                }),
                BlobEntry::TraceBody { bytes: body } => serde_json::json!({
                    "kind": "trace_body",
                    "bytes": size,
                    "trace_bytes": body.len(),
                }),
            });
        }
        total += posted.length;
        blobs.push(serde_json::json!({
            "height": posted.pointer.height,
            "commitment": posted.pointer.commitment,
            "bytes": posted.length,
            "framing_bytes": posted.length - held,
            "parts": parts,
        }));
    }
    Ok(serde_json::json!({ "bytes": total, "blobs": blobs }))
}
