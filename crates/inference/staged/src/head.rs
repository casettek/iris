//! Finalization over verified Raster pages, without an owned full-head matrix.
use crate::{
    cache::{MaterializationCache, MaterializationCacheKey},
    verified_weights::{fields, PageRegion, VerifiedSelections},
};
use anyhow::{ensure, Context, Result};
use inference_kernels::weight_budget::RetainedWeightBytes;
use prefill_finalize::input::{
    dot, rms_norm, softcap, unpack_i32s, ActivationSequence, FinalHeadParams, LogitEntry,
    PrefillLogits,
};
use raster::{BytesPage, List};
use raster_runtime::EntryArgumentsBinding;
use rayon::prelude::*;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

pub(crate) const BATCH_PAGES: usize = 32;
pub(crate) struct StaticHead {
    params: FinalHeadParams,
    region: PageRegion,
}
impl RetainedWeightBytes for StaticHead {
    fn retained_weight_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.params.norm_weights.allocated_bytes()
    }
}

/// The source belongs to this load/run invocation, never to the run weight cache.
/// Selection stays on the invoking thread; only verified page values enter Rayon.
pub(crate) struct PagedHead {
    pub(crate) metadata: Arc<StaticHead>,
    source: VerifiedSelections,
}

impl StaticHead {
    fn load(source: &VerifiedSelections) -> Result<Self> {
        let params: FinalHeadParams = source.select(fields(&["head", "params"]))?;
        ensure!(
            params.hidden_size > 0,
            "output head requires a non-zero hidden_size"
        );
        ensure!(
            params.norm_eps >= 0,
            "rms-norm epsilon must be non-negative"
        );
        ensure!(
            params.softcap >= 0,
            "final logit softcap must be non-negative (0 disables it)"
        );
        ensure!(
            params.norm_weights.len() as u64 == u64::from(params.hidden_size) * 4,
            "final norm weights width mismatch"
        );
        let region = PageRegion::load(source, &["head", "projection"])?;
        Ok(Self { params, region })
    }
}

pub(crate) fn load(
    binding: EntryArgumentsBinding,
    input: Option<&str>,
    manifest: Option<&str>,
    cache: Option<&MaterializationCache>,
    key: Option<MaterializationCacheKey>,
) -> Result<PagedHead> {
    let source = VerifiedSelections::new(binding, input, manifest)?;
    let metadata = match cache {
        Some(cache) => cache.get_or_try_insert_with(key, || StaticHead::load(&source))?,
        None => Arc::new(StaticHead::load(&source)?),
    };
    Ok(PagedHead { metadata, source })
}

#[derive(Default, Debug)]
pub(crate) struct RunStats {
    pub pages: u64,
    pub peak_batch_pages: usize,
    pub peak_batch_payload_bytes: usize,
    pub load_verify: Duration,
    pub arithmetic: Duration,
}

pub(crate) fn run(activations: &ActivationSequence, head: &PagedHead) -> Result<PrefillLogits> {
    run_measured(activations, head).map(|(output, _)| output)
}

pub(crate) fn run_measured(
    activations: &ActivationSequence,
    head: &PagedHead,
) -> Result<(PrefillLogits, RunStats)> {
    let mut stats = RunStats::default();
    let started = Instant::now();
    let params = &head.metadata.params;
    let position = activations
        .rows
        .iter()
        .last()
        .context("prefill produced no activation rows; there is no final position to score")?;
    let mut normalized = unpack_i32s(&position.values).map_err(anyhow::Error::msg)?;
    ensure!(
        normalized.len() == params.hidden_size as usize,
        "final position width mismatch"
    );
    rms_norm(
        &mut normalized,
        &unpack_i32s(&params.norm_weights).map_err(anyhow::Error::msg)?,
        params.norm_eps,
    )
    .map_err(anyhow::Error::msg)?;
    stats.arithmetic += started.elapsed();
    let parallel = std::env::var("RASTER_PARALLELISM")
        .map(|v| !v.eq_ignore_ascii_case("off"))
        .unwrap_or(true);
    let region = &head.metadata.region;
    let mut logits = Vec::new();
    let mut next = 0;
    while next < region.page_count {
        let started = Instant::now();
        let end = next
            .saturating_add(BATCH_PAGES as u64)
            .min(region.page_count);
        let pages = (next..end)
            .map(|index| region.page(&head.source, &["head", "projection"], index))
            .collect::<Result<Vec<_>>>()?;
        stats.pages += pages.len() as u64;
        stats.peak_batch_pages = stats.peak_batch_pages.max(pages.len());
        stats.peak_batch_payload_bytes = stats
            .peak_batch_payload_bytes
            .max(pages.iter().map(BytesPage::len).sum());
        stats.load_verify += started.elapsed();
        let started = Instant::now();
        let score = |page: &BytesPage| -> Result<Vec<LogitEntry>> {
            let weights = unpack_i32s(page).map_err(anyhow::Error::msg)?;
            let hidden = params.hidden_size as usize;
            let stride = u64::from(params.hidden_size) * 4;
            ensure!(
                weights.len() % hidden == 0,
                "projection page is not a whole number of hidden-width rows"
            );
            ensure!(
                page.offset() % stride == 0,
                "projection page offset is not row-aligned"
            );
            let first_id = (page.offset() / stride) as u32;
            weights
                .chunks_exact(hidden)
                .enumerate()
                .map(|(local, row)| {
                    let value = dot(&normalized, row).map_err(anyhow::Error::msg)?;
                    Ok(LogitEntry {
                        token_id: first_id + local as u32,
                        value: softcap(value, params.softcap),
                    })
                })
                .collect()
        };
        // Indexed parallel collection preserves page order. Dot products keep
        // Raster's left-to-right saturating MAC order within each row.
        let scored = if parallel && rayon::current_num_threads() > 1 {
            pages.par_iter().map(score).collect::<Result<Vec<_>>>()?
        } else {
            pages.iter().map(score).collect::<Result<Vec<_>>>()?
        };
        for page in scored {
            logits.extend(page);
        }
        stats.arithmetic += started.elapsed();
        next = end;
        // `pages` and decoded matrices die here, before the next batch is read.
    }
    Ok((
        PrefillLogits {
            decode_position: activations.start_position + activations.rows.len() as u32,
            logits: List::from(logits),
            errors: List::new(),
        },
        stats,
    ))
}
