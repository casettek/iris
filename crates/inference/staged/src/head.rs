//! Scoring over verified Raster pages, without an owned full-head matrix.
use crate::{
    cache::{MaterializationCache, MaterializationCacheKey},
    verified_weights::{fields, PageRegion, VerifiedSelections},
};
use anyhow::{ensure, Context, Result};
use inference_kernels::kernels::sequence_check::{score_row, Argmax, Selection};
use inference_kernels::weight_budget::RetainedWeightBytes;
use raster::BytesPage;
use raster_runtime::EntryArgumentsBinding;
use rayon::prelude::*;
use sequence_check::input::FinalHeadParams;
use std::sync::Arc;

pub(crate) const BATCH_PAGES: usize = 32;
pub(crate) struct StaticHead {
    params: FinalHeadParams,
    region: PageRegion,
}
impl RetainedWeightBytes for StaticHead {
    fn retained_weight_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.params.norm_weights.len()
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
        let region = PageRegion::load(
            source,
            &["head", "projection"],
            crate::verified_weights::Q8_HEAD_PAGE_SIZE,
        )?;
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

/// What one sweep of the head read.
#[derive(Default, Debug)]
pub(crate) struct RunStats {
    pub pages: u64,
    pub peak_batch_pages: usize,
    pub peak_batch_payload_bytes: usize,
}

impl PagedHead {
    pub(crate) fn params(&self) -> &FinalHeadParams {
        &self.metadata.params
    }
}

/// Greedy selection for every normalised position in one sweep of the head.
///
/// Each page is read and verified once and scored against all positions.
/// Every position folds its pages in page order and its rows in token-id
/// order, keeping only strictly greater scores, so each selection is the one
/// direct generation makes for that row.
pub(crate) fn select_positions(positions: &[&[i32]], head: &PagedHead) -> Result<Vec<Selection>> {
    select_positions_measured(positions, head).map(|(selections, _)| selections)
}

pub(crate) fn select_positions_measured(
    positions: &[&[i32]],
    head: &PagedHead,
) -> Result<(Vec<Selection>, RunStats)> {
    let mut stats = RunStats::default();
    let params = &head.metadata.params;
    let hidden = params.hidden_size as usize;
    let stride = det_num::q8::row_bytes(hidden).context("head width is not whole Q8_0 blocks")?;
    for position in positions {
        ensure!(
            position.len() == hidden,
            "normalised position width mismatch"
        );
    }
    let parallel = std::env::var("RASTER_PARALLELISM")
        .map(|v| !v.eq_ignore_ascii_case("off"))
        .unwrap_or(true);
    let region = &head.metadata.region;
    let mut argmax = vec![Argmax::default(); positions.len()];
    let mut next = 0;
    while next < region.page_count {
        let end = next
            .saturating_add(BATCH_PAGES as u64)
            .min(region.page_count);
        let pages = (next..end)
            .map(|index| {
                let page = region.page(&head.source, &["head", "projection"], index)?;
                ensure!(
                    page.len() % stride == 0,
                    "projection page is not a whole number of hidden-width rows"
                );
                ensure!(
                    page.offset() % stride as u64 == 0,
                    "projection page offset is not row-aligned"
                );
                Ok(((page.offset() / stride as u64) as u32, page))
            })
            .collect::<Result<Vec<_>>>()?;
        stats.pages += pages.len() as u64;
        stats.peak_batch_pages = stats.peak_batch_pages.max(pages.len());
        stats.peak_batch_payload_bytes = stats
            .peak_batch_payload_bytes
            .max(pages.iter().map(|(_, page)| BytesPage::len(page)).sum());
        let fold = |(position, best): (&&[i32], &mut Argmax)| -> Result<()> {
            for (first_id, page) in &pages {
                for (local, row) in page.as_slice().chunks_exact(stride).enumerate() {
                    best.observe(
                        first_id + local as u32,
                        score_row(position, row, params.softcap)?,
                    );
                }
            }
            Ok(())
        };
        if parallel && rayon::current_num_threads() > 1 {
            positions
                .par_iter()
                .zip(argmax.par_iter_mut())
                .try_for_each(fold)?;
        } else {
            positions.iter().zip(argmax.iter_mut()).try_for_each(fold)?;
        }
        next = end;
    }
    let selections = argmax
        .into_iter()
        .map(|best| {
            best.finish()
                .ok_or_else(|| anyhow::anyhow!("the head scored no token"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((selections, stats))
}
