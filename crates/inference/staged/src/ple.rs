//! Selective PLE loading for the staged host only. Raster remains the oracle.
use std::{collections::BTreeMap, sync::Arc};

use anyhow::{bail, ensure, Context, Result};
use inference_kernels::{
    tensor::{matvec_from_source, Matrix},
    weight_budget::RetainedWeightBytes,
};
use prefill_prepare_aux::input::{
    add_sat, pack_i32s, rms_norm, scale_row, unpack_i32s, unpack_i32s_at, ActivationSequence,
    PleLayerInputs, PleLayerParams, PleRow, PAGE_SIZE,
};
use raster::{Bytes, BytesPage, List};
use raster_core::input::{SelectorPath, SelectorSegment};
use raster_runtime::EntryArgumentsBinding;
use rayon::prelude::*;
use crate::verified_weights::{fields, VerifiedSelections};

use crate::cache::{MaterializationCache, MaterializationCacheKey};

pub(crate) const LOADING_ENV: &str = "STAGED_INFER_PLE_LOADING";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Loading {
    Selective,
    Eager,
}

impl Loading {
    pub(crate) fn from_env() -> Result<Self> {
        match std::env::var(LOADING_ENV) {
            Err(std::env::VarError::NotPresent) => Ok(Self::Selective),
            Ok(value) => Self::parse(&value),
            Err(error) => Err(error).context(LOADING_ENV),
        }
    }
    fn parse(value: &str) -> Result<Self> {
        match value {
            "selective" => Ok(Self::Selective),
            "eager" => Ok(Self::Eager),
            _ => bail!("{LOADING_ENV} must be selective or eager"),
        }
    }
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Selective => "selective",
            Self::Eager => "eager",
        }
    }
}

/// A small, verified subset of a PLE artifact; no embedding table is retained.
pub(crate) struct StaticWeights {
    params: PleLayerParams,
    projection: Bytes<PAGE_SIZE>,
    embedding_byte_len: u64,
    embedding_page_size: u64,
}

impl RetainedWeightBytes for StaticWeights {
    fn retained_weight_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.params.norm_weights.allocated_bytes()
            + self.projection.pages().capacity() * std::mem::size_of::<BytesPage>()
            + self
                .projection
                .pages()
                .iter()
                .map(BytesPage::allocated_bytes)
                .sum::<usize>()
    }
}

pub(crate) struct SelectedWeights {
    weights: Arc<StaticWeights>,
    pages: BTreeMap<u64, BytesPage>,
}

fn embedding_page(index: u64) -> SelectorPath {
    let mut path = fields(&["layer", "embeddings", "pages"]);
    path.segments.push(SelectorSegment::Index(index));
    path
}

impl StaticWeights {
    fn load(source: &VerifiedSelections) -> Result<Self> {
        let params: PleLayerParams = source.select(fields(&["layer", "params"]))?;
        ensure!(
            params.hidden_size > 0 && params.ple_width > 0,
            "PLE layer requires non-zero hidden_size and ple_width"
        );
        ensure!(
            params.norm_weights.len() as u64 == u64::from(params.ple_width) * 4,
            "PLE norm weights width mismatch"
        );
        ensure!(
            params.norm_eps >= 0,
            "PLE rms-norm epsilon must be non-negative"
        );
        let embedding_page_size = source.select(fields(&["layer", "embeddings", "page_size"]))?;
        ensure!(
            embedding_page_size == PAGE_SIZE,
            "PLE embedding page size mismatch"
        );
        let embedding_byte_len = source.select(fields(&["layer", "embeddings", "byte_len"]))?;
        let projection: Bytes<PAGE_SIZE> = source.select(fields(&["layer", "projection"]))?;
        ensure!(
            projection.page_size() == PAGE_SIZE,
            "PLE projection page size mismatch"
        );
        let stride = u64::from(params.hidden_size) * 4;
        let expected = stride
            .checked_mul(u64::from(params.ple_width))
            .context("PLE projection shape overflow")?;
        ensure!(
            projection.byte_len() == expected,
            "PLE projection byte length mismatch"
        );
        ensure!(
            projection.pages().len() as u64 == expected.div_ceil(PAGE_SIZE),
            "PLE projection page count mismatch"
        );
        for (i, page) in projection.pages().iter().enumerate() {
            let offset = i as u64 * PAGE_SIZE;
            ensure!(
                page.index() == i as u64
                    && page.offset() == offset
                    && page.len() as u64 == PAGE_SIZE.min(expected - offset),
                "PLE projection page geometry mismatch"
            );
            // The authored Raster MAC accepts only whole, aligned matrix rows.
            ensure!(
                offset % stride == 0 && page.len() as u64 % stride == 0,
                "PLE projection page does not contain whole output rows"
            );
        }
        Ok(Self {
            params,
            projection,
            embedding_byte_len,
            embedding_page_size,
        })
    }
}

pub(crate) fn load(
    binding: EntryArgumentsBinding,
    input: Option<&str>,
    manifest: Option<&str>,
    embedded: &ActivationSequence,
    cache: Option<&MaterializationCache>,
    key: Option<MaterializationCacheKey>,
) -> Result<SelectedWeights> {
    let source = VerifiedSelections::new(binding, input, manifest)?;
    let weights = match cache {
        Some(cache) => cache.get_or_try_insert_with(key, || StaticWeights::load(&source))?,
        None => Arc::new(StaticWeights::load(&source)?),
    };
    let mut pages = BTreeMap::new();
    let width_bytes = u64::from(weights.params.ple_width) * 4;
    for row in embedded.rows.iter() {
        let offset = u64::from(row.token_id)
            .checked_mul(width_bytes)
            .context("PLE row offset overflow")?;
        ensure!(
            offset
                .checked_add(width_bytes)
                .is_some_and(|end| end <= weights.embedding_byte_len),
            "PLE row missing for token {}",
            row.token_id
        );
        let index = offset / weights.embedding_page_size;
        if let std::collections::btree_map::Entry::Vacant(entry) = pages.entry(index) {
            let page: BytesPage = source.select(embedding_page(index))?;
            ensure!(
                page.index() == index
                    && page.offset() == index * weights.embedding_page_size
                    && page.len() as u64
                        == weights
                            .embedding_page_size
                            .min(weights.embedding_byte_len - page.offset()),
                "PLE embedding page geometry mismatch"
            );
            entry.insert(page);
        }
        // Do not stitch a row spanning pages: Raster selects exactly one page.
        unpack_i32s_at(&pages[&index], offset, weights.params.ple_width)
            .map_err(anyhow::Error::msg)?;
    }
    Ok(SelectedWeights { weights, pages })
}

pub(crate) fn run(
    embedded: &ActivationSequence,
    selected: &SelectedWeights,
) -> Result<PleLayerInputs> {
    let weights = &selected.weights;
    let params = &weights.params;
    let projection = Matrix::from_region(
        "projection",
        &weights.projection,
        params.ple_width as usize,
        params.hidden_size as usize,
    )?;
    let norm = unpack_i32s(&params.norm_weights).map_err(anyhow::Error::msg)?;
    let prepare = |row: &prefill_prepare_aux::input::ActivationRow| -> Result<PleRow> {
        let activation = unpack_i32s(&row.values).map_err(anyhow::Error::msg)?;
        let mut projected = matvec_from_source(&projection, &activation)?;
        scale_row(&mut projected, params.projection_scalar);
        rms_norm(&mut projected, &norm, params.norm_eps).map_err(anyhow::Error::msg)?;
        let offset = u64::from(row.token_id) * u64::from(params.ple_width) * 4;
        let page = selected
            .pages
            .get(&(offset / weights.embedding_page_size))
            .context("PLE selected page missing")?;
        let mut values =
            unpack_i32s_at(page, offset, params.ple_width).map_err(anyhow::Error::msg)?;
        scale_row(&mut values, params.embedding_scale);
        for (value, projection) in values.iter_mut().zip(projected) {
            *value = add_sat(*value, projection);
        }
        scale_row(&mut values, params.input_scale);
        Ok(PleRow {
            values: pack_i32s(&values),
        })
    };
    let parallel = std::env::var("RASTER_PARALLELISM")
        .map(|v| !v.eq_ignore_ascii_case("off"))
        .unwrap_or(true);
    let rows = if parallel && rayon::current_num_threads() > 1 {
        embedded
            .rows
            .par_iter()
            .map(prepare)
            .collect::<Result<Vec<_>>>()?
    } else {
        embedded
            .rows
            .iter()
            .map(prepare)
            .collect::<Result<Vec<_>>>()?
    };
    Ok(PleLayerInputs {
        layer_idx: params.layer_idx,
        rows: List::from(rows),
        errors: List::new(),
    })
}

#[cfg(test)]
mod tests;
