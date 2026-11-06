//! Select only the committed main embedding pages touched by this invocation.
use crate::{
    cache::{MaterializationCache, MaterializationCacheKey},
    verified_weights::{fields, PageRegion, VerifiedSelections},
};
use anyhow::{Context, Result};
use det_num::{ops::mul_sat, Act};
use inference_kernels::weight_budget::RetainedWeightBytes;
use raster::BytesPage;
use raster_runtime::EntryArgumentsBinding;
use std::{collections::BTreeMap, sync::Arc};

pub(crate) struct StaticEmbedding {
    hidden_size: u32,
    scale: i32,
    region: PageRegion,
}

impl RetainedWeightBytes for StaticEmbedding {
    fn retained_weight_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

pub(crate) struct SelectedEmbedding {
    pub(crate) metadata: Arc<StaticEmbedding>,
    pub(crate) pages: BTreeMap<u64, BytesPage>,
}

impl StaticEmbedding {
    fn row_offset(&self, token: u32) -> u64 {
        input_embedding::row_byte_offset(token, self.hidden_size, self.region.byte_len)
    }

    fn load(source: &VerifiedSelections) -> Result<Self> {
        Ok(Self {
            hidden_size: source.select(fields(&["embedding", "hidden_size"]))?,
            scale: source.select(fields(&["embedding", "embedding_scale"]))?,
            region: PageRegion::load(
                source,
                &["embedding", "values"],
                crate::verified_weights::Q8_PAGE_SIZE,
            )?,
        })
    }
}

pub(crate) fn load(
    binding: EntryArgumentsBinding,
    input: Option<&str>,
    manifest: Option<&str>,
    tokens: impl IntoIterator<Item = u32>,
    cache: Option<&MaterializationCache>,
    key: Option<MaterializationCacheKey>,
) -> Result<SelectedEmbedding> {
    let source = VerifiedSelections::new(binding, input, manifest)?;
    let metadata = match cache {
        Some(cache) => cache.get_or_try_insert_with(key, || StaticEmbedding::load(&source))?,
        None => Arc::new(StaticEmbedding::load(&source)?),
    };
    let mut pages = BTreeMap::new();
    for token in tokens {
        let offset = metadata.row_offset(token);
        let index = offset / metadata.region.page_size;
        if let std::collections::btree_map::Entry::Vacant(entry) = pages.entry(index) {
            entry.insert(
                metadata
                    .region
                    .page(&source, &["embedding", "values"], index)?,
            );
        }
        // Raster selects one page per token, including for zero-width rows.
        input_embedding::input::q8_row_at(&pages[&index], offset, metadata.hidden_size)
            .map_err(anyhow::Error::msg)?;
    }
    Ok(SelectedEmbedding { metadata, pages })
}

impl SelectedEmbedding {
    pub(crate) fn row(&self, token: u32) -> Result<Vec<i32>> {
        let offset = self.metadata.row_offset(token);
        let page = self
            .pages
            .get(&(offset / self.metadata.region.page_size))
            .context("selected embedding page missing")?;
        let values = input_embedding::input::q8_row_at(page, offset, self.metadata.hidden_size)
            .map_err(anyhow::Error::msg)?;
        let scale = Act::from_bits(self.metadata.scale);
        Ok(values
            .into_iter()
            .map(|bits| mul_sat(Act::from_bits(bits), scale).to_bits())
            .collect())
    }
}
