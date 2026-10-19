use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct MaterializationCacheKey {
    pub param: String,
    pub path: PathBuf,
    pub index_path: PathBuf,
    pub commitment: String,
    pub type_name: &'static str,
}

impl MaterializationCacheKey {
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.param.capacity()
            + self.commitment.capacity()
            + self.path.capacity()
            + self.index_path.capacity()
    }
}
