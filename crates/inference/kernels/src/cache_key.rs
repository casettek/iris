use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct MaterializationCacheKey {
    pub param: String,
    pub path: PathBuf,
    pub index_path: PathBuf,
    pub commitment: String,
    pub type_name: &'static str,
}
