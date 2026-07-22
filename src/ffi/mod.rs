/// Raw C ABI declarations generated from the bundled DLPack headers.
#[allow(
    missing_docs,
    rustdoc::broken_intra_doc_links,
    rustdoc::invalid_html_tags
)]
mod raw;

pub mod data_type;
pub mod device;
pub mod managed_tensor;

pub use raw::*;
