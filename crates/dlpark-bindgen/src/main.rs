use bindgen::callbacks::ParseCallbacks;
use snafu::{ResultExt, Whatever};

fn main() -> Result<(), Whatever> {
    let bindings = bindgen::builder()
        .header("dlpack/include/dlpack/dlpack.h")
        .allowlist_item("DL.*")
        .newtype_enum("DL.*")
        .parse_callbacks(Box::new(DlpackCallbacks))
        .generate()
        .whatever_context("fail to generate bindings")?
        .to_string();

    let post_processed = bindings
        .replace(
            "pub struct DLDataTypeCode(pub ::std::os::raw::c_uint);",
            "pub struct DLDataTypeCode(pub u8);",
        )
        .replace("code: u8", "code: DLDataTypeCode")
        .replace("SetError", "set_error")
        .replace("flags: u64", "flags: crate::DlpackFlags");

    std::fs::write("src/ffi.rs", &post_processed).whatever_context("failed to write file")?;

    Ok(())
}

#[derive(Debug)]
struct DlpackCallbacks;

impl ParseCallbacks for DlpackCallbacks {
    fn enum_variant_name(
        &self,
        _enum_name: Option<&str>,
        original_variant_name: &str,
        _variant_value: bindgen::callbacks::EnumVariantValue,
    ) -> Option<String> {
        let prefix = "kDL";
        if original_variant_name.starts_with(prefix) {
            Some(original_variant_name[prefix.len()..].to_uppercase())
        } else {
            None
        }
    }
}
