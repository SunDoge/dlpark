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
        .replace("flags: u64", "flags: crate::DlpackFlags")
        .replace("\\\\brief ", "")
        // bindgen preserves Doxygen code commands in doc attributes. Rustdoc
        // otherwise treats their indented C/C++ contents as Rust doctests.
        .replace("\\\\code{.c}", "```c")
        .replace("\\\\code", "```text")
        .replace("\\\\endcode", "```")
        // bindgen emits struct size/offset layout tests using the host pointer
        // width, so they fail to compile on 32-bit targets (e.g. wasm32) where
        // pointer-containing structs are smaller. Gate every layout-test block
        // on a 64-bit pointer width so the checks still run where they were
        // generated but don't break 32-bit builds.
        .replace(
            "#[allow(clippy::unnecessary_operation, clippy::identity_op)]",
            "#[cfg(target_pointer_width = \"64\")]\n#[allow(clippy::unnecessary_operation, clippy::identity_op)]",
        );

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
        original_variant_name
            .strip_prefix(prefix)
            .map(str::to_uppercase)
    }
}
