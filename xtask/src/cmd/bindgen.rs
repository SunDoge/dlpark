use bindgen::callbacks::ParseCallbacks;
use snafu::{ResultExt, Whatever, whatever};
use std::path::Path;

use crate::fs::atomic_write;

pub(crate) fn run(check: bool) -> Result<(), Whatever> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask must be located directly inside the workspace root");
    let header = workspace.join("vendor/dlpack/dlpack.h");
    let output = workspace.join("src/ffi.rs");

    if !header.is_file() {
        whatever!(
            "vendored DLPack header not found at {}; run `cargo xtask vendor-dlpack`",
            header.display()
        );
    }

    let bindings = bindgen::builder()
        .header(header.to_string_lossy())
        .allowlist_item("DL.*")
        .newtype_enum("DL.*")
        .parse_callbacks(Box::new(DlpackCallbacks))
        .generate()
        .whatever_context("fail to generate bindings")?
        .to_string();

    let bindings = replace_required(
        bindings,
        "pub struct DLDataTypeCode(pub ::std::os::raw::c_uint);",
        "pub struct DLDataTypeCode(pub u8);",
    )?;
    let bindings = replace_required(bindings, "code: u8", "code: DLDataTypeCode")?;
    let bindings = replace_required(bindings, "SetError", "set_error")?;
    let bindings = replace_required(bindings, "flags: u64", "flags: crate::DlpackFlags")?;
    let bindings = bindings
        .replace("\\\\brief ", "")
        .replace("\\\\code{.c}", "```c")
        .replace("\\\\code", "```text")
        .replace("\\\\endcode", "```");
    let bindings = replace_required(
        bindings,
        "#[allow(clippy::unnecessary_operation, clippy::identity_op)]",
        "#[cfg(target_pointer_width = \"64\")]\n#[allow(clippy::unnecessary_operation, clippy::identity_op)]",
    )?;

    if check {
        let existing = std::fs::read_to_string(&output)
            .whatever_context("failed to read existing bindings")?;
        if existing != bindings {
            whatever!(
                "{} is out of date; run `cargo xtask bindgen`",
                output.display()
            );
        }
    } else {
        atomic_write(&output, bindings.as_bytes())?;
    }
    Ok(())
}

fn replace_required(input: String, from: &str, to: &str) -> Result<String, Whatever> {
    if !input.contains(from) {
        whatever!("required bindgen post-processing pattern was not found: {from:?}");
    }
    Ok(input.replace(from, to))
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
        original_variant_name
            .strip_prefix("kDL")
            .map(str::to_uppercase)
    }
}
