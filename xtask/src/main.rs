use bindgen::callbacks::ParseCallbacks;
use snafu::{ResultExt, Whatever, whatever};
use std::{io::Write, path::Path};

fn main() -> Result<(), Whatever> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match args.as_slice() {
        [command] if command == "bindgen" => generate_bindings(false),
        [command, flag] if command == "bindgen" && flag == "--check" => generate_bindings(true),
        _ => {
            eprintln!("usage: cargo xtask bindgen [--check]");
            std::process::exit(2);
        }
    }
}

fn generate_bindings(check: bool) -> Result<(), Whatever> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask must be located directly inside the workspace root");
    let header = workspace.join("dlpack/include/dlpack/dlpack.h");
    let output = workspace.join("src/ffi.rs");

    if !header.is_file() {
        whatever!(
            "DLPack header not found at {}; initialize the submodule with `git submodule update --init`",
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

    let post_processed = replace_required(
        bindings,
        "pub struct DLDataTypeCode(pub ::std::os::raw::c_uint);",
        "pub struct DLDataTypeCode(pub u8);",
    )?;
    let post_processed = replace_required(post_processed, "code: u8", "code: DLDataTypeCode")?;
    let post_processed = replace_required(post_processed, "SetError", "set_error")?;
    let post_processed =
        replace_required(post_processed, "flags: u64", "flags: crate::DlpackFlags")?;
    let post_processed = post_processed
        .replace("\\\\brief ", "")
        // bindgen preserves Doxygen code commands in doc attributes. Rustdoc
        // otherwise treats their indented C/C++ contents as Rust doctests.
        .replace("\\\\code{.c}", "```c")
        .replace("\\\\code", "```text")
        .replace("\\\\endcode", "```");
    let post_processed = replace_required(
        post_processed,
        "#[allow(clippy::unnecessary_operation, clippy::identity_op)]",
        "#[cfg(target_pointer_width = \"64\")]\n#[allow(clippy::unnecessary_operation, clippy::identity_op)]",
    )?;

    if check {
        let existing = std::fs::read_to_string(&output)
            .whatever_context("failed to read existing bindings")?;
        if existing != post_processed {
            whatever!(
                "{} is out of date; run `cargo xtask bindgen`",
                output.display()
            );
        }
        return Ok(());
    }

    let output_dir = output
        .parent()
        .expect("bindings output must have a parent directory");
    let mut temporary = tempfile::NamedTempFile::new_in(output_dir)
        .whatever_context("failed to create temp file")?;
    temporary
        .write_all(post_processed.as_bytes())
        .whatever_context("failed to write generated bindings")?;
    temporary
        .persist(&output)
        .map_err(|error| error.error)
        .whatever_context("failed to replace generated bindings")?;

    Ok(())
}

fn replace_required(input: String, from: &str, to: &str) -> Result<String, Whatever> {
    let count = input.matches(from).count();
    if count == 0 {
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
        let prefix = "kDL";
        original_variant_name
            .strip_prefix(prefix)
            .map(str::to_uppercase)
    }
}
