use snafu::{ResultExt, Whatever, whatever};
use std::{path::Path, process::Command};

use crate::fs::atomic_write;

const VERSION: &str = "v1.3";
const REVISION: &str = "84d107bf416c6bab9ae68ad285876600d230490d";

pub(crate) fn run() -> Result<(), Whatever> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask must be located directly inside the workspace root");
    let vendor = workspace.join("vendor/dlpack");
    std::fs::create_dir_all(&vendor).whatever_context("failed to create vendor directory")?;

    verify_release()?;
    download("include/dlpack/dlpack.h", &vendor.join("dlpack.h"))?;
    download("LICENSE", &vendor.join("LICENSE"))?;

    let provenance = format!(
        "# DLPack header\n\n\
         This directory vendors the single upstream header needed to regenerate dlpark's FFI bindings.\n\n\
         - Upstream: https://github.com/dmlc/dlpack\n\
         - Version: `{VERSION}`\n\
         - Commit: `{REVISION}`\n\
         - Verification: GitHub reports a valid commit signature\n\n\
         To refresh these files from the pinned release, run `mise run vendor:dlpack`.\n\
         To upgrade DLPack, update the version and full commit SHA in `xtask/src/cmd/vendor_dlpack.rs`, then run the same task and review both the vendored and generated binding diffs.\n"
    );
    atomic_write(&vendor.join("README.md"), provenance.as_bytes())
}

fn verify_release() -> Result<(), Whatever> {
    let tag = gh_api(&[
        &format!("repos/dmlc/dlpack/git/ref/tags/{VERSION}"),
        "--jq",
        ".object.sha",
    ])?;
    if tag != format!("{REVISION}\n").as_bytes() {
        whatever!("DLPack tag {VERSION} does not resolve to {REVISION}");
    }

    let verified = gh_api(&[
        &format!("repos/dmlc/dlpack/commits/{REVISION}"),
        "--jq",
        ".commit.verification.verified",
    ])?;
    if verified != b"true\n" {
        whatever!("GitHub did not report a valid signature for DLPack commit {REVISION}");
    }
    Ok(())
}

fn download(relative: &str, destination: &Path) -> Result<(), Whatever> {
    let contents = gh_api(&[
        &format!("repos/dmlc/dlpack/contents/{relative}?ref={REVISION}"),
        "--header",
        "Accept: application/vnd.github.raw+json",
    ])?;
    atomic_write(destination, &contents)
}

fn gh_api(args: &[&str]) -> Result<Vec<u8>, Whatever> {
    let output = Command::new("gh")
        .arg("api")
        .args(args)
        .output()
        .whatever_context("failed to run `gh`; install and authenticate GitHub CLI")?;
    if !output.status.success() {
        whatever!("`gh api {}` failed with {}", args.join(" "), output.status);
    }
    Ok(output.stdout)
}
