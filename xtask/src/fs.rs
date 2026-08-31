use snafu::{ResultExt, Whatever};
use std::{io::Write, path::Path};

pub(crate) fn atomic_write(destination: &Path, contents: &[u8]) -> Result<(), Whatever> {
    let parent = destination
        .parent()
        .expect("output must have a parent directory");
    let mut temporary =
        tempfile::NamedTempFile::new_in(parent).whatever_context("failed to create temp file")?;
    temporary
        .write_all(contents)
        .whatever_context("failed to write temp file")?;
    temporary
        .persist(destination)
        .map_err(|error| error.error)
        .whatever_context("failed to replace output")?;
    Ok(())
}
