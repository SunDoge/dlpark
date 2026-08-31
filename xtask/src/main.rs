use clap::{Parser, Subcommand};
use snafu::Whatever;

mod cmd;
mod fs;

#[derive(Debug, Parser)]
#[command(about = "Workspace maintenance tasks")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Generate or check the DLPack FFI bindings.
    Bindgen {
        /// Check that the committed bindings are current without changing them.
        #[arg(long)]
        check: bool,
    },
    /// Refresh the vendored DLPack header and regenerate bindings.
    VendorDlpack,
}

fn main() -> Result<(), Whatever> {
    match Cli::parse().command {
        Command::Bindgen { check } => cmd::bindgen::run(check),
        Command::VendorDlpack => {
            cmd::vendor_dlpack::run()?;
            cmd::bindgen::run(false)
        }
    }
}
