use snafu::Whatever;
use usage_rs::{Cli, Subcommands};

mod cmd;
mod fs;

/// Workspace maintenance tasks.
#[derive(Debug, Cli)]
#[usage(bin = "xtask", unknown_flags = "error", args_override_self = false)]
struct Cli {
    #[usage(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommands)]
enum Command {
    /// Generate or check the DLPack FFI bindings.
    Bindgen {
        /// Check that the committed bindings are current without changing them.
        #[usage(long)]
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
