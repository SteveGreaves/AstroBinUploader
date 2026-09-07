//! Command-line interface, mirroring `AstroBinUpload.py`'s argparse setup.
//!
//! Flag names, defaults and arity are part of the parity surface: the golden
//! harness invokes the binary as
//! `<exe> <scenario_dir> --test <csv> --config <ini>`, so anything that
//! changes how those parse changes whether the harness can drive the port.

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "astrobin-upload",
    version,
    about = "AstroBin Upload Utility - a high-performance ETL pipeline for astronomical metadata.",
    long_about = None,
    after_help = "\
Example Usage:
  astrobin-upload /path/to/my/images
  astrobin-upload /path/to/my/images /path/to/my/calibrationfiles
  astrobin-upload /images /calibration_dir --debug
  astrobin-upload . --test my_headers.csv"
)]
pub struct Cli {
    /// One or more directory paths to recursively scan for FITS (.fits, .fit,
    /// .fts) or XISF (.xisf) files.
    #[arg(required = true, num_args = 1..)]
    pub directory_paths: Vec<PathBuf>,

    /// Diagnostic Mode: instead of scanning disk, inject metadata from a
    /// pre-processed CSV file.
    #[arg(long, value_name = "CSV_FILE")]
    pub test: Option<PathBuf>,

    /// Enable verbose debug logging and preserve intermediate dataframes.
    #[arg(long)]
    pub debug: bool,

    /// Specify a custom configuration file.
    #[arg(long, short = 'c', value_name = "CONFIG_FILE", default_value = "config.ini")]
    pub config: PathBuf,

    /// Print a canonical dump of the parsed config (and, with --test, the
    /// ingested CSV's inferred dtypes) and exit. Used by
    /// golden_tests/check_rust_parity.py to diff this port's parsing against
    /// configobj and pandas on the same inputs. Hidden: a debugging aid, not
    /// part of the Python CLI surface.
    #[arg(long, hide = true)]
    pub dump_parity: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn clap_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn parses_the_invocation_the_golden_harness_uses() {
        let cli = Cli::try_parse_from([
            "astrobin-upload",
            "/tmp/Sadr_Region",
            "--test",
            "/tmp/Sadr_Region/raw.csv",
            "--config",
            "golden_tests/golden_config.ini",
        ])
        .unwrap();
        assert_eq!(cli.directory_paths, vec![PathBuf::from("/tmp/Sadr_Region")]);
        assert_eq!(cli.test, Some(PathBuf::from("/tmp/Sadr_Region/raw.csv")));
        assert_eq!(
            cli.config,
            PathBuf::from("golden_tests/golden_config.ini")
        );
        assert!(!cli.debug);
    }

    #[test]
    fn config_defaults_to_config_ini() {
        let cli = Cli::try_parse_from(["astrobin-upload", "/images"]).unwrap();
        assert_eq!(cli.config, PathBuf::from("config.ini"));
    }

    #[test]
    fn multiple_directories_are_accepted_and_at_least_one_is_required() {
        let cli = Cli::try_parse_from(["astrobin-upload", "/lights", "/cal"]).unwrap();
        assert_eq!(cli.directory_paths.len(), 2);
        assert!(Cli::try_parse_from(["astrobin-upload"]).is_err());
    }
}
