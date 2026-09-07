//! Rust port of AstroBinUpload.py — Phase 1.
//!
//! Parity target: Python `v2.1.1` (see RUST_PORT_PLAN.md). This phase covers
//! the CLI, the configobj-compatible config parser, and the `--test` CSV
//! ingest path with pandas-equivalent dtype inference. The six pipeline steps
//! (Phase 2), the report/exporter formatting (Phase 3) and the FITS/XISF
//! readers (Phase 4) are not implemented yet; the binary reports what it
//! loaded and exits rather than pretending otherwise.

mod cli;
mod config;
mod table;

use anyhow::{bail, Context, Result};
use clap::Parser;

use crate::cli::Cli;
use crate::config::ConfigFile;
use crate::table::Table;

fn main() -> Result<()> {
    let args = Cli::parse();

    // B10 in REMEDIATION_PLAN.md: the Python side validates directory
    // arguments before use, because an unvalidated typo reached
    // os.makedirs() and silently created the typo'd tree.
    for dir in &args.directory_paths {
        if !dir.exists() {
            bail!("directory path does not exist: {}", dir.display());
        }
        if !dir.is_dir() {
            bail!("path is not a directory: {}", dir.display());
        }
    }

    // Unlike Python, a missing config is an error rather than a prompt to
    // generate a template: template generation is an interactive convenience
    // that has no place in a binary the differential harness drives.
    if !args.config.exists() {
        bail!(
            "configuration file not found: {} (the Rust port does not \
             auto-generate one; use the Python entry point for that)",
            args.config.display()
        );
    }
    let cfg = ConfigFile::parse_file(&args.config)
        .with_context(|| format!("parsing {}", args.config.display()))?;

    if args.dump_parity {
        dump_parity(&cfg, args.test.as_deref())?;
        return Ok(());
    }

    println!("astrobin-upload {} (Phase 1)", env!("CARGO_PKG_VERSION"));
    println!("config: {}", args.config.display());
    for name in ["defaults", "override", "equipmentoverrides", "filters", "sites"] {
        if let Some(sec) = cfg.section(name) {
            println!(
                "  [{name}] {} value(s), {} subsection(s)",
                sec.values.len(),
                sec.sections.len()
            );
        }
    }

    match &args.test {
        Some(csv) => {
            let t = Table::read_csv_upper(csv)
                .with_context(|| format!("ingesting {}", csv.display()))?;
            println!(
                "--test ingest: {} rows x {} columns from {}",
                t.n_rows,
                t.columns.len(),
                csv.display()
            );
            // Print the columns whose dtype is load-bearing for output
            // formatting, so a parity mismatch is visible immediately.
            for name in ["GAIN", "XBINNING", "NUMBER", "EGAIN", "EXPOSURE", "IMAGETYP"] {
                if let Some(c) = t.column(name) {
                    println!("    {name:<9} {:?}", c.dtype);
                }
            }
        }
        None => {
            println!("disk scanning is Phase 4; re-run with --test <csv> for now");
        }
    }

    bail!("Phase 1 only: the pipeline, reports and file readers are not implemented yet")
}

/// Canonical, line-oriented dump of everything Phase 1 parses.
///
/// Deliberately boring and sorted so it can be diffed byte-for-byte against
/// the equivalent dump produced from configobj and pandas.
fn dump_parity(cfg: &ConfigFile, test_csv: Option<&std::path::Path>) -> Result<()> {
    use crate::config::{Section, Value};

    fn render(v: &Value) -> String {
        match v {
            Value::Str(s) => format!("str\t{s}"),
            Value::List(items) => format!("list\t{}", items.join("\u{1f}")),
        }
    }

    fn walk(prefix: &str, sec: &Section, out: &mut Vec<String>) {
        for (k, v) in &sec.values {
            out.push(format!("{prefix}\t{k}\t{}", render(v)));
        }
        for (name, sub) in &sec.sections {
            walk(&format!("{prefix}/{name}"), sub, out);
        }
    }

    let mut lines = Vec::new();
    for (name, sec) in &cfg.sections {
        walk(name, sec, &mut lines);
    }
    lines.sort();
    for l in &lines {
        println!("CONFIG\t{l}");
    }

    if let Some(csv) = test_csv {
        let t = Table::read_csv_upper(csv)?;
        println!("CSV\trows\t{}", t.n_rows);
        let mut cols: Vec<String> = t
            .columns
            .iter()
            .map(|c| {
                let d = match c.dtype {
                    crate::table::DType::Int => "int64",
                    crate::table::DType::Float => "float64",
                    crate::table::DType::Bool => "bool",
                    crate::table::DType::Str => "object",
                };
                let nulls = c.cells.iter().filter(|x| x.is_null()).count();
                format!("CSV\tcol\t{}\t{d}\t{nulls}", c.name)
            })
            .collect();
        cols.sort();
        for c in cols {
            println!("{c}");
        }
    }
    Ok(())
}
