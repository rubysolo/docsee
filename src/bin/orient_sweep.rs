//! Sweep the orientation early-exit threshold over a set of files.
//!
//! One engine per process (pdfium's init lock is process-global and never
//! released), so a sweep is one process per threshold — the shell loops, this
//! binary handles every file at one setting. Prints the chosen angle and the
//! vote's cost, and nothing from the documents themselves, so it is safe to run
//! over real corpus files.
//!
//!   orient_sweep --early-exit-conf 90 file1.pdf file2.png

use anyhow::{bail, Result};
use docsee::Engine;
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut conf: Option<i32> = None;
    let mut files: Vec<PathBuf> = Vec::new();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--early-exit-conf" => {
                conf = Some(
                    args.next()
                        .and_then(|v| v.parse().ok())
                        .ok_or_else(|| anyhow::anyhow!("--early-exit-conf needs an integer"))?,
                )
            }
            other => files.push(PathBuf::from(other)),
        }
    }

    if files.is_empty() {
        bail!("usage: orient_sweep [--early-exit-conf N] <file>...");
    }

    let mut builder = Engine::builder().auto_rotate(true);
    if let Some(conf) = conf {
        builder = builder.orientation_early_exit_conf(conf);
    }
    let mut engine = builder.build()?;

    for path in &files {
        let result = engine.extract(path);
        let orient_us: u64 = result.timings.iter().filter_map(|t| t.orient_us).sum();
        let name = path.file_name().unwrap_or_default().to_string_lossy();

        println!(
            "{}\t{}\t{:?}\t{}\t{}",
            conf.map(|c| c.to_string())
                .unwrap_or_else(|| "default".into()),
            name,
            result.page_rotations,
            orient_us,
            result.error.unwrap_or_default()
        );
    }
    Ok(())
}
