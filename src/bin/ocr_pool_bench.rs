//! What `ocr_threads` buys on a real multi-page scan.
//!
//! One engine per process (pdfium's init lock is process-global), so a sweep is
//! one process per thread count — the shell loops.
//!
//!   ocr_pool_bench --assemble <image> <pages> <out.pdf>   # build a scan fixture
//!   ocr_pool_bench --threads N <file.pdf>                 # time an extraction
//!
//! Prints timings only, never recognized text.

use anyhow::{bail, Result};
use docsee::{Engine, PageRef};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.first().map(String::as_str) == Some("--assemble") {
        let [_, image, pages, out] = &args[..] else {
            bail!("usage: ocr_pool_bench --assemble <image> <pages> <out.pdf>");
        };
        let n: usize = pages.parse()?;
        let path = PathBuf::from(image);
        let refs: Vec<PageRef> = (0..n)
            .map(|_| PageRef {
                path: &path,
                page_number: 1,
            })
            .collect();

        let mut engine = Engine::builder().build()?;
        std::fs::write(out, engine.assemble_pdf(&refs)?)?;
        println!("assembled {n} page(s) -> {out}");
        return Ok(());
    }

    let (threads, file) = match &args[..] {
        [flag, n, file] if flag == "--threads" => (n.parse::<usize>()?, file.clone()),
        [file] => (1, file.clone()),
        _ => bail!("usage: ocr_pool_bench [--threads N] <file.pdf>"),
    };

    let mut engine = Engine::builder()
        .auto_rotate(true)
        .ocr_threads(threads)
        .build()?;

    let started = Instant::now();
    let result = engine.extract(Path::new(&file));
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;

    if let Some(err) = &result.error {
        bail!("extraction failed: {err}");
    }

    let engine_us: u64 = result.total_us;
    let sum_pages: u64 = result.timings.iter().map(|t| t.total_us).sum();
    let orient: u64 = result.timings.iter().filter_map(|t| t.orient_us).sum();
    let ocr: u64 = result.timings.iter().filter_map(|t| t.ocr_us).sum();

    println!(
        "{threads:>2} thread(s)  wall {wall_ms:8.0}ms  engine {:8.0}ms  \
         Σpages {:8.0}ms  orient {:7.0}ms  ocr {:7.0}ms  overlap {:.2}x  rotations {:?}",
        engine_us as f64 / 1000.0,
        sum_pages as f64 / 1000.0,
        orient as f64 / 1000.0,
        ocr as f64 / 1000.0,
        sum_pages as f64 / engine_us as f64,
        result.page_rotations,
    );
    Ok(())
}
