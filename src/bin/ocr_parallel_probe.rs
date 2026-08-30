//! Does tesseract actually scale across threads?
//!
//! The engine holds one `LepTess` and OCRs pages one at a time, so an extract
//! uses one core no matter how many the box has. Whether pooling instances is
//! worth building rests entirely on whether N instances on N threads do N
//! pages in the time one instance does one — this measures that, and nothing
//! else. No pdfium, no Engine, no page pipeline.
//!
//!   ocr_parallel_probe <image> [reps_per_worker] [max_workers]
//!
//! Prints only timings, never recognized text.

use anyhow::{bail, Result};
use leptess::LepTess;
use std::time::Instant;

/// `LepTess` holds raw tesseract/leptonica pointers, so it is `!Send` by
/// default. Moving one to a worker is sound for the same reason the dashboard's
/// NIF already relies on: neither library pins state to the thread that created
/// it (no TLS), and each instance here is touched by exactly one thread.
struct SendTess(LepTess);
unsafe impl Send for SendTess {}

fn ocr_once(t: &mut LepTess, png: &[u8]) -> Result<usize> {
    t.set_image_from_mem(png)?;
    Ok(t.get_utf8_text()?.len())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_default();
    if path.is_empty() {
        bail!("usage: ocr_parallel_probe <image> [reps_per_worker] [max_workers]");
    }
    let reps: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(2);
    let max_workers: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(4);
    let png = std::fs::read(&path)?;

    println!(
        "image {} ({} KB), {reps} rep(s) per worker\n",
        path,
        png.len() / 1024
    );

    let mut baseline_ms = 0f64;
    for workers in 1..=max_workers {
        // Built up front and sequentially: instances are independent once
        // live, but racing `TessBaseAPIInit` is not worth finding out about.
        let init = Instant::now();
        let mut pool: Vec<SendTess> = Vec::new();
        for _ in 0..workers {
            pool.push(SendTess(LepTess::new(None, "eng")?));
        }
        let init_ms = init.elapsed().as_secs_f64() * 1000.0;

        let started = Instant::now();
        std::thread::scope(|scope| {
            for tess in pool.iter_mut() {
                let png = &png;
                scope.spawn(move || {
                    for _ in 0..reps {
                        let _ = ocr_once(&mut tess.0, png);
                    }
                });
            }
        });
        let wall_ms = started.elapsed().as_secs_f64() * 1000.0;

        let pages = workers * reps;
        let per_page = wall_ms / pages as f64;
        if workers == 1 {
            baseline_ms = per_page;
        }

        println!(
            "{workers} worker(s): {pages} pages in {wall_ms:7.0}ms  \
             {per_page:6.0}ms/page  speedup {:.2}x  (init {init_ms:.0}ms)",
            baseline_ms / per_page
        );
    }
    Ok(())
}
