use anyhow::Result;
use clap::Parser;
use docsee::{Engine, Extraction, Extractor, MIN_CHARS_PER_PAGE, OCR_DPI, OCR_LANGUAGE};
use std::path::PathBuf;
use std::time::Instant;

/// Extract text from a PDF or image: native PDF text via pdfium, with
/// tesseract OCR for images and low-text PDFs.
#[derive(Parser)]
#[command(version, about)]
struct Args {
    /// PDF or image file to extract
    file: PathBuf,

    /// Print the full extraction record as JSON instead of plain text
    #[arg(long)]
    json: bool,

    /// Force OCR, skipping native PDF text extraction
    #[arg(long)]
    ocr: bool,

    /// DPI at which PDF pages are rendered for OCR
    #[arg(long, default_value_t = OCR_DPI)]
    dpi: f32,

    /// Tesseract language code (the matching tessdata pack must be installed)
    #[arg(long, default_value = OCR_LANGUAGE)]
    lang: String,

    /// Chars-per-page threshold below which native extraction falls back to OCR
    #[arg(long, default_value_t = MIN_CHARS_PER_PAGE)]
    min_chars_per_page: usize,

    /// Automatically detect and correct image orientation before OCR
    #[arg(long)]
    auto_rotate: bool,

    /// Print a per-page, per-phase timing table to stderr
    #[arg(long)]
    timings: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut engine = Engine::builder()
        .ocr_language(&args.lang)
        .ocr_dpi(args.dpi)
        .min_chars_per_page(args.min_chars_per_page)
        .auto_rotate(args.auto_rotate)
        .build()?;

    let result = if args.ocr && !has_image_extension(&args.file) {
        // Forced OCR on a PDF: render and OCR every page directly. This skips
        // `extract`, so it also skips the per-page timings it collects; only
        // the wall clock for the whole run is available here.
        let start = Instant::now();
        let mut out = Extraction::default();
        match engine.extract_pdf_ocr(&args.file, args.dpi) {
            Ok((pages, chars)) => {
                out.extractor = Some(Extractor::OcrPdf);
                out.n_pages = pages.len() as u32;
                out.extracted_chars = chars;
                out.pages = pages;
            }
            Err(e) => out.error = Some(format!("ocr_failed: {e:?}")),
        }
        out.total_us = start.elapsed().as_micros() as u64;
        out
    } else {
        engine.extract(&args.file)
    };

    if let Some(err) = &result.error {
        eprintln!("error: {err}");
        std::process::exit(1);
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        let text = result
            .pages
            .iter()
            .map(|p| p.text.trim_end())
            .collect::<Vec<_>>()
            .join("\n\n");
        println!("{text}");
    }

    // To stderr, so it stays out of the extracted text (or the JSON) when
    // stdout is redirected to a file.
    if args.timings {
        print_timings(&result);
    }

    Ok(())
}

/// Per-page, per-phase table: where a slow file spent its time, without
/// piping `--json` through `jq`.
fn print_timings(result: &Extraction) {
    if result.timings.is_empty() {
        eprintln!(
            "total {} (no per-page timings: --ocr bypasses the path that records them)",
            duration(result.total_us)
        );
        return;
    }

    eprintln!(
        "{:>4}  {:<9} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "page", "extractor", "start", "total", "render", "orient", "encode", "ocr"
    );
    for t in &result.timings {
        eprintln!(
            "{:>4}  {:<9} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            t.page,
            label(t.extractor),
            duration(t.started_us),
            duration(t.total_us),
            phase(t.render_us),
            phase(t.orient_us),
            phase(t.encode_us),
            phase(t.ocr_us),
        );
    }

    let in_pages: u64 = result.timings.iter().map(|t| t.total_us).sum();
    eprintln!(
        "\ntotal {} — {} in pages, {} outside them (document open, the type \
         sniff, any abandoned native attempt)",
        duration(result.total_us),
        duration(in_pages),
        duration(result.total_us.saturating_sub(in_pages)),
    );
}

fn label(extractor: Extractor) -> &'static str {
    match extractor {
        Extractor::Native => "native",
        Extractor::OcrPdf => "ocr_pdf",
        Extractor::OcrImage => "ocr_image",
    }
}

/// A phase that did not run on this page reads as a dash, not a zero.
fn phase(us: Option<u64>) -> String {
    us.map(duration).unwrap_or_else(|| "-".to_string())
}

/// Microseconds at a scale a person can compare at a glance.
fn duration(us: u64) -> String {
    match us {
        0..=999 => format!("{us}us"),
        1_000..=999_999 => format!("{:.1}ms", us as f64 / 1_000.0),
        _ => format!("{:.2}s", us as f64 / 1_000_000.0),
    }
}

// Images are always OCR'd, so --ocr only changes behavior for PDFs.
fn has_image_extension(path: &std::path::Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .as_deref(),
        Some("jpg" | "jpeg" | "png" | "tif" | "tiff" | "bmp")
    )
}
