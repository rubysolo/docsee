use anyhow::Result;
use clap::Parser;
use docsee::{Engine, Extraction, Extractor, MIN_CHARS_PER_PAGE, OCR_DPI, OCR_LANGUAGE};
use std::path::PathBuf;

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
        // Forced OCR on a PDF: render and OCR every page directly.
        match engine.extract_pdf_ocr(&args.file, args.dpi) {
            Ok((pages, chars)) => Extraction {
                extractor: Some(Extractor::OcrPdf),
                n_pages: pages.len() as u32,
                extracted_chars: chars,
                needs_ocr: false,
                pages,
                error: None,
                ..Default::default()
            },
            Err(e) => Extraction {
                error: Some(format!("ocr_failed: {e:?}")),
                ..Default::default()
            },
        }
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

    Ok(())
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
