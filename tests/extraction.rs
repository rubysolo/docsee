//! Integration tests. These need libpdfium (run `./get_pdfium.sh`) and
//! tesseract with English tessdata installed, same as the build itself.

use docsee::{Engine, Extraction, Extractor};
use std::path::Path;

const EXPECTED: &str = "The quick brown fox jumps over the lazy dog.";

fn fixture(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn full_text(result: &Extraction) -> String {
    result
        .pages
        .iter()
        .map(|p| p.text.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn digital_pdf_extracts_natively() {
    let mut engine = Engine::new().unwrap();
    let result = engine.extract(&fixture("digital.pdf"));

    assert_eq!(result.error, None);
    assert_eq!(result.extractor, Some(Extractor::Native));
    assert_eq!(result.n_pages, 1);
    assert!(!result.needs_ocr);
    assert!(full_text(&result).contains(EXPECTED));
}

#[test]
fn image_only_pdf_falls_back_to_ocr() {
    let mut engine = Engine::new().unwrap();
    let result = engine.extract(&fixture("scanned.pdf"));

    assert_eq!(result.error, None);
    assert_eq!(result.extractor, Some(Extractor::OcrPdf));
    assert_eq!(result.n_pages, 1);
    assert!(!result.needs_ocr);
    assert!(full_text(&result).contains(EXPECTED));
}

#[test]
fn image_file_is_ocrd_directly() {
    let mut engine = Engine::new().unwrap();
    let result = engine.extract(&fixture("sample.png"));

    assert_eq!(result.error, None);
    assert_eq!(result.extractor, Some(Extractor::OcrImage));
    assert_eq!(result.n_pages, 1);
    assert!(full_text(&result).contains(EXPECTED));
}

#[test]
fn builder_options_apply() {
    let mut engine = Engine::builder()
        .ocr_dpi(300.0)
        .min_chars_per_page(10)
        .build()
        .unwrap();
    let result = engine.extract(&fixture("scanned.pdf"));

    assert_eq!(result.extractor, Some(Extractor::OcrPdf));
    assert!(full_text(&result).contains(EXPECTED));
}

#[test]
fn extraction_serializes_with_snake_case_labels() {
    let mut engine = Engine::new().unwrap();
    let result = engine.extract(&fixture("digital.pdf"));

    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains(r#""extractor":"native""#));

    let back: Extraction = serde_json::from_str(&json).unwrap();
    assert_eq!(back.extractor, Some(Extractor::Native));
    assert_eq!(back.n_pages, result.n_pages);
}
