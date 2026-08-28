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
fn exif_rotated_jpeg_corrected_before_ocr() {
    // Pixels are physically 90° CCW; EXIF orientation=6 tells the decoder to
    // rotate 90° CW.  normalize_image_for_ocr_png applies that correction so
    // the text is upright when Tesseract sees it.
    let mut engine = Engine::new().unwrap();
    let result = engine.extract(&fixture("exif_rotated.jpg"));

    assert_eq!(result.error, None);
    assert_eq!(result.extractor, Some(Extractor::OcrImage));
    assert!(full_text(&result).contains(EXPECTED));
}

#[test]
fn transparent_png_composites_over_white() {
    // Background pixels have alpha=0; normalize_image_for_ocr_png blends over
    // a white canvas so black text stays visible for OCR.
    let mut engine = Engine::new().unwrap();
    let result = engine.extract(&fixture("transparent.png"));

    assert_eq!(result.error, None);
    assert_eq!(result.extractor, Some(Extractor::OcrImage));
    assert!(full_text(&result).contains(EXPECTED));
}

#[test]
fn multipage_pdf_reports_correct_page_count() {
    let mut engine = Engine::new().unwrap();
    let result = engine.extract(&fixture("multipage.pdf"));

    assert_eq!(result.error, None);
    assert_eq!(result.n_pages, 2);
    assert_eq!(result.pages.len(), 2);
    assert!(full_text(&result).contains(EXPECTED));
}

#[test]
fn content_sniff_routes_extensionless_image_to_ocr() {
    // image.dat is a PNG renamed to a non-image extension.  sniffs_as_image
    // reads the magic bytes and still routes to OcrImage.
    let mut engine = Engine::new().unwrap();
    let result = engine.extract(&fixture("image.dat"));

    assert_eq!(result.error, None);
    assert_eq!(result.extractor, Some(Extractor::OcrImage));
    assert!(full_text(&result).contains(EXPECTED));
}

#[test]
fn extract_with_hint_true_forces_image_path() {
    // Passing treat_as_image=true short-circuits the sniff and extension check,
    // routing directly to extract_image_ocr.
    let mut engine = Engine::new().unwrap();
    let result = engine.extract_with_hint(&fixture("image.dat"), true);

    assert_eq!(result.error, None);
    assert_eq!(result.extractor, Some(Extractor::OcrImage));
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

// ---------------------------------------------------------------------------
// Per-page timings.
//
// These assert structure and presence only — never a duration. A test that
// asserts a number of microseconds is a flaky test on someone else's CI.

/// The invariants that hold on every path: one timing per page, in page order,
/// each starting no earlier than the last, and all of them fitting inside the
/// call's own wall clock (the slack is document open, the type sniff, and any
/// abandoned native attempt).
fn assert_timings_are_coherent(result: &Extraction) {
    assert_eq!(
        result.timings.len(),
        result.n_pages as usize,
        "one timing per page"
    );

    let mut previous_start = 0;
    for (i, t) in result.timings.iter().enumerate() {
        assert_eq!(t.page, (i + 1) as u32, "timings are in page order");
        assert!(
            t.started_us >= previous_start,
            "page {} starts before page {} did",
            t.page,
            i
        );
        previous_start = t.started_us;
    }

    let in_pages: u64 = result.timings.iter().map(|t| t.total_us).sum();
    assert!(
        in_pages <= result.total_us,
        "pages account for {in_pages}us of a {}us call",
        result.total_us
    );
}

#[test]
fn native_pages_are_timed_with_no_ocr_phases() {
    let mut engine = Engine::new().unwrap();
    let result = engine.extract(&fixture("multipage.pdf"));

    assert_eq!(result.extractor, Some(Extractor::Native));
    assert_timings_are_coherent(&result);

    for t in &result.timings {
        assert_eq!(t.extractor, Extractor::Native);
        // Nothing was rasterized, rotated, encoded or OCR'd.
        assert_eq!(t.render_us, None);
        assert_eq!(t.orient_us, None);
        assert_eq!(t.encode_us, None);
        assert_eq!(t.ocr_us, None);
    }
}

#[test]
fn ocr_pages_are_timed_by_phase() {
    let mut engine = Engine::new().unwrap();
    let result = engine.extract(&fixture("scanned.pdf"));

    assert_eq!(result.extractor, Some(Extractor::OcrPdf));
    assert_timings_are_coherent(&result);

    for t in &result.timings {
        assert_eq!(t.extractor, Extractor::OcrPdf);
        assert!(t.render_us.is_some(), "the page was rasterized");
        assert!(t.encode_us.is_some(), "the page image was encoded");
        assert!(t.ocr_us.is_some(), "the page was OCR'd");
        // Auto-rotate is off, so no vote ran.
        assert_eq!(t.orient_us, None);
    }
}

#[test]
fn the_orientation_vote_is_timed_only_when_it_runs() {
    // The regression guard for the rotation work: the vote is up to four extra
    // tesseract passes per page, so if a future change reintroduces one where
    // it isn't wanted — or drops one that is — the difference shows up here
    // rather than as an unexplained slowdown in someone's queue.
    //
    // Each engine gets its own scope: only one `Engine` can be alive at a time
    // (see `Engine`'s docs), so the first must be dropped before the second is
    // built.
    let rotating = {
        let mut engine = Engine::builder().auto_rotate(true).build().unwrap();
        engine.extract(&fixture("scanned.pdf"))
    };

    assert_eq!(rotating.extractor, Some(Extractor::OcrPdf));
    assert_timings_are_coherent(&rotating);
    for t in &rotating.timings {
        assert!(
            t.orient_us.is_some(),
            "auto-rotate is on, so page {} was voted on",
            t.page
        );
    }

    let plain = {
        let mut engine = Engine::new().unwrap();
        engine.extract(&fixture("scanned.pdf"))
    };
    assert!(plain.timings.iter().all(|t| t.orient_us.is_none()));
}

#[test]
fn an_image_yields_one_timing() {
    let mut engine = Engine::new().unwrap();
    let result = engine.extract(&fixture("sample.png"));

    assert_eq!(result.extractor, Some(Extractor::OcrImage));
    assert_timings_are_coherent(&result);

    let t = &result.timings[0];
    assert_eq!(t.extractor, Extractor::OcrImage);
    assert_eq!(t.page, 1);
    // Normalization stands in for the rasterization a PDF page would need.
    assert!(t.render_us.is_some());
    assert!(t.encode_us.is_some());
    assert!(t.ocr_us.is_some());
    assert_eq!(t.orient_us, None, "auto-rotate is off");
}

#[test]
fn timings_survive_a_serde_round_trip() {
    let mut engine = Engine::new().unwrap();
    let result = engine.extract(&fixture("scanned.pdf"));

    let json = serde_json::to_string(&result).unwrap();
    let back: Extraction = serde_json::from_str(&json).unwrap();

    assert_eq!(back.timings.len(), result.timings.len());
    assert_eq!(back.total_us, result.total_us);
    assert_eq!(back.timings[0].ocr_us, result.timings[0].ocr_us);
}

#[test]
fn records_written_before_timings_existed_still_deserialize() {
    // A 0.1.x `--json` record, or anything an embedder archived from one.
    let legacy = r#"{
        "extractor": "native",
        "n_pages": 1,
        "extracted_chars": 44,
        "needs_ocr": false,
        "pages": [{"page": 1, "text": "hello"}],
        "error": null
    }"#;

    let back: Extraction = serde_json::from_str(legacy).unwrap();
    assert_eq!(back.extractor, Some(Extractor::Native));
    assert!(back.timings.is_empty());
    assert_eq!(back.total_us, 0);
}
