use docsee::Engine;
use std::path::Path;

#[test]
fn test_auto_rotate_90() -> anyhow::Result<()> {
    let mut engine = Engine::builder().auto_rotate(true).build()?;
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rotated_90.png");

    let result = engine.extract(&fixture);
    assert!(
        result.error.is_none(),
        "Extraction failed: {:?}",
        result.error
    );
    assert_eq!(result.pages.len(), 1);

    let text = &result.pages[0].text;
    assert!(
        text.contains("brown fox"),
        "Text not found or poorly recognized: {}",
        text
    );
    Ok(())
}

#[test]
fn test_auto_rotate_180() -> anyhow::Result<()> {
    let mut engine = Engine::builder().auto_rotate(true).build()?;
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rotated_180.png");

    let result = engine.extract(&fixture);
    assert!(
        result.error.is_none(),
        "Extraction failed: {:?}",
        result.error
    );
    assert_eq!(result.pages.len(), 1);

    let text = &result.pages[0].text;
    assert!(
        text.contains("brown fox"),
        "Text not found or poorly recognized: {}",
        text
    );
    Ok(())
}

#[test]
fn test_auto_rotate_270() -> anyhow::Result<()> {
    let mut engine = Engine::builder().auto_rotate(true).build()?;
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rotated_270.png");

    let result = engine.extract(&fixture);
    assert!(
        result.error.is_none(),
        "Extraction failed: {:?}",
        result.error
    );
    assert_eq!(result.pages.len(), 1);

    let text = &result.pages[0].text;
    assert!(
        text.contains("brown fox"),
        "Text not found or poorly recognized: {}",
        text
    );
    Ok(())
}

#[test]
fn test_no_auto_rotate_fails_on_rotated() -> anyhow::Result<()> {
    let mut engine = Engine::builder().auto_rotate(false).build()?;
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rotated_90.png");

    let result = engine.extract(&fixture);
    let text = &result.pages[0].text;

    // Without auto-rotate, Tesseract should fail to find the text or get it very wrong.
    assert!(
        !text.contains("brown fox"),
        "Surprisingly found text without rotation: {}",
        text
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Reusing a detected rotation instead of re-deriving it.
//
// Detecting orientation is up to four extra tesseract passes per page, and a
// caller that OCRs a document and then renders the same pages for thumbnails
// used to pay for it twice. `Extraction::page_rotations` reports what the OCR
// pass decided; the `*_with_rotations` render entry points take it back. These
// tests pin the two properties that make that safe: the angle is reported, and
// feeding it back produces the same image the vote would have.

#[test]
fn reports_the_rotation_it_applied() -> anyhow::Result<()> {
    let mut engine = Engine::builder().auto_rotate(true).build()?;
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rotated_180.png");

    let result = engine.extract(&fixture);

    assert!(
        result.error.is_none(),
        "extraction failed: {:?}",
        result.error
    );
    assert_eq!(
        result.page_rotations.len(),
        result.pages.len(),
        "one rotation per page"
    );
    assert_eq!(
        result.page_rotations[0], 180,
        "a 180-degree fixture should report the 180 it corrected"
    );
    Ok(())
}

#[test]
fn reports_no_rotation_when_auto_rotate_is_off() -> anyhow::Result<()> {
    let mut engine = Engine::builder().auto_rotate(false).build()?;
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rotated_180.png");

    let result = engine.extract(&fixture);

    // Nothing was rotated, so nothing is claimed to have been.
    assert_eq!(result.page_rotations, vec![0]);
    Ok(())
}

#[test]
fn native_text_reports_no_rotations() -> anyhow::Result<()> {
    // The native path never rasterizes, so there is no angle to report — an
    // empty slice, which the render entry points read as "detect normally".
    let mut engine = Engine::builder().auto_rotate(true).build()?;
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/digital.pdf");

    let result = engine.extract(&fixture);

    assert_eq!(result.extractor, Some(docsee::Extractor::Native));
    assert!(result.page_rotations.is_empty());
    Ok(())
}

#[test]
fn a_supplied_rotation_is_the_rotation_applied() -> anyhow::Result<()> {
    // The contract the reuse API actually owes its callers: the angle you pass
    // is the angle you get. Deliberately NOT phrased as "the same bytes the
    // vote would have produced" — the vote is a confidence comparison, not a
    // pure function of the page, and it can land differently at a different
    // render DPI. Pinning it to that would be testing the detector's luck.
    let mut engine = Engine::builder().auto_rotate(true).build()?;
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scanned.pdf");

    for angle in [0u32, 90, 180, 270] {
        let via_slice = engine.render_pages_png_with_rotations(&fixture, 100.0, &[angle])?;
        let via_single =
            engine.render_pdf_page_png_with_rotation(&fixture, 1, 100.0, Some(angle))?;

        assert_eq!(
            via_slice[0], via_single,
            "the batch and single-page entry points disagreed at {angle} degrees"
        );
    }
    Ok(())
}

#[test]
fn reuse_costs_no_orientation_vote() -> anyhow::Result<()> {
    // The reason the API exists. Not a wall-clock assertion — CI machines are
    // noisy — but the vote is up to four OCR passes per page, so if it were
    // still running, supplying rotations could not be dramatically cheaper.
    use std::time::Instant;

    let mut engine = Engine::builder().auto_rotate(true).build()?;
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scanned.pdf");

    // Warm pdfium/tesseract so the first call doesn't pay one-time costs.
    let _ = engine.render_pages_png_with_rotations(&fixture, 100.0, &[0])?;

    let t = Instant::now();
    engine.render_pages_png(&fixture, 100.0)?;
    let detecting = t.elapsed();

    let t = Instant::now();
    engine.render_pages_png_with_rotations(&fixture, 100.0, &[0])?;
    let reusing = t.elapsed();

    assert!(
        reusing < detecting,
        "supplying the rotation ({reusing:?}) should beat detecting it ({detecting:?})"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// The orientation vote only fires on evidence.
//
// `detect_orientation` takes the best of four `mean_text_conf` scores. On a
// genuinely rotated page the winner is unmistakable (the 180 fixture scores 95
// against 26 upright). On an upright page there is nothing to find and all four
// scores land in a noise band, where the winner is arbitrary — an upright
// 4-page scan measured 35 / 33 / 27 / 42 and got turned 270 degrees, which
// destroys the dense small print. These pin both directions.

#[test]
fn an_upright_page_is_left_alone() -> anyhow::Result<()> {
    let mut engine = Engine::builder().auto_rotate(true).build()?;
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sample.png");

    let result = engine.extract(&fixture);

    assert_eq!(
        result.page_rotations,
        vec![0],
        "an already-upright page must not be rotated on a coin flip"
    );
    assert!(
        result.pages[0].text.contains("brown fox"),
        "upright text should read cleanly: {}",
        result.pages[0].text
    );
    Ok(())
}

#[test]
fn an_upright_scan_is_left_alone() -> anyhow::Result<()> {
    // The OCR-fallback path (a PDF with no text layer), which is where the
    // misfire was found — a scanned page is noisier than a clean PNG, so its
    // four scores sit closer together.
    let mut engine = Engine::builder().auto_rotate(true).build()?;
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scanned.pdf");

    let result = engine.extract(&fixture);

    assert!(
        result.page_rotations.iter().all(|&r| r == 0),
        "upright scan was rotated: {:?}",
        result.page_rotations
    );
    Ok(())
}

#[test]
fn an_empty_rotation_slice_is_the_old_behavior() -> anyhow::Result<()> {
    // Callers that don't have angles (and the plain entry points, which
    // delegate with `&[]`) must be completely unaffected.
    let mut engine = Engine::builder().auto_rotate(true).build()?;
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scanned.pdf");

    let plain = engine.render_pages_png(&fixture, 100.0)?;
    let empty = engine.render_pages_png_with_rotations(&fixture, 100.0, &[])?;

    assert_eq!(plain, empty);
    Ok(())
}

#[test]
fn a_short_rotation_slice_falls_back_per_page() -> anyhow::Result<()> {
    // A slice shorter than the document must not panic or silently leave later
    // pages unrotated by accident — uncovered pages detect as usual.
    let mut engine = Engine::builder().auto_rotate(true).build()?;
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/multipage.pdf");

    let plain = engine.render_pages_png(&fixture, 100.0)?;
    let short = engine.render_pages_png_with_rotations(&fixture, 100.0, &[0])?;

    assert_eq!(plain.len(), short.len());
    assert_eq!(plain, short);
    Ok(())
}

#[test]
fn single_page_render_accepts_a_known_rotation() -> anyhow::Result<()> {
    let mut engine = Engine::builder().auto_rotate(true).build()?;
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scanned.pdf");

    let extraction = engine.extract(&fixture);
    let angle = extraction.page_rotations.first().copied();

    let detected = engine.render_pdf_page_png(&fixture, 1, 100.0)?;
    let reused = engine.render_pdf_page_png_with_rotation(&fixture, 1, 100.0, angle)?;

    assert_eq!(detected, reused);
    Ok(())
}

#[test]
fn a_tiny_orientation_probe_finds_no_evidence() -> anyhow::Result<()> {
    // The knob has to be observable, or it's decoration. The vote reads text to
    // decide which way is up, so starving it of pixels must cost it the answer:
    // `test_auto_rotate_180` above shows the default probe correcting this same
    // fixture, and here a probe too small to read anything leaves it alone.
    //
    // Split across two tests rather than asserted in one on purpose: building a
    // second Engine in the same process deadlocks in `FPDF_InitLibrary`, which
    // takes a process-global lock the first engine never releases. One engine
    // per test, as everywhere else in this file.
    let mut engine = Engine::builder()
        .auto_rotate(true)
        .orientation_probe_dim(32)
        .build()?;
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rotated_180.png");

    assert_eq!(
        engine.extract(&fixture).page_rotations,
        vec![0],
        "a 32px probe cannot read anything, so the vote should find no evidence \
         and leave the page alone"
    );
    Ok(())
}

#[test]
fn a_zero_probe_dim_does_not_panic() -> anyhow::Result<()> {
    // `image::resize` to a zero dimension panics; 0 is clamped to 1 so a caller
    // asking for "as small as possible" gets a useless vote, not a crash.
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rotated_180.png");
    let mut engine = Engine::builder()
        .auto_rotate(true)
        .orientation_probe_dim(0)
        .build()?;

    let result = engine.extract(&fixture);

    assert!(
        result.error.is_none(),
        "extraction errored: {:?}",
        result.error
    );
    Ok(())
}

#[test]
fn a_pool_preserves_page_order() -> anyhow::Result<()> {
    // The pool finishes out of order by design, so the thing worth pinning is
    // that the result does not. Assembled from two different images so a
    // scrambled result is visible: page 1 needs correcting and page 2 does not,
    // and [0, 180] would be the same multiset as the right answer reversed.
    //
    // One engine per test — a second in the same process deadlocks in
    // `FPDF_InitLibrary` — so this builds a pooled engine and uses it for both
    // the assembly and the extraction.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut engine = Engine::builder().auto_rotate(true).ocr_threads(2).build()?;

    let upside_down = dir.join("rotated_180.png");
    let upright = dir.join("sample.png");
    let pdf = engine.assemble_pdf(&[
        docsee::PageRef {
            path: &upside_down,
            page_number: 1,
        },
        docsee::PageRef {
            path: &upright,
            page_number: 1,
        },
    ])?;

    let scan = std::env::temp_dir().join("docsee_pool_order.pdf");
    std::fs::write(&scan, pdf)?;

    let result = engine.extract(&scan);
    let _ = std::fs::remove_file(&scan);

    assert!(
        result.error.is_none(),
        "extraction failed: {:?}",
        result.error
    );
    assert_eq!(result.pages.len(), 2);
    assert_eq!(result.pages[0].page, 1);
    assert_eq!(result.pages[1].page, 2);
    assert_eq!(
        result.page_rotations,
        vec![180, 0],
        "pooled pages came back in the wrong order"
    );
    Ok(())
}
