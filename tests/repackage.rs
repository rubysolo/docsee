use docsee::{Engine, Extractor, PageRef, PdfPagePaperSize, PdfPoints};
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

// Pixel dimensions from a PNG's IHDR chunk (width/height are big-endian u32 at
// byte offsets 16 and 20). Lets us check rendered page geometry without pulling
// in an image-decoding dev-dependency.
fn png_dimensions(bytes: &[u8]) -> (u32, u32) {
    assert!(
        bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]),
        "not a PNG"
    );
    let w = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
    let h = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
    (w, h)
}

// Save assembled bytes and re-extract them, so we check real PDF structure.
fn reextract(engine: &mut Engine, bytes: &[u8], tag: &str) -> docsee::Extraction {
    assert!(bytes.starts_with(b"%PDF"), "assembled output is a PDF");
    let out = std::env::temp_dir().join(format!("docsee-asm-{tag}.pdf"));
    std::fs::write(&out, bytes).unwrap();
    let result = engine.extract(&out);
    let _ = std::fs::remove_file(&out);
    result
}

#[test]
fn assembles_selected_pages_into_a_native_pdf() -> anyhow::Result<()> {
    let mut engine = Engine::new()?;
    let multi = fixture("multipage.pdf");

    // pick page 2 then page 1 — a 2-page result
    let bytes = engine.assemble_pdf(&[
        PageRef {
            path: &multi,
            page_number: 2,
        },
        PageRef {
            path: &multi,
            page_number: 1,
        },
    ])?;

    let result = reextract(&mut engine, &bytes, "order");
    assert!(
        result.error.is_none(),
        "re-extract failed: {:?}",
        result.error
    );
    assert_eq!(result.pages.len(), 2);
    // native content survived (not rasterized) — text is still extractable
    assert_eq!(result.extractor, Some(Extractor::Native));
    assert!(result.pages[0].text.contains("brown fox"));
    Ok(())
}

#[test]
fn selects_a_subset_and_allows_repeats() -> anyhow::Result<()> {
    let mut engine = Engine::new()?;
    let multi = fixture("multipage.pdf");

    // one page
    let one = engine.assemble_pdf(&[PageRef {
        path: &multi,
        page_number: 1,
    }])?;
    assert_eq!(reextract(&mut engine, &one, "one").pages.len(), 1);

    // same page twice -> two pages
    let dup = engine.assemble_pdf(&[
        PageRef {
            path: &multi,
            page_number: 1,
        },
        PageRef {
            path: &multi,
            page_number: 1,
        },
    ])?;
    assert_eq!(reextract(&mut engine, &dup, "dup").pages.len(), 2);
    Ok(())
}

#[test]
fn assembles_pages_from_multiple_sources() -> anyhow::Result<()> {
    let mut engine = Engine::new()?;
    let multi = fixture("multipage.pdf");
    let digital = fixture("digital.pdf");

    let bytes = engine.assemble_pdf(&[
        PageRef {
            path: &digital,
            page_number: 1,
        },
        PageRef {
            path: &multi,
            page_number: 1,
        },
    ])?;

    let result = reextract(&mut engine, &bytes, "multi");
    assert!(
        result.error.is_none(),
        "re-extract failed: {:?}",
        result.error
    );
    assert_eq!(result.pages.len(), 2);
    Ok(())
}

#[test]
fn assembles_an_image_only_document() -> anyhow::Result<()> {
    let mut engine = Engine::new()?;

    let bytes = engine.assemble_pdf(&[PageRef {
        path: &fixture("sample.png"),
        page_number: 1,
    }])?;

    let result = reextract(&mut engine, &bytes, "img-only");
    assert!(
        result.error.is_none(),
        "re-extract failed: {:?}",
        result.error
    );
    // one image source -> a valid one-page PDF
    assert_eq!(result.pages.len(), 1);
    Ok(())
}

#[test]
fn interleaves_pdf_and_image_sources_in_order() -> anyhow::Result<()> {
    let mut engine = Engine::new()?;
    let multi = fixture("multipage.pdf");
    let image = fixture("sample.png");

    // [pdf p1, image, pdf p2] -> a 3-page result in that exact order
    let bytes = engine.assemble_pdf(&[
        PageRef {
            path: &multi,
            page_number: 1,
        },
        PageRef {
            path: &image,
            page_number: 1,
        },
        PageRef {
            path: &multi,
            page_number: 2,
        },
    ])?;

    let result = reextract(&mut engine, &bytes, "interleave");
    assert!(
        result.error.is_none(),
        "re-extract failed: {:?}",
        result.error
    );
    assert_eq!(result.pages.len(), 3);
    // The image (page 2) is the only source carrying "Docsee"; the pdf pages
    // carry "Page text". Order is preserved only if the image landed in the
    // middle.
    let lower: Vec<String> = result.pages.iter().map(|p| p.text.to_lowercase()).collect();
    assert!(
        lower[1].contains("docsee"),
        "image content expected on page 2, got: {:?}",
        result.pages[1].text
    );
    assert!(
        !lower[0].contains("docsee") && !lower[2].contains("docsee"),
        "image content leaked onto a pdf page"
    );
    Ok(())
}

#[test]
fn flattens_alpha_over_white_so_page_is_not_black() -> anyhow::Result<()> {
    let mut engine = Engine::new()?;

    // A transparent PNG would render as dark-on-black (OCRs to nothing) if alpha
    // were dropped instead of composited over white.
    let bytes = engine.assemble_pdf(&[PageRef {
        path: &fixture("transparent.png"),
        page_number: 1,
    }])?;

    let result = reextract(&mut engine, &bytes, "alpha");
    assert!(
        result.error.is_none(),
        "re-extract failed: {:?}",
        result.error
    );
    assert_eq!(result.pages.len(), 1);
    // Text is legible only because the alpha was flattened over white.
    assert!(
        result.pages[0].text.to_lowercase().contains("docsee"),
        "alpha likely not composited over white; text: {:?}",
        result.pages[0].text
    );
    Ok(())
}

#[test]
fn applies_exif_orientation_without_auto_rotate() -> anyhow::Result<()> {
    // Default engine: auto_rotate is OFF. The JPEG is only upright (and thus
    // legible) if the deterministic EXIF orientation is honored during assembly.
    let mut engine = Engine::new()?;

    let bytes = engine.assemble_pdf(&[PageRef {
        path: &fixture("exif_rotated.jpg"),
        page_number: 1,
    }])?;

    let result = reextract(&mut engine, &bytes, "exif");
    assert_eq!(result.pages.len(), 1);
    assert!(
        result.pages[0].text.to_lowercase().contains("brown fox"),
        "EXIF orientation not applied; text: {:?}",
        result.pages[0].text
    );
    Ok(())
}

#[test]
fn routes_image_bytes_saved_under_a_pdf_name() -> anyhow::Result<()> {
    let mut engine = Engine::new()?;

    // Bytes are a PNG but the path ends in .pdf; the content sniff must carry it
    // down the image branch rather than hitting pdfium's FormatError.
    let bytes = engine.assemble_pdf(&[PageRef {
        path: &fixture("image_named.pdf"),
        page_number: 1,
    }])?;

    let result = reextract(&mut engine, &bytes, "misnamed");
    assert!(
        result.error.is_none(),
        "re-extract failed: {:?}",
        result.error
    );
    assert_eq!(result.pages.len(), 1);
    Ok(())
}

#[test]
fn native_pdf_path_is_unchanged() -> anyhow::Result<()> {
    // Regression: a PDF-only assembly still copies natively (text survives, not
    // rasterized) and opens cleanly.
    let mut engine = Engine::new()?;
    let multi = fixture("multipage.pdf");

    let bytes = engine.assemble_pdf(&[
        PageRef {
            path: &multi,
            page_number: 1,
        },
        PageRef {
            path: &multi,
            page_number: 2,
        },
    ])?;

    let result = reextract(&mut engine, &bytes, "native-regress");
    assert!(
        result.error.is_none(),
        "re-extract failed: {:?}",
        result.error
    );
    assert_eq!(result.pages.len(), 2);
    assert_eq!(result.extractor, Some(Extractor::Native));
    assert!(result.pages[0].text.contains("brown fox"));
    Ok(())
}

#[test]
fn corrupt_source_errors_without_panicking() {
    let mut engine = Engine::new().unwrap();

    // Garbage bytes under an image extension: decode fails, but the method stays
    // total — Err, not a panic.
    let bad = std::env::temp_dir().join("docsee-corrupt-source.png");
    std::fs::write(&bad, b"this is not an image").unwrap();

    let result = engine.assemble_pdf(&[PageRef {
        path: &bad,
        page_number: 1,
    }]);
    let _ = std::fs::remove_file(&bad);

    assert!(result.is_err(), "corrupt source should return Err");
}

// The two page-size cases are deliberately separate tests rather than one:
// each builds its own Engine, and two live Engines in one process would each
// hold a Pdfium instance, which deadlocks without the `thread-safe-pdfium`
// feature. Run serially, only one Engine is alive at a time.

#[test]
fn default_assembly_page_size_is_letter_portrait() -> anyhow::Result<()> {
    let mut engine = Engine::new()?;
    let bytes = engine.assemble_pdf(&[PageRef {
        path: &fixture("sample.png"),
        page_number: 1,
    }])?;
    let png = reextract_render(&mut engine, &bytes, "size-portrait")?;
    let (w, h) = png_dimensions(&png);
    assert!(h > w, "default Letter page should be portrait, got {w}x{h}");
    Ok(())
}

#[test]
fn custom_assembly_page_size_controls_page_geometry() -> anyhow::Result<()> {
    // A landscape custom size: rendered page is wider than tall, proving the
    // builder option took effect (vs. the portrait default above).
    let mut engine = Engine::builder()
        .assembly_page_size(PdfPagePaperSize::Custom(
            PdfPoints::new(792.0),
            PdfPoints::new(612.0),
        ))
        .build()?;
    let bytes = engine.assemble_pdf(&[PageRef {
        path: &fixture("sample.png"),
        page_number: 1,
    }])?;
    let png = reextract_render(&mut engine, &bytes, "size-landscape")?;
    let (w, h) = png_dimensions(&png);
    assert!(w > h, "custom page should be landscape, got {w}x{h}");
    Ok(())
}

// Save assembled bytes, reopen, and render page 1 to PNG at 72 DPI so page
// geometry maps 1pt -> 1px.
fn reextract_render(engine: &mut Engine, bytes: &[u8], tag: &str) -> anyhow::Result<Vec<u8>> {
    assert!(bytes.starts_with(b"%PDF"), "assembled output is a PDF");
    let out = std::env::temp_dir().join(format!("docsee-asm-{tag}.pdf"));
    std::fs::write(&out, bytes)?;
    let png = engine.render_pdf_page_png(&out, 1, 72.0);
    let _ = std::fs::remove_file(&out);
    png
}
