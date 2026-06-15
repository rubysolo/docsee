use docsee::{Engine, Extractor, PageRef};
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
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
