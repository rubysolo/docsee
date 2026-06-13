use docsee::Engine;
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn is_png(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a])
}

#[test]
fn renders_every_pdf_page_to_png() -> anyhow::Result<()> {
    let mut engine = Engine::new()?;
    // High-level entry dispatches a PDF to per-page rendering.
    let pngs = engine.render_pages_png(&fixture("multipage.pdf"), 110.0)?;

    assert!(
        pngs.len() >= 2,
        "multipage.pdf should yield multiple pages, got {}",
        pngs.len()
    );
    assert!(pngs.iter().all(|p| is_png(p)), "every output is a PNG");
    assert!(pngs.iter().all(|p| !p.is_empty()));
    Ok(())
}

#[test]
fn renders_image_file_as_a_single_png() -> anyhow::Result<()> {
    let mut engine = Engine::new()?;
    // An image upload has no PDF to load; it normalizes to one PNG "page".
    let pngs = engine.render_pages_png(&fixture("sample.png"), 110.0)?;
    assert_eq!(pngs.len(), 1);
    assert!(is_png(&pngs[0]));
    Ok(())
}

#[test]
fn renders_a_single_page_to_png() -> anyhow::Result<()> {
    let mut engine = Engine::new()?;
    let png = engine.render_pdf_page_png(&fixture("digital.pdf"), 1, 110.0)?;
    assert!(is_png(&png), "output is a PNG");
    Ok(())
}

#[test]
fn higher_dpi_produces_more_bytes() -> anyhow::Result<()> {
    let mut engine = Engine::new()?;
    let lo = engine.render_pdf_page_png(&fixture("digital.pdf"), 1, 72.0)?;
    let hi = engine.render_pdf_page_png(&fixture("digital.pdf"), 1, 200.0)?;
    assert!(
        hi.len() > lo.len(),
        "200 DPI ({}) should be larger than 72 DPI ({})",
        hi.len(),
        lo.len()
    );
    Ok(())
}
