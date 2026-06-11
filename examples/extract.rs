//! Extract text from a PDF or image: `cargo run --example extract -- file.pdf`

use docsee::Engine;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: extract <pdf-or-image>");

    let mut engine = Engine::builder()
        .ocr_language("eng")
        .ocr_dpi(200.0)
        .min_chars_per_page(50)
        .build()?;

    let result = engine.extract(Path::new(&path));
    if let Some(err) = &result.error {
        anyhow::bail!("extraction failed: {err}");
    }

    eprintln!(
        "extractor: {:?}, pages: {}, chars: {}",
        result.extractor, result.n_pages, result.extracted_chars
    );
    for page in &result.pages {
        println!("--- page {} ---\n{}", page.page, page.text);
    }
    Ok(())
}
