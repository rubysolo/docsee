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
