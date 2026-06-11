# docsee

Extract text from PDFs and images in Rust — native PDF text via [pdfium](https://pdfium.googlesource.com/pdfium/)
(Chrome's PDF engine), with automatic [tesseract](https://github.com/tesseract-ocr/tesseract) OCR
fallback for scanned documents and images.

**No subprocesses.** Everything runs in-process through linked libraries, so the
pipeline can be embedded in any host application:

| Stage | Library | Linkage |
|-------|---------|---------|
| PDF text extraction | pdfium | `libpdfium` loaded dynamically at startup |
| PDF page rendering for OCR | pdfium | same |
| OCR | tesseract | `libtesseract` + `libleptonica` linked at build time |

## How it works

- **Images** (jpg, png, tiff, bmp — by extension or content sniff) are normalized
  (EXIF orientation applied, alpha flattened, converted to RGB) and OCR'd directly.
- **PDFs** get native text extraction first. If that yields fewer than 50 characters
  per page (configurable), the pages are rendered at 200 DPI (configurable) and OCR'd.
- If native extraction fails outright, OCR is tried as a recovery path.

## Installation

System dependencies first:

```bash
# macOS
brew install tesseract pkg-config
# Debian/Ubuntu
apt-get install libtesseract-dev libleptonica-dev tesseract-ocr-eng pkg-config clang
```

pdfium ships as a prebuilt dynamic library; fetch it for your platform into `./lib/`:

```bash
./get_pdfium.sh
```

Then:

```bash
cargo install docsee        # CLI
cargo add docsee            # library
```

## CLI usage

```bash
docsee document.pdf                   # extracted text to stdout
docsee document.pdf --json            # full extraction record as JSON
docsee scan.pdf --ocr --dpi 300       # force OCR at 300 DPI
docsee brief.pdf --lang deu           # German tessdata
docsee photo.jpg                      # images are OCR'd directly
```

JSON output:

```json
{
  "extractor": "native",
  "n_pages": 3,
  "extracted_chars": 1234,
  "needs_ocr": false,
  "pages": [{"page": 1, "text": "..."}],
  "error": null
}
```

`extractor` is `native` (pdfium text), `ocr_pdf` (rendered pages OCR'd), or
`ocr_image` (image file OCR'd).

## Library usage

```rust
use docsee::{Engine, Extractor};

let mut engine = Engine::builder()
    .ocr_language("eng")      // tesseract language (default "eng")
    .ocr_dpi(200.0)           // render DPI for OCR (default 200)
    .min_chars_per_page(50)   // OCR-fallback threshold (default 50)
    .build()?;

let result = engine.extract(std::path::Path::new("document.pdf"));
match result.extractor {
    Some(Extractor::Native) => println!("digital PDF"),
    Some(Extractor::OcrPdf | Extractor::OcrImage) => println!("OCR'd"),
    None => eprintln!("failed: {:?}", result.error),
}
for page in &result.pages {
    println!("{}", page.text);
}
```

`Engine::new()` is shorthand for all defaults. The lower-level methods
`extract_pdf_native`, `extract_pdf_ocr`, and `extract_image_ocr` are public for
callers that want a single path with no fallback logic, and `extract_with_hint`
lets callers that know the file type (e.g. from a MIME type) skip detection.

## Configuration

| Knob | Where | Effect |
|------|-------|--------|
| `PDFIUM_LIB_DIR` | env | Directory searched for `libpdfium` |
| `TESSDATA_PREFIX` | env | Override tessdata location |
| `.pdfium_lib_dir(...)` | builder | Directory tried for `libpdfium` before all others |
| `.ocr_language(...)` | builder / `--lang` | Tesseract language code |
| `.ocr_dpi(...)` | builder / `--dpi` | PDF render DPI for OCR |
| `.min_chars_per_page(...)` | builder / `--min-chars-per-page` | OCR-fallback threshold |

`libpdfium` search order: the builder override, `$PDFIUM_LIB_DIR`, `./lib` next to
the executable, `./lib` under the cwd, then system library paths.

## Embedding in host processes

The crate is pure Rust with no framework dependencies, so the engine can be embedded
in servers, FFI bridges, or language runtimes. Two pieces of API exist for this:

- `Engine::builder().pdfium_lib_dir(...)` — host processes need an explicit
  `libpdfium` location because the exe-relative search resolves against the host
  binary, not this crate.
- The `thread-safe-pdfium` feature — passthrough to `pdfium-render`'s `thread_safe`
  feature, wrapping every pdfium FFI call in a lock. Enable it when calling the
  engine from more than one OS thread.

Deployment needs `libpdfium` shipped alongside the host and tesseract (lib +
tessdata) installed on it.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
