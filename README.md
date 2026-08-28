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
- **Auto-Rotation**: Optional content-based orientation detection. When enabled,
  it tries rotating the image (0, 90, 180, 270 degrees) and picks the one with
  the highest Tesseract confidence. This catches upside-down scans that lack
  EXIF metadata.
- **PDFs** get native text extraction first.
 If that yields fewer than 50 characters
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
docsee scan.jpg --auto-rotate         # detect and fix orientation
docsee scan.pdf --timings             # per-page, per-phase timing table
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
  "error": null,
  "page_rotations": [],
  "timings": [
    {
      "page": 1,
      "extractor": "native",
      "started_us": 91,
      "total_us": 2104,
      "render_us": null,
      "orient_us": null,
      "encode_us": null,
      "ocr_us": null
    }
  ],
  "total_us": 9781
}
```

`extractor` is `native` (pdfium text), `ocr_pdf` (rendered pages OCR'd), or
`ocr_image` (image file OCR'd).

### Where the time went

Every extraction reports what each page cost, by phase. `--timings` prints it
as a table on stderr, so it stays out of the extracted text when stdout is
redirected:

```
$ docsee scan.pdf --auto-rotate --timings > scan.txt
page  extractor      start      total     render     orient     encode        ocr
   1  ocr_pdf         86us      2.31s     41.2ms      1.55s     18.4ms    699.1ms
   2  ocr_pdf        2.31s      2.20s     39.8ms      1.48s     17.9ms    661.3ms

total 4.52s — 4.51s in pages, 8.1ms outside them (document open, the type sniff, any abandoned native attempt)
```

A phase that did not run on a page reads as `-`: a natively-extracted page has
no `ocr`, and `orient` appears only under `--auto-rotate`. That column is the
one to watch — the orientation vote is up to four extra tesseract passes per
page and is usually the largest number on an OCR row, so it is what
`--auto-rotate` actually costs you.

The same numbers are on every `--json` record, as `timings` and `total_us`.

## Library usage

```rust
use docsee::{Engine, Extractor};

let mut engine = Engine::builder()
    .ocr_language("eng")      // tesseract language (default "eng")
    .ocr_dpi(200.0)           // render DPI for OCR (default 200)
    .min_chars_per_page(50)   // OCR-fallback threshold (default 50)
    .auto_rotate(true)        // detect and fix orientation (default false)
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

`Extraction` also carries a `timings` entry per page — `started_us` and
`total_us` plus the `render_us` / `orient_us` / `encode_us` / `ocr_us` phases,
all in microseconds, `None` for a phase that did not run — and `total_us` for
the call as a whole, which additionally covers document open, the type sniff,
and a native attempt that was abandoned for OCR. There is nothing to enable:
reading the clock costs tens of nanoseconds against a tesseract pass measured
in hundreds of milliseconds. `Extraction` is `#[non_exhaustive]`, so build one
from `Extraction::default()` rather than a struct literal.

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
| `.auto_rotate(...)` | builder / `--auto-rotate` | Enable auto-orientation |
| `--timings` | CLI | Print the per-page, per-phase timing table to stderr |

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

One constraint to design around: **there can only be one live `Engine` per
process.** pdfium holds a process-wide lock for the lifetime of a binding, so a
second engine cannot be built until the first is dropped — `build` called from
another thread blocks until then, and `thread-safe-pdfium` does not change that
(it locks FFI calls, not the library init). On a thread that already owns an
engine the wait could never end, since that thread is the only one that could
release the lock, so `build` returns an error there instead of hanging.

Build one engine, keep it, and drop it before building a replacement. Hosts
serving concurrent work should share it behind a mutex rather than pool several
— a pool cannot run in parallel here, it can only queue on construction.

Deployment needs `libpdfium` shipped alongside the host and tesseract (lib +
tessdata) installed on it.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
