//! In-process text extraction from PDFs and images.
//!
//! PDFs get native text extraction first; pages that yield too little text
//! fall back to OCR. Images go straight to OCR. No subprocesses: PDF
//! parsing/rendering goes through libpdfium (dynamically loaded), OCR goes
//! through libtesseract/libleptonica (linked at build time), so the whole
//! pipeline can be embedded in any host process.
//!
//! ```no_run
//! use docsee::Engine;
//!
//! # fn main() -> anyhow::Result<()> {
//! let mut engine = Engine::new()?;
//! let result = engine.extract(std::path::Path::new("document.pdf"));
//! for page in &result.pages {
//!     println!("--- page {} ---\n{}", page.page, page.text);
//! }
//! # Ok(())
//! # }
//! ```

use anyhow::{Context, Result};
use image::{DynamicImage, ImageDecoder, ImageReader};
use leptess::LepTess;
use pdfium_render::prelude::*;
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use std::path::Path;

/// Default DPI at which PDF pages are rendered before OCR.
pub const OCR_DPI: f32 = 200.0;
/// Default OCR-fallback threshold: native extraction yielding fewer
/// characters per page than this is sent to OCR.
pub const MIN_CHARS_PER_PAGE: usize = 50;
/// Default tesseract language.
pub const OCR_LANGUAGE: &str = "eng";

/// Which extraction path produced the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Extractor {
    /// Native PDF text extraction via pdfium (no OCR).
    Native,
    /// PDF pages rendered and OCR'd with tesseract.
    OcrPdf,
    /// Image file OCR'd directly with tesseract.
    OcrImage,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PageText {
    pub page: u32,
    pub text: String,
}

/// Result of extracting one file. `extractor` is `None` when no extraction
/// path succeeded (see `error`).
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Extraction {
    pub extractor: Option<Extractor>,
    pub n_pages: u32,
    pub extracted_chars: usize,
    pub needs_ocr: bool,
    pub pages: Vec<PageText>,
    pub error: Option<String>,
}

/// Configures and builds an [`Engine`]. Created via [`Engine::builder`].
#[derive(Debug, Clone, Default)]
pub struct EngineBuilder {
    ocr_language: Option<String>,
    ocr_dpi: Option<f32>,
    min_chars_per_page: Option<usize>,
    pdfium_lib_dir: Option<String>,
}

impl EngineBuilder {
    /// Tesseract language code, e.g. `"eng"`, `"deu"` (default: [`OCR_LANGUAGE`]).
    /// The matching tessdata pack must be installed.
    pub fn ocr_language(mut self, lang: impl Into<String>) -> Self {
        self.ocr_language = Some(lang.into());
        self
    }

    /// DPI at which PDF pages are rendered before OCR (default: [`OCR_DPI`]).
    pub fn ocr_dpi(mut self, dpi: f32) -> Self {
        self.ocr_dpi = Some(dpi);
        self
    }

    /// OCR-fallback threshold in characters per page (default:
    /// [`MIN_CHARS_PER_PAGE`]). Native extraction yielding less falls back to OCR.
    pub fn min_chars_per_page(mut self, chars: usize) -> Self {
        self.min_chars_per_page = Some(chars);
        self
    }

    /// Directory tried for `libpdfium` before all other candidates. Embedders
    /// need this: the exe-relative search resolves against the host process's
    /// binary, not this crate.
    pub fn pdfium_lib_dir(mut self, dir: impl Into<String>) -> Self {
        self.pdfium_lib_dir = Some(dir.into());
        self
    }

    /// Bind to libpdfium and initialize a tesseract instance.
    ///
    /// pdfium search order: the [`pdfium_lib_dir`](Self::pdfium_lib_dir)
    /// override, `$PDFIUM_LIB_DIR`, `./lib` next to the executable, `./lib`
    /// under the cwd, then the system library path. tesseract uses the default
    /// tessdata location (`$TESSDATA_PREFIX` to override).
    pub fn build(self) -> Result<Engine> {
        let mut candidates: Vec<String> = Vec::new();
        if let Some(dir) = &self.pdfium_lib_dir {
            candidates.push(dir.clone());
        }
        if let Ok(dir) = std::env::var("PDFIUM_LIB_DIR") {
            candidates.push(dir);
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                candidates.push(dir.join("lib").display().to_string());
                // target/{debug,release}/binary -> sibling of target/
                candidates.push(dir.join("../../lib").display().to_string());
            }
        }
        candidates.push("./lib".to_string());

        let mut bindings = None;
        for dir in &candidates {
            if let Ok(b) =
                Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path(dir))
            {
                bindings = Some(b);
                break;
            }
        }
        let bindings = match bindings {
            Some(b) => b,
            None => Pdfium::bind_to_system_library().with_context(|| {
                format!(
                    "libpdfium not found; checked {:?} and system paths. Run get_pdfium.sh",
                    candidates
                )
            })?,
        };

        let language = self.ocr_language.as_deref().unwrap_or(OCR_LANGUAGE);
        let ocr = LepTess::new(None, language).with_context(|| {
            format!(
                "tesseract init failed for language {language:?} \
                 (is tessdata installed? set TESSDATA_PREFIX)"
            )
        })?;

        Ok(Engine {
            pdfium: Pdfium::new(bindings),
            ocr,
            ocr_dpi: self.ocr_dpi.unwrap_or(OCR_DPI),
            min_chars_per_page: self.min_chars_per_page.unwrap_or(MIN_CHARS_PER_PAGE),
        })
    }
}

pub struct Engine {
    pdfium: Pdfium,
    ocr: LepTess,
    ocr_dpi: f32,
    min_chars_per_page: usize,
}

impl Engine {
    /// An engine with all defaults; shorthand for `Engine::builder().build()`.
    pub fn new() -> Result<Self> {
        Self::builder().build()
    }

    /// Configure OCR language, DPI, fallback threshold, or the libpdfium
    /// location before building.
    pub fn builder() -> EngineBuilder {
        EngineBuilder::default()
    }

    /// Native (non-OCR) text extraction via pdfium.
    pub fn extract_pdf_native(&mut self, path: &Path) -> Result<(Vec<PageText>, usize)> {
        let doc = self
            .pdfium
            .load_pdf_from_file(path, None)
            .context("pdfium failed to open document")?;
        let mut pages = Vec::new();
        let mut total = 0;
        for (i, page) in doc.pages().iter().enumerate() {
            // pdfium emits \r\n line breaks; normalize to \n
            let text = page
                .text()
                .map(|t| t.all().replace("\r\n", "\n").replace('\r', "\n"))
                .unwrap_or_default();
            total += text.chars().count();
            pages.push(PageText {
                page: (i + 1) as u32,
                text,
            });
        }
        Ok((pages, total))
    }

    /// Render each PDF page via pdfium and OCR the bitmap with tesseract.
    pub fn extract_pdf_ocr(&mut self, path: &Path, dpi: f32) -> Result<(Vec<PageText>, usize)> {
        let doc = self
            .pdfium
            .load_pdf_from_file(path, None)
            .context("pdfium failed to open document for OCR")?;
        let config = PdfRenderConfig::new().scale_page_by_factor(dpi / 72.0);

        let mut pages = Vec::new();
        let mut total = 0;
        for (i, page) in doc.pages().iter().enumerate() {
            let bitmap = page
                .render_with_config(&config)
                .with_context(|| format!("pdfium render failed on page {}", i + 1))?;
            let image = bitmap.as_image();

            let mut png = Vec::new();
            image
                .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
                .context("PNG encode failed")?;

            let text = ocr_png_bytes(&mut self.ocr, &png, dpi as i32)?;
            total += text.chars().count();
            pages.push(PageText {
                page: (i + 1) as u32,
                text,
            });
        }
        Ok((pages, total))
    }

    /// OCR an image file after normalizing it (EXIF orientation applied,
    /// converted to RGB).
    pub fn extract_image_ocr(&mut self, path: &Path) -> Result<(Vec<PageText>, usize)> {
        let text = match normalize_image_for_ocr_png(path) {
            Ok(png) => ocr_bytes(&mut self.ocr, &png)?,
            Err(normalize_err) => {
                let bytes = std::fs::read(path).context("read image file")?;
                ocr_bytes(&mut self.ocr, &bytes).with_context(|| {
                    format!("image normalization failed first: {normalize_err:?}")
                })?
            }
        };
        let chars = text.chars().count();
        Ok((vec![PageText { page: 1, text }], chars))
    }

    /// Extract one file, detecting its type automatically: images (by
    /// extension or content sniff) go straight to OCR; anything else is
    /// treated as a PDF — native text extraction first, falling back to OCR
    /// when the result is empty/low-text.
    pub fn extract(&mut self, path: &Path) -> Extraction {
        self.extract_with_hint(path, has_image_extension(path))
    }

    /// Like [`extract`](Self::extract), but the caller asserts the file type:
    /// `treat_as_image: true` forces the image OCR path (useful when the type
    /// is known from a MIME type rather than the file name). Content sniffing
    /// still catches images with misleading names.
    pub fn extract_with_hint(&mut self, path: &Path, treat_as_image: bool) -> Extraction {
        let mut out = Extraction::default();

        if treat_as_image || sniffs_as_image(path) {
            match self.extract_image_ocr(path) {
                Ok((pages, chars)) => {
                    out.extractor = Some(Extractor::OcrImage);
                    out.n_pages = 1;
                    out.extracted_chars = chars;
                    out.pages = pages;
                }
                Err(e) => out.error = Some(format!("image_ocr_failed: {e:?}")),
            }
            return out;
        }

        match self.extract_pdf_native(path) {
            Ok((pages, chars)) => {
                out.extractor = Some(Extractor::Native);
                out.n_pages = pages.len() as u32;
                out.extracted_chars = chars;
                out.needs_ocr = chars < self.min_chars_per_page * pages.len().max(1);
                out.pages = pages;
            }
            Err(e) => {
                out.error = Some(format!("pdf_native_failed: {e:?}"));
                out.needs_ocr = true; // try OCR as a recovery path
            }
        }

        if out.needs_ocr {
            match self.extract_pdf_ocr(path, self.ocr_dpi) {
                Ok((pages, chars)) => {
                    out.extractor = Some(Extractor::OcrPdf);
                    out.n_pages = pages.len() as u32;
                    out.extracted_chars = chars;
                    out.pages = pages;
                    out.needs_ocr = false;
                    out.error = None; // OCR rescued us
                }
                Err(e) => {
                    let prior = out.error.take().unwrap_or_default();
                    let sep = if prior.is_empty() { "" } else { " ; " };
                    out.error = Some(format!("{prior}{sep}ocr_failed: {e:?}"));
                }
            }
        }

        out
    }
}

fn ocr_png_bytes(ocr: &mut LepTess, png: &[u8], dpi: i32) -> Result<String> {
    ocr.set_image_from_mem(png)
        .context("tesseract set_image failed")?;
    ocr.set_source_resolution(dpi);
    ocr.get_utf8_text().context("tesseract OCR failed")
}

fn ocr_bytes(ocr: &mut LepTess, bytes: &[u8]) -> Result<String> {
    ocr.set_image_from_mem(bytes)
        .context("tesseract set_image failed")?;
    ocr.get_utf8_text().context("tesseract OCR failed")
}

fn normalize_image_for_ocr_png(path: &Path) -> Result<Vec<u8>> {
    let reader = ImageReader::open(path)
        .context("open image file")?
        .with_guessed_format()
        .context("guess image format")?;
    let mut decoder = reader.into_decoder().context("create image decoder")?;
    let orientation = decoder.orientation().context("read image orientation")?;

    let mut image = DynamicImage::from_decoder(decoder).context("decode image")?;
    image.apply_orientation(orientation);

    // Composite alpha over white: dropping the channel would leave dark text
    // on a dark background for transparent images, which OCRs to nothing.
    let rgb = if image.color().has_alpha() {
        let mut canvas = image::RgbaImage::from_pixel(
            image.width(),
            image.height(),
            image::Rgba([255, 255, 255, 255]),
        );
        image::imageops::overlay(&mut canvas, &image.to_rgba8(), 0, 0);
        DynamicImage::ImageRgba8(canvas).to_rgb8()
    } else {
        image.to_rgb8()
    };

    let mut png = Vec::new();
    DynamicImage::ImageRgb8(rgb)
        .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
        .context("PNG re-encode failed")?;
    Ok(png)
}

fn has_image_extension(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .as_deref(),
        Some("jpg" | "jpeg" | "png" | "tif" | "tiff" | "bmp")
    )
}

/// Cheap content sniff for image files that arrive without an image
/// extension/MIME.
fn sniffs_as_image(path: &Path) -> bool {
    let mut buf = [0u8; 16];
    let n = match std::fs::File::open(path) {
        Ok(mut f) => std::io::Read::read(&mut f, &mut buf).unwrap_or(0),
        Err(_) => return false,
    };
    image::guess_format(&buf[..n]).is_ok()
}
