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

// Re-exported so callers can name the type accepted by
// [`EngineBuilder::assembly_page_size`] without depending on `pdfium-render`.
pub use pdfium_render::prelude::{PdfPagePaperSize, PdfPagePaperStandardSize, PdfPoints};

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
    /// Rotation, in degrees clockwise (0, 90, 180 or 270), that auto-rotate
    /// applied to each page before OCR — in page order, index 0 = page 1.
    ///
    /// Empty when auto-rotate is off, and when the native text path answered
    /// (nothing was rasterized, so nothing was rotated). Detecting orientation
    /// is the dominant cost of an OCR pass — up to four extra tesseract runs per
    /// page — so a caller that renders the same pages afterwards (for
    /// thumbnails, say) should feed these back through
    /// [`Engine::render_pages_png_with_rotations`] rather than pay the vote a
    /// second time for an answer it already has.
    #[serde(default)]
    pub page_rotations: Vec<u32>,
}

/// One page selected from a source PDF for re-packaging via
/// [`Engine::assemble_pdf`].
#[derive(Debug, Clone, Copy)]
pub struct PageRef<'a> {
    /// Path to the source PDF.
    pub path: &'a Path,
    /// 1-based page number within that PDF.
    pub page_number: u16,
}

/// Configures and builds an [`Engine`]. Created via [`Engine::builder`].
#[derive(Debug, Clone, Default)]
pub struct EngineBuilder {
    ocr_language: Option<String>,
    ocr_dpi: Option<f32>,
    min_chars_per_page: Option<usize>,
    pdfium_lib_dir: Option<String>,
    auto_rotate: bool,
    assembly_page_size: Option<PdfPagePaperSize>,
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

    /// Enable automatic orientation detection (OSD-like) via multi-angle
    /// confidence check. This will try rotating the image 0, 90, 180, and 270
    /// degrees and pick the one with highest Tesseract confidence.
    pub fn auto_rotate(mut self, enable: bool) -> Self {
        self.auto_rotate = enable;
        self
    }

    /// Page size used when [`assemble_pdf`](Engine::assemble_pdf) embeds an
    /// image source as a full page (default: US-Letter portrait). Each image is
    /// contain-fit and centered on a page of this size, so it has no effect on
    /// PDF sources, which are copied at their native page size. Use
    /// [`PdfPagePaperSize::Custom`] for an arbitrary size in points.
    pub fn assembly_page_size(mut self, size: PdfPagePaperSize) -> Self {
        self.assembly_page_size = Some(size);
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
            auto_rotate: self.auto_rotate,
            assembly_page_size: self
                .assembly_page_size
                .unwrap_or_else(default_assembly_page_size),
        })
    }
}

pub struct Engine {
    pdfium: Pdfium,
    ocr: LepTess,
    ocr_dpi: f32,
    min_chars_per_page: usize,
    auto_rotate: bool,
    assembly_page_size: PdfPagePaperSize,
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
        let (pages, total, _rotations) = self.extract_pdf_ocr_rotations(path, dpi)?;
        Ok((pages, total))
    }

    /// [`extract_pdf_ocr`](Self::extract_pdf_ocr), also reporting the rotation
    /// auto-rotate applied to each page. Private because the angles reach
    /// callers on [`Extraction::page_rotations`], which is where they are useful.
    fn extract_pdf_ocr_rotations(
        &mut self,
        path: &Path,
        dpi: f32,
    ) -> Result<(Vec<PageText>, usize, Vec<u32>)> {
        let doc = self
            .pdfium
            .load_pdf_from_file(path, None)
            .context("pdfium failed to open document for OCR")?;
        let config = PdfRenderConfig::new().scale_page_by_factor(dpi / 72.0);

        let mut pages = Vec::new();
        let mut rotations = Vec::new();
        let mut total = 0;
        for (i, page) in doc.pages().iter().enumerate() {
            let (png, angle) =
                render_page_to_png(&mut self.ocr, self.auto_rotate, &page, &config, None)
                    .with_context(|| format!("render failed on page {}", i + 1))?;
            let text = ocr_png_bytes(&mut self.ocr, &png, dpi as i32)?;
            total += text.chars().count();
            rotations.push(angle);
            pages.push(PageText {
                page: (i + 1) as u32,
                text,
            });
        }
        Ok((pages, total, rotations))
    }

    /// Render every PDF page to PNG bytes at `dpi`, honoring the engine's
    /// auto-rotate setting. Returned in page order (index 0 = page 1).
    ///
    /// This is the rasterization the OCR path performs internally, surfaced
    /// for callers that need the page image itself (e.g. thumbnails) rather
    /// than its OCR text. `dpi` trades size for sharpness independently of the
    /// OCR DPI; thumbnails typically want less than [`OCR_DPI`].
    pub fn render_pdf_pages_png(&mut self, path: &Path, dpi: f32) -> Result<Vec<Vec<u8>>> {
        self.render_pdf_pages_png_with_rotations(path, dpi, &[])
    }

    /// [`render_pdf_pages_png`](Self::render_pdf_pages_png), reusing rotations
    /// already known for these pages instead of detecting them again.
    ///
    /// `rotations` is in page order (index 0 = page 1) — pass
    /// [`Extraction::page_rotations`] straight through. Pages it doesn't cover
    /// fall back to the engine's normal behavior, so a short or empty slice is
    /// safe and `&[]` is exactly [`render_pdf_pages_png`](Self::render_pdf_pages_png).
    ///
    /// This exists because rendering the same pages twice — once to OCR them,
    /// once for thumbnails — otherwise runs the orientation vote twice, and the
    /// vote is up to four tesseract passes per page. The angle does not depend
    /// on `dpi`, so the second answer can only agree with the first.
    pub fn render_pdf_pages_png_with_rotations(
        &mut self,
        path: &Path,
        dpi: f32,
        rotations: &[u32],
    ) -> Result<Vec<Vec<u8>>> {
        let doc = self
            .pdfium
            .load_pdf_from_file(path, None)
            .context("pdfium failed to open document for render")?;
        let config = PdfRenderConfig::new().scale_page_by_factor(dpi / 72.0);

        let mut out = Vec::with_capacity(doc.pages().len() as usize);
        for (i, page) in doc.pages().iter().enumerate() {
            let known = rotations.get(i).copied();
            let (png, _angle) =
                render_page_to_png(&mut self.ocr, self.auto_rotate, &page, &config, known)
                    .with_context(|| format!("render failed on page {}", i + 1))?;
            out.push(png);
        }
        Ok(out)
    }

    /// Render a file to PNG page images, detecting its type the same way
    /// [`extract`](Self::extract) does: an image file yields a single
    /// normalized PNG (EXIF orientation applied, alpha composited over white);
    /// a PDF yields one PNG per page at `dpi`, auto-rotate honored. Page order;
    /// index 0 = page 1. This is the high-level entry thumbnail callers want —
    /// one call regardless of whether the upload was a scan or a photo.
    pub fn render_pages_png(&mut self, path: &Path, dpi: f32) -> Result<Vec<Vec<u8>>> {
        self.render_pages_png_with_rotations(path, dpi, &[])
    }

    /// [`render_pages_png`](Self::render_pages_png), reusing rotations already
    /// known for these pages — see
    /// [`render_pdf_pages_png_with_rotations`](Self::render_pdf_pages_png_with_rotations).
    ///
    /// This is the entry thumbnail callers want: hand it the
    /// [`Extraction::page_rotations`] from the extract that just ran and the
    /// render costs a rasterization instead of a rasterization plus an
    /// orientation vote. `rotations` is ignored for image sources, which are
    /// normalized rather than rotated (EXIF orientation, alpha flattened).
    pub fn render_pages_png_with_rotations(
        &mut self,
        path: &Path,
        dpi: f32,
        rotations: &[u32],
    ) -> Result<Vec<Vec<u8>>> {
        if has_image_extension(path) || sniffs_as_image(path) {
            Ok(vec![normalize_image_for_ocr_png(path)?])
        } else {
            self.render_pdf_pages_png_with_rotations(path, dpi, rotations)
        }
    }

    /// Render a single 1-based PDF page to PNG bytes at `dpi` (auto-rotate
    /// honored). Useful as a render-on-demand path when only one page is
    /// needed.
    pub fn render_pdf_page_png(
        &mut self,
        path: &Path,
        page_number: u16,
        dpi: f32,
    ) -> Result<Vec<u8>> {
        self.render_pdf_page_png_with_rotation(path, page_number, dpi, None)
    }

    /// [`render_pdf_page_png`](Self::render_pdf_page_png), with the page's
    /// rotation supplied rather than detected.
    ///
    /// `Some(angle)` skips the orientation vote — the render-on-demand
    /// counterpart to
    /// [`render_pdf_pages_png_with_rotations`](Self::render_pdf_pages_png_with_rotations),
    /// for a caller re-rendering one page whose angle it recorded earlier.
    /// `None` behaves exactly like `render_pdf_page_png`.
    pub fn render_pdf_page_png_with_rotation(
        &mut self,
        path: &Path,
        page_number: u16,
        dpi: f32,
        rotation: Option<u32>,
    ) -> Result<Vec<u8>> {
        let doc = self
            .pdfium
            .load_pdf_from_file(path, None)
            .context("pdfium failed to open document for render")?;
        let config = PdfRenderConfig::new().scale_page_by_factor(dpi / 72.0);
        let page = doc
            .pages()
            .get(page_number.saturating_sub(1))
            .with_context(|| format!("no page {page_number} in document"))?;
        let (png, _angle) =
            render_page_to_png(&mut self.ocr, self.auto_rotate, &page, &config, rotation)?;
        Ok(png)
    }

    /// Re-package selected pages from one or more sources into a single new PDF.
    /// Pages appear in the order given and sources may interleave; returns the
    /// new PDF's bytes.
    ///
    /// Sources are detected per [`PageRef`] (by extension or content sniff, the
    /// same way [`extract`](Self::extract) detects them):
    ///
    /// - **PDF source** — the page is copied natively (no rasterization, so text
    ///   and vectors survive) at its original page size.
    /// - **Image source** (PNG/JPG/TIFF/BMP, including image bytes saved under a
    ///   `.pdf` name) — embedded as a full-page image: EXIF orientation applied
    ///   and any alpha composited over white (deterministic; `auto_rotate` is
    ///   *not* applied, since assembly produces a canonical artifact rather than
    ///   maximizing OCR confidence), then contain-fit and centered on a page of
    ///   the configured [`assembly_page_size`](EngineBuilder::assembly_page_size)
    ///   (default US-Letter). `page_number` is ignored for image sources — an
    ///   image is single-page, so the one image is always emitted.
    ///
    /// Each source is reopened per page it contributes — fine for the occasional
    /// assembly this is built for (e.g. a reviewed document spanning uploads);
    /// cache by path if it ever runs hot.
    pub fn assemble_pdf(&mut self, pages: &[PageRef]) -> Result<Vec<u8>> {
        let mut dest = self
            .pdfium
            .create_new_pdf()
            .context("pdfium failed to create the output document")?;

        for page in pages {
            if has_image_extension(page.path) || sniffs_as_image(page.path) {
                append_image_page(&mut dest, page.path, self.assembly_page_size)
                    .with_context(|| format!("embed image source {:?}", page.path))?;
            } else {
                let src = self
                    .pdfium
                    .load_pdf_from_file(page.path, None)
                    .with_context(|| format!("open source pdf {:?}", page.path))?;

                let at = dest.pages().len();
                dest.pages_mut()
                    .copy_page_from_document(&src, page.page_number.saturating_sub(1), at)
                    .with_context(|| {
                        format!("copy page {} of {:?}", page.page_number, page.path)
                    })?;
            }
        }

        dest.save_to_bytes().context("save the assembled pdf")
    }

    /// OCR an image file after normalizing it (EXIF orientation applied,
    /// converted to RGB).
    pub fn extract_image_ocr(&mut self, path: &Path) -> Result<(Vec<PageText>, usize)> {
        let (pages, chars, _rotation) = self.extract_image_ocr_rotation(path)?;
        Ok((pages, chars))
    }

    /// [`extract_image_ocr`](Self::extract_image_ocr), also reporting the
    /// rotation auto-rotate applied. Private for the same reason as
    /// `extract_pdf_ocr_rotations`: the angle reaches callers on
    /// [`Extraction::page_rotations`].
    fn extract_image_ocr_rotation(&mut self, path: &Path) -> Result<(Vec<PageText>, usize, u32)> {
        let (text, chars, rotation) = if self.auto_rotate {
            let image = normalize_image_for_ocr(path).or_else(|_| {
                let bytes = std::fs::read(path).context("read image file")?;
                image::load_from_memory(&bytes).context("load image from memory")
            })?;
            let angle = detect_orientation(&mut self.ocr, &image)?;
            let rotated = rotate_image(image, angle);
            let mut png = Vec::new();
            rotated
                .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
                .context("PNG encode failed")?;
            let text = ocr_bytes(&mut self.ocr, &png)?;
            let chars = text.chars().count();
            (text, chars, angle)
        } else {
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
            (text, chars, 0)
        };
        Ok((vec![PageText { page: 1, text }], chars, rotation))
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
            match self.extract_image_ocr_rotation(path) {
                Ok((pages, chars, rotation)) => {
                    out.extractor = Some(Extractor::OcrImage);
                    out.n_pages = 1;
                    out.extracted_chars = chars;
                    out.pages = pages;
                    out.page_rotations = vec![rotation];
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
            match self.extract_pdf_ocr_rotations(path, self.ocr_dpi) {
                Ok((pages, chars, rotations)) => {
                    out.extractor = Some(Extractor::OcrPdf);
                    out.n_pages = pages.len() as u32;
                    out.extracted_chars = chars;
                    out.pages = pages;
                    out.page_rotations = rotations;
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

/// Render one pdfium page to PNG bytes, applying auto-rotation when enabled.
/// Returns the bytes and the angle applied, so a caller that renders the same
/// page again can reuse the answer instead of re-deriving it.
///
/// `known_angle` short-circuits detection: `Some(a)` rotates by `a` without
/// consulting tesseract at all. Orientation is a property of the page's
/// content, not of the resolution it was rasterized at, so an angle detected
/// during an OCR pass stays valid for a later render at a different DPI — and
/// re-deriving it costs up to four extra OCR runs.
///
/// A free function (rather than a method) on purpose: it borrows `ocr` while
/// the caller still holds the `PdfDocument` borrowed from `self.pdfium`, the
/// same disjoint-field-borrow the OCR paths rely on. `self.method()` would
/// borrow all of `self` and conflict.
fn render_page_to_png(
    ocr: &mut LepTess,
    auto_rotate: bool,
    page: &PdfPage,
    config: &PdfRenderConfig,
    known_angle: Option<u32>,
) -> Result<(Vec<u8>, u32)> {
    let bitmap = page
        .render_with_config(config)
        .context("pdfium render failed")?;
    let mut image = bitmap.as_image();

    let angle = match known_angle {
        Some(angle) => angle,
        None if auto_rotate => detect_orientation(ocr, &image)?,
        None => 0,
    };
    image = rotate_image(image, angle);

    let mut png = Vec::new();
    image
        .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
        .context("PNG encode failed")?;
    Ok((png, angle))
}

/// Default page size for image sources embedded by [`Engine::assemble_pdf`]:
/// US-Letter portrait (612 x 792 pt).
fn default_assembly_page_size() -> PdfPagePaperSize {
    PdfPagePaperSize::new_portrait(PdfPagePaperStandardSize::USLetterAnsiA)
}

/// Append a single image file to `dest` as a full page: normalize (EXIF +
/// alpha-over-white), contain-fit to `page_size`, centered.
///
/// A free function (not a method) so it can borrow `dest` mutably while the
/// caller still holds `self.pdfium` borrowed (`dest` is created from it) — the
/// same disjoint-borrow shape [`render_page_to_png`] relies on.
fn append_image_page(
    dest: &mut PdfDocument,
    path: &Path,
    page_size: PdfPagePaperSize,
) -> Result<()> {
    let img = normalize_image_for_ocr(path)?;

    let page_w = page_size.width().value;
    let page_h = page_size.height().value;

    let (iw, ih) = (img.width() as f32, img.height() as f32);
    // contain: scale to fit inside the page, preserving aspect ratio
    let scale = (page_w / iw).min(page_h / ih);
    let draw_w = iw * scale;
    let draw_h = ih * scale;
    let offset_x = (page_w - draw_w) / 2.0;
    let offset_y = (page_h - draw_h) / 2.0;

    let mut pdf_page = dest
        .pages_mut()
        .create_page_at_end(page_size)
        .context("create image page")?;
    pdf_page
        .objects_mut()
        .create_image_object(
            PdfPoints::new(offset_x),
            PdfPoints::new(offset_y),
            &img,
            Some(PdfPoints::new(draw_w)),
            Some(PdfPoints::new(draw_h)),
        )
        .context("place image on page")?;
    Ok(())
}

/// Rotate by a detected angle (90/180/270); any other value is a no-op.
fn rotate_image(image: DynamicImage, angle: u32) -> DynamicImage {
    match angle {
        90 => image.rotate90(),
        180 => image.rotate180(),
        270 => image.rotate270(),
        _ => image,
    }
}

/// Largest dimension of the image used for the orientation vote
/// (~160 DPI on a US-Letter page).
///
/// The vote reads text to decide which way is up, so the probe has to be
/// legible. At 1000 it isn't, for the documents that most need the answer: a
/// 10x15in scan downscales to 667x1000, where tesseract scores **25-45 at every
/// angle** — pure noise, and the correct angle survives only by winning a
/// tie-break. Measured on a 4-page scanned form, correct-angle confidence by
/// probe size:
///
/// | probe | correct-angle confidence |
/// |-------|--------------------------|
/// | 1000  | 25 / 45 / 35 / 25        |
/// | 1600  | 69 / 72 / 65 / 72        |
/// | 2400  | 69 / 79 / 73 / 77        |
///
/// 1600 is where dense scans clear [`MIN_ORIENTATION_CONFIDENCE`] on their own
/// merits instead of relying on the floor to rescue them; 2400 buys a few more
/// points for another 50% of the time. It costs real work — the vote is four
/// OCR passes and 1600 is 2.56x the pixels of 1000 — but callers that reuse the
/// answer via [`Engine::render_pages_png_with_rotations`] now run the vote once
/// per document instead of once per render, which more than covers it.
const DETECT_MAX_DIM: u32 = 1600;

/// Mean-confidence a non-zero angle must reach before we believe it.
///
/// The vote compares tesseract's `mean_text_conf` across four rotations and
/// takes the best. On a genuinely rotated page the right angle is unmistakable
/// — the `rotated_180` fixture scores 26 / 54 / **95**. On an upright page
/// there is no signal to find and all four land in a noise band, where the
/// winner is arbitrary: a 4-page upright scan measured 35 / 33 / 27 / **42**,
/// handing 270 degrees to a page that was already the right way up.
///
/// So a bare argmax rotates upright documents on a coin flip. Requiring the
/// winner to clear this floor keeps the confident corrections and discards the
/// noise. Set between the two populations, nearer the noise: a real rotation
/// clears it easily, and anything that doesn't was not evidence.
const MIN_ORIENTATION_CONFIDENCE: i32 = 60;

/// Extra confidence a non-zero angle must have *over leaving the page alone*.
///
/// Belt and braces with the floor above, for a page whose every orientation
/// scores well: rotating is only worth it if it is clearly better than not
/// rotating, and 0 degrees is the answer that needs no evidence.
const MIN_ORIENTATION_MARGIN: i32 = 10;

fn detect_orientation(ocr: &mut LepTess, image: &DynamicImage) -> Result<u32> {
    // Orientation is a coarse best-of-four vote; it doesn't need full OCR
    // resolution. Detection is the dominant OCR cost (up to 4 passes per page
    // at OCR DPI), so vote on a downscaled copy — the extraction pass that
    // follows still runs on the full-resolution image.
    let probe = if image.width().max(image.height()) > DETECT_MAX_DIM {
        image.resize(
            DETECT_MAX_DIM,
            DETECT_MAX_DIM,
            image::imageops::FilterType::Triangle,
        )
    } else {
        image.clone()
    };

    let angles = [0, 90, 180, 270];
    let mut best_angle = 0;
    let mut best_conf = -1;
    let mut upright_conf = 0;

    for &angle in &angles {
        let rotated = match angle {
            0 => probe.clone(),
            90 => probe.rotate90(),
            180 => probe.rotate180(),
            270 => probe.rotate270(),
            _ => unreachable!(),
        };

        let mut png = Vec::new();
        rotated
            .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
            .context("PNG encode for OSD check failed")?;

        ocr.set_image_from_mem(&png)
            .context("tesseract set_image for OSD check failed")?;
        let _ = ocr.get_utf8_text().context("OSD check OCR failed")?;
        let conf = ocr.mean_text_conf();

        if angle == 0 {
            upright_conf = conf;
        }

        if conf > best_conf {
            best_conf = conf;
            best_angle = angle;
        }

        // Optimization: if confidence is high enough, we can stop early
        if best_conf > 90 {
            break;
        }
    }

    // Only act on a win that is actually evidence. Rotating is a destructive
    // guess — an upright page turned on its side OCRs to noise, losing exactly
    // the dense small print (serial numbers, dates) that matters most — so the
    // burden of proof sits on rotating, not on leaving the page alone.
    if best_angle != 0
        && (best_conf < MIN_ORIENTATION_CONFIDENCE
            || best_conf - upright_conf < MIN_ORIENTATION_MARGIN)
    {
        return Ok(0);
    }

    Ok(best_angle)
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
    let image = normalize_image_for_ocr(path)?;
    let mut png = Vec::new();
    image
        .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
        .context("PNG re-encode failed")?;
    Ok(png)
}

fn normalize_image_for_ocr(path: &Path) -> Result<DynamicImage> {
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
    Ok(DynamicImage::ImageRgb8(rgb))
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
