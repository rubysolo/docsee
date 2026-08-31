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

use anyhow::{bail, Context, Result};
use image::{DynamicImage, ImageDecoder, ImageReader};
use leptess::LepTess;
use pdfium_render::prelude::*;
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::io::Cursor;
use std::path::Path;
use std::sync::mpsc;
use std::sync::Mutex;
use std::time::Instant;

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

/// Wall-clock cost of one page's extraction, by phase. Phases that did not run
/// on this page are `None` — a natively-extracted page has no `ocr`, and a page
/// whose rotation was supplied by the caller has no `orient`.
///
/// The phases do not have to add up to `total_us`: they cover the expensive
/// work, not every instruction between them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageTiming {
    pub page: u32,
    /// Which path produced this page's text.
    pub extractor: Extractor,
    /// Offset from the start of the extraction call, in microseconds.
    pub started_us: u64,
    /// Total for this page, in microseconds.
    pub total_us: u64,
    /// pdfium rasterization to an image. On the image path, the normalization
    /// that stands in for it (decode, EXIF, alpha-over-white).
    pub render_us: Option<u64>,
    /// Orientation vote ([`EngineBuilder::auto_rotate`]) — up to four
    /// tesseract passes, and usually the dominant cost of an OCR page.
    pub orient_us: Option<u64>,
    /// PNG encode of the page image, including the rotation applied to it.
    pub encode_us: Option<u64>,
    /// The tesseract pass that produced the text.
    pub ocr_us: Option<u64>,
}

/// Result of extracting one file. `extractor` is `None` when no extraction
/// path succeeded (see `error`).
#[derive(Debug, Default, Serialize, Deserialize)]
#[non_exhaustive]
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
    /// Per-page cost breakdown, in page order. Always populated by
    /// [`Engine::extract`] and [`Engine::extract_with_hint`].
    ///
    /// This is what makes a slow file legible: an OCR page splits into
    /// rasterization, the orientation vote, the PNG encode and the tesseract
    /// pass, which differ by an order of magnitude and have completely
    /// different fixes.
    #[serde(default)]
    pub timings: Vec<PageTiming>,
    /// Total wall clock for the whole `extract` call, in microseconds.
    /// Includes document open, the type sniff, and — when the native path was
    /// tried and rejected — the cost of that abandoned attempt, which no page
    /// timing covers. So `sum(timings.total_us) <= total_us`, and the
    /// difference is the per-document overhead.
    #[serde(default)]
    pub total_us: u64,
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

thread_local! {
    /// Set while this thread owns a live [`Engine`].
    ///
    /// pdfium holds a process-wide lock for the lifetime of a binding, so a
    /// second engine cannot be built until the first is dropped. From another
    /// thread that is an ordinary wait; on a thread that already owns one it
    /// never ends, because the only thread that could release the lock is the
    /// one blocked on it. This flag lets [`EngineBuilder::build`] recognize
    /// that case and fail instead of hanging.
    ///
    /// A thread-local is sound here precisely because `Engine` is `!Send`
    /// (pdfium's bindings are, with or without `thread-safe-pdfium`): an engine
    /// is always dropped on the thread that built it, so the flag cannot drift.
    static ENGINE_LIVE: Cell<bool> = const { Cell::new(false) };
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
    orientation_probe_dim: Option<u32>,
    ocr_threads: Option<usize>,
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

    /// Largest dimension, in pixels, of the downscaled copy the orientation
    /// vote runs on (default: [`DETECT_MAX_DIM`]). No effect unless
    /// [`auto_rotate`](Self::auto_rotate) is on.
    ///
    /// This trades detection accuracy against time, and the trade is steep in
    /// both directions. The vote reads text to decide which way is up, so too
    /// small and it is voting on an illegible image — a 10x15in scan at 1000px
    /// scores 25-45 at every angle, which is noise. Too large and you pay for
    /// it four times over, once per candidate angle.
    ///
    /// Lower it if your inputs are large-print or you have already established
    /// that orientation is reliable at a smaller probe; raise it if you handle
    /// dense small print. Values below a few hundred pixels make the vote
    /// meaningless for ordinary documents.
    pub fn orientation_probe_dim(mut self, px: u32) -> Self {
        self.orientation_probe_dim = Some(px);
        self
    }

    /// How many pages of a PDF may be OCR'd at once (default:
    /// [`default_ocr_threads`], the machine's parallelism capped at 4).
    ///
    /// Rasterizing a page needs pdfium and is serialized regardless, but it is
    /// a small share of an OCR'd page: the orientation vote and the recognition
    /// pass are tesseract, they dominate, and tesseract is single-threaded. So
    /// an extract uses one core however many the machine has. Above 1, pages
    /// are rendered in order on the calling thread and recognized on a pool of
    /// this many independent tesseract instances. The calling thread only
    /// rasterizes, so this is the recognition width, not a total thread count.
    ///
    /// Costs an OCR instance per worker — roughly 30ms to build and ~55MB of
    /// language model held resident — and holds up to `ocr_threads` rendered
    /// page images in memory at once. That is why the pool is built on the
    /// first multi-threaded OCR of a PDF rather than at
    /// [`build`](Self::build): an engine that only ever handles image sources,
    /// native-text PDFs, or renders pays none of it.
    ///
    /// Set it to the cores you are willing to spend. Going wider than the
    /// machine buys nothing, because the work is CPU-bound — and on a host that
    /// is also serving traffic, spending every core on an extraction is a
    /// choice about latency elsewhere, which is why this is a knob and not an
    /// assumption. `1` restores strictly sequential recognition.
    ///
    /// Page order is preserved in the result no matter which worker finishes
    /// first. Per-page timings are still each page's own, so with a pool their
    /// `started_us` offsets overlap — that is the parallelism, visible.
    pub fn ocr_threads(mut self, threads: usize) -> Self {
        self.ocr_threads = Some(threads);
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
    ///
    /// Errors if this thread already has a live [`Engine`] — see that type for
    /// why building a second one there could only hang.
    pub fn build(self) -> Result<Engine> {
        if ENGINE_LIVE.get() {
            bail!(
                "this thread already has a live Engine; pdfium holds a \
                 process-wide lock for the lifetime of a binding, so building a \
                 second one here would block until the first is dropped — which \
                 this thread cannot do while it is blocked. Drop the existing \
                 Engine first, or share it rather than building another."
            );
        }

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

        let ocr_threads = self.ocr_threads.unwrap_or_else(default_ocr_threads).max(1);

        // Last: every `?` above leaves the claim unset, so a failed build does
        // not lock this thread out of trying again.
        ENGINE_LIVE.set(true);
        Ok(Engine {
            pdfium: Pdfium::new(bindings),
            ocr,
            ocr_dpi: self.ocr_dpi.unwrap_or(OCR_DPI),
            min_chars_per_page: self.min_chars_per_page.unwrap_or(MIN_CHARS_PER_PAGE),
            auto_rotate: self.auto_rotate,
            assembly_page_size: self
                .assembly_page_size
                .unwrap_or_else(default_assembly_page_size),
            // Clamped to at least 1: `image::resize` to a zero dimension panics,
            // and a caller passing 0 means "as small as possible", not "crash".
            orientation_probe_dim: self.orientation_probe_dim.unwrap_or(DETECT_MAX_DIM).max(1),
            ocr_threads,
            ocr_language: language.to_string(),
            ocr_pool: Vec::new(),
        })
    }
}

/// What one OCR run over a document produced. Internal: its parts reach
/// callers on [`Extraction`], which is where they are useful.
struct OcrRun {
    pages: Vec<PageText>,
    chars: usize,
    rotations: Vec<u32>,
    timings: Vec<PageTiming>,
}

/// A bound libpdfium plus a live tesseract instance.
///
/// **One live engine per process.** pdfium holds a process-wide lock for the
/// lifetime of a binding, so a second engine cannot be built until the first is
/// dropped. From another thread [`EngineBuilder::build`] simply blocks until
/// that happens; on a thread that already owns an engine it would block forever
/// — the only thread that could release the lock is the one waiting on it — so
/// `build` recognizes that case and returns an error instead of hanging. The
/// `thread-safe-pdfium` feature does not change any of this: it wraps FFI calls
/// in a lock, not the library init.
///
/// Build an engine, keep it, and drop it before building a replacement. A host
/// serving concurrent work should share one engine behind its own mutex (the
/// extraction methods take `&mut self` anyway) rather than pool several — a
/// pool cannot run in parallel here, it can only queue on construction.
pub struct Engine {
    pdfium: Pdfium,
    ocr: LepTess,
    ocr_dpi: f32,
    min_chars_per_page: usize,
    auto_rotate: bool,
    assembly_page_size: PdfPagePaperSize,
    orientation_probe_dim: u32,
    /// How wide recognition may go; see [`EngineBuilder::ocr_threads`].
    ocr_threads: usize,
    /// Kept so the pool can be built after construction.
    ocr_language: String,
    /// Recognition instances, one per worker. Empty until the first
    /// multi-threaded OCR of a PDF actually needs them.
    ocr_pool: Vec<SendTess>,
}

/// A recognition instance that may be moved to a worker thread.
///
/// `LepTess` holds raw tesseract/leptonica pointers and so is `!Send`. Moving
/// one is sound here for a specific reason: neither library pins state to the
/// thread that created it (no TLS), instances share nothing with each other,
/// and each is owned exclusively by one worker for the life of a scope. This
/// is emphatically not a claim that a single instance is thread-safe — it is
/// that separate instances are independent.
struct SendTess(LepTess);

// SAFETY: see `SendTess`.
unsafe impl Send for SendTess {}

impl Drop for Engine {
    /// Releases this thread's claim. Dropping the fields is what releases
    /// pdfium's own lock, and so what actually unblocks anyone waiting on it.
    fn drop(&mut self) {
        ENGINE_LIVE.set(false);
    }
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
        let (pages, total, _timings) = self.extract_pdf_native_timed(path, Instant::now())?;
        Ok((pages, total))
    }

    /// [`extract_pdf_native`](Self::extract_pdf_native), also reporting what
    /// each page cost.
    ///
    /// `start` is the instant the whole extraction began, so
    /// [`PageTiming::started_us`] places each page within that call rather than
    /// within this method. The document open that precedes the loop is
    /// deliberately outside every page timing; it shows up only in the
    /// difference against [`Extraction::total_us`].
    fn extract_pdf_native_timed(
        &mut self,
        path: &Path,
        start: Instant,
    ) -> Result<(Vec<PageText>, usize, Vec<PageTiming>)> {
        let doc = self
            .pdfium
            .load_pdf_from_file(path, None)
            .context("pdfium failed to open document")?;
        let mut pages = Vec::new();
        let mut timings = Vec::new();
        let mut total = 0;
        for (i, page) in doc.pages().iter().enumerate() {
            let started_us = elapsed_us(start);
            let page_start = Instant::now();
            // pdfium emits \r\n line breaks; normalize to \n
            let text = page
                .text()
                .map(|t| t.all().replace("\r\n", "\n").replace('\r', "\n"))
                .unwrap_or_default();
            total += text.chars().count();
            timings.push(PageTiming {
                page: (i + 1) as u32,
                extractor: Extractor::Native,
                started_us,
                total_us: elapsed_us(page_start),
                render_us: None,
                orient_us: None,
                encode_us: None,
                ocr_us: None,
            });
            pages.push(PageText {
                page: (i + 1) as u32,
                text,
            });
        }
        Ok((pages, total, timings))
    }

    /// Render each PDF page via pdfium and OCR the bitmap with tesseract.
    pub fn extract_pdf_ocr(&mut self, path: &Path, dpi: f32) -> Result<(Vec<PageText>, usize)> {
        let run = self.extract_pdf_ocr_run(path, dpi, Instant::now())?;
        Ok((run.pages, run.chars))
    }

    /// [`extract_pdf_ocr`](Self::extract_pdf_ocr), also reporting the rotation
    /// auto-rotate applied to each page and what each page cost. Private
    /// because both reach callers on [`Extraction`], which is where they are
    /// useful. `start` carries the same meaning as in
    /// [`extract_pdf_native_timed`](Self::extract_pdf_native_timed).
    fn extract_pdf_ocr_run(&mut self, path: &Path, dpi: f32, start: Instant) -> Result<OcrRun> {
        self.ensure_ocr_pool()?;
        if self.ocr_pool.is_empty() {
            self.extract_pdf_ocr_serial(path, dpi, start)
        } else {
            self.extract_pdf_ocr_pooled(path, dpi, start)
        }
    }

    /// Build the recognition pool if this engine is allowed one and has not
    /// built it yet.
    ///
    /// Deferred to the first OCR'd PDF because the cost is real — an instance
    /// per worker, each ~30ms and ~55MB — and most work never reaches it: image
    /// sources are single-page, native-text PDFs never recognize anything, and
    /// renders reuse a known angle.
    ///
    /// Built sequentially. Instances are independent once live, but whether
    /// `TessBaseAPIInit` may be raced is not a question worth having an opinion
    /// about, and this happens once per engine.
    fn ensure_ocr_pool(&mut self) -> Result<()> {
        if self.ocr_threads <= 1 || !self.ocr_pool.is_empty() {
            return Ok(());
        }
        // One instance per worker. The calling thread rasterizes and does not
        // recognize, so this is `ocr_threads`, not `ocr_threads - 1` — sizing it
        // one short would silently make `ocr_threads(2)` no wider than serial.
        for _ in 0..self.ocr_threads {
            let language = self.ocr_language.clone();
            self.ocr_pool
                .push(SendTess(LepTess::new(None, &language).with_context(
                    || {
                        format!(
                            "tesseract init failed for language {language:?} building the OCR pool"
                        )
                    },
                )?));
        }
        Ok(())
    }

    /// One page at a time on the calling thread. What the engine did before
    /// [`EngineBuilder::ocr_threads`], and what `ocr_threads(1)` restores.
    fn extract_pdf_ocr_serial(&mut self, path: &Path, dpi: f32, start: Instant) -> Result<OcrRun> {
        let doc = self
            .pdfium
            .load_pdf_from_file(path, None)
            .context("pdfium failed to open document for OCR")?;
        let config = PdfRenderConfig::new().scale_page_by_factor(dpi / 72.0);

        let mut pages = Vec::new();
        let mut rotations = Vec::new();
        let mut timings = Vec::new();
        let mut total = 0;
        for (i, page) in doc.pages().iter().enumerate() {
            let started_us = elapsed_us(start);
            let page_start = Instant::now();
            let rendered = render_page_to_png(
                &mut self.ocr,
                self.auto_rotate,
                &page,
                &config,
                None,
                self.orientation_probe_dim,
            )
            .with_context(|| format!("render failed on page {}", i + 1))?;

            let ocr_start = Instant::now();
            let text = ocr_png_bytes(&mut self.ocr, &rendered.png, dpi as i32)?;
            let ocr_us = elapsed_us(ocr_start);

            total += text.chars().count();
            rotations.push(rendered.angle);
            timings.push(PageTiming {
                page: (i + 1) as u32,
                extractor: Extractor::OcrPdf,
                started_us,
                total_us: elapsed_us(page_start),
                render_us: Some(rendered.render_us),
                orient_us: rendered.orient_us,
                encode_us: Some(rendered.encode_us),
                ocr_us: Some(ocr_us),
            });
            pages.push(PageText {
                page: (i + 1) as u32,
                text,
            });
        }
        Ok(OcrRun {
            pages,
            chars: total,
            rotations,
            timings,
        })
    }

    /// Rasterize on this thread, recognize on the pool.
    ///
    /// The split is where the work actually is: rasterizing needs pdfium and
    /// cannot be shared, but it is a small share of an OCR'd page — the
    /// orientation vote and the recognition pass are tesseract and dominate.
    /// So pages are rendered here in order and handed to workers that own an
    /// instance each.
    ///
    /// The hand-off channel is bounded by the pool size, which is what keeps
    /// memory flat: at most one rendered page per worker is ever waiting, so a
    /// long document costs no more than a short one. Results carry their page
    /// index and are reordered at the end, so output order never depends on
    /// which worker won.
    fn extract_pdf_ocr_pooled(&mut self, path: &Path, dpi: f32, start: Instant) -> Result<OcrRun> {
        let Self {
            pdfium,
            ocr_pool,
            auto_rotate,
            orientation_probe_dim,
            ..
        } = self;
        let (auto_rotate, probe_dim) = (*auto_rotate, *orientation_probe_dim);

        let doc = pdfium
            .load_pdf_from_file(path, None)
            .context("pdfium failed to open document for OCR")?;
        let config = PdfRenderConfig::new().scale_page_by_factor(dpi / 72.0);

        // (index, rendered page, offset when its work began, rasterize cost)
        type Job = (usize, DynamicImage, u64, u64);
        let (tx_job, rx_job) = mpsc::sync_channel::<Job>(ocr_pool.len());
        let (tx_done, rx_done) = mpsc::channel::<Result<PageWork>>();
        let rx_job = Mutex::new(rx_job);

        let mut work = std::thread::scope(|scope| -> Result<Vec<PageWork>> {
            for tess in ocr_pool.iter_mut() {
                let rx_job = &rx_job;
                let tx_done = tx_done.clone();
                scope.spawn(move || {
                    // The lock is released before the page is recognized: it
                    // guards taking the next job, not doing it.
                    while let Ok(job) = {
                        let next = rx_job.lock().expect("job queue poisoned").recv();
                        next
                    } {
                        let (index, image, started_us, render_us) = job;
                        let done = ocr_one_page(
                            &mut tess.0,
                            index,
                            image,
                            started_us,
                            render_us,
                            auto_rotate,
                            probe_dim,
                            dpi,
                        );
                        if tx_done.send(done).is_err() {
                            break;
                        }
                    }
                });
            }
            // Only the workers may hold a sender past this point, or draining
            // below would wait on a sender that will never send.
            drop(tx_done);

            let mut render_err = None;
            for (i, page) in doc.pages().iter().enumerate() {
                let started_us = elapsed_us(start);
                let render_start = Instant::now();
                match page.render_with_config(&config) {
                    Ok(bitmap) => {
                        let image = bitmap.as_image();
                        let render_us = elapsed_us(render_start);
                        if tx_job.send((i, image, started_us, render_us)).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        render_err = Some(
                            anyhow::Error::new(e)
                                .context(format!("render failed on page {}", i + 1)),
                        );
                        break;
                    }
                }
            }
            drop(tx_job);

            // Drained before reporting a render failure so no worker is left
            // blocked on a send into a channel nobody is reading.
            let mut out = Vec::new();
            for done in rx_done {
                out.push(done?);
            }
            if let Some(err) = render_err {
                return Err(err);
            }
            Ok(out)
        })?;

        work.sort_by_key(|w| w.index);

        let chars = work.iter().map(|w| w.text.chars().count()).sum();
        let rotations = work.iter().map(|w| w.angle).collect();
        let timings = work
            .iter()
            .map(|w| PageTiming {
                page: (w.index + 1) as u32,
                extractor: Extractor::OcrPdf,
                started_us: w.started_us,
                total_us: w.total_us,
                render_us: Some(w.render_us),
                orient_us: w.orient_us,
                encode_us: Some(w.encode_us),
                ocr_us: Some(w.ocr_us),
            })
            .collect();
        let pages = work
            .into_iter()
            .map(|w| PageText {
                page: (w.index + 1) as u32,
                text: w.text,
            })
            .collect();

        Ok(OcrRun {
            pages,
            chars,
            rotations,
            timings,
        })
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
            let rendered = render_page_to_png(
                &mut self.ocr,
                self.auto_rotate,
                &page,
                &config,
                known,
                self.orientation_probe_dim,
            )
            .with_context(|| format!("render failed on page {}", i + 1))?;
            out.push(rendered.png);
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
        let rendered = render_page_to_png(
            &mut self.ocr,
            self.auto_rotate,
            &page,
            &config,
            rotation,
            self.orientation_probe_dim,
        )?;
        Ok(rendered.png)
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
        let run = self.extract_image_ocr_run(path, Instant::now())?;
        Ok((run.pages, run.chars))
    }

    /// [`extract_image_ocr`](Self::extract_image_ocr), also reporting the
    /// rotation auto-rotate applied and what the page cost. Private for the
    /// same reason as `extract_pdf_ocr_run`: both reach callers on
    /// [`Extraction`].
    ///
    /// An image is one page, so this emits exactly one [`PageTiming`]. Nothing
    /// is rasterized here, so `render_us` times the normalization that stands
    /// in for it — the decode-and-flatten that produces the image both the vote
    /// and the OCR pass read.
    fn extract_image_ocr_run(&mut self, path: &Path, start: Instant) -> Result<OcrRun> {
        let started_us = elapsed_us(start);
        let page_start = Instant::now();
        let mut orient_us = None;
        let mut encode_us = None;

        let (text, rotation, render_us, ocr_us) = if self.auto_rotate {
            let render_start = Instant::now();
            let image = normalize_image_for_ocr(path).or_else(|_| {
                let bytes = std::fs::read(path).context("read image file")?;
                image::load_from_memory(&bytes).context("load image from memory")
            })?;
            let render_us = elapsed_us(render_start);

            let orient_start = Instant::now();
            let angle = detect_orientation(&mut self.ocr, &image, self.orientation_probe_dim)?;
            orient_us = Some(elapsed_us(orient_start));

            let encode_start = Instant::now();
            let png = encode_png(&rotate_image(image, angle))?;
            encode_us = Some(elapsed_us(encode_start));

            let ocr_start = Instant::now();
            let text = ocr_bytes(&mut self.ocr, &png)?;
            (text, angle, render_us, elapsed_us(ocr_start))
        } else {
            let render_start = Instant::now();
            let normalized = normalize_image_for_ocr(path);
            let render_us = elapsed_us(render_start);

            let png = match normalized {
                Ok(image) => {
                    let encode_start = Instant::now();
                    let png = encode_png(&image);
                    encode_us = Some(elapsed_us(encode_start));
                    png
                }
                Err(normalize_err) => Err(normalize_err),
            };

            let (text, ocr_us) = match png {
                Ok(png) => {
                    let ocr_start = Instant::now();
                    let text = ocr_bytes(&mut self.ocr, &png)?;
                    (text, elapsed_us(ocr_start))
                }
                // Normalization is a convenience, not a requirement: hand the
                // raw file to tesseract rather than fail on something `image`
                // could not decode.
                Err(normalize_err) => {
                    let bytes = std::fs::read(path).context("read image file")?;
                    let ocr_start = Instant::now();
                    let text = ocr_bytes(&mut self.ocr, &bytes).with_context(|| {
                        format!("image normalization failed first: {normalize_err:?}")
                    })?;
                    (text, elapsed_us(ocr_start))
                }
            };
            (text, 0, render_us, ocr_us)
        };

        let chars = text.chars().count();
        Ok(OcrRun {
            pages: vec![PageText { page: 1, text }],
            chars,
            rotations: vec![rotation],
            timings: vec![PageTiming {
                page: 1,
                extractor: Extractor::OcrImage,
                started_us,
                total_us: elapsed_us(page_start),
                render_us: Some(render_us),
                orient_us,
                encode_us,
                ocr_us: Some(ocr_us),
            }],
        })
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
        // Before the sniff: the type detection is part of what the call cost.
        let start = Instant::now();
        let mut out = Extraction::default();

        if treat_as_image || sniffs_as_image(path) {
            match self.extract_image_ocr_run(path, start) {
                Ok(run) => {
                    out.extractor = Some(Extractor::OcrImage);
                    out.n_pages = 1;
                    out.extracted_chars = run.chars;
                    out.pages = run.pages;
                    out.page_rotations = run.rotations;
                    out.timings = run.timings;
                }
                Err(e) => out.error = Some(format!("image_ocr_failed: {e:?}")),
            }
            out.total_us = elapsed_us(start);
            return out;
        }

        match self.extract_pdf_native_timed(path, start) {
            Ok((pages, chars, timings)) => {
                out.extractor = Some(Extractor::Native);
                out.n_pages = pages.len() as u32;
                out.extracted_chars = chars;
                out.needs_ocr = chars < self.min_chars_per_page * pages.len().max(1);
                out.pages = pages;
                out.timings = timings;
            }
            Err(e) => {
                out.error = Some(format!("pdf_native_failed: {e:?}"));
                out.needs_ocr = true; // try OCR as a recovery path
            }
        }

        if out.needs_ocr {
            match self.extract_pdf_ocr_run(path, self.ocr_dpi, start) {
                Ok(run) => {
                    out.extractor = Some(Extractor::OcrPdf);
                    out.n_pages = run.pages.len() as u32;
                    out.extracted_chars = run.chars;
                    out.pages = run.pages;
                    out.page_rotations = run.rotations;
                    // The native attempt's timings describe text that was
                    // thrown away; only its cost survives, in `total_us`.
                    out.timings = run.timings;
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

        out.total_us = elapsed_us(start);
        out
    }
}

/// One rendered page: the bytes, the angle applied, and what each phase cost.
///
/// The timings are returned rather than written into the engine because
/// [`render_page_to_png`] is a free function holding a disjoint borrow of
/// `Engine::ocr` — it has no `self` to write to.
struct RenderedPage {
    png: Vec<u8>,
    /// The rotation applied, so a caller that renders the same page again can
    /// reuse the answer instead of re-deriving it.
    angle: u32,
    /// pdfium rasterization.
    render_us: u64,
    /// The orientation vote; `None` when the angle was known or auto-rotate is
    /// off, which is exactly the cost that reuse avoids.
    orient_us: Option<u64>,
    /// Rotation of the full-resolution image plus its PNG encode.
    encode_us: u64,
}

/// Render one pdfium page to PNG bytes, applying auto-rotation when enabled.
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
    probe_dim: u32,
) -> Result<RenderedPage> {
    let render_start = Instant::now();
    let bitmap = page
        .render_with_config(config)
        .context("pdfium render failed")?;
    let image = bitmap.as_image();
    let render_us = elapsed_us(render_start);

    let (angle, orient_us) = match known_angle {
        Some(angle) => (angle, None),
        None if auto_rotate => {
            let orient_start = Instant::now();
            let angle = detect_orientation(ocr, &image, probe_dim)?;
            (angle, Some(elapsed_us(orient_start)))
        }
        None => (0, None),
    };

    let encode_start = Instant::now();
    let png = encode_png(&rotate_image(image, angle))?;
    let encode_us = elapsed_us(encode_start);

    Ok(RenderedPage {
        png,
        angle,
        render_us,
        orient_us,
        encode_us,
    })
}

/// Recognition width when the caller has not chosen one: what the machine says
/// it has, capped at 4.
///
/// The cap is about memory, not speed — recognition scales nearly linearly, but
/// each worker holds a language model, so an uncapped default would reserve
/// hundreds of megabytes on a large build machine for work that may never need
/// it. On Linux, `available_parallelism` reflects the cgroup CPU quota, so a
/// container sees its own budget rather than the host's cores.
pub fn default_ocr_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(4)
}

/// One page's recognized text and what each phase of it cost. Carries its own
/// index because a pool finishes out of order.
struct PageWork {
    index: usize,
    text: String,
    angle: u32,
    started_us: u64,
    total_us: u64,
    render_us: u64,
    orient_us: Option<u64>,
    encode_us: u64,
    ocr_us: u64,
}

/// Everything an already-rasterized page still needs: the orientation vote,
/// the rotate-and-encode, and recognition. Touches no pdfium and no shared
/// state, which is what makes it safe to run on a worker.
#[allow(clippy::too_many_arguments)]
fn ocr_one_page(
    ocr: &mut LepTess,
    index: usize,
    image: DynamicImage,
    started_us: u64,
    render_us: u64,
    auto_rotate: bool,
    probe_dim: u32,
    dpi: f32,
) -> Result<PageWork> {
    let page_start = Instant::now();

    let (angle, orient_us) = if auto_rotate {
        let orient_start = Instant::now();
        let angle = detect_orientation(ocr, &image, probe_dim)?;
        (angle, Some(elapsed_us(orient_start)))
    } else {
        (0, None)
    };

    let encode_start = Instant::now();
    let png = encode_png(&rotate_image(image, angle))?;
    let encode_us = elapsed_us(encode_start);

    let ocr_start = Instant::now();
    let text = ocr_png_bytes(ocr, &png, dpi as i32)?;
    let ocr_us = elapsed_us(ocr_start);

    Ok(PageWork {
        index,
        text,
        angle,
        started_us,
        // The page's whole cost, rasterization included, even though that
        // happened on another thread — it is still what this page took.
        total_us: render_us + elapsed_us(page_start),
        render_us,
        orient_us,
        encode_us,
        ocr_us,
    })
}

/// Microseconds elapsed since `since`, saturating well past any plausible
/// extraction (u64 microseconds is ~584,000 years).
fn elapsed_us(since: Instant) -> u64 {
    since.elapsed().as_micros() as u64
}

/// PNG-encode an in-memory image.
fn encode_png(image: &DynamicImage) -> Result<Vec<u8>> {
    let mut png = Vec::new();
    image
        .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
        .context("PNG encode failed")?;
    Ok(png)
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

fn detect_orientation(ocr: &mut LepTess, image: &DynamicImage, max_dim: u32) -> Result<u32> {
    // Orientation is a coarse best-of-four vote; it doesn't need full OCR
    // resolution. Detection is the dominant OCR cost (up to 4 passes per page
    // at OCR DPI), so vote on a downscaled copy — the extraction pass that
    // follows still runs on the full-resolution image.
    let probe = if image.width().max(image.height()) > max_dim {
        image.resize(max_dim, max_dim, image::imageops::FilterType::Triangle)
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
    encode_png(&normalize_image_for_ocr(path)?)
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
