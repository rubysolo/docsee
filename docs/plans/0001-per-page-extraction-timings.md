# Plan 0001 — Per-page extraction timings

Status: **proposed** (2026-08-28).

## Goal

Report **where an extraction spent its time**, at page resolution and phase
resolution, so an embedder can render a waterfall of a single file's extraction
instead of one opaque bar.

Today `Extraction` says *what* happened (`extractor`, `n_pages`,
`extracted_chars`, `needs_ocr`, `page_rotations`) but nothing about *when* or
*how long*. A 60-page scan is a single multi-second call, and the caller cannot
tell whether the cost was pdfium rasterization, the orientation vote, or
tesseract itself — even though those differ by an order of magnitude and have
completely different fixes.

The orientation probe is the specific reason this matters. With `auto_rotate`
on, `detect_orientation` runs **up to four tesseract passes per page** before the
real OCR pass. That is usually the dominant cost of an OCR extraction, it is
already the subject of three merged optimizations (low-DPI probe, reuse of the
detected rotation, configurable probe dimension), and it is currently
unmeasurable from outside the crate. Anyone tuning `orientation_probe_dim` is
guessing.

### Non-goals

- **No live streaming in this pass.** Timings are returned with the result, not
  pushed during it. A callback/observer hook is a natural follow-up (see
  "Deferred") but it drags lifetimes, thread-safety, and a second public trait
  into a crate that currently has none of that.
- **No tracing/log crate dependency.** `docsee` has seven dependencies and no
  logging facade; adding `tracing` to answer this would push a runtime choice
  onto every embedder. Plain data on the result keeps the crate inert.
- **No feature flag.** A `cfg`-gated struct field breaks under feature
  unification in a workspace and would make `Extraction`'s shape depend on who
  else in the dependency graph enabled it. `Instant::now()` costs tens of
  nanoseconds against a tesseract pass measured in hundreds of milliseconds, so
  always-on is the honest default.

## Design

### `PageTiming` on the result

```rust
/// Wall-clock cost of one page's extraction, by phase. Phases that did not run
/// on this page are `None` — a natively-extracted page has no `ocr`, and a page
/// whose rotation was supplied by the caller has no `orient`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageTiming {
    pub page: u32,
    /// Which path produced this page's text.
    pub extractor: Extractor,
    /// Offset from the start of the extraction call, in microseconds.
    pub started_us: u64,
    /// Total for this page, in microseconds.
    pub total_us: u64,
    /// pdfium rasterization to an image.
    pub render_us: Option<u64>,
    /// Orientation vote (`detect_orientation`) — up to four tesseract passes.
    pub orient_us: Option<u64>,
    /// PNG encode of the (possibly rotated) page image.
    pub encode_us: Option<u64>,
    /// The tesseract pass that produced the text.
    pub ocr_us: Option<u64>,
}
```

and on `Extraction`:

```rust
/// Per-page cost breakdown, in page order. Always populated.
#[serde(default)]
pub timings: Vec<PageTiming>,
/// Total wall clock for the whole `extract` call, in microseconds. Includes
/// document open, the type sniff, and — when the native path was tried and
/// rejected — the cost of that abandoned attempt, which no page timing covers.
pub total_us: u64,
```

`started_us` is an offset rather than an absolute instant on purpose: the
embedder places the whole extraction on its own time axis and needs the pages
positioned *within* it, not a second clock to reconcile.

**Why data-return rather than a callback.** The consumer is a waterfall rendered
after the fact. Returning data needs no trait, no lifetime, no `Send + Sync`
bound on a user type, and survives the FFI boundary (`docsee` → `doxie_extract`
→ a Rustler NIF → the BEAM) as plain serde, which a callback does not.

### Phase boundaries

Mapped to the code as it stands:

| Phase | Where | Notes |
|---|---|---|
| type sniff | `extract_with_hint` | extension check + `sniffs_as_image` |
| native attempt | `extract_pdf_native` | per page: pdfium text. Timed per page; the whole attempt is also charged to `total_us` when it is abandoned for OCR |
| render | `render_page_to_png` (pdfium `render_with_config`) | PDF OCR path only |
| orient | `detect_orientation` | skipped when `known_angle` is supplied or `auto_rotate` is off — this is where the reuse optimization pays, so it must be visible |
| encode | `render_page_to_png` (PNG write) | |
| ocr | `ocr_png_bytes` / `ocr_bytes` | the tesseract pass that yields text |

The image path (`extract_image_ocr_rotation`) emits exactly one `PageTiming`
with `extractor: OcrImage`; its normalize step folds into `render_us`, which is
the closest honest analogue (it is the rasterization that precedes the vote).

`render_page_to_png` is a free function taking `&mut LepTess` specifically to
keep the disjoint-field borrow the OCR paths rely on. It must therefore *return*
its sub-timings alongside `(png, angle)` rather than write into `self` — a
three-tuple grown to a small struct.

### Compatibility

`Extraction` has public fields and derives `Default`, so adding fields breaks
struct-literal construction downstream. Mitigations, all in one release:

- Mark `Extraction` **`#[non_exhaustive]`** so this is the last time an added
  field is a breaking change.
- `#[serde(default)]` on `timings` keeps previously serialized JSON records
  (the CLI's `--json` output, anything an embedder archived) deserializable.
- Bump to **0.2.0**. At 0.1.0 with one known embedder plus the CLI, this is
  cheap now and expensive later.

`doxie_extract` maps `docsee::Extraction` into its own `Extraction` via `From`,
so it takes the new fields as a passthrough plus a rev bump — no behavior change
and no effect on the Python-compatible `extractor` labels that keep the two
extraction runs diffable.

### CLI

`--json` includes `timings` and `total_us` with no flag. Add `--timings` to
print a human-readable per-page table for the plain-text path, since the whole
point is to make a slow file legible without piping through `jq`.

## Tests

`tests/extraction.rs` gains, over the existing fixtures:

- a native-text PDF yields one `PageTiming` per page, each `extractor: Native`,
  with `ocr_us`/`orient_us` `None`;
- an OCR'd PDF yields `render_us`, `encode_us` and `ocr_us` on every page, and
  `orient_us` **only** when `auto_rotate` is on (this is the regression guard
  for the rotation-reuse optimization — if a future change reintroduces the
  redundant vote, this test fails);
- `timings.len() == n_pages` on every path, image included;
- `started_us` is monotonically non-decreasing across pages, and
  `sum(total_us per page) <= Extraction::total_us` (the difference is document
  open, the sniff, and any abandoned native attempt).

Assert **structure and presence**, never durations. A timing test that asserts a
number is a flaky test on someone else's CI.

## Slices

1. **`PageTiming` + `total_us`, native path.** Struct, `#[non_exhaustive]`,
   serde defaults, timings through `extract_pdf_native`, version bump.
2. **OCR paths.** `render_page_to_png` returns sub-timings; PDF-OCR and image
   paths populate render/orient/encode/ocr.
3. **CLI `--timings`** table, README section.

## Deferred

- **Observer hook.** A `Engine::builder().observer(...)` taking
  `Fn(Event) + Send` would let a host show progress *during* a long OCR rather
  than after it. Worth doing when someone wants a live progress bar; the data
  returned here is a strict subset of what such a hook would emit, so it is not
  wasted work.
- **Sub-phase detail inside the orientation vote** (per-angle cost). Only
  interesting if the four-angle strategy itself is being replaced.
