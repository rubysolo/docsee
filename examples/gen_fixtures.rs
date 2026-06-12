//! Generates synthetic test fixtures in tests/fixtures/.  Run once and commit
//! the output; tests reference the pre-built files.
//!
//!   cargo run --example gen_fixtures

use image::codecs::jpeg::JpegEncoder;
use std::fmt::Write as _;
use std::io::Cursor;
use std::path::{Path, PathBuf};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn main() {
    let dir = fixtures();
    gen_exif_rotated_jpg(&dir.join("exif_rotated.jpg"));
    gen_transparent_png(&dir.join("transparent.png"));
    gen_multipage_pdf(&dir.join("multipage.pdf"));
    gen_image_dat(&dir.join("image.dat"));
    gen_rotated_fixtures(&dir);
    println!("fixtures written to {}", dir.display());
}

fn gen_rotated_fixtures(dir: &Path) {
    let src = image::open(dir.join("sample.png")).expect("sample.png");

    // Convert to RGBA and composite over white to ensure clean rotation
    let mut canvas =
        image::RgbaImage::from_pixel(src.width(), src.height(), image::Rgba([255, 255, 255, 255]));
    image::imageops::overlay(&mut canvas, &src.to_rgba8(), 0, 0);
    let white_src = image::DynamicImage::ImageRgba8(canvas);

    white_src
        .rotate90()
        .save(dir.join("rotated_90.png"))
        .expect("save rotated_90.png");
    white_src
        .rotate180()
        .save(dir.join("rotated_180.png"))
        .expect("save rotated_180.png");
    white_src
        .rotate270()
        .save(dir.join("rotated_270.png"))
        .expect("save rotated_270.png");

    eprintln!("  {}/rotated_*.png", dir.display());
}

/// JPEG whose pixels are physically rotated 90° CCW but carry EXIF orientation = 6
/// (rotate 90° CW to display).  normalize_image_for_ocr_png applies the correction
/// before OCR, so text ends up upright.
fn gen_exif_rotated_jpg(path: &Path) {
    let src = image::open(fixtures().join("sample.png")).expect("sample.png");

    // sample.png is RGBA with black text on a transparent background (all R/G/B=0).
    // JPEG has no alpha channel, so composite over white before encoding, otherwise
    // the transparent background becomes solid black and OCR returns empty.
    let rgba = src.to_rgba8();
    let mut canvas = image::RgbaImage::from_pixel(
        rgba.width(),
        rgba.height(),
        image::Rgba([255, 255, 255, 255]),
    );
    image::imageops::overlay(&mut canvas, &rgba, 0, 0);
    let white_src = image::DynamicImage::ImageRgba8(canvas);

    // rotate270 = 90° CCW: text is now physically sideways
    let rotated = white_src.rotate270();

    let mut jpeg = Vec::new();
    let mut cur = Cursor::new(&mut jpeg);
    let enc = JpegEncoder::new_with_quality(&mut cur, 92);
    rotated.write_with_encoder(enc).expect("JPEG encode");

    // Minimal EXIF APP1 for orientation = 6.
    // Layout: FF E1 [len 2B BE] "Exif\0\0" [TIFF-LE header] [1 IFD entry] [next=0]
    // len counts itself + payload = 2 + (6+8+2+12+4) = 34 = 0x0022
    let app1: &[u8] = &[
        0xFF, 0xE1, 0x00, 0x22, // APP1 marker, length=34
        0x45, 0x78, 0x69, 0x66, 0x00, 0x00, // "Exif\0\0"
        0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00, // TIFF-LE, IFD@8
        0x01, 0x00, // 1 IFD entry
        0x12, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, // tag=0x0112 SHORT count=1
        0x06, 0x00, 0x00, 0x00, // value=6 (rotate 90° CW)
        0x00, 0x00, 0x00, 0x00, // next IFD = none
    ];

    // Splice APP1 right after the SOI marker (FF D8).
    let mut out = Vec::with_capacity(jpeg.len() + app1.len());
    out.extend_from_slice(&jpeg[..2]); // SOI
    out.extend_from_slice(app1);
    out.extend_from_slice(&jpeg[2..]);

    std::fs::write(path, out).expect("write exif_rotated.jpg");
    eprintln!("  {}", path.display());
}

/// PNG visually identical to sample.png but with background pixels fully
/// transparent.  Exercises the alpha-compositing branch (black text over
/// transparent → white canvas blend) in normalize_image_for_ocr_png.
fn gen_transparent_png(path: &Path) {
    let src = image::open(fixtures().join("sample.png")).expect("sample.png");
    let mut rgba = src.to_rgba8();
    for px in rgba.pixels_mut() {
        let [r, g, b, _] = px.0;
        if r > 200 && g > 200 && b > 200 {
            px.0[3] = 0; // background → fully transparent
        }
    }
    rgba.save(path).expect("write transparent.png");
    eprintln!("  {}", path.display());
}

/// Hand-rolled 2-page PDF using the built-in Helvetica Type1 font.
/// No extra crates: xref offsets are computed from the accumulated String length.
fn gen_multipage_pdf(path: &Path) {
    // Stream body: no trailing newline — the separator before endstream is added
    // separately so it is not counted in /Length (per PDF spec §7.3.8.1).
    // The string must exceed MIN_CHARS_PER_PAGE (50) so pdfium uses Native extraction.
    let body =
        "BT /F1 12 Tf 72 720 Td (The quick brown fox jumps over the lazy dog. Page text.) Tj ET";
    let blen = body.len();

    let mut s = String::new();
    writeln!(s, "%PDF-1.4").unwrap();

    let mut off = [0usize; 8]; // off[1..=7]: byte offsets of each object

    off[1] = s.len();
    write!(s, "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n").unwrap();

    off[2] = s.len();
    write!(
        s,
        "2 0 obj\n<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>\nendobj\n"
    )
    .unwrap();

    off[3] = s.len();
    write!(
        s,
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
         /Contents 5 0 R /Resources << /Font << /F1 6 0 R >> >> >>\nendobj\n"
    )
    .unwrap();

    off[4] = s.len();
    write!(
        s,
        "4 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
         /Contents 7 0 R /Resources << /Font << /F1 6 0 R >> >> >>\nendobj\n"
    )
    .unwrap();

    off[5] = s.len();
    write!(
        s,
        "5 0 obj\n<< /Length {blen} >>\nstream\n{body}\nendstream\nendobj\n"
    )
    .unwrap();

    off[6] = s.len();
    write!(
        s,
        "6 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    )
    .unwrap();

    off[7] = s.len();
    write!(
        s,
        "7 0 obj\n<< /Length {blen} >>\nstream\n{body}\nendstream\nendobj\n"
    )
    .unwrap();

    let xref_pos = s.len();
    // xref entries are exactly 20 bytes each (10-digit offset + " 00000 n \n")
    write!(s, "xref\n0 8\n0000000000 65535 f \n").unwrap();
    for o in off.iter().skip(1) {
        writeln!(s, "{:010} 00000 n ", o).unwrap();
    }
    write!(
        s,
        "trailer\n<< /Size 8 /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
    )
    .unwrap();

    std::fs::write(path, s.as_bytes()).expect("write multipage.pdf");
    eprintln!("  {}", path.display());
}

/// Copy of sample.png with a non-image extension.  Tests the sniffs_as_image
/// fallback: extract() reads the magic bytes and still routes to OcrImage.
fn gen_image_dat(path: &Path) {
    std::fs::copy(fixtures().join("sample.png"), path).expect("copy sample.png");
    eprintln!("  {}", path.display());
}
