//! Engine lifecycle. pdfium holds a process-wide lock for the lifetime of a
//! binding, which makes a second live engine impossible and a second engine
//! *on the same thread* impossible to even wait for — see `Engine`'s docs.

use docsee::Engine;

#[test]
fn a_second_engine_on_this_thread_is_refused_rather_than_hanging() {
    let first = Engine::new().unwrap();

    // `Engine` is not `Debug`, so discard it before asserting on the error.
    let err = Engine::builder()
        .build()
        .map(|_| ())
        .expect_err("a second engine on this thread can only block forever");
    let message = format!("{err:#}");
    assert!(
        message.contains("already has a live Engine"),
        "the error should say why: {message}"
    );

    // The claim is released with the engine, so the thread is not poisoned.
    drop(first);
    Engine::new().expect("a replacement engine builds once the first is gone");
}

#[test]
fn a_failed_build_does_not_claim_the_thread() {
    // No tessdata for this language, so the build fails after the pdfium bind.
    let failed = Engine::builder()
        .ocr_language("zzzz-not-a-language")
        .build();
    assert!(failed.is_err(), "the language should not resolve");

    Engine::new().expect("a failed build leaves the thread free to try again");
}
