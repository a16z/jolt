use tracing::info;

/// Executes the large-allocation roundtrip guest under the tracer (no
/// proving) and fails loudly if the guest panicked. Manual harness for
/// ZeroOS's mmap/munmap region accounting: each guest allocation is past
/// musl mallocng's individual-mmap threshold, so every `drop` issues a real
/// `munmap` against the guest kernel's heap. The CI regression test is
/// `large_alloc_munmap_trace` in jolt-prover-legacy's prover tests.
pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_large_alloc_roundtrip(target_dir);

    let (_lazy, trace, _memory, io_device) = program.trace(&[], &[], &[]);
    info!("guest executed {} trace rows", trace.len());
    assert!(
        !io_device.panic,
        "large-alloc guest panicked: the runtime failed a large free/munmap"
    );
    info!("outputs: {:?}", io_device.outputs);
    info!("large-alloc roundtrip completed cleanly");
}
