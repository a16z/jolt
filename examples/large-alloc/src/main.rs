use tracing::info;

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
