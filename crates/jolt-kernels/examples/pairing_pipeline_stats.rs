//! W3-st8 dispatch-geometry probe: compiler occupancy limits for the
//! pairing-family pipelines.
//!
//! ```text
//! cargo run --release -p jolt-kernels --example pairing_pipeline_stats --features metal
//! ```
//!
//! `max_threads` is `maxTotalThreadsPerThreadgroup` — the Metal compiler
//! lowers it below the 1024 device cap as per-thread register footprint
//! grows, so it is the only public register-pressure reading on a host
//! without the `metal` CLI toolchain.

#![expect(
    clippy::print_stdout,
    clippy::expect_used,
    reason = "diagnostic probe: report to stdout, fail loudly"
)]

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    use jolt_kernels::metal::{KernelId, MetalContext};

    let ctx = MetalContext::global().expect("Metal context");
    println!("device: {}", ctx.device_name());
    println!(
        "{:<22} {:>11} {:>10}",
        "kernel", "max_threads", "simd_width"
    );
    for kernel in [
        KernelId::Fq6Mul,
        KernelId::Fq6Sqr,
        KernelId::Fq12Mul,
        KernelId::Fq12Sqr,
        KernelId::Fq12Mul034,
        KernelId::MillerTable,
        KernelId::MillerFly,
        KernelId::MillerFlyLines,
        KernelId::MillerFlyFold,
        KernelId::G1ProjectiveMulAdd,
        KernelId::G2ProjectiveMulAdd,
        KernelId::FrBind,
    ] {
        let (max_threads, simd_width) = ctx.pipeline_stats(kernel);
        println!("{:<22} {max_threads:>11} {simd_width:>10}", kernel.name());
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    println!("pairing_pipeline_stats requires --features metal on macOS");
}
