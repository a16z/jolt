//! Setup shared by the `jolt-metal` benchmarks.

#![expect(clippy::expect_used, reason = "a benchmark aborts on any failure")]

use std::time::Duration;

use jolt_field::solinas::Prime128OffsetA7F7;
use jolt_field::Ring;
use jolt_metal::runtime::{
    host_name, Batch, Binding, Device, Grid, LibrarySpec, Pipeline, ShaderLibrary,
};
use jolt_metal::shaders::FIELD_HEADERS;

/// The field every benchmark measures.
pub type F = Prime128OffsetA7F7;

/// Compiles the field headers and `sources`, with every template in
/// `templates` instantiated for [`F`] and every kernel in `kernels` as is.
pub fn library(
    device: &Device,
    sources: &[(&str, &str)],
    templates: &[&str],
    kernels: &[&str],
) -> ShaderLibrary {
    let spec = FIELD_HEADERS
        .iter()
        .chain(sources)
        .fold(LibrarySpec::new(), |spec, (name, text)| {
            spec.source(name, text)
        });
    let spec = templates
        .iter()
        .fold(spec, |spec, template| spec.instantiate::<F>(template));
    let spec = kernels
        .iter()
        .fold(spec, |spec, kernel| spec.kernel(kernel));
    ShaderLibrary::compile(device, &spec).expect("benchmark library compiles")
}

/// The pipeline of `kernel`, a template instantiated for [`F`] or, failing
/// that, a plain kernel.
pub fn pipeline<'l>(library: &'l ShaderLibrary, kernel: &str) -> &'l Pipeline {
    library
        .pipeline(&host_name::<F>(kernel))
        .or_else(|_| library.pipeline(kernel))
        .expect("kernel is in the library")
}

/// The elementwise threadgroup size used by the tests.
pub fn threadgroup(pipeline: &Pipeline) -> usize {
    (pipeline.thread_execution_width() * 8).min(pipeline.max_total_threads_per_threadgroup())
}

/// Fixed-seed 64-bit words (SplitMix64).
pub fn words(seed: u64, len: usize) -> Vec<u64> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        })
        .collect()
}

/// Fixed-seed field elements.
pub fn elements(seed: u64, len: usize) -> Vec<F> {
    words(seed, 2 * len)
        .chunks_exact(2)
        .map(|pair| F::from_u128((u128::from(pair[0]) << 64) | u128::from(pair[1])))
        .collect()
}

/// Runs `repeats` copies of one dispatch as one batch and returns its GPU
/// time.
pub fn dispatch(
    device: &Device,
    pipeline: &Pipeline,
    bindings: &[Binding<'_>],
    grid: Grid,
    repeats: usize,
) -> Duration {
    let mut batch = Batch::new(device).expect("command batch");
    for _ in 0..repeats {
        batch
            .dispatch(pipeline, bindings, grid)
            .expect("valid dispatch");
    }
    batch.commit_and_wait().expect("batch completes")
}
