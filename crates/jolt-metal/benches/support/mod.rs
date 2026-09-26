//! Setup shared by the `jolt-metal` benchmarks.

#![expect(clippy::expect_used, reason = "a benchmark aborts on any failure")]

use std::time::Duration;

use jolt_field::solinas::{Ext2, Fp128, Fp64};
use jolt_field::Ring;
use jolt_metal::runtime::{
    host_name, Batch, Binding, Device, Grid, LibrarySpec, Pipeline, ShaderLibrary,
};
use jolt_metal::shaders::FIELD_HEADERS;
use jolt_metal::MetalField;

/// Compiles the field headers and `sources`, with every template in
/// `templates` instantiated for `T` and every kernel in `kernels` as is.
pub fn library<T: MetalField>(
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
        .fold(spec, |spec, template| spec.instantiate::<T>(template));
    let spec = kernels
        .iter()
        .fold(spec, |spec, kernel| spec.kernel(kernel));
    ShaderLibrary::compile(device, &spec).expect("benchmark library compiles")
}

/// The pipeline of `kernel`, a template instantiated for `T` or, failing
/// that, a plain kernel.
pub fn pipeline<'l, T: MetalField>(library: &'l ShaderLibrary, kernel: &str) -> &'l Pipeline {
    library
        .pipeline(&host_name::<T>(kernel))
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

/// A benchmarked field: fixed-seed elements from 64-bit words.
pub trait Sample: MetalField + Ring + Send + Sync {
    /// Words per element.
    const WORDS: usize;

    /// The element made from `words`, which has `WORDS` entries.
    fn from_words(words: &[u64]) -> Self;
}

impl<const P: u128> Sample for Fp128<P> {
    const WORDS: usize = 2;

    fn from_words(words: &[u64]) -> Self {
        Self::from_u128((u128::from(words[0]) << 64) | u128::from(words[1]))
    }
}

impl<const P: u64> Sample for Fp64<P>
where
    Self: MetalField,
{
    const WORDS: usize = 1;

    fn from_words(words: &[u64]) -> Self {
        Self::from_u64(words[0])
    }
}

impl<F: Sample> Sample for Ext2<F>
where
    Self: MetalField,
{
    const WORDS: usize = 2 * F::WORDS;

    fn from_words(words: &[u64]) -> Self {
        let (c0, c1) = words.split_at(F::WORDS);
        Self::new(F::from_words(c0), F::from_words(c1))
    }
}

/// Fixed-seed field elements.
pub fn elements<T: Sample>(seed: u64, len: usize) -> Vec<T> {
    words(seed, T::WORDS * len)
        .chunks_exact(T::WORDS)
        .map(T::from_words)
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
