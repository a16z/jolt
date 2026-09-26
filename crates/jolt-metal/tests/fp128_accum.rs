//! Conformance of the `jolt::Fp128<C>` accumulators and the accumulator
//! reductions with `jolt_field`.
//!
//! The oracle is field arithmetic: each threadgroup's reduced sum must equal
//! the sum of its terms computed with `jolt_field` operations, on edge vectors
//! and on 2^20 fixed-seed random terms, for every threadgroup shape
//! `threadgroup_merge` must handle, and in every lane.
//!
//! Capacity is checked at the boundary: exactly `CAPACITY` worst-case terms
//! reduce correctly, one more term reduces correctly through the documented
//! pre-reduction path, and one more term without it does not. The last check
//! shows that the worst case really reaches the representation's limit.

#![cfg(target_os = "macos")]
#![expect(clippy::unwrap_used, reason = "tests may panic on assertion failures")]

#[path = "support/field.rs"]
mod field;
mod support;

mod gpu {
    use jolt_field::solinas::{Prime128Offset275, Prime128OffsetA7F7};
    use jolt_metal::runtime::{
        host_name, Batch, Binding, Device, DeviceBuffer, Grid, LibrarySpec, Pipeline, ShaderLibrary,
    };
    use jolt_metal::shaders::FIELD_HEADERS;

    use super::field::{edges, element, modulus, random_elements, TestField};
    use super::support::{gpu, SplitMix64};

    const ACCUM_OPS: &str = include_str!("shaders/accum_ops.metal");
    const FMADD: &str = "jolt_test_accum_fmadd";
    const SMALL_SCALAR: &str = "jolt_test_accum_small_scalar";
    const FILL: &str = "jolt_test_accum_fill";
    const SMALL_SCALAR_FILL: &str = "jolt_test_accum_small_scalar_fill";
    const MERGE: &str = "jolt_test_accum_merge";
    const SMALL_SCALAR_MERGE: &str = "jolt_test_accum_small_scalar_merge";
    const KERNELS: [&str; 6] = [
        FMADD,
        SMALL_SCALAR,
        FILL,
        SMALL_SCALAR_FILL,
        MERGE,
        SMALL_SCALAR_MERGE,
    ];

    /// Random terms per accumulator kind.
    const RANDOM: usize = 1 << 20;

    /// `CAPACITY` of `Fp128Accumulator` and `Fp128SignedAccumulator` in
    /// `fp128_accum.h`.
    const CAPACITY: u64 = 1 << 32;
    const SMALL_SCALAR_CAPACITY: u64 = 1 << 31;

    /// Threads of the capacity fill kernels; each takes an equal share.
    const FILL_THREADS: usize = 1 << 20;

    /// `Fp128Accumulator<C>` and `Fp128SignedAccumulator<C>` as device words.
    type RawAccumulator = [u32; 9];
    type RawSmallScalarAccumulator = [u32; 7];

    fn library<F: TestField>(device: &Device) -> ShaderLibrary {
        let spec = FIELD_HEADERS
            .iter()
            .fold(LibrarySpec::new(), |spec, (name, text)| {
                spec.source(name, text)
            })
            .source("accum_ops.metal", ACCUM_OPS);
        let spec = KERNELS
            .iter()
            .fold(spec, |spec, kernel| spec.instantiate::<F>(kernel));
        ShaderLibrary::compile(device, &spec).unwrap()
    }

    fn pipeline<'l, F: TestField>(library: &'l ShaderLibrary, kernel: &str) -> &'l Pipeline {
        library.pipeline(&host_name::<F>(kernel)).unwrap()
    }

    /// Threadgroup sizes for `threadgroup_merge`: one simdgroup, two, three
    /// (not a power of two), eight, and the largest whole number of
    /// simdgroups the pipeline allows.
    fn group_sizes(pipeline: &Pipeline) -> Vec<usize> {
        let width = pipeline.thread_execution_width();
        let largest = pipeline.max_total_threads_per_threadgroup() / width * width;
        let mut sizes: Vec<usize> = [1, 2, 3, 8]
            .iter()
            .map(|simdgroups| simdgroups * width)
            .filter(|&size| size <= largest)
            .collect();
        sizes.push(largest);
        sizes.dedup();
        sizes
    }

    fn dispatch(device: &Device, pipeline: &Pipeline, bindings: &[Binding<'_>], grid: Grid) {
        let mut batch = Batch::new(device).unwrap();
        batch.dispatch(pipeline, bindings, grid).unwrap();
        let _ = batch.commit_and_wait().unwrap();
    }

    /// Runs a conformance kernel with `terms` terms per thread in groups of
    /// `group` threads and checks every thread's output against the sum of
    /// its threadgroup's terms.
    fn check_groups<F: TestField>(
        device: &Device,
        pipeline: &Pipeline,
        inputs: &[Binding<'_>],
        cpu_terms: &[F],
        terms: usize,
        group: usize,
    ) {
        let threads = cpu_terms.len().div_ceil(terms).next_multiple_of(group);
        let n = u32::try_from(cpu_terms.len()).unwrap();
        let terms_u32 = u32::try_from(terms).unwrap();
        let mut out = DeviceBuffer::<F>::zeroed(device, threads).unwrap();
        let mut bindings = inputs.to_vec();
        bindings.extend([
            Binding::value(&n),
            Binding::value(&terms_u32),
            Binding::buffer(&out),
        ]);
        dispatch(device, pipeline, &bindings, Grid::linear(threads, group));
        let got = out.read().unwrap();
        let per_group = terms * group;
        for (g, outputs) in got.chunks(group).enumerate() {
            let start = (g * per_group).min(cpu_terms.len());
            let end = ((g + 1) * per_group).min(cpu_terms.len());
            let want = cpu_terms[start..end]
                .iter()
                .fold(F::zero(), |sum, term| sum + *term);
            if let Some(lane) = outputs.iter().position(|value| *value != want) {
                assert_eq!(
                    outputs[lane],
                    want,
                    "{}: {terms} terms per thread, groups of {group}: group {g} lane {lane}",
                    pipeline.name(),
                );
            }
        }
    }

    fn fmadd_conformance<F: TestField>(device: &Device, library: &ShaderLibrary, seed: u64) {
        let edges = edges::<F>();
        let mut words = SplitMix64(seed);
        let mut a: Vec<u128> = Vec::new();
        let mut b: Vec<u128> = Vec::new();
        for &x in &edges {
            for &y in &edges {
                a.push(x);
                b.push(y);
            }
        }
        a.extend(random_elements::<F>(&mut words, RANDOM));
        b.extend(random_elements::<F>(&mut words, RANDOM));
        let a: Vec<F> = a.into_iter().map(element).collect();
        let b: Vec<F> = b.into_iter().map(element).collect();
        let cpu_terms: Vec<F> = a
            .iter()
            .zip(&b)
            .enumerate()
            .map(|(t, (x, y))| if t % 4 == 3 { *x } else { *x * *y })
            .collect();
        let (a_dev, b_dev) = (
            DeviceBuffer::from_slice(device, &a).unwrap(),
            DeviceBuffer::from_slice(device, &b).unwrap(),
        );
        let pipeline = pipeline::<F>(library, FMADD);
        for group in group_sizes(pipeline) {
            for terms in [1, 16] {
                check_groups(
                    device,
                    pipeline,
                    &[Binding::buffer(&a_dev), Binding::buffer(&b_dev)],
                    &cpu_terms,
                    terms,
                    group,
                );
            }
        }
    }

    fn small_scalar_conformance<F: TestField>(device: &Device, library: &ShaderLibrary, seed: u64) {
        let edges = edges::<F>();
        let scalars = [
            0,
            1,
            2,
            (1 << 32) - 1,
            1 << 32,
            (1 << 63) - 1,
            1 << 63,
            (1 << 63) + 1,
            u64::MAX - 1,
            u64::MAX,
        ];
        let mut words = SplitMix64(seed);
        let mut a: Vec<u128> = Vec::new();
        let mut s: Vec<u64> = Vec::new();
        // Four copies of each pair, so each pair meets every operation.
        for &x in &edges {
            for &scalar in &scalars {
                for _ in 0..4 {
                    a.push(x);
                    s.push(scalar);
                }
            }
        }
        a.extend(random_elements::<F>(&mut words, RANDOM));
        s.extend((&mut words).take(RANDOM));
        let a: Vec<F> = a.into_iter().map(element).collect();
        let cpu_terms: Vec<F> = a
            .iter()
            .zip(&s)
            .enumerate()
            .map(|(t, (&x, &scalar))| match t % 4 {
                0 => x * F::from_i64(scalar as i64),
                1 => x * F::from_u64(scalar),
                2 if t % 8 < 4 => x * F::from_u64(scalar),
                2 => -(x * F::from_u64(scalar)),
                _ => x,
            })
            .collect();
        let (a_dev, s_dev) = (
            DeviceBuffer::from_slice(device, &a).unwrap(),
            DeviceBuffer::from_slice(device, &s).unwrap(),
        );
        let pipeline = pipeline::<F>(library, SMALL_SCALAR);
        for group in group_sizes(pipeline) {
            for terms in [1, 16] {
                check_groups(
                    device,
                    pipeline,
                    &[Binding::buffer(&a_dev), Binding::buffer(&s_dev)],
                    &cpu_terms,
                    terms,
                    group,
                );
            }
        }
    }

    /// Merges the filled accumulators with one threadgroup and returns the
    /// reduced sum, after one more term when `extra` is set.
    fn merge<F: TestField>(
        device: &Device,
        pipeline: &Pipeline,
        leading: &[Binding<'_>],
        trailing: &[Binding<'_>],
    ) -> F {
        let group = group_sizes(pipeline).pop().unwrap();
        let mut out = DeviceBuffer::<F>::zeroed(device, 1).unwrap();
        let mut bindings = leading.to_vec();
        bindings.extend_from_slice(trailing);
        bindings.push(Binding::buffer(&out));
        dispatch(device, pipeline, &bindings, Grid::linear(group, group));
        out.read().unwrap()[0]
    }

    fn capacity<F: TestField>(device: &Device, library: &ShaderLibrary) {
        let x: F = element(modulus::<F>() - 1);
        let x_dev = DeviceBuffer::from_slice(device, &[x]).unwrap();
        let count = u32::try_from(FILL_THREADS).unwrap();
        let fill_group = |pipeline: &Pipeline| pipeline.thread_execution_width() * 8;

        // (p - 1)^2 = 1, so k terms sum to k.
        let fill = pipeline::<F>(library, FILL);
        let terms = u32::try_from(CAPACITY / FILL_THREADS as u64).unwrap();
        let partials = DeviceBuffer::<RawAccumulator>::zeroed(device, FILL_THREADS).unwrap();
        dispatch(
            device,
            fill,
            &[
                Binding::buffer(&x_dev),
                Binding::value(&terms),
                Binding::buffer(&partials),
            ],
            Grid::linear(FILL_THREADS, fill_group(fill)),
        );
        let merge_pipeline = pipeline::<F>(library, MERGE);
        let leading = [
            Binding::buffer(&partials),
            Binding::value(&count),
            Binding::buffer(&x_dev),
        ];
        let run = |extra: bool, pre_reduce: bool| -> F {
            merge(
                device,
                merge_pipeline,
                &leading,
                &[Binding::value(&extra), Binding::value(&pre_reduce)],
            )
        };
        assert_eq!(run(false, false), F::from_u64(CAPACITY));
        assert_eq!(run(true, true), F::from_u64(CAPACITY + 1));
        assert_ne!(run(true, false), F::from_u64(CAPACITY + 1));

        // (p - 1)(2^64 - 1) = -(2^64 - 1), so k positive terms sum to
        // -k (2^64 - 1) and k negative terms to k (2^64 - 1).
        let fill = pipeline::<F>(library, SMALL_SCALAR_FILL);
        let merge_pipeline = pipeline::<F>(library, SMALL_SCALAR_MERGE);
        let terms = u32::try_from(SMALL_SCALAR_CAPACITY / FILL_THREADS as u64).unwrap();
        let term = F::from_u64(u64::MAX);
        for negative in [false, true] {
            let partials =
                DeviceBuffer::<RawSmallScalarAccumulator>::zeroed(device, FILL_THREADS).unwrap();
            dispatch(
                device,
                fill,
                &[
                    Binding::buffer(&x_dev),
                    Binding::value(&terms),
                    Binding::value(&negative),
                    Binding::buffer(&partials),
                ],
                Grid::linear(FILL_THREADS, fill_group(fill)),
            );
            let leading = [
                Binding::buffer(&partials),
                Binding::value(&count),
                Binding::buffer(&x_dev),
            ];
            let run = |extra: bool, pre_reduce: bool| -> F {
                merge(
                    device,
                    merge_pipeline,
                    &leading,
                    &[
                        Binding::value(&extra),
                        Binding::value(&pre_reduce),
                        Binding::value(&negative),
                    ],
                )
            };
            let sum = |k: u64| {
                let magnitude = F::from_u64(k) * term;
                if negative {
                    magnitude
                } else {
                    -magnitude
                }
            };
            assert_eq!(run(false, false), sum(SMALL_SCALAR_CAPACITY));
            assert_eq!(run(true, true), sum(SMALL_SCALAR_CAPACITY + 1));
            assert_ne!(run(true, false), sum(SMALL_SCALAR_CAPACITY + 1));
        }
    }

    fn accumulators<F: TestField>(test: &'static str, seed: u64) {
        let (_guard, device) = gpu(test);
        let library = library::<F>(&device);
        fmadd_conformance::<F>(&device, &library, seed);
        small_scalar_conformance::<F>(&device, &library, seed + 1);
    }

    fn capacity_test<F: TestField>(test: &'static str) {
        let (_guard, device) = gpu(test);
        let library = library::<F>(&device);
        capacity::<F>(&device, &library);
    }

    #[test]
    fn fp128_a7f7_accumulators_match_jolt_field() {
        accumulators::<Prime128OffsetA7F7>("fp128_a7f7_accumulators_match_jolt_field", 3);
    }

    #[test]
    fn fp128_275_accumulators_match_jolt_field() {
        accumulators::<Prime128Offset275>("fp128_275_accumulators_match_jolt_field", 5);
    }

    #[test]
    fn fp128_a7f7_accumulator_capacity_is_exact() {
        capacity_test::<Prime128OffsetA7F7>("fp128_a7f7_accumulator_capacity_is_exact");
    }

    #[test]
    fn fp128_275_accumulator_capacity_is_exact() {
        capacity_test::<Prime128Offset275>("fp128_275_accumulator_capacity_is_exact");
    }
}
