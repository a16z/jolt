#![expect(clippy::unwrap_used, reason = "tests may panic on assertion failures")]

#[cfg(target_os = "macos")]
mod support;

use jolt_metal::runtime::Device;
use jolt_metal::{ErrorClass, MetalError};

#[cfg(not(target_os = "macos"))]
#[test]
fn device_is_unavailable_off_macos() {
    let error = Device::system_default().err().unwrap();
    assert!(matches!(error, MetalError::Unavailable { .. }), "{error}");
    assert_eq!(error.class(), ErrorClass::Unavailable);
}

#[cfg(target_os = "macos")]
mod gpu {
    use std::time::{Duration, Instant};

    use jolt_metal::runtime::{
        host_name, Batch, Binding, DeviceBuffer, Grid, LibrarySpec, Pipeline, ShaderLibrary,
    };
    use jolt_metal::CapacityLimit;

    use super::support::{gpu, SplitMix64};
    use super::*;

    const RUNTIME_METAL: &str = include_str!("shaders/runtime.metal");
    const VEC_ADD: &str = "jolt_test_vec_add";
    const FILL_BYTES: &str = "jolt_test_fill_bytes";

    fn runtime_library(device: &Device) -> ShaderLibrary {
        let spec = LibrarySpec::new()
            .source("runtime.metal", RUNTIME_METAL)
            .instantiate::<u32>(VEC_ADD)
            .instantiate::<u64>(VEC_ADD)
            .kernel(FILL_BYTES);
        ShaderLibrary::compile(device, &spec).unwrap()
    }

    fn setup_error(device: &Device, spec: &LibrarySpec) -> MetalError {
        let error = ShaderLibrary::compile(device, spec).err().unwrap();
        assert_eq!(error.class(), ErrorClass::Setup, "{error}");
        error
    }

    fn fault(result: Result<(), MetalError>) -> MetalError {
        let error = result.unwrap_err();
        assert_eq!(error.class(), ErrorClass::Fault, "{error}");
        error
    }

    fn group(pipeline: &Pipeline) -> usize {
        (pipeline.thread_execution_width() * 8).min(pipeline.max_total_threads_per_threadgroup())
    }

    /// `2^20 + 3` elements: not a multiple of any threadgroup size, so the
    /// partial last threadgroup is exercised.
    const LEN: usize = (1 << 20) + 3;

    #[test]
    fn vec_add_matches_wrapping_add_for_every_instance() {
        let (_gpu, device) = gpu("vec_add_matches_wrapping_add_for_every_instance");
        assert!(device.limits().apple_family >= 7);
        let library = runtime_library(&device);

        let mut words = SplitMix64(0x6a6f_6c74_6d65_7461);
        let mut a: Vec<u64> = words.by_ref().take(LEN).collect();
        let b: Vec<u64> = words.by_ref().take(LEN).collect();
        a[0] = u64::MAX;
        a[1] = u64::from(u32::MAX);
        let a32: Vec<u32> = a.iter().map(|&x| x as u32).collect();
        let b32: Vec<u32> = b.iter().map(|&x| x as u32).collect();

        let pipeline64 = library.pipeline(&host_name::<u64>(VEC_ADD)).unwrap();
        let pipeline32 = library.pipeline(&host_name::<u32>(VEC_ADD)).unwrap();
        let (a_dev, b_dev) = (
            DeviceBuffer::from_slice(&device, &a).unwrap(),
            DeviceBuffer::from_slice(&device, &b).unwrap(),
        );
        let (a32_dev, b32_dev) = (
            DeviceBuffer::from_slice(&device, &a32).unwrap(),
            DeviceBuffer::from_slice(&device, &b32).unwrap(),
        );
        let mut out = DeviceBuffer::<u64>::zeroed(&device, LEN).unwrap();
        let mut out32 = DeviceBuffer::<u32>::zeroed(&device, LEN).unwrap();

        let mut batch = Batch::new(&device).unwrap();
        let bindings = [
            Binding::buffer(&a_dev),
            Binding::buffer(&b_dev),
            Binding::buffer(&out),
        ];
        batch
            .dispatch(pipeline64, &bindings, Grid::linear(LEN, group(pipeline64)))
            .unwrap();
        let bindings32 = [
            Binding::buffer(&a32_dev),
            Binding::buffer(&b32_dev),
            Binding::buffer(&out32),
        ];
        batch
            .dispatch(
                pipeline32,
                &bindings32,
                Grid::linear(LEN, group(pipeline32)),
            )
            .unwrap();
        let _ = batch.commit_and_wait().unwrap();

        let expected: Vec<u64> = a.iter().zip(&b).map(|(x, y)| x.wrapping_add(*y)).collect();
        let expected32: Vec<u32> = a32
            .iter()
            .zip(&b32)
            .map(|(x, y)| x.wrapping_add(*y))
            .collect();
        assert_eq!(out.read().unwrap(), expected);
        assert_eq!(out32.read().unwrap(), expected32);
    }

    /// A later dispatch in the same batch reads what an earlier one wrote.
    #[test]
    fn dispatches_in_a_batch_run_in_order() {
        let (_gpu, device) = gpu("dispatches_in_a_batch_run_in_order");
        let library = runtime_library(&device);
        let pipeline = library.pipeline(&host_name::<u32>(VEC_ADD)).unwrap();
        let a: Vec<u32> = SplitMix64(7).take(LEN).map(|x| x as u32).collect();
        let a_dev = DeviceBuffer::from_slice(&device, &a).unwrap();
        let sum = DeviceBuffer::<u32>::zeroed(&device, LEN).unwrap();
        let mut triple = DeviceBuffer::<u32>::zeroed(&device, LEN).unwrap();

        let grid = Grid::linear(LEN, group(pipeline));
        let mut batch = Batch::new(&device).unwrap();
        let first = [
            Binding::buffer(&a_dev),
            Binding::buffer(&a_dev),
            Binding::buffer(&sum),
        ];
        batch.dispatch(pipeline, &first, grid).unwrap();
        let second = [
            Binding::buffer(&sum),
            Binding::buffer(&a_dev),
            Binding::buffer(&triple),
        ];
        batch.dispatch(pipeline, &second, grid).unwrap();
        let submitted = Instant::now();
        let gpu_time = batch.commit_and_wait().unwrap();
        let wall_time = submitted.elapsed();

        let expected: Vec<u32> = a.iter().map(|x| x.wrapping_mul(3)).collect();
        assert_eq!(triple.read().unwrap(), expected);
        // GPU time is measured inside the host's wait for the batch.
        assert!(
            Duration::ZERO < gpu_time && gpu_time <= wall_time,
            "gpu {gpu_time:?}, wall {wall_time:?}"
        );
    }

    #[test]
    fn invalid_dispatches_are_faults_that_leave_the_batch_usable() {
        let (_gpu, device) = gpu("invalid_dispatches_are_faults_that_leave_the_batch_usable");
        let library = runtime_library(&device);
        let pipeline = library.pipeline(&host_name::<u32>(VEC_ADD)).unwrap();
        let max_group = pipeline.max_total_threads_per_threadgroup();
        let ones = DeviceBuffer::from_slice(&device, &[1u32; 64]).unwrap();
        let wide = DeviceBuffer::from_slice(&device, &[1u64; 64]).unwrap();
        let mut out = DeviceBuffer::<u32>::zeroed(&device, 64).unwrap();
        let grid = Grid::linear(64, 64);

        let mut batch = Batch::new(&device).unwrap();
        let rejected = [
            (
                vec![Binding::buffer(&ones), Binding::buffer(&out)],
                grid,
                "buffer arguments",
            ),
            (
                vec![
                    Binding::buffer(&ones),
                    Binding::buffer(&wide),
                    Binding::buffer(&out),
                ],
                grid,
                "expects 4-byte data",
            ),
            (
                vec![
                    Binding::buffer(&ones),
                    Binding::value(&1u32),
                    Binding::buffer(&out),
                ],
                Grid::linear(64, 0),
                "threadgroup size 0",
            ),
            (
                vec![
                    Binding::buffer(&ones),
                    Binding::buffer(&ones),
                    Binding::buffer(&out),
                ],
                Grid::linear(64, max_group + 1),
                "outside 1..=",
            ),
            (
                vec![
                    Binding::buffer(&ones),
                    Binding::buffer(&ones),
                    Binding::buffer(&out),
                ],
                Grid::linear(1 << 32, 64),
                "32-bit thread position",
            ),
        ];
        for (bindings, grid, reason) in &rejected {
            let error = fault(batch.dispatch(pipeline, bindings, *grid));
            assert!(
                matches!(&error, MetalError::InvalidDispatch { .. })
                    && error.to_string().contains(reason),
                "expected `{reason}`, got: {error}"
            );
        }
        let error = library.pipeline("jolt_test_vec_add").err().unwrap();
        assert!(matches!(error, MetalError::UnknownPipeline { .. }));
        assert_eq!(error.class(), ErrorClass::Fault);

        let valid = [
            Binding::buffer(&ones),
            Binding::buffer(&ones),
            Binding::buffer(&out),
        ];
        batch.dispatch(pipeline, &valid, grid).unwrap();
        let _ = batch.commit_and_wait().unwrap();
        assert_eq!(out.read().unwrap(), [2u32; 64]);
    }

    /// Invariant 6: an invalid bit pattern written by a kernel is caught on
    /// read-back, not handed to the caller.
    #[test]
    fn invalid_readback_is_a_fault() {
        let (_gpu, device) = gpu("invalid_readback_is_a_fault");
        let library = runtime_library(&device);
        let pipeline = library.pipeline(FILL_BYTES).unwrap();
        let mut flags = DeviceBuffer::<bool>::zeroed(&device, 100).unwrap();
        assert_eq!(flags.read().unwrap(), [false; 100]);

        for (value, valid) in [(1u8, true), (2u8, false)] {
            let mut batch = Batch::new(&device).unwrap();
            let bindings = [Binding::buffer(&flags), Binding::value(&value)];
            batch
                .dispatch(pipeline, &bindings, Grid::linear(100, 32))
                .unwrap();
            let _ = batch.commit_and_wait().unwrap();
            match flags.read() {
                Ok(read) => assert!(valid && read == [true; 100]),
                Err(error) => {
                    assert!(!valid);
                    assert_eq!(error.class(), ErrorClass::Fault);
                    assert!(error.to_string().contains("element 0"), "{error}");
                }
            }
        }
    }

    #[test]
    fn library_failures_are_setup_errors() {
        let (_gpu, device) = gpu("library_failures_are_setup_errors");
        let with = |source: &str| LibrarySpec::new().source("case.metal", source);
        let header = "#include <metal_stdlib>\nusing namespace metal;\n";

        let error = setup_error(&device, &with("kernel void k( {").kernel("k"));
        assert!(
            matches!(error, MetalError::ShaderCompile { ref log } if log.contains("case.metal"))
        );

        let needs_wide = format!(
            "{header}template <typename T> kernel void wide(device T* out [[buffer(0)]]) {{ \
             static_assert(sizeof(T) == 8, \"needs a 64-bit type\"); out[0] = 0; }}"
        );
        let error = setup_error(&device, &with(&needs_wide).instantiate::<u32>("wide"));
        assert!(
            matches!(error, MetalError::ShaderCompile { ref log } if log.contains("needs a 64-bit type"))
        );
        let _ =
            ShaderLibrary::compile(&device, &with(&needs_wide).instantiate::<u64>("wide")).unwrap();

        let error = setup_error(&device, &with(RUNTIME_METAL).kernel("missing"));
        assert!(matches!(error, MetalError::Pipeline { .. }), "{error}");

        let gap = format!(
            "{header}kernel void gap(device uint* a [[buffer(0)]], device uint* b [[buffer(2)]]) \
             {{ a[0] = b[0]; }}"
        );
        let error = setup_error(&device, &with(&gap).kernel("gap"));
        assert!(error.to_string().contains("contiguous"), "{error}");

        let scratch = format!(
            "{header}kernel void scratch(device uint* out [[buffer(0)]], \
             threadgroup uint* tmp [[threadgroup(0)]]) {{ tmp[0] = 1; out[0] = tmp[0]; }}"
        );
        let error = setup_error(&device, &with(&scratch).kernel("scratch"));
        assert!(error.to_string().contains("not a buffer"), "{error}");
    }

    #[test]
    fn oversized_buffers_are_refused_before_allocation() {
        let (_gpu, device) = gpu("oversized_buffers_are_refused_before_allocation");
        let limits = device.limits();
        let cases = [
            (
                DeviceBuffer::<u32>::zeroed(&device, limits.max_buffer_length / 4 + 1).err(),
                CapacityLimit::MaxBufferLength,
            ),
            (
                DeviceBuffer::<u64>::zeroed(&device, usize::MAX).err(),
                CapacityLimit::AddressSpace,
            ),
        ];
        for (error, expected) in cases {
            let error = error.unwrap();
            assert_eq!(error.class(), ErrorClass::Capacity);
            assert!(
                matches!(error, MetalError::CapacityExceeded { limit, .. } if limit == expected),
                "{error}"
            );
        }
    }

    #[test]
    fn empty_buffers_and_grids_are_no_ops() {
        let (_gpu, device) = gpu("empty_buffers_and_grids_are_no_ops");
        let library = runtime_library(&device);
        let pipeline = library.pipeline(&host_name::<u32>(VEC_ADD)).unwrap();
        let empty = DeviceBuffer::<u32>::from_slice(&device, &[]).unwrap();
        let mut out = DeviceBuffer::<u32>::zeroed(&device, 0).unwrap();
        assert!(out.is_empty());

        let mut batch = Batch::new(&device).unwrap();
        let bindings = [
            Binding::buffer(&empty),
            Binding::buffer(&empty),
            Binding::buffer(&out),
        ];
        batch
            .dispatch(pipeline, &bindings, Grid::linear(0, 1))
            .unwrap();
        let _ = batch.commit_and_wait().unwrap();
        assert!(out.read().unwrap().is_empty());
        // A batch dropped without committing runs nothing and must not raise.
        drop(Batch::new(&device).unwrap());
    }
}
