//! Direct kernel tests requiring a usable CUDA device.
#![cfg(all(feature = "cuda", not(feature = "akita")))]
#![expect(
    clippy::expect_used,
    reason = "kernel fixtures must fail on device or launch errors"
)]

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use jolt_field::{Fr, Ring};
use jolt_kernels::cuda::LIMBS;

const POISON: u64 = 0xA5A5_A5A5_A5A5_A5A5;

struct Kernels {
    stream: Arc<CudaStream>,
    scatter: CudaFunction,
    fold: CudaFunction,
}

impl Kernels {
    fn new() -> Self {
        let context = CudaContext::new(0).expect("a CUDA device");
        let module = context
            .load_module(Ptx::from_binary(
                include_bytes!(concat!(env!("OUT_DIR"), "/kernels.cubin")).to_vec(),
            ))
            .expect("load the compiled production module");
        Self {
            stream: context.default_stream(),
            scatter: module
                .load_function("pcr_scatter_kernel")
                .expect("scatter kernel"),
            fold: module
                .load_function("pcr_value_fold_kernel")
                .expect("value fold kernel"),
        }
    }
}

fn launch_config(count: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (count.div_ceil(256).max(1), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn values(len: usize) -> Vec<Fr> {
    let wide = Fr::from_u128(u128::MAX);
    (0..len)
        .map(|i| match i % 5 {
            0 => Fr::from_u64(0),
            1 => Fr::from_u64(1),
            2 => -Fr::from_u64(1),
            3 => wide,
            _ => wide * Fr::from_u128(u128::MAX - i as u128),
        })
        .collect()
}

fn pack(values: &[Fr]) -> Vec<u64> {
    values
        .iter()
        .flat_map(|value| value.inner_limbs())
        .collect()
}

#[test]
fn scatter_preserves_unselected_cells_and_inputs() {
    let kernels = Kernels::new();
    for count in [0u32, 1, 31, 32, 33, 255, 256, 257, 513] {
        let table_len = count + 17;
        let mut indices: Vec<u32> = (0..table_len).collect();
        indices.rotate_left(table_len as usize / 3);
        indices.reverse();
        indices.truncate(count.max(1) as usize);
        let values = values(count.max(1) as usize);
        let packed = pack(&values);
        let device_indices = kernels.stream.clone_htod(&indices).expect("indices");
        let device_values = kernels.stream.clone_htod(&packed).expect("values");

        for row in [0u32, 3] {
            let initial = vec![POISON; (4 * table_len as usize + 2) * LIMBS];
            let mut expected = initial.clone();
            for (&index, value) in indices.iter().zip(&values).take(count as usize) {
                let start = (1 + row as usize * table_len as usize + index as usize) * LIMBS;
                expected[start..start + LIMBS].copy_from_slice(&value.inner_limbs());
            }
            let mut output = kernels.stream.clone_htod(&initial).expect("output");
            {
                let mut target = output.slice_mut(LIMBS..);
                let mut call = kernels.stream.launch_builder(&kernels.scatter);
                let _ = call.arg(&device_indices);
                let _ = call.arg(&device_values);
                let _ = call.arg(&count);
                let _ = call.arg(&table_len);
                let _ = call.arg(&row);
                let _ = call.arg(&mut target);
                // SAFETY: indices are distinct and below table_len, both inputs
                // cover count, and row selects one of four allocated rows.
                // The output view also retains a trailing guard; buffers do not alias.
                let _ = unsafe { call.launch(launch_config(count)) }.expect("scatter");
            }
            kernels.stream.synchronize().expect("scatter completion");
            assert_eq!(
                kernels.stream.clone_dtoh(&output).expect("output"),
                expected,
                "count={count}, row={row}"
            );
        }
        assert_eq!(
            kernels.stream.clone_dtoh(&device_indices).expect("indices"),
            indices
        );
        assert_eq!(
            kernels.stream.clone_dtoh(&device_values).expect("values"),
            packed
        );
    }
}

#[test]
fn value_fold_matches_factored_field_expression() {
    let kernels = Kernels::new();
    for table_len in [0u32, 1, 31, 32, 33, 255, 256, 257] {
        for chunk_count in [0u32, 1, 3, 7] {
            let row = values(table_len as usize);
            let scales: Vec<Fr> = (0..chunk_count)
                .map(|c| Fr::from_u64(u64::from(c) + 2))
                .collect();
            let shifts = values(chunk_count as usize + 3)[3..].to_vec();
            let weights = values(chunk_count as usize + 1)[1..].to_vec();
            let chunks: Vec<Fr> = scales
                .iter()
                .zip(&shifts)
                .flat_map(|(&scale, &shift)| row.iter().map(move |&value| value * scale + shift))
                .collect();
            let scale: Fr = scales.iter().zip(&weights).map(|(&s, &w)| s * w).sum();
            let shift: Fr = shifts.iter().zip(&weights).map(|(&s, &w)| s * w).sum();
            let mut expected = vec![POISON; (table_len as usize + 2) * LIMBS];
            for (i, &value) in row.iter().enumerate() {
                expected[(i + 1) * LIMBS..(i + 2) * LIMBS]
                    .copy_from_slice(&(value * scale + shift).inner_limbs());
            }

            let mut packed_chunks = pack(&chunks);
            let mut packed_weights = pack(&weights);
            // Empty domains still get valid pointers, so a nonzero grid can
            // exercise the kernel's guards without relying on zero-size allocations.
            packed_chunks.resize(packed_chunks.len().max(LIMBS), POISON);
            packed_weights.resize(packed_weights.len().max(LIMBS), POISON);
            let device_chunks = kernels.stream.clone_htod(&packed_chunks).expect("chunks");
            let device_weights = kernels.stream.clone_htod(&packed_weights).expect("weights");
            let mut output = kernels
                .stream
                .clone_htod(&vec![POISON; expected.len()])
                .expect("output");
            {
                let mut target = output.slice_mut(LIMBS..);
                let mut call = kernels.stream.launch_builder(&kernels.fold);
                let _ = call.arg(&device_chunks);
                let _ = call.arg(&device_weights);
                let _ = call.arg(&chunk_count);
                let _ = call.arg(&table_len);
                let _ = call.arg(&mut target);
                // SAFETY: chunks contains chunk_count complete rows, weights
                // covers every chunk, and target covers table_len plus a guard.
                // Distinct allocations prevent input/output aliasing.
                let _ = unsafe { call.launch(launch_config(table_len)) }.expect("value fold");
            }
            kernels.stream.synchronize().expect("fold completion");
            assert_eq!(
                kernels.stream.clone_dtoh(&output).expect("output"),
                expected,
                "table_len={table_len}, chunk_count={chunk_count}"
            );
            assert_eq!(
                kernels.stream.clone_dtoh(&device_chunks).expect("chunks"),
                packed_chunks
            );
            assert_eq!(
                kernels.stream.clone_dtoh(&device_weights).expect("weights"),
                packed_weights
            );
        }
    }
}
