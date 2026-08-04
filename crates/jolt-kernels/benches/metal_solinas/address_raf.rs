use std::{env, hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput};
use jolt_field::{AdditiveAccumulator, AkitaAccumulator, AkitaField, MulPrimitiveInt};
use jolt_kernels::metal::solinas::{
    AddressRafScanConfig, AddressRafScanRow, Fp128, SolinasMetal, ADDRESS_RAF_BINS,
    ADDRESS_RAF_LANES,
};
use jolt_lookup_tables::uninterleave_bits;
use rayon::prelude::*;

const DEFAULT_ELEMENTS: [usize; 3] = [1 << 16, 1 << 20, 1 << 22];

pub fn bench(c: &mut Criterion, context: &SolinasMetal) {
    let config = config();
    let mut group = c.benchmark_group("metal_sumcheck/instruction_read_raf_address_phase");
    let _ = group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5));

    for elements in cases() {
        let (rows, weights) = inputs(elements);
        let invocation = context
            .prepare_address_raf_scan(&rows, &weights, config)
            .expect("address RAF pipeline should prepare");
        invocation
            .execute()
            .expect("address RAF scan should execute");
        let expected = cpu_scan(&rows, &weights, config.suffix_len);
        assert_eq!(
            invocation
                .read_output()
                .expect("address RAF output should be readable")
                .as_flat_slice(),
            expected
        );
        eprintln!(
            "address RAF: elements={elements}, threadgroups={}, threads={}, static TG bytes={}",
            invocation.threadgroup_count(),
            invocation.threads_per_threadgroup(),
            invocation
                .pipeline_limits()
                .static_threadgroup_memory_length,
        );

        let _ = group.throughput(Throughput::Elements(elements as u64));
        let suffix = format!(
            "n{elements}_r{}_t{}",
            config.rows_per_threadgroup,
            invocation.threads_per_threadgroup()
        );
        let _ = group.bench_function(BenchmarkId::new("cpu_optimized_scan", &suffix), |b| {
            b.iter(|| black_box(cpu_scan(&rows, &weights, config.suffix_len)));
        });

        let mut output = vec![Fp128::ZERO; ADDRESS_RAF_LANES * ADDRESS_RAF_BINS];
        let _ = group.bench_function(BenchmarkId::new("metal_resident_wall", &suffix), |b| {
            b.iter(|| {
                invocation
                    .execute()
                    .expect("address RAF scan should execute");
                invocation
                    .read_output_into(&mut output)
                    .expect("address RAF output should be canonical");
                let _ = black_box(&output);
            });
        });
        let _ = group.bench_function(BenchmarkId::new("metal_resident_active", &suffix), |b| {
            b.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    elapsed += invocation
                        .execute_timed()
                        .expect("timed address RAF scan should execute");
                }
                elapsed
            });
        });
    }
    group.finish();
}

fn cpu_scan(rows: &[AddressRafScanRow], weights: &[Fp128], suffix_len: u32) -> Vec<Fp128> {
    let threads = rayon::current_num_threads();
    let chunk_len = rows.len().div_ceil(threads).max(1024);
    let suffix_mask = if suffix_len == 0 {
        0
    } else {
        (1u128 << suffix_len) - 1
    };
    let upper_bits = suffix_len.saturating_sub(64);
    let sums = rows
        .par_chunks(chunk_len)
        .zip(weights.par_chunks(chunk_len))
        .map(|(rows, weights)| {
            let mut sums = vec![AkitaAccumulator::default(); ADDRESS_RAF_LANES * ADDRESS_RAF_BINS];
            for (&row, &weight) in rows.iter().zip(weights) {
                let lookup_index = row.lookup_index();
                let chunk = ((lookup_index >> suffix_len) as usize) & (ADDRESS_RAF_BINS - 1);
                let suffix = lookup_index & suffix_mask;
                let weight: AkitaField = weight.into_jolt_field();
                if row.raf_flag() {
                    sums[3 * ADDRESS_RAF_BINS + chunk].add(weight);
                    if suffix != 0 {
                        sums[4 * ADDRESS_RAF_BINS + chunk].add(weight.mul_u128(suffix));
                    }
                    let upper_mask = if upper_bits == 0 {
                        0
                    } else {
                        (1u128 << upper_bits) - 1
                    };
                    if upper_bits == 0 || suffix >> 64 == upper_mask {
                        sums[5 * ADDRESS_RAF_BINS + chunk].add(weight);
                    }
                } else {
                    let (left, right) = uninterleave_bits(suffix);
                    sums[chunk].add(weight);
                    if left != 0 {
                        sums[ADDRESS_RAF_BINS + chunk].add(weight.mul_u64(left));
                    }
                    if right != 0 {
                        sums[2 * ADDRESS_RAF_BINS + chunk].add(weight.mul_u64(right));
                    }
                }
            }
            sums
        })
        .reduce(
            || vec![AkitaAccumulator::default(); ADDRESS_RAF_LANES * ADDRESS_RAF_BINS],
            |mut lhs, rhs| {
                for (lhs, rhs) in lhs.iter_mut().zip(rhs) {
                    lhs.merge(rhs);
                }
                lhs
            },
        );
    sums.into_iter()
        .map(|sum| Fp128::from_jolt_field(&sum.reduce()))
        .collect()
}

fn inputs(elements: usize) -> (Vec<AddressRafScanRow>, Vec<Fp128>) {
    let mut state = 0x243f_6a88_85a3_08d3;
    let mut rows = Vec::with_capacity(elements);
    let mut weights = Vec::with_capacity(elements);
    for index in 0..elements {
        let lookup_index =
            (u128::from(splitmix(&mut state)) << 64) | u128::from(splitmix(&mut state));
        rows.push(AddressRafScanRow::new(lookup_index, index % 3 == 0));
        let weight = u128::from(splitmix(&mut state))
            | (u128::from(splitmix(&mut state) & 0x7fff_ffff_ffff_ffff) << 64);
        weights.push(Fp128::from_u128(weight));
    }
    (rows, weights)
}

fn cases() -> Vec<usize> {
    env::var("JOLT_SOLINAS_BENCH_ELEMENTS").map_or_else(
        |_| DEFAULT_ELEMENTS.to_vec(),
        |value| {
            let elements = value
                .parse::<usize>()
                .expect("benchmark element count should be an integer");
            assert!(elements >= 32 && elements.is_power_of_two());
            vec![elements]
        },
    )
}

fn config() -> AddressRafScanConfig {
    let rows_per_threadgroup =
        env::var("JOLT_METAL_ADDRESS_ROWS_PER_THREADGROUP").map_or(1 << 16, |value| {
            value
                .parse::<usize>()
                .expect("rows per threadgroup should be an integer")
        });
    let threads_per_threadgroup = env::var("JOLT_METAL_ADDRESS_THREADS").map_or(128, |value| {
        value
            .parse::<usize>()
            .expect("threadgroup width should be an integer")
    });
    let suffix_len = env::var("JOLT_METAL_ADDRESS_SUFFIX_LEN").map_or(120, |value| {
        value
            .parse::<u32>()
            .expect("suffix length should be an integer")
    });
    AddressRafScanConfig {
        suffix_len,
        rows_per_threadgroup,
        threads_per_threadgroup: Some(threads_per_threadgroup),
    }
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
