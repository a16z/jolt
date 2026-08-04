use std::{env, hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput};
use jolt_field::{
    AdditiveAccumulator, AkitaAccumulator, AkitaField, FromPrimitiveInt, MulPrimitiveInt,
};
use jolt_kernels::metal::solinas::{
    AddressRafScanConfig, AddressRafScanRow, Fp128, SolinasMetal, ADDRESS_SUFFIX_BINS,
    ADDRESS_SUFFIX_TABLES,
};
use jolt_lookup_tables::{tables::Suffixes, LookupBits, LookupTableKind, XLEN as RISCV_XLEN};
use rayon::prelude::*;

const DEFAULT_ELEMENTS: [usize; 3] = [1 << 16, 1 << 20, 1 << 22];

pub fn bench_one(c: &mut Criterion, context: &SolinasMetal) {
    let config = config();
    let mut group = c.benchmark_group("metal_sumcheck/instruction_read_raf_address_suffix_one");
    let _ = group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5));

    for elements in cases() {
        let (rows, weights, buckets) = inputs(elements);
        let metal_weights: Vec<_> = weights.iter().map(Fp128::from_jolt_field).collect();
        let invocation = context
            .prepare_address_suffix_one(&rows, &metal_weights, config)
            .expect("address suffix pipeline should prepare");
        drop(metal_weights);
        invocation
            .execute()
            .expect("address suffix pipeline should execute");
        let expected = cpu_suffix_one(&rows, &weights, &buckets, config.suffix_len);
        assert_eq!(
            invocation
                .read_output()
                .expect("address suffix output should be readable")
                .as_flat_slice(),
            expected
        );
        eprintln!(
            "address suffix one: elements={elements}, jobs={}, threads={}, partial_bytes={}",
            invocation.job_count(),
            invocation.threads_per_threadgroup(),
            invocation.intermediate_partial_bytes(),
        );

        let _ = group.throughput(Throughput::Elements(elements as u64));
        let suffix = format!(
            "n{elements}_r{}_t{}",
            config.rows_per_threadgroup,
            invocation.threads_per_threadgroup()
        );
        let _ = group.bench_function(BenchmarkId::new("cpu_optimized_buckets", &suffix), |b| {
            b.iter(|| black_box(cpu_suffix_one(&rows, &weights, &buckets, config.suffix_len)));
        });
        let _ = group.bench_function(BenchmarkId::new("metal_resident_wall", &suffix), |b| {
            b.iter(|| {
                invocation
                    .execute()
                    .expect("address suffix pipeline should execute");
                black_box(
                    invocation
                        .read_output()
                        .expect("address suffix output should be readable"),
                )
            });
        });
        let _ = group.bench_function(BenchmarkId::new("metal_resident_active", &suffix), |b| {
            b.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    elapsed += invocation
                        .execute_timed()
                        .expect("timed address suffix pipeline should execute");
                }
                elapsed
            });
        });
    }
    group.finish();
}

pub fn bench_full(c: &mut Criterion, context: &SolinasMetal) {
    let config = config();
    let mut group = c.benchmark_group("metal_sumcheck/instruction_read_raf_address_suffix_full");
    let _ = group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5));

    for elements in cases() {
        let (rows, weights, buckets) = inputs(elements);
        let metal_weights: Vec<_> = weights.iter().map(Fp128::from_jolt_field).collect();
        let invocation = context
            .prepare_address_suffix_full(&rows, &metal_weights, config)
            .expect("full address suffix pipeline should prepare");
        drop(metal_weights);
        invocation
            .execute()
            .expect("full address suffix pipeline should execute");
        let expected = cpu_suffix_full(&rows, &weights, &buckets, config.suffix_len);
        assert_eq!(
            invocation
                .read_output()
                .expect("full address suffix output should be readable")
                .as_flat_slice(),
            expected
        );
        eprintln!(
            "full address suffix: elements={elements}, jobs={}, threads={}, partial_bytes={}",
            invocation.job_count(),
            invocation.threads_per_threadgroup(),
            invocation.intermediate_partial_bytes(),
        );

        let _ = group.throughput(Throughput::Elements(elements as u64));
        let suffix = format!(
            "n{elements}_r{}_t{}",
            config.rows_per_threadgroup,
            invocation.threads_per_threadgroup()
        );
        let _ = group.bench_function(BenchmarkId::new("cpu_optimized_buckets", &suffix), |b| {
            b.iter(|| {
                black_box(cpu_suffix_full(
                    &rows,
                    &weights,
                    &buckets,
                    config.suffix_len,
                ))
            });
        });
        let _ = group.bench_function(BenchmarkId::new("metal_resident_wall", &suffix), |b| {
            b.iter(|| {
                invocation
                    .execute()
                    .expect("full address suffix pipeline should execute");
                black_box(
                    invocation
                        .read_output()
                        .expect("full address suffix output should be readable"),
                )
            });
        });
        let _ = group.bench_function(BenchmarkId::new("metal_resident_active", &suffix), |b| {
            b.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    elapsed += invocation
                        .execute_timed()
                        .expect("timed full address suffix pipeline should execute");
                }
                elapsed
            });
        });
    }
    group.finish();
}

fn cpu_suffix_one(
    rows: &[AddressRafScanRow],
    weights: &[AkitaField],
    buckets: &[Vec<u32>],
    suffix_len: u32,
) -> Vec<Fp128> {
    buckets
        .par_iter()
        .flat_map_iter(|bucket| {
            let chunk_len = bucket
                .len()
                .div_ceil(rayon::current_num_threads())
                .max(1024);
            bucket
                .par_chunks(chunk_len)
                .map(|chunk| {
                    let mut sums = vec![AkitaAccumulator::default(); ADDRESS_SUFFIX_BINS];
                    for &row in chunk {
                        let row = row as usize;
                        let key = ((rows[row].lookup_index() >> suffix_len) as usize)
                            & (ADDRESS_SUFFIX_BINS - 1);
                        sums[key].add(weights[row]);
                    }
                    sums
                })
                .reduce(
                    || vec![AkitaAccumulator::default(); ADDRESS_SUFFIX_BINS],
                    |mut lhs, rhs| {
                        for (lhs, rhs) in lhs.iter_mut().zip(rhs) {
                            lhs.merge(rhs);
                        }
                        lhs
                    },
                )
                .into_iter()
                .map(|sum| Fp128::from_jolt_field(&sum.reduce()))
        })
        .collect()
}

fn cpu_suffix_full(
    rows: &[AddressRafScanRow],
    weights: &[AkitaField],
    buckets: &[Vec<u32>],
    suffix_len: u32,
) -> Vec<Fp128> {
    let tables: Vec<_> = LookupTableKind::<RISCV_XLEN>::iter().collect();
    let suffix_mask = if suffix_len == 0 {
        0
    } else {
        (1u128 << suffix_len) - 1
    };
    tables
        .par_iter()
        .flat_map_iter(|table| {
            let suffixes = table.suffixes();
            let one_position = suffixes
                .iter()
                .position(|suffix| matches!(suffix, Suffixes::One));
            let bucket = &buckets[table.index()];
            let chunk_len = bucket
                .len()
                .div_ceil(rayon::current_num_threads())
                .max(1024);
            bucket
                .par_chunks(chunk_len)
                .map(|chunk| {
                    let mut sums =
                        vec![AkitaAccumulator::default(); suffixes.len() * ADDRESS_SUFFIX_BINS];
                    for &row in chunk {
                        let row = row as usize;
                        let lookup = rows[row].lookup_index();
                        let key = ((lookup >> suffix_len) as usize) & (ADDRESS_SUFFIX_BINS - 1);
                        let suffix_bits =
                            LookupBits::new(lookup & suffix_mask, suffix_len as usize);
                        for (suffix_index, suffix) in suffixes.iter().enumerate() {
                            let slot = &mut sums[suffix_index * ADDRESS_SUFFIX_BINS + key];
                            if one_position == Some(suffix_index) {
                                slot.add(weights[row]);
                            } else if suffix.is_01_valued() {
                                if suffix.suffix_mle(suffix_bits) == 1 {
                                    slot.add(weights[row]);
                                }
                            } else {
                                let scalar = suffix.suffix_mle(suffix_bits);
                                if scalar != 0 {
                                    slot.add(weights[row].mul_u64(scalar));
                                }
                            }
                        }
                    }
                    sums
                })
                .reduce(
                    || vec![AkitaAccumulator::default(); suffixes.len() * ADDRESS_SUFFIX_BINS],
                    |mut lhs, rhs| {
                        for (lhs, rhs) in lhs.iter_mut().zip(rhs) {
                            lhs.merge(rhs);
                        }
                        lhs
                    },
                )
                .into_iter()
                .map(|sum| Fp128::from_jolt_field(&sum.reduce()))
        })
        .collect()
}

fn inputs(elements: usize) -> (Vec<AddressRafScanRow>, Vec<AkitaField>, Vec<Vec<u32>>) {
    let mut state = 0x243f_6a88_85a3_08d3;
    let mut rows = Vec::with_capacity(elements);
    let mut weights = Vec::with_capacity(elements);
    let mut buckets: Vec<Vec<u32>> = (0..ADDRESS_SUFFIX_TABLES)
        .map(|_| Vec::with_capacity(elements / ADDRESS_SUFFIX_TABLES))
        .collect();
    for index in 0..elements {
        let lookup_index =
            (u128::from(splitmix(&mut state)) << 64) | u128::from(splitmix(&mut state));
        let table = index % ADDRESS_SUFFIX_TABLES;
        rows.push(AddressRafScanRow::new_with_table(
            lookup_index,
            Some(table),
            index % 3 == 0,
        ));
        buckets[table].push(index as u32);
        let weight = u128::from(splitmix(&mut state))
            | (u128::from(splitmix(&mut state) & 0x7fff_ffff_ffff_ffff) << 64);
        weights.push(AkitaField::from_u128(weight));
    }
    (rows, weights, buckets)
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
        env::var("JOLT_METAL_ADDRESS_ROWS_PER_THREADGROUP").map_or(1 << 15, |value| {
            value
                .parse::<usize>()
                .expect("rows per threadgroup should be an integer")
        });
    let threads_per_threadgroup = env::var("JOLT_METAL_ADDRESS_THREADS").map_or(1024, |value| {
        value
            .parse::<usize>()
            .expect("threadgroup width should be an integer")
    });
    let suffix_len = env::var("JOLT_METAL_ADDRESS_SUFFIX_LEN").map_or(112, |value| {
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
