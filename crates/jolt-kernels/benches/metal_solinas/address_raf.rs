use std::{env, hint::black_box, time::Duration};

use criterion::{measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion, Throughput};
use jolt_field::{
    AdditiveAccumulator, AkitaAccumulator, AkitaField, FromPrimitiveInt, MulPrimitiveInt,
};
use jolt_kernels::metal::solinas::{
    AddressRafDirectInvocation, AddressRafScanConfig, AddressRafScanInvocation, AddressRafScanRow,
    AddressRafSums, Fp128, PipelineLimits, SolinasMetal, ADDRESS_RAF_BINS, ADDRESS_RAF_LANES,
};
use jolt_lookup_tables::uninterleave_bits;
use rayon::prelude::*;

const DEFAULT_ELEMENTS: [usize; 3] = [1 << 16, 1 << 20, 1 << 22];

pub fn bench(c: &mut Criterion, context: &SolinasMetal) {
    bench_impl(c, context, false, false);
}

pub fn bench_condensed(c: &mut Criterion, context: &SolinasMetal) {
    bench_impl(c, context, true, false);
}

pub fn bench_direct_condensed(c: &mut Criterion, context: &SolinasMetal) {
    bench_impl(c, context, true, true);
}

enum AddressBenchmarkInvocation<'a> {
    Contribution(AddressRafScanInvocation<'a>),
    Direct(AddressRafDirectInvocation<'a>),
}

impl AddressBenchmarkInvocation<'_> {
    fn execute(&self) {
        match self {
            Self::Contribution(invocation) => invocation.execute(),
            Self::Direct(invocation) => invocation.execute(),
        }
        .expect("address RAF scan should execute");
    }

    fn execute_timed(&self) -> Duration {
        match self {
            Self::Contribution(invocation) => invocation.execute_timed(),
            Self::Direct(invocation) => invocation.execute_timed(),
        }
        .expect("timed address RAF scan should execute")
    }

    fn read_output(&self) -> AddressRafSums {
        match self {
            Self::Contribution(invocation) => invocation.read_output(),
            Self::Direct(invocation) => invocation.read_output(),
        }
        .expect("address RAF output should be readable")
    }

    fn read_output_into(&self, output: &mut [Fp128]) {
        match self {
            Self::Contribution(invocation) => invocation.read_output_into(output),
            Self::Direct(invocation) => invocation.read_output_into(output),
        }
        .expect("address RAF output should be canonical");
    }

    fn threadgroup_count(&self) -> usize {
        match self {
            Self::Contribution(invocation) => invocation.threadgroup_count(),
            Self::Direct(invocation) => invocation.threadgroup_count(),
        }
    }

    fn threads_per_threadgroup(&self) -> usize {
        match self {
            Self::Contribution(invocation) => invocation.threads_per_threadgroup(),
            Self::Direct(invocation) => invocation.threads_per_threadgroup(),
        }
    }

    fn pipeline_limits(&self) -> PipelineLimits {
        match self {
            Self::Contribution(invocation) => invocation.pipeline_limits(),
            Self::Direct(invocation) => invocation.pipeline_limits(),
        }
    }
}

fn bench_impl(c: &mut Criterion, context: &SolinasMetal, condense: bool, direct: bool) {
    let config = config(condense, direct);
    let metal_first = metal_first();
    let group_name = if direct {
        "metal_sumcheck/instruction_read_raf_address_direct_condensed_phase"
    } else if condense {
        "metal_sumcheck/instruction_read_raf_address_condensed_phase"
    } else {
        "metal_sumcheck/instruction_read_raf_address_phase"
    };
    let mut group = c.benchmark_group(group_name);
    let _ = group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5));

    for elements in cases() {
        let (rows, weights) = inputs(elements);
        let metal_weights: Vec<_> = weights.iter().map(Fp128::from_jolt_field).collect();
        let previous_phase_table = previous_phase_table();
        let metal_previous_phase_table =
            previous_phase_table.map(|value| Fp128::from_jolt_field(&value));
        let invocation = if direct {
            AddressBenchmarkInvocation::Direct(
                context
                    .prepare_direct_condensed_address_raf_scan(
                        &rows,
                        &metal_weights,
                        &metal_previous_phase_table,
                        config,
                    )
                    .expect("direct address RAF pipeline should prepare"),
            )
        } else if condense {
            AddressBenchmarkInvocation::Contribution(
                context
                    .prepare_condensed_address_raf_scan(
                        &rows,
                        &metal_weights,
                        &metal_previous_phase_table,
                        config,
                    )
                    .expect("address RAF pipeline should prepare"),
            )
        } else {
            AddressBenchmarkInvocation::Contribution(
                context
                    .prepare_address_raf_scan(&rows, &metal_weights, config)
                    .expect("address RAF pipeline should prepare"),
            )
        };
        drop(metal_weights);
        invocation.execute();
        let mut expected_weights = weights.clone();
        if condense {
            condense_weights(
                &rows,
                &mut expected_weights,
                &previous_phase_table,
                config.suffix_len,
            );
        }
        let expected = cpu_scan(&rows, &expected_weights, config.suffix_len);
        assert_eq!(invocation.read_output().as_flat_slice(), expected);
        drop(expected_weights);
        drop(expected);
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
        if metal_first {
            benchmark_metal(&mut group, &suffix, &invocation);
            benchmark_cpu(
                &mut group,
                &suffix,
                &rows,
                &weights,
                &previous_phase_table,
                config,
                condense,
            );
        } else {
            benchmark_cpu(
                &mut group,
                &suffix,
                &rows,
                &weights,
                &previous_phase_table,
                config,
                condense,
            );
            benchmark_metal(&mut group, &suffix, &invocation);
        }
    }
    group.finish();
}

fn benchmark_cpu(
    group: &mut BenchmarkGroup<'_, WallTime>,
    suffix: &str,
    rows: &[AddressRafScanRow],
    weights: &[AkitaField],
    previous_phase_table: &[AkitaField; ADDRESS_RAF_BINS],
    config: AddressRafScanConfig,
    condense: bool,
) {
    if condense {
        let mut cpu_weights = weights.to_vec();
        let _ = group.bench_function(
            BenchmarkId::new("cpu_optimized_condense_then_scan", suffix),
            |b| {
                b.iter(|| {
                    condense_weights(
                        rows,
                        &mut cpu_weights,
                        previous_phase_table,
                        config.suffix_len,
                    );
                    black_box(cpu_scan(rows, &cpu_weights, config.suffix_len))
                });
            },
        );
    } else {
        let _ = group.bench_function(BenchmarkId::new("cpu_optimized_scan", suffix), |b| {
            b.iter(|| black_box(cpu_scan(rows, weights, config.suffix_len)));
        });
    }
}

fn benchmark_metal(
    group: &mut BenchmarkGroup<'_, WallTime>,
    suffix: &str,
    invocation: &AddressBenchmarkInvocation<'_>,
) {
    let mut output = vec![Fp128::ZERO; ADDRESS_RAF_LANES * ADDRESS_RAF_BINS];
    let _ = group.bench_function(BenchmarkId::new("metal_resident_wall", suffix), |b| {
        b.iter(|| {
            invocation.execute();
            invocation.read_output_into(&mut output);
            let _ = black_box(&output);
        });
    });
    let _ = group.bench_function(BenchmarkId::new("metal_resident_active", suffix), |b| {
        b.iter_custom(|iterations| {
            let mut elapsed = Duration::ZERO;
            for _ in 0..iterations {
                elapsed += invocation.execute_timed();
            }
            elapsed
        });
    });
}

fn cpu_scan(rows: &[AddressRafScanRow], weights: &[AkitaField], suffix_len: u32) -> Vec<Fp128> {
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

fn condense_weights(
    rows: &[AddressRafScanRow],
    weights: &mut [AkitaField],
    previous_phase_table: &[AkitaField; ADDRESS_RAF_BINS],
    suffix_len: u32,
) {
    weights
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, weight)| {
            let previous_chunk = ((rows[index].lookup_index() >> (suffix_len + 8)) as usize)
                & (ADDRESS_RAF_BINS - 1);
            *weight *= previous_phase_table[previous_chunk];
        });
}

fn previous_phase_table() -> [AkitaField; ADDRESS_RAF_BINS] {
    let mut state = 0x1319_8a2e_0370_7344;
    std::array::from_fn(|_| {
        let value = u128::from(splitmix(&mut state))
            | (u128::from(splitmix(&mut state) & 0x7fff_ffff_ffff_ffff) << 64);
        AkitaField::from_u128(value)
    })
}

fn inputs(elements: usize) -> (Vec<AddressRafScanRow>, Vec<AkitaField>) {
    let mut state = 0x243f_6a88_85a3_08d3;
    let mut rows = Vec::with_capacity(elements);
    let mut weights = Vec::with_capacity(elements);
    for index in 0..elements {
        let lookup_index =
            (u128::from(splitmix(&mut state)) << 64) | u128::from(splitmix(&mut state));
        rows.push(AddressRafScanRow::new(lookup_index, index % 3 == 0));
        let weight = u128::from(splitmix(&mut state))
            | (u128::from(splitmix(&mut state) & 0x7fff_ffff_ffff_ffff) << 64);
        weights.push(AkitaField::from_u128(weight));
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

fn metal_first() -> bool {
    env::var("JOLT_METAL_ADDRESS_BENCH_ORDER").is_ok_and(|value| match value.as_str() {
        "cpu-first" => false,
        "metal-first" => true,
        _ => panic!("benchmark order should be `cpu-first` or `metal-first`"),
    })
}

fn config(condense: bool, direct: bool) -> AddressRafScanConfig {
    let default_rows_per_threadgroup = if direct { 1 << 15 } else { 1 << 16 };
    let rows_per_threadgroup = env::var("JOLT_METAL_ADDRESS_ROWS_PER_THREADGROUP").map_or(
        default_rows_per_threadgroup,
        |value| {
            value
                .parse::<usize>()
                .expect("rows per threadgroup should be an integer")
        },
    );
    let threads_per_threadgroup = env::var("JOLT_METAL_ADDRESS_THREADS").map_or(1024, |value| {
        value
            .parse::<usize>()
            .expect("threadgroup width should be an integer")
    });
    let default_suffix_len = if condense { 112 } else { 120 };
    let suffix_len =
        env::var("JOLT_METAL_ADDRESS_SUFFIX_LEN").map_or(default_suffix_len, |value| {
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
