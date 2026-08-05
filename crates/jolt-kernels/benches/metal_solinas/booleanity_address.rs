use std::{
    env,
    hint::black_box,
    mem::{self, size_of},
    time::{Duration, Instant},
};

use criterion::{BenchmarkId, Criterion, Throughput};
use jolt_field::{
    AdditiveAccumulator, AkitaAccumulator, AkitaField, FromPrimitiveInt, RingAccumulator,
};
use jolt_kernels::metal::solinas::{
    BooleanityAddressPushforwardConfig, BooleanityRow, BooleanitySelector, SolinasMetal,
};
use jolt_poly::TensorEqTable;
use rayon::prelude::*;

const K: usize = 256;
const CHUNK_BITS: usize = 8;
const POLYS: usize = 29;
const DEFAULT_ELEMENTS: [usize; 3] = [1 << 16, 1 << 20, 1 << 22];

struct CpuState {
    partial: Vec<AkitaAccumulator>,
    block: Vec<AkitaAccumulator>,
}

pub fn bench(c: &mut Criterion, context: &SolinasMetal) {
    let config = config();
    let mut group = c.benchmark_group("metal_sumcheck/booleanity_address_pushforward");
    let _ = group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5));

    for elements in cases(config.inner_log2) {
        let log_n = elements.ilog2() as usize;
        let selectors = selectors();
        assert_eq!(selectors.len(), POLYS);
        let rows = rows(elements, 1);
        let reference_cycle = point(log_n, 0xc1c1_e5e5);
        let resident_rows = context
            .prepare_booleanity_rows(&rows)
            .expect("Booleanity address rows should become resident");
        let resident_identity = resident_rows.allocation_identity();
        let invocation = context
            .prepare_booleanity_address_pushforward(
                resident_rows.clone(),
                &selectors,
                &reference_cycle,
                config,
            )
            .expect("Booleanity address pushforward should prepare");
        assert_eq!(invocation.resident_row_identity(), resident_identity);

        let expected = cpu_pushforward(&rows, &selectors, &reference_cycle);
        invocation
            .execute()
            .expect("Booleanity address validation should execute");
        let mut readback = vec![AkitaField::zero(); invocation.output_elements()];
        invocation
            .read_masses_into(&mut readback)
            .expect("Booleanity address validation masses should be readable");
        assert_eq!(readback, expected);
        assert_eq!(resident_rows.allocation_identity(), resident_identity);
        drop(expected);
        drop(rows);

        let useful_field_adds = u64::try_from(elements * selectors.len())
            .expect("Booleanity address useful work should fit in u64");
        let resident_row_bytes = u64::try_from(elements * size_of::<BooleanityRow>())
            .expect("Booleanity address resident rows should fit in u64");
        let row_scan_bytes =
            u64::try_from(elements * size_of::<BooleanityRow>() * invocation.selector_tiles())
                .expect("Booleanity address row traffic should fit in u64");
        let readback_bytes = u64::try_from(invocation.output_elements() * size_of::<AkitaField>())
            .expect("Booleanity address readback should fit in u64");
        eprintln!(
            "booleanity_address log_n={log_n} rows={elements} selectors={} selector_tiles={} criterion_element=selector_row_field_add useful_field_adds={useful_field_adds} resident_row_bytes={resident_row_bytes} logical_row_scan_bytes={row_scan_bytes} readback_bytes={readback_bytes} inner_log2={} e_in_elements={} e_out_elements={} tile_threads={} finalize_threads={} production_specialized={} exact=true",
            selectors.len(),
            invocation.selector_tiles(),
            config.inner_log2,
            invocation.e_in_length(),
            invocation.e_out_length(),
            invocation.tile_threads_per_threadgroup(),
            invocation.finalize_threads_per_threadgroup(),
            invocation.uses_production_specialization(),
        );

        let _ = group.throughput(Throughput::Elements(useful_field_adds));
        let id = BenchmarkId::new(
            "metal_prepared_execute_readback_wall",
            format!(
                "log{log_n}_i{}_s{}_tile{}_final{}",
                config.inner_log2,
                config.selectors_per_tile,
                invocation.tile_threads_per_threadgroup(),
                invocation.finalize_threads_per_threadgroup(),
            ),
        );
        let _ = group.bench_function(id, |bench| {
            bench.iter_custom(|iterations| {
                let mut measured = Duration::ZERO;
                for _ in 0..iterations {
                    let started = Instant::now();
                    invocation
                        .execute()
                        .expect("Booleanity address pushforward should execute");
                    invocation
                        .read_masses_into(&mut readback)
                        .expect("Booleanity address masses should be readable");
                    measured += started.elapsed();
                    let _ = black_box(readback[0]);
                }
                measured
            });
        });
    }
    group.finish();
}

fn config() -> BooleanityAddressPushforwardConfig {
    let defaults = BooleanityAddressPushforwardConfig::default();
    BooleanityAddressPushforwardConfig {
        inner_log2: env_usize(
            "JOLT_METAL_BOOLEANITY_ADDRESS_INNER_LOG2",
            defaults.inner_log2,
        ),
        selectors_per_tile: env_usize(
            "JOLT_METAL_BOOLEANITY_ADDRESS_SELECTORS_PER_TILE",
            defaults.selectors_per_tile,
        ),
        tile_threads_per_threadgroup: Some(env_usize(
            "JOLT_METAL_BOOLEANITY_ADDRESS_TILE_THREADS",
            defaults
                .tile_threads_per_threadgroup
                .expect("default tile width should be explicit"),
        )),
        finalize_threads_per_threadgroup: Some(env_usize(
            "JOLT_METAL_BOOLEANITY_ADDRESS_FINALIZE_THREADS",
            defaults
                .finalize_threads_per_threadgroup
                .expect("default finalize width should be explicit"),
        )),
    }
}

fn selectors() -> Vec<BooleanitySelector> {
    let mut selectors = (0..16)
        .map(|index| BooleanitySelector::Lookup {
            shift: (CHUNK_BITS * (15 - index)) as u32,
        })
        .collect::<Vec<_>>();
    selectors.extend([8, 0].map(|shift| BooleanitySelector::Bytecode { shift }));
    selectors.extend([8, 0].map(|shift| BooleanitySelector::Ram { shift }));
    selectors.extend((0..8).map(|index| BooleanitySelector::FusedInc {
        shift: (CHUNK_BITS * index) as u32,
    }));
    selectors.push(BooleanitySelector::FusedIncMsb);
    selectors
}

fn rows(count: usize, seed: u64) -> Vec<BooleanityRow> {
    (0..count)
        .into_par_iter()
        .map(|row| {
            let mut state = u128::from(splitmix(seed ^ row as u64))
                | (u128::from(splitmix(!seed ^ row.rotate_left(17) as u64)) << 64);
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 43;
            let mapped_pc =
                (!row.is_multiple_of(7)).then_some(((state >> 61) as u64) & ((1 << 55) - 2));
            let ram_address = (!row.is_multiple_of(11)).then_some((state as u64) & (u64::MAX - 1));
            let fused_inc = match row % 6 {
                0 => -(u64::MAX as i128),
                1 => -((1i128 << 63) + row as i128),
                2 => u64::MAX as i128 - row as i128,
                3 => (1i128 << 63) + row as i128,
                4 => row as i128,
                _ => -(row as i128),
            };
            BooleanityRow::new(state, mapped_pc, ram_address, fused_inc)
                .expect("benchmark row should be representable")
        })
        .collect()
}

fn point(count: usize, seed: u64) -> Vec<AkitaField> {
    (0..count)
        .map(|index| {
            let low = splitmix(seed ^ index as u64);
            let high = splitmix(!seed ^ (index as u64).rotate_left(23)) & 0x7fff_ffff_ffff_ffff;
            AkitaField::from_u128(u128::from(low) | (u128::from(high) << 64))
        })
        .collect()
}

fn splitmix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn cpu_pushforward(
    rows: &[BooleanityRow],
    selectors: &[BooleanitySelector],
    reference_cycle: &[AkitaField],
) -> Vec<AkitaField> {
    let eq = TensorEqTable::new(reference_cycle);
    let e_out = eq.e_out();
    let e_in = eq.e_in();
    let fields = selectors.len() * K;
    let zero = || CpuState {
        partial: vec![AkitaAccumulator::default(); fields],
        block: vec![AkitaAccumulator::default(); fields],
    };
    let scatter = |mut state: CpuState, x_out: usize| {
        let base = x_out * e_in.len();
        for (x_in, weight) in e_in.iter().copied().enumerate() {
            let row = rows[base + x_in];
            for (selector_index, selector) in selectors.iter().copied().enumerate() {
                if let Some(hot) = hot_index(row, selector) {
                    state.block[selector_index * K + hot].add(weight);
                }
            }
        }
        let outer = e_out[x_out];
        for (partial, block) in state.partial.iter_mut().zip(&mut state.block) {
            let value = mem::take(block).reduce();
            if value != AkitaField::zero() {
                partial.fmadd(outer, value);
            }
        }
        state
    };
    let merge = |mut left: CpuState, right: CpuState| {
        for (left, right) in left.partial.iter_mut().zip(right.partial) {
            left.merge(right);
        }
        left
    };
    (0..e_out.len())
        .into_par_iter()
        .fold(zero, scatter)
        .reduce(zero, merge)
        .partial
        .into_iter()
        .map(|accumulator| accumulator.reduce())
        .collect()
}

fn hot_index(row: BooleanityRow, selector: BooleanitySelector) -> Option<usize> {
    let words = row.words();
    let mask = (K - 1) as u64;
    match selector {
        BooleanitySelector::Lookup { shift } => {
            let word = if shift < 64 { words[0] } else { words[1] };
            let shift = if shift < 64 { shift } else { shift - 64 };
            Some(((word >> shift) & mask) as usize)
        }
        BooleanitySelector::Bytecode { shift } => {
            let plus_one = words[4] & 0x00ff_ffff_ffff_ffff;
            (plus_one != 0).then(|| (((plus_one - 1) >> shift) & mask) as usize)
        }
        BooleanitySelector::Ram { shift } => {
            (words[2] != 0).then(|| (((words[2] - 1) >> shift) & mask) as usize)
        }
        BooleanitySelector::FusedInc { shift } => {
            let (biased, _) = biased_inc(words);
            let standard = (biased >> shift) & mask;
            Some(((standard + (K / 2) as u64) & mask) as usize)
        }
        BooleanitySelector::FusedIncMsb => {
            let (_, carry) = biased_inc(words);
            Some((carry as usize) & (K - 1))
        }
    }
}

fn biased_inc(words: [u64; 5]) -> (u64, i32) {
    let radix = 1u128 << CHUNK_BITS;
    let bias = ((radix / 2) * (u128::from(u64::MAX) / (radix - 1))) as u64;
    let magnitude = words[3];
    if words[4] >> 63 != 0 {
        (
            bias.wrapping_sub(magnitude),
            if magnitude > bias { -1 } else { 0 },
        )
    } else {
        let biased = bias.wrapping_add(magnitude);
        (biased, i32::from(biased < bias))
    }
}

fn cases(inner_log2: usize) -> Vec<usize> {
    let log_n = env::var("JOLT_METAL_BOOLEANITY_ADDRESS_LOG_N").ok();
    let elements = env::var("JOLT_SOLINAS_BENCH_ELEMENTS").ok();
    assert!(
        log_n.is_none() || elements.is_none(),
        "set either JOLT_METAL_BOOLEANITY_ADDRESS_LOG_N or JOLT_SOLINAS_BENCH_ELEMENTS"
    );
    let cases = if let Some(log_n) = log_n {
        let log_n = log_n
            .parse::<usize>()
            .expect("Booleanity address log_n should be a positive integer");
        assert!(log_n < usize::BITS as usize, "log_n is too large");
        vec![1usize << log_n]
    } else if let Some(elements) = elements {
        vec![elements
            .parse::<usize>()
            .expect("JOLT_SOLINAS_BENCH_ELEMENTS should be a positive integer")]
    } else {
        DEFAULT_ELEMENTS.to_vec()
    };
    assert!(
        cases
            .iter()
            .all(|elements| elements.is_power_of_two() && elements.ilog2() as usize >= inner_log2),
        "Booleanity address sizes must be powers of two with log_n >= inner_log2"
    );
    cases
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name).map_or(default, |value| {
        value
            .parse()
            .expect("Booleanity address geometry should be a positive integer")
    })
}
