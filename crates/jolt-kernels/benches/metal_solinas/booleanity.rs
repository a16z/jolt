use std::{env, hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput};
use jolt_field::{AdditiveAccumulator, AkitaAccumulator, AkitaField, RingAccumulator};
use jolt_kernels::metal::solinas::{
    BooleanityRow, BooleanitySelector, BooleanitySequenceConfig, SolinasMetal,
};
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial};

const K: usize = 256;
const CHUNK_BITS: usize = 8;
const DEFAULT_ELEMENTS: [usize; 3] = [1 << 16, 1 << 20, 1 << 22];

pub fn bench_message(c: &mut Criterion, context: &SolinasMetal) {
    let mut group = c.benchmark_group("metal_sumcheck/booleanity_initial_message");
    let _ = group
        .sample_size(12)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5));

    for elements in cases() {
        let selectors = selectors();
        let rows = rows(elements, 1);
        let cycle_point = values(elements.ilog2() as usize, 0x6a09_e667_f3bc_c909);
        let address_point = values(CHUNK_BITS, 0xbb67_ae85_84ca_a73b);
        let (rho, base_tables) = base_tables(selectors.len(), &address_point);
        let gruen = GruenSplitEqPolynomial::new(&cycle_point, BindingOrder::LowToHigh);
        let mut sequence = context
            .prepare_booleanity_sequence(
                &rows,
                &selectors,
                &base_tables,
                &rho,
                K,
                gruen.e_in_current_len(),
                gruen.e_out_current_len(),
                BooleanitySequenceConfig {
                    threads_per_threadgroup: Some(256),
                    dense_threads_per_threadgroup: Some(128),
                    materialize_width: 2,
                },
            )
            .expect("Booleanity sequence should prepare");
        let expected = cpu_message(&rows, &selectors, &base_tables, &rho, &gruen);
        let actual = sequence
            .message(gruen.e_in_current(), gruen.e_out_current())
            .expect("Booleanity message should execute");
        assert_eq!(actual, expected);

        let useful =
            (selectors.len() as u64 + 1) * elements as u64 + 2 * gruen.e_out_current_len() as u64;
        let _ = group.throughput(Throughput::Elements(useful));
        let suffix = format!("n{elements}_p{}_k{K}", selectors.len());
        let cpu_first = env::var("JOLT_SOLINAS_BENCH_ORDER").as_deref() == Ok("cpu-first");
        let add_cpu =
            |group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>| {
                let _ = group.bench_function(BenchmarkId::new("cpu_optimized", &suffix), |bench| {
                    bench.iter(|| {
                        black_box(cpu_message(&rows, &selectors, &base_tables, &rho, &gruen))
                    });
                });
            };
        let mut add_gpu =
            |group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>| {
                let _ = group.bench_function(BenchmarkId::new("metal_wall", &suffix), |bench| {
                    bench.iter(|| {
                        black_box(
                            sequence
                                .message(gruen.e_in_current(), gruen.e_out_current())
                                .expect("Booleanity message should execute"),
                        )
                    });
                });
                let _ = group.bench_function(BenchmarkId::new("metal_active", &suffix), |bench| {
                    bench.iter_custom(|iterations| {
                        let before = sequence.gpu_active_time();
                        for _ in 0..iterations {
                            let _ = black_box(
                                sequence
                                    .message(gruen.e_in_current(), gruen.e_out_current())
                                    .expect("Booleanity message should execute"),
                            );
                        }
                        sequence
                            .gpu_active_time()
                            .checked_sub(before)
                            .expect("GPU active time should be monotonic")
                    });
                });
            };
        if cpu_first {
            add_cpu(&mut group);
            add_gpu(&mut group);
        } else {
            add_gpu(&mut group);
            add_cpu(&mut group);
        }
    }
    group.finish();
}

fn cpu_message(
    rows: &[BooleanityRow],
    selectors: &[BooleanitySelector],
    base_tables: &[AkitaField],
    rho: &[AkitaField],
    gruen: &GruenSplitEqPolynomial<AkitaField>,
) -> [AkitaField; 2] {
    struct Scratch {
        lanes: [AkitaAccumulator; 2],
        pairs: Vec<(AkitaField, AkitaField)>,
    }

    let lanes = gruen.par_fold_out_in(
        || Scratch {
            lanes: [AkitaAccumulator::default(); 2],
            pairs: vec![(AkitaField::zero(), AkitaField::zero()); selectors.len()],
        },
        |scratch, row, _x_in, e_in| {
            for (poly, pair) in scratch.pairs.iter_mut().enumerate() {
                let table = &base_tables[poly * K..(poly + 1) * K];
                pair.0 = hot_index(rows[2 * row], selectors[poly])
                    .map_or_else(AkitaField::zero, |hot| table[hot]);
                pair.1 = hot_index(rows[2 * row + 1], selectors[poly])
                    .map_or_else(AkitaField::zero, |hot| table[hot]);
            }
            let mut constant = AkitaAccumulator::default();
            let mut leading = AkitaAccumulator::default();
            for ((h_0, h_1), rho) in scratch.pairs.iter().zip(rho) {
                let delta = *h_1 - *h_0;
                constant.fmadd(*h_0, *h_0 - *rho);
                leading.fmadd(delta, delta);
            }
            scratch.lanes[0].fmadd(e_in, constant.reduce());
            scratch.lanes[1].fmadd(e_in, leading.reduce());
        },
        |_x_out, e_out, scratch| {
            let mut output = [AkitaAccumulator::default(); 2];
            output[0].fmadd(e_out, scratch.lanes[0].reduce());
            output[1].fmadd(e_out, scratch.lanes[1].reduce());
            output
        },
        |mut lhs, rhs| {
            lhs[0].merge(rhs[0]);
            lhs[1].merge(rhs[1]);
            lhs
        },
    );
    lanes.map(AdditiveAccumulator::reduce)
}

fn selectors() -> Vec<BooleanitySelector> {
    let mut selectors = (0..16)
        .map(|index| BooleanitySelector::Lookup {
            shift: (CHUNK_BITS * index) as u32,
        })
        .collect::<Vec<_>>();
    selectors.push(BooleanitySelector::Bytecode { shift: 0 });
    selectors.extend([0, 8, 56].map(|shift| BooleanitySelector::Ram { shift }));
    selectors.extend((0..8).map(|index| BooleanitySelector::FusedInc {
        shift: (CHUNK_BITS * index) as u32,
    }));
    selectors.push(BooleanitySelector::FusedIncMsb);
    selectors
}

fn rows(count: usize, seed: u64) -> Vec<BooleanityRow> {
    let mut state = u128::from(seed) | (u128::from(!seed) << 64);
    (0..count)
        .map(|row| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 43;
            let mapped_pc = (row % 7 != 0).then_some(((state >> 49) as u64) & ((1 << 55) - 2));
            let ram_address = (row % 11 != 0).then_some((state as u64) & (u64::MAX - 1));
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

fn hot_index(row: BooleanityRow, selector: BooleanitySelector) -> Option<usize> {
    let words = row.words();
    match selector {
        BooleanitySelector::Lookup { shift } => {
            let lookup = u128::from(words[0]) | (u128::from(words[1]) << 64);
            Some(((lookup >> shift) as usize) & (K - 1))
        }
        BooleanitySelector::Bytecode { shift } => {
            let pc_plus_one = words[4] & ((1 << 56) - 1);
            pc_plus_one
                .checked_sub(1)
                .map(|pc| ((pc >> shift) as usize) & (K - 1))
        }
        BooleanitySelector::Ram { shift } => words[2]
            .checked_sub(1)
            .map(|address| ((address >> shift) as usize) & (K - 1)),
        BooleanitySelector::FusedInc { shift } => {
            let standard = ((biased_fused_inc(words) >> shift) as usize) & (K - 1);
            Some((standard + K / 2) & (K - 1))
        }
        BooleanitySelector::FusedIncMsb => {
            let carry = biased_fused_inc(words) >> 64;
            Some(carry.rem_euclid(K as i128) as usize)
        }
    }
}

fn biased_fused_inc(words: [u64; 5]) -> i128 {
    let magnitude = i128::from(words[3]);
    let value = if words[4] >> 63 == 0 {
        magnitude
    } else {
        -magnitude
    };
    let radix = 1i128 << CHUNK_BITS;
    value + (radix / 2) * (((1i128 << 64) - 1) / (radix - 1))
}

fn base_tables(
    selectors: usize,
    address_point: &[AkitaField],
) -> (Vec<AkitaField>, Vec<AkitaField>) {
    let eq_address = EqPolynomial::<AkitaField>::evals(address_point, None);
    let mut rho = Vec::with_capacity(selectors);
    let mut tables = Vec::with_capacity(selectors * K);
    let gamma = AkitaField::from_u64(31);
    let mut power = AkitaField::one();
    for _ in 0..selectors {
        rho.push(power);
        tables.extend(eq_address.iter().map(|value| power * *value));
        power *= gamma;
    }
    (rho, tables)
}

fn values(count: usize, seed: u64) -> Vec<AkitaField> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut value = state;
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            AkitaField::from_u64(value ^ (value >> 31))
        })
        .collect()
}

fn cases() -> Vec<usize> {
    let cases = env::var("JOLT_SOLINAS_BENCH_ELEMENTS").map_or_else(
        |_| DEFAULT_ELEMENTS.to_vec(),
        |value| {
            vec![value
                .parse()
                .expect("JOLT_SOLINAS_BENCH_ELEMENTS should be a positive integer")]
        },
    );
    assert!(
        cases.iter().all(|elements| elements.is_power_of_two()),
        "Booleanity benchmark sizes must be powers of two"
    );
    cases
}
