use std::{env, hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput};
use jolt_field::AkitaField;
use jolt_kernels::metal::solinas::{
    evaluate_spartan_outer_uniskip_cpu, SolinasMetal, SpartanOuterUniskipConfig,
    SpartanOuterUniskipRow,
};
use jolt_poly::EqPolynomial;
use rayon::prelude::*;

const DEFAULT_ELEMENTS: [usize; 3] = [1 << 16, 1 << 20, 1 << 22];

pub fn bench(c: &mut Criterion, context: &SolinasMetal) {
    let mut group = c.benchmark_group("metal_sumcheck/spartan_outer_uniskip");
    let _ = group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5));

    for elements in cases() {
        let rows = (0..elements)
            .into_par_iter()
            .map(|index| synthetic_row(index, 1))
            .collect::<Vec<_>>();
        let point = values(elements.ilog2() as usize + 1, 0x6a09_e667_f3bc_c909);
        let split = point.len() / 2;
        let e_out = EqPolynomial::<AkitaField>::evals(&point[..split], None);
        let e_in = EqPolynomial::<AkitaField>::evals(&point[split..], None);
        let resident = context
            .prepare_spartan_outer_uniskip_rows(&rows)
            .expect("Spartan outer rows should prepare");
        let invocation = context
            .prepare_spartan_outer_uniskip_with_rows(
                &resident,
                &e_in,
                &e_out,
                SpartanOuterUniskipConfig::default(),
            )
            .expect("Spartan outer invocation should prepare");
        let expected = evaluate_spartan_outer_uniskip_cpu(&rows, &e_in, &e_out)
            .expect("Spartan outer CPU oracle should evaluate");
        invocation
            .execute()
            .expect("Spartan outer invocation should execute");
        assert_eq!(
            invocation
                .read_output()
                .expect("Spartan outer output should be readable"),
            expected
        );

        let _ = group.throughput(Throughput::Elements(18 * elements as u64));
        let suffix = format!("n{elements}_tg{}", invocation.threads_per_threadgroup());
        let cpu_first = env::var("JOLT_SOLINAS_BENCH_ORDER").as_deref() == Ok("cpu-first");
        let add_cpu =
            |group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>| {
                let _ = group.bench_function(BenchmarkId::new("cpu_optimized", &suffix), |bench| {
                    bench.iter(|| {
                        black_box(
                            evaluate_spartan_outer_uniskip_cpu(&rows, &e_in, &e_out)
                                .expect("Spartan outer CPU oracle should evaluate"),
                        )
                    });
                });
            };
        let add_gpu = |group: &mut criterion::BenchmarkGroup<
            '_,
            criterion::measurement::WallTime,
        >| {
            let _ =
                group.bench_function(BenchmarkId::new("metal_wall_resident", &suffix), |bench| {
                    bench.iter(|| {
                        invocation
                            .execute()
                            .expect("Spartan outer invocation should execute");
                        black_box(
                            invocation
                                .read_output()
                                .expect("Spartan outer output should be readable"),
                        )
                    });
                });
            let _ = group.bench_function(
                BenchmarkId::new("metal_active_resident", &suffix),
                |bench| {
                    bench.iter_custom(|iterations| {
                        let mut active = Duration::ZERO;
                        for _ in 0..iterations {
                            active += invocation
                                .execute_timed()
                                .expect("timed Spartan outer invocation should execute");
                        }
                        active
                    });
                },
            );
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
        cases
            .iter()
            .all(|elements| elements.is_power_of_two() && *elements >= 2),
        "Spartan outer benchmark sizes must be powers of two"
    );
    cases
}

fn values(count: usize, seed: u64) -> Vec<AkitaField> {
    (0..count)
        .map(|index| AkitaField::from_u64(splitmix(seed ^ index as u64) & ((1u64 << 56) - 1)))
        .collect()
}

fn synthetic_row(index: usize, seed: u64) -> SpartanOuterUniskipRow {
    let mut words = [0u64; 20];
    for (word, value) in words[..19].iter_mut().enumerate() {
        *value = splitmix(seed ^ index as u64 ^ (word as u64).wrapping_mul(0x1000_0001));
    }
    words[2] &= (1 << 24) - 1;
    words[4] &= (1 << 24) - 1;
    words[8] = 0;
    words[15] &= (1 << 24) - 1;
    let selector = splitmix(seed ^ index as u64 ^ 0xa5a5_5a5a);
    let mut flags = 0u64;
    match selector % 3 {
        1 => flags |= 1 << 0,
        2 => flags |= 1 << 1,
        _ => {}
    }
    match (selector >> 2) % 4 {
        1 => flags |= 1 << 2,
        2 => flags |= 1 << 3,
        3 => flags |= 1 << 4,
        _ => {}
    }
    for bit in 5..=16 {
        flags |= ((selector >> (bit + 7)) & 1) << bit;
    }
    flags |= ((selector >> 40) & 1) << 17;
    flags |= ((selector >> 41) & 1) << 18;
    flags |= ((selector >> 42) & 1) << 19;
    words[19] = flags;
    SpartanOuterUniskipRow::from_words(words)
}

fn splitmix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
