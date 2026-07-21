use std::time::Instant;

use criterion::{black_box, Criterion, Throughput};
use p3_baby_bear::BabyBear;
use p3_field::extension::{BinomialExtensionField, QuinticTrinomialExtensionField};
use p3_field::{
    BasedVectorSpace, ExtensionField, Field, PackedField, PackedFieldExtension, PackedValue,
    PrimeCharacteristicRing,
};
use p3_koala_bear::KoalaBear;
use p3_mersenne_31::Mersenne31;
use rand::{rngs::StdRng, RngCore, SeedableRng};

use super::data::duration_per_logical_op;
use super::params::ArithmeticBenchParams;

fn sample_base<F: PrimeCharacteristicRing + Copy>(rng: &mut StdRng) -> F {
    F::from_u64(rng.next_u64())
}

fn sample_ext<Base: Field, EF: ExtensionField<Base> + BasedVectorSpace<Base>>(
    rng: &mut StdRng,
) -> EF {
    EF::from_basis_coefficients_fn(|_| sample_base::<Base>(rng))
}

pub(crate) fn bench_p3_base_case<F>(
    c: &mut Criterion,
    family: &str,
    label: &str,
    seed: u64,
    params: ArithmeticBenchParams,
) where
    F: Field + Copy,
    F::Packing: PackedField<Scalar = F> + Copy,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let scalar_latency_inputs: Vec<F> = (0..params.latency_iters)
        .map(|_| sample_base(&mut rng))
        .collect();
    let packed_latency_inputs: Vec<F::Packing> = (0..params.latency_iters)
        .map(|_| F::Packing::from_fn(|_| sample_base(&mut rng)))
        .collect();
    let scalar_stream_lanes: Vec<(F, F)> = (0..params.streams)
        .map(|_| (sample_base(&mut rng), sample_base(&mut rng)))
        .collect();
    let packed_stream_lanes: Vec<(F::Packing, F::Packing)> = (0..params.streams)
        .map(|_| {
            (
                F::Packing::from_fn(|_| sample_base(&mut rng)),
                F::Packing::from_fn(|_| sample_base(&mut rng)),
            )
        })
        .collect();

    let width = <F::Packing as PackedValue>::WIDTH;

    let mut latency_group = c.benchmark_group(format!(
        "field_arith/{family}/latency_chain/{label}_w{width}"
    ));

    p3_bench_scalar_suite_latency(&mut latency_group, params, &scalar_latency_inputs);

    let packed_zero = F::Packing::broadcast(F::ZERO);
    let packed_one = F::Packing::broadcast(F::ONE);

    p3_bench_packed_latency(
        &mut latency_group,
        width,
        "add",
        params.latency_iters,
        &packed_latency_inputs,
        |acc, x| acc + x,
        packed_zero,
    );
    p3_bench_packed_latency(
        &mut latency_group,
        width,
        "sub",
        params.latency_iters,
        &packed_latency_inputs,
        |acc, x| acc - x,
        packed_zero,
    );
    p3_bench_packed_unary_latency(
        &mut latency_group,
        width,
        "neg",
        params.latency_iters,
        &packed_latency_inputs,
        |acc| packed_zero - acc,
    );
    p3_bench_packed_unary_latency(
        &mut latency_group,
        width,
        "double",
        params.latency_iters,
        &packed_latency_inputs,
        |acc| acc + acc,
    );
    p3_bench_packed_latency(
        &mut latency_group,
        width,
        "add_neg",
        params.latency_iters,
        &packed_latency_inputs,
        |acc, x| packed_zero - (acc + x),
        packed_zero,
    );
    p3_bench_packed_latency(
        &mut latency_group,
        width,
        "double_add",
        params.latency_iters,
        &packed_latency_inputs,
        |acc, x| acc + acc + x,
        packed_zero,
    );
    p3_bench_packed_latency(
        &mut latency_group,
        width,
        "mul",
        params.latency_iters,
        &packed_latency_inputs,
        |acc, x| acc * x,
        packed_one,
    );
    p3_bench_packed_latency(
        &mut latency_group,
        width,
        "mul_add",
        params.latency_iters,
        &packed_latency_inputs,
        |acc, x| acc * x + acc,
        packed_one,
    );
    p3_bench_packed_unary_latency(
        &mut latency_group,
        width,
        "square",
        params.latency_iters,
        &packed_latency_inputs,
        |acc| acc.square(),
    );
    p3_bench_packed_unary_latency(
        &mut latency_group,
        width,
        "mul_self",
        params.latency_iters,
        &packed_latency_inputs,
        |acc| acc * acc,
    );

    latency_group.throughput(Throughput::Elements(1));
    latency_group.bench_function(
        format!(
            "packed_inverse_chain/{}x{width}_ns_lane",
            params.inverse_latency_iters
        ),
        |b| {
            b.iter_custom(|iters| {
                let inputs = black_box(&packed_latency_inputs[..params.inverse_latency_iters]);
                let mut acc = packed_one;
                let start = Instant::now();
                for _ in 0..iters {
                    for x in inputs {
                        acc = F::Packing::from_fn(|lane| {
                            (PackedValue::extract(&acc, lane) + PackedValue::extract(x, lane))
                                .inverse()
                        });
                    }
                }
                black_box(PackedValue::extract(&acc, 0));
                duration_per_logical_op(
                    start.elapsed(),
                    (params.inverse_latency_iters * width) as u64,
                )
            })
        },
    );

    latency_group.finish();

    let mut throughput_group = c.benchmark_group(format!(
        "field_arith/{family}/throughput_stream/{label}_w{width}"
    ));

    p3_bench_scalar_suite_throughput(&mut throughput_group, params, &scalar_stream_lanes);

    p3_bench_packed_throughput(
        &mut throughput_group,
        width,
        "add",
        params,
        &packed_stream_lanes,
        |acc, x| acc + x,
        |a, b| a + b,
    );
    p3_bench_packed_throughput(
        &mut throughput_group,
        width,
        "sub",
        params,
        &packed_stream_lanes,
        |acc, x| acc - x,
        |a, b| a - b,
    );
    p3_bench_packed_throughput(
        &mut throughput_group,
        width,
        "mul",
        params,
        &packed_stream_lanes,
        |acc, x| acc * x,
        |a, b| a * b,
    );
    p3_bench_packed_throughput(
        &mut throughput_group,
        width,
        "square",
        params,
        &packed_stream_lanes,
        |acc, _| acc.square(),
        |a, _| a.square(),
    );

    throughput_group.throughput(Throughput::Elements(1));
    throughput_group.bench_function(
        format!(
            "packed_inverse_stream/{}x{width}x{}_ns_lane",
            params.streams, params.inverse_throughput_iters
        ),
        |b| {
            b.iter_custom(|iters| {
                let lanes = black_box(&packed_stream_lanes);
                let mut acc: Vec<F::Packing> = lanes.iter().map(|(a, _)| *a).collect();
                let start = Instant::now();
                for _ in 0..iters {
                    for _ in 0..params.inverse_throughput_iters {
                        for (acc_i, lane) in acc.iter_mut().zip(lanes.iter()) {
                            let next = F::Packing::from_fn(|i| {
                                (PackedValue::extract(acc_i, i) + PackedValue::extract(&lane.0, i))
                                    .inverse()
                            });
                            *acc_i = next;
                        }
                    }
                }
                black_box(PackedValue::extract(&acc[0], 0));
                duration_per_logical_op(
                    start.elapsed(),
                    (params.streams * width * params.inverse_throughput_iters) as u64,
                )
            })
        },
    );

    throughput_group.finish();
}

pub(crate) fn bench_p3_ext_case<Base, EF>(
    c: &mut Criterion,
    family: &str,
    label: &str,
    seed: u64,
    params: ArithmeticBenchParams,
) where
    Base: Field + Copy,
    Base::Packing: PackedField<Scalar = Base> + Copy,
    EF: ExtensionField<Base> + BasedVectorSpace<Base> + Copy,
    EF::ExtensionPacking: PackedFieldExtension<Base, EF> + Copy,
{
    let width = <Base::Packing as PackedValue>::WIDTH;

    let mut rng = StdRng::seed_from_u64(seed);
    let scalar_latency_inputs: Vec<EF> = (0..params.latency_iters)
        .map(|_| sample_ext::<Base, EF>(&mut rng))
        .collect();
    let packed_latency_inputs: Vec<EF::ExtensionPacking> = (0..params.latency_iters)
        .map(|_| {
            let ext_vals: Vec<EF> = (0..width)
                .map(|_| sample_ext::<Base, EF>(&mut rng))
                .collect();
            EF::ExtensionPacking::from_ext_slice(&ext_vals)
        })
        .collect();
    let scalar_stream_lanes: Vec<(EF, EF)> = (0..params.streams)
        .map(|_| (sample_ext(&mut rng), sample_ext(&mut rng)))
        .collect();
    let packed_stream_lanes: Vec<(EF::ExtensionPacking, EF::ExtensionPacking)> = (0..params
        .streams)
        .map(|_| {
            let a: Vec<EF> = (0..width)
                .map(|_| sample_ext::<Base, EF>(&mut rng))
                .collect();
            let b: Vec<EF> = (0..width)
                .map(|_| sample_ext::<Base, EF>(&mut rng))
                .collect();
            (
                EF::ExtensionPacking::from_ext_slice(&a),
                EF::ExtensionPacking::from_ext_slice(&b),
            )
        })
        .collect();

    let mut latency_group = c.benchmark_group(format!(
        "field_arith/{family}/latency_chain/{label}_w{width}"
    ));

    p3_bench_scalar_suite_latency(&mut latency_group, params, &scalar_latency_inputs);

    let packed_zero = broadcast_ext::<Base, EF>(EF::ZERO, width);
    let packed_one = broadcast_ext::<Base, EF>(EF::ONE, width);

    p3_bench_packed_ext_latency(
        &mut latency_group,
        width,
        "add",
        params.latency_iters,
        &packed_latency_inputs,
        |acc, x| acc + x,
        packed_zero,
    );
    p3_bench_packed_ext_latency(
        &mut latency_group,
        width,
        "sub",
        params.latency_iters,
        &packed_latency_inputs,
        |acc, x| acc - x,
        packed_zero,
    );
    p3_bench_packed_ext_unary_latency(
        &mut latency_group,
        width,
        "neg",
        params.latency_iters,
        &packed_latency_inputs,
        |acc| packed_zero - acc,
    );
    p3_bench_packed_ext_unary_latency(
        &mut latency_group,
        width,
        "double",
        params.latency_iters,
        &packed_latency_inputs,
        |acc| acc + acc,
    );
    p3_bench_packed_ext_latency(
        &mut latency_group,
        width,
        "add_neg",
        params.latency_iters,
        &packed_latency_inputs,
        |acc, x| packed_zero - (acc + x),
        packed_zero,
    );
    p3_bench_packed_ext_latency(
        &mut latency_group,
        width,
        "double_add",
        params.latency_iters,
        &packed_latency_inputs,
        |acc, x| acc + acc + x,
        packed_zero,
    );
    p3_bench_packed_ext_latency(
        &mut latency_group,
        width,
        "mul",
        params.latency_iters,
        &packed_latency_inputs,
        |acc, x| acc * x,
        packed_one,
    );
    p3_bench_packed_ext_latency(
        &mut latency_group,
        width,
        "mul_add",
        params.latency_iters,
        &packed_latency_inputs,
        |acc, x| acc * x + acc,
        packed_one,
    );
    p3_bench_packed_ext_unary_latency(
        &mut latency_group,
        width,
        "square",
        params.latency_iters,
        &packed_latency_inputs,
        |acc| acc.square(),
    );
    p3_bench_packed_ext_unary_latency(
        &mut latency_group,
        width,
        "mul_self",
        params.latency_iters,
        &packed_latency_inputs,
        |acc| acc * acc,
    );

    latency_group.finish();

    let mut throughput_group = c.benchmark_group(format!(
        "field_arith/{family}/throughput_stream/{label}_w{width}"
    ));

    p3_bench_scalar_suite_throughput(&mut throughput_group, params, &scalar_stream_lanes);

    p3_bench_packed_ext_throughput(
        &mut throughput_group,
        width,
        "add",
        params,
        &packed_stream_lanes,
        |acc, x| acc + x,
        |a, b| a + b,
    );
    p3_bench_packed_ext_throughput(
        &mut throughput_group,
        width,
        "sub",
        params,
        &packed_stream_lanes,
        |acc, x| acc - x,
        |a, b| a - b,
    );
    p3_bench_packed_ext_throughput(
        &mut throughput_group,
        width,
        "mul",
        params,
        &packed_stream_lanes,
        |acc, x| acc * x,
        |a, b| a * b,
    );
    p3_bench_packed_ext_throughput(
        &mut throughput_group,
        width,
        "square",
        params,
        &packed_stream_lanes,
        |acc, _| acc.square(),
        |a, _| a.square(),
    );

    throughput_group.finish();
}

fn broadcast_ext<Base: Field, EF: ExtensionField<Base> + BasedVectorSpace<Base>>(
    value: EF,
    width: usize,
) -> EF::ExtensionPacking
where
    EF::ExtensionPacking: PackedFieldExtension<Base, EF>,
{
    EF::ExtensionPacking::from_ext_slice(&(0..width).map(|_| value).collect::<Vec<_>>())
}

pub(crate) fn bench_p3_base_matrix(c: &mut Criterion) {
    let params = ArithmeticBenchParams::from_env("AKITA_BENCH_BASE_ARITH", 2048, 256);

    bench_p3_base_case::<Mersenne31>(c, "base", "p3_mersenne31", 0xba5e_3131_0003, params);
    bench_p3_base_case::<BabyBear>(c, "base", "p3_baby_bear", 0xba5e_babe_0003, params);
    bench_p3_base_case::<KoalaBear>(c, "base", "p3_koala_bear", 0xba5e_c0a1_a003, params);
}

pub(crate) fn bench_p3_ext4_matrix(c: &mut Criterion) {
    let params = ArithmeticBenchParams::from_env("AKITA_BENCH_EXT4_ARITH", 512, 128);

    bench_p3_ext_case::<BabyBear, BinomialExtensionField<BabyBear, 4>>(
        c,
        "ext4",
        "p3_baby_bear_ext4",
        0xe400_babe_0004,
        params,
    );
    bench_p3_ext_case::<KoalaBear, BinomialExtensionField<KoalaBear, 4>>(
        c,
        "ext4",
        "p3_koala_bear_ext4",
        0xe400_c0a1_a004,
        params,
    );
}

pub(crate) fn bench_p3_ext5_matrix(c: &mut Criterion) {
    let params = ArithmeticBenchParams::from_env("AKITA_BENCH_EXT5_ARITH", 512, 128);

    bench_p3_ext_case::<BabyBear, BinomialExtensionField<BabyBear, 5>>(
        c,
        "ext5",
        "p3_baby_bear_ext5",
        0xe500_babe_0005,
        params,
    );
    bench_p3_ext_case::<KoalaBear, QuinticTrinomialExtensionField<KoalaBear>>(
        c,
        "ext5",
        "p3_koala_bear_ext5",
        0xe500_c0a1_a005,
        params,
    );
}

/// Full scalar latency-chain op set, shared by the base and extension matrices
/// (both operate on a `Field`, only the concrete type differs).
fn p3_bench_scalar_suite_latency<S: Field + Copy>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    params: ArithmeticBenchParams,
    inputs: &[S],
) {
    p3_bench_scalar_latency(
        group,
        "add",
        params.latency_iters,
        inputs,
        |acc, x| acc + x,
        S::ZERO,
    );
    p3_bench_scalar_latency(
        group,
        "sub",
        params.latency_iters,
        inputs,
        |acc, x| acc - x,
        S::ZERO,
    );
    p3_bench_scalar_unary_latency(group, "neg", params.latency_iters, inputs, |acc| -acc);
    p3_bench_scalar_unary_latency(group, "double", params.latency_iters, inputs, |acc| {
        acc.double()
    });
    p3_bench_scalar_latency(
        group,
        "add_neg",
        params.latency_iters,
        inputs,
        |acc, x| -(acc + x),
        S::ZERO,
    );
    p3_bench_scalar_latency(
        group,
        "double_add",
        params.latency_iters,
        inputs,
        |acc, x| acc + acc + x,
        S::ZERO,
    );
    p3_bench_scalar_latency(
        group,
        "mul",
        params.latency_iters,
        inputs,
        |acc, x| acc * x,
        S::ONE,
    );
    p3_bench_scalar_latency(
        group,
        "mul_add",
        params.latency_iters,
        inputs,
        |acc, x| acc * x + acc,
        S::ONE,
    );

    group.throughput(Throughput::Elements(1));
    group.bench_function(
        format!("scalar_square_chain/{}_ns_per_op", params.latency_iters),
        |b| {
            b.iter_custom(|iters| {
                let mut acc = black_box(inputs[0]);
                let start = Instant::now();
                for _ in 0..iters {
                    for _ in 0..params.latency_iters {
                        acc = acc.square();
                    }
                }
                black_box(acc);
                duration_per_logical_op(start.elapsed(), params.latency_iters as u64)
            })
        },
    );

    group.throughput(Throughput::Elements(1));
    group.bench_function(
        format!("scalar_mul_self_chain/{}_ns_per_op", params.latency_iters),
        |b| {
            b.iter_custom(|iters| {
                let mut acc = black_box(inputs[0]);
                let start = Instant::now();
                for _ in 0..iters {
                    for _ in 0..params.latency_iters {
                        acc = acc * acc;
                    }
                }
                black_box(acc);
                duration_per_logical_op(start.elapsed(), params.latency_iters as u64)
            })
        },
    );

    group.throughput(Throughput::Elements(1));
    group.bench_function(
        format!(
            "scalar_inverse_chain/{}_ns_per_op",
            params.inverse_latency_iters
        ),
        |b| {
            b.iter_custom(|iters| {
                let inputs = black_box(&inputs[..params.inverse_latency_iters]);
                let mut acc = S::ONE;
                let start = Instant::now();
                for _ in 0..iters {
                    for x in inputs {
                        acc = (acc + *x).inverse();
                    }
                }
                black_box(acc);
                duration_per_logical_op(start.elapsed(), params.inverse_latency_iters as u64)
            })
        },
    );
}

/// Full scalar throughput-stream op set, shared by the base and extension matrices.
fn p3_bench_scalar_suite_throughput<S: Field + Copy>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    params: ArithmeticBenchParams,
    lanes: &[(S, S)],
) {
    p3_bench_scalar_throughput(group, "add", params, lanes, |acc, x| acc + x, |a, b| a + b);
    p3_bench_scalar_throughput(group, "sub", params, lanes, |acc, x| acc - x, |a, b| a - b);
    p3_bench_scalar_throughput(group, "mul", params, lanes, |acc, x| acc * x, |a, b| a * b);
    p3_bench_scalar_throughput(
        group,
        "square",
        params,
        lanes,
        |acc, _| acc.square(),
        |a, _| a.square(),
    );

    group.throughput(Throughput::Elements(1));
    group.bench_function(
        format!(
            "scalar_inverse_stream/{}x{}_ns_per_op",
            params.streams, params.inverse_throughput_iters
        ),
        |b| {
            b.iter_custom(|iters| {
                let lanes = black_box(lanes);
                let mut acc: Vec<S> = lanes.iter().map(|(a, _)| *a).collect();
                let start = Instant::now();
                for _ in 0..iters {
                    for _ in 0..params.inverse_throughput_iters {
                        for (acc_i, lane) in acc.iter_mut().zip(lanes.iter()) {
                            *acc_i = (*acc_i + lane.0).inverse();
                        }
                    }
                }
                black_box(acc[0]);
                duration_per_logical_op(
                    start.elapsed(),
                    (params.streams * params.inverse_throughput_iters) as u64,
                )
            })
        },
    );
}

fn p3_bench_scalar_latency<F: Copy>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    op: &str,
    latency_iters: usize,
    inputs: &[F],
    step: impl Fn(F, F) -> F,
    init: F,
) {
    group.throughput(Throughput::Elements(1));
    group.bench_function(
        format!("scalar_{op}_chain/{latency_iters}_ns_per_op"),
        |b| {
            b.iter_custom(|iters| {
                let inputs = black_box(inputs);
                let mut acc = init;
                let start = Instant::now();
                for _ in 0..iters {
                    for x in inputs {
                        acc = step(acc, *x);
                    }
                }
                black_box(acc);
                duration_per_logical_op(start.elapsed(), latency_iters as u64)
            })
        },
    );
}

fn p3_bench_scalar_unary_latency<F: Copy>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    op: &str,
    latency_iters: usize,
    inputs: &[F],
    step: impl Fn(F) -> F,
) {
    group.throughput(Throughput::Elements(1));
    group.bench_function(
        format!("scalar_{op}_chain/{latency_iters}_ns_per_op"),
        |b| {
            b.iter_custom(|iters| {
                let mut acc = black_box(inputs[0]);
                let start = Instant::now();
                for _ in 0..iters {
                    for _ in 0..latency_iters {
                        acc = step(acc);
                    }
                }
                black_box(acc);
                duration_per_logical_op(start.elapsed(), latency_iters as u64)
            })
        },
    );
}

fn p3_bench_packed_latency<PF: PackedField + PackedValue + Copy>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    width: usize,
    op: &str,
    latency_iters: usize,
    inputs: &[PF],
    step: impl Fn(PF, PF) -> PF,
    init: PF,
) {
    group.throughput(Throughput::Elements(1));
    group.bench_function(
        format!("packed_{op}_chain/{latency_iters}x{width}_ns_lane"),
        |b| {
            b.iter_custom(|iters| {
                let inputs = black_box(inputs);
                let mut acc = init;
                let start = Instant::now();
                for _ in 0..iters {
                    for x in inputs {
                        acc = step(acc, *x);
                    }
                }
                black_box(<PF as PackedValue>::extract(&acc, 0));
                duration_per_logical_op(start.elapsed(), (latency_iters * width) as u64)
            })
        },
    );
}

fn p3_bench_packed_unary_latency<PF: PackedField + PackedValue + Copy>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    width: usize,
    op: &str,
    latency_iters: usize,
    inputs: &[PF],
    step: impl Fn(PF) -> PF,
) {
    group.throughput(Throughput::Elements(1));
    group.bench_function(
        format!("packed_{op}_chain/{latency_iters}x{width}_ns_lane"),
        |b| {
            b.iter_custom(|iters| {
                let mut acc = black_box(inputs[0]);
                let start = Instant::now();
                for _ in 0..iters {
                    for _ in 0..latency_iters {
                        acc = step(acc);
                    }
                }
                black_box(<PF as PackedValue>::extract(&acc, 0));
                duration_per_logical_op(start.elapsed(), (latency_iters * width) as u64)
            })
        },
    );
}

fn p3_bench_scalar_throughput<F: Copy>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    op: &str,
    params: ArithmeticBenchParams,
    lanes: &[(F, F)],
    step: impl Fn(F, F) -> F,
    init: impl Fn(F, F) -> F,
) {
    group.throughput(Throughput::Elements(1));
    group.bench_function(
        format!(
            "scalar_{op}_stream/{}x{}_ns_per_op",
            params.streams, params.throughput_iters
        ),
        |b| {
            b.iter_custom(|iters| {
                let lanes = black_box(lanes);
                let mut acc: Vec<F> = lanes.iter().map(|(a, b)| init(*a, *b)).collect();
                let start = Instant::now();
                for _ in 0..iters {
                    for _ in 0..params.throughput_iters {
                        for (acc_i, lane) in acc.iter_mut().zip(lanes.iter()) {
                            *acc_i = step(*acc_i, lane.0);
                        }
                    }
                }
                black_box(acc[0]);
                duration_per_logical_op(
                    start.elapsed(),
                    (params.streams * params.throughput_iters) as u64,
                )
            })
        },
    );
}

fn p3_bench_packed_throughput<PF: PackedField + PackedValue + Copy>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    width: usize,
    op: &str,
    params: ArithmeticBenchParams,
    lanes: &[(PF, PF)],
    step: impl Fn(PF, PF) -> PF,
    init: impl Fn(PF, PF) -> PF,
) {
    group.throughput(Throughput::Elements(1));
    group.bench_function(
        format!(
            "packed_{op}_stream/{}x{width}x{}_ns_lane",
            params.streams, params.throughput_iters
        ),
        |b| {
            b.iter_custom(|iters| {
                let lanes = black_box(lanes);
                let mut acc: Vec<PF> = lanes.iter().map(|(a, b)| init(*a, *b)).collect();
                let start = Instant::now();
                for _ in 0..iters {
                    for _ in 0..params.throughput_iters {
                        for (acc_i, lane) in acc.iter_mut().zip(lanes.iter()) {
                            *acc_i = step(*acc_i, lane.0);
                        }
                    }
                }
                black_box(<PF as PackedValue>::extract(&acc[0], 0));
                duration_per_logical_op(
                    start.elapsed(),
                    (params.streams * width * params.throughput_iters) as u64,
                )
            })
        },
    );
}

fn p3_bench_packed_ext_latency<Base, EF, EP>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    width: usize,
    op: &str,
    latency_iters: usize,
    inputs: &[EP],
    step: impl Fn(EP, EP) -> EP,
    init: EP,
) where
    Base: Field,
    EF: ExtensionField<Base, ExtensionPacking = EP>,
    EP: PackedFieldExtension<Base, EF> + Copy,
{
    group.throughput(Throughput::Elements(1));
    group.bench_function(
        format!("packed_{op}_chain/{latency_iters}x{width}_ns_lane"),
        |b| {
            b.iter_custom(|iters| {
                let inputs = black_box(inputs);
                let mut acc = init;
                let start = Instant::now();
                for _ in 0..iters {
                    for x in inputs {
                        acc = step(acc, *x);
                    }
                }
                black_box(PackedFieldExtension::extract(&acc, 0));
                duration_per_logical_op(start.elapsed(), (latency_iters * width) as u64)
            })
        },
    );
}

fn p3_bench_packed_ext_unary_latency<Base, EF, EP>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    width: usize,
    op: &str,
    latency_iters: usize,
    inputs: &[EP],
    step: impl Fn(EP) -> EP,
) where
    Base: Field,
    EF: ExtensionField<Base, ExtensionPacking = EP>,
    EP: PackedFieldExtension<Base, EF> + Copy,
{
    group.throughput(Throughput::Elements(1));
    group.bench_function(
        format!("packed_{op}_chain/{latency_iters}x{width}_ns_lane"),
        |b| {
            b.iter_custom(|iters| {
                let mut acc = black_box(inputs[0]);
                let start = Instant::now();
                for _ in 0..iters {
                    for _ in 0..latency_iters {
                        acc = step(acc);
                    }
                }
                black_box(PackedFieldExtension::extract(&acc, 0));
                duration_per_logical_op(start.elapsed(), (latency_iters * width) as u64)
            })
        },
    );
}

fn p3_bench_packed_ext_throughput<Base, EF, EP>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    width: usize,
    op: &str,
    params: ArithmeticBenchParams,
    lanes: &[(EP, EP)],
    step: impl Fn(EP, EP) -> EP,
    init: impl Fn(EP, EP) -> EP,
) where
    Base: Field,
    EF: ExtensionField<Base, ExtensionPacking = EP>,
    EP: PackedFieldExtension<Base, EF> + Copy,
{
    group.throughput(Throughput::Elements(1));
    group.bench_function(
        format!(
            "packed_{op}_stream/{}x{width}x{}_ns_lane",
            params.streams, params.throughput_iters
        ),
        |b| {
            b.iter_custom(|iters| {
                let lanes = black_box(lanes);
                let mut acc: Vec<EP> = lanes.iter().map(|(a, b)| init(*a, *b)).collect();
                let start = Instant::now();
                for _ in 0..iters {
                    for _ in 0..params.throughput_iters {
                        for (acc_i, lane) in acc.iter_mut().zip(lanes.iter()) {
                            *acc_i = step(*acc_i, lane.0);
                        }
                    }
                }
                black_box(acc[0].extract(0));
                duration_per_logical_op(
                    start.elapsed(),
                    (params.streams * width * params.throughput_iters) as u64,
                )
            })
        },
    );
}
