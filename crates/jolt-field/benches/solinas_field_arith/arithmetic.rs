use std::ops::{AddAssign, MulAssign, SubAssign};
use std::time::Instant;

use criterion::{black_box, Criterion, Throughput};
use jolt_field::packed::PackedField;
use jolt_field::{FieldCore, RingCore};
use rand::{rngs::StdRng, SeedableRng};

use super::data::duration_per_logical_op;
use super::params::ArithmeticBenchParams;

pub(crate) fn bench_arithmetic_case<F, PF>(
    c: &mut Criterion,
    family: &str,
    label: &str,
    seed: u64,
    params: ArithmeticBenchParams,
) where
    F: FieldCore + FieldCore + RingCore + FieldCore + AddAssign + SubAssign + MulAssign + 'static,
    PF: PackedField<Scalar = F> + Copy + 'static,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let scalar_latency_inputs: Vec<F> = (0..params.latency_iters)
        .map(|_| F::random(&mut rng))
        .collect();
    let packed_latency_inputs: Vec<PF> = (0..params.latency_iters)
        .map(|_| PF::from_fn(|_| F::random(&mut rng)))
        .collect();
    let scalar_stream_lanes: Vec<(F, F)> = (0..params.streams)
        .map(|_| (F::random(&mut rng), F::random(&mut rng)))
        .collect();
    let packed_stream_lanes: Vec<(PF, PF)> = (0..params.streams)
        .map(|_| {
            (
                PF::from_fn(|_| F::random(&mut rng)),
                PF::from_fn(|_| F::random(&mut rng)),
            )
        })
        .collect();

    let mut latency_group = c.benchmark_group(format!(
        "field_arith/{family}/latency_chain/{label}_w{}",
        PF::WIDTH
    ));

    bench_scalar_latency::<F>(
        &mut latency_group,
        "add",
        params.latency_iters,
        &scalar_latency_inputs,
        |mut acc, x| {
            acc += x;
            acc
        },
        F::zero(),
    );
    bench_scalar_latency::<F>(
        &mut latency_group,
        "sub",
        params.latency_iters,
        &scalar_latency_inputs,
        |mut acc, x| {
            acc -= x;
            acc
        },
        F::zero(),
    );
    bench_scalar_unary_latency::<F>(
        &mut latency_group,
        "neg",
        params.latency_iters,
        &scalar_latency_inputs,
        |acc| -acc,
    );
    bench_scalar_unary_latency::<F>(
        &mut latency_group,
        "double",
        params.latency_iters,
        &scalar_latency_inputs,
        |acc| acc + acc,
    );
    bench_scalar_latency::<F>(
        &mut latency_group,
        "add_neg",
        params.latency_iters,
        &scalar_latency_inputs,
        |acc, x| -(acc + x),
        F::zero(),
    );
    bench_scalar_latency::<F>(
        &mut latency_group,
        "double_add",
        params.latency_iters,
        &scalar_latency_inputs,
        |acc, x| acc + acc + x,
        F::zero(),
    );
    bench_scalar_latency::<F>(
        &mut latency_group,
        "mul",
        params.latency_iters,
        &scalar_latency_inputs,
        |mut acc, x| {
            acc *= x;
            acc
        },
        F::one(),
    );
    bench_scalar_latency::<F>(
        &mut latency_group,
        "mul_add",
        params.latency_iters,
        &scalar_latency_inputs,
        |acc, x| acc * x + acc,
        F::one(),
    );

    latency_group.throughput(Throughput::Elements(1));
    latency_group.bench_function(
        format!("scalar_square_chain/{}_ns_per_op", params.latency_iters),
        |b| {
            b.iter_custom(|iters| {
                let mut acc = black_box(scalar_latency_inputs[0]);
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

    latency_group.throughput(Throughput::Elements(1));
    latency_group.bench_function(
        format!("scalar_mul_self_chain/{}_ns_per_op", params.latency_iters),
        |b| {
            b.iter_custom(|iters| {
                let mut acc = black_box(scalar_latency_inputs[0]);
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

    latency_group.throughput(Throughput::Elements(1));
    latency_group.bench_function(
        format!(
            "scalar_inverse_chain/{}_ns_per_op",
            params.inverse_latency_iters
        ),
        |b| {
            b.iter_custom(|iters| {
                let inputs = black_box(&scalar_latency_inputs[..params.inverse_latency_iters]);
                let mut acc = F::one();
                let start = Instant::now();
                for _ in 0..iters {
                    for x in inputs {
                        acc = (acc + *x).inverse().unwrap_or_else(F::one);
                    }
                }
                black_box(acc);
                duration_per_logical_op(start.elapsed(), params.inverse_latency_iters as u64)
            })
        },
    );

    bench_packed_latency::<F, PF>(
        &mut latency_group,
        "add",
        params.latency_iters,
        &packed_latency_inputs,
        |acc, x| acc + x,
        PF::broadcast(F::zero()),
    );
    bench_packed_latency::<F, PF>(
        &mut latency_group,
        "sub",
        params.latency_iters,
        &packed_latency_inputs,
        |acc, x| acc - x,
        PF::broadcast(F::zero()),
    );
    let packed_zero = PF::broadcast(F::zero());
    bench_packed_unary_latency::<F, PF>(
        &mut latency_group,
        "neg",
        params.latency_iters,
        &packed_latency_inputs,
        |acc| packed_zero - acc,
    );
    bench_packed_unary_latency::<F, PF>(
        &mut latency_group,
        "double",
        params.latency_iters,
        &packed_latency_inputs,
        |acc| acc + acc,
    );
    bench_packed_latency::<F, PF>(
        &mut latency_group,
        "add_neg",
        params.latency_iters,
        &packed_latency_inputs,
        |acc, x| packed_zero - (acc + x),
        packed_zero,
    );
    bench_packed_latency::<F, PF>(
        &mut latency_group,
        "double_add",
        params.latency_iters,
        &packed_latency_inputs,
        |acc, x| acc + acc + x,
        PF::broadcast(F::zero()),
    );
    bench_packed_latency::<F, PF>(
        &mut latency_group,
        "mul",
        params.latency_iters,
        &packed_latency_inputs,
        |acc, x| acc * x,
        PF::broadcast(F::one()),
    );
    bench_packed_latency::<F, PF>(
        &mut latency_group,
        "mul_add",
        params.latency_iters,
        &packed_latency_inputs,
        |acc, x| acc * x + acc,
        PF::broadcast(F::one()),
    );
    bench_packed_unary_latency::<F, PF>(
        &mut latency_group,
        "square",
        params.latency_iters,
        &packed_latency_inputs,
        |acc| acc.square(),
    );
    bench_packed_unary_latency::<F, PF>(
        &mut latency_group,
        "mul_self",
        params.latency_iters,
        &packed_latency_inputs,
        |acc| acc * acc,
    );
    latency_group.throughput(Throughput::Elements(1));
    latency_group.bench_function(
        format!(
            "packed_inverse_chain/{}x{}_ns_lane",
            params.inverse_latency_iters,
            PF::WIDTH
        ),
        |b| {
            b.iter_custom(|iters| {
                let inputs = black_box(&packed_latency_inputs[..params.inverse_latency_iters]);
                let mut acc = PF::broadcast(F::one());
                let start = Instant::now();
                for _ in 0..iters {
                    for x in inputs {
                        acc = (acc + *x)
                            .inverse()
                            .unwrap_or_else(|| PF::broadcast(F::one()));
                    }
                }
                black_box(acc.extract(0));
                duration_per_logical_op(
                    start.elapsed(),
                    (params.inverse_latency_iters * PF::WIDTH) as u64,
                )
            })
        },
    );

    latency_group.finish();

    let mut throughput_group = c.benchmark_group(format!(
        "field_arith/{family}/throughput_stream/{label}_w{}",
        PF::WIDTH
    ));

    bench_scalar_throughput::<F>(
        &mut throughput_group,
        "add",
        params,
        &scalar_stream_lanes,
        |mut acc, x| {
            acc += x;
            acc
        },
        |a, b| a + b,
    );
    bench_scalar_throughput::<F>(
        &mut throughput_group,
        "sub",
        params,
        &scalar_stream_lanes,
        |mut acc, x| {
            acc -= x;
            acc
        },
        |a, b| a - b,
    );
    bench_scalar_throughput::<F>(
        &mut throughput_group,
        "mul",
        params,
        &scalar_stream_lanes,
        |mut acc, x| {
            acc *= x;
            acc
        },
        |a, b| a * b,
    );

    throughput_group.throughput(Throughput::Elements(1));
    throughput_group.bench_function(
        format!(
            "scalar_square_stream/{}x{}_ns_per_op",
            params.streams, params.throughput_iters
        ),
        |b| {
            b.iter_custom(|iters| {
                let lanes = black_box(&scalar_stream_lanes);
                let mut acc: Vec<F> = lanes.iter().map(|(a, _)| *a).collect();
                let start = Instant::now();
                for _ in 0..iters {
                    for _ in 0..params.throughput_iters {
                        for acc_i in acc.iter_mut() {
                            *acc_i = acc_i.square();
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

    throughput_group.throughput(Throughput::Elements(1));
    throughput_group.bench_function(
        format!(
            "scalar_inverse_stream/{}x{}_ns_per_op",
            params.streams, params.inverse_throughput_iters
        ),
        |b| {
            b.iter_custom(|iters| {
                let lanes = black_box(&scalar_stream_lanes);
                let mut acc: Vec<F> = lanes.iter().map(|(a, _)| *a).collect();
                let start = Instant::now();
                for _ in 0..iters {
                    for _ in 0..params.inverse_throughput_iters {
                        for (acc_i, lane) in acc.iter_mut().zip(lanes.iter()) {
                            *acc_i = (*acc_i + lane.0).inverse().unwrap_or_else(F::one);
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

    bench_packed_throughput::<F, PF>(
        &mut throughput_group,
        "add",
        params,
        &packed_stream_lanes,
        |acc, x| acc + x,
        |a, b| a + b,
    );
    bench_packed_throughput::<F, PF>(
        &mut throughput_group,
        "sub",
        params,
        &packed_stream_lanes,
        |acc, x| acc - x,
        |a, b| a - b,
    );
    bench_packed_throughput::<F, PF>(
        &mut throughput_group,
        "mul",
        params,
        &packed_stream_lanes,
        |acc, x| acc * x,
        |a, b| a * b,
    );

    throughput_group.throughput(Throughput::Elements(1));
    throughput_group.bench_function(
        format!(
            "packed_square_stream/{}x{}x{}_ns_lane",
            params.streams,
            PF::WIDTH,
            params.throughput_iters
        ),
        |b| {
            b.iter_custom(|iters| {
                let lanes = black_box(&packed_stream_lanes);
                let mut acc: Vec<PF> = lanes.iter().map(|(a, _)| *a).collect();
                let start = Instant::now();
                for _ in 0..iters {
                    for _ in 0..params.throughput_iters {
                        for acc_i in acc.iter_mut() {
                            *acc_i = acc_i.square();
                        }
                    }
                }
                black_box(acc[0].extract(0));
                duration_per_logical_op(
                    start.elapsed(),
                    (params.streams * PF::WIDTH * params.throughput_iters) as u64,
                )
            })
        },
    );

    throughput_group.throughput(Throughput::Elements(1));
    throughput_group.bench_function(
        format!(
            "packed_mul_self_stream/{}x{}x{}_ns_lane",
            params.streams,
            PF::WIDTH,
            params.throughput_iters
        ),
        |b| {
            b.iter_custom(|iters| {
                let lanes = black_box(&packed_stream_lanes);
                let mut acc: Vec<PF> = lanes.iter().map(|(a, _)| *a).collect();
                let start = Instant::now();
                for _ in 0..iters {
                    for _ in 0..params.throughput_iters {
                        for acc_i in acc.iter_mut() {
                            let x = *acc_i;
                            *acc_i = x * x;
                        }
                    }
                }
                black_box(acc[0].extract(0));
                duration_per_logical_op(
                    start.elapsed(),
                    (params.streams * PF::WIDTH * params.throughput_iters) as u64,
                )
            })
        },
    );

    throughput_group.throughput(Throughput::Elements(1));
    throughput_group.bench_function(
        format!(
            "packed_inverse_stream/{}x{}x{}_ns_lane",
            params.streams,
            PF::WIDTH,
            params.inverse_throughput_iters
        ),
        |b| {
            b.iter_custom(|iters| {
                let lanes = black_box(&packed_stream_lanes);
                let mut acc: Vec<PF> = lanes.iter().map(|(a, _)| *a).collect();
                let start = Instant::now();
                for _ in 0..iters {
                    for _ in 0..params.inverse_throughput_iters {
                        for (acc_i, lane) in acc.iter_mut().zip(lanes.iter()) {
                            *acc_i = (*acc_i + lane.0)
                                .inverse()
                                .unwrap_or_else(|| PF::broadcast(F::one()));
                        }
                    }
                }
                black_box(acc[0].extract(0));
                duration_per_logical_op(
                    start.elapsed(),
                    (params.streams * PF::WIDTH * params.inverse_throughput_iters) as u64,
                )
            })
        },
    );

    throughput_group.finish();
}

fn bench_scalar_latency<F>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    op: &str,
    latency_iters: usize,
    inputs: &[F],
    step: impl Fn(F, F) -> F,
    init: F,
) where
    F: FieldCore,
{
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

fn bench_scalar_unary_latency<F>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    op: &str,
    latency_iters: usize,
    inputs: &[F],
    step: impl Fn(F) -> F,
) where
    F: FieldCore,
{
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

fn bench_packed_latency<F, PF>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    op: &str,
    latency_iters: usize,
    inputs: &[PF],
    step: impl Fn(PF, PF) -> PF,
    init: PF,
) where
    F: FieldCore,
    PF: PackedField<Scalar = F> + Copy,
{
    group.throughput(Throughput::Elements(1));
    group.bench_function(
        format!("packed_{op}_chain/{latency_iters}x{}_ns_lane", PF::WIDTH),
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
                black_box(acc.extract(0));
                duration_per_logical_op(start.elapsed(), (latency_iters * PF::WIDTH) as u64)
            })
        },
    );
}

fn bench_packed_unary_latency<F, PF>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    op: &str,
    latency_iters: usize,
    inputs: &[PF],
    step: impl Fn(PF) -> PF,
) where
    F: FieldCore,
    PF: PackedField<Scalar = F> + Copy,
{
    group.throughput(Throughput::Elements(1));
    group.bench_function(
        format!("packed_{op}_chain/{latency_iters}x{}_ns_lane", PF::WIDTH),
        |b| {
            b.iter_custom(|iters| {
                let mut acc = black_box(inputs[0]);
                let start = Instant::now();
                for _ in 0..iters {
                    for _ in 0..latency_iters {
                        acc = step(acc);
                    }
                }
                black_box(acc.extract(0));
                duration_per_logical_op(start.elapsed(), (latency_iters * PF::WIDTH) as u64)
            })
        },
    );
}

fn bench_scalar_throughput<F>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    op: &str,
    params: ArithmeticBenchParams,
    lanes: &[(F, F)],
    step: impl Fn(F, F) -> F,
    init: impl Fn(F, F) -> F,
) where
    F: FieldCore,
{
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

fn bench_packed_throughput<F, PF>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    op: &str,
    params: ArithmeticBenchParams,
    lanes: &[(PF, PF)],
    step: impl Fn(PF, PF) -> PF,
    init: impl Fn(PF, PF) -> PF,
) where
    F: FieldCore,
    PF: PackedField<Scalar = F> + Copy,
{
    group.throughput(Throughput::Elements(1));
    group.bench_function(
        format!(
            "packed_{op}_stream/{}x{}x{}_ns_lane",
            params.streams,
            PF::WIDTH,
            params.throughput_iters
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
                black_box(acc[0].extract(0));
                duration_per_logical_op(
                    start.elapsed(),
                    (params.streams * PF::WIDTH * params.throughput_iters) as u64,
                )
            })
        },
    );
}
