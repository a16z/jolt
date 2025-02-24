use ark_bn254::Fr;
use ark_std::{test_rng, UniformRand};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use jolt_core::field::JoltField;
use rand::Rng;
use rayon::prelude::*;

fn multiply_u64(a: u64, b: u64) -> u64 {
    a * b
}

fn multiply_u128(a: u128, b: u128) -> u128 {
    a * b
}

// Field multiplication
fn multiply_field(a: Fr, b: Fr) -> Fr {
    a * b
}

fn benchmark_parallel_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_operations");
    let mut rng = rand::thread_rng();
    let mut field_rng = test_rng();

    let sizes = [100_000, 1_000_000];

    for size in sizes {
        // Generate test data
        let arr_a_u64: Vec<u64> = (0..size).map(|_| rng.gen()).collect();
        let arr_b_u64: Vec<u64> = (0..size).map(|_| rng.gen()).collect();
        let arr_a_fr: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut field_rng)).collect();
        let arr_b_fr: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut field_rng)).collect();

        // Sequential u64 multiplication
        group.bench_with_input(BenchmarkId::new("Sequential::u64", size), &size, |b, _| {
            b.iter(|| {
                arr_a_u64
                    .iter()
                    .zip(arr_b_u64.iter())
                    .map(|(&x, &y)| multiply_u64(x, y))
                    .collect::<Vec<_>>()
            })
        });

        // Parallel u64 multiplication
        group.bench_with_input(BenchmarkId::new("Parallel::u64", size), &size, |b, _| {
            b.iter(|| {
                arr_a_u64
                    .par_iter()
                    .zip(arr_b_u64.par_iter())
                    .map(|(&x, &y)| multiply_u64(x, y))
                    .collect::<Vec<_>>()
            })
        });

        // Sequential field multiplication
        group.bench_with_input(
            BenchmarkId::new("Sequential::field_mul", size),
            &size,
            |b, _| {
                b.iter(|| {
                    arr_a_fr
                        .iter()
                        .zip(arr_b_fr.iter())
                        .map(|(&x, &y)| multiply_field(x, y))
                        .collect::<Vec<_>>()
                })
            },
        );

        // Parallel field multiplication
        group.bench_with_input(
            BenchmarkId::new("Parallel::field_mul", size),
            &size,
            |b, _| {
                b.iter(|| {
                    arr_a_fr
                        .par_iter()
                        .zip(arr_b_fr.par_iter())
                        .map(|(&x, &y)| multiply_field(x, y))
                        .collect::<Vec<_>>()
                })
            },
        );

        // Sequential field squaring
        group.bench_with_input(
            BenchmarkId::new("Sequential::field_square", size),
            &size,
            |b, _| b.iter(|| arr_a_fr.iter().map(|x| x.square()).collect::<Vec<_>>()),
        );

        // Parallel field squaring
        group.bench_with_input(
            BenchmarkId::new("Parallel::field_square", size),
            &size,
            |b, _| b.iter(|| arr_a_fr.par_iter().map(|x| x.square()).collect::<Vec<_>>()),
        );

        // Sequential field * u64
        group.bench_with_input(
            BenchmarkId::new("Sequential::field_mul_u64", size),
            &size,
            |b, _| {
                b.iter(|| {
                    arr_a_fr
                        .iter()
                        .zip(arr_b_u64.iter())
                        .map(|(&x, &y)| x.mul_u64(y))
                        .collect::<Vec<_>>()
                })
            },
        );

        // Parallel field * u64
        group.bench_with_input(
            BenchmarkId::new("Parallel::field_mul_u64", size),
            &size,
            |b, _| {
                b.iter(|| {
                    arr_a_fr
                        .par_iter()
                        .zip(arr_b_u64.par_iter())
                        .map(|(&x, &y)| x.mul_u64(y))
                        .collect::<Vec<_>>()
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_millis(500));
    targets = benchmark_parallel_operations
);
criterion_main!(benches);

fn benchmark_single_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_multiplication");
    let mut rng = rand::thread_rng();
    let mut field_rng = test_rng();

    // Native integer multiplication benchmarks
    group.bench_function("SingleMultiply::u8(8 bit integers)", |b| {
        let x: u8 = rng.gen();
        let y: u8 = rng.gen();
        b.iter(|| multiply_u8(criterion::black_box(x), criterion::black_box(y)))
    });

    group.bench_function("SingleMultiply::u16(16 bit integers)", |b| {
        let x: u16 = rng.gen();
        let y: u16 = rng.gen();
        b.iter(|| multiply_u16(criterion::black_box(x), criterion::black_box(y)))
    });

    group.bench_function("SingleMultiply::u32(32 bit integers)", |b| {
        let x: u32 = rng.gen();
        let y: u32 = rng.gen();
        b.iter(|| multiply_u32(criterion::black_box(x), criterion::black_box(y)))
    });

    group.bench_function("SingleMultiply::u64(64 bit integers)", |b| {
        let x: u64 = rng.gen();
        let y: u64 = rng.gen();
        b.iter(|| multiply_u64(criterion::black_box(x), criterion::black_box(y)))
    });

    group.bench_function("SingleMultiply::u128(128 bit integers)", |b| {
        let x: u128 = rng.gen();
        let y: u128 = rng.gen();
        b.iter(|| multiply_u128(criterion::black_box(x), criterion::black_box(y)))
    });

    // Field multiplication benchmark
    group.bench_function("SingleMultiply::Fr(BN254 field element)", |b| {
        let x: Fr = Fr::rand(&mut field_rng);
        let y: Fr = Fr::rand(&mut field_rng);
        b.iter(|| multiply_field(criterion::black_box(x), criterion::black_box(y)))
    });

    group.finish();
}

fn benchmark_bulk_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("bulk_multiplication");
    let mut rng = rand::thread_rng();
    let mut field_rng = test_rng();

    let sizes = [100000];

    for size in sizes {
        // // u8 bulk operations
        // let arr_a: Vec<u8> = (0..size).map(|_| rng.gen()).collect();
        // let arr_b: Vec<u8> = (0..size).map(|_| rng.gen()).collect();
        // group.bench_with_input(
        //     BenchmarkId::new("BulkMultiply::u8(8 bit arrays)", size),
        //     &size,
        //     |b, _| {
        //         b.iter(|| {
        //             arr_a
        //                 .iter()
        //                 .zip(arr_b.iter())
        //                 .map(|(&x, &y)| {
        //                     multiply_u8(criterion::black_box(x), criterion::black_box(y))
        //                 })
        //                 .collect::<Vec<_>>()
        //         })
        //     },
        // );

        // // u32 bulk operations
        // let arr_a: Vec<u32> = (0..size).map(|_| rng.gen()).collect();
        // let arr_b: Vec<u32> = (0..size).map(|_| rng.gen()).collect();
        // group.bench_with_input(
        //     BenchmarkId::new("BulkMultiply::u32(32 bit arrays)", size),
        //     &size,
        //     |b, _| {
        //         b.iter(|| {
        //             arr_a
        //                 .iter()
        //                 .zip(arr_b.iter())
        //                 .map(|(&x, &y)| {
        //                     multiply_u32(criterion::black_box(x), criterion::black_box(y))
        //                 })
        //                 .collect::<Vec<_>>()
        //         })
        //     },
        // );

        // u64 bulk operations
        let arr_a_u64: Vec<u64> = (0..size).map(|_| rng.gen()).collect();
        let arr_b_u64: Vec<u64> = (0..size).map(|_| rng.gen()).collect();
        group.bench_with_input(
            BenchmarkId::new("BulkMultiply::u64(64 bit arrays)", size),
            &size,
            |b, _| {
                b.iter(|| {
                    arr_a_u64
                        .iter()
                        .zip(arr_b_u64.iter())
                        .map(|(&x, &y)| {
                            multiply_u64(criterion::black_box(x), criterion::black_box(y))
                        })
                        .collect::<Vec<_>>()
                })
            },
        );

        // u128 bulk operations
        let arr_a: Vec<u128> = (0..size).map(|_| rng.gen()).collect();
        let arr_b: Vec<u128> = (0..size).map(|_| rng.gen()).collect();
        group.bench_with_input(
            BenchmarkId::new("BulkMultiply::u128(128 bit arrays)", size),
            &size,
            |b, _| {
                b.iter(|| {
                    arr_a
                        .iter()
                        .zip(arr_b.iter())
                        .map(|(&x, &y)| {
                            multiply_u128(criterion::black_box(x), criterion::black_box(y))
                        })
                        .collect::<Vec<_>>()
                })
            },
        );
        // Field bulk operations
        let arr_a: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut field_rng)).collect();
        let arr_b: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut field_rng)).collect();
        group.bench_with_input(
            BenchmarkId::new("BulkAdd::Fr(BN254 field element arrays)", size),
            &size,
            |b, _| {
                b.iter(|| {
                    arr_a
                        .iter()
                        .zip(arr_b.iter())
                        .map(|(&x, &y)| criterion::black_box(x) + criterion::black_box(y))
                        .collect::<Vec<_>>()
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("BulkMultiply::Fr(BN254 field element arrays)", size),
            &size,
            |b, _| {
                b.iter(|| {
                    arr_a
                        .iter()
                        .zip(arr_b.iter())
                        .map(|(&x, &y)| {
                            multiply_field(criterion::black_box(x), criterion::black_box(y))
                        })
                        .collect::<Vec<_>>()
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("BulkSquare::Fr(BN254 field element arrays)", size),
            &size,
            |b, _| b.iter(|| arr_a.iter().map(|x| x.square()).collect::<Vec<_>>()),
        );
        // group.bench_with_input(
        //     BenchmarkId::new("BulkInverse::Fr(BN254 field element arrays)", size),
        //     &size,
        //     |b, _| b.iter(|| arr_a.iter().map(|x| x.inverse()).collect::<Vec<_>>()),
        // );
        group.bench_with_input(
            BenchmarkId::new("BulkMulU64::Fr(BN254 field element arrays)", size),
            &size,
            |b, _| {
                b.iter(|| {
                    arr_a
                        .iter()
                        .zip(arr_b_u64.iter())
                        .map(|(x, y)| x.mul_u64(criterion::black_box(*y)))
                        .collect::<Vec<_>>()
                })
            },
        );
    }

    group.finish();
}

fn main() {
    let mut criterion = Criterion::default()
        .configure_from_args()
        .sample_size(20)
        .warm_up_time(std::time::Duration::from_secs(2));

    // benchmark_single_multiplication(&mut criterion);
    benchmark_bulk_multiplication(&mut criterion);

    criterion.final_summary();
}
