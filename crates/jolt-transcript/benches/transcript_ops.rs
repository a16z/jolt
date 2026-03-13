#![allow(unused_results)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use jolt_field::Fr;
use jolt_transcript::{Blake2bTranscript, KeccakTranscript, Transcript};

fn bench_append_bytes(c: &mut Criterion) {
    let mut group = c.benchmark_group("append_bytes");
    let data_32 = [0xABu8; 32];
    let data_256 = [0xCDu8; 256];

    for (label, data) in [("32B", &data_32[..]), ("256B", &data_256[..])] {
        group.bench_with_input(BenchmarkId::new("Blake2b", label), data, |bench, data| {
            bench.iter_batched(
                || Blake2bTranscript::<Fr>::new(b"bench"),
                |mut t| {
                    t.append_bytes(black_box(data));
                    t
                },
                criterion::BatchSize::SmallInput,
            );
        });
        group.bench_with_input(BenchmarkId::new("Keccak", label), data, |bench, data| {
            bench.iter_batched(
                || KeccakTranscript::<Fr>::new(b"bench"),
                |mut t| {
                    t.append_bytes(black_box(data));
                    t
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_challenge(c: &mut Criterion) {
    let mut group = c.benchmark_group("challenge");

    group.bench_function("Blake2b", |bench| {
        bench.iter_batched(
            || {
                let mut t = Blake2bTranscript::<Fr>::new(b"bench");
                t.append_bytes(&[42u8; 32]);
                t
            },
            |mut t| t.challenge(),
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("Keccak", |bench| {
        bench.iter_batched(
            || {
                let mut t = KeccakTranscript::<Fr>::new(b"bench");
                t.append_bytes(&[42u8; 32]);
                t
            },
            |mut t| t.challenge(),
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_append_bytes, bench_challenge);
criterion_main!(benches);
