#![expect(unused_results)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use jolt_field::Fr;
use jolt_transcript::{
    prover_transcript, Blake2b512, FieldTranscript, Keccak, PoseidonSponge, ProverState, Transcript,
};
use spongefish::DuplexSpongeInterface;

fn fresh<H: DuplexSpongeInterface<U = u8> + spongefish::DuplexSpongeInit>() -> ProverState<H> {
    prover_transcript::<H>(b"bench", b"bench")
}

fn bench_append_bytes_with<H>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &str,
    label: &str,
    data: &[u8],
) where
    H: DuplexSpongeInterface<U = u8> + spongefish::DuplexSpongeInit,
{
    group.bench_with_input(BenchmarkId::new(name, label), data, |bench, data| {
        bench.iter_batched(
            fresh::<H>,
            |mut t| {
                t.append_bytes(black_box(data));
                t
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

fn bench_append_bytes(c: &mut Criterion) {
    let mut group = c.benchmark_group("append_bytes");
    let data_32 = [0xABu8; 32];
    let data_256 = [0xCDu8; 256];

    for (label, data) in [("32B", &data_32[..]), ("256B", &data_256[..])] {
        bench_append_bytes_with::<Blake2b512>(&mut group, "Blake2b", label, data);
        bench_append_bytes_with::<Keccak>(&mut group, "Keccak", label, data);
        bench_append_bytes_with::<PoseidonSponge>(&mut group, "Poseidon", label, data);
    }
    group.finish();
}

fn bench_challenge(c: &mut Criterion) {
    let mut group = c.benchmark_group("challenge");

    group.bench_function("Blake2b", |bench| {
        bench.iter_batched(
            || {
                let mut t = fresh::<Blake2b512>();
                t.append_bytes(&[42u8; 32]);
                t
            },
            |mut t| FieldTranscript::<Fr>::challenge(&mut t),
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("Keccak", |bench| {
        bench.iter_batched(
            || {
                let mut t = fresh::<Keccak>();
                t.append_bytes(&[42u8; 32]);
                t
            },
            |mut t| FieldTranscript::<Fr>::challenge(&mut t),
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_append_bytes, bench_challenge);
criterion_main!(benches);
