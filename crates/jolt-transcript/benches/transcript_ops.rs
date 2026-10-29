#![expect(unused_results)]

use std::hint::black_box;

use criterion::measurement::WallTime;
use criterion::{
    criterion_group, criterion_main, BatchSize, BenchmarkGroup, BenchmarkId, Criterion,
};
use jolt_field::Fr;
use jolt_transcript::{
    protocol_tag, prover_transcript, Blake2b512, FieldTranscript, Keccak, PoseidonSponge,
    ProverState, SpongeUnit, Transcript,
};
use spongefish::DuplexSpongeInit;

fn fresh<H: DuplexSpongeInit<U: SpongeUnit>>() -> ProverState<H> {
    prover_transcript::<H>(&protocol_tag(b"bench"), b"bench")
}

fn bench_append_bytes_with<H>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    label: &str,
    data: &[u8],
) where
    H: DuplexSpongeInit<U: SpongeUnit>,
{
    group.bench_with_input(BenchmarkId::new(name, label), data, |bench, data| {
        bench.iter_batched(
            fresh::<H>,
            |mut t| {
                t.append_bytes(black_box(data));
                t
            },
            BatchSize::SmallInput,
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
            BatchSize::SmallInput,
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
            BatchSize::SmallInput,
        );
    });

    group.bench_function("Poseidon", |bench| {
        bench.iter_batched(
            || {
                let mut t = fresh::<PoseidonSponge>();
                t.append_bytes(&[42u8; 32]);
                t
            },
            |mut t| FieldTranscript::<Fr>::challenge(&mut t),
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_append_bytes, bench_challenge);
criterion_main!(benches);
