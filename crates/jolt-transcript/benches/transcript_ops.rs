#![expect(unused_results)]

use std::hint::black_box;

use criterion::measurement::WallTime;
use criterion::{
    criterion_group, criterion_main, BatchSize, BenchmarkGroup, BenchmarkId, Criterion,
};
use jolt_field::Fr;
use jolt_transcript::{
    Blake2b512, Channel, Keccak, PoseidonSponge, ProtocolId, ProverTranscript, Sponge,
};

fn transcript<H: Sponge>() -> ProverTranscript<H> {
    ProverTranscript::new(&ProtocolId::new::<H>("bench"), b"")
}

fn bench_send<H: Sponge>(group: &mut BenchmarkGroup<'_, WallTime>, name: &str) {
    let values: Vec<Fr> = (0..8u64).map(Fr::from).collect();
    group.bench_with_input(BenchmarkId::new(name, "8xFr"), &values, |bench, values| {
        bench.iter_batched(
            transcript::<H>,
            |mut t| {
                t.send_all(black_box(values));
                t
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_public_bytes<H: Sponge>(group: &mut BenchmarkGroup<'_, WallTime>, name: &str) {
    for (label, data) in [("32B", &[0xABu8; 32][..]), ("256B", &[0xCDu8; 256][..])] {
        group.bench_with_input(BenchmarkId::new(name, label), data, |bench, data| {
            bench.iter_batched(
                transcript::<H>,
                |mut t| {
                    t.public_bytes(black_box(data));
                    t
                },
                BatchSize::SmallInput,
            );
        });
    }
}

fn bench_challenge<H: Sponge>(group: &mut BenchmarkGroup<'_, WallTime>, name: &str) {
    let seeded = || {
        let mut t = transcript::<H>();
        t.public_bytes(&[42u8; 32]);
        t
    };
    group.bench_function(BenchmarkId::new(name, "uniform"), |bench| {
        bench.iter_batched(seeded, |mut t| t.challenge::<Fr>(), BatchSize::SmallInput);
    });
    group.bench_function(BenchmarkId::new(name, "small"), |bench| {
        bench.iter_batched(
            seeded,
            |mut t| t.challenge_small::<Fr>(),
            BatchSize::SmallInput,
        );
    });
}

fn bench_send_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("send");
    bench_send::<Blake2b512>(&mut group, "Blake2b");
    bench_send::<Keccak>(&mut group, "Keccak");
    bench_send::<PoseidonSponge>(&mut group, "Poseidon");
    group.finish();
}

fn bench_public(c: &mut Criterion) {
    let mut group = c.benchmark_group("public_bytes");
    bench_public_bytes::<Blake2b512>(&mut group, "Blake2b");
    bench_public_bytes::<Keccak>(&mut group, "Keccak");
    bench_public_bytes::<PoseidonSponge>(&mut group, "Poseidon");
    group.finish();
}

fn bench_challenges(c: &mut Criterion) {
    let mut group = c.benchmark_group("challenge");
    bench_challenge::<Blake2b512>(&mut group, "Blake2b");
    bench_challenge::<Keccak>(&mut group, "Keccak");
    bench_challenge::<PoseidonSponge>(&mut group, "Poseidon");
    group.finish();
}

criterion_group!(benches, bench_send_all, bench_public, bench_challenges);
criterion_main!(benches);
