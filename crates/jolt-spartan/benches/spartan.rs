#![allow(unused_results)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use jolt_field::{Field, Fr};
use jolt_openings::mock::MockCommitmentScheme;
use jolt_openings::CommitmentScheme;
use jolt_spartan::{FirstRoundStrategy, SimpleR1CS, SpartanKey, SpartanProver, SpartanVerifier};
use jolt_transcript::{Blake2bTranscript, Transcript};

type MockPCS = MockCommitmentScheme<Fr>;

fn commit_witness(witness: &[Fr], padded_len: usize, transcript: &mut Blake2bTranscript) {
    let mut padded = vec![Fr::from_u64(0); padded_len];
    let copy_len = witness.len().min(padded_len);
    padded[..copy_len].copy_from_slice(&witness[..copy_len]);
    let (commitment, ()) = MockPCS::commit(&padded, &());
    transcript.append_bytes(format!("{commitment:?}").as_bytes());
}

/// Builds a chain-multiplication R1CS with `n_constraints` constraints.
///
/// z = [1, x, x^2, x^3, ..., x^{n+1}]
/// Constraint i: z[i+1] * z[1] = z[i+2]
fn chain_mul_circuit(n_constraints: usize) -> (SimpleR1CS<Fr>, Vec<Fr>) {
    let one = Fr::from_u64(1);
    let n_vars = n_constraints + 2; // 1 + x + n_constraints outputs

    let a_entries: Vec<_> = (0..n_constraints).map(|i| (i, i + 1, one)).collect();
    let b_entries: Vec<_> = (0..n_constraints).map(|i| (i, 1, one)).collect();
    let c_entries: Vec<_> = (0..n_constraints).map(|i| (i, i + 2, one)).collect();

    let r1cs = SimpleR1CS::new(n_constraints, n_vars, a_entries, b_entries, c_entries);

    // x = 3
    let x = Fr::from_u64(3);
    let mut witness = Vec::with_capacity(n_vars);
    witness.push(Fr::from_u64(1)); // z[0] = 1
    let mut power = Fr::from_u64(1);
    for _ in 0..n_vars - 1 {
        power *= x;
        witness.push(power);
    }

    (r1cs, witness)
}

fn bench_prove(c: &mut Criterion) {
    let mut group = c.benchmark_group("SpartanProver::prove");
    for n_constraints in [4, 16, 64, 256] {
        let (r1cs, witness) = chain_mul_circuit(n_constraints);
        let key = SpartanKey::from_r1cs(&r1cs);

        group.bench_with_input(
            BenchmarkId::from_parameter(n_constraints),
            &n_constraints,
            |bench, _| {
                bench.iter_batched(
                    || {
                        let mut t = Blake2bTranscript::new(b"bench");
                        commit_witness(&witness, key.num_variables_padded, &mut t);
                        t
                    },
                    |mut transcript| {
                        SpartanProver::prove(
                            black_box(&r1cs),
                            black_box(&key),
                            black_box(&witness),
                            &mut transcript,
                            FirstRoundStrategy::Standard,
                        )
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("SpartanVerifier::verify");
    for n_constraints in [4, 16, 64, 256] {
        let (r1cs, witness) = chain_mul_circuit(n_constraints);
        let key = SpartanKey::from_r1cs(&r1cs);

        let mut pt = Blake2bTranscript::new(b"bench");
        commit_witness(&witness, key.num_variables_padded, &mut pt);
        let proof =
            SpartanProver::prove(&r1cs, &key, &witness, &mut pt, FirstRoundStrategy::Standard)
                .expect("proving should succeed");

        group.bench_with_input(
            BenchmarkId::from_parameter(n_constraints),
            &n_constraints,
            |bench, _| {
                bench.iter_batched(
                    || {
                        let mut t = Blake2bTranscript::new(b"bench");
                        commit_witness(&witness, key.num_variables_padded, &mut t);
                        t
                    },
                    |mut transcript| {
                        SpartanVerifier::verify(black_box(&key), black_box(&proof), &mut transcript)
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_key_gen(c: &mut Criterion) {
    let mut group = c.benchmark_group("SpartanKey::from_r1cs");
    for n_constraints in [4, 16, 64, 256] {
        let (r1cs, _) = chain_mul_circuit(n_constraints);

        group.bench_with_input(
            BenchmarkId::from_parameter(n_constraints),
            &n_constraints,
            |bench, _| {
                bench.iter(|| SpartanKey::from_r1cs(black_box(&r1cs)));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_prove, bench_verify, bench_key_gen);
criterion_main!(benches);
