#![allow(unused_results)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use jolt_blindfold::{
    BakedPublicInputs, BlindFoldAccumulator, BlindFoldProver, BlindFoldVerifier,
    CommittedRoundData, RelaxedWitness, StageConfig,
};
use jolt_crypto::arkworks::bn254::Bn254G1;
use jolt_crypto::Pedersen;
use jolt_field::{Field, Fr};
use jolt_openings::mock::MockCommitmentScheme;
use jolt_openings::CommitmentScheme;
use jolt_poly::{Polynomial, UnivariatePoly};
use jolt_spartan::{SpartanKey, SpartanProver, SpartanVerifier, R1CS};
use jolt_sumcheck::{
    ClearRoundHandler, RoundHandler, SumcheckClaim, SumcheckCompute, SumcheckProver,
};
use jolt_transcript::{Blake2bTranscript, Transcript};
use num_traits::{One, Zero};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type TestVC = Pedersen<Bn254G1>;
type MockPCS = MockCommitmentScheme<Fr>;

struct IpWitness {
    a: Polynomial<Fr>,
    b: Polynomial<Fr>,
}

impl SumcheckCompute<Fr> for IpWitness {
    fn round_polynomial(&self) -> UnivariatePoly<Fr> {
        let half = self.a.evaluations().len() / 2;
        let a = self.a.evaluations();
        let b = self.b.evaluations();
        let mut evals = [Fr::zero(); 3];
        for i in 0..half {
            let a_lo = a[i];
            let a_hi = a[i + half];
            let b_lo = b[i];
            let b_hi = b[i + half];
            let a_delta = a_hi - a_lo;
            let b_delta = b_hi - b_lo;
            for (t, eval) in evals.iter_mut().enumerate() {
                let x = Fr::from_u64(t as u64);
                *eval += (a_lo + x * a_delta) * (b_lo + x * b_delta);
            }
        }
        let points: Vec<(Fr, Fr)> = (0..3).map(|t| (Fr::from_u64(t as u64), evals[t])).collect();
        UnivariatePoly::interpolate(&points)
    }

    fn bind(&mut self, challenge: Fr) {
        self.a.bind(challenge);
        self.b.bind(challenge);
    }
}

struct RecordingHandler {
    inner: ClearRoundHandler<Fr>,
    challenges: Vec<Fr>,
    round_polys: Vec<Vec<Fr>>,
}

impl RecordingHandler {
    fn new(cap: usize) -> Self {
        Self {
            inner: ClearRoundHandler::with_capacity(cap),
            challenges: Vec::with_capacity(cap),
            round_polys: Vec::with_capacity(cap),
        }
    }
}

impl RoundHandler<Fr> for RecordingHandler {
    type Proof = (
        jolt_sumcheck::proof::SumcheckProof<Fr>,
        Vec<Fr>,
        Vec<Vec<Fr>>,
    );

    fn absorb_round_poly(&mut self, poly: &UnivariatePoly<Fr>, transcript: &mut impl Transcript) {
        self.round_polys.push(poly.coefficients().to_vec());
        self.inner.absorb_round_poly(poly, transcript);
    }

    fn on_challenge(&mut self, challenge: Fr) {
        self.challenges.push(challenge);
    }

    fn finalize(
        self,
    ) -> (
        jolt_sumcheck::proof::SumcheckProof<Fr>,
        Vec<Fr>,
        Vec<Vec<Fr>>,
    ) {
        (self.inner.finalize(), self.challenges, self.round_polys)
    }
}

fn run_ip_stage(
    a_vals: Vec<Fr>,
    b_vals: Vec<Fr>,
    claimed_sum: Fr,
    transcript: &mut Blake2bTranscript,
) -> (StageConfig<Fr>, Vec<Fr>, Vec<Vec<Fr>>) {
    let num_vars = a_vals.len().trailing_zeros() as usize;
    let mut witness = IpWitness {
        a: Polynomial::new(a_vals),
        b: Polynomial::new(b_vals),
    };
    let claim = SumcheckClaim {
        num_vars,
        degree: 2,
        claimed_sum,
    };
    let handler = RecordingHandler::new(num_vars);
    let (_proof, challenges, round_polys) = SumcheckProver::prove_with_handler(
        &claim,
        &mut witness,
        transcript,
        |c: u128| Fr::from_u128(c),
        handler,
    );
    let config = StageConfig {
        num_rounds: num_vars,
        degree: 2,
        claimed_sum,
    };
    (config, challenges, round_polys)
}

/// Generates a single sumcheck stage with `num_rounds` rounds (2^num_rounds evaluations).
fn generate_stage(
    num_rounds: usize,
) -> (
    Vec<StageConfig<Fr>>,
    Vec<Fr>,
    BlindFoldAccumulator<Fr, TestVC>,
) {
    let n = 1 << num_rounds;
    let mut transcript = Blake2bTranscript::new(b"bench-stages");
    let a_vals: Vec<Fr> = (1..=n as u64).map(Fr::from_u64).collect();
    let b_vals: Vec<Fr> = vec![Fr::one(); n];
    let claimed_sum: Fr = a_vals.iter().copied().sum();

    let (config, challenges, round_polys) =
        run_ip_stage(a_vals, b_vals, claimed_sum, &mut transcript);

    let mut acc = BlindFoldAccumulator::new();
    acc.push_stage(CommittedRoundData {
        round_commitments: vec![Bn254G1::default(); round_polys.len()],
        poly_coeffs: round_polys,
        blinding_factors: vec![Fr::zero(); challenges.len()],
        poly_degrees: vec![2; challenges.len()],
        challenges: challenges.clone(),
    });

    (vec![config], challenges, acc)
}

fn pad_to(data: &[Fr], target_len: usize) -> Vec<Fr> {
    let mut v = vec![Fr::zero(); target_len];
    let copy_len = data.len().min(target_len);
    v[..copy_len].copy_from_slice(&data[..copy_len]);
    v
}

fn bench_build_verifier_r1cs(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_verifier_r1cs");
    for total_rounds in [2, 4, 8, 16] {
        let (configs, all_challenges, _acc) = generate_stage(total_rounds);
        let baked = BakedPublicInputs {
            challenges: all_challenges,
        };

        group.bench_with_input(
            BenchmarkId::from_parameter(total_rounds),
            &total_rounds,
            |bench, _| {
                bench.iter(|| {
                    let r1cs = jolt_blindfold::verifier_r1cs::build_verifier_r1cs(
                        black_box(&configs),
                        black_box(&baked),
                    );
                    SpartanKey::from_r1cs(black_box(&r1cs))
                });
            },
        );
    }
    group.finish();
}

fn bench_assign_witness(c: &mut Criterion) {
    let mut group = c.benchmark_group("assign_witness");
    for total_rounds in [2, 4, 8, 16] {
        let (configs, all_challenges, acc) = generate_stage(total_rounds);
        let baked = BakedPublicInputs {
            challenges: all_challenges,
        };

        let stage_coefficients: Vec<Vec<Vec<Fr>>> = acc
            .into_stages()
            .into_iter()
            .map(|s| s.round_data.poly_coeffs)
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(total_rounds),
            &total_rounds,
            |bench, _| {
                bench.iter(|| {
                    jolt_blindfold::verifier_r1cs::assign_witness(
                        black_box(&configs),
                        black_box(&baked),
                        black_box(&stage_coefficients),
                    )
                });
            },
        );
    }
    group.finish();
}

fn bench_nova_folding(c: &mut Criterion) {
    use jolt_blindfold::folding::{
        compute_cross_term, fold_scalar, fold_witnesses, sample_random_witness,
    };

    let mut group = c.benchmark_group("nova_folding");
    for total_rounds in [2, 4, 8, 16] {
        let (configs, all_challenges, acc) = generate_stage(total_rounds);
        let baked = BakedPublicInputs {
            challenges: all_challenges,
        };

        let stage_coefficients: Vec<Vec<Vec<Fr>>> = acc
            .into_stages()
            .into_iter()
            .map(|s| s.round_data.poly_coeffs)
            .collect();

        let r1cs =
            jolt_blindfold::verifier_r1cs::build_verifier_r1cs(&configs, &baked);
        let z_real =
            jolt_blindfold::verifier_r1cs::assign_witness(&configs, &baked, &stage_coefficients);
        let u_real = Fr::one();

        group.bench_with_input(
            BenchmarkId::from_parameter(total_rounds),
            &total_rounds,
            |bench, _| {
                bench.iter_batched(
                    || ChaCha20Rng::seed_from_u64(42),
                    |mut rng| {
                        let (u_rand, w_rand) =
                            sample_random_witness(black_box(&r1cs), &mut rng);
                        let cross =
                            compute_cross_term(&r1cs, &z_real, u_real, &w_rand.w, u_rand);
                        let w_real = RelaxedWitness {
                            w: z_real.clone(),
                            e: vec![Fr::zero(); r1cs.num_constraints()],
                        };
                        let _u_folded = fold_scalar(u_real, u_rand, Fr::from_u64(17));
                        fold_witnesses(&w_real, &w_rand, &cross, Fr::from_u64(17))
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_prove_relaxed(c: &mut Criterion) {
    use jolt_blindfold::folding::{
        compute_cross_term, fold_scalar, fold_witnesses, sample_random_witness,
    };

    let mut group = c.benchmark_group("prove_relaxed");
    for total_rounds in [2, 4, 8, 16] {
        let (configs, all_challenges, acc) = generate_stage(total_rounds);
        let baked = BakedPublicInputs {
            challenges: all_challenges,
        };

        let stage_coefficients: Vec<Vec<Vec<Fr>>> = acc
            .into_stages()
            .into_iter()
            .map(|s| s.round_data.poly_coeffs)
            .collect();

        let r1cs =
            jolt_blindfold::verifier_r1cs::build_verifier_r1cs(&configs, &baked);
        let z_real =
            jolt_blindfold::verifier_r1cs::assign_witness(&configs, &baked, &stage_coefficients);
        let key = SpartanKey::from_r1cs(&r1cs);

        let u_real = Fr::one();
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let (u_rand, w_rand) = sample_random_witness(&r1cs, &mut rng);
        let cross = compute_cross_term(&r1cs, &z_real, u_real, &w_rand.w, u_rand);
        let w_real = RelaxedWitness {
            w: z_real,
            e: vec![Fr::zero(); r1cs.num_constraints()],
        };
        let folded = fold_witnesses(&w_real, &w_rand, &cross, Fr::from_u64(17));
        let u_folded = fold_scalar(u_real, u_rand, Fr::from_u64(17));

        let w_padded = pad_to(&folded.w, key.num_variables_padded);
        let e_padded = pad_to(&folded.e, key.num_constraints_padded);
        let (w_com, ()) = MockPCS::commit(&w_padded, &());
        let (e_com, ()) = MockPCS::commit(&e_padded, &());

        group.bench_with_input(
            BenchmarkId::from_parameter(total_rounds),
            &total_rounds,
            |bench, _| {
                bench.iter_batched(
                    || Blake2bTranscript::new(b"bench"),
                    |mut transcript| {
                        SpartanProver::prove_relaxed::<MockPCS, _>(
                            black_box(&r1cs),
                            black_box(&key),
                            black_box(u_folded),
                            black_box(&folded.w),
                            black_box(&folded.e),
                            black_box(&w_com),
                            black_box(&e_com),
                            &(),
                            &mut transcript,
                        )
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_verify_relaxed(c: &mut Criterion) {
    use jolt_blindfold::folding::{
        compute_cross_term, fold_scalar, fold_witnesses, sample_random_witness,
    };

    let mut group = c.benchmark_group("verify_relaxed");
    for total_rounds in [2, 4, 8, 16] {
        let (configs, all_challenges, acc) = generate_stage(total_rounds);
        let baked = BakedPublicInputs {
            challenges: all_challenges,
        };

        let stage_coefficients: Vec<Vec<Vec<Fr>>> = acc
            .into_stages()
            .into_iter()
            .map(|s| s.round_data.poly_coeffs)
            .collect();

        let r1cs =
            jolt_blindfold::verifier_r1cs::build_verifier_r1cs(&configs, &baked);
        let z_real =
            jolt_blindfold::verifier_r1cs::assign_witness(&configs, &baked, &stage_coefficients);
        let key = SpartanKey::from_r1cs(&r1cs);

        let u_real = Fr::one();
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let (u_rand, w_rand) = sample_random_witness(&r1cs, &mut rng);
        let cross = compute_cross_term(&r1cs, &z_real, u_real, &w_rand.w, u_rand);
        let w_real = RelaxedWitness {
            w: z_real,
            e: vec![Fr::zero(); r1cs.num_constraints()],
        };
        let folded = fold_witnesses(&w_real, &w_rand, &cross, Fr::from_u64(17));
        let u_folded = fold_scalar(u_real, u_rand, Fr::from_u64(17));

        let w_padded = pad_to(&folded.w, key.num_variables_padded);
        let e_padded = pad_to(&folded.e, key.num_constraints_padded);
        let (w_com, ()) = MockPCS::commit(&w_padded, &());
        let (e_com, ()) = MockPCS::commit(&e_padded, &());

        let mut pt = Blake2bTranscript::new(b"bench");
        let proof = SpartanProver::prove_relaxed::<MockPCS, _>(
            &r1cs, &key, u_folded, &folded.w, &folded.e, &w_com, &e_com, &(), &mut pt,
        )
        .expect("prove_relaxed must succeed");

        group.bench_with_input(
            BenchmarkId::from_parameter(total_rounds),
            &total_rounds,
            |bench, _| {
                bench.iter_batched(
                    || Blake2bTranscript::new(b"bench"),
                    |mut transcript| {
                        SpartanVerifier::verify_relaxed::<MockPCS, _>(
                            black_box(&key),
                            black_box(u_folded),
                            black_box(&w_com),
                            black_box(&e_com),
                            black_box(&proof),
                            &(),
                            &mut transcript,
                        )
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_blindfold_e2e_prove(c: &mut Criterion) {
    let mut group = c.benchmark_group("BlindFoldProver::prove");
    for total_rounds in [2, 4, 8, 16] {
        let (configs, _all_challenges, acc) = generate_stage(total_rounds);

        group.bench_with_input(
            BenchmarkId::from_parameter(total_rounds),
            &total_rounds,
            |bench, _| {
                bench.iter_batched(
                    || {
                        (
                            acc.clone(),
                            Blake2bTranscript::new(b"bench"),
                            ChaCha20Rng::seed_from_u64(42),
                        )
                    },
                    |(acc, mut transcript, mut rng)| {
                        BlindFoldProver::prove::<TestVC, MockPCS, _>(
                            black_box(acc),
                            black_box(&configs),
                            &(),
                            &mut transcript,
                            &mut rng,
                        )
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_blindfold_e2e_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("BlindFoldVerifier::verify");
    for total_rounds in [2, 4, 8, 16] {
        let (configs, all_challenges, acc) = generate_stage(total_rounds);

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let mut prove_transcript = Blake2bTranscript::new(b"bench");
        let proof = BlindFoldProver::prove::<TestVC, MockPCS, _>(
            acc,
            &configs,
            &(),
            &mut prove_transcript,
            &mut rng,
        )
        .expect("prove must succeed for verify benchmark");

        let baked = BakedPublicInputs {
            challenges: all_challenges,
        };

        group.bench_with_input(
            BenchmarkId::from_parameter(total_rounds),
            &total_rounds,
            |bench, _| {
                bench.iter_batched(
                    || Blake2bTranscript::new(b"bench"),
                    |mut transcript| {
                        BlindFoldVerifier::verify::<MockPCS, _>(
                            black_box(&proof),
                            black_box(&configs),
                            black_box(&baked),
                            &(),
                            &mut transcript,
                        )
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_build_verifier_r1cs,
    bench_assign_witness,
    bench_nova_folding,
    bench_prove_relaxed,
    bench_verify_relaxed,
    bench_blindfold_e2e_prove,
    bench_blindfold_e2e_verify,
);
criterion_main!(benches);
