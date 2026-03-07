//! Integration tests exercising the Spartan ↔ BlindFold boundary.
//!
//! These tests go beyond the unit tests in `protocol.rs` by covering:
//! - Higher-degree sumcheck stages (degree 3)
//! - Many-stage accumulation (4+ stages)
//! - Larger polynomial sizes (8 sumcheck rounds)
//! - Proof serialization round-trips
//! - Additional tampering vectors (witness/error commitments, baked challenges)
//! - Cross-RNG-seed determinism
//! - Mixed-degree multi-stage pipelines
//! - Relaxed Spartan across various R1CS shapes

use jolt_blindfold::{
    BakedPublicInputs, BlindFoldAccumulator, BlindFoldProof, BlindFoldProver, BlindFoldVerifier,
    CommittedRoundData, RelaxedWitness, StageConfig,
};
use jolt_crypto::arkworks::bn254::Bn254G1;
use jolt_crypto::Pedersen;
use jolt_field::{Field, Fr};
use jolt_openings::mock::MockCommitmentScheme;
use jolt_openings::CommitmentScheme;
use jolt_poly::{Polynomial, UnivariatePoly};
use jolt_spartan::{SimpleR1CS, SpartanKey, SpartanProver, SpartanVerifier, R1CS};
use jolt_sumcheck::{
    ClearRoundHandler, RoundHandler, SumcheckClaim, SumcheckCompute, SumcheckProver,
};
use jolt_transcript::{Blake2bTranscript, Transcript};
use num_traits::{One, Zero};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type TestVC = Pedersen<Bn254G1>;
type MockPCS = MockCommitmentScheme<Fr>;

/// Degree-2 inner-product witness: g(x) = a(x) * b(x).
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

/// Degree-3 witness: g(x) = a(x) * b(x) * c(x).
struct CubicWitness {
    a: Polynomial<Fr>,
    b: Polynomial<Fr>,
    c: Polynomial<Fr>,
}

impl SumcheckCompute<Fr> for CubicWitness {
    fn round_polynomial(&self) -> UnivariatePoly<Fr> {
        let half = self.a.evaluations().len() / 2;
        let a = self.a.evaluations();
        let b = self.b.evaluations();
        let c = self.c.evaluations();
        // degree-3 round poly needs 4 evaluation points
        let mut evals = [Fr::zero(); 4];
        for i in 0..half {
            let a_lo = a[i];
            let a_hi = a[i + half];
            let b_lo = b[i];
            let b_hi = b[i + half];
            let c_lo = c[i];
            let c_hi = c[i + half];
            let a_delta = a_hi - a_lo;
            let b_delta = b_hi - b_lo;
            let c_delta = c_hi - c_lo;
            for (t, eval) in evals.iter_mut().enumerate() {
                let x = Fr::from_u64(t as u64);
                *eval += (a_lo + x * a_delta) * (b_lo + x * b_delta) * (c_lo + x * c_delta);
            }
        }
        let points: Vec<(Fr, Fr)> = (0..4).map(|t| (Fr::from_u64(t as u64), evals[t])).collect();
        UnivariatePoly::interpolate(&points)
    }

    fn bind(&mut self, challenge: Fr) {
        self.a.bind(challenge);
        self.b.bind(challenge);
        self.c.bind(challenge);
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

/// Runs a degree-2 sumcheck stage and returns (StageConfig, challenges, round_polys).
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

/// Runs a degree-3 sumcheck stage.
fn run_cubic_stage(
    a_vals: Vec<Fr>,
    b_vals: Vec<Fr>,
    c_vals: Vec<Fr>,
    claimed_sum: Fr,
    transcript: &mut Blake2bTranscript,
) -> (StageConfig<Fr>, Vec<Fr>, Vec<Vec<Fr>>) {
    let num_vars = a_vals.len().trailing_zeros() as usize;
    let mut witness = CubicWitness {
        a: Polynomial::new(a_vals),
        b: Polynomial::new(b_vals),
        c: Polynomial::new(c_vals),
    };
    let claim = SumcheckClaim {
        num_vars,
        degree: 3,
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
        degree: 3,
        claimed_sum,
    };
    (config, challenges, round_polys)
}

/// Builds a single-stage BlindFoldAccumulator from recorded round data.
fn build_single_accumulator(
    challenges: &[Fr],
    round_polys: &[Vec<Fr>],
    degree: usize,
) -> BlindFoldAccumulator<Fr, TestVC> {
    let mut acc = BlindFoldAccumulator::new();
    acc.push_stage(CommittedRoundData {
        round_commitments: vec![Bn254G1::default(); round_polys.len()],
        poly_coeffs: round_polys.to_vec(),
        blinding_factors: vec![Fr::zero(); round_polys.len()],
        poly_degrees: vec![degree; round_polys.len()],
        challenges: challenges.to_vec(),
    });
    acc
}

/// Full BlindFold prove-and-verify pipeline with a given RNG seed.
/// Returns the proof for tests that need to inspect or serialize it.
fn prove_and_verify(
    acc: BlindFoldAccumulator<Fr, TestVC>,
    stage_configs: &[StageConfig<Fr>],
    all_challenges: &[Fr],
    seed: u64,
) -> Result<BlindFoldProof<Fr, MockPCS>, jolt_blindfold::BlindFoldError> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut prove_transcript = Blake2bTranscript::new(b"integ-test-prove");
    let proof = BlindFoldProver::prove::<TestVC, MockPCS, _>(
        acc,
        stage_configs,
        &(),
        &mut prove_transcript,
        &mut rng,
    )?;

    let baked = BakedPublicInputs {
        challenges: all_challenges.to_vec(),
    };
    let mut verify_transcript = Blake2bTranscript::new(b"integ-test-prove");
    BlindFoldVerifier::verify::<MockPCS, _>(
        &proof,
        stage_configs,
        &baked,
        &(),
        &mut verify_transcript,
    )?;

    Ok(proof)
}

/// Prove-and-verify without returning the proof (avoids unused result warnings).
fn assert_prove_and_verify(
    acc: BlindFoldAccumulator<Fr, TestVC>,
    stage_configs: &[StageConfig<Fr>],
    all_challenges: &[Fr],
    seed: u64,
    msg: &str,
) {
    let _ = prove_and_verify(acc, stage_configs, all_challenges, seed)
        .unwrap_or_else(|e| panic!("{msg}: {e}"));
}

/// Pads a slice to target_len with zeros.
fn pad(data: &[Fr], target_len: usize) -> Vec<Fr> {
    let mut v = vec![Fr::zero(); target_len];
    let copy_len = data.len().min(target_len);
    v[..copy_len].copy_from_slice(&data[..copy_len]);
    v
}

#[test]
fn higher_degree_stage() {
    // Degree-3: g(x) = a(x) * b(x) * c(x)
    // a=[1,2,3,4], b=[1,1,1,1], c=[2,3,4,5]
    // sum = 1*1*2 + 2*1*3 + 3*1*4 + 4*1*5 = 2+6+12+20 = 40
    let a = vec![1u64, 2, 3, 4].into_iter().map(Fr::from_u64).collect();
    let b = vec![1u64, 1, 1, 1].into_iter().map(Fr::from_u64).collect();
    let c = vec![2u64, 3, 4, 5].into_iter().map(Fr::from_u64).collect();
    let claimed_sum = Fr::from_u64(40);

    let mut sc_transcript = Blake2bTranscript::new(b"integ-cubic");
    let (config, challenges, round_polys) =
        run_cubic_stage(a, b, c, claimed_sum, &mut sc_transcript);

    let acc = build_single_accumulator(&challenges, &round_polys, 3);
    assert_prove_and_verify(acc, &[config], &challenges, 42, "degree-3 pipeline");
}

#[test]
fn many_stages_accumulated() {
    let mut sc_transcript = Blake2bTranscript::new(b"integ-many");
    let mut all_configs = Vec::new();
    let mut all_challenges = Vec::new();
    let mut acc = BlindFoldAccumulator::<Fr, TestVC>::new();

    // Stage 0: 2 vars, a=[1,2], b=[3,4], sum=1*3+2*4=11
    let (cfg, ch, rp) = run_ip_stage(
        vec![Fr::from_u64(1), Fr::from_u64(2)],
        vec![Fr::from_u64(3), Fr::from_u64(4)],
        Fr::from_u64(11),
        &mut sc_transcript,
    );
    all_configs.push(cfg);
    all_challenges.extend_from_slice(&ch);
    acc.push_stage(CommittedRoundData {
        round_commitments: vec![Bn254G1::default(); rp.len()],
        poly_coeffs: rp.clone(),
        blinding_factors: vec![Fr::zero(); ch.len()],
        poly_degrees: vec![2; ch.len()],
        challenges: ch,
    });

    // Stage 1: 2 vars, a=[5,6,7,8], b=[1,1,1,1], sum=26
    let (cfg, ch, rp) = run_ip_stage(
        vec![
            Fr::from_u64(5),
            Fr::from_u64(6),
            Fr::from_u64(7),
            Fr::from_u64(8),
        ],
        vec![
            Fr::from_u64(1),
            Fr::from_u64(1),
            Fr::from_u64(1),
            Fr::from_u64(1),
        ],
        Fr::from_u64(26),
        &mut sc_transcript,
    );
    all_configs.push(cfg);
    all_challenges.extend_from_slice(&ch);
    acc.push_stage(CommittedRoundData {
        round_commitments: vec![Bn254G1::default(); rp.len()],
        poly_coeffs: rp.clone(),
        blinding_factors: vec![Fr::zero(); ch.len()],
        poly_degrees: vec![2; ch.len()],
        challenges: ch,
    });

    // Stage 2: 2 vars, a=[10,20], b=[2,3], sum=80
    let (cfg, ch, rp) = run_ip_stage(
        vec![Fr::from_u64(10), Fr::from_u64(20)],
        vec![Fr::from_u64(2), Fr::from_u64(3)],
        Fr::from_u64(80),
        &mut sc_transcript,
    );
    all_configs.push(cfg);
    all_challenges.extend_from_slice(&ch);
    acc.push_stage(CommittedRoundData {
        round_commitments: vec![Bn254G1::default(); rp.len()],
        poly_coeffs: rp.clone(),
        blinding_factors: vec![Fr::zero(); ch.len()],
        poly_degrees: vec![2; ch.len()],
        challenges: ch,
    });

    // Stage 3: 2 vars, a=[4,4,4,4], b=[1,2,3,4], sum=40
    let (cfg, ch, rp) = run_ip_stage(
        vec![
            Fr::from_u64(4),
            Fr::from_u64(4),
            Fr::from_u64(4),
            Fr::from_u64(4),
        ],
        vec![
            Fr::from_u64(1),
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(4),
        ],
        Fr::from_u64(40),
        &mut sc_transcript,
    );
    all_configs.push(cfg);
    all_challenges.extend_from_slice(&ch);
    acc.push_stage(CommittedRoundData {
        round_commitments: vec![Bn254G1::default(); rp.len()],
        poly_coeffs: rp.clone(),
        blinding_factors: vec![Fr::zero(); ch.len()],
        poly_degrees: vec![2; ch.len()],
        challenges: ch,
    });

    assert_prove_and_verify(acc, &all_configs, &all_challenges, 123, "4-stage pipeline");
}

#[test]
fn large_sumcheck_many_rounds() {
    // 8-variable sumcheck (256 evaluations, 8 rounds)
    let n = 256usize;
    let num_vars = 8;
    let a_vals: Vec<Fr> = (1..=n as u64).map(Fr::from_u64).collect();
    let b_vals: Vec<Fr> = vec![Fr::from_u64(1); n];
    let claimed_sum: Fr = a_vals.iter().zip(b_vals.iter()).map(|(&a, &b)| a * b).sum();

    let mut sc_transcript = Blake2bTranscript::new(b"integ-large");
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
        &mut sc_transcript,
        |c: u128| Fr::from_u128(c),
        handler,
    );

    let config = StageConfig {
        num_rounds: num_vars,
        degree: 2,
        claimed_sum,
    };
    let acc = build_single_accumulator(&challenges, &round_polys, 2);

    assert_prove_and_verify(acc, &[config], &challenges, 77, "8-round pipeline");
}

#[test]
fn proof_serialization_roundtrip() {
    let a = vec![
        Fr::from_u64(1),
        Fr::from_u64(2),
        Fr::from_u64(3),
        Fr::from_u64(4),
    ];
    let b = vec![
        Fr::from_u64(5),
        Fr::from_u64(6),
        Fr::from_u64(7),
        Fr::from_u64(8),
    ];
    let claimed_sum = Fr::from_u64(70);

    let mut sc_transcript = Blake2bTranscript::new(b"integ-serde");
    let (config, challenges, round_polys) =
        run_ip_stage(a, b, claimed_sum, &mut sc_transcript);

    let acc = build_single_accumulator(&challenges, &round_polys, 2);
    let proof = prove_and_verify(acc, &[config.clone()], &challenges, 42)
        .expect("initial prove should succeed");

    // Serialize → deserialize
    let json = serde_json::to_string(&proof).expect("serialization should succeed");
    let deserialized: BlindFoldProof<Fr, MockPCS> =
        serde_json::from_str(&json).expect("deserialization should succeed");

    // Verify deserialized proof
    let baked = BakedPublicInputs {
        challenges: challenges.clone(),
    };
    let mut verify_transcript = Blake2bTranscript::new(b"integ-test-prove");
    BlindFoldVerifier::verify::<MockPCS, _>(
        &deserialized,
        &[config],
        &baked,
        &(),
        &mut verify_transcript,
    )
    .expect("deserialized proof should verify");
}

#[test]
fn tampered_witness_commitment_rejected() {
    let a = vec![
        Fr::from_u64(1),
        Fr::from_u64(2),
        Fr::from_u64(3),
        Fr::from_u64(4),
    ];
    let b = vec![
        Fr::from_u64(5),
        Fr::from_u64(6),
        Fr::from_u64(7),
        Fr::from_u64(8),
    ];
    let claimed_sum = Fr::from_u64(70);

    let mut sc_transcript = Blake2bTranscript::new(b"integ-tamper-wcom");
    let (config, challenges, round_polys) =
        run_ip_stage(a, b, claimed_sum, &mut sc_transcript);

    let acc = build_single_accumulator(&challenges, &round_polys, 2);

    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let mut prove_transcript = Blake2bTranscript::new(b"integ-tamper-wcom-prove");
    let mut proof = BlindFoldProver::prove::<TestVC, MockPCS, _>(
        acc,
        &[config.clone()],
        &(),
        &mut prove_transcript,
        &mut rng,
    )
    .expect("prove should succeed");

    // Tamper the real witness commitment
    proof.real_w_commitment = MockPCS::commit(&[Fr::from_u64(999)], &()).0;

    let baked = BakedPublicInputs {
        challenges: challenges.clone(),
    };
    let mut verify_transcript = Blake2bTranscript::new(b"integ-tamper-wcom-prove");
    let result = BlindFoldVerifier::verify::<MockPCS, _>(
        &proof,
        &[config],
        &baked,
        &(),
        &mut verify_transcript,
    );
    assert!(
        result.is_err(),
        "tampered witness commitment should be rejected"
    );
}

#[test]
fn tampered_error_commitment_rejected() {
    let a = vec![
        Fr::from_u64(1),
        Fr::from_u64(2),
        Fr::from_u64(3),
        Fr::from_u64(4),
    ];
    let b = vec![
        Fr::from_u64(5),
        Fr::from_u64(6),
        Fr::from_u64(7),
        Fr::from_u64(8),
    ];
    let claimed_sum = Fr::from_u64(70);

    let mut sc_transcript = Blake2bTranscript::new(b"integ-tamper-ecom");
    let (config, challenges, round_polys) =
        run_ip_stage(a, b, claimed_sum, &mut sc_transcript);

    let acc = build_single_accumulator(&challenges, &round_polys, 2);

    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let mut prove_transcript = Blake2bTranscript::new(b"integ-tamper-ecom-prove");
    let mut proof = BlindFoldProver::prove::<TestVC, MockPCS, _>(
        acc,
        &[config.clone()],
        &(),
        &mut prove_transcript,
        &mut rng,
    )
    .expect("prove should succeed");

    // Tamper the real error commitment
    proof.real_e_commitment = MockPCS::commit(&[Fr::from_u64(888)], &()).0;

    let baked = BakedPublicInputs {
        challenges: challenges.clone(),
    };
    let mut verify_transcript = Blake2bTranscript::new(b"integ-tamper-ecom-prove");
    let result = BlindFoldVerifier::verify::<MockPCS, _>(
        &proof,
        &[config],
        &baked,
        &(),
        &mut verify_transcript,
    );
    assert!(
        result.is_err(),
        "tampered error commitment should be rejected"
    );
}

#[test]
fn wrong_baked_challenges_rejected() {
    let a = vec![
        Fr::from_u64(1),
        Fr::from_u64(2),
        Fr::from_u64(3),
        Fr::from_u64(4),
    ];
    let b = vec![
        Fr::from_u64(5),
        Fr::from_u64(6),
        Fr::from_u64(7),
        Fr::from_u64(8),
    ];
    let claimed_sum = Fr::from_u64(70);

    let mut sc_transcript = Blake2bTranscript::new(b"integ-wrong-baked");
    let (config, challenges, round_polys) =
        run_ip_stage(a, b, claimed_sum, &mut sc_transcript);

    let acc = build_single_accumulator(&challenges, &round_polys, 2);

    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let mut prove_transcript = Blake2bTranscript::new(b"integ-wrong-baked-prove");
    let proof = BlindFoldProver::prove::<TestVC, MockPCS, _>(
        acc,
        &[config.clone()],
        &(),
        &mut prove_transcript,
        &mut rng,
    )
    .expect("prove should succeed");

    // Verifier uses wrong challenges
    let mut wrong_challenges = challenges.clone();
    wrong_challenges[0] += Fr::from_u64(1);
    let baked = BakedPublicInputs {
        challenges: wrong_challenges,
    };

    let mut verify_transcript = Blake2bTranscript::new(b"integ-wrong-baked-prove");
    let result = BlindFoldVerifier::verify::<MockPCS, _>(
        &proof,
        &[config],
        &baked,
        &(),
        &mut verify_transcript,
    );
    assert!(
        result.is_err(),
        "wrong baked challenges should cause verification failure"
    );
}

#[test]
fn deterministic_across_rng_seeds() {
    // Same sumcheck data, same transcript prefix, different RNG seeds → both verify
    let a = vec![
        Fr::from_u64(1),
        Fr::from_u64(2),
        Fr::from_u64(3),
        Fr::from_u64(4),
    ];
    let b = vec![
        Fr::from_u64(5),
        Fr::from_u64(6),
        Fr::from_u64(7),
        Fr::from_u64(8),
    ];
    let claimed_sum = Fr::from_u64(70);

    for seed in [0u64, 1, 42, 999, u64::MAX] {
        let mut sc_transcript = Blake2bTranscript::new(b"integ-rng");
        let (config, challenges, round_polys) = run_ip_stage(
            a.clone(),
            b.clone(),
            claimed_sum,
            &mut sc_transcript,
        );
        let acc = build_single_accumulator(&challenges, &round_polys, 2);
        assert_prove_and_verify(
            acc,
            &[config],
            &challenges,
            seed,
            &format!("proof with seed {seed}"),
        );
    }
}

#[test]
fn relaxed_spartan_various_r1cs_shapes() {
    use jolt_blindfold::{check_relaxed_satisfaction, sample_random_witness};
    use jolt_blindfold::folding::{compute_cross_term, fold_scalar, fold_witnesses};

    // Shape 1: x * x = y (1 constraint, 3 variables)
    let r1cs_1 = SimpleR1CS::new(
        1,
        3,
        vec![(0, 1, Fr::from_u64(1))],
        vec![(0, 1, Fr::from_u64(1))],
        vec![(0, 2, Fr::from_u64(1))],
    );
    let z1 = vec![Fr::from_u64(1), Fr::from_u64(5), Fr::from_u64(25)];

    // Shape 2: two constraints — x*x=y, y*1=y
    let r1cs_2 = SimpleR1CS::new(
        2,
        3,
        vec![(0, 1, Fr::one()), (1, 2, Fr::one())],
        vec![(0, 1, Fr::one()), (1, 0, Fr::one())],
        vec![(0, 2, Fr::one()), (1, 2, Fr::one())],
    );
    let z2 = vec![Fr::from_u64(1), Fr::from_u64(4), Fr::from_u64(16)];

    // Shape 3: verifier R1CS from 3-round degree-2 sumcheck (6 constraints, many vars)
    let stages = vec![StageConfig {
        num_rounds: 3,
        degree: 2,
        claimed_sum: Fr::from_u64(44),
    }];
    let ch = vec![Fr::from_u64(11), Fr::from_u64(22), Fr::from_u64(33)];
    let baked = BakedPublicInputs {
        challenges: ch.clone(),
    };
    let r1cs_3 = jolt_blindfold::verifier_r1cs::build_verifier_r1cs(&stages, &baked);
    // Build valid witness
    let mut stage_coeffs = Vec::new();
    let mut running_sum = Fr::from_u64(44);
    for &r_i in &ch {
        let c0 = Fr::from_u64(5);
        let c2 = Fr::from_u64(1);
        let c1 = running_sum - Fr::from_u64(2) * c0 - c2;
        stage_coeffs.push(vec![c0, c1, c2]);
        running_sum = c0 + r_i * c1 + r_i * r_i * c2;
    }
    let z3 = jolt_blindfold::verifier_r1cs::assign_witness(&stages, &baked, &[stage_coeffs]);

    /// Folds a real witness with a random instance and runs relaxed Spartan prove+verify.
    fn fold_and_prove_relaxed(r1cs: &SimpleR1CS<Fr>, z_real: &[Fr], label: &str) {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let u_real = Fr::one();
        let w_real = RelaxedWitness {
            w: z_real.to_vec(),
            e: vec![Fr::zero(); r1cs.num_constraints()],
        };
        let (u_rand, w_rand) = sample_random_witness(r1cs, &mut rng);
        let cross = compute_cross_term(r1cs, z_real, u_real, &w_rand.w, u_rand);
        let r = Fr::from_u64(17);
        let folded = fold_witnesses(&w_real, &w_rand, &cross, r);
        let u_folded = fold_scalar(u_real, u_rand, r);

        check_relaxed_satisfaction(r1cs, u_folded, &folded)
            .unwrap_or_else(|i| panic!("{label}: folded not satisfied at constraint {i}"));

        let key = SpartanKey::from_r1cs(r1cs);
        let w_padded = pad(&folded.w, key.num_variables_padded);
        let e_padded = pad(&folded.e, key.num_constraints_padded);
        let (w_com, ()) = MockPCS::commit(&w_padded, &());
        let (e_com, ()) = MockPCS::commit(&e_padded, &());

        let mut pt = Blake2bTranscript::new(b"relaxed-shapes");
        let proof = SpartanProver::prove_relaxed::<MockPCS, _>(
            r1cs,
            &key,
            u_folded,
            &folded.w,
            &folded.e,
            &w_com,
            &e_com,
            &(),
            &mut pt,
        )
        .unwrap_or_else(|e| panic!("{label}: prove_relaxed failed: {e}"));

        let mut vt = Blake2bTranscript::new(b"relaxed-shapes");
        SpartanVerifier::verify_relaxed::<MockPCS, _>(
            &key,
            u_folded,
            &w_com,
            &e_com,
            &proof,
            &(),
            &mut vt,
        )
        .unwrap_or_else(|e| panic!("{label}: verify_relaxed failed: {e}"));
    }

    fold_and_prove_relaxed(&r1cs_1, &z1, "1-constraint");
    fold_and_prove_relaxed(&r1cs_2, &z2, "2-constraint");
    fold_and_prove_relaxed(&r1cs_3, &z3, "verifier-r1cs");
}

#[test]
fn mixed_degree_multi_stage() {
    let mut sc_transcript = Blake2bTranscript::new(b"integ-mixed");

    // Stage 0: degree-2, 2 rounds (4-element polys)
    let (cfg0, ch0, rp0) = run_ip_stage(
        vec![
            Fr::from_u64(1),
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(4),
        ],
        vec![
            Fr::from_u64(1),
            Fr::from_u64(1),
            Fr::from_u64(1),
            Fr::from_u64(1),
        ],
        Fr::from_u64(10),
        &mut sc_transcript,
    );

    // Stage 1: degree-3, 1 round (2-element polys)
    // a=[2,3], b=[1,1], c=[4,5] → sum = 2*1*4 + 3*1*5 = 23
    let (cfg1, ch1, rp1) = run_cubic_stage(
        vec![Fr::from_u64(2), Fr::from_u64(3)],
        vec![Fr::from_u64(1), Fr::from_u64(1)],
        vec![Fr::from_u64(4), Fr::from_u64(5)],
        Fr::from_u64(23),
        &mut sc_transcript,
    );

    let mut acc = BlindFoldAccumulator::<Fr, TestVC>::new();
    acc.push_stage(CommittedRoundData {
        round_commitments: vec![Bn254G1::default(); rp0.len()],
        poly_coeffs: rp0,
        blinding_factors: vec![Fr::zero(); ch0.len()],
        poly_degrees: vec![2; ch0.len()],
        challenges: ch0.clone(),
    });
    acc.push_stage(CommittedRoundData {
        round_commitments: vec![Bn254G1::default(); rp1.len()],
        poly_coeffs: rp1,
        blinding_factors: vec![Fr::zero(); ch1.len()],
        poly_degrees: vec![3; ch1.len()],
        challenges: ch1.clone(),
    });

    let stage_configs = vec![cfg0, cfg1];
    let mut all_challenges = ch0;
    all_challenges.extend_from_slice(&ch1);

    assert_prove_and_verify(acc, &stage_configs, &all_challenges, 88, "mixed-degree pipeline");
}
