//! BlindFold protocol orchestrator: prover and verifier.
//!
//! Ties together verifier R1CS construction, Nova folding, and relaxed Spartan
//! to produce a ZK proof that all committed sumcheck stages were executed correctly.

use jolt_crypto::{HomomorphicCommitment, JoltCommitment};
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_spartan::{SpartanKey, SpartanProver, SpartanVerifier, R1CS};
use jolt_transcript::Transcript;
use num_traits::{One, Zero};
use rand_core::CryptoRngCore;

use crate::accumulator::BlindFoldAccumulator;
use crate::error::BlindFoldError;
use crate::folding::{
    compute_cross_term, fold_scalar, fold_witnesses, sample_random_witness, RelaxedWitness,
};
use crate::proof::BlindFoldProof;
use crate::verifier_r1cs::{assign_witness, build_verifier_r1cs, BakedPublicInputs, StageConfig};

/// Pads a slice to `target_len` with zeros.
fn pad_to<F: Field>(data: &[F], target_len: usize) -> Vec<F> {
    let mut v = vec![F::zero(); target_len];
    let copy_len = data.len().min(target_len);
    v[..copy_len].copy_from_slice(&data[..copy_len]);
    v
}

/// Stateless BlindFold prover.
pub struct BlindFoldProver;

impl BlindFoldProver {
    /// Produces a BlindFold proof from accumulated committed sumcheck data.
    ///
    /// # Arguments
    ///
    /// * `accumulator` — Collected round data from all committed sumcheck stages.
    /// * `stage_configs` — Stage structure (rounds, degrees, claimed sums).
    /// * `pcs_setup` — PCS prover setup for witness/error commitments and openings.
    /// * `transcript` — Fiat-Shamir transcript (shared with verifier).
    /// * `rng` — Cryptographic RNG for sampling the random masking instance.
    #[allow(clippy::type_complexity)]
    pub fn prove<VC, PCS, T>(
        accumulator: BlindFoldAccumulator<PCS::Field, VC>,
        stage_configs: &[StageConfig<PCS::Field>],
        pcs_setup: &PCS::ProverSetup,
        transcript: &mut T,
        rng: &mut impl CryptoRngCore,
    ) -> Result<BlindFoldProof<PCS::Field, PCS>, BlindFoldError>
    where
        VC: JoltCommitment,
        PCS: CommitmentScheme,
        PCS::Output: HomomorphicCommitment<PCS::Field>,
        T: Transcript<Challenge = u128>,
    {
        let stages = accumulator.into_stages();
        if stages.len() != stage_configs.len() {
            return Err(BlindFoldError::StageCountMismatch {
                expected: stage_configs.len(),
                actual: stages.len(),
            });
        }

        // Extract challenges and coefficients from accumulated stage data
        let mut all_challenges = Vec::new();
        let mut stage_coefficients = Vec::with_capacity(stages.len());

        for stage_data in &stages {
            all_challenges.extend_from_slice(&stage_data.round_data.challenges);
            stage_coefficients.push(stage_data.round_data.poly_coeffs.clone());
        }

        let baked = BakedPublicInputs {
            challenges: all_challenges,
        };

        // Build verifier R1CS and assign witness
        let r1cs = build_verifier_r1cs(stage_configs, &baked);
        let z_real = assign_witness(stage_configs, &baked, &stage_coefficients);

        let key = SpartanKey::from_r1cs(&r1cs);

        // Spartan internally pads to power-of-two, so commitments must match
        let w_padded_len = key.num_variables_padded;
        let e_padded_len = key.num_constraints_padded;

        // Commit to the real witness and error (zero error for satisfying instance)
        let real_w_padded = pad_to(&z_real, w_padded_len);
        let (real_w_commitment, _) = PCS::commit(&real_w_padded, pcs_setup);
        let zero_error = vec![PCS::Field::zero(); r1cs.num_constraints()];
        let zero_e_padded = pad_to(&zero_error, e_padded_len);
        let (real_e_commitment, _) = PCS::commit(&zero_e_padded, pcs_setup);

        // Sample random satisfying relaxed instance for masking
        let (u_rand, w_rand) = sample_random_witness(&r1cs, rng);

        let rand_w_padded = pad_to(&w_rand.w, w_padded_len);
        let (random_w_commitment, _) = PCS::commit(&rand_w_padded, pcs_setup);
        let rand_e_padded = pad_to(&w_rand.e, e_padded_len);
        let (random_e_commitment, _) = PCS::commit(&rand_e_padded, pcs_setup);

        // Compute cross-term and commit
        let u_real = PCS::Field::one();
        let cross_term = compute_cross_term(&r1cs, &z_real, u_real, &w_rand.w, u_rand);
        let cross_t_padded = pad_to(&cross_term, e_padded_len);
        let (cross_term_commitment, _) = PCS::commit(&cross_t_padded, pcs_setup);

        // Append all commitments and random_u to transcript
        transcript.append_bytes(format!("{real_w_commitment:?}").as_bytes());
        transcript.append_bytes(format!("{real_e_commitment:?}").as_bytes());
        transcript.append(&u_rand);
        transcript.append_bytes(format!("{random_w_commitment:?}").as_bytes());
        transcript.append_bytes(format!("{random_e_commitment:?}").as_bytes());
        transcript.append_bytes(format!("{cross_term_commitment:?}").as_bytes());

        // Sample folding challenge
        let r = PCS::Field::from_u128(transcript.challenge());

        // Fold witnesses
        let w_real = RelaxedWitness {
            w: z_real,
            e: zero_error,
        };
        let folded_wit = fold_witnesses(&w_real, &w_rand, &cross_term, r);
        let u_folded = fold_scalar(u_real, u_rand, r);

        // Fold instance commitments via HomomorphicCommitment
        let w_com_folded =
            PCS::Output::linear_combine(&real_w_commitment, &random_w_commitment, &r);
        let r2 = r * r;
        let e_intermediate =
            PCS::Output::linear_combine(&real_e_commitment, &cross_term_commitment, &r);
        let e_com_folded = PCS::Output::linear_combine(&e_intermediate, &random_e_commitment, &r2);

        // Produce relaxed Spartan proof
        let spartan_proof = SpartanProver::prove_relaxed::<PCS, T>(
            &r1cs,
            &key,
            u_folded,
            &folded_wit.w,
            &folded_wit.e,
            &w_com_folded,
            &e_com_folded,
            pcs_setup,
            transcript,
        )?;

        Ok(BlindFoldProof {
            real_w_commitment,
            real_e_commitment,
            random_u: u_rand,
            random_w_commitment,
            random_e_commitment,
            cross_term_commitment,
            spartan_proof,
        })
    }
}

/// Stateless BlindFold verifier.
pub struct BlindFoldVerifier;

impl BlindFoldVerifier {
    /// Verifies a BlindFold proof.
    ///
    /// Reconstructs the verifier R1CS and Spartan key, replays the folding
    /// challenge derivation, folds the instance commitments, and delegates
    /// to the relaxed Spartan verifier.
    pub fn verify<PCS, T>(
        proof: &BlindFoldProof<PCS::Field, PCS>,
        stage_configs: &[StageConfig<PCS::Field>],
        baked: &BakedPublicInputs<PCS::Field>,
        pcs_verifier_setup: &PCS::VerifierSetup,
        transcript: &mut T,
    ) -> Result<(), BlindFoldError>
    where
        PCS: CommitmentScheme,
        PCS::Output: HomomorphicCommitment<PCS::Field>,
        T: Transcript<Challenge = u128>,
    {
        let r1cs = build_verifier_r1cs(stage_configs, baked);
        let key = SpartanKey::from_r1cs(&r1cs);

        // Replay commitment absorption (same order as prover)
        transcript.append_bytes(format!("{:?}", proof.real_w_commitment).as_bytes());
        transcript.append_bytes(format!("{:?}", proof.real_e_commitment).as_bytes());
        transcript.append(&proof.random_u);
        transcript.append_bytes(format!("{:?}", proof.random_w_commitment).as_bytes());
        transcript.append_bytes(format!("{:?}", proof.random_e_commitment).as_bytes());
        transcript.append_bytes(format!("{:?}", proof.cross_term_commitment).as_bytes());

        // Derive folding challenge
        let r = PCS::Field::from_u128(transcript.challenge());

        // Fold instance commitments
        let u_folded = fold_scalar(PCS::Field::one(), proof.random_u, r);
        let w_com_folded =
            PCS::Output::linear_combine(&proof.real_w_commitment, &proof.random_w_commitment, &r);
        let r2 = r * r;
        let e_intermediate =
            PCS::Output::linear_combine(&proof.real_e_commitment, &proof.cross_term_commitment, &r);
        let e_com_folded =
            PCS::Output::linear_combine(&e_intermediate, &proof.random_e_commitment, &r2);

        // Verify relaxed Spartan proof
        SpartanVerifier::verify_relaxed::<PCS, T>(
            &key,
            u_folded,
            &w_com_folded,
            &e_com_folded,
            &proof.spartan_proof,
            pcs_verifier_setup,
            transcript,
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accumulator::BlindFoldAccumulator;
    use crate::proof::CommittedRoundData;
    use jolt_crypto::arkworks::bn254::Bn254G1;
    use jolt_crypto::Pedersen;
    use jolt_field::{Field, Fr};
    use jolt_openings::mock::MockCommitmentScheme;
    use jolt_poly::{Polynomial, UnivariatePoly};
    use jolt_sumcheck::{RoundHandler, SumcheckClaim, SumcheckCompute, SumcheckProver};
    use jolt_transcript::Blake2bTranscript;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    type TestVC = Pedersen<Bn254G1>;
    type MockPCS = MockCommitmentScheme<Fr>;

    /// Inner-product sumcheck witness: g(x) = a(x) * b(x)
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
            let points: Vec<(Fr, Fr)> =
                (0..3).map(|t| (Fr::from_u64(t as u64), evals[t])).collect();
            UnivariatePoly::interpolate(&points)
        }

        fn bind(&mut self, challenge: Fr) {
            self.a.bind(challenge);
            self.b.bind(challenge);
        }
    }

    /// Handler that records challenges and round polynomials for test extraction.
    struct RecordingHandler {
        inner: jolt_sumcheck::ClearRoundHandler<Fr>,
        challenges: Vec<Fr>,
        round_polys: Vec<Vec<Fr>>,
    }

    impl RecordingHandler {
        fn new(cap: usize) -> Self {
            Self {
                inner: jolt_sumcheck::ClearRoundHandler::with_capacity(cap),
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

        fn absorb_round_poly(
            &mut self,
            poly: &UnivariatePoly<Fr>,
            transcript: &mut impl Transcript,
        ) {
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

    /// Runs a single sumcheck stage and returns stage config, challenges, and coefficients.
    fn run_sumcheck_stage(
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

    /// Builds a BlindFoldAccumulator from recorded round data.
    fn build_accumulator(
        challenges: &[Fr],
        round_polys: &[Vec<Fr>],
        degree: usize,
    ) -> BlindFoldAccumulator<Fr, TestVC> {
        let mut acc = BlindFoldAccumulator::new();
        let round_data = CommittedRoundData {
            round_commitments: vec![Bn254G1::default(); round_polys.len()],
            poly_coeffs: round_polys.to_vec(),
            blinding_factors: vec![Fr::zero(); round_polys.len()],
            poly_degrees: vec![degree; round_polys.len()],
            challenges: challenges.to_vec(),
        };
        acc.push_stage(round_data);
        acc
    }

    #[test]
    fn single_stage_e2e() {
        // a = [1, 2, 3, 4], b = [5, 6, 7, 8], sum = 70
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

        let mut prover_transcript = Blake2bTranscript::new(b"blindfold-e2e");
        let (config, challenges, round_polys) =
            run_sumcheck_stage(a, b, claimed_sum, &mut prover_transcript);

        let acc = build_accumulator(&challenges, &round_polys, 2);

        let stage_configs = vec![config];

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let mut prove_transcript = Blake2bTranscript::new(b"blindfold-prove");
        let proof = BlindFoldProver::prove::<TestVC, MockPCS, _>(
            acc,
            &stage_configs,
            &(),
            &mut prove_transcript,
            &mut rng,
        )
        .expect("BlindFold prove should succeed");

        let baked = BakedPublicInputs {
            challenges: challenges.clone(),
        };
        let mut verify_transcript = Blake2bTranscript::new(b"blindfold-prove");
        BlindFoldVerifier::verify::<MockPCS, _>(
            &proof,
            &stage_configs,
            &baked,
            &(),
            &mut verify_transcript,
        )
        .expect("BlindFold verify should succeed");
    }

    #[test]
    fn multi_stage_e2e() {
        // Stage 0: a=[1,2,3,4], b=[1,1,1,1], sum=10
        // Stage 1: a=[2,3], b=[4,5], sum=23
        let mut prover_transcript = Blake2bTranscript::new(b"blindfold-multi");

        let (config0, challenges0, polys0) = run_sumcheck_stage(
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
            &mut prover_transcript,
        );

        let (config1, challenges1, polys1) = run_sumcheck_stage(
            vec![Fr::from_u64(2), Fr::from_u64(3)],
            vec![Fr::from_u64(4), Fr::from_u64(5)],
            Fr::from_u64(23),
            &mut prover_transcript,
        );

        // Build accumulator with two stages
        let mut acc = BlindFoldAccumulator::new();
        acc.push_stage(CommittedRoundData {
            round_commitments: vec![Bn254G1::default(); polys0.len()],
            poly_coeffs: polys0,
            blinding_factors: vec![Fr::zero(); challenges0.len()],
            poly_degrees: vec![2; challenges0.len()],
            challenges: challenges0.clone(),
        });
        acc.push_stage(CommittedRoundData {
            round_commitments: vec![Bn254G1::default(); polys1.len()],
            poly_coeffs: polys1,
            blinding_factors: vec![Fr::zero(); challenges1.len()],
            poly_degrees: vec![2; challenges1.len()],
            challenges: challenges1.clone(),
        });

        let stage_configs = vec![config0, config1];

        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let mut prove_transcript = Blake2bTranscript::new(b"blindfold-multi-prove");
        let proof = BlindFoldProver::prove::<TestVC, MockPCS, _>(
            acc,
            &stage_configs,
            &(),
            &mut prove_transcript,
            &mut rng,
        )
        .expect("BlindFold multi-stage prove should succeed");

        let mut all_challenges = challenges0;
        all_challenges.extend_from_slice(&challenges1);
        let baked = BakedPublicInputs {
            challenges: all_challenges,
        };
        let mut verify_transcript = Blake2bTranscript::new(b"blindfold-multi-prove");
        BlindFoldVerifier::verify::<MockPCS, _>(
            &proof,
            &stage_configs,
            &baked,
            &(),
            &mut verify_transcript,
        )
        .expect("BlindFold multi-stage verify should succeed");
    }

    #[test]
    fn tampered_cross_term_rejected() {
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

        let mut prover_transcript = Blake2bTranscript::new(b"blindfold-tamper-ct");
        let (config, challenges, round_polys) =
            run_sumcheck_stage(a, b, claimed_sum, &mut prover_transcript);

        let acc = build_accumulator(&challenges, &round_polys, 2);
        let stage_configs = vec![config];

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let mut prove_transcript = Blake2bTranscript::new(b"blindfold-tamper-ct-prove");
        let mut proof = BlindFoldProver::prove::<TestVC, MockPCS, _>(
            acc,
            &stage_configs,
            &(),
            &mut prove_transcript,
            &mut rng,
        )
        .expect("BlindFold prove should succeed");

        // Tamper cross-term commitment
        proof.cross_term_commitment = MockPCS::commit(&[Fr::from_u64(999)], &()).0;

        let baked = BakedPublicInputs {
            challenges: challenges.clone(),
        };
        let mut verify_transcript = Blake2bTranscript::new(b"blindfold-tamper-ct-prove");
        let result = BlindFoldVerifier::verify::<MockPCS, _>(
            &proof,
            &stage_configs,
            &baked,
            &(),
            &mut verify_transcript,
        );
        assert!(
            result.is_err(),
            "tampered cross-term commitment should be rejected"
        );
    }

    #[test]
    fn tampered_random_u_rejected() {
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

        let mut prover_transcript = Blake2bTranscript::new(b"blindfold-tamper-u");
        let (config, challenges, round_polys) =
            run_sumcheck_stage(a, b, claimed_sum, &mut prover_transcript);

        let acc = build_accumulator(&challenges, &round_polys, 2);
        let stage_configs = vec![config];

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let mut prove_transcript = Blake2bTranscript::new(b"blindfold-tamper-u-prove");
        let mut proof = BlindFoldProver::prove::<TestVC, MockPCS, _>(
            acc,
            &stage_configs,
            &(),
            &mut prove_transcript,
            &mut rng,
        )
        .expect("BlindFold prove should succeed");

        // Tamper random_u
        proof.random_u += Fr::from_u64(1);

        let baked = BakedPublicInputs {
            challenges: challenges.clone(),
        };
        let mut verify_transcript = Blake2bTranscript::new(b"blindfold-tamper-u-prove");
        let result = BlindFoldVerifier::verify::<MockPCS, _>(
            &proof,
            &stage_configs,
            &baked,
            &(),
            &mut verify_transcript,
        );
        assert!(result.is_err(), "tampered random_u should be rejected");
    }

    #[test]
    fn stage_count_mismatch_rejected() {
        let acc = BlindFoldAccumulator::<Fr, TestVC>::new();
        let stage_configs = vec![StageConfig {
            num_rounds: 2,
            degree: 2,
            claimed_sum: Fr::from_u64(10),
        }];

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let mut transcript = Blake2bTranscript::new(b"blindfold-mismatch");
        let result = BlindFoldProver::prove::<TestVC, MockPCS, _>(
            acc,
            &stage_configs,
            &(),
            &mut transcript,
            &mut rng,
        );

        assert!(
            matches!(
                result,
                Err(BlindFoldError::StageCountMismatch {
                    expected: 1,
                    actual: 0
                })
            ),
            "stage count mismatch should be detected"
        );
    }

    /// Relaxed Spartan test with folded verifier R1CS instance (from jolt-blindfold folding).
    #[test]
    fn prove_relaxed_folded_verifier_r1cs() {
        use crate::folding::{check_relaxed_satisfaction, sample_random_witness};
        use jolt_openings::CommitmentScheme;
        use jolt_spartan::{SpartanKey, SpartanProver, SpartanVerifier, R1CS};

        // Build a verifier R1CS from a sumcheck stage
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

        let mut sumcheck_transcript = Blake2bTranscript::new(b"relaxed-folded");
        let (config, challenges, round_polys) =
            run_sumcheck_stage(a, b, claimed_sum, &mut sumcheck_transcript);

        let baked = BakedPublicInputs {
            challenges: challenges.clone(),
        };
        let r1cs = build_verifier_r1cs(&[config], &baked);
        let z_real = assign_witness(
            &[StageConfig {
                num_rounds: 2,
                degree: 2,
                claimed_sum,
            }],
            &baked,
            &[round_polys],
        );

        let key = SpartanKey::from_r1cs(&r1cs);

        // Fold with random
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let u_real = Fr::one();
        let w_real = RelaxedWitness {
            w: z_real.clone(),
            e: vec![Fr::zero(); r1cs.num_constraints()],
        };
        let (u_rand, w_rand) = sample_random_witness(&r1cs, &mut rng);
        let cross_term = compute_cross_term(&r1cs, &z_real, u_real, &w_rand.w, u_rand);
        let r = Fr::from_u64(17);
        let folded = fold_witnesses(&w_real, &w_rand, &cross_term, r);
        let u_folded = fold_scalar(u_real, u_rand, r);

        check_relaxed_satisfaction(&r1cs, u_folded, &folded).expect("folded instance must satisfy");

        // Pad and commit
        let w_padded = {
            let mut v = vec![Fr::zero(); key.num_variables_padded];
            v[..folded.w.len()].copy_from_slice(&folded.w);
            v
        };
        let e_padded = {
            let mut v = vec![Fr::zero(); key.num_constraints_padded];
            v[..folded.e.len()].copy_from_slice(&folded.e);
            v
        };
        let (w_com, ()) = MockPCS::commit(&w_padded, &());
        let (e_com, ()) = MockPCS::commit(&e_padded, &());

        let mut transcript = Blake2bTranscript::new(b"relaxed-folded-prove");
        let proof = SpartanProver::prove_relaxed::<MockPCS, _>(
            &r1cs,
            &key,
            u_folded,
            &folded.w,
            &folded.e,
            &w_com,
            &e_com,
            &(),
            &mut transcript,
        )
        .expect("relaxed proving should succeed");

        let mut vt = Blake2bTranscript::new(b"relaxed-folded-prove");
        SpartanVerifier::verify_relaxed::<MockPCS, _>(
            &key,
            u_folded,
            &w_com,
            &e_com,
            &proof,
            &(),
            &mut vt,
        )
        .expect("relaxed verification should succeed");
    }
}
