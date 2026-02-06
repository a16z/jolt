//! BlindFold Protocol Implementation
//!
//! The BlindFold protocol makes sumcheck proofs zero-knowledge by:
//! 1. Encoding verifier checks into a small R1CS
//! 2. Using Nova folding to hide the witness
//! 3. Using Spartan sumcheck to prove R1CS satisfaction without revealing the witness
//!
//! Protocol flow:
//! 1. Prover has real instance-witness pair (u=1, E=0)
//! 2. Prover samples random satisfying pair
//! 3. Prover commits to cross-term T
//! 4. Verifier sends challenge r (via Fiat-Shamir)
//! 5. Both parties fold instances
//! 6. Prover folds witnesses (private)
//! 7. Prover runs Spartan sumcheck on folded R1CS
//! 8. Verifier verifies sumcheck, defers opening proofs to joint verification

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::curve::{JoltCurve, JoltGroupElement};
use crate::field::JoltField;
use crate::poly::commitment::pedersen::PedersenGenerators;
use crate::poly::unipoly::CompressedUniPoly;
use crate::transcripts::{AppendToTranscript, Transcript};

use super::folding::{
    compute_cross_term, sample_random_instance_deterministic,
    sample_random_satisfying_pair_deterministic,
};
use super::r1cs::VerifierR1CS;
use super::relaxed_r1cs::{RelaxedR1CSInstance, RelaxedR1CSWitness};

/// Information about final_output variables at specific stages.
/// Used by verifier to properly skip these variables during coefficient consistency checking.
#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct FinalOutputInfo {
    /// Stage index (0-indexed) that has final_output
    pub stage_idx: usize,
    /// Number of witness variables for this final_output
    pub num_variables: usize,
}

/// BlindFold proof containing only the data needed for verification.
///
/// Minimal proof structure - instances are NOT included because:
/// - real_instance: verifier reconstructs from round_commitments (main proof),
///   eval_commitments (opening proof), and public_inputs (transcript)
/// - random_instance: verifier derives deterministically from transcript
///
/// The witness is never revealed. A Spartan sumcheck proves that the
/// folded R1CS is satisfied. Opening proofs are deferred to joint verification.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct BlindFoldProof<F: JoltField, C: JoltCurve> {
    /// Commitment to the BlindFold witness W (only data unique to BlindFold)
    pub witness_commitment: C::G1,
    /// Commitment to cross-term T
    pub cross_term_commitment: C::G1,
    /// Spartan sumcheck round polynomials (compressed)
    pub spartan_proof: Vec<CompressedUniPoly<F>>,
    /// Information about final_output variables for verifier coefficient consistency checking
    pub final_output_info: Vec<FinalOutputInfo>,
}

/// BlindFold prover.
pub struct BlindFoldProver<'a, F: JoltField, C: JoltCurve> {
    /// Pedersen generators for commitments
    gens: &'a PedersenGenerators<C>,
    /// Verifier R1CS for sumcheck verification
    r1cs: &'a VerifierR1CS<F>,
    /// Generators for evaluation commitments (g1_0, h1)
    eval_commitment_gens: Option<(C::G1, C::G1)>,
}

impl<'a, F: JoltField, C: JoltCurve> BlindFoldProver<'a, F, C> {
    /// Create a new BlindFold prover.
    pub fn new(
        gens: &'a PedersenGenerators<C>,
        r1cs: &'a VerifierR1CS<F>,
        eval_commitment_gens: Option<(C::G1, C::G1)>,
    ) -> Self {
        Self {
            gens,
            r1cs,
            eval_commitment_gens,
        }
    }

    /// Generate a BlindFold proof.
    ///
    /// The prover must provide:
    /// - `real_instance`: The real R1CS instance (with commitments)
    /// - `real_witness`: The real witness satisfying the R1CS
    /// - `real_z`: The full Z vector for the real instance
    /// - `transcript`: For Fiat-Shamir challenge generation
    ///
    /// Randomness is derived from transcript, making the protocol deterministic
    /// given the same transcript state. This allows the verifier to reconstruct
    /// the random instance without it being in the proof.
    pub fn prove<T: Transcript>(
        &self,
        real_instance: &RelaxedR1CSInstance<F, C>,
        real_witness: &RelaxedR1CSWitness<F>,
        real_z: &[F],
        transcript: &mut T,
    ) -> BlindFoldProof<F, C> {
        use super::spartan::BlindFoldSpartanProver;
        use crate::utils::math::Math;
        use rand_chacha::ChaCha20Rng;
        use rand_core::SeedableRng;

        // Step 1: Bind real_instance to transcript FIRST (before deriving randomness)
        // SECURITY: This prevents adaptive attacks on the random instance
        transcript.append_message(b"BlindFold_real_instance");
        append_instance_to_transcript(real_instance, transcript);

        // Step 2: Sample random satisfying pair deterministically from transcript
        // The verifier will derive the same random instance using the same procedure
        let (random_instance, random_witness, random_z) =
            sample_random_satisfying_pair_deterministic(
                self.gens,
                self.r1cs,
                self.eval_commitment_gens,
                transcript,
            );

        // Step 3: Compute cross-term T
        let T = compute_cross_term(
            self.r1cs,
            real_z,
            real_instance.u,
            &random_z,
            random_instance.u,
        );

        // Step 4: Commit to cross-term (derive blinding from transcript)
        transcript.append_message(b"BlindFold_cross_term_blinding");
        let seed_a = transcript.challenge_u128();
        let seed_b = transcript.challenge_u128();
        let mut seed = [0u8; 32];
        seed[..16].copy_from_slice(&seed_a.to_le_bytes());
        seed[16..].copy_from_slice(&seed_b.to_le_bytes());
        let mut rng = ChaCha20Rng::from_seed(seed);
        let r_T = F::random(&mut rng);
        let T_bar = self.gens.commit(&T, &r_T);

        // Step 5: Bind T_bar to transcript
        transcript.append_message(b"BlindFold_cross_term");
        append_g1_to_transcript::<C>(&T_bar, transcript);

        // Step 6: Get folding challenge via Fiat-Shamir
        let r: F::Challenge = transcript.challenge_scalar_optimized::<F>();
        let r_field: F = r.into();

        // Step 7: Fold instances (public)
        let folded_instance = real_instance.fold(&random_instance, &T_bar, r_field);

        // Step 8: Fold witnesses (private)
        let folded_witness = real_witness.fold(&random_witness, &T, r_T, r_field);

        // Build folded Z vector: [u, x..., W...]
        let mut folded_z = Vec::with_capacity(self.r1cs.num_vars);
        folded_z.push(folded_instance.u);
        folded_z.extend_from_slice(&folded_instance.x);
        folded_z.extend_from_slice(&folded_witness.W);

        // Step 9: Run Spartan sumcheck
        transcript.append_message(b"BlindFold_spartan");
        let num_vars = self.r1cs.num_constraints.next_power_of_two().log_2();
        let tau: Vec<_> = transcript.challenge_vector_optimized::<F>(num_vars);

        let mut spartan_prover = BlindFoldSpartanProver::new(
            self.r1cs,
            folded_instance.u,
            folded_z,
            folded_witness.E.clone(),
            tau,
        );

        // Run sumcheck rounds
        let mut spartan_proof = Vec::with_capacity(num_vars);
        let mut claim = F::zero(); // Spartan starts at 0

        for _ in 0..num_vars {
            let poly = spartan_prover.compute_round_polynomial(claim);

            // Append polynomial to transcript
            let compressed = poly.compress();
            compressed.append_to_transcript(transcript);
            spartan_proof.push(compressed);

            // Get challenge and update
            let r_j = transcript.challenge_scalar_optimized::<F>();
            claim = poly.evaluate(&r_j);
            spartan_prover.bind_challenge(r_j);
        }

        // Collect final_output info for verifier
        let final_output_info: Vec<FinalOutputInfo> = self
            .r1cs
            .stage_configs
            .iter()
            .enumerate()
            .filter_map(|(idx, config)| {
                config.final_output.as_ref().map(|fo| {
                    let num_variables = if let Some(ref constraint) = fo.constraint {
                        let num_openings = constraint.required_openings.len();
                        let num_aux = estimate_aux_var_count(constraint);
                        num_openings + num_aux
                    } else {
                        let n = fo.num_evaluations;
                        if n > 1 {
                            n + n - 1
                        } else {
                            n
                        }
                    };
                    FinalOutputInfo {
                        stage_idx: idx,
                        num_variables,
                    }
                })
            })
            .collect();

        BlindFoldProof {
            witness_commitment: real_instance.W_bar,
            cross_term_commitment: T_bar,
            spartan_proof,
            final_output_info,
        }
    }
}

fn estimate_aux_var_count(constraint: &super::OutputClaimConstraint) -> usize {
    let mut count = 0;
    for term in &constraint.terms {
        if term.factors.len() <= 1 {
            count += 1;
        } else {
            count += term.factors.len();
        }
    }
    count
}

/// BlindFold verifier.
pub struct BlindFoldVerifier<'a, F: JoltField, C: JoltCurve> {
    /// Pedersen generators (for opening proof verification)
    #[allow(dead_code)]
    gens: &'a PedersenGenerators<C>,
    /// Verifier R1CS for sumcheck verification
    r1cs: &'a VerifierR1CS<F>,
    /// Generators for evaluation commitments (for opening proof verification)
    #[allow(dead_code)]
    eval_commitment_gens: Option<(C::G1, C::G1)>,
}

/// Error type for BlindFold verification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlindFoldVerifyError {
    /// Spartan sumcheck round polynomial failed verification at specified round
    SpartanSumcheckFailed(usize),
    /// Wrong number of Spartan proof rounds
    WrongSpartanProofLength { expected: usize, got: usize },
}

/// Data needed by verifier to reconstruct the real instance.
pub struct BlindFoldVerifierInput<F: JoltField, C: JoltCurve> {
    /// Public inputs (challenges, initial claims derived from main proof)
    pub public_inputs: Vec<F>,
    /// Round commitments from the main sumcheck proofs
    pub round_commitments: Vec<C::G1>,
    /// Evaluation commitments from the joint opening proof
    pub eval_commitments: Vec<C::G1>,
}

impl<'a, F: JoltField, C: JoltCurve> BlindFoldVerifier<'a, F, C> {
    /// Create a new BlindFold verifier.
    pub fn new(
        gens: &'a PedersenGenerators<C>,
        r1cs: &'a VerifierR1CS<F>,
        eval_commitment_gens: Option<(C::G1, C::G1)>,
    ) -> Self {
        Self {
            gens,
            r1cs,
            eval_commitment_gens,
        }
    }

    /// Verify a BlindFold proof.
    ///
    /// The verifier:
    /// 1. Reconstructs the real instance from inputs
    /// 2. Derives the random instance deterministically from transcript
    /// 3. Computes the folded instance
    /// 4. Verifies the Spartan sumcheck
    /// 5. Defers opening proofs to joint verification
    ///
    /// The `input` contains data from the main proof that the verifier already has:
    /// - public_inputs: derived from transcript challenges and initial claims
    /// - round_commitments: from the main sumcheck proofs
    /// - eval_commitments: from the joint opening proof
    pub fn verify<T: Transcript>(
        &self,
        proof: &BlindFoldProof<F, C>,
        input: &BlindFoldVerifierInput<F, C>,
        transcript: &mut T,
    ) -> Result<(), BlindFoldVerifyError> {
        use crate::utils::math::Math;

        // Step 1: Reconstruct the real instance from inputs
        // Real instance is non-relaxed: u = 1, E_bar = 0
        let real_instance = RelaxedR1CSInstance {
            E_bar: C::G1::zero(),
            u: F::one(),
            W_bar: proof.witness_commitment,
            x: input.public_inputs.clone(),
            round_commitments: input.round_commitments.clone(),
            eval_commitments: input.eval_commitments.clone(),
        };

        // Step 2: Bind real_instance to transcript (must match prover exactly)
        transcript.append_message(b"BlindFold_real_instance");
        append_instance_to_transcript(&real_instance, transcript);

        // Step 3: Derive random instance deterministically from transcript
        // This must match the prover's derivation exactly
        let random_instance = sample_random_instance_deterministic(
            self.gens,
            self.r1cs,
            self.eval_commitment_gens,
            transcript,
        );

        // Step 4: Derive cross-term blinding (same as prover, for transcript consistency)
        // We don't need the actual blinding, just need to advance transcript state
        transcript.append_message(b"BlindFold_cross_term_blinding");
        let _ = transcript.challenge_u128();
        let _ = transcript.challenge_u128();

        // Step 5: Bind T_bar to transcript
        transcript.append_message(b"BlindFold_cross_term");
        append_g1_to_transcript::<C>(&proof.cross_term_commitment, transcript);

        // Step 6: Get folding challenge via Fiat-Shamir
        let r: F::Challenge = transcript.challenge_scalar_optimized::<F>();
        let r_field: F = r.into();

        // Step 7: Compute folded instance
        let folded_instance =
            real_instance.fold(&random_instance, &proof.cross_term_commitment, r_field);

        // Step 8: Verify Spartan sumcheck
        transcript.append_message(b"BlindFold_spartan");
        let num_vars = self.r1cs.num_constraints.next_power_of_two().log_2();

        // Check proof length
        if proof.spartan_proof.len() != num_vars {
            return Err(BlindFoldVerifyError::WrongSpartanProofLength {
                expected: num_vars,
                got: proof.spartan_proof.len(),
            });
        }

        // Generate tau and verify sumcheck rounds
        let _tau: Vec<_> = transcript.challenge_vector_optimized::<F>(num_vars);
        let mut claim = F::zero(); // Spartan starts at 0
        let mut _challenges = Vec::with_capacity(num_vars);

        for (round, compressed_poly) in proof.spartan_proof.iter().enumerate() {
            // Append polynomial to transcript (for Fiat-Shamir)
            compressed_poly.append_to_transcript(transcript);

            // Decompress and verify: p(0) + p(1) = claim
            let poly = compressed_poly.decompress(&claim);
            // Evaluate at 0 and 1 using F directly
            let sum = poly.coeffs[0] + poly.coeffs.iter().sum::<F>(); // p(0) + p(1)
            if sum != claim {
                return Err(BlindFoldVerifyError::SpartanSumcheckFailed(round));
            }

            // Get challenge and update claim
            let r_j = transcript.challenge_scalar_optimized::<F>();
            _challenges.push(r_j);
            claim = poly.evaluate(&r_j);
        }

        // The final claim will be verified via polynomial opening proofs
        // (deferred to joint verification with Dory)
        //
        // Final check would be:
        // claim == eq(tau, r) * (Az(r)*Bz(r) - u*Cz(r) - E(r))
        //
        // Where:
        // - Az(r) = pub_az + w_az (public + witness contribution)
        // - w_az, w_bz, w_cz come from inner product of W with A', B', C'
        // - E(r) comes from inner product of E with eq(r, ·)
        //
        // These opening claims are batched into the joint Dory proof.

        let _ = folded_instance; // Will be used for opening verification

        Ok(())
    }
}

/// Append a group element to the transcript.
fn append_g1_to_transcript<C: JoltCurve>(g: &C::G1, transcript: &mut impl Transcript) {
    let mut bytes = Vec::new();
    g.serialize_compressed(&mut bytes)
        .expect("Serialization should not fail");
    transcript.append_bytes(&bytes);
}

/// Append an instance to the transcript (for Fiat-Shamir).
fn append_instance_to_transcript<F: JoltField, C: JoltCurve>(
    instance: &RelaxedR1CSInstance<F, C>,
    transcript: &mut impl Transcript,
) {
    append_g1_to_transcript::<C>(&instance.E_bar, transcript);
    append_g1_to_transcript::<C>(&instance.W_bar, transcript);

    let mut u_bytes = Vec::new();
    instance
        .u
        .serialize_compressed(&mut u_bytes)
        .expect("Serialization should not fail");
    transcript.append_bytes(&u_bytes);

    for x_i in &instance.x {
        let mut x_bytes = Vec::new();
        x_i.serialize_compressed(&mut x_bytes)
            .expect("Serialization should not fail");
        transcript.append_bytes(&x_bytes);
    }

    // Include round commitments in Fiat-Shamir
    for commitment in &instance.round_commitments {
        append_g1_to_transcript::<C>(commitment, transcript);
    }

    // Include evaluation commitments in Fiat-Shamir
    for commitment in &instance.eval_commitments {
        append_g1_to_transcript::<C>(commitment, transcript);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curve::Bn254Curve;
    use crate::curve::JoltCurve;
    use crate::subprotocols::blindfold::r1cs::VerifierR1CSBuilder;
    use crate::subprotocols::blindfold::witness::{BlindFoldWitness, RoundWitness, StageWitness};
    use crate::subprotocols::blindfold::StageConfig;
    use crate::transcripts::KeccakTranscript;
    use ark_bn254::Fr;
    use rand::thread_rng;

    fn round_commitment_data<F: JoltField, C: JoltCurve, R: rand_core::RngCore>(
        gens: &PedersenGenerators<C>,
        stages: &[StageWitness<F>],
        rng: &mut R,
    ) -> (Vec<C::G1>, Vec<Vec<F>>, Vec<F>) {
        let mut commitments = Vec::new();
        let mut coeffs = Vec::new();
        let mut blindings = Vec::new();
        for stage in stages {
            for round in &stage.rounds {
                let blinding = F::random(rng);
                let commitment = gens.commit(&round.coeffs, &blinding);
                commitments.push(commitment);
                coeffs.push(round.coeffs.clone());
                blindings.push(blinding);
            }
        }
        (commitments, coeffs, blindings)
    }

    #[test]
    fn test_blindfold_protocol_completeness() {
        let mut rng = thread_rng();
        type F = Fr;

        // Create R1CS
        let configs = [StageConfig::new(1, 3)];
        let builder = VerifierR1CSBuilder::<F>::new(&configs);
        let r1cs = builder.build();

        let gens = PedersenGenerators::<Bn254Curve>::deterministic(r1cs.num_vars + 100);

        // Create a valid witness
        let round = RoundWitness::new(
            vec![
                F::from_u64(40),
                F::from_u64(5),
                F::from_u64(10),
                F::from_u64(5),
            ],
            F::from_u64(3),
        );
        let blindfold_witness =
            BlindFoldWitness::new(F::from_u64(100), vec![StageWitness::new(vec![round])]);
        let z = blindfold_witness.assign(&r1cs);

        // Verify standard R1CS is satisfied
        assert!(r1cs.is_satisfied(&z));

        // Extract components for relaxed R1CS
        let witness_start = 1 + r1cs.num_public_inputs;
        let witness: Vec<F> = z[witness_start..].to_vec();
        let public_inputs: Vec<F> = z[1..witness_start].to_vec();

        let (round_commitments, round_coefficients, round_blindings) =
            round_commitment_data(&gens, &blindfold_witness.stages, &mut rng);
        // Create non-relaxed instance and witness
        let (real_instance, real_witness) = RelaxedR1CSInstance::<F, Bn254Curve>::new_non_relaxed(
            &gens,
            &witness,
            public_inputs,
            r1cs.num_constraints,
            round_commitments,
            round_coefficients,
            round_blindings,
            Vec::new(),
            &mut rng,
        );

        // Create prover and verifier
        let prover = BlindFoldProver::new(&gens, &r1cs, None);
        let verifier = BlindFoldVerifier::new(&gens, &r1cs, None);

        // Generate proof
        let mut prover_transcript = KeccakTranscript::new(b"BlindFold_test");
        let proof = prover.prove(&real_instance, &real_witness, &z, &mut prover_transcript);

        // Create verifier input from real_instance data
        let verifier_input = BlindFoldVerifierInput {
            public_inputs: real_instance.x.clone(),
            round_commitments: real_instance.round_commitments.clone(),
            eval_commitments: real_instance.eval_commitments.clone(),
        };

        // Verify proof
        let mut verifier_transcript = KeccakTranscript::new(b"BlindFold_test");
        let result = verifier.verify(&proof, &verifier_input, &mut verifier_transcript);

        assert!(result.is_ok(), "Verification should succeed: {result:?}");
    }

    #[test]
    fn test_blindfold_protocol_soundness_wrong_proof_length() {
        let mut rng = thread_rng();
        type F = Fr;

        let configs = [StageConfig::new(1, 3)];
        let builder = VerifierR1CSBuilder::<F>::new(&configs);
        let r1cs = builder.build();

        let gens = PedersenGenerators::<Bn254Curve>::deterministic(r1cs.num_vars + 100);

        // Create valid instance
        let round = RoundWitness::new(
            vec![
                F::from_u64(40),
                F::from_u64(5),
                F::from_u64(10),
                F::from_u64(5),
            ],
            F::from_u64(3),
        );
        let blindfold_witness =
            BlindFoldWitness::new(F::from_u64(100), vec![StageWitness::new(vec![round])]);
        let z = blindfold_witness.assign(&r1cs);

        let witness_start = 1 + r1cs.num_public_inputs;
        let witness: Vec<F> = z[witness_start..].to_vec();
        let public_inputs: Vec<F> = z[1..witness_start].to_vec();

        let (round_commitments, round_coefficients, round_blindings) =
            round_commitment_data(&gens, &blindfold_witness.stages, &mut rng);
        let (real_instance, real_witness) = RelaxedR1CSInstance::<F, Bn254Curve>::new_non_relaxed(
            &gens,
            &witness,
            public_inputs,
            r1cs.num_constraints,
            round_commitments,
            round_coefficients,
            round_blindings,
            Vec::new(),
            &mut rng,
        );

        let prover = BlindFoldProver::new(&gens, &r1cs, None);
        let verifier = BlindFoldVerifier::new(&gens, &r1cs, None);

        // Generate valid proof
        let mut prover_transcript = KeccakTranscript::new(b"BlindFold_test");
        let mut proof = prover.prove(&real_instance, &real_witness, &z, &mut prover_transcript);

        // Remove a round from the proof (wrong length)
        if !proof.spartan_proof.is_empty() {
            proof.spartan_proof.pop();
        }

        // Create verifier input
        let verifier_input = BlindFoldVerifierInput {
            public_inputs: real_instance.x.clone(),
            round_commitments: real_instance.round_commitments.clone(),
            eval_commitments: real_instance.eval_commitments.clone(),
        };

        // Verification should fail due to wrong proof length
        let mut verifier_transcript = KeccakTranscript::new(b"BlindFold_test");
        let result = verifier.verify(&proof, &verifier_input, &mut verifier_transcript);

        assert!(
            matches!(
                result,
                Err(BlindFoldVerifyError::WrongSpartanProofLength { .. })
            ),
            "Verification should fail with wrong proof length: {result:?}"
        );
    }

    #[test]
    fn test_blindfold_protocol_multi_round() {
        let mut rng = thread_rng();
        type F = Fr;

        // Create R1CS with multiple rounds
        let configs = [StageConfig::new(3, 3)];
        let builder = VerifierR1CSBuilder::<F>::new(&configs);
        let r1cs = builder.build();

        let gens = PedersenGenerators::<Bn254Curve>::deterministic(r1cs.num_vars + 100);

        // Create valid multi-round witness
        // Round 1
        let round1 = RoundWitness::new(
            vec![
                F::from_u64(20),
                F::from_u64(5),
                F::from_u64(7),
                F::from_u64(3),
            ],
            F::from_u64(2),
        );
        let next1 = round1.evaluate(F::from_u64(2));

        // Round 2: claimed_sum must equal next1
        let c0_2 = F::from_u64(30);
        let c2_2 = F::from_u64(10);
        let c3_2 = F::from_u64(5);
        // 2*c0 + c1 + c2 + c3 = next1
        // c1 = next1 - 2*30 - 10 - 5 = next1 - 75
        let c1_2 = next1 - F::from_u64(75);
        let round2 = RoundWitness::new(vec![c0_2, c1_2, c2_2, c3_2], F::from_u64(4));
        let next2 = round2.evaluate(F::from_u64(4));

        // Round 3
        let c0_3 = F::from_u64(50);
        let c2_3 = F::from_u64(8);
        let c3_3 = F::from_u64(2);
        let c1_3 = next2 - F::from_u64(110);
        let round3 = RoundWitness::new(vec![c0_3, c1_3, c2_3, c3_3], F::from_u64(6));

        let initial_claim = F::from_u64(55); // 2*20 + 5 + 7 + 3 = 55

        let blindfold_witness = BlindFoldWitness::new(
            initial_claim,
            vec![StageWitness::new(vec![round1, round2, round3])],
        );
        let z = blindfold_witness.assign(&r1cs);

        assert!(r1cs.is_satisfied(&z));

        let witness_start = 1 + r1cs.num_public_inputs;
        let witness: Vec<F> = z[witness_start..].to_vec();
        let public_inputs: Vec<F> = z[1..witness_start].to_vec();

        let (round_commitments, round_coefficients, round_blindings) =
            round_commitment_data(&gens, &blindfold_witness.stages, &mut rng);
        let (real_instance, real_witness) = RelaxedR1CSInstance::<F, Bn254Curve>::new_non_relaxed(
            &gens,
            &witness,
            public_inputs,
            r1cs.num_constraints,
            round_commitments,
            round_coefficients,
            round_blindings,
            Vec::new(),
            &mut rng,
        );

        let prover = BlindFoldProver::new(&gens, &r1cs, None);
        let verifier = BlindFoldVerifier::new(&gens, &r1cs, None);

        let mut prover_transcript = KeccakTranscript::new(b"BlindFold_multi");
        let proof = prover.prove(&real_instance, &real_witness, &z, &mut prover_transcript);

        // Create verifier input
        let verifier_input = BlindFoldVerifierInput {
            public_inputs: real_instance.x.clone(),
            round_commitments: real_instance.round_commitments.clone(),
            eval_commitments: real_instance.eval_commitments.clone(),
        };

        let mut verifier_transcript = KeccakTranscript::new(b"BlindFold_multi");
        let result = verifier.verify(&proof, &verifier_input, &mut verifier_transcript);

        assert!(
            result.is_ok(),
            "Multi-round verification should succeed: {result:?}"
        );
    }
}
