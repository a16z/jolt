//! BlindFold Protocol Implementation
//!
//! The BlindFold protocol makes sumcheck proofs zero-knowledge by:
//! 1. Encoding verifier checks into a small R1CS
//! 2. Using Nova folding to hide the witness
//!
//! Protocol flow:
//! 1. Prover has real instance-witness pair (u=1, E=0)
//! 2. Prover samples random satisfying pair
//! 3. Prover commits to cross-term T
//! 4. Verifier sends challenge r (via Fiat-Shamir)
//! 5. Both parties fold instances
//! 6. Prover folds witnesses
//! 7. Prover sends folded witness
//! 8. Verifier checks folded witness satisfies folded instance

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand_core::CryptoRngCore;

use crate::curve::JoltCurve;
use crate::field::JoltField;
use crate::poly::commitment::pedersen::PedersenGenerators;
use crate::transcripts::Transcript;

use super::folding::{compute_cross_term, sample_random_satisfying_pair};
use super::r1cs::VerifierR1CS;
use super::relaxed_r1cs::{RelaxedR1CSInstance, RelaxedR1CSWitness};

/// BlindFold proof containing the data needed for verification.
///
/// The proof reveals the folded witness, but this is zero-knowledge because
/// the folded witness = real_witness + r * random_witness is masked by
/// the uniformly random random_witness.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct BlindFoldProof<F: JoltField, C: JoltCurve> {
    /// Real instance (commitments to the actual sumcheck witness)
    pub real_instance: RelaxedR1CSInstance<F, C>,
    /// Random instance (commitments only, no private data)
    pub random_instance: RelaxedR1CSInstance<F, C>,
    /// Commitment to cross-term T
    pub cross_term_commitment: C::G1,
    /// Folded witness (masked by random witness)
    pub folded_witness: RelaxedR1CSWitness<F>,
}

/// BlindFold prover.
pub struct BlindFoldProver<'a, F: JoltField, C: JoltCurve> {
    /// Pedersen generators for commitments
    gens: &'a PedersenGenerators<C>,
    /// Verifier R1CS for sumcheck verification
    r1cs: &'a VerifierR1CS<F>,
}

impl<'a, F: JoltField, C: JoltCurve> BlindFoldProver<'a, F, C> {
    /// Create a new BlindFold prover.
    pub fn new(gens: &'a PedersenGenerators<C>, r1cs: &'a VerifierR1CS<F>) -> Self {
        Self { gens, r1cs }
    }

    /// Generate a BlindFold proof.
    ///
    /// The prover must provide:
    /// - `real_instance`: The real R1CS instance (with commitments)
    /// - `real_witness`: The real witness satisfying the R1CS
    /// - `real_z`: The full Z vector for the real instance
    /// - `transcript`: For Fiat-Shamir challenge generation
    /// - `rng`: Random number generator
    pub fn prove<T: Transcript, R: CryptoRngCore>(
        &self,
        real_instance: &RelaxedR1CSInstance<F, C>,
        real_witness: &RelaxedR1CSWitness<F>,
        real_z: &[F],
        transcript: &mut T,
        rng: &mut R,
    ) -> BlindFoldProof<F, C> {
        // Step 1: Sample random satisfying pair
        let (random_instance, random_witness, random_z) =
            sample_random_satisfying_pair(self.gens, self.r1cs, rng);

        // Step 2: Compute cross-term T
        let T = compute_cross_term(
            self.r1cs,
            real_z,
            real_instance.u,
            &random_z,
            random_instance.u,
        );

        // Step 3: Commit to cross-term
        let r_T = F::random(rng);
        let T_bar = self.gens.commit(&T, &r_T);

        // Step 4: Append data to transcript for Fiat-Shamir
        // SECURITY: real_instance must be bound to prevent adaptive attacks
        transcript.append_message(b"BlindFold_real_instance");
        append_instance_to_transcript(real_instance, transcript);

        transcript.append_message(b"BlindFold_random_instance");
        append_instance_to_transcript(&random_instance, transcript);

        transcript.append_message(b"BlindFold_cross_term");
        append_g1_to_transcript::<C>(&T_bar, transcript);

        // Step 5: Get challenge via Fiat-Shamir
        let r: F::Challenge = transcript.challenge_scalar_optimized::<F>();
        let r_field: F = r.into();

        // Step 6: Fold instances
        let folded_instance = real_instance.fold(&random_instance, &T_bar, r_field);

        // Step 7: Fold witnesses
        let folded_witness = real_witness.fold(&random_witness, &T, r_T, r_field);

        // Verify commitment consistency before returning proof
        assert!(
            self.gens.verify(
                &folded_instance.W_bar,
                &folded_witness.W,
                &folded_witness.r_W
            ),
            "Internal error: W commitment mismatch"
        );
        assert!(
            self.gens.verify(
                &folded_instance.E_bar,
                &folded_witness.E,
                &folded_witness.r_E
            ),
            "Internal error: E commitment mismatch"
        );

        BlindFoldProof {
            real_instance: real_instance.clone(),
            random_instance,
            cross_term_commitment: T_bar,
            folded_witness,
        }
    }
}

/// BlindFold verifier.
pub struct BlindFoldVerifier<'a, F: JoltField, C: JoltCurve> {
    /// Pedersen generators for commitment verification
    gens: &'a PedersenGenerators<C>,
    /// Verifier R1CS for sumcheck verification
    r1cs: &'a VerifierR1CS<F>,
}

/// Error type for BlindFold verification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlindFoldVerifyError {
    /// E commitment opening failed
    ECommitmentMismatch,
    /// W commitment opening failed
    WCommitmentMismatch,
    /// R1CS constraint not satisfied
    R1CSConstraintFailed(usize),
}

impl<'a, F: JoltField, C: JoltCurve> BlindFoldVerifier<'a, F, C> {
    /// Create a new BlindFold verifier.
    pub fn new(gens: &'a PedersenGenerators<C>, r1cs: &'a VerifierR1CS<F>) -> Self {
        Self { gens, r1cs }
    }

    /// Verify a BlindFold proof.
    ///
    /// The verifier:
    /// 1. Recomputes the folded instance from public data
    /// 2. Verifies commitment openings
    /// 3. Verifies R1CS satisfaction
    pub fn verify<T: Transcript>(
        &self,
        proof: &BlindFoldProof<F, C>,
        transcript: &mut T,
    ) -> Result<(), BlindFoldVerifyError> {
        // Step 1: Replay transcript to get challenge
        // SECURITY: real_instance must be bound to prevent adaptive attacks
        transcript.append_message(b"BlindFold_real_instance");
        append_instance_to_transcript(&proof.real_instance, transcript);

        transcript.append_message(b"BlindFold_random_instance");
        append_instance_to_transcript(&proof.random_instance, transcript);

        transcript.append_message(b"BlindFold_cross_term");
        append_g1_to_transcript::<C>(&proof.cross_term_commitment, transcript);

        let r: F::Challenge = transcript.challenge_scalar_optimized::<F>();
        let r_field: F = r.into();

        // Step 2: Recompute folded instance using real_instance from proof
        let folded_instance = proof.real_instance.fold(
            &proof.random_instance,
            &proof.cross_term_commitment,
            r_field,
        );

        // Step 3: Verify commitment openings
        // Check W̄_folded = Com(W_folded, r_W_folded)
        if !self.gens.verify(
            &folded_instance.W_bar,
            &proof.folded_witness.W,
            &proof.folded_witness.r_W,
        ) {
            return Err(BlindFoldVerifyError::WCommitmentMismatch);
        }

        // Check Ē_folded = Com(E_folded, r_E_folded)
        if !self.gens.verify(
            &folded_instance.E_bar,
            &proof.folded_witness.E,
            &proof.folded_witness.r_E,
        ) {
            return Err(BlindFoldVerifyError::ECommitmentMismatch);
        }

        // Step 4: Verify R1CS satisfaction
        // Check: (A·Z') ∘ (B·Z') = u_folded*(C·Z') + E_folded
        let result = proof.folded_witness.check_satisfaction(
            self.r1cs,
            folded_instance.u,
            &folded_instance.x,
        );

        result.map_err(BlindFoldVerifyError::R1CSConstraintFailed)
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curve::Bn254Curve;
    use crate::subprotocols::blindfold::r1cs::VerifierR1CSBuilder;
    use crate::subprotocols::blindfold::witness::{BlindFoldWitness, RoundWitness, StageWitness};
    use crate::subprotocols::blindfold::StageConfig;
    use crate::transcripts::KeccakTranscript;
    use ark_bn254::Fr;
    use ark_std::UniformRand;
    use rand::thread_rng;

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

        // Create non-relaxed instance and witness
        let (real_instance, real_witness) = RelaxedR1CSInstance::<F, Bn254Curve>::new_non_relaxed(
            &gens,
            &witness,
            public_inputs,
            r1cs.num_constraints,
            &mut rng,
        );

        // Create prover and verifier
        let prover = BlindFoldProver::new(&gens, &r1cs);
        let verifier = BlindFoldVerifier::new(&gens, &r1cs);

        // Generate proof
        let mut prover_transcript = KeccakTranscript::new(b"BlindFold_test");
        let proof = prover.prove(
            &real_instance,
            &real_witness,
            &z,
            &mut prover_transcript,
            &mut rng,
        );

        // Verify proof
        let mut verifier_transcript = KeccakTranscript::new(b"BlindFold_test");
        let result = verifier.verify(&proof, &mut verifier_transcript);

        assert!(result.is_ok(), "Verification should succeed: {result:?}");
    }

    #[test]
    fn test_blindfold_protocol_soundness_bad_witness() {
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

        let (real_instance, real_witness) = RelaxedR1CSInstance::<F, Bn254Curve>::new_non_relaxed(
            &gens,
            &witness,
            public_inputs,
            r1cs.num_constraints,
            &mut rng,
        );

        let prover = BlindFoldProver::new(&gens, &r1cs);
        let verifier = BlindFoldVerifier::new(&gens, &r1cs);

        // Generate valid proof
        let mut prover_transcript = KeccakTranscript::new(b"BlindFold_test");
        let mut proof = prover.prove(
            &real_instance,
            &real_witness,
            &z,
            &mut prover_transcript,
            &mut rng,
        );

        // Tamper with the folded witness
        proof.folded_witness.W[0] = F::rand(&mut rng);

        // Verification should fail
        let mut verifier_transcript = KeccakTranscript::new(b"BlindFold_test");
        let result = verifier.verify(&proof, &mut verifier_transcript);

        assert!(
            result.is_err(),
            "Verification should fail with tampered witness"
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

        let (real_instance, real_witness) = RelaxedR1CSInstance::<F, Bn254Curve>::new_non_relaxed(
            &gens,
            &witness,
            public_inputs,
            r1cs.num_constraints,
            &mut rng,
        );

        let prover = BlindFoldProver::new(&gens, &r1cs);
        let verifier = BlindFoldVerifier::new(&gens, &r1cs);

        let mut prover_transcript = KeccakTranscript::new(b"BlindFold_multi");
        let proof = prover.prove(
            &real_instance,
            &real_witness,
            &z,
            &mut prover_transcript,
            &mut rng,
        );

        let mut verifier_transcript = KeccakTranscript::new(b"BlindFold_multi");
        let result = verifier.verify(&proof, &mut verifier_transcript);

        assert!(
            result.is_ok(),
            "Multi-round verification should succeed: {result:?}"
        );
    }
}
