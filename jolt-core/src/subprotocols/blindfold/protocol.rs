//! BlindFold Protocol Implementation
//!
//! The BlindFold protocol makes sumcheck proofs zero-knowledge by:
//! 1. Encoding verifier checks into a small R1CS
//! 2. Using Nova folding to hide the witness
//! 3. Using Spartan sumcheck to prove R1CS satisfaction without revealing the witness
//! 4. Inner sumcheck + Pedersen/Dory decomposition to verify the final Spartan claim
//!
//! Protocol flow:
//! 1. Prover has real instance-witness pair (u=1, E=0)
//! 2. Prover samples random satisfying pair
//! 3. Prover commits to cross-term T
//! 4. Verifier sends challenge r (via Fiat-Shamir)
//! 5. Both parties fold instances; prover folds witnesses
//! 6. Prover commits E and W_dory via Dory
//! 7. Outer Spartan sumcheck → rx, outer_claim
//! 8. Inner sumcheck reduces matrix claims to point evaluation of W
//! 9. Pedersen + Dory openings verify W(ry_w) decomposition

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::curve::{JoltCurve, JoltGroupElement};
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::pedersen::PedersenGenerators;
use crate::poly::unipoly::CompressedUniPoly;
use crate::transcripts::Transcript;

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

/// BlindFold proof containing the data needed for verification.
///
/// Instances are NOT included because:
/// - real_instance: verifier reconstructs from round_commitments, eval_commitments, public_inputs
/// - random_instance: verifier derives deterministically from transcript
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct BlindFoldProof<F: JoltField, C: JoltCurve, PCS: CommitmentScheme<Field = F>> {
    pub witness_commitment: C::G1,
    pub cross_term_commitment: C::G1,
    pub spartan_proof: Vec<CompressedUniPoly<F>>,
    pub final_output_info: Vec<FinalOutputInfo>,
    pub e_commitment: PCS::Commitment,
    pub e_claim: F,
    pub e_opening_proof: PCS::Proof,
    pub az_r: F,
    pub bz_r: F,
    pub cz_r: F,
    pub inner_sumcheck_proof: Vec<CompressedUniPoly<F>>,
    pub round_coefficients_folded: Vec<Vec<F>>,
    pub round_blindings_folded: Vec<F>,
    pub w_dory_commitment: PCS::Commitment,
    pub w_dory_claim: F,
    pub w_dory_opening_proof: PCS::Proof,
}

/// BlindFold prover.
pub struct BlindFoldProver<'a, F: JoltField, C: JoltCurve, PCS: CommitmentScheme<Field = F>> {
    gens: &'a PedersenGenerators<C>,
    r1cs: &'a VerifierR1CS<F>,
    eval_commitment_gens: Option<(C::G1, C::G1)>,
    pcs_setup: &'a PCS::ProverSetup,
}

impl<'a, F: JoltField, C: JoltCurve, PCS: CommitmentScheme<Field = F>>
    BlindFoldProver<'a, F, C, PCS>
{
    pub fn new(
        gens: &'a PedersenGenerators<C>,
        r1cs: &'a VerifierR1CS<F>,
        eval_commitment_gens: Option<(C::G1, C::G1)>,
        pcs_setup: &'a PCS::ProverSetup,
    ) -> Self {
        Self {
            gens,
            r1cs,
            eval_commitment_gens,
            pcs_setup,
        }
    }

    pub fn prove<T: Transcript>(
        &self,
        real_instance: &RelaxedR1CSInstance<F, C>,
        real_witness: &RelaxedR1CSWitness<F>,
        real_z: &[F],
        transcript: &mut T,
    ) -> BlindFoldProof<F, C, PCS> {
        use super::spartan::{
            coefficient_positions, BlindFoldInnerSumcheckProver, BlindFoldSpartanProver,
        };
        use crate::poly::commitment::dory::{DoryContext, DoryGlobals};
        use crate::poly::dense_mlpoly::DensePolynomial;
        use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial};
        use crate::utils::math::Math;
        use rand_chacha::ChaCha20Rng;
        use rand_core::SeedableRng;

        // Step 1: Bind real_instance to transcript
        transcript.raw_append_label(b"BlindFold_real_instance");
        append_instance_to_transcript(real_instance, transcript);

        // Step 2: Sample random satisfying pair deterministically from transcript
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

        // Step 4: Commit to cross-term
        transcript.raw_append_label(b"BlindFold_cross_term_blinding");
        let seed_a = transcript.challenge_u128();
        let seed_b = transcript.challenge_u128();
        let mut seed = [0u8; 32];
        seed[..16].copy_from_slice(&seed_a.to_le_bytes());
        seed[16..].copy_from_slice(&seed_b.to_le_bytes());
        let mut rng = ChaCha20Rng::from_seed(seed);
        let r_T = F::random(&mut rng);
        let T_bar = self.gens.commit(&T, &r_T);

        // Step 5: Bind T_bar to transcript
        transcript.raw_append_label(b"BlindFold_cross_term");
        append_g1_to_transcript::<C>(&T_bar, transcript);

        // Step 6: Get folding challenge via Fiat-Shamir
        let r: F::Challenge = transcript.challenge_scalar_optimized::<F>();
        let r_field: F = r.into();

        // Step 7: Fold instances (public)
        let folded_instance = real_instance.fold(&random_instance, &T_bar, r_field);

        // Step 8: Fold witnesses (private)
        let folded_witness = real_witness.fold(&random_witness, &T, r_T, r_field);

        // Destructure folded witness
        let folded_E = folded_witness.E;
        let folded_W = folded_witness.W;
        let round_coefficients_folded = folded_witness.round_coefficients;
        let round_blindings_folded = folded_witness.round_blindings;

        // Build folded Z vector: [u, x..., W...]
        let mut folded_z = Vec::with_capacity(self.r1cs.num_vars);
        folded_z.push(folded_instance.u);
        folded_z.extend_from_slice(&folded_instance.x);
        folded_z.extend_from_slice(&folded_W);

        // --- Commit E and W_dory before outer sumcheck (bound to transcript) ---

        let padded_e_len = self.r1cs.num_constraints.next_power_of_two();
        let mut e_padded = folded_E.clone();
        e_padded.resize(padded_e_len, F::zero());

        DoryGlobals::initialize_context(1, padded_e_len, DoryContext::BlindFoldE, None);
        let e_poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(e_padded));
        let (e_commitment, e_hint) = {
            let _guard = DoryGlobals::with_context(DoryContext::BlindFoldE);
            PCS::commit(&e_poly, self.pcs_setup)
        };

        let witness_len = folded_W.len();
        let w_padded_len = witness_len.next_power_of_two();
        let coeff_positions = coefficient_positions(self.r1cs);

        let mut w_padded = folded_W;
        w_padded.resize(w_padded_len, F::zero());

        // W_dory: zero at coefficient positions, actual witness elsewhere
        let mut w_dory_vals = w_padded.clone();
        for &(start, num_coeffs) in &coeff_positions {
            for k in 0..num_coeffs {
                if start + k < w_padded_len {
                    w_dory_vals[start + k] = F::zero();
                }
            }
        }

        DoryGlobals::initialize_context(1, w_padded_len, DoryContext::BlindFoldW, None);
        let w_dory_poly =
            MultilinearPolynomial::LargeScalars(DensePolynomial::new(w_dory_vals.clone()));
        let (w_dory_commitment, w_dory_hint) = {
            let _guard = DoryGlobals::with_context(DoryContext::BlindFoldW);
            PCS::commit(&w_dory_poly, self.pcs_setup)
        };

        // Append commitments to transcript before tau generation
        let mut e_commitment_bytes = Vec::new();
        e_commitment
            .serialize_compressed(&mut e_commitment_bytes)
            .expect("Serialization should not fail");
        transcript.append_bytes(b"blindfold_e_commitment", &e_commitment_bytes);

        let mut w_dory_commitment_bytes = Vec::new();
        w_dory_commitment
            .serialize_compressed(&mut w_dory_commitment_bytes)
            .expect("Serialization should not fail");
        transcript.append_bytes(b"blindfold_w_dory_com", &w_dory_commitment_bytes);

        // --- Outer Spartan sumcheck ---

        transcript.raw_append_label(b"BlindFold_spartan");
        let num_vars = self.r1cs.num_constraints.next_power_of_two().log_2();
        let tau: Vec<_> = transcript.challenge_vector_optimized::<F>(num_vars);

        let mut spartan_prover =
            BlindFoldSpartanProver::new(self.r1cs, folded_instance.u, folded_z, folded_E, tau);

        let mut spartan_proof = Vec::with_capacity(num_vars);
        let mut spartan_challenges: Vec<F::Challenge> = Vec::with_capacity(num_vars);
        let mut claim = F::zero();

        for _ in 0..num_vars {
            let poly = spartan_prover.compute_round_polynomial(claim);
            let compressed = poly.compress();
            transcript.append_scalars(b"sumcheck_poly", &compressed.coeffs_except_linear_term);
            spartan_proof.push(compressed);

            let r_j = transcript.challenge_scalar_optimized::<F>();
            claim = poly.evaluate(&r_j);
            spartan_prover.bind_challenge(r_j);
            spartan_challenges.push(r_j);
        }

        let final_claims = spartan_prover.final_claims();
        let e_claim = final_claims.e_r;
        let az_r = final_claims.az_r;
        let bz_r = final_claims.bz_r;
        let cz_r = final_claims.cz_r;

        // --- Inner sumcheck ---

        transcript.append_scalars(b"blindfold_az_bz_cz", &[az_r, bz_r, cz_r]);

        let ra: F = transcript.challenge_scalar_optimized::<F>().into();
        let rb: F = transcript.challenge_scalar_optimized::<F>().into();
        let rc: F = transcript.challenge_scalar_optimized::<F>().into();

        let mut inner_prover =
            BlindFoldInnerSumcheckProver::new(self.r1cs, &spartan_challenges, w_padded, ra, rb, rc);

        let inner_num_vars = inner_prover.num_vars();
        let mut inner_proof = Vec::with_capacity(inner_num_vars);
        let mut inner_challenges: Vec<F::Challenge> = Vec::with_capacity(inner_num_vars);

        let (w_az, w_bz, w_cz) = spartan_prover.witness_contributions(&spartan_challenges);
        let mut inner_claim = ra * w_az + rb * w_bz + rc * w_cz;

        for _ in 0..inner_num_vars {
            let poly = inner_prover.compute_round_polynomial();
            debug_assert_eq!(
                poly.coeffs[0] + poly.coeffs.iter().sum::<F>(),
                inner_claim,
                "Inner sumcheck round polynomial p(0)+p(1) != claim"
            );

            let compressed = poly.compress();
            transcript.append_scalars(
                b"inner_sumcheck_poly",
                &compressed.coeffs_except_linear_term,
            );
            inner_proof.push(compressed);

            let r_j = transcript.challenge_scalar_optimized::<F>();
            inner_claim = poly.evaluate(&r_j);
            inner_prover.bind_challenge(r_j);
            inner_challenges.push(r_j);
        }

        // --- Opening proofs ---

        // E opening at outer sumcheck point
        let (e_opening_proof, _) = {
            let _guard = DoryGlobals::with_context(DoryContext::BlindFoldE);
            PCS::prove(
                self.pcs_setup,
                &e_poly,
                &spartan_challenges,
                Some(e_hint),
                transcript,
            )
        };

        // W_dory: evaluate at inner sumcheck point, then prove
        let mut w_dory_eval = DensePolynomial::new(w_dory_vals);
        for &c in &inner_challenges {
            w_dory_eval.bind_parallel(c, BindingOrder::HighToLow);
        }
        let w_dory_claim: F = w_dory_eval[0];

        let (w_dory_opening_proof, _) = {
            let _guard = DoryGlobals::with_context(DoryContext::BlindFoldW);
            PCS::prove(
                self.pcs_setup,
                &w_dory_poly,
                &inner_challenges,
                Some(w_dory_hint),
                transcript,
            )
        };

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
            e_commitment,
            e_claim,
            e_opening_proof,
            az_r,
            bz_r,
            cz_r,
            inner_sumcheck_proof: inner_proof,
            round_coefficients_folded,
            round_blindings_folded,
            w_dory_commitment,
            w_dory_claim,
            w_dory_opening_proof,
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
pub struct BlindFoldVerifier<'a, F: JoltField, C: JoltCurve, PCS: CommitmentScheme<Field = F>> {
    gens: &'a PedersenGenerators<C>,
    r1cs: &'a VerifierR1CS<F>,
    #[allow(dead_code)]
    eval_commitment_gens: Option<(C::G1, C::G1)>,
    pcs_setup: &'a PCS::VerifierSetup,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlindFoldVerifyError {
    SpartanSumcheckFailed(usize),
    WrongSpartanProofLength { expected: usize, got: usize },
    EOpeningFailed,
    OuterClaimMismatch,
    WrongInnerSumcheckLength { expected: usize, got: usize },
    InnerSumcheckFailed(usize),
    PedersenCommitmentMismatch(usize),
    WDoryOpeningFailed,
    FinalClaimMismatch,
}

/// Data needed by verifier to reconstruct the real instance.
pub struct BlindFoldVerifierInput<F: JoltField, C: JoltCurve> {
    pub public_inputs: Vec<F>,
    pub round_commitments: Vec<C::G1>,
    pub eval_commitments: Vec<C::G1>,
}

impl<'a, F: JoltField, C: JoltCurve, PCS: CommitmentScheme<Field = F>>
    BlindFoldVerifier<'a, F, C, PCS>
{
    pub fn new(
        gens: &'a PedersenGenerators<C>,
        r1cs: &'a VerifierR1CS<F>,
        eval_commitment_gens: Option<(C::G1, C::G1)>,
        pcs_setup: &'a PCS::VerifierSetup,
    ) -> Self {
        Self {
            gens,
            r1cs,
            eval_commitment_gens,
            pcs_setup,
        }
    }

    pub fn verify<T: Transcript>(
        &self,
        proof: &BlindFoldProof<F, C, PCS>,
        input: &BlindFoldVerifierInput<F, C>,
        transcript: &mut T,
    ) -> Result<(), BlindFoldVerifyError> {
        use super::spartan::{
            coefficient_positions, compute_L_w_at_ry, compute_W_ped_at_ry, BlindFoldSpartanVerifier,
        };
        use crate::poly::commitment::dory::{DoryContext, DoryGlobals};
        use crate::utils::math::Math;

        // Step 1: Reconstruct the real instance (non-relaxed: u=1, E_bar=0)
        let real_instance = RelaxedR1CSInstance {
            E_bar: C::G1::zero(),
            u: F::one(),
            W_bar: proof.witness_commitment,
            x: input.public_inputs.clone(),
            round_commitments: input.round_commitments.clone(),
            eval_commitments: input.eval_commitments.clone(),
        };

        // Step 2: Bind real_instance to transcript (must match prover exactly)
        transcript.raw_append_label(b"BlindFold_real_instance");
        append_instance_to_transcript(&real_instance, transcript);

        // Step 3: Derive random instance deterministically from transcript
        let random_instance = sample_random_instance_deterministic(
            self.gens,
            self.r1cs,
            self.eval_commitment_gens,
            transcript,
        );

        // Step 4: Advance transcript for cross-term blinding
        transcript.raw_append_label(b"BlindFold_cross_term_blinding");
        let _ = transcript.challenge_u128();
        let _ = transcript.challenge_u128();

        // Step 5: Bind T_bar to transcript
        transcript.raw_append_label(b"BlindFold_cross_term");
        append_g1_to_transcript::<C>(&proof.cross_term_commitment, transcript);

        // Step 6: Get folding challenge
        let r: F::Challenge = transcript.challenge_scalar_optimized::<F>();
        let r_field: F = r.into();

        // Step 7: Compute folded instance
        let folded_instance =
            real_instance.fold(&random_instance, &proof.cross_term_commitment, r_field);

        // --- Absorb commitments (must match prover order) ---

        let mut e_commitment_bytes = Vec::new();
        proof
            .e_commitment
            .serialize_compressed(&mut e_commitment_bytes)
            .expect("Serialization should not fail");
        transcript.append_bytes(b"blindfold_e_commitment", &e_commitment_bytes);

        let mut w_dory_commitment_bytes = Vec::new();
        proof
            .w_dory_commitment
            .serialize_compressed(&mut w_dory_commitment_bytes)
            .expect("Serialization should not fail");
        transcript.append_bytes(b"blindfold_w_dory_com", &w_dory_commitment_bytes);

        // --- Outer Spartan sumcheck ---

        transcript.raw_append_label(b"BlindFold_spartan");
        let num_vars = self.r1cs.num_constraints.next_power_of_two().log_2();

        if proof.spartan_proof.len() != num_vars {
            return Err(BlindFoldVerifyError::WrongSpartanProofLength {
                expected: num_vars,
                got: proof.spartan_proof.len(),
            });
        }

        let tau: Vec<_> = transcript.challenge_vector_optimized::<F>(num_vars);
        let mut claim = F::zero();
        let mut challenges: Vec<F::Challenge> = Vec::with_capacity(num_vars);

        for (round, compressed_poly) in proof.spartan_proof.iter().enumerate() {
            transcript.append_scalars(b"sumcheck_poly", &compressed_poly.coeffs_except_linear_term);

            let poly = compressed_poly.decompress(&claim);
            let sum = poly.coeffs[0] + poly.coeffs.iter().sum::<F>();
            if sum != claim {
                return Err(BlindFoldVerifyError::SpartanSumcheckFailed(round));
            }

            let r_j = transcript.challenge_scalar_optimized::<F>();
            challenges.push(r_j);
            claim = poly.evaluate(&r_j);
        }

        // --- Outer claim check ---

        let az_r = proof.az_r;
        let bz_r = proof.bz_r;
        let cz_r = proof.cz_r;

        transcript.append_scalars(b"blindfold_az_bz_cz", &[az_r, bz_r, cz_r]);

        let spartan_verifier = BlindFoldSpartanVerifier::new(
            self.r1cs,
            tau,
            folded_instance.u,
            folded_instance.x.clone(),
        );

        let eq_tau_r = spartan_verifier.eq_tau_at_r(&challenges);
        let expected_outer = eq_tau_r * (az_r * bz_r - folded_instance.u * cz_r - proof.e_claim);
        if claim != expected_outer {
            return Err(BlindFoldVerifyError::OuterClaimMismatch);
        }

        // --- Inner sumcheck ---

        let ra: F = transcript.challenge_scalar_optimized::<F>().into();
        let rb: F = transcript.challenge_scalar_optimized::<F>().into();
        let rc: F = transcript.challenge_scalar_optimized::<F>().into();

        let (pub_az, pub_bz, pub_cz) = spartan_verifier.public_contributions(&challenges);
        let mut inner_claim = ra * (az_r - pub_az) + rb * (bz_r - pub_bz) + rc * (cz_r - pub_cz);

        let witness_len = self
            .r1cs
            .num_vars
            .saturating_sub(1 + self.r1cs.num_public_inputs);
        let w_padded_len = witness_len.next_power_of_two();
        let inner_num_vars = w_padded_len.log_2();

        if proof.inner_sumcheck_proof.len() != inner_num_vars {
            return Err(BlindFoldVerifyError::WrongInnerSumcheckLength {
                expected: inner_num_vars,
                got: proof.inner_sumcheck_proof.len(),
            });
        }

        let mut inner_challenges: Vec<F::Challenge> = Vec::with_capacity(inner_num_vars);

        for (round, compressed_poly) in proof.inner_sumcheck_proof.iter().enumerate() {
            transcript.append_scalars(
                b"inner_sumcheck_poly",
                &compressed_poly.coeffs_except_linear_term,
            );

            let poly = compressed_poly.decompress(&inner_claim);
            let sum = poly.coeffs[0] + poly.coeffs.iter().sum::<F>();
            if sum != inner_claim {
                return Err(BlindFoldVerifyError::InnerSumcheckFailed(round));
            }

            let r_j = transcript.challenge_scalar_optimized::<F>();
            inner_challenges.push(r_j);
            inner_claim = poly.evaluate(&r_j);
        }

        // inner_claim is now the final evaluation claim

        // --- E opening proof ---

        let padded_e_len = self.r1cs.num_constraints.next_power_of_two();
        DoryGlobals::initialize_context(1, padded_e_len, DoryContext::BlindFoldE, None);
        {
            let _guard = DoryGlobals::with_context(DoryContext::BlindFoldE);
            PCS::verify(
                &proof.e_opening_proof,
                self.pcs_setup,
                transcript,
                &challenges,
                &proof.e_claim,
                &proof.e_commitment,
            )
            .map_err(|_| BlindFoldVerifyError::EOpeningFailed)?;
        }

        // --- W_dory opening proof ---

        DoryGlobals::initialize_context(1, w_padded_len, DoryContext::BlindFoldW, None);
        {
            let _guard = DoryGlobals::with_context(DoryContext::BlindFoldW);
            PCS::verify(
                &proof.w_dory_opening_proof,
                self.pcs_setup,
                transcript,
                &inner_challenges,
                &proof.w_dory_claim,
                &proof.w_dory_commitment,
            )
            .map_err(|_| BlindFoldVerifyError::WDoryOpeningFailed)?;
        }

        // --- Pedersen verification ---

        let coeff_positions = coefficient_positions(self.r1cs);
        let num_rounds: usize = self.r1cs.stage_configs.iter().map(|s| s.num_rounds).sum();

        if proof.round_coefficients_folded.len() != num_rounds
            || proof.round_blindings_folded.len() != num_rounds
        {
            return Err(BlindFoldVerifyError::PedersenCommitmentMismatch(0));
        }

        for i in 0..num_rounds {
            let expected = self.gens.commit(
                &proof.round_coefficients_folded[i],
                &proof.round_blindings_folded[i],
            );
            if expected != folded_instance.round_commitments[i] {
                return Err(BlindFoldVerifyError::PedersenCommitmentMismatch(i));
            }
        }

        // --- Final claim verification ---

        let w_ped_at_ry = compute_W_ped_at_ry(
            &proof.round_coefficients_folded,
            &coeff_positions,
            &inner_challenges,
        );

        let l_w_at_ry = compute_L_w_at_ry(self.r1cs, &challenges, &inner_challenges, ra, rb, rc);

        let expected_inner_final = l_w_at_ry * (w_ped_at_ry + proof.w_dory_claim);
        if inner_claim != expected_inner_final {
            return Err(BlindFoldVerifyError::FinalClaimMismatch);
        }

        Ok(())
    }
}

fn append_g1_to_transcript<C: JoltCurve>(g: &C::G1, transcript: &mut impl Transcript) {
    let mut bytes = Vec::new();
    g.serialize_compressed(&mut bytes)
        .expect("Serialization should not fail");
    transcript.append_bytes(b"blindfold_g1", &bytes);
}

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
    transcript.append_bytes(b"blindfold_u", &u_bytes);

    for x_i in &instance.x {
        let mut x_bytes = Vec::new();
        x_i.serialize_compressed(&mut x_bytes)
            .expect("Serialization should not fail");
        transcript.append_bytes(b"blindfold_x", &x_bytes);
    }

    for commitment in &instance.round_commitments {
        append_g1_to_transcript::<C>(commitment, transcript);
    }

    for commitment in &instance.eval_commitments {
        append_g1_to_transcript::<C>(commitment, transcript);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curve::Bn254Curve;
    use crate::curve::JoltCurve;
    use crate::poly::commitment::mock::MockCommitScheme;
    use crate::subprotocols::blindfold::r1cs::VerifierR1CSBuilder;
    use crate::subprotocols::blindfold::witness::{BlindFoldWitness, RoundWitness, StageWitness};
    use crate::subprotocols::blindfold::StageConfig;
    use crate::transcripts::KeccakTranscript;
    use ark_bn254::Fr;
    use rand::thread_rng;

    type TestPCS = MockCommitScheme<Fr>;

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

    fn make_test_instance(
        configs: &[StageConfig],
        blindfold_witness: &BlindFoldWitness<Fr>,
        z: &[Fr],
    ) -> (
        RelaxedR1CSInstance<Fr, Bn254Curve>,
        RelaxedR1CSWitness<Fr>,
        VerifierR1CS<Fr>,
        PedersenGenerators<Bn254Curve>,
    ) {
        type F = Fr;
        let mut rng = thread_rng();

        let builder = VerifierR1CSBuilder::<F>::new(configs);
        let r1cs = builder.build();
        let gens = PedersenGenerators::<Bn254Curve>::deterministic(r1cs.num_vars + 100);

        assert!(r1cs.is_satisfied(z));

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

        (real_instance, real_witness, r1cs, gens)
    }

    fn prove_and_verify(
        r1cs: &VerifierR1CS<Fr>,
        gens: &PedersenGenerators<Bn254Curve>,
        real_instance: &RelaxedR1CSInstance<Fr, Bn254Curve>,
        real_witness: &RelaxedR1CSWitness<Fr>,
        z: &[Fr],
        label: &'static [u8],
    ) -> Result<(), BlindFoldVerifyError> {
        let pcs_prover_setup = ();
        let pcs_verifier_setup = ();
        let prover = BlindFoldProver::<_, _, TestPCS>::new(gens, r1cs, None, &pcs_prover_setup);
        let verifier =
            BlindFoldVerifier::<_, _, TestPCS>::new(gens, r1cs, None, &pcs_verifier_setup);

        let mut prover_transcript = KeccakTranscript::new(label);
        let proof = prover.prove(real_instance, real_witness, z, &mut prover_transcript);

        let verifier_input = BlindFoldVerifierInput {
            public_inputs: real_instance.x.clone(),
            round_commitments: real_instance.round_commitments.clone(),
            eval_commitments: real_instance.eval_commitments.clone(),
        };

        let mut verifier_transcript = KeccakTranscript::new(label);
        verifier.verify(&proof, &verifier_input, &mut verifier_transcript)
    }

    #[test]
    fn test_blindfold_protocol_completeness() {
        type F = Fr;

        let configs = [StageConfig::new(1, 3)];
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

        let builder = VerifierR1CSBuilder::<F>::new(&configs);
        let r1cs = builder.build();
        let z = blindfold_witness.assign(&r1cs);

        let (real_instance, real_witness, r1cs, gens) =
            make_test_instance(&configs, &blindfold_witness, &z);

        let result = prove_and_verify(
            &r1cs,
            &gens,
            &real_instance,
            &real_witness,
            &z,
            b"BlindFold_test",
        );
        assert!(result.is_ok(), "Verification should succeed: {result:?}");
    }

    #[test]
    fn test_blindfold_protocol_soundness_wrong_proof_length() {
        type F = Fr;

        let configs = [StageConfig::new(1, 3)];
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

        let builder = VerifierR1CSBuilder::<F>::new(&configs);
        let r1cs = builder.build();
        let z = blindfold_witness.assign(&r1cs);

        let (real_instance, real_witness, r1cs, gens) =
            make_test_instance(&configs, &blindfold_witness, &z);

        let pcs_prover_setup = ();
        let pcs_verifier_setup = ();
        let prover = BlindFoldProver::<_, _, TestPCS>::new(&gens, &r1cs, None, &pcs_prover_setup);
        let verifier =
            BlindFoldVerifier::<_, _, TestPCS>::new(&gens, &r1cs, None, &pcs_verifier_setup);

        let mut prover_transcript = KeccakTranscript::new(b"BlindFold_test");
        let mut proof = prover.prove(&real_instance, &real_witness, &z, &mut prover_transcript);

        if !proof.spartan_proof.is_empty() {
            proof.spartan_proof.pop();
        }

        let verifier_input = BlindFoldVerifierInput {
            public_inputs: real_instance.x.clone(),
            round_commitments: real_instance.round_commitments.clone(),
            eval_commitments: real_instance.eval_commitments.clone(),
        };

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
        type F = Fr;

        let configs = [StageConfig::new(3, 3)];

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

        let c0_2 = F::from_u64(30);
        let c2_2 = F::from_u64(10);
        let c3_2 = F::from_u64(5);
        let c1_2 = next1 - F::from_u64(75);
        let round2 = RoundWitness::new(vec![c0_2, c1_2, c2_2, c3_2], F::from_u64(4));
        let next2 = round2.evaluate(F::from_u64(4));

        let c0_3 = F::from_u64(50);
        let c2_3 = F::from_u64(8);
        let c3_3 = F::from_u64(2);
        let c1_3 = next2 - F::from_u64(110);
        let round3 = RoundWitness::new(vec![c0_3, c1_3, c2_3, c3_3], F::from_u64(6));

        let initial_claim = F::from_u64(55);
        let blindfold_witness = BlindFoldWitness::new(
            initial_claim,
            vec![StageWitness::new(vec![round1, round2, round3])],
        );

        let builder = VerifierR1CSBuilder::<F>::new(&configs);
        let r1cs = builder.build();
        let z = blindfold_witness.assign(&r1cs);

        let (real_instance, real_witness, r1cs, gens) =
            make_test_instance(&configs, &blindfold_witness, &z);

        let result = prove_and_verify(
            &r1cs,
            &gens,
            &real_instance,
            &real_witness,
            &z,
            b"BlindFold_multi",
        );
        assert!(
            result.is_ok(),
            "Multi-round verification should succeed: {result:?}"
        );
    }

    #[test]
    fn test_blindfold_soundness_tampered_az_r() {
        type F = Fr;

        let configs = [StageConfig::new(1, 3)];
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

        let builder = VerifierR1CSBuilder::<F>::new(&configs);
        let r1cs = builder.build();
        let z = blindfold_witness.assign(&r1cs);

        let (real_instance, real_witness, r1cs, gens) =
            make_test_instance(&configs, &blindfold_witness, &z);

        let pcs_prover_setup = ();
        let pcs_verifier_setup = ();
        let prover = BlindFoldProver::<_, _, TestPCS>::new(&gens, &r1cs, None, &pcs_prover_setup);
        let verifier =
            BlindFoldVerifier::<_, _, TestPCS>::new(&gens, &r1cs, None, &pcs_verifier_setup);

        let mut prover_transcript = KeccakTranscript::new(b"BlindFold_test");
        let mut proof = prover.prove(&real_instance, &real_witness, &z, &mut prover_transcript);

        proof.az_r += F::from_u64(1);

        let verifier_input = BlindFoldVerifierInput {
            public_inputs: real_instance.x.clone(),
            round_commitments: real_instance.round_commitments.clone(),
            eval_commitments: real_instance.eval_commitments.clone(),
        };

        let mut verifier_transcript = KeccakTranscript::new(b"BlindFold_test");
        let result = verifier.verify(&proof, &verifier_input, &mut verifier_transcript);
        assert!(
            result.is_err(),
            "Tampered az_r should cause verification failure"
        );
    }

    #[test]
    fn test_blindfold_soundness_tampered_round_coefficients() {
        type F = Fr;

        let configs = [StageConfig::new(1, 3)];
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

        let builder = VerifierR1CSBuilder::<F>::new(&configs);
        let r1cs = builder.build();
        let z = blindfold_witness.assign(&r1cs);

        let (real_instance, real_witness, r1cs, gens) =
            make_test_instance(&configs, &blindfold_witness, &z);

        let pcs_prover_setup = ();
        let pcs_verifier_setup = ();
        let prover = BlindFoldProver::<_, _, TestPCS>::new(&gens, &r1cs, None, &pcs_prover_setup);
        let verifier =
            BlindFoldVerifier::<_, _, TestPCS>::new(&gens, &r1cs, None, &pcs_verifier_setup);

        let mut prover_transcript = KeccakTranscript::new(b"BlindFold_test");
        let mut proof = prover.prove(&real_instance, &real_witness, &z, &mut prover_transcript);

        if !proof.round_coefficients_folded.is_empty()
            && !proof.round_coefficients_folded[0].is_empty()
        {
            proof.round_coefficients_folded[0][0] += F::from_u64(1);
        }

        let verifier_input = BlindFoldVerifierInput {
            public_inputs: real_instance.x.clone(),
            round_commitments: real_instance.round_commitments.clone(),
            eval_commitments: real_instance.eval_commitments.clone(),
        };

        let mut verifier_transcript = KeccakTranscript::new(b"BlindFold_test");
        let result = verifier.verify(&proof, &verifier_input, &mut verifier_transcript);
        assert!(
            matches!(
                result,
                Err(BlindFoldVerifyError::PedersenCommitmentMismatch(_))
            ),
            "Tampered round coefficients should fail Pedersen check: {result:?}"
        );
    }

    #[test]
    fn test_blindfold_soundness_tampered_w_dory_claim() {
        type F = Fr;

        let configs = [StageConfig::new(1, 3)];
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

        let builder = VerifierR1CSBuilder::<F>::new(&configs);
        let r1cs = builder.build();
        let z = blindfold_witness.assign(&r1cs);

        let (real_instance, real_witness, r1cs, gens) =
            make_test_instance(&configs, &blindfold_witness, &z);

        let pcs_prover_setup = ();
        let pcs_verifier_setup = ();
        let prover = BlindFoldProver::<_, _, TestPCS>::new(&gens, &r1cs, None, &pcs_prover_setup);
        let verifier =
            BlindFoldVerifier::<_, _, TestPCS>::new(&gens, &r1cs, None, &pcs_verifier_setup);

        let mut prover_transcript = KeccakTranscript::new(b"BlindFold_test");
        let mut proof = prover.prove(&real_instance, &real_witness, &z, &mut prover_transcript);

        proof.w_dory_claim += F::from_u64(1);

        let verifier_input = BlindFoldVerifierInput {
            public_inputs: real_instance.x.clone(),
            round_commitments: real_instance.round_commitments.clone(),
            eval_commitments: real_instance.eval_commitments.clone(),
        };

        let mut verifier_transcript = KeccakTranscript::new(b"BlindFold_test");
        let result = verifier.verify(&proof, &verifier_input, &mut verifier_transcript);
        assert!(
            result.is_err(),
            "Tampered W_dory claim should cause verification failure: {result:?}"
        );
    }
}
