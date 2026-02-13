//! BlindFold Protocol Implementation
//!
//! The BlindFold protocol makes sumcheck proofs zero-knowledge by:
//! 1. Encoding verifier checks into a small R1CS
//! 2. Using Nova folding to hide the witness
//! 3. Using Spartan sumcheck to prove R1CS satisfaction without revealing the witness
//! 4. Hyrax-style openings to verify W(ry) and E(rx) evaluations

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::curve::{JoltCurve, JoltGroupElement};
use crate::field::JoltField;
use crate::poly::commitment::pedersen::PedersenGenerators;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::unipoly::CompressedUniPoly;
use crate::transcripts::Transcript;
use crate::utils::math::Math;

use super::folding::{
    commit_cross_term_rows, compute_cross_term, sample_random_instance_deterministic,
    sample_random_satisfying_pair_deterministic,
};
use super::r1cs::VerifierR1CS;
use super::relaxed_r1cs::{RelaxedR1CSInstance, RelaxedR1CSWitness};
use super::spartan::{hyrax_combined_blinding, hyrax_combined_row, hyrax_evaluate};

/// Information about final_output variables at specific stages.
#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct FinalOutputInfo {
    pub stage_idx: usize,
    pub num_variables: usize,
}

/// BlindFold proof with Hyrax-style openings for W and E.
///
/// Instances are NOT included because:
/// - real_instance: verifier reconstructs from round_commitments, eval_commitments, public_inputs
/// - random_instance: verifier derives deterministically from transcript
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct BlindFoldProof<F: JoltField, C: JoltCurve> {
    /// Non-coefficient W row commitments from the real instance
    pub noncoeff_row_commitments: Vec<C::G1>,
    /// Cross-term T row commitments (E grid layout)
    pub cross_term_row_commitments: Vec<C::G1>,
    pub spartan_proof: Vec<CompressedUniPoly<F>>,
    pub final_output_info: Vec<FinalOutputInfo>,
    pub az_r: F,
    pub bz_r: F,
    pub cz_r: F,
    pub inner_sumcheck_proof: Vec<CompressedUniPoly<F>>,
    /// Hyrax W opening: combined row (C elements)
    pub w_combined_row: Vec<F>,
    pub w_combined_blinding: F,
    /// Hyrax E opening: combined row (C_E elements)
    pub e_combined_row: Vec<F>,
    pub e_combined_blinding: F,
}

pub struct BlindFoldProver<'a, F: JoltField, C: JoltCurve> {
    gens: &'a PedersenGenerators<C>,
    r1cs: &'a VerifierR1CS<F>,
    eval_commitment_gens: Option<(C::G1, C::G1)>,
}

impl<'a, F: JoltField, C: JoltCurve> BlindFoldProver<'a, F, C> {
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

    pub fn prove<T: Transcript>(
        &self,
        real_instance: &RelaxedR1CSInstance<F, C>,
        real_witness: &RelaxedR1CSWitness<F>,
        real_z: &[F],
        transcript: &mut T,
    ) -> BlindFoldProof<F, C> {
        use super::spartan::{BlindFoldInnerSumcheckProver, BlindFoldSpartanProver};
        use rand_chacha::ChaCha20Rng;
        use rand_core::SeedableRng;

        transcript.raw_append_label(b"BlindFold_real_instance");
        append_instance_to_transcript(real_instance, transcript);

        let (random_instance, random_witness, random_z) =
            sample_random_satisfying_pair_deterministic(
                self.gens,
                self.r1cs,
                self.eval_commitment_gens,
                transcript,
            );

        let T = compute_cross_term(
            self.r1cs,
            real_z,
            real_instance.u,
            &random_z,
            random_instance.u,
        );

        let (R_E, C_E) = self.r1cs.hyrax.e_grid(self.r1cs.num_constraints);

        let mut cross_term_rng = {
            transcript.raw_append_label(b"BlindFold_cross_term_blinding");
            let seed_a = transcript.challenge_u128();
            let seed_b = transcript.challenge_u128();
            let mut seed = [0u8; 32];
            seed[..16].copy_from_slice(&seed_a.to_le_bytes());
            seed[16..].copy_from_slice(&seed_b.to_le_bytes());
            ChaCha20Rng::from_seed(seed)
        };
        let (t_row_commitments, t_row_blindings) =
            commit_cross_term_rows(self.gens, &T, R_E, C_E, &mut cross_term_rng);

        transcript.raw_append_label(b"BlindFold_cross_term");
        for com in &t_row_commitments {
            append_g1_to_transcript::<C>(com, transcript);
        }

        let r: F::Challenge = transcript.challenge_scalar_optimized::<F>();
        let r_field: F = r.into();

        let folded_instance = real_instance.fold(&random_instance, &t_row_commitments, r_field);
        let folded_witness = real_witness.fold(&random_witness, &T, &t_row_blindings, r_field);

        let RelaxedR1CSWitness {
            E: folded_E,
            W: folded_W,
            w_row_blindings,
            e_row_blindings,
        } = folded_witness;

        let mut folded_z = Vec::with_capacity(self.r1cs.num_vars);
        folded_z.push(folded_instance.u);
        folded_z.extend_from_slice(&folded_instance.x);
        folded_z.extend_from_slice(&folded_W);

        let padded_e_len = self.r1cs.num_constraints.next_power_of_two();
        let mut e_padded = folded_E;
        e_padded.resize(padded_e_len, F::zero());
        let e_for_hyrax = e_padded.clone();

        // --- Outer Spartan sumcheck ---

        transcript.raw_append_label(b"BlindFold_spartan");
        let num_vars = padded_e_len.log_2();
        let tau: Vec<_> = transcript.challenge_vector_optimized::<F>(num_vars);

        let mut spartan_prover =
            BlindFoldSpartanProver::new(self.r1cs, folded_instance.u, folded_z, e_padded, tau);

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
        let az_r = final_claims.az_r;
        let bz_r = final_claims.bz_r;
        let cz_r = final_claims.cz_r;

        // --- Inner sumcheck ---

        transcript.append_scalars(b"blindfold_az_bz_cz", &[az_r, bz_r, cz_r]);

        let ra: F = transcript.challenge_scalar_optimized::<F>().into();
        let rb: F = transcript.challenge_scalar_optimized::<F>().into();
        let rc: F = transcript.challenge_scalar_optimized::<F>().into();

        let w_for_inner = folded_W.clone();
        let mut inner_prover = BlindFoldInnerSumcheckProver::new(
            self.r1cs,
            &spartan_challenges,
            w_for_inner,
            ra,
            rb,
            rc,
        );

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

        // --- Hyrax openings ---

        let hyrax = &self.r1cs.hyrax;

        // E opening at rx
        let log_R_E = R_E.log_2();
        let rx: Vec<F> = spartan_challenges.iter().map(|c| (*c).into()).collect();
        let e_combined_row = hyrax_combined_row(&e_for_hyrax, C_E, &rx[..log_R_E]);
        let e_combined_blinding = hyrax_combined_blinding(&e_row_blindings, &rx[..log_R_E]);

        // W opening at ry_w
        let log_R_prime = hyrax.log_R_prime();
        let ry_w: Vec<F> = inner_challenges.iter().map(|c| (*c).into()).collect();
        let w_combined_row = hyrax_combined_row(&folded_W, hyrax.C, &ry_w[..log_R_prime]);
        let w_combined_blinding = hyrax_combined_blinding(&w_row_blindings, &ry_w[..log_R_prime]);

        let final_output_info = collect_final_output_info(&self.r1cs.stage_configs);

        BlindFoldProof {
            noncoeff_row_commitments: real_instance.noncoeff_row_commitments.clone(),
            cross_term_row_commitments: t_row_commitments,
            spartan_proof,
            final_output_info,
            az_r,
            bz_r,
            cz_r,
            inner_sumcheck_proof: inner_proof,
            w_combined_row,
            w_combined_blinding,
            e_combined_row,
            e_combined_blinding,
        }
    }
}

fn collect_final_output_info(stage_configs: &[super::StageConfig]) -> Vec<FinalOutputInfo> {
    stage_configs
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
        .collect()
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlindFoldVerifyError {
    SpartanSumcheckFailed(usize),
    WrongSpartanProofLength { expected: usize, got: usize },
    EOpeningFailed,
    OuterClaimMismatch,
    WrongInnerSumcheckLength { expected: usize, got: usize },
    InnerSumcheckFailed(usize),
    WOpeningFailed,
    FinalClaimMismatch,
}

/// Data needed by verifier to reconstruct the real instance.
pub struct BlindFoldVerifierInput<F: JoltField, C: JoltCurve> {
    pub public_inputs: Vec<F>,
    pub round_commitments: Vec<C::G1>,
    pub eval_commitments: Vec<C::G1>,
}

pub struct BlindFoldVerifier<'a, F: JoltField, C: JoltCurve> {
    gens: &'a PedersenGenerators<C>,
    r1cs: &'a VerifierR1CS<F>,
    eval_commitment_gens: Option<(C::G1, C::G1)>,
}

impl<'a, F: JoltField, C: JoltCurve> BlindFoldVerifier<'a, F, C> {
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

    pub fn verify<T: Transcript>(
        &self,
        proof: &BlindFoldProof<F, C>,
        input: &BlindFoldVerifierInput<F, C>,
        transcript: &mut T,
    ) -> Result<(), BlindFoldVerifyError> {
        use super::spartan::{compute_L_w_at_ry, BlindFoldSpartanVerifier};

        let hyrax = &self.r1cs.hyrax;
        let (R_E, _C_E) = hyrax.e_grid(self.r1cs.num_constraints);

        // Reconstruct real instance (non-relaxed: u=1, E=0)
        let real_instance = RelaxedR1CSInstance {
            u: F::one(),
            x: input.public_inputs.clone(),
            round_commitments: input.round_commitments.clone(),
            noncoeff_row_commitments: proof.noncoeff_row_commitments.clone(),
            e_row_commitments: vec![C::G1::zero(); R_E],
            eval_commitments: input.eval_commitments.clone(),
        };

        transcript.raw_append_label(b"BlindFold_real_instance");
        append_instance_to_transcript(&real_instance, transcript);

        let random_instance = sample_random_instance_deterministic(
            self.gens,
            self.r1cs,
            self.eval_commitment_gens,
            transcript,
        );

        // Advance transcript past cross-term blinding seed
        transcript.raw_append_label(b"BlindFold_cross_term_blinding");
        let _ = transcript.challenge_u128();
        let _ = transcript.challenge_u128();

        transcript.raw_append_label(b"BlindFold_cross_term");
        for com in &proof.cross_term_row_commitments {
            append_g1_to_transcript::<C>(com, transcript);
        }

        let r: F::Challenge = transcript.challenge_scalar_optimized::<F>();
        let r_field: F = r.into();

        let folded_instance =
            real_instance.fold(&random_instance, &proof.cross_term_row_commitments, r_field);

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

        let az_r = proof.az_r;
        let bz_r = proof.bz_r;
        let cz_r = proof.cz_r;

        transcript.append_scalars(b"blindfold_az_bz_cz", &[az_r, bz_r, cz_r]);

        let ra: F = transcript.challenge_scalar_optimized::<F>().into();
        let rb: F = transcript.challenge_scalar_optimized::<F>().into();
        let rc: F = transcript.challenge_scalar_optimized::<F>().into();

        // --- Inner sumcheck ---

        let spartan_verifier = BlindFoldSpartanVerifier::new(
            self.r1cs,
            tau,
            folded_instance.u,
            folded_instance.x.clone(),
        );

        let (pub_az, pub_bz, pub_cz) = spartan_verifier.public_contributions(&challenges);
        let mut inner_claim = ra * (az_r - pub_az) + rb * (bz_r - pub_bz) + rc * (cz_r - pub_cz);

        let w_padded_len = hyrax.R_prime * hyrax.C;
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

        // --- E Hyrax opening ---

        let log_R_E = R_E.log_2();
        let rx: Vec<F> = challenges.iter().map(|c| (*c).into()).collect();
        let (rx_row, rx_col) = rx.split_at(log_R_E);

        let eq_rx_row: Vec<F> = EqPolynomial::evals(rx_row);
        let mut c_combined_e = C::G1::zero();
        for (i, com) in folded_instance.e_row_commitments.iter().enumerate() {
            c_combined_e += com.scalar_mul(&eq_rx_row[i]);
        }
        let expected_e_com = self
            .gens
            .commit(&proof.e_combined_row, &proof.e_combined_blinding);
        if c_combined_e != expected_e_com {
            return Err(BlindFoldVerifyError::EOpeningFailed);
        }
        let e_r = hyrax_evaluate(&proof.e_combined_row, rx_col);

        // --- Outer claim check ---

        let eq_tau_r = spartan_verifier.eq_tau_at_r(&challenges);
        let expected_outer = eq_tau_r * (az_r * bz_r - folded_instance.u * cz_r - e_r);
        if claim != expected_outer {
            return Err(BlindFoldVerifyError::OuterClaimMismatch);
        }

        // --- W Hyrax opening ---

        let log_R_prime = hyrax.log_R_prime();
        let ry_w: Vec<F> = inner_challenges.iter().map(|c| (*c).into()).collect();
        let (ry_row, ry_col) = ry_w.split_at(log_R_prime);

        let all_w_rows =
            folded_instance.all_w_row_commitments(hyrax.total_rounds, hyrax.R_coeff, hyrax.R_prime);
        let eq_ry_row: Vec<F> = EqPolynomial::evals(ry_row);
        let mut c_combined_w = C::G1::zero();
        for (i, com) in all_w_rows.iter().enumerate() {
            c_combined_w += com.scalar_mul(&eq_ry_row[i]);
        }
        let expected_w_com = self
            .gens
            .commit(&proof.w_combined_row, &proof.w_combined_blinding);
        if c_combined_w != expected_w_com {
            return Err(BlindFoldVerifyError::WOpeningFailed);
        }
        let w_ry = hyrax_evaluate(&proof.w_combined_row, ry_col);

        // --- Final claim check: inner_claim == L_w(ry) · W(ry) ---

        let l_w_at_ry = compute_L_w_at_ry(self.r1cs, &challenges, &inner_challenges, ra, rb, rc);
        let expected_inner_final = l_w_at_ry * w_ry;
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

    for commitment in &instance.noncoeff_row_commitments {
        append_g1_to_transcript::<C>(commitment, transcript);
    }

    for commitment in &instance.e_row_commitments {
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
    use crate::subprotocols::blindfold::r1cs::VerifierR1CSBuilder;
    use crate::subprotocols::blindfold::witness::{BlindFoldWitness, RoundWitness, StageWitness};
    use crate::subprotocols::blindfold::StageConfig;
    use crate::transcripts::KeccakTranscript;
    use ark_bn254::Fr;
    use ark_std::Zero;
    use rand::thread_rng;

    fn make_test_instance(
        configs: &[StageConfig],
        _blindfold_witness: &BlindFoldWitness<Fr>,
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
        let gens = PedersenGenerators::<Bn254Curve>::deterministic(r1cs.hyrax.C + 1);

        assert!(r1cs.is_satisfied(z));

        let witness_start = 1 + r1cs.num_public_inputs;
        let witness: Vec<F> = z[witness_start..].to_vec();
        let public_inputs: Vec<F> = z[1..witness_start].to_vec();

        let hyrax = &r1cs.hyrax;
        let hyrax_C = hyrax.C;
        let R_coeff = hyrax.R_coeff;
        let R_prime = hyrax.R_prime;

        let mut round_commitments = Vec::new();
        let mut w_row_blindings = vec![F::zero(); R_prime];

        for round_idx in 0..hyrax.total_rounds {
            let row_start = round_idx * hyrax_C;
            let blinding = F::random(&mut rng);
            let commitment = gens.commit(&witness[row_start..row_start + hyrax_C], &blinding);
            w_row_blindings[round_idx] = blinding;
            round_commitments.push(commitment);
        }

        let noncoeff_rows_count = hyrax.noncoeff_rows();
        let mut noncoeff_row_commitments = Vec::new();
        for row in 0..noncoeff_rows_count {
            let start = R_coeff * hyrax_C + row * hyrax_C;
            let end = (start + hyrax_C).min(witness.len());
            let blinding = F::random(&mut rng);
            noncoeff_row_commitments.push(gens.commit(&witness[start..end], &blinding));
            w_row_blindings[R_coeff + row] = blinding;
        }

        let (real_instance, real_witness) = RelaxedR1CSInstance::<F, Bn254Curve>::new_non_relaxed(
            &witness,
            public_inputs,
            r1cs.num_constraints,
            hyrax_C,
            round_commitments,
            noncoeff_row_commitments,
            Vec::new(),
            w_row_blindings,
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
        let prover = BlindFoldProver::new(gens, r1cs, None);
        let verifier = BlindFoldVerifier::new(gens, r1cs, None);

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

        let prover = BlindFoldProver::new(&gens, &r1cs, None);
        let verifier = BlindFoldVerifier::new(&gens, &r1cs, None);

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

        let prover = BlindFoldProver::new(&gens, &r1cs, None);
        let verifier = BlindFoldVerifier::new(&gens, &r1cs, None);

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
    fn test_blindfold_soundness_tampered_w_combined_row() {
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

        let prover = BlindFoldProver::new(&gens, &r1cs, None);
        let verifier = BlindFoldVerifier::new(&gens, &r1cs, None);

        let mut prover_transcript = KeccakTranscript::new(b"BlindFold_test");
        let mut proof = prover.prove(&real_instance, &real_witness, &z, &mut prover_transcript);

        if !proof.w_combined_row.is_empty() {
            proof.w_combined_row[0] += F::from_u64(1);
        }

        let verifier_input = BlindFoldVerifierInput {
            public_inputs: real_instance.x.clone(),
            round_commitments: real_instance.round_commitments.clone(),
            eval_commitments: real_instance.eval_commitments.clone(),
        };

        let mut verifier_transcript = KeccakTranscript::new(b"BlindFold_test");
        let result = verifier.verify(&proof, &verifier_input, &mut verifier_transcript);
        assert!(
            matches!(result, Err(BlindFoldVerifyError::WOpeningFailed)),
            "Tampered w_combined_row should fail W Hyrax check: {result:?}"
        );
    }

    #[test]
    fn test_blindfold_soundness_tampered_e_combined_row() {
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

        let prover = BlindFoldProver::new(&gens, &r1cs, None);
        let verifier = BlindFoldVerifier::new(&gens, &r1cs, None);

        let mut prover_transcript = KeccakTranscript::new(b"BlindFold_test");
        let mut proof = prover.prove(&real_instance, &real_witness, &z, &mut prover_transcript);

        if !proof.e_combined_row.is_empty() {
            proof.e_combined_row[0] += F::from_u64(1);
        }

        let verifier_input = BlindFoldVerifierInput {
            public_inputs: real_instance.x.clone(),
            round_commitments: real_instance.round_commitments.clone(),
            eval_commitments: real_instance.eval_commitments.clone(),
        };

        let mut verifier_transcript = KeccakTranscript::new(b"BlindFold_test");
        let result = verifier.verify(&proof, &verifier_input, &mut verifier_transcript);
        assert!(
            matches!(result, Err(BlindFoldVerifyError::EOpeningFailed)),
            "Tampered e_combined_row should fail E Hyrax check: {result:?}"
        );
    }
}
