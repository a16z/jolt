#![allow(clippy::len_without_is_empty)]

use std::marker::PhantomData;

use crate::field::JoltField;
use crate::jolt::vm::JoltCommitments;
use crate::jolt::vm::JoltPolynomials;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::opening_proof::ProverOpeningAccumulator;
use crate::poly::opening_proof::VerifierOpeningAccumulator;
use crate::poly::split_eq_poly::SplitEqPolynomial;
use crate::r1cs::key::UniformSpartanKey;
use crate::utils::math::Math;
use crate::utils::thread::drop_in_background_thread;

use crate::utils::transcript::Transcript;
use ark_serialize::CanonicalDeserialize;
use ark_serialize::CanonicalSerialize;

use rayon::prelude::*;
use thiserror::Error;

use crate::{
    poly::{dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial},
    subprotocols::sumcheck::SumcheckInstanceProof,
};

use super::builder::CombinedUniformBuilder;
use super::inputs::ConstraintInput;

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum SpartanError {
    /// returned if the supplied row or col in (row,col,val) tuple is out of range
    #[error("InvalidIndex")]
    InvalidIndex,

    /// returned when an invalid sum-check proof is provided
    #[error("InvalidSumcheckProof")]
    InvalidSumcheckProof,

    /// returned when the recusive sumcheck proof fails
    #[error("InvalidOuterSumcheckProof")]
    InvalidOuterSumcheckProof,

    /// returned when the final sumcheck opening proof fails
    #[error("InvalidOuterSumcheckClaim")]
    InvalidOuterSumcheckClaim,

    /// returned when the recusive sumcheck proof fails
    #[error("InvalidInnerSumcheckProof")]
    InvalidInnerSumcheckProof,

    /// returned when the final sumcheck opening proof fails
    #[error("InvalidInnerSumcheckClaim")]
    InvalidInnerSumcheckClaim,

    /// returned if the supplied witness is not of the right length
    #[error("InvalidWitnessLength")]
    InvalidWitnessLength,

    /// returned when an invalid PCS proof is provided
    #[error("InvalidPCSProof")]
    InvalidPCSProof,
}

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct UniformSpartanProof<
    const C: usize,
    I: ConstraintInput,
    F: JoltField,
    ProofTranscript: Transcript,
> {
    _inputs: PhantomData<I>,
    pub(crate) outer_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub(crate) outer_sumcheck_claims: (F, F, F),
    pub(crate) inner_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub(crate) claimed_witness_evals: Vec<F>,
    _marker: PhantomData<ProofTranscript>,
}

impl<const C: usize, I, F, ProofTranscript> UniformSpartanProof<C, I, F, ProofTranscript>
where
    I: ConstraintInput,
    F: JoltField,
    ProofTranscript: Transcript,
{
    #[tracing::instrument(skip_all, name = "Spartan::setup")]
    pub fn setup(
        constraint_builder: &CombinedUniformBuilder<C, F, I>,
        padded_num_steps: usize,
    ) -> UniformSpartanKey<C, I, F> {
        assert_eq!(
            padded_num_steps,
            constraint_builder.uniform_repeat().next_power_of_two()
        );
        UniformSpartanKey::from_builder(constraint_builder)
    }

    #[tracing::instrument(skip_all, name = "Spartan::prove")]
    pub fn prove<PCS>(
        constraint_builder: &CombinedUniformBuilder<C, F, I>,
        key: &UniformSpartanKey<C, I, F>,
        polynomials: &JoltPolynomials<F>,
        opening_accumulator: &mut ProverOpeningAccumulator<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<Self, SpartanError>
    where
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
    {
        let flattened_polys: Vec<&DensePolynomial<F>> = I::flatten::<C>()
            .iter()
            .map(|var| var.get_ref(polynomials))
            .collect();

        let num_rounds_x = key.num_rows_total().log_2();
        let num_rounds_y = key.num_cols_total().log_2();

        // outer sum-check
        let tau = (0..num_rounds_x)
            .map(|_i| transcript.challenge_scalar())
            .collect::<Vec<F>>();
        let mut eq_tau = SplitEqPolynomial::new(&tau);

        let (mut az, mut bz, mut cz) =
            constraint_builder.compute_spartan_Az_Bz_Cz::<PCS, ProofTranscript>(&flattened_polys);

        let (outer_sumcheck_proof, outer_sumcheck_r, outer_sumcheck_claims) =
            SumcheckInstanceProof::prove_spartan_cubic(
                &F::zero(), // claim is zero
                num_rounds_x,
                &mut eq_tau,
                &mut az,
                &mut bz,
                &mut cz,
                transcript,
            );
        let outer_sumcheck_r: Vec<F> = outer_sumcheck_r.into_iter().rev().collect();
        drop_in_background_thread((az, bz, cz, eq_tau));

        ProofTranscript::append_scalars(transcript, &outer_sumcheck_claims);
        // claims from the end of sum-check
        // claim_Az is the (scalar) value v_A = \sum_y A(r_x, y) * z(r_x) where r_x is the sumcheck randomness
        let (claim_Az, claim_Bz, claim_Cz): (F, F, F) = (
            outer_sumcheck_claims[0],
            outer_sumcheck_claims[1],
            outer_sumcheck_claims[2],
        );

        // inner sum-check
        let r_inner_sumcheck_RLC: F = transcript.challenge_scalar();
        let claim_inner_joint = claim_Az
            + r_inner_sumcheck_RLC * claim_Bz
            + r_inner_sumcheck_RLC * r_inner_sumcheck_RLC * claim_Cz;

        // this is the polynomial extended from the vector r_A * A(r_x, y) + r_B * B(r_x, y) + r_C * C(r_x, y) for all y
        let num_steps_bits = constraint_builder
            .uniform_repeat()
            .next_power_of_two()
            .ilog2();
        let (rx_con, rx_ts) =
            outer_sumcheck_r.split_at(outer_sumcheck_r.len() - num_steps_bits as usize);
        let mut poly_ABC =
            DensePolynomial::new(key.evaluate_r1cs_mle_rlc(rx_con, rx_ts, r_inner_sumcheck_RLC));

        let (inner_sumcheck_proof, inner_sumcheck_r, _claims_inner) =
            SumcheckInstanceProof::prove_spartan_quadratic(
                &claim_inner_joint, // r_A * v_A + r_B * v_B + r_C * v_C
                num_rounds_y,
                &mut poly_ABC, // r_A * A(r_x, y) + r_B * B(r_x, y) + r_C * C(r_x, y) for all y
                &flattened_polys,
                transcript,
            );
        drop_in_background_thread(poly_ABC);

        // Requires 'r_col_segment_bits' to index the (const, segment). Within that segment we index the step using 'r_col_step'
        let r_col_segment_bits = key.uniform_r1cs.num_vars.next_power_of_two().log_2() + 1;
        let r_col_step = &inner_sumcheck_r[r_col_segment_bits..];

        let chi = EqPolynomial::evals(r_col_step);
        let claimed_witness_evals: Vec<_> = flattened_polys
            .par_iter()
            .map(|poly| poly.evaluate_at_chi_low_optimized(&chi))
            .collect();

        opening_accumulator.append(
            &flattened_polys,
            DensePolynomial::new(chi),
            r_col_step.to_vec(),
            &claimed_witness_evals.iter().collect::<Vec<_>>(),
            transcript,
        );

        // Outer sumcheck claims: [A(r_x), B(r_x), C(r_x)]
        let outer_sumcheck_claims = (
            outer_sumcheck_claims[0],
            outer_sumcheck_claims[1],
            outer_sumcheck_claims[2],
        );
        Ok(UniformSpartanProof {
            _inputs: PhantomData,
            outer_sumcheck_proof,
            outer_sumcheck_claims,
            inner_sumcheck_proof,
            claimed_witness_evals,
            _marker: PhantomData,
        })
    }

    #[tracing::instrument(skip_all, name = "Spartan::verify")]
    pub fn verify<PCS>(
        &self,
        key: &UniformSpartanKey<C, I, F>,
        commitments: &JoltCommitments<PCS, ProofTranscript>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), SpartanError>
    where
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        ProofTranscript: Transcript,
    {
        let num_rounds_x = key.num_rows_total().log_2();
        let num_rounds_y = key.num_cols_total().log_2();

        // outer sum-check
        let tau = (0..num_rounds_x)
            .map(|_i| transcript.challenge_scalar())
            .collect::<Vec<F>>();

        let (claim_outer_final, r_x) = self
            .outer_sumcheck_proof
            .verify(F::zero(), num_rounds_x, 3, transcript)
            .map_err(|_| SpartanError::InvalidOuterSumcheckProof)?;

        // Outer sumcheck is bound from the top, reverse the fiat shamir randomness
        let r_x: Vec<F> = r_x.into_iter().rev().collect();

        // verify claim_outer_final
        let (claim_Az, claim_Bz, claim_Cz) = self.outer_sumcheck_claims;
        let taus_bound_rx = EqPolynomial::new(tau).evaluate(&r_x);
        let claim_outer_final_expected = taus_bound_rx * (claim_Az * claim_Bz - claim_Cz);
        if claim_outer_final != claim_outer_final_expected {
            return Err(SpartanError::InvalidOuterSumcheckClaim);
        }

        transcript.append_scalars(
            [
                self.outer_sumcheck_claims.0,
                self.outer_sumcheck_claims.1,
                self.outer_sumcheck_claims.2,
            ]
            .as_slice(),
        );

        // inner sum-check
        let r_inner_sumcheck_RLC: F = transcript.challenge_scalar();
        let claim_inner_joint = self.outer_sumcheck_claims.0
            + r_inner_sumcheck_RLC * self.outer_sumcheck_claims.1
            + r_inner_sumcheck_RLC * r_inner_sumcheck_RLC * self.outer_sumcheck_claims.2;

        let (claim_inner_final, inner_sumcheck_r) = self
            .inner_sumcheck_proof
            .verify(claim_inner_joint, num_rounds_y, 2, transcript)
            .map_err(|_| SpartanError::InvalidInnerSumcheckProof)?;

        // n_prefix = n_segments + 1
        let n_prefix = key.uniform_r1cs.num_vars.next_power_of_two().log_2() + 1;

        let eval_Z = key.evaluate_z_mle(&self.claimed_witness_evals, &inner_sumcheck_r);

        let r_y = inner_sumcheck_r.clone();
        let r = [r_x, r_y].concat();
        let (eval_a, eval_b, eval_c) = key.evaluate_r1cs_matrix_mles(&r);

        let left_expected = eval_a
            + r_inner_sumcheck_RLC * eval_b
            + r_inner_sumcheck_RLC * r_inner_sumcheck_RLC * eval_c;
        let right_expected = eval_Z;
        let claim_inner_final_expected = left_expected * right_expected;
        if claim_inner_final != claim_inner_final_expected {
            return Err(SpartanError::InvalidInnerSumcheckClaim);
        }

        let flattened_commitments: Vec<_> = I::flatten::<C>()
            .iter()
            .map(|var| var.get_ref(commitments))
            .collect();
        let r_y_point = &inner_sumcheck_r[n_prefix..];
        opening_accumulator.append(
            &flattened_commitments,
            r_y_point.to_vec(),
            &self.claimed_witness_evals.iter().collect::<Vec<_>>(),
            transcript,
        );

        Ok(())
    }
}

// #[cfg(test)]
// mod test {
//     use ark_bn254::Fr;
//     use ark_std::One;

//     use crate::poly::commitment::{commitment_scheme::CommitShape, hyrax::HyraxScheme};

//     use super::*;

//     #[test]
//     fn integration() {
//         let (builder, key) = simp_test_builder_key();
//         let witness_segments: Vec<Vec<Fr>> = vec![
//             vec![Fr::one(), Fr::from(5), Fr::from(9), Fr::from(13)], /* Q */
//             vec![Fr::one(), Fr::from(5), Fr::from(9), Fr::from(13)], /* R */
//             vec![Fr::one(), Fr::from(5), Fr::from(9), Fr::from(13)], /* S */
//         ];

//         // Create a witness and commit
//         let witness_segments_ref: Vec<&[Fr]> = witness_segments
//             .iter()
//             .map(|segment| segment.as_slice())
//             .collect();
//         let gens = HyraxScheme::setup(&[CommitShape::new(16, BatchType::Small)]);
//         let witness_commitment =
//             HyraxScheme::batch_commit(&witness_segments_ref, &gens, BatchType::Small);

//         // Prove spartan!
//         let mut prover_transcript = ProofTranscript::new(b"stuff");
//         let proof =
//             UniformSpartanProof::<Fr, HyraxScheme<ark_bn254::G1Projective>>::prove_precommitted::<
//                 SimpTestIn,
//             >(
//                 &gens,
//                 builder,
//                 &key,
//                 witness_segments,
//                 todo!("opening accumulator"),
//                 &mut prover_transcript,
//             )
//             .unwrap();

//         let mut verifier_transcript = ProofTranscript::new(b"stuff");
//         let witness_commitment_ref: Vec<&_> = witness_commitment.iter().collect();
//         proof
//             .verify_precommitted(
//                 &key,
//                 witness_commitment_ref,
//                 &gens,
//                 &mut verifier_transcript,
//             )
//             .expect("Spartan verifier failed");
//     }
// }
