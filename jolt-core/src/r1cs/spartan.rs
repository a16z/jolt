#![allow(clippy::len_without_is_empty)]

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::BatchType;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::r1cs::key::UniformSpartanKey;
use crate::utils::compute_dotproduct_low_optimized;
use crate::utils::math::Math;
use crate::utils::thread::drop_in_background_thread;

use crate::utils::transcript::ProofTranscript;
use ark_serialize::CanonicalDeserialize;
use ark_serialize::CanonicalSerialize;
use rayon::prelude::*;

use thiserror::Error;

use crate::{
    poly::{dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial},
    subprotocols::sumcheck::SumcheckInstanceProof,
};

use super::builder::CombinedUniformBuilder;
use super::ops::ConstraintInput;

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

// TODO: Rather than use these adhoc virtual indexable polys – create a DensePolynomial which takes any impl Index<usize> inner
// and can run all the normal DensePolynomial ops.
#[derive(Clone)]
pub struct SegmentedPaddedWitness<F: JoltField> {
    total_len: usize,
    segments: Vec<Vec<F>>,
    segment_len: usize,
    zero: F,
}

impl<F: JoltField> SegmentedPaddedWitness<F> {
    pub fn new(total_len: usize, segments: Vec<Vec<F>>) -> Self {
        let segment_len = segments[0].len();
        assert!(segment_len.is_power_of_two());
        for segment in &segments {
            assert_eq!(
                segment.len(),
                segment_len,
                "All segments must be the same length"
            );
        }
        SegmentedPaddedWitness {
            total_len,
            segments,
            segment_len,
            zero: F::zero(),
        }
    }

    pub fn len(&self) -> usize {
        self.total_len
    }

    #[tracing::instrument(skip_all, name = "SegmentedPaddedWitness::evaluate_all")]
    pub fn evaluate_all(&self, point: Vec<F>) -> Vec<F> {
        let chi = EqPolynomial::evals(&point);
        assert!(chi.len() >= self.segment_len);

        let evals = self
            .segments
            .par_iter()
            .map(|segment| compute_dotproduct_low_optimized(&chi[0..self.segment_len], segment))
            .collect();
        drop_in_background_thread(chi);
        evals
    }

    pub fn into_dense_polys(self) -> Vec<DensePolynomial<F>> {
        self.segments
            .into_iter()
            .map(|poly| DensePolynomial::new(poly))
            .collect()
    }
}

impl<F: JoltField> std::ops::Index<usize> for SegmentedPaddedWitness<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.segments.len() * self.segment_len {
            &self.zero
        } else if index >= self.total_len {
            panic!("index too high");
        } else {
            let segment_index = index / self.segment_len;
            let inner_index = index % self.segment_len;
            &self.segments[segment_index][inner_index]
        }
    }
}

impl<F: JoltField> IndexablePoly<F> for SegmentedPaddedWitness<F> {
    fn len(&self) -> usize {
        self.total_len
    }
}

pub trait IndexablePoly<F: JoltField>: std::ops::Index<usize, Output = F> + Sync {
    fn len(&self) -> usize;
}

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct UniformSpartanProof<F: JoltField, C: CommitmentScheme<Field = F>> {
    outer_sumcheck_proof: SumcheckInstanceProof<F>,
    outer_sumcheck_claims: (F, F, F),
    inner_sumcheck_proof: SumcheckInstanceProof<F>,
    claimed_witness_evals: Vec<F>,
    opening_proof: C::BatchedProof,
}

impl<F: JoltField, C: CommitmentScheme<Field = F>> UniformSpartanProof<F, C> {
    #[tracing::instrument(skip_all, name = "UniformSpartanProof::setup_precommitted")]
    pub fn setup_precommitted<I: ConstraintInput>(
        constraint_builder: &CombinedUniformBuilder<F, I>,
        padded_num_steps: usize,
    ) -> UniformSpartanKey<F> {
        assert_eq!(
            padded_num_steps,
            constraint_builder.uniform_repeat().next_power_of_two()
        );
        UniformSpartanKey::from_builder(constraint_builder)
    }

    /// produces a succinct proof of satisfiability of a `RelaxedR1CS` instance
    #[tracing::instrument(skip_all, name = "UniformSpartanProof::prove_precommitted")]
    pub fn prove_precommitted<I: ConstraintInput>(
        generators: &C::Setup,
        constraint_builder: CombinedUniformBuilder<F, I>,
        key: &UniformSpartanKey<F>,
        witness_segments: Vec<Vec<F>>,
        transcript: &mut ProofTranscript,
    ) -> Result<Self, SpartanError> {
        assert_eq!(witness_segments.len(), key.uniform_r1cs.num_vars);
        witness_segments
            .iter()
            .for_each(|segment| assert_eq!(segment.len(), key.num_steps));

        let segmented_padded_witness =
            SegmentedPaddedWitness::new(key.num_vars_total(), witness_segments);

        let num_rounds_x = key.num_rows_total().log_2();
        let num_rounds_y = key.num_cols_total().log_2();

        // outer sum-check
        let tau = (0..num_rounds_x)
            .map(|_i| transcript.challenge_scalar(b"t"))
            .collect::<Vec<F>>();
        let mut poly_tau = DensePolynomial::new(EqPolynomial::evals(&tau));

        let inputs = &segmented_padded_witness.segments[0..I::COUNT];
        let aux = &segmented_padded_witness.segments[I::COUNT..];
        let (az, bz, cz) = constraint_builder.compute_spartan_Az_Bz_Cz(inputs, aux);
        // TODO: Do not require these padded, Sumcheck should handle sparsity.
        assert!(az.len().is_power_of_two());
        assert!(bz.len().is_power_of_two());
        assert!(cz.len().is_power_of_two());

        let mut poly_Az = DensePolynomial::new(az);
        let mut poly_Bz = DensePolynomial::new(bz);
        let mut poly_Cz = DensePolynomial::new(cz);

        #[cfg(test)]
        {
            // Check that Z is a satisfying assignment
            for (i, ((az, bz), cz)) in poly_Az
                .evals_ref()
                .iter()
                .zip(poly_Bz.evals_ref())
                .zip(poly_Cz.evals_ref())
                .enumerate()
            {
                if *az * *bz != *cz {
                    let padded_segment_len = segmented_padded_witness.segment_len;
                    let error_segment_index = i / padded_segment_len;
                    let error_step_index = i % padded_segment_len;
                    panic!("witness is not a satisfying assignment. Failed on segment {error_segment_index} at step {error_step_index}");
                }
            }
        }

        let comb_func_outer = |A: &F, B: &F, C: &F, D: &F| -> F {
            // Below is an optimized form of: *A * (*B * *C - *D)
            if B.is_zero() || C.is_zero() {
                if D.is_zero() {
                    F::zero()
                } else {
                    *A * (-(*D))
                }
            } else {
                *A * (*B * *C - *D)
            }
        };

        let (outer_sumcheck_proof, outer_sumcheck_r, outer_sumcheck_claims) =
            SumcheckInstanceProof::prove_spartan_cubic::<_>(
                &F::zero(), // claim is zero
                num_rounds_x,
                &mut poly_tau,
                &mut poly_Az,
                &mut poly_Bz,
                &mut poly_Cz,
                comb_func_outer,
                transcript,
            );
        drop_in_background_thread(poly_Az);
        drop_in_background_thread(poly_Bz);
        drop_in_background_thread(poly_Cz);
        drop_in_background_thread(poly_tau);

        // claims from the end of sum-check
        // claim_Az is the (scalar) value v_A = \sum_y A(r_x, y) * z(r_x) where r_x is the sumcheck randomness
        let (claim_Az, claim_Bz, claim_Cz): (F, F, F) = (
            outer_sumcheck_claims[1],
            outer_sumcheck_claims[2],
            outer_sumcheck_claims[3],
        );
        ProofTranscript::append_scalars(
            transcript,
            b"claims_outer",
            [claim_Az, claim_Bz, claim_Cz].as_slice(),
        );

        // inner sum-check
        let r_inner_sumcheck_RLC: F = transcript.challenge_scalar(b"r");
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
            SumcheckInstanceProof::prove_spartan_quadratic::<SegmentedPaddedWitness<F>>(
                &claim_inner_joint, // r_A * v_A + r_B * v_B + r_C * v_C
                num_rounds_y,
                &mut poly_ABC, // r_A * A(r_x, y) + r_B * B(r_x, y) + r_C * C(r_x, y) for all y
                &segmented_padded_witness,
                transcript,
            );
        drop_in_background_thread(poly_ABC);

        // Requires 'r_col_segment_bits' to index the (const, segment). Within that segment we index the step using 'r_col_step'
        let r_col_segment_bits = key.uniform_r1cs.num_vars.next_power_of_two().log_2() + 1;
        let r_col_step = &inner_sumcheck_r[r_col_segment_bits..];
        let witness_evals = segmented_padded_witness.evaluate_all(r_col_step.to_owned());

        let witness_segment_polys: Vec<DensePolynomial<F>> =
            segmented_padded_witness.into_dense_polys();
        let witness_segment_polys_ref: Vec<&DensePolynomial<F>> =
            witness_segment_polys.iter().collect();
        let opening_proof = C::batch_prove(
            generators,
            &witness_segment_polys_ref,
            r_col_step,
            &witness_evals,
            BatchType::Big,
            transcript,
        );

        drop_in_background_thread(witness_segment_polys);

        // Outer sumcheck claims: [eq(r_x), A(r_x), B(r_x), C(r_x)]
        let outer_sumcheck_claims = (
            outer_sumcheck_claims[1],
            outer_sumcheck_claims[2],
            outer_sumcheck_claims[3],
        );
        Ok(UniformSpartanProof {
            outer_sumcheck_proof,
            outer_sumcheck_claims,
            inner_sumcheck_proof,
            claimed_witness_evals: witness_evals,
            opening_proof,
        })
    }

    #[tracing::instrument(skip_all, name = "SNARK::verify")]
    /// verifies a proof of satisfiability of a `RelaxedR1CS` instance
    pub fn verify_precommitted(
        &self,
        key: &UniformSpartanKey<F>,
        witness_segment_commitments: Vec<&C::Commitment>,
        generators: &C::Setup,
        transcript: &mut ProofTranscript,
    ) -> Result<(), SpartanError> {
        let num_rounds_x = key.num_rows_total().log_2();
        let num_rounds_y = key.num_cols_total().log_2();

        // outer sum-check
        let tau = (0..num_rounds_x)
            .map(|_i| transcript.challenge_scalar(b"t"))
            .collect::<Vec<F>>();

        let (claim_outer_final, r_x) = self
            .outer_sumcheck_proof
            .verify(F::zero(), num_rounds_x, 3, transcript)
            .map_err(|_| SpartanError::InvalidOuterSumcheckProof)?;

        // verify claim_outer_final
        let (claim_Az, claim_Bz, claim_Cz) = self.outer_sumcheck_claims;
        let taus_bound_rx = EqPolynomial::new(tau).evaluate(&r_x);
        let claim_outer_final_expected = taus_bound_rx * (claim_Az * claim_Bz - claim_Cz);
        if claim_outer_final != claim_outer_final_expected {
            return Err(SpartanError::InvalidOuterSumcheckClaim);
        }

        transcript.append_scalars(
            b"claims_outer",
            [
                self.outer_sumcheck_claims.0,
                self.outer_sumcheck_claims.1,
                self.outer_sumcheck_claims.2,
            ]
            .as_slice(),
        );

        // inner sum-check
        let r_inner_sumcheck_RLC: F = transcript.challenge_scalar(b"r");
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

        let r_y_point = &inner_sumcheck_r[n_prefix..];
        C::batch_verify(
            &self.opening_proof,
            generators,
            r_y_point,
            &self.claimed_witness_evals,
            &witness_segment_commitments,
            transcript,
        )
        .map_err(|_| SpartanError::InvalidPCSProof)?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::{
        poly::commitment::{commitment_scheme::CommitShape, hyrax::HyraxScheme},
        r1cs::test::{simp_test_builder_key, SimpTestIn},
    };

    use super::*;

    #[test]
    fn integration() {
        let (builder, key) = simp_test_builder_key();
        let witness_segments: Vec<Vec<Fr>> = vec![
            vec![Fr::one(), Fr::from(5), Fr::from(9), Fr::from(13)], /* Q */
            vec![Fr::one(), Fr::from(5), Fr::from(9), Fr::from(13)], /* R */
            vec![Fr::one(), Fr::from(5), Fr::from(9), Fr::from(13)], /* S */
        ];

        // Create a witness and commit
        let witness_segments_ref: Vec<&[Fr]> = witness_segments
            .iter()
            .map(|segment| segment.as_slice())
            .collect();
        let gens = HyraxScheme::setup(&[CommitShape::new(16, BatchType::Small)]);
        let witness_commitment =
            HyraxScheme::batch_commit(&witness_segments_ref, &gens, BatchType::Small);

        // Prove spartan!
        let mut prover_transcript = ProofTranscript::new(b"stuff");
        let proof =
            UniformSpartanProof::<Fr, HyraxScheme<ark_bn254::G1Projective>>::prove_precommitted::<
                SimpTestIn,
            >(
                &gens,
                builder,
                &key,
                witness_segments,
                &mut prover_transcript,
            )
            .unwrap();

        let mut verifier_transcript = ProofTranscript::new(b"stuff");
        let witness_commitment_ref: Vec<&_> = witness_commitment.iter().collect();
        proof
            .verify_precommitted(
                &key,
                witness_commitment_ref,
                &gens,
                &mut verifier_transcript,
            )
            .expect("Spartan verifier failed");
    }
}
