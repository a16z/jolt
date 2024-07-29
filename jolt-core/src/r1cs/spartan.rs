#![allow(clippy::len_without_is_empty)]

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::BatchType;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::r1cs::key::UniformSpartanKey;
use crate::r1cs::special_polys::SegmentedPaddedWitness;
use crate::subprotocols::sumcheck::CurveSpartanSumcheckBackend;
use crate::subprotocols::sumcheck::SpartanSumcheckBackend;
use crate::utils::math::Math;
use crate::utils::thread::drop_in_background_thread;

use crate::utils::transcript::ProofTranscript;
use ark_serialize::CanonicalDeserialize;
use ark_serialize::CanonicalSerialize;

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
    pub fn prove_precommitted<I: ConstraintInput, S: SpartanSumcheckBackend<F>>(
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
            .map(|_i| transcript.challenge_scalar())
            .collect::<Vec<F>>();
        let mut poly_tau = DensePolynomial::new(EqPolynomial::evals(&tau));

        let inputs = &segmented_padded_witness.segments[0..I::COUNT];
        let aux = &segmented_padded_witness.segments[I::COUNT..];
        let (mut az, mut bz, mut cz) = constraint_builder.compute_spartan_Az_Bz_Cz(inputs, aux);

        let (outer_sumcheck_proof, outer_sumcheck_r, outer_sumcheck_claims) = S::prove_spartan_cubic(&F::zero(), num_rounds_x, &mut poly_tau, &mut az, &mut bz, &mut cz, transcript);
        let outer_sumcheck_r: Vec<F> = outer_sumcheck_r.into_iter().rev().collect();
        drop_in_background_thread((az, bz, cz, poly_tau));

        // claims from the end of sum-check
        // claim_Az is the (scalar) value v_A = \sum_y A(r_x, y) * z(r_x) where r_x is the sumcheck randomness
        let (claim_Az, claim_Bz, claim_Cz): (F, F, F) = (
            outer_sumcheck_claims[1],
            outer_sumcheck_claims[2],
            outer_sumcheck_claims[3],
        );
        ProofTranscript::append_scalars(transcript, [claim_Az, claim_Bz, claim_Cz].as_slice());

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

        let (inner_sumcheck_proof, inner_sumcheck_r, _claims_inner) = S::prove_spartan_quadratic(&claim_inner_joint, num_rounds_y, &mut poly_ABC, &segmented_padded_witness, transcript);
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

pub mod bench {
    use ark_std::{log2, test_rng};

    use crate::r1cs::special_polys::SparsePolynomial;

    use super::*;

    pub fn cubic_sumcheck<F: JoltField, PCS: CommitmentScheme<Field = F>, S: SpartanSumcheckBackend<F>>(az: Vec<u64>, bz: Vec<u64>, cz: Vec<u64>){
        // Test sum_{x \ in {0,1}^n}{eq(r, x) * [Az(x) * Bz(x) - Cz(x)} = 0

        // 1. Compute eq_table = eq(r, _)
        // 2. Assign Az, Bz, Cz
        // 3. Test Az * Bz - Cz == 0
        // 4. Sumcheck
        //      - Evaluate at {0, 1, 2, 3}
        //      - Bind
        // 5. Sumcheck verifier

        let len: usize = az.len();
        assert_eq!(bz.len(), len);
        assert_eq!(cz.len(), len);
        let log_len: usize = log2(len) as usize;


        let mut rng = test_rng();
        let r_eq: Vec<F> = (0..log_len).into_iter().map(|_| F::random(&mut rng)).collect();

        // 1. Compute eq_table = eq(r, _)
        let mut tau = DensePolynomial::new(EqPolynomial::evals(&r_eq));

        // 2. Assign Az, Bz, Cz
        let az = az.into_iter().map(|item| F::from_u64(item).unwrap()).collect();
        let bz = bz.into_iter().map(|item| F::from_u64(item).unwrap()).collect();
        let cz = cz.into_iter().map(|item| F::from_u64(item).unwrap()).collect();
        let sparsify = |vec: Vec<F>| -> Vec<(F, usize)> {
            vec.into_iter().enumerate().flat_map(|(index, item)| {
                if item.is_zero() {
                    None
                } else {
                    Some((item, index))
                }
            }).collect()
        };
        let mut az_poly = SparsePolynomial::new(log_len, sparsify(az));
        let mut bz_poly = SparsePolynomial::new(log_len, sparsify(bz) );
        let mut cz_poly = SparsePolynomial::new(log_len, sparsify(cz));

        let v_tau = tau.clone();
        let v_az_poly = az_poly.clone().to_dense();
        let v_bz_poly = bz_poly.clone().to_dense();
        let v_cz_poly = cz_poly.clone().to_dense();

        // 3. Test Az * Bz - Cz == 0
        for ((az, bz), cz) in v_az_poly.evals_ref().iter().zip(v_bz_poly.evals_ref()).zip(v_cz_poly.evals_ref()) {
            assert_eq!(*az * bz, *cz);
        }

        // 4. Sumcheck
        //      - Evaluate at {0, 1, 2, 3}
        //      - Bind

        let mut transcript = ProofTranscript::new(b"test");
        let (outer_sumcheck_proof, outer_sumcheck_r, outer_sumcheck_claims) = S::prove_spartan_cubic(
                &F::from_u64(0).unwrap(),
                log_len,
                &mut tau,
                &mut az_poly,
                &mut bz_poly,
                &mut cz_poly,
                &mut transcript,
            );
        assert_eq!(outer_sumcheck_claims.len(), 4);

        let mut verify_transcript = ProofTranscript::new(b"test");
        let (f_r, r) = outer_sumcheck_proof.verify(F::from_u64(0).unwrap(), log_len, 3, &mut verify_transcript).unwrap();
        assert_eq!(outer_sumcheck_r, r);
        assert_eq!(outer_sumcheck_claims.len(), 4);

        let rev_r: Vec<F> = r.clone().into_iter().rev().collect();
        let tau_r = v_tau.evaluate(&rev_r);
        let az_r = v_az_poly.evaluate(&rev_r);
        let bz_r = v_bz_poly.evaluate(&rev_r);
        let cz_r = v_cz_poly.evaluate(&rev_r);
        assert_eq!(outer_sumcheck_claims[0], tau_r);
        assert_eq!(outer_sumcheck_claims[1], az_r);
        assert_eq!(outer_sumcheck_claims[2], bz_r);
        assert_eq!(outer_sumcheck_claims[3], cz_r);
        let verifier_eval = tau_r * (az_r * bz_r - cz_r);

        assert_eq!(f_r, verifier_eval);
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_std::{test_rng, One};

    use binius_field::BinaryField128bPolyval as BF;

    use crate::{
        field::binius::BiniusField, poly::commitment::{commitment_scheme::CommitShape, hyrax::HyraxScheme, mock::MockCommitScheme}, r1cs::{special_polys::SparsePolynomial, test::{simp_test_builder_key, SimpTestIn}}, subprotocols::sumcheck::BiniusSpartanSumcheckBackend
    };

    use super::*;

    fn integration_test<F: JoltField, PCS: CommitmentScheme<Field = F>, S: SpartanSumcheckBackend<F>>() {
        let (builder, key) = simp_test_builder_key();
        let witness_segments: Vec<Vec<F>> = vec![
            vec![F::one(), F::from_u64(5).unwrap(), F::from_u64(9).unwrap(), F::from_u64(13).unwrap()], /* Q */
            vec![F::one(), F::from_u64(5).unwrap(), F::from_u64(9).unwrap(), F::from_u64(13).unwrap()], /* R */
            vec![F::one(), F::from_u64(5).unwrap(), F::from_u64(9).unwrap(), F::from_u64(13).unwrap()], /* S */
        ];

        // Create a witness and commit
        let witness_segments_ref: Vec<&[F]> = witness_segments
            .iter()
            .map(|segment| segment.as_slice())
            .collect();
        let gens = PCS::setup(&[CommitShape::new(16, BatchType::Small)]);
        let witness_commitment =
            PCS::batch_commit(&witness_segments_ref, &gens, BatchType::Small);

        // Prove spartan!
        let mut prover_transcript = ProofTranscript::new(b"stuff");
        let proof =
            UniformSpartanProof::<F, PCS>::prove_precommitted::<
                SimpTestIn,
                S
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

    #[test]
    fn curve_integration_test() {
        integration_test::<Fr, HyraxScheme<ark_bn254::G1Projective>, CurveSpartanSumcheckBackend>();
    }

    // TODO(sragss): 
    // - Doesn't work because SimpTest non_uniform_constraints includes integer arthmetic
    // - Make seperate working constraint system

    // #[test]
    // fn binius_integration_test() {
    //     type Field = BiniusField<BF>;
    //     integration_test::<Field, MockCommitScheme<Field>, BiniusSpartanSumcheckBackend>();
    // }

    fn cubic_sumcheck_test<F: JoltField, PCS: CommitmentScheme<Field = F>, S: SpartanSumcheckBackend<F>>(){
        const LOG_LEN: usize = 4;
        const LEN: usize = 1 << LOG_LEN;

        let az: Vec<u64> = vec![1; LEN];
        let bz: Vec<u64> = (0..LEN).into_iter().map(|i| i as u64).collect();
        let cz: Vec<u64> = (0..LEN).into_iter().map(|i| i as u64).collect();

        bench::cubic_sumcheck::<F, PCS, S>(az, bz, cz);
    }

    #[test]
    fn curve_cubic_sumcheck_test() {
        cubic_sumcheck_test::<Fr, HyraxScheme<ark_bn254::G1Projective>, CurveSpartanSumcheckBackend>();
    }

    #[test]
    fn binius_cubic_sumcheck_test() {
        type Field = BiniusField<BF>;
        cubic_sumcheck_test::<Field, MockCommitScheme<Field>, BiniusSpartanSumcheckBackend>();
    }
}
