#![allow(clippy::len_without_is_empty)]

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::BatchType;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::r1cs::key::UniformSpartanKey;
use crate::r1cs::special_polys::SegmentedPaddedWitness;
use crate::utils::index_to_field_bitvector;
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
            .map(|_i| transcript.challenge_scalar())
            .collect::<Vec<F>>();
        let mut poly_tau = DensePolynomial::new(EqPolynomial::evals(&tau));

        let inputs = &segmented_padded_witness.segments[0..I::COUNT];
        let aux = &segmented_padded_witness.segments[I::COUNT..];
        let (mut az, mut bz, mut cz) = constraint_builder.compute_spartan_Az_Bz_Cz(inputs, aux);

        let comb_func_outer = |eq: &F, az: &F, bz: &F, cz: &F| -> F {
            // Below is an optimized form of: *A * (*B * *C - *D)
            if az.is_zero() || bz.is_zero() {
                if cz.is_zero() {
                    F::zero()
                } else {
                    *eq * (-(*cz))
                }
            } else {
                let inner = *az * *bz - *cz;
                if inner.is_zero() {
                    F::zero()
                } else {
                    *eq * inner
                }
            }
        };

        let (outer_sumcheck_proof, outer_sumcheck_r, outer_sumcheck_claims) =
            SumcheckInstanceProof::prove_spartan_cubic::<_>(
                &F::zero(), // claim is zero
                num_rounds_x,
                &mut poly_tau,
                &mut az,
                &mut bz,
                &mut cz,
                comb_func_outer,
                transcript,
            );
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
        println!("Prover claim_inner_joint: {claim_inner_joint:?}");
        println!("claim_Az: {claim_Az:?}");
        println!("claim_Bz: {claim_Bz:?}");
        println!("claim_Cz: {claim_Cz:?}");


        // TODO(sragss):
        // - Variables for bit lens + magnitudes
        // - Concat Z
        // - Eval MLE
        // - Sumcheck (arbitrary)
        let height = constraint_builder.constraint_rows();
        let num_constraints = constraint_builder.num_constraints();
        let num_steps = constraint_builder.uniform_repeat();
        let num_vars = constraint_builder.num_vars();
        assert_eq!(height.next_power_of_two(), num_constraints.next_power_of_two() * num_steps.next_power_of_two());
        let width = (num_vars.next_power_of_two() * num_steps.next_power_of_two()) * 2;
        let num_constraints_bits = num_constraints.log_2();
        let r_x_step = &outer_sumcheck_r[num_constraints_bits..];

        let mut z: Vec<F> = segmented_padded_witness.clone().segments.into_iter().map(|mut segment| {
            segment.resize(segment.len().next_power_of_two(), F::zero());
            segment
        }).flatten().collect();
        z.resize(z.len().next_power_of_two(), F::zero());

        let mut poly_z = DensePolynomial::new(z);
        println!("r_x_step {r_x_step:?}");
        // poly_z.bound_poly_var_bot(&F::zero());
        for r_s in r_x_step.iter().rev() {
            poly_z.bound_poly_var_bot(r_s);
        }
        let mut evals = poly_z.evals();
        evals.push(F::one());
        evals.resize(evals.len().next_power_of_two(), F::zero());
        poly_z = DensePolynomial::new(evals);

        println!("poly_z, {poly_z:?}");
        assert_eq!(poly_z.len(), num_vars.next_power_of_two() * 2);

        // this is the polynomial extended from the vector r_A * A(r_x, y) + r_B * B(r_x, y) + r_C * C(r_x, y) for all y
        let num_steps_bits = constraint_builder
            .uniform_repeat()
            .next_power_of_two()
            .ilog2();
        let (rx_con, rx_ts) =
            outer_sumcheck_r.split_at(outer_sumcheck_r.len() - num_steps_bits as usize);
        let poly_ABC =
            DensePolynomial::new(key.evaluate_r1cs_mle_rlc(rx_con, rx_ts, r_inner_sumcheck_RLC));
        assert_eq!(poly_z.len(), poly_ABC.len());

        // RLC_ABC(r_x_constraint, r_x_step, r_y_const, r_y_variable, r_x_step)
        // Degrees of freedom: r_y_const, r_y_variable, so evaluate over the boolean hypercube
        // TODO(sragss): Test more of the points on polyABC vs full evaluation.
        // println!("Prover RLC_ABC(rx_con, rx_ts) -> rx_con = {rx_con:?}, rx_ts = {rx_ts:?}");
        // println!("Prover polyABC[0,..,0]= {:?}", poly_ABC[0]);
        // println!("Prover polyABC[1]= {:?}", poly_ABC[1]);
        // println!("Prover polyABC[1,0,..,0]= {:?}", poly_ABC[poly_ABC.len() / 2]);
        // println!("Prover polyABC[0,1,..,1]= {:?}", poly_ABC[(poly_ABC.len() / 2) - 1]);
        // println!("Prover polyABC[1,..,1]= {:?}", poly_ABC[poly_ABC.len() - 1]);

        // TODO(sragss): z could be wrong!!!! EVALUATE ALL OF Z

        let num_rounds = (num_vars.next_power_of_two() * 2).log_2();
        let mut polys = vec![poly_ABC, poly_z];
        let comb_func = |stuff: &[F]| -> F {
            assert_eq!(stuff.len(), 2);
            stuff[0] * stuff[1]
        };
        let (inner_sumcheck_proof, inner_sumcheck_r, _claims_inner) =
            SumcheckInstanceProof::prove_arbitrary(
                &claim_inner_joint, 
                num_rounds, 
                &mut polys, 
                comb_func, 
                2, 
                transcript);
        println!("Prover inner_sumcheck_r: {inner_sumcheck_r:?}");
        println!("Prover RLC(x,y'): {:?}", _claims_inner[0]);
        println!("Prover Z(y'): {:?}", _claims_inner[1]);
        println!("Prove claims combined: {:?}", _claims_inner[0] * _claims_inner[1]);
        // TODO(sragss): Why is inner_sumcheck_r unused
        

        // let (inner_sumcheck_proof, inner_sumcheck_r, _claims_inner) =
        //     SumcheckInstanceProof::prove_spartan_quadratic::<SegmentedPaddedWitness<F>>(
        //         &claim_inner_joint, // r_A * v_A + r_B * v_B + r_C * v_C
        //         num_rounds_y,
        //         &mut poly_ABC, // r_A * A(r_x, y) + r_B * B(r_x, y) + r_C * C(r_x, y) for all y
        //         &segmented_padded_witness,
        //         transcript,
        //     );
        drop_in_background_thread(polys);

        // Requires 'r_col_segment_bits' to index the (const, segment). Within that segment we index the step using 'r_col_step'
        // let r_col_segment_bits = key.uniform_r1cs.num_vars.next_power_of_two().log_2() + 1;
        // let r_col_step = &inner_sumcheck_r[r_col_segment_bits..];
        let r_z = r_x_step;
        let witness_evals = segmented_padded_witness.evaluate_all(r_z.to_owned());

        let witness_segment_polys: Vec<DensePolynomial<F>> =
            segmented_padded_witness.into_dense_polys();
        let witness_segment_polys_ref: Vec<&DensePolynomial<F>> =
            witness_segment_polys.iter().collect();
        let opening_proof = C::batch_prove(
            generators,
            &witness_segment_polys_ref,
            r_z,
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
        println!("Verifier claim_inner_joint: {claim_inner_joint:?}");

        let num_rounds = (key.num_vars() * 2).next_power_of_two().log_2();
        // TODO(sragss): claim_inner_final does not agree with the self calculation nor the prover claim
        let (claim_inner_final, inner_sumcheck_r) = self
            .inner_sumcheck_proof
            .verify(claim_inner_joint, num_rounds, 2, transcript)
            .map_err(|_| SpartanError::InvalidInnerSumcheckProof)?;
        println!("Verifier inner_sumcheck_r {inner_sumcheck_r:?}");

        // n_prefix = n_segments + 1
        let n_prefix = key.uniform_r1cs.num_vars.next_power_of_two().log_2() + 1;

        // TODO(sragss): Prepend outersumcheck r_step
        let constraint_bits = key.uniform_r1cs.num_rows.next_power_of_two().log_2();
        let outer_sumcheck_r_step = &r_x[constraint_bits..];
        let y_prime = [inner_sumcheck_r.to_owned(), outer_sumcheck_r_step.to_owned()].concat();
        let eval_Z = key.evaluate_z_mle(&self.claimed_witness_evals, &y_prime);

        // let r_y = inner_sumcheck_r.clone();
        println!("y_prime.len() {}", y_prime.len());
        // let var_and_const_bits = key.uniform_r1cs.num_vars.next_power_of_two().log_2() + 1;
        // let variable_zero= index_to_field_bitvector(1, var_and_const_bits);
        // let r_0: Vec<F> = [&r_x, &variable_zero, outer_sumcheck_r_step].concat();
        // let (a_0, b_0, c_0) = key.evaluate_r1cs_matrix_mles(&r_0);
        // let rlc = a_0
        //     + r_inner_sumcheck_RLC * b_0
        //     + r_inner_sumcheck_RLC * r_inner_sumcheck_RLC * c_0;
        // println!("Verifier rlc[0] = {rlc:?}");




        let r = [r_x.clone(), y_prime.clone()].concat(); // TODO(sragss): This is questionable at best.
        // TODO(sragss): This is now unhappy with what we're delivering
        let (eval_a, eval_b, eval_c) = key.evaluate_r1cs_matrix_mles(&r);
        println!("Verifier RLC_ABC(r) -> r = {r:?}");

        let left_expected = eval_a
            + r_inner_sumcheck_RLC * eval_b
            + r_inner_sumcheck_RLC * r_inner_sumcheck_RLC * eval_c;
        let right_expected = eval_Z;
        println!("Verifier RLC(x,y'): {left_expected:?}");
        println!("Verifier Z(y'): {right_expected:?}");
        let claim_inner_final_expected = left_expected * right_expected;

        if claim_inner_final != claim_inner_final_expected {
            return Err(SpartanError::InvalidInnerSumcheckClaim);
        }

        println!("LORD ALMIGHTY THE MLE EVALUATION BY THE VERIFIER PASSED"); 

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
    use ark_std::One;

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
