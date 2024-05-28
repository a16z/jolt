#![allow(clippy::len_without_is_empty)]

use crate::poly::commitment::commitment_scheme::BatchType;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::field::JoltField;
use crate::r1cs::spartan::IndexablePoly;
use crate::utils::compute_dotproduct_low_optimized;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::r1cs::new_r1cs::key::UniformSpartanKey;
use crate::utils::transcript::ProofTranscript;
use ark_serialize::CanonicalDeserialize;
use ark_serialize::CanonicalSerialize;
use rayon::prelude::*;
use sha3::Digest;
use sha3::Sha3_256;
use thiserror::Error;
use strum::EnumCount;

use crate::{
    poly::{dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial},
    subprotocols::sumcheck::SumcheckInstanceProof,
};

use super::builder::CombinedUniformBuilder;
use super::jolt_constraints::JoltInputs;
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
pub struct SegmentedPaddedWitness<F: JoltField> {
    total_len: usize,
    segments: Vec<Vec<F>>,
    segment_len: usize,
    zero: F,
}

impl<F: JoltField> SegmentedPaddedWitness<F> {
    pub fn new(total_len: usize, segments: Vec<Vec<F>>) -> Self {
        let segment_len = segments[0].len();
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

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct UniformSpartanProof<F: JoltField, C: CommitmentScheme<Field = F>> {
    outer_sumcheck_proof: SumcheckInstanceProof<F>,
    outer_sumcheck_claims: (F, F, F),
    inner_sumcheck_proof: SumcheckInstanceProof<F>,
    eval_arg: Vec<F>, // TODO(arasuarun / sragss): better name
    claimed_witnesss_evals: Vec<F>,
    opening_proof: C::BatchedProof,
}

impl<F: JoltField, C: CommitmentScheme<Field = F>> UniformSpartanProof<F, C> {
    #[tracing::instrument(skip_all, name = "UniformSpartanProof::setup_precommitted")]
    pub fn setup_precommitted<I: ConstraintInput>(
        constraint_builder: &CombinedUniformBuilder<F, I>,
        padded_num_steps: usize,
    ) -> UniformSpartanKey<F> {
        todo!()
        // assert_eq!(padded_num_steps, constraint_builder.uniform_repeat().next_power_of_two());
        // let single_step_shape = constraint_builder.materialize_uniform();

        // let padded_num_constraints = constraint_builder.constraint_rows().next_power_of_two();
        // let padded_num_vars = constraint_builder.total_columns().next_power_of_two();

        // UniformSpartanKey::construct(
        //     single_step_shape, 
        //     padded_num_constraints, 
        //     padded_num_vars, 
        //     padded_num_steps, 
        //     constraint_builder.uniform_rows(), 
        //     constraint_builder.uniform_columns()
        // )
    }

    /// produces a succinct proof of satisfiability of a `RelaxedR1CS` instance
    #[tracing::instrument(skip_all, name = "UniformSpartanProof::prove_precommitted")]
    pub fn prove_precommitted(
        constraint_builder: CombinedUniformBuilder<F, JoltInputs>,
        witness_segments: Vec<Vec<F>>,
        transcript: &mut ProofTranscript,
    ) -> Result<Self, SpartanError> {
        todo!();
        /*println!("\n UPGRADED SPARTAN");
        // TODO(sragss): Can drop constraint_builder in favor of UniformSpartanKey if we pass in Az, Bz, Cz.
        let mut verifier_transcript = transcript.clone();
        let padded_num_columns = constraint_builder.total_columns().next_power_of_two();
        let padded_num_rows = constraint_builder.constraint_rows().next_power_of_two();
        witness_segments.iter().for_each(|segment| assert_eq!(segment.len(), constraint_builder.uniform_repeat()));
        let poly_ABC_len = 2 * padded_num_columns; // TODO(sragss): Why?? I think this is just a shitty way of making space for const.

        let segmented_padded_witness = SegmentedPaddedWitness::new(padded_num_columns, witness_segments);

        // TODO(sragss): Should x be based on padded_num_rows. Why?
        // TODO(sragss): Arasu decided "x" was vertical. Convert to 'rows / columns'
        let num_rounds_x = padded_num_rows.ilog2() as usize; // (A,B,C) rows
        let num_rounds_y = padded_num_columns.ilog2() as usize; // (A,B,C) cols
        println!("num_rounds_x {num_rounds_x}");
        println!("num_rounds_y {num_rounds_y}");

        // outer sum-check

        let tau = (0..num_rounds_x)
            .map(|_i| transcript.challenge_scalar(b"t"))
            .collect::<Vec<F>>();
        let mut poly_tau = DensePolynomial::new(EqPolynomial::evals(&tau));

        // TODO(sragss): This snippet makes prove JoltInputs dependent.
        let inputs = &segmented_padded_witness.segments[0..JoltInputs::COUNT];
        let aux = &segmented_padded_witness.segments[JoltInputs::COUNT..];
        let (mut az, mut bz, mut cz) = constraint_builder.compute_spartan(&inputs, &aux);

        // TODO(sragss): Explicitly left because this is wasteful. Should likely deal with through more intelligent DensePaddedPolynomial.
        az.resize(padded_num_rows, F::zero());
        bz.resize(padded_num_rows, F::zero());
        cz.resize(padded_num_rows, F::zero());

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
                if *az * bz != *cz {
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


        // VERIFIER VERIFIER VERIFIER VERIFIER VERIFIER
        // TODO(sragss): rm
        // VERIFIER VERIFIER VERIFIER VERIFIER VERIFIER
        let verifier_tau = (0..num_rounds_x)
            .map(|_i| verifier_transcript.challenge_scalar(b"t"))
            .collect::<Vec<F>>();
        assert_eq!(tau, verifier_tau);
        let (claim_outer_final, r_x) = outer_sumcheck_proof
            .verify(F::zero(), num_rounds_x, 3, &mut verifier_transcript)
            .expect("Sumcheck doesn't verify bruv.");
            // .map_err(|_| SpartanError::InvalidOuterSumcheckProof)?;
        assert_eq!(outer_sumcheck_r, r_x);
        // verify claim_outer_final
        verifier_transcript.append_scalars(b"claims_outer", [claim_Az, claim_Bz, claim_Cz].as_slice());
        let taus_bound_rx = EqPolynomial::new(tau).evaluate(&r_x);
        let claim_outer_final_expected = taus_bound_rx * (claim_Az * claim_Bz - claim_Cz);
        if claim_outer_final != claim_outer_final_expected {
            return Err(SpartanError::InvalidOuterSumcheckClaim);
        }


        // inner sum-check
        let r_inner_sumcheck_RLC: F = transcript.challenge_scalar(b"r");
        let claim_inner_joint = claim_Az
            + r_inner_sumcheck_RLC * claim_Bz
            + r_inner_sumcheck_RLC * r_inner_sumcheck_RLC * claim_Cz;

        let span = tracing::span!(tracing::Level::TRACE, "poly_ABC");
        let _enter = span.enter();

        // this is the polynomial extended from the vector r_A * A(r_x, y) + r_B * B(r_x, y) + r_C * C(r_x, y) for all y
        let poly_ABC = {
            let num_steps_bits = constraint_builder.uniform_repeat().next_power_of_two().ilog2();
            println!("outer_sumcheck_r.len(): {}", outer_sumcheck_r.len());
            println!("num_steps_bits: {}", num_steps_bits);
            let (rx_con, rx_ts) =
                outer_sumcheck_r.split_at(outer_sumcheck_r.len() - num_steps_bits as usize);
            let (eq_rx_con, eq_rx_ts) = rayon::join(
                || EqPolynomial::evals(rx_con),
                || EqPolynomial::evals(rx_ts),
            );
            assert_eq!(eq_rx_con.len(), constraint_builder.uniform_columns().next_power_of_two());
            // TODO(sragss): Assert size of eq_rx_ts
            let n_steps = constraint_builder.uniform_repeat();

            // With uniformity, each entry of the RLC of A, B, C can be expressed using
            // the RLC of the small_A, small_B, small_C matrices.

            // 1. Evaluate \tilde smallM(r_x, y) for all y. Here, \tilde smallM(r_x, y) = \sum_{x} eq(r_x, x) * smallM(x, y)
            let compute_eval_table_sparse_single = |small_M: &[(usize, usize, F)], columns: usize| -> Vec<F> {
                let mut small_M_evals = vec![F::zero(); columns];
                for (row, col, val) in small_M.iter() {
                    small_M_evals[*col] += eq_rx_con[*row] * val;
                }
                small_M_evals
            };

            // TODO(sragss): Switch to key.uniform_r1cs
            let (a_uniform, b_uniform, c_uniform) = constraint_builder.materialize_uniform();
            let columns = constraint_builder.uniform_columns() + 1; // TODO(sragss): Additional column is for constants??

            let (small_A_evals, (small_B_evals, small_C_evals)) = rayon::join(
                || compute_eval_table_sparse_single(&a_uniform, columns),
                || {
                    rayon::join(
                        || compute_eval_table_sparse_single(&b_uniform, columns),
                        || compute_eval_table_sparse_single(&c_uniform, columns),
                    )
                },
            );

            let span = tracing::span!(tracing::Level::TRACE, "poly_ABC_small_RLC_evals");
            let _enter = span.enter();
            let r_sq = r_inner_sumcheck_RLC * r_inner_sumcheck_RLC;
            let small_RLC_evals = (0..small_A_evals.len())
                .into_par_iter()
                .map(|i| {
                    small_A_evals[i]
                        + small_B_evals[i] * r_inner_sumcheck_RLC
                        + small_C_evals[i] * r_sq
                })
                .collect::<Vec<F>>();
            drop(_enter);

            // 2. Obtains the MLE evaluation for each variable y in the full matrix.
            // We first handle all entries but the last one with the constant 1 variable.
            // Each entry is just the small_RLC_evals for the corresponding variable multiplied with eq_rx_tx[timestamp of variable]
            let other_span = tracing::span!(tracing::Level::TRACE, "poly_ABC_wait_alloc_complete");
            let _other_enter = other_span.enter();
            let mut RLC_evals = unsafe_allocate_zero_vec(poly_ABC_len);
            drop(_other_enter);

            let span = tracing::span!(tracing::Level::TRACE, "poly_ABC_big_RLC_evals");
            let _enter = span.enter();

            // Handle all variables but pc_out and the constant
            RLC_evals
                .par_chunks_mut(constraint_builder.uniform_repeat())
                .take(constraint_builder.total_columns() / constraint_builder.uniform_repeat()) // Note that this ignores the last variable which is the constant
                .enumerate()
                .for_each(|(var_index, var_chunk)| {
                    if var_index != 1 && !small_RLC_evals[var_index].is_zero() { // ignore pc_out (var_index = 1) 
                        for (ts, item) in var_chunk.iter_mut().enumerate() {
                            *item = eq_rx_ts[ts] * small_RLC_evals[var_index];
                        }
                    }
                });
            drop(_enter);

            // Handle pc_out
            RLC_evals[1..constraint_builder.uniform_repeat()]
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, rlc)| {
                    *rlc += eq_rx_ts[i] * small_RLC_evals[1]; // take the intended mle eval at pc_out and add it instead to pc_in
                });

            // Handle the constant
            // TODO(sragss): These functions are named like shit.
            RLC_evals[constraint_builder.total_columns() - 1] = small_RLC_evals[constraint_builder.uniform_columns()]; // constant

            RLC_evals
        };
        drop(_enter);
        drop(span);
        let mut poly_ABC = DensePolynomial::new(poly_ABC);
        let (inner_sumcheck_proof, inner_sumcheck_r, _claims_inner) =
            SumcheckInstanceProof::prove_spartan_quadratic::<SegmentedPaddedWitness<F>>(
                &claim_inner_joint, // r_A * v_A + r_B * v_B + r_C * v_C
                num_rounds_y,
                &mut poly_ABC, // r_A * A(r_x, y) + r_B * B(r_x, y) + r_C * C(r_x, y) for all y
                &segmented_padded_witness,
                transcript,
            );
        drop_in_background_thread(poly_ABC);
        println!("claims_inner: {_claims_inner:?}");
        println!("inner_sumcheck_r: {inner_sumcheck_r:?}");

        // The number of prefix bits needed to identify a segment within the witness vector
        // assuming that num_vars_total is a power of 2 and each segment has length num_steps, which is also a power of 2.
        // The +1 is for the first element in r_y used as indicator between input or witness.

        // TODO(sragss): There was a +1 here before. I removed it bc .evaluate_all(r_y) fails otherwise.
        let n_prefix = padded_num_columns.ilog2() as usize - constraint_builder.uniform_repeat().ilog2() as usize;
        let r_y_point = &inner_sumcheck_r[n_prefix..];

        // Evaluate each segment on r_y_point
        let span = tracing::span!(tracing::Level::TRACE, "evaluate_segments");
        let _enter = span.enter();
        let witness_evals = segmented_padded_witness.evaluate_all(r_y_point.to_owned());
        drop(_enter);

        let witness_segment_polys: Vec<DensePolynomial<F>> =
            segmented_padded_witness.into_dense_polys();
        let witness_segment_polys_ref: Vec<&DensePolynomial<F>> =
            witness_segment_polys.iter().collect();
        let opening_proof = C::batch_prove(
            &witness_segment_polys_ref,
            r_y_point,
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
            eval_arg: vec![],
            claimed_witnesss_evals: witness_evals,
            opening_proof,
        })*/
    }

    #[tracing::instrument(skip_all, name = "SNARK::verify")]
    /// verifies a proof of satisfiability of a `RelaxedR1CS` instance
    pub fn verify_precommitted(
        &self,
        key: UniformSpartanKey<F>,
        witness_segment_commitments: Vec<&C::Commitment>,
        generators: &C::Setup,
        transcript: &mut ProofTranscript,
    ) -> Result<(), SpartanError> {
        println!("\n\nVERIFIER TIME");
        let N_SEGMENTS = witness_segment_commitments.len();

        let num_rounds_x = key.num_cons_total.ilog2() as usize;
        let num_rounds_y = key.num_vars_total().ilog2() as usize;
        println!("num_rounds_x {num_rounds_x}");
        println!("num_rounds_y {num_rounds_y}");

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
        println!("inner_sumcheck_r: {inner_sumcheck_r:?}");

        // n_prefix = n_segments + 1
        let n_prefix = (key.num_vars_total().ilog2() as usize - key.num_steps.ilog2() as usize) + 1;

        let eval_Z = key.evaluate_z_mle(&self.claimed_witnesss_evals, &inner_sumcheck_r);

        println!("EVAL Z: {eval_Z:?}");

        /* MLE evaluation */
        let num_steps_bits = key.num_steps.ilog2();
        let (rx_con, rx_ts) = r_x.split_at(r_x.len() - num_steps_bits as usize);

        let r_y = inner_sumcheck_r.clone();
        let (ry_var, ry_ts) = r_y.split_at(r_y.len() - num_steps_bits as usize);

        let eq_rx_con = EqPolynomial::evals(rx_con);
        let eq_ry_var = EqPolynomial::evals(ry_var);

        let eq_rx_ry_ts = EqPolynomial::new(rx_ts.to_vec()).evaluate(ry_ts);
        let eq_ry_0 =
            EqPolynomial::new(ry_ts.to_vec()).evaluate(vec![F::zero(); ry_ts.len()].as_slice());

        /* This MLE is 1 if y = x + 1 for x in the range [0... 2^l-2].
        That is, it ignores the case where x is all 1s, outputting 0.
        Assumes x and y are provided big-endian. */
        let plus_1_mle = |x: &[F], y: &[F], l: usize| -> F {
            let one = F::from_u64(1_u64).unwrap();
            let _two = F::from_u64(2_u64).unwrap();

            /* If y+1 = x, then the two bit vectors are of the following form.
                Let k be the longest suffix of 1s in x.
                In y, those k bits are 0.
                Then, the next bit in x is 0 and the next bit in y is 1.
                The remaining higher bits are the same in x and y.
            */
            (0..l)
                .into_par_iter()
                .map(|k| {
                    let lower_bits_product = (0..k)
                        .map(|i| x[l - 1 - i] * (F::one() - y[l - 1 - i]))
                        .product::<F>();
                    let kth_bit_product = (F::one() - x[l - 1 - k]) * y[l - 1 - k];
                    let higher_bits_product = ((k + 1)..l)
                        .map(|i| {
                            x[l - 1 - i] * y[l - 1 - i]
                                + (one - x[l - 1 - i]) * (one - y[l - 1 - i])
                        })
                        .product::<F>();
                    lower_bits_product * kth_bit_product * higher_bits_product
                })
                .sum()
        };

        let y_eq_x_plus_1 = plus_1_mle(rx_ts, ry_ts, num_steps_bits as usize);

        // compute evaluations of R1CS matrices
        let multi_evaluate_uniform = |M_vec: &[&[(usize, usize, F)]]| -> Vec<F> {
            let evaluate_with_table_uniform = |M: &[(usize, usize, F)]| -> F {
                (0..M.len())
                    .into_par_iter()
                    .map(|i| {
                        let (row, col, val) = M[i];
                        val * eq_rx_con[row] *
                            // * if col == 1 {
                                // pc_out (col 1) is redirected to pc_in (col 0)
                                // eq_ry_var[0] * y_eq_x_plus_1
                            // } else 
                            if col == key.uniform_r1cs.num_vars {
                                eq_ry_var[col] * eq_ry_0
                            } else {
                                eq_ry_var[col] * eq_rx_ry_ts
                            }
                    })
                    .sum()
            };

            (0..M_vec.len())
                .into_par_iter()
                .map(|i| evaluate_with_table_uniform(M_vec[i]))
                .collect()
        };

        todo!();
        /*let evals = multi_evaluate_uniform(&[
            &key.uniform_r1cs.a,
            &key.uniform_r1cs.b,
            &key.uniform_r1cs.c,
        ]);
        println!("evals: {evals:?}");

        let left_expected = evals[0]
            + r_inner_sumcheck_RLC * evals[1]
            + r_inner_sumcheck_RLC * r_inner_sumcheck_RLC * evals[2];
        println!("evals combined: {left_expected:?}");
        println!("r_inner_sumcheck_RLC {r_inner_sumcheck_RLC:?}");
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
            &self.claimed_witnesss_evals,
            &witness_segment_commitments,
            transcript,
        )
        .map_err(|_| SpartanError::InvalidPCSProof)?;

        Ok(())*/
    }
}

struct SparsePolynomial<F: JoltField> {
    num_vars: usize,
    Z: Vec<(usize, F)>,
}

impl<Scalar: JoltField> SparsePolynomial<Scalar> {
    pub fn new(num_vars: usize, Z: Vec<(usize, Scalar)>) -> Self {
        SparsePolynomial { num_vars, Z }
    }

    /// Computes the $\tilde{eq}$ extension polynomial.
    /// return 1 when a == r, otherwise return 0.
    fn compute_chi(a: &[bool], r: &[Scalar]) -> Scalar {
        assert_eq!(a.len(), r.len());
        let mut chi_i = Scalar::one();
        for j in 0..r.len() {
            if a[j] {
                chi_i *= r[j];
            } else {
                chi_i *= Scalar::one() - r[j];
            }
        }
        chi_i
    }

    // Takes O(n log n)
    pub fn evaluate(&self, r: &[Scalar]) -> Scalar {
        assert_eq!(self.num_vars, r.len());

        (0..self.Z.len())
            .into_par_iter()
            .map(|i| {
                let bits = get_bits(self.Z[0].0, r.len());
                SparsePolynomial::compute_chi(&bits, r) * self.Z[i].1
            })
            .sum()
    }
}

/// Returns the `num_bits` from n in a canonical order
fn get_bits(operand: usize, num_bits: usize) -> Vec<bool> {
    (0..num_bits)
        .map(|shift_amount| ((operand & (1 << (num_bits - shift_amount - 1))) > 0))
        .collect::<Vec<bool>>()
}

#[cfg(test)]
mod tests {
    fn piecewise_mle() {

    }
}
