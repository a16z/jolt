use std::marker::PhantomData;

use super::builder::{
    eval_offset_lc, shard_last_step_eval_offset_lc, CombinedUniformBuilder, Constraint,
    OffsetEqConstraint,
};

use crate::field::JoltField;
use crate::jolt::instruction::JoltInstructionSet;
use crate::jolt::vm::JoltPolynomials;
use crate::jolt::vm::JoltStuff;
use crate::jolt::vm::{JoltCommitments, JoltTraceStep};
use crate::jolt::vm::{JoltOracle, JoltProverPreprocessing};
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::eq_poly::StreamingEqPolynomial;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::multilinear_polynomial::PolynomialEvaluation;
use crate::poly::opening_proof::ProverOpeningAccumulator;
use crate::poly::opening_proof::VerifierOpeningAccumulator;
use crate::poly::split_eq_poly::SplitEqPolynomial;
use crate::r1cs::constraints::R1CSConstraints;
use crate::r1cs::inputs::{ConstraintInput, JoltR1CSInputs};
use crate::r1cs::key::UniformSpartanKey;
use crate::subprotocols::sumcheck::{OracleItem, Stream};
use crate::utils::math::Math;
use crate::utils::streaming::Oracle;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::Transcript;
use crate::{
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::{EqPlusOnePolynomial, EqPolynomial},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
};
use ark_ff::Zero;
use ark_serialize::CanonicalDeserialize;
use ark_serialize::CanonicalSerialize;
use common::rv_trace::JoltDevice;
use itertools::Itertools;
use rayon::prelude::*;
use thiserror::Error;
use tracing::{span, Level};

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum SpartanError {
    /// returned if the supplied row or col in (row,col,val) tuple is out of range
    #[error("InvalidIndex")]
    InvalidIndex,

    /// returned when an invalid sum-check proof is provided
    #[error("InvalidSumcheckProof")]
    InvalidSumcheckProof,

    /// returned when the recursive sumcheck proof fails
    #[error("InvalidOuterSumcheckProof")]
    InvalidOuterSumcheckProof,

    /// returned when the final sumcheck opening proof fails
    #[error("InvalidOuterSumcheckClaim")]
    InvalidOuterSumcheckClaim,

    /// returned when the recursive sumcheck proof fails
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

#[derive(Debug)]
pub struct AzBzCz {
    pub interleaved_az_bz_cz: Vec<(usize, i128)>,
}
impl AzBzCz {
    pub fn uninterleave(self) -> (Vec<(usize, i128)>, Vec<(usize, i128)>, Vec<(usize, i128)>) {
        let mut az = Vec::<(usize, i128)>::new();
        let mut bz = Vec::<(usize, i128)>::new();
        let mut cz = Vec::<(usize, i128)>::new();

        for entry in self.interleaved_az_bz_cz {
            match entry.0 % 3 {
                0 => az.push((entry.0 / 3, entry.1)),
                1 => bz.push((entry.0 / 3, entry.1)),
                2 => cz.push((entry.0 / 3, entry.1)),
                _ => unreachable!(),
            }
        }
        (az, bz, cz)
    }
}
pub struct AzBzCzOracle<'a, F: JoltField, InstructionSet: JoltInstructionSet> {
    pub jolt_oracle: JoltOracle<'a, F, InstructionSet>,
    pub func: Box<
        dyn (Fn(
                usize,
                JoltStuff<MultilinearPolynomial<F>>,
                JoltStuff<MultilinearPolynomial<F>>,
            ) -> AzBzCz)
            + 'a,
    >,
}

impl<'a, F: JoltField, InstructionSet: JoltInstructionSet> AzBzCzOracle<'a, F, InstructionSet> {
    pub fn new<const C: usize, I: ConstraintInput>(
        uniform_constraints: &'a [Constraint],
        cross_step_constraints: &'a [OffsetEqConstraint],
        padded_num_constraints: usize,
        jolt_oracle: JoltOracle<'a, F, InstructionSet>,
    ) -> Self {
        let total_num_steps = jolt_oracle.get_len();

        let polynomial_stream =
            move |shard_idx: usize,
                  shard: JoltStuff<MultilinearPolynomial<F>>,
                  extra_eval: JoltStuff<MultilinearPolynomial<F>>| {
                let shard_length = shard.bytecode.a_read_write.len();

                let streaming_z: Vec<&MultilinearPolynomial<F>> = I::flatten::<C>()
                    .iter()
                    .map(|var| var.get_ref(&shard))
                    .collect();

                // TODO: Put in a seperate function.
                let num_steps = streaming_z[0].len();

                let num_chunks = rayon::current_num_threads().next_power_of_two() * 4;
                let chunk_size = num_steps.div_ceil(num_chunks);

                let interleaved_az_bz_cz: Vec<(usize, i128)> = (0..num_chunks)
                .into_par_iter()
                .flat_map_iter(|chunk_index| {
                    let mut coeffs = Vec::with_capacity(3 * chunk_size * padded_num_constraints);

                    for step_index in chunk_size * chunk_index..chunk_size * (chunk_index + 1) {
                        // Uniform constraints
                        for (constraint_index, constraint) in uniform_constraints
                            .iter()
                            .enumerate() {
                            let global_index =
                                3 *
                                ((step_index + shard_idx * shard_length) * padded_num_constraints +
                                    constraint_index);

                            // Az
                            let mut az_coeff = 0;
                            if !constraint.a.terms().is_empty() {
                                az_coeff = constraint.a.evaluate_row(&streaming_z, step_index);
                                if !az_coeff.is_zero() {
                                    coeffs.push((global_index, az_coeff));
                                }
                            }
                            // Bz
                            let mut bz_coeff = 0;
                            if !constraint.b.terms().is_empty() {
                                bz_coeff = constraint.b.evaluate_row(&streaming_z, step_index);
                                if !bz_coeff.is_zero() {
                                    coeffs.push((global_index + 1, bz_coeff));
                                }
                            }
                            // Cz = Az ⊙ Bz
                            if !az_coeff.is_zero() && !bz_coeff.is_zero() {
                                let cz_coeff = az_coeff * bz_coeff;
                                #[cfg(test)]
                                {
                                    if
                                        cz_coeff !=
                                        constraint.c.evaluate_row(&streaming_z, step_index)
                                    {
                                        let mut constraint_string = String::new();
                                        let _ = constraint.pretty_fmt::<4, JoltR1CSInputs, F>(
                                            &mut constraint_string,
                                            &streaming_z,
                                            step_index
                                        );
                                        println!("{constraint_string}");
                                        panic!(
                                            "Uniform constraint {constraint_index} violated at step {step_index}"
                                        );
                                    }
                                }
                                coeffs.push((global_index + 2, cz_coeff));
                            }
                        }

                        let next_step_index = if
                            step_index + shard_idx * shard_length + 1 < total_num_steps
                        {
                            Some(step_index + 1)
                        } else {
                            None
                        };

                        for (constraint_index, constraint) in cross_step_constraints
                            .iter()
                            .enumerate() {
                            let global_index =
                                3 *
                                ((step_index + shard_idx * shard_length) * padded_num_constraints +
                                    uniform_constraints.len() +
                                    constraint_index);

                            if
                                next_step_index.is_none() ||
                                (next_step_index.is_some() &&
                                    next_step_index.unwrap() < shard_length)
                            {
                                // Az
                                let eq_a_eval = eval_offset_lc(
                                    &constraint.a,
                                    &streaming_z,
                                    step_index,
                                    next_step_index
                                );
                                let eq_b_eval = eval_offset_lc(
                                    &constraint.b,
                                    &streaming_z,
                                    step_index,
                                    next_step_index
                                );
                                let az_coeff = eq_a_eval - eq_b_eval;
                                if !az_coeff.is_zero() {
                                    coeffs.push((global_index, az_coeff));
                                    // If Az != 0, then the condition must be false (i.e. Bz = 0)
                                    #[cfg(test)]
                                    {
                                        let bz_coeff = eval_offset_lc(
                                            &constraint.cond,
                                            &streaming_z,
                                            step_index,
                                            next_step_index
                                        );
                                        assert_eq!(
                                            bz_coeff,
                                            0,
                                            "Cross-step constraint {constraint_index} violated at step {step_index}"
                                        );
                                    }
                                } else {
                                    let bz_coeff = eval_offset_lc(
                                        &constraint.cond,
                                        &streaming_z,
                                        step_index,
                                        next_step_index
                                    );
                                    if !bz_coeff.is_zero() {
                                        coeffs.push((global_index + 1, bz_coeff));
                                    }
                                }
                            } else {
                                let extra_eval_z: Vec<&MultilinearPolynomial<F>> = I::flatten::<C>()
                                    .iter()
                                    .map(|var| var.get_ref(&extra_eval))
                                    .collect();

                                let eq_a_eval = shard_last_step_eval_offset_lc(
                                    &constraint.a,
                                    &streaming_z,
                                    &extra_eval_z,
                                    step_index,
                                    next_step_index
                                );
                                let eq_b_eval = shard_last_step_eval_offset_lc(
                                    &constraint.b,
                                    &streaming_z,
                                    &extra_eval_z,
                                    step_index,
                                    next_step_index
                                );

                                let az_coeff = eq_a_eval - eq_b_eval;
                                if !az_coeff.is_zero() {
                                    coeffs.push((global_index, az_coeff));
                                } else {
                                    let bz_coeff = shard_last_step_eval_offset_lc(
                                        &constraint.cond,
                                        &streaming_z,
                                        &extra_eval_z,
                                        step_index,
                                        next_step_index
                                    );
                                    if !bz_coeff.is_zero() {
                                        coeffs.push((global_index + 1, bz_coeff));
                                    }
                                }
                            }
                        }
                    }
                    coeffs
                })
                .collect();

                AzBzCz {
                    interleaved_az_bz_cz,
                }
            };

        AzBzCzOracle {
            jolt_oracle,
            func: Box::new(polynomial_stream),
        }
    }
}

impl<'a, F: JoltField, InstructionSet: JoltInstructionSet> Oracle
    for AzBzCzOracle<'a, F, InstructionSet>
{
    type Item = AzBzCz;
    fn next_shard(&mut self, shard_len: usize) -> Self::Item {
        let shard_idx = self.jolt_oracle.get_step() / shard_len;
        let jolt_shard = self.jolt_oracle.next_shard(shard_len);

        if self.jolt_oracle.peek().is_some() {
            let jolt_peek = self.jolt_oracle.peek().unwrap();
            (self.func)(shard_idx, jolt_shard, jolt_peek)
        } else {
            (self.func)(shard_idx, jolt_shard, Default::default())
        }
    }

    fn reset(&mut self) {
        self.jolt_oracle.reset();
    }

    fn get_len(&self) -> usize {
        self.jolt_oracle.get_len()
    }

    fn get_step(&self) -> usize {
        self.jolt_oracle.get_step()
    }
}

pub struct BindZRyVarOracle<'a, F: JoltField, InstructionSet: JoltInstructionSet> {
    pub jolt_oracle: JoltOracle<'a, F, InstructionSet>,
    pub func: Box<dyn (Fn(JoltStuff<MultilinearPolynomial<F>>) -> MultilinearPolynomial<F>) + 'a>,
}

impl<'a, F: JoltField, InstructionSet: JoltInstructionSet> BindZRyVarOracle<'a, F, InstructionSet> {
    pub fn new<const C: usize, I: ConstraintInput>(
        jolt_oracle: JoltOracle<'a, F, InstructionSet>,
        eq_ry_var: &'a [F],
        eq_ry_var_r2: &'a [F],
    ) -> Self {
        let polynomial_stream = move |shard: JoltStuff<MultilinearPolynomial<F>>| {
            let streaming_z: Vec<&MultilinearPolynomial<F>> = I::flatten::<C>()
                .iter()
                .map(|var| var.get_ref(&shard))
                .collect();
            let shard_length = shard.bytecode.a_read_write.len();

            let sum_vec: Vec<F> = (0..shard_length)
                .map(|t| {
                    streaming_z
                        .iter()
                        .enumerate()
                        .fold(F::zero(), |sum, (i, poly)| {
                            sum + poly.scale_coeff(t, eq_ry_var[i], eq_ry_var_r2[i])
                        })
                })
                .collect();

            MultilinearPolynomial::from(sum_vec)
        };

        BindZRyVarOracle {
            jolt_oracle,
            func: Box::new(polynomial_stream),
        }
    }
}

impl<'a, F: JoltField, InstructionSet: JoltInstructionSet> Oracle
    for BindZRyVarOracle<'a, F, InstructionSet>
{
    type Item = MultilinearPolynomial<F>;

    fn next_shard(&mut self, shard_len: usize) -> Self::Item {
        (self.func)(self.jolt_oracle.next_shard(shard_len))
    }

    fn reset(&mut self) {
        self.jolt_oracle.reset();
    }

    fn get_len(&self) -> usize {
        self.jolt_oracle.get_len()
    }

    fn get_step(&self) -> usize {
        self.jolt_oracle.get_step()
    }
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
    pub(crate) shift_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub(crate) shift_sumcheck_claim: F,
    pub(crate) claimed_witness_evals: Vec<F>,
    pub(crate) shift_sumcheck_witness_evals: Vec<F>,
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
        let flattened_polys: Vec<&MultilinearPolynomial<F>> = I::flatten::<C>()
            .iter()
            .map(|var| var.get_ref(polynomials))
            .collect();

        let num_rounds_x = key.num_rows_bits();

        /* Sumcheck 1: Outer sumcheck */

        let tau = (0..num_rounds_x)
            .map(|_i| transcript.challenge_scalar())
            .collect::<Vec<F>>();

        let mut eq_tau = SplitEqPolynomial::new(&tau);

        let mut az_bz_cz_poly = constraint_builder.compute_spartan_Az_Bz_Cz(&flattened_polys);

        let (outer_sumcheck_proof, outer_sumcheck_r, outer_sumcheck_claims) =
            SumcheckInstanceProof::prove_spartan_cubic(
                num_rounds_x,
                &mut eq_tau,
                &mut az_bz_cz_poly,
                transcript,
            );
        let outer_sumcheck_r: Vec<F> = outer_sumcheck_r.into_iter().rev().collect();
        drop_in_background_thread((az_bz_cz_poly, eq_tau));

        ProofTranscript::append_scalars(transcript, &outer_sumcheck_claims);
        // claims from the end of sum-check
        // claim_Az is the (scalar) value v_A = \sum_y A(r_x, y) * z(r_x) where r_x is the sumcheck randomness
        let (claim_Az, claim_Bz, claim_Cz): (F, F, F) = (
            outer_sumcheck_claims[0],
            outer_sumcheck_claims[1],
            outer_sumcheck_claims[2],
        );

        /* Sumcheck 2: Inner sumcheck
            RLC of claims Az, Bz, Cz
            where claim_Az = \sum_{y_var} A(rx, y_var || rx_step) * z(y_var || rx_step)
                                + A_shift(..) * z_shift(..)
            and shift denotes the values at the next time step "rx_step+1" for cross-step constraints
            - A_shift(rx, y_var || rx_step) = \sum_t A(rx, y_var || t) * eq_plus_one(rx_step, t)
            - z_shift(y_var || rx_step) = \sum z(y_var || rx_step) * eq_plus_one(rx_step, t)
        */

        let num_steps = key.num_steps;
        let num_steps_bits = num_steps.ilog2() as usize;
        let num_vars_uniform = key.num_vars_uniform_padded().next_power_of_two();

        let inner_sumcheck_RLC: F = transcript.challenge_scalar();
        let claim_inner_joint = claim_Az
            + inner_sumcheck_RLC * claim_Bz
            + inner_sumcheck_RLC * inner_sumcheck_RLC * claim_Cz;

        let (rx_step, rx_constr) = outer_sumcheck_r.split_at(num_steps_bits);

        let (eq_rx_step, eq_plus_one_rx_step) = EqPlusOnePolynomial::evals(rx_step, None);
        let (eq_rx_step_r2, eq_plus_one_rx_step_r2) =
            EqPlusOnePolynomial::evals(rx_step, F::montgomery_r2());

        /* Compute the two polynomials provided as input to the second sumcheck:
           - poly_ABC: A(r_x, y_var || rx_step), A_shift(..) at all variables y_var
           - poly_z: z(y_var || rx_step), z_shift(..)
        */

        let poly_ABC = DensePolynomial::new(key.evaluate_matrix_mle_partial(
            rx_constr,
            rx_step,
            inner_sumcheck_RLC,
        ));

        // Binding z and z_shift polynomials at point rx_step
        let span = span!(Level::INFO, "binding_z_and_shift_z");
        let _guard = span.enter();

        let mut bind_z = vec![F::zero(); num_vars_uniform * 2];
        let mut bind_shift_z = vec![F::zero(); num_vars_uniform * 2];

        flattened_polys
            .par_iter()
            .zip(bind_z.par_iter_mut().zip(bind_shift_z.par_iter_mut()))
            .for_each(|(poly, (eval, eval_shifted))| {
                *eval = poly.dot_product(Some(&eq_rx_step), Some(&eq_rx_step_r2));
                *eval_shifted =
                    poly.dot_product(Some(&eq_plus_one_rx_step), Some(&eq_plus_one_rx_step_r2));
            });

        bind_z[num_vars_uniform] = F::one();

        drop(_guard);
        drop(span);

        let poly_z =
            DensePolynomial::new(bind_z.into_iter().chain(bind_shift_z.into_iter()).collect());
        assert_eq!(poly_z.len(), poly_ABC.len());

        let num_rounds_inner_sumcheck = poly_ABC.len().log_2();

        let mut polys = vec![
            MultilinearPolynomial::LargeScalars(poly_ABC),
            MultilinearPolynomial::LargeScalars(poly_z),
        ];

        let comb_func = |poly_evals: &[F]| -> F {
            assert_eq!(poly_evals.len(), 2);
            poly_evals[0] * poly_evals[1]
        };

        let (inner_sumcheck_proof, inner_sumcheck_r, _claims_inner) =
            SumcheckInstanceProof::prove_arbitrary(
                &claim_inner_joint,
                num_rounds_inner_sumcheck,
                &mut polys,
                comb_func,
                2,
                transcript,
            );

        drop_in_background_thread(polys);

        /*  Sumcheck 3: Shift sumcheck
            sumcheck claim is = z_shift(ry_var || rx_step) = \sum_t z(ry_var || t) * eq_plus_one(rx_step, t)
        */

        let ry_var = inner_sumcheck_r[1..].to_vec();
        let eq_ry_var = EqPolynomial::evals(&ry_var);
        let eq_ry_var_r2 = EqPolynomial::evals_with_r2(&ry_var);

        let mut bind_z_ry_var: Vec<F> = Vec::with_capacity(num_steps);
        let span = span!(Level::INFO, "bind_z_ry_var");
        let _guard = span.enter();

        let num_steps_unpadded = constraint_builder.uniform_repeat();
        (0..num_steps_unpadded) // unpadded number of steps is sufficient
            .into_par_iter()
            .map(|t| {
                flattened_polys
                    .iter()
                    .enumerate()
                    .map(|(i, poly)| poly.scale_coeff(t, eq_ry_var[i], eq_ry_var_r2[i]))
                    .sum()
            })
            .collect_into_vec(&mut bind_z_ry_var);
        drop(_guard);
        drop(span);

        let num_rounds_shift_sumcheck = num_steps_bits;
        assert_eq!(bind_z_ry_var.len(), eq_plus_one_rx_step.len());

        let mut shift_sumcheck_polys = vec![
            MultilinearPolynomial::from(bind_z_ry_var),
            MultilinearPolynomial::from(eq_plus_one_rx_step),
        ];

        let shift_sumcheck_claim = (0..1 << num_rounds_shift_sumcheck)
            .into_par_iter()
            .map(|i| {
                let params: Vec<F> = shift_sumcheck_polys
                    .iter()
                    .map(|poly| poly.get_coeff(i))
                    .collect();
                comb_func(&params)
            })
            .reduce(|| F::zero(), |acc, x| acc + x);

        let (shift_sumcheck_proof, shift_sumcheck_r, _shift_sumcheck_claims) =
            SumcheckInstanceProof::prove_arbitrary(
                &shift_sumcheck_claim,
                num_rounds_shift_sumcheck,
                &mut shift_sumcheck_polys,
                comb_func,
                2,
                transcript,
            );
        drop_in_background_thread(shift_sumcheck_polys);

        // Inner sumcheck evaluations: evaluate z on rx_step
        let (claimed_witness_evals, chis) =
            MultilinearPolynomial::batch_evaluate(&flattened_polys, rx_step);

        opening_accumulator.append(
            &flattened_polys,
            DensePolynomial::new(chis),
            rx_step.to_vec(),
            &claimed_witness_evals,
            transcript,
        );

        // Shift sumcheck evaluations: evaluate z on ry_var
        let (shift_sumcheck_witness_evals, chis2) =
            MultilinearPolynomial::batch_evaluate(&flattened_polys, &shift_sumcheck_r);

        opening_accumulator.append(
            &flattened_polys,
            DensePolynomial::new(chis2),
            shift_sumcheck_r.to_vec(),
            &shift_sumcheck_witness_evals,
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
            shift_sumcheck_proof,
            shift_sumcheck_claim,
            claimed_witness_evals,
            shift_sumcheck_witness_evals,
            _marker: PhantomData,
        })
    }

    // TODO: Change the return type back to Result<Self, SpartanError>.
    #[tracing::instrument(skip_all, name = "Spartan::streaming_prove")]
    pub fn streaming_prove<PCS, InstructionSet, const M: usize>(
        num_shards: usize,
        shard_length: usize,
        preprocessing: &JoltProverPreprocessing<C, F, PCS, ProofTranscript>,
        program_io: &JoltDevice,
        trace: &Vec<JoltTraceStep<InstructionSet>>,
        constraint_builder: &CombinedUniformBuilder<C, F, I>,
        key: &UniformSpartanKey<C, I, F>,
        polynomials: &JoltPolynomials<F>,

        opening_accumulator: &mut ProverOpeningAccumulator<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<Self, SpartanError>
    where
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        InstructionSet: JoltInstructionSet,
    {
        let flattened_polys: Vec<&MultilinearPolynomial<F>> = I::flatten::<C>()
            .iter()
            .map(|var| var.get_ref(polynomials))
            .collect();

        let num_rounds_x = key.num_rows_bits();

        let tau = (0..num_rounds_x)
            .map(|_i| transcript.challenge_scalar())
            .collect::<Vec<F>>();

        let num_padded_rows = constraint_builder.padded_rows_per_step();

        let jolt_oracle = JoltOracle::new::<C, M, PCS, ProofTranscript, I>(
            preprocessing,
            program_io,
            constraint_builder,
            trace,
        );

        let mut streaming_az_bz_cz_poly = AzBzCzOracle::new::<C, I>(
            &constraint_builder.uniform_builder.constraints,
            &constraint_builder.offset_equality_constraints,
            num_padded_rows,
            jolt_oracle,
        );

        #[cfg(test)]
        {
            let mut az_bz_cz_poly = constraint_builder.compute_spartan_Az_Bz_Cz(&flattened_polys);

            let mut streamed_polys_vec: Vec<AzBzCz> = Vec::new();
            for n in 0..num_shards {
                let streamed_polys = streaming_az_bz_cz_poly.next_shard(shard_length);
                streamed_polys_vec.push(streamed_polys);
            }

            let mut j = 0;
            for n in 0..num_shards {
                let len = streamed_polys_vec[n].interleaved_az_bz_cz.len();
                for i in 0..len {
                    assert_eq!(
                        streamed_polys_vec[n].interleaved_az_bz_cz[i].0,
                        az_bz_cz_poly.unbound_coeffs[j].index
                    );
                    assert_eq!(
                        streamed_polys_vec[n].interleaved_az_bz_cz[i].1,
                        az_bz_cz_poly.unbound_coeffs[j].value
                    );
                    j += 1;
                }
            }
            assert_eq!(j, az_bz_cz_poly.unbound_coeffs.len());
            streaming_az_bz_cz_poly.reset();
        }

        let (outer_sumcheck_proof, outer_sumcheck_r, outer_sumcheck_claims) =
            SumcheckInstanceProof::stream_prove_cubic(
                num_shards,
                num_rounds_x,
                &mut streaming_az_bz_cz_poly,
                shard_length,
                num_padded_rows,
                tau.clone(),
                transcript,
            );

        let outer_sumcheck_r: Vec<F> = outer_sumcheck_r.into_iter().rev().collect();

        ProofTranscript::append_scalars(transcript, &outer_sumcheck_claims);

        // claims from the end of sum-check
        // claim_Az is the (scalar) value v_A = \sum_y A(r_x, y) * z(r_x) where r_x is the sumcheck randomness
        let (claim_Az, claim_Bz, claim_Cz): (F, F, F) = (
            outer_sumcheck_claims[0],
            outer_sumcheck_claims[1],
            outer_sumcheck_claims[2],
        );

        /* Sumcheck 2: Inner sumcheck
            RLC of claims Az, Bz, Cz
            where claim_Az = \sum_{y_var} A(rx, y_var || rx_step) * z(y_var || rx_step)
                                + A_shift(..) * z_shift(..)
            and shift denotes the values at the next time step "rx_step+1" for cross-step constraints
            - A_shift(rx, y_var || rx_step) = \sum_t A(rx, y_var || t) * eq_plus_one(rx_step, t)
            - z_shift(y_var || rx_step) = \sum z(y_var || rx_step) * eq_plus_one(rx_step, t)
        */

        let num_steps = key.num_steps;
        let num_steps_bits = num_steps.ilog2() as usize;
        let num_vars_uniform = key.num_vars_uniform_padded().next_power_of_two();

        let inner_sumcheck_RLC: F = transcript.challenge_scalar();
        let claim_inner_joint = claim_Az
            + inner_sumcheck_RLC * claim_Bz
            + inner_sumcheck_RLC * inner_sumcheck_RLC * claim_Cz;

        let (rx_step, rx_constr) = outer_sumcheck_r.split_at(num_steps_bits);

        /* Compute the two polynomials provided as input to the second sumcheck:
           - poly_ABC: A(r_x, y_var || rx_step), A_shift(..) at all variables y_var
           - poly_z: z(y_var || rx_step), z_shift(..)
        */

        let poly_ABC = DensePolynomial::new(key.evaluate_matrix_mle_partial(
            rx_constr,
            rx_step,
            inner_sumcheck_RLC,
        ));

        // Binding z and z_shift polynomials at point rx_step
        let span = span!(Level::INFO, "binding_z_and_shift_z");
        let _guard = span.enter();

        let mut jolt_oracle = JoltOracle::new::<C, M, PCS, ProofTranscript, I>(
            preprocessing,
            program_io,
            constraint_builder,
            trace,
        );

        let mut bind_z_stream = vec![F::zero(); num_vars_uniform * 2];
        let mut bind_shift_z_stream = vec![F::zero(); num_vars_uniform * 2];

        let reverse_rx_step: Vec<F> = rx_step.iter().rev().copied().collect();
        let mut eq_rx_step_stream =
            StreamingEqPolynomial::new(reverse_rx_step.to_vec(), reverse_rx_step.len(), None, true);
        let mut eq_rx_step_r2_stream = StreamingEqPolynomial::new(
            reverse_rx_step.to_vec(),
            reverse_rx_step.len(),
            F::montgomery_r2(),
            true,
        );
        let mut eq_plus_one_rx_step_stream = StreamingEqPolynomial::new(
            reverse_rx_step.to_vec(),
            reverse_rx_step.len(),
            None,
            false,
        );
        let mut eq_plus_one_rx_step_r2_stream = StreamingEqPolynomial::new(
            reverse_rx_step.to_vec(),
            reverse_rx_step.len(),
            F::montgomery_r2(),
            false,
        );

        for _ in 0..num_shards {
            let polynomials = jolt_oracle.next_shard(shard_length);
            let (eq_rx_step_shard, eq_rx_step_r2_shard) = (
                eq_rx_step_stream.next_shard(shard_length),
                eq_rx_step_r2_stream.next_shard(shard_length),
            );

            let (eq_plus_one_rx_step_shard, eq_plus_one_rx_step_r2_shard) = (
                eq_plus_one_rx_step_stream.next_shard(shard_length),
                eq_plus_one_rx_step_r2_stream.next_shard(shard_length),
            );

            let flattened_polys: Vec<&MultilinearPolynomial<F>> = I::flatten::<C>()
                .iter()
                .map(|var| var.get_ref(&polynomials))
                .collect();

            let (partial_bind_z, partial_bind_shift_z): (Vec<F>, Vec<F>) = flattened_polys
                .par_iter()
                .map(|poly| {
                    let eval1 =
                        poly.dot_product(Some(&eq_rx_step_shard), Some(&eq_rx_step_r2_shard));
                    let eval2 = poly.dot_product(
                        Some(&eq_plus_one_rx_step_shard),
                        Some(&eq_plus_one_rx_step_r2_shard),
                    );
                    (eval1, eval2)
                })
                .collect();

            partial_bind_z
                .iter()
                .zip(partial_bind_shift_z.iter())
                .zip(bind_z_stream.iter_mut().zip(bind_shift_z_stream.iter_mut()))
                .for_each(
                    |((partial_eval, partial_bind_shift_eval), (eval, eval_shifted))| {
                        *eval += *partial_eval;
                        *eval_shifted += *partial_bind_shift_eval;
                    },
                );
        }

        bind_z_stream[num_vars_uniform] = F::one();
        jolt_oracle.reset();

        drop(_guard);
        drop(span);

        let poly_z = DensePolynomial::new(
            bind_z_stream
                .into_iter()
                .chain(bind_shift_z_stream.into_iter())
                .collect(),
        );
        assert_eq!(poly_z.len(), poly_ABC.len());

        let num_rounds_inner_sumcheck = poly_ABC.len().log_2();

        let mut polys = vec![
            MultilinearPolynomial::LargeScalars(poly_ABC),
            MultilinearPolynomial::LargeScalars(poly_z),
        ];

        let comb_func = |poly_evals: &[F]| -> F {
            assert_eq!(poly_evals.len(), 2);
            poly_evals[0] * poly_evals[1]
        };

        let (inner_sumcheck_proof, inner_sumcheck_r, _claims_inner) =
            SumcheckInstanceProof::prove_arbitrary(
                &claim_inner_joint,
                num_rounds_inner_sumcheck,
                &mut polys,
                comb_func,
                2,
                transcript,
            );

        drop_in_background_thread(polys);

        /*  Sumcheck 3: Shift sumcheck
            sumcheck claim is = z_shift(ry_var || rx_step) = \sum_t z(ry_var || t) * eq_plus_one(rx_step, t)
        */

        let ry_var = inner_sumcheck_r[1..].to_vec();
        let eq_ry_var = EqPolynomial::evals(&ry_var);
        let eq_ry_var_r2 = EqPolynomial::evals_with_r2(&ry_var);

        let span = span!(Level::INFO, "bind_z_ry_var");
        let _guard = span.enter();

        let num_rounds_shift_sumcheck = num_steps_bits;

        let bind_z_ry_var_oracle =
            BindZRyVarOracle::new::<C, I>(jolt_oracle, &eq_ry_var, &eq_ry_var_r2);

        let eq_plus_one_rx_step_stream =
            StreamingEqPolynomial::new(reverse_rx_step.to_vec(), rx_step.len(), None, false);
        let mut oracle =
            Stream::SpartanSumCheck((bind_z_ry_var_oracle, eq_plus_one_rx_step_stream));

        let extract_poly_fn = |stream_data: &OracleItem<F>| -> Vec<MultilinearPolynomial<F>> {
            match stream_data {
                OracleItem::SpartanSumCheck(stream) => stream.to_vec(),
                _ => vec![],
            }
        };

        let shift_sumcheck_claim: F = (0..num_shards)
            .map(|_| {
                let shards = oracle.next_shard(shard_length);
                let polys = extract_poly_fn(&shards);
                (0..shard_length)
                    .map(|j| {
                        let params: Vec<F> = polys.iter().map(|poly| poly.get_coeff(j)).collect();
                        comb_func(&params)
                    })
                    .fold(F::zero(), |acc, x| acc + x)
            })
            .sum();

        oracle.reset();

        drop(_guard);
        drop(span);

        let (shift_sumcheck_proof, shift_sumcheck_r_rev, _shift_sumcheck_claims) =
            SumcheckInstanceProof::stream_prove_arbitrary(
                num_rounds_shift_sumcheck,
                &mut oracle,
                extract_poly_fn,
                comb_func,
                2,
                shard_length,
                2,
                transcript,
            );

        let shift_sumcheck_r: Vec<F> = shift_sumcheck_r_rev.iter().rev().copied().collect();

        // Inner sumcheck evaluations: evaluate z on rx_step
        let (claimed_witness_evals, chis) =
            MultilinearPolynomial::batch_evaluate(&flattened_polys, rx_step);

        opening_accumulator.append(
            &flattened_polys,
            DensePolynomial::new(chis),
            rx_step.to_vec(),
            &claimed_witness_evals,
            transcript,
        );

        let mut jolt_oracle = JoltOracle::new::<C, M, PCS, ProofTranscript, I>(
            preprocessing,
            program_io,
            constraint_builder,
            trace,
        );

        #[cfg(test)]
        {
            let claimed_witness_eval2 = MultilinearPolynomial::stream_batch_evaluate::<
                C,
                InstructionSet,
                I,
            >(
                &mut jolt_oracle, rx_step, num_shards, shard_length
            );

            assert_eq!(
                claimed_witness_evals, claimed_witness_eval2,
                "stream claimed witness evals are incorrect "
            );
        }
        jolt_oracle.reset();

        // Shift sumcheck evaluations: evaluate z on ry_var
        let (shift_sumcheck_witness_evals, chis2) =
            MultilinearPolynomial::batch_evaluate(&flattened_polys, &shift_sumcheck_r);

        #[cfg(test)]
        {
            let shift_sumcheck_witness_evals2 =
                MultilinearPolynomial::stream_batch_evaluate::<C, InstructionSet, I>(
                    &mut jolt_oracle,
                    &shift_sumcheck_r,
                    num_shards,
                    shard_length,
                );

            assert_eq!(
                shift_sumcheck_witness_evals, shift_sumcheck_witness_evals2,
                "stream shift sum check witness are incorrect "
            );
        }

        opening_accumulator.append(
            &flattened_polys,
            DensePolynomial::new(chis2),
            shift_sumcheck_r.to_vec(),
            &shift_sumcheck_witness_evals,
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
            shift_sumcheck_proof,
            shift_sumcheck_claim,
            claimed_witness_evals,
            shift_sumcheck_witness_evals,
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

        /* Sumcheck 1: Outer sumcheck
         */
        let tau = (0..num_rounds_x)
            .map(|_i| transcript.challenge_scalar())
            .collect::<Vec<F>>();

        let (claim_outer_final, outer_sumcheck_r) = self
            .outer_sumcheck_proof
            .verify(F::zero(), num_rounds_x, 3, transcript)
            .map_err(|_| SpartanError::InvalidOuterSumcheckProof)?;

        // Outer sumcheck is bound from the top, reverse the fiat shamir randomness
        let outer_sumcheck_r: Vec<F> = outer_sumcheck_r.into_iter().rev().collect();

        let (claim_Az, claim_Bz, claim_Cz) = self.outer_sumcheck_claims;
        let taus_bound_rx = EqPolynomial::new(tau).evaluate(&outer_sumcheck_r);
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

        /* Sumcheck 2: Inner sumcheck
           - claim is an RLC of claims_Az, Bz, Cz
           where claim_Az = \sum_{y_var} A(rx, y_var || rx_step) * z(y_var || rx_step)
                               + A_shift(..) * z_shift(..)
           - verifying it involves computing each term with randomness ry_var
        */
        let inner_sumcheck_RLC: F = transcript.challenge_scalar();
        let claim_inner_joint = self.outer_sumcheck_claims.0
            + inner_sumcheck_RLC * self.outer_sumcheck_claims.1
            + inner_sumcheck_RLC * inner_sumcheck_RLC * self.outer_sumcheck_claims.2;

        let num_rounds_inner_sumcheck = (2 * key.num_vars_uniform_padded()).log_2() + 1; // +1 for shift evals
        let (claim_inner_final, inner_sumcheck_r) = self
            .inner_sumcheck_proof
            .verify(claim_inner_joint, num_rounds_inner_sumcheck, 2, transcript)
            .map_err(|_| SpartanError::InvalidInnerSumcheckProof)?;
        let num_steps_bits = key.num_steps.log_2();

        let (rx_step, rx_constr) = outer_sumcheck_r.split_at(num_steps_bits);

        let r_is_cross_step = inner_sumcheck_r[0];
        let ry_var = inner_sumcheck_r[1..].to_vec();
        let eval_z =
            key.evaluate_z_mle_with_segment_evals(&self.claimed_witness_evals, &ry_var, true);

        let (eval_a, eval_b, eval_c) =
            key.evaluate_matrix_mle_full(rx_constr, &ry_var, &r_is_cross_step);

        let left_expected =
            eval_a + inner_sumcheck_RLC * eval_b + inner_sumcheck_RLC * inner_sumcheck_RLC * eval_c;
        let right_expected =
            (F::one() - r_is_cross_step) * eval_z + r_is_cross_step * self.shift_sumcheck_claim;

        let claim_inner_final_expected = left_expected * right_expected;
        if claim_inner_final != claim_inner_final_expected {
            return Err(SpartanError::InvalidInnerSumcheckClaim);
        }

        /* Sumcheck 3: Shift sumcheck
            - claim = \sum_t z(ry_var || t) * eq_plus_one(rx_step, t)
            - verifying it involves checking that claim = z(ry_var || r_t) * eq_plus_one(rx_step, r_t)
            where r_t = shift_sumcheck_r
        */

        let num_rounds_shift_sumcheck = num_steps_bits;
        let (claim_shift_sumcheck, shift_sumcheck_r_rev) = self
            .shift_sumcheck_proof
            .verify(
                self.shift_sumcheck_claim,
                num_rounds_shift_sumcheck,
                2,
                transcript,
            )
            .map_err(|_| SpartanError::InvalidInnerSumcheckProof)?;
        let shift_sumcheck_r: Vec<F> = shift_sumcheck_r_rev.iter().rev().copied().collect();

        let eval_z_shift_sumcheck = key.evaluate_z_mle_with_segment_evals(
            &self.shift_sumcheck_witness_evals,
            &ry_var,
            false,
        );

        let eq_plus_one_shift_sumcheck =
            EqPlusOnePolynomial::new(rx_step.to_vec()).evaluate(&shift_sumcheck_r);
        let claim_shift_sumcheck_expected = eval_z_shift_sumcheck * eq_plus_one_shift_sumcheck;

        if claim_shift_sumcheck != claim_shift_sumcheck_expected {
            return Err(SpartanError::InvalidInnerSumcheckClaim);
        }

        let flattened_commitments: Vec<_> = I::flatten::<C>()
            .iter()
            .map(|var| var.get_ref(commitments))
            .collect();

        opening_accumulator.append(
            &flattened_commitments,
            rx_step.to_vec(),
            &self.claimed_witness_evals.iter().collect::<Vec<_>>(),
            transcript,
        );

        opening_accumulator.append(
            &flattened_commitments,
            shift_sumcheck_r.to_vec(),
            &self.shift_sumcheck_witness_evals.iter().collect::<Vec<_>>(),
            transcript,
        );

        Ok(())
    }
}

// #[cfg(test)]
// mod test {
//     use ark_bn254::Fr;
//     use ark_std::One;

//     use crate::poly::commitment::{ commitment_scheme::CommitShape, hyrax::HyraxScheme };

//     use super::*;

//     #[test]
//     fn integration() {
//         let (builder, key) = simp_test_builder_key();
//         let witness_segments: Vec<Vec<Fr>> = vec![
//             vec![Fr::one(), Fr::from(5), Fr::from(9), Fr::from(13)] /* Q */,
//             vec![Fr::one(), Fr::from(5), Fr::from(9), Fr::from(13)] /* R */,
//             vec![Fr::one(), Fr::from(5), Fr::from(9), Fr::from(13)] /* S */
//         ];

//         // Create a witness and commit
//         let witness_segments_ref: Vec<&[Fr]> = witness_segments
//             .iter()
//             .map(|segment| segment.as_slice())
//             .collect();
//         let gens = HyraxScheme::setup(&[CommitShape::new(16, BatchType::Small)]);
//         let witness_commitment = HyraxScheme::batch_commit(
//             &witness_segments_ref,
//             &gens,
//             BatchType::Small
//         );

//         // Prove spartan!
//         let mut prover_transcript = ProofTranscript::new(b"stuff");
//         let proof = UniformSpartanProof::<Fr, HyraxScheme<ark_bn254::G1Projective>>
//             ::prove_precommitted::<SimpTestIn>(
//                 &gens,
//                 builder,
//                 &key,
//                 witness_segments,
//                 todo!("opening accumulator"),
//                 &mut prover_transcript
//             )
//             .unwrap();

//         let mut verifier_transcript = ProofTranscript::new(b"stuff");
//         let witness_commitment_ref: Vec<&_> = witness_commitment.iter().collect();
//         proof
//             .verify_precommitted(&key, witness_commitment_ref, &gens, &mut verifier_transcript)
//             .expect("Spartan verifier failed");
//     }
// }
