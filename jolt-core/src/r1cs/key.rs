use std::marker::PhantomData;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use sha3::Sha3_256;

use crate::{
    field::JoltField,
    poly::eq_poly::{eq_plus_one, EqPolynomial},
    utils::{index_to_field_bitvector, mul_0_1_optimized, thread::unsafe_allocate_zero_vec},
};

use super::{builder::CombinedUniformBuilder, inputs::ConstraintInput};
use sha3::Digest;

use crate::utils::math::Math;

use rayon::prelude::*;

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct UniformSpartanKey<const C: usize, I: ConstraintInput, F: JoltField> {
    _inputs: PhantomData<I>,
    pub uniform_r1cs: UniformR1CS<F>,

    pub offset_eq_r1cs: NonUniformR1CS<F>,

    /// Number of constraints across all steps padded to nearest power of 2
    pub num_cons_total: usize,

    /// Number of steps padded to the nearest power of 2
    pub num_steps: usize,

    /// Digest of verifier key
    pub(crate) vk_digest: F,
}

/// (row, col, value)
pub type Coeff<F> = (usize, usize, F);

/// Sparse representation of a single R1CS matrix.
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct SparseConstraints<F: JoltField> {
    /// Non-zero, non-constant coefficients
    pub vars: Vec<Coeff<F>>,

    /// Non-zero constant coefficients stored as (uniform_row_index, coeff)
    pub consts: Vec<(usize, F)>,
}

impl<F: JoltField> SparseConstraints<F> {
    pub fn empty_with_capacity(vars: usize, consts: usize) -> Self {
        Self {
            vars: Vec::with_capacity(vars),
            consts: Vec::with_capacity(consts),
        }
    }
}

/// Sparse representation of all 3 uniform R1CS matrices. Uniform matrices can be repeated over a number of steps
/// and efficiently evaluated by taking advantage of the structure.
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct UniformR1CS<F: JoltField> {
    pub a: SparseConstraints<F>,
    pub b: SparseConstraints<F>,
    pub c: SparseConstraints<F>,

    /// Unpadded number of variables in uniform instance.
    pub num_vars: usize,

    /// Unpadded number of rows in uniform instance.
    pub num_rows: usize,
}

/// NonUniformR1CSConstraint only supports a single additional equality constraint. 'a' holds the equality (something minus something),
/// 'b' holds the condition. 'a' * 'b' == 0. Each SparseEqualityItem stores a uniform_column (pointing to a variable) and an offset
/// suggesting which other step to point to.
#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct NonUniformR1CSConstraint<F: JoltField> {
    pub eq: SparseEqualityItem<F>,
    pub condition: SparseEqualityItem<F>,
}

impl<F: JoltField> NonUniformR1CSConstraint<F> {
    pub fn new(eq: SparseEqualityItem<F>, condition: SparseEqualityItem<F>) -> Self {
        Self { eq, condition }
    }

    pub fn empty() -> Self {
        Self {
            eq: SparseEqualityItem::empty(),
            condition: SparseEqualityItem::empty(),
        }
    }
}

/// NonUniformR1CS stores a vector of NonUniformR1CSConstraint
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct NonUniformR1CS<F: JoltField> {
    pub constraints: Vec<NonUniformR1CSConstraint<F>>,
}

impl<F: JoltField> NonUniformR1CS<F> {
    /// Returns a tuple of (eq_constants, condition_constants)
    fn constants(&self) -> (Vec<F>, Vec<F>) {
        let mut eq_constants = Vec::with_capacity(self.constraints.len());
        let mut condition_constants = Vec::with_capacity(self.constraints.len());

        for constraint in &self.constraints {
            eq_constants.push(constraint.eq.constant);
            condition_constants.push(constraint.condition.constant);
        }

        (eq_constants, condition_constants)
    }

    /// Unpadded number of non-uniform constraints.
    fn num_constraints(&self) -> usize {
        self.constraints.len()
    }
}

/// Represents a single constraint row where the variables are either from the current step (offset = false)
/// or from the proceeding step (offset = true).
#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, PartialEq)]
pub struct SparseEqualityItem<F: JoltField> {
    /// (uniform_col, offset, val)
    pub offset_vars: Vec<(usize, bool, F)>,

    pub constant: F,
}

impl<F: JoltField> SparseEqualityItem<F> {
    pub fn empty() -> Self {
        Self {
            offset_vars: vec![],
            constant: F::zero(),
        }
    }
}

impl<const C: usize, F: JoltField, I: ConstraintInput> UniformSpartanKey<C, I, F> {
    pub fn from_builder(constraint_builder: &CombinedUniformBuilder<C, F, I>) -> Self {
        let uniform_r1cs = constraint_builder.materialize_uniform();
        let offset_eq_r1cs = constraint_builder.materialize_offset_eq();

        let total_rows = constraint_builder.constraint_rows().next_power_of_two();
        let num_steps = constraint_builder.uniform_repeat().next_power_of_two(); // TODO(JP): Number of steps no longer need to be padded.

        let vk_digest = Self::digest(&uniform_r1cs, &offset_eq_r1cs, num_steps);

        Self {
            _inputs: PhantomData,
            uniform_r1cs,
            offset_eq_r1cs,
            num_cons_total: total_rows,
            num_steps,
            vk_digest,
        }
    }

    fn full_z_len(&self) -> usize {
        2 * self.num_steps * self.uniform_r1cs.num_vars.next_power_of_two()
    }

    /// Number of variables across all steps padded to next power of two.
    pub fn num_vars_total(&self) -> usize {
        self.num_steps * self.uniform_r1cs.num_vars.next_power_of_two()
    }

    /// Number of columns across all steps + constant column padded to next power of two.
    pub fn num_cols_total(&self) -> usize {
        2 * self.num_vars_total()
    }

    pub fn num_rows_total(&self) -> usize {
        self.num_cons_total
    }

    /// Padded number of constraint rows per step.
    pub fn padded_row_constraint_per_step(&self) -> usize {
        // JP: This is redundant with `padded_rows_per_step`. Can we reuse that instead?
        (self.uniform_r1cs.num_rows + self.offset_eq_r1cs.num_constraints()).next_power_of_two()
    }

    /// Number of bits needed for all rows.
    pub fn num_rows_bits(&self) -> usize {
        let row_count = self.num_steps * self.padded_row_constraint_per_step();
        row_count.next_power_of_two().log_2()
    }

    /// Evaluates A(r_x, y) + r_rlc * B(r_x, y) + r_rlc^2 * C(r_x, y) where r_x = r_constr || r_step for all y.
    #[tracing::instrument(skip_all, name = "UniformSpartanKey::evaluate_r1cs_mle_rlc")]
    pub fn evaluate_r1cs_mle_rlc(&self, r_constr: &[F], r_step: &[F], r_rlc: F) -> Vec<F> {
        assert_eq!(
            r_constr.len(),
            (self.uniform_r1cs.num_rows + 1).next_power_of_two().log_2()
        );
        assert_eq!(r_step.len(), self.num_steps.log_2());

        let eq_rx_step = EqPolynomial::evals(r_step);
        let eq_rx_constr = EqPolynomial::evals(r_constr);
        let first_non_uniform_row = self.uniform_r1cs.num_rows;
        let constant_column = self.uniform_r1cs.num_vars;

        // Computation strategy
        // 1. Compute the RLC of the repeated terms in A, B, C, and the constant column
        // 2. Expand this RLC to the full column y by multiplying by eq(rx_step, step_index) for each step
        // 3. Add the non uniform constraint rows

        let compute_repeated =
            |constraints: &SparseConstraints<F>, non_uni_constants: Option<Vec<F>>| -> Vec<F> {
                // +1 for constant
                let mut evals = unsafe_allocate_zero_vec(self.uniform_r1cs.num_vars + 1);
                for (row, col, val) in constraints.vars.iter() {
                    evals[*col] += mul_0_1_optimized(val, &eq_rx_constr[*row]);
                }

                for (row, val) in constraints.consts.iter() {
                    evals[constant_column] += mul_0_1_optimized(val, &eq_rx_constr[*row]);
                }

                if let Some(non_uni_constants) = non_uni_constants {
                    for (i, non_uni_constant) in non_uni_constants.iter().enumerate() {
                        evals[constant_column] +=
                            eq_rx_constr[first_non_uniform_row + i] * non_uni_constant;
                    }
                }

                evals
            };

        let (eq_constants, condition_constants) = self.offset_eq_r1cs.constants();
        let sm_a_r = compute_repeated(&self.uniform_r1cs.a, Some(eq_constants)); // V var entries
        let sm_b_r = compute_repeated(&self.uniform_r1cs.b, Some(condition_constants)); // V var entries
        let sm_c_r = compute_repeated(&self.uniform_r1cs.c, None); // V var entries

        let r_rlc_sq = r_rlc.square();
        let sm_rlc = sm_a_r
            .iter()
            .zip(sm_b_r.iter())
            .zip(sm_c_r.iter())
            .map(|((a, b), c)| *a + mul_0_1_optimized(b, &r_rlc) + mul_0_1_optimized(c, &r_rlc_sq))
            .collect::<Vec<F>>();

        let mut rlc = unsafe_allocate_zero_vec(self.num_cols_total());

        {
            let span = tracing::span!(tracing::Level::INFO, "big_rlc_computation");
            let _guard = span.enter();
            rlc.par_chunks_mut(self.num_steps)
                .take(self.uniform_r1cs.num_vars)
                .enumerate()
                .for_each(|(var_index, var_chunk)| {
                    if !sm_rlc[var_index].is_zero() {
                        for (step_index, item) in var_chunk.iter_mut().enumerate() {
                            *item = mul_0_1_optimized(&eq_rx_step[step_index], &sm_rlc[var_index]);
                        }
                    }
                });
        }

        rlc[self.num_vars_total()] = sm_rlc[self.uniform_r1cs.num_vars]; // constant

        // Handle non-uniform constraints
        let update_non_uni = |rlc: &mut Vec<F>,
                              offset: &SparseEqualityItem<F>,
                              non_uni_constraint_index: usize,
                              r: F| {
            for (col, is_offset, coeff) in offset.offset_vars.iter() {
                let offset = if *is_offset { 1 } else { 0 };

                // Ignores the offset overflow at the last step
                let y_index_range = *col * self.num_steps + offset..(*col + 1) * self.num_steps;
                let steps = (0..self.num_steps).into_par_iter();

                rlc[y_index_range]
                    .par_iter_mut()
                    .zip(steps)
                    .for_each(|(rlc_col, step_index)| {
                        *rlc_col += mul_0_1_optimized(&r, coeff)
                            * eq_rx_step[step_index]
                            * eq_rx_constr[first_non_uniform_row + non_uni_constraint_index];
                    });
            }
        };

        {
            let span = tracing::span!(tracing::Level::INFO, "update_non_uniform");
            let _guard = span.enter();
            for (i, constraint) in self.offset_eq_r1cs.constraints.iter().enumerate() {
                update_non_uni(&mut rlc, &constraint.eq, i, F::one());
                update_non_uni(&mut rlc, &constraint.condition, i, r_rlc);
            }
        }

        rlc
    }

    /// Evaluates the full expanded witness vector at 'r' using evaluations of segments.
    #[tracing::instrument(skip_all, name = "UniformSpartanKey::evaluate_z_mle")]
    pub fn evaluate_z_mle(&self, segment_evals: &[F], r: &[F]) -> F {
        assert_eq!(self.uniform_r1cs.num_vars, segment_evals.len());
        assert_eq!(r.len(), self.full_z_len().log_2());

        // Z can be computed in two halves, [Variables, (constant) 1, 0 , ...] indexed by the first bit.
        let r_const = r[0];
        let r_rest = &r[1..];
        assert_eq!(r_rest.len(), self.num_vars_total().log_2());

        // Don't need the last log2(num_steps) bits, they've been evaluated already.
        let var_bits = self.uniform_r1cs.num_vars.next_power_of_two().log_2();
        let r_var = &r_rest[..var_bits];

        let r_var_eq = EqPolynomial::evals(r_var);
        let eval_variables: F = (0..self.uniform_r1cs.num_vars)
            .map(|var_index| r_var_eq[var_index] * segment_evals[var_index])
            .sum();
        let eval_const: F = r_rest.iter().map(|r_i| F::one() - r_i).product();

        (F::one() - r_const) * eval_variables + r_const * eval_const
    }

    /// Evaluates A(r), B(r), C(r) efficiently using their small uniform representations.
    #[tracing::instrument(skip_all, name = "UniformSpartanKey::evaluate_r1cs_matrix_mles")]
    pub fn evaluate_r1cs_matrix_mles(&self, r_row: &[F], r_col: &[F]) -> (F, F, F) {
        let total_rows_bits = r_row.len();
        let total_cols_bits = r_col.len();
        let constraint_rows_bits = self.padded_row_constraint_per_step().log_2();
        let steps_bits: usize = total_rows_bits - constraint_rows_bits;
        let (r_row_step, r_row_constr) = r_row.split_at(total_rows_bits - constraint_rows_bits); // TMP
        let uniform_cols_bits = self.uniform_r1cs.num_vars.next_power_of_two().log_2();
        let (r_col_var, r_col_step) = r_col.split_at(uniform_cols_bits + 1);
        assert_eq!(r_row_step.len(), r_col_step.len());

        let eq_rx_ry_step = EqPolynomial::new(r_row_step.to_vec()).evaluate(r_col_step);
        let eq_rx_constr = EqPolynomial::evals(r_row_constr);
        let eq_ry_var = EqPolynomial::evals(r_col_var);

        let constant_column = index_to_field_bitvector(self.num_cols_total() / 2, total_cols_bits);
        let col_eq_constant = EqPolynomial::new(r_col.to_vec()).evaluate(&constant_column);

        let compute_uniform_matrix_mle = |constraints: &SparseConstraints<F>| -> F {
            let mut full_mle_evaluation: F = constraints
                .vars
                .iter()
                .map(|(row, col, coeff)| *coeff * eq_rx_constr[*row] * eq_ry_var[*col])
                .sum::<F>()
                * eq_rx_ry_step;

            full_mle_evaluation += constraints
                .consts
                .iter()
                .map(|(constraint_row, constant_coeff)| {
                    *constant_coeff * eq_rx_constr[*constraint_row]
                })
                .sum::<F>()
                * col_eq_constant;

            full_mle_evaluation
        };

        let mut a_mle = compute_uniform_matrix_mle(&self.uniform_r1cs.a);
        let mut b_mle = compute_uniform_matrix_mle(&self.uniform_r1cs.b);
        let c_mle = compute_uniform_matrix_mle(&self.uniform_r1cs.c);

        // Non-uniform constraints
        let eq_step_offset_1 = eq_plus_one(r_row_step, r_col_step, steps_bits);
        let compute_non_uniform = |non_uni: &SparseEqualityItem<F>| -> F {
            let mut non_uni_mle = non_uni
                .offset_vars
                .iter()
                .map(|(col, offset, coeff)| {
                    if !offset {
                        *coeff * eq_ry_var[*col] * eq_rx_ry_step
                    } else {
                        *coeff * eq_ry_var[*col] * eq_step_offset_1
                    }
                })
                .sum::<F>();

            non_uni_mle += non_uni.constant * col_eq_constant;

            non_uni_mle
        };

        for (i, constraint) in self.offset_eq_r1cs.constraints.iter().enumerate() {
            let non_uni_a = compute_non_uniform(&constraint.eq);
            let non_uni_b = compute_non_uniform(&constraint.condition);

            let non_uni_constraint_index =
                index_to_field_bitvector(self.uniform_r1cs.num_rows + i, constraint_rows_bits);
            let row_constr_eq_non_uni =
                EqPolynomial::new(r_row_constr.to_vec()).evaluate(&non_uni_constraint_index);

            assert_eq!(
                row_constr_eq_non_uni,
                eq_rx_constr[self.uniform_r1cs.num_rows + i]
            );

            a_mle += non_uni_a * row_constr_eq_non_uni;
            b_mle += non_uni_b * row_constr_eq_non_uni;
        }

        (a_mle, b_mle, c_mle)
    }

    /// Returns the digest of the r1cs shape
    fn digest(uniform_r1cs: &UniformR1CS<F>, offset_eq: &NonUniformR1CS<F>, num_steps: usize) -> F {
        let mut hash_bytes = Vec::new();
        uniform_r1cs.serialize_compressed(&mut hash_bytes).unwrap();
        let mut offset_eq_bytes = Vec::new();
        offset_eq
            .serialize_compressed(&mut offset_eq_bytes)
            .unwrap();
        hash_bytes.extend(offset_eq_bytes);
        hash_bytes.extend(num_steps.to_be_bytes().to_vec());
        let mut hasher = Sha3_256::new();
        hasher.update(hash_bytes);

        let map_to_field = |digest: &[u8]| -> F {
            let bv = (0..250).map(|i| {
                let (byte_pos, bit_pos) = (i / 8, i % 8);
                let bit = (digest[byte_pos] >> bit_pos) & 1;
                bit == 1
            });

            // turn the bit vector into a scalar
            let mut digest = F::zero();
            let mut coeff = F::one();
            for bit in bv {
                if bit {
                    digest += coeff;
                }
                coeff += coeff;
            }
            digest
        };
        map_to_field(&hasher.finalize())
    }
}
