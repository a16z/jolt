use std::marker::PhantomData;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use sha3::Sha3_256;

use crate::{
    field::JoltField,
    poly::eq_poly::EqPolynomial,
    utils::{index_to_field_bitvector, mul_0_1_optimized, thread::unsafe_allocate_zero_vec},
};

use super::{builder::CombinedUniformBuilder, inputs::ConstraintInput};
use sha3::Digest;

use crate::utils::math::Math;

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

    pub fn num_vars_uniform_padded(&self) -> usize {
        self.uniform_r1cs.num_vars.next_power_of_two()
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

    /// (Prover) Evaluates RLC over A, B, C of: [A(r_x, y_var || r_x_step), A_shift(..)] for all y_var
    #[tracing::instrument(skip_all, name = "UniformSpartanKey::evaluate_r1cs_mle_rlc")]
    pub fn evaluate_matrix_mle_partial(&self, r_constr: &[F], r_step: &[F], r_rlc: F) -> Vec<F> {
        assert_eq!(
            r_constr.len(),
            (self.uniform_r1cs.num_rows + 1).next_power_of_two().log_2()
        );
        assert_eq!(r_step.len(), self.num_steps.log_2());

        let eq_rx_constr = EqPolynomial::evals(r_constr);
        let first_non_uniform_row = self.uniform_r1cs.num_rows;
        let constant_column = self.uniform_r1cs.num_vars.next_power_of_two();

        // Computation strategy:
        // 1. Compute A(r_x, y_var || r_x_step) for each y_var by iterating over terms in uniform (small per-step) matrix
        // 2. Incorporate just the constant values from non-uniform constraints here
        let compute_repeated =
            |constraints: &SparseConstraints<F>, non_uni_constants: Option<Vec<F>>| -> Vec<F> {
                // evals structure: [inputs, aux ... 1, cross_inputs, cross_aux ...] where ... indicates padding to next power of 2
                let mut evals =
                    unsafe_allocate_zero_vec(self.uniform_r1cs.num_vars.next_power_of_two() * 4); // *4 instead of *2 to accommodate cross-step constraints
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
        let sm_a_r = compute_repeated(&self.uniform_r1cs.a, Some(eq_constants));
        let sm_b_r = compute_repeated(&self.uniform_r1cs.b, Some(condition_constants));
        let sm_c_r = compute_repeated(&self.uniform_r1cs.c, None);

        let r_rlc_sq = r_rlc.square();
        let mut sm_rlc = sm_a_r
            .iter()
            .zip(sm_b_r.iter())
            .zip(sm_c_r.iter())
            .map(|((a, b), c)| *a + mul_0_1_optimized(b, &r_rlc) + mul_0_1_optimized(c, &r_rlc_sq))
            .collect::<Vec<F>>();

        // 3. Add non-constant variables from non-uniform constraints here,
        // depending on which type of variables (current step or next) they involve.
        let update_non_uni = |rlc: &mut Vec<F>,
                              offset: &SparseEqualityItem<F>,
                              non_uni_constraint_index: usize,
                              r: F| {
            for (col, is_offset, coeff) in offset.offset_vars.iter() {
                let offset = if *is_offset { 1 } else { 0 };
                let col = *col + offset * constant_column * 2;
                rlc[col] += mul_0_1_optimized(&r, coeff)
                    * eq_rx_constr[first_non_uniform_row + non_uni_constraint_index];
            }
        };

        for (i, constraint) in self.offset_eq_r1cs.constraints.iter().enumerate() {
            update_non_uni(&mut sm_rlc, &constraint.eq, i, F::one());
            update_non_uni(&mut sm_rlc, &constraint.condition, i, r_rlc);
        }

        sm_rlc
    }

    /// (Verifier) Evaluates the full expanded witness vector at 'r' using evaluations of segments.
    #[tracing::instrument(
        skip_all,
        name = "UniformSpartanKey::evaluate_z_mle_with_segment_evals"
    )]
    pub fn evaluate_z_mle_with_segment_evals(
        &self,
        segment_evals: &[F],
        r: &[F],
        with_const: bool,
    ) -> F {
        assert_eq!(self.uniform_r1cs.num_vars, segment_evals.len());
        assert_eq!(r.len(), self.num_vars_uniform_padded().log_2() + 1);

        // Variables vector is [vars, ..., 1, ...] where ... denotes padding to power of 2
        let num_vars = self.num_vars_uniform_padded();
        let var_bits = num_vars.log_2();
        let var_and_const_bits = var_bits + 1;

        let r_const = r[0];
        let r_var = &r[1..var_bits + 1];

        let eq_ry_var = EqPolynomial::evals(r_var);
        let eval_variables: F = (0..self.uniform_r1cs.num_vars)
            .map(|var_index| eq_ry_var[var_index] * segment_evals[var_index])
            .sum();

        let eq_const = EqPolynomial::new(r[..var_and_const_bits].to_vec()).evaluate(
            &index_to_field_bitvector(1 << (var_and_const_bits - 1), var_and_const_bits),
        );

        let const_coeff = if with_const { F::one() } else { F::zero() };

        ((F::one() - r_const) * eval_variables) + eq_const * const_coeff
    }

    /// (Verifier) Evaluates uniform and non-uniform matrix MLEs with all variables fixed.
    #[tracing::instrument(skip_all, name = "UniformSpartanKey::evaluate_matrix_mle_full")]
    pub fn evaluate_matrix_mle_full(
        &self,
        rx_constr: &[F],
        ry_var: &[F],
        r_non_uni: &F,
    ) -> (F, F, F) {
        let constraint_rows_bits = (self.uniform_r1cs.num_rows + 1).next_power_of_two().log_2();
        let num_vars_bits = self.num_vars_uniform_padded().log_2();

        let eq_rx_constr = EqPolynomial::evals(rx_constr);
        let eq_ry_var = EqPolynomial::evals(ry_var);

        let constant_column = index_to_field_bitvector(
            self.num_cols_total() / 2 / self.num_steps,
            num_vars_bits + 1,
        );

        let col_eq_constant = EqPolynomial::new(ry_var.to_vec()).evaluate(&constant_column);

        let compute_uniform_matrix_mle = |constraints: &SparseConstraints<F>| -> F {
            let mut full_mle_evaluation: F = constraints
                .vars
                .iter()
                .map(|(row, col, coeff)| *coeff * eq_rx_constr[*row] * eq_ry_var[*col])
                .sum::<F>();

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
        let mut c_mle = compute_uniform_matrix_mle(&self.uniform_r1cs.c);

        // Non-uniform constraints
        let mut non_uni_a_mle = F::zero();
        let mut non_uni_b_mle = F::zero();

        let compute_non_uniform =
            |uni_mle: &mut F, non_uni_mle: &mut F, non_uni: &SparseEqualityItem<F>, eq_rx: F| {
                for (col, offset, coeff) in &non_uni.offset_vars {
                    if !offset {
                        *uni_mle += *coeff * eq_ry_var[*col] * eq_rx;
                    } else {
                        *non_uni_mle += *coeff * eq_ry_var[*col] * eq_rx;
                    }
                }
            };

        for (i, constraint) in self.offset_eq_r1cs.constraints.iter().enumerate() {
            let non_uni_constraint_index =
                index_to_field_bitvector(self.uniform_r1cs.num_rows + i, constraint_rows_bits);
            let row_constr_eq_non_uni =
                EqPolynomial::new(rx_constr.to_vec()).evaluate(&non_uni_constraint_index);

            assert_eq!(
                row_constr_eq_non_uni,
                eq_rx_constr[self.uniform_r1cs.num_rows + i]
            );

            compute_non_uniform(
                &mut a_mle,
                &mut non_uni_a_mle,
                &constraint.eq,
                row_constr_eq_non_uni,
            );
            compute_non_uniform(
                &mut b_mle,
                &mut non_uni_b_mle,
                &constraint.condition,
                row_constr_eq_non_uni,
            );
        }

        // Need to handle constants because they're defined separately in the non-uniform constraints
        let compute_non_uni_constants = |uni_mle: &mut F, non_uni_constants: Option<Vec<F>>| {
            if let Some(non_uni_constants) = non_uni_constants {
                for (i, non_uni_constant) in non_uni_constants.iter().enumerate() {
                    let first_non_uniform_row = self.uniform_r1cs.num_rows;
                    *uni_mle += eq_rx_constr[first_non_uniform_row + i]
                        * non_uni_constant
                        * col_eq_constant;
                }
            }
        };

        let (eq_constants, condition_constants) = self.offset_eq_r1cs.constants();
        compute_non_uni_constants(&mut a_mle, Some(eq_constants));
        compute_non_uni_constants(&mut b_mle, Some(condition_constants));

        a_mle = (F::one() - r_non_uni) * a_mle + *r_non_uni * non_uni_a_mle;
        b_mle = (F::one() - r_non_uni) * b_mle + *r_non_uni * non_uni_b_mle;
        c_mle = (F::one() - r_non_uni) * c_mle;

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
