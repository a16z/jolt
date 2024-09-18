use std::marker::PhantomData;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use sha3::Sha3_256;

use crate::{
    field::JoltField,
    poly::eq_poly::EqPolynomial,
    r1cs::special_polys::{eq_plus_one, SparsePolynomial},
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
        let num_steps = constraint_builder.uniform_repeat().next_power_of_two();

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
        let sm_a_r = compute_repeated(&self.uniform_r1cs.a, Some(eq_constants));
        let sm_b_r = compute_repeated(&self.uniform_r1cs.b, Some(condition_constants));
        let sm_c_r = compute_repeated(&self.uniform_r1cs.c, None);

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
        let const_poly = SparsePolynomial::new(self.num_vars_total().log_2(), vec![(F::one(), 0)]);
        let eval_const = const_poly.evaluate(r_rest);

        (F::one() - r_const) * eval_variables + r_const * eval_const
    }

    /// Evaluates A(r), B(r), C(r) efficiently using their small uniform representations.
    #[tracing::instrument(skip_all, name = "UniformSpartanKey::evaluate_r1cs_matrix_mles")]
    pub fn evaluate_r1cs_matrix_mles(&self, r: &[F]) -> (F, F, F) {
        let total_rows_bits = self.num_rows_total().log_2();
        let total_cols_bits = self.num_cols_total().log_2();
        let steps_bits: usize = self.num_steps.log_2();
        let constraint_rows_bits = (self.uniform_r1cs.num_rows + 1).next_power_of_two().log_2();
        let uniform_cols_bits = self.uniform_r1cs.num_vars.next_power_of_two().log_2();
        assert_eq!(r.len(), total_rows_bits + total_cols_bits);
        assert_eq!(total_rows_bits - steps_bits, constraint_rows_bits);

        // Deconstruct 'r' into representitive bits
        let (r_row, r_col) = r.split_at(total_rows_bits);
        let (r_row_constr, r_row_step) = r_row.split_at(constraint_rows_bits);
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

// #[cfg(test)]
// mod test {
//     use super::*;
//     use ark_bn254::Fr;
//     use ark_std::{One, Zero};

//     use crate::{
//         poly::dense_mlpoly::DensePolynomial,
//         r1cs::builder::{R1CSBuilder, R1CSConstraintBuilder},
//         utils::{index_to_field_bitvector, math::Math},
//     };

//     #[test]
//     fn materialize() {
//         let mut uniform_builder = R1CSBuilder::<Fr, TestInputs>::new();
//         // OpFlags0 * OpFlags1 == 12
//         struct TestConstraints();
//         impl<const C: usize, F: JoltField> R1CSConstraints<C, F> for TestConstraints {
//             type Inputs = TestInputs;
//             fn build_constraints(&self, builder: &mut R1CSBuilder<F, Self::Inputs>) {
//                 builder.constrain_prod(TestInputs::OpFlags0, TestInputs::OpFlags1, 12);
//             }
//         }

//         let constraints = TestConstraints();
//         constraints.build_constraints(&mut uniform_builder);
//         let _num_steps: usize = 3;
//         let num_steps_pad = 4;
//         let combined_builder =
//             CombinedUniformBuilder::construct(uniform_builder, num_steps_pad, vec![]);
//         let key = UniformSpartanKey::from_builder(&combined_builder);

//         let materialized_a = materialize_full_uniform(&key, &key.uniform_r1cs.a);
//         let materialized_b = materialize_full_uniform(&key, &key.uniform_r1cs.b);
//         let materialized_c = materialize_full_uniform(&key, &key.uniform_r1cs.c);

//         let row_width =
//             (TestInputs::COUNT.next_power_of_two() * num_steps_pad).next_power_of_two() * 2;
//         let op_flags_0_pos = (TestInputs::OpFlags0 as usize) * num_steps_pad;
//         assert_eq!(materialized_a[op_flags_0_pos], Fr::one());
//         assert_eq!(
//             materialized_b[(TestInputs::OpFlags1 as usize) * num_steps_pad],
//             Fr::one()
//         );
//         let const_col_index = row_width / 2;
//         assert_eq!(materialized_c[const_col_index], Fr::from(12));
//         assert_eq!(materialized_a[row_width + op_flags_0_pos + 1], Fr::one());
//         assert_eq!(materialized_c[row_width + const_col_index], Fr::from(12));
//         assert_eq!(
//             materialized_c[2 * row_width + const_col_index],
//             Fr::from(12)
//         );
//         assert_eq!(
//             materialized_c[3 * row_width + const_col_index],
//             Fr::from(12)
//         );
//     }

//     #[test]
//     fn evaluate_r1cs_mle_rlc() {
//         let (_builder, key) = simp_test_builder_key();
//         let (a, b, c) = simp_test_big_matrices();
//         let a = DensePolynomial::new(a);
//         let b = DensePolynomial::new(b);
//         let c = DensePolynomial::new(c);

//         let r_row_constr_len = (key.uniform_r1cs.num_rows + 1).next_power_of_two().log_2();
//         let r_col_step_len = key.num_steps.log_2();

//         let r_row_constr = vec![Fr::from(100), Fr::from(200)];
//         let r_row_step = vec![Fr::from(100), Fr::from(200)];
//         assert_eq!(r_row_constr.len(), r_row_constr_len);
//         assert_eq!(r_row_step.len(), r_col_step_len);
//         let r_rlc = Fr::from(1000);

//         let rlc = key.evaluate_r1cs_mle_rlc(&r_row_constr, &r_row_step, r_rlc);

//         // let row_coordinate_len = key.num_rows_total().log_2();
//         let col_coordinate_len = key.num_cols_total().log_2();
//         let row_coordinate: Vec<Fr> = [r_row_constr, r_row_step].concat();
//         for i in 0..key.num_cols_total() {
//             let col_coordinate = index_to_field_bitvector(i, col_coordinate_len);

//             let coordinate: Vec<Fr> = [row_coordinate.clone(), col_coordinate].concat();
//             let expected_rlc = a.evaluate(&coordinate)
//                 + r_rlc * b.evaluate(&coordinate)
//                 + r_rlc * r_rlc * c.evaluate(&coordinate);

//             assert_eq!(expected_rlc, rlc[i], "Failed at {i}");
//         }
//     }

//     #[test]
//     fn r1cs_matrix_mles_offset_constraints() {
//         let (_builder, key) = simp_test_builder_key();
//         let (big_a, big_b, big_c) = simp_test_big_matrices();

//         // Evaluate over boolean hypercube
//         let total_size = key.num_cols_total() * key.num_rows_total();
//         let r_len = total_size.log_2();
//         for i in 0..total_size {
//             let r = index_to_field_bitvector(i, r_len);
//             let (a_r, b_r, c_r) = key.evaluate_r1cs_matrix_mles(&r);

//             assert_eq!(big_a[i], a_r, "Error at index {}", i);
//             assert_eq!(big_b[i], b_r, "Error at index {}", i);
//             assert_eq!(big_c[i], c_r, "Error at index {}", i);
//         }

//         // Evaluate outside boolean hypercube
//         let mut r_outside = Vec::new();
//         for i in 0..9 {
//             r_outside.push(Fr::from(100 + i * 100));
//         }
//         let (a_r, b_r, c_r) = key.evaluate_r1cs_matrix_mles(&r_outside);
//         assert_eq!(
//             DensePolynomial::new(big_a.clone()).evaluate(&r_outside),
//             a_r
//         );
//         assert_eq!(
//             DensePolynomial::new(big_b.clone()).evaluate(&r_outside),
//             b_r
//         );
//         assert_eq!(
//             DensePolynomial::new(big_c.clone()).evaluate(&r_outside),
//             c_r
//         );
//     }

//     #[test]
//     fn z_mle() {
//         let mut uniform_builder = R1CSBuilder::<Fr, TestInputs>::new();
//         // OpFlags0 * OpFlags1 == 12
//         struct TestConstraints();
//         impl<const C: usize, F: JoltField> R1CSConstraints<C, F> for TestConstraints {
//             type Inputs = TestInputs;
//             fn build_constraints(&self, builder: &mut R1CSBuilder<F, Self::Inputs>) {
//                 builder.constrain_prod(TestInputs::OpFlags0, TestInputs::OpFlags1, 12);
//             }
//         }

//         let constraints = TestConstraints();
//         constraints.build_constraints(&mut uniform_builder);
//         let num_steps_pad = 4;
//         let combined_builder =
//             CombinedUniformBuilder::construct(uniform_builder, num_steps_pad, vec![]);
//         let mut inputs = vec![vec![Fr::zero(); num_steps_pad]; TestInputs::COUNT];

//         inputs[TestInputs::OpFlags0 as usize][0] = Fr::from(1);
//         inputs[TestInputs::OpFlags1 as usize][0] = Fr::from(12);

//         inputs[TestInputs::OpFlags0 as usize][1] = Fr::from(2);
//         inputs[TestInputs::OpFlags1 as usize][1] = Fr::from(6);

//         inputs[TestInputs::OpFlags0 as usize][2] = Fr::from(3);
//         inputs[TestInputs::OpFlags1 as usize][2] = Fr::from(4);

//         inputs[TestInputs::OpFlags0 as usize][3] = Fr::from(4);
//         inputs[TestInputs::OpFlags1 as usize][3] = Fr::from(3);

//         // Confirms validity of constraints
//         let (_az, _bz, _cz) = combined_builder.compute_spartan_Az_Bz_Cz(&inputs, &[]);

//         let key = UniformSpartanKey::from_builder(&combined_builder);

//         // Z's full padded length is 2 * (num_vars * num_steps.next_power_of_two())
//         let z_pad_len = 2 * num_steps_pad * TestInputs::COUNT.next_power_of_two();
//         let z_bits = z_pad_len.log_2();
//         assert_eq!(z_bits, 8);

//         // 1 bit to index const
//         // 5 bits to index variable
//         // 2 bits to index step
//         let r_const = vec![Fr::from(100)];
//         let r_var = vec![
//             Fr::from(200),
//             Fr::from(300),
//             Fr::from(400),
//             Fr::from(500),
//             Fr::from(600),
//         ];
//         let r_step = vec![Fr::from(100), Fr::from(200)];
//         let r = [r_const, r_var, r_step.clone()].concat();

//         let z_segment_evals: Vec<Fr> = inputs
//             .iter()
//             .map(|input_vec| {
//                 let poly = DensePolynomial::new_padded(input_vec.clone());
//                 assert_eq!(poly.len(), num_steps_pad);
//                 poly.evaluate(&r_step)
//             })
//             .collect();

//         // Construct the fully materialized version of 'z'
//         // Expecting form of Z
//         // [TestInputs::PCIn[0], ... PcIn[num_steps.next_pow_2 - 1],
//         //  TestInputs::PCOut[0], ... PcOut[num_steps.next_pow_2 - 1],
//         //  0 padding to num_vars.next_pow_2 * num_steps.next_pow_2
//         //  1
//         //  0 padding to 2 * num_vars.next_pow_2 * num_steps.next_pow_2
//         // ]
//         //
//         let mut z = Vec::with_capacity(z_pad_len);
//         for var_across_steps in inputs {
//             let new_padded_len = z.len() + num_steps_pad;
//             z.extend(var_across_steps);
//             z.resize(new_padded_len, Fr::zero());
//         }
//         let const_index = z_pad_len / 2;
//         z.resize(const_index, Fr::zero());
//         z.push(Fr::one());
//         z.resize(z_pad_len, Fr::zero());

//         let actual = key.evaluate_z_mle(&z_segment_evals, &r);
//         let expected = DensePolynomial::new(z).evaluate(&r);
//         assert_eq!(expected, actual);
//     }
// }
