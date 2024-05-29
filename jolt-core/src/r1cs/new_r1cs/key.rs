use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use sha3::Sha3_256;

use crate::{
    poly::{eq_poly::EqPolynomial, field::JoltField},
    utils::thread::unsafe_allocate_zero_vec,
};

use super::{
    builder::{CombinedUniformBuilder, SparseConstraints, UniformR1CS},
    ops::ConstraintInput,
};
use digest::Digest;

use crate::utils::math::Math;

use rayon::prelude::*;

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct UniformSpartanKey<F: JoltField> {
    pub uniform_r1cs: UniformR1CS<F>,

    /// Number of constraints across all steps padded to nearest power of 2
    pub num_cons_total: usize,

    /// Number of steps padded to the nearest power of 2
    pub num_steps: usize,

    /// Digest of verifier key
    pub(crate) vk_digest: F,
}

impl<F: JoltField> UniformSpartanKey<F> {
    pub fn from_builder<I: ConstraintInput>(
        constraint_builder: &CombinedUniformBuilder<F, I>,
    ) -> Self {
        let uniform_r1cs = constraint_builder.materialize_uniform();

        let total_rows = constraint_builder.constraint_rows().next_power_of_two();
        let num_steps = constraint_builder.uniform_repeat().next_power_of_two();

        // TODO(sragss): Need to digest non-uniform constraints as well.
        let vk_digest = Self::digest(&uniform_r1cs, num_steps);

        Self {
            uniform_r1cs,
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

    #[tracing::instrument(skip_all, name = "UniformSpartanKey::evaluate_r1cs_mle_rlc")]
    pub fn evaluate_r1cs_mle_rlc(&self, r_constr: &[F], r_step: &[F], r_rlc: F) -> Vec<F> {
        assert_eq!(
            r_constr.len(),
            self.uniform_r1cs.num_rows.next_power_of_two().log_2()
        );
        assert_eq!(r_step.len(), self.num_steps.log_2());

        let (sm_a_r, sm_b_r, sm_c_r) = self.evaluate_uniform_r1cs_at_row(r_constr);

        let r_rlc_sq = r_rlc.square();
        let sm_rlc = sm_a_r
            .iter()
            .zip(sm_b_r.iter())
            .zip(sm_c_r.iter())
            .map(|((a, b), c)| *a + r_rlc * b + r_rlc_sq * c)
            .collect::<Vec<F>>();

        let mut rlc = unsafe_allocate_zero_vec(self.num_cols_total());
        let eq_r_var_step = EqPolynomial::evals(r_step);

        rlc.par_chunks_mut(self.num_steps)
            .take(self.uniform_r1cs.num_vars)
            .enumerate()
            .for_each(|(var_index, var_chunk)| {
                if !sm_rlc[var_index].is_zero() {
                    for (step_index, item) in var_chunk.iter_mut().enumerate() {
                        *item = eq_r_var_step[step_index] * sm_rlc[var_index];
                    }
                }
            });

        rlc[self.num_vars_total()] = sm_rlc[self.uniform_r1cs.num_vars]; // constant

        rlc
    }

    /// Evaluates uniformA(r_x, _), uniformB(r_x, _), uniformC(r_x, _) assuming tight packing, no padding and tightly packed constant column.
    pub fn evaluate_uniform_r1cs_at_row(&self, r_row: &[F]) -> (Vec<F>, Vec<F>, Vec<F>) {
        assert_eq!(
            r_row.len(),
            self.uniform_r1cs.num_rows.next_power_of_two().log_2()
        );

        let eq_r_row = EqPolynomial::evals(r_row);

        let compute = |constraints: &SparseConstraints<F>| -> Vec<F> {
            // +1 for constant
            let mut evals = vec![F::zero(); self.uniform_r1cs.num_vars + 1];
            for (row, col, val) in constraints.vars.iter() {
                evals[*col] += eq_r_row[*row] * val;
            }

            for (row, val) in constraints.consts.iter() {
                evals[self.uniform_r1cs.num_vars] += eq_r_row[*row] * val;
            }

            evals
        };

        (
            compute(&self.uniform_r1cs.a),
            compute(&self.uniform_r1cs.b),
            compute(&self.uniform_r1cs.c),
        )
    }

    /// Evaluates the full expanded witness vector at 'r' using evaluations of segments.
    pub fn evaluate_z_mle(&self, segment_evals: &[F], r: &[F]) -> F {
        assert_eq!(self.uniform_r1cs.num_vars, segment_evals.len());
        assert_eq!(r.len(), self.full_z_len().log_2());

        println!("constant index: {}", self.num_vars_total());
        println!("witness semgnts len: {}", self.num_vars_total());
        println!("total len {}", self.full_z_len());

        // Z can be computed in two halves, [Variables, (constant) 1, 0 , ...] indexed by the first bit.
        let r_const = r[0];
        let r_rest = &r[1..];
        assert_eq!(r_rest.len(), self.num_vars_total().log_2());

        // Don't need the last log2(num_steps) bits, they've been evaluated already.
        let var_bits = self.uniform_r1cs.num_vars.next_power_of_two().log_2();
        let r_var = &r_rest[..var_bits];

        let eval_variables = (0..self.uniform_r1cs.num_vars)
            .map(|var_index| {
                // write var_index in binary
                let bin = format!("{var_index:0width$b}", width = var_bits);

                let product = bin.chars().enumerate().fold(F::one(), |acc, (j, bit)| {
                    acc * if bit == '0' {
                        F::one() - r_var[j]
                    } else {
                        r_var[j]
                    }
                });

                product * segment_evals[var_index]
            })
            .sum::<F>();
        let const_poly = SparsePolynomial::new(self.num_vars_total().log_2(), vec![(0, F::one())]);
        let eval_const = const_poly.evaluate(r_rest);

        (F::one() - r_const) * eval_variables + r_const * eval_const
    }

    /// Evaluates A(r), B(r), C(r) efficiently using their small uniform representations.
    pub fn evaluate_r1cs_matrix_mles(&self, r: &[F]) -> (F, F, F) {
        let total_rows_bits = self.num_rows_total().log_2();
        let total_cols_bits = self.num_cols_total().log_2();
        let steps_bits = self.num_steps.log_2();
        let uniform_rows_bits = self.uniform_r1cs.num_rows.next_power_of_two().log_2();
        let uniform_cols_bits = self.uniform_r1cs.num_vars.next_power_of_two().log_2();
        assert_eq!(r.len(), total_rows_bits + total_cols_bits);
        assert_eq!(total_rows_bits - steps_bits, uniform_rows_bits);

        // Deconstruct 'r' into representitive bits
        let (r_row, r_col) = r.split_at(total_rows_bits);
        let (r_row_constr, r_row_step) = r_row.split_at(uniform_rows_bits);
        let (r_col_var, r_col_step) = r_col.split_at(uniform_cols_bits + 1); // TODO(sragss): correct?
        assert_eq!(r_row_step.len(), r_col_step.len());

        let eq_rx_ry_ts = EqPolynomial::new(r_row_step.to_vec()).evaluate(r_col_step);
        let eq_rx_con = EqPolynomial::evals(&r_row_constr);
        let eq_ry_var = EqPolynomial::evals(&r_col_var);

        // TODO(sragss): Must be able to dedupe
        let eq_r_row = EqPolynomial::evals(r_row);

        let compute = |constraints: &SparseConstraints<F>| -> F {
            let var_evaluation: F = constraints
                .vars
                .iter()
                .map(|(row, col, coeff)| {
                    // Note: row indexes constraints, col indexes vars
                    *coeff * eq_rx_ry_ts * eq_rx_con[*row] * eq_ry_var[*col]
                })
                .sum();

            // Constant (second half of each row)
            let r_col_const: F = r_col[0]
                * (1..total_cols_bits)
                    .into_iter()
                    .map(|i| F::one() - r_col[i])
                    .product::<F>();
            let mut sum = F::zero();
            for (constraint_row, constant_coeff) in &constraints.consts {
                let mut eq_summation = F::zero();
                for step_index in 0..self.num_steps {
                    // TODO(sragss): Are these bounds right? Verifier now linear.
                    let packed_row_index = constraint_row * (1 << steps_bits) + step_index;
                    eq_summation += eq_r_row[packed_row_index];
                }
                sum += *constant_coeff * eq_summation;
            }
            var_evaluation + r_col_const * sum
        };

        (
            compute(&self.uniform_r1cs.a),
            compute(&self.uniform_r1cs.b),
            compute(&self.uniform_r1cs.c),
        )
    }

    /// Returns the digest of the r1cs shape
    fn digest(uniform_r1cs: &UniformR1CS<F>, num_steps: usize) -> F {
        let mut compressed_bytes = Vec::new();
        uniform_r1cs
            .serialize_compressed(&mut compressed_bytes)
            .unwrap();
        compressed_bytes.extend(num_steps.to_be_bytes().to_vec());
        let mut hasher = Sha3_256::new();
        hasher.input(compressed_bytes);

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
        map_to_field(&hasher.result())
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
mod test {
    use super::*;
    use ark_bn254::Fr;

    use crate::{
        poly::dense_mlpoly::DensePolynomial,
        r1cs::new_r1cs::{
            builder::{R1CSBuilder, R1CSConstraintBuilder, SparseConstraints},
            test::TestInputs,
        },
        utils::{index_to_field_bitvector, math::Math},
    };
    use strum::EnumCount;

    fn materialize_full<F: JoltField>(
        key: &UniformSpartanKey<F>,
        sparse_constraints: &SparseConstraints<F>,
    ) -> Vec<F> {
        let row_width = 2 * key.num_vars_total().next_power_of_two();
        let col_height = key.num_cons_total;
        let total_size = row_width * col_height;
        assert!(total_size.is_power_of_two());
        let mut materialized = vec![F::zero(); total_size];

        for (row, col, val) in sparse_constraints.vars.iter() {
            for step_index in 0..key.num_steps {
                let x = col * key.num_steps + step_index;
                let y = row * key.num_steps + step_index;
                let i = y * row_width + x;
                materialized[i] = *val;
            }
        }

        let const_col_index = key.num_vars_total();
        for (row, val) in sparse_constraints.consts.iter() {
            for step_index in 0..key.num_steps {
                let y = row * key.num_steps + step_index;
                let i = y * row_width + const_col_index;
                materialized[i] = *val;
            }
        }

        materialized
    }

    fn materialize_all<F: JoltField>(key: &UniformSpartanKey<F>) -> (Vec<F>, Vec<F>, Vec<F>) {
        (
            materialize_full(&key, &key.uniform_r1cs.a),
            materialize_full(&key, &key.uniform_r1cs.b),
            materialize_full(&key, &key.uniform_r1cs.c),
        )
    }

    #[test]
    fn materialize() {
        let mut uniform_builder = R1CSBuilder::<Fr, TestInputs>::new();
        // OpFlags0 * OpFlags1 == 12
        struct TestConstraints();
        impl<F: JoltField> R1CSConstraintBuilder<F> for TestConstraints {
            type Inputs = TestInputs;
            fn build_constraints(&self, builder: &mut R1CSBuilder<F, Self::Inputs>) {
                builder.constrain_prod(TestInputs::OpFlags0, TestInputs::OpFlags1, 12);
            }
        }

        let constraints = TestConstraints();
        constraints.build_constraints(&mut uniform_builder);
        let num_steps: usize = 3;
        let num_steps_pad = 4;
        let combined_builder =
            CombinedUniformBuilder::construct(uniform_builder, num_steps, vec![]);
        let key = UniformSpartanKey::from_builder(&combined_builder);

        let materialized_a = materialize_full(&key, &key.uniform_r1cs.a);
        let materialized_b = materialize_full(&key, &key.uniform_r1cs.b);
        let materialized_c = materialize_full(&key, &key.uniform_r1cs.c);

        let row_width =
            (TestInputs::COUNT.next_power_of_two() * num_steps_pad).next_power_of_two() * 2;
        let op_flags_0_pos = (TestInputs::OpFlags0 as usize) * num_steps_pad;
        assert_eq!(materialized_a[op_flags_0_pos], Fr::one());
        assert_eq!(
            materialized_b[(TestInputs::OpFlags1 as usize) * num_steps_pad],
            Fr::one()
        );
        let const_col_index = row_width / 2;
        assert_eq!(materialized_c[const_col_index], Fr::from(12));
        assert_eq!(materialized_a[row_width + op_flags_0_pos + 1], Fr::one());
        assert_eq!(materialized_c[row_width + const_col_index], Fr::from(12));
        assert_eq!(
            materialized_c[2 * row_width + const_col_index],
            Fr::from(12)
        );
        assert_eq!(
            materialized_c[3 * row_width + const_col_index],
            Fr::from(12)
        );
    }

    #[test]
    fn evaluate_r1cs_mle_rlc() {
        let mut uniform_builder = R1CSBuilder::<Fr, TestInputs>::new();
        // PcIn * PcOut == 12
        // BytecodeA == BytecodeVOpcode + PCIn
        struct TestConstraints();
        impl<F: JoltField> R1CSConstraintBuilder<F> for TestConstraints {
            type Inputs = TestInputs;
            fn build_constraints(&self, builder: &mut R1CSBuilder<F, Self::Inputs>) {
                builder.constrain_prod(TestInputs::PcIn, TestInputs::PcOut, 12);
                builder.constrain_eq(
                    TestInputs::BytecodeA,
                    TestInputs::BytecodeVOpcode + TestInputs::PcIn,
                );
            }
        }

        let constraints = TestConstraints();
        constraints.build_constraints(&mut uniform_builder);
        let num_steps: usize = 3;
        let _num_steps_pad = 4;
        let combined_builder =
            CombinedUniformBuilder::construct(uniform_builder, num_steps, vec![]);
        let key = UniformSpartanKey::from_builder(&combined_builder);

        let (a, b, c) = materialize_all(&key);
        let (a, b, c) = (
            DensePolynomial::new(a),
            DensePolynomial::new(b),
            DensePolynomial::new(c),
        );

        let r_row_constr_len = key.uniform_r1cs.num_rows.next_power_of_two().log_2();
        let r_col_step_len = key.num_steps.log_2();

        let r_row_constr = vec![Fr::from(100)];
        let r_row_step = vec![Fr::from(100), Fr::from(200)];
        assert_eq!(r_row_constr.len(), r_row_constr_len);
        assert_eq!(r_row_step.len(), r_col_step_len);
        let r_rlc = Fr::from(1000);

        let rlc = key.evaluate_r1cs_mle_rlc(&r_row_constr, &r_row_step, r_rlc);

        // let row_coordinate_len = key.num_rows_total().log_2();
        let col_coordinate_len = key.num_cols_total().log_2();
        let row_coordinate: Vec<Fr> = [r_row_constr, r_row_step].concat();
        for i in 0..key.num_cols_total() {
            let col_coordinate = index_to_field_bitvector(i, col_coordinate_len);

            let coordinate: Vec<Fr> = [row_coordinate.clone(), col_coordinate].concat();
            let expected_rlc = a.evaluate(&coordinate)
                + r_rlc * b.evaluate(&coordinate)
                + r_rlc * r_rlc * c.evaluate(&coordinate);

            assert_eq!(expected_rlc, rlc[i], "Failed at {i}");
        }
    }

    #[test]
    fn r1cs_matrix_mles() {
        let mut uniform_builder = R1CSBuilder::<Fr, TestInputs>::new();
        // OpFlags0 * OpFlags1 == 12
        struct TestConstraints();
        impl<F: JoltField> R1CSConstraintBuilder<F> for TestConstraints {
            type Inputs = TestInputs;
            fn build_constraints(&self, builder: &mut R1CSBuilder<F, Self::Inputs>) {
                builder.constrain_prod(TestInputs::OpFlags0, TestInputs::OpFlags1, 12);
            }
        }

        let constraints = TestConstraints();
        constraints.build_constraints(&mut uniform_builder);
        let num_steps: usize = 3;
        let _num_steps_pad = 4;
        let combined_builder =
            CombinedUniformBuilder::construct(uniform_builder, num_steps, vec![]);
        let key = UniformSpartanKey::from_builder(&combined_builder);

        let big_a = DensePolynomial::new(materialize_full(&key, &key.uniform_r1cs.a));
        let big_b = DensePolynomial::new(materialize_full(&key, &key.uniform_r1cs.b));
        let big_c = DensePolynomial::new(materialize_full(&key, &key.uniform_r1cs.c));

        let r_len = (key.num_cols_total() * key.num_rows_total()).log_2();
        let r = vec![
            Fr::from(100),
            Fr::from(200),
            Fr::from(300),
            Fr::from(400),
            Fr::from(500),
            Fr::from(600),
            Fr::from(700),
            Fr::from(800),
            Fr::from(900),
            Fr::from(1000),
        ];
        assert_eq!(r.len(), r_len);
        let (a_r, b_r, c_r) = key.evaluate_r1cs_matrix_mles(&r);

        assert_eq!(big_a.evaluate(&r), a_r);
        assert_eq!(big_b.evaluate(&r), b_r);
        assert_eq!(big_c.evaluate(&r), c_r);
    }

    #[test]
    fn z_mle() {
        let mut uniform_builder = R1CSBuilder::<Fr, TestInputs>::new();
        // OpFlags0 * OpFlags1 == 12
        struct TestConstraints();
        impl<F: JoltField> R1CSConstraintBuilder<F> for TestConstraints {
            type Inputs = TestInputs;
            fn build_constraints(&self, builder: &mut R1CSBuilder<F, Self::Inputs>) {
                builder.constrain_prod(TestInputs::OpFlags0, TestInputs::OpFlags1, 12);
            }
        }

        let constraints = TestConstraints();
        constraints.build_constraints(&mut uniform_builder);
        let num_steps: usize = 3;
        let num_steps_pad = 4;
        let combined_builder =
            CombinedUniformBuilder::construct(uniform_builder, num_steps, vec![]);
        let mut inputs = vec![vec![Fr::zero(); num_steps]; TestInputs::COUNT];

        inputs[TestInputs::OpFlags0 as usize][0] = Fr::from(1);
        inputs[TestInputs::OpFlags1 as usize][0] = Fr::from(12);

        inputs[TestInputs::OpFlags0 as usize][1] = Fr::from(2);
        inputs[TestInputs::OpFlags1 as usize][1] = Fr::from(6);

        inputs[TestInputs::OpFlags0 as usize][2] = Fr::from(3);
        inputs[TestInputs::OpFlags1 as usize][2] = Fr::from(4);

        // Confirms validity of constraints
        let (_az, _bz, _cz) = combined_builder.compute_spartan(&inputs, &vec![]);

        let key = UniformSpartanKey::from_builder(&combined_builder);

        // Z's full padded length is 2 * (num_vars * num_steps.next_power_of_two())
        let z_pad_len = 2 * num_steps_pad * TestInputs::COUNT.next_power_of_two();
        let z_bits = z_pad_len.log_2();
        assert_eq!(z_bits, 8);

        // 1 bit to index const
        // 5 bits to index variable
        // 2 bits to index step
        let r_const = vec![Fr::from(100)];
        let r_var = vec![
            Fr::from(200),
            Fr::from(300),
            Fr::from(400),
            Fr::from(500),
            Fr::from(600),
        ];
        let r_step = vec![Fr::from(100), Fr::from(200)];
        let r = [r_const, r_var, r_step.clone()].concat();

        let z_segment_evals: Vec<Fr> = inputs
            .iter()
            .map(|input_vec| {
                let poly = DensePolynomial::new_padded(input_vec.clone());
                assert_eq!(poly.len(), num_steps_pad);
                poly.evaluate(&r_step)
            })
            .collect();

        // Construct the fully materialized version of 'z'
        // Expecting form of Z
        // [TestInputs::PCIn[0], ... PcIn[num_steps.next_pow_2 - 1],
        //  TestInputs::PCOut[0], ... PcOut[num_steps.next_pow_2 - 1],
        //  0 padding to num_vars.next_pow_2 * num_steps.next_pow_2
        //  1
        //  0 padding to 2 * num_vars.next_pow_2 * num_steps.next_pow_2
        // ]
        //
        let mut z = Vec::with_capacity(z_pad_len);
        for var_across_steps in inputs {
            let new_padded_len = z.len() + num_steps_pad;
            z.extend(var_across_steps);
            z.resize(new_padded_len, Fr::zero());
        }
        let const_index = z_pad_len / 2;
        z.resize(const_index, Fr::zero());
        z.push(Fr::one());
        z.resize(z_pad_len, Fr::zero());

        let actual = key.evaluate_z_mle(&z_segment_evals, &r);
        let expected = DensePolynomial::new(z).evaluate(&r);
        assert_eq!(expected, actual);
    }
}
