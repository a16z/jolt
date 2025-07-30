use sha3::Sha3_256;

use super::builder::CombinedUniformBuilder;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_core::{
    field::JoltField,
    poly::eq_poly::EqPolynomial,
    r1cs::key::SparseConstraints,
    utils::{index_to_field_bitvector, mul_0_1_optimized, thread::unsafe_allocate_zero_vec},
};
use sha3::Digest;

use jolt_core::utils::math::Math;

pub struct UniformSpartanKey<F: JoltField> {
    pub uniform_r1cs: UniformR1CS<F>,

    /// Number of constraints across all steps padded to nearest power of 2
    pub num_cons_total: usize,

    /// Number of steps padded to the nearest power of 2
    pub num_steps: usize,

    /// Digest of verifier key
    pub vk_digest: F,
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

impl<F: JoltField> UniformSpartanKey<F> {
    pub fn from_builder(constraint_builder: &CombinedUniformBuilder<F>) -> Self {
        let uniform_r1cs = constraint_builder.materialize_uniform();

        let total_rows = constraint_builder.constraint_rows().next_power_of_two();
        let num_steps = constraint_builder.uniform_repeat().next_power_of_two(); // TODO(JP): Number of steps no longer need to be padded.

        let vk_digest = Self::digest(&uniform_r1cs, num_steps);

        Self {
            uniform_r1cs,
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
        self.uniform_r1cs.num_rows.next_power_of_two()
    }

    /// Number of bits needed for all rows.
    pub fn num_rows_bits(&self) -> usize {
        let row_count = self.num_steps * self.padded_row_constraint_per_step();
        row_count.next_power_of_two().log_2()
    }

    /// Evaluate the RLC of A_small, B_small, C_small matrices at (r_constr, y_var)
    /// This function only handles uniform constraints, ignoring cross-step constraints
    /// Returns evaluations for each y_var
    #[tracing::instrument(skip_all, name = "UniformSpartanKey::evaluate_small_matrix_rlc")]
    pub fn evaluate_small_matrix_rlc(&self, r_constr: &[F], r_rlc: F) -> Vec<F> {
        assert_eq!(
            r_constr.len(),
            (self.uniform_r1cs.num_rows + 1).next_power_of_two().log_2()
        );

        let eq_rx_constr = EqPolynomial::evals(r_constr);
        let num_vars_padded = self.uniform_r1cs.num_vars.next_power_of_two();
        // The constant column is at position num_vars (within the padded allocation)
        let constant_column = self.uniform_r1cs.num_vars;

        // Helper function to evaluate a single small matrix
        let evaluate_small_matrix = |constraints: &SparseConstraints<F>| -> Vec<F> {
            // Allocate vector with power-of-2 size
            let mut evals = unsafe_allocate_zero_vec(num_vars_padded);

            // Evaluate non-constant terms
            for (row, col, val) in constraints.vars.iter() {
                evals[*col] += mul_0_1_optimized(val, &eq_rx_constr[*row]);
            }

            // Evaluate constant terms
            for (row, val) in constraints.consts.iter() {
                evals[constant_column] += mul_0_1_optimized(val, &eq_rx_constr[*row]);
            }

            evals
        };

        // Evaluate A_small, B_small, C_small
        let a_small_evals = evaluate_small_matrix(&self.uniform_r1cs.a);
        let b_small_evals = evaluate_small_matrix(&self.uniform_r1cs.b);
        let c_small_evals = evaluate_small_matrix(&self.uniform_r1cs.c);

        // Compute RLC: A_small + r_rlc * B_small + r_rlc^2 * C_small
        let r_rlc_sq = r_rlc.square();
        a_small_evals
            .iter()
            .zip(b_small_evals.iter())
            .zip(c_small_evals.iter())
            .map(|((a, b), c)| *a + mul_0_1_optimized(b, &r_rlc) + mul_0_1_optimized(c, &r_rlc_sq))
            .collect()
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
        assert_eq!(r.len(), self.num_vars_uniform_padded().log_2());

        // Variables vector is [vars, ..., 1, ...] where ... denotes padding to power of 2
        let num_vars = self.num_vars_uniform_padded();
        let var_bits = num_vars.log_2();

        let eq_ry_var = EqPolynomial::evals(r);
        let eval_variables: F = (0..self.uniform_r1cs.num_vars)
            .map(|var_index| eq_ry_var[var_index] * segment_evals[var_index])
            .sum();

        // Evaluate at the constant position if it exists within the padded space
        let const_eval = if self.uniform_r1cs.num_vars < num_vars && with_const {
            let const_position_bits =
                index_to_field_bitvector(self.uniform_r1cs.num_vars as u64, var_bits);
            EqPolynomial::mle(r, &const_position_bits)
        } else {
            F::zero()
        };

        eval_variables + const_eval
    }

    /// Evaluate uniform matrix A at a specific point (rx_constr, ry_var)
    pub fn evaluate_uniform_a_at_point(&self, rx_constr: &[F], ry_var: &[F]) -> F {
        self.evaluate_uniform_matrix_at_point(&self.uniform_r1cs.a, rx_constr, ry_var)
    }

    /// Evaluate uniform matrix B at a specific point (rx_constr, ry_var)
    pub fn evaluate_uniform_b_at_point(&self, rx_constr: &[F], ry_var: &[F]) -> F {
        self.evaluate_uniform_matrix_at_point(&self.uniform_r1cs.b, rx_constr, ry_var)
    }

    /// Evaluate uniform matrix C at a specific point (rx_constr, ry_var)
    pub fn evaluate_uniform_c_at_point(&self, rx_constr: &[F], ry_var: &[F]) -> F {
        self.evaluate_uniform_matrix_at_point(&self.uniform_r1cs.c, rx_constr, ry_var)
    }

    /// Helper function to evaluate a uniform matrix at a specific point
    fn evaluate_uniform_matrix_at_point(
        &self,
        constraints: &SparseConstraints<F>,
        rx_constr: &[F],
        ry_var: &[F],
    ) -> F {
        let mut eval = F::zero();

        // Evaluate non-constant terms
        for (row, col, val) in constraints.vars.iter() {
            let row_bits = index_to_field_bitvector(*row as u64, rx_constr.len());
            let col_bits = index_to_field_bitvector(*col as u64, ry_var.len());
            eval += *val
                * EqPolynomial::mle(rx_constr, &row_bits)
                * EqPolynomial::mle(ry_var, &col_bits);
        }

        // Evaluate constant terms
        let constant_column = self.uniform_r1cs.num_vars;
        let const_col_bits = index_to_field_bitvector(constant_column as u64, ry_var.len());
        let eq_ry_const = EqPolynomial::mle(ry_var, &const_col_bits);

        for (row, val) in constraints.consts.iter() {
            let row_bits = index_to_field_bitvector(*row as u64, rx_constr.len());
            eval += *val * EqPolynomial::mle(rx_constr, &row_bits) * eq_ry_const;
        }

        eval
    }

    /// Returns the digest of the r1cs shape
    fn digest(uniform_r1cs: &UniformR1CS<F>, num_steps: usize) -> F {
        let mut hash_bytes = Vec::new();
        uniform_r1cs.serialize_compressed(&mut hash_bytes).unwrap();
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
