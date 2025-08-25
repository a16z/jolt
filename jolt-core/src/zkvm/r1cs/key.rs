use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use sha3::Sha3_256;

use crate::{
    field::JoltField,
    poly::eq_poly::EqPolynomial,
    utils::{index_to_field_bitvector, thread::unsafe_allocate_zero_vec},
};

use sha3::Digest;

use super::constraints::{Constraint, LC, UNIFORM_R1CS};
use crate::utils::math::Math;
use crate::zkvm::r1cs::inputs::JoltR1CSInputs;

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct UniformSpartanKey<F: JoltField> {
    /// Number of constraints across all steps padded to nearest power of 2
    pub num_cons_total: usize,

    /// Number of steps padded to the nearest power of 2
    pub num_steps: usize,

    /// Digest of verifier key
    pub(crate) vk_digest: F,
}

/// (row, col, value)
pub type Coeff<F> = (usize, usize, F);

/// (Deprecated) Sparse representation of a single R1CS matrix.
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

/// (Deprecated) Sparse representation of all 3 uniform R1CS matrices.
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

/// (Optional legacy) Represents a single constraint row with possible offsets.
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
    pub fn new(num_steps: usize) -> Self {
        assert!(num_steps.is_power_of_two());
        let rows_per_step_padded = Self::num_rows_per_step().next_power_of_two();
        let total_rows = (num_steps * rows_per_step_padded).next_power_of_two();
        let vk_digest = Self::digest(num_steps);
        Self {
            num_cons_total: total_rows,
            num_steps,
            vk_digest,
        }
    }

    #[inline]
    fn num_vars() -> usize {
        JoltR1CSInputs::num_inputs()
    }

    #[inline]
    fn num_rows_per_step() -> usize {
        UNIFORM_R1CS.len()
    }

    pub fn num_vars_uniform_padded(&self) -> usize {
        Self::num_vars().next_power_of_two()
    }

    /// Number of variables across all steps padded to next power of two.
    pub fn num_vars_total(&self) -> usize {
        self.num_steps * Self::num_vars().next_power_of_two()
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
        Self::num_rows_per_step().next_power_of_two()
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
            (Self::num_rows_per_step() + 1).next_power_of_two().log_2()
        );

        let eq_rx = EqPolynomial::evals(r_constr);
        let num_vars = Self::num_vars();
        let num_vars_padded = num_vars.next_power_of_two();

        // Allocate output vector and precompute rlc powers
        let mut evals = unsafe_allocate_zero_vec(num_vars_padded);
        let r_sq = r_rlc.square();

        // Accumulate directly: for each row, add wr * (A_terms + r*B_terms + r^2*C_terms)
        for (row_idx, row_named) in UNIFORM_R1CS.iter().enumerate() {
            let row = &row_named.cons;
            let wr = eq_rx[row_idx];

            row.a.accumulate_evaluations(&mut evals, wr, num_vars);
            row.b
                .accumulate_evaluations(&mut evals, wr * r_rlc, num_vars);
            row.c
                .accumulate_evaluations(&mut evals, wr * r_sq, num_vars);
        }

        evals
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
        assert_eq!(Self::num_vars(), segment_evals.len());
        assert_eq!(r.len(), self.num_vars_uniform_padded().log_2());

        // Variables vector is [vars, ..., 1, ...] where ... denotes padding to power of 2
        let num_vars = self.num_vars_uniform_padded();
        let var_bits = num_vars.log_2();

        let eq_ry_var = EqPolynomial::evals(r);
        let eval_variables: F = (0..Self::num_vars())
            .map(|var_index| eq_ry_var[var_index] * segment_evals[var_index])
            .sum();

        // Evaluate at the constant position if it exists within the padded space
        let const_eval = if Self::num_vars() < num_vars && with_const {
            let const_position_bits = index_to_field_bitvector(Self::num_vars() as u128, var_bits);
            EqPolynomial::mle(r, &const_position_bits)
        } else {
            F::zero()
        };

        eval_variables + const_eval
    }

    /// Evaluate uniform matrix A at a specific point (rx_constr, ry_var)
    pub fn evaluate_uniform_a_at_point(&self, rx_constr: &[F], ry_var: &[F]) -> F {
        self.evaluate_uniform_matrix_at_point(|row| &row.a, rx_constr, ry_var)
    }

    /// Evaluate uniform matrix B at a specific point (rx_constr, ry_var)
    pub fn evaluate_uniform_b_at_point(&self, rx_constr: &[F], ry_var: &[F]) -> F {
        self.evaluate_uniform_matrix_at_point(|row| &row.b, rx_constr, ry_var)
    }

    /// Evaluate uniform matrix C at a specific point (rx_constr, ry_var)
    pub fn evaluate_uniform_c_at_point(&self, rx_constr: &[F], ry_var: &[F]) -> F {
        self.evaluate_uniform_matrix_at_point(|row| &row.c, rx_constr, ry_var)
    }

    /// Helper function to evaluate a uniform matrix at a specific point
    fn evaluate_uniform_matrix_at_point(
        &self,
        select: impl Fn(&Constraint) -> &LC,
        rx_constr: &[F],
        ry_var: &[F],
    ) -> F {
        // Precompute eq tables for rows and columns once
        let eq_rx = EqPolynomial::evals(rx_constr);
        let eq_ry = EqPolynomial::evals(ry_var);

        let num_vars = JoltR1CSInputs::num_inputs();

        debug_assert!(UNIFORM_R1CS.len() <= eq_rx.len());
        debug_assert!(num_vars < eq_ry.len());

        let mut acc = F::zero();
        for (row_idx, row_named) in UNIFORM_R1CS.iter().enumerate() {
            let row = &row_named.cons;
            let wr = eq_rx[row_idx];
            let lc = select(row);
            let col_contrib = lc.dot_eq_ry::<F>(&eq_ry, num_vars);
            acc += wr * col_contrib;
        }

        acc
    }

    /// Returns the digest of the R1CS "shape" derived from compile-time constants
    /// Canonical serialization of constants:
    /// - domain tag
    /// - num_steps (u64 BE)
    /// - num_vars (u32 BE)
    /// - for each row in UNIFORM_R1CS:
    ///   - tag 'A' | row.a terms (sorted by input_index asc) + const term
    ///   - tag 'B' | row.b terms (sorted by input_index asc) + const term
    ///   - tag 'C' | row.c terms (sorted by input_index asc) + const term
    fn digest(num_steps: usize) -> F {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(b"JOLT_UNIFORM_R1CS");
        bytes.extend_from_slice(&num_steps.to_be_bytes());

        let num_vars: u32 = JoltR1CSInputs::num_inputs() as u32;
        bytes.extend_from_slice(&num_vars.to_be_bytes());

        for row_named in UNIFORM_R1CS.iter() {
            let row = &row_named.cons;
            row.a.serialize_canonical(b'A', &mut bytes);
            row.b.serialize_canonical(b'B', &mut bytes);
            row.c.serialize_canonical(b'C', &mut bytes);
        }

        let mut hasher = Sha3_256::new();
        hasher.update(&bytes);

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
