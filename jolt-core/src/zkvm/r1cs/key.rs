use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use sha3::Sha3_256;

use crate::{
    field::JoltField,
    poly::{eq_poly::EqPolynomial, lagrange_poly::LagrangePolynomial},
    utils::{index_to_field_bitvector, thread::unsafe_allocate_zero_vec},
};

use sha3::Digest;

use super::constraints::{
    Constraint, LC, UNIFORM_R1CS, UNIFORM_R1CS_FIRST_GROUP, UNIFORM_R1CS_SECOND_GROUP,
    UNIVARIATE_SKIP_DOMAIN_SIZE,
};
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

    /// Number of cycle variables, e.g. number of bits needed to represent all cycles in the trace
    pub fn num_cycle_vars(&self) -> usize {
        self.num_steps.next_power_of_two().log_2()
    }

    /// Number of bits needed for all rows.
    /// With univariate skip, this is the number of cycle variables plus two (one for univariate skip of degree ~13-15, and one for the streaming round)
    pub fn num_rows_bits(&self) -> usize {
        self.num_cycle_vars() + 2
    }

    /// Evaluate the RLC of A_small, B_small, C_small matrices at (r_constr, y_var)
    /// This function only handles uniform constraints, ignoring cross-step constraints
    /// Returns evaluations for each y_var
    #[tracing::instrument(skip_all, name = "UniformSpartanKey::evaluate_small_matrix_rlc")]
    pub fn evaluate_small_matrix_rlc(
        &self,
        r_constr: &[F::Challenge],
        r_rlc: F::Challenge,
    ) -> Vec<F> {
        // With univariate skip, `r_constr` consists of two challenges in the canonical order:
        // - r_constr[0] = r_stream: selector for the second (odd) group in the row split
        // - r_constr[1] = r0:       challenge for the univariate-skip first-round (Lagrange basis)
        assert_eq!(r_constr.len(), 2);

        let r_stream = r_constr[0];
        let lag_evals = LagrangePolynomial::<F>::evals::<F::Challenge, UNIVARIATE_SKIP_DOMAIN_SIZE>(
            &r_constr[1],
        );
        let w_group0 = F::one() - r_stream; // weight for first group
        let w_group1 = r_stream; // weight for second group

        let num_vars = Self::num_vars();
        let num_vars_padded = num_vars.next_power_of_two();

        // Allocate output vector and precompute rlc powers
        let mut evals = unsafe_allocate_zero_vec(num_vars_padded);

        // Accumulate using explicit FIRST and SECOND groups
        // First group weighted by (1 - r_stream)
        for (i, row_named) in UNIFORM_R1CS_FIRST_GROUP.iter().enumerate() {
            let row = &row_named.cons;
            let wr = w_group0 * lag_evals[i];
            row.a.accumulate_evaluations(&mut evals, wr, num_vars);
            row.b
                .accumulate_evaluations(&mut evals, wr * r_rlc, num_vars);
        }
        // Second group weighted by r_stream
        for (i, row_named) in UNIFORM_R1CS_SECOND_GROUP.iter().enumerate() {
            let row = &row_named.cons;
            let wr = w_group1 * lag_evals[i];
            row.a.accumulate_evaluations(&mut evals, wr, num_vars);
            row.b
                .accumulate_evaluations(&mut evals, wr * r_rlc, num_vars);
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
        r: &[F::Challenge],
        with_const: bool,
    ) -> F {
        assert_eq!(Self::num_vars(), segment_evals.len());
        assert_eq!(r.len(), self.num_vars_uniform_padded().log_2());

        // Variables vector is [vars, ..., 1, ...] where ... denotes padding to power of 2
        let num_vars = self.num_vars_uniform_padded();
        let var_bits = num_vars.log_2();

        let eq_ry_var = EqPolynomial::<F>::evals(r);
        let eval_variables: F = (0..Self::num_vars())
            .map(|var_index| eq_ry_var[var_index] * segment_evals[var_index])
            .sum();

        // Evaluate at the constant position if it exists within the padded space
        let const_eval = if Self::num_vars() < num_vars && with_const {
            let const_position_bits: Vec<F> =
                index_to_field_bitvector(Self::num_vars() as u128, var_bits);
            EqPolynomial::mle(r, &const_position_bits)
        } else {
            F::zero()
        };

        eval_variables + const_eval
    }

    /// Evaluate uniform matrix A at a specific point (rx_constr, ry_var)
    pub fn evaluate_uniform_a_at_point(
        &self,
        rx_constr: &[F::Challenge],
        ry_var: &[F::Challenge],
    ) -> F {
        self.evaluate_uniform_matrix_at_point(|row| &row.a, rx_constr, ry_var)
    }

    /// Evaluate uniform matrix B at a specific point (rx_constr, ry_var)
    pub fn evaluate_uniform_b_at_point(
        &self,
        rx_constr: &[F::Challenge],
        ry_var: &[F::Challenge],
    ) -> F {
        self.evaluate_uniform_matrix_at_point(|row| &row.b, rx_constr, ry_var)
    }

    /// Helper function to evaluate a uniform matrix at a specific point
    /// Uses univariate-skip semantics on the row axis: split rows into two groups,
    /// weight them by (1 - r_stream) and r_stream respectively, and use Lagrange basis
    /// for the first-round (size-UNIVARIATE_SKIP_DOMAIN_SIZE) row domain.
    fn evaluate_uniform_matrix_at_point(
        &self,
        select: impl Fn(&Constraint) -> &LC,
        rx_constr: &[F::Challenge],
        ry_var: &[F::Challenge],
    ) -> F {
        // Row axis: r_constr = [r_stream, r0]; use Lagrange basis for first-round
        // (half the number of R1CS constraints)
        // and linear blend for the two groups using r_stream
        debug_assert!(rx_constr.len() >= 2);
        let r_stream = rx_constr[0];
        let r0 = rx_constr[1];

        // Lagrange basis over symmetric domain for first-round rows
        let lag_basis =
            LagrangePolynomial::<F>::evals::<F::Challenge, UNIVARIATE_SKIP_DOMAIN_SIZE>(&r0);

        // Column axis: standard eq basis over variables
        let eq_ry = EqPolynomial::<F>::evals(ry_var);

        let num_vars = JoltR1CSInputs::num_inputs();
        debug_assert!(num_vars < eq_ry.len());

        let mut acc_first_group = F::zero();
        // First group: 14 rows evaluated with Lagrange basis in group order
        for (i, row_named) in UNIFORM_R1CS_FIRST_GROUP.iter().enumerate() {
            let row = &row_named.cons;
            let lc = select(row);
            let col_contrib = lc.dot_eq_ry::<F>(&eq_ry, num_vars);
            acc_first_group += lag_basis[i] * col_contrib;
        }

        let mut acc_second_group = F::zero();
        // Second group: remaining 13 rows, uniformly weighted by r_stream in group order
        for (i, row_named) in UNIFORM_R1CS_SECOND_GROUP.iter().enumerate() {
            let row = &row_named.cons;
            let lc = select(row);
            let col_contrib = lc.dot_eq_ry::<F>(&eq_ry, num_vars);
            acc_second_group += lag_basis[i] * col_contrib;
        }

        acc_first_group + r_stream * (acc_second_group - acc_first_group)
    }

    /// Returns the digest of the R1CS "shape" derived from compile-time constants
    /// Canonical serialization of constants:
    /// - domain tag
    /// - num_steps (u64 BE)
    /// - num_vars (u32 BE)
    /// - for each row in UNIFORM_R1CS:
    ///   - tag 'A' | row.a terms (sorted by input_index asc) + const term
    ///   - tag 'B' | row.b terms (sorted by input_index asc) + const term
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
