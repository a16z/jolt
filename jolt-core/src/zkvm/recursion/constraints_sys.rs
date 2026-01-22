//! Batches constraints into a single indexed polynomial F(z, x) = Σ_i eq(z, i) * C_i(x)

use crate::{
    field::JoltField,
    poly::{
        commitment::{
            commitment_scheme::RecursionExt,
            dory::{
                recursion::JoltGtMulWitness, ArkDoryProof, ArkworksVerifierSetup,
                DoryCommitmentScheme,
            },
        },
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
    zkvm::recursion::{
        stage1::packed_gt_exp::PackedGtExpPublicInputs,
        witness::{GTCombineWitness, GTMulOpWitness},
    },
};
use ark_bn254::{Fq, Fr};
use ark_ff::{One, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use dory::backends::arkworks::ArkGT;
use jolt_optimizations::{fq12_to_multilinear_evals, get_g_mle};

pub mod constraint_config;
use self::constraint_config::CONFIG;

/// Convert index to binary representation as field elements (little-endian)
pub fn index_to_binary<F: JoltField>(index: usize, num_vars: usize) -> Vec<F> {
    let mut binary = Vec::with_capacity(num_vars);
    let mut idx = index;

    for _ in 0..num_vars {
        binary.push(if idx & 1 == 1 { F::one() } else { F::zero() });
        idx >>= 1;
    }

    // binary.reverse();
    binary
}

/// Builder for RecursionConstraintMetadata that encapsulates all metadata extraction
#[derive(Clone)]
pub struct RecursionMetadataBuilder {
    constraint_system: ConstraintSystem,
}

impl RecursionMetadataBuilder {
    /// Create a new builder from a constraint system
    pub fn from_constraint_system(constraint_system: ConstraintSystem) -> Self {
        Self { constraint_system }
    }

    /// Build the metadata, extracting all necessary information
    pub fn build(self) -> crate::zkvm::proof_serialization::RecursionConstraintMetadata {
        // Extract constraint types
        let constraint_types: Vec<_> = self
            .constraint_system
            .constraints
            .iter()
            .map(|c| c.constraint_type.clone())
            .collect();

        // Build dense polynomial and get bijection info
        let (dense_poly, jagged_bijection, jagged_mapping) =
            self.constraint_system.build_dense_polynomial();
        let dense_num_vars = dense_poly.get_num_vars();

        // Compute matrix rows for the verifier
        let num_polynomials = jagged_bijection.num_polynomials();
        let mut matrix_rows = Vec::with_capacity(num_polynomials);

        for poly_idx in 0..num_polynomials {
            let (constraint_idx, poly_type) = jagged_mapping.decode(poly_idx);
            let matrix_row = self
                .constraint_system
                .matrix
                .row_index(poly_type, constraint_idx);
            matrix_rows.push(matrix_row);
        }

        crate::zkvm::proof_serialization::RecursionConstraintMetadata {
            constraint_types,
            jagged_bijection,
            jagged_mapping,
            matrix_rows,
            dense_num_vars,
            packed_gt_exp_public_inputs: self.constraint_system.packed_gt_exp_public_inputs.clone(),
        }
    }
}

/// Type-safe array indexed by PolyType, eliminating the need for `as usize` casting
#[derive(Clone)]
pub struct PolyTypeArray<T>([T; PolyType::NUM_TYPES]);

impl<T: Default> Default for PolyTypeArray<T> {
    fn default() -> Self {
        Self(std::array::from_fn(|_| T::default()))
    }
}

impl<T> std::ops::Index<PolyType> for PolyTypeArray<T> {
    type Output = T;

    #[inline]
    fn index(&self, poly_type: PolyType) -> &Self::Output {
        unsafe {
            // SAFETY: PolyType values are guaranteed to be 0..NUM_TYPES
            self.0.get_unchecked(poly_type as usize)
        }
    }
}

impl<T> std::ops::IndexMut<PolyType> for PolyTypeArray<T> {
    #[inline]
    fn index_mut(&mut self, poly_type: PolyType) -> &mut Self::Output {
        unsafe {
            // SAFETY: PolyType values are guaranteed to be 0..NUM_TYPES
            self.0.get_unchecked_mut(poly_type as usize)
        }
    }
}

/// Polynomial types stored in the matrix
/// Note: Base and digit bits are NOT stored in the matrix - verifier computes them from public inputs
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum PolyType {
    // Packed GT Exponentiation polynomials (11-var each, one constraint per GT exp)
    // Note: base(x) and digit bits are public inputs, not committed polynomials
    // Note: rho_next is no longer committed - verified via shift sumcheck
    RhoPrev = 0,  // rho(s,x) - packed intermediate results (11-var)
    Quotient = 1, // quotient(s,x) - packed quotients (11-var)

    // GT Multiplication polynomials
    MulLhs = 2,
    MulRhs = 3,
    MulResult = 4,
    MulQuotient = 5,

    // G1 Scalar Multiplication polynomials
    G1ScalarMulXA = 6, // x-coord of accumulator A_i (contains A_0, A_1, ..., A_n)
    G1ScalarMulYA = 7, // y-coord of accumulator A_i (contains A_0, A_1, ..., A_n)
    G1ScalarMulXT = 8, // x-coord of doubled point T_i
    G1ScalarMulYT = 9, // y-coord of doubled point T_i
    G1ScalarMulXANext = 10, // x-coord of A_{i+1} (shifted by 1)
    G1ScalarMulYANext = 11, // y-coord of A_{i+1} (shifted by 1)
    G1ScalarMulIndicator = 12, // Indicator for T_i = O (point at infinity)
}

impl PolyType {
    pub const NUM_TYPES: usize = 13;

    /// All polynomial types as a constant array
    pub const ALL: [PolyType; 13] = [
        PolyType::RhoPrev,
        PolyType::Quotient,
        PolyType::MulLhs,
        PolyType::MulRhs,
        PolyType::MulResult,
        PolyType::MulQuotient,
        PolyType::G1ScalarMulXA,
        PolyType::G1ScalarMulYA,
        PolyType::G1ScalarMulXT,
        PolyType::G1ScalarMulYT,
        PolyType::G1ScalarMulXANext,
        PolyType::G1ScalarMulYANext,
        PolyType::G1ScalarMulIndicator,
    ];

    pub fn all() -> [PolyType; 13] {
        Self::ALL
    }

    /// Iterator over all polynomial types
    pub fn iter() -> impl Iterator<Item = PolyType> {
        Self::ALL.iter().copied()
    }

    /// Get polynomial type from row index
    pub fn from_row_index(row_idx: usize, num_constraints: usize) -> Self {
        match row_idx / num_constraints {
            0 => PolyType::RhoPrev,
            1 => PolyType::Quotient,
            2 => PolyType::MulLhs,
            3 => PolyType::MulRhs,
            4 => PolyType::MulResult,
            5 => PolyType::MulQuotient,
            6 => PolyType::G1ScalarMulXA,
            7 => PolyType::G1ScalarMulYA,
            8 => PolyType::G1ScalarMulXT,
            9 => PolyType::G1ScalarMulYT,
            10 => PolyType::G1ScalarMulXANext,
            11 => PolyType::G1ScalarMulYANext,
            12 => PolyType::G1ScalarMulIndicator,
            _ => panic!("Invalid row index"),
        }
    }
}

/// Giant multilinear matrix M(s, x) that stores all Dory polynomials in a single structure.
///
/// Layout: M(s, x) where s is the row index and x are the constraint variables
/// Physical layout: rows are organized as [all base] [all rho_prev] [all rho_curr] [all quotient]
/// Row index = poly_type * num_constraints_padded + constraint_index
#[derive(Clone)]
pub struct DoryMultilinearMatrix {
    /// Number of s variables (log2(num_rows))
    pub num_s_vars: usize,

    /// Number of constraint variables (x) - fixed at 4 for Fq12
    pub num_constraint_vars: usize,

    /// Number of constraint index variables (bits needed to index constraints)
    pub num_constraint_index_vars: usize,

    /// Number of constraints (before padding)
    pub num_constraints: usize,

    /// Number of constraints padded to power of 2
    pub num_constraints_padded: usize,

    /// Total number of rows: 4 * num_constraints_padded
    pub num_rows: usize,

    /// Total M variables: num_s_vars + num_constraint_vars
    pub num_vars: usize,

    /// Flattened storage: rows concatenated together
    /// Each row contains 2^num_constraint_vars evaluations
    /// Total size: num_rows * 2^num_constraint_vars
    pub evaluations: Vec<Fq>,
}

impl DoryMultilinearMatrix {
    /// Get row index for a given polynomial type and constraint index
    pub fn row_index(&self, poly_type: PolyType, constraint_idx: usize) -> usize {
        (poly_type as usize) * self.num_constraints_padded + constraint_idx
    }

    /// Get the storage offset for accessing a specific row's polynomial
    pub fn storage_offset(&self, row_index: usize) -> usize {
        row_index * (1 << self.num_constraint_vars)
    }

    /// Evaluate a specific row's polynomial at point x
    /// Note: constraint_vars is in little-endian (LSB first), but DensePolynomial::evaluate
    /// uses big-endian (MSB first), so we reverse the point.
    pub fn evaluate_row(&self, row: usize, constraint_vars: &[Fq]) -> Fq {
        let offset = self.storage_offset(row);
        let row_evals = &self.evaluations[offset..offset + (1 << self.num_constraint_vars)];

        let poly = DensePolynomial::new(row_evals.to_vec());
        // Reverse for big-endian convention used by DensePolynomial::evaluate
        let reversed: Vec<Fq> = constraint_vars.iter().rev().copied().collect();
        poly.evaluate(&reversed)
    }

    /// Evaluate M(s, x) where s selects the row and x is the evaluation point
    pub fn evaluate(&self, s_vars: &[Fq], constraint_vars: &[Fq]) -> Fq {
        assert_eq!(s_vars.len(), self.num_s_vars);
        assert_eq!(constraint_vars.len(), self.num_constraint_vars);

        let mut result = Fq::zero();
        for row in 0..self.num_rows {
            let row_binary = index_to_binary::<Fq>(row, self.num_s_vars);
            let eq_eval = EqPolynomial::mle(&row_binary, s_vars);

            let row_poly_eval = self.evaluate_row(row, constraint_vars);
            result += eq_eval * row_poly_eval;
        }
        result
    }
}

/// Builder for constructing the giant multilinear matrix.
///
/// Physical layout: rows are organized by polynomial type
/// Row index = poly_type * num_constraints_padded + constraint_index
pub struct DoryMatrixBuilder {
    num_constraint_vars: usize,
    /// Rows grouped by polynomial type using type-safe array
    rows_by_type: PolyTypeArray<Vec<Vec<Fq>>>,
    /// Constraint types for each constraint
    constraint_types: Vec<ConstraintType>,
}

impl DoryMatrixBuilder {
    pub fn new(num_constraint_vars: usize) -> Self {
        Self {
            num_constraint_vars,
            rows_by_type: PolyTypeArray::default(),
            constraint_types: Vec::new(),
        }
    }

    /// Get the current number of constraints added to the builder
    pub fn constraint_count(&self) -> usize {
        self.constraint_types.len()
    }

    /// Generic zero padding from one variable count to another
    /// Fills remaining positions with zeros to maintain true jaggedness
    pub fn pad_with_zeros(data: &[Fq], from_vars: usize, to_vars: usize) -> Vec<Fq> {
        assert_eq!(
            data.len(),
            1 << from_vars,
            "Input must have {} elements for {}-var MLE",
            1 << from_vars,
            from_vars
        );
        assert!(
            to_vars >= from_vars,
            "Cannot pad from {} vars to {} vars",
            from_vars,
            to_vars
        );

        let mut result = Vec::with_capacity(1 << to_vars);
        result.extend_from_slice(data);
        result.resize(1 << to_vars, Fq::zero());
        result
    }

    /// Generic replication padding from one variable count to another
    /// Replicates values across new dimensions (used for g(x) style padding)
    pub fn pad_with_replication(data: &[Fq], from_vars: usize, to_vars: usize) -> Vec<Fq> {
        assert_eq!(
            data.len(),
            1 << from_vars,
            "Input must have {} elements for {}-var MLE",
            1 << from_vars,
            from_vars
        );
        assert!(
            to_vars >= from_vars,
            "Cannot pad from {} vars to {} vars",
            from_vars,
            to_vars
        );

        let mut result = Vec::with_capacity(1 << to_vars);
        let repeat_factor = 1 << (to_vars - from_vars);

        for &val in data {
            for _ in 0..repeat_factor {
                result.push(val);
            }
        }

        assert_eq!(result.len(), 1 << to_vars);
        result
    }

    /// Special padding for packed GT exp layout
    /// Data layout: index = x * 128 + s (s in low 7 bits, x in high 4 bits)
    /// g(x) doesn't depend on s, so we replicate each g[x] across all 128 s values
    pub fn pad_packed_gt_exp(data: &[Fq], element_vars: usize, step_vars: usize) -> Vec<Fq> {
        assert_eq!(
            data.len(),
            1 << element_vars,
            "Input must have {} elements for {}-var MLE",
            1 << element_vars,
            element_vars
        );

        let total_vars = element_vars + step_vars;
        let mut result = vec![Fq::zero(); 1 << total_vars];
        let step_count = 1 << step_vars;

        // For each x value, replicate g[x] across all s values
        // index = x * step_count + s
        for x in 0..data.len() {
            let g_x = data[x];
            for s in 0..step_count {
                result[x * step_count + s] = g_x;
            }
        }

        result
    }

    /// Pad a 4-variable MLE to 8 variables using zero padding for true jaggedness.
    pub fn pad_4var_to_8var_zero_padding(mle_4var: &[Fq]) -> Vec<Fq> {
        Self::pad_with_zeros(mle_4var, 4, 8)
    }

    /// Pad a 4-variable MLE to 11 variables using zero padding for true jaggedness.
    /// For GT mul: index = s * 16 + x (x in low bits).
    pub fn pad_4var_to_11var_zero_padding(mle_4var: &[Fq]) -> Vec<Fq> {
        Self::pad_with_zeros(mle_4var, 4, 11)
    }

    /// Pad a 4-variable MLE to 11 variables for packed GT exp layout.
    /// Data layout: index = x * 128 + s (s in low 7 bits, x in high 4 bits).
    /// g(x) doesn't depend on s, so we replicate each g[x] across all 128 s values.
    pub fn pad_4var_to_11var_replicated(mle_4var: &[Fq]) -> Vec<Fq> {
        Self::pad_packed_gt_exp(mle_4var, 4, 7) // 4 element vars, 7 step vars
    }

    /// Pad an 8-variable MLE to 11 variables using zero padding.
    /// For G1 scalar mul: 8 vars → 11 vars.
    pub fn pad_8var_to_11var_zero_padding(mle_8var: &[Fq]) -> Vec<Fq> {
        Self::pad_with_zeros(mle_8var, 8, 11)
    }

    /// Add a packed GT exponentiation witness.
    /// Creates ONE constraint per GT exp with packed polynomials (all 11-var):
    /// - Base: base(x) - 4-var padded to 11-var (public input, not committed)
    /// - RhoPrev: rho(s,x) - all intermediate results packed
    /// - RhoNext: rho_next(s,x) - shifted intermediates (NOT COMMITTED - verified via shift sumcheck)
    /// - Quotient: quotient(s,x) - all quotients packed
    /// - Digit bits: digit_lo/hi(s) - scalar digits (7-var padded to 11-var, public input)
    pub fn add_packed_gt_exp_witness(
        &mut self,
        witness: &super::stage1::packed_gt_exp::PackedGtExpWitness,
    ) {
        assert_eq!(
            self.num_constraint_vars, CONFIG.packed_vars,
            "Packed GT exp requires {} constraint variables",
            CONFIG.packed_vars
        );

        let row_size = 1 << self.num_constraint_vars; // 2048

        // All packed polynomials are already 11-var (2048 elements)
        assert_eq!(witness.base_packed.len(), row_size);
        assert_eq!(witness.rho_packed.len(), row_size);
        assert_eq!(witness.rho_next_packed.len(), row_size);
        assert_eq!(witness.quotient_packed.len(), row_size);
        assert_eq!(witness.digit_lo_packed.len(), row_size);
        assert_eq!(witness.digit_hi_packed.len(), row_size);
        assert_eq!(witness.base2_packed.len(), row_size);
        assert_eq!(witness.base3_packed.len(), row_size);

        // Add only the 2 committed GT exp polynomials (base/digits/rho_next are not committed)
        self.rows_by_type[PolyType::RhoPrev].push(witness.rho_packed.clone());
        self.rows_by_type[PolyType::Quotient].push(witness.quotient_packed.clone());

        // Add empty rows for GT mul and G1 types to maintain consistent indexing
        let zero_row = vec![Fq::zero(); row_size];
        self.rows_by_type[PolyType::MulLhs].push(zero_row.clone());
        self.rows_by_type[PolyType::MulRhs].push(zero_row.clone());
        self.rows_by_type[PolyType::MulResult].push(zero_row.clone());
        self.rows_by_type[PolyType::MulQuotient].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulXA].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulYA].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulXT].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulYT].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulXANext].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulYANext].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulIndicator].push(zero_row);

        // Store ONE constraint entry for this packed GT exp
        self.constraint_types.push(ConstraintType::PackedGtExp);
    }

    /// Add constraint from a GT multiplication witness.
    /// Creates one constraint using:
    /// - lhs: the left operand a
    /// - rhs: the right operand b
    /// - result: the product c = a * b
    /// - quotient: Q such that a(x) * b(x) - c(x) = Q(x) * g(x)
    pub fn add_gt_mul_witness(&mut self, witness: &JoltGtMulWitness) {
        let lhs_mle_4var = fq12_to_multilinear_evals(&witness.lhs);
        let rhs_mle_4var = fq12_to_multilinear_evals(&witness.rhs);
        let result_mle_4var = fq12_to_multilinear_evals(&witness.result);
        let quotient_mle_4var = witness.quotient_mle.clone();

        assert_eq!(
            lhs_mle_4var.len(),
            16,
            "GT mul witness should have 4-variable MLEs"
        );
        assert_eq!(
            rhs_mle_4var.len(),
            16,
            "GT mul witness should have 4-variable MLEs"
        );
        assert_eq!(
            result_mle_4var.len(),
            16,
            "GT mul witness should have 4-variable MLEs"
        );
        assert_eq!(
            quotient_mle_4var.len(),
            16,
            "GT mul witness should have 4-variable MLEs"
        );

        // Handle padding from 4-var to target vars
        let (lhs_mle, rhs_mle, result_mle, quotient_mle) =
            if self.num_constraint_vars == CONFIG.packed_vars {
                // Pad 4-variable MLEs to 11 variables using zero padding
                (
                    Self::pad_4var_to_11var_zero_padding(&lhs_mle_4var),
                    Self::pad_4var_to_11var_zero_padding(&rhs_mle_4var),
                    Self::pad_4var_to_11var_zero_padding(&result_mle_4var),
                    Self::pad_4var_to_11var_zero_padding(&quotient_mle_4var),
                )
            } else if self.num_constraint_vars == CONFIG.g1_vars {
                // Pad 4-variable MLEs to 8 variables using zero padding
                (
                    Self::pad_4var_to_8var_zero_padding(&lhs_mle_4var),
                    Self::pad_4var_to_8var_zero_padding(&rhs_mle_4var),
                    Self::pad_4var_to_8var_zero_padding(&result_mle_4var),
                    Self::pad_4var_to_8var_zero_padding(&quotient_mle_4var),
                )
            } else if self.num_constraint_vars == CONFIG.element_vars {
                // Use MLEs as-is
                (
                    lhs_mle_4var,
                    rhs_mle_4var,
                    result_mle_4var,
                    quotient_mle_4var,
                )
            } else {
                panic!(
                    "Unsupported number of constraint variables: {}",
                    self.num_constraint_vars
                );
            };

        assert_eq!(lhs_mle.len(), 1 << self.num_constraint_vars);
        assert_eq!(rhs_mle.len(), 1 << self.num_constraint_vars);
        assert_eq!(result_mle.len(), 1 << self.num_constraint_vars);
        assert_eq!(quotient_mle.len(), 1 << self.num_constraint_vars);

        // Add rows for GT mul polynomials (keeping GT exp rows empty)
        self.rows_by_type[PolyType::MulLhs].push(lhs_mle);
        self.rows_by_type[PolyType::MulRhs].push(rhs_mle);
        self.rows_by_type[PolyType::MulResult].push(result_mle);
        self.rows_by_type[PolyType::MulQuotient].push(quotient_mle);

        // Add empty rows for GT exp and G1 types to maintain consistent indexing
        let zero_row = vec![Fq::zero(); 1 << self.num_constraint_vars];
        self.rows_by_type[PolyType::RhoPrev].push(zero_row.clone());
        self.rows_by_type[PolyType::Quotient].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulXA].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulYA].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulXT].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulYT].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulXANext].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulYANext].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulIndicator].push(zero_row);

        // Store constraint type
        self.constraint_types.push(ConstraintType::GtMul);
    }

    /// Add constraints from a G1 scalar multiplication witness.
    /// Unlike GT exp which has one MLE per step, G1 has one MLE per variable type
    /// that contains all steps.
    pub fn add_g1_scalar_mul_witness(
        &mut self,
        witness: &crate::poly::commitment::dory::recursion::JoltG1ScalarMulWitness,
    ) {
        let _n = witness.bits.len();

        // The witness MLEs contain all steps in one MLE
        assert_eq!(witness.x_a_mles.len(), 1, "Expected single MLE for x_a");
        assert_eq!(witness.y_a_mles.len(), 1, "Expected single MLE for y_a");
        assert_eq!(witness.x_t_mles.len(), 1, "Expected single MLE for x_t");
        assert_eq!(witness.y_t_mles.len(), 1, "Expected single MLE for y_t");
        assert_eq!(
            witness.x_a_next_mles.len(),
            1,
            "Expected single MLE for x_a_next"
        );
        assert_eq!(
            witness.y_a_next_mles.len(),
            1,
            "Expected single MLE for y_a_next"
        );

        // Each MLE should have 256 evaluations for 8 variables
        assert_eq!(
            witness.x_a_mles[0].len(),
            CONFIG.g1_size,
            "Expected 256 evaluations for 8 variables"
        );
        assert_eq!(witness.y_a_mles[0].len(), CONFIG.g1_size);
        assert_eq!(witness.x_t_mles[0].len(), CONFIG.g1_size);
        assert_eq!(witness.y_t_mles[0].len(), CONFIG.g1_size);
        assert_eq!(witness.x_a_next_mles[0].len(), CONFIG.g1_size);
        assert_eq!(witness.y_a_next_mles[0].len(), CONFIG.g1_size);

        // Compute infinity indicator from T coordinates
        use ark_ff::Zero;
        let t_is_infinity: Vec<Fq> = witness.x_t_mles[0]
            .iter()
            .zip(witness.y_t_mles[0].iter())
            .map(|(x_t, y_t)| {
                if x_t.is_zero() && y_t.is_zero() {
                    Fq::from(1u64)
                } else {
                    Fq::zero()
                }
            })
            .collect();
        assert_eq!(t_is_infinity.len(), CONFIG.g1_size);

        // Pad 8-var MLEs to target constraint vars if needed
        let (x_a, y_a, x_t, y_t, x_a_next, y_a_next, indicator) = if self.num_constraint_vars
            == CONFIG.packed_vars
        {
            (
                Self::pad_8var_to_11var_zero_padding(&witness.x_a_mles[0]),
                Self::pad_8var_to_11var_zero_padding(&witness.y_a_mles[0]),
                Self::pad_8var_to_11var_zero_padding(&witness.x_t_mles[0]),
                Self::pad_8var_to_11var_zero_padding(&witness.y_t_mles[0]),
                Self::pad_8var_to_11var_zero_padding(&witness.x_a_next_mles[0]),
                Self::pad_8var_to_11var_zero_padding(&witness.y_a_next_mles[0]),
                Self::pad_8var_to_11var_zero_padding(&t_is_infinity),
            )
        } else if self.num_constraint_vars == CONFIG.g1_vars {
            (
                witness.x_a_mles[0].clone(),
                witness.y_a_mles[0].clone(),
                witness.x_t_mles[0].clone(),
                witness.y_t_mles[0].clone(),
                witness.x_a_next_mles[0].clone(),
                witness.y_a_next_mles[0].clone(),
                t_is_infinity,
            )
        } else {
            panic!(
                "G1 scalar multiplication requires 8 or 11 constraint variables, but builder has {}",
                self.num_constraint_vars
            );
        };

        // Add the entire MLEs (one per variable type for this scalar multiplication)
        self.rows_by_type[PolyType::G1ScalarMulXA].push(x_a);
        self.rows_by_type[PolyType::G1ScalarMulYA].push(y_a);
        self.rows_by_type[PolyType::G1ScalarMulXT].push(x_t);
        self.rows_by_type[PolyType::G1ScalarMulYT].push(y_t);
        self.rows_by_type[PolyType::G1ScalarMulXANext].push(x_a_next);
        self.rows_by_type[PolyType::G1ScalarMulYANext].push(y_a_next);
        self.rows_by_type[PolyType::G1ScalarMulIndicator].push(indicator);

        // Add empty rows for GT types to maintain consistent indexing
        let zero_row = vec![Fq::zero(); 1 << self.num_constraint_vars];
        self.rows_by_type[PolyType::RhoPrev].push(zero_row.clone());
        self.rows_by_type[PolyType::Quotient].push(zero_row.clone());
        self.rows_by_type[PolyType::MulLhs].push(zero_row.clone());
        self.rows_by_type[PolyType::MulRhs].push(zero_row.clone());
        self.rows_by_type[PolyType::MulResult].push(zero_row.clone());
        self.rows_by_type[PolyType::MulQuotient].push(zero_row);

        // Store one constraint entry for this G1 scalar multiplication
        // The constraint evaluation will handle accessing different indices within the MLEs
        self.constraint_types.push(ConstraintType::G1ScalarMul {
            base_point: (witness.point_base.x, witness.point_base.y),
        });
    }

    /// Add constraint from a per-operation GT multiplication witness (from combine_commitments).
    /// This is the same as `add_gt_mul_witness` but accepts `GTMulOpWitness` type.
    pub fn add_gt_mul_op_witness(&mut self, witness: &GTMulOpWitness) {
        // Skip witnesses with empty quotient MLEs
        if witness.quotient_mle.is_empty() {
            tracing::debug!(
                "[Homomorphic Combine] Skipping GT mul witness with empty quotient_mle"
            );
            return;
        }

        let lhs_mle_4var = fq12_to_multilinear_evals(&witness.lhs);
        let rhs_mle_4var = fq12_to_multilinear_evals(&witness.rhs);
        let result_mle_4var = fq12_to_multilinear_evals(&witness.result);
        let quotient_mle_4var = witness.quotient_mle.clone();

        assert_eq!(
            lhs_mle_4var.len(),
            16,
            "GT mul witness should have 4-variable MLEs"
        );
        assert_eq!(
            rhs_mle_4var.len(),
            16,
            "GT mul witness should have 4-variable MLEs"
        );
        assert_eq!(
            result_mle_4var.len(),
            16,
            "GT mul witness should have 4-variable MLEs"
        );
        assert_eq!(
            quotient_mle_4var.len(),
            16,
            "GT mul witness should have 4-variable MLEs"
        );

        let (lhs_mle, rhs_mle, result_mle, quotient_mle) =
            if self.num_constraint_vars == CONFIG.packed_vars {
                (
                    Self::pad_4var_to_11var_zero_padding(&lhs_mle_4var),
                    Self::pad_4var_to_11var_zero_padding(&rhs_mle_4var),
                    Self::pad_4var_to_11var_zero_padding(&result_mle_4var),
                    Self::pad_4var_to_11var_zero_padding(&quotient_mle_4var),
                )
            } else if self.num_constraint_vars == CONFIG.g1_vars {
                (
                    Self::pad_4var_to_8var_zero_padding(&lhs_mle_4var),
                    Self::pad_4var_to_8var_zero_padding(&rhs_mle_4var),
                    Self::pad_4var_to_8var_zero_padding(&result_mle_4var),
                    Self::pad_4var_to_8var_zero_padding(&quotient_mle_4var),
                )
            } else if self.num_constraint_vars == CONFIG.element_vars {
                (
                    lhs_mle_4var,
                    rhs_mle_4var,
                    result_mle_4var,
                    quotient_mle_4var,
                )
            } else {
                panic!(
                    "Unsupported number of constraint variables: {}",
                    self.num_constraint_vars
                );
            };

        assert_eq!(lhs_mle.len(), 1 << self.num_constraint_vars);
        assert_eq!(rhs_mle.len(), 1 << self.num_constraint_vars);
        assert_eq!(result_mle.len(), 1 << self.num_constraint_vars);
        assert_eq!(quotient_mle.len(), 1 << self.num_constraint_vars);

        self.rows_by_type[PolyType::MulLhs].push(lhs_mle);
        self.rows_by_type[PolyType::MulRhs].push(rhs_mle);
        self.rows_by_type[PolyType::MulResult].push(result_mle);
        self.rows_by_type[PolyType::MulQuotient].push(quotient_mle);

        let zero_row = vec![Fq::zero(); 1 << self.num_constraint_vars];
        self.rows_by_type[PolyType::RhoPrev].push(zero_row.clone());
        self.rows_by_type[PolyType::Quotient].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulXA].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulYA].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulXT].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulYT].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulXANext].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulYANext].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulIndicator].push(zero_row);

        self.constraint_types.push(ConstraintType::GtMul);
    }

    /// Add all constraints from a GTCombineWitness (homomorphic combine offloading).
    /// Returns the packed GT exp witnesses that were created.
    pub fn add_combine_witness(
        &mut self,
        witness: &GTCombineWitness,
    ) -> Vec<super::stage1::packed_gt_exp::PackedGtExpWitness> {
        use super::stage1::packed_gt_exp::PackedGtExpWitness;
        let mut packed_witnesses = Vec::new();

        tracing::info!(
            "[add_combine_witness] Processing {} GT exp witnesses",
            witness.exp_witnesses.len()
        );

        // Convert and add GT exp witnesses in packed format
        for (idx, exp_wit) in witness.exp_witnesses.iter().enumerate() {
            // Handle edge cases where exponent is 0 or 1 (no bits)
            if exp_wit.bits.is_empty() {
                tracing::info!(
                    "[add_combine_witness] GT exp witness {} has empty bits, creating minimal witness",
                    idx
                );
                let base_mle = fq12_to_multilinear_evals(&exp_wit.base);
                let base2_mle = fq12_to_multilinear_evals(&(exp_wit.base * exp_wit.base));
                let base3_mle =
                    fq12_to_multilinear_evals(&(exp_wit.base * exp_wit.base * exp_wit.base));

                let rho_mles = if exp_wit.rho_mles.is_empty() {
                    vec![fq12_to_multilinear_evals(&exp_wit.result)]
                } else {
                    exp_wit.rho_mles.clone()
                };
                let quotient_mles = exp_wit.quotient_mles.clone();

                let packed = PackedGtExpWitness::from_steps(
                    &rho_mles,
                    &quotient_mles,
                    &exp_wit.bits,
                    &base_mle,
                    &base2_mle,
                    &base3_mle,
                );
                self.add_packed_gt_exp_witness(&packed);
                packed_witnesses.push(packed);
                continue;
            }

            // Convert base Fq12 to 4-var MLE
            let base_mle = fq12_to_multilinear_evals(&exp_wit.base);
            let base2_mle = fq12_to_multilinear_evals(&(exp_wit.base * exp_wit.base));
            let base3_mle =
                fq12_to_multilinear_evals(&(exp_wit.base * exp_wit.base * exp_wit.base));

            // Validate and fix witness data if needed
            let num_steps = (exp_wit.bits.len() + 1) / 2;
            let (rho_mles, quotient_mles) = if exp_wit.quotient_mles.len() != num_steps {
                // Fix mismatched sizes
                let mut fixed_quotients = exp_wit.quotient_mles.clone();

                // Ensure we have exactly num_steps quotient MLEs
                fixed_quotients.resize(num_steps, vec![Fq::zero(); 16]);

                // Ensure we have exactly num_steps + 1 rho MLEs
                let mut fixed_rhos = exp_wit.rho_mles.clone();
                if fixed_rhos.len() < num_steps + 1 {
                    // Pad with result MLE if needed
                    let result_mle = fq12_to_multilinear_evals(&exp_wit.result);
                    while fixed_rhos.len() < num_steps + 1 {
                        fixed_rhos.push(result_mle.clone());
                    }
                }

                (fixed_rhos, fixed_quotients)
            } else {
                (exp_wit.rho_mles.clone(), exp_wit.quotient_mles.clone())
            };

            // Create packed witness
            let packed = PackedGtExpWitness::from_steps(
                &rho_mles,
                &quotient_mles,
                &exp_wit.bits,
                &base_mle,
                &base2_mle,
                &base3_mle,
            );

            // Add to matrix
            self.add_packed_gt_exp_witness(&packed);
            packed_witnesses.push(packed);
        }

        // Add GT mul witnesses
        tracing::info!(
            "[add_combine_witness] Processing {} GT mul witnesses",
            witness.mul_witnesses.len()
        );
        for (idx, mul_wit) in witness.mul_witnesses.iter().enumerate() {
            tracing::debug!("[add_combine_witness] Adding GT mul witness {}", idx);
            self.add_gt_mul_op_witness(mul_wit);
        }

        tracing::info!(
            "[add_combine_witness] Total constraints after combine witness: {}",
            self.constraint_count()
        );

        packed_witnesses
    }

    pub fn build(self) -> (DoryMultilinearMatrix, Vec<MatrixConstraint>) {
        let num_constraints = self.rows_by_type[PolyType::RhoPrev].len();
        assert!(num_constraints > 0, "No constraints added");

        // Debug: print constraint counts for each type
        for poly_type in PolyType::all() {
            let count = self.rows_by_type[poly_type].len();
            if count != num_constraints {
                eprintln!(
                    "Row type {:?} has {} constraints, expected {}",
                    poly_type, count, num_constraints
                );
            }
        }

        for poly_type in PolyType::all() {
            assert_eq!(
                self.rows_by_type[poly_type].len(),
                num_constraints,
                "Row type {:?} has wrong number of constraints",
                poly_type
            );
        }
        assert_eq!(
            self.constraint_types.len(),
            num_constraints,
            "Number of constraint types must match number of constraints"
        );

        // Pad num_constraints to next power of 2
        let num_constraints_bits = (num_constraints as f64).log2().ceil() as usize;
        let num_constraints_padded = 1 << num_constraints_bits;

        // Total rows = NUM_TYPES × num_constraints_padded
        let num_rows_unpadded = PolyType::NUM_TYPES * num_constraints_padded;

        // Pad num_rows to next power of 2 for the matrix
        let num_s_vars = (num_rows_unpadded as f64).log2().ceil() as usize;
        let num_s_vars = num_s_vars.max(11);
        let num_rows = 1 << num_s_vars;

        // Sanity checks
        assert!(
            num_constraints_padded >= num_constraints,
            "Padded constraints must be at least as large as actual constraints"
        );
        assert_eq!(num_rows, 1 << num_s_vars, "num_rows must be a power of 2");
        assert!(
            num_rows >= PolyType::NUM_TYPES * num_constraints_padded,
            "num_rows must be at least NUM_TYPES * num_constraints_padded"
        );
        assert_eq!(
            1 << num_constraints_bits,
            num_constraints_padded,
            "Constraints padding must be power of 2"
        );

        let row_size = 1 << self.num_constraint_vars;
        let capacity = num_rows * row_size;

        // Pre-allocate the exact size and initialize to zero
        let mut evaluations = vec![Fq::zero(); capacity];

        // Use unsafe for faster copying without bounds checks
        unsafe {
            let eval_ptr = evaluations.as_mut_ptr();
            let mut offset = 0;

            for poly_type in PolyType::all() {
                let rows = &self.rows_by_type[poly_type];

                // Copy actual rows
                for row in rows {
                    std::ptr::copy_nonoverlapping(row.as_ptr(), eval_ptr.add(offset), row_size);
                    offset += row_size;
                }

                // Skip zero padding (already initialized to zero)
                offset += (num_constraints_padded - rows.len()) * row_size;
            }
        }

        let matrix = DoryMultilinearMatrix {
            num_s_vars,
            num_constraint_vars: self.num_constraint_vars,
            num_constraint_index_vars: num_constraints_bits,
            num_constraints,
            num_constraints_padded,
            num_rows,
            num_vars: num_s_vars + self.num_constraint_vars,
            evaluations,
        };

        let constraints: Vec<MatrixConstraint> = self
            .constraint_types
            .into_iter()
            .enumerate()
            .map(|(idx, constraint_type)| MatrixConstraint {
                constraint_index: idx,
                constraint_type,
            })
            .collect();

        (matrix, constraints)
    }
}

/// Type of constraint
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConstraintType {
    /// Packed GT exponentiation constraint (one per GT exp, covers all 254 steps)
    PackedGtExp,
    /// GT multiplication constraint
    GtMul,
    /// G1 scalar multiplication constraint with base point
    G1ScalarMul {
        base_point: (Fq, Fq), // (x_p, y_p)
    },
}

/// Constraint metadata for matrix-based evaluation.
/// Row indices are computed from constraint_index using the matrix layout.
#[derive(Clone, Debug)]
pub struct MatrixConstraint {
    /// Index of this constraint (0 to num_constraints-1)
    pub constraint_index: usize,
    /// Type of constraint (GT exp or GT mul)
    pub constraint_type: ConstraintType,
}

/// Trait for any witness that can be added to the constraint system
/// Constraint system using a giant multilinear matrix for all witness polynomials
#[derive(Clone)]
pub struct ConstraintSystem {
    /// The giant matrix M(s, x)
    pub matrix: DoryMultilinearMatrix,

    /// g(x) polynomial - precomputed as DensePolynomial
    pub g_poly: DensePolynomial<Fq>,

    /// Constraint metadata: maps constraint index to matrix rows it references
    pub constraints: Vec<MatrixConstraint>,

    /// Packed GT exp witnesses for Stage 1 prover (base-4 steps packed into 11-var MLEs)
    pub packed_gt_exp_witnesses: Vec<super::stage1::packed_gt_exp::PackedGtExpWitness>,

    /// Public inputs for packed GT exp (base Fq12 and scalar bits) - used by verifier
    /// and Stage 2 to compute digit/base evaluations directly
    pub packed_gt_exp_public_inputs: Vec<PackedGtExpPublicInputs>,
}

impl ConstraintSystem {
    /// Create constraint system from witness data
    pub fn from_witness(
        constraint_types: Vec<ConstraintType>,
        g_poly: DensePolynomial<Fq>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // For now, create a simple matrix with padding
        let _num_constraints = constraint_types.len();
        let num_constraint_vars = CONFIG.packed_vars; // Using packed vars for packed GT exp base-4

        let mut builder = DoryMatrixBuilder::new(num_constraint_vars);

        // Add dummy data for each constraint type
        // This is a placeholder - should be replaced with actual witness data
        let zero_row = vec![Fq::zero(); 1 << num_constraint_vars];

        for constraint_type in &constraint_types {
            match constraint_type {
                ConstraintType::PackedGtExp => {
                    // Add packed GT exp rows (2 committed polynomials - base/digits/rho_next are not committed)
                    builder.rows_by_type[PolyType::RhoPrev].push(zero_row.clone());
                    builder.rows_by_type[PolyType::Quotient].push(zero_row.clone());

                    // Add empty rows for other types (GT mul + G1)
                    for poly_type in PolyType::iter().skip(2) {
                        builder.rows_by_type[poly_type].push(zero_row.clone());
                    }

                    builder.constraint_types.push(constraint_type.clone());
                }
                ConstraintType::GtMul => {
                    // Add empty rows for packed GT exp types
                    builder.rows_by_type[PolyType::RhoPrev].push(zero_row.clone());
                    builder.rows_by_type[PolyType::Quotient].push(zero_row.clone());

                    // Add GT mul rows
                    builder.rows_by_type[PolyType::MulLhs].push(zero_row.clone());
                    builder.rows_by_type[PolyType::MulRhs].push(zero_row.clone());
                    builder.rows_by_type[PolyType::MulResult].push(zero_row.clone());
                    builder.rows_by_type[PolyType::MulQuotient].push(zero_row.clone());

                    // Add empty rows for G1 types
                    for poly_type in PolyType::iter().skip(PolyType::G1ScalarMulXA as usize) {
                        builder.rows_by_type[poly_type].push(zero_row.clone());
                    }

                    builder.constraint_types.push(constraint_type.clone());
                }
                ConstraintType::G1ScalarMul { .. } => {
                    // Add empty rows for GT types (packed GT exp + GT mul)
                    for poly_type in PolyType::iter().take(PolyType::G1ScalarMulXA as usize) {
                        builder.rows_by_type[poly_type].push(zero_row.clone());
                    }

                    // Add G1 scalar mul rows
                    for poly_type in PolyType::iter().skip(PolyType::G1ScalarMulXA as usize) {
                        builder.rows_by_type[poly_type].push(zero_row.clone());
                    }

                    builder.constraint_types.push(constraint_type.clone());
                }
            }
        }

        let (matrix, constraints) = builder.build();

        Ok(Self {
            matrix,
            g_poly,
            constraints,
            packed_gt_exp_witnesses: Vec::new(), // No actual witnesses in from_witness (test helper)
            packed_gt_exp_public_inputs: Vec::new(), // No actual public inputs in from_witness (test helper)
        })
    }

    /// Get the number of variables in the constraint system
    pub fn num_vars(&self) -> usize {
        self.matrix.num_vars
    }

    /// Get the number of constraints
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Get the number of s variables (for virtualization)
    pub fn num_s_vars(&self) -> usize {
        self.matrix.num_s_vars
    }

    pub fn new<T>(
        proof: &ArkDoryProof,
        setup: &ArkworksVerifierSetup,
        transcript: &mut T,
        point: &[<Fr as JoltField>::Challenge],
        evaluation: &Fr,
        commitment: &ArkGT,
    ) -> Result<(Self, <DoryCommitmentScheme as RecursionExt<Fr>>::Hint), ProofVerifyError>
    where
        T: Transcript,
    {
        let (witnesses, hints) = <DoryCommitmentScheme as RecursionExt<Fr>>::witness_gen(
            proof, setup, transcript, point, evaluation, commitment,
        )?;

        // Always use 11 variables for uniform matrix structure
        // GT operations (4-var) and G1 operations (8-var) will be padded to 11 vars
        let mut builder = DoryMatrixBuilder::new(11);

        // Create packed GT exp witnesses and public inputs, add to matrix
        let mut packed_gt_exp_witnesses = Vec::with_capacity(witnesses.gt_exp.len());
        let mut packed_gt_exp_public_inputs = Vec::with_capacity(witnesses.gt_exp.len());
        for (_op_id, witness) in witnesses.gt_exp.iter() {
            let base_mle = fq12_to_multilinear_evals(&witness.base);
            let base2_mle = fq12_to_multilinear_evals(&(witness.base * witness.base));
            let base3_mle =
                fq12_to_multilinear_evals(&(witness.base * witness.base * witness.base));
            let packed = super::stage1::packed_gt_exp::PackedGtExpWitness::from_steps(
                &witness.rho_mles,
                &witness.quotient_mles,
                &witness.bits,
                &base_mle,
                &base2_mle,
                &base3_mle,
            );
            builder.add_packed_gt_exp_witness(&packed);
            packed_gt_exp_witnesses.push(packed);
            packed_gt_exp_public_inputs.push(PackedGtExpPublicInputs::new(
                witness.base,
                witness.bits.clone(),
            ));
        }

        for (_op_id, witness) in witnesses.gt_mul.iter() {
            builder.add_gt_mul_witness(witness);
        }

        // Add G1 scalar multiplication witnesses
        for (_op_id, witness) in witnesses.g1_scalar_mul.iter() {
            builder.add_g1_scalar_mul_witness(witness);
        }

        let (matrix, constraints) = builder.build();

        // Get the 4-variable g(x) polynomial
        let g_mle_4var = get_g_mle();

        // Pad g(x) to match constraint vars
        let g_poly = if matrix.num_constraint_vars == CONFIG.packed_vars {
            let padded_g = DoryMatrixBuilder::pad_4var_to_11var_zero_padding(&g_mle_4var);
            DensePolynomial::new(padded_g)
        } else if matrix.num_constraint_vars == CONFIG.g1_vars {
            let padded_g = DoryMatrixBuilder::pad_4var_to_8var_zero_padding(&g_mle_4var);
            DensePolynomial::new(padded_g)
        } else {
            DensePolynomial::new(g_mle_4var)
        };

        Ok((
            Self {
                matrix,
                g_poly,
                constraints,
                packed_gt_exp_witnesses,
                packed_gt_exp_public_inputs,
            },
            hints,
        ))
    }

    /// Extract GT mul constraint data for gt_mul sumcheck
    pub fn extract_gt_mul_constraints(&self) -> Vec<(usize, Vec<Fq>, Vec<Fq>, Vec<Fq>, Vec<Fq>)> {
        let num_constraint_vars = self.matrix.num_constraint_vars;
        let row_size = 1 << num_constraint_vars;

        // Pre-allocate with exact capacity
        let gt_mul_count = self
            .constraints
            .iter()
            .filter(|c| matches!(c.constraint_type, ConstraintType::GtMul))
            .count();
        let mut constraints = Vec::with_capacity(gt_mul_count);

        for (idx, constraint) in self.constraints.iter().enumerate() {
            if let ConstraintType::GtMul = constraint.constraint_type {
                let lhs = self.extract_row_poly(PolyType::MulLhs, idx, row_size);
                let rhs = self.extract_row_poly(PolyType::MulRhs, idx, row_size);
                let result = self.extract_row_poly(PolyType::MulResult, idx, row_size);
                let quotient = self.extract_row_poly(PolyType::MulQuotient, idx, row_size);

                constraints.push((constraint.constraint_index, lhs, rhs, result, quotient));
            }
        }

        constraints
    }

    /// Extract G1 scalar mul constraint data for g1_scalar_mul sumcheck
    pub fn extract_g1_scalar_mul_constraints(
        &self,
    ) -> Vec<(
        usize,
        (Fq, Fq),
        Vec<Fq>,
        Vec<Fq>,
        Vec<Fq>,
        Vec<Fq>,
        Vec<Fq>,
        Vec<Fq>,
        Vec<Fq>,
    )> {
        let num_constraint_vars = self.matrix.num_constraint_vars;
        let row_size = 1 << num_constraint_vars;

        // Pre-allocate with exact capacity
        let g1_scalar_mul_count = self
            .constraints
            .iter()
            .filter(|c| matches!(c.constraint_type, ConstraintType::G1ScalarMul { .. }))
            .count();
        let mut constraints = Vec::with_capacity(g1_scalar_mul_count);

        for (idx, constraint) in self.constraints.iter().enumerate() {
            if let ConstraintType::G1ScalarMul { base_point } = constraint.constraint_type {
                // Extract the MLEs for this G1 scalar multiplication
                let x_a = self.extract_row_poly(PolyType::G1ScalarMulXA, idx, row_size);
                let y_a = self.extract_row_poly(PolyType::G1ScalarMulYA, idx, row_size);
                let x_t = self.extract_row_poly(PolyType::G1ScalarMulXT, idx, row_size);
                let y_t = self.extract_row_poly(PolyType::G1ScalarMulYT, idx, row_size);
                let x_a_next = self.extract_row_poly(PolyType::G1ScalarMulXANext, idx, row_size);
                let y_a_next = self.extract_row_poly(PolyType::G1ScalarMulYANext, idx, row_size);
                let t_is_infinity =
                    self.extract_row_poly(PolyType::G1ScalarMulIndicator, idx, row_size);

                constraints.push((
                    constraint.constraint_index,
                    base_point,
                    x_a,
                    y_a,
                    x_t,
                    y_t,
                    x_a_next,
                    y_a_next,
                    t_is_infinity,
                ));
            }
        }

        constraints
    }

    /// Extract packed GT exp constraint data for packed_gt_exp sumcheck
    /// Returns: (constraint_index, rho, quotient) for each PackedGtExp constraint
    /// Note: digit bits, base, and rho_next are public inputs/virtual claims
    pub fn extract_packed_gt_exp_constraints(&self) -> Vec<(usize, Vec<Fq>, Vec<Fq>)> {
        let num_constraint_vars = self.matrix.num_constraint_vars;
        let row_size = 1 << num_constraint_vars;

        // Pre-allocate with exact capacity
        let packed_gt_exp_count = self
            .constraints
            .iter()
            .filter(|c| matches!(c.constraint_type, ConstraintType::PackedGtExp))
            .count();
        let mut constraints = Vec::with_capacity(packed_gt_exp_count);

        for (idx, constraint) in self.constraints.iter().enumerate() {
            if let ConstraintType::PackedGtExp = constraint.constraint_type {
                // Extract the 2 committed MLEs for this packed GT exp
                // Note: RhoPrev = rho in the packed convention
                // Base, digit bits, and rho_next are not committed
                let rho = self.extract_row_poly(PolyType::RhoPrev, idx, row_size);
                let quotient = self.extract_row_poly(PolyType::Quotient, idx, row_size);

                constraints.push((constraint.constraint_index, rho, quotient));
            }
        }

        constraints
    }

    /// Helper to extract a row polynomial from the matrix
    fn extract_row_poly(
        &self,
        poly_type: PolyType,
        constraint_idx: usize,
        row_size: usize,
    ) -> Vec<Fq> {
        let type_start = (poly_type as usize) * self.matrix.num_constraints_padded * row_size;
        let row_start = type_start + constraint_idx * row_size;
        let row_end = row_start + row_size;
        self.matrix.evaluations[row_start..row_end].to_vec()
    }

    /// Evaluate the full constraint system at a point (for testing)
    ///
    /// Point structure: [x_vars, s_vars]
    /// Returns F(x, s) where s selects which constraint to evaluate
    pub fn evaluate(&self, point: &[Fq]) -> Fq {
        let num_x_vars = self.matrix.num_constraint_vars;
        let num_s_vars = self.matrix.num_s_vars;

        assert_eq!(point.len(), num_x_vars + num_s_vars);

        // Split point: [x_vars, s_vars]
        let (x_vars, s_vars) = point.split_at(num_x_vars);

        // We only evaluate constraints here, not the matrix rows
        // So we need to map s to constraint index
        let num_constraint_bits = (self.num_constraints() as f64).log2().ceil() as usize;
        let num_constraints_padded = 1 << num_constraint_bits;

        let mut result = Fq::zero();

        // For each constraint, check if s selects it
        for constraint in self.constraints.iter() {
            // s encodes both poly type and constraint index
            // For constraint evaluation, we treat all poly types of a constraint the same
            let constraint_padded_idx = constraint.constraint_index;

            // Check all 4 row indices that correspond to this constraint
            for poly_type in PolyType::all() {
                let row_idx = (poly_type as usize) * num_constraints_padded + constraint_padded_idx;
                let row_binary = index_to_binary::<Fq>(row_idx, num_s_vars);
                let eq_eval = EqPolynomial::mle(&row_binary, s_vars);

                let constraint_eval = self.evaluate_constraint(constraint, x_vars);
                result += eq_eval * constraint_eval;
            }
        }

        result
    }

    /// Helper to evaluate g(x) based on constraint type and variable count
    fn evaluate_g_poly(&self, x: &[Fq], constraint_type: &ConstraintType) -> Fq {
        match constraint_type {
            ConstraintType::PackedGtExp if x.len() == CONFIG.packed_vars => {
                // For packed GT exp, g(x) only depends on element vars (high 4 bits)
                let x_elem_reversed: Vec<Fq> = x[CONFIG.step_vars..CONFIG.packed_vars]
                    .iter()
                    .rev()
                    .copied()
                    .collect();
                let g_4var = get_g_mle();
                DensePolynomial::new(g_4var).evaluate(&x_elem_reversed)
            }
            ConstraintType::GtMul if x.len() == CONFIG.packed_vars => {
                // For GT mul with zero padding, element vars are in low 4 bits
                let x_elem_reversed: Vec<Fq> =
                    x[0..CONFIG.element_vars].iter().rev().copied().collect();
                let g_4var = get_g_mle();
                DensePolynomial::new(g_4var).evaluate(&x_elem_reversed)
            }
            _ => {
                let x_reversed: Vec<Fq> = x.iter().rev().copied().collect();
                self.g_poly.evaluate(&x_reversed)
            }
        }
    }

    /// Helper to get polynomial evaluations for a constraint
    fn get_poly_evaluations(
        &self,
        constraint_idx: usize,
        x: &[Fq],
        poly_types: &[PolyType],
    ) -> Vec<Fq> {
        poly_types
            .iter()
            .map(|&pt| {
                let row = self.matrix.row_index(pt, constraint_idx);
                self.matrix.evaluate_row(row, x)
            })
            .collect()
    }

    /// Evaluate a single constraint C_i(x) using the matrix layout.
    fn evaluate_constraint(&self, constraint: &MatrixConstraint, x: &[Fq]) -> Fq {
        let idx = constraint.constraint_index;

        match constraint.constraint_type {
            ConstraintType::PackedGtExp => {
                // Evaluate g(x) using helper
                let g_eval = self.evaluate_g_poly(x, &constraint.constraint_type);

                // Get polynomial evaluations
                let evals =
                    self.get_poly_evaluations(idx, x, &[PolyType::RhoPrev, PolyType::Quotient]);
                let rho_prev = evals[0];
                let quotient = evals[1];

                // Compute rho_next on the fly from packed witness
                // During constraint evaluation, we use the precomputed rho_next_packed
                // Need to find GT exp witness index (count PackedGtExp constraints before this one)
                let gt_exp_idx = self
                    .constraints
                    .iter()
                    .take(idx)
                    .filter(|c| matches!(c.constraint_type, ConstraintType::PackedGtExp))
                    .count();
                let packed = &self.packed_gt_exp_witnesses[gt_exp_idx];
                // Reverse for big-endian convention used by DensePolynomial::evaluate
                let x_reversed: Vec<Fq> = x.iter().rev().copied().collect();
                let rho_curr =
                    DensePolynomial::new(packed.rho_next_packed.clone()).evaluate(&x_reversed);
                let base_eval =
                    DensePolynomial::new(packed.base_packed.clone()).evaluate(&x_reversed);
                let base2_eval =
                    DensePolynomial::new(packed.base2_packed.clone()).evaluate(&x_reversed);
                let base3_eval =
                    DensePolynomial::new(packed.base3_packed.clone()).evaluate(&x_reversed);
                let digit_lo_eval =
                    DensePolynomial::new(packed.digit_lo_packed.clone()).evaluate(&x_reversed);
                let digit_hi_eval =
                    DensePolynomial::new(packed.digit_hi_packed.clone()).evaluate(&x_reversed);

                let u = digit_lo_eval;
                let v = digit_hi_eval;
                let w0 = (Fq::one() - u) * (Fq::one() - v);
                let w1 = u * (Fq::one() - v);
                let w2 = (Fq::one() - u) * v;
                let w3 = u * v;
                let base_power = w0 + w1 * base_eval + w2 * base2_eval + w3 * base3_eval;
                let rho2 = rho_prev * rho_prev;
                let rho4 = rho2 * rho2;

                rho_curr - rho4 * base_power - quotient * g_eval
            }
            ConstraintType::GtMul => {
                let g_eval = self.evaluate_g_poly(x, &constraint.constraint_type);

                // Get all GT mul polynomial evaluations at once
                let evals = self.get_poly_evaluations(
                    idx,
                    x,
                    &[
                        PolyType::MulLhs,
                        PolyType::MulRhs,
                        PolyType::MulResult,
                        PolyType::MulQuotient,
                    ],
                );

                // GT mul constraint: lhs * rhs - result - quotient * g
                evals[0] * evals[1] - evals[2] - evals[3] * g_eval
            }
            ConstraintType::G1ScalarMul { base_point } => {
                // Keep existing G1 logic mostly unchanged but use helper for poly evaluations
                let poly_types = [
                    PolyType::G1ScalarMulXA,
                    PolyType::G1ScalarMulYA,
                    PolyType::G1ScalarMulXT,
                    PolyType::G1ScalarMulYT,
                    PolyType::G1ScalarMulXANext,
                    PolyType::G1ScalarMulYANext,
                ];
                let evals = self.get_poly_evaluations(idx, x, &poly_types);

                let (x_a, y_a, x_t, y_t, x_a_next, y_a_next) =
                    (evals[0], evals[1], evals[2], evals[3], evals[4], evals[5]);

                // Extract base point coordinates
                let (x_p, y_p) = base_point;

                // C1: Doubling x-coordinate constraint
                // 4y_A^2(x_T + 2x_A) - 9x_A^4 = 0
                let c1 = {
                    let four = Fq::from(4u64);
                    let two = Fq::from(2u64);
                    let nine = Fq::from(9u64);

                    let y_a_sq = y_a * y_a;
                    let x_a_sq = x_a * x_a;
                    let x_a_fourth = x_a_sq * x_a_sq;

                    four * y_a_sq * (x_t + two * x_a) - nine * x_a_fourth
                };

                // C2: Doubling y-coordinate constraint
                // 3x_A^2(x_T - x_A) + 2y_A(y_T + y_A) = 0
                let c2 = {
                    let three = Fq::from(3u64);
                    let two = Fq::from(2u64);

                    let x_a_sq = x_a * x_a;
                    three * x_a_sq * (x_t - x_a) + two * y_a * (y_t + y_a)
                };

                // C3: Conditional addition x-coordinate constraint
                // Reformulated without explicit bit values:
                // C3 = (x_A' - x_T) * [(x_A' + x_T + x_P)(x_P - x_T)^2 - (y_P - y_T)^2]
                let c3 = {
                    let x_diff = x_p - x_t;
                    let y_diff = y_p - y_t;
                    let x_a_diff = x_a_next - x_t;

                    x_a_diff * ((x_a_next + x_t + x_p) * x_diff * x_diff - y_diff * y_diff)
                };

                // C4: Conditional addition y-coordinate constraint
                // Reformulated without explicit bit values:
                // C4 = (y_A' - y_T) * [x_T(y_P + y_A') - x_P(y_T + y_A') + x_A'(y_T - y_P)]
                let c4 = {
                    let y_a_diff = y_a_next - y_t;

                    y_a_diff
                        * (x_t * (y_p + y_a_next) - x_p * (y_t + y_a_next) + x_a_next * (y_t - y_p))
                };

                // Return sum of all constraints (should be 0 when valid)
                c1 + c2 + c3 + c4
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::{
        commitment::{
            commitment_scheme::CommitmentScheme,
            dory::{DoryCommitmentScheme, DoryGlobals},
        },
        dense_mlpoly::DensePolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    };
    use ark_bn254::Fr;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_dory_witness_constraint_evaluation() {
        use ark_ff::UniformRand;
        use rand::thread_rng;

        DoryGlobals::reset();
        DoryGlobals::initialize(1 << 2, 1 << 2);
        let num_vars = 4;
        let mut rng = thread_rng();

        let prover_setup = DoryCommitmentScheme::setup_prover(num_vars);
        let verifier_setup = DoryCommitmentScheme::setup_verifier(&prover_setup);

        let coefficients: Vec<Fr> = (0..(1 << num_vars)).map(|_| Fr::rand(&mut rng)).collect();
        let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coefficients));
        let (commitment, hint) = DoryCommitmentScheme::commit(&poly, &prover_setup);

        let point: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
            .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
            .collect();

        let mut prover_transcript = crate::transcripts::Blake2bTranscript::new(b"test");
        let proof = DoryCommitmentScheme::prove(
            &prover_setup,
            &poly,
            &point,
            Some(hint),
            &mut prover_transcript,
        );

        let evaluation = PolynomialEvaluation::evaluate(&poly, &point);
        let mut extract_transcript = crate::transcripts::Blake2bTranscript::new(b"test");

        let _start = std::time::Instant::now();

        let (system, hints) = ConstraintSystem::new(
            &proof,
            &verifier_setup,
            &mut extract_transcript,
            &point,
            &evaluation,
            &commitment,
        )
        .expect("System creation should succeed");

        let _elapsed = _start.elapsed();
        // Count constraints by type
        let mut gt_exp_count = 0;
        let mut gt_mul_count = 0;
        let mut g1_scalar_mul_count = 0;

        for constraint in &system.constraints {
            match &constraint.constraint_type {
                ConstraintType::PackedGtExp => gt_exp_count += 1,
                ConstraintType::GtMul { .. } => gt_mul_count += 1,
                ConstraintType::G1ScalarMul { .. } => g1_scalar_mul_count += 1,
            }
        }

        let _ = (gt_exp_count, gt_mul_count, g1_scalar_mul_count);
        // Instead of evaluating the full system, just evaluate constraints at random points
        use rand::{rngs::StdRng, Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(42);
        let num_x_vars = system.matrix.num_constraint_vars;

        // For each constraint, test it at random x-variable points
        for (idx, constraint) in system.constraints.iter().enumerate() {
            // Test this constraint at 5 random points
            for _trial in 0..5 {
                let mut x_point = Vec::with_capacity(num_x_vars);
                for _ in 0..num_x_vars {
                    x_point.push(if rng.gen_bool(0.5) {
                        Fq::one()
                    } else {
                        Fq::zero()
                    });
                }

                let eval = system.evaluate_constraint(constraint, &x_point);
                assert_eq!(
                    eval,
                    Fq::zero(),
                    "Constraint {} ({:?}) should evaluate to 0 at boolean points",
                    idx,
                    constraint.constraint_type
                );
            }
        }
        let mut verify_transcript = crate::transcripts::Blake2bTranscript::new(b"test");
        DoryCommitmentScheme::verify_with_hint(
            &proof,
            &verifier_setup,
            &mut verify_transcript,
            &point,
            &evaluation,
            &commitment,
            &hints,
        )
        .expect("Verification with hint should succeed");

        let mut verify_transcript_no_hint = crate::transcripts::Blake2bTranscript::new(b"test");
        DoryCommitmentScheme::verify(
            &proof,
            &verifier_setup,
            &mut verify_transcript_no_hint,
            &point,
            &evaluation,
            &commitment,
        )
        .expect("Verification without hint should also succeed");
    }
}

// Manual serialization implementations for enums

use ark_serialize::{Compress, SerializationError, Valid};

impl CanonicalSerialize for PolyType {
    fn serialize_with_mode<W: ark_serialize::Write>(
        &self,
        writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        (*self as u8).serialize_with_mode(writer, _compress)
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        1 // u8 size
    }
}

impl CanonicalDeserialize for PolyType {
    fn deserialize_with_mode<R: ark_serialize::Read>(
        reader: R,
        _compress: Compress,
        _validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        let val = u8::deserialize_with_mode(reader, _compress, _validate)?;
        match val {
            0 => Ok(PolyType::RhoPrev),
            1 => Ok(PolyType::Quotient),
            2 => Ok(PolyType::MulLhs),
            3 => Ok(PolyType::MulRhs),
            4 => Ok(PolyType::MulResult),
            5 => Ok(PolyType::MulQuotient),
            6 => Ok(PolyType::G1ScalarMulXA),
            7 => Ok(PolyType::G1ScalarMulYA),
            8 => Ok(PolyType::G1ScalarMulXT),
            9 => Ok(PolyType::G1ScalarMulYT),
            10 => Ok(PolyType::G1ScalarMulXANext),
            11 => Ok(PolyType::G1ScalarMulYANext),
            12 => Ok(PolyType::G1ScalarMulIndicator),
            _ => Err(SerializationError::InvalidData),
        }
    }
}

impl Valid for PolyType {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalSerialize for ConstraintType {
    fn serialize_with_mode<W: ark_serialize::Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            ConstraintType::PackedGtExp => {
                0u8.serialize_with_mode(&mut writer, compress)?;
            }
            ConstraintType::GtMul => {
                1u8.serialize_with_mode(&mut writer, compress)?;
            }
            ConstraintType::G1ScalarMul { base_point } => {
                2u8.serialize_with_mode(&mut writer, compress)?;
                base_point.0.serialize_with_mode(&mut writer, compress)?;
                base_point.1.serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match self {
            ConstraintType::PackedGtExp => 1,
            ConstraintType::GtMul => 1,
            ConstraintType::G1ScalarMul { base_point } => {
                1 + base_point.0.serialized_size(compress) + base_point.1.serialized_size(compress)
            }
        }
    }
}

impl CanonicalDeserialize for ConstraintType {
    fn deserialize_with_mode<R: ark_serialize::Read>(
        mut reader: R,
        compress: Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        let variant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        match variant {
            0 => Ok(ConstraintType::PackedGtExp),
            1 => Ok(ConstraintType::GtMul),
            2 => {
                let x = Fq::deserialize_with_mode(&mut reader, compress, validate)?;
                let y = Fq::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(ConstraintType::G1ScalarMul { base_point: (x, y) })
            }
            _ => Err(SerializationError::InvalidData),
        }
    }
}

impl Valid for ConstraintType {
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            ConstraintType::G1ScalarMul { base_point } => {
                base_point.0.check()?;
                base_point.1.check()?;
            }
            _ => {}
        }
        Ok(())
    }
}
