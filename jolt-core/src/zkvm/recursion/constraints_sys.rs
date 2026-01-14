//! Batches constraints into a single indexed polynomial F(z, x) = Σ_i eq(z, i) * C_i(x)

use crate::{
    field::JoltField,
    poly::{
        commitment::{
            commitment_scheme::RecursionExt,
            dory::{
                recursion::JoltGtMulWitness,
                ArkDoryProof, ArkworksVerifierSetup, DoryCommitmentScheme,
            },
        },
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
    zkvm::recursion::witness::{GTCombineWitness, GTMulOpWitness},
};
use ark_bn254::{Fq, Fr};
use ark_ff::{One, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use dory::backends::arkworks::ArkGT;
use jolt_optimizations::{fq12_to_multilinear_evals, get_g_mle};

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
        }
    }
}

/// Compute constraint formula: ρ_curr - ρ_prev² × base^{b} - quotient × g
pub fn compute_constraint_formula(
    rho_curr: Fq,
    rho_prev: Fq,
    base: Fq,
    quotient: Fq,
    g_val: Fq,
    bit: bool,
) -> Fq {
    let base_power = if bit { base } else { Fq::one() };
    rho_curr - rho_prev.square() * base_power - quotient * g_val
}

/// Polynomial types stored in the matrix
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum PolyType {
    // Packed GT Exponentiation polynomials (12-var each, one constraint per GT exp)
    Base = 0,     // base(x) - 4-var padded to 12-var
    RhoPrev = 1,  // rho(s,x) - packed intermediate results (12-var)
    RhoCurr = 2,  // rho_next(s,x) - shifted intermediates (12-var)
    Quotient = 3, // quotient(s,x) - packed quotients (12-var)
    Bit = 4,      // bit(s) - scalar bits (8-var padded to 12-var)

    // GT Multiplication polynomials
    MulLhs = 5,
    MulRhs = 6,
    MulResult = 7,
    MulQuotient = 8,

    // G1 Scalar Multiplication polynomials
    G1ScalarMulXA = 9,  // x-coord of accumulator A_i (contains A_0, A_1, ..., A_n)
    G1ScalarMulYA = 10, // y-coord of accumulator A_i (contains A_0, A_1, ..., A_n)
    G1ScalarMulXT = 11, // x-coord of doubled point T_i
    G1ScalarMulYT = 12, // y-coord of doubled point T_i
    G1ScalarMulXANext = 13, // x-coord of A_{i+1} (shifted by 1)
    G1ScalarMulYANext = 14, // y-coord of A_{i+1} (shifted by 1)
    G1ScalarMulIndicator = 15, // Indicator for T_i = O (point at infinity)
}

impl PolyType {
    pub const NUM_TYPES: usize = 16;

    pub fn all() -> [PolyType; 16] {
        [
            PolyType::Base,
            PolyType::RhoPrev,
            PolyType::RhoCurr,
            PolyType::Quotient,
            PolyType::Bit,
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
        ]
    }

    /// Get polynomial type from row index
    pub fn from_row_index(row_idx: usize, num_constraints: usize) -> Self {
        match row_idx / num_constraints {
            0 => PolyType::Base,
            1 => PolyType::RhoPrev,
            2 => PolyType::RhoCurr,
            3 => PolyType::Quotient,
            4 => PolyType::Bit,
            5 => PolyType::MulLhs,
            6 => PolyType::MulRhs,
            7 => PolyType::MulResult,
            8 => PolyType::MulQuotient,
            9 => PolyType::G1ScalarMulXA,
            10 => PolyType::G1ScalarMulYA,
            11 => PolyType::G1ScalarMulXT,
            12 => PolyType::G1ScalarMulYT,
            13 => PolyType::G1ScalarMulXANext,
            14 => PolyType::G1ScalarMulYANext,
            15 => PolyType::G1ScalarMulIndicator,
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
    /// Rows grouped by polynomial type (16 types total)
    rows_by_type: [Vec<Vec<Fq>>; 16],
    /// Constraint types for each constraint
    constraint_types: Vec<ConstraintType>,
}

impl DoryMatrixBuilder {
    pub fn new(num_constraint_vars: usize) -> Self {
        Self {
            num_constraint_vars,
            rows_by_type: [
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ],
            constraint_types: Vec::new(),
        }
    }

    /// Get the current number of constraints added to the builder
    pub fn constraint_count(&self) -> usize {
        self.constraint_types.len()
    }

    /// Pad a 4-variable MLE to 8 variables by repeating the values.
    /// For a 4-var MLE f(x0,x1,x2,x3), the 8-var version is f(x0,x1,x2,x3,0,0,0,0).
    /// This means we repeat each 4-var evaluation 2^4 = 16 times.
    pub fn pad_4var_to_8var(mle_4var: &[Fq]) -> Vec<Fq> {
        assert_eq!(mle_4var.len(), 16, "Input must be a 4-variable MLE");
        let mut mle_8var = Vec::with_capacity(256);

        // For each 4-var evaluation, repeat it 16 times
        // This corresponds to all possible values of the last 4 variables
        for &val in mle_4var {
            for _ in 0..16 {
                mle_8var.push(val);
            }
        }

        assert_eq!(mle_8var.len(), 256);
        mle_8var
    }

    /// Pad a 4-variable MLE to 8 variables using zero padding for true jaggedness.
    pub fn pad_4var_to_8var_zero_padding(mle_4var: &[Fq]) -> Vec<Fq> {
        assert_eq!(mle_4var.len(), 16, "Input must be a 4-variable MLE");
        let mut mle_8var = Vec::with_capacity(256);

        // Copy original 16 values at the beginning
        mle_8var.extend_from_slice(mle_4var);

        // Pad with zeros for the remaining positions
        // This creates true jaggedness - non-zero values only at the start
        mle_8var.resize(256, Fq::zero());

        assert_eq!(mle_8var.len(), 256);
        mle_8var
    }

    /// Pad a 4-variable MLE to 12 variables using zero padding for true jaggedness.
    /// For GT mul: index = s * 16 + x (x in low bits).
    pub fn pad_4var_to_12var_zero_padding(mle_4var: &[Fq]) -> Vec<Fq> {
        assert_eq!(mle_4var.len(), 16, "Input must be a 4-variable MLE");
        let mut mle_12var = Vec::with_capacity(4096);

        // Copy original 16 values at the beginning
        mle_12var.extend_from_slice(mle_4var);

        // Pad with zeros for the remaining positions
        mle_12var.resize(4096, Fq::zero());

        assert_eq!(mle_12var.len(), 4096);
        mle_12var
    }

    /// Pad a 4-variable MLE to 12 variables for packed GT exp layout.
    /// Data layout: index = x * 256 + s (s in low 8 bits, x in high 4 bits).
    /// g(x) doesn't depend on s, so we replicate each g[x] across all 256 s values.
    pub fn pad_4var_to_12var_replicated(mle_4var: &[Fq]) -> Vec<Fq> {
        assert_eq!(mle_4var.len(), 16, "Input must be a 4-variable MLE");
        let mut mle_12var = vec![Fq::zero(); 4096];

        // For each x value (0-15), replicate g[x] across all s values (0-255)
        // index = x * 256 + s
        for x in 0..16 {
            let g_x = mle_4var[x];
            for s in 0..256 {
                mle_12var[x * 256 + s] = g_x;
            }
        }

        mle_12var
    }

    /// Pad an 8-variable MLE to 12 variables using zero padding.
    /// For G1 scalar mul: 8 vars → 12 vars.
    pub fn pad_8var_to_12var_zero_padding(mle_8var: &[Fq]) -> Vec<Fq> {
        assert_eq!(mle_8var.len(), 256, "Input must be an 8-variable MLE");
        let mut mle_12var = Vec::with_capacity(4096);

        // Copy original 256 values at the beginning
        mle_12var.extend_from_slice(mle_8var);

        // Pad with zeros for the remaining positions
        mle_12var.resize(4096, Fq::zero());

        assert_eq!(mle_12var.len(), 4096);
        mle_12var
    }

    /// Add a packed GT exponentiation witness.
    /// Creates ONE constraint per GT exp with 5 packed polynomials (all 12-var):
    /// - Base: base(x) - 4-var padded to 12-var
    /// - RhoPrev: rho(s,x) - all intermediate results packed
    /// - RhoCurr: rho_next(s,x) - shifted intermediates
    /// - Quotient: quotient(s,x) - all quotients packed
    /// - Bit: bit(s) - scalar bits (8-var padded to 12-var)
    pub fn add_packed_gt_exp_witness(
        &mut self,
        witness: &super::stage1::packed_gt_exp::PackedGtExpWitness,
    ) {
        assert_eq!(
            self.num_constraint_vars, 12,
            "Packed GT exp requires 12 constraint variables"
        );

        let row_size = 1 << self.num_constraint_vars; // 4096

        // All packed polynomials are already 12-var (4096 elements)
        assert_eq!(witness.base_packed.len(), row_size);
        assert_eq!(witness.rho_packed.len(), row_size);
        assert_eq!(witness.rho_next_packed.len(), row_size);
        assert_eq!(witness.quotient_packed.len(), row_size);
        assert_eq!(witness.bit_packed.len(), row_size);

        // Add the 5 GT exp polynomials
        self.rows_by_type[PolyType::Base as usize].push(witness.base_packed.clone());
        self.rows_by_type[PolyType::RhoPrev as usize].push(witness.rho_packed.clone());
        self.rows_by_type[PolyType::RhoCurr as usize].push(witness.rho_next_packed.clone());
        self.rows_by_type[PolyType::Quotient as usize].push(witness.quotient_packed.clone());
        self.rows_by_type[PolyType::Bit as usize].push(witness.bit_packed.clone());

        // Add empty rows for GT mul and G1 types to maintain consistent indexing
        let zero_row = vec![Fq::zero(); row_size];
        self.rows_by_type[PolyType::MulLhs as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::MulRhs as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::MulResult as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::MulQuotient as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulXA as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulYA as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulXT as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulYT as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulXANext as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulYANext as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulIndicator as usize].push(zero_row);

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
        let (lhs_mle, rhs_mle, result_mle, quotient_mle) = if self.num_constraint_vars == 12 {
            // Pad 4-variable MLEs to 12 variables using zero padding
            (
                Self::pad_4var_to_12var_zero_padding(&lhs_mle_4var),
                Self::pad_4var_to_12var_zero_padding(&rhs_mle_4var),
                Self::pad_4var_to_12var_zero_padding(&result_mle_4var),
                Self::pad_4var_to_12var_zero_padding(&quotient_mle_4var),
            )
        } else if self.num_constraint_vars == 8 {
            // Pad 4-variable MLEs to 8 variables using zero padding
            (
                Self::pad_4var_to_8var_zero_padding(&lhs_mle_4var),
                Self::pad_4var_to_8var_zero_padding(&rhs_mle_4var),
                Self::pad_4var_to_8var_zero_padding(&result_mle_4var),
                Self::pad_4var_to_8var_zero_padding(&quotient_mle_4var),
            )
        } else if self.num_constraint_vars == 4 {
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
        self.rows_by_type[PolyType::MulLhs as usize].push(lhs_mle);
        self.rows_by_type[PolyType::MulRhs as usize].push(rhs_mle);
        self.rows_by_type[PolyType::MulResult as usize].push(result_mle);
        self.rows_by_type[PolyType::MulQuotient as usize].push(quotient_mle);

        // Add empty rows for GT exp and G1 types to maintain consistent indexing
        let zero_row = vec![Fq::zero(); 1 << self.num_constraint_vars];
        self.rows_by_type[PolyType::Base as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::RhoPrev as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::RhoCurr as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::Quotient as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::Bit as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulXA as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulYA as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulXT as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulYT as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulXANext as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulYANext as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulIndicator as usize].push(zero_row);

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
            1 << 8,
            "Expected 256 evaluations for 8 variables"
        );
        assert_eq!(witness.y_a_mles[0].len(), 1 << 8);
        assert_eq!(witness.x_t_mles[0].len(), 1 << 8);
        assert_eq!(witness.y_t_mles[0].len(), 1 << 8);
        assert_eq!(witness.x_a_next_mles[0].len(), 1 << 8);
        assert_eq!(witness.y_a_next_mles[0].len(), 1 << 8);

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
        assert_eq!(t_is_infinity.len(), 1 << 8);

        // Pad 8-var MLEs to target constraint vars if needed
        let (x_a, y_a, x_t, y_t, x_a_next, y_a_next, indicator) = if self.num_constraint_vars == 12
        {
            (
                Self::pad_8var_to_12var_zero_padding(&witness.x_a_mles[0]),
                Self::pad_8var_to_12var_zero_padding(&witness.y_a_mles[0]),
                Self::pad_8var_to_12var_zero_padding(&witness.x_t_mles[0]),
                Self::pad_8var_to_12var_zero_padding(&witness.y_t_mles[0]),
                Self::pad_8var_to_12var_zero_padding(&witness.x_a_next_mles[0]),
                Self::pad_8var_to_12var_zero_padding(&witness.y_a_next_mles[0]),
                Self::pad_8var_to_12var_zero_padding(&t_is_infinity),
            )
        } else if self.num_constraint_vars == 8 {
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
                "G1 scalar multiplication requires 8 or 12 constraint variables, but builder has {}",
                self.num_constraint_vars
            );
        };

        // Add the entire MLEs (one per variable type for this scalar multiplication)
        self.rows_by_type[PolyType::G1ScalarMulXA as usize].push(x_a);
        self.rows_by_type[PolyType::G1ScalarMulYA as usize].push(y_a);
        self.rows_by_type[PolyType::G1ScalarMulXT as usize].push(x_t);
        self.rows_by_type[PolyType::G1ScalarMulYT as usize].push(y_t);
        self.rows_by_type[PolyType::G1ScalarMulXANext as usize].push(x_a_next);
        self.rows_by_type[PolyType::G1ScalarMulYANext as usize].push(y_a_next);
        self.rows_by_type[PolyType::G1ScalarMulIndicator as usize].push(indicator);

        // Add empty rows for GT types to maintain consistent indexing
        let zero_row = vec![Fq::zero(); 1 << self.num_constraint_vars];
        self.rows_by_type[PolyType::Base as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::RhoPrev as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::RhoCurr as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::Quotient as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::Bit as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::MulLhs as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::MulRhs as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::MulResult as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::MulQuotient as usize].push(zero_row);

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

        assert_eq!(lhs_mle_4var.len(), 16, "GT mul witness should have 4-variable MLEs");
        assert_eq!(rhs_mle_4var.len(), 16, "GT mul witness should have 4-variable MLEs");
        assert_eq!(result_mle_4var.len(), 16, "GT mul witness should have 4-variable MLEs");
        assert_eq!(quotient_mle_4var.len(), 16, "GT mul witness should have 4-variable MLEs");

        let (lhs_mle, rhs_mle, result_mle, quotient_mle) = if self.num_constraint_vars == 12 {
            (
                Self::pad_4var_to_12var_zero_padding(&lhs_mle_4var),
                Self::pad_4var_to_12var_zero_padding(&rhs_mle_4var),
                Self::pad_4var_to_12var_zero_padding(&result_mle_4var),
                Self::pad_4var_to_12var_zero_padding(&quotient_mle_4var),
            )
        } else if self.num_constraint_vars == 8 {
            (
                Self::pad_4var_to_8var_zero_padding(&lhs_mle_4var),
                Self::pad_4var_to_8var_zero_padding(&rhs_mle_4var),
                Self::pad_4var_to_8var_zero_padding(&result_mle_4var),
                Self::pad_4var_to_8var_zero_padding(&quotient_mle_4var),
            )
        } else if self.num_constraint_vars == 4 {
            (lhs_mle_4var, rhs_mle_4var, result_mle_4var, quotient_mle_4var)
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

        self.rows_by_type[PolyType::MulLhs as usize].push(lhs_mle);
        self.rows_by_type[PolyType::MulRhs as usize].push(rhs_mle);
        self.rows_by_type[PolyType::MulResult as usize].push(result_mle);
        self.rows_by_type[PolyType::MulQuotient as usize].push(quotient_mle);

        let zero_row = vec![Fq::zero(); 1 << self.num_constraint_vars];
        self.rows_by_type[PolyType::Base as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::RhoPrev as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::RhoCurr as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::Quotient as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::Bit as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulXA as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulYA as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulXT as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulYT as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulXANext as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulYANext as usize].push(zero_row.clone());
        self.rows_by_type[PolyType::G1ScalarMulIndicator as usize].push(zero_row);

        self.constraint_types.push(ConstraintType::GtMul);
    }

    /// Add all constraints from a GTCombineWitness (homomorphic combine offloading).
    /// Returns the packed GT exp witnesses that were created.
    pub fn add_combine_witness(&mut self, witness: &GTCombineWitness) -> Vec<super::stage1::packed_gt_exp::PackedGtExpWitness> {
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
                // For trivial exponents (0 or 1), create a minimal packed witness
                // with one step to maintain consistency
                let base_mle = fq12_to_multilinear_evals(&exp_wit.base);

                // Create minimal witness data
                let bits = vec![false]; // One dummy bit
                let rho_mles = if exp_wit.rho_mles.is_empty() {
                    // Convert Fq12 to MLE for base and result
                    let result_mle = fq12_to_multilinear_evals(&exp_wit.result);
                    vec![base_mle.clone(), result_mle]
                } else {
                    exp_wit.rho_mles.clone()
                };
                let quotient_mles = if exp_wit.quotient_mles.is_empty() {
                    vec![vec![Fq::zero(); 16]] // One zero quotient
                } else {
                    exp_wit.quotient_mles.clone()
                };

                let packed = PackedGtExpWitness::from_steps(
                    &rho_mles,
                    &quotient_mles,
                    &bits,
                    &base_mle,
                );
                self.add_packed_gt_exp_witness(&packed);
                packed_witnesses.push(packed);
                continue;
            }

            // Convert base Fq12 to 4-var MLE
            let base_mle = fq12_to_multilinear_evals(&exp_wit.base);

            // Validate and fix witness data if needed
            let (rho_mles, quotient_mles) = if exp_wit.quotient_mles.len() != exp_wit.bits.len() {
                // Fix mismatched sizes
                let num_steps = exp_wit.bits.len();
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
        let num_constraints = self.rows_by_type[0].len();
        assert!(num_constraints > 0, "No constraints added");

        // Debug: print constraint counts for each type
        for poly_type in PolyType::all() {
            let count = self.rows_by_type[poly_type as usize].len();
            if count != num_constraints {
                eprintln!(
                    "Row type {:?} has {} constraints, expected {}",
                    poly_type, count, num_constraints
                );
            }
        }

        for poly_type in PolyType::all() {
            assert_eq!(
                self.rows_by_type[poly_type as usize].len(),
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
                let rows = &self.rows_by_type[poly_type as usize];

                // Copy actual rows
                for row in rows {
                    std::ptr::copy_nonoverlapping(
                        row.as_ptr(),
                        eval_ptr.add(offset),
                        row_size
                    );
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

/// Constraint system using a giant multilinear matrix for all witness polynomials
#[derive(Clone)]
pub struct ConstraintSystem {
    /// The giant matrix M(s, x)
    pub matrix: DoryMultilinearMatrix,

    /// g(x) polynomial - precomputed as DensePolynomial
    pub g_poly: DensePolynomial<Fq>,

    /// Constraint metadata: maps constraint index to matrix rows it references
    pub constraints: Vec<MatrixConstraint>,

    /// Packed GT exp witnesses for Stage 1 prover (all 254 steps packed into 12-var MLEs)
    pub packed_gt_exp_witnesses: Vec<super::stage1::packed_gt_exp::PackedGtExpWitness>,
}

impl ConstraintSystem {
    /// Create constraint system from witness data
    pub fn from_witness(
        constraint_types: Vec<ConstraintType>,
        g_poly: DensePolynomial<Fq>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // For now, create a simple matrix with padding
        let _num_constraints = constraint_types.len();
        let num_constraint_vars = 12; // Using 12 for packed GT exp compatibility

        let mut builder = DoryMatrixBuilder::new(num_constraint_vars);

        // Add dummy data for each constraint type
        // This is a placeholder - should be replaced with actual witness data
        let zero_row = vec![Fq::zero(); 1 << num_constraint_vars];

        for constraint_type in &constraint_types {
            match constraint_type {
                ConstraintType::PackedGtExp => {
                    // Add packed GT exp rows (5 polynomials)
                    builder.rows_by_type[PolyType::Base as usize].push(zero_row.clone());
                    builder.rows_by_type[PolyType::RhoPrev as usize].push(zero_row.clone());
                    builder.rows_by_type[PolyType::RhoCurr as usize].push(zero_row.clone());
                    builder.rows_by_type[PolyType::Quotient as usize].push(zero_row.clone());
                    builder.rows_by_type[PolyType::Bit as usize].push(zero_row.clone());

                    // Add empty rows for other types (GT mul + G1)
                    for poly_type in 5..16 {
                        builder.rows_by_type[poly_type].push(zero_row.clone());
                    }

                    builder.constraint_types.push(constraint_type.clone());
                }
                ConstraintType::GtMul => {
                    // Add empty rows for packed GT exp types
                    for poly_type in 0..5 {
                        builder.rows_by_type[poly_type].push(zero_row.clone());
                    }

                    // Add GT mul rows
                    builder.rows_by_type[PolyType::MulLhs as usize].push(zero_row.clone());
                    builder.rows_by_type[PolyType::MulRhs as usize].push(zero_row.clone());
                    builder.rows_by_type[PolyType::MulResult as usize].push(zero_row.clone());
                    builder.rows_by_type[PolyType::MulQuotient as usize].push(zero_row.clone());

                    // Add empty rows for G1 types
                    for poly_type in 9..16 {
                        builder.rows_by_type[poly_type].push(zero_row.clone());
                    }

                    builder.constraint_types.push(constraint_type.clone());
                }
                ConstraintType::G1ScalarMul { .. } => {
                    // Add empty rows for GT types (packed GT exp + GT mul)
                    for poly_type in 0..9 {
                        builder.rows_by_type[poly_type].push(zero_row.clone());
                    }

                    // Add G1 scalar mul rows
                    for poly_type in 9..16 {
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

        // Always use 12 variables for uniform matrix structure
        // GT operations (4-var) and G1 operations (8-var) will be padded to 12 vars
        let mut builder = DoryMatrixBuilder::new(12);

        // Create packed GT exp witnesses and add to matrix
        let mut packed_gt_exp_witnesses = Vec::with_capacity(witnesses.gt_exp.len());
        for (_op_id, witness) in witnesses.gt_exp.iter() {
            let base_mle = fq12_to_multilinear_evals(&witness.base);
            let packed = super::stage1::packed_gt_exp::PackedGtExpWitness::from_steps(
                &witness.rho_mles,
                &witness.quotient_mles,
                &witness.bits,
                &base_mle,
            );
            builder.add_packed_gt_exp_witness(&packed);
            packed_gt_exp_witnesses.push(packed);
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
        let g_poly = if matrix.num_constraint_vars == 12 {
            let padded_g = DoryMatrixBuilder::pad_4var_to_12var_zero_padding(&g_mle_4var);
            DensePolynomial::new(padded_g)
        } else if matrix.num_constraint_vars == 8 {
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
            },
            hints,
        ))
    }

    /// Extract constraint polynomials for square-and-multiply sumcheck (Stage 1)
    /// DEPRECATED: This is for the old per-step GT exp approach.
    #[deprecated(note = "Use extract_packed_gt_exp_constraints for packed GT exp approach")]
    pub fn extract_constraint_polynomials(
        &self,
    ) -> Vec<crate::zkvm::recursion::stage1::square_and_multiply::ConstraintPolynomials<Fq>> {
        // Packed GT exp doesn't use per-step constraints
        // This function is kept for backwards compatibility but returns empty
        Vec::new()
    }

    /// Extract GT mul constraint data for gt_mul sumcheck
    pub fn extract_gt_mul_constraints(&self) -> Vec<(usize, Vec<Fq>, Vec<Fq>, Vec<Fq>, Vec<Fq>)> {
        let num_constraint_vars = self.matrix.num_constraint_vars;
        let row_size = 1 << num_constraint_vars;

        // Pre-allocate with exact capacity
        let gt_mul_count = self.constraints.iter()
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
        let g1_scalar_mul_count = self.constraints.iter()
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
    /// Returns: (constraint_index, rho, rho_next, quotient, bit, base) for each PackedGtExp constraint
    pub fn extract_packed_gt_exp_constraints(
        &self,
    ) -> Vec<(usize, Vec<Fq>, Vec<Fq>, Vec<Fq>, Vec<Fq>, Vec<Fq>)> {
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
                // Extract the 5 MLEs for this packed GT exp
                // Note: RhoPrev = rho, RhoCurr = rho_next in the packed convention
                let rho = self.extract_row_poly(PolyType::RhoPrev, idx, row_size);
                let rho_next = self.extract_row_poly(PolyType::RhoCurr, idx, row_size);
                let quotient = self.extract_row_poly(PolyType::Quotient, idx, row_size);
                let bit = self.extract_row_poly(PolyType::Bit, idx, row_size);
                let base = self.extract_row_poly(PolyType::Base, idx, row_size);

                constraints.push((
                    constraint.constraint_index,
                    rho,
                    rho_next,
                    quotient,
                    bit,
                    base,
                ));
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

    /// Debug helper to print constraint evaluation components
    #[allow(dead_code)]
    pub fn debug_constraint_eval(&self, constraint: &MatrixConstraint, x: &[Fq]) {
        let idx = constraint.constraint_index;
        match constraint.constraint_type {
            ConstraintType::PackedGtExp => {
                let g_eval = if x.len() == 12 {
                    let x_elem_reversed: Vec<Fq> = x[8..12].iter().rev().copied().collect();
                    let g_4var = get_g_mle();
                    DensePolynomial::new(g_4var).evaluate(&x_elem_reversed)
                } else {
                    let x_reversed: Vec<Fq> = x.iter().rev().copied().collect();
                    self.g_poly.evaluate(&x_reversed)
                };

                let base_row = self.matrix.row_index(PolyType::Base, idx);
                let rho_prev_row = self.matrix.row_index(PolyType::RhoPrev, idx);
                let rho_curr_row = self.matrix.row_index(PolyType::RhoCurr, idx);
                let quotient_row = self.matrix.row_index(PolyType::Quotient, idx);
                let bit_row = self.matrix.row_index(PolyType::Bit, idx);

                let base_eval = self.matrix.evaluate_row(base_row, x);
                let rho_prev = self.matrix.evaluate_row(rho_prev_row, x);
                let rho_curr = self.matrix.evaluate_row(rho_curr_row, x);
                let quotient = self.matrix.evaluate_row(quotient_row, x);
                let bit_eval = self.matrix.evaluate_row(bit_row, x);

                let base_power = Fq::one() + (base_eval - Fq::one()) * bit_eval;
                let constraint_eval =
                    rho_curr - rho_prev.square() * base_power - quotient * g_eval;

                // Convert point to index
                let mut index = 0usize;
                for (i, &b) in x.iter().enumerate() {
                    if b == Fq::one() {
                        index |= 1 << i;
                    }
                }
                let s_index = index & 0xFF; // low 8 bits
                let x_index = (index >> 8) & 0xF; // high 4 bits

                println!("  s_index={}, x_index={}", s_index, x_index);
                println!("  base_eval = {:?}", base_eval);
                println!("  rho_prev = {:?}", rho_prev);
                println!("  rho_curr = {:?}", rho_curr);
                println!("  quotient = {:?}", quotient);
                println!("  bit_eval = {:?}", bit_eval);
                println!("  g_eval = {:?}", g_eval);
                println!("  base_power = {:?}", base_power);
                println!("  constraint = {:?}", constraint_eval);

                // Also check raw data at the specific index
                let packed = &self.packed_gt_exp_witnesses[idx];
                println!("  --- Raw packed data at index {} ---", index);
                println!("  rho_packed[{}] = {:?}", index, packed.rho_packed.get(index));
                println!(
                    "  rho_next_packed[{}] = {:?}",
                    index,
                    packed.rho_next_packed.get(index)
                );
                println!(
                    "  quotient_packed[{}] = {:?}",
                    index,
                    packed.quotient_packed.get(index)
                );
                println!("  bit_packed[{}] = {:?}", index, packed.bit_packed.get(index));
                println!(
                    "  base_packed[{}] = {:?}",
                    index,
                    packed.base_packed.get(index)
                );

                // Check if matrix data matches packed data
                println!("  --- Matrix data check ---");
                let storage_offset = self.matrix.storage_offset(base_row);
                println!(
                    "  base_row={}, storage_offset={}",
                    base_row, storage_offset
                );
                println!(
                    "  matrix.evaluations[{}] = {:?}",
                    storage_offset + index,
                    self.matrix.evaluations.get(storage_offset + index)
                );
                println!(
                    "  matrix row_evals[{}] = {:?}",
                    index,
                    self.matrix.evaluations.get(storage_offset..storage_offset + 4096)
                        .and_then(|s| s.get(index))
                );
            }
            _ => {
                println!("  Non-PackedGtExp constraint, skipping debug");
            }
        }
    }

    /// Evaluate μ at a given point s
    /// This is used for testing/debugging - the actual evaluation should use the multilinear polynomial
    pub fn evaluate_mu_at_binary_point(
        base_claim: Fq,
        rho_prev_claim: Fq,
        rho_curr_claim: Fq,
        quotient_claim: Fq,
        s_binary: &[Fq],
        num_constraints_padded: usize,
    ) -> Fq {
        // Convert binary point to index (for testing only)
        let mut s_index = 0usize;
        for (i, &bit) in s_binary.iter().enumerate() {
            if bit == Fq::one() {
                s_index |= 1 << i;
            }
        }

        // Determine which polynomial type this row index corresponds to
        let poly_type = PolyType::from_row_index(s_index, num_constraints_padded);

        match poly_type {
            PolyType::Base => base_claim,
            PolyType::RhoPrev => rho_prev_claim,
            PolyType::RhoCurr => rho_curr_claim,
            PolyType::Quotient => quotient_claim,
            PolyType::Bit => Fq::zero(), // Bit handled by packed prover
            PolyType::MulLhs => Fq::zero(),
            PolyType::MulRhs => Fq::zero(),
            PolyType::MulResult => Fq::zero(),
            PolyType::MulQuotient => Fq::zero(),
            PolyType::G1ScalarMulXA => Fq::zero(),
            PolyType::G1ScalarMulYA => Fq::zero(),
            PolyType::G1ScalarMulXT => Fq::zero(),
            PolyType::G1ScalarMulYT => Fq::zero(),
            PolyType::G1ScalarMulXANext => Fq::zero(),
            PolyType::G1ScalarMulYANext => Fq::zero(),
            PolyType::G1ScalarMulIndicator => Fq::zero(),
        }
    }

    /// Evaluate the constraint system for Stage 1 sumcheck
    ///
    /// Takes only x variables and evaluates F(x) = Σ_i γ^i * C_i(x)
    /// This is used in the square-and-multiply sumcheck.
    pub fn evaluate_constraints_batched(&self, x_vars: &[Fq], gamma: Fq) -> Fq {
        assert_eq!(x_vars.len(), self.matrix.num_constraint_vars);

        let mut result = Fq::zero();
        let mut gamma_power = gamma;

        for constraint in self.constraints.iter() {
            let constraint_eval = self.evaluate_constraint(constraint, x_vars);
            result += gamma_power * constraint_eval;
            gamma_power *= gamma;
        }

        result
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

    /// Evaluate a single constraint C_i(x) using the matrix layout.
    fn evaluate_constraint(&self, constraint: &MatrixConstraint, x: &[Fq]) -> Fq {
        let idx = constraint.constraint_index;

        match constraint.constraint_type {
            ConstraintType::PackedGtExp => {
                // For packed GT exp, g(x) only depends on element variables (high 4 bits)
                // Data layout: index = x_elem * 256 + s (s in low 8 bits, x_elem in high 4 bits)
                // So for a 12-var point, element vars are x[8..12]
                let g_eval = if x.len() == 12 {
                    // Extract element variables (high 4 bits) and evaluate 4-var g
                    // Reverse for big-endian convention used by DensePolynomial::evaluate
                    let x_elem_reversed: Vec<Fq> = x[8..12].iter().rev().copied().collect();
                    let g_4var = get_g_mle();
                    DensePolynomial::new(g_4var).evaluate(&x_elem_reversed)
                } else {
                    let x_reversed: Vec<Fq> = x.iter().rev().copied().collect();
                    self.g_poly.evaluate(&x_reversed)
                };

                let base_row = self.matrix.row_index(PolyType::Base, idx);
                let rho_prev_row = self.matrix.row_index(PolyType::RhoPrev, idx);
                let rho_curr_row = self.matrix.row_index(PolyType::RhoCurr, idx);
                let quotient_row = self.matrix.row_index(PolyType::Quotient, idx);
                let bit_row = self.matrix.row_index(PolyType::Bit, idx);

                let base_eval = self.matrix.evaluate_row(base_row, x);
                let rho_prev = self.matrix.evaluate_row(rho_prev_row, x);
                let rho_curr = self.matrix.evaluate_row(rho_curr_row, x);
                let quotient = self.matrix.evaluate_row(quotient_row, x);
                let bit_eval = self.matrix.evaluate_row(bit_row, x);

                // base^bit = 1 + (base - 1) * bit (linear interpolation for bit ∈ {0,1})
                let base_power = Fq::one() + (base_eval - Fq::one()) * bit_eval;

                rho_curr - rho_prev.square() * base_power - quotient * g_eval
            }
            ConstraintType::GtMul => {
                // GT mul uses 4-var polynomials with ZERO PADDING to 12-var
                // Zero padding: data in low indices 0-15, zeros in 16-4095
                // So element variables are in the LOW 4 bits: x[0..4]
                // Reverse for big-endian convention used by DensePolynomial::evaluate
                let g_eval = if x.len() == 12 {
                    let x_elem_reversed: Vec<Fq> = x[0..4].iter().rev().copied().collect();
                    let g_4var = get_g_mle();
                    DensePolynomial::new(g_4var).evaluate(&x_elem_reversed)
                } else {
                    let x_reversed: Vec<Fq> = x.iter().rev().copied().collect();
                    self.g_poly.evaluate(&x_reversed)
                };

                let lhs_row = self.matrix.row_index(PolyType::MulLhs, idx);
                let rhs_row = self.matrix.row_index(PolyType::MulRhs, idx);
                let result_row = self.matrix.row_index(PolyType::MulResult, idx);
                let quotient_row = self.matrix.row_index(PolyType::MulQuotient, idx);

                let lhs_eval = self.matrix.evaluate_row(lhs_row, x);
                let rhs_eval = self.matrix.evaluate_row(rhs_row, x);
                let result_eval = self.matrix.evaluate_row(result_row, x);
                let quotient_eval = self.matrix.evaluate_row(quotient_row, x);

                // GT mul constraint: lhs * rhs - result - quotient * g
                lhs_eval * rhs_eval - result_eval - quotient_eval * g_eval
            }
            ConstraintType::G1ScalarMul { base_point } => {
                // G1 scalar multiplication constraint evaluation
                // We evaluate all 4 constraints (C1-C4) at the given point x

                // Get the row indices for our MLEs
                let x_a_row = self.matrix.row_index(PolyType::G1ScalarMulXA, idx);
                let y_a_row = self.matrix.row_index(PolyType::G1ScalarMulYA, idx);
                let x_t_row = self.matrix.row_index(PolyType::G1ScalarMulXT, idx);
                let y_t_row = self.matrix.row_index(PolyType::G1ScalarMulYT, idx);
                let x_a_next_row = self.matrix.row_index(PolyType::G1ScalarMulXANext, idx);
                let y_a_next_row = self.matrix.row_index(PolyType::G1ScalarMulYANext, idx);

                // Evaluate MLEs at point x
                let x_a = self.matrix.evaluate_row(x_a_row, x);
                let y_a = self.matrix.evaluate_row(y_a_row, x);
                let x_t = self.matrix.evaluate_row(x_t_row, x);
                let y_t = self.matrix.evaluate_row(y_t_row, x);
                let x_a_next = self.matrix.evaluate_row(x_a_next_row, x);
                let y_a_next = self.matrix.evaluate_row(y_a_next_row, x);

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

    #[cfg(test)]
    pub fn verify_constraints_are_zero(&self) {
        // Verify that each constraint evaluates to 0 over the entire hypercube
        let num_x_points = 1 << self.matrix.num_constraint_vars;

        for constraint in &self.constraints {
            let idx = constraint.constraint_index;

            for x_val in 0..num_x_points {
                let mut x_binary = Vec::with_capacity(self.matrix.num_constraint_vars);
                let mut x = x_val;
                for _ in 0..self.matrix.num_constraint_vars {
                    x_binary.push(if x & 1 == 1 { Fq::one() } else { Fq::zero() });
                    x >>= 1;
                }

                let constraint_eval = self.evaluate_constraint(constraint, &x_binary);

                assert!(
                    constraint_eval == Fq::zero(),
                    "Constraint {} failed at x={:?}: got {}, expected 0",
                    idx,
                    x_binary,
                    constraint_eval
                );
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
        let num_x_vars = system.matrix.num_constraint_vars; // 8 variables

        // For each constraint, test it at random 12-variable points
        for (idx, constraint) in system.constraints.iter().enumerate() {
            // Test this constraint at 5 random points
            for trial in 0..5 {
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
                    "Constraint {} should evaluate to 0 at boolean points",
                    idx
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
            0 => Ok(PolyType::Base),
            1 => Ok(PolyType::RhoPrev),
            2 => Ok(PolyType::RhoCurr),
            3 => Ok(PolyType::Quotient),
            4 => Ok(PolyType::MulLhs),
            5 => Ok(PolyType::MulRhs),
            6 => Ok(PolyType::MulResult),
            7 => Ok(PolyType::MulQuotient),
            8 => Ok(PolyType::G1ScalarMulXA),
            9 => Ok(PolyType::G1ScalarMulYA),
            10 => Ok(PolyType::G1ScalarMulXT),
            11 => Ok(PolyType::G1ScalarMulYT),
            12 => Ok(PolyType::G1ScalarMulXANext),
            13 => Ok(PolyType::G1ScalarMulYANext),
            14 => Ok(PolyType::G1ScalarMulIndicator),
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
