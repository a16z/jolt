//! Generic jagged-to-dense polynomial bijection transform
//!
//! This module implements a generic bijection between sparse "jagged" multilinear polynomials
//! and dense representations, following the approach from "Jagged Polynomial Commitments".
//!
//! The "jaggedness" in this context comes from polynomials having different numbers of variables
//! (e.g., 4-var vs 8-var), requiring padding in the sparse representation. The dense representation
//! excludes these redundant padded values to achieve compression.

use crate::field::JoltField;
use ark_bn254::Fq;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use super::constraints_sys::{ConstraintSystem, ConstraintType, MatrixConstraint, PolyType};

/// Core trait for jagged transforms with variable-count based heights
pub trait JaggedTransform<F: JoltField> {
    /// Get the polynomial index (row in sparse representation) for a dense index
    fn row(&self, dense_idx: usize) -> usize;

    /// Get the evaluation index (col in sparse representation) for a dense index
    fn col(&self, dense_idx: usize) -> usize;

    /// Map sparse (row, col) to dense index, returns None if outside bounds
    fn sparse_to_dense(&self, row: usize, col: usize) -> Option<usize>;

    /// Total non-redundant entries in dense representation
    fn dense_size(&self) -> usize;

    /// Get native variable count for a polynomial at given row
    fn poly_num_vars(&self, poly_idx: usize) -> usize;
}

/// Represents a polynomial with a specific number of variables
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct JaggedPolynomial {
    /// Number of variables (e.g., 4 or 8)
    pub num_vars: usize,
    /// Native size: 2^num_vars
    pub native_size: usize,
}

impl JaggedPolynomial {
    pub fn new(num_vars: usize) -> Self {
        Self {
            num_vars,
            native_size: 1 << num_vars,
        }
    }
}

/// Main bijection implementation for variable-count based jaggedness
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct VarCountJaggedBijection {
    /// Information for each polynomial
    polynomials: Vec<JaggedPolynomial>,
    /// Cumulative sizes for dense indexing
    cumulative_sizes: Vec<usize>,
    /// Total dense size
    total_size: usize,
}

impl VarCountJaggedBijection {
    /// Create a new bijection from polynomial specifications
    pub fn new(polynomials: Vec<JaggedPolynomial>) -> Self {
        let mut cumulative_sizes = Vec::with_capacity(polynomials.len());
        let mut total = 0;

        for poly in &polynomials {
            total += poly.native_size;
            cumulative_sizes.push(total);
        }

        Self {
            polynomials,
            cumulative_sizes,
            total_size: total,
        }
    }

    /// Get the number of polynomials
    pub fn num_polynomials(&self) -> usize {
        self.polynomials.len()
    }

    /// Get the cumulative size at the given index
    pub fn cumulative_size(&self, idx: usize) -> usize {
        self.cumulative_sizes[idx]
    }

    /// Get the cumulative size before the given index (0 if idx is 0)
    pub fn cumulative_size_before(&self, idx: usize) -> usize {
        if idx == 0 {
            0
        } else {
            self.cumulative_sizes[idx - 1]
        }
    }
}

impl<F: JoltField> JaggedTransform<F> for VarCountJaggedBijection {
    fn row(&self, dense_idx: usize) -> usize {
        if dense_idx >= self.total_size {
            panic!(
                "Dense index {} out of bounds (total size: {})",
                dense_idx, self.total_size
            );
        }

        // Binary search to find which polynomial contains this index
        self.cumulative_sizes
            .binary_search(&(dense_idx + 1))
            .unwrap_or_else(|x| x)
    }

    fn col(&self, dense_idx: usize) -> usize {
        let poly_idx = <Self as JaggedTransform<F>>::row(self, dense_idx);
        let poly_start = if poly_idx == 0 {
            0
        } else {
            self.cumulative_sizes[poly_idx - 1]
        };
        dense_idx - poly_start
    }

    fn sparse_to_dense(&self, row: usize, col: usize) -> Option<usize> {
        if row >= self.polynomials.len() {
            return None;
        }

        let poly = &self.polynomials[row];
        if col >= poly.native_size {
            return None;
        }

        let poly_start = if row == 0 {
            0
        } else {
            self.cumulative_sizes[row - 1]
        };

        Some(poly_start + col)
    }

    fn dense_size(&self) -> usize {
        self.total_size
    }

    fn poly_num_vars(&self, poly_idx: usize) -> usize {
        self.polynomials
            .get(poly_idx)
            .map(|p| p.num_vars)
            .unwrap_or(0)
    }
}

/// Maps between polynomial indices and constraint system structure
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ConstraintMapping {
    /// Maps polynomial index to (constraint_idx, poly_type)
    poly_to_constraint: Vec<(usize, PolyType)>,
}

impl ConstraintMapping {
    /// Decode polynomial index to constraint index and polynomial type
    pub fn decode(&self, poly_idx: usize) -> (usize, PolyType) {
        self.poly_to_constraint[poly_idx]
    }

    /// Create from list of (constraint_idx, poly_type, num_vars) tuples
    pub fn from_tuples(polynomials: &[(usize, PolyType, usize)]) -> Self {
        let poly_to_constraint = polynomials
            .iter()
            .map(|(idx, poly_type, _)| (*idx, *poly_type))
            .collect();

        Self { poly_to_constraint }
    }

    /// Get the number of polynomials
    pub fn num_polynomials(&self) -> usize {
        self.poly_to_constraint.len()
    }
}

/// Builder for creating jagged bijection from constraint system
pub struct ConstraintSystemJaggedBuilder {
    /// List of (constraint_idx, poly_type, num_vars) for each polynomial
    pub polynomials: Vec<(usize, PolyType, usize)>,
}

impl ConstraintSystemJaggedBuilder {
    /// Create builder from constraint system constraints
    /// This matches the matrix layout where rows are organized by polynomial type first
    pub fn from_constraints(constraints: &[MatrixConstraint]) -> Self {
        let mut polynomials = Vec::new();

        // Get all polynomial types that are actually used
        let mut used_poly_types = std::collections::HashSet::new();
        for constraint in constraints.iter() {
            match &constraint.constraint_type {
                ConstraintType::PackedGtExp => {
                    // Base, Bit, and RhoNext are not committed polynomials
                    used_poly_types.insert(PolyType::RhoPrev);
                    used_poly_types.insert(PolyType::Quotient);
                }
                ConstraintType::GtMul => {
                    used_poly_types.insert(PolyType::MulLhs);
                    used_poly_types.insert(PolyType::MulRhs);
                    used_poly_types.insert(PolyType::MulResult);
                    used_poly_types.insert(PolyType::MulQuotient);
                }
                ConstraintType::G1ScalarMul { .. } => {
                    used_poly_types.insert(PolyType::G1ScalarMulXA);
                    used_poly_types.insert(PolyType::G1ScalarMulYA);
                    used_poly_types.insert(PolyType::G1ScalarMulXT);
                    used_poly_types.insert(PolyType::G1ScalarMulYT);
                    used_poly_types.insert(PolyType::G1ScalarMulXANext);
                    used_poly_types.insert(PolyType::G1ScalarMulYANext);
                    used_poly_types.insert(PolyType::G1ScalarMulIndicator);
                }
            }
        }

        // Iterate through polynomial types in order (matching matrix layout)
        for poly_type in PolyType::all() {
            if !used_poly_types.contains(&poly_type) {
                continue;
            }

            // For each constraint, check if it uses this polynomial type
            for (idx, constraint) in constraints.iter().enumerate() {
                let num_vars = match &constraint.constraint_type {
                    ConstraintType::PackedGtExp => {
                        // Packed GT exp uses RhoPrev and Quotient (all 11-var)
                        // Base, Bit, and RhoNext are not committed polynomials
                        match poly_type {
                            PolyType::RhoPrev | PolyType::Quotient => Some(11),
                            _ => None,
                        }
                    }
                    ConstraintType::GtMul => {
                        // GT mul uses MulLhs, MulRhs, MulResult, MulQuotient (4-var padded to 11)
                        match poly_type {
                            PolyType::MulLhs
                            | PolyType::MulRhs
                            | PolyType::MulResult
                            | PolyType::MulQuotient => Some(4),
                            _ => None,
                        }
                    }
                    ConstraintType::G1ScalarMul { .. } => {
                        // G1 scalar mul uses all G1ScalarMul* types (8-var padded to 11)
                        match poly_type {
                            PolyType::G1ScalarMulXA
                            | PolyType::G1ScalarMulYA
                            | PolyType::G1ScalarMulXT
                            | PolyType::G1ScalarMulYT
                            | PolyType::G1ScalarMulXANext
                            | PolyType::G1ScalarMulYANext
                            | PolyType::G1ScalarMulIndicator => Some(8),
                            _ => None,
                        }
                    }
                };

                if let Some(num_vars) = num_vars {
                    polynomials.push((idx, poly_type, num_vars));
                }
            }
        }

        Self { polynomials }
    }

    /// Build the bijection and constraint mapping
    pub fn build(self) -> (VarCountJaggedBijection, ConstraintMapping) {
        let jagged_polys: Vec<JaggedPolynomial> = self
            .polynomials
            .iter()
            .map(|(_, _, num_vars)| JaggedPolynomial::new(*num_vars))
            .collect();

        let bijection = VarCountJaggedBijection::new(jagged_polys);
        let mapping = ConstraintMapping::from_tuples(&self.polynomials);

        (bijection, mapping)
    }
}

/// Extension methods for ConstraintSystem to build dense polynomial
impl ConstraintSystem {
    /// Build dense polynomial using generic jagged transform
    /// Returns the dense polynomial, the bijection, and the mapping used to create it
    pub fn build_dense_polynomial(
        &self,
    ) -> (
        crate::poly::dense_mlpoly::DensePolynomial<Fq>,
        VarCountJaggedBijection,
        ConstraintMapping,
    ) {
        let builder = ConstraintSystemJaggedBuilder::from_constraints(&self.constraints);
        let (bijection, mapping) = builder.build();

        // Pre-allocate for exact dense size
        let dense_size = <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&bijection);
        let mut dense_evals = Vec::with_capacity(dense_size);

        // Extract only native (non-padded) evaluations
        for dense_idx in 0..dense_size {
            let poly_idx =
                <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&bijection, dense_idx);
            let eval_idx =
                <VarCountJaggedBijection as JaggedTransform<Fq>>::col(&bijection, dense_idx);

            let (constraint_idx, poly_type) = mapping.decode(poly_idx);

            // Get the number of variables for this polynomial
            let _num_vars = <VarCountJaggedBijection as JaggedTransform<Fq>>::poly_num_vars(
                &bijection, poly_idx,
            );

            // Get the row in the matrix
            let matrix_row = self.matrix.row_index(poly_type, constraint_idx);
            let offset = self.matrix.storage_offset(matrix_row);

            // For 4-var polynomials with zero padding, values are stored directly
            // at the beginning of the padded array (no repetition).
            let sparse_idx = eval_idx; // Direct indexing for both 4-var and 8-var

            dense_evals.push(self.matrix.evaluations[offset + sparse_idx]);
        }

        // Pad to power of 2 for multilinear polynomial
        use ark_ff::Zero;
        let padded_size = dense_evals.len().next_power_of_two();
        dense_evals.resize(padded_size, Fq::zero());

        (
            crate::poly::dense_mlpoly::DensePolynomial::new(dense_evals),
            bijection,
            mapping,
        )
    }
}
