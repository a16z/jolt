//! On-demand computation of derived polynomials.
//!
//! [`DerivedSource`] materializes [`PolySource::Derived`] polynomials from
//! the per-cycle R1CS witness vector. These are polynomials that require
//! non-trivial reshaping or combination of witness data — neither direct
//! witness columns nor R1CS matrix-vector products.
//!
//! Two sources of data:
//! - **Computed from witness**: ProductLeft/ProductRight — domain-indexed
//!   product factors extracted from the per-cycle R1CS witness.
//! - **Precomputed buffers**: RamVal, RamCombinedRa, RamValFinal, RamRafRa —
//!   computed externally from trace data and inserted before proving.

use std::borrow::Cow;
use std::collections::HashMap;

use jolt_compiler::PolynomialId;
use jolt_field::Field;
use jolt_r1cs::constraints::rv64::*;

/// Maps a [`PolynomialId`] to its R1CS witness variable index, or `None`
/// if the polynomial is not a direct column of the per-cycle witness.
fn witness_var(poly_id: PolynomialId) -> Option<usize> {
    // First try the compiler's canonical mapping (covers R1CS inputs + OpFlags).
    if let Some(idx) = poly_id.r1cs_variable_index() {
        return Some(idx);
    }
    // Product factor variables live outside the R1CS input range.
    match poly_id {
        PolynomialId::BranchFlag => Some(V_BRANCH),
        PolynomialId::NextIsNoop => Some(V_NEXT_IS_NOOP),
        _ => None,
    }
}

/// Product constraint stride: next power of two of 3 product constraints.
const PRODUCT_STRIDE: usize = NUM_PRODUCT_CONSTRAINTS.next_power_of_two(); // 4

/// A-row variable indices for each product constraint (k = 0, 1, 2).
///
/// Maps directly to the R1CS product constraint definitions:
///   k=0: Product = LeftInstructionInput × RightInstructionInput
///   k=1: ShouldBranch = LookupOutput × Branch
///   k=2: ShouldJump = Jump × (1 − NextIsNoop)
const PRODUCT_A_VARS: [usize; NUM_PRODUCT_CONSTRAINTS] = [
    V_LEFT_INSTRUCTION_INPUT,
    V_LOOKUP_OUTPUT,
    V_FLAG_JUMP,
];

/// Computes derived polynomials from the flat R1CS witness and
/// precomputed buffers inserted before proving.
pub struct DerivedSource<'a, F> {
    witness: &'a [F],
    num_cycles: usize,
    vars_padded: usize,
    precomputed: HashMap<PolynomialId, Vec<F>>,
}

impl<'a, F: Field> DerivedSource<'a, F> {
    pub fn new(witness: &'a [F], num_cycles: usize, vars_padded: usize) -> Self {
        Self {
            witness,
            num_cycles,
            vars_padded,
            precomputed: HashMap::new(),
        }
    }

    /// Insert a precomputed derived polynomial buffer.
    ///
    /// Used for polynomials that require external data (trace, committed
    /// polys, initial memory) beyond the R1CS witness.
    pub fn insert(&mut self, poly_id: PolynomialId, data: Vec<F>) {
        let _ = self.precomputed.insert(poly_id, data);
    }

    /// Compute or retrieve a derived polynomial by ID.
    pub fn compute(&self, poly_id: PolynomialId) -> Cow<'_, [F]> {
        if let Some(data) = self.precomputed.get(&poly_id) {
            return Cow::Borrowed(data);
        }
        match poly_id {
            PolynomialId::ProductLeft => Cow::Owned(self.product_left()),
            PolynomialId::ProductRight => Cow::Owned(self.product_right()),
            other => {
                if let Some(var) = witness_var(other) {
                    Cow::Owned(self.extract_column(var))
                } else {
                    panic!(
                        "DerivedSource: {other:?} is PolySource::Derived but has no compute \
                         method and was not precomputed"
                    )
                }
            }
        }
    }

    /// Extract a single R1CS variable column as a T-element vector.
    fn extract_column(&self, var: usize) -> Vec<F> {
        (0..self.num_cycles)
            .map(|c| self.witness[c * self.vars_padded + var])
            .collect()
    }

    /// Domain-indexed left (A-row) factors: `buf[c * 4 + k]` = A_k(c).
    fn product_left(&self) -> Vec<F> {
        let mut buf = vec![F::zero(); self.num_cycles * PRODUCT_STRIDE];
        for c in 0..self.num_cycles {
            let w = c * self.vars_padded;
            for (k, &var) in PRODUCT_A_VARS.iter().enumerate() {
                buf[c * PRODUCT_STRIDE + k] = self.witness[w + var];
            }
        }
        buf
    }

    /// Domain-indexed right (B-row) factors: `buf[c * 4 + k]` = B_k(c).
    ///
    /// k=2 (ShouldJump) has B = 1 − NextIsNoop, not a single variable.
    fn product_right(&self) -> Vec<F> {
        let mut buf = vec![F::zero(); self.num_cycles * PRODUCT_STRIDE];
        for c in 0..self.num_cycles {
            let w = c * self.vars_padded;
            buf[c * PRODUCT_STRIDE] = self.witness[w + V_RIGHT_INSTRUCTION_INPUT];
            buf[c * PRODUCT_STRIDE + 1] = self.witness[w + V_BRANCH];
            buf[c * PRODUCT_STRIDE + 2] =
                self.witness[w + V_CONST] - self.witness[w + V_NEXT_IS_NOOP];
        }
        buf
    }
}
