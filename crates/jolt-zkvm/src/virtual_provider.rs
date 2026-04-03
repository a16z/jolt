//! On-demand computation of virtual polynomials from the R1CS witness.
//!
//! [`VirtualProvider`] materializes polynomials that are derived from the
//! per-cycle R1CS witness vector but are NOT direct column slices. The main
//! example is the product factors: `ProductLeft` and `ProductRight` are
//! domain-indexed buffers (T × stride) whose entries are the A-row and B-row
//! evaluations of the 3 product constraints at each cycle.

use jolt_compiler::PolynomialId;
use jolt_compute::{Buf, ComputeBackend, DeviceBuffer};
use jolt_field::Field;
use jolt_r1cs::constraints::rv64::*;

/// Product constraint domain: 3 constraints, padded stride of 4.
const NUM_PRODUCT: usize = NUM_PRODUCT_CONSTRAINTS;
const PRODUCT_STRIDE: usize = NUM_PRODUCT.next_power_of_two(); // 4

/// R1CS variable indices for the left (A-row) factor of each product constraint.
const PRODUCT_LEFT_VARS: [usize; NUM_PRODUCT] = [
    V_LEFT_INSTRUCTION_INPUT, // k=0: LeftInstructionInput
    V_LOOKUP_OUTPUT,          // k=1: LookupOutput
    V_FLAG_JUMP,              // k=2: Jump flag
];


/// Provides virtual polynomial buffers computed from the R1CS witness.
///
/// Dispatches on [`PolynomialId`] to compute domain-indexed product factor
/// buffers on demand. Holds an immutable reference to the flat R1CS witness
/// and the per-cycle variable stride.
pub struct VirtualProvider<'a, F> {
    witness: &'a [F],
    num_cycles: usize,
    vars_padded: usize,
}

impl<'a, F: Field> VirtualProvider<'a, F> {
    pub fn new(witness: &'a [F], num_cycles: usize, vars_padded: usize) -> Self {
        Self {
            witness,
            num_cycles,
            vars_padded,
        }
    }

    /// Compute `ProductLeft`: domain-indexed buffer of length `num_cycles × PRODUCT_STRIDE`.
    ///
    /// Layout: `buf[c * PRODUCT_STRIDE + k]` = A-row factor of product constraint k at cycle c.
    fn compute_product_left(&self) -> Vec<F> {
        let mut buf = vec![F::zero(); self.num_cycles * PRODUCT_STRIDE];
        for c in 0..self.num_cycles {
            let w = c * self.vars_padded;
            for k in 0..NUM_PRODUCT {
                buf[c * PRODUCT_STRIDE + k] = self.witness[w + PRODUCT_LEFT_VARS[k]];
            }
        }
        buf
    }

    /// Compute `ProductRight`: domain-indexed buffer of length `num_cycles × PRODUCT_STRIDE`.
    ///
    /// Layout: `buf[c * PRODUCT_STRIDE + k]` = B-row factor of product constraint k at cycle c.
    ///
    /// Constraint k=2 (ShouldJump) has B = 1 − NextIsNoop, computed inline.
    fn compute_product_right(&self) -> Vec<F> {
        let mut buf = vec![F::zero(); self.num_cycles * PRODUCT_STRIDE];
        for c in 0..self.num_cycles {
            let w = c * self.vars_padded;
            // k=0: RightInstructionInput
            buf[c * PRODUCT_STRIDE] = self.witness[w + V_RIGHT_INSTRUCTION_INPUT];
            // k=1: Branch
            buf[c * PRODUCT_STRIDE + 1] = self.witness[w + V_BRANCH];
            // k=2: 1 − NextIsNoop
            buf[c * PRODUCT_STRIDE + 2] =
                self.witness[w + V_CONST] - self.witness[w + V_NEXT_IS_NOOP];
        }
        buf
    }

    /// Load a virtual polynomial by ID, uploading to the backend.
    pub fn load<B: ComputeBackend>(&self, poly_id: PolynomialId, backend: &B) -> Buf<B, F> {
        let data = match poly_id {
            PolynomialId::ProductLeft => self.compute_product_left(),
            PolynomialId::ProductRight => self.compute_product_right(),
            other => panic!(
                "VirtualProvider does not handle {other:?} — \
                 add a compute method or route through another provider"
            ),
        };
        DeviceBuffer::Field(backend.upload(&data))
    }

    /// Returns true if this provider handles the given polynomial.
    pub fn handles(&self, poly_id: PolynomialId) -> bool {
        matches!(
            poly_id,
            PolynomialId::ProductLeft | PolynomialId::ProductRight
        )
    }
}
