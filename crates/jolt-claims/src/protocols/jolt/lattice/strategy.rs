//! Canonical layout of the prefix-packed Akita `OneHotTrace` commitment.
//!
//! The protocol fixes the semantic column order and selector capacity. Every
//! column has the same `(cycle || address)` point, so
//! [`jolt_openings::PrefixPackedLayout`] reduces the columns directly to one
//! opening of one physical polynomial.

use std::ops::Range;

use blake2::{digest::consts::U32, Blake2b, Digest};
use jolt_field::Field;
use jolt_openings::{OpeningsError, PrefixPackedClaims, PrefixPackedLayout};

use super::super::JoltCommittedPolynomial;
use super::packing::{one_hot_trace_column_capacity, one_hot_trace_columns, OneHotTraceShape};

/// `OneHotTrace` is committed as one prefix-packed physical polynomial.
pub const ONE_HOT_TRACE_LAYOUT: OneHotTraceLayout = OneHotTraceLayout;

/// The one protocol layout for the per-proof `OneHotTrace` commitment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OneHotTraceLayout;

/// Semantic column ranges in the packed selector domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneHotTraceColumnRanges {
    pub instruction: Range<usize>,
    pub bytecode: Range<usize>,
    pub ram: Range<usize>,
    pub balanced_inc: Range<usize>,
    pub balanced_inc_carry: usize,
}

/// Canonical column order and packed geometry for one proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneHotTraceLayoutPlan {
    packing: PrefixPackedLayout<JoltCommittedPolynomial>,
    ranges: OneHotTraceColumnRanges,
    layout_digest: [u8; 32],
}

/// The commitment-object setup shape the layout requires.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OneHotTraceSetupShape {
    pub num_vars: usize,
    pub num_polys: usize,
}

impl OneHotTraceLayout {
    /// The canonical object layout for `shape`.
    pub fn plan(&self, shape: &OneHotTraceShape) -> Result<OneHotTraceLayoutPlan, OpeningsError> {
        let columns = one_hot_trace_columns(shape)
            .map_err(|error| OpeningsError::InvalidBatch(error.to_string()))?;
        let column_capacity = one_hot_trace_column_capacity(shape.log_k_chunk)
            .map_err(|error| OpeningsError::InvalidBatch(error.to_string()))?;
        if columns.len() > column_capacity {
            return Err(OpeningsError::InvalidBatch(
                super::geometry::LatticeGeometryError::TooManyOneHotTraceColumns {
                    chunk_width: shape.log_k_chunk,
                    actual: columns.len(),
                    capacity: column_capacity,
                }
                .to_string(),
            ));
        }

        let instruction_end = shape.ra_layout.instruction();
        let balanced_inc_end =
            instruction_end + super::geometry::FUSED_INC_BITS / shape.log_k_chunk;
        let balanced_inc_carry = balanced_inc_end;
        let bytecode_start = balanced_inc_carry + 1;
        let bytecode_end = bytecode_start + shape.ra_layout.bytecode();
        let ram_end = bytecode_end + shape.ra_layout.ram();
        debug_assert_eq!(ram_end, columns.len());

        let packing =
            PrefixPackedLayout::new(shape.log_k_chunk + shape.log_t, column_capacity, columns)?;
        let layout_digest = layout_digest(shape, &packing)?;
        Ok(OneHotTraceLayoutPlan {
            packing,
            ranges: OneHotTraceColumnRanges {
                instruction: 0..instruction_end,
                bytecode: bytecode_start..bytecode_end,
                ram: bytecode_end..ram_end,
                balanced_inc: instruction_end..balanced_inc_end,
                balanced_inc_carry,
            },
            layout_digest,
        })
    }

    /// The commitment-object setup shape.
    pub fn setup_shape(
        &self,
        shape: &OneHotTraceShape,
    ) -> Result<OneHotTraceSetupShape, OpeningsError> {
        let plan = self.plan(shape)?;
        Ok(OneHotTraceSetupShape {
            num_vars: plan.packing.packed_num_vars(),
            num_polys: 1,
        })
    }

    /// Digest binding the ordered columns, dimensions, and layout version.
    pub fn layout_digest(&self, shape: &OneHotTraceShape) -> Result<[u8; 32], OpeningsError> {
        Ok(self.plan(shape)?.layout_digest)
    }

    /// Maps a column's leaf-claim point from `(address || cycle)` to the
    /// row-major committed order `(cycle || address)`.
    pub fn column_point<F: Field>(
        &self,
        polynomial: JoltCommittedPolynomial,
        chunk_width: usize,
        leaf_point: &[F],
    ) -> Result<Vec<F>, OpeningsError> {
        let invalid = |message: String| OpeningsError::InvalidBatch(message);
        if !matches!(
            polynomial,
            JoltCommittedPolynomial::InstructionRa(_)
                | JoltCommittedPolynomial::BytecodeRa(_)
                | JoltCommittedPolynomial::RamRa(_)
                | JoltCommittedPolynomial::BalancedIncDigit(_)
                | JoltCommittedPolynomial::BalancedIncCarry
        ) {
            return Err(invalid(format!(
                "polynomial {polynomial:?} is not a OneHotTrace column"
            )));
        }
        if leaf_point.len() < chunk_width {
            return Err(invalid(format!(
                "OneHotTrace leaf point has {} variables, below its \
                 {chunk_width}-variable address block",
                leaf_point.len()
            )));
        }
        let (address, cycle) = leaf_point.split_at(chunk_width);
        let mut point = Vec::with_capacity(leaf_point.len());
        point.extend_from_slice(cycle);
        point.extend_from_slice(address);
        Ok(point)
    }
}

impl OneHotTraceLayoutPlan {
    /// Generic prefix layout with the protocol's ordered column identifiers.
    pub const fn packing(&self) -> &PrefixPackedLayout<JoltCommittedPolynomial> {
        &self.packing
    }

    /// Column-family ranges used when constructing the packed witness.
    pub const fn ranges(&self) -> &OneHotTraceColumnRanges {
        &self.ranges
    }

    /// Protocol digest checked against the commitment and setup metadata.
    pub const fn layout_digest(&self) -> [u8; 32] {
        self.layout_digest
    }

    /// Constructs the semantic statement consumed by the generic reduction.
    pub fn packed_claims<F: Field>(
        &self,
        point: Vec<F>,
        evaluations: Vec<F>,
    ) -> PrefixPackedClaims<F> {
        PrefixPackedClaims::new(self.layout_digest, point, evaluations)
    }
}

fn layout_digest(
    shape: &OneHotTraceShape,
    packing: &PrefixPackedLayout<JoltCommittedPolynomial>,
) -> Result<[u8; 32], OpeningsError> {
    let mut hasher = Blake2b::<U32>::new();
    hasher.update(b"jolt/akita/one_hot_trace/digit-zero-mu-one-full-ram/v7");
    append_usize(&mut hasher, packing.logical_num_vars());
    append_usize(&mut hasher, packing.packed_num_vars());
    append_usize(&mut hasher, packing.slot_capacity());
    append_usize(&mut hasher, packing.selector_num_vars());
    append_usize(&mut hasher, packing.ids().len());
    append_usize(&mut hasher, shape.log_t);
    append_usize(&mut hasher, shape.log_k_chunk);
    append_usize(&mut hasher, shape.ra_layout.instruction());
    append_usize(&mut hasher, shape.ra_layout.bytecode());
    append_usize(&mut hasher, shape.ra_layout.ram());
    for column in packing.ids() {
        match column {
            JoltCommittedPolynomial::InstructionRa(index) => {
                hasher.update([0]);
                append_usize(&mut hasher, *index);
            }
            JoltCommittedPolynomial::BytecodeRa(index) => {
                hasher.update([1]);
                append_usize(&mut hasher, *index);
            }
            JoltCommittedPolynomial::RamRa(index) => {
                hasher.update([2]);
                append_usize(&mut hasher, *index);
            }
            JoltCommittedPolynomial::BalancedIncDigit(index) => {
                hasher.update([3]);
                append_usize(&mut hasher, *index);
            }
            JoltCommittedPolynomial::BalancedIncCarry => hasher.update([4]),
            other => {
                return Err(OpeningsError::InvalidBatch(format!(
                    "non-OneHotTrace polynomial {other:?} in packed one-hot layout"
                )));
            }
        }
    }
    Ok(hasher.finalize().into())
}

fn append_usize(hasher: &mut Blake2b<U32>, value: usize) {
    hasher.update((value as u64).to_le_bytes());
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
    use jolt_field::{Fr, Ring};

    fn shape(log_t: usize) -> OneHotTraceShape {
        OneHotTraceShape {
            ra_layout: JoltRaPolynomialLayout::new(16, 1, 1).unwrap(),
            log_t,
            log_k_chunk: 8,
        }
    }

    #[test]
    fn packed_layout_has_fixed_capacity_and_is_digest_bound() {
        let plan = ONE_HOT_TRACE_LAYOUT.plan(&shape(5)).unwrap();
        assert_eq!(plan.packing().logical_num_vars(), 13);
        assert_eq!(plan.packing().packed_num_vars(), 18);
        assert_eq!(plan.packing().slot_capacity(), 32);
        assert_eq!(plan.packing().selector_num_vars(), 5);
        assert_eq!(plan.ranges().instruction, 0..16);
        assert_eq!(plan.ranges().balanced_inc, 16..24);
        assert_eq!(plan.ranges().balanced_inc_carry, 24);
        assert_eq!(plan.ranges().bytecode, 25..26);
        assert_eq!(plan.ranges().ram, 26..27);
        assert_eq!(
            plan.packing().ids().last(),
            Some(&JoltCommittedPolynomial::RamRa(0))
        );
        assert_eq!(
            ONE_HOT_TRACE_LAYOUT.setup_shape(&shape(5)).unwrap(),
            OneHotTraceSetupShape {
                num_vars: 18,
                num_polys: 1,
            }
        );

        let digest = ONE_HOT_TRACE_LAYOUT.layout_digest(&shape(5)).unwrap();
        assert_ne!(digest, [0; 32]);
        assert_ne!(
            digest,
            ONE_HOT_TRACE_LAYOUT.layout_digest(&shape(6)).unwrap()
        );
    }

    #[test]
    fn every_column_uses_the_same_point_permutation() {
        let leaf = (0..5).map(Fr::from_u64).collect::<Vec<_>>();
        let expected = [2, 3, 4, 0, 1]
            .into_iter()
            .map(Fr::from_u64)
            .collect::<Vec<_>>();
        for polynomial in [
            JoltCommittedPolynomial::InstructionRa(0),
            JoltCommittedPolynomial::BytecodeRa(0),
            JoltCommittedPolynomial::RamRa(0),
            JoltCommittedPolynomial::BalancedIncDigit(0),
            JoltCommittedPolynomial::BalancedIncCarry,
        ] {
            assert_eq!(
                ONE_HOT_TRACE_LAYOUT
                    .column_point(polynomial, 2, &leaf)
                    .unwrap(),
                expected
            );
        }
    }
}
