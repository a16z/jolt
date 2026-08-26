//! Canonical prefix-packed layout of the FR limb-column commitment object
//! (`FieldIncLimbs`): every balanced digit and carry column of every u64 limb
//! of `FieldRdInc`'s canonical representative, packed into one physical
//! polynomial exactly like the jolt `OneHotTrace`
//! (`specs/field-inline-portability.md`, Axis 1).
//!
//! Columns are committed row-major (`cycle ‖ digit-value`), and leaf claims
//! arrive at `(digit-value ‖ cycle)` points — the same permutation as the
//! `OneHotTrace` columns, so [`FieldIncLimbPackingPlan::column_point`]
//! mirrors that map.

use blake2::{digest::consts::U32, Blake2b, Digest};
use jolt_field::Field;
use jolt_openings::{OpeningsError, PrefixPackedLayout};

use super::geometry::FieldIncLimbGeometryError;
use crate::lattice::BalancedIncChunking;
use crate::protocols::field_inline::FieldInlineCommittedPolynomial;

/// Shape of the per-proof FR limb object: the limb count of the proof
/// field's canonical representative ([`super::field_inc_limb_count`]), the
/// trace arity, and the shared one-hot chunk size (the digit width, equal to
/// the `Ra` families' by the shared-final-point convention).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldIncLimbShape {
    pub limbs: usize,
    pub log_t: usize,
    pub log_k_chunk: usize,
}

/// The canonical ordered limb columns: limb-major, digits little-endian, the
/// limb's carry last — each limb group is the fused-inc column order applied
/// to that limb.
pub fn field_inc_limb_columns(
    shape: &FieldIncLimbShape,
) -> Result<Vec<FieldInlineCommittedPolynomial>, FieldIncLimbGeometryError> {
    if shape.limbs == 0 {
        return Err(FieldIncLimbGeometryError::ZeroLimbCount);
    }
    let chunking = BalancedIncChunking::new(shape.log_k_chunk)?;
    let mut columns = Vec::with_capacity(shape.limbs * (chunking.chunk_count() + 1));
    for limb in 0..shape.limbs {
        columns.extend(
            (0..chunking.chunk_count())
                .map(|index| FieldInlineCommittedPolynomial::FieldIncLimbDigit { limb, index }),
        );
        columns.push(FieldInlineCommittedPolynomial::FieldIncLimbCarry { limb });
    }
    Ok(columns)
}

/// Canonical column order and packed geometry of the FR limb object. Slot
/// capacity is the column count rounded up to a power of two — at two limbs
/// (a 128-bit proof field) this coincides with the `OneHotTrace` capacity at
/// both supported chunk widths, so the packed physical shape sits in the
/// already-catalogued Akita shape class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldIncLimbPackingPlan {
    packing: PrefixPackedLayout<FieldInlineCommittedPolynomial>,
    chunk_width: usize,
    layout_digest: [u8; 32],
}

impl FieldIncLimbPackingPlan {
    pub fn new(shape: &FieldIncLimbShape) -> Result<Self, FieldIncLimbGeometryError> {
        let columns = field_inc_limb_columns(shape)?;
        let capacity = columns.len().next_power_of_two();
        let logical_num_vars = shape.log_k_chunk.checked_add(shape.log_t).ok_or_else(|| {
            OpeningsError::InvalidSetup("field-inc limb variable count exceeds usize".to_owned())
        })?;
        let packing = PrefixPackedLayout::new(logical_num_vars, capacity, columns)?;
        let layout_digest = layout_digest(shape, &packing)?;
        Ok(Self {
            packing,
            chunk_width: shape.log_k_chunk,
            layout_digest,
        })
    }

    pub const fn packing(&self) -> &PrefixPackedLayout<FieldInlineCommittedPolynomial> {
        &self.packing
    }

    /// The digit-value (address) variable count of every limb column.
    pub const fn chunk_width(&self) -> usize {
        self.chunk_width
    }

    /// Protocol digest binding the ordered columns, dimensions, and layout
    /// version; checked against the commitment and setup metadata.
    pub const fn layout_digest(&self) -> [u8; 32] {
        self.layout_digest
    }

    /// Maps a limb column's leaf-claim point from `(digit-value ‖ cycle)` to
    /// the row-major committed order `(cycle ‖ digit-value)`.
    pub fn column_point<F: Field>(
        &self,
        polynomial: FieldInlineCommittedPolynomial,
        leaf_point: &[F],
    ) -> Result<Vec<F>, OpeningsError> {
        if self.packing.slot_index(&polynomial).is_none() {
            return Err(OpeningsError::InvalidBatch(format!(
                "polynomial {polynomial:?} is not a field-inc limb column"
            )));
        }
        let (address, cycle) = leaf_point
            .split_at_checked(self.chunk_width)
            .ok_or_else(|| {
                OpeningsError::InvalidBatch(format!(
                    "field-inc limb leaf point has {} variables, below its \
                 {}-variable digit-value block",
                    leaf_point.len(),
                    self.chunk_width
                ))
            })?;
        let mut point = Vec::with_capacity(leaf_point.len());
        point.extend_from_slice(cycle);
        point.extend_from_slice(address);
        Ok(point)
    }
}

fn layout_digest(
    shape: &FieldIncLimbShape,
    packing: &PrefixPackedLayout<FieldInlineCommittedPolynomial>,
) -> Result<[u8; 32], OpeningsError> {
    let mut hasher = Blake2b::<U32>::new();
    hasher.update(b"jolt/field-inline/akita/field-inc-limbs/v1");
    append_usize(&mut hasher, packing.logical_num_vars());
    append_usize(&mut hasher, packing.packed_num_vars());
    append_usize(&mut hasher, packing.slot_capacity());
    append_usize(&mut hasher, packing.selector_num_vars());
    append_usize(&mut hasher, packing.ids().len());
    append_usize(&mut hasher, shape.limbs);
    append_usize(&mut hasher, shape.log_t);
    append_usize(&mut hasher, shape.log_k_chunk);
    for column in packing.ids() {
        match column {
            FieldInlineCommittedPolynomial::FieldIncLimbDigit { limb, index } => {
                hasher.update([0]);
                append_usize(&mut hasher, *limb);
                append_usize(&mut hasher, *index);
            }
            FieldInlineCommittedPolynomial::FieldIncLimbCarry { limb } => {
                hasher.update([1]);
                append_usize(&mut hasher, *limb);
            }
            FieldInlineCommittedPolynomial::FieldRdInc => {
                return Err(OpeningsError::InvalidBatch(
                    "non-limb polynomial in the field-inc limb layout".to_owned(),
                ));
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
    use jolt_field::{Fr, Ring};

    fn shape(limbs: usize, log_t: usize, log_k_chunk: usize) -> FieldIncLimbShape {
        FieldIncLimbShape {
            limbs,
            log_t,
            log_k_chunk,
        }
    }

    #[test]
    fn columns_are_limb_major_digits_then_carry() {
        let columns = field_inc_limb_columns(&shape(2, 5, 8)).unwrap();
        assert_eq!(columns.len(), 2 * 9);
        assert_eq!(
            columns.first(),
            Some(&FieldInlineCommittedPolynomial::FieldIncLimbDigit { limb: 0, index: 0 })
        );
        assert_eq!(
            columns.get(8),
            Some(&FieldInlineCommittedPolynomial::FieldIncLimbCarry { limb: 0 })
        );
        assert_eq!(
            columns.get(9),
            Some(&FieldInlineCommittedPolynomial::FieldIncLimbDigit { limb: 1, index: 0 })
        );
        assert_eq!(
            columns.last(),
            Some(&FieldInlineCommittedPolynomial::FieldIncLimbCarry { limb: 1 })
        );
        assert_eq!(
            field_inc_limb_columns(&shape(0, 5, 8)),
            Err(FieldIncLimbGeometryError::ZeroLimbCount)
        );
    }

    /// The two-limb plan's physical shape at both supported chunk widths:
    /// capacity rounds the column count up to the `OneHotTrace` capacities
    /// (32 at K=256, 64 at K=16), so packed arity matches the catalogued
    /// shape class; the digest binds shape and order.
    #[test]
    fn packed_shape_and_digest_are_pinned() {
        let plan = FieldIncLimbPackingPlan::new(&shape(2, 5, 8)).unwrap();
        assert_eq!(plan.packing().slot_capacity(), 32);
        assert_eq!(plan.packing().logical_num_vars(), 13);
        assert_eq!(plan.packing().packed_num_vars(), 18);
        assert_eq!(plan.chunk_width(), 8);

        let k16 = FieldIncLimbPackingPlan::new(&shape(2, 5, 4)).unwrap();
        assert_eq!(k16.packing().ids().len(), 2 * 17);
        assert_eq!(k16.packing().slot_capacity(), 64);
        assert_eq!(k16.packing().packed_num_vars(), 6 + 4 + 5);

        let digest = plan.layout_digest();
        assert_ne!(digest, [0; 32]);
        assert_ne!(
            digest,
            FieldIncLimbPackingPlan::new(&shape(2, 6, 8))
                .unwrap()
                .layout_digest()
        );
        assert_ne!(
            digest,
            FieldIncLimbPackingPlan::new(&shape(4, 5, 8))
                .unwrap()
                .layout_digest()
        );
    }

    #[test]
    fn column_points_use_the_one_hot_trace_permutation() {
        let plan = FieldIncLimbPackingPlan::new(&shape(2, 3, 2)).unwrap();
        let leaf = (0..5).map(Fr::from_u64).collect::<Vec<_>>();
        let expected = [2, 3, 4, 0, 1]
            .into_iter()
            .map(Fr::from_u64)
            .collect::<Vec<_>>();
        for polynomial in [
            FieldInlineCommittedPolynomial::FieldIncLimbDigit { limb: 1, index: 0 },
            FieldInlineCommittedPolynomial::FieldIncLimbCarry { limb: 0 },
        ] {
            assert_eq!(plan.column_point(polynomial, &leaf).unwrap(), expected);
        }
        assert!(plan
            .column_point(FieldInlineCommittedPolynomial::FieldRdInc, &leaf)
            .is_err());
        assert!(plan
            .column_point(
                FieldInlineCommittedPolynomial::FieldIncLimbCarry { limb: 0 },
                &leaf[..1],
            )
            .is_err());
    }
}
