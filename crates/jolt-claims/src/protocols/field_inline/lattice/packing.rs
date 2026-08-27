//! Canonical dense prefix packing of the FR limb-word group.
//!
//! `FieldRdInc`'s per-cycle canonical u64 limbs form
//! [`field_inc_limb_count`](super::geometry::field_inc_limb_count) dense
//! columns over the trace domain, committed as ONE independent dense Akita
//! group (the advice-object treatment) and opened in the same heterogeneous
//! batch as advice and `OneHotTrace`. The stage-6b reduced `FieldRdInc`
//! claim binds to the group by the single linear identity
//! [`recompose_limbs`](super::geometry::recompose_limbs).

use blake2::{digest::consts::U32, Blake2b, Digest};
use jolt_field::Field;
#[cfg(feature = "akita")]
use jolt_openings::PrecommittedRole;
use jolt_openings::{OpeningsError, PrefixPackedClaims, PrefixPackedLayout};

use crate::lattice::min_dense_slot_capacity;

/// One limb-word column of the packed FR group, by little-endian limb index.
/// Local to the packing plan: the columns never appear as protocol opening
/// ids (no relation consumes them — only the stage-8 batch statement).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct FieldIncLimbWord(pub usize);

/// Shape of the per-proof FR limb group: the proof field's canonical limb
/// count and the trace arity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldIncLimbShape {
    pub limbs: usize,
    pub log_t: usize,
}

/// The canonical dense layout of the FR limb group: the limb-word columns in
/// little-endian limb order, prefix-packed into one physical polynomial at
/// the shared dense schedule floor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldIncLimbPackingPlan {
    packing: PrefixPackedLayout<FieldIncLimbWord>,
    layout_digest: [u8; 32],
}

impl FieldIncLimbPackingPlan {
    pub fn new(shape: &FieldIncLimbShape) -> Result<Self, OpeningsError> {
        let slot_capacity = min_dense_slot_capacity(shape.limbs, shape.log_t);
        let packing = PrefixPackedLayout::new(
            shape.log_t,
            slot_capacity,
            (0..shape.limbs).map(FieldIncLimbWord),
        )?;
        let layout_digest = layout_digest(&packing);
        Ok(Self {
            packing,
            layout_digest,
        })
    }

    pub const fn packing(&self) -> &PrefixPackedLayout<FieldIncLimbWord> {
        &self.packing
    }

    /// Protocol digest checked against the commitment and setup metadata.
    pub const fn layout_digest(&self) -> [u8; 32] {
        self.layout_digest
    }

    /// The semantic statement consumed by the generic reduction: every limb
    /// column's evaluation at the one shared point (the stage-6b reduced
    /// `FieldRdInc` opening point), in little-endian limb order.
    pub fn packed_claims<F: Field>(
        &self,
        point: Vec<F>,
        limb_evaluations: Vec<F>,
    ) -> Result<PrefixPackedClaims<F>, OpeningsError> {
        if point.len() != self.packing.logical_num_vars() {
            return Err(OpeningsError::InvalidBatch(format!(
                "FR limb point has {} variables, expected {}",
                point.len(),
                self.packing.logical_num_vars()
            )));
        }
        if limb_evaluations.len() != self.packing.ids().len() {
            return Err(OpeningsError::InvalidBatch(format!(
                "FR limb statement has {} evaluations for {} limb columns",
                limb_evaluations.len(),
                self.packing.ids().len()
            )));
        }
        Ok(PrefixPackedClaims::new(
            self.layout_digest,
            point,
            limb_evaluations,
        ))
    }
}

/// Role descriptor of the FR limb group in the final heterogeneous Akita
/// opening. WARNING: the order and transcript label are wire-shape-affecting
/// (canonical batch position after `TrustedAdvice(1)` and before the final
/// trace group, grouped schedule-row keying, transcript domain separation)
/// and frozen. The advice roles 0/1 live on the jolt protocol's
/// `JoltAdviceKind::precommitted_role`; the families share only the
/// [`PrecommittedRole`] total order.
#[cfg(feature = "akita")]
pub const fn field_inc_limbs_precommitted_role() -> PrecommittedRole {
    PrecommittedRole::new(2, b"field_inc_limbs", "field-inc-limbs")
}

fn layout_digest(packing: &PrefixPackedLayout<FieldIncLimbWord>) -> [u8; 32] {
    let mut hasher = Blake2b::<U32>::new();
    hasher.update(b"jolt/field-inline/akita/inc-limb-words/v1");
    append_usize(&mut hasher, packing.logical_num_vars());
    append_usize(&mut hasher, packing.packed_num_vars());
    append_usize(&mut hasher, packing.slot_capacity());
    append_usize(&mut hasher, packing.ids().len());
    for FieldIncLimbWord(limb) in packing.ids() {
        append_usize(&mut hasher, *limb);
    }
    hasher.finalize().into()
}

fn append_usize(hasher: &mut Blake2b<U32>, value: usize) {
    hasher.update((value as u64).to_le_bytes());
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use jolt_field::{Fr, Ring};

    use super::super::geometry::{field_inc_limb_count, recompose_limbs};
    use super::*;
    use crate::lattice::MIN_AUXILIARY_PACKED_NUM_VARS;

    fn shape(log_t: usize) -> FieldIncLimbShape {
        FieldIncLimbShape { limbs: 2, log_t }
    }

    /// Sub-floor trace arities pad through empty prefix slots to the dense
    /// schedule floor; above it the group carries exactly one selector
    /// variable per limb-count doubling.
    #[test]
    fn physical_arity_pads_to_the_dense_floor() {
        for (log_t, expected_physical) in [
            (11, MIN_AUXILIARY_PACKED_NUM_VARS),
            (13, MIN_AUXILIARY_PACKED_NUM_VARS),
            (16, 17),
            (24, 25),
        ] {
            let plan = FieldIncLimbPackingPlan::new(&shape(log_t)).unwrap();
            assert_eq!(plan.packing().logical_num_vars(), log_t);
            assert_eq!(plan.packing().packed_num_vars(), expected_physical);
            assert_eq!(plan.packing().ids().len(), 2);
        }
    }

    #[test]
    fn layout_digest_binds_the_shape() {
        let base = FieldIncLimbPackingPlan::new(&shape(16)).unwrap();
        assert_ne!(base.layout_digest(), [0; 32]);
        assert_ne!(
            base.layout_digest(),
            FieldIncLimbPackingPlan::new(&shape(17))
                .unwrap()
                .layout_digest()
        );
        assert_ne!(
            base.layout_digest(),
            FieldIncLimbPackingPlan::new(&FieldIncLimbShape {
                limbs: 4,
                log_t: 16
            })
            .unwrap()
            .layout_digest()
        );
    }

    #[test]
    fn packed_claims_reject_mismatched_shapes() {
        let plan = FieldIncLimbPackingPlan::new(&shape(4)).unwrap();
        let point = vec![Fr::from_u64(3); 4];
        assert!(plan
            .packed_claims(point.clone(), vec![Fr::from_u64(1); 2])
            .is_ok());
        assert!(plan
            .packed_claims(vec![Fr::from_u64(3); 5], vec![Fr::from_u64(1); 2])
            .is_err());
        assert!(plan.packed_claims(point, vec![Fr::from_u64(1); 3]).is_err());
    }

    /// The batch position and transcript label are frozen wire shape: the FR
    /// group follows the advice roles (0, 1) and precedes the final trace
    /// group, which is always last.
    #[cfg(feature = "akita")]
    #[test]
    fn precommitted_role_is_frozen() {
        let role = field_inc_limbs_precommitted_role();
        assert_eq!(role.order(), 2);
        assert_eq!(role.transcript_label(), b"field_inc_limbs");
    }

    /// The spec's Axis 1 identity holds for the recomposition helper the
    /// stage-8 linear check evaluates (against BN254's 4-limb count here; the
    /// 2-limb packed field pins itself where that field is visible).
    #[test]
    fn recompose_limbs_matches_the_limb_count() {
        let limbs: Vec<Fr> = (0..field_inc_limb_count::<Fr>())
            .map(|limb| Fr::from_u64(limb as u64 + 1))
            .collect();
        let expected = Fr::from_u64(1)
            + Fr::pow2(64) * Fr::from_u64(2)
            + Fr::pow2(128) * Fr::from_u64(3)
            + Fr::pow2(192) * Fr::from_u64(4);
        assert_eq!(recompose_limbs(&limbs), expected);
    }
}
