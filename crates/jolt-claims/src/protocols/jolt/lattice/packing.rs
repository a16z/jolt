//! Canonical fixed-capacity prefix layouts for Akita commitment objects.
//!
//! The physical packing primitive lives in `jolt-openings`. This module owns
//! Jolt's semantic column order, zero-prefix embeddings, and layout digests.

use std::collections::BTreeMap;

use blake2::{digest::consts::U32, Blake2b, Digest};
use jolt_field::JoltField;
use jolt_lookup_tables::XLEN;
use jolt_openings::{
    CommitmentGroupRole, EvaluationClaim, OpeningsError, PrefixPackedClaims, PrefixPackedLayout,
};
use jolt_poly::eq_index_msb;

use super::super::geometry::claim_reductions::bytecode::bytecode_total_vars;
use super::super::geometry::ra::JoltRaPolynomialLayout;
use super::super::{JoltAdviceKind, JoltCommittedPolynomial, TracePolynomialOrder};
use super::geometry::{BalancedIncChunking, LatticeGeometryError};

pub use crate::lattice::MIN_DENSE_OBJECT_NUM_VARS;

/// Shape of the per-proof `OneHotTrace`: the canonical committed Jolt data —
/// `Ra` families, balanced increment chunks, and signed carry as semantic
/// columns of one native commitment batch. Instruction, bytecode, and increment
/// columns omit row zero; RAM commits every row.
/// Advice word columns are their own commitment objects
/// ([`advice_packing_plan`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OneHotTraceShape {
    pub ra_layout: JoltRaPolynomialLayout,
    pub log_t: usize,
    /// Shared one-hot chunk size: the address bits of each `Ra` family and
    /// the width of each increment digit (equal by the shared-final-point
    /// convention).
    pub log_k_chunk: usize,
}

/// One physical fixed-capacity prefix-packed polynomial and the logical
/// arity of each semantic column before zero-prefix embedding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixPackedObjectPlan {
    role: CommitmentGroupRole,
    packing: PrefixPackedLayout<JoltCommittedPolynomial>,
    logical_num_vars: BTreeMap<JoltCommittedPolynomial, usize>,
    layout_digest: [u8; 32],
}

/// Direct committed-program layouts: whole bytecode followed by the initial image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrecommittedPackingPlan {
    pub bytecode: PrefixPackedObjectPlan,
    pub program_image: PrefixPackedObjectPlan,
}

/// Returns the canonical ordered one-hot columns of `OneHotTrace`.
pub fn one_hot_trace_columns(
    shape: &OneHotTraceShape,
) -> Result<Vec<JoltCommittedPolynomial>, LatticeGeometryError> {
    let chunking = BalancedIncChunking::new(shape.log_k_chunk)?;
    if !matches!(shape.log_k_chunk, 4 | 8) {
        return Err(LatticeGeometryError::UnsupportedOneHotTraceChunkWidth {
            chunk_width: shape.log_k_chunk,
        });
    }
    let instruction_columns = 2 * XLEN / shape.log_k_chunk;
    if shape.ra_layout.instruction() != instruction_columns {
        return Err(
            LatticeGeometryError::UnexpectedOneHotTraceInstructionColumns {
                chunk_width: shape.log_k_chunk,
                actual: shape.ra_layout.instruction(),
                expected: instruction_columns,
            },
        );
    }
    let mut polynomials = (0..instruction_columns)
        .map(JoltCommittedPolynomial::InstructionRa)
        .collect::<Vec<_>>();
    polynomials.extend((0..chunking.chunk_count()).map(JoltCommittedPolynomial::BalancedIncDigit));
    polynomials.push(JoltCommittedPolynomial::BalancedIncCarry);
    polynomials.extend((0..shape.ra_layout.bytecode()).map(JoltCommittedPolynomial::BytecodeRa));
    polynomials.extend((0..shape.ra_layout.ram()).map(JoltCommittedPolynomial::RamRa));
    Ok(polynomials)
}

/// Whole-bytecode and initial-image objects, each with its own local arity.
pub fn committed_program_packing_plan(
    bytecode_len: usize,
    program_image_len_words: usize,
    trace_order: TracePolynomialOrder,
) -> Result<PrecommittedPackingPlan, LatticeGeometryError> {
    let bytecode_vars = bytecode_total_vars(bytecode_len)
        .map_err(|error| OpeningsError::InvalidSetup(error.to_string()))?;
    let image_words = program_image_len_words
        .checked_next_power_of_two()
        .ok_or_else(|| {
            OpeningsError::InvalidSetup("program-image word count overflows".to_owned())
        })?
        .max(2);
    let image_vars = image_words.ilog2() as usize;
    if bytecode_vars > DIRECT_PROGRAM_MAX_PHYSICAL_VARS
        || image_vars > DIRECT_PROGRAM_MAX_PHYSICAL_VARS
    {
        return Err(OpeningsError::InvalidSetup(format!(
            "direct committed-program arity exceeds {DIRECT_PROGRAM_MAX_PHYSICAL_VARS} variables"
        ))
        .into());
    }
    Ok(PrecommittedPackingPlan {
        bytecode: PrefixPackedObjectPlan::new_with_trace_order(
            direct_program_role(JoltCommittedPolynomial::ProgramBytecode, 2)?,
            b"program-bytecode-whole-v1",
            vec![(JoltCommittedPolynomial::ProgramBytecode, bytecode_vars)],
            trace_order,
        )?,
        program_image: PrefixPackedObjectPlan::new(
            direct_program_role(JoltCommittedPolynomial::ProgramImageInit, 3)?,
            b"program-image-init-v1",
            vec![(JoltCommittedPolynomial::ProgramImageInit, image_vars)],
        )?,
    })
}

pub const ADVICE_MIN_PHYSICAL_VARS: usize = 14;
pub const ADVICE_MAX_PHYSICAL_VARS: usize = 34;
pub const DIRECT_PROGRAM_MAX_PHYSICAL_VARS: usize = 34;

/// Single-column advice-word layout, padded through empty prefix slots
/// when its logical arity is below Akita's dense schedule floor.
pub fn advice_packing_plan(
    kind: JoltAdviceKind,
    word_vars: usize,
) -> Result<PrefixPackedObjectPlan, LatticeGeometryError> {
    let physical_vars = word_vars.max(ADVICE_MIN_PHYSICAL_VARS);
    if physical_vars > ADVICE_MAX_PHYSICAL_VARS {
        return Err(OpeningsError::InvalidSetup(format!(
            "advice physical arity {physical_vars} is outside the supported {}..={} range",
            ADVICE_MIN_PHYSICAL_VARS, ADVICE_MAX_PHYSICAL_VARS
        ))
        .into());
    }
    let selector_vars = physical_vars - word_vars;
    let slot_capacity = 1usize.checked_shl(selector_vars as u32).ok_or_else(|| {
        OpeningsError::InvalidSetup("advice slot capacity exceeds usize".to_owned())
    })?;
    let polynomial = match kind {
        JoltAdviceKind::Trusted => JoltCommittedPolynomial::TrustedAdvice,
        JoltAdviceKind::Untrusted => JoltCommittedPolynomial::UntrustedAdvice,
    };
    Ok(PrefixPackedObjectPlan::new_with_slot_capacity(
        kind.group_role(),
        b"advice-dense-words-v2",
        vec![(polynomial, word_vars)],
        slot_capacity,
    )?)
}

impl PrefixPackedObjectPlan {
    fn new(
        role: CommitmentGroupRole,
        domain: &[u8],
        columns: Vec<(JoltCommittedPolynomial, usize)>,
    ) -> Result<Self, OpeningsError> {
        let slot_capacity = columns.len().next_power_of_two();
        Self::new_with_slot_capacity(role, domain, columns, slot_capacity)
    }

    fn new_with_slot_capacity(
        role: CommitmentGroupRole,
        domain: &[u8],
        columns: Vec<(JoltCommittedPolynomial, usize)>,
        slot_capacity: usize,
    ) -> Result<Self, OpeningsError> {
        if columns.is_empty() {
            return Err(OpeningsError::InvalidSetup(
                "prefix-packed object requires at least one column".to_string(),
            ));
        }
        let packed_logical_num_vars = columns
            .iter()
            .map(|(_, num_vars)| *num_vars)
            .max()
            .ok_or_else(|| {
                OpeningsError::InvalidSetup(
                    "prefix-packed object requires at least one column".to_string(),
                )
            })?;
        let slot_capacity = slot_capacity.max(crate::lattice::min_dense_slot_capacity(
            columns.len(),
            packed_logical_num_vars,
        ));
        let ids = columns.iter().map(|(id, _)| *id).collect::<Vec<_>>();
        let packing = PrefixPackedLayout::new(packed_logical_num_vars, slot_capacity, ids)?;
        let logical_num_vars = columns.iter().copied().collect::<BTreeMap<_, _>>();
        if logical_num_vars.len() != columns.len() {
            return Err(OpeningsError::InvalidSetup(
                "prefix-packed object contains a duplicate column".to_string(),
            ));
        }
        let layout_digest = packed_object_layout_digest(domain, &packing, &logical_num_vars, None)?;
        Ok(Self {
            role,
            packing,
            logical_num_vars,
            layout_digest,
        })
    }

    fn new_with_trace_order(
        role: CommitmentGroupRole,
        domain: &[u8],
        columns: Vec<(JoltCommittedPolynomial, usize)>,
        trace_order: TracePolynomialOrder,
    ) -> Result<Self, OpeningsError> {
        let mut plan = Self::new(role, domain, columns)?;
        plan.layout_digest = packed_object_layout_digest(
            domain,
            &plan.packing,
            &plan.logical_num_vars,
            Some(trace_order),
        )?;
        Ok(plan)
    }

    pub const fn packing(&self) -> &PrefixPackedLayout<JoltCommittedPolynomial> {
        &self.packing
    }

    pub const fn group_role(&self) -> CommitmentGroupRole {
        self.role
    }

    pub const fn layout_digest(&self) -> [u8; 32] {
        self.layout_digest
    }

    pub fn logical_num_vars(&self, id: JoltCommittedPolynomial) -> Option<usize> {
        self.logical_num_vars.get(&id).copied()
    }

    /// Aligns suffix-compatible logical claims at the widest point. A shorter
    /// polynomial is embedded under a zero prefix, so its evaluation is
    /// multiplied by `eq(common_prefix, 0)`.
    pub fn packed_claims<F: JoltField>(
        &self,
        claims: &BTreeMap<JoltCommittedPolynomial, EvaluationClaim<F>>,
    ) -> Result<PrefixPackedClaims<F>, OpeningsError> {
        let common_num_vars = self.packing.logical_num_vars();
        let common_point = self
            .packing
            .ids()
            .iter()
            .find_map(|id| {
                (self.logical_num_vars(*id) == Some(common_num_vars))
                    .then(|| claims.get(id))
                    .flatten()
            })
            .ok_or_else(|| {
                OpeningsError::InvalidBatch(
                    "prefix-packed object is missing a widest logical claim".to_string(),
                )
            })?
            .point
            .as_slice()
            .to_vec();

        let evaluations = self
            .packing
            .ids()
            .iter()
            .map(|id| {
                let claim = claims.get(id).ok_or_else(|| {
                    OpeningsError::InvalidBatch(format!("missing prefix-packed claim for {id:?}"))
                })?;
                let own_num_vars = self.logical_num_vars(*id).ok_or_else(|| {
                    OpeningsError::InvalidBatch(format!(
                        "missing logical arity for prefix-packed claim {id:?}"
                    ))
                })?;
                if claim.point.len() != own_num_vars {
                    return Err(OpeningsError::InvalidBatch(format!(
                        "claim for {id:?} has {} variables, expected {own_num_vars}",
                        claim.point.len()
                    )));
                }
                let prefix_len = common_num_vars - own_num_vars;
                if &common_point[prefix_len..] != claim.point.as_slice() {
                    return Err(OpeningsError::InvalidBatch(format!(
                        "claim for {id:?} is not a suffix of the common packed point"
                    )));
                }
                Ok(eq_index_msb(&common_point[..prefix_len], 0) * claim.value)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PrefixPackedClaims::new(
            self.layout_digest,
            common_point,
            evaluations,
        ))
    }
}

fn direct_program_role(
    id: JoltCommittedPolynomial,
    order: usize,
) -> Result<CommitmentGroupRole, OpeningsError> {
    let order = u64::try_from(order).map_err(|_| {
        OpeningsError::InvalidSetup("direct program role order exceeds u64".to_owned())
    })?;
    match id {
        JoltCommittedPolynomial::ProgramBytecode => Ok(CommitmentGroupRole::new(
            order,
            b"program_bytecode",
            "program-bytecode",
        )),
        JoltCommittedPolynomial::ProgramImageInit => Ok(CommitmentGroupRole::new(
            order,
            b"program_image_init",
            "program-image-init",
        )),
        _ => Err(OpeningsError::InvalidSetup(
            "direct program role requires a committed-program polynomial".to_owned(),
        )),
    }
}

impl PrecommittedPackingPlan {
    pub fn objects(&self) -> impl Iterator<Item = &PrefixPackedObjectPlan> {
        [&self.bytecode, &self.program_image].into_iter()
    }
}

fn packed_object_layout_digest(
    domain: &[u8],
    packing: &PrefixPackedLayout<JoltCommittedPolynomial>,
    logical_num_vars: &BTreeMap<JoltCommittedPolynomial, usize>,
    trace_order: Option<TracePolynomialOrder>,
) -> Result<[u8; 32], OpeningsError> {
    let mut hasher = Blake2b::<U32>::new();
    hasher.update(b"jolt/akita/fixed-prefix-object/v1");
    hasher.update((domain.len() as u64).to_le_bytes());
    hasher.update(domain);
    hasher.update([trace_order.map_or(u8::MAX, |order| order.transcript_scalar() as u8)]);
    append_usize(&mut hasher, packing.logical_num_vars());
    append_usize(&mut hasher, packing.packed_num_vars());
    append_usize(&mut hasher, packing.slot_capacity());
    append_usize(&mut hasher, packing.ids().len());
    for id in packing.ids() {
        append_packed_object_id(&mut hasher, *id)?;
        append_usize(
            &mut hasher,
            *logical_num_vars.get(id).ok_or_else(|| {
                OpeningsError::InvalidSetup("missing logical column arity".to_string())
            })?,
        );
    }
    Ok(hasher.finalize().into())
}

fn append_packed_object_id(
    hasher: &mut Blake2b<U32>,
    id: JoltCommittedPolynomial,
) -> Result<(), OpeningsError> {
    let (tag, index, secondary) = match id {
        JoltCommittedPolynomial::TrustedAdvice => (10, 0, 0),
        JoltCommittedPolynomial::UntrustedAdvice => (11, 0, 0),
        JoltCommittedPolynomial::ProgramBytecode => (14, 0, 0),
        JoltCommittedPolynomial::ProgramImageInit => (13, 0, 0),
        other => {
            return Err(OpeningsError::InvalidSetup(format!(
                "unsupported polynomial {other:?} in packed object layout"
            )))
        }
    };
    hasher.update([tag]);
    append_usize(hasher, index);
    append_usize(hasher, secondary);
    Ok(())
}

fn append_usize(hasher: &mut Blake2b<U32>, value: usize) {
    hasher.update((value as u64).to_le_bytes());
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn one_hot_trace_shape() -> OneHotTraceShape {
        OneHotTraceShape {
            ra_layout: JoltRaPolynomialLayout::new(16, 1, 1).unwrap(),
            log_t: 5,
            log_k_chunk: 8,
        }
    }

    #[test]
    fn one_hot_trace_columns_cover_every_committed_lattice_polynomial() {
        let columns = one_hot_trace_columns(&one_hot_trace_shape()).unwrap();

        assert_eq!(columns.len(), 16 + 8 + 1 + 2);
        assert_eq!(columns[0], JoltCommittedPolynomial::InstructionRa(0));
        assert!(columns.contains(&JoltCommittedPolynomial::BalancedIncDigit(7)));
        assert_eq!(columns[16], JoltCommittedPolynomial::BalancedIncDigit(0));
        assert_eq!(columns[24], JoltCommittedPolynomial::BalancedIncCarry);
        assert_eq!(columns[25], JoltCommittedPolynomial::BytecodeRa(0));
        assert_eq!(columns.last(), Some(&JoltCommittedPolynomial::RamRa(0)));
    }

    #[test]
    fn advice_packing_uses_the_schedule_floor() {
        for kind in [JoltAdviceKind::Untrusted, JoltAdviceKind::Trusted] {
            let id = match kind {
                JoltAdviceKind::Trusted => JoltCommittedPolynomial::TrustedAdvice,
                JoltAdviceKind::Untrusted => JoltCommittedPolynomial::UntrustedAdvice,
            };
            let plan = advice_packing_plan(kind, 9).unwrap();
            assert_eq!(plan.packing().ids().len(), 1);
            assert_eq!(plan.packing().ids(), &[id]);
            assert_eq!(plan.packing().selector_num_vars(), 5);
            assert_eq!(plan.packing().logical_num_vars(), 9);
            assert_eq!(plan.packing().packed_num_vars(), 14);
            assert_eq!(plan.packing().slot_capacity(), 32);
            assert_eq!(plan.logical_num_vars(id), Some(9));

            let large = advice_packing_plan(kind, 20).unwrap();
            assert_eq!(large.packing().selector_num_vars(), 0);
            assert_eq!(large.packing().packed_num_vars(), 20);
            assert_eq!(large.packing().slot_capacity(), 1);
        }
    }

    #[test]
    fn tiny_precommitted_objects_pad_slot_capacity_to_the_planner_floor() {
        for kind in [JoltAdviceKind::Untrusted, JoltAdviceKind::Trusted] {
            let plan = advice_packing_plan(kind, 0).unwrap();
            assert_eq!(plan.packing().ids().len(), 1);
            assert_eq!(plan.packing().logical_num_vars(), 0);
            assert_eq!(plan.packing().slot_capacity(), 1 << 14);
            assert_eq!(plan.packing().packed_num_vars(), MIN_DENSE_OBJECT_NUM_VARS);
        }

        let plan = advice_packing_plan(JoltAdviceKind::Untrusted, 13).unwrap();
        assert_eq!(plan.packing().logical_num_vars(), 13);
        assert_eq!(plan.packing().slot_capacity(), 2);
        assert_eq!(plan.packing().packed_num_vars(), MIN_DENSE_OBJECT_NUM_VARS);

        let plan = advice_packing_plan(JoltAdviceKind::Trusted, 14).unwrap();
        assert_eq!(plan.packing().slot_capacity(), 1);
        assert_eq!(plan.packing().packed_num_vars(), MIN_DENSE_OBJECT_NUM_VARS);

        let image_plan =
            committed_program_packing_plan(128, 2, TracePolynomialOrder::CycleMajor).unwrap();
        let image = &image_plan.program_image;
        assert_eq!(image.packing().logical_num_vars(), 1);
        assert_eq!(image.packing().slot_capacity(), 1 << 13);
        assert_eq!(image.packing().packed_num_vars(), MIN_DENSE_OBJECT_NUM_VARS);
    }

    #[test]
    fn padded_capacity_claims_reduce_to_the_slot_zero_embedding() {
        use jolt_field::{Fr, Ring};

        let plan = advice_packing_plan(JoltAdviceKind::Untrusted, 9).unwrap();
        let id = JoltCommittedPolynomial::UntrustedAdvice;
        let point = (0..plan.packing().logical_num_vars())
            .map(|index| Fr::from_u64(index as u64 + 3))
            .collect::<Vec<_>>();
        let value = Fr::from_u64(41);
        let claims = BTreeMap::from([(id, EvaluationClaim::new(point.clone(), value))]);

        let packed = plan.packed_claims(&claims).unwrap();
        assert_eq!(packed.point(), point.as_slice());
        assert_eq!(packed.evaluations(), &[value]);
    }

    #[test]
    fn one_hot_trace_columns_reject_invalid_chunk_widths() {
        let shape = OneHotTraceShape {
            log_k_chunk: 7,
            ..one_hot_trace_shape()
        };
        assert_eq!(
            one_hot_trace_columns(&shape),
            Err(LatticeGeometryError::ChunkWidthMisaligned { chunk_width: 7 })
        );
    }

    #[test]
    fn precommitted_packing_has_whole_bytecode_and_image() {
        let plan =
            committed_program_packing_plan(128, 513, TracePolynomialOrder::CycleMajor).unwrap();
        assert_eq!(plan.objects().count(), 2);
        assert_eq!(
            plan.bytecode.packing().ids(),
            [JoltCommittedPolynomial::ProgramBytecode]
        );
        assert_eq!(plan.bytecode.packing().logical_num_vars(), 16);
        let role = plan.bytecode.group_role();
        assert_eq!(role.order(), 2);
        assert_eq!(role.transcript_label(), b"program_bytecode");
        assert_eq!(role.transcript_index(), None);
        assert_eq!(
            plan.program_image.packing().ids(),
            [JoltCommittedPolynomial::ProgramImageInit]
        );
        assert_eq!(plan.program_image.group_role().order(), 3);
        assert_eq!(
            plan.program_image
                .logical_num_vars(JoltCommittedPolynomial::ProgramImageInit),
            Some(10)
        );
    }

    #[test]
    fn committed_program_plan_rejects_invalid_public_dimensions() {
        for rows in [0, 3, 127] {
            assert!(
                committed_program_packing_plan(rows, 2, TracePolynomialOrder::CycleMajor).is_err()
            );
        }
    }

    #[test]
    fn bytecode_layout_digest_binds_trace_order() {
        let cycle =
            committed_program_packing_plan(128, 2, TracePolynomialOrder::CycleMajor).unwrap();
        let address =
            committed_program_packing_plan(128, 2, TracePolynomialOrder::AddressMajor).unwrap();
        assert_ne!(
            cycle.bytecode.layout_digest(),
            address.bytecode.layout_digest()
        );
    }

    #[test]
    fn direct_program_plan_rejects_arity_above_34() {
        assert!(
            committed_program_packing_plan(1 << 26, 2, TracePolynomialOrder::CycleMajor).is_err()
        );
        assert!(
            committed_program_packing_plan(2, 1 << 35, TracePolynomialOrder::CycleMajor).is_err()
        );
        let boundary =
            committed_program_packing_plan(1 << 25, 2, TracePolynomialOrder::CycleMajor).unwrap();
        assert_eq!(boundary.bytecode.packing().packed_num_vars(), 34);
    }
}
