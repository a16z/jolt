//! Canonical fixed-capacity prefix layouts for Akita commitment objects.
//!
//! The physical packing primitive lives in `jolt-openings`. This module owns
//! Jolt's semantic column order, zero-prefix embeddings, and layout digests.

use std::collections::BTreeMap;

use blake2::{digest::consts::U32, Blake2b, Digest};
use jolt_field::JoltField;
use jolt_lookup_tables::XLEN;
use jolt_openings::{EvaluationClaim, OpeningsError, PrefixPackedClaims, PrefixPackedLayout};
use jolt_poly::eq_index_msb;

use super::super::geometry::claim_reductions::bytecode::{
    committed_lane_vars, MAX_COMMITTED_BYTECODE_CHUNK_COUNT,
};
use super::super::geometry::ra::JoltRaPolynomialLayout;
use super::super::{JoltAdviceKind, JoltCommittedPolynomial, TracePolynomialOrder};
use super::geometry::{BalancedIncChunking, LatticeGeometryError};

/// Fixed selector capacity of the packed trace polynomial at K=16.
pub const ONE_HOT_TRACE_K16_CAPACITY: usize = 64;
/// Fixed selector capacity of the packed trace polynomial at K=256.
pub const ONE_HOT_TRACE_K256_CAPACITY: usize = 32;

/// Minimum physical arity of a bounded-dense commitment object (advice words,
/// program bytecode/image). Akita's dense DP planner admits no fold
/// schedule below 2^13 coefficients for these single-polynomial groups; one
/// variable of headroom over the current floor absorbs upstream repricing.
/// [`PrefixPackedObjectPlan::new`] pads slot capacity — never column arity —
/// up to this bound, so claim reduction is unchanged. Like any unused slot,
/// the padding is unconstrained committed data whose contribution to the
/// single reduced opening is zero w.h.p. under the sampled selector; nothing
/// may assume the padded region is identically zero.
pub const MIN_DENSE_OBJECT_NUM_VARS: usize = 14;

/// Shape of the per-proof `OneHotTrace`: the canonical committed Jolt data —
/// `Ra` families, balanced increment chunks, and signed carry as semantic
/// columns of one packed polynomial. Instruction, bytecode, and increment
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

/// Shape of the preprocessing-time direct bounded-dense committed-program
/// objects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrecommittedPackingShape {
    pub bytecode_chunks: usize,
    /// Log of the row count of one bytecode chunk.
    pub log_bytecode_rows: usize,
    pub trace_order: TracePolynomialOrder,
    pub program_image_log_words: Option<usize>,
}

/// One physical fixed-capacity prefix-packed polynomial and the logical
/// arity of each semantic column before zero-prefix embedding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixPackedObjectPlan {
    packing: PrefixPackedLayout<JoltCommittedPolynomial>,
    logical_num_vars: BTreeMap<JoltCommittedPolynomial, usize>,
    layout_digest: [u8; 32],
}

/// Direct committed-program layouts: one singleton object per bytecode chunk,
/// followed by the singleton program-image object when present.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrecommittedPackingPlan {
    bytecode_chunks: Vec<PrefixPackedObjectPlan>,
    program_image: Option<PrefixPackedObjectPlan>,
}

/// Returns the canonical ordered one-hot columns of `OneHotTrace`.
pub fn one_hot_trace_columns(
    shape: &OneHotTraceShape,
) -> Result<Vec<JoltCommittedPolynomial>, LatticeGeometryError> {
    let chunking = BalancedIncChunking::new(shape.log_k_chunk)?;
    let _capacity = one_hot_trace_column_capacity(shape.log_k_chunk)?;
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

/// Number of selector slots in the packed `OneHotTrace`.
pub const fn one_hot_trace_column_capacity(
    log_k_chunk: usize,
) -> Result<usize, LatticeGeometryError> {
    match log_k_chunk {
        4 => Ok(ONE_HOT_TRACE_K16_CAPACITY),
        8 => Ok(ONE_HOT_TRACE_K256_CAPACITY),
        chunk_width => Err(LatticeGeometryError::UnsupportedOneHotTraceChunkWidth { chunk_width }),
    }
}

/// Canonical committed-program packing plan.
pub fn precommitted_packing_plan(
    shape: &PrecommittedPackingShape,
) -> Result<PrecommittedPackingPlan, LatticeGeometryError> {
    if shape.bytecode_chunks == 0
        || !shape.bytecode_chunks.is_power_of_two()
        || shape.bytecode_chunks > MAX_COMMITTED_BYTECODE_CHUNK_COUNT
    {
        return Err(OpeningsError::InvalidSetup(format!(
            "direct bytecode chunk count must be a power of two in 1..={MAX_COMMITTED_BYTECODE_CHUNK_COUNT}"
        ))
        .into());
    }
    let chunk_num_vars = committed_lane_vars() + shape.log_bytecode_rows;
    if chunk_num_vars > DIRECT_PROGRAM_MAX_PHYSICAL_VARS
        || shape
            .program_image_log_words
            .is_some_and(|num_vars| num_vars > DIRECT_PROGRAM_MAX_PHYSICAL_VARS)
    {
        return Err(OpeningsError::InvalidSetup(format!(
            "direct committed-program arity exceeds {DIRECT_PROGRAM_MAX_PHYSICAL_VARS} variables"
        ))
        .into());
    }
    let bytecode_chunks = (0..shape.bytecode_chunks)
        .map(|chunk| {
            PrefixPackedObjectPlan::new_with_trace_order(
                b"program-bytecode-chunk-v1",
                vec![(
                    JoltCommittedPolynomial::BytecodeChunk(chunk),
                    chunk_num_vars,
                )],
                shape.trace_order,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let program_image = shape
        .program_image_log_words
        .map(|log_words| {
            PrefixPackedObjectPlan::new(
                b"program-image-init-v1",
                vec![(JoltCommittedPolynomial::ProgramImageInit, log_words)],
            )
        })
        .transpose()?;
    Ok(PrecommittedPackingPlan {
        bytecode_chunks,
        program_image,
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
        b"advice-dense-words-v2",
        vec![(polynomial, word_vars)],
        slot_capacity,
    )?)
}

impl PrefixPackedObjectPlan {
    fn new(
        domain: &[u8],
        columns: Vec<(JoltCommittedPolynomial, usize)>,
    ) -> Result<Self, OpeningsError> {
        let slot_capacity = columns.len().next_power_of_two();
        Self::new_with_slot_capacity(domain, columns, slot_capacity)
    }

    fn new_with_slot_capacity(
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
        let slot_capacity = slot_capacity
            .max(columns.len().next_power_of_two())
            .max(1usize << MIN_DENSE_OBJECT_NUM_VARS.saturating_sub(packed_logical_num_vars));
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
            packing,
            logical_num_vars,
            layout_digest,
        })
    }

    fn new_with_trace_order(
        domain: &[u8],
        columns: Vec<(JoltCommittedPolynomial, usize)>,
        trace_order: TracePolynomialOrder,
    ) -> Result<Self, OpeningsError> {
        let mut plan = Self::new(domain, columns)?;
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

impl PrecommittedPackingPlan {
    pub fn bytecode_chunks(&self) -> &[PrefixPackedObjectPlan] {
        &self.bytecode_chunks
    }

    pub const fn program_image(&self) -> Option<&PrefixPackedObjectPlan> {
        self.program_image.as_ref()
    }

    pub fn objects(&self) -> impl Iterator<Item = &PrefixPackedObjectPlan> {
        self.bytecode_chunks.iter().chain(self.program_image.iter())
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
        JoltCommittedPolynomial::BytecodeChunk(chunk) => (12, chunk, 0),
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

    fn precommitted_shape() -> PrecommittedPackingShape {
        PrecommittedPackingShape {
            bytecode_chunks: 2,
            log_bytecode_rows: 6,
            trace_order: TracePolynomialOrder::CycleMajor,
            program_image_log_words: Some(10),
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
    fn tiny_auxiliary_objects_pad_slot_capacity_to_the_planner_floor() {
        // A one-coefficient advice polynomial is below the planner
        // floor, so the plan widens its otherwise-empty selector capacity.
        for kind in [JoltAdviceKind::Untrusted, JoltAdviceKind::Trusted] {
            let plan = advice_packing_plan(kind, 0).unwrap();
            assert_eq!(plan.packing().ids().len(), 1);
            assert_eq!(plan.packing().logical_num_vars(), 0);
            assert_eq!(plan.packing().slot_capacity(), 1 << 14);
            assert_eq!(plan.packing().packed_num_vars(), MIN_DENSE_OBJECT_NUM_VARS);
        }

        // One variable below the floor, capacity doubles to reach it.
        let plan = advice_packing_plan(JoltAdviceKind::Untrusted, 13).unwrap();
        assert_eq!(plan.packing().logical_num_vars(), 13);
        assert_eq!(plan.packing().slot_capacity(), 2);
        assert_eq!(plan.packing().packed_num_vars(), MIN_DENSE_OBJECT_NUM_VARS);

        // At the floor exactly, capacity stays a single slot.
        let plan = advice_packing_plan(JoltAdviceKind::Trusted, 14).unwrap();
        assert_eq!(plan.packing().slot_capacity(), 1);
        assert_eq!(plan.packing().packed_num_vars(), MIN_DENSE_OBJECT_NUM_VARS);

        // A two-word program image pads the same way.
        let shape = PrecommittedPackingShape {
            program_image_log_words: Some(1),
            ..precommitted_shape()
        };
        let image_plan = precommitted_packing_plan(&shape).unwrap();
        let image = image_plan.program_image().unwrap();
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
    fn precommitted_packing_has_indexed_direct_singletons() {
        let plan = precommitted_packing_plan(&precommitted_shape()).unwrap();
        assert_eq!(plan.bytecode_chunks().len(), 2);
        for (index, chunk) in plan.bytecode_chunks().iter().enumerate() {
            assert_eq!(
                chunk.packing().ids(),
                [JoltCommittedPolynomial::BytecodeChunk(index)]
            );
            assert_eq!(
                chunk.packing().logical_num_vars(),
                committed_lane_vars() + 6
            );
            assert_eq!(chunk.packing().slot_capacity(), 1);
        }
        let image = plan.program_image().unwrap();
        assert_eq!(
            image.packing().ids(),
            [JoltCommittedPolynomial::ProgramImageInit]
        );
        assert_eq!(image.packing().logical_num_vars(), 10);
        assert_eq!(image.packing().packed_num_vars(), 14);
    }

    #[test]
    fn bytecode_layout_digest_binds_trace_order() {
        let cycle = precommitted_packing_plan(&precommitted_shape()).unwrap();
        let address = precommitted_packing_plan(&PrecommittedPackingShape {
            trace_order: TracePolynomialOrder::AddressMajor,
            ..precommitted_shape()
        })
        .unwrap();
        assert_ne!(
            cycle.bytecode_chunks()[0].layout_digest(),
            address.bytecode_chunks()[0].layout_digest()
        );
    }

    #[test]
    fn direct_program_plan_accepts_the_256_chunk_boundary() {
        let plan = precommitted_packing_plan(&PrecommittedPackingShape {
            bytecode_chunks: MAX_COMMITTED_BYTECODE_CHUNK_COUNT,
            log_bytecode_rows: 0,
            trace_order: TracePolynomialOrder::CycleMajor,
            program_image_log_words: Some(1),
        })
        .unwrap();
        assert_eq!(plan.bytecode_chunks().len(), 256);
        assert_eq!(plan.objects().count(), 257);
        assert!(plan
            .objects()
            .all(|object| object.packing().packed_num_vars() == MIN_DENSE_OBJECT_NUM_VARS));
    }

    #[test]
    fn direct_program_plan_rejects_arity_above_34() {
        assert!(precommitted_packing_plan(&PrecommittedPackingShape {
            bytecode_chunks: 1,
            log_bytecode_rows: 26,
            trace_order: TracePolynomialOrder::CycleMajor,
            program_image_log_words: Some(1),
        })
        .is_err());
        assert!(precommitted_packing_plan(&PrecommittedPackingShape {
            bytecode_chunks: 1,
            log_bytecode_rows: 0,
            trace_order: TracePolynomialOrder::CycleMajor,
            program_image_log_words: Some(35),
        })
        .is_err());
    }
}
