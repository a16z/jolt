use serde::{Deserialize, Serialize};

use crate::protocols::jolt::{JoltCommittedPolynomial, JoltOpeningId, JoltRelationId};

use super::families::JoltPackingFamilyId;
use super::UNSIGNED_INC_BITS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct UnsignedIncChunking {
    pub chunk_count: usize,
    pub alphabet_size: usize,
    pub radix: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LatticeFinalOpeningRequirement {
    PackingLayoutFamily {
        family: JoltPackingFamilyId,
        relation: JoltRelationId,
    },
    LogicalOnly,
}

pub fn inc_virtualization_relation() -> JoltRelationId {
    JoltRelationId::IncVirtualization
}

pub fn unsigned_inc_claim_reduction_relation() -> JoltRelationId {
    JoltRelationId::UnsignedIncClaimReduction
}

pub fn unsigned_inc_chunk_reconstruction_relation() -> JoltRelationId {
    JoltRelationId::UnsignedIncChunkReconstruction
}

pub fn inc_virtualization_input_openings() -> [JoltOpeningId; 4] {
    [
        inc_virtualization_ram_read_write_opening(),
        inc_virtualization_ram_val_check_opening(),
        inc_virtualization_rd_read_write_opening(),
        inc_virtualization_rd_val_evaluation_opening(),
    ]
}

pub fn inc_virtualization_output_openings() -> [JoltOpeningId; 2] {
    [
        inc_virtualization_inc_opening(),
        inc_virtualization_store_opening(),
    ]
}

pub fn inc_virtualization_ram_read_write_opening() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RamInc,
        JoltRelationId::RamReadWriteChecking,
    )
}

pub fn inc_virtualization_ram_val_check_opening() -> JoltOpeningId {
    JoltOpeningId::committed(JoltCommittedPolynomial::RamInc, JoltRelationId::RamValCheck)
}

pub fn inc_virtualization_rd_read_write_opening() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RdInc,
        JoltRelationId::RegistersReadWriteChecking,
    )
}

pub fn inc_virtualization_rd_val_evaluation_opening() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RdInc,
        JoltRelationId::RegistersValEvaluation,
    )
}

pub fn inc_virtualization_inc_opening() -> JoltOpeningId {
    JoltOpeningId::lattice(JoltRelationId::IncVirtualization, 0)
}

pub fn inc_virtualization_store_opening() -> JoltOpeningId {
    JoltOpeningId::lattice(JoltRelationId::IncVirtualization, 1)
}

pub fn unsigned_inc_input_opening() -> JoltOpeningId {
    inc_virtualization_inc_opening()
}

pub fn unsigned_inc_opening() -> JoltOpeningId {
    JoltOpeningId::lattice(JoltRelationId::UnsignedIncClaimReduction, 0)
}

pub fn unsigned_inc_msb_opening() -> JoltOpeningId {
    JoltOpeningId::lattice(JoltRelationId::UnsignedIncClaimReduction, 1)
}

pub fn unsigned_inc_chunk_opening(index: usize) -> JoltOpeningId {
    JoltOpeningId::lattice(JoltRelationId::UnsignedIncClaimReduction, 2 + index)
}

pub fn unsigned_inc_lower_chunk_count(log_k_chunk: usize) -> Option<usize> {
    Some(unsigned_inc_chunking(log_k_chunk)?.chunk_count)
}

pub(crate) fn unsigned_inc_chunking(log_k_chunk: usize) -> Option<UnsignedIncChunking> {
    if log_k_chunk != 0 && UNSIGNED_INC_BITS.is_multiple_of(log_k_chunk) {
        let shift = u32::try_from(log_k_chunk).ok()?;
        Some(UnsignedIncChunking {
            chunk_count: UNSIGNED_INC_BITS / log_k_chunk,
            alphabet_size: 1usize.checked_shl(shift)?,
            radix: 1u64.checked_shl(shift)?,
        })
    } else {
        None
    }
}

pub fn final_opening_lattice_requirement(
    polynomial: JoltCommittedPolynomial,
) -> LatticeFinalOpeningRequirement {
    match polynomial {
        JoltCommittedPolynomial::RdInc | JoltCommittedPolynomial::RamInc => {
            LatticeFinalOpeningRequirement::LogicalOnly
        }
        JoltCommittedPolynomial::InstructionRa(index) => packed_family_requirement(
            JoltPackingFamilyId::InstructionRa { index },
            JoltRelationId::HammingWeightClaimReduction,
        ),
        JoltCommittedPolynomial::BytecodeRa(index) => packed_family_requirement(
            JoltPackingFamilyId::BytecodeRa { index },
            JoltRelationId::HammingWeightClaimReduction,
        ),
        JoltCommittedPolynomial::RamRa(index) => packed_family_requirement(
            JoltPackingFamilyId::RamRa { index },
            JoltRelationId::HammingWeightClaimReduction,
        ),
        JoltCommittedPolynomial::TrustedAdvice => packed_family_requirement(
            JoltPackingFamilyId::AdviceBytes {
                kind: crate::protocols::jolt::JoltAdviceKind::Trusted,
                index: 0,
            },
            JoltRelationId::AdviceClaimReduction,
        ),
        JoltCommittedPolynomial::UntrustedAdvice => packed_family_requirement(
            JoltPackingFamilyId::AdviceBytes {
                kind: crate::protocols::jolt::JoltAdviceKind::Untrusted,
                index: 0,
            },
            JoltRelationId::AdviceClaimReduction,
        ),
        JoltCommittedPolynomial::BytecodeChunk(index) => packed_family_requirement(
            JoltPackingFamilyId::BytecodeChunk { index },
            JoltRelationId::BytecodeClaimReduction,
        ),
        JoltCommittedPolynomial::ProgramImageInit => packed_family_requirement(
            JoltPackingFamilyId::ProgramImageInit,
            JoltRelationId::ProgramImageClaimReduction,
        ),
    }
}

fn packed_family_requirement(
    family: JoltPackingFamilyId,
    relation: JoltRelationId,
) -> LatticeFinalOpeningRequirement {
    LatticeFinalOpeningRequirement::PackingLayoutFamily { family, relation }
}
