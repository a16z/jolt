//! The packed-witness semantics in one place: the canonical packing
//! registration and the final-opening map.
//!
//! `jolt-openings::PrefixPacking` is the single source of truth for slot
//! assignment — [`proof_packing`]/[`precommitted_packing`] register the
//! canonical polynomial orderings with it and hand back the packing object
//! itself; this module never does its own offset arithmetic. A packing is
//! keyed by [`JoltCommittedPolynomial`] directly: every logical polynomial
//! of a packed witness is a (lattice-mode) committed polynomial.

use jolt_lookup_tables::{LookupTableKind, XLEN};
use jolt_openings::PrefixPacking;
use jolt_poly::math::Math;
use jolt_riscv::{NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS};

use super::super::geometry::committed_openings::final_opening_id;
use super::super::geometry::dimensions::REGISTER_ADDRESS_BITS;
use super::super::geometry::ra::JoltRaPolynomialLayout;
use super::super::{BytecodeRegisterLane, JoltAdviceKind, JoltCommittedPolynomial, JoltOpeningId};
use super::geometry::{
    byte_num_vars, word_byte_num_vars, LatticeGeometryError, UnsignedIncChunking,
};

/// Shape of the per-proof packed commitment: every prover-supplied
/// polynomial is a slot of one packed witness, so a single Akita opening covers the
/// whole set (Akita has no commitment homomorphism to RLC separate
/// commitments with).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProofPackingShape {
    pub ra_layout: JoltRaPolynomialLayout,
    pub log_t: usize,
    /// Shared one-hot chunk size: the address bits of each `Ra` family and
    /// the width of each unsigned-inc chunk (equal by the shared-final-point
    /// convention).
    pub log_k_chunk: usize,
    /// `Some(word_vars)` when untrusted advice is committed.
    pub untrusted_advice_word_vars: Option<usize>,
}

/// Shape of the preprocessing-time packed commitment (committed-program
/// mode): the per-lane bytecode decompositions, program image bytes, trusted advice
/// bytes. All public or verifier-trusted, so their one-hot structure is
/// checked offline rather than by in-protocol relations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrecommittedPackingShape {
    pub bytecode_chunks: usize,
    /// Log of the row count of one bytecode chunk.
    pub log_bytecode_rows: usize,
    /// Byte places of one immediate lane (the field's canonical byte width).
    pub imm_byte_width: usize,
    pub program_image_log_words: Option<usize>,
    pub trusted_advice_word_vars: Option<usize>,
}

/// Registers the canonical per-proof polynomial ordering and returns the
/// packing.
pub fn proof_packing(
    shape: &ProofPackingShape,
) -> Result<PrefixPacking<JoltCommittedPolynomial>, LatticeGeometryError> {
    Ok(PrefixPacking::new(proof_polynomials(shape)?)?)
}

/// Registers the canonical precommitted polynomial ordering and returns the
/// packing.
pub fn precommitted_packing(
    shape: &PrecommittedPackingShape,
) -> Result<PrefixPacking<JoltCommittedPolynomial>, LatticeGeometryError> {
    Ok(PrefixPacking::new(precommitted_polynomials(shape)?)?)
}

fn proof_polynomials(
    shape: &ProofPackingShape,
) -> Result<Vec<(JoltCommittedPolynomial, usize)>, LatticeGeometryError> {
    let chunking = UnsignedIncChunking::new(shape.log_k_chunk)?;
    let one_hot_vars = shape.log_k_chunk + shape.log_t;

    let mut polynomials = Vec::new();
    polynomials.extend(
        shape
            .ra_layout
            .committed_polynomials()
            .map(|polynomial| (polynomial, one_hot_vars)),
    );
    polynomials.extend((0..chunking.chunk_count()).map(|index| {
        (
            JoltCommittedPolynomial::UnsignedIncChunk(index),
            one_hot_vars,
        )
    }));
    polynomials.push((JoltCommittedPolynomial::UnsignedIncMsb, shape.log_t));
    if let Some(word_vars) = shape.untrusted_advice_word_vars {
        polynomials.push((
            JoltCommittedPolynomial::advice_bytes(JoltAdviceKind::Untrusted),
            word_byte_num_vars(word_vars),
        ));
    }
    Ok(polynomials)
}

fn precommitted_polynomials(
    shape: &PrecommittedPackingShape,
) -> Result<Vec<(JoltCommittedPolynomial, usize)>, LatticeGeometryError> {
    let log_rows = shape.log_bytecode_rows;
    let selector_vars = REGISTER_ADDRESS_BITS + log_rows;
    let lookup_vars = LookupTableKind::<XLEN>::COUNT.log_2() + log_rows;

    let mut polynomials = Vec::new();
    for chunk in 0..shape.bytecode_chunks {
        polynomials.extend(BytecodeRegisterLane::ALL.into_iter().map(|lane| {
            (
                JoltCommittedPolynomial::BytecodeRegisterSelector { chunk, lane },
                selector_vars,
            )
        }));
        polynomials.extend((0..NUM_CIRCUIT_FLAGS).map(|flag| {
            (
                JoltCommittedPolynomial::BytecodeCircuitFlag { chunk, flag },
                log_rows,
            )
        }));
        polynomials.extend((0..NUM_INSTRUCTION_FLAGS).map(|flag| {
            (
                JoltCommittedPolynomial::BytecodeInstructionFlag { chunk, flag },
                log_rows,
            )
        }));
        polynomials.push((
            JoltCommittedPolynomial::BytecodeLookupSelector { chunk },
            lookup_vars,
        ));
        polynomials.push((JoltCommittedPolynomial::BytecodeRafFlag { chunk }, log_rows));
        polynomials.push((
            JoltCommittedPolynomial::BytecodeUnexpandedPcBytes { chunk },
            word_byte_num_vars(log_rows),
        ));
        polynomials.push((
            JoltCommittedPolynomial::BytecodeImmBytes { chunk },
            byte_num_vars(shape.imm_byte_width, log_rows)?,
        ));
    }
    if let Some(log_words) = shape.program_image_log_words {
        polynomials.push((
            JoltCommittedPolynomial::ProgramImageBytes,
            word_byte_num_vars(log_words),
        ));
    }
    if let Some(word_vars) = shape.trusted_advice_word_vars {
        polynomials.push((
            JoltCommittedPolynomial::advice_bytes(JoltAdviceKind::Trusted),
            word_byte_num_vars(word_vars),
        ));
    }
    Ok(polynomials)
}

/// How one committed polynomial's final opening is resolved in lattice mode —
/// the replacement for the base stage-8 RLC batch (which needs the commitment
/// homomorphism Akita lacks).
///
/// Claims flow through the relation DAG until, per polynomial, one claim
/// remains that no relation consumes — its **final claim**. In lattice mode
/// that claim is either settled by the packed witness opening (`Packed`) or
/// never exists because every claim on the polynomial is consumed by a
/// lattice relation (`Virtualized`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LatticeFinalOpening {
    /// One evaluation claim on the packed witness. `final_claim` names the
    /// producing relation output — the `(polynomial, relation)` opening id
    /// whose bound point and claimed value become this slot's
    /// `(id, EvaluationClaim)` in the packed statement. The packed witness
    /// admits exactly one claim per slot, so this is total — a packed
    /// polynomial without a producing relation cannot exist.
    Packed { final_claim: JoltOpeningId },
    /// Never PCS-opened: the polynomial's consumer claims leave the base PIOP
    /// through lattice relations (the `IncVirtualization` chain, the
    /// advice / bytecode-chunk / program-image reconstruction relations).
    Virtualized,
}

/// The final-opening resolution for each committed polynomial. Total over
/// `JoltCommittedPolynomial` so a new committed polynomial cannot land in
/// lattice mode without a resolution decision. The leaves derive from the
/// base `committed_openings::final_opening_id`, the single owner of the
/// polynomial→relation mapping.
pub fn final_opening(polynomial: JoltCommittedPolynomial) -> LatticeFinalOpening {
    match polynomial {
        // Consumed by the IncVirtualization chain.
        JoltCommittedPolynomial::RdInc | JoltCommittedPolynomial::RamInc => {
            LatticeFinalOpening::Virtualized
        }
        // Word/lane-valued polynomials whose claims are settled against their
        // one-hot decompositions by the reconstruction relations.
        JoltCommittedPolynomial::TrustedAdvice
        | JoltCommittedPolynomial::UntrustedAdvice
        | JoltCommittedPolynomial::BytecodeChunk(_)
        | JoltCommittedPolynomial::ProgramImageInit => LatticeFinalOpening::Virtualized,
        // Packed polynomials: one claim each, produced by the named relation.
        JoltCommittedPolynomial::InstructionRa(_)
        | JoltCommittedPolynomial::BytecodeRa(_)
        | JoltCommittedPolynomial::RamRa(_)
        | JoltCommittedPolynomial::UnsignedIncChunk(_)
        | JoltCommittedPolynomial::UnsignedIncMsb
        | JoltCommittedPolynomial::TrustedAdviceBytes
        | JoltCommittedPolynomial::UntrustedAdviceBytes
        | JoltCommittedPolynomial::BytecodeRegisterSelector { .. }
        | JoltCommittedPolynomial::BytecodeCircuitFlag { .. }
        | JoltCommittedPolynomial::BytecodeInstructionFlag { .. }
        | JoltCommittedPolynomial::BytecodeLookupSelector { .. }
        | JoltCommittedPolynomial::BytecodeRafFlag { .. }
        | JoltCommittedPolynomial::BytecodeUnexpandedPcBytes { .. }
        | JoltCommittedPolynomial::BytecodeImmBytes { .. }
        | JoltCommittedPolynomial::ProgramImageBytes => LatticeFinalOpening::Packed {
            final_claim: final_opening_id(polynomial),
        },
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::super::relations::advice_reconstruction::{
        trusted_advice_bytes_opening, untrusted_advice_bytes_opening,
    };
    use super::super::relations::bytecode_reconstruction::{
        bytecode_imm_bytes_opening, bytecode_raf_flag_opening, bytecode_register_selector_opening,
    };
    use super::super::relations::program_image_reconstruction::program_image_bytes_opening;
    use super::*;

    fn proof_shape() -> ProofPackingShape {
        ProofPackingShape {
            ra_layout: JoltRaPolynomialLayout::new(2, 1, 1).unwrap(),
            log_t: 5,
            log_k_chunk: 8,
            untrusted_advice_word_vars: Some(4),
        }
    }

    fn precommitted_shape() -> PrecommittedPackingShape {
        PrecommittedPackingShape {
            bytecode_chunks: 2,
            log_bytecode_rows: 6,
            imm_byte_width: 16,
            program_image_log_words: Some(10),
            trusted_advice_word_vars: Some(4),
        }
    }

    #[test]
    fn proof_packing_covers_every_committed_lattice_polynomial() {
        let packing = proof_packing(&proof_shape()).unwrap();

        // 4 Ra polynomials + 8 inc chunks + msb + untrusted advice bytes.
        assert_eq!(packing.iter().count(), 4 + 8 + 1 + 1);
        assert_eq!(
            packing[&JoltCommittedPolynomial::InstructionRa(0)].num_vars,
            13
        );
        assert_eq!(
            packing[&JoltCommittedPolynomial::UnsignedIncChunk(7)].num_vars,
            13
        );
        assert_eq!(
            packing[&JoltCommittedPolynomial::UnsignedIncMsb].num_vars,
            5
        );
        assert_eq!(
            packing[&JoltCommittedPolynomial::UntrustedAdviceBytes].num_vars,
            8 + 3 + 4
        );
    }

    #[test]
    fn proof_packing_rejects_invalid_chunk_widths() {
        let shape = ProofPackingShape {
            log_k_chunk: 7,
            ..proof_shape()
        };
        assert_eq!(
            proof_packing(&shape),
            Err(LatticeGeometryError::ChunkWidthMisaligned { chunk_width: 7 })
        );
    }

    #[test]
    fn precommitted_packing_covers_every_bytecode_lane() {
        let packing = precommitted_packing(&precommitted_shape()).unwrap();

        let per_chunk = 3 + NUM_CIRCUIT_FLAGS + NUM_INSTRUCTION_FLAGS + 4;
        assert_eq!(packing.iter().count(), 2 * per_chunk + 2);
        assert_eq!(
            packing[&JoltCommittedPolynomial::BytecodeRegisterSelector {
                chunk: 1,
                lane: BytecodeRegisterLane::Rd,
            }]
                .num_vars,
            REGISTER_ADDRESS_BITS + 6,
        );
        assert_eq!(
            packing[&JoltCommittedPolynomial::BytecodeCircuitFlag { chunk: 0, flag: 0 }].num_vars,
            6
        );
        assert_eq!(
            packing[&JoltCommittedPolynomial::BytecodeUnexpandedPcBytes { chunk: 1 }].num_vars,
            8 + 3 + 6
        );
        assert_eq!(
            packing[&JoltCommittedPolynomial::BytecodeImmBytes { chunk: 0 }].num_vars,
            8 + 4 + 6
        );
        assert_eq!(
            packing[&JoltCommittedPolynomial::ProgramImageBytes].num_vars,
            8 + 3 + 10
        );
    }

    #[test]
    fn word_and_lane_valued_polynomials_are_virtualized() {
        for polynomial in [
            JoltCommittedPolynomial::RdInc,
            JoltCommittedPolynomial::RamInc,
            JoltCommittedPolynomial::TrustedAdvice,
            JoltCommittedPolynomial::UntrustedAdvice,
            JoltCommittedPolynomial::BytecodeChunk(3),
            JoltCommittedPolynomial::ProgramImageInit,
        ] {
            assert_eq!(final_opening(polynomial), LatticeFinalOpening::Virtualized);
        }
    }

    /// Every packed polynomial's claim is produced by its reconstruction /
    /// reduction relation — the final claim is total, including the
    /// precommitted bytecode/program-image decompositions.
    #[test]
    fn packed_final_claims_name_the_producing_relations() {
        for (polynomial, final_claim) in [
            (
                JoltCommittedPolynomial::UntrustedAdviceBytes,
                untrusted_advice_bytes_opening(),
            ),
            (
                JoltCommittedPolynomial::TrustedAdviceBytes,
                trusted_advice_bytes_opening(),
            ),
            (
                JoltCommittedPolynomial::ProgramImageBytes,
                program_image_bytes_opening(),
            ),
            (
                JoltCommittedPolynomial::BytecodeRegisterSelector {
                    chunk: 1,
                    lane: BytecodeRegisterLane::Rd,
                },
                bytecode_register_selector_opening(1, BytecodeRegisterLane::Rd),
            ),
            (
                JoltCommittedPolynomial::BytecodeRafFlag { chunk: 0 },
                bytecode_raf_flag_opening(0),
            ),
            (
                JoltCommittedPolynomial::BytecodeImmBytes { chunk: 2 },
                bytecode_imm_bytes_opening(2),
            ),
        ] {
            assert_eq!(
                final_opening(polynomial),
                LatticeFinalOpening::Packed { final_claim }
            );
        }
    }

    #[test]
    fn every_proof_packed_polynomial_has_a_claim_source() {
        let packing = proof_packing(&proof_shape()).unwrap();
        for (polynomial, _) in &packing {
            assert!(matches!(
                final_opening(*polynomial),
                LatticeFinalOpening::Packed { .. }
            ));
        }
    }

    #[test]
    fn every_precommitted_packed_polynomial_has_a_claim_source() {
        let packing = precommitted_packing(&precommitted_shape()).unwrap();
        for (polynomial, _) in &packing {
            assert!(matches!(
                final_opening(*polynomial),
                LatticeFinalOpening::Packed { .. }
            ));
        }
    }
}
