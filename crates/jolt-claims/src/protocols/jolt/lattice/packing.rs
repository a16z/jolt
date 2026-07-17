//! Canonical auxiliary-object packings and the native Wjolt member registry.
//!
//! `jolt-openings::PrefixPacking` is the single source of truth for slot
//! assignment within auxiliary advice and committed-program objects. Wjolt is
//! not prefix-packed: [`wjolt_members`] returns the exact ordered members of
//! its native same-point one-hot commitment group.

use jolt_lookup_tables::{LookupTableKind, XLEN};
use jolt_openings::PrefixPacking;
use jolt_poly::math::Math;
use jolt_riscv::{NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS};

use super::super::geometry::dimensions::REGISTER_ADDRESS_BITS;
use super::super::geometry::ra::JoltRaPolynomialLayout;
use super::super::{BytecodeRegisterLane, JoltAdviceKind, JoltCommittedPolynomial};
use super::geometry::{
    byte_num_vars, word_byte_num_vars, LatticeGeometryError, UnsignedIncChunking,
};

/// Shape of the per-proof native commitment group (`W_jolt`): the canonical
/// committed Jolt data — `Ra` families, unsigned-inc chunks, and MSB — as
/// uniform one-hot members opened together at one point.
/// Advice byte columns are their own commitment objects
/// ([`advice_bytes_packing`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WJoltShape {
    pub ra_layout: JoltRaPolynomialLayout,
    pub log_t: usize,
    /// Shared one-hot chunk size: the address bits of each `Ra` family and
    /// the width of each unsigned-inc chunk (equal by the shared-final-point
    /// convention).
    pub log_k_chunk: usize,
}

/// Shape of the preprocessing-time packed commitment (`W_prog`,
/// committed-program mode): the per-lane bytecode decompositions and program
/// image bytes. All public or verifier-trusted, so their one-hot structure
/// is checked offline rather than by in-protocol relations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrecommittedPackingShape {
    pub bytecode_chunks: usize,
    /// Log of the row count of one bytecode chunk.
    pub log_bytecode_rows: usize,
    /// Byte places of one immediate lane (the field's canonical byte width).
    pub imm_byte_width: usize,
    pub program_image_log_words: Option<usize>,
}

/// Returns the canonical ordered native one-hot members of Wjolt.
pub fn wjolt_members(
    shape: &WJoltShape,
) -> Result<Vec<JoltCommittedPolynomial>, LatticeGeometryError> {
    let chunking = UnsignedIncChunking::new(shape.log_k_chunk)?;
    let mut polynomials = shape.ra_layout.committed_polynomials().collect::<Vec<_>>();
    polynomials.extend((0..chunking.chunk_count()).map(JoltCommittedPolynomial::UnsignedIncChunk));
    polynomials.push(JoltCommittedPolynomial::UnsignedIncMsb);
    Ok(polynomials)
}

/// Registers the canonical precommitted polynomial ordering and returns the
/// packing.
pub fn precommitted_packing(
    shape: &PrecommittedPackingShape,
) -> Result<PrefixPacking<JoltCommittedPolynomial>, LatticeGeometryError> {
    Ok(PrefixPacking::new(precommitted_polynomials(shape)?)?)
}

/// Registers an advice byte column as its own single-slot packing: advice is
/// committed per kind as a standalone object (untrusted per proof, trusted
/// precommitted), each carrying one byte one-hot column at the identity slot.
/// Routing every commitment object through a packing keeps the
/// packed-opening machinery uniform across objects.
pub fn advice_bytes_packing(
    kind: JoltAdviceKind,
    word_vars: usize,
) -> Result<PrefixPacking<JoltCommittedPolynomial>, LatticeGeometryError> {
    Ok(PrefixPacking::new([(
        JoltCommittedPolynomial::advice_bytes(kind),
        word_byte_num_vars(word_vars),
    )])?)
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
    Ok(polynomials)
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn wjolt_shape() -> WJoltShape {
        WJoltShape {
            ra_layout: JoltRaPolynomialLayout::new(2, 1, 1).unwrap(),
            log_t: 5,
            log_k_chunk: 8,
        }
    }

    fn precommitted_shape() -> PrecommittedPackingShape {
        PrecommittedPackingShape {
            bytecode_chunks: 2,
            log_bytecode_rows: 6,
            imm_byte_width: 16,
            program_image_log_words: Some(10),
        }
    }

    #[test]
    fn wjolt_members_cover_every_committed_lattice_polynomial() {
        let members = wjolt_members(&wjolt_shape()).unwrap();

        // 4 Ra polynomials + 8 inc chunks + msb.
        assert_eq!(members.len(), 4 + 8 + 1);
        assert_eq!(members[0], JoltCommittedPolynomial::InstructionRa(0));
        assert!(members.contains(&JoltCommittedPolynomial::UnsignedIncChunk(7)));
        assert_eq!(
            members.last(),
            Some(&JoltCommittedPolynomial::UnsignedIncMsb)
        );
    }

    /// Each advice byte column is its own commitment object: a single-slot
    /// packing whose slot is the whole domain (empty prefix).
    #[test]
    fn advice_bytes_packing_is_the_identity_slot() {
        for kind in [JoltAdviceKind::Untrusted, JoltAdviceKind::Trusted] {
            let packing = advice_bytes_packing(kind, 4).unwrap();
            assert_eq!(packing.iter().count(), 1);
            let slot = &packing[&JoltCommittedPolynomial::advice_bytes(kind)];
            assert!(slot.prefix.is_empty());
            assert_eq!(slot.num_vars, 8 + 3 + 4);
        }
    }

    #[test]
    fn wjolt_members_reject_invalid_chunk_widths() {
        let shape = WJoltShape {
            log_k_chunk: 7,
            ..wjolt_shape()
        };
        assert_eq!(
            wjolt_members(&shape),
            Err(LatticeGeometryError::ChunkWidthMisaligned { chunk_width: 7 })
        );
    }

    #[test]
    fn precommitted_packing_covers_every_bytecode_lane() {
        let packing = precommitted_packing(&precommitted_shape()).unwrap();

        let per_chunk = 3 + NUM_CIRCUIT_FLAGS + NUM_INSTRUCTION_FLAGS + 4;
        assert_eq!(packing.iter().count(), 2 * per_chunk + 1);
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
}
