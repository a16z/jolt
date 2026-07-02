use jolt_lookup_tables::{LookupTableKind, XLEN};
use jolt_riscv::{NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS};

use super::super::geometry::dimensions::REGISTER_ADDRESS_BITS;
use super::super::geometry::ra::JoltRaPolynomialLayout;
use super::super::{JoltAdviceKind, JoltCommittedPolynomial};
use super::geometry::{
    byte_column_vars, ceil_log2, one_hot_column_vars, LatticeGeometryError, UnsignedIncChunking,
    WORD_BYTE_LIMBS,
};
use super::ids::{BytecodeRegisterLane, LatticeColumn};

/// Shape of the per-proof packed commitment: every prover-supplied column is
/// a slot of one packed witness, so a single Akita opening discharges the
/// whole set (Akita has no commitment homomorphism to RLC separate
/// commitments with).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProofPackingShape {
    pub ra_layout: JoltRaPolynomialLayout,
    pub log_t: usize,
    /// Shared one-hot chunk size: the address bits of each `Ra` column and
    /// the width of each unsigned-inc chunk (equal by the shared-final-point
    /// invariant).
    pub log_k_chunk: usize,
    /// `Some(word_vars)` when untrusted advice is committed.
    pub untrusted_advice_word_vars: Option<usize>,
}

/// The per-proof packed column list, in declaration order. Slot assignment
/// (sorting, prefix addresses, zero-fill) is `jolt-openings::PrefixPacking`'s
/// job; feed this list to `PrefixPacking::new` verbatim.
pub fn proof_packed_columns(
    shape: &ProofPackingShape,
) -> Result<Vec<(LatticeColumn, usize)>, LatticeGeometryError> {
    let chunking = UnsignedIncChunking::new(shape.log_k_chunk)?;
    let one_hot_vars = shape.log_k_chunk + shape.log_t;

    let mut columns = Vec::new();
    columns.extend(
        shape
            .ra_layout
            .committed_polynomials()
            .map(|polynomial| (LatticeColumn::from(polynomial), one_hot_vars)),
    );
    columns.extend((0..chunking.chunk_count()).map(|index| {
        (
            LatticeColumn::from(JoltCommittedPolynomial::UnsignedIncChunk(index)),
            one_hot_vars,
        )
    }));
    columns.push((
        LatticeColumn::from(JoltCommittedPolynomial::UnsignedIncMsb),
        shape.log_t,
    ));
    if let Some(word_vars) = shape.untrusted_advice_word_vars {
        columns.push((
            LatticeColumn::advice_bytes(JoltAdviceKind::Untrusted),
            byte_column_vars(WORD_BYTE_LIMBS, word_vars)?,
        ));
    }
    Ok(columns)
}

/// Shape of the preprocessing-time packed commitment (committed-program
/// mode): bytecode lane sub-columns, program image bytes, trusted advice
/// bytes. All public or verifier-trusted, so their one-hot structure is
/// checked offline rather than by in-protocol relations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrecommittedPackingShape {
    pub bytecode_chunks: usize,
    /// Log of the row count of one bytecode chunk.
    pub log_bytecode_rows: usize,
    /// Byte limbs of one immediate lane (the field's canonical byte width).
    pub imm_byte_width: usize,
    pub program_image_log_words: Option<usize>,
    pub trusted_advice_word_vars: Option<usize>,
}

pub fn precommitted_packed_columns(
    shape: &PrecommittedPackingShape,
) -> Result<Vec<(LatticeColumn, usize)>, LatticeGeometryError> {
    let log_rows = shape.log_bytecode_rows;
    let selector_vars = one_hot_column_vars(REGISTER_ADDRESS_BITS, 1, log_rows)?;
    let lookup_vars = ceil_log2(LookupTableKind::<XLEN>::COUNT.next_power_of_two()) + log_rows;

    let mut columns = Vec::new();
    for chunk in 0..shape.bytecode_chunks {
        columns.extend(BytecodeRegisterLane::ALL.into_iter().map(|lane| {
            (
                LatticeColumn::BytecodeRegisterSelector { chunk, lane },
                selector_vars,
            )
        }));
        columns.extend(
            (0..NUM_CIRCUIT_FLAGS)
                .map(|flag| (LatticeColumn::BytecodeCircuitFlag { chunk, flag }, log_rows)),
        );
        columns.extend((0..NUM_INSTRUCTION_FLAGS).map(|flag| {
            (
                LatticeColumn::BytecodeInstructionFlag { chunk, flag },
                log_rows,
            )
        }));
        columns.push((LatticeColumn::BytecodeLookupSelector { chunk }, lookup_vars));
        columns.push((LatticeColumn::BytecodeRafFlag { chunk }, log_rows));
        columns.push((
            LatticeColumn::BytecodeUnexpandedPcBytes { chunk },
            byte_column_vars(WORD_BYTE_LIMBS, log_rows)?,
        ));
        columns.push((
            LatticeColumn::BytecodeImmBytes { chunk },
            byte_column_vars(shape.imm_byte_width, log_rows)?,
        ));
    }
    if let Some(log_words) = shape.program_image_log_words {
        columns.push((
            LatticeColumn::ProgramImageBytes,
            byte_column_vars(WORD_BYTE_LIMBS, log_words)?,
        ));
    }
    if let Some(word_vars) = shape.trusted_advice_word_vars {
        columns.push((
            LatticeColumn::advice_bytes(JoltAdviceKind::Trusted),
            byte_column_vars(WORD_BYTE_LIMBS, word_vars)?,
        ));
    }
    Ok(columns)
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn proof_shape() -> ProofPackingShape {
        ProofPackingShape {
            ra_layout: JoltRaPolynomialLayout::new(2, 1, 1).unwrap(),
            log_t: 5,
            log_k_chunk: 8,
            untrusted_advice_word_vars: Some(4),
        }
    }

    #[test]
    fn proof_columns_cover_every_committed_lattice_polynomial() {
        let columns = proof_packed_columns(&proof_shape()).unwrap();

        // 4 Ra columns + 8 inc chunks + msb + untrusted advice bytes.
        assert_eq!(columns.len(), 4 + 8 + 1 + 1);
        assert_eq!(
            columns[0],
            (
                LatticeColumn::from(JoltCommittedPolynomial::InstructionRa(0)),
                13
            )
        );
        assert!(columns.contains(&(
            LatticeColumn::from(JoltCommittedPolynomial::UnsignedIncChunk(7)),
            13
        )));
        assert!(columns.contains(&(
            LatticeColumn::from(JoltCommittedPolynomial::UnsignedIncMsb),
            5
        )));
        assert!(columns.contains(&(
            LatticeColumn::from(JoltCommittedPolynomial::UntrustedAdviceBytes),
            8 + 3 + 4
        )));
    }

    #[test]
    fn proof_columns_reject_invalid_chunk_widths() {
        let shape = ProofPackingShape {
            log_k_chunk: 7,
            ..proof_shape()
        };
        assert_eq!(
            proof_packed_columns(&shape),
            Err(LatticeGeometryError::ChunkWidthMisaligned { chunk_width: 7 })
        );
    }

    #[test]
    fn precommitted_columns_cover_every_bytecode_lane() {
        let shape = PrecommittedPackingShape {
            bytecode_chunks: 2,
            log_bytecode_rows: 6,
            imm_byte_width: 16,
            program_image_log_words: Some(10),
            trusted_advice_word_vars: None,
        };
        let columns = precommitted_packed_columns(&shape).unwrap();

        let per_chunk = 3 + NUM_CIRCUIT_FLAGS + NUM_INSTRUCTION_FLAGS + 4;
        assert_eq!(columns.len(), 2 * per_chunk + 1);
        assert!(columns.contains(&(
            LatticeColumn::BytecodeRegisterSelector {
                chunk: 1,
                lane: BytecodeRegisterLane::Rd,
            },
            REGISTER_ADDRESS_BITS + 6,
        )));
        assert!(columns.contains(&(LatticeColumn::BytecodeCircuitFlag { chunk: 0, flag: 0 }, 6)));
        assert!(columns.contains(&(
            LatticeColumn::BytecodeUnexpandedPcBytes { chunk: 1 },
            8 + 3 + 6
        )));
        assert!(columns.contains(&(LatticeColumn::BytecodeImmBytes { chunk: 0 }, 8 + 4 + 6)));
        assert!(columns.contains(&(LatticeColumn::ProgramImageBytes, 8 + 3 + 10)));
    }

    #[test]
    fn packed_column_lists_are_deterministic() {
        assert_eq!(
            proof_packed_columns(&proof_shape()).unwrap(),
            proof_packed_columns(&proof_shape()).unwrap()
        );
    }
}
