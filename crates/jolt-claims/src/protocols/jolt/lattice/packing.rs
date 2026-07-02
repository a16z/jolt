//! The packed-witness semantics in one place: the column ids, the canonical
//! packing registration, the decode views over decomposed columns, and the
//! final-opening discharge map.
//!
//! `jolt-openings::PrefixPacking` is the single source of truth for slot
//! assignment — [`proof_packing`]/[`precommitted_packing`] register the
//! canonical column orderings with it and hand back the packing object
//! itself; this module never does its own offset arithmetic.

use jolt_field::{Field, FromPrimitiveInt};
use jolt_lookup_tables::{LookupTableKind, XLEN};
use jolt_openings::PrefixPacking;
use jolt_poly::EqPolynomial;
use jolt_riscv::{NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS};
use serde::{Deserialize, Serialize};

use super::super::geometry::claim_reductions::bytecode::{
    committed_lane_vars, BYTECODE_LANE_LAYOUT,
};
use super::super::geometry::dimensions::REGISTER_ADDRESS_BITS;
use super::super::geometry::error::JoltFormulaPointError;
use super::super::geometry::ra::JoltRaPolynomialLayout;
use super::super::{JoltAdviceKind, JoltCommittedPolynomial, JoltOpeningId, JoltRelationId};
use super::geometry::{
    byte_column_vars, ceil_log2, one_hot_column_vars, LatticeGeometryError, UnsignedIncChunking,
    WORD_BYTE_LIMBS,
};

/// The register-selector lanes of a bytecode row, in committed lane order.
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BytecodeRegisterLane {
    Rs1,
    Rs2,
    Rd,
}

impl BytecodeRegisterLane {
    pub const ALL: [Self; 3] = [Self::Rs1, Self::Rs2, Self::Rd];
}

/// One column of a packed lattice witness — the `Id` fed to
/// `jolt-openings::PrefixPacking`.
///
/// A column with in-protocol openings is a (lattice-mode) committed
/// polynomial and wraps its [`JoltCommittedPolynomial`] id; a precommitted
/// sub-column is only ever reached through a decode view and gets its own
/// variant with no opening-id identity.
///
/// WARNING: `Ord` is protocol data — `PrefixPacking` assigns slots by
/// `(arity, Id)` order, so reordering variants here (or in
/// `JoltCommittedPolynomial`) silently changes the packed witness layout.
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LatticeColumn {
    Committed(JoltCommittedPolynomial),
    BytecodeRegisterSelector {
        chunk: usize,
        lane: BytecodeRegisterLane,
    },
    BytecodeCircuitFlag {
        chunk: usize,
        flag: usize,
    },
    BytecodeInstructionFlag {
        chunk: usize,
        flag: usize,
    },
    BytecodeLookupSelector {
        chunk: usize,
    },
    BytecodeRafFlag {
        chunk: usize,
    },
    BytecodeUnexpandedPcBytes {
        chunk: usize,
    },
    BytecodeImmBytes {
        chunk: usize,
    },
    ProgramImageBytes,
}

impl LatticeColumn {
    pub fn advice_bytes(kind: JoltAdviceKind) -> Self {
        Self::Committed(advice_bytes_polynomial(kind))
    }
}

impl From<JoltCommittedPolynomial> for LatticeColumn {
    fn from(polynomial: JoltCommittedPolynomial) -> Self {
        Self::Committed(polynomial)
    }
}

pub fn advice_bytes_polynomial(kind: JoltAdviceKind) -> JoltCommittedPolynomial {
    match kind {
        JoltAdviceKind::Trusted => JoltCommittedPolynomial::TrustedAdviceBytes,
        JoltAdviceKind::Untrusted => JoltCommittedPolynomial::UntrustedAdviceBytes,
    }
}

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

/// Registers the canonical per-proof column ordering and returns the packing.
pub fn proof_packing(
    shape: &ProofPackingShape,
) -> Result<PrefixPacking<LatticeColumn>, LatticeGeometryError> {
    Ok(PrefixPacking::new(proof_columns(shape)?)?)
}

/// Registers the canonical precommitted column ordering and returns the
/// packing.
pub fn precommitted_packing(
    shape: &PrecommittedPackingShape,
) -> Result<PrefixPacking<LatticeColumn>, LatticeGeometryError> {
    Ok(PrefixPacking::new(precommitted_columns(shape)?)?)
}

fn proof_columns(
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

fn precommitted_columns(
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

/// One weighted cell of a decode view: the logical value being decoded equals
/// `Σ_terms weight · Column(cell ‖ row_point)`, where `cell` indexes the
/// column's `(symbol ‖ limb)` prefix (msb-first, limbs rounded to a power of
/// two).
///
/// A weighted sum with non-`eq` weights is not a single packed evaluation
/// claim, so views are discharged by a reduction over the cell variables (or
/// folded into an existing reduction) on the verifier side; the term list is
/// the semantics both sides must agree on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodeTerm<F> {
    pub column: LatticeColumn,
    pub cell: usize,
    pub weight: F,
}

impl<F> DecodeTerm<F> {
    pub fn new(column: LatticeColumn, cell: usize, weight: F) -> Self {
        Self {
            column,
            cell,
            weight,
        }
    }
}

/// Little-endian byte decode: `Σ_{limb, symbol} 256^limb · symbol ·
/// Column(symbol ‖ limb ‖ row)`, each term additionally scaled by `scale`.
/// The zero symbol carries weight zero and is omitted.
pub fn scaled_byte_decode_terms<F: Field + FromPrimitiveInt>(
    column: LatticeColumn,
    limbs: usize,
    scale: F,
) -> Vec<DecodeTerm<F>> {
    let limb_bits = ceil_log2(limbs.next_power_of_two());
    let mut terms = Vec::with_capacity(limbs * 256);
    let mut place = scale;
    for limb in 0..limbs {
        for symbol in 1..256usize {
            terms.push(DecodeTerm::new(
                column,
                (symbol << limb_bits) | limb,
                place * F::from_u64(symbol as u64),
            ));
        }
        place *= F::from_u64(256);
    }
    terms
}

pub fn byte_decode_terms<F: Field + FromPrimitiveInt>(
    column: LatticeColumn,
    limbs: usize,
) -> Vec<DecodeTerm<F>> {
    scaled_byte_decode_terms(column, limbs, F::one())
}

/// `Σ_symbol weight[symbol] · Column(symbol ‖ row)` for a single-limb one-hot
/// column.
pub fn weighted_symbol_terms<F>(
    column: LatticeColumn,
    weights: impl IntoIterator<Item = F>,
) -> Vec<DecodeTerm<F>> {
    weights
        .into_iter()
        .enumerate()
        .map(|(symbol, weight)| DecodeTerm::new(column, symbol, weight))
        .collect()
}

/// The advice word view: `advice(word) = Σ_limb 256^limb · byte(word, limb)`.
pub fn advice_word_decode_terms<F: Field + FromPrimitiveInt>(
    kind: JoltAdviceKind,
) -> Vec<DecodeTerm<F>> {
    byte_decode_terms(LatticeColumn::advice_bytes(kind), WORD_BYTE_LIMBS)
}

pub fn program_image_word_decode_terms<F: Field + FromPrimitiveInt>() -> Vec<DecodeTerm<F>> {
    byte_decode_terms(LatticeColumn::ProgramImageBytes, WORD_BYTE_LIMBS)
}

/// The lane decode of one committed bytecode chunk: reconstructs
/// `BytecodeChunk(chunk)(lane_point ‖ row_point)` as a weighted sum over the
/// chunk's sub-columns, weighting each lane by the `eq` evaluation of
/// `lane_point` at its lane index.
pub fn bytecode_chunk_decode_terms<F: Field + FromPrimitiveInt>(
    chunk: usize,
    lane_point: &[F],
    imm_byte_width: usize,
) -> Result<Vec<DecodeTerm<F>>, JoltFormulaPointError> {
    let lane_vars = committed_lane_vars();
    if lane_point.len() != lane_vars {
        return Err(JoltFormulaPointError::OpeningPointLengthMismatch {
            expected: lane_vars,
            got: lane_point.len(),
        });
    }
    let lane_weights = EqPolynomial::<F>::evals(lane_point, None);
    let layout = BYTECODE_LANE_LAYOUT;
    let register_count = 1usize << REGISTER_ADDRESS_BITS;

    let mut terms = Vec::new();
    for (lane, start) in [
        (BytecodeRegisterLane::Rs1, layout.rs1_start),
        (BytecodeRegisterLane::Rs2, layout.rs2_start),
        (BytecodeRegisterLane::Rd, layout.rd_start),
    ] {
        terms.extend(weighted_symbol_terms(
            LatticeColumn::BytecodeRegisterSelector { chunk, lane },
            lane_weights[start..start + register_count].iter().copied(),
        ));
    }
    terms.extend(scaled_byte_decode_terms(
        LatticeColumn::BytecodeUnexpandedPcBytes { chunk },
        WORD_BYTE_LIMBS,
        lane_weights[layout.unexp_pc_idx],
    ));
    terms.extend(scaled_byte_decode_terms(
        LatticeColumn::BytecodeImmBytes { chunk },
        imm_byte_width,
        lane_weights[layout.imm_idx],
    ));
    for flag in 0..NUM_CIRCUIT_FLAGS {
        terms.push(DecodeTerm::new(
            LatticeColumn::BytecodeCircuitFlag { chunk, flag },
            0,
            lane_weights[layout.circuit_start + flag],
        ));
    }
    for flag in 0..NUM_INSTRUCTION_FLAGS {
        terms.push(DecodeTerm::new(
            LatticeColumn::BytecodeInstructionFlag { chunk, flag },
            0,
            lane_weights[layout.instr_start + flag],
        ));
    }
    terms.extend(weighted_symbol_terms(
        LatticeColumn::BytecodeLookupSelector { chunk },
        lane_weights[layout.lookup_start..layout.lookup_start + LookupTableKind::<XLEN>::COUNT]
            .iter()
            .copied(),
    ));
    terms.push(DecodeTerm::new(
        LatticeColumn::BytecodeRafFlag { chunk },
        0,
        lane_weights[layout.raf_flag_idx],
    ));

    Ok(terms)
}

/// How one committed polynomial's final claim is discharged in lattice mode —
/// the replacement for the base stage-8 RLC batch (which needs the commitment
/// homomorphism Akita lacks).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LatticeFinalOpening {
    /// One `PrefixPackedClaim` on the packed witness. `leaf` names the
    /// relation output the claim's value comes from; `None` means the claim
    /// is produced by the view-discharge reduction rather than a relation
    /// (trusted-advice bytes and the precommitted sub-columns).
    Packed {
        column: LatticeColumn,
        leaf: Option<JoltOpeningId>,
    },
    /// A word-valued advice polynomial, decoded through
    /// [`advice_word_decode_terms`].
    AdviceWordView { kind: JoltAdviceKind },
    /// A committed `BytecodeChunk(chunk)`, decoded through
    /// [`bytecode_chunk_decode_terms`].
    BytecodeChunkLanesView { chunk: usize },
    /// A program-image word, decoded through
    /// [`program_image_word_decode_terms`].
    ProgramImageWordView,
    /// Never PCS-opened: the polynomial's consumer claims leave the base PIOP
    /// through lattice relations (the `IncVirtualization` chain).
    Virtualized,
}

/// The discharge for each committed polynomial's final opening. Total over
/// `JoltCommittedPolynomial` so a new committed polynomial cannot land in
/// lattice mode without a discharge decision.
pub fn final_opening(polynomial: JoltCommittedPolynomial) -> LatticeFinalOpening {
    match polynomial {
        JoltCommittedPolynomial::RdInc | JoltCommittedPolynomial::RamInc => {
            LatticeFinalOpening::Virtualized
        }
        JoltCommittedPolynomial::InstructionRa(_)
        | JoltCommittedPolynomial::BytecodeRa(_)
        | JoltCommittedPolynomial::RamRa(_)
        | JoltCommittedPolynomial::UnsignedIncChunk(_)
        | JoltCommittedPolynomial::UnsignedIncMsb
        | JoltCommittedPolynomial::TrustedAdviceBytes
        | JoltCommittedPolynomial::UntrustedAdviceBytes => LatticeFinalOpening::Packed {
            column: LatticeColumn::from(polynomial),
            leaf: packed_column_leaf(LatticeColumn::from(polynomial)),
        },
        JoltCommittedPolynomial::TrustedAdvice => LatticeFinalOpening::AdviceWordView {
            kind: JoltAdviceKind::Trusted,
        },
        JoltCommittedPolynomial::UntrustedAdvice => LatticeFinalOpening::AdviceWordView {
            kind: JoltAdviceKind::Untrusted,
        },
        JoltCommittedPolynomial::BytecodeChunk(chunk) => {
            LatticeFinalOpening::BytecodeChunkLanesView { chunk }
        }
        JoltCommittedPolynomial::ProgramImageInit => LatticeFinalOpening::ProgramImageWordView,
    }
}

/// The relation output supplying a packed column's claim value, or `None`
/// when the view-discharge reduction produces it (view-only sub-columns and
/// trusted-advice bytes, which have no in-protocol validity relation).
pub fn packed_column_leaf(column: LatticeColumn) -> Option<JoltOpeningId> {
    let LatticeColumn::Committed(polynomial) = column else {
        return None;
    };
    let relation = match polynomial {
        JoltCommittedPolynomial::InstructionRa(_)
        | JoltCommittedPolynomial::BytecodeRa(_)
        | JoltCommittedPolynomial::RamRa(_) => JoltRelationId::HammingWeightClaimReduction,
        JoltCommittedPolynomial::UnsignedIncChunk(_) => {
            JoltRelationId::UnsignedIncChunkReconstruction
        }
        JoltCommittedPolynomial::UnsignedIncMsb => JoltRelationId::Booleanity,
        JoltCommittedPolynomial::UntrustedAdviceBytes => JoltRelationId::AdviceBytesValidity,
        _ => return None,
    };
    Some(JoltOpeningId::committed(polynomial, relation))
}

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::super::super::geometry::committed_openings;
    use super::*;
    use jolt_field::Fr;

    fn proof_shape() -> ProofPackingShape {
        ProofPackingShape {
            ra_layout: JoltRaPolynomialLayout::new(2, 1, 1).unwrap(),
            log_t: 5,
            log_k_chunk: 8,
            untrusted_advice_word_vars: Some(4),
        }
    }

    #[test]
    fn proof_packing_covers_every_committed_lattice_polynomial() {
        let packing = proof_packing(&proof_shape()).unwrap();

        // 4 Ra columns + 8 inc chunks + msb + untrusted advice bytes.
        assert_eq!(packing.iter().count(), 4 + 8 + 1 + 1);
        assert_eq!(
            packing[&LatticeColumn::from(JoltCommittedPolynomial::InstructionRa(0))].num_vars,
            13
        );
        assert_eq!(
            packing[&LatticeColumn::from(JoltCommittedPolynomial::UnsignedIncChunk(7))].num_vars,
            13
        );
        assert_eq!(
            packing[&LatticeColumn::from(JoltCommittedPolynomial::UnsignedIncMsb)].num_vars,
            5
        );
        assert_eq!(
            packing[&LatticeColumn::from(JoltCommittedPolynomial::UntrustedAdviceBytes)].num_vars,
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
        let shape = PrecommittedPackingShape {
            bytecode_chunks: 2,
            log_bytecode_rows: 6,
            imm_byte_width: 16,
            program_image_log_words: Some(10),
            trusted_advice_word_vars: None,
        };
        let packing = precommitted_packing(&shape).unwrap();

        let per_chunk = 3 + NUM_CIRCUIT_FLAGS + NUM_INSTRUCTION_FLAGS + 4;
        assert_eq!(packing.iter().count(), 2 * per_chunk + 1);
        assert_eq!(
            packing[&LatticeColumn::BytecodeRegisterSelector {
                chunk: 1,
                lane: BytecodeRegisterLane::Rd,
            }]
                .num_vars,
            REGISTER_ADDRESS_BITS + 6,
        );
        assert_eq!(
            packing[&LatticeColumn::BytecodeCircuitFlag { chunk: 0, flag: 0 }].num_vars,
            6
        );
        assert_eq!(
            packing[&LatticeColumn::BytecodeUnexpandedPcBytes { chunk: 1 }].num_vars,
            8 + 3 + 6
        );
        assert_eq!(
            packing[&LatticeColumn::BytecodeImmBytes { chunk: 0 }].num_vars,
            8 + 4 + 6
        );
        assert_eq!(
            packing[&LatticeColumn::ProgramImageBytes].num_vars,
            8 + 3 + 10
        );
    }

    #[test]
    fn packing_registration_is_deterministic() {
        assert_eq!(
            proof_packing(&proof_shape()).unwrap(),
            proof_packing(&proof_shape()).unwrap()
        );
    }

    #[test]
    fn incs_are_virtualized_and_never_opened() {
        assert_eq!(
            final_opening(JoltCommittedPolynomial::RdInc),
            LatticeFinalOpening::Virtualized
        );
        assert_eq!(
            final_opening(JoltCommittedPolynomial::RamInc),
            LatticeFinalOpening::Virtualized
        );
    }

    #[test]
    fn decomposed_polynomials_discharge_through_views() {
        assert_eq!(
            final_opening(JoltCommittedPolynomial::BytecodeChunk(3)),
            LatticeFinalOpening::BytecodeChunkLanesView { chunk: 3 }
        );
        assert_eq!(
            final_opening(JoltCommittedPolynomial::UntrustedAdvice),
            LatticeFinalOpening::AdviceWordView {
                kind: JoltAdviceKind::Untrusted
            }
        );
        assert_eq!(
            final_opening(JoltCommittedPolynomial::ProgramImageInit),
            LatticeFinalOpening::ProgramImageWordView
        );
    }

    #[test]
    fn every_proof_packed_column_has_a_claim_source() {
        let packing = proof_packing(&proof_shape()).unwrap();
        for (column, _) in &packing {
            let leaf = packed_column_leaf(*column).unwrap();
            let LatticeColumn::Committed(polynomial) = column else {
                panic!("proof packed columns are committed polynomials");
            };
            assert_eq!(
                final_opening(*polynomial),
                LatticeFinalOpening::Packed {
                    column: *column,
                    leaf: Some(leaf)
                }
            );
        }
    }

    /// The base-mode `final_opening_relation` gained lattice arms when the
    /// committed enum grew; they must agree with the discharge leaves
    /// (trusted-advice bytes intentionally diverge: their claim is
    /// view-produced, so the base map's arm is only a scheduling hint).
    #[test]
    fn packed_leaves_agree_with_base_final_opening_relation() {
        let packing = proof_packing(&proof_shape()).unwrap();
        for (column, _) in &packing {
            let LatticeColumn::Committed(polynomial) = column else {
                panic!("proof packed columns are committed polynomials");
            };
            assert_eq!(
                packed_column_leaf(*column).unwrap(),
                JoltOpeningId::committed(
                    *polynomial,
                    committed_openings::final_opening_relation(*polynomial)
                )
            );
        }
        assert_eq!(
            packed_column_leaf(LatticeColumn::Committed(
                JoltCommittedPolynomial::TrustedAdviceBytes
            )),
            None
        );
    }

    #[test]
    fn view_only_columns_have_no_relation_leaf() {
        assert_eq!(packed_column_leaf(LatticeColumn::ProgramImageBytes), None);
        assert_eq!(
            final_opening(JoltCommittedPolynomial::TrustedAdviceBytes),
            LatticeFinalOpening::Packed {
                column: LatticeColumn::Committed(JoltCommittedPolynomial::TrustedAdviceBytes),
                leaf: None,
            }
        );
    }

    #[test]
    fn byte_decode_terms_weight_little_endian_places() {
        let terms = byte_decode_terms::<Fr>(LatticeColumn::ProgramImageBytes, 8);

        // 255 nonzero symbols per limb; the zero symbol contributes nothing.
        assert_eq!(terms.len(), 8 * 255);
        let term = &terms[6]; // limb 0, symbol 7
        assert_eq!(term.cell, 7 << 3, "cell packs (symbol ‖ limb) msb-first");
        assert_eq!(term.weight, Fr::from_u64(7));

        let second_limb = &terms[255 + 2]; // limb 1, symbol 3
        assert_eq!(second_limb.cell, (3 << 3) | 1);
        assert_eq!(second_limb.weight, Fr::from_u64(3 * 256));
    }

    #[test]
    fn byte_decode_reconstructs_words() {
        let word: u64 = 0x0123_4567_89ab_cdef;
        let terms = advice_word_decode_terms::<Fr>(JoltAdviceKind::Untrusted);

        // Evaluate the view against the one-hot indicator of `word`'s bytes.
        let value = terms
            .iter()
            .filter(|term| {
                let limb = term.cell & 0b111;
                let symbol = term.cell >> 3;
                (word >> (8 * limb)) & 0xff == symbol as u64
            })
            .fold(Fr::from_u64(0), |acc, term| acc + term.weight);
        assert_eq!(value, Fr::from_u64(word));
    }

    #[test]
    fn bytecode_chunk_decode_covers_every_lane_exactly_once() {
        let lane_point = (0..committed_lane_vars())
            .map(|i| Fr::from_u64(3 + i as u64))
            .collect::<Vec<_>>();
        let terms = bytecode_chunk_decode_terms::<Fr>(1, &lane_point, 16).unwrap();

        let register_count = 1usize << REGISTER_ADDRESS_BITS;
        let expected = 3 * register_count
            + 255 * WORD_BYTE_LIMBS
            + 255 * 16
            + NUM_CIRCUIT_FLAGS
            + NUM_INSTRUCTION_FLAGS
            + LookupTableKind::<XLEN>::COUNT
            + 1;
        assert_eq!(terms.len(), expected);
        assert!(terms
            .iter()
            .all(|term| !matches!(term.column, LatticeColumn::Committed(_))));

        let lane_weights = EqPolynomial::<Fr>::evals(&lane_point, None);
        let raf = terms
            .iter()
            .find(|term| matches!(term.column, LatticeColumn::BytecodeRafFlag { chunk: 1 }))
            .unwrap();
        assert_eq!(raf.weight, lane_weights[BYTECODE_LANE_LAYOUT.raf_flag_idx]);
    }

    #[test]
    fn bytecode_chunk_decode_rejects_wrong_lane_arity() {
        let lane_point = vec![Fr::from_u64(1); 3];
        assert_eq!(
            bytecode_chunk_decode_terms::<Fr>(0, &lane_point, 16),
            Err(JoltFormulaPointError::OpeningPointLengthMismatch {
                expected: committed_lane_vars(),
                got: 3,
            })
        );
    }
}
