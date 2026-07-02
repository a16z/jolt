use jolt_field::{Field, FromPrimitiveInt};
use jolt_lookup_tables::{LookupTableKind, XLEN};
use jolt_poly::EqPolynomial;
use jolt_riscv::{NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS};
use serde::{Deserialize, Serialize};

use super::super::geometry::claim_reductions::bytecode::{
    committed_lane_vars, BYTECODE_LANE_LAYOUT,
};
use super::super::geometry::dimensions::REGISTER_ADDRESS_BITS;
use super::super::geometry::error::JoltFormulaPointError;
use super::super::JoltAdviceKind;
use super::geometry::{ceil_log2, WORD_BYTE_LIMBS};
use super::ids::{BytecodeRegisterLane, LatticeColumn};

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

/// A decomposed logical polynomial and the view reconstructing it. Discharge
/// targets for [`LatticeFinalOpening::Decoded`](super::discharge::LatticeFinalOpening).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LatticeView {
    /// A word-valued advice polynomial from its byte one-hot column.
    AdviceWord { kind: JoltAdviceKind },
    /// A committed `BytecodeChunk(chunk)` lane value from the chunk's lane
    /// sub-columns.
    BytecodeChunkLanes { chunk: usize },
    /// A program-image word from the byte one-hot column.
    ProgramImageWord,
}

/// Little-endian byte decode: `Σ_{limb, symbol} 256^limb · symbol ·
/// Column(symbol ‖ limb ‖ row)`, each term additionally scaled by `scale`.
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

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_field::Fr;

    #[test]
    fn byte_decode_terms_weight_little_endian_places() {
        let terms = byte_decode_terms::<Fr>(LatticeColumn::ProgramImageBytes, 8);

        // 255 nonzero symbols per limb; the zero symbol contributes nothing.
        assert_eq!(terms.len(), 8 * 255);
        let term = &terms[6]; // limb 0, symbol 7
        assert_eq!(term.cell, (7 << 3), "cell packs (symbol ‖ limb) msb-first");
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

        // The lane weights sum to 1 over the full eq table; spot-check the raf
        // flag term carries its lane's weight.
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
