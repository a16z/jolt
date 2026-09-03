//! What the table hands the stream: its committed columns by commitment
//! phase (packing order = [`col`] index order), the VK-committed public
//! columns, and its stage-A members.

use jolt_field::{Fr, Ring};

use super::columns::{Columns, CHUNK_COLUMNS, HELPER_COLUMNS, LIMBS};
use super::digit_link::LinkMember;
use super::layout::LOG_ROWS;
use super::lookup::{LookupColumns, PublicColumns, DIGIT_BITS};
use super::relation::{Col, RowSumcheck, SLOTS};
use super::schedule::Layout;

/// When a column is committed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Before any challenge: chunks, digit bits, multiplicities.
    One,
    /// After `ξ, α, β, γ, …`: operands, LogUp helpers, lookup helpers, fingerprints.
    Two,
    /// Fixed public columns committed in the verifying key.
    Vk,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnSpec {
    pub name: &'static str,
    pub phase: Phase,
    pub count: usize,
    /// First index in the claimed-column order.
    pub first: usize,
}

/// The column list in claimed-index order.
pub fn columns() -> Vec<ColumnSpec> {
    let spec = |name, phase, count, first| ColumnSpec {
        name,
        phase,
        count,
        first,
    };
    vec![
        spec("chunk", Phase::One, CHUNK_COLUMNS, Col::CHUNKS),
        spec("digit_bit", Phase::One, DIGIT_BITS, Col::DIGITS),
        spec("digit_value", Phase::One, 1, Col::D),
        spec("lookup_mult_pos", Phase::One, 1, Col::M_POS),
        spec("lookup_mult_neg", Phase::One, 1, Col::M_NEG),
        spec("range_mult", Phase::One, 1, Col::MULT),
        spec("operand_x", Phase::Two, SLOTS, Col::X),
        spec("operand_y", Phase::Two, SLOTS, Col::Y),
        spec("range_helper", Phase::Two, HELPER_COLUMNS, Col::HELPERS),
        spec("range_inverse", Phase::Two, 1, Col::INV),
        spec("lookup_read", Phase::Two, 1, Col::H),
        spec("lookup_table_pos", Phase::Two, 1, Col::G_POS),
        spec("lookup_table_neg", Phase::Two, 1, Col::G_NEG),
        spec("fingerprint_pos", Phase::Two, 1, Col::F_POS),
        spec("fingerprint_neg", Phase::Two, 1, Col::F_NEG),
        spec("pin", Phase::Vk, 1, Col::PIN),
        spec("pin_limb", Phase::Vk, LIMBS, Col::PIN_LIMBS),
        spec("free", Phase::Vk, 1, Col::FREE),
    ]
}

pub fn count(phase: Phase) -> usize {
    columns()
        .iter()
        .filter(|c| c.phase == phase)
        .map(|c| c.count)
        .sum()
}

/// A stage-A member: rounds, round-polynomial degree, offset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemberSpec {
    pub name: &'static str,
    pub rounds: usize,
    pub degree: usize,
    pub offset: usize,
}

pub fn members() -> [MemberSpec; 2] {
    [
        MemberSpec {
            name: "row",
            rounds: LOG_ROWS,
            degree: RowSumcheck::degree(),
            offset: 0,
        },
        MemberSpec {
            name: "digit_link",
            rounds: LOG_ROWS,
            degree: LinkMember::degree(),
            offset: 0,
        },
    ]
}

/// The claimed columns as `Col::CLAIMED` vectors in index order.
pub struct ClaimedColumns {
    pub columns: Vec<Vec<Fr>>,
}

impl ClaimedColumns {
    #[expect(
        clippy::too_many_arguments,
        reason = "one assembly of every column source"
    )]
    pub fn assemble(
        chunks: &Columns,
        public: &PublicColumns,
        operands: Vec<Vec<Fr>>,
        helpers: Vec<Vec<Fr>>,
        range_mult: Vec<Fr>,
        inverse_table: Vec<Fr>,
        lookup: LookupColumns,
        fingerprints: (Vec<Fr>, Vec<Fr>),
        pins: (Vec<Fr>, [Vec<Fr>; LIMBS]),
        free: Vec<Fr>,
    ) -> Self {
        let rows = chunks.rows();
        let mut columns: Vec<Vec<Fr>> = Vec::with_capacity(Col::CLAIMED);
        for j in 0..CHUNK_COLUMNS {
            columns.push((0..rows).map(|r| chunks.chunk(r, j)).collect());
        }
        for bits in &public.digits {
            columns.push(bits.iter().map(|b| Fr::from_u64(u64::from(*b))).collect());
        }
        columns.push(public.digit_values.clone());
        columns.push(lookup.m_pos);
        columns.push(lookup.m_neg);
        columns.push(range_mult);
        assert_eq!(operands.len(), 2 * SLOTS);
        columns.extend(operands);
        assert_eq!(helpers.len(), HELPER_COLUMNS);
        columns.extend(helpers);
        columns.push(inverse_table);
        columns.push(lookup.h);
        columns.push(lookup.g_pos);
        columns.push(lookup.g_neg);
        columns.push(fingerprints.0);
        columns.push(fingerprints.1);
        columns.push(pins.0);
        columns.extend(pins.1);
        columns.push(free);
        assert_eq!(columns.len(), Col::CLAIMED);
        Self { columns }
    }

    /// Phase-1 committed columns (indices `0..Col::PHASE1_END`).
    pub fn phase_one(&self) -> &[Vec<Fr>] {
        &self.columns[..Col::PHASE1_END]
    }

    pub fn phase_two(&self) -> &[Vec<Fr>] {
        &self.columns[Col::PHASE1_END..Col::PHASE2_END]
    }

    pub fn vk(&self) -> &[Vec<Fr>] {
        &self.columns[Col::COMMITTED..Col::CLAIMED]
    }
}

/// The VK `free` column: inputs and public constants.
pub fn free_column(layout: &Layout) -> Vec<Fr> {
    use jolt_field::Zero;
    let mut free = vec![Fr::zero(); 1usize << LOG_ROWS];
    for row in layout.program.free_rows() {
        free[row] = Fr::from_u64(1);
    }
    free
}

/// The VK pin columns of a layout: the pin indicator and the pinned limbs.
pub fn pin_columns(layout: &Layout) -> (Vec<Fr>, [Vec<Fr>; LIMBS]) {
    use jolt_field::Zero;
    let rows = 1usize << LOG_ROWS;
    let mut pin = vec![Fr::zero(); rows];
    let mut limbs: [Vec<Fr>; LIMBS] = std::array::from_fn(|_| vec![Fr::zero(); rows]);
    for (row, value) in layout.program.pinned_rows() {
        pin[row] = Fr::from_u64(1);
        for (column, limb) in limbs.iter_mut().zip(super::columns::fq_limbs(&value)) {
            column[row] = limb;
        }
    }
    (pin, limbs)
}
