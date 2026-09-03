//! What the table hands the stream: its committed columns by commitment
//! phase (packing order = [`col`] index order), the VK-committed public
//! columns, and its stage-A members.

use std::ops::Range;

use jolt_field::{Fr, Ring};

use super::columns::{Columns, CHUNK_COLUMNS, HELPER_COLUMNS, LIMBS};
use super::digit_link::LinkMember;
use super::layout::LOG_ROWS;
use super::lookup::{LookupColumns, PublicColumns, DIGIT_BITS};
use super::relation::{Col, RowSumcheck, SLOTS};
use super::schedule::Layout;

/// When a column is committed. Every prover-chosen value is committed
/// before the challenge that fingerprints or batches it; only helper columns
/// (functions of earlier challenges and committed values) follow a challenge.
/// The table has no phase-1a columns: everything depends on the offset
/// challenge `θ` (drawn after T1's phase 1a), which the digits encode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// After `θ`, before `ξ, α`: chunks, digit bits and values, multiplicities, sign flags.
    OneB,
    /// After `ξ, α`, before `fp_root`: operands `X, Y`, range helpers, the inverse table.
    TwoA,
    /// After `fp_root`, before `β, fp_combine, copy_root`: the table fingerprints.
    TwoB,
    /// After `β, fp_combine`, before `τ, γ, λ, λ_lookup, constancy_root`: `h, g±`.
    TwoC,
    /// Fixed public columns committed in the verifying key.
    Vk,
}

/// A committed phase and the challenges drawn right before it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhaseSpec {
    pub phase: Phase,
    pub challenges_before: &'static [&'static str],
    /// Index range of the phase's columns in the claimed-column order.
    pub columns: Range<usize>,
}

/// The committed phases in transcript order; the stage-A challenges
/// (`tau, gamma, lambda, lambda_lookup, constancy_root`) follow the last one.
pub fn phases() -> [PhaseSpec; 4] {
    [
        PhaseSpec {
            phase: Phase::OneB,
            challenges_before: &["theta"],
            columns: 0..Col::PHASE_1B_END,
        },
        PhaseSpec {
            phase: Phase::TwoA,
            challenges_before: &["xi", "alpha"],
            columns: Col::PHASE_1B_END..Col::PHASE_2A_END,
        },
        PhaseSpec {
            phase: Phase::TwoB,
            challenges_before: &["fp_root"],
            columns: Col::PHASE_2A_END..Col::PHASE_2B_END,
        },
        PhaseSpec {
            phase: Phase::TwoC,
            challenges_before: &["beta", "fp_combine", "copy_root"],
            columns: Col::PHASE_2B_END..Col::PHASE_2C_END,
        },
    ]
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
        spec("chunk", Phase::OneB, CHUNK_COLUMNS, Col::CHUNKS),
        spec("digit_bit", Phase::OneB, DIGIT_BITS, Col::DIGITS),
        spec("digit_value", Phase::OneB, 1, Col::D),
        spec("lookup_mult_pos", Phase::OneB, 1, Col::M_POS),
        spec("lookup_mult_neg", Phase::OneB, 1, Col::M_NEG),
        spec("range_mult", Phase::OneB, 1, Col::MULT),
        spec("sign_flag", Phase::OneB, 1, Col::FLAG),
        spec("operand_x", Phase::TwoA, SLOTS, Col::X),
        spec("operand_y", Phase::TwoA, SLOTS, Col::Y),
        spec("range_helper", Phase::TwoA, HELPER_COLUMNS, Col::HELPERS),
        spec("range_inverse", Phase::TwoA, 1, Col::INV),
        spec("fingerprint_pos", Phase::TwoB, 1, Col::F_POS),
        spec("fingerprint_neg", Phase::TwoB, 1, Col::F_NEG),
        spec("lookup_read", Phase::TwoC, 1, Col::H),
        spec("lookup_table_pos", Phase::TwoC, 1, Col::G_POS),
        spec("lookup_table_neg", Phase::TwoC, 1, Col::G_NEG),
        spec("pin", Phase::Vk, 1, Col::PIN),
        spec("pin_limb", Phase::Vk, LIMBS, Col::PIN_LIMBS),
        spec("free", Phase::Vk, 1, Col::FREE),
        spec("exact", Phase::Vk, 1, Col::EXACT),
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
        exact: Vec<Fr>,
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
        columns.push(
            chunks
                .flags
                .iter()
                .map(|f| Fr::from_u64(u64::from(*f)))
                .collect(),
        );
        assert_eq!(operands.len(), 2 * SLOTS);
        columns.extend(operands);
        assert_eq!(helpers.len(), HELPER_COLUMNS);
        columns.extend(helpers);
        columns.push(inverse_table);
        columns.push(fingerprints.0);
        columns.push(fingerprints.1);
        columns.push(lookup.h);
        columns.push(lookup.g_pos);
        columns.push(lookup.g_neg);
        columns.push(pins.0);
        columns.extend(pins.1);
        columns.push(free);
        columns.push(exact);
        assert_eq!(columns.len(), Col::CLAIMED);
        Self { columns }
    }

    /// The columns of one committed phase.
    pub fn phase(&self, spec: &PhaseSpec) -> &[Vec<Fr>] {
        &self.columns[spec.columns.clone()]
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

/// The VK `exact` column: rows whose limb identity has no quotient.
pub fn exact_column(layout: &Layout) -> Vec<Fr> {
    use jolt_field::Zero;
    let mut exact = vec![Fr::zero(); 1usize << LOG_ROWS];
    for row in layout.program.exact_rows() {
        exact[row] = Fr::from_u64(1);
    }
    exact
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
