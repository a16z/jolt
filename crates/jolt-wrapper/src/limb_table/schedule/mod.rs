//! The fixed layout of the deferred check over `2^18` rows (`2^14` cells of
//! 16 rows): GT operations occupy rows `0..12` of a cell, four-row G1
//! operations ride in rows `12..16` of GT cells, eight-row G2 operations
//! pack two per cell. Every region is a bit-field box of the row index, so
//! each operand relation is a kernel ([`super::layout::Kernel`]) the
//! verifier evaluates in `O(bits)`; the digit-selected table reads are proven
//! by the operand lookup ([`super::lookup`]); the few irregular
//! final-exponentiation glue rows use explicit `Table` edge lists (`≤ 64`
//! edges per family).

use ark_bn254::{Config as Bn254Config, Fq, Fq12, Fq2, G2Affine};
use ark_ec::bn::{BnConfig, G2Prepared};
use ark_ec::short_weierstrass::SWCurveConfig;
use ark_ff::{AdditiveGroup, Field};
use std::collections::HashMap;
use std::ops::Range;

use crate::relation::DoryScalar;

use super::digits::{digits, WINDOWS};
use super::dory::{DorySetupInputs, FlattenedCheck, GtBase, InputElement, Wire, WireValues};
use super::layout::{Bits, Factor, Piece, Rel, LOG_ROWS};
use super::ops::{
    ell, g2_psi, gt_difference_pins, gt_frobenius, gt_inverse_pin, gt_inverse_witness, gt_mul,
    gt_norm_one, miller_add_step, miller_double_step, GtOperand, ADD_LINE, CONST_LINE, DOUBLE_LINE,
};
use super::program::{Program, RowId};
use super::relation::FP_SLOTS_GT;
use super::template::{
    fingerprint_maps, DigitRule, ElemRel, ElemWiring, Family, Template, ONE_ROW,
};
use super::tower::{fq12_coords, frobenius_form};
use super::wiring::{FingerprintGroup, ReadKind, TableRead};

mod ec;
mod final_exp;
mod gt;
mod miller;

/// Row-count profile of one opening: `σ` reduction rounds, `n` commitments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Profile {
    pub sigma: usize,
    pub n: usize,
}

impl Profile {
    pub const fn gt_bases(self) -> usize {
        9 * self.sigma + self.n + 4
    }
}

/// Cell indices (`row = 16·cell + c`) of every region; each box is aligned
/// so its index fields are bit-fields of the row index.
/// Cell (and half-cell) bases of the fixed layout.
pub struct Cells;

impl Cells {
    /// `64·k + w`: GT Straus mults (`k < B`) and per-window squarings (`k ∈ B..B+4`).
    pub const GT_ONLINE: u32 = 0;
    /// Public constants, one row each (`1024` rows).
    pub const CONSTANTS: Range<u32> = 9536..9600;
    pub const FINAL: u32 = 9600;
    /// Structured constant cells (points in output layout).
    pub const POINT_CONSTANTS: u32 = 9601;
    /// `9728 + k`: the norm-one check of the GT input base `k`.
    pub const GT_NORM: u32 = 9728;
    /// Final-exponentiation glue and the `ψ` points (explicit edges).
    pub const GLUE: u32 = 9888;
    pub const GLUE_END: u32 = 10048;
    /// The G2 offset chain `θ·G2`: `HI2 = 4`, `k` slots `58..64` (cells `10048..10240`).
    pub const G2_OFFSET_HI: u32 = 4;
    pub const G2_OFFSET_K: u32 = 58;
    /// Half-cell index `row >> 3 = 20480 + 64·k + w`.
    pub const G2_ONLINE_HALF: u32 = 20480;
    /// `11776 + 8·t + s`: `s = 0` squaring, `s = 1 + p` doubling-line `ell`.
    pub const MILLER_DBL_GT: u32 = 11776;
    /// `12288 + 8·k + e`: `X_k^{e+1}` (`e = 0` is the input/setup element).
    pub const GT_TABLE: u32 = 12288;
    /// `13312 + 16·b + j` (rows 12–15): `(j−8)·P_b + Z0`.
    pub const G1_TABLE: u32 = 13312;
    /// `15232 + 4·a + p`: addition-line `ell`.
    pub const MILLER_ADD_GT: u32 = 15232;
    /// `13248 + b` (rows 12–15): G1 base points.
    pub const G1_INPUT: u32 = 13248;
    /// `14080 + 4·t + 2·p + i`: doubling steps of pairs `p < 2` (32 rows).
    pub const MILLER_DBL_LINES: u32 = 14080;
    /// Half-cell index `28672 + 16·b + j`.
    pub const G2_TABLE_HALF: u32 = 28672;
    /// `14720 + 4·a + 2·p + i`: addition steps of pairs `p < 2`.
    pub const MILLER_ADD_LINES: u32 = 14720;
    /// Half-cell index `29696 + b`.
    pub const G2_INPUT_HALF: u32 = 29696;
    /// `14976 + 2·t + (p − 2)`: public doubling lines of pairs `p ≥ 2`.
    pub const CONST_LINES_DBL: u32 = 14976;
    /// `15104 + 2·a + (p − 2)`.
    pub const CONST_LINES_ADD: u32 = 15104;
    /// `15360 + 128·chain + 2·step + slot`: `f^{-x}` chains.
    pub const FE_CHAINS: u32 = 15360;
    /// `15872 + 256·chain + local`: the G2 subgroup checks of the two
    /// proof-derived Miller `Q` inputs (`2^i·P` at half cells, guarded adds
    /// and the `ψ` tail at cells).
    pub const PSI_CHAIN: u32 = 15872;
    /// `15168 + b` (row 0) / `14912 + b` (rows 0–5): sign rows of the
    /// byte-linked G1 / G2 base points `b`.
    pub const G1_SIGN: u32 = 15168;
    pub const G2_SIGN: u32 = 14912;
}

const C: Bits = Bits::new(0, 4);
const C3: Bits = Bits::new(0, 3);
const C5: Bits = Bits::new(0, 5);
const CELL: Bits = Bits::new(4, 18);
/// Index of a 32-row (two-cell) group.
const GROUP: Bits = Bits::new(5, 18);
// GT online.
const W_ON: Bits = Bits::new(4, 10);
const K_ON: Bits = Bits::new(10, 18);
// GT table.
const E_T: Bits = Bits::new(4, 7);
const K_T: Bits = Bits::new(7, 15);
const HI_T: Bits = Bits::new(15, 18);
// G2 online (half cells) / table / input.
const W2: Bits = Bits::new(3, 9);
const K2: Bits = Bits::new(9, 15);
const HI2: Bits = Bits::new(15, 18);
const J2: Bits = Bits::new(3, 7);
const B2: Bits = Bits::new(7, 13);
const HI2T: Bits = Bits::new(13, 18);
const B2I: Bits = Bits::new(3, 9);
const HI2I: Bits = Bits::new(9, 18);
// G1 online / table / input.
const W1: Bits = Bits::new(4, 10);
const KM1: Bits = Bits::new(10, 18);
const K1: Bits = Bits::new(10, 16);
const M1: Bits = Bits::new(16, 18);
const J1: Bits = Bits::new(4, 8);
const B1: Bits = Bits::new(8, 14);
const HI1T: Bits = Bits::new(14, 18);
const B1I: Bits = Bits::new(4, 10);
const HI1I: Bits = Bits::new(10, 18);
// Miller.
const S_MD: Bits = Bits::new(4, 7);
const T_MD: Bits = Bits::new(7, 13);
const HI_MD: Bits = Bits::new(13, 18);
const P_MA: Bits = Bits::new(4, 6);
const A_MA: Bits = Bits::new(6, 11);
const HI_MA: Bits = Bits::new(11, 18);
const P_LD: Bits = Bits::new(5, 6);
const T_LD: Bits = Bits::new(6, 12);
const HI_LD: Bits = Bits::new(12, 18);
const P_LA: Bits = Bits::new(5, 6);
const A_LA: Bits = Bits::new(6, 11);
const HI_LA: Bits = Bits::new(11, 18);
const P_CD: Bits = Bits::new(4, 5);
const T_CD: Bits = Bits::new(5, 11);
const HI_CD: Bits = Bits::new(11, 18);
const P_CA: Bits = Bits::new(4, 5);
const A_CA: Bits = Bits::new(5, 10);
const HI_CA: Bits = Bits::new(10, 18);
// Final exponentiation chains.
const SLOT_FE: Bits = Bits::new(4, 5);
const STEP_FE: Bits = Bits::new(5, 11);
const CHAIN_FE: Bits = Bits::new(11, 13);
const HI_FE: Bits = Bits::new(13, 18);

/// Row 0 of a cell.
const fn row(cell: u32) -> RowId {
    cell * 16
}

/// Row 0 of a half cell (`row >> 3` index).
const fn half_row(half: u32) -> RowId {
    half * 8
}

const fn hi(cell: u32, field: Bits) -> u32 {
    (cell * 16) >> field.lo
}

/// Per-family placement statistics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FamilyStats {
    pub name: &'static str,
    pub ops: usize,
    pub rows: usize,
    pub fixed_pieces: usize,
    pub selected_pieces: usize,
}

/// One digit-driven op: its first slotted row, the number of slotted rows,
/// what it looks up, its digit `j`, digit-base index `kd` and window `w`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DigitOp {
    pub first_row: RowId,
    pub rows: u8,
    pub kind: ReadKind,
    /// Index into [`Layout::selected`].
    pub family: u8,
    pub j: u8,
    pub kd: u32,
    /// Index of this `(chain, base)` occurrence among every chain's bases:
    /// the digit link weighs the op `ρ^link`, binding each occurrence's
    /// recoding to its scalar on its own.
    pub link: u32,
    pub w: u32,
}

/// `S0(x) = x + constant + k_coeff·k(x) + w_coeff·w(x)`: the row of the
/// reference table entry (`e = 0` for GT, `d = 0` for EC) aligned with row `x`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeyBase {
    pub constant: i64,
    pub k_coeff: i64,
    pub w_coeff: i64,
}

/// A family of digit-selected ops: the domain of its slotted rows (bit-field
/// restrictions), its fields, the key base, and the digit-base index of every
/// admitted `k` field value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectedFamily {
    pub kind: ReadKind,
    pub domain: Vec<Factor>,
    pub c_bits: Bits,
    /// Slotted rows of an op are coordinates `first_c..first_c + rows`.
    pub first_c: u32,
    pub rows: u32,
    pub k_bits: Bits,
    pub w_bits: Bits,
    pub key: KeyBase,
    /// `(k field value, link occurrence index)` of every admitted `k`.
    pub digit_base: Vec<(u32, u32)>,
}

/// The built layout: explicit rows for the prover, kernels for the verifier.
pub struct Layout {
    pub profile: Profile,
    pub check: FlattenedCheck,
    pub program: Program,
    /// Fixed copies: every operand column's kernel groups.
    pub copies: Vec<ElemWiring>,
    /// Fingerprint kernels of the table regions and the table rows' reads.
    pub fingerprints: Vec<FingerprintGroup>,
    pub table_reads: Vec<TableRead>,
    pub selected: Vec<SelectedFamily>,
    pub digit_ops: Vec<DigitOp>,
    pub families: Vec<FamilyStats>,
    /// The `one` element cell (rows `0..12` hold `1, 0, …, 0`).
    pub one_cell: u32,
    /// Digit bases: the named wires in the published order, the constant one,
    /// then the offset challenge `θ`.
    pub digit_bases: u32,
    /// Chain-base occurrences: the digit link's `ρ` powers (`DigitOp::link`).
    pub link_occurrences: u32,
    /// Committed input elements in the order of the `Input` rows (the T1 link order).
    pub input_order: Vec<InputElement>,
    /// The sign row of every byte-linked G1/G2 point: its `flag` column entry
    /// is `[y > −y]` (Fq2: lexicographic on `(y1, y0)`), the compressed
    /// encoding's sign bit the T1 link binds.
    pub sign_rows: Vec<(InputElement, RowId)>,
    /// The pinned rows `lhs_c − rhs_c = 0`.
    pub final_check: [RowId; 12],
    /// Cells of the four pairing G1 points (rows 14–15) and half cells of the
    /// two computed-line `Q` points (rows 0–3).
    pub pairing_points: [u32; 4],
    pub q_halves: [u32; 2],
    /// Cells of the corrected G1 chain outputs (`E1_acc, A3, A1, A4`; rows 14–15)
    /// and half cells of the G2 chain outputs (`E2_acc', B2`; rows 4–7).
    pub g1_outputs: [u32; 4],
    pub g2_outputs: [u32; 2],
    /// Rows of the Miller loop output and the final exponentiation output.
    pub miller: [RowId; 12],
    pub lhs: [RowId; 12],
    pub rhs: [RowId; 12],
}

impl Layout {
    pub fn used_rows(&self) -> usize {
        self.program.emitted()
    }

    /// Every fixed-copy kernel as a flat piece list.
    pub fn pieces(&self) -> Vec<Piece> {
        self.copies.iter().flat_map(ElemWiring::pieces).collect()
    }
}

/// Accumulates rows, kernels and statistics while the regions are placed.
struct Builder {
    program: Program,
    copies: Vec<ElemWiring>,
    fingerprints: Vec<FingerprintGroup>,
    table_reads: Vec<TableRead>,
    selected: Vec<SelectedFamily>,
    digit_ops: Vec<DigitOp>,
    link_occurrences: u32,
    families: Vec<FamilyStats>,
    input_order: Vec<InputElement>,
    sign_rows: Vec<(InputElement, RowId)>,
    glue_next: u32,
    wire_index: HashMap<DoryScalar, u32>,
    one_cell: u32,
}

/// What a GT leaf cell holds.
enum GtLeaf<'a> {
    Input(InputElement),
    Constant(&'a Fq12),
}

/// A GT value: the cell holding its twelve coordinates (rows `0..12`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GtCell(u32);

impl GtCell {
    fn rows(self) -> [RowId; 12] {
        std::array::from_fn(|c| row(self.0) + c as u32)
    }

    /// The element relation of a fixed GT cell read by every op of a family.
    fn fixed(self) -> ElemRel {
        ElemRel::structured(vec![Factor::constant(CELL, self.0)], C, 0)
    }
}

impl Builder {
    fn new(wire_order: &[DoryScalar]) -> Self {
        let wire_index = wire_order
            .iter()
            .enumerate()
            .map(|(i, wire)| (wire.clone(), i as u32))
            .collect();
        let mut program = Program::new(row(Cells::CONSTANTS.start)..row(Cells::CONSTANTS.end));
        assert_eq!(program.one, ONE_ROW);
        // The `one` element cell shares the point-constants cell (rows 0..12).
        let one_cell = Cells::POINT_CONSTANTS;
        for c in 0..12u32 {
            let value = if c == 0 { Fq::ONE } else { Fq::ZERO };
            program.pinned_constant_at(row(one_cell) + c, value);
        }
        Self {
            program,
            copies: Vec::new(),
            fingerprints: Vec::new(),
            table_reads: Vec::new(),
            selected: Vec::new(),
            digit_ops: Vec::new(),
            link_occurrences: 0,
            families: Vec::new(),
            input_order: Vec::new(),
            sign_rows: Vec::new(),
            glue_next: Cells::GLUE,
            wire_index,
            one_cell,
        }
    }

    /// Digit-base index of a wire: its position in the published order, then
    /// the constant one, then the offset challenge `θ`.
    pub(super) fn digit_index(&self, wire: &Wire) -> u32 {
        match wire {
            Wire::Named(scalar) => self.wire_index[scalar],
            Wire::One => self.wire_index.len() as u32,
            Wire::Offset => self.wire_index.len() as u32 + 1,
        }
    }

    /// Reserves the link occurrence indices of a chain's `n` bases.
    fn link_base(&mut self, n: u32) -> u32 {
        let base = self.link_occurrences;
        self.link_occurrences += n;
        base
    }

    /// Registers a table region: its fingerprint kernels (rows read their
    /// own cell through the reading template's fingerprinted `Y` maps) and
    /// the explicit reads of every entry (`first_rows` are the entries'
    /// coordinate-0 rows).
    #[expect(clippy::too_many_arguments, reason = "one region descriptor")]
    fn table_region(
        &mut self,
        domain: Vec<Factor>,
        cell_bits: Bits,
        first_rows: impl Iterator<Item = RowId>,
        template: &Template,
        own_bits: Bits,
        own_offset: u32,
        coord_offset: u32,
        fp_slots: usize,
    ) {
        let (maps, conj_maps) = fingerprint_maps(
            template,
            2,
            own_bits,
            own_offset,
            own_bits,
            coord_offset,
            fp_slots,
        );
        // Rows read their own cell: `same` on every field of the cell index,
        // restricted where the domain restricts it.
        let mut cell: Vec<Factor> = Vec::new();
        let mut lo = cell_bits.lo;
        let mut restricted: Vec<&Factor> = domain.iter().collect();
        restricted.sort_by_key(|f| f.u.lo);
        for factor in restricted {
            assert!(factor.u.lo >= lo, "overlapping domain fields");
            if factor.u.lo > lo {
                let gap = Bits::new(lo, factor.u.lo);
                cell.push(Factor::same(gap, gap));
            }
            let mut same = Factor::same(factor.u, factor.u);
            same.range.clone_from(&factor.range);
            cell.push(same);
            lo = factor.u.hi;
        }
        if lo < cell_bits.hi {
            let gap = Bits::new(lo, cell_bits.hi);
            cell.push(Factor::same(gap, gap));
        }
        // The conjugated map's weight is the one owner of the `f_neg` sign.
        for first in first_rows {
            for ((slot, map), (_, conj_map)) in maps.iter().zip(&conj_maps) {
                let (
                    Factor {
                        rel: Rel::Map(entries),
                        ..
                    },
                    Factor {
                        rel: Rel::Map(conj),
                        ..
                    },
                ) = (map, conj_map)
                else {
                    unreachable!("fingerprint maps are maps")
                };
                for (c, (entry, conj_entry)) in entries.iter().zip(conj).enumerate() {
                    let Some((coord, _)) = entry else { continue };
                    let Some((_, sign)) = conj_entry else {
                        unreachable!("conjugated map covers the same rows")
                    };
                    self.table_reads.push(TableRead {
                        row: first + c as u32,
                        slot: *slot,
                        src: first + coord,
                        conjugated: *sign < 0,
                    });
                }
            }
        }
        self.fingerprints.push(FingerprintGroup {
            cell,
            maps,
            conj_maps,
        });
    }

    fn one(&self) -> RowId {
        self.program.one
    }

    fn constant(&mut self, value: Fq) -> RowId {
        self.program.constant(value)
    }

    fn fq2_constant(&mut self, value: Fq2) -> [RowId; 2] {
        [self.constant(value.c0), self.constant(value.c1)]
    }

    /// The `(1,)` constants element.
    fn ones(&self) -> ElemRel {
        ElemRel::Rows(vec![self.program.one])
    }

    fn glue_cell(&mut self) -> u32 {
        let cell = self.glue_next;
        self.glue_next += 1;
        assert!(cell < Cells::GLUE_END, "glue region full");
        cell
    }

    /// Places every op of `family` (coordinate-0 row and digit, in
    /// evaluation order) and records its kernels.
    fn place(&mut self, family: &Family, ops: &[(RowId, Option<u8>)], selected: bool) {
        for &(base, digit) in ops {
            family.emit(&mut self.program, base, digit);
        }
        if family.domain.is_empty() {
            // Scattered glue ops: the row set is exactly the placed cells, so
            // constant and `ONE` operands must not leak onto other cells.
            let mut mask = vec![0i32; 1 << CELL.width()];
            for &(base, _) in ops {
                mask[CELL.extract(base) as usize] = 1;
            }
            let scattered = Family {
                domain: vec![Factor::weight(CELL, mask)],
                ..family.clone()
            };
            self.record(&scattered, ops.len(), selected);
        } else {
            self.record(family, ops.len(), selected);
        }
    }

    /// Records the kernels and statistics of an already emitted family; the
    /// digit-selected reads of a `selected` family are proven by the lookup.
    fn record(&mut self, family: &Family, ops: usize, selected: bool) {
        let (fixed, looked_up) = family.wiring();
        assert_eq!(
            looked_up.is_empty(),
            !selected,
            "{}: selected elements need the lookup",
            family.name
        );
        self.families.push(FamilyStats {
            name: family.name,
            ops,
            rows: ops * family.template.rows.len(),
            fixed_pieces: fixed.iter().map(|w| w.maps.len()).sum(),
            selected_pieces: looked_up.len(),
        });
        self.copies.extend(fixed);
    }
}

/// Non-adjacent form of `value`, least significant digit first.
fn naf(mut value: u128) -> Vec<i8> {
    let mut out = Vec::new();
    while value != 0 {
        let digit = if value & 1 == 1 {
            2 - (value % 4) as i8
        } else {
            0
        };
        out.push(digit);
        value = if digit < 0 {
            value.div_ceil(2)
        } else {
            (value - digit as u128) / 2
        };
    }
    out
}

impl Builder {
    fn table_elem(u: Bits, v: Bits, pairs: Vec<(u32, u32)>, coord: Bits, offset: u32) -> ElemRel {
        ElemRel::structured(vec![Factor::table(u, v, pairs)], coord, offset)
    }

    // ----- Miller loop ---------------------------------------------------
}

/// Builds the fixed layout for `check` with the digits of `values`;
/// `setup` supplies the pinned verifier-key constants.
pub fn build(
    check: &FlattenedCheck,
    values: &WireValues,
    setup: &DorySetupInputs,
    wire_order: &[DoryScalar],
) -> Layout {
    let profile = Profile {
        sigma: check.sigma,
        n: check.n,
    };
    assert!(
        profile.gt_bases() + 4 <= 150,
        "GT online region holds at most 150 bases"
    );
    let mut b = Builder::new(wire_order);
    // H1 in output layout (rows 14–15) so every pairing point shares one elem shape.
    let h1_cell = Cells::POINT_CONSTANTS;
    b.program.pinned_constant_at(row(h1_cell) + 14, setup.h1.x);
    b.program.pinned_constant_at(row(h1_cell) + 15, setup.h1.y);

    b.gt_tables(check, setup);
    b.gt_norm_checks(check.gt.bases.len());
    let rhs = b.gt_online(check, values);
    let g1_out = b.g1(check, values, setup);
    let (g2_out, q_halves) = b.g2(check, values, setup);
    b.g2_subgroup_checks(q_halves);
    // Pairs: (A1, E2_fin), (H1, B2), (A3, H2), (A4, Γ2_0).
    let p_cells = [g1_out[2], h1_cell, g1_out[1], g1_out[3]];
    let miller = b.miller(q_halves, &[setup.h2, setup.g2_0], p_cells);
    let lhs = b.final_exponentiation(miller);
    let pins = gt_difference_pins();
    b.place(
        &Family {
            name: "final_check",
            template: &pins,
            elems: vec![lhs.fixed(), rhs.fixed(), b.ones()],
            own_bits: C,
            own_offset: 0,
            domain: vec![Factor::restrict(CELL, Cells::FINAL..Cells::FINAL + 1)],
        },
        &[(row(Cells::FINAL), None)],
        false,
    );
    let final_check = GtCell(Cells::FINAL).rows();
    Layout {
        profile,
        check: check.clone(),
        program: b.program,
        copies: b.copies,
        fingerprints: b.fingerprints,
        table_reads: b.table_reads,
        selected: b.selected,
        digit_ops: b.digit_ops,
        families: b.families,
        one_cell: b.one_cell,
        digit_bases: b.wire_index.len() as u32 + 2,
        link_occurrences: b.link_occurrences,
        input_order: b.input_order,
        sign_rows: b.sign_rows,
        pairing_points: p_cells,
        q_halves,
        g1_outputs: g1_out,
        g2_outputs: g2_out,
        final_check,
        miller: miller.rows(),
        lhs: lhs.rows(),
        rhs: rhs.rows(),
    }
}
