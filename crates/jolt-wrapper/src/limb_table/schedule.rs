//! The fixed layout of the deferred check over `2^18` rows (`2^14` cells of
//! 16 rows): GT operations occupy rows `0..12` of a cell, four-row G1
//! operations ride in rows `12..16` of GT cells, eight-row G2 operations
//! pack two per cell. Every region is a bit-field box of the row index, so
//! each operand relation is a kernel ([`super::layout::Kernel`]) the
//! verifier evaluates in `O(bits)`; the digit-selected table reads are proven
//! by the operand lookup ([`super::lookup`]); the few irregular
//! final-exponentiation glue rows use explicit `Table` edge lists (`≤ 64`
//! edges per family).

use ark_bn254::{Config as Bn254Config, Fq, Fq12, Fq2, G1Affine, G2Affine};
use ark_ec::bn::{BnConfig, G2Prepared};
use ark_ec::short_weierstrass::SWCurveConfig;
use ark_ec::{AffineRepr, CurveGroup, PrimeGroup};
use ark_ff::{AdditiveGroup, BigInteger, Field, PrimeField};
use num_bigint::BigUint;
use std::collections::HashMap;
use std::ops::Range;

use crate::relation::DoryScalar;

use super::digits::{digits, WINDOWS};
use super::dory::{
    DorySetupInputs, FlattenedCheck, G1Base, G2Base, GtBase, InputElement, Wire, WireValues,
};
use super::layout::{Bits, Factor, Piece, Rel, LOG_ROWS};
use super::ops::{
    ell, g1_add, g1_copy, g1_dbl, g1_on_curve, g2_add, g2_copy, g2_dbl, g2_on_curve, g2_psi,
    gt_difference_pins, gt_frobenius, gt_inverse_pin, gt_inverse_witness, gt_mul, miller_add_step,
    miller_double_step, GtOperand, ADD_LINE, CONST_LINE, DOUBLE_LINE,
};
use super::program::{Program, RowId};
use super::relation::{FP_SLOTS_G1, FP_SLOTS_G2, FP_SLOTS_GT};
use super::template::{
    fingerprint_maps, DigitRule, ElemRel, ElemWiring, Family, Template, ONE_ROW,
};
use super::tower::{fq12_coords, frobenius_form};
use super::wiring::{FingerprintGroup, ReadKind, TableRead};

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
    /// Final-exponentiation glue and the `ψ` points (explicit edges).
    pub const GLUE: u32 = 9728;
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
    /// `(k field value, digit-base index)`.
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
    /// Digit bases: the named wires in the published order, then the constant one.
    pub digit_bases: u32,
    /// Committed input elements in the order of the `Input` rows (the T1 link order).
    pub input_order: Vec<InputElement>,
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
    families: Vec<FamilyStats>,
    input_order: Vec<InputElement>,
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
            families: Vec::new(),
            input_order: Vec::new(),
            glue_next: Cells::GLUE,
            wire_index,
            one_cell,
        }
    }

    /// Digit-base index of a wire: its position in the published order, or
    /// the extra index after every named wire for the constant one.
    pub(super) fn digit_index(&self, wire: &Wire) -> u32 {
        match wire {
            Wire::Named(scalar) => self.wire_index[scalar],
            Wire::One => self.wire_index.len() as u32,
        }
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
        assert!(cell < Cells::GLUE + 128, "glue region full");
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

    /// A GT leaf cell: committed input (twelve `Input` rows) or a pinned
    /// verifier-key constant.
    fn gt_leaf(&mut self, cell: u32, leaf: GtLeaf<'_>) {
        match leaf {
            GtLeaf::Input(element) => {
                for c in 0..12u32 {
                    let index = self.program.input_rows.len();
                    self.program.input_at(row(cell) + c, index);
                }
                self.input_order.push(element);
            }
            GtLeaf::Constant(value) => {
                for (c, coord) in fq12_coords(value).iter().enumerate() {
                    self.program
                        .pinned_constant_at(row(cell) + c as u32, *coord);
                }
            }
        }
    }

    // ----- GT: tables and Straus online ---------------------------------

    /// `X_k^{e+1}` for `e ∈ 1..8` behind each base's leaf cell.
    fn gt_tables(&mut self, check: &FlattenedCheck, setup: &DorySetupInputs) {
        let bases = check.gt.bases.len();
        for (k, (base, _)) in check.gt.bases.iter().enumerate() {
            let cell = Cells::GT_TABLE + 8 * k as u32;
            match base {
                GtBase::Input(element) => self.gt_leaf(cell, GtLeaf::Input(*element)),
                GtBase::Chi(i) => self.gt_leaf(cell, GtLeaf::Constant(&setup.chi[*i])),
                GtBase::Delta1R(i) => self.gt_leaf(cell, GtLeaf::Constant(&setup.delta_1r[*i])),
                GtBase::Delta2R(i) => self.gt_leaf(cell, GtLeaf::Constant(&setup.delta_2r[*i])),
                GtBase::Ht => self.gt_leaf(cell, GtLeaf::Constant(&setup.ht)),
            }
        }
        let template = gt_mul(GtOperand::dense(1), GtOperand::dense(2));
        let reading = template.clone().padded(2);
        let family = Family {
            name: "gt_table",
            template: &template,
            elems: vec![
                ElemRel::structured(
                    vec![
                        Factor::shift(E_T, E_T, -1).with_range(1..8),
                        Factor::same(K_T, K_T),
                        Factor::constant(HI_T, hi(Cells::GT_TABLE, HI_T)),
                    ],
                    C,
                    0,
                ),
                ElemRel::structured(
                    vec![
                        Factor::constant(E_T, 0),
                        Factor::same(K_T, K_T),
                        Factor::constant(HI_T, hi(Cells::GT_TABLE, HI_T)),
                    ],
                    C,
                    0,
                ),
            ],
            own_bits: C,
            own_offset: 0,
            domain: vec![
                Factor::restrict(E_T, 1..8),
                Factor::restrict(K_T, 0..bases as u32),
                Factor::restrict(
                    HI_T,
                    hi(Cells::GT_TABLE, HI_T)..hi(Cells::GT_TABLE, HI_T) + 1,
                ),
            ],
        };
        let ops: Vec<(RowId, Option<u8>)> = (0..bases as u32)
            .flat_map(|k| (1..8u32).map(move |e| (row(Cells::GT_TABLE + 8 * k + e), None)))
            .collect();
        self.place(&family, &ops, false);
        // Fingerprints of every entry (`e ∈ 0..8`) and of the `one` cell.
        let hi_t = hi(Cells::GT_TABLE, HI_T);
        let entries = (0..bases as u32)
            .flat_map(|k| (0..8u32).map(move |e| row(Cells::GT_TABLE + 8 * k + e)));
        self.table_region(
            vec![
                Factor::restrict(K_T, 0..bases as u32),
                Factor::restrict(HI_T, hi_t..hi_t + 1),
            ],
            CELL,
            entries,
            &reading,
            C,
            0,
            0,
            FP_SLOTS_GT,
        );
        let one_cell = self.one_cell;
        self.table_region(
            vec![Factor::restrict(CELL, one_cell..one_cell + 1)],
            CELL,
            std::iter::once(row(one_cell)),
            &reading,
            C,
            0,
            0,
            FP_SLOTS_GT,
        );
    }

    /// Straus over the 64 windows: per window four squarings of the
    /// accumulator (`k ∈ B..B+4`) then one selected-entry mult per base.
    /// Returns the accumulator cell (`RHS`).
    fn gt_online(&mut self, check: &FlattenedCheck, values: &WireValues) -> GtCell {
        let bases = check.gt.bases.len() as u32;
        let b = bases;
        let scalars = values.scalars(&check.gt);
        let digit_table: Vec<[u8; WINDOWS]> = scalars.iter().map(|s| digits(*s)).collect();
        let cell = |k: u32, w: u32| Cells::GT_ONLINE + 64 * k + w;
        let one_row = row(self.one_cell);
        let table_entry = ElemRel::Selected {
            factors: vec![
                Factor::same(K_ON, K_T),
                Factor::constant(HI_T, hi(Cells::GT_TABLE, HI_T)),
            ],
            coord_bits: C,
            offset: 0,
            entry_bits: E_T,
            rule: DigitRule::Gt { one: one_row },
        };
        let dense = gt_mul(GtOperand::dense(1), GtOperand::dense(1));
        let dense2 = gt_mul(GtOperand::dense(1), GtOperand::dense(2)).padded(2);
        let identity = gt_mul(GtOperand::one(1), GtOperand::one(1));
        let restrict_k = |lo: u32, hi_: u32| Factor::restrict(K_ON, lo..hi_);
        let const_k = |value: u32, at: u32| Factor {
            u: K_ON,
            v: K_ON,
            rel: Rel::Const(value),
            range: Some(at..at + 1),
        };
        let sq_init = Family {
            name: "gt_sq_init",
            template: &identity,
            elems: vec![ElemRel::Rows(vec![one_row])],
            own_bits: C,
            own_offset: 0,
            domain: vec![Factor::restrict(W_ON, 0..1), restrict_k(b, b + 1)],
        };
        let sq0 = Family {
            name: "gt_sq0",
            template: &dense,
            elems: vec![ElemRel::structured(
                vec![const_k(b - 1, b), Factor::shift(W_ON, W_ON, -1)],
                C,
                0,
            )],
            own_bits: C,
            own_offset: 0,
            domain: vec![Factor::restrict(W_ON, 1..64), restrict_k(b, b + 1)],
        };
        let sq = Family {
            name: "gt_sq",
            template: &dense,
            elems: vec![ElemRel::structured(
                vec![
                    Factor::shift(K_ON, K_ON, -1).with_range(b + 1..b + 4),
                    Factor::same(W_ON, W_ON),
                ],
                C,
                0,
            )],
            own_bits: C,
            own_offset: 0,
            domain: vec![restrict_k(b + 1, b + 4)],
        };
        let mult0 = Family {
            name: "gt_mult0",
            template: &dense2,
            elems: vec![
                ElemRel::structured(vec![const_k(b + 3, 0), Factor::same(W_ON, W_ON)], C, 0),
                table_entry.clone(),
            ],
            own_bits: C,
            own_offset: 0,
            domain: vec![restrict_k(0, 1)],
        };
        let mult = Family {
            name: "gt_mult",
            template: &dense2,
            elems: vec![
                ElemRel::structured(
                    vec![
                        Factor::shift(K_ON, K_ON, -1).with_range(1..b),
                        Factor::same(W_ON, W_ON),
                    ],
                    C,
                    0,
                ),
                table_entry,
            ],
            own_bits: C,
            own_offset: 0,
            domain: vec![restrict_k(1, b)],
        };
        let (mut o_init, mut o_sq0, mut o_sq, mut o_m0, mut o_m) =
            (vec![], vec![], vec![], vec![], vec![]);
        for w in 0..WINDOWS as u32 {
            if w == 0 {
                o_init.push((row(cell(b, w)), None));
            } else {
                o_sq0.push((row(cell(b, w)), None));
            }
            for i in 1..4 {
                o_sq.push((row(cell(b + i, w)), None));
            }
            for k in 0..bases {
                let j = digit_table[k as usize][WINDOWS - 1 - w as usize];
                self.digit_ops.push(DigitOp {
                    first_row: row(cell(k, w)),
                    rows: 12,
                    kind: ReadKind::Gt,
                    family: self.selected.len() as u8,
                    j,
                    kd: self.digit_index(&check.gt.bases[k as usize].1),
                    w,
                });
                if k == 0 {
                    o_m0.push((row(cell(k, w)), Some(j)));
                } else {
                    o_m.push((row(cell(k, w)), Some(j)));
                }
            }
        }
        // Emission order is window-major; `place` emits a family at once, so
        // interleave by hand.
        for w in 0..WINDOWS as u32 {
            let fam = if w == 0 { &sq_init } else { &sq0 };
            fam.emit(&mut self.program, row(cell(b, w)), None);
            for i in 1..4 {
                sq.emit(&mut self.program, row(cell(b + i, w)), None);
            }
            for k in 0..bases {
                let j = digit_table[k as usize][WINDOWS - 1 - w as usize];
                let fam = if k == 0 { &mult0 } else { &mult };
                fam.emit(&mut self.program, row(cell(k, w)), Some(j));
            }
        }
        for (family, ops, selected) in [
            (&sq_init, &o_init, false),
            (&sq0, &o_sq0, false),
            (&sq, &o_sq, false),
            (&mult0, &o_m0, true),
            (&mult, &o_m, true),
        ] {
            self.record(family, ops.len(), selected);
        }
        // Reference entry `e = 0` of base `k`: row `16·(GT_TABLE + 8k) + c`
        // from op row `16·(GT_ONLINE + 64k + w) + c`.
        self.selected.push(SelectedFamily {
            kind: ReadKind::Gt,
            domain: vec![Factor::restrict(C, 0..12), restrict_k(0, b)],
            c_bits: C,
            first_c: 0,
            rows: 12,
            k_bits: K_ON,
            w_bits: W_ON,
            key: KeyBase {
                constant: 16 * (i64::from(Cells::GT_TABLE) - i64::from(Cells::GT_ONLINE)),
                k_coeff: -(16 * 64 - 16 * 8),
                w_coeff: -16,
            },
            digit_base: (0..b)
                .map(|k| (k, self.digit_index(&check.gt.bases[k as usize].1)))
                .collect(),
        });
        GtCell(cell(b - 1, WINDOWS as u32 - 1))
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

/// Public offsets of an affine Straus chain: the accumulator starts at `R`
/// and every table entry carries `+Z0`, so no operand is ever the identity;
/// the total `16^64·R + n·((16^64 − 1)/15)·Z0` is subtracted at the end.
struct Offsets<A: AffineRepr> {
    r: A,
    z0: A,
}

impl<A: AffineRepr> Offsets<A> {
    fn new() -> Self {
        let r = A::generator();
        let z0 = (r.into_group() + r.into_group()).into_affine();
        Self { r, z0 }
    }

    fn correction(&self, bases: usize) -> A {
        let order = BigUint::from_bytes_be(
            &<<A as AffineRepr>::ScalarField as PrimeField>::MODULUS.to_bytes_be(),
        );
        let sixteen_pow = BigUint::from(1u32) << (4 * WINDOWS);
        let r_scale = &sixteen_pow % &order;
        let z_scale = (&sixteen_pow - BigUint::from(1u32)) / BigUint::from(15u32)
            * BigUint::from(bases)
            % &order;
        let total = self.r.into_group().mul_bigint(r_scale.to_u64_digits())
            + self.z0.into_group().mul_bigint(z_scale.to_u64_digits());
        (-total).into_affine()
    }
}

/// One EC chain's placement: global `k` of its first add and its digits.
struct EcChain {
    kbase: u32,
    bases: usize,
    digits: Vec<[u8; WINDOWS]>,
}

impl Builder {
    // ----- G1 (four-row ops in rows 12–15 of GT cells) ------------------

    /// Base points, tables, the four Straus chains and their corrections;
    /// returns the corrected chain outputs (cells whose rows 14–15 hold the point).
    fn g1(
        &mut self,
        check: &FlattenedCheck,
        values: &WireValues,
        setup: &DorySetupInputs,
    ) -> [u32; 4] {
        let offsets = Offsets::<G1Affine>::new();
        let [rx, ry] = [self.constant(offsets.r.x), self.constant(offsets.r.y)];
        let [zx, zy] = [self.constant(offsets.z0.x), self.constant(offsets.z0.y)];
        let three = self.constant(Fq::from(3u64));
        let one = self.one();
        let curve = g1_on_curve();
        let copy = g1_copy(false);
        let copy_neg = g1_copy(true);
        let add = g1_add(false);
        let sub = g1_add(true);
        let dbl = g1_dbl();
        let chains = check.g1_chains();
        let mut table_base = 0u32;
        let mut first_input: HashMap<InputElement, u32> = HashMap::new();
        let mut outputs = [0u32; 4];
        let mut fresh_inputs: Vec<u32> = Vec::new();
        for (m, msm) in chains.iter().enumerate() {
            let m32 = m as u32;
            let chain = EcChain {
                kbase: 64 * m32,
                bases: msm.bases.len(),
                digits: values.scalars(msm).iter().map(|s| digits(*s)).collect(),
            };
            let cell = |k: u32, w: u32| 4096 * m32 + 64 * k + w;
            // Inputs.
            for (i, (base, _)) in msm.bases.iter().enumerate() {
                let b = table_base + i as u32;
                let input_cell = Cells::G1_INPUT + b;
                let point_rows = [row(input_cell) + 12, row(input_cell) + 13];
                match base {
                    G1Base::Input(element) => {
                        if let Some(&source) = first_input.get(element) {
                            let family = Family {
                                name: "g1_input_copy",
                                template: &copy,
                                elems: vec![
                                    ElemRel::Rows(vec![row(source) + 12, row(source) + 13]),
                                    self.ones(),
                                ],
                                own_bits: C,
                                own_offset: 12,
                                domain: vec![Factor::restrict(CELL, input_cell..input_cell + 1)],
                            };
                            self.place(&family, &[(row(input_cell), None)], false);
                        } else {
                            let _ = first_input.insert(*element, input_cell);
                            for r in point_rows {
                                let index = self.program.input_rows.len();
                                self.program.input_at(r, index);
                            }
                            self.input_order.push(*element);
                            fresh_inputs.push(b);
                        }
                    }
                    G1Base::Gamma1Zero => {
                        self.program.pinned_constant_at(point_rows[0], setup.g1_0.x);
                        self.program.pinned_constant_at(point_rows[1], setup.g1_0.y);
                    }
                    G1Base::NegAcc => {
                        let source = outputs[0];
                        let family = Family {
                            name: "g1_neg_acc",
                            template: &copy_neg,
                            elems: vec![
                                ElemRel::Rows(vec![row(source) + 14, row(source) + 15]),
                                self.ones(),
                            ],
                            own_bits: C,
                            own_offset: 12,
                            domain: vec![Factor::restrict(CELL, input_cell..input_cell + 1)],
                        };
                        self.place(&family, &[(row(input_cell), None)], false);
                    }
                }
            }
            // Tables: `(j − 8)·P + Z0` at rows 14–15 of `G1_TABLE + 16·b + j`.
            let b_range = table_base..table_base + msm.bases.len() as u32;
            let tcell = |b: u32, j: u32| Cells::G1_TABLE + 16 * b + j;
            let hi1 = hi(Cells::G1_TABLE, HI1T);
            let z0_family = Family {
                name: "g1_table_z0",
                template: &copy,
                elems: vec![ElemRel::Rows(vec![zx, zy]), self.ones()],
                own_bits: C,
                own_offset: 14,
                domain: vec![
                    Factor::restrict(J1, 8..9),
                    Factor::restrict(B1, b_range.clone()),
                    Factor::restrict(HI1T, hi1..hi1 + 1),
                ],
            };
            let p_elem = ElemRel::structured(
                vec![
                    Factor::same(B1, B1I),
                    Factor::constant(HI1I, hi(Cells::G1_INPUT, HI1I)),
                ],
                C,
                12,
            );
            let prev = |delta: i64, range: Range<u32>| {
                ElemRel::structured(
                    vec![
                        Factor::shift(J1, J1, delta).with_range(range),
                        Factor::same(B1, B1),
                        Factor::constant(HI1T, hi1),
                    ],
                    C,
                    14,
                )
            };
            let up = Family {
                name: "g1_table_up",
                template: &add,
                elems: vec![prev(-1, 9..16), p_elem.clone(), self.ones()],
                own_bits: C,
                own_offset: 12,
                domain: vec![
                    Factor::restrict(J1, 9..16),
                    Factor::restrict(B1, b_range.clone()),
                    Factor::restrict(HI1T, hi1..hi1 + 1),
                ],
            };
            let down = Family {
                name: "g1_table_down",
                template: &sub,
                elems: vec![prev(1, 0..8), p_elem, self.ones()],
                own_bits: C,
                own_offset: 12,
                domain: vec![
                    Factor::restrict(J1, 0..8),
                    Factor::restrict(B1, b_range.clone()),
                    Factor::restrict(HI1T, hi1..hi1 + 1),
                ],
            };
            for b in b_range.clone() {
                z0_family.emit(&mut self.program, row(tcell(b, 8)), None);
                for j in 9..16 {
                    up.emit(&mut self.program, row(tcell(b, j)), None);
                }
                for j in (0..8).rev() {
                    down.emit(&mut self.program, row(tcell(b, j)), None);
                }
            }
            let nb = msm.bases.len();
            self.record(&z0_family, nb, false);
            self.record(&up, 7 * nb, false);
            self.record(&down, 8 * nb, false);
            let entries = b_range
                .clone()
                .flat_map(|b| (0..16u32).map(move |j| row(tcell(b, j))));
            self.table_region(
                vec![
                    Factor::restrict(B1, b_range.clone()),
                    Factor::restrict(HI1T, hi1..hi1 + 1),
                ],
                CELL,
                entries,
                &add,
                C,
                12,
                14,
                FP_SLOTS_G1,
            );

            // Online: doublings then adds per window, then the correction.
            let n = chain.bases as u32;
            let k_field = |k: u32| 64 * m32 + k;
            let restrict = |lo: u32, hi_: u32| Factor::restrict(KM1, k_field(lo)..k_field(hi_));
            let const_k = |value: u32, at: u32| Factor {
                u: KM1,
                v: KM1,
                rel: Rel::Const(k_field(value)),
                range: Some(k_field(at)..k_field(at) + 1),
            };
            let acc_prev = |delta: i64, lo: u32, hi_: u32| {
                ElemRel::structured(
                    vec![
                        Factor::shift(KM1, KM1, delta).with_range(k_field(lo)..k_field(hi_)),
                        Factor::same(W1, W1),
                    ],
                    C,
                    14,
                )
            };
            let selected = ElemRel::Selected {
                factors: vec![
                    Factor::shift(K1, B1, i64::from(table_base)).with_range(0..n),
                    Factor::restrict(M1, m32..m32 + 1),
                    Factor::constant(HI1T, hi1),
                ],
                coord_bits: C,
                offset: 14,
                entry_bits: J1,
                rule: DigitRule::Ec,
            };
            let add0 = Family {
                name: "g1_add0",
                template: &add,
                elems: vec![
                    ElemRel::structured(vec![const_k(n + 3, 0), Factor::same(W1, W1)], C, 14),
                    selected.clone(),
                    self.ones(),
                ],
                own_bits: C,
                own_offset: 12,
                domain: vec![restrict(0, 1)],
            };
            let adds = Family {
                name: "g1_add",
                template: &add,
                elems: vec![acc_prev(-1, 1, n), selected, self.ones()],
                own_bits: C,
                own_offset: 12,
                domain: vec![restrict(1, n)],
            };
            let dbl_init = Family {
                name: "g1_dbl_init",
                template: &dbl,
                elems: vec![ElemRel::Rows(vec![rx, ry]), self.ones()],
                own_bits: C,
                own_offset: 12,
                domain: vec![restrict(n, n + 1), Factor::restrict(W1, 0..1)],
            };
            let dbl0 = Family {
                name: "g1_dbl0",
                template: &dbl,
                elems: vec![
                    ElemRel::structured(vec![const_k(n - 1, n), Factor::shift(W1, W1, -1)], C, 14),
                    self.ones(),
                ],
                own_bits: C,
                own_offset: 12,
                domain: vec![restrict(n, n + 1), Factor::restrict(W1, 1..64)],
            };
            let dbls = Family {
                name: "g1_dbl",
                template: &dbl,
                elems: vec![acc_prev(-1, n + 1, n + 4), self.ones()],
                own_bits: C,
                own_offset: 12,
                domain: vec![restrict(n + 1, n + 4)],
            };
            let correction = offsets.correction(chain.bases);
            let [cx, cy] = [self.constant(correction.x), self.constant(correction.y)];
            let corr = Family {
                name: "g1_corr",
                template: &add,
                elems: vec![
                    ElemRel::structured(vec![const_k(n - 1, n + 4), Factor::same(W1, W1)], C, 14),
                    ElemRel::Rows(vec![cx, cy]),
                    self.ones(),
                ],
                own_bits: C,
                own_offset: 12,
                domain: vec![restrict(n + 4, n + 5), Factor::restrict(W1, 63..64)],
            };
            let (mut c_init, mut c_dbl0, mut c_dbl, mut c_add0, mut c_add) = (0, 0, 0, 0, 0);
            for w in 0..WINDOWS as u32 {
                if w == 0 {
                    dbl_init.emit(&mut self.program, row(cell(n, w)), None);
                    c_init += 1;
                } else {
                    dbl0.emit(&mut self.program, row(cell(n, w)), None);
                    c_dbl0 += 1;
                }
                for i in 1..4 {
                    dbls.emit(&mut self.program, row(cell(n + i, w)), None);
                    c_dbl += 1;
                }
                for k in 0..n {
                    let j = chain.digits[k as usize][WINDOWS - 1 - w as usize];
                    self.digit_ops.push(DigitOp {
                        first_row: row(cell(k, w)) + 13,
                        rows: 3,
                        kind: ReadKind::G1,
                        family: self.selected.len() as u8,
                        j,
                        kd: self.digit_index(&msm.bases[k as usize].1),
                        w,
                    });
                    if k == 0 {
                        add0.emit(&mut self.program, row(cell(k, w)), Some(j));
                        c_add0 += 1;
                    } else {
                        adds.emit(&mut self.program, row(cell(k, w)), Some(j));
                        c_add += 1;
                    }
                }
            }
            corr.emit(&mut self.program, row(cell(n + 4, 63)), None);
            self.record(&dbl_init, c_init, false);
            self.record(&dbl0, c_dbl0, false);
            self.record(&dbls, c_dbl, false);
            self.record(&add0, c_add0, true);
            self.record(&adds, c_add, true);
            self.record(&corr, 1, false);
            // Reference entry `j = 8` of base `b = table_base + k`: row
            // `16·(G1_TABLE + 16b + 8) + c` from op row `16·(4096m + 64k + w) + c`,
            // with the `KM1` field holding `64m + k`.
            self.selected.push(SelectedFamily {
                kind: ReadKind::G1,
                domain: vec![Factor::restrict(C, 13..16), restrict(0, n)],
                c_bits: C,
                first_c: 13,
                rows: 3,
                k_bits: KM1,
                w_bits: W1,
                key: KeyBase {
                    constant: 16 * i64::from(Cells::G1_TABLE) + 128 + 256 * i64::from(table_base)
                        - 16384 * i64::from(m32),
                    k_coeff: -768,
                    w_coeff: -16,
                },
                digit_base: (0..n)
                    .map(|k| (k_field(k), self.digit_index(&msm.bases[k as usize].1)))
                    .collect(),
            });
            outputs[m] = cell(n + 4, 63);
            table_base += msm.bases.len() as u32;
        }
        let mut mask = vec![0i32; 64];
        for b in &fresh_inputs {
            mask[*b as usize] = 1;
        }
        let hi_i = hi(Cells::G1_INPUT, HI1I);
        let family = Family {
            name: "g1_on_curve",
            template: &curve,
            elems: vec![
                ElemRel::structured(vec![Factor::same(CELL, CELL)], C, 12),
                ElemRel::Rows(vec![three, one]),
            ],
            own_bits: C,
            own_offset: 14,
            domain: vec![
                Factor::weight(B1I, mask),
                Factor::restrict(HI1I, hi_i..hi_i + 1),
            ],
        };
        let ops: Vec<(RowId, Option<u8>)> = fresh_inputs
            .iter()
            .map(|b| (row(Cells::G1_INPUT + b), None))
            .collect();
        self.place(&family, &ops, false);
        outputs
    }
}

impl Builder {
    // ----- G2 (eight-row ops, two per cell) ------------------------------

    /// Base points (`b < 3σ+3`: the two chains' bases, then `E2_fin` and the
    /// `B2` copy the Miller loop reads), tables, chains and corrections.
    /// Returns the half cells of the corrected chain outputs (point at rows
    /// 4–7) and of the two Miller `Q` inputs (point at rows 0–3).
    fn g2(
        &mut self,
        check: &FlattenedCheck,
        values: &WireValues,
        setup: &DorySetupInputs,
    ) -> ([u32; 2], [u32; 2]) {
        let offsets = Offsets::<G2Affine>::new();
        let r = [
            self.fq2_constant(offsets.r.x),
            self.fq2_constant(offsets.r.y),
        ]
        .concat();
        let z0 = [
            self.fq2_constant(offsets.z0.x),
            self.fq2_constant(offsets.z0.y),
        ]
        .concat();
        let b_curve = self.fq2_constant(<ark_bn254::g2::Config as SWCurveConfig>::COEFF_B);
        let one = self.one();
        let curve = g2_on_curve();
        let copy = g2_copy(false);
        let copy_neg = g2_copy(true);
        let add = g2_add(false);
        let sub = g2_add(true);
        let dbl = g2_dbl();
        let chains = check.g2_chains();
        let hi_t = hi(Cells::G2_TABLE_HALF / 2, HI2T);
        let hi_i = hi(Cells::G2_INPUT_HALF / 2, HI2I);
        let hi_o = hi(Cells::G2_ONLINE_HALF / 2, HI2);
        let input_half = |b: u32| Cells::G2_INPUT_HALF + b;
        let mut table_base = 0u32;
        let mut first_input: HashMap<InputElement, u32> = HashMap::new();
        let mut outputs = [0u32; 2];
        let mut fresh_halves: Vec<u32> = Vec::new();
        let kbases = [0u32, 39];
        for (m, msm) in chains.iter().enumerate() {
            let chain = EcChain {
                kbase: kbases[m],
                bases: msm.bases.len(),
                digits: values.scalars(msm).iter().map(|s| digits(*s)).collect(),
            };
            let half = |k: u32, w: u32| Cells::G2_ONLINE_HALF + 64 * k + w;
            for (i, (base, _)) in msm.bases.iter().enumerate() {
                let b = table_base + i as u32;
                let h = input_half(b);
                match base {
                    G2Base::Input(element) => {
                        if let Some(&source) = first_input.get(element) {
                            let family = Family {
                                name: "g2_input_copy",
                                template: &copy,
                                elems: vec![
                                    ElemRel::Rows((0..4).map(|c| half_row(source) + c).collect()),
                                    self.ones(),
                                ],
                                own_bits: C3,
                                own_offset: 0,
                                domain: vec![Factor::restrict(Bits::new(3, 18), h..h + 1)],
                            };
                            self.place(&family, &[(half_row(h), None)], false);
                        } else {
                            let _ = first_input.insert(*element, h);
                            self.g2_input_leaf(h, *element);
                            fresh_halves.push(h);
                        }
                    }
                    G2Base::Gamma2Zero => {
                        for (c, v) in [
                            setup.g2_0.x.c0,
                            setup.g2_0.x.c1,
                            setup.g2_0.y.c0,
                            setup.g2_0.y.c1,
                        ]
                        .iter()
                        .enumerate()
                        {
                            self.program.pinned_constant_at(half_row(h) + c as u32, *v);
                        }
                    }
                    G2Base::NegAcc => {
                        let source = outputs[0];
                        let family = Family {
                            name: "g2_neg_acc",
                            template: &copy_neg,
                            elems: vec![
                                ElemRel::Rows((4..8).map(|c| half_row(source) + c).collect()),
                                self.ones(),
                            ],
                            own_bits: C3,
                            own_offset: 0,
                            domain: vec![Factor::restrict(Bits::new(3, 18), h..h + 1)],
                        };
                        self.place(&family, &[(half_row(h), None)], false);
                    }
                }
            }
            // Tables at half cells `G2_TABLE_HALF + 16·b + j`, points at rows 4–7.
            let b_range = table_base..table_base + msm.bases.len() as u32;
            let thalf = |b: u32, j: u32| Cells::G2_TABLE_HALF + 16 * b + j;
            let z0_family = Family {
                name: "g2_table_z0",
                template: &copy,
                elems: vec![ElemRel::Rows(z0.clone()), self.ones()],
                own_bits: C3,
                own_offset: 4,
                domain: vec![
                    Factor::restrict(J2, 8..9),
                    Factor::restrict(B2, b_range.clone()),
                    Factor::restrict(HI2T, hi_t..hi_t + 1),
                ],
            };
            let p_elem = ElemRel::structured(
                vec![Factor::same(B2, B2I), Factor::constant(HI2I, hi_i)],
                C3,
                0,
            );
            let prev = |delta: i64, range: Range<u32>| {
                ElemRel::structured(
                    vec![
                        Factor::shift(J2, J2, delta).with_range(range),
                        Factor::same(B2, B2),
                        Factor::constant(HI2T, hi_t),
                    ],
                    C3,
                    4,
                )
            };
            let up = Family {
                name: "g2_table_up",
                template: &add,
                elems: vec![prev(-1, 9..16), p_elem.clone(), self.ones()],
                own_bits: C3,
                own_offset: 0,
                domain: vec![
                    Factor::restrict(J2, 9..16),
                    Factor::restrict(B2, b_range.clone()),
                    Factor::restrict(HI2T, hi_t..hi_t + 1),
                ],
            };
            let down = Family {
                name: "g2_table_down",
                template: &sub,
                elems: vec![prev(1, 0..8), p_elem, self.ones()],
                own_bits: C3,
                own_offset: 0,
                domain: vec![
                    Factor::restrict(J2, 0..8),
                    Factor::restrict(B2, b_range.clone()),
                    Factor::restrict(HI2T, hi_t..hi_t + 1),
                ],
            };
            for b in b_range.clone() {
                z0_family.emit(&mut self.program, half_row(thalf(b, 8)), None);
                for j in 9..16 {
                    up.emit(&mut self.program, half_row(thalf(b, j)), None);
                }
                for j in (0..8).rev() {
                    down.emit(&mut self.program, half_row(thalf(b, j)), None);
                }
            }
            let nb = msm.bases.len();
            self.record(&z0_family, nb, false);
            self.record(&up, 7 * nb, false);
            self.record(&down, 8 * nb, false);
            let entries = b_range
                .clone()
                .flat_map(|b| (0..16u32).map(move |j| half_row(thalf(b, j))));
            self.table_region(
                vec![
                    Factor::restrict(B2, b_range.clone()),
                    Factor::restrict(HI2T, hi_t..hi_t + 1),
                ],
                Bits::new(3, LOG_ROWS as u8),
                entries,
                &add,
                C3,
                0,
                4,
                FP_SLOTS_G2,
            );

            let n = chain.bases as u32;
            let kb = chain.kbase;
            let restrict = |lo: u32, hi_: u32| Factor::restrict(K2, kb + lo..kb + hi_);
            let const_k = |value: u32, at: u32| Factor {
                u: K2,
                v: K2,
                rel: Rel::Const(kb + value),
                range: Some(kb + at..kb + at + 1),
            };
            let acc_prev = |delta: i64, lo: u32, hi_: u32| {
                ElemRel::structured(
                    vec![
                        Factor::shift(K2, K2, delta).with_range(kb + lo..kb + hi_),
                        Factor::same(W2, W2),
                        Factor::constant(HI2, hi_o),
                    ],
                    C3,
                    4,
                )
            };
            let selected = ElemRel::Selected {
                factors: vec![
                    Factor::shift(K2, B2, i64::from(table_base) - i64::from(kb))
                        .with_range(kb..kb + n),
                    Factor::constant(HI2T, hi_t),
                ],
                coord_bits: C3,
                offset: 4,
                entry_bits: J2,
                rule: DigitRule::Ec,
            };
            let region = Factor::restrict(HI2, hi_o..hi_o + 1);
            let add0 = Family {
                name: "g2_add0",
                template: &add,
                elems: vec![
                    ElemRel::structured(
                        vec![
                            const_k(n + 3, 0),
                            Factor::same(W2, W2),
                            Factor::constant(HI2, hi_o),
                        ],
                        C3,
                        4,
                    ),
                    selected.clone(),
                    self.ones(),
                ],
                own_bits: C3,
                own_offset: 0,
                domain: vec![restrict(0, 1), region.clone()],
            };
            let adds = Family {
                name: "g2_add",
                template: &add,
                elems: vec![acc_prev(-1, 1, n), selected, self.ones()],
                own_bits: C3,
                own_offset: 0,
                domain: vec![restrict(1, n), region.clone()],
            };
            let dbl_init = Family {
                name: "g2_dbl_init",
                template: &dbl,
                elems: vec![ElemRel::Rows(r.clone()), self.ones()],
                own_bits: C3,
                own_offset: 0,
                domain: vec![
                    restrict(n, n + 1),
                    Factor::restrict(W2, 0..1),
                    region.clone(),
                ],
            };
            let dbl0 = Family {
                name: "g2_dbl0",
                template: &dbl,
                elems: vec![
                    ElemRel::structured(
                        vec![
                            const_k(n - 1, n),
                            Factor::shift(W2, W2, -1),
                            Factor::constant(HI2, hi_o),
                        ],
                        C3,
                        4,
                    ),
                    self.ones(),
                ],
                own_bits: C3,
                own_offset: 0,
                domain: vec![
                    restrict(n, n + 1),
                    Factor::restrict(W2, 1..64),
                    region.clone(),
                ],
            };
            let dbls = Family {
                name: "g2_dbl",
                template: &dbl,
                elems: vec![acc_prev(-1, n + 1, n + 4), self.ones()],
                own_bits: C3,
                own_offset: 0,
                domain: vec![restrict(n + 1, n + 4), region.clone()],
            };
            let correction = offsets.correction(chain.bases);
            let c_rows = [
                self.fq2_constant(correction.x),
                self.fq2_constant(correction.y),
            ]
            .concat();
            let corr = Family {
                name: "g2_corr",
                template: &add,
                elems: vec![
                    ElemRel::structured(
                        vec![
                            const_k(n - 1, n + 4),
                            Factor::same(W2, W2),
                            Factor::constant(HI2, hi_o),
                        ],
                        C3,
                        4,
                    ),
                    ElemRel::Rows(c_rows),
                    self.ones(),
                ],
                own_bits: C3,
                own_offset: 0,
                domain: vec![
                    restrict(n + 4, n + 5),
                    Factor::restrict(W2, 63..64),
                    region.clone(),
                ],
            };
            let (mut c_init, mut c_dbl0, mut c_dbl, mut c_add0, mut c_add) = (0, 0, 0, 0, 0);
            for w in 0..WINDOWS as u32 {
                if w == 0 {
                    dbl_init.emit(&mut self.program, half_row(half(kb + n, w)), None);
                    c_init += 1;
                } else {
                    dbl0.emit(&mut self.program, half_row(half(kb + n, w)), None);
                    c_dbl0 += 1;
                }
                for i in 1..4 {
                    dbls.emit(&mut self.program, half_row(half(kb + n + i, w)), None);
                    c_dbl += 1;
                }
                for k in 0..n {
                    let j = chain.digits[k as usize][WINDOWS - 1 - w as usize];
                    self.digit_ops.push(DigitOp {
                        first_row: half_row(half(kb + k, w)) + 2,
                        rows: 6,
                        kind: ReadKind::G2,
                        family: self.selected.len() as u8,
                        j,
                        kd: self.digit_index(&msm.bases[k as usize].1),
                        w,
                    });
                    if k == 0 {
                        add0.emit(&mut self.program, half_row(half(kb + k, w)), Some(j));
                        c_add0 += 1;
                    } else {
                        adds.emit(&mut self.program, half_row(half(kb + k, w)), Some(j));
                        c_add += 1;
                    }
                }
            }
            corr.emit(&mut self.program, half_row(half(kb + n + 4, 63)), None);
            self.record(&dbl_init, c_init, false);
            self.record(&dbl0, c_dbl0, false);
            self.record(&dbls, c_dbl, false);
            self.record(&add0, c_add0, true);
            self.record(&adds, c_add, true);
            self.record(&corr, 1, false);
            // Reference entry `j = 8` of base `b = table_base + k`: row
            // `8·(G2_TABLE_HALF + 16b + 8) + c` from op row
            // `8·(G2_ONLINE_HALF + 64·k2 + w) + c`, `k2 = kb + k` the `K2` field.
            self.selected.push(SelectedFamily {
                kind: ReadKind::G2,
                domain: vec![Factor::restrict(C3, 2..8), restrict(0, n), region.clone()],
                c_bits: C3,
                first_c: 2,
                rows: 6,
                k_bits: K2,
                w_bits: W2,
                key: KeyBase {
                    constant: 8
                        * (i64::from(Cells::G2_TABLE_HALF) - i64::from(Cells::G2_ONLINE_HALF))
                        + 64
                        + 128 * (i64::from(table_base) - i64::from(kb)),
                    k_coeff: -(8 * 64 - 8 * 16),
                    w_coeff: -8,
                },
                digit_base: (0..n)
                    .map(|k| (kb + k, self.digit_index(&msm.bases[k as usize].1)))
                    .collect(),
            });
            outputs[m] = half(kb + n + 4, 63);
            table_base += msm.bases.len() as u32;
        }
        // Miller `Q` inputs: `E2_fin` (committed) and a copy of `B2` in input layout.
        let e2_fin = input_half(table_base);
        self.g2_input_leaf(e2_fin, InputElement::FinalE2);
        fresh_halves.push(e2_fin);
        self.g2_on_curve_family(&fresh_halves, &curve, &b_curve, one);
        let b2 = input_half(table_base + 1);
        let family = Family {
            name: "g2_b2_copy",
            template: &copy,
            elems: vec![
                ElemRel::Rows((4..8).map(|c| half_row(outputs[1]) + c).collect()),
                self.ones(),
            ],
            own_bits: C3,
            own_offset: 0,
            domain: vec![Factor::restrict(Bits::new(3, 18), b2..b2 + 1)],
        };
        self.place(&family, &[(half_row(b2), None)], false);
        (outputs, [e2_fin, b2])
    }

    /// A committed G2 point at rows 0–3 of half cell `h` (its on-curve check
    /// is placed once for all inputs by [`Self::g2_on_curve_family`]).
    fn g2_input_leaf(&mut self, h: u32, element: InputElement) {
        for c in 0..4 {
            let index = self.program.input_rows.len();
            self.program.input_at(half_row(h) + c, index);
        }
        self.input_order.push(element);
    }

    fn g2_on_curve_family(
        &mut self,
        halves: &[u32],
        curve: &Template,
        b_curve: &[RowId; 2],
        one: RowId,
    ) {
        let mut mask = vec![0i32; 64];
        for h in halves {
            mask[(h - Cells::G2_INPUT_HALF) as usize] = 1;
        }
        let hi_i = hi(Cells::G2_INPUT_HALF / 2, HI2I);
        let family = Family {
            name: "g2_on_curve",
            template: curve,
            elems: vec![
                ElemRel::structured(
                    vec![Factor::same(Bits::new(3, 18), Bits::new(3, 18))],
                    C3,
                    0,
                ),
                ElemRel::Rows(vec![b_curve[0], b_curve[1], one]),
            ],
            own_bits: C3,
            own_offset: 4,
            domain: vec![
                Factor::weight(B2I, mask),
                Factor::restrict(HI2I, hi_i..hi_i + 1),
            ],
        };
        let ops: Vec<(RowId, Option<u8>)> = halves.iter().map(|h| (half_row(*h), None)).collect();
        self.place(&family, &ops, false);
    }
}

/// Non-adjacent form of `value`, least significant digit first.
fn naf(mut value: u64) -> Vec<i8> {
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
            (value - digit as u64) / 2
        };
    }
    out
}

/// Whether the ate loop adds after doubling step `t` (`ate[63 − t] ≠ 0`),
/// the addition index `a(t)` of each such step, and the sign of each
/// addition (`−1`: the step adds `−Q`).
fn ate_schedule() -> (Vec<bool>, Vec<Option<u32>>, Vec<i8>) {
    let ate = Bn254Config::ATE_LOOP_COUNT;
    let steps = ate.len() - 1;
    let mut add_after = Vec::with_capacity(steps);
    let mut add_index = Vec::with_capacity(steps);
    let mut signs = Vec::new();
    let mut a = 0u32;
    for t in 0..steps {
        let digit = ate[steps - 1 - t];
        let adds = digit != 0;
        add_after.push(adds);
        add_index.push(adds.then_some(a));
        if adds {
            signs.push(digit);
        }
        a += u32::from(adds);
    }
    (add_after, add_index, signs)
}

/// Rows of a Miller step's output point inside its 32-row group.
const DBL_OUT: u32 = 20;
const ADD_OUT: u32 = 16;

impl Builder {
    fn table_elem(u: Bits, v: Bits, pairs: Vec<(u32, u32)>, coord: Bits, offset: u32) -> ElemRel {
        ElemRel::structured(vec![Factor::table(u, v, pairs)], coord, offset)
    }

    // ----- Miller loop ---------------------------------------------------

    /// Line computations for pairs 0–1 (`Q` at the given half cells, point at
    /// rows 0–3), public lines for pairs 2–3, and the GT accumulation over
    /// the pairing points `p_cells` (rows 14–15). Returns the Miller output.
    fn miller(&mut self, q_halves: [u32; 2], const_q: &[G2Affine; 2], p_cells: [u32; 4]) -> GtCell {
        let (add_after, add_index, signs) = ate_schedule();
        let steps = add_after.len() as u32;
        let loop_adds = add_index.iter().flatten().count() as u32;
        let adds = loop_adds + 2;
        let one = self.one();
        let zero = self.program.zero;
        let two_inv = self.constant(Fq::from(2u64).inverse().unwrap_or_else(|| unreachable!()));
        let three_b = self.fq2_constant(
            <ark_bn254::g2::Config as SWCurveConfig>::COEFF_B * Fq2::new(Fq::from(3u64), Fq::ZERO),
        );
        let dbl_consts = ElemRel::Rows(vec![two_inv, three_b[0], three_b[1], one]);
        let dbl_step = miller_double_step();
        let add_step = miller_add_step(false);
        let add_step_neg = miller_add_step(true);
        let dcell = |t: u32, p: u32| Cells::MILLER_DBL_LINES + 4 * t + 2 * p;
        let acell = |a: u32, p: u32| Cells::MILLER_ADD_LINES + 4 * a + 2 * p;
        let (hi_ld, hi_la) = (
            hi(Cells::MILLER_DBL_LINES, HI_LD),
            hi(Cells::MILLER_ADD_LINES, HI_LA),
        );

        // ψ points for the two final additions.
        let (cx, cy) = super::ops::psi_coefficients(1);
        let psi_consts = ElemRel::Rows([self.fq2_constant(cx), self.fq2_constant(cy)].concat());
        let q1_cells = [self.glue_cell(), self.glue_cell()];
        let q2_cells = [self.glue_cell(), self.glue_cell()];
        let psi = g2_psi(false);
        let psi_neg = g2_psi(true);
        let q1 = Family {
            name: "ml_psi",
            template: &psi,
            elems: vec![
                Self::table_elem(
                    CELL,
                    Bits::new(3, 18),
                    (0..2).map(|p| (q1_cells[p], q_halves[p])).collect(),
                    C3,
                    0,
                ),
                psi_consts.clone(),
            ],
            own_bits: C,
            own_offset: 0,
            domain: vec![],
        };
        self.place(&q1, &q1_cells.map(|c| (row(c), None)), false);
        let q2 = Family {
            name: "ml_psi_neg",
            template: &psi_neg,
            elems: vec![
                Self::table_elem(
                    CELL,
                    CELL,
                    (0..2).map(|p| (q2_cells[p], q1_cells[p])).collect(),
                    C,
                    0,
                ),
                psi_consts,
            ],
            own_bits: C,
            own_offset: 0,
            domain: vec![],
        };
        self.place(&q2, &q2_cells.map(|c| (row(c), None)), false);

        // Doubling steps.
        let first: Vec<Family> = (0..2u32)
            .map(|p| Family {
                name: "ml_dbl_first",
                template: &dbl_step,
                elems: vec![
                    ElemRel::Rows(vec![
                        half_row(q_halves[p as usize]),
                        half_row(q_halves[p as usize]) + 1,
                        half_row(q_halves[p as usize]) + 2,
                        half_row(q_halves[p as usize]) + 3,
                        one,
                        zero,
                    ]),
                    dbl_consts.clone(),
                ],
                own_bits: C5,
                own_offset: 0,
                // A 32-row op: its block of two cells.
                domain: vec![Factor::restrict(
                    GROUP,
                    dcell(0, p) / 2..dcell(0, p) / 2 + 1,
                )],
            })
            .collect();
        let after_dbl_mask: Vec<i32> = (0..64)
            .map(|t| i32::from(t >= 1 && t < steps && !add_after[t as usize - 1]))
            .collect();
        let after_add_mask: Vec<i32> = (0..64)
            .map(|t| i32::from(t >= 1 && t < steps && add_after[t as usize - 1]))
            .collect();
        let after_dbl = Family {
            name: "ml_dbl_after_dbl",
            template: &dbl_step,
            elems: vec![
                ElemRel::structured(
                    vec![
                        Factor::shift(T_LD, T_LD, -1),
                        Factor::same(P_LD, P_LD),
                        Factor::constant(HI_LD, hi_ld),
                    ],
                    C5,
                    DBL_OUT,
                ),
                dbl_consts.clone(),
            ],
            own_bits: C5,
            own_offset: 0,
            domain: vec![
                Factor::weight(T_LD, after_dbl_mask),
                Factor::restrict(HI_LD, hi_ld..hi_ld + 1),
            ],
        };
        let after_add_pairs: Vec<(u32, u32)> = (1..steps)
            .filter_map(|t| add_index[t as usize - 1].map(|a| (t, a)))
            .collect();
        let after_add = Family {
            name: "ml_dbl_after_add",
            template: &dbl_step,
            elems: vec![
                ElemRel::structured(
                    vec![
                        Factor::table(T_LD, A_LA, after_add_pairs.clone()),
                        Factor::same(P_LD, P_LA),
                        Factor::constant(HI_LA, hi_la),
                    ],
                    C5,
                    ADD_OUT,
                ),
                dbl_consts,
            ],
            own_bits: C5,
            own_offset: 0,
            domain: vec![
                Factor::weight(T_LD, after_add_mask),
                Factor::restrict(HI_LD, hi_ld..hi_ld + 1),
            ],
        };
        // Addition steps.
        let q_elem = Self::table_elem(
            P_LA,
            Bits::new(3, 18),
            (0..2).map(|p| (p, q_halves[p as usize])).collect(),
            C3,
            0,
        );
        let loop_pairs: Vec<(u32, u32)> = (0..steps)
            .filter_map(|t| add_index[t as usize].map(|a| (a, t)))
            .collect();
        let sign_mask = |sign: i8| -> Vec<i32> {
            (0..32)
                .map(|a| i32::from((a as usize) < signs.len() && signs[a as usize] == sign))
                .collect()
        };
        let add_loop_family = |name, template, sign| Family {
            name,
            template,
            elems: vec![
                ElemRel::structured(
                    vec![
                        Factor::table(A_LA, T_LD, loop_pairs.clone()),
                        Factor::same(P_LA, P_LD),
                        Factor::constant(HI_LD, hi_ld),
                    ],
                    C5,
                    DBL_OUT,
                ),
                q_elem.clone(),
                self.ones(),
            ],
            own_bits: C5,
            own_offset: 0,
            domain: vec![
                Factor::weight(A_LA, sign_mask(sign)),
                Factor::restrict(HI_LA, hi_la..hi_la + 1),
            ],
        };
        let add_loop = add_loop_family("ml_add", &add_step, 1);
        let add_loop_neg = add_loop_family("ml_add_neg", &add_step_neg, -1);
        let last_t = steps - 1;
        let (end_cell, end_offset): (Box<dyn Fn(u32) -> u32>, u32) = if add_after[last_t as usize] {
            (Box::new(move |p| acell(loop_adds - 1, p)), ADD_OUT)
        } else {
            (Box::new(move |p| dcell(last_t, p)), DBL_OUT)
        };
        let add_q1 = Family {
            name: "ml_add_q1",
            template: &add_step,
            elems: vec![
                Self::table_elem(
                    P_LA,
                    GROUP,
                    (0..2).map(|p| (p, end_cell(p) >> 1)).collect(),
                    C5,
                    end_offset,
                ),
                Self::table_elem(
                    P_LA,
                    CELL,
                    (0..2).map(|p| (p, q1_cells[p as usize])).collect(),
                    C,
                    0,
                ),
                self.ones(),
            ],
            own_bits: C5,
            own_offset: 0,
            domain: vec![
                Factor::restrict(A_LA, loop_adds..loop_adds + 1),
                Factor::restrict(HI_LA, hi_la..hi_la + 1),
            ],
        };
        let add_q2 = Family {
            name: "ml_add_q2",
            template: &add_step,
            elems: vec![
                ElemRel::structured(
                    vec![
                        Factor::shift(A_LA, A_LA, -1).with_range(loop_adds + 1..loop_adds + 2),
                        Factor::same(P_LA, P_LA),
                        Factor::constant(HI_LA, hi_la),
                    ],
                    C5,
                    ADD_OUT,
                ),
                Self::table_elem(
                    P_LA,
                    CELL,
                    (0..2).map(|p| (p, q2_cells[p as usize])).collect(),
                    C,
                    0,
                ),
                self.ones(),
            ],
            own_bits: C5,
            own_offset: 0,
            domain: vec![
                Factor::restrict(A_LA, loop_adds + 1..loop_adds + 2),
                Factor::restrict(HI_LA, hi_la..hi_la + 1),
            ],
        };
        let (mut n_first, mut n_ad, mut n_aa, mut n_loop) = (0, 0, 0, 0);
        for t in 0..steps {
            for p in 0..2u32 {
                let base = row(dcell(t, p));
                if t == 0 {
                    first[p as usize].emit(&mut self.program, base, None);
                    n_first += 1;
                } else if add_after[t as usize - 1] {
                    after_add.emit(&mut self.program, base, None);
                    n_aa += 1;
                } else {
                    after_dbl.emit(&mut self.program, base, None);
                    n_ad += 1;
                }
            }
            if let Some(a) = add_index[t as usize] {
                let family = if signs[a as usize] > 0 {
                    &add_loop
                } else {
                    &add_loop_neg
                };
                for p in 0..2u32 {
                    family.emit(&mut self.program, row(acell(a, p)), None);
                    n_loop += 1;
                }
            }
        }
        for p in 0..2u32 {
            add_q1.emit(&mut self.program, row(acell(loop_adds, p)), None);
        }
        for p in 0..2u32 {
            add_q2.emit(&mut self.program, row(acell(loop_adds + 1, p)), None);
        }
        for family in &first {
            self.record(family, n_first / 2, false);
        }
        self.record(&after_dbl, n_ad, false);
        self.record(&after_add, n_aa, false);
        let n_neg = 2 * signs.iter().filter(|s| **s < 0).count();
        self.record(&add_loop, n_loop - n_neg, false);
        self.record(&add_loop_neg, n_neg, false);
        self.record(&add_q1, 2, false);
        self.record(&add_q2, 2, false);

        // Public lines of pairs 2–3.
        for (i, q) in const_q.iter().enumerate() {
            let prepared = G2Prepared::<Bn254Config>::from(*q);
            let mut coeffs = prepared.ell_coeffs.iter();
            let mut write = |cell: u32, coeff: &(Fq2, Fq2, Fq2)| {
                for (c, v) in [
                    coeff.0.c0, coeff.0.c1, coeff.1.c0, coeff.1.c1, coeff.2.c0, coeff.2.c1,
                ]
                .iter()
                .enumerate()
                {
                    self.program.pinned_constant_at(row(cell) + c as u32, *v);
                }
            };
            for t in 0..steps {
                write(
                    Cells::CONST_LINES_DBL + 2 * t + i as u32,
                    coeffs
                        .next()
                        .unwrap_or_else(|| unreachable!("G2Prepared has one line per step")),
                );
                if let Some(a) = add_index[t as usize] {
                    write(
                        Cells::CONST_LINES_ADD + 2 * a + i as u32,
                        coeffs
                            .next()
                            .unwrap_or_else(|| unreachable!("G2Prepared has one line per step")),
                    );
                }
            }
            for a in loop_adds..adds {
                write(
                    Cells::CONST_LINES_ADD + 2 * a + i as u32,
                    coeffs
                        .next()
                        .unwrap_or_else(|| unreachable!("G2Prepared has one line per step")),
                );
            }
            assert!(coeffs.next().is_none(), "line schedule mismatch");
        }
        self.miller_gt(&add_after, &add_index, loop_adds, p_cells)
    }

    /// `f ← f²`, then `ell` per pair, over the 64 doubling steps and the
    /// `adds` addition steps; returns the final `f`.
    fn miller_gt(
        &mut self,
        add_after: &[bool],
        add_index: &[Option<u32>],
        loop_adds: u32,
        p_cells: [u32; 4],
    ) -> GtCell {
        let steps = add_after.len() as u32;
        let one = self.one();
        let dcell = |t: u32, s: u32| Cells::MILLER_DBL_GT + 8 * t + s;
        let acell = |a: u32, p: u32| Cells::MILLER_ADD_GT + 4 * a + p;
        let (hi_md, hi_ma) = (
            hi(Cells::MILLER_DBL_GT, HI_MD),
            hi(Cells::MILLER_ADD_GT, HI_MA),
        );
        let (hi_ld, hi_la) = (
            hi(Cells::MILLER_DBL_LINES, HI_LD),
            hi(Cells::MILLER_ADD_LINES, HI_LA),
        );
        let (hi_cd, hi_ca) = (
            hi(Cells::CONST_LINES_DBL, HI_CD),
            hi(Cells::CONST_LINES_ADD, HI_CA),
        );
        let identity = gt_mul(GtOperand::one(1), GtOperand::one(1));
        let dense = gt_mul(GtOperand::dense(1), GtOperand::dense(1));
        let ell_dbl = ell(DOUBLE_LINE);
        let ell_add = ell(ADD_LINE);
        let ell_const = ell(CONST_LINE);
        let p_dbl = Self::table_elem(
            S_MD,
            CELL,
            (1..5).map(|s| (s, p_cells[s as usize - 1])).collect(),
            C,
            14,
        );
        let p_add = Self::table_elem(
            P_MA,
            CELL,
            (0..4).map(|p| (p, p_cells[p as usize])).collect(),
            C,
            14,
        );
        let region_d = Factor::restrict(HI_MD, hi_md..hi_md + 1);
        let region_a = Factor::restrict(HI_MA, hi_ma..hi_ma + 1);

        let sq_init = Family {
            name: "mg_sq_init",
            template: &identity,
            elems: vec![ElemRel::Rows(vec![one])],
            own_bits: C,
            own_offset: 0,
            domain: vec![Factor::restrict(CELL, dcell(0, 0)..dcell(0, 0) + 1)],
        };
        let after_dbl_mask: Vec<i32> = (0..64)
            .map(|t| i32::from(t >= 1 && t < steps && !add_after[t as usize - 1]))
            .collect();
        let after_add_mask: Vec<i32> = (0..64)
            .map(|t| i32::from(t >= 1 && t < steps && add_after[t as usize - 1]))
            .collect();
        let sq_after_dbl = Family {
            name: "mg_sq_after_dbl",
            template: &dense,
            elems: vec![ElemRel::structured(
                vec![
                    Factor::shift(T_MD, T_MD, -1),
                    Factor {
                        u: S_MD,
                        v: S_MD,
                        rel: Rel::Const(4),
                        range: Some(0..1),
                    },
                    Factor::constant(HI_MD, hi_md),
                ],
                C,
                0,
            )],
            own_bits: C,
            own_offset: 0,
            domain: vec![
                Factor::restrict(S_MD, 0..1),
                Factor::weight(T_MD, after_dbl_mask),
                region_d.clone(),
            ],
        };
        let after_add_pairs: Vec<(u32, u32)> = (1..steps)
            .filter_map(|t| add_index[t as usize - 1].map(|a| (t, a)))
            .collect();
        let sq_after_add = Family {
            name: "mg_sq_after_add",
            template: &dense,
            elems: vec![ElemRel::structured(
                vec![
                    Factor::table(T_MD, A_MA, after_add_pairs),
                    Factor {
                        u: S_MD,
                        v: P_MA,
                        rel: Rel::Const(3),
                        range: Some(0..1),
                    },
                    Factor::constant(HI_MA, hi_ma),
                ],
                C,
                0,
            )],
            own_bits: C,
            own_offset: 0,
            domain: vec![
                Factor::restrict(S_MD, 0..1),
                Factor::weight(T_MD, after_add_mask),
                region_d.clone(),
            ],
        };
        let prev_slot = ElemRel::structured(
            vec![
                Factor::shift(S_MD, S_MD, -1).with_range(1..5),
                Factor::same(T_MD, T_MD),
                Factor::constant(HI_MD, hi_md),
            ],
            C,
            0,
        );
        let ell_dbl_computed = Family {
            name: "mg_ell_dbl",
            template: &ell_dbl,
            elems: vec![
                prev_slot.clone(),
                ElemRel::structured(
                    vec![
                        Factor::map(
                            S_MD,
                            P_LD,
                            vec![
                                None,
                                Some((0, 1)),
                                Some((1, 1)),
                                None,
                                None,
                                None,
                                None,
                                None,
                            ],
                        ),
                        Factor::same(T_MD, T_LD),
                        Factor::constant(HI_LD, hi_ld),
                    ],
                    C5,
                    0,
                ),
                p_dbl.clone(),
            ],
            own_bits: C,
            own_offset: 0,
            domain: vec![
                Factor::restrict(S_MD, 1..3),
                Factor::restrict(T_MD, 0..steps),
                region_d.clone(),
            ],
        };
        let ell_dbl_const = Family {
            name: "mg_ell_dbl_const",
            template: &ell_const,
            elems: vec![
                prev_slot,
                ElemRel::structured(
                    vec![
                        Factor::map(
                            S_MD,
                            P_CD,
                            vec![
                                None,
                                None,
                                None,
                                Some((0, 1)),
                                Some((1, 1)),
                                None,
                                None,
                                None,
                            ],
                        ),
                        Factor::same(T_MD, T_CD),
                        Factor::constant(HI_CD, hi_cd),
                    ],
                    C,
                    0,
                ),
                p_dbl,
            ],
            own_bits: C,
            own_offset: 0,
            domain: vec![
                Factor::restrict(S_MD, 3..5),
                Factor::restrict(T_MD, 0..steps),
                region_d,
            ],
        };
        // Addition ells: slot 0 reads the loop's `f` through an edge table.
        let last_t = steps - 1;
        let mut f_pairs: Vec<(u32, u32)> = (0..steps)
            .filter_map(|t| add_index[t as usize].map(|a| (a, dcell(t, 4))))
            .collect();
        f_pairs.push((
            loop_adds,
            if add_after[last_t as usize] {
                acell(loop_adds - 1, 3)
            } else {
                dcell(last_t, 4)
            },
        ));
        f_pairs.push((loop_adds + 1, acell(loop_adds, 3)));
        let computed_line = ElemRel::structured(
            vec![
                Factor::same(A_MA, A_LA),
                Factor::map(P_MA, P_LA, vec![Some((0, 1)), Some((1, 1)), None, None]),
                Factor::constant(HI_LA, hi_la),
            ],
            C5,
            0,
        );
        let ell_add0 = Family {
            name: "ma_ell0",
            template: &ell_add,
            elems: vec![
                Self::table_elem(A_MA, CELL, f_pairs, C, 0),
                computed_line.clone(),
                p_add.clone(),
            ],
            own_bits: C,
            own_offset: 0,
            domain: vec![
                Factor::restrict(P_MA, 0..1),
                Factor::restrict(A_MA, 0..loop_adds + 2),
                region_a.clone(),
            ],
        };
        let prev_p = ElemRel::structured(
            vec![
                Factor::shift(P_MA, P_MA, -1).with_range(1..4),
                Factor::same(A_MA, A_MA),
                Factor::constant(HI_MA, hi_ma),
            ],
            C,
            0,
        );
        let ell_add1 = Family {
            name: "ma_ell1",
            template: &ell_add,
            elems: vec![prev_p.clone(), computed_line, p_add.clone()],
            own_bits: C,
            own_offset: 0,
            domain: vec![
                Factor::restrict(P_MA, 1..2),
                Factor::restrict(A_MA, 0..loop_adds + 2),
                region_a.clone(),
            ],
        };
        let ell_add_const = Family {
            name: "ma_ell_const",
            template: &ell_const,
            elems: vec![
                prev_p,
                ElemRel::structured(
                    vec![
                        Factor::same(A_MA, A_CA),
                        Factor::map(P_MA, P_CA, vec![None, None, Some((0, 1)), Some((1, 1))]),
                        Factor::constant(HI_CA, hi_ca),
                    ],
                    C,
                    0,
                ),
                p_add,
            ],
            own_bits: C,
            own_offset: 0,
            domain: vec![
                Factor::restrict(P_MA, 2..4),
                Factor::restrict(A_MA, 0..loop_adds + 2),
                region_a,
            ],
        };
        let mut counts = [0usize; 8];
        let emit_adds = |this: &mut Self, a: u32, counts: &mut [usize; 8]| {
            ell_add0.emit(&mut this.program, row(acell(a, 0)), None);
            ell_add1.emit(&mut this.program, row(acell(a, 1)), None);
            ell_add_const.emit(&mut this.program, row(acell(a, 2)), None);
            ell_add_const.emit(&mut this.program, row(acell(a, 3)), None);
            counts[5] += 1;
            counts[6] += 1;
            counts[7] += 2;
        };
        for t in 0..steps {
            if t == 0 {
                sq_init.emit(&mut self.program, row(dcell(0, 0)), None);
                counts[0] += 1;
            } else if add_after[t as usize - 1] {
                sq_after_add.emit(&mut self.program, row(dcell(t, 0)), None);
                counts[2] += 1;
            } else {
                sq_after_dbl.emit(&mut self.program, row(dcell(t, 0)), None);
                counts[1] += 1;
            }
            for s in 1..3 {
                ell_dbl_computed.emit(&mut self.program, row(dcell(t, s)), None);
                counts[3] += 1;
            }
            for s in 3..5 {
                ell_dbl_const.emit(&mut self.program, row(dcell(t, s)), None);
                counts[4] += 1;
            }
            if let Some(a) = add_index[t as usize] {
                emit_adds(self, a, &mut counts);
            }
        }
        emit_adds(self, loop_adds, &mut counts);
        emit_adds(self, loop_adds + 1, &mut counts);
        for (family, count) in [
            (&sq_init, counts[0]),
            (&sq_after_dbl, counts[1]),
            (&sq_after_add, counts[2]),
            (&ell_dbl_computed, counts[3]),
            (&ell_dbl_const, counts[4]),
            (&ell_add0, counts[5]),
            (&ell_add1, counts[6]),
            (&ell_add_const, counts[7]),
        ] {
            self.record(family, count, false);
        }
        GtCell(acell(loop_adds + 1, 3))
    }
}

impl Builder {
    // ----- Final exponentiation -------------------------------------------

    /// A glue family: one template over ops whose operand cells are listed
    /// explicitly (`(op cell, x cell, y cell)`), wired through edge tables.
    fn glue(&mut self, name: &'static str, template: &Template, ops: &[(u32, u32, Option<u32>)]) {
        let x_pairs: Vec<(u32, u32)> = ops.iter().map(|(op, x, _)| (*op, *x)).collect();
        let mut elems = vec![Self::table_elem(CELL, CELL, x_pairs, C, 0)];
        if ops.iter().any(|(_, _, y)| y.is_some()) {
            let y_pairs: Vec<(u32, u32)> = ops
                .iter()
                .filter_map(|(op, _, y)| y.map(|y| (*op, y)))
                .collect();
            elems.push(Self::table_elem(CELL, CELL, y_pairs, C, 0));
        }
        let family = Family {
            name,
            template,
            elems,
            own_bits: C,
            own_offset: 0,
            domain: vec![],
        };
        let placed: Vec<(RowId, Option<u8>)> =
            ops.iter().map(|(op, _, _)| (row(*op), None)).collect();
        self.place(&family, &placed, false);
    }

    fn frobenius(&mut self, name: &'static str, power: usize, ops: &[(u32, u32)]) {
        let form = frobenius_form(power);
        let (template, constants) = gt_frobenius(form);
        let mut rows: Vec<RowId> = Vec::with_capacity(constants.len());
        for c in constants {
            rows.push(self.constant(c));
        }
        let family = Family {
            name,
            template: &template,
            elems: vec![
                Self::table_elem(CELL, CELL, ops.to_vec(), C, 0),
                ElemRel::Rows(rows),
            ],
            own_bits: C,
            own_offset: 0,
            domain: vec![],
        };
        let placed: Vec<(RowId, Option<u8>)> = ops.iter().map(|(op, _)| (row(*op), None)).collect();
        self.place(&family, &placed, false);
    }

    /// Arkworks' BN final exponentiation of `f`: easy part with an inverse
    /// witness, then the Fuentes–Castañeda hard part with three `f^{-x}`
    /// chains. Returns the result cell.
    fn final_exponentiation(&mut self, f: GtCell) -> GtCell {
        let one = self.one();
        let mut cell = || self.glue_cell();
        let (inv_w, inv_pin, r, r2, f_cyc) = (cell(), cell(), cell(), cell(), cell());
        let (y1, y2, y3, y5, y7, y8, y9, y10, y11) = (
            cell(),
            cell(),
            cell(),
            cell(),
            cell(),
            cell(),
            cell(),
            cell(),
            cell(),
        );
        let (y12, y13, y8f, y14, y15a, y15, result) =
            (cell(), cell(), cell(), cell(), cell(), cell(), cell());
        let naf_x = naf(Bn254Config::X[0]);
        let chain_steps = naf_x.len() as u32 - 1;
        let chain_out = |chain: u32| Cells::FE_CHAINS + 128 * chain + 2 * (chain_steps - 1) + 1;
        let [res0, res1, res2] = [chain_out(0), chain_out(1), chain_out(2)];

        // Easy part.
        let witness = gt_inverse_witness();
        self.place(
            &Family {
                name: "fe_inv_witness",
                template: &witness,
                elems: vec![f.fixed()],
                own_bits: C,
                own_offset: 0,
                domain: vec![Factor::restrict(CELL, inv_w..inv_w + 1)],
            },
            &[(row(inv_w), None)],
            false,
        );
        let pin = gt_inverse_pin();
        self.place(
            &Family {
                name: "fe_inv_pin",
                template: &pin,
                elems: vec![f.fixed(), GtCell(inv_w).fixed()],
                own_bits: C,
                own_offset: 0,
                domain: vec![Factor::restrict(CELL, inv_pin..inv_pin + 1)],
            },
            &[(row(inv_pin), None)],
            false,
        );
        let conj_dense = gt_mul(GtOperand::conj(1), GtOperand::dense(2));
        let dense_dense = gt_mul(GtOperand::dense(1), GtOperand::dense(2));
        let conj_conj = gt_mul(GtOperand::conj(1), GtOperand::conj(2));
        let dense_conj = gt_mul(GtOperand::dense(1), GtOperand::conj(2));
        // r = conj(f)·f⁻¹; f_cyc = frob²(r)·r.
        self.glue("fe_mul_conj_dense_a", &conj_dense, &[(r, f.0, Some(inv_w))]);
        self.frobenius("fe_frob2_a", 2, &[(r2, r)]);
        self.glue("fe_mul_a", &dense_dense, &[(f_cyc, r2, Some(r))]);
        // Hard part, interleaved with the chains it depends on.
        let chains = [(0u32, f_cyc), (1, y3), (2, y5)];
        self.fe_chain(&naf_x, chains[0]);
        self.glue("fe_mul_conj_conj_a", &conj_conj, &[(y1, res0, Some(res0))]);
        self.glue(
            "fe_mul_b",
            &dense_dense,
            &[(y2, y1, Some(y1)), (y3, y2, Some(y1))],
        );
        self.fe_chain(&naf_x, chains[1]);
        self.glue("fe_mul_conj_conj_b", &conj_conj, &[(y5, res1, Some(res1))]);
        self.fe_chain(&naf_x, chains[2]);
        self.glue(
            "fe_mul_dense_conj",
            &dense_conj,
            &[
                (y7, res2, Some(res1)),
                (y8, y7, Some(y3)),
                (y10, y8, Some(res1)),
            ],
        );
        self.glue(
            "fe_mul_c",
            &dense_dense,
            &[(y9, y8, Some(y1)), (y11, y10, Some(f_cyc))],
        );
        self.frobenius("fe_frob1", 1, &[(y12, y9)]);
        self.frobenius("fe_frob2_b", 2, &[(y8f, y8)]);
        self.glue(
            "fe_mul_d",
            &dense_dense,
            &[(y13, y12, Some(y11)), (y14, y8f, Some(y13))],
        );
        self.glue(
            "fe_mul_conj_dense_b",
            &conj_dense,
            &[(y15a, f_cyc, Some(y9))],
        );
        self.frobenius("fe_frob3", 3, &[(y15, y15a)]);
        self.glue("fe_mul_e", &dense_dense, &[(result, y15, Some(y14))]);
        let _ = one;
        GtCell(result)
    }

    /// `base^{-x}` as its own conjugate: NAF square-and-multiply with the
    /// base, its conjugate, or the identity per step (public digits).
    fn fe_chain(&mut self, naf_x: &[i8], (chain, base): (u32, u32)) {
        let steps = naf_x.len() - 1;
        let one = self.one();
        let hi_fe = hi(Cells::FE_CHAINS, HI_FE);
        let cell = |step: u32, slot: u32| Cells::FE_CHAINS + 128 * chain + 2 * step + slot;
        // Digit of step `s` (processing from the most significant): naf[steps − 1 − s].
        let digit_mask = |value: i8| -> Vec<i32> {
            (0..64)
                .map(|s| i32::from((s as usize) < steps && naf_x[steps - 1 - s as usize] == value))
                .collect()
        };
        let base_elem = Self::table_elem(CHAIN_FE, CELL, vec![(chain, base)], C, 0);
        let region = Factor::restrict(HI_FE, hi_fe..hi_fe + 1);
        let this_chain = Factor::restrict(CHAIN_FE, chain..chain + 1);
        let dense = gt_mul(GtOperand::dense(1), GtOperand::dense(1));
        let dense2 = gt_mul(GtOperand::dense(1), GtOperand::dense(2));
        let dense_conj = gt_mul(GtOperand::dense(1), GtOperand::conj(2));
        let dense_one = gt_mul(GtOperand::dense(1), GtOperand::one(2));
        let sq0 = Family {
            name: "fe_sq0",
            template: &dense,
            elems: vec![base_elem.clone()],
            own_bits: C,
            own_offset: 0,
            domain: vec![
                Factor::restrict(STEP_FE, 0..1),
                Factor::restrict(SLOT_FE, 0..1),
                this_chain.clone(),
                region.clone(),
            ],
        };
        let sq = Family {
            name: "fe_sq",
            template: &dense,
            elems: vec![ElemRel::structured(
                vec![
                    Factor::shift(STEP_FE, STEP_FE, -1).with_range(1..steps as u32),
                    Factor {
                        u: SLOT_FE,
                        v: SLOT_FE,
                        rel: Rel::Const(1),
                        range: Some(0..1),
                    },
                    Factor::same(CHAIN_FE, CHAIN_FE),
                    Factor::constant(HI_FE, hi_fe),
                ],
                C,
                0,
            )],
            own_bits: C,
            own_offset: 0,
            domain: vec![
                Factor::restrict(STEP_FE, 1..steps as u32),
                Factor::restrict(SLOT_FE, 0..1),
                this_chain.clone(),
                region.clone(),
            ],
        };
        let sq_elem = ElemRel::structured(
            vec![
                Factor {
                    u: SLOT_FE,
                    v: SLOT_FE,
                    rel: Rel::Const(0),
                    range: Some(1..2),
                },
                Factor::same(STEP_FE, STEP_FE),
                Factor::same(CHAIN_FE, CHAIN_FE),
                Factor::constant(HI_FE, hi_fe),
            ],
            C,
            0,
        );
        let mul = |name, template, y: ElemRel, value: i8| Family {
            name,
            template,
            elems: vec![sq_elem.clone(), y],
            own_bits: C,
            own_offset: 0,
            domain: vec![
                Factor::restrict(SLOT_FE, 1..2),
                Factor::weight(STEP_FE, digit_mask(value)),
                this_chain.clone(),
                region.clone(),
            ],
        };
        let mul_pos = mul("fe_mul_pos", &dense2, base_elem.clone(), 1);
        let mul_neg = mul("fe_mul_neg", &dense_conj, base_elem, -1);
        let mul_zero = mul("fe_mul_zero", &dense_one, ElemRel::Rows(vec![one]), 0);
        let mut counts = [0usize; 5];
        for s in 0..steps as u32 {
            if s == 0 {
                sq0.emit(&mut self.program, row(cell(s, 0)), None);
                counts[0] += 1;
            } else {
                sq.emit(&mut self.program, row(cell(s, 0)), None);
                counts[1] += 1;
            }
            let family = match naf_x[steps - 1 - s as usize] {
                1 => {
                    counts[2] += 1;
                    &mul_pos
                }
                -1 => {
                    counts[3] += 1;
                    &mul_neg
                }
                _ => {
                    counts[4] += 1;
                    &mul_zero
                }
            };
            family.emit(&mut self.program, row(cell(s, 1)), None);
        }
        for (family, count) in [
            (&sq0, counts[0]),
            (&sq, counts[1]),
            (&mul_pos, counts[2]),
            (&mul_neg, counts[3]),
            (&mul_zero, counts[4]),
        ] {
            self.record(family, count, false);
        }
    }
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
    let rhs = b.gt_online(check, values);
    let g1_out = b.g1(check, values, setup);
    let (g2_out, q_halves) = b.g2(check, values, setup);
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
        digit_bases: b.wire_index.len() as u32 + 1,
        input_order: b.input_order,
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
