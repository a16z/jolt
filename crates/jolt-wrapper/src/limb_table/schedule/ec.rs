//! The G1 and G2 lanes: affine Straus chains over the flattened check's
//! MSMs with transcript-randomized offsets.
//!
//! Every chain starts at `R = θ·G` and reads table entries `d·P + Z0` with
//! `Z0 = φ(R) = θ·φ(G)`, `φ` the GLV endomorphism (`φ(G) = [λ]G`), `θ` the
//! wrapper's offset challenge drawn after the phase-1a commitments (the
//! points `P` among them). Before the add of window `w`, base `k`, the
//! accumulator is `θ·(16^{w+1} + λ(nw + k))·G + H` with `H` the honest,
//! θ-independent partial sum, and the entry is `d·P + θλ·G`; the exceptional
//! affine case `acc = ±entry` is therefore the linear equation
//! `θ·(16^{w+1} + λ(nw + k ∓ 1))·G = ∓d·P − H` in `θ`, with exactly one root
//! because its coefficient is a nonzero scalar ([`tests::offsets_are_nondegenerate`]
//! sweeps every `(w, k, n)` of the layout); a doubling of the identity is
//! `16^w + λnw ≡ 0`, likewise excluded, and neither curve has 2-torsion. The
//! table entries `(d ∓ 1)·P + θλG ± P` degenerate only when `θλG` equals a
//! fixed point, one root each. The offsets are removed by one extra base
//! `−K` per chain with scalar `θ`, `K = 16^64·G + n'·(16^64 − 1)/15·φ(G)`
//! (`n'` bases including itself), so no correction depends on the prover.
//!
//! `R` itself is a fixed-base Straus chain of `θ` over `G` with constant
//! offsets `R'' = G`, `Z'' = 9G` and a constant correction: the accumulator
//! multiplier before the add of window `w` is the integer
//! `16^{w+1} + Σ_{i<w} (d_i + 9)·16^{w−i} ∈ (16^{w+1}, 2.07·16^{w+1})` and
//! the entry multiplier `d_w + 9 ∈ [1, 16]`, so `acc = ±entry` needs a wrap
//! modulo `r`, impossible below `2^254` (`w ≤ 62`) and a single-residue
//! event on the last window (`≤ 32` digit strings) and on the correction
//! (`θ ∈ {0, −2·(16^64 + 9(16^64 − 1)/15)}`).

use ark_bn254::{
    g1::Config as G1Config, g2::Config as G2Config, Config as Bn254Config, Fq, G1Affine, G2Affine,
};
use ark_ec::bn::BnConfig;
use ark_ec::scalar_mul::glv::GLVConfig;
use ark_ec::short_weierstrass::SWCurveConfig;
use ark_ec::{AffineRepr, CurveGroup};
use num_bigint::BigUint;
use std::collections::HashMap;
use std::ops::Range;

use super::super::digits::{digits, WINDOWS};
use super::super::dory::{FlattenedCheck, G1Base, G2Base, InputElement, Wire, WireValues};
use super::super::layout::{Bits, Factor, Rel};
use super::super::ops::{
    g1_add, g1_copy, g1_dbl, g1_endo, g1_on_curve, g1_sign, g2_add, g2_add_guarded, g2_copy,
    g2_dbl, g2_endo, g2_negation_pins, g2_on_curve, g2_psi, g2_sign, psi_coefficients,
};
use super::super::program::{half_plus_one, RowId};
use super::super::relation::{FP_SLOTS_G1, FP_SLOTS_G2};
use super::super::template::{DigitRule, ElemRel, Family, Template};
use super::super::wiring::ReadKind;
use super::{
    half_row, hi, naf, row, Builder, Cells, DigitOp, DorySetupInputs, KeyBase, SelectedFamily, B1,
    B1I, B2, B2I, C, C3, CELL, HI1I, HI1T, HI2, HI2I, HI2T, J1, J2, K1, K2, KM1, LOG_ROWS, M1, W1,
    W2,
};

/// The fixed-base chain's table offset `Z'' = 9·G` (every entry `d + 9 ≥ 1`).
const FIXED_TABLE_OFFSET: u64 = 9;

/// `(16^64 − 1)/15`: the offset count a base's 64 window adds accumulate to.
fn window_sum() -> BigUint {
    ((BigUint::from(1u32) << (4 * WINDOWS)) - BigUint::from(1u32)) / BigUint::from(15u32)
}

fn scale<A: AffineRepr>(point: A, scalar: &BigUint) -> A::Group {
    point.mul_bigint(scalar.to_u64_digits())
}

/// `−K = −(16^64·R'' + n·(16^64 − 1)/15·Z'')`: what a chain of `n` bases
/// accumulates from its offsets, negated.
fn offset_correction<A: AffineRepr>(r: A, z: A, bases: usize) -> A {
    let sixteen_pow = BigUint::from(1u32) << (4 * WINDOWS);
    let total = scale(r, &sixteen_pow) + scale(z, &(window_sum() * BigUint::from(bases)));
    (-total).into_affine()
}

/// A base of a chain: one of the check's MSM bases or a public constant point.
enum Operand<B, A> {
    Base(B),
    Constant(A),
}

/// One chain: bases with scalar wires, first `k` slot, accumulator start rows,
/// table offset rows and the constant correction of a fixed-base chain.
struct Chain<B, A> {
    bases: Vec<(Operand<B, A>, Wire)>,
    kbase: u32,
    init: Vec<RowId>,
    z0: Vec<RowId>,
    correction: Option<A>,
}

impl<B, A> Chain<B, A> {
    /// `k` slots used: bases, four doublings, the correction.
    fn slots(&self) -> u32 {
        self.bases.len() as u32 + 4 + u32::from(self.correction.is_some())
    }
}

/// Shared state of a lane's chains: the next table index, the first cell of
/// every input element and the fresh inputs' table indices.
struct Lane {
    table_base: u32,
    first_input: HashMap<InputElement, u32>,
    /// Fresh inputs (table index for G1, half cell for G2) and their elements.
    fresh: Vec<u32>,
    fresh_elements: Vec<InputElement>,
    acc_output: Option<u32>,
}

impl Lane {
    fn new() -> Self {
        Self {
            table_base: 0,
            first_input: HashMap::new(),
            fresh: Vec::new(),
            fresh_elements: Vec::new(),
            acc_output: None,
        }
    }
}

/// The G1 templates of a lane.
struct G1Templates {
    copy: Template,
    copy_neg: Template,
    add: Template,
    sub: Template,
    dbl: Template,
}

impl G1Templates {
    fn new() -> Self {
        Self {
            copy: g1_copy(false),
            copy_neg: g1_copy(true),
            add: g1_add(false),
            sub: g1_add(true),
            dbl: g1_dbl(),
        }
    }
}

/// The G2 templates of a lane.
struct G2Templates {
    copy: Template,
    copy_neg: Template,
    add: Template,
    sub: Template,
    dbl: Template,
}

impl G2Templates {
    fn new() -> Self {
        Self {
            copy: g2_copy(false),
            copy_neg: g2_copy(true),
            add: g2_add(false),
            sub: g2_add(true),
            dbl: g2_dbl(),
        }
    }
}

impl Builder {
    // ----- G1 (four-row ops in rows 12–15 of GT cells) ------------------

    /// The fixed-base chain `R = θ·G`, `Z0 = φ(R)`, then the four Straus
    /// chains; returns the cells whose rows 14–15 hold the chain outputs.
    pub(super) fn g1(
        &mut self,
        check: &FlattenedCheck,
        values: &WireValues,
        setup: &DorySetupInputs,
    ) -> [u32; 4] {
        let g = G1Affine::generator();
        let g_endo = G1Config::endomorphism_affine(&g);
        let z_fixed = scale(g, &BigUint::from(FIXED_TABLE_OFFSET)).into_affine();
        let chains = check.g1_chains();
        let templates = G1Templates::new();
        let mut lane = Lane::new();

        let r_chain = Chain {
            bases: vec![(Operand::Constant(g), Wire::Offset)],
            kbase: chains[0].bases.len() as u32 + 1 + 4,
            init: vec![self.constant(g.x), self.constant(g.y)],
            z0: vec![self.constant(z_fixed.x), self.constant(z_fixed.y)],
            correction: Some(offset_correction(g, z_fixed, 1)),
        };
        assert!(
            r_chain.kbase + r_chain.slots() <= 64,
            "G1 chain 0 leaves no room for the offset chain"
        );
        let r_cell = self.g1_chain(&r_chain, values, setup, &mut lane, &templates);
        self.g1_table_region(0..1, &r_chain.z0, &templates);
        let r_rows = vec![row(r_cell) + 14, row(r_cell) + 15];
        let z0_rows = self.g1_endo_op(r_cell);

        let main_tables = lane.table_base;
        let mut outputs = [0u32; 4];
        for (m, msm) in chains.iter().enumerate() {
            let mut bases: Vec<(Operand<G1Base, G1Affine>, Wire)> = msm
                .bases
                .iter()
                .map(|(base, wire)| (Operand::Base(*base), wire.clone()))
                .collect();
            let correction = offset_correction(g, g_endo, bases.len() + 1);
            bases.push((Operand::Constant(correction), Wire::Offset));
            let chain = Chain {
                bases,
                kbase: 64 * m as u32,
                init: r_rows.clone(),
                z0: z0_rows.to_vec(),
                correction: None,
            };
            assert!(chain.slots() <= 64);
            outputs[m] = self.g1_chain(&chain, values, setup, &mut lane, &templates);
            if m == 0 {
                lane.acc_output = Some(outputs[0]);
            }
        }
        // One table family set for the main chains (they share `Z0`).
        self.g1_table_region(main_tables..lane.table_base, &z0_rows, &templates);

        let three = self.constant(Fq::from(3u64));
        let one = self.one();
        let curve = g1_on_curve();
        let mut mask = vec![0i32; 64];
        for b in &lane.fresh {
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
        let ops: Vec<(RowId, Option<u8>)> = lane
            .fresh
            .iter()
            .map(|b| (row(Cells::G1_INPUT + b), None))
            .collect();
        self.place(&family, &ops, false);
        self.g1_sign_rows(&lane);
        outputs
    }

    /// Sign rows of the fresh G1 inputs at `G1_SIGN + b` (row 0).
    fn g1_sign_rows(&mut self, lane: &Lane) {
        let half = self.constant(half_plus_one());
        let one = self.one();
        let sign = g1_sign();
        let mut mask = vec![0i32; 64];
        for b in &lane.fresh {
            mask[*b as usize] = 1;
        }
        let hi_s = hi(Cells::G1_SIGN, HI1I);
        let family = Family {
            name: "g1_sign",
            template: &sign,
            elems: vec![
                ElemRel::structured(
                    vec![
                        Factor::same(B1I, B1I),
                        Factor::constant(HI1I, hi(Cells::G1_INPUT, HI1I)),
                    ],
                    C,
                    12,
                ),
                ElemRel::Rows(vec![half, one]),
            ],
            own_bits: C,
            own_offset: 0,
            domain: vec![
                Factor::weight(B1I, mask),
                Factor::restrict(HI1I, hi_s..hi_s + 1),
            ],
        };
        let ops: Vec<(RowId, Option<u8>)> = lane
            .fresh
            .iter()
            .map(|b| (row(Cells::G1_SIGN + b), None))
            .collect();
        self.place(&family, &ops, false);
        for (b, element) in lane.fresh.iter().zip(&lane.fresh_elements) {
            self.sign_rows.push((*element, row(Cells::G1_SIGN + b)));
        }
    }

    /// `Z0 = φ(R)` in rows 0–1 of a glue cell.
    fn g1_endo_op(&mut self, r_cell: u32) -> [RowId; 2] {
        let zeta = self.constant(G1Config::ENDO_COEFFS[0]);
        let one = self.one();
        let cell = self.glue_cell();
        let endo = g1_endo();
        let family = Family {
            name: "g1_endo",
            template: &endo,
            elems: vec![
                Self::table_elem(CELL, CELL, vec![(cell, r_cell)], C, 14),
                ElemRel::Rows(vec![zeta, one]),
            ],
            own_bits: C,
            own_offset: 0,
            domain: vec![],
        };
        self.place(&family, &[(row(cell), None)], false);
        [row(cell), row(cell) + 1]
    }

    /// Base cells, the table and the online ops of one G1 chain at `k` slots
    /// `kbase..`; returns the cell whose rows 14–15 hold the output.
    fn g1_chain(
        &mut self,
        chain: &Chain<G1Base, G1Affine>,
        values: &WireValues,
        setup: &DorySetupInputs,
        lane: &mut Lane,
        templates: &G1Templates,
    ) -> u32 {
        let G1Templates {
            copy,
            copy_neg,
            add,
            dbl,
            ..
        } = templates;
        let table_base = lane.table_base;
        let n = chain.bases.len() as u32;
        let kbase = chain.kbase;
        let m32 = kbase / 64;
        let cell = |k: u32, w: u32| 64 * (kbase + k) + w;
        let digit_table: Vec<[u8; WINDOWS]> = chain
            .bases
            .iter()
            .map(|(_, wire)| digits(values.get(wire)))
            .collect();
        // Inputs.
        for (i, (base, _)) in chain.bases.iter().enumerate() {
            let b = table_base + i as u32;
            let input_cell = Cells::G1_INPUT + b;
            let point_rows = [row(input_cell) + 12, row(input_cell) + 13];
            let pin = |builder: &mut Self, point: G1Affine| {
                builder.program.pinned_constant_at(point_rows[0], point.x);
                builder.program.pinned_constant_at(point_rows[1], point.y);
            };
            let copy_from = |builder: &mut Self, name, template, source_rows: Vec<RowId>| {
                let family = Family {
                    name,
                    template,
                    elems: vec![ElemRel::Rows(source_rows), builder.ones()],
                    own_bits: C,
                    own_offset: 12,
                    domain: vec![Factor::restrict(CELL, input_cell..input_cell + 1)],
                };
                builder.place(&family, &[(row(input_cell), None)], false);
            };
            match base {
                Operand::Base(G1Base::Input(element)) => {
                    if let Some(&source) = lane.first_input.get(element) {
                        copy_from(
                            self,
                            "g1_input_copy",
                            copy,
                            vec![row(source) + 12, row(source) + 13],
                        );
                    } else {
                        let _ = lane.first_input.insert(*element, input_cell);
                        for r in point_rows {
                            let index = self.program.input_rows.len();
                            self.program.input_at(r, index);
                        }
                        self.input_order.push(*element);
                        lane.fresh.push(b);
                        lane.fresh_elements.push(*element);
                    }
                }
                Operand::Base(G1Base::Gamma1Zero) => pin(self, setup.g1_0),
                Operand::Base(G1Base::NegAcc) => {
                    let source = lane
                        .acc_output
                        .unwrap_or_else(|| unreachable!("the accumulator chain runs first"));
                    copy_from(
                        self,
                        "g1_neg_acc",
                        copy_neg,
                        vec![row(source) + 14, row(source) + 15],
                    );
                }
                Operand::Constant(point) => pin(self, *point),
            }
        }
        // Tables: `(j − 8)·P + Z0` at rows 14–15 of `G1_TABLE + 16·b + j`
        // (families recorded per lane, see [`Self::g1_table_region`]).
        let b_range = table_base..table_base + n;
        self.g1_table_ops(b_range.clone(), &chain.z0, templates);
        let hi1 = hi(Cells::G1_TABLE, HI1T);

        // Online: doublings then adds per window, then the fixed correction.
        let k_field = |k: u32| kbase + k;
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
                Factor::shift(K1, B1, i64::from(table_base) - i64::from(kbase % 64))
                    .with_range(kbase % 64..kbase % 64 + n),
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
            template: add,
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
            template: add,
            elems: vec![acc_prev(-1, 1, n), selected, self.ones()],
            own_bits: C,
            own_offset: 12,
            domain: vec![restrict(1, n)],
        };
        let dbl_init = Family {
            name: "g1_dbl_init",
            template: dbl,
            elems: vec![ElemRel::Rows(chain.init.clone()), self.ones()],
            own_bits: C,
            own_offset: 12,
            domain: vec![restrict(n, n + 1), Factor::restrict(W1, 0..1)],
        };
        let dbl0 = Family {
            name: "g1_dbl0",
            template: dbl,
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
            template: dbl,
            elems: vec![acc_prev(-1, n + 1, n + 4), self.ones()],
            own_bits: C,
            own_offset: 12,
            domain: vec![restrict(n + 1, n + 4)],
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
                let j = digit_table[k as usize][WINDOWS - 1 - w as usize];
                self.digit_ops.push(DigitOp {
                    first_row: row(cell(k, w)) + 13,
                    rows: 3,
                    kind: ReadKind::G1,
                    family: self.selected.len() as u8,
                    j,
                    kd: self.digit_index(&chain.bases[k as usize].1),
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
        self.record(&dbl_init, c_init, false);
        self.record(&dbl0, c_dbl0, false);
        self.record(&dbls, c_dbl, false);
        self.record(&add0, c_add0, true);
        self.record(&adds, c_add, true);
        let mut output = cell(n - 1, 63);
        if let Some(correction) = chain.correction {
            let [cx, cy] = [self.constant(correction.x), self.constant(correction.y)];
            let corr = Family {
                name: "g1_corr",
                template: add,
                elems: vec![
                    ElemRel::structured(vec![const_k(n - 1, n + 4), Factor::same(W1, W1)], C, 14),
                    ElemRel::Rows(vec![cx, cy]),
                    self.ones(),
                ],
                own_bits: C,
                own_offset: 12,
                domain: vec![restrict(n + 4, n + 5), Factor::restrict(W1, 63..64)],
            };
            output = cell(n + 4, 63);
            corr.emit(&mut self.program, row(output), None);
            self.record(&corr, 1, false);
        }
        // Reference entry `j = 8` of base `b = table_base + k`: row
        // `16·(G1_TABLE + 16b + 8) + c` from op row `16·(64·(kbase + k) + w) + c`,
        // with the `KM1` field holding `kbase + k`.
        self.selected.push(SelectedFamily {
            kind: ReadKind::G1,
            domain: vec![Factor::restrict(C, 13..16), restrict(0, n)],
            c_bits: C,
            first_c: 13,
            rows: 3,
            k_bits: KM1,
            w_bits: W1,
            key: KeyBase {
                constant: 16 * i64::from(Cells::G1_TABLE)
                    + 128
                    + 256 * (i64::from(table_base) - i64::from(kbase)),
                k_coeff: -768,
                w_coeff: -16,
            },
            digit_base: (0..n)
                .map(|k| (k_field(k), self.digit_index(&chain.bases[k as usize].1)))
                .collect(),
        });
        lane.table_base += n;
        output
    }
}

impl Builder {
    /// The `Z0` copy, `+P` and `−P` table families over the bases `b_range`.
    fn g1_table_families<'t>(
        &self,
        b_range: Range<u32>,
        z0: &[RowId],
        templates: &'t G1Templates,
    ) -> [Family<'t>; 3] {
        let hi1 = hi(Cells::G1_TABLE, HI1T);
        let in_table = |j: Range<u32>| {
            vec![
                Factor::restrict(J1, j),
                Factor::restrict(B1, b_range.clone()),
                Factor::restrict(HI1T, hi1..hi1 + 1),
            ]
        };
        let z0_family = Family {
            name: "g1_table_z0",
            template: &templates.copy,
            elems: vec![ElemRel::Rows(z0.to_vec()), self.ones()],
            own_bits: C,
            own_offset: 14,
            domain: in_table(8..9),
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
            template: &templates.add,
            elems: vec![prev(-1, 9..16), p_elem.clone(), self.ones()],
            own_bits: C,
            own_offset: 12,
            domain: in_table(9..16),
        };
        let down = Family {
            name: "g1_table_down",
            template: &templates.sub,
            elems: vec![prev(1, 0..8), p_elem, self.ones()],
            own_bits: C,
            own_offset: 12,
            domain: in_table(0..8),
        };
        [z0_family, up, down]
    }

    /// Emits the table ops of `b_range`.
    fn g1_table_ops(&mut self, b_range: Range<u32>, z0: &[RowId], templates: &G1Templates) {
        let tcell = |b: u32, j: u32| Cells::G1_TABLE + 16 * b + j;
        let [z0_family, up, down] = self.g1_table_families(b_range.clone(), z0, templates);
        for b in b_range {
            z0_family.emit(&mut self.program, row(tcell(b, 8)), None);
            for j in 9..16 {
                up.emit(&mut self.program, row(tcell(b, j)), None);
            }
            for j in (0..8).rev() {
                down.emit(&mut self.program, row(tcell(b, j)), None);
            }
        }
    }

    /// Records the table families and the fingerprinted region of `b_range`.
    fn g1_table_region(&mut self, b_range: Range<u32>, z0: &[RowId], templates: &G1Templates) {
        let nb = b_range.len();
        let [z0_family, up, down] = self.g1_table_families(b_range.clone(), z0, templates);
        self.record(&z0_family, nb, false);
        self.record(&up, 7 * nb, false);
        self.record(&down, 8 * nb, false);
        let hi1 = hi(Cells::G1_TABLE, HI1T);
        let entries = b_range
            .clone()
            .flat_map(|b| (0..16u32).map(move |j| row(Cells::G1_TABLE + 16 * b + j)));
        self.table_region(
            vec![
                Factor::restrict(B1, b_range),
                Factor::restrict(HI1T, hi1..hi1 + 1),
            ],
            CELL,
            entries,
            &templates.add,
            C,
            12,
            14,
            FP_SLOTS_G1,
        );
    }
}

impl Builder {
    // ----- G2 (eight-row ops, two per cell) ------------------------------

    /// The fixed-base chain `R = θ·G2`, `Z0 = φ(R)`, the two Straus chains,
    /// then the Miller `Q` inputs (`E2_fin` committed, `B2` copied into input
    /// layout). Returns the half cells of the chain outputs (point at rows
    /// 4–7) and of the two `Q` inputs (rows 0–3).
    pub(super) fn g2(
        &mut self,
        check: &FlattenedCheck,
        values: &WireValues,
        setup: &DorySetupInputs,
    ) -> ([u32; 2], [u32; 2]) {
        let g = G2Affine::generator();
        let g_endo = G2Config::endomorphism_affine(&g);
        let z_fixed = scale(g, &BigUint::from(FIXED_TABLE_OFFSET)).into_affine();
        let chains = check.g2_chains();
        let templates = G2Templates::new();
        let mut lane = Lane::new();

        let r_chain = Chain {
            bases: vec![(Operand::Constant(g), Wire::Offset)],
            kbase: Cells::G2_OFFSET_K,
            init: [self.fq2_constant(g.x), self.fq2_constant(g.y)].concat(),
            z0: [self.fq2_constant(z_fixed.x), self.fq2_constant(z_fixed.y)].concat(),
            correction: Some(offset_correction(g, z_fixed, 1)),
        };
        assert!(r_chain.kbase + r_chain.slots() <= 64);
        let r_half = self.g2_chain(
            &r_chain,
            Cells::G2_OFFSET_HI,
            values,
            setup,
            &mut lane,
            &templates,
        );
        self.g2_table_region(0..1, &r_chain.z0, &templates);
        let r_rows: Vec<RowId> = (4..8).map(|c| half_row(r_half) + c).collect();
        let z0_rows = self.g2_endo_op(r_half);

        let hi_o = hi(Cells::G2_ONLINE_HALF / 2, HI2);
        let main_tables = lane.table_base;
        let mut outputs = [0u32; 2];
        let mut kbase = 0u32;
        for (m, msm) in chains.iter().enumerate() {
            let mut bases: Vec<(Operand<G2Base, G2Affine>, Wire)> = msm
                .bases
                .iter()
                .map(|(base, wire)| (Operand::Base(*base), wire.clone()))
                .collect();
            let correction = offset_correction(g, g_endo, bases.len() + 1);
            bases.push((Operand::Constant(correction), Wire::Offset));
            let chain = Chain {
                bases,
                kbase,
                init: r_rows.clone(),
                z0: z0_rows.to_vec(),
                correction: None,
            };
            kbase += chain.slots();
            assert!(kbase <= 64, "G2 chains exceed the online region");
            outputs[m] = self.g2_chain(&chain, hi_o, values, setup, &mut lane, &templates);
            if m == 0 {
                lane.acc_output = Some(outputs[0]);
            }
        }
        self.g2_table_region(main_tables..lane.table_base, &z0_rows, &templates);

        // Miller `Q` inputs: `E2_fin` (committed) and a copy of `B2` in input layout.
        let input_half = |b: u32| Cells::G2_INPUT_HALF + b;
        let e2_fin = input_half(lane.table_base);
        self.g2_input_leaf(e2_fin, InputElement::FinalE2);
        lane.fresh.push(e2_fin);
        lane.fresh_elements.push(InputElement::FinalE2);
        let b_curve = self.fq2_constant(<G2Config as SWCurveConfig>::COEFF_B);
        let one = self.one();
        let curve = g2_on_curve();
        self.g2_on_curve_family(&lane.fresh, &curve, &b_curve, one);
        self.g2_sign_rows(&lane);
        let b2 = input_half(lane.table_base + 1);
        let family = Family {
            name: "g2_b2_copy",
            template: &templates.copy,
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

    /// Sign rows of the fresh G2 inputs at cells `G2_SIGN + b` (rows 0–5).
    fn g2_sign_rows(&mut self, lane: &Lane) {
        let half = self.constant(half_plus_one());
        let one = self.one();
        let sign = g2_sign();
        let mut mask = vec![0i32; 64];
        for h in &lane.fresh {
            mask[(h - Cells::G2_INPUT_HALF) as usize] = 1;
        }
        let hi_s = hi(Cells::G2_SIGN, HI1I);
        let hi_i = hi(Cells::G2_INPUT_HALF / 2, HI2I);
        let family = Family {
            name: "g2_sign",
            template: &sign,
            elems: vec![
                ElemRel::structured(
                    vec![Factor::same(B1I, B2I), Factor::constant(HI2I, hi_i)],
                    C3,
                    0,
                ),
                ElemRel::Rows(vec![half, one]),
            ],
            own_bits: C,
            own_offset: 0,
            domain: vec![
                Factor::weight(B1I, mask),
                Factor::restrict(HI1I, hi_s..hi_s + 1),
            ],
        };
        let ops: Vec<(RowId, Option<u8>)> = lane
            .fresh
            .iter()
            .map(|h| (row(Cells::G2_SIGN + (h - Cells::G2_INPUT_HALF)), None))
            .collect();
        self.place(&family, &ops, false);
        for (h, element) in lane.fresh.iter().zip(&lane.fresh_elements) {
            let cell = Cells::G2_SIGN + (h - Cells::G2_INPUT_HALF);
            self.sign_rows.push((*element, row(cell) + 5));
        }
    }

    /// `Z0 = φ(R)` in rows 0–3 of a glue cell.
    fn g2_endo_op(&mut self, r_half: u32) -> [RowId; 4] {
        let omega = self.fq2_constant(G2Config::ENDO_COEFFS[0]);
        let one = self.one();
        let cell = self.glue_cell();
        let endo = g2_endo();
        let family = Family {
            name: "g2_endo",
            template: &endo,
            elems: vec![
                Self::table_elem(CELL, Bits::new(3, 18), vec![(cell, r_half)], C3, 4),
                ElemRel::Rows(vec![omega[0], omega[1], one]),
            ],
            own_bits: C,
            own_offset: 0,
            domain: vec![],
        };
        self.place(&family, &[(row(cell), None)], false);
        std::array::from_fn(|c| row(cell) + c as u32)
    }

    /// Base half cells, the table and the online ops of one G2 chain in the
    /// `HI2 = hi` region at `k` slots `kbase..`; returns the half cell whose
    /// rows 4–7 hold the output.
    fn g2_chain(
        &mut self,
        chain: &Chain<G2Base, G2Affine>,
        hi_o: u32,
        values: &WireValues,
        setup: &DorySetupInputs,
        lane: &mut Lane,
        templates: &G2Templates,
    ) -> u32 {
        let G2Templates {
            copy,
            copy_neg,
            add,
            dbl,
            ..
        } = templates;
        let hi_t = hi(Cells::G2_TABLE_HALF / 2, HI2T);
        let input_half = |b: u32| Cells::G2_INPUT_HALF + b;
        let region_half = hi_o << 12;
        let table_base = lane.table_base;
        let n = chain.bases.len() as u32;
        let kb = chain.kbase;
        let half = |k: u32, w: u32| region_half + 64 * (kb + k) + w;
        let digit_table: Vec<[u8; WINDOWS]> = chain
            .bases
            .iter()
            .map(|(_, wire)| digits(values.get(wire)))
            .collect();
        for (i, (base, _)) in chain.bases.iter().enumerate() {
            let b = table_base + i as u32;
            let h = input_half(b);
            let pin = |builder: &mut Self, point: G2Affine| {
                for (c, v) in [point.x.c0, point.x.c1, point.y.c0, point.y.c1]
                    .iter()
                    .enumerate()
                {
                    builder
                        .program
                        .pinned_constant_at(half_row(h) + c as u32, *v);
                }
            };
            let copy_from = |builder: &mut Self, name, template, source_rows: Vec<RowId>| {
                let family = Family {
                    name,
                    template,
                    elems: vec![ElemRel::Rows(source_rows), builder.ones()],
                    own_bits: C3,
                    own_offset: 0,
                    domain: vec![Factor::restrict(Bits::new(3, 18), h..h + 1)],
                };
                builder.place(&family, &[(half_row(h), None)], false);
            };
            match base {
                Operand::Base(G2Base::Input(element)) => {
                    if let Some(&source) = lane.first_input.get(element) {
                        copy_from(
                            self,
                            "g2_input_copy",
                            copy,
                            (0..4).map(|c| half_row(source) + c).collect(),
                        );
                    } else {
                        let _ = lane.first_input.insert(*element, h);
                        self.g2_input_leaf(h, *element);
                        lane.fresh.push(h);
                        lane.fresh_elements.push(*element);
                    }
                }
                Operand::Base(G2Base::Gamma2Zero) => pin(self, setup.g2_0),
                Operand::Base(G2Base::NegAcc) => {
                    let source = lane
                        .acc_output
                        .unwrap_or_else(|| unreachable!("the accumulator chain runs first"));
                    copy_from(
                        self,
                        "g2_neg_acc",
                        copy_neg,
                        (4..8).map(|c| half_row(source) + c).collect(),
                    );
                }
                Operand::Constant(point) => pin(self, *point),
            }
        }
        // Tables at half cells `G2_TABLE_HALF + 16·b + j`, points at rows 4–7
        // (families recorded per lane, see [`Self::g2_table_region`]).
        let b_range = table_base..table_base + n;
        self.g2_table_ops(b_range.clone(), &chain.z0, templates);

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
                Factor::shift(K2, B2, i64::from(table_base) - i64::from(kb)).with_range(kb..kb + n),
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
            template: add,
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
            template: add,
            elems: vec![acc_prev(-1, 1, n), selected, self.ones()],
            own_bits: C3,
            own_offset: 0,
            domain: vec![restrict(1, n), region.clone()],
        };
        let dbl_init = Family {
            name: "g2_dbl_init",
            template: dbl,
            elems: vec![ElemRel::Rows(chain.init.clone()), self.ones()],
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
            template: dbl,
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
            template: dbl,
            elems: vec![acc_prev(-1, n + 1, n + 4), self.ones()],
            own_bits: C3,
            own_offset: 0,
            domain: vec![restrict(n + 1, n + 4), region.clone()],
        };
        let (mut c_init, mut c_dbl0, mut c_dbl, mut c_add0, mut c_add) = (0, 0, 0, 0, 0);
        for w in 0..WINDOWS as u32 {
            if w == 0 {
                dbl_init.emit(&mut self.program, half_row(half(n, w)), None);
                c_init += 1;
            } else {
                dbl0.emit(&mut self.program, half_row(half(n, w)), None);
                c_dbl0 += 1;
            }
            for i in 1..4 {
                dbls.emit(&mut self.program, half_row(half(n + i, w)), None);
                c_dbl += 1;
            }
            for k in 0..n {
                let j = digit_table[k as usize][WINDOWS - 1 - w as usize];
                self.digit_ops.push(DigitOp {
                    first_row: half_row(half(k, w)) + 2,
                    rows: 6,
                    kind: ReadKind::G2,
                    family: self.selected.len() as u8,
                    j,
                    kd: self.digit_index(&chain.bases[k as usize].1),
                    w,
                });
                if k == 0 {
                    add0.emit(&mut self.program, half_row(half(k, w)), Some(j));
                    c_add0 += 1;
                } else {
                    adds.emit(&mut self.program, half_row(half(k, w)), Some(j));
                    c_add += 1;
                }
            }
        }
        self.record(&dbl_init, c_init, false);
        self.record(&dbl0, c_dbl0, false);
        self.record(&dbls, c_dbl, false);
        self.record(&add0, c_add0, true);
        self.record(&adds, c_add, true);
        let mut output = half(n - 1, 63);
        if let Some(correction) = chain.correction {
            let c_rows = [
                self.fq2_constant(correction.x),
                self.fq2_constant(correction.y),
            ]
            .concat();
            let corr = Family {
                name: "g2_corr",
                template: add,
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
            output = half(n + 4, 63);
            corr.emit(&mut self.program, half_row(output), None);
            self.record(&corr, 1, false);
        }
        // Reference entry `j = 8` of base `b = table_base + k`: row
        // `8·(G2_TABLE_HALF + 16b + 8) + c` from op row
        // `8·(region_half + 64·k2 + w) + c`, `k2 = kb + k` the `K2` field.
        self.selected.push(SelectedFamily {
            kind: ReadKind::G2,
            domain: vec![Factor::restrict(C3, 2..8), restrict(0, n), region.clone()],
            c_bits: C3,
            first_c: 2,
            rows: 6,
            k_bits: K2,
            w_bits: W2,
            key: KeyBase {
                constant: 8 * (i64::from(Cells::G2_TABLE_HALF) - i64::from(region_half))
                    + 64
                    + 128 * (i64::from(table_base) - i64::from(kb)),
                k_coeff: -(8 * 64 - 8 * 16),
                w_coeff: -8,
            },
            digit_base: (0..n)
                .map(|k| (kb + k, self.digit_index(&chain.bases[k as usize].1)))
                .collect(),
        });
        lane.table_base += n;
        output
    }

    //    /// The `Z0` copy, `+P` and `−P` table families over the bases `b_range`.
    fn g2_table_families<'t>(
        &self,
        b_range: Range<u32>,
        z0: &[RowId],
        templates: &'t G2Templates,
    ) -> [Family<'t>; 3] {
        let hi_t = hi(Cells::G2_TABLE_HALF / 2, HI2T);
        let hi_i = hi(Cells::G2_INPUT_HALF / 2, HI2I);
        let in_table = |j: Range<u32>| {
            vec![
                Factor::restrict(J2, j),
                Factor::restrict(B2, b_range.clone()),
                Factor::restrict(HI2T, hi_t..hi_t + 1),
            ]
        };
        let z0_family = Family {
            name: "g2_table_z0",
            template: &templates.copy,
            elems: vec![ElemRel::Rows(z0.to_vec()), self.ones()],
            own_bits: C3,
            own_offset: 4,
            domain: in_table(8..9),
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
            template: &templates.add,
            elems: vec![prev(-1, 9..16), p_elem.clone(), self.ones()],
            own_bits: C3,
            own_offset: 0,
            domain: in_table(9..16),
        };
        let down = Family {
            name: "g2_table_down",
            template: &templates.sub,
            elems: vec![prev(1, 0..8), p_elem, self.ones()],
            own_bits: C3,
            own_offset: 0,
            domain: in_table(0..8),
        };
        [z0_family, up, down]
    }

    /// Emits the table ops of `b_range`.
    fn g2_table_ops(&mut self, b_range: Range<u32>, z0: &[RowId], templates: &G2Templates) {
        let thalf = |b: u32, j: u32| Cells::G2_TABLE_HALF + 16 * b + j;
        let [z0_family, up, down] = self.g2_table_families(b_range.clone(), z0, templates);
        for b in b_range {
            z0_family.emit(&mut self.program, half_row(thalf(b, 8)), None);
            for j in 9..16 {
                up.emit(&mut self.program, half_row(thalf(b, j)), None);
            }
            for j in (0..8).rev() {
                down.emit(&mut self.program, half_row(thalf(b, j)), None);
            }
        }
    }

    /// Records the table families and the fingerprinted region of `b_range`.
    fn g2_table_region(&mut self, b_range: Range<u32>, z0: &[RowId], templates: &G2Templates) {
        let nb = b_range.len();
        let [z0_family, up, down] = self.g2_table_families(b_range.clone(), z0, templates);
        self.record(&z0_family, nb, false);
        self.record(&up, 7 * nb, false);
        self.record(&down, 8 * nb, false);
        let hi_t = hi(Cells::G2_TABLE_HALF / 2, HI2T);
        let entries = b_range
            .clone()
            .flat_map(|b| (0..16u32).map(move |j| half_row(Cells::G2_TABLE_HALF + 16 * b + j)));
        self.table_region(
            vec![
                Factor::restrict(B2, b_range),
                Factor::restrict(HI2T, hi_t..hi_t + 1),
            ],
            Bits::new(3, LOG_ROWS as u8),
            entries,
            &templates.add,
            C3,
            0,
            4,
            FP_SLOTS_G2,
        );
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

// ----- G2 subgroup checks (ψ-chains) ----------------------------------

/// Local half cell within a chain's `256`-cell box.
const PSI_LH: Bits = Bits::new(3, 12);
/// Local cell within the box.
const PSI_CL: Bits = Bits::new(4, 12);
/// Local half / cell with the chain bit.
const PSI_LHC: Bits = Bits::new(3, 13);
const PSI_HI: Bits = Bits::new(13, 18);
const HALF_PARITY: Bits = Bits::new(3, 4);
const HALF: Bits = Bits::new(3, 18);

impl Builder {
    /// `ψ²(P) + ψ([6x+3]P) + [6x+1]P = 0` for both proof-derived Miller `Q`
    /// inputs `P` (rows 0–3 of `q_halves`), the G2 membership identity of the
    /// BN twist. `[6x+1]P` is accumulated from the powers `2^i·P` (doublings
    /// at consecutive half cells) by guarded adds of the NAF terms, each
    /// pinned to the generic affine case by `inv·(x2 − x1) = 1`: a point of
    /// the twist outside G2 fails the identity, and no exceptional add can
    /// hide that because every add has a witness only when `x2 ≠ x1` (a
    /// doubling has one unless the point has order 2, which the twist's odd
    /// group order excludes). `[6x+3]P = [6x+1]P + 2P`; the tail applies `ψ²`
    /// and `ψ`, adds them (guarded) and pins the sum to `−[6x+1]P`.
    pub(super) fn g2_subgroup_checks(&mut self, q_halves: [u32; 2]) {
        let x = u128::from(Bn254Config::X[0]);
        let terms = naf(6 * x + 1);
        let top = terms.len() - 1;
        assert_eq!(terms[top], 1);
        // `P` sits at half `start` so the top power lands on an even half
        // (the first half of its cell, where the adds' `prev` element reads).
        let start = (top % 2) as u32;
        let power_half = |i: usize| start + i as u32;
        let mut adds: Vec<(u32, bool)> = terms[..top]
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, d)| **d != 0)
            .map(|(i, d)| (power_half(i), *d < 0))
            .collect();
        // `[6x+3]P = [6x+1]P + 2P`.
        adds.push((power_half(1), false));
        let first_add_cell = u32::midpoint(start, top as u32) + 1;
        let a_cell = first_add_cell + adds.len() as u32 - 2;
        let b_cell = a_cell + 1;
        let (t0, t1, t2) = (b_cell + 1, b_cell + 2, b_cell + 3);
        assert!(t2 < 256, "ψ-chain exceeds its box");

        let one = self.one();
        let hi_c = hi(Cells::PSI_CHAIN, PSI_HI);
        let half_of = |chain: u32, local: u32| 2 * Cells::PSI_CHAIN + 512 * chain + local;
        let cell_of = |chain: u32, local: u32| Cells::PSI_CHAIN + 256 * chain + local;
        let in_box = Factor::restrict(PSI_HI, hi_c..hi_c + 1);
        let copy = g2_copy(false);
        let dbl = g2_dbl();
        let add = g2_add_guarded(false);
        let sub = g2_add_guarded(true);
        let (cx1, cy1) = psi_coefficients(1);
        let (cx2, cy2) = psi_coefficients(2);
        let psi1_consts = ElemRel::Rows([self.fq2_constant(cx1), self.fq2_constant(cy1)].concat());
        let psi2_consts = ElemRel::Rows([self.fq2_constant(cx2), self.fq2_constant(cy2)].concat());
        let psi = g2_psi(false, true);
        let psi_even = g2_psi(false, false);
        let pins = g2_negation_pins();

        // `P` copies (rows 4–7 of half `start`).
        let mut copy_mask = vec![0i32; 1 << PSI_LHC.width()];
        let mut copy_pairs = Vec::new();
        for (chain, q) in q_halves.iter().enumerate() {
            let local = PSI_LHC.extract(half_row(half_of(chain as u32, start)));
            copy_mask[local as usize] = 1;
            copy_pairs.push((local, *q));
        }
        let p_copy = Family {
            name: "psi_p_copy",
            template: &copy,
            elems: vec![
                ElemRel::structured(vec![Factor::table(PSI_LHC, HALF, copy_pairs)], C3, 0),
                self.ones(),
            ],
            own_bits: C3,
            own_offset: 4,
            domain: vec![Factor::weight(PSI_LHC, copy_mask), in_box.clone()],
        };
        // Doublings `2^i·P` at halves `start + i`, `i = 1..=top`.
        let dbl_range = start + 1..start + top as u32 + 1;
        let dbls = Family {
            name: "psi_dbl",
            template: &dbl,
            elems: vec![
                ElemRel::structured(
                    vec![
                        Factor::shift(PSI_LH, PSI_LH, -1).with_range(dbl_range.clone()),
                        Factor::same(Bits::new(12, 13), Bits::new(12, 13)),
                        Factor::constant(PSI_HI, hi_c),
                    ],
                    C3,
                    4,
                ),
                self.ones(),
            ],
            own_bits: C3,
            own_offset: 0,
            domain: vec![Factor::restrict(PSI_LH, dbl_range), in_box.clone()],
        };
        // Guarded adds at cells `first_add_cell..`: `prev` is the first half
        // of the previous cell (the top power, then the previous sum), the
        // term is the power's half.
        let prev = ElemRel::structured(
            vec![
                Factor::shift(PSI_CL, PSI_CL, -1),
                Factor::constant(HALF_PARITY, 0),
                Factor::same(Bits::new(12, 13), Bits::new(12, 13)),
                Factor::constant(PSI_HI, hi_c),
            ],
            C3,
            4,
        );
        // Both chains share the local cell pattern: masks and tables are over
        // the local fields, the chain bit carries over unchanged.
        let mut masks = [
            vec![0i32; 1 << PSI_CL.width()],
            vec![0i32; 1 << PSI_CL.width()],
        ];
        let mut pairs: [Vec<(u32, u32)>; 2] = [Vec::new(), Vec::new()];
        for (j, (power, negate)) in adds.iter().enumerate() {
            let local = first_add_cell + j as u32;
            let side = usize::from(*negate);
            masks[side][local as usize] = 1;
            pairs[side].push((local, *power));
        }
        let add_family = |name, template, mask: Vec<i32>, pairs: Vec<(u32, u32)>| Family {
            name,
            template,
            elems: vec![
                prev.clone(),
                ElemRel::structured(
                    vec![
                        Factor::table(PSI_CL, PSI_LH, pairs),
                        Factor::same(Bits::new(12, 13), Bits::new(12, 13)),
                        Factor::constant(PSI_HI, hi_c),
                    ],
                    C3,
                    4,
                ),
                ElemRel::Rows(vec![one]),
            ],
            own_bits: C,
            own_offset: 0,
            domain: vec![Factor::weight(PSI_CL, mask), in_box.clone()],
        };
        let [mask_add, mask_sub] = masks;
        let [pairs_add, pairs_sub] = pairs;
        let add_ops = add_family("psi_add", &add, mask_add, pairs_add);
        let sub_ops = add_family("psi_sub", &sub, mask_sub, pairs_sub);
        // Tail: `ψ²(P)` (rows 0–3 of `t0`), `ψ(B)` (rows 4–7), `S = ψ²(P) + ψ(B)`
        // (`t1`), pins `S + A = 0` (`t2`).
        let pow2 = Family {
            name: "psi_pow2",
            template: &psi_even,
            elems: vec![
                Self::table_elem(
                    CELL,
                    HALF,
                    (0..2)
                        .map(|c| (cell_of(c as u32, t0), q_halves[c]))
                        .collect(),
                    C3,
                    0,
                ),
                psi2_consts,
            ],
            own_bits: C,
            own_offset: 0,
            domain: vec![],
        };
        let psi_b = Family {
            name: "psi_b",
            template: &psi,
            elems: vec![
                ElemRel::structured(
                    vec![
                        Factor::table(
                            CELL,
                            CELL,
                            (0..2u32)
                                .map(|c| (cell_of(c, t0), cell_of(c, b_cell)))
                                .collect(),
                        ),
                        Factor::constant(HALF_PARITY, 0),
                    ],
                    C3,
                    4,
                ),
                psi1_consts,
            ],
            own_bits: C,
            own_offset: 4,
            domain: vec![],
        };
        let sum = Family {
            name: "psi_sum",
            template: &add,
            elems: vec![
                Self::table_elem(
                    CELL,
                    CELL,
                    (0..2u32)
                        .map(|c| (cell_of(c, t1), cell_of(c, t0)))
                        .collect(),
                    C,
                    0,
                ),
                Self::table_elem(
                    CELL,
                    CELL,
                    (0..2u32)
                        .map(|c| (cell_of(c, t1), cell_of(c, t0)))
                        .collect(),
                    C,
                    4,
                ),
                ElemRel::Rows(vec![one]),
            ],
            own_bits: C,
            own_offset: 0,
            domain: vec![],
        };
        let negation = Family {
            name: "psi_pins",
            template: &pins,
            elems: vec![
                Self::table_elem(
                    CELL,
                    CELL,
                    (0..2u32)
                        .map(|c| (cell_of(c, t2), cell_of(c, t1)))
                        .collect(),
                    C,
                    4,
                ),
                ElemRel::structured(
                    vec![
                        Factor::table(
                            CELL,
                            CELL,
                            (0..2u32)
                                .map(|c| (cell_of(c, t2), cell_of(c, a_cell)))
                                .collect(),
                        ),
                        Factor::constant(HALF_PARITY, 0),
                    ],
                    C3,
                    4,
                ),
                ElemRel::Rows(vec![one]),
            ],
            own_bits: C,
            own_offset: 0,
            domain: vec![],
        };
        let (mut n_copy, mut n_dbl, mut n_add, mut n_sub) = (0, 0, 0, 0);
        for chain in 0..2u32 {
            p_copy.emit(&mut self.program, half_row(half_of(chain, start)), None);
            n_copy += 1;
            for i in 1..=top as u32 {
                dbls.emit(&mut self.program, half_row(half_of(chain, start + i)), None);
                n_dbl += 1;
            }
            for (j, (_, negate)) in adds.iter().enumerate() {
                let base = row(cell_of(chain, first_add_cell + j as u32));
                if *negate {
                    sub_ops.emit(&mut self.program, base, None);
                    n_sub += 1;
                } else {
                    add_ops.emit(&mut self.program, base, None);
                    n_add += 1;
                }
            }
        }
        self.record(&p_copy, n_copy, false);
        self.record(&dbls, n_dbl, false);
        self.record(&add_ops, n_add, false);
        self.record(&sub_ops, n_sub, false);
        let tail: Vec<(RowId, Option<u8>)> =
            (0..2u32).map(|c| (row(cell_of(c, t0)), None)).collect();
        self.place(&pow2, &tail, false);
        self.place(&psi_b, &tail, false);
        let tail: Vec<(RowId, Option<u8>)> =
            (0..2u32).map(|c| (row(cell_of(c, t1)), None)).collect();
        self.place(&sum, &tail, false);
        let tail: Vec<(RowId, Option<u8>)> =
            (0..2u32).map(|c| (row(cell_of(c, t2)), None)).collect();
        self.place(&negation, &tail, false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::{BigInteger, One, PrimeField, Zero};

    fn modulus<A: AffineRepr>() -> BigUint {
        BigUint::from_bytes_be(&<A::ScalarField as PrimeField>::MODULUS.to_bytes_be())
    }

    /// `φ(G) = [λ]G` on both groups (the endomorphism the offsets rely on).
    #[test]
    fn glv_endomorphism_is_lambda() {
        let g1 = G1Affine::generator();
        assert_eq!(
            G1Config::endomorphism_affine(&g1),
            g1.mul_bigint(G1Config::LAMBDA.into_bigint()).into_affine()
        );
        let g2 = G2Affine::generator();
        assert_eq!(
            G2Config::endomorphism_affine(&g2),
            g2.mul_bigint(G2Config::LAMBDA.into_bigint()).into_affine()
        );
    }

    /// No `(window, add, chain size)` of the layout makes an exceptional
    /// affine case θ-independent: `16^{w+1} + λ(nw + k ∓ 1) ≠ 0` and
    /// `16^w + λnw ≠ 0` for every `w < 64`, `k < n ≤ 64`.
    #[test]
    fn offsets_are_nondegenerate() {
        for lambda in [G1Config::LAMBDA, G2Config::LAMBDA] {
            assert!(!lambda.is_zero());
            let sixteen = Fr::from(16u64);
            for n in 1..=64u64 {
                let mut power = Fr::one();
                for w in 0..WINDOWS as u64 {
                    let zeros = Fr::from(n * w);
                    assert_ne!(power + lambda * zeros, Fr::zero(), "doubling n={n} w={w}");
                    let before_add = power * sixteen;
                    for k in 0..n {
                        for sign in [Fr::one(), -Fr::one()] {
                            let coefficient = before_add + lambda * (zeros + Fr::from(k) - sign);
                            assert_ne!(coefficient, Fr::zero(), "add n={n} w={w} k={k}");
                        }
                    }
                    power *= sixteen;
                }
            }
        }
    }

    /// The G2 membership identity the ψ-chains pin: `ψ²(P) + ψ([6x+3]P) +
    /// [6x+1]P = 0` on the subgroup, false on a random point of the twist's
    /// cofactor torsion.
    #[test]
    fn psi_identity_holds_on_g2_only() {
        use super::super::super::ops::psi_coefficients;
        use ark_bn254::Fq2;
        use ark_ff::UniformRand;
        use rand_chacha::ChaCha20Rng;
        use rand_core::SeedableRng;
        let psi = |p: G2Affine, power: usize| -> G2Affine {
            let (cx, cy) = psi_coefficients(power);
            let conj = |mut v: Fq2| {
                for _ in 0..power {
                    let _ = v.conjugate_in_place();
                }
                v
            };
            G2Affine::new_unchecked(conj(p.x) * cx, conj(p.y) * cy)
        };
        let x = u128::from(<ark_bn254::Config as ark_ec::bn::BnConfig>::X[0]);
        let limbs = |s: u128| [s as u64, (s >> 64) as u64];
        let check = |p: G2Affine| {
            let a = p.mul_bigint(limbs(6 * x + 3)).into_affine();
            let b = p.mul_bigint(limbs(6 * x + 1)).into_affine();
            (psi(p, 2).into_group() + psi(a, 1).into_group() + b.into_group()).is_zero()
        };
        let mut rng = ChaCha20Rng::seed_from_u64(0x9D1);
        for _ in 0..4 {
            assert!(check(G2Affine::rand(&mut rng)));
        }
        // A point of the twist outside G2: random x, solve for y, no cofactor clearing.
        let outside = loop {
            let x = Fq2::rand(&mut rng);
            if let Some(p) = G2Affine::get_point_from_x_unchecked(x, true) {
                if !p.is_in_correct_subgroup_assuming_on_curve() {
                    break p;
                }
            }
        };
        assert!(!check(outside));
    }

    /// The fixed-base chain's integer regime: the accumulator multiplier of
    /// window `w ≤ 62` stays below `r`, so no wrap can align it with an entry.
    #[test]
    fn fixed_base_multipliers_stay_below_the_modulus() {
        let r = modulus::<G1Affine>();
        let sixteen = BigUint::from(16u32);
        // Largest multiplier before the add of window 62: every digit `7`.
        let mut multiplier = BigUint::one();
        for _ in 0..62 {
            multiplier = &multiplier * &sixteen + BigUint::from(7u32 + FIXED_TABLE_OFFSET as u32);
        }
        multiplier *= &sixteen;
        assert!(multiplier < r);
        assert_eq!(
            offset_correction(G1Affine::generator(), G1Affine::generator(), 1),
            (-(scale(G1Affine::generator(), &(BigUint::one() << 256))
                + scale(G1Affine::generator(), &window_sum())))
            .into_affine()
        );
    }
}
