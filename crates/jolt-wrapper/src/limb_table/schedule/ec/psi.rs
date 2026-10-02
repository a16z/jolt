//! G2 subgroup membership of the proof-derived pairing inputs through the
//! endomorphism identity `ψ²(P) + ψ([6x+3]P) + [6x+1]P = 0`.

use super::{
    g2_add_guarded, g2_copy, g2_dbl, g2_negation_pins, g2_psi, half_row, hi, naf, psi_coefficients,
    row, Bits, Bn254Config, BnConfig, Builder, Cells, ElemRel, Factor, Family, RowId, C, C3, CELL,
};

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
    pub(in crate::limb_table::schedule) fn g2_subgroup_checks(&mut self, q_halves: [u32; 2]) {
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
