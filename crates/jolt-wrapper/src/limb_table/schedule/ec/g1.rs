//! The G1 lane: the θ-offset Straus chains over the G1 pairing inputs,
//! their fixed-base tables and on-curve checks.

use super::{
    digits, g1_endo, g1_on_curve, hi, offset_correction, row, scale, AffineRepr, BigUint, Builder,
    Cells, Chain, CurveGroup, DigitOp, DigitRule, DorySetupInputs, ElemRel, Factor, Family,
    FlattenedCheck, Fq, G1Affine, G1Base, G1Config, G1Templates, GLVConfig, KeyBase, Lane, Operand,
    Range, ReadKind, Rel, RowId, SelectedFamily, Wire, WireValues, B1, B1I, C, CELL,
    FIXED_TABLE_OFFSET, FP_SLOTS_G1, HI1I, HI1T, J1, K1, KM1, M1, W1, WINDOWS,
};

impl Builder {
    // ----- G1 (four-row ops in rows 12–15 of GT cells) ------------------

    /// The fixed-base chain `R = θ·G`, `Z0 = φ(R)`, then the four Straus
    /// chains; returns the cells whose rows 14–15 hold the chain outputs.
    pub(in crate::limb_table::schedule) fn g1(
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
