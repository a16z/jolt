//! The G2 lane: the θ-offset Straus chains over the G2 pairing inputs,
//! their fixed-base tables, on-curve checks and input leaves.

use super::{
    digits, g2_endo, g2_on_curve, half_row, hi, offset_correction, row, scale, AffineRepr, BigUint,
    Bits, Builder, Cells, Chain, CurveGroup, DigitOp, DigitRule, DorySetupInputs, ElemRel, Factor,
    Family, FlattenedCheck, G2Affine, G2Base, G2Config, G2Templates, GLVConfig, InputElement,
    KeyBase, Lane, Operand, Range, ReadKind, Rel, RowId, SWCurveConfig, SelectedFamily, Template,
    Wire, WireValues, B2, B2I, C, C3, CELL, FIXED_TABLE_OFFSET, FP_SLOTS_G2, HI2, HI2I, HI2T, J2,
    K2, LOG_ROWS, W2, WINDOWS,
};

impl Builder {
    // ----- G2 (eight-row ops, two per cell) ------------------------------

    /// The fixed-base chain `R = θ·G2`, `Z0 = φ(R)`, the two Straus chains,
    /// then the Miller `Q` inputs (`E2_fin` committed, `B2` copied into input
    /// layout). Returns the half cells of the chain outputs (point at rows
    /// 4–7) and of the two `Q` inputs (rows 0–3).
    pub(in crate::limb_table::schedule) fn g2(
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
            guard: false,
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
                guard: true,
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
            guard,
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
        if chain.guard {
            // Slot `n + 4`: the guard of the last base's add (slot `n − 1`),
            // reading its `λ`/`x3` rows and the accumulator it added to.
            let guarded = |delta: i64, offset: u32| {
                ElemRel::structured(
                    vec![
                        Factor::shift(K2, K2, delta).with_range(kb + n + 4..kb + n + 5),
                        Factor::same(W2, W2),
                        Factor::constant(HI2, hi_o),
                    ],
                    C3,
                    offset,
                )
            };
            let guards = Family {
                name: "g2_guard",
                template: guard,
                elems: vec![guarded(-5, 0), guarded(-6, 4), self.ones()],
                own_bits: C3,
                own_offset: 0,
                domain: vec![restrict(n + 4, n + 5), region.clone()],
            };
            for w in 0..WINDOWS as u32 {
                guards.emit(&mut self.program, half_row(half(n + 4, w)), None);
            }
            self.record(&guards, WINDOWS, false);
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
