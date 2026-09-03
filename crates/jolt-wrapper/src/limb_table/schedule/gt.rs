//! The GT lane: raw GT inputs, their norm-one checks, the fixed GT tables
//! and the online GT multiplications of the deferred check.

use super::{
    digits, fq12_coords, gt_mul, gt_norm_one, hi, row, Bits, Builder, Cells, DigitOp, DigitRule,
    DorySetupInputs, ElemRel, Factor, Family, FlattenedCheck, GtBase, GtCell, GtLeaf, GtOperand,
    KeyBase, ReadKind, Rel, RowId, SelectedFamily, WireValues, C, CELL, E_T, FP_SLOTS_GT, HI_T,
    K_ON, K_T, WINDOWS, W_ON,
};

impl Builder {
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
    pub(super) fn gt_tables(&mut self, check: &FlattenedCheck, setup: &DorySetupInputs) {
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
        let hi_t = hi(Cells::GT_TABLE, HI_T);
        let family = Family {
            name: "gt_table",
            template: &template,
            elems: vec![
                ElemRel::structured(
                    vec![
                        Factor::shift(E_T, E_T, -1).with_range(1..8),
                        Factor::same(K_T, K_T),
                        Factor::constant(HI_T, hi_t),
                    ],
                    C,
                    0,
                ),
                ElemRel::structured(
                    vec![
                        Factor::constant(E_T, 0),
                        Factor::same(K_T, K_T),
                        Factor::constant(HI_T, hi_t),
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
                Factor::restrict(HI_T, hi_t..hi_t + 1),
            ],
        };
        let ops: Vec<(RowId, Option<u8>)> = (0..bases as u32)
            .flat_map(|k| (1..8u32).map(move |e| (row(Cells::GT_TABLE + 8 * k + e), None)))
            .collect();
        self.place(&family, &ops, false);
        // Fingerprints of every entry (`e ∈ 0..8`).
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
        // The `one` cell's fingerprints.
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

    /// `x · conj(x) = 1` for every byte-linked GT input: raw `Fq12` inputs
    /// land in the norm-one torus, where the conjugation the negative digits
    /// use is inversion. Order-`r` membership is not needed: the torus is the
    /// direct product of the target group and a subgroup of order coprime to
    /// `r`, every verifier equation splits component-wise, and the target-group
    /// component equations are the honest ones under the same challenges.
    /// `x·conj(x) = 1` for every GT base (the byte-linked inputs; the setup
    /// constants ride along in the same range), at `GT_NORM + k` for base `k`
    /// (its table entry `e = 0` is the leaf cell).
    pub(super) fn gt_norm_checks(&mut self, bases: usize) {
        let norm = gt_norm_one();
        let k_op = Bits::new(4, 12);
        let hi_op = Bits::new(12, 18);
        let hi_n = hi(Cells::GT_NORM, hi_op);
        let family = Family {
            name: "gt_norm_one",
            template: &norm,
            elems: vec![ElemRel::structured(
                vec![
                    Factor::same(k_op, K_T),
                    Factor::constant(E_T, 0),
                    Factor::constant(HI_T, hi(Cells::GT_TABLE, HI_T)),
                ],
                C,
                0,
            )],
            own_bits: C,
            own_offset: 0,
            domain: vec![
                Factor::restrict(k_op, 0..bases as u32),
                Factor::restrict(hi_op, hi_n..hi_n + 1),
            ],
        };
        let ops: Vec<(RowId, Option<u8>)> = (0..bases as u32)
            .map(|k| (row(Cells::GT_NORM + k), None))
            .collect();
        self.place(&family, &ops, false);
    }

    /// Straus over the 64 windows: per window four squarings of the
    /// accumulator (`k ∈ B..B+4`) then one selected-entry mult per base.
    /// Returns the accumulator cell (`RHS`).
    pub(super) fn gt_online(&mut self, check: &FlattenedCheck, values: &WireValues) -> GtCell {
        let bases = check.gt.bases.len() as u32;
        let b = bases;
        let link_base = self.link_base(bases);
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
                    link: link_base + k,
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
            digit_base: (0..b).map(|k| (k, link_base + k)).collect(),
        });
        GtCell(cell(b - 1, WINDOWS as u32 - 1))
    }
}
