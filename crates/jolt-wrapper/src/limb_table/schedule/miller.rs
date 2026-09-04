//! The optimal ate Miller loop over the two G2 pairing inputs and the
//! constant G2 bases, with its line evaluations at the G1 inputs.

use super::{
    ell, g2_psi, gt_mul, half_row, hi, miller_add_step, miller_double_step, row, AdditiveGroup,
    Bits, Bn254Config, BnConfig, Builder, Cells, ElemRel, Factor, Family, Field, Fq, Fq2, G2Affine,
    G2Prepared, GtCell, GtOperand, Rel, SWCurveConfig, ADD_LINE, A_CA, A_LA, A_MA, C, C3, C5, CELL,
    CONST_LINE, DOUBLE_LINE, GROUP, HI_CA, HI_CD, HI_LA, HI_LD, HI_MA, HI_MD, P_CA, P_CD, P_LA,
    P_LD, P_MA, S_MD, T_CD, T_LD, T_MD,
};

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
    /// Line computations for pairs 0–1 (`Q` at the given half cells, point at
    /// rows 0–3), public lines for pairs 2–3, and the GT accumulation over
    /// the pairing points `p_cells` (rows 14–15). Returns the Miller output.
    pub(super) fn miller(
        &mut self,
        q_halves: [u32; 2],
        const_q: &[G2Affine; 2],
        p_cells: [u32; 4],
    ) -> GtCell {
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
        let (cx, cy) = super::super::ops::psi_coefficients(1);
        let psi_consts = ElemRel::Rows([self.fq2_constant(cx), self.fq2_constant(cy)].concat());
        let q1_cells = [self.glue_cell(), self.glue_cell()];
        let q2_cells = [self.glue_cell(), self.glue_cell()];
        let psi = g2_psi(false, true);
        let psi_neg = g2_psi(true, true);
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
