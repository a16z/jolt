//! The final exponentiation: Frobenius maps, glue multiplications and the
//! hard-part addition chains over the Miller output.

use super::{
    frobenius_form, gt_frobenius, gt_inverse_pin, gt_inverse_witness, gt_mul, hi, naf, row,
    Bn254Config, BnConfig, Builder, Cells, ElemRel, Factor, Family, GtCell, GtOperand, Rel, RowId,
    Template, C, CELL, CHAIN_FE, HI_FE, SLOT_FE, STEP_FE,
};

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
    pub(super) fn final_exponentiation(&mut self, f: GtCell) -> GtCell {
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
        let naf_x = naf(u128::from(Bn254Config::X[0]));
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
