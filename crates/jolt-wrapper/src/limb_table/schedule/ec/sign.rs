//! Canonical sign flags of the G1/G2 proof points (arkworks `y > -y`).

use super::{
    g1_sign, g2_sign, half_plus_one, hi, row, Builder, Cells, ElemRel, Factor, Family, Lane, RowId,
    B1I, B2I, C, C3, HI1I, HI2I,
};

impl Builder {
    /// Sign rows of the fresh G1 inputs at `G1_SIGN + b` (row 0).
    pub(super) fn g1_sign_rows(&mut self, lane: &Lane) {
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

    /// Sign rows of the fresh G2 inputs at cells `G2_SIGN + b` (rows 0–5).
    pub(super) fn g2_sign_rows(&mut self, lane: &Lane) {
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
}
