//! The template catalog: every operation's rows as slots over its own rows
//! and its input elements (arkworks' formulas row for row). Element
//! coordinate conventions: GT = the twelve tower coordinates; G1 = `(x, y)`;
//! G2 affine = `(x0, x1, y0, y1)`; G2 homogeneous projective = `(x0, x1, y0,
//! y1, z0, z1)`; `Fq2` values = `(re, im)`.

use ark_bn254::Fq;
use ark_ff::{AdditiveGroup, Field};

use super::template::{at, conjugated, own, Ref, RefSlots, RowKind, Template, TemplateRow};
use super::tower::{mul_form, FrobeniusForm, LINE_COORDS};

type Slots = RefSlots;

/// A GT operand: which of its twelve coordinates are structurally nonzero
/// and whether it enters conjugated (coordinates `≥ 6` negated).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GtOperand {
    pub elem: u8,
    pub nonzero: [bool; 12],
    pub conj: bool,
}

impl GtOperand {
    pub const fn dense(elem: u8) -> Self {
        Self {
            elem,
            nonzero: [true; 12],
            conj: false,
        }
    }

    pub const fn conj(elem: u8) -> Self {
        Self {
            elem,
            nonzero: [true; 12],
            conj: true,
        }
    }

    /// The identity: only coordinate 0.
    pub const fn one(elem: u8) -> Self {
        let mut nonzero = [false; 12];
        nonzero[0] = true;
        Self {
            elem,
            nonzero,
            conj: false,
        }
    }

    fn sign(&self, coord: u8) -> i32 {
        if self.conj && conjugated(coord) {
            -1
        } else {
            1
        }
    }
}

/// Slots of the twelve product coordinates `z_c = Σ κ x_a y_b`, with `y`
/// coordinates resolved through `y_ref` (lines mix own rows and step rows).
fn gt_product_rows(
    x: GtOperand,
    y_nonzero: [bool; 12],
    y_conj: bool,
    y_ref: impl Fn(u8) -> Ref,
) -> Vec<TemplateRow> {
    mul_form()
        .iter()
        .map(|terms| {
            let slots = terms
                .iter()
                .filter(|t| x.nonzero[t.a as usize] && y_nonzero[t.b as usize])
                .map(|t| {
                    let y_sign = if y_conj && conjugated(t.b) { -1 } else { 1 };
                    (
                        at(x.elem, t.a),
                        y_ref(t.b),
                        i32::from(t.kappa) * x.sign(t.a) * y_sign,
                    )
                })
                .collect();
            TemplateRow::compute(slots)
        })
        .collect()
}

/// `x · y` over full GT elements (twelve rows).
pub fn gt_mul(x: GtOperand, y: GtOperand) -> Template {
    Template::new(gt_product_rows(x, y.nonzero, y.conj, |b| at(y.elem, b)))
}

/// Frobenius power of element 1: each coordinate is a linear form over public
/// tower constants held by element 2 (`constants[i]` is its coordinate `i`).
pub fn gt_frobenius(form: &FrobeniusForm) -> (Template, Vec<Fq>) {
    let mut constants: Vec<Fq> = Vec::new();
    let rows = form
        .iter()
        .map(|terms| {
            let slots = terms
                .iter()
                .map(|(a, constant)| {
                    let index = constants
                        .iter()
                        .position(|c| c == constant)
                        .unwrap_or_else(|| {
                            constants.push(*constant);
                            constants.len() - 1
                        });
                    (at(1, *a), at(2, index as u8), 1)
                })
                .collect();
            TemplateRow::compute(slots)
        })
        .collect();
    (Template::new(rows), constants)
}

/// Witness `x⁻¹` (element 1 = `x`): twelve witness rows.
pub fn gt_inverse_witness() -> Template {
    Template::new(
        (0..12u8)
            .map(|coord| TemplateRow::witness(RowKind::InverseFq12 { coord }))
            .collect(),
    )
}

/// The pinned product `x · x⁻¹ = 1` (element 1 = `x`, element 2 = the witness).
pub fn gt_inverse_pin() -> Template {
    let mut rows = gt_product_rows(GtOperand::dense(1), [true; 12], false, |b| at(2, b));
    for (c, row) in rows.iter_mut().enumerate() {
        row.pin = Some(if c == 0 { Fq::ONE } else { Fq::ZERO });
    }
    Template::new(rows)
}

/// Which step rows hold a line's three `Fq2` coefficients.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LineRows {
    pub c0: [u8; 2],
    pub c1: [u8; 2],
    pub c2: [u8; 2],
    /// Signs folding arkworks' `(-h, 3j, i)` / `(λ, -θ, j)` into the rows.
    pub scale: [i32; 3],
}

/// Doubling-step lines `(-h, 3j, i)` in [`miller_double_step`]'s rows.
pub const DOUBLE_LINE: LineRows = LineRows {
    c0: [12, 13],
    c1: [14, 15],
    c2: [18, 19],
    scale: [-1, 3, 1],
};

/// Addition-step lines `(λ, -θ, j)` in [`miller_add_step`]'s rows.
pub const ADD_LINE: LineRows = LineRows {
    c0: [2, 3],
    c1: [0, 1],
    c2: [22, 23],
    scale: [1, -1, 1],
};

/// Public line coefficients as six pinned rows `(c0, c1, c2)`.
pub const CONST_LINE: LineRows = LineRows {
    c0: [0, 1],
    c1: [2, 3],
    c2: [4, 5],
    scale: [1, 1, 1],
};

/// `Bn::ell` (TwistType::D): rows 12–15 scale `c0` by `P.y` and `c1` by
/// `P.x`, rows 0–11 multiply `f` (element 1) by the sparse line
/// `(c0·P.y, c1·P.x, c2)` at coordinates `LINE_COORDS`. Element 2 holds
/// the line rows, element 3 the point `P = (x, y)`.
pub fn ell(line: LineRows) -> Template {
    let mut nonzero = [false; 12];
    for pair in LINE_COORDS {
        nonzero[pair[0]] = true;
        nonzero[pair[1]] = true;
    }
    let y_ref = |b: u8| -> Ref {
        let b = usize::from(b);
        if LINE_COORDS[0].contains(&b) {
            own(12 + (b - LINE_COORDS[0][0]) as u8)
        } else if LINE_COORDS[1].contains(&b) {
            own(14 + (b - LINE_COORDS[1][0]) as u8)
        } else {
            at(2, line.c2[b - LINE_COORDS[2][0]])
        }
    };
    let mut rows = gt_product_rows(GtOperand::dense(1), nonzero, false, y_ref);
    // c2 enters with its sign through the mult slots.
    if line.scale[2] != 1 {
        for row in &mut rows {
            for slot in &mut row.slots {
                if slot.1.elem == 2 {
                    slot.2 *= line.scale[2];
                }
            }
        }
    }
    for i in 0..2u8 {
        rows.push(TemplateRow::compute(vec![(
            at(2, line.c0[i as usize]),
            at(3, 1),
            line.scale[0],
        )]));
    }
    for i in 0..2u8 {
        rows.push(TemplateRow::compute(vec![(
            at(2, line.c1[i as usize]),
            at(3, 0),
            line.scale[1],
        )]));
    }
    // The scaled coefficients (rows 12–15) feed the product rows.
    Template::new(rows).with_order((12..16).chain(0..12).collect())
}

fn fq2_mul(a: [Ref; 2], b: [Ref; 2], kappa: i32) -> [Slots; 2] {
    [
        vec![(a[0], b[0], kappa), (a[1], b[1], -kappa)],
        vec![(a[0], b[1], kappa), (a[1], b[0], kappa)],
    ]
}

fn extend2(rows: &mut [Slots; 2], extra: [Slots; 2]) {
    for (row, more) in rows.iter_mut().zip(extra) {
        row.extend(more);
    }
}

/// Element 1 = `r = (x, y, z)`, element 2 = constants `(1/2, 3b.re, 3b.im, 1)`.
/// Outputs `x3 y3 z3` at rows 20–25; lines per [`DOUBLE_LINE`].
pub fn miller_double_step() -> Template {
    let (x, y, z) = (
        [at(1, 0), at(1, 1)],
        [at(1, 2), at(1, 3)],
        [at(1, 4), at(1, 5)],
    );
    let (two_inv, three_b, one) = (at(2, 0), [at(2, 1), at(2, 2)], at(2, 3));
    let pair = |base: u8| [own(base), own(base + 1)];
    let (xy, a, b, c, e, g, h, e2) = (
        pair(0),
        pair(2),
        pair(4),
        pair(6),
        pair(8),
        pair(10),
        pair(12),
        pair(16),
    );
    let mut rows: Vec<[Slots; 2]> = Vec::with_capacity(13);
    rows.push(fq2_mul(x, y, 1));
    rows.push([vec![(xy[0], two_inv, 1)], vec![(xy[1], two_inv, 1)]]);
    rows.push(fq2_mul(y, y, 1));
    rows.push(fq2_mul(z, z, 1));
    rows.push(fq2_mul(three_b, c, 1));
    rows.push([
        vec![(b[0], two_inv, 1), (e[0], two_inv, 3)],
        vec![(b[1], two_inv, 1), (e[1], two_inv, 3)],
    ]);
    rows.push(fq2_mul(y, z, 2));
    rows.push(fq2_mul(x, x, 1));
    rows.push(fq2_mul(e, e, 1));
    rows.push([
        vec![(e[0], one, 1), (b[0], one, -1)],
        vec![(e[1], one, 1), (b[1], one, -1)],
    ]);
    let mut x3 = fq2_mul(a, b, 1);
    extend2(&mut x3, fq2_mul(a, e, -3));
    rows.push(x3);
    let mut y3 = fq2_mul(g, g, 1);
    extend2(&mut y3, [vec![(e2[0], one, -3)], vec![(e2[1], one, -3)]]);
    rows.push(y3);
    rows.push(fq2_mul(b, h, 1));
    Template::new(
        rows.into_iter()
            .flatten()
            .map(TemplateRow::compute)
            .collect(),
    )
}

/// Element 1 = `r = (x, y, z)`, element 2 = `q = (x0, x1, y0, y1)` affine
/// (negated when `neg_q`), element 3 = constants `(1,)`. Outputs `x3 y3 z3`
/// at rows 16–21; lines per [`ADD_LINE`].
pub fn miller_add_step(neg_q: bool) -> Template {
    let s = if neg_q { -1 } else { 1 };
    let (x, y, z) = (
        [at(1, 0), at(1, 1)],
        [at(1, 2), at(1, 3)],
        [at(1, 4), at(1, 5)],
    );
    let (qx, qy) = ([at(2, 0), at(2, 1)], [at(2, 2), at(2, 3)]);
    let one = at(3, 0);
    let pair = |base: u8| [own(base), own(base + 1)];
    let (theta, lambda, c, d, e, f, g, h) = (
        pair(0),
        pair(2),
        pair(4),
        pair(6),
        pair(8),
        pair(10),
        pair(12),
        pair(14),
    );
    let mut rows: Vec<[Slots; 2]> = Vec::with_capacity(12);
    let mut th = [vec![(y[0], one, 1)], vec![(y[1], one, 1)]];
    extend2(&mut th, fq2_mul(qy, z, -s));
    rows.push(th);
    let mut la = [vec![(x[0], one, 1)], vec![(x[1], one, 1)]];
    extend2(&mut la, fq2_mul(qx, z, -1));
    rows.push(la);
    rows.push(fq2_mul(theta, theta, 1));
    rows.push(fq2_mul(lambda, lambda, 1));
    rows.push(fq2_mul(lambda, d, 1));
    rows.push(fq2_mul(z, c, 1));
    rows.push(fq2_mul(x, d, 1));
    rows.push([
        vec![(e[0], one, 1), (f[0], one, 1), (g[0], one, -2)],
        vec![(e[1], one, 1), (f[1], one, 1), (g[1], one, -2)],
    ]);
    rows.push(fq2_mul(lambda, h, 1));
    let mut y3 = fq2_mul(theta, g, 1);
    extend2(&mut y3, fq2_mul(theta, h, -1));
    extend2(&mut y3, fq2_mul(e, y, -1));
    rows.push(y3);
    rows.push(fq2_mul(z, e, 1));
    let mut jj = fq2_mul(theta, qx, 1);
    extend2(&mut jj, fq2_mul(lambda, qy, -s));
    rows.push(jj);
    Template::new(
        rows.into_iter()
            .flatten()
            .map(TemplateRow::compute)
            .collect(),
    )
}

/// `ψ(P) = (conj(x)·cx, conj(y)·cy)` (arkworks' `mul_by_char`), element 1 =
/// `P` affine, element 2 = constants `(cx.re, cx.im, cy.re, cy.im)`;
/// `negate_y` folds the sign of `q2 = -ψ(q1)`.
pub fn g2_psi(negate_y: bool) -> Template {
    let (x, y) = ([at(1, 0), at(1, 1)], [at(1, 2), at(1, 3)]);
    let (cx, cy) = ([at(2, 0), at(2, 1)], [at(2, 2), at(2, 3)]);
    let conj_mul = |v: [Ref; 2], c: [Ref; 2], s: i32| -> [Slots; 2] {
        [
            vec![(v[0], c[0], s), (v[1], c[1], s)],
            vec![(v[0], c[1], s), (v[1], c[0], -s)],
        ]
    };
    let sy = if negate_y { -1 } else { 1 };
    let rows = [conj_mul(x, cx, 1), conj_mul(y, cy, sy)];
    Template::new(
        rows.into_iter()
            .flatten()
            .map(TemplateRow::compute)
            .collect(),
    )
}

/// Affine G2 addition `p ± q` (elements 1, 2; element 3 = `(1,)`): rows
/// `λ` (witness, 0–1), pin `λ·(x2−x1) − (±y2−y1) = 0` (2–3), `x3` (4–5), `y3` (6–7).
pub fn g2_add(neg_q: bool) -> Template {
    let (x1, y1) = ([at(1, 0), at(1, 1)], [at(1, 2), at(1, 3)]);
    let (x2, y2) = ([at(2, 0), at(2, 1)], [at(2, 2), at(2, 3)]);
    let one = at(3, 0);
    let s = if neg_q { -1 } else { 1 };
    let lambda = [own(0), own(1)];
    let x3 = [own(4), own(5)];
    let num: [Slots; 2] = [
        vec![(y2[0], one, s), (y1[0], one, -1)],
        vec![(y2[1], one, s), (y1[1], one, -1)],
    ];
    let den: [Slots; 2] = [
        vec![(x2[0], one, 1), (x1[0], one, -1)],
        vec![(x2[1], one, 1), (x1[1], one, -1)],
    ];
    let mut rows = vec![
        TemplateRow::witness(RowKind::QuotientFq2 {
            num: num.clone(),
            den: den.clone(),
            coord: 0,
        }),
        TemplateRow::witness(RowKind::QuotientFq2 { num, den, coord: 1 }),
    ];
    let mut pin = fq2_mul(lambda, x2, 1);
    extend2(&mut pin, fq2_mul(lambda, x1, -1));
    extend2(
        &mut pin,
        [
            vec![(y2[0], one, -s), (y1[0], one, 1)],
            vec![(y2[1], one, -s), (y1[1], one, 1)],
        ],
    );
    rows.extend(
        pin.into_iter()
            .map(|slots| TemplateRow::pinned(slots, Fq::ZERO)),
    );
    let mut x3_rows = fq2_mul(lambda, lambda, 1);
    extend2(
        &mut x3_rows,
        [
            vec![(x1[0], one, -1), (x2[0], one, -1)],
            vec![(x1[1], one, -1), (x2[1], one, -1)],
        ],
    );
    rows.extend(x3_rows.into_iter().map(TemplateRow::compute));
    let mut y3 = fq2_mul(lambda, x1, 1);
    extend2(&mut y3, fq2_mul(lambda, x3, -1));
    extend2(&mut y3, [vec![(y1[0], one, -1)], vec![(y1[1], one, -1)]]);
    rows.extend(y3.into_iter().map(TemplateRow::compute));
    Template::new(rows)
}

/// Affine G2 doubling (element 1 = `p`, element 2 = `(1,)`): `λ = 3x²/(2y)`,
/// pin `2y·λ − 3x² = 0` (rows 2–3), `x3` (4–5), `y3` (6–7).
pub fn g2_dbl() -> Template {
    let (x, y) = ([at(1, 0), at(1, 1)], [at(1, 2), at(1, 3)]);
    let one = at(2, 0);
    let lambda = [own(0), own(1)];
    let x3 = [own(4), own(5)];
    let num = fq2_mul(x, x, 3);
    let den: [Slots; 2] = [vec![(y[0], one, 2)], vec![(y[1], one, 2)]];
    let mut rows = vec![
        TemplateRow::witness(RowKind::QuotientFq2 {
            num: num.clone(),
            den: den.clone(),
            coord: 0,
        }),
        TemplateRow::witness(RowKind::QuotientFq2 { num, den, coord: 1 }),
    ];
    let mut pin = fq2_mul(y, lambda, 2);
    extend2(&mut pin, fq2_mul(x, x, -3));
    rows.extend(
        pin.into_iter()
            .map(|slots| TemplateRow::pinned(slots, Fq::ZERO)),
    );
    let mut x3_rows = fq2_mul(lambda, lambda, 1);
    extend2(&mut x3_rows, [vec![(x[0], one, -2)], vec![(x[1], one, -2)]]);
    rows.extend(x3_rows.into_iter().map(TemplateRow::compute));
    let mut y3 = fq2_mul(lambda, x, 1);
    extend2(&mut y3, fq2_mul(lambda, x3, -1));
    extend2(&mut y3, [vec![(y[0], one, -1)], vec![(y[1], one, -1)]]);
    rows.extend(y3.into_iter().map(TemplateRow::compute));
    Template::new(rows)
}

/// Copy of an affine G2 point (element 1) with optional `y` negation;
/// element 2 = `(1,)`.
pub fn g2_copy(negate_y: bool) -> Template {
    let one = at(2, 0);
    let sy = if negate_y { -1 } else { 1 };
    Template::new(vec![
        TemplateRow::compute(vec![(at(1, 0), one, 1)]),
        TemplateRow::compute(vec![(at(1, 1), one, 1)]),
        TemplateRow::compute(vec![(at(1, 2), one, sy)]),
        TemplateRow::compute(vec![(at(1, 3), one, sy)]),
    ])
}

/// On-curve check of a G2 point (element 1 = `x0 x1 y0 y1`): `t = x²`
/// (rows 0–1), pin `y² − t·x − b = 0` (rows 2–3); element 2 = constants
/// `(b.re, b.im, 1)`.
pub fn g2_on_curve() -> Template {
    let (x, y) = ([at(1, 0), at(1, 1)], [at(1, 2), at(1, 3)]);
    let t = [own(0), own(1)];
    let (b, one) = ([at(2, 0), at(2, 1)], at(2, 2));
    let mut rows: Vec<TemplateRow> = fq2_mul(x, x, 1)
        .into_iter()
        .map(TemplateRow::compute)
        .collect();
    let mut pin = fq2_mul(y, y, 1);
    extend2(&mut pin, fq2_mul(t, x, -1));
    extend2(&mut pin, [vec![(b[0], one, -1)], vec![(b[1], one, -1)]]);
    rows.extend(
        pin.into_iter()
            .map(|slots| TemplateRow::pinned(slots, Fq::ZERO)),
    );
    Template::new(rows)
}

/// Affine G1 addition `p ± q` (elements 1, 2; element 3 = `(1,)`) in the
/// four spare rows of a GT cell: `λ` (witness, row 0), pin
/// `λ·(x2−x1) − (±y2−y1) = 0` (1), `x3` (2), `y3` (3).
pub fn g1_add(neg_q: bool) -> Template {
    let (x1, y1, x2, y2) = (at(1, 0), at(1, 1), at(2, 0), at(2, 1));
    let one = at(3, 0);
    let s = if neg_q { -1 } else { 1 };
    let (lambda, x3) = (own(0), own(2));
    Template::new(vec![
        TemplateRow::witness(RowKind::Quotient {
            num: vec![(y2, one, s), (y1, one, -1)],
            den: vec![(x2, one, 1), (x1, one, -1)],
        }),
        TemplateRow::pinned(
            vec![
                (lambda, x2, 1),
                (lambda, x1, -1),
                (y2, one, -s),
                (y1, one, 1),
            ],
            Fq::ZERO,
        ),
        TemplateRow::compute(vec![(lambda, lambda, 1), (x1, one, -1), (x2, one, -1)]),
        TemplateRow::compute(vec![(lambda, x1, 1), (lambda, x3, -1), (y1, one, -1)]),
    ])
}

/// Affine G1 doubling (element 1 = `p`, element 2 = `(1,)`): `λ = 3x²/(2y)`.
pub fn g1_dbl() -> Template {
    let (x, y) = (at(1, 0), at(1, 1));
    let one = at(2, 0);
    let (lambda, x3) = (own(0), own(2));
    Template::new(vec![
        TemplateRow::witness(RowKind::Quotient {
            num: vec![(x, x, 3)],
            den: vec![(y, one, 2)],
        }),
        TemplateRow::pinned(vec![(y, lambda, 2), (x, x, -3)], Fq::ZERO),
        TemplateRow::compute(vec![(lambda, lambda, 1), (x, one, -2)]),
        TemplateRow::compute(vec![(lambda, x, 1), (lambda, x3, -1), (y, one, -1)]),
    ])
}

/// Copy of a G1 point (element 1) with optional `y` negation; element 2 = `(1,)`.
pub fn g1_copy(negate_y: bool) -> Template {
    let one = at(2, 0);
    let sy = if negate_y { -1 } else { 1 };
    Template::new(vec![
        TemplateRow::compute(vec![(at(1, 0), one, 1)]),
        TemplateRow::compute(vec![(at(1, 1), one, sy)]),
    ])
}

/// On-curve check of a G1 point (element 1 = `(x, y)`): `t = x²` (row 0),
/// pin `y² − t·x − 3 = 0` (row 1); element 2 = constants `(3, 1)`.
pub fn g1_on_curve() -> Template {
    let (x, y, t) = (at(1, 0), at(1, 1), own(0));
    let (b, one) = (at(2, 0), at(2, 1));
    Template::new(vec![
        TemplateRow::compute(vec![(x, x, 1)]),
        TemplateRow::pinned(vec![(y, y, 1), (t, x, -1), (b, one, -1)], Fq::ZERO),
    ])
}

/// `lhs_c − rhs_c = 0` over two GT elements (elements 1, 2; element 3 = `(1,)`).
pub fn gt_difference_pins() -> Template {
    let one = at(3, 0);
    Template::new(
        (0..12u8)
            .map(|c| TemplateRow::pinned(vec![(at(1, c), one, 1), (at(2, c), one, -1)], Fq::ZERO))
            .collect(),
    )
}

/// `ψ^power(x, y) = (conj^power(x)·cx, conj^power(y)·cy)`, folded from
/// arkworks' `TWIST_MUL_BY_Q_{X,Y}` (`ψ = mul_by_char`).
pub fn psi_coefficients(power: usize) -> (ark_bn254::Fq2, ark_bn254::Fq2) {
    use ark_bn254::Config as Bn254Config;
    use ark_ec::bn::BnConfig;
    let (mut cx, mut cy) = (ark_bn254::Fq2::ONE, ark_bn254::Fq2::ONE);
    for _ in 0..power {
        let mut cx_conj = cx;
        let mut cy_conj = cy;
        let _ = cx_conj.conjugate_in_place();
        let _ = cy_conj.conjugate_in_place();
        cx = cx_conj * Bn254Config::TWIST_MUL_BY_Q_X;
        cy = cy_conj * Bn254Config::TWIST_MUL_BY_Q_Y;
    }
    (cx, cy)
}
