//! Symbolic tower and curve arithmetic over [`Program`] rows: `Fq2`/`Fq12`
//! products from the probed bilinear forms, affine short-Weierstrass group
//! law with inverse witnesses, GLV endomorphisms and Frobenius maps.

use ark_bn254::{Config as Bn254Config, Fq, Fq12, Fq2, G1Affine, G2Affine};
use ark_ec::bn::BnConfig;
use ark_ec::scalar_mul::glv::GLVConfig;
use ark_ff::{AdditiveGroup, Field, Zero};

use super::program::{
    lin, lin_add, lin_neg, lin_scale, lin_sub, mul_terms, Lin, Program, Slot, Source,
};
use super::tower::{conjugated, fq12_coords, frobenius_form, mul_form, LINE_COORDS};

pub type Lin2 = [Lin; 2];
pub type Lin12 = [Lin; 12];
/// Affine point `(x, y)`.
pub type G1Point = [Lin; 2];
pub type G2Point = [Lin2; 2];

/// Slots of `kappa · a · b` over `Fq2` (`u² = -1`).
pub fn fq2_mul_terms(a: &Lin2, b: &Lin2, kappa: i32) -> [Vec<Slot>; 2] {
    let mut re = mul_terms(&a[0], &b[0], kappa);
    re.extend(mul_terms(&a[1], &b[1], -kappa));
    let mut im = mul_terms(&a[0], &b[1], kappa);
    im.extend(mul_terms(&a[1], &b[0], kappa));
    [re, im]
}

pub fn fq2_add(a: &Lin2, b: &Lin2) -> Lin2 {
    [lin_add(&a[0], &b[0]), lin_add(&a[1], &b[1])]
}

pub fn fq2_sub(a: &Lin2, b: &Lin2) -> Lin2 {
    [lin_sub(&a[0], &b[0]), lin_sub(&a[1], &b[1])]
}

pub fn fq2_neg(a: &Lin2) -> Lin2 {
    [lin_neg(&a[0]), lin_neg(&a[1])]
}

pub fn fq2_scale(a: &Lin2, k: i32) -> Lin2 {
    [lin_scale(&a[0], k), lin_scale(&a[1], k)]
}

pub fn fq2_conj(a: &Lin2) -> Lin2 {
    [a[0].clone(), lin_neg(&a[1])]
}

/// Conjugation (`Fq12.c1` negated): the cyclotomic inverse of a unitary element.
pub fn gt_conj(x: &Lin12) -> Lin12 {
    std::array::from_fn(|c| {
        if conjugated(c) {
            lin_neg(&x[c])
        } else {
            x[c].clone()
        }
    })
}

/// The sparse line element `(c0, c3, c4)` of `mul_by_034` as twelve coordinates.
pub fn gt_line(c0: &Lin2, c3: &Lin2, c4: &Lin2) -> Lin12 {
    let mut out: Lin12 = Default::default();
    for (coords, value) in LINE_COORDS.iter().zip([c0, c3, c4]) {
        out[coords[0]].clone_from(&value[0]);
        out[coords[1]].clone_from(&value[1]);
    }
    out
}

impl Program {
    pub fn emit(&mut self, slots: Vec<Slot>) -> Lin {
        lin(self.compute(slots))
    }

    pub fn emit2(&mut self, slots: [Vec<Slot>; 2]) -> Lin2 {
        let [re, im] = slots;
        [self.emit(re), self.emit(im)]
    }

    /// Slots of `kappa · l` (each term times the constant one).
    pub fn linear(&self, l: &Lin, kappa: i32) -> Vec<Slot> {
        mul_terms(l, &lin(self.one), kappa)
    }

    pub fn fq2_constant(&mut self, value: Fq2) -> Lin2 {
        [self.fq_lin(value.c0), self.fq_lin(value.c1)]
    }

    /// Constant as a linear combination; zero is the empty combination.
    pub fn fq_lin(&mut self, value: Fq) -> Lin {
        if value.is_zero() {
            Vec::new()
        } else {
            self.constant_lin(value)
        }
    }

    pub fn fq2_mul(&mut self, a: &Lin2, b: &Lin2) -> Lin2 {
        self.emit2(fq2_mul_terms(a, b, 1))
    }

    pub fn fq2_sqr(&mut self, a: &Lin2) -> Lin2 {
        self.fq2_mul(a, a)
    }

    /// `Fq2 × Fq`.
    pub fn fq2_mul_fq(&mut self, a: &Lin2, b: &Lin) -> Lin2 {
        [
            self.emit(mul_terms(&a[0], b, 1)),
            self.emit(mul_terms(&a[1], b, 1)),
        ]
    }

    /// Witness `d⁻¹` bound by the pinned row `d · d⁻¹ = 1`.
    pub fn fq_inverse(&mut self, d: &Lin) -> Lin {
        let inverse = lin(self.witness(Source::Inverse(d.clone())));
        let _ = self.pinned(mul_terms(d, &inverse, 1), Fq::ONE);
        inverse
    }

    /// Witness `d⁻¹ ∈ Fq2` bound by the pinned rows `d · d⁻¹ = 1 + 0u`.
    pub fn fq2_inverse(&mut self, d: &Lin2) -> Lin2 {
        let inverse: Lin2 = std::array::from_fn(|coord| {
            lin(self.witness(Source::InverseFq2 {
                re: d[0].clone(),
                im: d[1].clone(),
                coord: coord as u8,
            }))
        });
        let [re, im] = fq2_mul_terms(d, &inverse, 1);
        let _ = self.pinned(re, Fq::ONE);
        let _ = self.pinned(im, Fq::ZERO);
        inverse
    }

    pub fn gt_one(&self) -> Lin12 {
        let mut out: Lin12 = Default::default();
        out[0] = lin(self.one);
        out
    }

    pub fn gt_constant(&mut self, value: Fq12) -> Lin12 {
        let coords = fq12_coords(&value);
        std::array::from_fn(|c| self.fq_lin(coords[c]))
    }

    /// Twelve rows of the product; structurally zero coordinates stay empty.
    pub fn gt_mul(&mut self, x: &Lin12, y: &Lin12) -> Lin12 {
        let form = mul_form();
        std::array::from_fn(|c| {
            let mut slots = Vec::new();
            for term in &form[c] {
                let (a, b) = (&x[term.a as usize], &y[term.b as usize]);
                if !a.is_empty() && !b.is_empty() {
                    slots.extend(mul_terms(a, b, i32::from(term.kappa)));
                }
            }
            if slots.is_empty() {
                Vec::new()
            } else {
                self.emit(slots)
            }
        })
    }

    pub fn gt_sqr(&mut self, x: &Lin12) -> Lin12 {
        self.gt_mul(x, x)
    }

    /// `Frob^power(x)`: each coordinate is a row over public tower constants.
    pub fn gt_frobenius(&mut self, x: &Lin12, power: usize) -> Lin12 {
        let form = frobenius_form(power);
        std::array::from_fn(|c| {
            let mut slots = Vec::new();
            for (a, constant) in &form[c] {
                let source = &x[*a as usize];
                if !source.is_empty() {
                    let constant = self.constant_lin(*constant);
                    slots.extend(mul_terms(source, &constant, 1));
                }
            }
            if slots.is_empty() {
                Vec::new()
            } else {
                self.emit(slots)
            }
        })
    }

    /// Witness `x⁻¹ ∈ Fq12` bound by twelve pinned product rows.
    pub fn gt_inverse(&mut self, x: &Lin12) -> Lin12 {
        let inverse: Lin12 = std::array::from_fn(|coord| {
            lin(self.witness(Source::InverseFq12 {
                coords: Box::new(x.clone()),
                coord: coord as u8,
            }))
        });
        for (c, terms) in mul_form().iter().enumerate() {
            let mut slots = Vec::new();
            for term in terms {
                let (a, b) = (&x[term.a as usize], &inverse[term.b as usize]);
                if !a.is_empty() {
                    slots.extend(mul_terms(a, b, i32::from(term.kappa)));
                }
            }
            let expected = if c == 0 { Fq::ONE } else { Fq::ZERO };
            let _ = self.pinned(slots, expected);
        }
        inverse
    }

    pub fn g1_constant(&mut self, point: G1Affine) -> G1Point {
        [self.fq_lin(point.x), self.fq_lin(point.y)]
    }

    /// Affine `p + q` (or `p - q`): `λ = Δy/Δx`, with `Δx⁻¹` a witness.
    pub fn g1_add(&mut self, p: &G1Point, q: &G1Point, neg_q: bool) -> G1Point {
        let qy = if neg_q { lin_neg(&q[1]) } else { q[1].clone() };
        let dx = lin_sub(&q[0], &p[0]);
        let dy = lin_sub(&qy, &p[1]);
        let inverse = self.fq_inverse(&dx);
        let lambda = self.emit(mul_terms(&dy, &inverse, 1));
        let mut x3 = mul_terms(&lambda, &lambda, 1);
        x3.extend(self.linear(&lin_add(&p[0], &q[0]), -1));
        let x3 = self.emit(x3);
        let mut y3 = mul_terms(&lambda, &lin_sub(&p[0], &x3), 1);
        y3.extend(self.linear(&p[1], -1));
        [x3, self.emit(y3)]
    }

    /// Affine doubling: `λ = 3x²/(2y)`.
    pub fn g1_dbl(&mut self, p: &G1Point) -> G1Point {
        let inverse = self.fq_inverse(&lin_scale(&p[1], 2));
        let xx = self.emit(mul_terms(&p[0], &p[0], 1));
        let lambda = self.emit(mul_terms(&xx, &inverse, 3));
        let mut x3 = mul_terms(&lambda, &lambda, 1);
        x3.extend(self.linear(&p[0], -2));
        let x3 = self.emit(x3);
        let mut y3 = mul_terms(&lambda, &lin_sub(&p[0], &x3), 1);
        y3.extend(self.linear(&p[1], -1));
        [x3, self.emit(y3)]
    }

    /// `φ(x, y) = (β·x, y)`, arkworks' GLV endomorphism (eigenvalue `LAMBDA`).
    pub fn g1_endomorphism(&mut self, p: &G1Point) -> G1Point {
        let beta = self.constant_lin(<ark_bn254::g1::Config as GLVConfig>::ENDO_COEFFS[0]);
        [self.emit(mul_terms(&p[0], &beta, 1)), p[1].clone()]
    }

    pub fn g2_constant(&mut self, point: G2Affine) -> G2Point {
        [self.fq2_constant(point.x), self.fq2_constant(point.y)]
    }

    pub fn g2_add(&mut self, p: &G2Point, q: &G2Point, neg_q: bool) -> G2Point {
        let qy = if neg_q { fq2_neg(&q[1]) } else { q[1].clone() };
        let dx = fq2_sub(&q[0], &p[0]);
        let dy = fq2_sub(&qy, &p[1]);
        let inverse = self.fq2_inverse(&dx);
        let lambda = self.fq2_mul(&dy, &inverse);
        let mut x3 = fq2_mul_terms(&lambda, &lambda, 1);
        let sum = fq2_add(&p[0], &q[0]);
        for (slots, coord) in x3.iter_mut().zip(&sum) {
            slots.extend(self.linear(coord, -1));
        }
        let x3 = self.emit2(x3);
        let mut y3 = fq2_mul_terms(&lambda, &fq2_sub(&p[0], &x3), 1);
        for (slots, coord) in y3.iter_mut().zip(&p[1]) {
            slots.extend(self.linear(coord, -1));
        }
        [x3, self.emit2(y3)]
    }

    pub fn g2_dbl(&mut self, p: &G2Point) -> G2Point {
        let inverse = self.fq2_inverse(&fq2_scale(&p[1], 2));
        let xx = self.fq2_sqr(&p[0]);
        let lambda = self.emit2(fq2_mul_terms(&xx, &inverse, 3));
        let mut x3 = fq2_mul_terms(&lambda, &lambda, 1);
        for (slots, coord) in x3.iter_mut().zip(&p[0]) {
            slots.extend(self.linear(coord, -2));
        }
        let x3 = self.emit2(x3);
        let mut y3 = fq2_mul_terms(&lambda, &fq2_sub(&p[0], &x3), 1);
        for (slots, coord) in y3.iter_mut().zip(&p[1]) {
            slots.extend(self.linear(coord, -1));
        }
        [x3, self.emit2(y3)]
    }

    /// `ψ^power`: arkworks' `mul_by_char` iterated (conjugate, then multiply by
    /// the twist Frobenius coefficients).
    pub fn g2_psi(&mut self, p: &G2Point, power: usize) -> G2Point {
        let (cx, cy) = psi_coefficients(power);
        let mut point = p.clone();
        if power % 2 == 1 {
            point = [fq2_conj(&point[0]), fq2_conj(&point[1])];
        }
        let cx = self.fq2_constant(cx);
        let cy = self.fq2_constant(cy);
        [self.fq2_mul(&point[0], &cx), self.fq2_mul(&point[1], &cy)]
    }
}

/// `ψ^power(x, y) = (conj^power(x)·cx, conj^power(y)·cy)`, folded from
/// arkworks' `TWIST_MUL_BY_Q_{X,Y}` (`ψ = mul_by_char`).
pub fn psi_coefficients(power: usize) -> (Fq2, Fq2) {
    let (mut cx, mut cy) = (Fq2::ONE, Fq2::ONE);
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

/// Native `ψ^power` on an affine point, the oracle for [`Program::g2_psi`].
pub fn psi_native(point: G2Affine, power: usize) -> G2Affine {
    let (cx, cy) = psi_coefficients(power);
    let (mut x, mut y) = (point.x, point.y);
    if power % 2 == 1 {
        let _ = x.conjugate_in_place();
        let _ = y.conjugate_in_place();
    }
    G2Affine::new_unchecked(x * cx, y * cy)
}
