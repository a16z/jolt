//! The fixed operation schedule of the deferred check as a [`Program`]:
//! input and public rows, signed-digit Straus multi-exponentiations (GT with
//! four Frobenius accumulators, G1/G2 affine with public offset points),
//! arkworks' Miller loop and final exponentiation, and the twelve pinned
//! rows equating both sides.

use std::collections::HashMap;

use ark_bn254::{Config as Bn254Config, Fq, Fq2, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::bn::BnConfig;
use ark_ec::scalar_mul::glv::GLVConfig;
use ark_ec::short_weierstrass::SWCurveConfig;
use ark_ec::{AffineRepr, CurveGroup, PrimeGroup};
use ark_ff::{AdditiveGroup, Field, One, Zero};
use num_bigint::BigUint;

use super::dory::{
    input_elements, Base, DorySetupInputs, ElementKind, FlattenedCheck, InputElement, MultiExp,
};
use super::glv::{Digits, G1_WINDOWS, G2_WINDOWS, GT_WINDOWS, TABLE_ENTRIES, WINDOW};
use super::ops::{
    fq2_mul_terms, fq2_neg, fq2_scale, fq2_sub, gt_conj, gt_line, psi_native, G1Point, G2Point,
    Lin12, Lin2,
};
use super::program::{lin, Lin, Program, RowId};

/// Group operations Straus needs, over the program's symbolic points.
pub trait EcOps {
    type Point: Clone;
    type Affine: Copy;
    type Projective: CurveGroup<Affine = Self::Affine>;

    const GENERATOR: fn() -> Self::Affine;
    const WINDOWS: usize;

    fn constant(program: &mut Program, point: Self::Affine) -> Self::Point;
    fn add(program: &mut Program, p: &Self::Point, q: &Self::Point, neg_q: bool) -> Self::Point;
    fn dbl(program: &mut Program, p: &Self::Point) -> Self::Point;
    /// The GLV endomorphism power matching dimension `power` of the decomposition.
    fn endomorphism(program: &mut Program, p: &Self::Point, power: usize) -> Self::Point;
    fn endomorphism_native(point: Self::Affine, power: usize) -> Self::Affine;
    fn digits(scalars: &[ark_bn254::Fr]) -> Digits;
    fn to_projective(point: Self::Affine) -> Self::Projective;
}

pub struct G1Ops;
pub struct G2Ops;

impl EcOps for G1Ops {
    type Point = G1Point;
    type Affine = G1Affine;
    type Projective = G1Projective;
    const GENERATOR: fn() -> G1Affine = G1Affine::generator;
    const WINDOWS: usize = G1_WINDOWS;

    fn constant(program: &mut Program, point: G1Affine) -> G1Point {
        program.g1_constant(point)
    }
    fn add(program: &mut Program, p: &G1Point, q: &G1Point, neg_q: bool) -> G1Point {
        program.g1_add(p, q, neg_q)
    }
    fn dbl(program: &mut Program, p: &G1Point) -> G1Point {
        program.g1_dbl(p)
    }
    fn endomorphism(program: &mut Program, p: &G1Point, power: usize) -> G1Point {
        assert_eq!(power, 1);
        program.g1_endomorphism(p)
    }
    fn endomorphism_native(point: G1Affine, power: usize) -> G1Affine {
        if power == 0 {
            point
        } else {
            <ark_bn254::g1::Config as GLVConfig>::endomorphism_affine(&point)
        }
    }
    fn digits(scalars: &[ark_bn254::Fr]) -> Digits {
        Digits::two_dimensional_g1(scalars, G1_WINDOWS)
    }
    fn to_projective(point: G1Affine) -> G1Projective {
        point.into()
    }
}

impl EcOps for G2Ops {
    type Point = G2Point;
    type Affine = G2Affine;
    type Projective = G2Projective;
    const GENERATOR: fn() -> G2Affine = G2Affine::generator;
    const WINDOWS: usize = G2_WINDOWS;

    fn constant(program: &mut Program, point: G2Affine) -> G2Point {
        program.g2_constant(point)
    }
    fn add(program: &mut Program, p: &G2Point, q: &G2Point, neg_q: bool) -> G2Point {
        program.g2_add(p, q, neg_q)
    }
    fn dbl(program: &mut Program, p: &G2Point) -> G2Point {
        program.g2_dbl(p)
    }
    fn endomorphism(program: &mut Program, p: &G2Point, power: usize) -> G2Point {
        program.g2_psi(p, power)
    }
    fn endomorphism_native(point: G2Affine, power: usize) -> G2Affine {
        psi_native(point, power)
    }
    fn digits(scalars: &[ark_bn254::Fr]) -> Digits {
        Digits::four_dimensional(scalars, G2_WINDOWS)
    }
    fn to_projective(point: G2Affine) -> G2Projective {
        point.into()
    }
}

/// Public offset points of an affine Straus accumulation: the accumulator
/// starts at `R = G` and a zero digit adds `Z0 = 2G`, so no operand is ever
/// the identity; the public multiple accumulated this way is subtracted at
/// the end.
fn offsets<G: EcOps>() -> (G::Affine, G::Affine) {
    let generator = (G::GENERATOR)();
    let doubled = G::to_projective(generator).double().into_affine();
    (generator, doubled)
}

/// Signed-digit Straus over affine points with public offsets; the schedule
/// (tables, `WINDOW` doublings and one addition per base per window,
/// correction, GLV recombination) is independent of the digits.
pub fn straus_ec<G: EcOps>(program: &mut Program, bases: &[G::Point], digits: &Digits) -> G::Point {
    assert_eq!(bases.len(), digits.digits.len());
    let dims = digits.dims();
    let windows = digits.windows;
    let (r_point, z0_point) = offsets::<G>();
    let tables: Vec<Vec<G::Point>> = bases
        .iter()
        .map(|base| {
            let mut table = Vec::with_capacity(TABLE_ENTRIES + 1);
            table.push(base.clone());
            table.push(base.clone());
            table.push(G::dbl(program, base));
            for _ in 3..=TABLE_ENTRIES {
                let next = G::add(program, &table[table.len() - 1], base, false);
                table.push(next);
            }
            table
        })
        .collect();
    let z0 = G::constant(program, z0_point);
    let mut accumulators: Vec<G::Point> =
        (0..dims).map(|_| G::constant(program, r_point)).collect();
    let mut zero_weight = vec![BigUint::ZERO; dims];
    for window in (0..windows).rev() {
        for acc in &mut accumulators {
            for _ in 0..WINDOW {
                *acc = G::dbl(program, acc);
            }
        }
        for (table, base_digits) in tables.iter().zip(&digits.digits) {
            for (p, acc) in accumulators.iter_mut().enumerate() {
                let digit = base_digits[p][window];
                let operand = if digit == 0 {
                    zero_weight[p] += BigUint::one() << (WINDOW * window);
                    &z0
                } else {
                    &table[digit.unsigned_abs() as usize]
                };
                *acc = G::add(program, acc, operand, digit < 0);
            }
        }
    }
    // GLV recombination first, one public correction last: a zero
    // mini-scalar (scalar 1 decomposes to (1, 0, ..)) leaves an accumulator at
    // exactly its public offset, which a per-accumulator correction would map
    // to the identity.
    let r_weight = BigUint::one() << (WINDOW * windows);
    let mut result = accumulators[0].clone();
    let mut offset_total = G::Projective::zero();
    for (p, acc) in accumulators.iter().enumerate() {
        let offset = G::to_projective(r_point).mul_bigint(r_weight.to_u64_digits())
            + G::to_projective(z0_point).mul_bigint(zero_weight[p].to_u64_digits());
        offset_total += G::to_projective(G::endomorphism_native(offset.into_affine(), p));
        if p > 0 {
            let component = G::endomorphism(program, acc, p);
            result = G::add(program, &result, &component, false);
        }
    }
    let correction = G::constant(program, (-offset_total).into_affine());
    G::add(program, &result, &correction, false)
}

/// Signed-digit Straus in GT with one accumulator per Frobenius dimension,
/// recombined as `((A_3^q · A_2)^q · A_1)^q · A_0`; zero digits multiply by
/// the identity and negative digits by the conjugate.
pub fn straus_gt(program: &mut Program, bases: &[Lin12], digits: &Digits) -> Lin12 {
    assert_eq!(bases.len(), digits.digits.len());
    let dims = digits.dims();
    let one = program.gt_one();
    let tables: Vec<Vec<Lin12>> = bases
        .iter()
        .map(|base| {
            let mut table = Vec::with_capacity(TABLE_ENTRIES + 1);
            table.push(one.clone());
            table.push(base.clone());
            for _ in 2..=TABLE_ENTRIES {
                let next = program.gt_mul(&table[table.len() - 1], base);
                table.push(next);
            }
            table
        })
        .collect();
    program.end_section("gt_tables");
    let mut accumulators = vec![one; dims];
    for window in (0..digits.windows).rev() {
        for acc in &mut accumulators {
            for _ in 0..WINDOW {
                *acc = program.gt_sqr(acc);
            }
        }
        for (table, base_digits) in tables.iter().zip(&digits.digits) {
            for (p, acc) in accumulators.iter_mut().enumerate() {
                let digit = base_digits[p][window];
                let entry = &table[digit.unsigned_abs() as usize];
                let operand = if digit < 0 {
                    gt_conj(entry)
                } else {
                    entry.clone()
                };
                *acc = program.gt_mul(acc, &operand);
            }
        }
    }
    program.end_section("gt_online");
    let mut result = accumulators[dims - 1].clone();
    for acc in accumulators[..dims - 1].iter().rev() {
        let mapped = program.gt_frobenius(&result, 1);
        result = program.gt_mul(&mapped, acc);
    }
    result
}

/// One pairing's line coefficients: arkworks' `G2Prepared` (homogeneous
/// projective doubling/addition steps, TwistType::D), row for row.
struct LineCoefficients {
    coefficients: Vec<[Lin2; 3]>,
}

fn miller_lines(program: &mut Program, q: &G2Point) -> LineCoefficients {
    let two_inv = program.constant_lin(Fq::from(2u64).inverse().unwrap_or_else(|| unreachable!()));
    let three_b =
        <ark_bn254::g2::Config as SWCurveConfig>::COEFF_B * Fq2::new(Fq::from(3u64), Fq::ZERO);
    let three_b = program.fq2_constant(three_b);
    let mut r = [q[0].clone(), q[1].clone(), [lin(program.one), Vec::new()]];
    let mut coefficients = Vec::with_capacity(Bn254Config::ATE_LOOP_COUNT.len() + 2);
    for bit in Bn254Config::ATE_LOOP_COUNT.iter().rev().skip(1) {
        coefficients.push(doubling_step(program, &mut r, &two_inv, &three_b));
        match bit {
            1 => coefficients.push(addition_step(program, &mut r, q, false)),
            -1 => coefficients.push(addition_step(program, &mut r, q, true)),
            _ => {}
        }
    }
    let q1 = program.g2_psi(q, 1);
    let mut q2 = program.g2_psi(&q1, 1);
    q2[1] = fq2_neg(&q2[1]);
    coefficients.push(addition_step(program, &mut r, &q1, false));
    coefficients.push(addition_step(program, &mut r, &q2, false));
    LineCoefficients { coefficients }
}

/// `G2HomProjective::double_in_place` with `(-h, 3j, i)` line coefficients.
fn doubling_step(
    program: &mut Program,
    r: &mut [Lin2; 3],
    two_inv: &Lin,
    three_b: &Lin2,
) -> [Lin2; 3] {
    let [x, y, z] = r.clone();
    let xy = program.fq2_mul(&x, &y);
    let a = program.fq2_mul_fq(&xy, two_inv);
    let b = program.fq2_sqr(&y);
    let c = program.fq2_sqr(&z);
    let e = program.fq2_mul(three_b, &c);
    let f = fq2_scale(&e, 3);
    let bf = [
        [b[0].clone(), f[0].clone()].concat(),
        [b[1].clone(), f[1].clone()].concat(),
    ];
    let g = program.fq2_mul_fq(&bf, two_inv);
    let h = program.emit2(fq2_mul_terms(&y, &z, 2));
    let i = fq2_sub(&e, &b);
    let j = program.fq2_sqr(&x);
    let e_square = program.fq2_sqr(&e);
    let x3 = program.fq2_mul(&a, &fq2_sub(&b, &f));
    let mut y3 = fq2_mul_terms(&g, &g, 1);
    for (slots, coord) in y3.iter_mut().zip(&e_square) {
        slots.extend(program.linear(coord, -3));
    }
    let y3 = program.emit2(y3);
    let z3 = program.fq2_mul(&b, &h);
    *r = [x3, y3, z3];
    [fq2_neg(&h), fq2_scale(&j, 3), i]
}

/// `G2HomProjective::add_in_place` with `(λ, -θ, j)` line coefficients.
fn addition_step(program: &mut Program, r: &mut [Lin2; 3], q: &G2Point, neg_q: bool) -> [Lin2; 3] {
    let [x, y, z] = r.clone();
    let qy = if neg_q { fq2_neg(&q[1]) } else { q[1].clone() };
    let mut theta = fq2_mul_terms(&qy, &z, -1);
    for (slots, coord) in theta.iter_mut().zip(&y) {
        slots.extend(program.linear(coord, 1));
    }
    let theta = program.emit2(theta);
    let mut lambda = fq2_mul_terms(&q[0], &z, -1);
    for (slots, coord) in lambda.iter_mut().zip(&x) {
        slots.extend(program.linear(coord, 1));
    }
    let lambda = program.emit2(lambda);
    let c = program.fq2_sqr(&theta);
    let d = program.fq2_sqr(&lambda);
    let e = program.fq2_mul(&lambda, &d);
    let f = program.fq2_mul(&z, &c);
    let g = program.fq2_mul(&x, &d);
    let h = fq2_sub(
        &fq2_sub(
            &[
                [e[0].clone(), f[0].clone()].concat(),
                [e[1].clone(), f[1].clone()].concat(),
            ],
            &fq2_scale(&g, 2),
        ),
        &[Vec::new(), Vec::new()],
    );
    let x3 = program.fq2_mul(&lambda, &h);
    let mut y3 = fq2_mul_terms(&theta, &fq2_sub(&g, &h), 1);
    let ey = fq2_mul_terms(&e, &y, -1);
    for (slots, extra) in y3.iter_mut().zip(ey) {
        slots.extend(extra);
    }
    let y3 = program.emit2(y3);
    let z3 = program.fq2_mul(&z, &e);
    let mut j = fq2_mul_terms(&theta, &q[0], 1);
    let lq = fq2_mul_terms(&lambda, &qy, -1);
    for (slots, extra) in j.iter_mut().zip(lq) {
        slots.extend(extra);
    }
    let j = program.emit2(j);
    *r = [x3, y3, z3];
    [lambda, fq2_neg(&theta), j]
}

/// `Bn::ell` for TwistType::D: scale `c0` by `P.y`, `c1` by `P.x`, then `mul_by_034`.
fn ell(program: &mut Program, f: &Lin12, coefficients: &[Lin2; 3], p: &G1Point) -> Lin12 {
    let c0 = program.fq2_mul_fq(&coefficients[0], &p[1]);
    let c1 = program.fq2_mul_fq(&coefficients[1], &p[0]);
    let line = gt_line(&c0, &c1, &coefficients[2]);
    program.gt_mul(f, &line)
}

/// Arkworks' four-pair `multi_miller_loop` (`X_IS_NEGATIVE = false`).
pub fn miller_loop(program: &mut Program, pairs: &[(G1Point, G2Point); 4]) -> Lin12 {
    let lines: Vec<LineCoefficients> = pairs
        .iter()
        .map(|(_, q)| miller_lines(program, q))
        .collect();
    program.end_section("miller_lines");
    let mut next = vec![0usize; 4];
    let mut f = program.gt_one();
    let ate = Bn254Config::ATE_LOOP_COUNT;
    let step = |program: &mut Program, f: &Lin12, next: &mut [usize]| -> Lin12 {
        let mut f = f.clone();
        for (i, ((p, _), line)) in pairs.iter().zip(&lines).enumerate() {
            f = ell(program, &f, &line.coefficients[next[i]], p);
            next[i] += 1;
        }
        f
    };
    for i in (1..ate.len()).rev() {
        if i != ate.len() - 1 {
            f = program.gt_sqr(&f);
        }
        f = step(program, &f, &mut next);
        if ate[i - 1] != 0 {
            f = step(program, &f, &mut next);
        }
    }
    f = step(program, &f, &mut next);
    f = step(program, &f, &mut next);
    for (i, line) in lines.iter().enumerate() {
        assert_eq!(next[i], line.coefficients.len());
    }
    program.end_section("miller_loop");
    f
}

/// Non-adjacent form of `value`, least significant digit first.
fn naf(mut value: u64) -> Vec<i8> {
    let mut digits = Vec::new();
    while value != 0 {
        let digit = if value & 1 == 1 {
            2 - (value % 4) as i8
        } else {
            0
        };
        digits.push(digit);
        value = if digit < 0 {
            value.wrapping_add(digit.unsigned_abs() as u64) / 2
        } else {
            (value - digit as u64) / 2
        };
    }
    digits
}

/// `f^{-X}` for the positive curve parameter: NAF exponentiation with the
/// conjugate for negative digits (`cyclotomic_exp` on a unitary element),
/// then conjugation.
fn exp_by_neg_x(program: &mut Program, f: &Lin12) -> Lin12 {
    let conj = gt_conj(f);
    let mut result: Option<Lin12> = None;
    for digit in naf(Bn254Config::X[0]).iter().rev() {
        if let Some(acc) = result.as_mut() {
            *acc = program.gt_sqr(acc);
        }
        if *digit != 0 {
            let operand = if *digit > 0 { f } else { &conj };
            result = Some(match result {
                None => operand.clone(),
                Some(acc) => program.gt_mul(&acc, operand),
            });
        }
    }
    gt_conj(&result.unwrap_or_else(|| unreachable!("X is nonzero")))
}

/// Arkworks' BN final exponentiation (easy part with an inverse witness, then
/// the Fuentes-Castañeda hard part), row for row.
pub fn final_exponentiation(program: &mut Program, f: &Lin12) -> Lin12 {
    let f1 = gt_conj(f);
    let f2 = program.gt_inverse(f);
    let r = program.gt_mul(&f1, &f2);
    let f2 = r.clone();
    let r = program.gt_frobenius(&r, 2);
    let f = program.gt_mul(&r, &f2);

    let y0 = exp_by_neg_x(program, &f);
    let y1 = program.gt_sqr(&y0);
    let y2 = program.gt_sqr(&y1);
    let y3 = program.gt_mul(&y2, &y1);
    let y4 = exp_by_neg_x(program, &y3);
    let y5 = program.gt_sqr(&y4);
    let y6 = exp_by_neg_x(program, &y5);
    let y3 = gt_conj(&y3);
    let y6 = gt_conj(&y6);
    let y7 = program.gt_mul(&y6, &y4);
    let y8 = program.gt_mul(&y7, &y3);
    let y9 = program.gt_mul(&y8, &y1);
    let y10 = program.gt_mul(&y8, &y4);
    let y11 = program.gt_mul(&y10, &f);
    let y12 = program.gt_frobenius(&y9, 1);
    let y13 = program.gt_mul(&y12, &y11);
    let y8 = program.gt_frobenius(&y8, 2);
    let y14 = program.gt_mul(&y8, &y13);
    let f = gt_conj(&f);
    let y15 = program.gt_mul(&f, &y9);
    let y15 = program.gt_frobenius(&y15, 3);
    program.gt_mul(&y15, &y14)
}

/// The built table program with the rows the tests and the linking lane read.
pub struct Layout {
    pub program: Program,
    pub sigma: usize,
    pub n_commitments: usize,
    /// `Σ_k s_k X_k` (twelve rows).
    pub rhs: Lin12,
    /// Miller loop output before the final exponentiation.
    pub miller: Lin12,
    /// Final-exponentiation output (twelve rows).
    pub lhs: Lin12,
    /// Pinned-to-zero rows `lhs_c - rhs_c`.
    pub final_check: [RowId; 12],
}

/// Builds the fixed program for `check` (public scalars/digits) and `setup`
/// (public constants); the committed inputs are rows in
/// [`input_elements`] order.
pub fn build(check: &FlattenedCheck, setup: &DorySetupInputs, sigma: usize, n: usize) -> Layout {
    let mut program = Program::new();
    let mut gt_inputs: HashMap<InputElement, Lin12> = HashMap::new();
    let mut g1_inputs: HashMap<InputElement, G1Point> = HashMap::new();
    let mut g2_inputs: HashMap<InputElement, G2Point> = HashMap::new();
    let mut index = 0;
    for element in input_elements(sigma, n) {
        let mut coord = || {
            let row = program.input(index);
            index += 1;
            lin(row)
        };
        match element.kind() {
            ElementKind::Gt => {
                let _ = gt_inputs.insert(element, std::array::from_fn(|_| coord()));
            }
            ElementKind::G1 => {
                let _ = g1_inputs.insert(element, [coord(), coord()]);
            }
            ElementKind::G2 => {
                let point = [[coord(), coord()], [coord(), coord()]];
                let _ = g2_inputs.insert(element, point);
            }
        }
    }
    program.end_section("inputs");

    let gt_bases: Vec<Lin12> = check
        .gt
        .bases
        .iter()
        .map(|base| match base {
            Base::Input(e) => gt_inputs[e].clone(),
            Base::Chi(k) => program.gt_constant(setup.chi[*k]),
            Base::Delta1R(k) => program.gt_constant(setup.delta_1r[*k]),
            Base::Delta2R(k) => program.gt_constant(setup.delta_2r[*k]),
            Base::Ht => program.gt_constant(setup.ht),
            Base::G1Zero | Base::G2Zero => unreachable!("not a GT base"),
        })
        .collect();
    let g1_bases = |program: &mut Program, m: &MultiExp| -> Vec<G1Point> {
        m.bases
            .iter()
            .map(|base| match base {
                Base::Input(e) => g1_inputs[e].clone(),
                Base::G1Zero => program.g1_constant(setup.g1_0),
                _ => unreachable!("not a G1 base"),
            })
            .collect()
    };
    let g2_bases = |program: &mut Program, m: &MultiExp| -> Vec<G2Point> {
        m.bases
            .iter()
            .map(|base| match base {
                Base::Input(e) => g2_inputs[e].clone(),
                Base::G2Zero => program.g2_constant(setup.g2_0),
                _ => unreachable!("not a G2 base"),
            })
            .collect()
    };
    let h1 = program.g1_constant(setup.h1);
    let h2 = program.g2_constant(setup.h2);
    let g2_0 = program.g2_constant(setup.g2_0);
    program.end_section("public");

    let gt_digits = Digits::four_dimensional(&check.gt.scalars, GT_WINDOWS);
    let rhs = straus_gt(&mut program, &gt_bases, &gt_digits);
    program.end_section("gt_combine");

    let g1: Vec<G1Point> = check
        .g1
        .iter()
        .map(|m| {
            let bases = g1_bases(&mut program, m);
            straus_ec::<G1Ops>(&mut program, &bases, &G1Ops::digits(&m.scalars))
        })
        .collect();
    program.end_section("g1");
    let g2: Vec<G2Point> = check
        .g2
        .iter()
        .map(|m| {
            let bases = g2_bases(&mut program, m);
            straus_ec::<G2Ops>(&mut program, &bases, &G2Ops::digits(&m.scalars))
        })
        .collect();
    program.end_section("g2");

    let pairs = [
        (g1[0].clone(), g2[0].clone()),
        (h1, g2[1].clone()),
        (g1[1].clone(), h2),
        (g1[2].clone(), g2_0),
    ];
    let miller = miller_loop(&mut program, &pairs);
    let lhs = final_exponentiation(&mut program, &miller);
    program.end_section("final_exp");
    let final_check: [RowId; 12] = std::array::from_fn(|c| {
        let mut slots = program.linear(&lhs[c], 1);
        slots.extend(program.linear(&rhs[c], -1));
        program.pinned(slots, Fq::ZERO)
    });
    program.end_section("final_check");
    Layout {
        program,
        sigma,
        n_commitments: n,
        rhs,
        miller,
        lhs,
        final_check,
    }
}
