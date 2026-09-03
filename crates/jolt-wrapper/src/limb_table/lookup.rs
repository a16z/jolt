//! The digit columns and the operand lookup: every slotted row of a
//! digit-selected op carries its digit bits `(zero, neg, e0, e1, e2)`, looks
//! up `(key, fingerprint)` in the table rows' `(row index, f_±)` with LogUp,
//! and the public multilinears the row member reads (`sel`, kind indicators,
//! `S0`, the constancy weights) are built here for the prover and evaluated
//! in closed form for the verifier.

use crate::stream::TermObserver;
use jolt_field::{Field, Fr, One, Ring, Zero};

use super::layout::{Bits, Factor, Rel, LOG_ROWS, ROWS};
use super::relation::{PublicEvals, RowRelation, FP_SLOTS_G1, FP_SLOTS_G2, FP_SLOTS_GT};
use super::schedule::{Layout, SelectedFamily};
use super::terms::{plain, powers_with, Mul};
use super::verifier::{Evaluator, Point};
use super::wiring::{copy_kernel_eval, ReadKind};
use ark_bn254::Fr as ArkFr;
use std::ops::Range;

pub const DIGIT_BITS: usize = 5;
const NEG_KEY_OFFSET: u64 = 1 << LOG_ROWS;

/// `(zero, neg, e0, e1, e2)` of digit `j` (`d = j − 8`, `e = |d| − 1`).
pub fn digit_bits(j: u8) -> [u8; DIGIT_BITS] {
    let d = i32::from(j) - 8;
    if d == 0 {
        return [1, 0, 0, 0, 0];
    }
    let e = d.unsigned_abs() - 1;
    [
        0,
        u8::from(d < 0),
        (e & 1) as u8,
        ((e >> 1) & 1) as u8,
        ((e >> 2) & 1) as u8,
    ]
}

/// Row stride of an EC table entry (`16·d` for G1 cells, `8·d` for G2 half cells).
pub const fn ec_stride(kind: ReadKind) -> u32 {
    match kind {
        ReadKind::G1 => 16,
        ReadKind::G2 => 8,
        ReadKind::None | ReadKind::Gt => 0,
    }
}

/// Prover-side public columns and per-row read kinds.
pub struct PublicColumns {
    pub kinds: Vec<ReadKind>,
    pub sel: Vec<Fr>,
    pub is_gt: Vec<Fr>,
    pub is_g1: Vec<Fr>,
    pub is_g2: Vec<Fr>,
    pub s0: Vec<Fr>,
    /// The own coordinate `c(x)` of every selected row.
    pub coord: Vec<Fr>,
    /// Rows whose digit bits must equal the previous row's.
    pub constancy: Vec<u8>,
    pub digits: [Vec<u8>; DIGIT_BITS],
    /// The digit value `d` per row.
    pub digit_values: Vec<Fr>,
    /// `(key row, conjugated)` per reading row.
    pub keys: Vec<Option<(u32, bool)>>,
}

impl SelectedFamily {
    /// `S0(x)` for a row of this family.
    pub fn s0(&self, x: u32) -> i64 {
        i64::from(x)
            + self.key.constant
            + self.key.k_coeff * i64::from(self.k_bits.extract(x))
            + self.key.w_coeff * i64::from(self.w_bits.extract(x))
    }
}

/// The lookup key row of a reading row from its digit (mirrors the row
/// member's key polynomial on integers).
pub fn key_row(kind: ReadKind, s0: i64, j: u8, one_row: u32, c: u32) -> (u32, bool) {
    let d = i64::from(j) - 8;
    match kind {
        ReadKind::Gt => {
            if d == 0 {
                (one_row + c, false)
            } else {
                let e = d.unsigned_abs() - 1;
                ((s0 + 16 * e as i64) as u32, d < 0)
            }
        }
        ReadKind::G1 | ReadKind::G2 => ((s0 + i64::from(ec_stride(kind)) * d) as u32, false),
        ReadKind::None => unreachable!("reading rows have a kind"),
    }
}

impl PublicColumns {
    pub fn new(layout: &Layout) -> Self {
        let mut kinds = vec![ReadKind::None; ROWS];
        let (mut sel, mut is_gt, mut is_g1, mut is_g2, mut s0) = (
            vec![Fr::zero(); ROWS],
            vec![Fr::zero(); ROWS],
            vec![Fr::zero(); ROWS],
            vec![Fr::zero(); ROWS],
            vec![Fr::zero(); ROWS],
        );
        let mut coord = vec![Fr::zero(); ROWS];
        let mut constancy = vec![0u8; ROWS];
        let mut digits: [Vec<u8>; DIGIT_BITS] = std::array::from_fn(|_| vec![0u8; ROWS]);
        let mut digit_values = vec![Fr::zero(); ROWS];
        let mut keys = vec![None; ROWS];
        let one_row = layout.one_cell * 16;
        for op in &layout.digit_ops {
            let family = &layout.selected[usize::from(op.family)];
            let bits = digit_bits(op.j);
            for i in 0..u32::from(op.rows) {
                let x = op.first_row + i;
                let r = x as usize;
                kinds[r] = op.kind;
                sel[r] = Fr::one();
                match op.kind {
                    ReadKind::Gt => is_gt[r] = Fr::one(),
                    ReadKind::G1 => is_g1[r] = Fr::one(),
                    ReadKind::G2 => is_g2[r] = Fr::one(),
                    ReadKind::None => unreachable!("digit ops read"),
                }
                let s0_x = family.s0(x);
                s0[r] = Fr::from_i64(s0_x);
                coord[r] = Fr::from_u64(u64::from(family.c_bits.extract(x)));
                if i > 0 {
                    constancy[r] = 1;
                }
                for (column, bit) in digits.iter_mut().zip(bits) {
                    column[r] = bit;
                }
                digit_values[r] = Fr::from_i64(i64::from(op.j) - 8);
                keys[r] = Some(key_row(
                    op.kind,
                    s0_x,
                    op.j,
                    one_row,
                    family.c_bits.extract(x),
                ));
            }
        }
        Self {
            kinds,
            sel,
            is_gt,
            is_g1,
            is_g2,
            s0,
            coord,
            constancy,
            digits,
            digit_values,
            keys,
        }
    }

    /// `W(x) = eq(τ,x)·cst(x) − eq(τ,x+1)·cst(x+1)`: with `Σ_x W(x)·bit(x) = 0`
    /// the digit bits are constant over each op's slotted rows.
    pub fn constancy_weights(&self, eq_tau: &[Fr]) -> Vec<Fr> {
        (0..ROWS)
            .map(|x| {
                let here = if self.constancy[x] == 1 {
                    eq_tau[x]
                } else {
                    Fr::zero()
                };
                let next = if x + 1 < ROWS && self.constancy[x + 1] == 1 {
                    eq_tau[x + 1]
                } else {
                    Fr::zero()
                };
                here - next
            })
            .collect()
    }

    /// `small(x) = [x < 2^16]` and `id(x) = x`.
    pub fn small_and_id() -> (Vec<Fr>, Vec<Fr>) {
        let small = (0..ROWS)
            .map(|x| if x < 1 << 16 { Fr::one() } else { Fr::zero() })
            .collect();
        let id = (0..ROWS as u64).map(Fr::from_u64).collect();
        (small, id)
    }

    /// The range inverse table column `1/(α − x)` for `x < 2^16`, zero above.
    pub fn inverse_table(alpha: Fr) -> Vec<Fr> {
        let mut values: Vec<ArkFr> = (0..1u64 << 16)
            .map(|v| ArkFr::from(alpha - Fr::from_u64(v)))
            .collect();
        ark_ff::batch_inversion(&mut values);
        let mut column: Vec<Fr> = values.into_iter().map(Fr::from).collect();
        column.resize(ROWS, Fr::zero());
        column
    }
}

/// The prover's lookup columns.
pub struct LookupColumns {
    pub m_pos: Vec<Fr>,
    pub m_neg: Vec<Fr>,
    pub h: Vec<Fr>,
    pub g_pos: Vec<Fr>,
    pub g_neg: Vec<Fr>,
}

fn batch_invert(values: &mut [Fr]) {
    let mut ark: Vec<ArkFr> = values.iter().map(|v| ArkFr::from(*v)).collect();
    ark_ff::batch_inversion(&mut ark);
    for (slot, v) in values.iter_mut().zip(ark) {
        *slot = Fr::from(v);
    }
}

impl LookupColumns {
    /// `operands[SLOTS + s]` is `Y_s`; `f_pos`, `f_neg` the fingerprint columns.
    pub fn new(
        public: &PublicColumns,
        operands: &[Vec<Fr>],
        f_pos: &[Fr],
        f_neg: &[Fr],
        relation: &RowRelation,
    ) -> Self {
        let ch = &relation.challenges;
        let slots = operands.len() / 2;
        let mut m_pos = vec![0u32; ROWS];
        let mut m_neg = vec![0u32; ROWS];
        let mut h = vec![Fr::zero(); ROWS];
        let mut h_rows = Vec::new();
        for (x, key) in public.keys.iter().enumerate() {
            let Some((row, conjugated)) = key else {
                continue;
            };
            if *conjugated {
                m_neg[*row as usize] += 1;
            } else {
                m_pos[*row as usize] += 1;
            }
            let n = public.kinds[x].fp_slots();
            let fingerprint = (0..n).fold(Fr::zero(), |acc, s| {
                acc + relation.fingerprint_weight(s) * operands[slots + s][x]
            });
            let offset = if *conjugated {
                Fr::from_u64(NEG_KEY_OFFSET)
            } else {
                Fr::zero()
            };
            h[x] = ch.beta + Fr::from_u64(u64::from(*row)) + offset + ch.fp_combine * fingerprint;
            h_rows.push(x);
        }
        let mut h_values: Vec<Fr> = h_rows.iter().map(|&x| h[x]).collect();
        batch_invert(&mut h_values);
        for (&x, v) in h_rows.iter().zip(h_values) {
            h[x] = v;
        }
        let table = |m: &[u32], f: &[Fr], offset: Fr| -> Vec<Fr> {
            let rows: Vec<usize> = (0..ROWS).filter(|&x| m[x] > 0).collect();
            let mut dens: Vec<Fr> = rows
                .iter()
                .map(|&x| ch.beta + Fr::from_u64(x as u64) + offset + ch.fp_combine * f[x])
                .collect();
            batch_invert(&mut dens);
            let mut g = vec![Fr::zero(); ROWS];
            for (&x, inv) in rows.iter().zip(dens) {
                g[x] = Fr::from_u64(u64::from(m[x])) * inv;
            }
            g
        };
        let g_pos = table(&m_pos, f_pos, Fr::zero());
        let g_neg = table(&m_neg, f_neg, Fr::from_u64(NEG_KEY_OFFSET));
        Self {
            m_pos: m_pos
                .into_iter()
                .map(|m| Fr::from_u64(u64::from(m)))
                .collect(),
            m_neg: m_neg
                .into_iter()
                .map(|m| Fr::from_u64(u64::from(m)))
                .collect(),
            h,
            g_pos,
            g_neg,
        }
    }
}

// ----- verifier closed forms ------------------------------------------------

/// Fields of the 18 row bits as `(bits, range)`: the family's restricted
/// fields (from its `restrict` factors) and the free gaps (full range).
fn field_partition(domain: &[Factor]) -> Vec<(Bits, Range<u32>, Option<Factor>)> {
    let mut fields: Vec<(Bits, Range<u32>, Option<Factor>)> = Vec::new();
    for factor in domain {
        assert!(factor.v.width() == 0, "domain factors have no source field");
        let range = match (&factor.rel, &factor.range) {
            (Rel::Const(_), Some(range)) => range.clone(),
            _ => 0..1 << factor.u.width(),
        };
        fields.push((factor.u, range, Some(factor.clone())));
    }
    fields.sort_by_key(|(bits, _, _)| bits.lo);
    let mut all = Vec::new();
    let mut lo = 0u8;
    for (bits, range, factor) in fields {
        assert!(bits.lo >= lo, "overlapping domain fields");
        if bits.lo > lo {
            let gap = Bits::new(lo, bits.lo);
            all.push((gap, 0..1 << gap.width(), None));
        }
        lo = bits.hi;
        all.push((bits, range, factor));
    }
    if lo < LOG_ROWS as u8 {
        let gap = Bits::new(lo, LOG_ROWS as u8);
        all.push((gap, 0..1 << gap.width(), None));
    }
    all
}

impl SelectedFamily {
    /// Per-field `(Σ eq, Σ eq·u)` over the family's rows at the evaluator's
    /// row point; `Weight` factors contribute their mask sum with a zero moment.
    fn moments<O: TermObserver + ?Sized>(&self, ev: &mut Evaluator<'_, O>) -> Vec<(Bits, Fr, Fr)> {
        field_partition(&self.domain)
            .into_iter()
            .map(|(bits, range, factor)| match factor {
                Some(f) if !matches!(f.rel, Rel::Const(_)) => {
                    (bits, ev.field(std::slice::from_ref(&f)), Fr::zero())
                }
                _ => {
                    let (sum, moment) = ev.field_moment(Point::Row, bits, range);
                    (bits, sum, moment)
                }
            })
            .collect()
    }

    /// `Σ_{x ∈ family} eq(r, x)`.
    pub fn indicator<O: TermObserver + ?Sized>(&self, ev: &mut Evaluator<'_, O>) -> Fr {
        ev.kernel(&self.domain)
    }

    /// `Σ_{x ∈ family} eq(r, x)·c(x)`.
    pub fn coord_eval<O: TermObserver + ?Sized>(&self, ev: &mut Evaluator<'_, O>) -> Fr {
        let moments = self.moments(ev);
        let mut total = Fr::zero();
        for (i, (bits, _, moment)) in moments.iter().enumerate() {
            if *bits != self.c_bits {
                continue;
            }
            let others = moments
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .fold(Fr::one(), |acc, (_, (_, sum, _))| acc * *sum);
            total += *moment * others;
        }
        total
    }

    /// `Σ_{x ∈ family} eq(r, x)·S0(x)`.
    pub fn s0_eval<O: TermObserver + ?Sized>(&self, ev: &mut Evaluator<'_, O>) -> Fr {
        let moments = self.moments(ev);
        let indicator = moments
            .iter()
            .fold(Fr::one(), |acc, (_, sum, _)| acc * *sum);
        let mut total = indicator * Fr::from_i64(self.key.constant);
        for (i, (bits, _, moment)) in moments.iter().enumerate() {
            let mut coefficient = Fr::from_u64(1u64 << bits.lo);
            if *bits == self.k_bits {
                coefficient += Fr::from_i64(self.key.k_coeff);
            }
            if *bits == self.w_bits {
                coefficient += Fr::from_i64(self.key.w_coeff);
            }
            let others = moments
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .fold(Fr::one(), |acc, (_, (_, sum, _))| acc * *sum);
            total += coefficient * *moment * others;
        }
        total
    }

    /// `Σ_{x ∈ family, c(x) = first_c} eq(r, x)·ρ^{kd(k(x))}/mult(kd)·16^{63 − w(x)}`.
    pub fn omega_eval<O: TermObserver + ?Sized>(
        &self,
        ev: &mut Evaluator<'_, O>,
        rho_weights: &[Fr],
    ) -> Fr {
        let mut product = Fr::one();
        for (bits, range, factor) in field_partition(&self.domain) {
            let part = if bits == self.c_bits {
                ev.eq_const(Point::Row, bits, self.first_c)
            } else if bits == self.k_bits {
                let weights = &self.digit_base;
                let f = |k: u32| -> Fr {
                    weights
                        .iter()
                        .find(|(kk, _)| *kk == k)
                        .map_or(Fr::zero(), |(_, kd)| rho_weights[*kd as usize])
                };
                let admitted: Vec<u32> = weights.iter().map(|(k, _)| *k).collect();
                let (lo, hi) = (
                    *admitted.iter().min().unwrap_or(&0),
                    *admitted.iter().max().unwrap_or(&0) + 1,
                );
                ev.field_sum(Point::Row, bits, lo..hi, &f)
            } else if bits == self.w_bits {
                if range == (0..1 << bits.width()) {
                    // `Σ_w eq(r_w, w)·16^{63−w} = 16^{63}·Π_i (1 − r_i + r_i·16^{−2^i})`.
                    let sixteen_inverse = Fr::pow2(4)
                        .inverse()
                        .unwrap_or_else(|| unreachable!("16 is invertible"));
                    ev.geometric(Point::Row, bits, sixteen_inverse) * Fr::pow2(4 * 63)
                } else {
                    let f = |w: u32| -> Fr { Fr::pow2(4 * (63 - w as usize)) };
                    ev.field_sum(Point::Row, bits, range, &f)
                }
            } else {
                match factor {
                    Some(f) => ev.field(std::slice::from_ref(&f)),
                    None => Fr::one(),
                }
            };
            product *= part;
        }
        product
    }

    /// The two constancy kernels at (`τ`, `r`): rows `c ∈ first_c+1..first_c+rows`
    /// reading themselves and their predecessor.
    pub fn constancy_kernels<O: TermObserver + ?Sized>(&self, ev: &mut Evaluator<'_, O>) -> Fr {
        let range = self.first_c + 1..self.first_c + self.rows;
        let kernel = |delta: i64| -> Vec<Factor> {
            field_partition(&self.domain)
                .into_iter()
                .map(|(bits, field_range, _)| {
                    if bits == self.c_bits {
                        Factor::shift(bits, bits, delta).with_range(range.clone())
                    } else {
                        Factor::same(bits, bits).with_range(field_range)
                    }
                })
                .collect()
        };
        ev.kernel(&kernel(0)) - ev.kernel(&kernel(-1))
    }
}

/// `ρ^{kd}/mult(kd)` per digit base (the digit link's per-base weight).
pub fn rho_weights(layout: &Layout, rho: Fr) -> Vec<Fr> {
    rho_weights_with(layout, rho, &mut plain)
}

/// [`rho_weights`] with every product observed by `mul` (the verifier's
/// derivation): the `ρ` powers, plus one scaling per base several MSMs share.
pub fn rho_weights_with(layout: &Layout, rho: Fr, mul: Mul<'_>) -> Vec<Fr> {
    let mut multiplicity = vec![0u64; layout.digit_bases as usize];
    for op in &layout.digit_ops {
        if op.w == 0 {
            multiplicity[op.kd as usize] += 1;
        }
    }
    let powers = powers_with(rho, layout.digit_bases as usize, mul);
    let mut inverses: Vec<(u64, Fr)> = Vec::new();
    powers
        .iter()
        .zip(&multiplicity)
        .map(|(power, &m)| match m {
            0 => Fr::zero(),
            1 => *power,
            _ => {
                let inverse = if let Some((_, v)) = inverses.iter().find(|(k, _)| *k == m) {
                    *v
                } else {
                    let v = Fr::from_u64(m)
                        .inverse()
                        .unwrap_or_else(|| unreachable!("nonzero multiplicity"));
                    inverses.push((m, v));
                    v
                };
                mul(*power, inverse)
            }
        })
        .collect()
}

/// The prover's `ω(x)` table: `ρ^{kd}/mult · 16^{63−w}` on every op's first slotted row.
pub fn omega_column(layout: &Layout, rho: Fr) -> Vec<Fr> {
    let weights = rho_weights(layout, rho);
    let mut omega = vec![Fr::zero(); ROWS];
    for op in &layout.digit_ops {
        omega[op.first_row as usize] = weights[op.kd as usize] * Fr::pow2(4 * (63 - op.w as usize));
    }
    omega
}

/// Verifier: `ω̃(r)` over every selected family.
pub fn omega_eval<O: TermObserver + ?Sized>(
    layout: &Layout,
    rho: Fr,
    r_le: &[Fr],
    observer: &mut O,
) -> Fr {
    let mut ev = Evaluator::new(r_le, r_le, observer);
    omega_eval_in(layout, rho, &mut ev)
}

fn omega_eval_in<O: TermObserver + ?Sized>(
    layout: &Layout,
    rho: Fr,
    ev: &mut Evaluator<'_, O>,
) -> Fr {
    let weights = rho_weights_with(layout, rho, &mut |a, b| ev.mul(a, b));
    layout.selected.iter().fold(Fr::zero(), |acc, family| {
        acc + family.omega_eval(ev, &weights)
    })
}

/// [`public_evals`] and `ω̃(r)` over one evaluator: the digit link reuses the
/// stage point's eq tables.
pub fn public_and_omega_evals<O: TermObserver + ?Sized>(
    layout: &Layout,
    relation: &RowRelation,
    tau_le: &[Fr],
    r_le: &[Fr],
    rho: Fr,
    observer: &mut O,
) -> (PublicEvals, Fr) {
    let (public, mut ev) = public_evals_with(layout, relation, tau_le, r_le, observer);
    let omega = omega_eval_in(layout, rho, &mut ev);
    (public, omega)
}

/// Verifier: the public multilinears of the row member at the stage point
/// `r` (little-endian) for the copy point `τ` (little-endian).
pub fn public_evals<O: TermObserver + ?Sized>(
    layout: &Layout,
    relation: &RowRelation,
    tau_le: &[Fr],
    r_le: &[Fr],
    observer: &mut O,
) -> PublicEvals {
    public_evals_with(layout, relation, tau_le, r_le, observer).0
}

fn public_evals_with<'o, O: TermObserver + ?Sized>(
    layout: &Layout,
    relation: &RowRelation,
    tau_le: &[Fr],
    r_le: &[Fr],
    observer: &'o mut O,
) -> (PublicEvals, Evaluator<'o, O>) {
    let eq_tau = tau_le.iter().zip(r_le).fold(Fr::one(), |acc, (t, r)| {
        acc * (*t * *r + (Fr::one() - *t) * (Fr::one() - *r))
    });
    let mut ev = Evaluator::new(tau_le, r_le, observer);
    let copy_kernel = copy_kernel_eval(&mut ev, &layout.copies, &layout.fingerprints, relation);
    let constancy = layout
        .selected
        .iter()
        .fold(Fr::zero(), |acc, f| acc + f.constancy_kernels(&mut ev));
    let mut ev = ev.rebase_row_to_src();
    let (mut sel, mut is_gt, mut is_g1, mut is_g2, mut s0, mut coord) = (
        Fr::zero(),
        Fr::zero(),
        Fr::zero(),
        Fr::zero(),
        Fr::zero(),
        Fr::zero(),
    );
    for family in &layout.selected {
        let indicator = family.indicator(&mut ev);
        sel += indicator;
        match family.kind {
            ReadKind::Gt => is_gt += indicator,
            ReadKind::G1 => is_g1 += indicator,
            ReadKind::G2 => is_g2 += indicator,
            ReadKind::None => unreachable!("selected families read"),
        }
        s0 += family.s0_eval(&mut ev);
        coord += family.coord_eval(&mut ev);
    }
    let small = r_le[16..]
        .iter()
        .fold(Fr::one(), |acc, r| acc * (Fr::one() - *r));
    let id = r_le
        .iter()
        .enumerate()
        .fold(Fr::zero(), |acc, (i, r)| acc + *r * Fr::from_u64(1u64 << i));
    let public = PublicEvals {
        eq_tau,
        copy_kernel,
        sel,
        is_gt,
        is_g1,
        is_g2,
        s0,
        coord,
        constancy,
        small,
        id,
    };
    (public, ev)
}

const _: () = assert!(FP_SLOTS_GT >= FP_SLOTS_G2 && FP_SLOTS_G2 >= FP_SLOTS_G1);
