//! The digit columns and the operand lookup: every slotted row of a
//! digit-selected op carries its digit bits `(zero, neg, e0, e1, e2)`, looks
//! up `(key, fingerprint)` in the table rows' `(row index, f_±)` with LogUp,
//! and the public multilinears the row member reads (`sel`, kind indicators,
//! `S0`, the constancy weights) are built here for the prover and evaluated
//! in closed form for the verifier.

use crate::stream::TermObserver;
use jolt_field::{Fr, One, Ring, Zero};

use super::digits::{WINDOWS, WINDOW_ROWS, WINDOW_TOP_DIGITS};
use super::layout::{Bits, Factor, Rel, LOG_ROWS, ROWS};
use super::literals::{SIXTEEN_INVERSE, SIXTEEN_POWERS, WINDOW_BOUND_FR};
use super::relation::{PublicEvals, RowRelation, FP_SLOTS_G1, FP_SLOTS_G2, FP_SLOTS_GT};
use super::schedule::{Layout, SelectedFamily, WINDOW_ROW_BASE};
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
    /// Lookup multiplicities: reads of every key row, plain and conjugated.
    pub m_pos: Vec<u32>,
    pub m_neg: Vec<u32>,
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
        let mut m_pos = vec![0u32; ROWS];
        let mut m_neg = vec![0u32; ROWS];
        for (row, conjugated) in keys.iter().flatten() {
            if *conjugated {
                m_neg[*row as usize] += 1;
            } else {
                m_pos[*row as usize] += 1;
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
            m_pos,
            m_neg,
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

/// The prover's phase-2c lookup columns.
pub struct LookupColumns {
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
    /// `y[s]` is the operand column `Y_s`; `f_pos`, `f_neg` the fingerprint
    /// columns; `fp_pow[s]` the fingerprint weight of slot `s`.
    pub fn new(
        public: &PublicColumns,
        y: &[&[Fr]],
        f_pos: &[Fr],
        f_neg: &[Fr],
        fp_pow: &[Fr],
        beta: Fr,
        fp_combine: Fr,
    ) -> Self {
        let mut h = vec![Fr::zero(); ROWS];
        let mut h_rows = Vec::new();
        for (x, key) in public.keys.iter().enumerate() {
            let Some((row, conjugated)) = key else {
                continue;
            };
            let n = public.kinds[x].fp_slots();
            let fingerprint = (0..n).fold(Fr::zero(), |acc, s| acc + fp_pow[s] * y[s][x]);
            let offset = if *conjugated {
                Fr::from_u64(NEG_KEY_OFFSET)
            } else {
                Fr::zero()
            };
            h[x] = beta + Fr::from_u64(u64::from(*row)) + offset + fp_combine * fingerprint;
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
                .map(|&x| beta + Fr::from_u64(x as u64) + offset + fp_combine * f[x])
                .collect();
            batch_invert(&mut dens);
            let mut g = vec![Fr::zero(); ROWS];
            for (&x, inv) in rows.iter().zip(dens) {
                g[x] = Fr::from_u64(u64::from(m[x])) * inv;
            }
            g
        };
        let g_pos = table(&public.m_pos, f_pos, Fr::zero());
        let g_neg = table(&public.m_neg, f_neg, Fr::from_u64(NEG_KEY_OFFSET));
        Self { h, g_pos, g_neg }
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

    /// The family's public multilinears at the evaluator's row point:
    /// `(Σ_x eq(r, x), Σ_x eq(r, x)·S0(x), Σ_x eq(r, x)·c(x))` over its rows,
    /// the per-field sums and moments combined through the all-but-one
    /// products the three share.
    pub fn public_evals<O: TermObserver + ?Sized>(
        &self,
        ev: &mut Evaluator<'_, O>,
    ) -> (Fr, Fr, Fr) {
        let moments = self.moments(ev);
        let sums: Vec<Fr> = moments.iter().map(|(_, sum, _)| *sum).collect();
        let (indicator, others) = ev.all_but_one_products(&sums);
        let mut s0 = if self.key.constant == 0 {
            Fr::zero()
        } else {
            ev.mul(indicator, Fr::from_i64(self.key.constant))
        };
        let mut coord = Fr::zero();
        for (i, (bits, _, moment)) in moments.iter().enumerate() {
            if moment.is_zero() {
                continue;
            }
            let mut coefficient = Fr::from_u64(1u64 << bits.lo);
            if *bits == self.k_bits {
                coefficient += Fr::from_i64(self.key.k_coeff);
            }
            if *bits == self.w_bits {
                coefficient += Fr::from_i64(self.key.w_coeff);
            }
            let weighted = ev.mul(coefficient, *moment);
            s0 += ev.mul(weighted, others[i]);
            if *bits == self.c_bits {
                coord = ev.mul(*moment, others[i]);
            }
        }
        (indicator, s0, coord)
    }

    /// The digit link's weight over the family at the evaluator's row point:
    /// `Σ_{x: c(x) = first_c} eq(r, x)·ρ^{link(k(x))}·(16^{63−w(x)} +
    /// [w(x) < 16]·ρ^M·16^{15−w(x)})` — the recoded integer's weight plus the
    /// window check's top-16-digit weight.
    pub fn omega_eval<O: TermObserver + ?Sized>(
        &self,
        ev: &mut Evaluator<'_, O>,
        powers: &LinkPowers,
    ) -> Fr {
        let mut product: Option<Fr> = None;
        for (bits, range, factor) in field_partition(&self.domain) {
            let part = if bits == self.c_bits {
                ev.eq_const(Point::Row, bits, self.first_c)
            } else if bits == self.k_bits {
                let bases = &self.digit_base;
                let f = |k: u32| -> Fr {
                    bases
                        .iter()
                        .find(|(kk, _)| *kk == k)
                        .map_or(Fr::zero(), |(_, link)| powers.occurrence[*link as usize])
                };
                let admitted: Vec<u32> = bases.iter().map(|(k, _)| *k).collect();
                let (lo, hi) = (
                    *admitted.iter().min().unwrap_or(&0),
                    *admitted.iter().max().unwrap_or(&0) + 1,
                );
                ev.field_sum(Point::Row, bits, lo..hi, &f)
            } else if bits == self.w_bits {
                let (all, top) = if range == (0..1 << bits.width()) {
                    // `Σ_w eq(r_w, w)·16^{63−w} = 16^{63}·Π_i (1 − r_i + r_i·16^{−2^i})`;
                    // the top 16 windows are the field's high bits zero.
                    let geometric = ev.geometric(Point::Row, bits, sixteen_inverse());
                    let all = ev.mul(geometric, sixteen_pow(WINDOWS - 1));
                    let low = Bits::new(bits.lo, bits.lo + 4);
                    let high = Bits::new(bits.lo + 4, bits.hi);
                    let low_geometric = ev.geometric(Point::Row, low, sixteen_inverse());
                    let high_zero = ev.eq_const(Point::Row, high, 0);
                    let top = ev.mul(low_geometric, high_zero);
                    (all, ev.mul(top, sixteen_pow(WINDOW_TOP_DIGITS - 1)))
                } else {
                    let all = ev.field_sum(Point::Row, bits, range.clone(), &|w| {
                        sixteen_pow(WINDOWS - 1 - w as usize)
                    });
                    let top_range = range.start..range.end.min(WINDOW_TOP_DIGITS as u32);
                    let top = if top_range.is_empty() {
                        Fr::zero()
                    } else {
                        ev.field_sum(Point::Row, bits, top_range, &|w| {
                            sixteen_pow(WINDOW_TOP_DIGITS - 1 - w as usize)
                        })
                    };
                    (all, top)
                };
                all + ev.mul(powers.window_digits, top)
            } else {
                let Some(f) = factor else { continue };
                ev.field(std::slice::from_ref(&f))
            };
            product = Some(match product {
                Some(p) => ev.mul(p, part),
                None => part,
            });
        }
        product.unwrap_or(Fr::one())
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

/// `16^k` from its literal (`k < WINDOWS`).
fn sixteen_pow(k: usize) -> Fr {
    Fr::from(SIXTEEN_POWERS[k])
}

/// `16^{−1}` from its literal.
fn sixteen_inverse() -> Fr {
    Fr::from(SIXTEEN_INVERSE)
}

/// The digit link's `ρ` powers for a layout with `M = link_occurrences`
/// chain-base occurrences: `ρ^o` for `o ≤ M`, the window check's bases `ρ^M`
/// (digit side) and `ρ^{M+256}` (chunk side), and `Σ_{o<256} ρ^o`.
pub struct LinkPowers {
    pub occurrence: Vec<Fr>,
    pub window_digits: Fr,
    pub window_chunks: Fr,
    pub window_sum: Fr,
}

impl LinkPowers {
    pub fn new(layout: &Layout, rho: Fr) -> Self {
        Self::new_with(layout, rho, &mut plain)
    }

    /// [`Self::new`] with every product observed by `mul`.
    pub fn new_with(layout: &Layout, rho: Fr, mul: Mul<'_>) -> Self {
        let m = layout.link_occurrences as usize;
        let occurrence = powers_with(rho, m + 1, mul);
        // `ρ^{2^i}` by squaring: `Σ_{o<256} ρ^o = Π_{i<8} (1 + ρ^{2^i})` and `ρ^256`.
        let mut window_sum = Fr::one() + rho;
        let mut square = mul(rho, rho);
        for _ in 1..8 {
            window_sum = mul(window_sum, Fr::one() + square);
            square = mul(square, square);
        }
        let window_digits = occurrence[m];
        Self {
            occurrence,
            window_digits,
            window_chunks: mul(window_digits, square),
            window_sum,
        }
    }

    /// `W_kd(ρ) = Σ ρ^o` over the chain-base occurrences `o` of every digit
    /// base `kd` (the named wires in the published order, the constant one,
    /// then `θ`).
    pub fn base_weights(&self, layout: &Layout) -> Vec<Fr> {
        let mut weights = vec![Fr::zero(); layout.digit_bases as usize];
        for op in layout.digit_ops.iter().filter(|op| op.w == 0) {
            weights[op.kd as usize] += self.occurrence[op.link as usize];
        }
        weights
    }

    /// The window checks' public constant `WINDOW_BOUND·ρ^{M+256}·Σ_{o<256} ρ^o`
    /// (the chunk-side identities `V(o) + V'(o) = WINDOW_BOUND` summed).
    pub fn window_constant(&self, mul: Mul<'_>) -> Fr {
        let scaled = mul(Fr::from(WINDOW_BOUND_FR), self.window_chunks);
        mul(scaled, self.window_sum)
    }
}

/// The digit link's weight of every digit base: `W_kd(ρ) = Σ ρ^o` over the
/// chain-base occurrences of `kd`. R's scalar link publishes `Σ_k W_k(ρ)·s_k`
/// over the named wires while T2 sums `ρ^o` per occurrence, so every chain's
/// recoding is bound to its scalar on its own (no averaging across the chains
/// sharing a scalar).
pub fn link_weights(layout: &Layout, rho: Fr) -> Vec<Fr> {
    link_weights_with(layout, rho, &mut plain)
}

/// [`link_weights`] with every product observed by `mul`.
pub fn link_weights_with(layout: &Layout, rho: Fr, mul: Mul<'_>) -> Vec<Fr> {
    LinkPowers::new_with(layout, rho, mul).base_weights(layout)
}

/// The digit link's public weight columns: `ω` on every op's first slotted
/// row (`ρ^o·16^{63−w}`, plus `ρ^{M+o}·16^{15−w}` on the top 16 windows) and
/// the window rows' chunk weights `κ = ρ^o·(ρ^{M+256} − ρ^M)` (on `V`) and
/// `κ' = ρ^{M+256+o}` (on `V'`).
pub struct LinkColumns {
    pub omega: Vec<Fr>,
    pub kappa: Vec<Fr>,
    pub kappa_prime: Vec<Fr>,
}

pub fn link_columns(layout: &Layout, rho: Fr) -> LinkColumns {
    let powers = LinkPowers::new(layout, rho);
    let mut omega = vec![Fr::zero(); ROWS];
    for op in &layout.digit_ops {
        let w = op.w as usize;
        let mut weight = sixteen_pow(WINDOWS - 1 - w);
        if w < WINDOW_TOP_DIGITS {
            weight += powers.window_digits * sixteen_pow(WINDOW_TOP_DIGITS - 1 - w);
        }
        omega[op.first_row as usize] = powers.occurrence[op.link as usize] * weight;
    }
    let mut kappa = vec![Fr::zero(); ROWS];
    let mut kappa_prime = vec![Fr::zero(); ROWS];
    let chunks_minus_digits = powers.window_chunks - powers.window_digits;
    for (o, rho_o) in powers_with(rho, WINDOW_ROWS, &mut plain)
        .into_iter()
        .enumerate()
    {
        let row = WINDOW_ROW_BASE as usize + o;
        kappa[row] = rho_o * chunks_minus_digits;
        kappa_prime[row] = rho_o * powers.window_chunks;
    }
    LinkColumns {
        omega,
        kappa,
        kappa_prime,
    }
}

/// The digit link's public multilinears at the stage point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinkEvals {
    pub omega: Fr,
    pub kappa: Fr,
    pub kappa_prime: Fr,
}

/// Verifier: the digit link's weights at `r` — `ω̃` over every selected
/// family, `κ̃`/`κ̃'` over the window block.
pub fn link_evals<O: TermObserver + ?Sized>(
    layout: &Layout,
    rho: Fr,
    r_le: &[Fr],
    observer: &mut O,
) -> LinkEvals {
    let mut ev = Evaluator::new(r_le, r_le, observer);
    link_evals_in(layout, rho, &mut ev)
}

fn link_evals_in<O: TermObserver + ?Sized>(
    layout: &Layout,
    rho: Fr,
    ev: &mut Evaluator<'_, O>,
) -> LinkEvals {
    let powers = LinkPowers::new_with(layout, rho, &mut |a, b| ev.mul(a, b));
    let omega = layout.selected.iter().fold(Fr::zero(), |acc, family| {
        acc + family.omega_eval(ev, &powers)
    });
    // Window rows `WINDOW_ROW_BASE + o`, `o < 256`: `eq(r_hi, base)·Σ_o ρ^o·eq(r_lo, o)`.
    let low = Bits::new(0, WINDOW_ROWS.trailing_zeros() as u8);
    let high = Bits::new(low.hi, LOG_ROWS as u8);
    let block = ev.eq_const(Point::Row, high, WINDOW_ROW_BASE >> low.hi);
    let geometric = ev.geometric(Point::Row, low, rho);
    let window = ev.mul(block, geometric);
    LinkEvals {
        omega,
        kappa: ev.mul(window, powers.window_chunks - powers.window_digits),
        kappa_prime: ev.mul(window, powers.window_chunks),
    }
}

/// [`public_evals`] and the digit link's weights over one evaluator: the link
/// reuses the stage point's eq tables.
pub fn public_and_link_evals<O: TermObserver + ?Sized>(
    layout: &Layout,
    relation: &RowRelation,
    tau_le: &[Fr],
    r_le: &[Fr],
    rho: Fr,
    observer: &mut O,
) -> (PublicEvals, LinkEvals) {
    let (public, mut ev) = public_evals_with(layout, relation, tau_le, r_le, observer);
    let link = link_evals_in(layout, rho, &mut ev);
    (public, link)
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
    let mut ev = Evaluator::new(tau_le, r_le, observer);
    let eq_tau = ev.eq_row_src();
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
        let (indicator, family_s0, family_coord) = family.public_evals(&mut ev);
        sel += indicator;
        match family.kind {
            ReadKind::Gt => is_gt += indicator,
            ReadKind::G1 => is_g1 += indicator,
            ReadKind::G2 => is_g2 += indicator,
            ReadKind::None => unreachable!("selected families read"),
        }
        s0 += family_s0;
        coord += family_coord;
    }
    // `small(r) = Π_{i ≥ 16} (1 − r_i)`; `id(r) = Σ_i 2^i·r_i` by doublings.
    let small = r_le[16..]
        .iter()
        .map(|r| Fr::one() - *r)
        .reduce(|a, b| ev.mul(a, b))
        .unwrap_or(Fr::one());
    let id = r_le.iter().rev().fold(Fr::zero(), |acc, r| acc + acc + *r);
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
