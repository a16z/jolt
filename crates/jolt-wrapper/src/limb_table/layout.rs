//! The fixed row layout and its wiring kernels. Rows are indexed by 18 bits:
//! `row = 16·cell + c` (`c` the coordinate inside a 16-row cell). Every
//! operand source is a function of the row index built from bit-field
//! relations, so the verifier evaluates a wiring kernel
//! `W(r, r') = Σ_row eq(r, row)·w(row)·eq(r', src(row))` as a product of
//! per-field factors in `O(bits)` each instead of walking an edge list.

use std::ops::Range;

use jolt_field::{Fr, One, Ring, Zero};

pub const LOG_ROWS: usize = 18;
pub const ROWS: usize = 1 << LOG_ROWS;
pub const LOG_CELL: u8 = 4;
pub const CELL_ROWS: u32 = 1 << LOG_CELL;
pub const LOG_CELLS: usize = LOG_ROWS - LOG_CELL as usize;
pub const CELLS: u32 = 1 << LOG_CELLS;

/// A bit-field `[lo, hi)` of the row index (bit 0 is the coordinate's low bit).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Bits {
    pub lo: u8,
    pub hi: u8,
}

impl Bits {
    pub const fn new(lo: u8, hi: u8) -> Self {
        assert!(lo <= hi && hi <= 32);
        Self { lo, hi }
    }

    pub const EMPTY: Self = Self { lo: 0, hi: 0 };

    pub const fn width(self) -> u8 {
        self.hi - self.lo
    }

    pub fn extract(self, index: u32) -> u32 {
        if self.width() == 0 {
            0
        } else {
            (index >> self.lo) & ((1u32 << self.width()) - 1)
        }
    }

    pub fn insert(self, index: &mut u32, value: u32) {
        debug_assert!(self.width() == 32 || value < 1u32 << self.width());
        *index |= value << self.lo;
    }

    fn slice(self, point: &[Fr]) -> &[Fr] {
        &point[usize::from(self.lo)..usize::from(self.hi)]
    }
}

/// How a source field is determined by a row field.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Rel {
    /// `v = u + delta`, no carry or borrow out of the field (`delta = 0`: `v = u`).
    Shift(i64),
    /// `v = value`; the row field is only restricted by the factor's range.
    Const(u32),
    /// `v = map[u].0` with weight `map[u].1`; `None` excludes the row.
    Map(Vec<Option<(u32, i32)>>),
    /// No source field: the row field carries a public weight per value
    /// (zero excludes the row).
    Weight(Vec<i32>),
    /// Sparse `(u, v)` pairs (an explicit edge list over the field, for the
    /// few irregular regions); rows whose `u` is absent are excluded.
    Table(Vec<(u32, u32)>),
}

/// One factor of a kernel: the relation between row field `u` and source
/// field `v`, restricted to rows whose `u` value lies in `range`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Factor {
    pub u: Bits,
    pub v: Bits,
    pub rel: Rel,
    pub range: Option<Range<u32>>,
}

impl Factor {
    pub fn same(u: Bits, v: Bits) -> Self {
        assert_eq!(u.width(), v.width());
        Self {
            u,
            v,
            rel: Rel::Shift(0),
            range: None,
        }
    }

    pub fn shift(u: Bits, v: Bits, delta: i64) -> Self {
        assert_eq!(u.width(), v.width());
        Self {
            u,
            v,
            rel: Rel::Shift(delta),
            range: None,
        }
    }

    /// `v = value` for every row (the row side is free).
    pub fn constant(v: Bits, value: u32) -> Self {
        Self {
            u: Bits::EMPTY,
            v,
            rel: Rel::Const(value),
            range: None,
        }
    }

    /// Rows with `u ∈ range` only (no source field).
    pub fn restrict(u: Bits, range: Range<u32>) -> Self {
        Self {
            u,
            v: Bits::EMPTY,
            rel: Rel::Const(0),
            range: Some(range),
        }
    }

    pub fn map(u: Bits, v: Bits, map: Vec<Option<(u32, i32)>>) -> Self {
        assert_eq!(map.len(), 1 << u.width());
        debug_assert!(map
            .iter()
            .flatten()
            .all(|(v_val, _)| *v_val < 1 << v.width()));
        Self {
            u,
            v,
            rel: Rel::Map(map),
            range: None,
        }
    }

    pub fn weight(u: Bits, weights: Vec<i32>) -> Self {
        assert_eq!(weights.len(), 1 << u.width());
        Self {
            u,
            v: Bits::EMPTY,
            rel: Rel::Weight(weights),
            range: None,
        }
    }

    pub fn table(u: Bits, v: Bits, pairs: Vec<(u32, u32)>) -> Self {
        Self {
            u,
            v,
            rel: Rel::Table(pairs),
            range: None,
        }
    }

    pub fn with_range(mut self, range: Range<u32>) -> Self {
        self.range = Some(range);
        self
    }

    /// The source field value and weight for row field value `u`, if the row
    /// is in this factor's domain.
    pub fn apply(&self, u: u32) -> Option<(u32, i32)> {
        if let Some(range) = &self.range {
            if !range.contains(&u) {
                return None;
            }
        }
        match &self.rel {
            Rel::Shift(delta) => {
                let v = i64::from(u) + delta;
                (v >= 0 && v < 1i64 << self.v.width()).then_some((v as u32, 1))
            }
            Rel::Const(value) => Some((*value, 1)),
            Rel::Map(map) => map[u as usize],
            Rel::Weight(weights) => {
                let w = weights[u as usize];
                (w != 0).then_some((0, w))
            }
            Rel::Table(pairs) => pairs
                .iter()
                .find(|(from, _)| *from == u)
                .map(|(_, to)| (*to, 1)),
        }
    }

    /// `Σ_{u,v} [rel(u) = v, u ∈ range]·w(u)·eq(r_u, u)·eq(r'_v, v)`.
    pub fn mle(&self, r: &[Fr], r_src: &[Fr]) -> Fr {
        let (ru, rv) = (self.u.slice(r), self.v.slice(r_src));
        match &self.rel {
            Rel::Shift(delta) => shift_mle(ru, rv, *delta, self.range.as_ref()),
            Rel::Const(value) => eq_const(rv, *value) * range_mle(ru, self.range.as_ref()),
            Rel::Map(map) => map
                .iter()
                .enumerate()
                .filter(|(u, _)| {
                    self.range
                        .as_ref()
                        .is_none_or(|range| range.contains(&(*u as u32)))
                })
                .filter_map(|(u, entry)| entry.map(|(v, w)| (u, v, w)))
                .fold(Fr::zero(), |acc, (u, v, w)| {
                    acc + eq_const(ru, u as u32) * eq_const(rv, v) * Fr::from_i64(i64::from(w))
                }),
            Rel::Weight(weights) => weights
                .iter()
                .enumerate()
                .filter(|(u, w)| {
                    **w != 0
                        && self
                            .range
                            .as_ref()
                            .is_none_or(|range| range.contains(&(*u as u32)))
                })
                .fold(Fr::zero(), |acc, (u, w)| {
                    acc + eq_const(ru, u as u32) * Fr::from_i64(i64::from(*w))
                }),
            Rel::Table(pairs) => pairs
                .iter()
                .filter(|(u, _)| self.range.as_ref().is_none_or(|range| range.contains(u)))
                .fold(Fr::zero(), |acc, (u, v)| {
                    acc + eq_const(ru, *u) * eq_const(rv, *v)
                }),
        }
    }
}

/// `eq(r, bits(value))` over a little-endian point.
pub fn eq_const(r: &[Fr], value: u32) -> Fr {
    r.iter().enumerate().fold(Fr::one(), |acc, (i, ri)| {
        acc * if (value >> i) & 1 == 1 {
            *ri
        } else {
            Fr::one() - *ri
        }
    })
}

/// Bitwise automaton for `Σ_{u ∈ range} eq(r_u, u)·eq(r_v, u + delta)`
/// (no carry out of the field): states are (carry, `u < lo` so far,
/// `u < hi` so far), read least significant bit first.
fn shift_mle(ru: &[Fr], rv: &[Fr], delta: i64, range: Option<&Range<u32>>) -> Fr {
    let width = ru.len();
    let magnitude = delta.unsigned_abs();
    if magnitude >= 1u64 << width {
        return Fr::zero();
    }
    let (lo, hi) = range.map_or((0, u64::MAX), |r| (u64::from(r.start), u64::from(r.end)));
    let mut states = [Fr::zero(); 8];
    states[0] = Fr::one();
    for i in 0..width {
        let d = (magnitude >> i) & 1;
        let (lo_i, hi_i) = ((lo >> i) & 1, (hi >> i) & 1);
        let mut next = [Fr::zero(); 8];
        for (state, weight) in states.iter().enumerate() {
            if weight.is_zero() {
                continue;
            }
            let (carry, lt_lo, lt_hi) = (state as u64 & 1, (state >> 1) & 1, (state >> 2) & 1);
            for u in 0..2u64 {
                let (v, carry_out) = if delta >= 0 {
                    let sum = u + d + carry;
                    (sum & 1, sum >> 1)
                } else {
                    let diff = i64::try_from(u).unwrap_or(0) - d as i64 - carry as i64;
                    (diff.rem_euclid(2) as u64, u64::from(diff < 0))
                };
                let step = |flag: u64, bound: u64| -> u64 {
                    match u.cmp(&bound) {
                        std::cmp::Ordering::Less => 1,
                        std::cmp::Ordering::Greater => 0,
                        std::cmp::Ordering::Equal => flag,
                    }
                };
                let next_state = (carry_out
                    | (step(lt_lo as u64, lo_i) << 1)
                    | (step(lt_hi as u64, hi_i) << 2)) as usize;
                let bit = |r: Fr, b: u64| if b == 1 { r } else { Fr::one() - r };
                next[next_state] += *weight * bit(ru[i], u) * bit(rv[i], v);
            }
        }
        states = next;
    }
    // Accept: no carry/borrow out, `u ≥ lo`, `u < hi` (the flags are exact
    // because `lo` and `hi` were compared bit by bit over the same width;
    // bounds at or above `2^width` accept everything on that side).
    let hi_open = hi >= 1u64 << width;
    (0..8)
        .filter(|state| {
            let (carry, lt_lo, lt_hi) = (state & 1, (state >> 1) & 1, (state >> 2) & 1);
            carry == 0 && lt_lo == 0 && (hi_open || lt_hi == 1)
        })
        .fold(Fr::zero(), |acc, state| acc + states[state])
}

/// `Σ_{u ∈ range} eq(r_u, u)` (one when unrestricted).
fn range_mle(ru: &[Fr], range: Option<&Range<u32>>) -> Fr {
    if ru.is_empty() {
        return Fr::one();
    }
    // Same automaton as `shift_mle` without the source side: states are the
    // two comparison flags.
    let width = ru.len();
    let (lo, hi) = range.map_or((0, u64::MAX), |r| (u64::from(r.start), u64::from(r.end)));
    let mut states = [Fr::zero(); 4];
    states[0] = Fr::one();
    for (i, &r_i) in ru.iter().enumerate() {
        let (lo_i, hi_i) = ((lo >> i) & 1, (hi >> i) & 1);
        let mut next = [Fr::zero(); 4];
        for (state, weight) in states.iter().enumerate() {
            if weight.is_zero() {
                continue;
            }
            let (lt_lo, lt_hi) = (state as u64 & 1, (state >> 1) as u64 & 1);
            for u in 0..2u64 {
                let step = |flag: u64, bound: u64| -> u64 {
                    match u.cmp(&bound) {
                        std::cmp::Ordering::Less => 1,
                        std::cmp::Ordering::Greater => 0,
                        std::cmp::Ordering::Equal => flag,
                    }
                };
                let next_state = (step(lt_lo, lo_i) | (step(lt_hi, hi_i) << 1)) as usize;
                let bit = if u == 1 { r_i } else { Fr::one() - r_i };
                next[next_state] += *weight * bit;
            }
        }
        states = next;
    }
    let hi_open = hi >= 1u64 << width;
    (0..4)
        .filter(|state| state & 1 == 0 && (hi_open || (state >> 1) & 1 == 1))
        .fold(Fr::zero(), |acc, state| acc + states[state])
}

/// Splits every unrestricted identity factor (`same(F, F)`) and every
/// single-value restriction at the field boundaries of the other factors, so
/// that overlapping row fields become identical ones ([`field_groups`] then
/// sums each field once). Multi-value restrictions must already be aligned.
pub fn normalize(factors: Vec<Factor>) -> Vec<Factor> {
    let cuts_of = |i: usize, field: Bits| -> Vec<u8> {
        let mut cuts: Vec<u8> = vec![field.lo, field.hi];
        for (j, other) in factors.iter().enumerate() {
            if i == j || other.u.width() == 0 {
                continue;
            }
            for edge in [other.u.lo, other.u.hi] {
                if edge > field.lo && edge < field.hi {
                    cuts.push(edge);
                }
            }
        }
        cuts.sort_unstable();
        cuts.dedup();
        cuts
    };
    let mut out = Vec::with_capacity(factors.len());
    for (i, factor) in factors.iter().enumerate() {
        let cuts = cuts_of(i, factor.u);
        if cuts.len() == 2 || factor.u.width() == 0 {
            out.push(factor.clone());
            continue;
        }
        let identity =
            factor.rel == Rel::Shift(0) && factor.range.is_none() && factor.u == factor.v;
        let single = factor.v.width() == 0
            && matches!(factor.rel, Rel::Const(_))
            && factor.range.as_ref().is_some_and(|r| r.end == r.start + 1);
        assert!(
            identity || single,
            "row field {:?} straddles other fields and cannot be split: {factor:?}",
            factor.u
        );
        for pair in cuts.windows(2) {
            let piece = Bits::new(pair[0], pair[1]);
            if identity {
                out.push(Factor::same(piece, piece));
            } else {
                let Some(range) = &factor.range else {
                    unreachable!("single-value restriction")
                };
                let value =
                    (range.start >> (piece.lo - factor.u.lo)) & ((1u32 << piece.width()) - 1);
                out.push(Factor::restrict(piece, value..value + 1));
            }
        }
    }
    out
}

/// Groups a kernel's factors by row field: factors on the same `u` field
/// restrict one sum (at most one of them has a source field); source-only
/// constants form singleton groups.
pub fn field_groups(factors: &[Factor]) -> Vec<Vec<Factor>> {
    let mut groups: Vec<Vec<Factor>> = Vec::new();
    for factor in factors {
        if factor.u.width() == 0 {
            groups.push(vec![factor.clone()]);
            continue;
        }
        debug_assert!(
            groups.iter().all(|g| {
                let u = g[0].u;
                u.width() == 0 || u == factor.u || u.hi <= factor.u.lo || factor.u.hi <= u.lo
            }),
            "overlapping row fields need identical bit ranges: {factor:?} in {factors:?}"
        );
        match groups
            .iter_mut()
            .find(|g| g[0].u.width() > 0 && g[0].u == factor.u)
        {
            Some(group) => {
                debug_assert!(
                    group.iter().filter(|f| f.v.width() > 0).count()
                        + usize::from(factor.v.width() > 0)
                        <= 1,
                    "two source relations on one field"
                );
                group.push(factor.clone());
            }
            None => groups.push(vec![factor.clone()]),
        }
    }
    groups
}

/// `Σ_u eq(r_u, u)·Π_f w_f(u)·eq(r'_v, v(u))` over the values every factor of
/// the group admits (brute force over the field; the test reference).
pub fn field_mle(group: &[Factor], r: &[Fr], r_src: &[Fr]) -> Fr {
    let u = group[0].u;
    if u.width() == 0 {
        return group.iter().fold(Fr::one(), |acc, f| acc * f.mle(r, r_src));
    }
    let ru = u.slice(r);
    let mut sum = Fr::zero();
    for value in 0..1u32 << u.width() {
        let mut weight = 1i32;
        let mut src = Fr::one();
        let mut admitted = true;
        for f in group {
            let Some((v, w)) = f.apply(value) else {
                admitted = false;
                break;
            };
            weight *= w;
            if f.v.width() > 0 {
                src *= eq_const(f.v.slice(r_src), v);
            }
        }
        if admitted && weight != 0 {
            sum += eq_const(ru, value) * src * Fr::from_i64(i64::from(weight));
        }
    }
    sum
}

/// A wiring kernel: the product of its factors. Source fields partition the
/// 18 row bits; row fields may repeat (a range on one field and a shift on
/// the same field are two factors) but at most one factor carries a `Map`
/// or `Weight` so the weight is a product of public integers.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Kernel {
    pub factors: Vec<Factor>,
}

impl Kernel {
    pub fn new(factors: Vec<Factor>) -> Self {
        let kernel = Self { factors };
        kernel.validate();
        kernel
    }

    fn validate(&self) {
        let mut covered = 0u32;
        for factor in &self.factors {
            let mask = if factor.v.width() == 0 {
                0
            } else {
                ((1u32 << factor.v.width()) - 1) << factor.v.lo
            };
            assert_eq!(covered & mask, 0, "source bits covered twice");
            covered |= mask;
        }
        assert_eq!(covered, (1u32 << LOG_ROWS) - 1, "source bits uncovered");
    }

    /// The source row and integer weight of `row`, or `None` when the row is
    /// outside this kernel's domain.
    pub fn apply(&self, row: u32) -> Option<(u32, i32)> {
        let mut src = 0u32;
        let mut weight = 1i32;
        for factor in &self.factors {
            let (v, w) = factor.apply(factor.u.extract(row))?;
            factor.v.insert(&mut src, v);
            weight *= w;
        }
        Some((src, weight))
    }

    /// `Σ_row eq(r, row)·w(row)·eq(r', src(row))` for little-endian points
    /// (`r` over the kernel's row index, `r_src` over the 18 row bits): the
    /// product over row fields of each field group's sum
    /// ([`field_groups`]); factors sharing a field restrict the same sum.
    pub fn mle(&self, r: &[Fr], r_src: &[Fr]) -> Fr {
        debug_assert_eq!(r_src.len(), LOG_ROWS);
        field_groups(&self.factors)
            .into_iter()
            .fold(Fr::one(), |acc, group| acc * field_mle(&group, r, r_src))
    }

    /// Native verifier cost in field multiplications, the reporting unit.
    pub fn cost(&self) -> usize {
        self.factors
            .iter()
            .map(|factor| match &factor.rel {
                Rel::Shift(_) => 32 * usize::from(factor.u.width()),
                Rel::Const(_) => usize::from(factor.v.width()) + 8 * usize::from(factor.u.width()),
                Rel::Map(map) => {
                    map.iter().flatten().count()
                        * usize::from(factor.u.width() + factor.v.width() + 1)
                }
                Rel::Weight(weights) => {
                    weights.iter().filter(|w| **w != 0).count() * usize::from(factor.u.width() + 1)
                }
                Rel::Table(pairs) => {
                    pairs.len() * usize::from(factor.u.width() + factor.v.width() + 1)
                }
            })
            .sum()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Side {
    X,
    Y,
}

/// One kernel of one operand side of one slot; a slot side is the sum of its pieces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Piece {
    pub slot: u8,
    pub side: Side,
    pub kernel: Kernel,
}

impl Factor {
    /// The row-field values this factor admits (all `2^width` values when
    /// unrestricted).
    pub fn admitted(&self) -> Vec<u32> {
        let all = 0..1u32 << self.u.width();
        let in_range = |u: &u32| self.range.as_ref().is_none_or(|range| range.contains(u));
        match &self.rel {
            Rel::Shift(delta) => all
                .filter(in_range)
                .filter(|u| {
                    let v = i64::from(*u) + delta;
                    v >= 0 && v < 1i64 << self.v.width()
                })
                .collect(),
            Rel::Const(_) => all.filter(in_range).collect(),
            Rel::Map(map) => all
                .filter(in_range)
                .filter(|u| map[*u as usize].is_some())
                .collect(),
            Rel::Weight(weights) => all
                .filter(in_range)
                .filter(|u| weights[*u as usize] != 0)
                .collect(),
            Rel::Table(pairs) => pairs.iter().map(|(u, _)| *u).filter(in_range).collect(),
        }
    }
}

impl Kernel {
    /// Every `(row, src, weight)` edge of the kernel: rows are enumerated
    /// field by field from the factors' admitted values (free bits take
    /// every value), so the cost is the edge count, not the row count.
    pub fn edges(&self) -> Vec<(u32, u32, i32)> {
        // Intersect admitted values per distinct row field; fields that
        // repeat across factors are intersected by re-checking `apply`.
        let mut fields: Vec<(Bits, Vec<u32>)> = Vec::new();
        let mut covered = 0u32;
        for factor in &self.factors {
            if factor.u.width() == 0 {
                continue;
            }
            let mask = ((1u32 << factor.u.width()) - 1) << factor.u.lo;
            if let Some((_, values)) = fields.iter_mut().find(|(bits, _)| *bits == factor.u) {
                let admitted = factor.admitted();
                values.retain(|u| admitted.contains(u));
            } else {
                assert_eq!(
                    covered & mask,
                    0,
                    "overlapping row fields need identical bit ranges"
                );
                covered |= mask;
                fields.push((factor.u, factor.admitted()));
            }
        }
        let free_bits: Vec<u8> = (0..LOG_ROWS as u8)
            .filter(|b| covered & (1 << b) == 0)
            .collect();
        let mut rows = vec![0u32];
        for (bits, values) in &fields {
            rows = rows
                .iter()
                .flat_map(|base| {
                    values.iter().map(move |v| {
                        let mut row = *base;
                        bits.insert(&mut row, *v);
                        row
                    })
                })
                .collect();
        }
        for bit in free_bits {
            rows = rows
                .iter()
                .flat_map(|base| [*base, *base | (1 << bit)])
                .collect();
        }
        rows.into_iter()
            .filter_map(|row| self.apply(row).map(|(src, w)| (row, src, w)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use jolt_poly::EqPolynomial;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    fn random_point(rng: &mut ChaCha20Rng, n: usize) -> Vec<Fr> {
        (0..n).map(|_| Fr::from(ark_bn254::Fr::rand(rng))).collect()
    }

    fn brute(kernel: &Kernel, r: &[Fr], r_src: &[Fr]) -> Fr {
        let eq_r = EqPolynomial::<Fr>::evals(&r.iter().rev().copied().collect::<Vec<_>>(), None);
        let eq_src =
            EqPolynomial::<Fr>::evals(&r_src.iter().rev().copied().collect::<Vec<_>>(), None);
        (0..ROWS as u32)
            .filter_map(|row| kernel.apply(row).map(|(src, w)| (row, src, w)))
            .fold(Fr::zero(), |acc, (row, src, w)| {
                acc + eq_r[row as usize] * eq_src[src as usize] * Fr::from_i64(i64::from(w))
            })
    }

    #[test]
    fn kernels_match_their_edge_lists() {
        let mut rng = ChaCha20Rng::seed_from_u64(3);
        let coord = Bits::new(0, 4);
        let cell = Bits::new(4, 18);
        let map: Vec<Option<(u32, i32)>> = (0..16u32)
            .map(|c| (c < 12).then_some(((c * 7) % 12, if c % 3 == 0 { -2 } else { 1 })))
            .collect();
        let kernels = [
            Kernel::new(vec![
                Factor::map(coord, coord, map.clone()),
                Factor::shift(Bits::new(4, 10), Bits::new(4, 10), -1),
                Factor::same(Bits::new(10, 18), Bits::new(10, 18)).with_range(3..149),
            ]),
            Kernel::new(vec![
                Factor::map(coord, coord, map.clone()),
                Factor::constant(Bits::new(4, 7), 5),
                Factor::same(Bits::new(10, 17), Bits::new(7, 14)).with_range(0..100),
                Factor::constant(Bits::new(14, 18), 9),
                Factor::weight(Bits::new(4, 10), (0..64).map(|t| (t % 5) - 2).collect()),
            ]),
            Kernel::new(vec![
                Factor::map(coord, coord, map),
                Factor::shift(cell, cell, 64 * 148).with_range(0..64),
            ]),
            Kernel::new(vec![
                Factor::map(coord, Bits::new(0, 18), vec![Some((152_580, 1)); 16]),
                Factor::table(
                    Bits::new(10, 18),
                    Bits::EMPTY,
                    vec![(3, 0), (9, 0), (77, 0)],
                ),
            ]),
        ];
        for kernel in &kernels {
            let r = random_point(&mut rng, LOG_ROWS);
            let r_src = random_point(&mut rng, LOG_ROWS);
            assert_eq!(kernel.mle(&r, &r_src), brute(kernel, &r, &r_src));
        }
    }

    #[test]
    fn shift_ranges_are_exact() {
        let mut rng = ChaCha20Rng::seed_from_u64(4);
        for (delta, range) in [
            (3i64, Some(2..40u32)),
            (-5, None),
            (0, Some(17..64)),
            (-64, Some(0..64)),
        ] {
            let ru = random_point(&mut rng, 6);
            let rv = random_point(&mut rng, 6);
            let expected = (0..64u32)
                .filter(|u| range.as_ref().is_none_or(|r| r.contains(u)))
                .filter_map(|u| {
                    let v = i64::from(u) + delta;
                    (0..64).contains(&v).then_some((u, v as u32))
                })
                .fold(Fr::zero(), |acc, (u, v)| {
                    acc + eq_const(&ru, u) * eq_const(&rv, v)
                });
            assert_eq!(shift_mle(&ru, &rv, delta, range.as_ref()), expected);
        }
    }
}
