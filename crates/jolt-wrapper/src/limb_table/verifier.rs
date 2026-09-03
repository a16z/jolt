//! Verifier-side evaluation of the wiring kernels at the stage point:
//! `K(τ, r) = Σ_row eq(τ,row)·w(row)·eq(r, src(row))` factorizes over the row
//! bit-fields, each field's factor being the sum over its admitted values of
//! `eq(τ_u, u)·w(u)·eq(r_v, v(u))`. Every distinct field group is evaluated
//! once (memoized), bit-field `eq` tables are shared, and every field
//! multiplication is reported to the observer.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::ops::Range;

use jolt_field::{Fr, One, Ring, Zero};
use jolt_hyperkzg::VerifierObserver;

use super::layout::{field_groups, Bits, Factor, Rel, LOG_ROWS};

/// Which point a bit-field is read from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Point {
    Row,
    Src,
}

/// Fields wider than this are evaluated as products of table lookups.
/// Block boundaries of the eq tables: a field within one block is a single
/// table; a field spanning blocks is the product of its per-block pieces.
const BLOCKS: [u8; 4] = [0, 6, 12, 18];

fn block_end(bit: u8) -> u8 {
    BLOCKS
        .iter()
        .copied()
        .find(|&b| b > bit)
        .unwrap_or_else(|| unreachable!("bit {bit} beyond the row index"))
}

pub struct Evaluator<'o, O: VerifierObserver> {
    row: Vec<Fr>,
    src: Vec<Fr>,
    tables: HashMap<(Point, Bits), Vec<Fr>>,
    consts: HashMap<(Point, Bits, u32), Fr>,
    groups: HashMap<Vec<Factor>, Fr>,
    pairs: HashMap<(Bits, Bits, u32, u32), Fr>,
    observer: &'o mut O,
}

impl<'o, O: VerifierObserver> Evaluator<'o, O> {
    /// `row_le`: the kernel's row point (the copy identity's `τ`), `src_le`:
    /// the source point (the stage point `r`), both little-endian.
    pub fn new(row_le: &[Fr], src_le: &[Fr], observer: &'o mut O) -> Self {
        assert_eq!(src_le.len(), LOG_ROWS);
        Self {
            row: row_le.to_vec(),
            src: src_le.to_vec(),
            tables: HashMap::new(),
            consts: HashMap::new(),
            groups: HashMap::new(),
            pairs: HashMap::new(),
            observer,
        }
    }

    pub fn mul(&mut self, a: Fr, b: Fr) -> Fr {
        self.observer.fr_mul(a, b)
    }

    fn point(&self, point: Point) -> &[Fr] {
        match point {
            Point::Row => &self.row,
            Point::Src => &self.src,
        }
    }

    /// `eq(p_bits, u)` for every `u < 2^width` (the field lies within one block).
    fn table(&mut self, point: Point, bits: Bits) -> &[Fr] {
        debug_assert!(
            block_end(bits.lo) >= bits.hi,
            "table field {bits:?} crosses a block"
        );
        if !self.tables.contains_key(&(point, bits)) {
            let coords: Vec<Fr> =
                self.point(point)[usize::from(bits.lo)..usize::from(bits.hi)].to_vec();
            let mut table = vec![Fr::one()];
            // Most significant coordinate first so index bit `i` pairs with `coords[i]`.
            for r in coords.into_iter().rev() {
                let mut next = Vec::with_capacity(table.len() * 2);
                for &entry in &table {
                    let hi = self.mul(entry, r);
                    next.push(entry - hi);
                    next.push(hi);
                }
                table = next;
            }
            let _ = self.tables.insert((point, bits), table);
        }
        &self.tables[&(point, bits)]
    }

    fn lookup(&mut self, point: Point, bits: Bits, value: u32) -> Fr {
        self.table(point, bits)[value as usize]
    }

    /// `eq(p_bits, value)`, split at the fixed block boundaries so fields
    /// sharing a block share its tables.
    pub fn eq_const(&mut self, point: Point, bits: Bits, value: u32) -> Fr {
        if bits.width() == 0 {
            return Fr::one();
        }
        if block_end(bits.lo) >= bits.hi {
            return self.lookup(point, bits, value);
        }
        if let Some(&v) = self.consts.get(&(point, bits, value)) {
            return v;
        }
        let mut product = Fr::one();
        let mut lo = bits.lo;
        let mut first = true;
        while lo < bits.hi {
            let hi = block_end(lo).min(bits.hi);
            let piece = Bits::new(lo, hi);
            let piece_value = (value >> (lo - bits.lo)) & ((1u32 << piece.width()) - 1);
            let part = self.lookup(point, piece, piece_value);
            product = if first { part } else { self.mul(product, part) };
            first = false;
            lo = hi;
        }
        let _ = self.consts.insert((point, bits, value), product);
        product
    }

    fn pair(&mut self, u: Bits, v: Bits, c: u32, t: u32) -> Fr {
        if let Some(&p) = self.pairs.get(&(u, v, c, t)) {
            return p;
        }
        let a = self.eq_const(Point::Row, u, c);
        let b = self.eq_const(Point::Src, v, t);
        let p = self.mul(a, b);
        let _ = self.pairs.insert((u, v, c, t), p);
        p
    }

    fn scaled(&mut self, value: Fr, weight: i32) -> Fr {
        match weight {
            1 => value,
            -1 => -value,
            w => self.mul(value, Fr::from_i64(i64::from(w))),
        }
    }

    /// The factor of one row field: `Σ_u eq(τ_u, u)·Π_f w_f(u)·eq(r_v, v(u))`
    /// over the values every factor of the group admits. At most one factor
    /// has a source field.
    pub fn field(&mut self, group: &[Factor]) -> Fr {
        let key: Vec<Factor> = group.to_vec();
        if let Some(&v) = self.groups.get(&key) {
            return v;
        }
        let value = self.field_uncached(group);
        let _ = self.groups.insert(key, value);
        value
    }

    fn field_uncached(&mut self, group: &[Factor]) -> Fr {
        let relation = group.iter().find(|f| f.v.width() > 0);
        let u = group[0].u;
        debug_assert!(group.iter().all(|f| f.u == u));
        if u.width() == 0 {
            // Source-only constant.
            let Some(f) = relation else { return Fr::one() };
            let Rel::Const(value) = f.rel else {
                unreachable!("a source-only factor is a constant")
            };
            return self.eq_const(Point::Src, f.v, value);
        }
        // Unrestricted `same` field: product of per-bit equalities (two
        // multiplications per bit beat one per admitted value from width 2).
        if let [f] = group {
            if f.rel == Rel::Shift(0) && f.range.is_none() && u.width() >= 2 {
                let mut product = Fr::one();
                for i in 0..u.width() {
                    let a = self.row[usize::from(u.lo + i)];
                    let b = self.src[usize::from(f.v.lo + i)];
                    let ab = self.mul(a, b);
                    let eq = ab + ab - a - b + Fr::one();
                    product = if i == 0 { eq } else { self.mul(product, eq) };
                }
                return product;
            }
        }
        // One shift (or identity) with range restrictions over a large range:
        // the bitwise automaton, `≤ 17` multiplications per bit.
        if let Some(rel) = relation {
            if let Rel::Shift(delta) = rel.rel {
                let restricts_only = group.iter().all(|f| {
                    std::ptr::eq(f, rel)
                        || (f.v.width() == 0 && matches!(f.rel, Rel::Const(_)) && f.range.is_some())
                });
                if restricts_only {
                    let mut range = 0..1u32 << u.width();
                    for f in group {
                        if let Some(r) = &f.range {
                            range.start = range.start.max(r.start);
                            range.end = range.end.min(r.end);
                        }
                    }
                    let values = range.end.saturating_sub(range.start) as usize;
                    if 3 * values > 17 * usize::from(u.width()) {
                        return self.shift_automaton(u, rel.v, delta, &range);
                    }
                }
            }
        }
        let mut sum = Fr::zero();
        for value in 0..1u32 << u.width() {
            let mut weight = 1i32;
            let mut src = None;
            let mut admitted = true;
            for f in group {
                let Some((v, w)) = f.apply(value) else {
                    admitted = false;
                    break;
                };
                weight *= w;
                if f.v.width() > 0 {
                    src = Some((f.v, v));
                }
            }
            if !admitted || weight == 0 {
                continue;
            }
            let term = match src {
                Some((v_bits, v)) => self.pair(u, v_bits, value, v),
                None => self.eq_const(Point::Row, u, value),
            };
            sum += self.scaled(term, weight);
        }
        sum
    }

    /// `Σ_{u ∈ range} eq(r_u, u)·eq(r_v, u + delta)` (no carry out of the
    /// field) by the bitwise automaton of [`super::layout`]'s `shift_mle`:
    /// states are (carry, `u < lo` so far, `u < hi` so far), read least
    /// significant bit first; one multiplication per bit for the four bit
    /// products and one per live transition.
    fn shift_automaton(&mut self, u: Bits, v: Bits, delta: i64, range: &Range<u32>) -> Fr {
        let width = usize::from(u.width());
        let magnitude = delta.unsigned_abs();
        if magnitude >= 1u64 << width {
            return Fr::zero();
        }
        let (lo, hi) = (u64::from(range.start), u64::from(range.end));
        let mut states = [Fr::zero(); 8];
        states[0] = Fr::one();
        for i in 0..width {
            let a = self.row[usize::from(u.lo) + i];
            let b = self.src[usize::from(v.lo) + i];
            let ab = self.mul(a, b);
            // `products[u][v] = bit(a, u)·bit(b, v)`.
            let products = [[Fr::one() - a - b + ab, b - ab], [a - ab, ab]];
            let d = (magnitude >> i) & 1;
            let (lo_i, hi_i) = ((lo >> i) & 1, (hi >> i) & 1);
            let mut next = [Fr::zero(); 8];
            for (state, &weight) in states.iter().enumerate() {
                if weight.is_zero() {
                    continue;
                }
                let (carry, lt_lo, lt_hi) = (
                    state as u64 & 1,
                    (state >> 1) as u64 & 1,
                    (state >> 2) as u64 & 1,
                );
                for bit_u in 0..2u64 {
                    let (bit_v, carry_out) = if delta >= 0 {
                        let sum = bit_u + d + carry;
                        (sum & 1, sum >> 1)
                    } else {
                        let diff = bit_u as i64 - d as i64 - carry as i64;
                        (diff.rem_euclid(2) as u64, u64::from(diff < 0))
                    };
                    let step = |flag: u64, bound: u64| -> u64 {
                        match bit_u.cmp(&bound) {
                            Ordering::Less => 1,
                            Ordering::Greater => 0,
                            Ordering::Equal => flag,
                        }
                    };
                    let next_state =
                        (carry_out | (step(lt_lo, lo_i) << 1) | (step(lt_hi, hi_i) << 2)) as usize;
                    let term = self.mul(weight, products[bit_u as usize][bit_v as usize]);
                    next[next_state] += term;
                }
            }
            states = next;
        }
        let hi_open = hi >= 1u64 << width;
        (0..8usize)
            .filter(|state| {
                let (carry, lt_lo, lt_hi) = (state & 1, (state >> 1) & 1, (state >> 2) & 1);
                carry == 0 && lt_lo == 0 && (hi_open || lt_hi == 1)
            })
            .fold(Fr::zero(), |acc, state| acc + states[state])
    }

    /// `Σ_u eq(p_bits, u)·q^u` over the whole field: `Π_i (1 − r_i + r_i·q^{2^i})`.
    pub fn geometric(&mut self, point: Point, bits: Bits, q: Fr) -> Fr {
        let coords = self.point(point)[usize::from(bits.lo)..usize::from(bits.hi)].to_vec();
        let mut power = q;
        let mut product = Fr::one();
        for (i, r) in coords.into_iter().enumerate() {
            if i > 0 {
                power = self.mul(power, power);
            }
            let term = self.mul(r, power - Fr::one()) + Fr::one();
            product = if i == 0 {
                term
            } else {
                self.mul(product, term)
            };
        }
        product
    }

    /// `Σ_{u ∈ range} eq(p_bits, u)·f(u)` (one multiplication per admitted value).
    pub fn field_sum(
        &mut self,
        point: Point,
        bits: Bits,
        range: Range<u32>,
        f: &dyn Fn(u32) -> Fr,
    ) -> Fr {
        let mut sum = Fr::zero();
        for u in range {
            let e = self.eq_const(point, bits, u);
            sum += self.mul(e, f(u));
        }
        sum
    }

    /// `(Σ_{u ∈ range} eq(p_bits, u), Σ_{u ∈ range} eq(p_bits, u)·u)` by bit
    /// decomposition (one multiplication per bit).
    pub fn field_moment(&mut self, point: Point, bits: Bits, range: Range<u32>) -> (Fr, Fr) {
        if bits.width() == 0 {
            return (Fr::one(), Fr::zero());
        }
        if range == (0..1 << bits.width()) {
            // Free field: `Σ_u eq = 1`, `Σ_u eq·u = Σ_i 2^i·r_i`.
            let coords = &self.point(point)[usize::from(bits.lo)..usize::from(bits.hi)];
            let moment = coords
                .iter()
                .enumerate()
                .fold(Fr::zero(), |acc, (i, r)| acc + *r * Fr::from_u64(1u64 << i));
            return (Fr::one(), moment);
        }
        let mut total = Fr::zero();
        let mut per_bit = vec![Fr::zero(); usize::from(bits.width())];
        for u in range {
            let e = self.eq_const(point, bits, u);
            total += e;
            for (i, acc) in per_bit.iter_mut().enumerate() {
                if (u >> i) & 1 == 1 {
                    *acc += e;
                }
            }
        }
        let mut moment = Fr::zero();
        for (i, acc) in per_bit.into_iter().enumerate() {
            moment += self.mul(acc, Fr::from_u64(1u64 << i));
        }
        (total, moment)
    }

    /// `Π_fields field(group)`: a kernel's MLE.
    pub fn kernel(&mut self, factors: &[Factor]) -> Fr {
        let mut product = Fr::one();
        for (i, group) in field_groups(factors).into_iter().enumerate() {
            let value = self.field(&group);
            product = if i == 0 {
                value
            } else {
                self.mul(product, value)
            };
        }
        product
    }

    /// Adds `Π_{other cell fields}·m_map` for every map into `buckets[index]`,
    /// the caller applying one weight per bucket; the cell factors sharing
    /// the maps' row field join each map's sum.
    pub fn group_into(&mut self, cell: &[Factor], maps: &[(usize, &Factor)], buckets: &mut [Fr]) {
        let Some((_, first)) = maps.first() else {
            return;
        };
        let map_field = first.u;
        let mut product: Option<Fr> = None;
        let mut shared: Vec<Factor> = Vec::new();
        for group in field_groups(cell) {
            if group[0].u == map_field {
                shared = group;
                continue;
            }
            let value = self.field(&group);
            product = Some(match product {
                Some(p) => self.mul(p, value),
                None => value,
            });
        }
        for (index, map) in maps {
            let mut group = shared.clone();
            group.push((*map).clone());
            let m = self.field(&group);
            buckets[*index] += match product {
                Some(p) => self.mul(p, m),
                None => m,
            };
        }
    }

    /// The same evaluator with the source point also as the row point,
    /// keeping the source tables (kernels over one point only).
    pub fn rebase_row_to_src(mut self) -> Self {
        self.row = self.src.clone();
        let src_tables: Vec<(Bits, Vec<Fr>)> = self
            .tables
            .iter()
            .filter(|((point, _), _)| *point == Point::Src)
            .map(|((_, bits), table)| (*bits, table.clone()))
            .collect();
        self.tables.retain(|(point, _), _| *point == Point::Src);
        for (bits, table) in src_tables {
            let _ = self.tables.insert((Point::Row, bits), table);
        }
        let src_consts: Vec<((Bits, u32), Fr)> = self
            .consts
            .iter()
            .filter(|((point, _, _), _)| *point == Point::Src)
            .map(|((_, bits, value), v)| ((*bits, *value), *v))
            .collect();
        self.consts.retain(|(point, _, _), _| *point == Point::Src);
        for ((bits, value), v) in src_consts {
            let _ = self.consts.insert((Point::Row, bits, value), v);
        }
        self.groups.clear();
        self.pairs.clear();
        self
    }
}
