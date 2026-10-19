//! Division-free reduction of centered field coefficients modulo CRT primes.
//!
//! A canonical coefficient `c < q < 2^128` has the centered value
//! `v = c - q·[c > ⌊q/2⌋]`. Cutting `c` and `q` into 26-bit slices `c_j` and
//! `q_j` gives signed limbs `l_j = c_j - q_j·[c > ⌊q/2⌋]` with `|l_j| < 2^26`
//! and `v = Σ_j 2^(26 j) l_j`. Every CRT prime shares these limbs. For a prime
//! `p` with Montgomery radix `R`, the centered scales
//! `s_j ≡ 2^(26 j) · R · 2^32 (mod p)` bound `|Σ_j l_j s_j| < L · 2^25 · p`, so
//! one signed 32-bit Montgomery reduction returns `v · R mod p` in `(-p, p)`,
//! the range [`NttPrime::from_canonical`] produces.

use std::array::from_fn;

use super::montgomery::{inverse_i32, reduce_i32};
use super::prime::{MontCoeff, NttPrime, PrimeWidth};

/// Bits per signed field limb.
const LIMB_BITS: u32 = 26;
const LIMB_MASK: u32 = (1 << LIMB_BITS) - 1;
/// Limbs that cover a 128-bit canonical coefficient.
pub(crate) const MAX_FIELD_LIMBS: usize = 5;

/// Montgomery constants that reduce signed field limbs modulo one CRT prime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FieldLimbScales {
    /// Prime modulus.
    pub(crate) p: i32,
    /// `p^{-1} mod 2^32`.
    pub(crate) pinv: i32,
    /// Centered `2^(26 j) · R · 2^32 mod p`.
    pub(crate) scales: [i32; MAX_FIELD_LIMBS],
}

impl FieldLimbScales {
    pub(crate) fn new<W: PrimeWidth>(prime: NttPrime<W>) -> Self {
        let p = prime.p.to_i64();
        let scales = from_fn(|j| {
            let exponent = LIMB_BITS * j as u32 + W::R_LOG + 32;
            let residue = (0..exponent).fold(1i64, |x, _| 2 * x % p);
            (if residue > p / 2 {
                residue - p
            } else {
                residue
            }) as i32
        });
        Self {
            p: p as i32,
            pinv: inverse_i32(p as i32),
            scales,
        }
    }

    /// Montgomery residue of `Σ_j 2^(26 j) limbs[j]` modulo `p`, in `(-p, p)`.
    #[inline(always)]
    fn reduce<const L: usize>(&self, limbs: [i32; L]) -> i32 {
        let t = limbs
            .iter()
            .zip(&self.scales)
            .map(|(&limb, &scale)| i64::from(limb) * i64::from(scale))
            .sum::<i64>();
        reduce_i32(t, self.p, self.pinv)
    }
}

/// A field modulus cut into the 26-bit slices that centering subtracts.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FieldModulusLimbs {
    /// `⌊q/2⌋`; larger canonical coefficients are centered by subtracting `q`.
    pub(crate) half: u128,
    /// The 26-bit slices of `q`.
    pub(crate) limbs: [i32; MAX_FIELD_LIMBS],
    /// Number of limbs that cover `q`.
    pub(crate) len: usize,
}

impl FieldModulusLimbs {
    pub(crate) fn new(modulus: u128) -> Self {
        let bits = u128::BITS - modulus.leading_zeros();
        Self {
            half: modulus / 2,
            limbs: slices(modulus).map(|slice| slice as i32),
            len: (bits.div_ceil(LIMB_BITS) as usize).clamp(1, MAX_FIELD_LIMBS),
        }
    }
}

/// The 26-bit slices of `x`; the last slice holds bits `104..128`.
#[inline(always)]
fn slices(x: u128) -> [u32; MAX_FIELD_LIMBS] {
    let w = [
        x as u32,
        (x >> 32) as u32,
        (x >> 64) as u32,
        (x >> 96) as u32,
    ];
    [
        w[0] & LIMB_MASK,
        (w[0] >> 26 | w[1] << 6) & LIMB_MASK,
        (w[1] >> 20 | w[2] << 12) & LIMB_MASK,
        (w[2] >> 14 | w[3] << 18) & LIMB_MASK,
        w[3] >> 8,
    ]
}

/// Coefficients staged as canonical `u128` values per residue-kernel call.
pub(crate) const FIELD_CHUNK: usize = 64;

/// Write the Montgomery residues of centered canonical coefficients modulo
/// every prime into `out[k][start..start + canonical.len()]`, using the first
/// `L` limbs of `modulus`.
///
/// Each coefficient's limbs stay in registers while every prime reduces them.
/// The coefficients lie below the modulus, `L` covers the modulus, and
/// `start + canonical.len() <= D`.
pub(crate) fn field_residues<W: PrimeWidth, const L: usize, const K: usize, const D: usize>(
    out: &mut [[MontCoeff<W>; D]; K],
    start: usize,
    canonical: &[u128],
    modulus: &FieldModulusLimbs,
    scales: &[FieldLimbScales; K],
) {
    for (index, &coefficient) in canonical.iter().enumerate() {
        let center = -i32::from(coefficient > modulus.half);
        let slices = slices(coefficient);
        let limbs = from_fn::<_, L, _>(|j| slices[j] as i32 - (modulus.limbs[j] & center));
        for (residues, scales) in out.iter_mut().zip(scales) {
            residues[start + index] =
                MontCoeff::from_raw(W::from_i64(i64::from(scales.reduce(limbs))));
        }
    }
}
