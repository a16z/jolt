//! Mixed-radix FFT over prime fields with smooth-order multiplicative subgroups.

#![expect(
    clippy::expect_used,
    reason = "constructed FFT plans establish the indexed root and factor invariants"
)]
//!
//! # Setting
//!
//! The protocol primes [`crate::Prime128Offset2355`]
//! and [`crate::Prime128OffsetA7F7`] are pseudo-Mersenne, so
//! `p − 1` is not a power of two; each is instead chosen so it carries
//! a large **smooth factor** — a product of small primes:
//!
//! - `p = 2^128 − 2355`: smooth order `14_700 = 2² · 3 · 5² · 7²`
//! - `p = 2^128 − 2^32 + 22_537`: smooth order `17_496 = 2³ · 3⁷`
//!
//! FFT domain sizes are divisors of that smooth order; there is no
//! power-of-two NTT to fall back on. The primary use case is FFT-based
//! Reed-Solomon encoding inside the protocol.
//!
//! # Algorithm
//!
//! Iterative Cooley-Tukey decimation-in-time (DIT). For a domain size
//! `n = f_0 · f_1 · … · f_{s−1}` (each `f_i` a small prime, ≤ 7 in
//! practice), the size-`n` DFT factors recursively into size-`f_i`
//! DFTs combined with twiddle multiplications. The iterative form
//! permutes the input by mixed-radix digit reversal up front, then
//! sweeps `s` stages bottom-up, running radix-`f_i` butterflies in
//! place at each stage.
//!
//! # Optimizations
//!
//! All precomputed once when a `SmoothDomain` is built and reused
//! across transforms:
//!
//! - **Stage plan** (`factorize`, `digit_reversal_permutation`): the
//!   per-stage radices and digit-reversal permutation are fixed at
//!   construction.
//! - **Twiddle tables** (`StageData::twiddle_table`): the `ω^{jk}`
//!   factor the DIT formula uses at every butterfly becomes a table
//!   lookup plus a small power-up loop, replacing a `field_pow` call
//!   per butterfly.
//! - **Ping-pong buffers** (`FftWorkspace`): the two length-`n` working
//!   buffers are pre-allocated, so the transform itself is allocation-free.
//! - **Low-multiplication radix kernels**
//!   (`FftWorkspace::butterfly_stages`): the size-`r` DFT inside each
//!   butterfly is hand-tuned per radix, taking the multiplication
//!   count from the naive `r²` down to `1, 2, 6, 18` for
//!   `r ∈ {2, 3, 5, 7}` (radix 3 uses `1 + ω + ω² = 0`; radix 5 / 7
//!   use Karatsuba on the conjugate-pair-symmetrized inputs, with
//!   the constants precomputed in `StageData::winograd`).
//! - **Smooth-subgroup-derived roots**
//!   ([`primitive_nth_root`](crate::fft::primitive_nth_root)):
//!   `ω_n` is one exponentiation of the field's compile-time
//!   `SmoothFftField::SMOOTH_OMEGA` literal — no runtime base scan.
//!
//! # Coset evaluation and RS-extend
//!
//! Reed-Solomon extension interpolates a polynomial through the `k`
//! known evaluations (one inverse FFT) then evaluates it on
//! `blowup − 1` cosets of the base subgroup. Each coset evaluation is
//! a coset FFT — pre-twist `c_i ← c_i · s^i` then run a plain forward
//! FFT — see [`SmoothDomain::coset_forward`](crate::fft::SmoothDomain::coset_forward)
//! and [`SmoothDomain::rs_extend_batch`](crate::fft::SmoothDomain::rs_extend_batch).

use crate::{FieldCore, FromPrimitiveInt, Invertible, SmoothFftField};

/// Compute `base^exp` by repeated squaring.
#[inline]
pub fn field_pow<F: FieldCore>(base: F, mut exp: u64) -> F {
    let mut result = F::one();
    let mut b = base;
    while exp > 0 {
        if exp & 1 == 1 {
            result *= b;
        }
        b *= b;
        exp >>= 1;
    }
    result
}

/// Compute `base^exp` for u128 exponents. Test-only scanner helper.
#[cfg(test)]
pub(crate) fn field_pow_u128<F: FieldCore>(base: F, mut exp: u128) -> F {
    let mut result = F::one();
    let mut b = base;
    while exp > 0 {
        if exp & 1 == 1 {
            result *= b;
        }
        b *= b;
        exp >>= 1;
    }
    result
}

/// Smallest prime factor of `n` (returns `n` itself if `n ≤ 1` or is prime).
fn smallest_prime_factor(n: usize) -> usize {
    if n <= 1 {
        return n;
    }
    for &p in &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31] {
        if n.is_multiple_of(p) {
            return p;
        }
    }
    let mut i = 37;
    while i * i <= n {
        if n.is_multiple_of(i) {
            return i;
        }
        i += 2;
    }
    n
}

/// Prime factorization of `n` (with multiplicity), in non-decreasing order.
///
/// The mixed-radix decomposition uses each prime factor as the radix of
/// one stage, so e.g. `n = 14_700 = 2² · 3 · 5² · 7²` becomes seven
/// stages `[2, 2, 3, 5, 5, 7, 7]`.
fn factorize(mut n: usize) -> Vec<usize> {
    let mut factors = Vec::new();
    while n > 1 {
        let p = smallest_prime_factor(n);
        factors.push(p);
        n /= p;
    }
    factors
}

/// Mixed-radix digit-reversal permutation, the analogue of bit-reversal
/// for power-of-two FFTs.
///
/// For `n = f_0 · f_1 · … · f_{s−1}`, write index `k` in mixed-radix
/// form `k = d_0 + d_1 · f_0 + d_2 · f_0 · f_1 + …`; then `perm[k]` is
/// the index whose digits are the reverse sequence. Permuting the
/// input by this table aligns the recursion's base cases at
/// consecutive indices, which is what lets the bottom-up DIT sweep
/// work in place.
fn digit_reversal_permutation(n: usize, factors: &[usize]) -> Vec<usize> {
    let s = factors.len();
    let mut perm = vec![0usize; n];
    for (k, perm_k) in perm.iter_mut().enumerate() {
        let mut digits = vec![0usize; s];
        let mut tmp = k;
        for (digit, &f) in digits.iter_mut().zip(factors.iter()) {
            *digit = tmp % f;
            tmp /= f;
        }
        let mut rev = 0usize;
        for (&f, &d) in factors.iter().zip(digits.iter()) {
            rev = rev * f + d;
        }
        *perm_k = rev;
    }
    perm
}

/// Read-only data the inner butterfly loop consults at every stage.
///
/// One per prime factor of `n`. Stage `i` combines `r` blocks of size
/// `block` (the product of all earlier stages' radices) into a single
/// block of size `block · r`.
struct StageData<F> {
    /// Radix of this stage — one of `{2, 3, 5, 7}` in practice.
    r: usize,
    /// Block size feeding this stage; output blocks are `block · r`.
    block: usize,
    /// `omega_r_pow[q] = ω_r^q` for `q ∈ 0..r`. Length fixed at 8 for
    /// stack storage; entries past `r` stay `1` and are unused.
    omega_r_pow: [F; 8],
    /// `twiddle_table[j] = ω_{block · r}^j`, indexed by lane within a
    /// group. The DIT butterfly's `ω^{jk}` factor decomposes as
    /// `(twiddle_table[j])^k`, with `tw^k` materialized on the fly.
    twiddle_table: Vec<F>,
    /// Precomputed Winograd constants for the low-mul radix-5 / 7
    /// kernels (empty for other radices). Layout:
    ///
    /// - `r == 5`: `[α/2, β/2, γ/2, δ/2, (α+β)/2, (γ+δ)/2]` where
    ///   `α = ω+ω⁴`, `β = ω²+ω³`, `γ = ω−ω⁴`, `δ = ω²−ω³`.
    /// - `r == 7`: 9 `α_{jk}` then 9 `β_{jk}`, row-major over
    ///   `(j, k) ∈ {1, 2, 3}²`, with
    ///   `α_{jk} = (ω^{jk} + ω^{−jk})/2` and
    ///   `β_{jk} = (ω^{jk} − ω^{−jk})/2`.
    ///
    /// The `/2` is folded into the stored values so the kernel doesn't
    /// halve at every butterfly.
    winograd: Vec<F>,
}

/// Build the per-stage tables consumed by the iterative FFT.
///
/// Walks `factors` in reverse so the resulting `Vec<StageData>` is
/// already in bottom-up sweep order. For each stage:
///
/// - `omega_new_block = omega^{n/(block · r)}` is the principal
///   `(block · r)`-th root of unity used to fill the twiddle table.
/// - `omega_r = omega_new_block^block` is the principal `r`-th root
///   used inside the size-`r` butterfly.
///
/// Called twice per domain — once with `omega = ω_n` for the forward
/// transform and once with `omega = ω_n^{−1}` for the inverse.
fn precompute_stages<F: FieldCore + FromPrimitiveInt + Invertible>(
    omega: F,
    n: usize,
    factors: &[usize],
) -> Vec<StageData<F>> {
    let mut stages = Vec::with_capacity(factors.len());
    let mut block = 1usize;

    for &r in factors.iter().rev() {
        debug_assert!(r <= 8, "radix {r} exceeds omega_r_pow capacity (max 8)");
        let new_block = block * r;
        let omega_new_block = field_pow(omega, (n / new_block) as u64);
        let omega_r = field_pow(omega_new_block, block as u64);

        let mut omega_r_pow = [F::one(); 8];
        for q in 1..r {
            omega_r_pow[q] = omega_r_pow[q - 1] * omega_r;
        }

        let mut twiddle_table = Vec::with_capacity(block);
        let mut tw = F::one();
        for _ in 0..block {
            twiddle_table.push(tw);
            tw *= omega_new_block;
        }

        let winograd = winograd_consts_for_radix::<F>(r, &omega_r_pow);

        stages.push(StageData {
            r,
            block,
            omega_r_pow,
            twiddle_table,
            winograd,
        });

        block = new_block;
    }
    stages
}

/// Precompute the Winograd constants consumed by the radix-5 / 7
/// kernels. Returns an empty vector for other radices. See the
/// doc-comment on `StageData::winograd` for the exact layout.
fn winograd_consts_for_radix<F: FieldCore + FromPrimitiveInt + Invertible>(
    r: usize,
    omega_r_pow: &[F; 8],
) -> Vec<F> {
    match r {
        5 => {
            let w1 = omega_r_pow[1];
            let w2 = omega_r_pow[2];
            let w3 = omega_r_pow[3];
            let w4 = omega_r_pow[4];
            let half = F::from_u64(2)
                .inverse()
                .expect("2 is invertible in a non-binary field");
            // α = ω+ω⁴, β = ω²+ω³, γ = ω−ω⁴, δ = ω²−ω³.
            let alpha_half = (w1 + w4) * half;
            let beta_half = (w2 + w3) * half;
            let gamma_half = (w1 - w4) * half;
            let delta_half = (w2 - w3) * half;
            // (α+β)/2 = (Σ_{q=1..4} ω^q)/2 = (-1)/2 since 1+ω+…+ω⁴ = 0.
            let ab_half = alpha_half + beta_half;
            let gd_half = gamma_half + delta_half;
            vec![
                alpha_half, beta_half, gamma_half, delta_half, ab_half, gd_half,
            ]
        }
        7 => {
            // ω^{−q} mod 7 = ω^{7−q}; map negative exponents through
            // `rem_euclid` so we can index the precomputed `omega_r_pow`
            // table for both signs.
            let w = omega_r_pow;
            let pow = |q: isize| -> F {
                let qq = q.rem_euclid(7) as usize;
                w[qq]
            };
            let half = F::from_u64(2)
                .inverse()
                .expect("2 is invertible in a non-binary field");
            let mut out = Vec::with_capacity(18);
            // α_{jk} = (ω^{jk} + ω^{−jk})/2, row-major in (j, k).
            for j in 1..=3 {
                for k in 1..=3 {
                    let jk = (j * k) as isize;
                    out.push((pow(jk) + pow(-jk)) * half);
                }
            }
            // β_{jk} = (ω^{jk} − ω^{−jk})/2, row-major in (j, k).
            for j in 1..=3 {
                for k in 1..=3 {
                    let jk = (j * k) as isize;
                    out.push((pow(jk) - pow(-jk)) * half);
                }
            }
            out
        }
        _ => Vec::new(),
    }
}

/// Pre-allocated ping-pong buffers for an iterative mixed-radix FFT.
///
/// `buf_a` is updated in place across all stages and holds the result
/// on return. `buf_b` is a scratch slot callers can pre-fill (see
/// `execute_from_b`); reused across the inverse and forward passes
/// inside `rs_extend_batch`.
struct FftWorkspace<F> {
    n: usize,
    buf_a: Vec<F>,
    buf_b: Vec<F>,
}

impl<F: FieldCore> FftWorkspace<F> {
    fn new(n: usize) -> Self {
        Self {
            n,
            buf_a: vec![F::zero(); n],
            buf_b: vec![F::zero(); n],
        }
    }

    /// Run an iterative mixed-radix Cooley-Tukey DIT FFT on `input`:
    /// digit-reverse into `buf_a`, then sweep `stages` bottom-up
    /// running radix-`r` butterflies in place. Returns a view into
    /// `buf_a`.
    fn execute(&mut self, input: &[F], stages: &[StageData<F>], digit_rev: &[usize]) -> &[F] {
        let n = self.n;
        debug_assert_eq!(input.len(), n);

        for (i, &rev_i) in digit_rev.iter().enumerate() {
            self.buf_a[rev_i] = input[i];
        }

        self.butterfly_stages(stages);
        &self.buf_a[..n]
    }

    /// Like [`Self::execute`], but reads the input from a `buf_b` the
    /// caller has already populated. Used by `coset_forward` to avoid
    /// an extra allocation for the twisted coefficient vector.
    fn execute_from_b(&mut self, stages: &[StageData<F>], digit_rev: &[usize]) -> &[F] {
        let n = self.n;

        for (i, &rev_i) in digit_rev.iter().enumerate() {
            self.buf_a[rev_i] = self.buf_b[i];
        }

        self.butterfly_stages(stages);
        &self.buf_a[..n]
    }

    /// Bottom-up FFT sweep. For each stage, the outer loop walks
    /// independent groups of `block · r` consecutive entries; the
    /// middle loop runs the `block` parallel butterflies inside one
    /// group; each butterfly does a twiddle phase (scale lane `k` by
    /// `twiddle_table[j]^k`) followed by a size-`r` DFT specialized
    /// per radix.
    fn butterfly_stages(&mut self, stages: &[StageData<F>]) {
        let n = self.n;
        for stage in stages {
            let r = stage.r;
            let block = stage.block;
            let new_block = block * r;
            let omega_r_pow = &stage.omega_r_pow;
            let twiddle_table = &stage.twiddle_table;

            for group_start in (0..n).step_by(new_block) {
                for (j, tw_entry) in twiddle_table.iter().enumerate() {
                    let base = group_start + j;

                    // Gather the `r` lanes of this butterfly into a
                    // stack array (cap of 8 is debug-asserted in
                    // `precompute_stages`).
                    let mut x = [F::zero(); 8];
                    for (ki, xi) in x[..r].iter_mut().enumerate() {
                        *xi = self.buf_a[base + ki * block];
                    }

                    if j > 0 {
                        // Twiddle phase: scale lane k by tw^k. The
                        // unrolled per-radix sequences below share
                        // tw², tw³, … across lanes; the generic loop
                        // covers radices we don't have a tuned kernel
                        // for. Skipped entirely when j == 0 (tw = 1).
                        let tw = *tw_entry;
                        let tw2 = tw * tw;
                        match r {
                            2 => {
                                x[1] *= tw;
                            }
                            3 => {
                                x[1] *= tw;
                                x[2] *= tw2;
                            }
                            5 => {
                                let tw3 = tw2 * tw;
                                let tw4 = tw2 * tw2;
                                x[1] *= tw;
                                x[2] *= tw2;
                                x[3] *= tw3;
                                x[4] *= tw4;
                            }
                            7 => {
                                let tw3 = tw2 * tw;
                                let tw4 = tw2 * tw2;
                                let tw5 = tw4 * tw;
                                let tw6 = tw3 * tw3;
                                x[1] *= tw;
                                x[2] *= tw2;
                                x[3] *= tw3;
                                x[4] *= tw4;
                                x[5] *= tw5;
                                x[6] *= tw6;
                            }
                            _ => {
                                let mut tw_k = tw;
                                for xi in &mut x[1..r] {
                                    *xi *= tw_k;
                                    tw_k *= tw;
                                }
                            }
                        }
                    }

                    // DFT phase: hand-tuned size-r kernel per radix.
                    match r {
                        2 => {
                            self.buf_a[base] = x[0] + x[1];
                            self.buf_a[base + block] = x[0] - x[1];
                        }
                        3 => {
                            // 2-mul DFT_3 from 1 + ω + ω² = 0:
                            //   S = x₁ + x₂, T = ω·x₁ + ω²·x₂
                            //   y₀ = x₀ + S, y₁ = x₀ + T, y₂ = x₀ − S − T
                            let w1 = omega_r_pow[1];
                            let w2 = omega_r_pow[2];
                            let s = x[1] + x[2];
                            let t = x[1] * w1 + x[2] * w2;
                            self.buf_a[base] = x[0] + s;
                            self.buf_a[base + block] = x[0] + t;
                            self.buf_a[base + 2 * block] = x[0] - s - t;
                        }
                        5 => {
                            // 6-mul DFT_5 via Karatsuba on the
                            // (A, B) = x_j ± x_{5−j} pairs. Constants
                            // come from winograd_consts_for_radix(5):
                            //   [α/2, β/2, γ/2, δ/2, (α+β)/2, (γ+δ)/2]
                            let cc = &stage.winograd;
                            debug_assert_eq!(cc.len(), 6);
                            let a_h = cc[0];
                            let b_h = cc[1];
                            let g_h = cc[2];
                            let d_h = cc[3];
                            let ab_h = cc[4];
                            let gd_h = cc[5];

                            let a = x[1] + x[4];
                            let b = x[2] + x[3];
                            let c = x[1] - x[4];
                            let d = x[2] - x[3];

                            // P-block (cosine, Karatsuba k₁+k₂+k₃):
                            //   P₁ = A·α/2 + B·β/2,  P₂ = A·β/2 + B·α/2
                            let k1 = a * a_h;
                            let k2 = b * b_h;
                            let k3 = (a + b) * ab_h;
                            let p1 = k1 + k2;
                            let p2 = k3 - k1 - k2;

                            // Q-block (sine, complex-mul Karatsuba):
                            //   Q₁ = C·γ/2 + D·δ/2, Q₂ = C·δ/2 − D·γ/2
                            let m1 = c * g_h;
                            let m2 = d * d_h;
                            let m3 = (c - d) * gd_h;
                            let q1 = m1 + m2;
                            let q2 = m3 - m1 + m2;

                            self.buf_a[base] = x[0] + a + b;
                            self.buf_a[base + block] = x[0] + p1 + q1;
                            self.buf_a[base + 2 * block] = x[0] + p2 + q2;
                            self.buf_a[base + 3 * block] = x[0] + p2 - q2;
                            self.buf_a[base + 4 * block] = x[0] + p1 - q1;
                        }
                        7 => {
                            // 18-mul DFT_7. Same conjugate-pair idea
                            // as DFT_5: pair x_j with x_{7−j} into
                            // A_j = x_j + x_{7−j} (symmetric) and
                            // B_j = x_j − x_{7−j} (antisymmetric), so
                            //
                            //   x_j·ω^{jk} + x_{7−j}·ω^{−jk}
                            //     = A_j · α_{jk} + B_j · β_{jk}
                            //
                            // with α_{jk}, β_{jk} (already including
                            // the /2) precomputed in `winograd`.
                            // Outputs y₄, y₅, y₆ recover by flipping
                            // the β sign.
                            let cc = &stage.winograd;
                            debug_assert_eq!(cc.len(), 18);

                            let a1 = x[1] + x[6];
                            let a2 = x[2] + x[5];
                            let a3 = x[3] + x[4];
                            let b1 = x[1] - x[6];
                            let b2 = x[2] - x[5];
                            let b3 = x[3] - x[4];

                            // α table at offset (j-1)*3 + (k-1).
                            let s1 = a1 * cc[0] + a2 * cc[3] + a3 * cc[6]; // k = 1
                            let s2 = a1 * cc[1] + a2 * cc[4] + a3 * cc[7]; // k = 2
                            let s3 = a1 * cc[2] + a2 * cc[5] + a3 * cc[8]; // k = 3

                            // β table at offset 9 + (j-1)*3 + (k-1).
                            let t1 = b1 * cc[9] + b2 * cc[12] + b3 * cc[15];
                            let t2 = b1 * cc[10] + b2 * cc[13] + b3 * cc[16];
                            let t3 = b1 * cc[11] + b2 * cc[14] + b3 * cc[17];

                            self.buf_a[base] = x[0] + a1 + a2 + a3;
                            self.buf_a[base + block] = x[0] + s1 + t1;
                            self.buf_a[base + 2 * block] = x[0] + s2 + t2;
                            self.buf_a[base + 3 * block] = x[0] + s3 + t3;
                            self.buf_a[base + 4 * block] = x[0] + s3 - t3;
                            self.buf_a[base + 5 * block] = x[0] + s2 - t2;
                            self.buf_a[base + 6 * block] = x[0] + s1 - t1;
                        }
                        _ => {
                            // Naive O(r²) fallback.
                            for (q, &wq) in omega_r_pow[..r].iter().enumerate() {
                                let mut val = x[0];
                                let mut w = wq;
                                for &xp in &x[1..r] {
                                    val += xp * w;
                                    w *= wq;
                                }
                                self.buf_a[base + q * block] = val;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Mixed-radix FFT domain backed by a smooth-order multiplicative subgroup.
///
/// Holds the immutable state for a fixed-size FFT (roots of unity,
/// digit-reversal permutation, per-stage twiddle tables for both
/// directions). Build once with [`SmoothDomain::new`] and reuse across
/// transforms; `Sync`-safe since all fields are read-only after
/// construction.
pub struct SmoothDomain<F> {
    /// Number of points in the FFT domain.
    pub n: usize,
    /// Primitive `n`-th root of unity that generates the domain.
    pub omega: F,
    /// `n⁻¹`, applied to normalize the inverse transform.
    n_inv: F,
    /// Mixed-radix digit-reversal permutation, length `n`.
    digit_rev: Vec<usize>,
    /// Per-stage tables for the forward transform (twiddles in `ω`).
    fwd_stages: Vec<StageData<F>>,
    /// Per-stage tables for the inverse transform (twiddles in `ω⁻¹`).
    inv_stages: Vec<StageData<F>>,
}

impl<F: FieldCore + FromPrimitiveInt + Invertible + std::fmt::Debug> SmoothDomain<F> {
    /// Build a domain of size `n` from a primitive `n`-th root of
    /// unity. Precomputes the digit-reversal permutation and per-stage
    /// tables for both forward and inverse transforms.
    ///
    /// # Panics
    /// If `omega` is zero or `n` is not invertible in the field.
    pub fn new(omega: F, n: usize) -> Self {
        debug_assert_primitive_nth_root(omega, n);
        let omega_inv = omega.inverse().expect("omega must be nonzero");
        let n_inv = F::from_u64(n as u64)
            .inverse()
            .expect("n must be invertible in field");
        let factors = factorize(n);
        let digit_rev = digit_reversal_permutation(n, &factors);
        let fwd_stages = precompute_stages(omega, n, &factors);
        let inv_stages = precompute_stages(omega_inv, n, &factors);
        Self {
            n,
            omega,
            n_inv,
            digit_rev,
            fwd_stages,
            inv_stages,
        }
    }

    /// Forward DFT: `Y[k] = Σ_{j=0}^{n-1} x[j] · ω^{jk}`.
    ///
    /// # Panics
    /// If `input.len() != n`.
    pub fn forward(&self, input: &[F]) -> Vec<F> {
        assert_eq!(input.len(), self.n);
        let mut ws = FftWorkspace::new(self.n);
        ws.execute(input, &self.fwd_stages, &self.digit_rev)
            .to_vec()
    }

    /// Inverse DFT: `x[j] = (1/n) · Σ_{k=0}^{n-1} Y[k] · ω^{-jk}`.
    ///
    /// # Panics
    /// If `input.len() != n`.
    pub fn inverse(&self, input: &[F]) -> Vec<F> {
        assert_eq!(input.len(), self.n);
        let mut ws: FftWorkspace<F> = FftWorkspace::new(self.n);
        let mut result = ws
            .execute(input, &self.inv_stages, &self.digit_rev)
            .to_vec();
        for v in &mut result {
            *v *= self.n_inv;
        }
        result
    }

    /// Evaluate a polynomial at the shifted coset
    /// `{shift · ω^i | i = 0, …, n−1}`.
    ///
    /// Reduces to a plain forward DFT on twisted coefficients via
    ///
    /// `P(shift · ω^i) = Σ_j (c_j · shift^j) · ω^{ij}`,
    ///
    /// so we pre-twist `c_j ← c_j · shift^j` into `buf_b`
    /// (zero-padding any unused tail) and forward-FFT from there.
    ///
    /// # Panics
    /// If `coeffs.len() > n`.
    pub fn coset_forward(&self, coeffs: &[F], shift: F) -> Vec<F> {
        assert!(coeffs.len() <= self.n);
        let mut ws: FftWorkspace<F> = FftWorkspace::new(self.n);
        let buf = &mut ws.buf_b[..self.n];
        let mut tw = F::one();
        for (i, &c) in coeffs.iter().enumerate() {
            buf[i] = c * tw;
            tw *= shift;
        }
        for v in &mut buf[coeffs.len()..] {
            *v = F::zero();
        }
        ws.execute_from_b(&self.fwd_stages, &self.digit_rev)
            .to_vec()
    }

    /// Reed-Solomon-extend `evals` from the base subgroup
    /// `K = {ω_K^i}` (with `ω_K = ω_n^{blowup}`, `k = self.n`) to the
    /// `blowup − 1` non-trivial cosets of `K` inside the larger
    /// size-`(k · blowup)` subgroup.
    ///
    /// One inverse FFT recovers the polynomial through `evals`, then
    /// `blowup − 1` coset forward FFTs (shifts `ω_n^j` for
    /// `j = 1, …, blowup − 1`) evaluate it on each extension coset.
    /// Returns `k · (blowup − 1)` values, coset-major; the original
    /// evaluations on `K` are not re-emitted. All transforms share a
    /// single workspace.
    ///
    /// # Panics
    /// If `evals.len() != n`.
    pub fn rs_extend_batch(&self, evals: &[F], omega_n: F, blowup: usize) -> Vec<F> {
        let k = self.n;
        assert_eq!(evals.len(), k);

        let mut ws: FftWorkspace<F> = FftWorkspace::new(self.n);

        let mut coeffs = ws
            .execute(evals, &self.inv_stages, &self.digit_rev)
            .to_vec();
        for v in &mut coeffs {
            *v *= self.n_inv;
        }

        let mut extension = Vec::with_capacity(k * (blowup - 1));
        for j in 1..blowup {
            let shift = field_pow(omega_n, j as u64);
            let buf = &mut ws.buf_b[..k];
            let mut tw = F::one();
            for (i, &c) in coeffs.iter().enumerate() {
                buf[i] = c * tw;
                tw *= shift;
            }
            let result = ws.execute_from_b(&self.fwd_stages, &self.digit_rev);
            extension.extend_from_slice(result);
        }
        extension
    }
}

/// Primitive `n`-th root of unity in `F`, derived from
/// [`SmoothFftField::SMOOTH_OMEGA`] as
/// `omega_n = SMOOTH_OMEGA ^ (SMOOTH_SUBGROUP_ORDER / n)`. Requires
/// `n | SMOOTH_SUBGROUP_ORDER`.
///
/// # Panics
/// If `n` does not divide [`SmoothFftField::SMOOTH_SUBGROUP_ORDER`], or
/// if `SMOOTH_OMEGA` is not in canonical form.
pub fn primitive_nth_root<F: SmoothFftField>(n: usize) -> F {
    assert!(n > 0, "n must be positive");
    assert_eq!(
        F::SMOOTH_SUBGROUP_ORDER % n,
        0,
        "n={n} must divide SMOOTH_SUBGROUP_ORDER={}",
        F::SMOOTH_SUBGROUP_ORDER
    );
    // Checked construction so a literal `≥ p` panics rather than
    // being silently reduced.
    let omega = F::from_canonical_u128_checked(F::SMOOTH_OMEGA)
        .expect("SMOOTH_OMEGA must be < p (canonical form)");
    field_pow(omega, (F::SMOOTH_SUBGROUP_ORDER / n) as u64)
}

/// Find a primitive `n`-th root of unity in `F` by scanning small
/// bases.
///
/// Verifies primitivity against every distinct prime factor of `n`, so
/// it remains correct when a base lands in a strict subgroup (e.g.
/// `g = 2` is a quadratic residue modulo `Prime128OffsetA7F7`, so
/// `2^{(p−1)/n}` has order `n/2`).
///
/// Used by per-prime tests as a drift guard on
/// [`SmoothFftField::SMOOTH_OMEGA`]; production code should call
/// [`primitive_nth_root`] instead.
///
/// # Panics
/// If `n` does not divide `p − 1`, or if no base in `{2, 3, …, 47}`
/// yields a primitive `n`-th root.
#[cfg(test)]
#[expect(
    clippy::panic,
    reason = "test-only primitive-root scanner fails loudly"
)]
pub(crate) fn find_primitive_nth_root<F: FieldCore + FromPrimitiveInt>(
    p_minus_1: u128,
    n: usize,
) -> F {
    assert_eq!(
        p_minus_1 % (n as u128),
        0,
        "n={n} must divide p-1={p_minus_1}"
    );
    let exp = p_minus_1 / (n as u128);
    let prime_factors = distinct_prime_factors(n);

    for &g in &[2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47] {
        let candidate = field_pow_u128(F::from_u64(g), exp);
        if !is_primitive_nth_root(candidate, n, &prime_factors) {
            continue;
        }
        return candidate;
    }
    panic!("no primitive {n}-th root of unity found in scanned bases");
}

/// Distinct prime factors of `n`, sorted ascending.
fn distinct_prime_factors(n: usize) -> Vec<usize> {
    let mut factors = factorize(n);
    factors.sort_unstable();
    factors.dedup();
    factors
}

/// Test whether `omega` has exact multiplicative order `n`, given a slice
/// containing every distinct prime factor of `n`.
fn is_primitive_nth_root<F: FieldCore>(omega: F, n: usize, distinct_factors: &[usize]) -> bool {
    if field_pow(omega, n as u64) != F::one() {
        return false;
    }
    distinct_factors
        .iter()
        .all(|&q| field_pow(omega, (n / q) as u64) != F::one())
}

/// Debug-only check that `omega` is a primitive `n`-th root of unity.
fn debug_assert_primitive_nth_root<F: FieldCore>(omega: F, n: usize) {
    if !cfg!(debug_assertions) {
        return;
    }
    let factors = distinct_prime_factors(n);
    assert!(
        is_primitive_nth_root(omega, n, &factors),
        "omega is not a primitive {n}-th root of unity (n's prime factors: {factors:?})"
    );
}

/// Free-function wrapper around [`SmoothDomain::rs_extend_batch`].
pub fn rs_extend_fft<F: FieldCore + FromPrimitiveInt + Invertible + std::fmt::Debug>(
    evals: &[F],
    domain_k: &SmoothDomain<F>,
    omega_n: F,
    blowup: usize,
) -> Vec<F> {
    domain_k.rs_extend_batch(evals, omega_n, blowup)
}

#[cfg(test)]
mod test_support {
    //! Prime-agnostic helpers shared by per-prime FFT parity tests.
    //!
    //! The two protocol primes (`Prime128Offset2355`, `Prime128OffsetA7F7`)
    //! have different smooth-subgroup factorizations, but the parity
    //! properties under test (FFT vs naive DFT, forward/inverse roundtrip,
    //! RS-extend consistency) are identical. Factor them out here so the
    //! per-prime modules carry only the size lattice that actually differs.
    //!
    //! All omegas come from [`super::primitive_nth_root`] (i.e. the
    //! field's `SmoothFftField::SMOOTH_OMEGA`); the scanner
    //! [`super::find_primitive_nth_root`] is only re-invoked by the
    //! `smooth_omega_matches_search` per-prime tests as a drift guard
    //! on the hardcoded literal.
    use super::*;
    use crate::FromPrimitiveInt;
    use std::fmt::Debug;
    use std::ops::{AddAssign, MulAssign};

    pub(super) use super::find_primitive_nth_root;

    /// O(n^2) naive DFT, used as oracle for the iterative FFT under test.
    fn naive_dft<F: FieldCore>(input: &[F], omega: F) -> Vec<F> {
        let n = input.len();
        let mut out = vec![F::zero(); n];
        for (k, ok) in out.iter_mut().enumerate() {
            for (j, &xj) in input.iter().enumerate() {
                *ok += xj * field_pow(omega, (j * k) as u64);
            }
        }
        out
    }

    /// For each `n` in `sizes` that divides `F::SMOOTH_SUBGROUP_ORDER`,
    /// assert the iterative FFT matches the naive DFT on a deterministic
    /// input vector. Sizes that do not divide are silently skipped so
    /// per-prime modules can share a single union of "interesting" sizes.
    pub(super) fn assert_fft_matches_naive_dft<F>(sizes: &[usize])
    where
        F: SmoothFftField + FromPrimitiveInt + Invertible + Debug,
    {
        for &n in sizes {
            if F::SMOOTH_SUBGROUP_ORDER % n != 0 {
                continue;
            }
            let omega = primitive_nth_root::<F>(n);
            let input: Vec<F> = (0..n).map(|i| F::from_u64((i + 1) as u64)).collect();
            let expected = naive_dft(&input, omega);

            let factors = factorize(n);
            let digit_rev = digit_reversal_permutation(n, &factors);
            let stages = precompute_stages(omega, n, &factors);
            let mut ws: FftWorkspace<F> = FftWorkspace::new(n);
            let got = ws.execute(&input, &stages, &digit_rev).to_vec();
            assert_eq!(got, expected, "FFT mismatch for n={n}");
        }
    }

    /// `forward(inverse(x)) == x` over a smooth domain of order `n`.
    pub(super) fn assert_forward_inverse_roundtrip<F>(n: usize)
    where
        F: SmoothFftField + FromPrimitiveInt + Invertible + Debug,
    {
        let omega = primitive_nth_root::<F>(n);
        let domain = SmoothDomain::new(omega, n);
        let input: Vec<F> = (0..n).map(|i| F::from_u64(i as u64 + 1)).collect();
        let transformed = domain.forward(&input);
        let recovered = domain.inverse(&transformed);
        assert_eq!(input, recovered);
    }

    /// `rs_extend_fft` matches direct evaluation of the interpolating
    /// polynomial on each of the `blowup - 1` extension cosets.
    pub(super) fn assert_rs_extend_consistency<F>(k: usize, blowup: usize)
    where
        F: SmoothFftField + FromPrimitiveInt + Invertible + Debug + AddAssign + MulAssign,
    {
        let n = k * blowup;
        let omega_n = primitive_nth_root::<F>(n);
        let omega_k = field_pow(omega_n, blowup as u64);
        let domain_k = SmoothDomain::new(omega_k, k);

        let evals: Vec<F> = (0..k).map(|i| F::from_u64((i * 7 + 3) as u64)).collect();
        let coeffs = domain_k.inverse(&evals);
        let extension = rs_extend_fft(&evals, &domain_k, omega_n, blowup);
        assert_eq!(extension.len(), k * (blowup - 1));

        for j in 1..blowup {
            for i in 0..k {
                let point = field_pow(omega_n, j as u64) * field_pow(omega_k, i as u64);
                let mut expected = F::zero();
                let mut x_pow = F::one();
                for &c in &coeffs {
                    expected += c * x_pow;
                    x_pow *= point;
                }
                assert_eq!(
                    extension[(j - 1) * k + i],
                    expected,
                    "mismatch at coset {j}, position {i}"
                );
            }
        }
    }
}

#[cfg(test)]
mod prime_2355_tests {
    //! `Prime128Offset2355` (`p = 2^128 - 2355`) has smooth multiplicative
    //! subgroup of order `14_700 = 2^2 * 3 * 5^2 * 7^2`, drawing sizes from
    //! the `{2, 3, 5, 7}` lattice.
    use super::test_support::*;
    use super::*;
    use crate::Prime128Offset2355;
    use crate::{CanonicalField, PseudoMersenneField};

    type F = Prime128Offset2355;

    /// Drift guard: re-derive the primitive `SMOOTH_SUBGROUP_ORDER`-th
    /// root of unity from a base scan and assert it equals the literal
    /// declared in [`crate::prime::fp128`]. Also validates the
    /// trait's structural invariant `SMOOTH_SUBGROUP_ORDER | (p − 1)`.
    #[test]
    fn smooth_omega_matches_search() {
        let p_minus_1 = u128::MAX - F::MODULUS_OFFSET;
        assert_eq!(
            p_minus_1 % (F::SMOOTH_SUBGROUP_ORDER as u128),
            0,
            "SMOOTH_SUBGROUP_ORDER must divide p − 1",
        );
        let derived = find_primitive_nth_root::<F>(p_minus_1, F::SMOOTH_SUBGROUP_ORDER);
        let declared =
            F::from_canonical_u128_checked(F::SMOOTH_OMEGA).expect("SMOOTH_OMEGA must be < p");
        assert_eq!(
            derived, declared,
            "SMOOTH_OMEGA literal has drifted from the scanner's primitive root"
        );
    }

    #[test]
    fn primitive_nth_root_has_correct_order_for_every_divisor() {
        // Every `n | SMOOTH_SUBGROUP_ORDER` should yield a primitive
        // n-th root via the trait derivation.
        for &n in &[
            2, 3, 4, 5, 6, 7, 10, 12, 14, 15, 20, 21, 25, 28, 30, 35, 42, 49, 50, 60, 70, 75, 84,
            98, 100, 105, 140, 147, 150, 175, 196, 210, 245, 294, 300, 350, 420, 490, 525, 588,
            700, 735, 980, 1050, 1225, 1470, 2100, 2450, 2940, 3675, 4900, 7350, 14700,
        ] {
            if F::SMOOTH_SUBGROUP_ORDER % n != 0 {
                continue;
            }
            let omega = primitive_nth_root::<F>(n);
            let factors = distinct_prime_factors(n);
            assert!(
                is_primitive_nth_root(omega, n, &factors),
                "primitive_nth_root failed primitivity check for n={n}"
            );
        }
    }

    #[test]
    fn small_fft_matches_naive_dft() {
        assert_fft_matches_naive_dft::<F>(&[
            2, 3, 4, 5, 6, 7, 10, 12, 14, 15, 20, 21, 25, 28, 42, 49, 50,
        ]);
    }

    #[test]
    fn forward_inverse_roundtrip_300() {
        assert_forward_inverse_roundtrip::<F>(300);
    }

    #[test]
    fn forward_inverse_roundtrip_1470() {
        assert_forward_inverse_roundtrip::<F>(1470);
    }

    #[test]
    fn rs_extend_consistency() {
        // k = 300 = 2^2 * 3 * 5^2, blowup = 7, so n = 2_100 | 14_700.
        assert_rs_extend_consistency::<F>(300, 7);
    }
}

#[cfg(test)]
mod prime_a7f7_tests {
    //! `Prime128OffsetA7F7` (`p = 2^128 - 2^32 + 22537`) has smooth
    //! multiplicative subgroup of order `2^3 * 3^7 = 17_496`, with a pure
    //! radix-3 substructure of order `3^7 = 2_187`. Sizes are drawn from
    //! the `{2, 3}` lattice instead of `{2, 3, 5, 7}`.
    use super::test_support::*;
    use super::*;
    use crate::Prime128OffsetA7F7;
    use crate::{CanonicalField, PseudoMersenneField};

    type F = Prime128OffsetA7F7;

    /// Cross-implementation check: the radix-3 GPU NTT in
    /// `gpu_bench/primeB_roots.hpp` bakes
    /// `OMEGA_2187 = 2^((p_B − 1)/2187)` into a separate constant table.
    /// The two implementations independently choose their generators (the
    /// GPU uses `g = 2`, the Rust scanner uses the smallest base whose
    /// `g^((p−1)/n)` reaches full order `n`), so the constants are not
    /// expected to be *equal*; they are expected to be *primitive
    /// 2187-th roots of unity in the same field*. We verify the GPU's
    /// limb table is a valid primitive 2187-th root under the Rust
    /// `Fp128` implementation, which is the meaningful invariant for
    /// cross-impl correctness.
    #[test]
    fn gpu_omega_2187_is_primitive_in_rust_field() {
        // Limbs from `gpu_bench/primeB_roots.hpp` OMEGA_2187, packed
        // little-endian into a u128.
        const GPU_OMEGA_2187: u128 = 0x44E6_6EEC_31E7_36A6_A030_9253_219B_CCCD;
        let omega = F::from_canonical_u128_checked(GPU_OMEGA_2187)
            .expect("GPU OMEGA_2187 must lie in [0, p_B)");
        let factors = distinct_prime_factors(2187);
        assert!(
            is_primitive_nth_root(omega, 2187, &factors),
            "gpu_bench OMEGA_2187 is not a primitive 2187-th root under Rust Fp128",
        );

        // The GPU also bakes OMEGA_3 = OMEGA_2187^729 (a primitive cube
        // root). Cross-check that limb table too.
        const GPU_OMEGA_3: u128 = 0x66F1_B0EE_0E4A_40F7_0F69_0C7F_0F66_39DD;
        let omega3 =
            F::from_canonical_u128_checked(GPU_OMEGA_3).expect("GPU OMEGA_3 must lie in [0, p_B)");
        assert_eq!(field_pow(omega, 729), omega3, "OMEGA_3 != OMEGA_2187^729");
        assert!(
            is_primitive_nth_root(omega3, 3, &distinct_prime_factors(3)),
            "gpu_bench OMEGA_3 is not a primitive cube root",
        );
    }

    /// Drift guard: see `prime_2355_tests::smooth_omega_matches_search`.
    #[test]
    fn smooth_omega_matches_search() {
        let p_minus_1 = u128::MAX - F::MODULUS_OFFSET;
        assert_eq!(
            p_minus_1 % (F::SMOOTH_SUBGROUP_ORDER as u128),
            0,
            "SMOOTH_SUBGROUP_ORDER must divide p − 1",
        );
        let derived = find_primitive_nth_root::<F>(p_minus_1, F::SMOOTH_SUBGROUP_ORDER);
        let declared =
            F::from_canonical_u128_checked(F::SMOOTH_OMEGA).expect("SMOOTH_OMEGA must be < p");
        assert_eq!(
            derived, declared,
            "SMOOTH_OMEGA literal has drifted from the scanner's primitive root"
        );
    }

    #[test]
    fn primitive_nth_root_has_correct_order_for_every_divisor() {
        for &n in &[
            2, 3, 6, 8, 9, 18, 24, 27, 54, 81, 162, 243, 486, 729, 1458, 2187, 4374, 8748, 17496,
        ] {
            if F::SMOOTH_SUBGROUP_ORDER % n != 0 {
                continue;
            }
            let omega = primitive_nth_root::<F>(n);
            let factors = distinct_prime_factors(n);
            assert!(
                is_primitive_nth_root(omega, n, &factors),
                "primitive_nth_root failed primitivity check for n={n}"
            );
        }
    }

    #[test]
    fn small_fft_matches_naive_dft() {
        assert_fft_matches_naive_dft::<F>(&[2, 3, 6, 8, 9, 18, 24, 27, 54, 81, 162, 243, 486, 729]);
    }

    #[test]
    fn forward_inverse_roundtrip_243() {
        assert_forward_inverse_roundtrip::<F>(243);
    }

    #[test]
    fn forward_inverse_roundtrip_1458() {
        assert_forward_inverse_roundtrip::<F>(1458);
    }

    #[test]
    fn forward_inverse_roundtrip_2187() {
        assert_forward_inverse_roundtrip::<F>(2187);
    }

    #[test]
    fn rs_extend_consistency() {
        // k = 243 (= 3^5), blowup = 9 (= 3^2), n = 3^7 = 2_187 | 17_496.
        assert_rs_extend_consistency::<F>(243, 9);
    }
}
