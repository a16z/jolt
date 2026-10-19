//! Exact operator-norm acceptance predicate `gamma_D(c) <= T`.
//!
//! For `c in Z[X]/(X^D + 1)` the negacyclic convolution operator norm is
//!
//! ```text
//! gamma_D(c) = max_{0 <= k < D} |c_hat(k)|,
//! c_hat(k)   = sum_{j} c_j * zeta_k^j,   zeta_k = exp((2k + 1) * pi * i / D),
//! ```
//!
//! and `accept(c) iff gamma_D(c) <= T iff max_k |c_hat(k)|^2 <= T^2`.
//!
//! Soundness contract (the only invariant callers depend on): a returned
//! [`Decision::Accept`] means `gamma_D(c) <= T` over the reals. Machine
//! floating-point FFTs cannot honor that contract (their rounding could turn a
//! just-over-threshold challenge into an unsound accept), so the decision is
//! integer-only:
//!
//! - The `2*D` distinct trig values `cos(pi*t/D), sin(pi*t/D)` are tabulated as
//!   signed fixed-point integers at scale `2^q`, each carried with a *sound*
//!   per-entry error bound `eps_root` (`|C[t] - 2^q cos(pi t/D)| <= eps_root`).
//!   The table is built by interval Taylor evaluation over a rigorous rational
//!   enclosure of `pi` (Machin series in `i128`); no floating point enters the
//!   table or the decision.
//! - For each frequency the predicate forms integer accumulators `R_k, I_k`,
//!   then the conservative upper bound
//!   `upper_k = R_k^2 + I_k^2 + 2(|R_k| + |I_k|) r + 2 r^2` with `r = ||c||_1 *
//!   eps_root`, which provably bounds `|2^q c_hat(k)|^2` from above.
//!   [`Decision::Accept`] is returned only when every `upper_k <= T^2 2^{2q}`,
//!   so acceptance is always sound. A frequency whose *lower* bound already
//!   exceeds the threshold forces [`Decision::Reject`]; anything in between is
//!   [`Decision::Indeterminate`]. Callers that fold the boundary band into a
//!   reject (see [`OpNormTable::accept_strict_parts`]) stay sound, at the cost of a
//!   slightly smaller accepted support.
//!
//! Only the `D/2` frequencies `k = 0..D/2` are scanned: `zeta_{D-1-k} =
//! conj(zeta_k)` and `c` is real, so `|c_hat(D-1-k)| = |c_hat(k)|`.
//!
//! This is the acceptance oracle for operator-norm rejection sampling of fold
//! challenges: the crate's private sparse sampler retains a sampled challenge
//! only if it passes [`OpNormTable::accept_strict_parts`].

use akita_error::AkitaError;

use super::op_norm_accumulate;

/// Internal fixed-point scale for the certified `pi` and trig interval math.
/// Chosen so that products of two scaled values stay within `i128`
/// (`2 * TRIG_SCALE < 127`).
const TRIG_SCALE: u32 = 61;
/// `1.0` at [`TRIG_SCALE`].
const TRIG_ONE: i128 = 1i128 << TRIG_SCALE;
/// Taylor terms for the small-angle (`<= pi/4`) cos/sin enclosures. The first
/// omitted term is below `2^-29` at [`TRIG_SCALE`] for `phi <= pi/4`, and the
/// largest factorial used for the remainder bound (`26!`) fits `i128`.
const NTERMS: usize = 12;
/// Decision of the operator-norm acceptance predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Decision {
    /// Proven `gamma_D(c) <= T`.
    Accept,
    /// Proven `gamma_D(c) > T`.
    Reject,
    /// Inside the certified uncertainty band; neither bound is decisive.
    Indeterminate,
}

/// Certified fixed-point cos/sin tables for one `(D, q)` plus the global sound
/// error bound `eps_root`, with accumulator overflow validated for a stated
/// `max_l1` / `max_t`.
#[derive(Debug, Clone)]
pub(crate) struct OpNormTable {
    d: usize,
    q: u32,
    /// Certified `2^q cos/sin(pi t / D)` source tables, retained only for the
    /// accuracy/reference unit tests; production reads `freq_*_at`.
    #[cfg(test)]
    base_cos: Vec<i128>,
    #[cfg(test)]
    base_sin: Vec<i128>,
    /// Sound bound `|base_*[t] - 2^q trig(pi t / D)| <= eps_root`.
    eps_root: i128,
    /// Largest `||c||_1` the overflow validation covered.
    max_l1: i128,
    /// Largest threshold `t` the overflow validation covered.
    max_t: i128,
    /// Transposed rows `freq_cos_at[pos * half_d + k]` for fast accumulation,
    /// the `i64` downcast of the certified `base_*` tables (`|entry| <= 2^q`).
    freq_cos_at: Vec<i64>,
    freq_sin_at: Vec<i64>,
}

impl OpNormTable {
    /// Build the certified table for ring degree `d` at fixed-point scale `q`,
    /// validating that the predicate's `i128` accumulators cannot overflow for
    /// any challenge with `||c||_1 <= max_l1` and any threshold `t <= max_t`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] if `d` is not a positive multiple of
    /// 4, if `q` is outside `1..=56`, or if the worst-case accumulator/square
    /// bounds for `(max_l1, max_t)` do not fit `i128`.
    #[doc(hidden)]
    pub(crate) fn new(d: usize, q: u32, max_l1: u64, max_t: u64) -> Result<Self, AkitaError> {
        if d < 4 || !d.is_multiple_of(4) {
            return Err(AkitaError::InvalidSetup(format!(
                "OpNormTable: ring degree d = {d} must be a positive multiple of 4"
            )));
        }
        if !(1..=56).contains(&q) {
            return Err(AkitaError::InvalidSetup(format!(
                "OpNormTable: scale q = {q} out of supported range 1..=56"
            )));
        }

        let (base_cos, base_sin, eps_root) = build_tables(d, q);

        let max_l1 = i128::from(max_l1);
        let max_t = i128::from(max_t);
        validate_no_overflow(q, eps_root, max_l1, max_t)?;
        validate_i64_accumulators(q, max_l1)?;

        let downcast = |table: &[i128], name: &str| -> Result<Vec<i64>, AkitaError> {
            table
                .iter()
                .map(|&v| {
                    i64::try_from(v).map_err(|_| {
                        AkitaError::InvalidSetup(format!(
                            "OpNormTable: {name} entry does not fit i64 accumulator path"
                        ))
                    })
                })
                .collect()
        };
        let base_cos_i64 = downcast(&base_cos, "base_cos")?;
        let base_sin_i64 = downcast(&base_sin, "base_sin")?;
        let (freq_cos_at, freq_sin_at) =
            op_norm_accumulate::build_freq_at_tables(d, &base_cos_i64, &base_sin_i64);

        Ok(Self {
            d,
            q,
            #[cfg(test)]
            base_cos,
            #[cfg(test)]
            base_sin,
            eps_root,
            max_l1,
            max_t,
            freq_cos_at,
            freq_sin_at,
        })
    }

    /// `i64` downcast of `base_cos[idx]` (certified entry, `|entry| <= 2^q`).
    #[cfg(test)]
    #[inline]
    pub(super) fn base_cos_i64(&self, idx: usize) -> i64 {
        self.base_cos[idx] as i64
    }

    /// `i64` downcast of `base_sin[idx]`.
    #[cfg(test)]
    #[inline]
    pub(super) fn base_sin_i64(&self, idx: usize) -> i64 {
        self.base_sin[idx] as i64
    }

    #[inline]
    pub(super) fn freq_row(&self, pos: usize, half_d: usize) -> (&[i64], &[i64]) {
        let row = pos * half_d;
        (
            &self.freq_cos_at[row..row + half_d],
            &self.freq_sin_at[row..row + half_d],
        )
    }

    /// Certified operator-norm predicate on stack-buffered `(positions, coeffs)`
    /// slices, scanning the `D/2` distinct frequencies (`zeta_{D-1-k} =
    /// conj(zeta_k)`) with the transposed `i64` accumulators.
    pub(crate) fn decide_parts(
        &self,
        positions: &[u32],
        coeffs: &[i8],
        t: u64,
    ) -> Result<Decision, AkitaError> {
        if positions.len() != coeffs.len() {
            return Err(AkitaError::InvalidInput(
                "operator-norm predicate: positions/coeffs length mismatch".to_string(),
            ));
        }
        let mut l1: i128 = 0;
        for (&pos, &coeff) in positions.iter().zip(coeffs.iter()) {
            if pos as usize >= self.d {
                return Err(AkitaError::InvalidInput(format!(
                    "operator-norm predicate: position {pos} out of range for D = {}",
                    self.d
                )));
            }
            l1 += i128::from(coeff.unsigned_abs());
        }
        if l1 > self.max_l1 {
            return Err(AkitaError::InvalidInput(format!(
                "operator-norm predicate: ||c||_1 = {l1} exceeds validated max_l1 = {}",
                self.max_l1
            )));
        }
        // `threshold = t^2 << 2q` is only proven to fit `i128` for `t <= max_t`
        // (the bound `new` validated). Reject larger thresholds rather than risk
        // a debug-overflow panic or a release wrap that would corrupt the
        // comparison.
        if i128::from(t) > self.max_t {
            return Err(AkitaError::InvalidInput(format!(
                "operator-norm predicate: threshold t = {t} exceeds validated max_t = {}",
                self.max_t
            )));
        }

        let r = l1 * self.eps_root;
        let two_r = 2 * r;
        let two_r_sq = 2 * r * r;
        let threshold = (i128::from(t) * i128::from(t)) << (2 * self.q);
        decide_transposed_fused(self, positions, coeffs, r, two_r, two_r_sq, threshold)
    }

    /// Slice form of the production (strict) predicate for stack-buffered rejection draws.
    pub(crate) fn accept_strict_parts(
        &self,
        positions: &[u32],
        coeffs: &[i8],
        t: u64,
    ) -> Result<bool, AkitaError> {
        Ok(self.decide_parts(positions, coeffs, t)? == Decision::Accept)
    }

    /// Prove that every challenge in the certificate's shrunken true-norm
    /// subset is accepted by the strict fixed-point predicate.
    ///
    /// Each real/imaginary accumulator has error at most `r`. The predicate's
    /// upper enclosure for true norm at most `strict_t - h / 2^q` is below
    /// `(strict_t * 2^q)^2` when `h > 2 sqrt(2) r`. The integer inequality
    /// `h^2 > 8 r^2` proves that strict containment without floating point.
    pub(crate) fn strict_threshold_contains_shrunken_subset(
        &self,
        fractional_bits: u32,
        root_coordinate_error_units: u32,
        strict_t: u64,
        rounding_margin_units: u32,
    ) -> bool {
        if self.q != fractional_bits
            || self.eps_root > i128::from(root_coordinate_error_units)
            || i128::from(strict_t) > self.max_t
        {
            return false;
        }
        let Some(r) = self
            .max_l1
            .checked_mul(i128::from(root_coordinate_error_units))
        else {
            return false;
        };
        let margin = i128::from(rounding_margin_units);
        let Some(strict_fixed) = i128::from(strict_t).checked_shl(self.q) else {
            return false;
        };
        if margin == 0 || margin >= strict_fixed {
            return false;
        }
        8i128
            .checked_mul(r)
            .and_then(|value| value.checked_mul(r))
            .zip(margin.checked_mul(margin))
            .is_some_and(|(lhs, rhs)| lhs < rhs)
    }
}

#[allow(clippy::too_many_arguments)]
fn decide_transposed_fused(
    table: &OpNormTable,
    positions: &[u32],
    coeffs: &[i8],
    r: i128,
    two_r: i128,
    two_r_sq: i128,
    threshold: i128,
) -> Result<Decision, AkitaError> {
    let num_freqs = table.d / 2;
    let mut indeterminate = false;
    let mut start_k = 0;
    while start_k < num_freqs {
        let len = (num_freqs - start_k).min(op_norm_accumulate::FUSED_CHUNK_FREQS);
        let (acc_re, acc_im) = op_norm_accumulate::accumulate_transposed_chunk(
            table, positions, coeffs, start_k, len, num_freqs,
        );
        for lane in 0..len {
            let re = i128::from(acc_re[lane]);
            let im = i128::from(acc_im[lane]);
            let abs_re = re.abs();
            let abs_im = im.abs();
            if decide_frequency(abs_re, abs_im, re, im, r, two_r, two_r_sq, threshold)? {
                return Ok(Decision::Reject);
            }
            if !frequency_accepts_upper(abs_re, abs_im, re, im, r, two_r, two_r_sq, threshold) {
                indeterminate = true;
            }
        }
        start_k += len;
    }
    if indeterminate {
        Ok(Decision::Indeterminate)
    } else {
        Ok(Decision::Accept)
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn frequency_accepts_upper(
    abs_re: i128,
    abs_im: i128,
    re: i128,
    im: i128,
    _r: i128,
    two_r: i128,
    two_r_sq: i128,
    threshold: i128,
) -> bool {
    let center = re * re + im * im;
    let upper = center + two_r * (abs_re + abs_im) + two_r_sq;
    upper <= threshold
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn decide_frequency(
    abs_re: i128,
    abs_im: i128,
    re: i128,
    im: i128,
    r: i128,
    two_r: i128,
    two_r_sq: i128,
    threshold: i128,
) -> Result<bool, AkitaError> {
    if frequency_accepts_upper(abs_re, abs_im, re, im, r, two_r, two_r_sq, threshold) {
        return Ok(false);
    }
    let low_re = (abs_re - r).max(0);
    let low_im = (abs_im - r).max(0);
    let lower = low_re * low_re + low_im * low_im;
    Ok(lower > threshold)
}

/// Validate worst-case `i64` accumulator magnitudes for the transposed path.
fn validate_i64_accumulators(q: u32, max_l1: i128) -> Result<(), AkitaError> {
    let overflow = || {
        AkitaError::InvalidSetup(
            "OpNormTable: worst-case i64 accumulator does not fit i64".to_string(),
        )
    };
    let max_acc = max_l1
        .checked_mul(1i128.checked_shl(q).ok_or_else(overflow)?)
        .ok_or_else(overflow)?;
    if max_acc > i128::from(i64::MAX) {
        return Err(overflow());
    }
    Ok(())
}

/// Validate the worst-case `i128` accumulator and square bounds for the
/// predicate, given a sound `eps_root` and the caller's `(max_l1, max_t)`.
fn validate_no_overflow(
    q: u32,
    eps_root: i128,
    max_l1: i128,
    max_t: i128,
) -> Result<(), AkitaError> {
    let overflow = || {
        AkitaError::InvalidSetup(
            "OpNormTable: worst-case accumulator does not fit i128".to_string(),
        )
    };
    // max |R_k| = max_l1 * 2^q  (triangle bound over base_* in [-2^q, 2^q]).
    let max_acc = max_l1
        .checked_mul(1i128.checked_shl(q).ok_or_else(overflow)?)
        .ok_or_else(overflow)?;
    let r = max_l1.checked_mul(eps_root).ok_or_else(overflow)?;
    // upper_k = R^2 + I^2 + 2(|R| + |I|) r + 2 r^2, worst case |R| = |I| = max_acc.
    let center = max_acc.checked_mul(max_acc).ok_or_else(overflow)?;
    let center2 = center.checked_mul(2).ok_or_else(overflow)?;
    let cross = max_acc
        .checked_mul(4)
        .and_then(|v| v.checked_mul(r))
        .ok_or_else(overflow)?;
    let slack = r
        .checked_mul(r)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(overflow)?;
    let upper = center2
        .checked_add(cross)
        .and_then(|v| v.checked_add(slack))
        .ok_or_else(overflow)?;
    // threshold = max_t^2 * 2^{2q}.
    let threshold = max_t
        .checked_mul(max_t)
        .and_then(|v| v.checked_shl(2 * q))
        .ok_or_else(overflow)?;
    let _ = upper.checked_add(threshold).ok_or_else(overflow)?;
    Ok(())
}

/// Build `base_cos`, `base_sin` at scale `2^q` for `t in 0..2d`, returning the
/// sound global error bound `eps_root`.
fn build_tables(d: usize, q: u32) -> (Vec<i128>, Vec<i128>, i128) {
    let two_d = 2 * d;
    let (pi_lo, pi_hi) = pi_enclosure();
    let shift = TRIG_SCALE - q;

    let mut base_cos = vec![0i128; two_d];
    let mut base_sin = vec![0i128; two_d];
    let mut eps_root: i128 = 0;

    for t in 0..two_d {
        let (r, swap, cos_sign, sin_sign) = octant_reduce(t, d);
        // phi = pi * r / d, enclosed at TRIG_SCALE; phi in [0, pi/4].
        let phi_lo = fdiv(pi_lo * r as i128, d as i128);
        let phi_hi = cdiv(pi_hi * r as i128, d as i128);
        let cos_phi = taylor_cos((phi_lo, phi_hi));
        let sin_phi = taylor_sin((phi_lo, phi_hi));
        let (cos_a, sin_a) = if swap {
            (sin_phi, cos_phi)
        } else {
            (cos_phi, sin_phi)
        };
        let cos_a = apply_sign(cos_a, cos_sign);
        let sin_a = apply_sign(sin_a, sin_sign);

        let (c_val, c_err) = downscale_round(cos_a, shift);
        let (s_val, s_err) = downscale_round(sin_a, shift);
        base_cos[t] = c_val;
        base_sin[t] = s_val;
        eps_root = eps_root.max(c_err).max(s_err);
    }
    (base_cos, base_sin, eps_root)
}

/// Reduce angle index `t` (`angle = pi t / d`, period `2d`) to the first octant:
/// returns `(r, swap, cos_sign, sin_sign)` with `r in [0, d/4]` such that
/// `cos(angle) = cos_sign * (swap ? sin : cos)(pi r / d)` and likewise for sin.
fn octant_reduce(t: usize, d: usize) -> (usize, bool, i128, i128) {
    let mut t = t % (2 * d);
    let mut cos_sign = 1i128;
    let mut sin_sign = 1i128;
    if t > d {
        // angle in (pi, 2pi): cos(2pi - a) = cos a, sin(2pi - a) = -sin a.
        t = 2 * d - t;
        sin_sign = -1;
    }
    if t > d / 2 {
        // angle in (pi/2, pi]: cos(pi - a) = -cos a, sin(pi - a) = sin a.
        t = d - t;
        cos_sign = -cos_sign;
    }
    let swap = if t > d / 4 {
        // angle in (pi/4, pi/2]: cos a = sin(pi/2 - a), sin a = cos(pi/2 - a).
        t = d / 2 - t;
        true
    } else {
        false
    };
    (t, swap, cos_sign, sin_sign)
}

/// `cos(phi)` enclosure at [`TRIG_SCALE`] for `phi in [0, pi/4]`.
fn taylor_cos(phi: (i128, i128)) -> (i128, i128) {
    let phi2 = imul(phi, phi);
    let mut sum = (TRIG_ONE, TRIG_ONE);
    let mut pow = (TRIG_ONE, TRIG_ONE); // phi^{2n}
    let mut fact: i128 = 1; // (2n)!
    for n in 1..=NTERMS {
        pow = imul(pow, phi2);
        let two_n = 2 * n as i128;
        fact = fact * (two_n - 1) * two_n;
        let term = (fdiv(pow.0, fact), cdiv(pow.1, fact));
        sum = if n % 2 == 1 {
            isub(sum, term)
        } else {
            iadd(sum, term)
        };
    }
    add_remainder(sum, phi2, pow, fact, 2 * NTERMS + 1)
}

/// `sin(phi)` enclosure at [`TRIG_SCALE`] for `phi in [0, pi/4]`.
fn taylor_sin(phi: (i128, i128)) -> (i128, i128) {
    let phi2 = imul(phi, phi);
    let mut sum = phi; // n = 0 term: phi^1 / 1!
    let mut pow = phi; // phi^{2n+1}
    let mut fact: i128 = 1; // (2n+1)!
    for n in 1..=NTERMS {
        pow = imul(pow, phi2);
        let two_n = 2 * n as i128;
        fact = fact * two_n * (two_n + 1);
        let term = (fdiv(pow.0, fact), cdiv(pow.1, fact));
        sum = if n % 2 == 1 {
            isub(sum, term)
        } else {
            iadd(sum, term)
        };
    }
    add_remainder(sum, phi2, pow, fact, 2 * NTERMS + 2)
}

/// Widen an alternating-series partial sum by the first omitted term magnitude,
/// a sound bound on the truncation error for a decreasing alternating series.
fn add_remainder(
    sum: (i128, i128),
    phi2: (i128, i128),
    pow: (i128, i128),
    fact: i128,
    next_factor: usize,
) -> (i128, i128) {
    let next_pow = imul(pow, phi2);
    let next_fact = fact * (next_factor as i128) * (next_factor as i128 + 1);
    let rem = cdiv(next_pow.1, next_fact);
    (sum.0 - rem, sum.1 + rem)
}

/// Negate / sign-apply an interval (`sign in {-1, +1}`).
fn apply_sign(v: (i128, i128), sign: i128) -> (i128, i128) {
    if sign < 0 {
        (-v.1, -v.0)
    } else {
        v
    }
}

/// Downscale an interval from [`TRIG_SCALE`] to scale `2^(TRIG_SCALE - shift)`
/// (i.e. to the table scale `2^q`), pick the midpoint integer entry, and return
/// `(entry, error_bound)` with `|entry - true_value| <= error_bound`.
fn downscale_round(v: (i128, i128), shift: u32) -> (i128, i128) {
    let lo = fshr(v.0, shift);
    let hi = cshr(v.1, shift);
    let entry = fdiv(lo + hi, 2);
    let err = (entry - lo).max(hi - entry);
    (entry, err)
}

/// Interval multiply at [`TRIG_SCALE`] for non-negative operands.
fn imul(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
    debug_assert!(a.0 >= 0 && b.0 >= 0);
    (fshr(a.0 * b.0, TRIG_SCALE), cshr(a.1 * b.1, TRIG_SCALE))
}

fn iadd(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
    (a.0 + b.0, a.1 + b.1)
}

fn isub(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
    (a.0 - b.1, a.1 - b.0)
}

/// Floor division (`b > 0`).
fn fdiv(a: i128, b: i128) -> i128 {
    a.div_euclid(b)
}

/// Ceil division (`b > 0`).
fn cdiv(a: i128, b: i128) -> i128 {
    -((-a).div_euclid(b))
}

/// Arithmetic floor shift `floor(x / 2^sh)`.
fn fshr(x: i128, sh: u32) -> i128 {
    x >> sh
}

/// Ceil shift `ceil(x / 2^sh)`.
fn cshr(x: i128, sh: u32) -> i128 {
    -((-x) >> sh)
}

/// Rigorous rational enclosure of `pi` at [`TRIG_SCALE`] via Machin's formula
/// `pi = 16 arctan(1/5) - 4 arctan(1/239)`, all in `i128` fixed point.
fn pi_enclosure() -> (i128, i128) {
    let a5 = arctan_inv(5);
    let a239 = arctan_inv(239);
    // pi = 16 a5 - 4 a239 (interval arithmetic, outward).
    let lo = 16 * a5.0 - 4 * a239.1;
    let hi = 16 * a5.1 - 4 * a239.0;
    (lo, hi)
}

/// Enclosure of `arctan(1/x)` at [`TRIG_SCALE`] from its alternating series
/// `sum_n (-1)^n / ((2n+1) x^{2n+1})`. Each scaled term is floor-divided, and
/// the result is widened by (terms summed + 1) ULPs to cover both the per-term
/// truncation and the alternating-series remainder.
fn arctan_inv(x: i128) -> (i128, i128) {
    let mut sum: i128 = 0;
    let mut x_pow = x; // x^{2n+1}
    let x2 = x * x;
    let mut count: i128 = 0;
    let mut n: i128 = 0;
    while let Some(denom) = (2 * n + 1).checked_mul(x_pow) {
        let term = TRIG_ONE / denom;
        if term == 0 {
            break;
        }
        if n % 2 == 0 {
            sum += term;
        } else {
            sum -= term;
        }
        count += 1;
        n += 1;
        match x_pow.checked_mul(x2) {
            Some(v) => x_pow = v,
            None => break,
        }
    }
    let slack = count + 1;
    (sum - slack, sum + slack)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SparseChallenge;
    use std::f64::consts::PI;

    const Q: u32 = 48;

    fn table(d: usize) -> OpNormTable {
        // ||c||_1 <= 2 * D covers every shipping shell at D <= 128; T <= 64.
        OpNormTable::new(d, Q, (2 * d) as u64, 64).unwrap()
    }

    fn decide_ch(
        table: &OpNormTable,
        challenge: &SparseChallenge,
        threshold: u64,
    ) -> Result<Decision, AkitaError> {
        table.decide_parts(&challenge.positions, &challenge.coeffs, threshold)
    }

    /// Test-only naive `k`-outer nested-`i128` reference over `num_freqs`
    /// frequencies, mirroring the production conservative-interval decision but
    /// without the transposed accumulation. Cross-checks the production `D/2`
    /// scan and confirms it equals the full `D`-frequency spectrum.
    fn decide_reference_nested(
        table: &OpNormTable,
        ch: &SparseChallenge,
        t: u64,
        num_freqs: usize,
    ) -> Decision {
        let two_d = 2 * table.d;
        let l1: i128 = ch.coeffs.iter().map(|c| i128::from(c.unsigned_abs())).sum();
        let r = l1 * table.eps_root;
        let two_r = 2 * r;
        let two_r_sq = 2 * r * r;
        let threshold = (i128::from(t) * i128::from(t)) << (2 * table.q);
        let mut indeterminate = false;
        for k in 0..num_freqs {
            let mult = (2 * k + 1) % two_d;
            let mut acc_re: i128 = 0;
            let mut acc_im: i128 = 0;
            for (&pos, &coeff) in ch.positions.iter().zip(ch.coeffs.iter()) {
                let idx = (mult * pos as usize) % two_d;
                let coeff = i128::from(coeff);
                acc_re += coeff * table.base_cos[idx];
                acc_im += coeff * table.base_sin[idx];
            }
            let (abs_re, abs_im) = (acc_re.abs(), acc_im.abs());
            if decide_frequency(
                abs_re, abs_im, acc_re, acc_im, r, two_r, two_r_sq, threshold,
            )
            .unwrap()
            {
                return Decision::Reject;
            }
            if !frequency_accepts_upper(
                abs_re, abs_im, acc_re, acc_im, r, two_r, two_r_sq, threshold,
            ) {
                indeterminate = true;
            }
        }
        if indeterminate {
            Decision::Indeterminate
        } else {
            Decision::Accept
        }
    }

    /// f64 reference, used ONLY to sanity-check the certified integer path.
    fn gamma_sq_f64(d: usize, ch: &SparseChallenge) -> f64 {
        let mut max = 0.0f64;
        for k in 0..d {
            let base = (2 * k + 1) as f64 * PI / d as f64;
            let mut re = 0.0;
            let mut im = 0.0;
            for (&pos, &coeff) in ch.positions.iter().zip(ch.coeffs.iter()) {
                let theta = base * pos as f64;
                re += coeff as f64 * theta.cos();
                im += coeff as f64 * theta.sin();
            }
            max = max.max(re * re + im * im);
        }
        max
    }

    #[test]
    fn pi_encloses_pi() {
        let (lo, hi) = pi_enclosure();
        let pi_scaled = PI * TRIG_ONE as f64;
        assert!((lo as f64) <= pi_scaled && pi_scaled <= (hi as f64));
        // Enclosure should be tight (well under 2^-40 absolute).
        assert!(hi - lo < (1i128 << (TRIG_SCALE - 40)));
    }

    #[test]
    fn table_entries_match_reference_within_eps() {
        for &d in &[4usize, 32, 64, 128] {
            let t = table(d);
            for idx in 0..(2 * d) {
                let angle = PI * idx as f64 / d as f64;
                let ref_cos = (angle.cos() * (1i128 << Q) as f64).round() as i128;
                let ref_sin = (angle.sin() * (1i128 << Q) as f64).round() as i128;
                assert!(
                    (t.base_cos[idx] - ref_cos).abs() <= t.eps_root + 1,
                    "cos mismatch d={d} idx={idx}: {} vs {ref_cos} (eps={})",
                    t.base_cos[idx],
                    t.eps_root
                );
                assert!(
                    (t.base_sin[idx] - ref_sin).abs() <= t.eps_root + 1,
                    "sin mismatch d={d} idx={idx}: {} vs {ref_sin} (eps={})",
                    t.base_sin[idx],
                    t.eps_root
                );
            }
            // eps_root stays tiny: a couple of units in 2^48.
            assert!(
                t.eps_root <= 4,
                "eps_root too large for d={d}: {}",
                t.eps_root
            );
        }
    }

    #[test]
    fn pythagorean_identity_holds() {
        let scale_sq = (1i128 << Q) * (1i128 << Q);
        for &d in &[32usize, 64, 128] {
            let t = table(d);
            for idx in 0..(2 * d) {
                let s = t.base_cos[idx] * t.base_cos[idx] + t.base_sin[idx] * t.base_sin[idx];
                let diff = (s - scale_sq).abs();
                // Each factor is within eps_root; product error is O(eps * 2^q).
                assert!(diff <= (t.eps_root + 2) * (1i128 << (Q + 2)));
            }
        }
    }

    #[test]
    fn unit_coefficient_has_unit_norm() {
        let d = 64;
        let t = table(d);
        let ch = SparseChallenge {
            positions: vec![0].into(),
            coeffs: vec![1].into(),
        };
        // A single coefficient of magnitude m gives gamma = m exactly, so the
        // predicate accepts strictly above, rejects strictly below, and reports
        // the exact tie (T = m) as the indeterminate boundary band.
        assert_eq!(decide_ch(&t, &ch, 0).unwrap(), Decision::Reject);
        assert_eq!(decide_ch(&t, &ch, 1).unwrap(), Decision::Indeterminate);
        assert_eq!(decide_ch(&t, &ch, 2).unwrap(), Decision::Accept);
        assert_eq!(decide_ch(&t, &ch, 5).unwrap(), Decision::Accept);
    }

    #[test]
    fn large_single_coefficient_rejects() {
        let d = 64;
        let t = table(d);
        // gamma = 5 exactly: reject below, indeterminate at the tie, accept above.
        let ch = SparseChallenge {
            positions: vec![3].into(),
            coeffs: vec![5].into(),
        };
        assert_eq!(decide_ch(&t, &ch, 4).unwrap(), Decision::Reject);
        assert_eq!(decide_ch(&t, &ch, 5).unwrap(), Decision::Indeterminate);
        assert_eq!(decide_ch(&t, &ch, 6).unwrap(), Decision::Accept);
    }

    /// Tiny deterministic PRNG (xorshift) so the test is fully reproducible.
    fn rng_next(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    fn random_shell(state: &mut u64, d: usize, mag1: usize, mag2: usize) -> SparseChallenge {
        let total = mag1 + mag2;
        let mut positions = Vec::with_capacity(total);
        while positions.len() < total {
            let p = (rng_next(state) % d as u64) as u32;
            if !positions.contains(&p) {
                positions.push(p);
            }
        }
        let mut coeffs = Vec::with_capacity(total);
        for i in 0..total {
            let sign = if rng_next(state) & 1 == 0 { 1i8 } else { -1 };
            let mag = if i < mag1 { 1 } else { 2 };
            coeffs.push(sign * mag);
        }
        SparseChallenge {
            positions: positions.into(),
            coeffs: coeffs.into(),
        }
    }

    #[test]
    fn production_matches_reference_and_reduced_scan_matches_full_spectrum() {
        let d = 64;
        let t = table(d);
        let mut state = 0x1234_5678_9abc_def0u64;
        for _ in 0..2000 {
            let ch = random_shell(&mut state, d, 31, 11);
            // Production transposed `D/2` scan equals the naive `D/2` reference,
            // which in turn equals the full `D`-frequency spectrum.
            let production = decide_ch(&t, &ch, 16).unwrap();
            let reduced = decide_reference_nested(&t, &ch, 16, d / 2);
            let full = decide_reference_nested(&t, &ch, 16, d);
            assert_eq!(production, reduced, "challenge {ch:?}");
            assert_eq!(reduced, full, "challenge {ch:?}");
        }
    }

    #[test]
    fn accept_implies_sound_and_reject_implies_over() {
        let d = 64;
        let t = table(d);
        let mut state = 0xfeed_face_dead_beefu64;
        let threshold = 16u64;
        let mut accepts = 0;
        let mut rejects = 0;
        for _ in 0..5000 {
            let ch = random_shell(&mut state, d, 31, 11);
            let g2 = gamma_sq_f64(d, &ch);
            match decide_ch(&t, &ch, threshold).unwrap() {
                Decision::Accept => {
                    accepts += 1;
                    assert!(
                        g2 <= (threshold * threshold) as f64 + 1e-3,
                        "unsound accept: gamma^2={g2} for {ch:?}"
                    );
                }
                Decision::Reject => {
                    rejects += 1;
                    assert!(
                        g2 >= (threshold * threshold) as f64 - 1e-3,
                        "unsound reject: gamma^2={g2} for {ch:?}"
                    );
                }
                Decision::Indeterminate => {}
            }
        }
        // The (31, 11) shell at T = 16 should produce a healthy mix.
        assert!(accepts > 0, "no accepts sampled");
        assert!(rejects > 0, "no rejects sampled");
    }

    #[test]
    fn production_strict_threshold_contains_the_certified_subset() {
        for &(d, mag1, mag2, strict, margin, min_numerator, denominator) in
            &[(64, 31, 11, 18, 600, 1, 4), (128, 31, 0, 13, 351, 1, 2)]
        {
            let t = OpNormTable::new(d, Q, (mag1 + 2 * mag2) as u64, strict).unwrap();
            assert!(t.strict_threshold_contains_shrunken_subset(48, 4, strict, margin));
            assert!(!t.strict_threshold_contains_shrunken_subset(47, 4, strict, margin));
            assert!(!t.strict_threshold_contains_shrunken_subset(48, 0, strict, margin));
            assert!(!t.strict_threshold_contains_shrunken_subset(48, 4, strict, 1));

            let mut state = 0xface_cafe_beef_0001u64 ^ d as u64;
            let samples = 10_000u64;
            let mut accepts = 0u64;
            for _ in 0..samples {
                let ch = random_shell(&mut state, d, mag1, mag2);
                if t.accept_strict_parts(&ch.positions, &ch.coeffs, strict)
                    .expect("strict predicate")
                {
                    accepts += 1;
                }
            }
            assert!(
                accepts * denominator > samples * min_numerator,
                "D{d} strict accept rate {accepts}/{samples} unexpectedly low",
            );
        }
    }

    #[test]
    fn d128_production_matches_reference_and_full_spectrum() {
        let d = 128;
        let t = table(d);
        let mut state = 0xd128_cafe_beef_0001u64;
        for _ in 0..1000 {
            let ch = random_shell(&mut state, d, 31, 0);
            let production = decide_ch(&t, &ch, 13).unwrap();
            let reduced = decide_reference_nested(&t, &ch, 13, d / 2);
            let full = decide_reference_nested(&t, &ch, 13, d);
            assert_eq!(production, reduced, "challenge {ch:?}");
            assert_eq!(reduced, full, "challenge {ch:?}");

            let gamma_sq = gamma_sq_f64(d, &ch);
            if production == Decision::Accept {
                assert!(gamma_sq <= 13.0f64.powi(2) + 1e-3);
            }
            if production == Decision::Reject {
                assert!(gamma_sq >= 13.0f64.powi(2) - 1e-3);
            }
        }
    }

    #[test]
    fn rejects_oversized_l1() {
        let d = 32;
        let t = OpNormTable::new(d, Q, 4, 16).unwrap();
        let ch = SparseChallenge {
            positions: vec![0, 1, 2, 3, 4].into(),
            coeffs: vec![1, 1, 1, 1, 1].into(),
        };
        assert!(decide_ch(&t, &ch, 8).is_err());
    }

    #[test]
    fn rejects_oversized_threshold() {
        let d = 32;
        // `||c||_1` stays within `max_l1`, so the threshold guard (not the L1
        // guard) is what rejects a `t` above the validated `max_t = 16`.
        let table = OpNormTable::new(d, Q, 64, 16).unwrap();
        let ch = SparseChallenge {
            positions: vec![0, 1].into(),
            coeffs: vec![1, 1].into(),
        };
        assert!(decide_ch(&table, &ch, 17).is_err());
        assert!(decide_ch(&table, &ch, u64::MAX).is_err());
        // A threshold at the validated boundary is still accepted as input.
        assert!(decide_ch(&table, &ch, 16).is_ok());
    }

    #[test]
    fn construction_rejects_bad_params() {
        assert!(OpNormTable::new(6, Q, 16, 16).is_err()); // not a multiple of 4
        assert!(OpNormTable::new(64, 60, 16, 16).is_err()); // q out of range
        assert!(OpNormTable::new(64, Q, u64::MAX, u64::MAX).is_err()); // overflow
    }
}
