use akita_algebra::{CrtNttParamSet, CyclotomicCrtNtt, DigitMontLut, MontCoeff};

/// Coefficients per committed row window of a `K = 16` one-hot ring.
pub(super) const WINDOW: usize = 16;
/// Row windows per `D = 512` ring.
pub(super) const ROWS: usize = 32;

/// Forward negacyclic NTT of a binary ring in which every row window sets at
/// most one coefficient, in a fixed permutation of the library's slot order.
///
/// Write `X^(WINDOW*r + v)` for the coefficient of row `r` and hot value `v`.
/// Every NTT slot evaluates at a root `alpha` with `alpha^D = -1`, and the
/// slots split into `ROWS` classes sharing `beta = alpha^WINDOW`. Within a
/// class, `alpha = gamma * omega^t` for a primitive `WINDOW`-th root `omega`,
/// so slot `(class, t)` equals `sum_v omega^(t v) * A[v][class]`, where
/// `A[v][class]` sums `scale * beta^r * gamma^v` over the rows with value `v`.
/// The row terms are tabulated, leaving vector additions per committed row and
/// one `WINDOW`-point DFT across classes.
///
/// The roots, Montgomery scale and slot order are read from the library's own
/// transform of `1` and `X`, so lane `i` agrees modulo `p` with the library's
/// slot [`Self::slot`]`(i)`. Pointwise products are order-agnostic: callers
/// permute their other operand with [`Self::to_lane_order`] and restore
/// library order with [`Self::to_library_order`] before an inverse NTT.
/// Outputs are canonical, within the `(-p, p)` contract of those products.
pub(super) struct OneHotNtt {
    p: u32,
    /// `[v][r]` lanes of `scale * beta_class^r * gamma_class^v`.
    row_terms: Vec<[u32; ROWS]>,
    /// `omega^j` and its Shoup quotient for the DIF twiddles `j < WINDOW / 2`.
    twiddles: [(u32, u32); WINDOW / 2],
    /// Library slot of lane `ROWS * i + class`, where DIF row `i` holds the
    /// bit-reversed DFT index.
    slots: Vec<u16>,
}

impl OneHotNtt {
    pub(super) fn new<const D: usize>(
        params: &CrtNttParamSet<i32, 1, D>,
        lut: &DigitMontLut<i32, 1>,
    ) -> Option<Self> {
        let p = u32::try_from(params.primes[0].p).ok()?;
        if D != WINDOW * ROWS || p >= 1 << 30 {
            return None;
        }
        let image = |shift: usize| {
            let mut digits = [0i8; D];
            digits[shift] = 1;
            let mut ring = CyclotomicCrtNtt::<i32, 1, D>::zero();
            ring.assign_i8_with_lut(&digits, params, lut);
            ring.limbs[0].map(|value| value.raw().rem_euclid(p as i32) as u32)
        };
        let one = image(0);
        let x = image(1);
        let scale = one[0];
        if scale == 0 || one.iter().any(|&value| value != scale) {
            return None;
        }
        let scale_inv = inverse(scale, p);
        let alphas = x.map(|value| mul(value, scale_inv, p));

        let mut class_of = [usize::MAX; D];
        let mut betas = Vec::with_capacity(ROWS);
        let mut gammas = Vec::with_capacity(ROWS);
        for (slot, &alpha) in alphas.iter().enumerate() {
            let beta = pow(alpha, WINDOW as u64, p);
            let class = betas
                .iter()
                .position(|&known| known == beta)
                .unwrap_or_else(|| {
                    betas.push(beta);
                    gammas.push(alpha);
                    betas.len() - 1
                });
            if class >= ROWS {
                return None;
            }
            class_of[slot] = class;
        }
        if betas.len() != ROWS {
            return None;
        }

        let omega = (0..D)
            .filter(|&slot| class_of[slot] == 0)
            .map(|slot| mul(alphas[slot], inverse(gammas[0], p), p))
            .find(|&ratio| pow(ratio, (WINDOW / 2) as u64, p) != 1)?;
        if pow(omega, WINDOW as u64, p) != 1 {
            return None;
        }
        let omega_powers: Vec<u32> = (0..WINDOW as u64).map(|j| pow(omega, j, p)).collect();

        let mut slots = vec![u16::MAX; D];
        for (slot, &alpha) in alphas.iter().enumerate() {
            let class = class_of[slot];
            let ratio = mul(alpha, inverse(gammas[class], p), p);
            let t = omega_powers.iter().position(|&power| power == ratio)?;
            let lane = &mut slots[ROWS * bit_reverse(t) + class];
            if *lane != u16::MAX {
                return None;
            }
            *lane = u16::try_from(slot).ok()?;
        }

        let mut row_terms = vec![[0u32; ROWS]; WINDOW * ROWS];
        for (class, (&beta, &gamma)) in betas.iter().zip(&gammas).enumerate() {
            let mut gamma_power = scale;
            for v in 0..WINDOW {
                let mut term = gamma_power;
                for r in 0..ROWS {
                    row_terms[v * ROWS + r][class] = term;
                    term = mul(term, beta, p);
                }
                gamma_power = mul(gamma_power, gamma, p);
            }
        }
        let twiddles = std::array::from_fn(|j| (omega_powers[j], shoup(omega_powers[j], p)));

        Some(Self {
            p,
            row_terms,
            twiddles,
            slots,
        })
    }

    /// Transform the binary ring with ones at `shifts`, each `WINDOW * row +
    /// value` with at most one shift per row, into `out` in lane order.
    pub(super) fn transform<const D: usize>(
        &self,
        shifts: &[usize],
        out: &mut [MontCoeff<i32>; D],
    ) {
        // Every lane stays in [0, 2p): additions reach at most 4p < 2^32,
        // Shoup products accept any u32 and return [0, 2p).
        let mut lanes = [[0u32; ROWS]; WINDOW];
        for &shift in shifts {
            let value = shift % WINDOW;
            self.add_terms(
                &mut lanes[value],
                &self.row_terms[value * ROWS + shift / WINDOW],
            );
        }
        for half in [8, 4, 2] {
            let stride = WINDOW / (2 * half);
            for start in (0..WINDOW).step_by(2 * half) {
                let (lo, hi) = lanes[start..start + 2 * half].split_at_mut(half);
                for (j, (a, b)) in lo.iter_mut().zip(hi).enumerate() {
                    self.butterfly(a, b, self.twiddles[j * stride]);
                }
            }
        }
        for (pair, out) in lanes.chunks_exact(2).zip(out.chunks_exact_mut(2 * ROWS)) {
            self.final_butterfly(&pair[0], &pair[1], out);
        }
    }

    // The lane helpers stay out of line and index fixed-length arrays:
    // inlined into `transform`, LLVM fully unrolls their 32-lane loops before
    // vectorizing and emits scalar code.

    #[inline(never)]
    fn add_terms(&self, lanes: &mut [u32; ROWS], terms: &[u32; ROWS]) {
        let p2 = 2 * self.p;
        for lane in 0..ROWS {
            lanes[lane] = reduce(lanes[lane] + terms[lane], p2);
        }
    }

    /// DIF butterfly `(a + b, (a - b) * w)`; the unit twiddle skips the
    /// product.
    #[inline(never)]
    fn butterfly(&self, a: &mut [u32; ROWS], b: &mut [u32; ROWS], (w, w_shoup): (u32, u32)) {
        let (p, p2) = (self.p, 2 * self.p);
        if w == 1 {
            for lane in 0..ROWS {
                let (x, y) = (a[lane], b[lane]);
                a[lane] = reduce(x + y, p2);
                b[lane] = reduce(x + p2 - y, p2);
            }
            return;
        }
        for lane in 0..ROWS {
            let (x, y) = (a[lane], b[lane]);
            let difference = x + p2 - y;
            a[lane] = reduce(x + y, p2);
            let quotient = ((u64::from(difference) * u64::from(w_shoup)) >> 32) as u32;
            b[lane] = difference
                .wrapping_mul(w)
                .wrapping_sub(quotient.wrapping_mul(p));
        }
    }

    /// The last DIF stage, whose twiddles are all one, written canonically.
    #[inline(never)]
    fn final_butterfly(&self, a: &[u32; ROWS], b: &[u32; ROWS], out: &mut [MontCoeff<i32>]) {
        let (p, p2) = (self.p, 2 * self.p);
        let (out_a, out_b) = out.split_at_mut(ROWS);
        for lane in 0..ROWS {
            let (x, y) = (a[lane], b[lane]);
            out_a[lane] = MontCoeff::from_raw(reduce(reduce(x + y, p2), p) as i32);
            out_b[lane] = MontCoeff::from_raw(reduce(reduce(x + p2 - y, p2), p) as i32);
        }
    }

    /// Permute a library-order NTT ring into lane order.
    pub(super) fn to_lane_order<const D: usize>(&self, ring: &mut [MontCoeff<i32>; D]) {
        let library = *ring;
        for (lane, &slot) in ring.iter_mut().zip(&self.slots) {
            *lane = library[usize::from(slot)];
        }
    }

    /// Permute a lane-order NTT ring back into library order.
    pub(super) fn to_library_order<const D: usize>(&self, ring: &mut [MontCoeff<i32>; D]) {
        let lanes = *ring;
        for (&value, &slot) in lanes.iter().zip(&self.slots) {
            ring[usize::from(slot)] = value;
        }
    }
}

#[inline(always)]
fn reduce(value: u32, bound: u32) -> u32 {
    value.min(value.wrapping_sub(bound))
}

fn bit_reverse(index: usize) -> usize {
    index.reverse_bits() >> (usize::BITS - WINDOW.trailing_zeros())
}

fn shoup(w: u32, p: u32) -> u32 {
    ((u64::from(w) << 32) / u64::from(p)) as u32
}

fn mul(a: u32, b: u32, p: u32) -> u32 {
    (u64::from(a) * u64::from(b) % u64::from(p)) as u32
}

fn pow(mut base: u32, mut exponent: u64, p: u32) -> u32 {
    let mut result = 1;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = mul(result, base, p);
        }
        base = mul(base, base, p);
        exponent >>= 1;
    }
    result
}

fn inverse(value: u32, p: u32) -> u32 {
    pow(value, u64::from(p - 2), p)
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "test parameters are valid")]

    use super::*;
    use akita_algebra::tables::q128_primes;

    const D: usize = WINDOW * ROWS;

    fn library(
        shifts: &[usize],
        params: &CrtNttParamSet<i32, 1, D>,
        lut: &DigitMontLut<i32, 1>,
    ) -> [i32; D] {
        let mut digits = [0i8; D];
        for &shift in shifts {
            digits[shift] = 1;
        }
        let mut ring = CyclotomicCrtNtt::<i32, 1, D>::zero();
        ring.assign_i8_with_lut(&digits, params, lut);
        ring.limbs[0].map(MontCoeff::raw)
    }

    #[test]
    fn matches_library_transform_modulo_prime() {
        let params = CrtNttParamSet::new([q128_primes()[0]]);
        let lut = DigitMontLut::new_with_digit_bound(&params, 2);
        let ntt = OneHotNtt::new(&params, &lut).unwrap();
        let p = params.primes[0].p;
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |bound: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % bound as u64) as usize
        };
        let mut cases: Vec<Vec<usize>> = vec![
            vec![],
            (0..ROWS).map(|row| row * WINDOW).collect(),
            (0..ROWS).map(|row| row * WINDOW + WINDOW - 1).collect(),
            vec![D - 1],
        ];
        for _ in 0..64 {
            let mut shifts = Vec::new();
            for row in 0..ROWS {
                if next(4) != 0 {
                    shifts.push(row * WINDOW + next(WINDOW));
                }
            }
            cases.push(shifts);
        }
        for shifts in cases {
            let expected = library(&shifts, &params, &lut);
            let mut actual = [MontCoeff::from_raw(0); D];
            ntt.transform(&shifts, &mut actual);
            for (lane, &value) in actual.iter().enumerate() {
                assert!(
                    (0..p).contains(&value.raw()),
                    "lane {lane} is not canonical"
                );
            }
            ntt.to_library_order(&mut actual);
            for (slot, (&expected, actual)) in expected.iter().zip(actual).enumerate() {
                let actual = actual.raw();
                assert_eq!(
                    (expected - actual).rem_euclid(p),
                    0,
                    "slot {slot} for {shifts:?}"
                );
            }
        }
    }
}
