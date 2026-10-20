//! The private vector `s` of the residual relation, as typed segments.
//!
//! Coefficients are stored in native order: a segment consumed by a matrix of
//! ring dimension `d′ = 64m` holds whole `R_{d′}` elements, each viewed as an
//! `m`-tuple over the challenge ring `R = F[X]/(X^64 + 1)` by stride packing
//! (native coefficient `m·i + k` is coefficient `i` of component `k`).

use std::ops::Range;

use jolt_algebra::{CyclotomicRing, Field};
use jolt_crypto::lattice::CHALLENGE_RING_DIM;

use super::WhiteoutError;

/// `[min, max]` of `digits` balanced digits in base `2^log_basis`, each in
/// `[−b/2, b/2 − 1]`: `min = −(b/2)(b^D − 1)/(b − 1)` and
/// `max = (b/2 − 1)(b^D − 1)/(b − 1)`. The one owner of this range: segment
/// alphabets, response digit counts, acceptance boxes and response bounds all
/// use it. `None` if it overflows `i128`.
pub fn balanced_range(log_basis: u32, digits: usize) -> Option<(i128, i128)> {
    if !(1..=8).contains(&log_basis) {
        return None;
    }
    let radix = 1i128 << log_basis;
    let span = radix
        .checked_pow(u32::try_from(digits).ok()?)?
        .checked_sub(1)?
        / (radix - 1);
    Some((-(radix / 2) * span, (radix / 2 - 1) * span))
}

/// What a segment of `s` holds (Whiteout Def. `wo:def:residual`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentRole {
    /// Digits of the residual witness `w_k`.
    Witness,
    /// Inner-commitment digits `t̂_a` of the residual commitment.
    InnerDigits,
    /// Randomness `r_a` of the residual commitment.
    ResidualRandomness,
    /// Radix-16 digits of the regular pads and of the mask `g`.
    PadDigits,
    /// Radix-256 digit polynomials of the operand pads.
    OperandDigits,
    /// Randomness `r_p` of the pad commitment.
    PadRandomness,
}

/// The honest form of every element of a segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentShape {
    Dense,
    /// Every challenge-ring element is a constant polynomial: in native
    /// order, only the first `native_dim / 64` coefficients of each native
    /// element may be nonzero.
    ConstantPoly,
}

/// Whether the Σ-protocol masks a segment with a Gaussian or leaves it to
/// the public response `c·s` (§4.2, "Public coordinates").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentMask {
    Gaussian,
    Public,
}

/// One segment of `s`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment {
    pub role: SegmentRole,
    /// Length in challenge-ring elements.
    pub len: usize,
    /// Ring dimension of the matrix that consumes the segment.
    pub native_dim: usize,
    /// The balanced alphabet `[−2^{log_basis−1}, 2^{log_basis−1} − 1]`.
    pub log_basis: u32,
    pub shape: SegmentShape,
    pub mask: SegmentMask,
}

impl Segment {
    /// Number of field coefficients, `64 · len`.
    pub fn num_coefficients(&self) -> usize {
        self.len * CHALLENGE_RING_DIM
    }

    /// `B = 2^{log_basis − 1}`, the largest digit magnitude: the alphabet is
    /// the one-digit balanced range `[−B, B − 1]`.
    pub fn digit_bound(&self) -> u64 {
        balanced_range(self.log_basis, 1).map_or(0, |(min, _)| min.unsigned_abs() as u64)
    }

    /// The honest bound on `‖s_b‖₂²`: `64·len·B²` for a dense segment and
    /// `len·B²` for constant polynomials. Whiteout's widths use this worst
    /// case (§6.2), and the alphabet and shape check enforces it.
    pub fn honest_l2_sq(&self) -> u128 {
        let elements = match self.shape {
            SegmentShape::Dense => self.num_coefficients(),
            SegmentShape::ConstantPoly => self.len,
        };
        elements as u128 * u128::from(self.digit_bound()).pow(2)
    }

    fn validate(&self, index: usize) -> Result<(), WhiteoutError> {
        let fail = |reason| Err(WhiteoutError::Segment { index, reason });
        if self.len == 0 {
            return fail("empty segment");
        }
        if !(1..=8).contains(&self.log_basis) {
            return fail("digits must be i8 digits of base 2..=256");
        }
        if !self.native_dim.is_power_of_two() || self.native_dim < CHALLENGE_RING_DIM {
            return fail("native ring dimension must be a power-of-two multiple of 64");
        }
        let coefficients =
            self.len
                .checked_mul(CHALLENGE_RING_DIM)
                .ok_or(WhiteoutError::Segment {
                    index,
                    reason: "segment length overflows",
                })?;
        if !coefficients.is_multiple_of(self.native_dim) {
            // Public ranges, and every block of the rejection rule, are
            // unions of whole native elements.
            return fail("segment must hold whole native ring elements");
        }
        Ok(())
    }
}

/// Where one segment sits in `s`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Placed {
    pub(crate) segment: Segment,
    /// First native coefficient.
    pub(crate) coefficient: usize,
    /// First challenge-ring element.
    pub(crate) ring: usize,
}

impl Placed {
    pub(crate) fn coefficients(&self) -> Range<usize> {
        self.coefficient..self.coefficient + self.segment.num_coefficients()
    }

    pub(crate) fn rings(&self) -> Range<usize> {
        self.ring..self.ring + self.segment.len
    }

    pub(crate) fn stride(&self) -> usize {
        self.segment.native_dim / CHALLENGE_RING_DIM
    }
}

/// The ordered segments of `s`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentTable {
    placed: Vec<Placed>,
    num_coefficients: usize,
}

impl SegmentTable {
    pub fn new(segments: Vec<Segment>) -> Result<Self, WhiteoutError> {
        let mut placed = Vec::with_capacity(segments.len());
        let mut coefficient = 0usize;
        for (index, segment) in segments.into_iter().enumerate() {
            segment.validate(index)?;
            placed.push(Placed {
                segment,
                coefficient,
                ring: coefficient / CHALLENGE_RING_DIM,
            });
            coefficient = coefficient.checked_add(segment.num_coefficients()).ok_or(
                WhiteoutError::Segment {
                    index,
                    reason: "total length overflows",
                },
            )?;
        }
        Ok(Self {
            placed,
            num_coefficients: coefficient,
        })
    }

    pub fn segments(&self) -> impl ExactSizeIterator<Item = &Segment> {
        self.placed.iter().map(|placed| &placed.segment)
    }

    pub(crate) fn placed(&self) -> &[Placed] {
        &self.placed
    }

    pub fn num_coefficients(&self) -> usize {
        self.num_coefficients
    }

    /// The only segment with `role`, if exactly one has it.
    pub(crate) fn unique(&self, role: SegmentRole) -> Option<&Placed> {
        let mut found = self
            .placed
            .iter()
            .filter(|placed| placed.segment.role == role);
        match (found.next(), found.next()) {
            (Some(placed), None) => Some(placed),
            _ => None,
        }
    }

    /// Challenge-ring element `ring` of `coefficients` (laid out like `s`):
    /// component `k` of native element `e` of its segment, holding native
    /// coefficients `m·i + k`. Every coefficient is lifted into `F` by
    /// [`lift_ct`], so a secret vector is read in constant time; the segment
    /// lookup depends only on the public index.
    pub(crate) fn ring<F: Field>(
        &self,
        coefficients: &[i64],
        ring: usize,
    ) -> Result<CyclotomicRing<F, CHALLENGE_RING_DIM>, WhiteoutError> {
        let placed = self
            .placed
            .iter()
            .find(|placed| placed.rings().contains(&ring))
            .ok_or(WhiteoutError::InvalidRelation("ring outside s"))?;
        let stride = placed.stride();
        let local = ring - placed.ring;
        let start =
            placed.coefficient + (local / stride) * placed.segment.native_dim + local % stride;
        let values = coefficients
            .get(start..start + (CHALLENGE_RING_DIM - 1) * stride + 1)
            .ok_or(WhiteoutError::WitnessLength {
                expected: self.num_coefficients,
                actual: coefficients.len(),
            })?;
        let mut out = CyclotomicRing::zero();
        for (dst, &value) in out
            .coefficients_mut()
            .iter_mut()
            .zip(values.iter().step_by(stride))
        {
            *dst = lift_ct(value);
        }
        Ok(out)
    }

    /// Release check that every coefficient of `s` lies in its segment's
    /// alphabet, and that constant-polynomial segments vanish off their
    /// constant positions.
    ///
    /// This is the zero-knowledge precondition `‖cs‖_σ ≤ 1/α`: the widths
    /// hold it only for vectors of the declared shapes, and the rejection
    /// rule's per-coordinate shift bound cannot detect a violation. The
    /// check reads every coefficient and branches only on the aggregate, so
    /// its time is independent of `s`.
    #[cfg(feature = "whiteout-prover")]
    pub fn check_private_vector(&self, s: &[i64]) -> Result<(), WhiteoutError> {
        if s.len() != self.num_coefficients {
            return Err(WhiteoutError::WitnessLength {
                expected: self.num_coefficients,
                actual: s.len(),
            });
        }
        let mut outside = 0i64;
        for placed in &self.placed {
            let segment = placed.segment;
            let values = s
                .get(placed.coefficients())
                .ok_or(WhiteoutError::InvalidRelation("segment outside s"))?;
            let (min, max) = balanced_range(segment.log_basis, 1)
                .ok_or(WhiteoutError::InvalidRelation("segment alphabet"))?;
            let (min, max) = (min as i64, max as i64);
            let constant_positions = match segment.shape {
                SegmentShape::Dense => segment.native_dim,
                SegmentShape::ConstantPoly => placed.stride(),
            };
            for element in values.chunks_exact(segment.native_dim) {
                let (free, fixed) = element.split_at(constant_positions);
                // Negative exactly when `x` leaves `[min, max]`.
                outside = free.iter().fold(outside, |acc, &x| {
                    acc | max.wrapping_sub(x) | x.wrapping_sub(min)
                });
                // Negative when a non-constant coefficient is nonzero.
                outside = fixed
                    .iter()
                    .fold(outside, |acc, &x| acc | x | x.wrapping_neg());
            }
        }
        if outside < 0 {
            Err(WhiteoutError::WitnessOutsideAlphabet)
        } else {
            Ok(())
        }
    }
}

/// `x` as a field element, from its magnitude and sign by arithmetic: no
/// branch on `x`, unlike `Field::from_i64`.
pub(crate) fn lift_ct<F: Field>(x: i64) -> F {
    let sign = x >> 63;
    let magnitude = F::from_u64(((x ^ sign).wrapping_sub(sign)) as u64);
    magnitude - (magnitude + magnitude) * F::from_u64((sign & 1) as u64)
}
