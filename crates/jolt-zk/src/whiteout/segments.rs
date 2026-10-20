//! The private vector `s` of the residual relation, as typed segments.
//!
//! Coefficients are stored in native order: a segment consumed by a matrix of
//! ring dimension `d′ = 64m` holds whole `R_{d′}` elements, each viewed as an
//! `m`-tuple over the challenge ring `R = F[X]/(X^64 + 1)` by stride packing
//! (native coefficient `m·i + k` is coefficient `i` of component `k`).

use jolt_algebra::{CyclotomicRing, Field};
use jolt_crypto::lattice::CHALLENGE_RING_DIM;

use super::WhiteoutError;

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

    /// `B = 2^{log_basis − 1}`, the largest digit magnitude.
    pub fn digit_bound(&self) -> u64 {
        1 << (self.log_basis - 1)
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

/// The ordered segments of `s`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentTable {
    segments: Vec<Segment>,
    num_coefficients: usize,
}

impl SegmentTable {
    pub fn new(segments: Vec<Segment>) -> Result<Self, WhiteoutError> {
        for (index, segment) in segments.iter().enumerate() {
            segment.validate(index)?;
        }
        let num_coefficients = segments
            .iter()
            .try_fold(0usize, |total, segment| {
                total.checked_add(segment.num_coefficients())
            })
            .ok_or(WhiteoutError::Segment {
                index: segments.len(),
                reason: "total length overflows",
            })?;
        Ok(Self {
            segments,
            num_coefficients,
        })
    }

    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    pub fn num_coefficients(&self) -> usize {
        self.num_coefficients
    }

    /// Number of challenge-ring elements of `s`.
    pub fn num_rings(&self) -> usize {
        self.num_coefficients / CHALLENGE_RING_DIM
    }

    /// For each segment, its first coefficient and its first
    /// challenge-ring element.
    pub fn starts(&self) -> Vec<(usize, usize)> {
        let mut coefficient = 0;
        self.segments
            .iter()
            .map(|segment| {
                let start = (coefficient, coefficient / CHALLENGE_RING_DIM);
                coefficient += segment.num_coefficients();
                start
            })
            .collect()
    }

    /// The index of the only segment with `role`, if exactly one has it.
    pub fn unique(&self, role: SegmentRole) -> Option<usize> {
        let mut found = self
            .segments
            .iter()
            .enumerate()
            .filter(|(_, segment)| segment.role == role)
            .map(|(index, _)| index);
        match (found.next(), found.next()) {
            (Some(index), None) => Some(index),
            _ => None,
        }
    }

    /// The challenge-ring view of native coefficients `coefficients`, laid
    /// out like `s`: native element `e` of a segment with `native_dim = 64m`
    /// becomes ring elements `m·e .. m·e + m`, component `k` holding native
    /// coefficients `m·i + k`.
    ///
    /// Every coefficient is lifted into `F` from its magnitude and sign by
    /// arithmetic, so the view of a secret vector is computed in constant
    /// time.
    pub fn challenge_view<F: Field>(
        &self,
        coefficients: &[i64],
    ) -> Result<Vec<CyclotomicRing<F, CHALLENGE_RING_DIM>>, WhiteoutError> {
        if coefficients.len() != self.num_coefficients {
            return Err(WhiteoutError::WitnessLength {
                expected: self.num_coefficients,
                actual: coefficients.len(),
            });
        }
        let mut view = Vec::with_capacity(self.num_rings());
        let mut rest = coefficients;
        for segment in &self.segments {
            let (values, tail) = rest.split_at(segment.num_coefficients());
            rest = tail;
            let stride = segment.native_dim / CHALLENGE_RING_DIM;
            for element in values.chunks_exact(segment.native_dim) {
                for component in 0..stride {
                    let ring: Vec<F> = element
                        .iter()
                        .skip(component)
                        .step_by(stride)
                        .map(|&value| lift_ct(value))
                        .collect();
                    view.push(CyclotomicRing::from_slice(&ring));
                }
            }
        }
        Ok(view)
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
        let mut rest = s;
        for segment in &self.segments {
            let (values, tail) = rest.split_at(segment.num_coefficients());
            rest = tail;
            let bound = segment.digit_bound() as i64;
            let constant_positions = match segment.shape {
                SegmentShape::Dense => segment.native_dim,
                SegmentShape::ConstantPoly => segment.native_dim / CHALLENGE_RING_DIM,
            };
            for element in values.chunks_exact(segment.native_dim) {
                let (free, fixed) = element.split_at(constant_positions);
                // Negative exactly when `x` leaves `[−B, B − 1]`.
                outside = free.iter().fold(outside, |acc, &x| {
                    acc | (bound - 1).wrapping_sub(x) | x.wrapping_add(bound)
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
