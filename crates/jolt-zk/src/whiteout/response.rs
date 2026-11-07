//! The response statement (Whiteout §4.3) and the `ResponseProver` seam.
//!
//! After an accepted attempt, the verifier holds `(c, ρ, cm(t_D))` and the
//! compressed commitment images, but never `z`, `V` or `θ`. The response
//! proof certifies the vector `e = (G_z⁻¹(z), G⁻¹(V), G⁻¹(θ), ξ)`, with `ξ`
//! the compression witnesses of the compressed targets of `N` and of `t_D`:
//!
//! 1. every digit of `e` lies in the balanced radix-16 alphabet;
//! 2. `t_D = A_D G⁻¹(V)`, and every `ξ` opens its payload under the response
//!    prover's compression map (Whiteout §2);
//! 3. `N z = f + c t`, on Gaussian and public coordinates alike;
//!
//! where every compressed image (`t_D` and the compressed `t`) is the one
//! its `ξ` reconstructs;
//! 4. `Q_μ(c, z) + L z = E_0 + c E_1 + c² h`, with `θ` the quotient of
//!    `Σ μ_i [X_i(c,z)][Y_i(c,z)]` by `X^64 + 1`;
//! 5. `ct(h) = 0`.
//!
//! Akita proves the statement (W5); [`ResponseStatement::check_witness`] is
//! its definition.

#[cfg(feature = "whiteout-prover")]
use jolt_algebra::balanced_decompose_coefficients_pow2_i8_into;
#[cfg(feature = "whiteout-prover")]
use jolt_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use rayon::prelude::*;
use std::ops::Range;

use jolt_algebra::{lift_ct, CyclotomicRing, JoltField};
use jolt_crypto::challenge::SparseChallenge;
use jolt_crypto::lattice::CHALLENGE_RING_DIM;
use jolt_transcript::{ProverState, VerifierState};
use spongefish::DuplexSpongeInterface;

use super::challenge::{challenge_ring, challenge_times_ring, challenge_times_ring_transpose};
use super::compression::ImageCompression;
use super::lift::{AggregatedRelation, ChallengeRing, LiftedRelation, ProductAdjointWeights};
use super::linear::BlockTarget;
use super::params::RESPONSE_LOG_BASIS;
use super::segments::balanced_range;
use super::sigma::SigmaSetup;
use super::WhiteoutError;
use crate::masking::ZkError;
use crate::secret::{wipe, SecretVec};

/// Radix-16 digits of a field element of `V` or `θ`.
pub const FIELD_LEVELS: usize = 32;
/// Coefficients of `θ`, which has degree below 63.
pub(crate) const THETA_COEFFICIENTS: usize = CHALLENGE_RING_DIM - 1;

/// One segment of `e`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponseSegment {
    /// The response on segment `segment` of `s`: `digits` radix-16 digits
    /// per coefficient, digit plane by digit plane.
    Response {
        segment: usize,
        coefficients: usize,
        digits: usize,
    },
    /// `G⁻¹(V)`, as `t_D` commits it: ring by ring, 32 digit planes each.
    V { coefficients: usize },
    /// `G⁻¹(θ)`: 32 digit planes of the 63 coefficients.
    Theta,
    /// The compression witness of compressed image `image`, of
    /// `coefficients` coefficients: the compressed targets of `N` in block
    /// order, then `t_D`.
    Compression {
        image: usize,
        coefficients: usize,
        digits: usize,
    },
}

impl ResponseSegment {
    /// How this segment stores its coefficients' digits; `None` for a
    /// compression witness, whose layout is the compression map's.
    pub(crate) fn planes(&self) -> Option<PlaneLayout> {
        match *self {
            Self::Response {
                coefficients,
                digits,
                ..
            } => Some(PlaneLayout {
                width: coefficients,
                planes: digits,
                coefficients,
            }),
            Self::V { coefficients } => Some(PlaneLayout {
                width: CHALLENGE_RING_DIM,
                planes: FIELD_LEVELS,
                coefficients,
            }),
            Self::Theta => Some(PlaneLayout::single(THETA_COEFFICIENTS, FIELD_LEVELS)),
            Self::Compression { .. } => None,
        }
    }

    fn digit_shape(&self) -> (usize, usize) {
        match *self {
            Self::Response {
                coefficients,
                digits,
                ..
            } => (coefficients, digits),
            Self::V { coefficients } => (coefficients, FIELD_LEVELS),
            Self::Theta => (THETA_COEFFICIENTS, FIELD_LEVELS),
            Self::Compression { digits, .. } => (digits, 1),
        }
    }

    /// The number of digits of `e` in this segment.
    pub fn num_digits(&self) -> usize {
        let (coefficients, digits) = self.digit_shape();
        coefficients * digits
    }
}

/// The position law of a planar segment of `e`: its `coefficients` are cut
/// into blocks of `width`, and each block is stored as `planes`
/// consecutive digit planes of `width` digits, plane `p` holding every
/// coefficient's radix-16 digit `p`. Every reader and writer of `e`'s
/// digits goes through [`Self::index`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlaneLayout {
    pub(crate) width: usize,
    pub(crate) planes: usize,
    pub(crate) coefficients: usize,
}

impl PlaneLayout {
    /// One block of `coefficients`.
    pub(crate) fn single(coefficients: usize, planes: usize) -> Self {
        Self {
            width: coefficients,
            planes,
            coefficients,
        }
    }

    /// The digit of plane `plane` of coefficient `coefficient`, counted from
    /// the segment's first digit.
    pub(crate) fn index(self, coefficient: usize, plane: usize) -> usize {
        let width = self.width.max(1);
        (coefficient / width) * width * self.planes + plane * width + coefficient % width
    }

    /// Every coefficient `Σ_p radix^p·digit[index(k, p)]`, over `F`.
    pub(crate) fn recompose<F: JoltField>(self, digits: &[i8]) -> Result<Vec<F>, WhiteoutError> {
        let radix = F::from_u64(1 << RESPONSE_LOG_BASIS);
        (0..self.coefficients)
            .map(|k| {
                (0..self.planes).rev().try_fold(F::zero(), |value, plane| {
                    digits
                        .get(self.index(k, plane))
                        .map(|&digit| value * radix + lift_ct::<F>(i64::from(digit)))
                        .ok_or(WhiteoutError::InvalidRelation(
                            "digits shorter than their planes",
                        ))
                })
            })
            .collect()
    }

    /// Writes every coefficient `Σ_p 16^p·digit[index(k, p)]`, as an
    /// integer, to `out`.
    pub(crate) fn recompose_integers(
        self,
        digits: &[i8],
        out: &mut [i64],
    ) -> Result<(), WhiteoutError> {
        if out.len() != self.coefficients || digits.len() < self.coefficients * self.planes {
            return Err(WhiteoutError::InvalidRelation(
                "digits shorter than their planes",
            ));
        }
        out.par_iter_mut().enumerate().for_each(|(k, dst)| {
            *dst = (0..self.planes).rev().fold(0i64, |value, plane| {
                value * (1 << RESPONSE_LOG_BASIS)
                    + digits
                        .get(self.index(k, plane))
                        .map_or(0, |&digit| i64::from(digit))
            });
        });
        Ok(())
    }

    /// The transpose of [`Self::recompose`]: adds `radix^p·weights[k]` to
    /// `out[index(k, p)]`.
    pub(crate) fn transpose<F: JoltField>(
        self,
        weights: &[F],
        out: &mut [F],
    ) -> Result<(), WhiteoutError> {
        if weights.len() != self.coefficients || out.len() != self.coefficients * self.planes {
            return Err(WhiteoutError::InvalidRelation("plane weights length"));
        }
        let radix = F::from_u64(1 << RESPONSE_LOG_BASIS);
        let powers: Vec<F> = std::iter::successors(Some(F::one()), |power| Some(*power * radix))
            .take(self.planes)
            .collect();
        let width = self.width.max(1);
        // Run `c` of `width` digits is plane `c mod planes` of block
        // `c / planes`: `index(block·width + k, plane) = c·width + k`.
        out.par_chunks_mut(width)
            .enumerate()
            .for_each(|(run, dst)| {
                let block = run / self.planes;
                let power = powers.get(run % self.planes).copied().unwrap_or_default();
                let weights = weights.get(block * width..).unwrap_or_default();
                dst.par_chunks_mut(PLANE_GRAIN)
                    .zip(weights.par_chunks(PLANE_GRAIN))
                    .for_each(|(dst, weights)| {
                        for (dst, weight) in dst.iter_mut().zip(weights) {
                            *dst += *weight * power;
                        }
                    });
            });
        Ok(())
    }

    /// Writes the balanced radix-16 digits of `values` (`planes` per field
    /// element) at their places. The decomposition kernel writes one block
    /// plane-major, which is [`Self::index`] within a block.
    #[cfg(feature = "whiteout-prover")]
    pub(crate) fn field_digits<F: JoltField>(
        self,
        values: &[F],
        digits: &mut [i8],
    ) -> Result<(), WhiteoutError> {
        if values.len() != self.coefficients || digits.len() != self.coefficients * self.planes {
            return Err(WhiteoutError::InvariantViolation("field digit count"));
        }
        let params = BalancedDecomposePow2Params::<F>::new(self.planes, RESPONSE_LOG_BASIS);
        for (block, out) in values
            .chunks(self.width.max(1))
            .zip(digits.chunks_mut(self.width.max(1) * self.planes))
        {
            balanced_decompose_coefficients_pow2_i8_into(block, out, &params);
        }
        Ok(())
    }

    /// Writes the balanced radix-16 digits of the integers `values`
    /// (`planes` each) at their places.
    #[cfg(feature = "whiteout-prover")]
    pub(crate) fn integer_digits(
        self,
        values: &[i64],
        digits: &mut [i8],
    ) -> Result<(), WhiteoutError> {
        let radix = 1i128 << RESPONSE_LOG_BASIS;
        let (min, top) = balanced_range(RESPONSE_LOG_BASIS, self.planes)
            .ok_or(WhiteoutError::InvariantViolation("response digit range"))?;
        let offset = -min;
        if values.len() != self.coefficients || digits.len() != values.len() * self.planes {
            return Err(WhiteoutError::InvariantViolation("response digit count"));
        }
        for (index, &value) in values.iter().enumerate() {
            let value = i128::from(value);
            if !(-offset..=top).contains(&value) {
                return Err(WhiteoutError::InvariantViolation(
                    "response coefficient outside its digit range",
                ));
            }
            let mut shifted = value + offset;
            for plane in 0..self.planes {
                if let Some(digit) = digits.get_mut(self.index(index, plane)) {
                    *digit = ((shifted % radix) - radix / 2) as i8;
                }
                shifted /= radix;
            }
        }
        Ok(())
    }
}

/// Digits per parallel task when reading or writing a plane.
const PLANE_GRAIN: usize = 1 << 14;

/// [`ResponseStatement::weigh`]'s `a`, one weight per digit of `e` in
/// layout order, held factored: a response segment of `n` coefficients
/// keeps its `n` coefficient weights `w` once, digit `p·n + k` weighing
/// `16^p·w_k` ([`PlaneLayout::transpose`] with one block); every other
/// segment is held densely. [`Self::fill`] writes any range of `a`.
pub struct ResponseWeights<F> {
    /// Each segment's first digit and weights, in layout order.
    segments: Vec<(usize, SegmentWeights<F>)>,
    /// `16^p` for every digit plane of the planar segments.
    powers: Vec<F>,
    len: usize,
}

enum SegmentWeights<F> {
    /// Digit `p·weights.len() + k` weighs `16^p·weights[k]`, for `p` below
    /// `planes`.
    Planar {
        weights: Vec<F>,
        planes: usize,
    },
    Dense(Vec<F>),
}

impl<F> SegmentWeights<F> {
    /// The segment of `planes`, a single block, with coefficient weights
    /// `weights`.
    fn planar(planes: PlaneLayout, weights: Vec<F>) -> Result<Self, WhiteoutError> {
        if planes.width != planes.coefficients || weights.len() != planes.coefficients {
            return Err(WhiteoutError::InvalidRelation("plane weights length"));
        }
        Ok(Self::Planar {
            weights,
            planes: planes.planes,
        })
    }

    fn len(&self) -> usize {
        match self {
            Self::Planar { weights, planes } => weights.len() * planes,
            Self::Dense(values) => values.len(),
        }
    }
}

impl<F: JoltField> ResponseWeights<F> {
    fn new(weights: Vec<SegmentWeights<F>>) -> Result<Self, WhiteoutError> {
        let mut segments = Vec::with_capacity(weights.len());
        let mut len = 0usize;
        let mut planes = 0;
        for segment in weights {
            if let SegmentWeights::Planar { planes: count, .. } = segment {
                planes = planes.max(count);
            }
            let start = len;
            len = len
                .checked_add(segment.len())
                .ok_or(WhiteoutError::Overflow)?;
            segments.push((start, segment));
        }
        let radix = F::from_u64(1 << RESPONSE_LOG_BASIS);
        let powers = std::iter::successors(Some(F::one()), |power| Some(*power * radix))
            .take(planes)
            .collect();
        Ok(Self {
            segments,
            powers,
            len,
        })
    }

    /// The number of digits of `e`.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether `e` has no digits.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Writes `a[start..start + out.len()]`, each weight equal to the one
    /// [`PlaneLayout::transpose`] writes.
    pub fn fill(&self, start: usize, out: &mut [F]) -> Result<(), WhiteoutError> {
        let end = start
            .checked_add(out.len())
            .filter(|&end| end <= self.len)
            .ok_or(WhiteoutError::InvalidRelation("row weights range"))?;
        for (offset, segment) in &self.segments {
            let (lo, hi) = (start.max(*offset), end.min(offset + segment.len()));
            if lo >= hi {
                continue;
            }
            let dst = out
                .get_mut(lo - start..hi - start)
                .ok_or(WhiteoutError::InvariantViolation("row weights range"))?;
            match segment {
                SegmentWeights::Dense(values) => dst.copy_from_slice(
                    values
                        .get(lo - offset..hi - offset)
                        .ok_or(WhiteoutError::InvariantViolation("row weights range"))?,
                ),
                SegmentWeights::Planar { weights, .. } => {
                    // Digits `lo..hi` of the segment, run by run: run `p`
                    // is plane `p`, `weights.len()` digits long.
                    let n = weights.len();
                    let mut digit = lo - offset;
                    let mut dst = dst;
                    while !dst.is_empty() {
                        let (plane, k) = (digit / n, digit % n);
                        let power = *self
                            .powers
                            .get(plane)
                            .ok_or(WhiteoutError::InvariantViolation("digit plane"))?;
                        let run = (n - k).min(dst.len());
                        let (head, tail) = dst.split_at_mut(run);
                        let weights = weights
                            .get(k..k + run)
                            .ok_or(WhiteoutError::InvariantViolation("plane weights"))?;
                        for (dst, weight) in head.iter_mut().zip(weights) {
                            *dst = *weight * power;
                        }
                        dst = tail;
                        digit += run;
                    }
                }
            }
        }
        Ok(())
    }
}

/// The logical layout of `e`, from the public setup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResponseLayout {
    pub segments: Vec<ResponseSegment>,
}

impl ResponseLayout {
    pub fn checked_len(&self) -> Result<usize, WhiteoutError> {
        self.segments.iter().try_fold(0usize, |sum, segment| {
            let (coefficients, digits) = segment.digit_shape();
            let len = coefficients
                .checked_mul(digits)
                .ok_or(WhiteoutError::Overflow)?;
            sum.checked_add(len).ok_or(WhiteoutError::Overflow)
        })
    }

    pub fn len(&self) -> usize {
        self.segments.iter().map(ResponseSegment::num_digits).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// `e` in digit form, in [`ResponseLayout`] order. Zeroed when dropped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResponseWitness {
    pub digits: Vec<i8>,
}

// Hand-written: the derived `ZeroizeOnDrop` wipes a `Vec` twice, byte by
// byte, and `e` is the largest buffer of a proof.
impl Drop for ResponseWitness {
    fn drop(&mut self) {
        wipe(&mut self.digits);
    }
}

/// What the response proof certifies.
pub struct ResponseStatement<'a, F: JoltField, const D: usize> {
    pub(crate) setup: &'a SigmaSetup<F, D>,
    pub(crate) relation: &'a LiftedRelation<F>,
    /// The response prover's compression map.
    pub(crate) compression: &'a dyn ImageCompression<F>,
    pub(crate) challenge: SparseChallenge,
    /// The aggregation challenges; `μ` follows from the relation's labels.
    pub(crate) rho: Vec<F>,
    /// The relation aggregated under `rho`, which the rows, their constant
    /// and their weight at a point all read.
    pub(crate) aggregated: AggregatedRelation<'a, F>,
    /// `cm(t_D)`.
    pub(crate) v_payload: Vec<F>,
}

/// A compressed image of the statement: where its witness sits in `e`,
/// its length, and its payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompressedTarget<'a, F> {
    pub digits: Range<usize>,
    pub image_len: usize,
    pub payload: &'a [F],
}

/// Verifies the response proof. The Σ-protocol absorbed the statement's
/// compressed targets of `N` before `ρ` ([`super::verify_sigma`]); the
/// response proof does not absorb them again.
///
/// The response proof runs in the proof's NARG string, on the concrete
/// spongefish states of the implementor's sponge, as Akita's opening does.
pub trait ResponseVerifier<F: JoltField, const D: usize> {
    /// The transcript sponge the response proof runs on.
    type Sponge: DuplexSpongeInterface<U = u8>;

    /// The compression map of the statement's images.
    fn compression(&self) -> &dyn ImageCompression<F>;

    /// Verifies the response proof of `statement`, reading it from
    /// `transcript`.
    ///
    /// # Errors
    ///
    /// [`ZkError`] if the proof is rejected.
    fn verify_response(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        transcript: &mut VerifierState<'_, Self::Sponge>,
    ) -> Result<(), ZkError>;
}

// TODO(akita#120): make `ResponseProver: ResponseVerifier` and drop its
// verifying half once Akita implements `ResponseVerifier` for its prover.
impl<F: JoltField, const D: usize, P: ResponseProver<F, D>> ResponseVerifier<F, D> for P {
    type Sponge = P::Sponge;

    fn compression(&self) -> &dyn ImageCompression<F> {
        ResponseProver::compression(self)
    }

    fn verify_response(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        transcript: &mut VerifierState<'_, Self::Sponge>,
    ) -> Result<(), ZkError> {
        ResponseProver::verify_response(self, statement, transcript)
    }
}

/// Proves the response statement. Akita implements it (W5).
pub trait ResponseProver<F: JoltField, const D: usize> {
    /// The transcript sponge the response proof runs on.
    type Sponge: DuplexSpongeInterface<U = u8>;

    /// The compression map of the statement's images.
    fn compression(&self) -> &dyn ImageCompression<F>;

    /// Proves `statement` for `witness`, writing the proof to `transcript`.
    ///
    /// # Errors
    ///
    /// [`ZkError`] if the prover fails.
    fn prove_response(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        witness: &ResponseWitness,
        transcript: &mut ProverState<Self::Sponge>,
    ) -> Result<(), ZkError>;

    /// Verifies the response proof of `statement`, reading it from
    /// `transcript`.
    ///
    /// # Errors
    ///
    /// [`ZkError`] if the proof is rejected.
    fn verify_response(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        transcript: &mut VerifierState<'_, Self::Sponge>,
    ) -> Result<(), ZkError>;
}

impl<'a, F: JoltField, const D: usize> ResponseStatement<'a, F, D> {
    /// The statement of `relation` under the aggregation challenges `rho`.
    ///
    /// # Errors
    ///
    /// [`WhiteoutError`] if `rho` does not match the relation's labels.
    pub(crate) fn new(
        setup: &'a SigmaSetup<F, D>,
        relation: &'a LiftedRelation<F>,
        compression: &'a dyn ImageCompression<F>,
        challenge: SparseChallenge,
        rho: Vec<F>,
        v_payload: Vec<F>,
    ) -> Result<Self, WhiteoutError> {
        let aggregated = relation.aggregate(&rho)?;
        Ok(Self {
            setup,
            relation,
            compression,
            challenge,
            rho,
            aggregated,
            v_payload,
        })
    }
}

impl<F: JoltField, const D: usize> ResponseStatement<'_, F, D> {
    pub fn layout(&self) -> Result<ResponseLayout, WhiteoutError> {
        self.setup.response_layout(self.compression)
    }

    /// `log₂` of `e`'s digit radix: Item 1 puts every digit in the
    /// one-digit balanced alphabet of this base.
    pub fn digit_log_basis(&self) -> u32 {
        RESPONSE_LOG_BASIS
    }

    /// The Σ-protocol challenge `c`.
    pub fn challenge(&self) -> &SparseChallenge {
        &self.challenge
    }

    /// The aggregation challenges `ρ`.
    pub fn rho(&self) -> &[F] {
        &self.rho
    }

    /// `cm(t_D)`, the compressed commitment to `V`.
    pub fn v_payload(&self) -> &[F] {
        &self.v_payload
    }

    /// The compressed images, in layout order.
    pub fn compressed_targets(&self) -> Result<Vec<CompressedTarget<'_, F>>, WhiteoutError> {
        let layout = self.layout()?;
        let payloads: Vec<(usize, &[F])> = self
            .setup
            .compressed_targets()
            .chain(std::iter::once((
                self.setup.v_image_len(),
                self.v_payload.as_slice(),
            )))
            .collect();
        let mut offset = 0;
        let mut targets = Vec::with_capacity(payloads.len());
        for segment in &layout.segments {
            let len = segment.num_digits();
            if let ResponseSegment::Compression {
                image,
                coefficients,
                ..
            } = *segment
            {
                let (image_len, payload) = *payloads
                    .get(image)
                    .ok_or(WhiteoutError::InvalidRelation("compressed image missing"))?;
                if image_len != coefficients {
                    return Err(WhiteoutError::InvalidRelation(
                        "compressed image length differs from its segment",
                    ));
                }
                if payload.len() != self.compression.payload_len(image_len)? {
                    return Err(WhiteoutError::ResponseRejected("compressed payload length"));
                }
                targets.push(CompressedTarget {
                    digits: offset..offset + len,
                    image_len,
                    payload,
                });
            }
            offset += len;
        }
        Ok(targets)
    }

    /// Whether `witness` satisfies items 1–5: the statement's definition.
    pub fn check_witness(&self, witness: &ResponseWitness) -> Result<(), WhiteoutError> {
        let decoded = self.decode(&witness.digits)?;
        let (z, v) = (&decoded.z, &decoded.v);

        // Item 2: every ξ opens its payload, and t_D = A_D G⁻¹(V), A_D
        // applied to e's V digits as given.
        for (target, witness) in self.compressed_targets()?.iter().zip(&decoded.witnesses) {
            self.compression
                .check(target.image_len, target.payload, witness)?;
        }
        let t_d = self
            .setup
            .v_matrix()
            .apply_centered(&decoded.v_digits, RESPONSE_DIGIT_BOUND)?;
        if t_d
            .iter()
            .flat_map(|ring| ring.coefficients())
            .ne(decoded.v_image()?)
        {
            return Err(WhiteoutError::ResponseRejected("V commitment"));
        }

        // Item 3: N z = f + c t.
        let c = self.challenge.to_dense_ct()?;
        let bound = self.setup.params().response_bound();
        let nz = self.setup.linear().apply(z, bound)?;
        let f = v.chunks_exact(D).map(CyclotomicRing::<F, D>::from_slice);
        for ((nz, f), t) in nz.iter().zip(f).zip(self.targets(&decoded)?.iter()) {
            if *nz != f + challenge_times_ring(&c, t) {
                return Err(WhiteoutError::ResponseRejected("ring-linear rows"));
            }
        }

        // Items 4 and 5.
        let [e0, e1, h] = self.v_tail(v)?;
        if !h.constant_term().is_zero() {
            return Err(WhiteoutError::ResponseRejected("ct(h)"));
        }
        let c_ring: ChallengeRing<F> = challenge_ring(&c);
        let aggregated = &self.aggregated;
        let tables = self.relation.claim_tables()?;
        let parts = aggregated.linear_parts(&tables, &z[..], false)?;
        let lhs = aggregated.quadratic(&c_ring, &parts) + self.setup.l_row().apply(z, bound)?;
        if lhs != e0 + c_ring * e1 + c_ring * c_ring * h {
            return Err(WhiteoutError::ResponseRejected("ring-quadratic identity"));
        }
        let product = aggregated.product_polynomial(&c_ring, &parts);
        if product.get(CHALLENGE_RING_DIM..) != Some(&decoded.theta[..]) {
            return Err(WhiteoutError::ResponseRejected("product quotient θ"));
        }
        Ok(())
    }

    /// The number of rows of each kind, as [`ResponseRows`] holds them.
    pub fn row_shape(&self) -> ResponseRowShape {
        ResponseRowShape {
            v_image: self.setup.v_matrix().shape().rows * CHALLENGE_RING_DIM,
            linear: self.setup.linear().num_outputs() * D,
            products: self.relation.num_products(),
        }
    }

    /// The value of every affine row of Items 2–5 on `digits` (`e` in
    /// layout order, every digit in the radix-16 alphabet), with Item 4
    /// reduced at `alpha` (Whiteout §4.3, "Reducing the products"). Items 2,
    /// 3 and 5 vanish exactly when they hold. For Item 4, `product_target`
    /// is `[H_e](α) + (α^64 + 1)θ(α)`, which equals `Σ_i U_i W_i` exactly
    /// when the product lift (4) holds at `α`.
    pub fn rows(&self, digits: &[i8], alpha: F) -> Result<ResponseRows<F>, WhiteoutError> {
        let decoded = self.decode(digits)?;
        let (z, v) = (&decoded.z, &decoded.v);
        let flat = |rings: Vec<CyclotomicRing<F, D>>| -> Vec<F> {
            rings
                .iter()
                .flat_map(|ring| ring.coefficients().iter().copied())
                .collect()
        };
        let mut v_image = self
            .setup
            .v_matrix()
            .apply_centered(&decoded.v_digits, RESPONSE_DIGIT_BOUND)?
            .iter()
            .flat_map(|ring| ring.coefficients().iter().copied())
            .collect::<Vec<F>>();
        for (value, target) in v_image.iter_mut().zip(decoded.v_image()?) {
            *value -= *target;
        }

        let c = self.challenge.to_dense_ct()?;
        let bound = self.setup.params().response_bound();
        let mut linear = flat(self.setup.linear().apply(z, bound)?.to_vec());
        let shifts = self
            .targets(&decoded)?
            .iter()
            .map(|t| challenge_times_ring(&c, t))
            .collect();
        for ((value, f), shift) in linear.iter_mut().zip(v).zip(flat(shifts)) {
            *value -= *f + shift;
        }

        let [e0, e1, h] = self.v_tail(v)?;
        let c_ring: ChallengeRing<F> = challenge_ring(&c);
        let aggregated = &self.aggregated;
        let tables = self.relation.claim_tables()?;
        let parts = aggregated.linear_parts(&tables, &z[..], false)?;
        let operands = aggregated.product_operands(&c_ring, &parts);
        let reduced = operands
            .iter()
            .fold(ChallengeRing::<F>::zero(), |sum, (x, y)| sum + *x * *y);
        let h_e = reduced
            - (aggregated.quadratic(&c_ring, &parts) + self.setup.l_row().apply(z, bound)?)
            + e0
            + c_ring * e1
            + c_ring * c_ring * h;
        let powers = alpha_powers(alpha);
        let theta = decoded
            .theta
            .iter()
            .zip(powers.coefficients())
            .fold(F::zero(), |sum, (theta, power)| sum + *theta * *power);
        Ok(ResponseRows {
            v_image,
            linear,
            ct_h: h.constant_term(),
            product_target: h_e.coefficient_inner_product(&powers)
                + (alpha_pow_ring_dim(alpha) + F::one()) * theta,
            product_x: operands
                .iter()
                .map(|(x, _)| x.coefficient_inner_product(&powers))
                .collect(),
            product_y: operands
                .iter()
                .map(|(_, y)| y.coefficient_inner_product(&powers))
                .collect(),
        })
    }

    /// The batched row `Σ_l w_l · row_l(e) = ⟨a, e⟩ + b` for row weights
    /// `weights`: returns `a`, one coefficient per digit of `e` in layout
    /// order, and `b = Σ_l w_l · row_l(0)`. `a` is the transpose of the rows'
    /// maps applied to `weights`, so the response matrix `H` is never built
    /// (Whiteout §4.3, "Evaluating the public weights"); it is returned
    /// factored ([`ResponseWeights`]), so no table of `e`'s length is built
    /// either.
    pub fn weigh(
        &self,
        weights: &ResponseRows<F>,
        alpha: F,
    ) -> Result<(ResponseWeights<F>, F), WhiteoutError> {
        if weights.shape() != self.row_shape() {
            return Err(WhiteoutError::InvalidRelation(
                "row weights differ from the statement's rows",
            ));
        }
        let layout = self.layout()?;
        let c = self.challenge.to_dense_ct()?;
        let c_ring: ChallengeRing<F> = challenge_ring(&c);
        let n_out = self.setup.linear().num_outputs() * D;
        // Weights on each compressed image, in layout order: Item 3's
        // `−c·t` on the compressed targets of `N`, then Item 2's `−t_D`.
        let mut images: Vec<Vec<F>> = Vec::new();
        let mut rest = weights.linear.as_slice();
        for (rows, target) in self.setup.linear().targets() {
            let (block, tail) = rest.split_at((rows * D).min(rest.len()));
            rest = tail;
            if let BlockTarget::Compressed { .. } = target {
                images.push(
                    block
                        .chunks_exact(D)
                        .flat_map(|ring| {
                            (-challenge_times_ring_transpose(
                                &c,
                                &CyclotomicRing::<F, D>::from_slice(ring),
                            ))
                            .coefficients()
                            .to_vec()
                        })
                        .collect(),
                );
            }
        }
        images.push(weights.v_image.iter().map(|w| -*w).collect());
        let mut z = vec![F::zero(); self.setup.table().num_coefficients()];
        let mut v = vec![F::zero(); n_out + 3 * CHALLENGE_RING_DIM];
        let mut theta = vec![F::zero(); THETA_COEFFICIENTS];

        // Item 2 acts on V's digits directly.
        let v_rows: Vec<ChallengeRing<F>> = weights
            .v_image
            .chunks_exact(CHALLENGE_RING_DIM)
            .map(CyclotomicRing::from_slice)
            .collect();
        let v_digit_columns = self.setup.v_matrix().apply_transposed(
            &v_rows
                .iter()
                .map(CyclotomicRing::sigma_m1)
                .collect::<Vec<_>>(),
        )?;

        // Item 3: N z − f.
        self.setup.linear().adjoint(&weights.linear, &mut z)?;
        for (dst, weight) in v.iter_mut().zip(&weights.linear) {
            *dst -= *weight;
        }

        // Item 5.
        let tail = |index: usize| n_out + index * CHALLENGE_RING_DIM;
        if let Some(dst) = v.get_mut(tail(2)) {
            *dst += weights.ct_h;
        }

        // Item 4: [H_e](α) + (α^64 + 1)θ(α) and the product factors.
        let powers = alpha_powers(alpha);
        let target = powers.scale(&weights.product_target);
        let times =
            |ring: &ChallengeRing<F>, by: &ChallengeRing<F>| (ring.sigma_m1() * *by).sigma_m1();
        let through_c = times(&target, &c_ring);
        for (index, ring) in [target, through_c, times(&through_c, &c_ring)]
            .into_iter()
            .enumerate()
        {
            let range = tail(index)..tail(index + 1);
            for (dst, w) in v
                .get_mut(range)
                .ok_or(WhiteoutError::InvalidRelation("V too short"))?
                .iter_mut()
                .zip(ring.coefficients())
            {
                *dst += *w;
            }
        }
        let theta_scale = (alpha_pow_ring_dim(alpha) + F::one()) * weights.product_target;
        for (dst, power) in theta.iter_mut().zip(powers.coefficients()) {
            *dst += theta_scale * *power;
        }
        self.setup.l_row().adjoint(&(-target), &mut z)?;
        let aggregated = &self.aggregated;
        let tables = self.relation.claim_tables()?;
        let x: Vec<ChallengeRing<F>> = weights.product_x.iter().map(|w| powers.scale(w)).collect();
        let y: Vec<ChallengeRing<F>> = weights.product_y.iter().map(|w| powers.scale(w)).collect();
        aggregated.product_adjoint(
            &tables,
            &c_ring,
            &ProductAdjointWeights {
                target,
                x: &x,
                y: &y,
            },
            &mut z,
        )?;

        // Back through the digit reconstruction to e.
        let mut segments = Vec::with_capacity(layout.segments.len());
        let mut z_rest = z.as_slice();
        for segment in &layout.segments {
            let dense = |planes: PlaneLayout, values: &[F]| -> Result<Vec<F>, WhiteoutError> {
                let mut out = vec![F::zero(); segment.num_digits()];
                planes.transpose(values, &mut out)?;
                Ok(out)
            };
            segments.push(match (*segment, segment.planes()) {
                (ResponseSegment::Response { coefficients, .. }, Some(planes)) => {
                    let (values, tail) = z_rest.split_at(coefficients.min(z_rest.len()));
                    z_rest = tail;
                    SegmentWeights::planar(planes, values.to_vec())?
                }
                (ResponseSegment::V { coefficients }, Some(planes)) => {
                    let mut out = dense(
                        planes,
                        v.get(..coefficients)
                            .ok_or(WhiteoutError::InvalidRelation("V weights too short"))?,
                    )?;
                    // A_D reads V's digits in e's order, one ring per column.
                    for (column, dst) in v_digit_columns
                        .iter()
                        .zip(out.chunks_exact_mut(CHALLENGE_RING_DIM))
                    {
                        for (dst, w) in dst.iter_mut().zip(column.sigma_m1().coefficients()) {
                            *dst += *w;
                        }
                    }
                    SegmentWeights::Dense(out)
                }
                (ResponseSegment::Theta, Some(planes)) => {
                    SegmentWeights::Dense(dense(planes, &theta)?)
                }
                (
                    ResponseSegment::Compression {
                        image,
                        coefficients,
                        ..
                    },
                    _,
                ) => SegmentWeights::Dense(
                    self.compression.reconstruct_transpose(
                        coefficients,
                        images
                            .get(image)
                            .ok_or(WhiteoutError::InvalidRelation("compressed image missing"))?,
                    )?,
                ),
                (_, None) => {
                    return Err(WhiteoutError::InvariantViolation(
                        "a planar segment without planes",
                    ))
                }
            });
        }
        let a = ResponseWeights::new(segments)?;
        if a.len() != layout.len() {
            return Err(WhiteoutError::InvariantViolation("row weights length"));
        }
        Ok((a, self.constant(weights, alpha)?))
    }

    /// `e`'s decoded parts: `z`, `V` as field coefficients and as digits,
    /// and `θ`. Checks the length and Item 1.
    fn decode<'d>(&self, digits: &'d [i8]) -> Result<Decoded<'d, F>, WhiteoutError> {
        let layout = self.layout()?;
        if digits.len() != layout.len() {
            return Err(WhiteoutError::ResponseRejected("response length"));
        }
        // Item 1.
        let (min, max) = balanced_range(RESPONSE_LOG_BASIS, 1)
            .ok_or(WhiteoutError::ResponseRejected("digit alphabet"))?;
        if digits
            .iter()
            .any(|&digit| !(min..=max).contains(&i128::from(digit)))
        {
            return Err(WhiteoutError::ResponseRejected(
                "digit outside its alphabet",
            ));
        }
        let mut decoded = Decoded {
            z: SecretVec::from(vec![0i64; self.setup.table().num_coefficients()]),
            v: SecretVec::from(Vec::new()),
            v_digits: SecretVec::from(Vec::new()),
            theta: SecretVec::from(Vec::new()),
            images: Vec::new(),
            witnesses: Vec::new(),
        };
        let mut z_offset = 0;
        let mut rest = digits;
        for segment in &layout.segments {
            let (digits, tail) = rest.split_at(segment.num_digits());
            rest = tail;
            match (*segment, segment.planes()) {
                (ResponseSegment::Response { .. }, Some(planes)) => {
                    let out = decoded
                        .z
                        .get_mut(z_offset..z_offset + planes.coefficients)
                        .ok_or(WhiteoutError::InvalidRelation("response longer than s"))?;
                    planes.recompose_integers(digits, out)?;
                    z_offset += planes.coefficients;
                }
                (ResponseSegment::V { .. }, Some(planes)) => {
                    decoded.v = SecretVec::from(planes.recompose(digits)?);
                    decoded.v_digits = SecretVec::from(
                        digits
                            .iter()
                            .map(|&digit| i64::from(digit))
                            .collect::<Vec<_>>(),
                    );
                }
                (ResponseSegment::Theta, Some(planes)) => {
                    decoded.theta = SecretVec::from(planes.recompose(digits)?);
                }
                (ResponseSegment::Compression { coefficients, .. }, _) => {
                    decoded.images.push(SecretVec::from(
                        self.compression.reconstruct(coefficients, digits)?,
                    ));
                    decoded.witnesses.push(digits);
                }
                (_, None) => {
                    return Err(WhiteoutError::InvariantViolation(
                        "a planar segment without planes",
                    ))
                }
            }
        }
        Ok(decoded)
    }

    /// Every target of `N`: the public ones, and the compressed ones as
    /// `e` reconstructs them.
    fn targets(
        &self,
        decoded: &Decoded<'_, F>,
    ) -> Result<Vec<CyclotomicRing<F, D>>, WhiteoutError> {
        let mut images = decoded.images.iter();
        let mut targets = Vec::with_capacity(self.setup.linear().num_outputs());
        for (_, target) in self.setup.linear().targets() {
            match target {
                BlockTarget::Public(image) => targets.extend_from_slice(image),
                BlockTarget::Compressed { .. } => targets.extend(
                    images
                        .next()
                        .ok_or(WhiteoutError::InvalidRelation("compressed image missing"))?
                        .chunks_exact(D)
                        .map(CyclotomicRing::from_slice),
                ),
            }
        }
        Ok(targets)
    }

    /// `(E_0, E_1, h)`, the last three challenge-ring elements of `V`.
    fn v_tail(&self, v: &[F]) -> Result<[ChallengeRing<F>; 3], WhiteoutError> {
        let tail = v
            .get(self.setup.linear().num_outputs() * D..)
            .ok_or(WhiteoutError::ResponseRejected("V too short"))?;
        Ok([0, 1, 2].map(|index| {
            CyclotomicRing::from_slice(
                tail.get(index * CHALLENGE_RING_DIM..(index + 1) * CHALLENGE_RING_DIM)
                    .unwrap_or_default(),
            )
        }))
    }
}

/// The largest digit magnitude of the radix-16 alphabet: the bound of the
/// Ajtai map on `V`'s digits.
const RESPONSE_DIGIT_BOUND: u64 = 1 << (RESPONSE_LOG_BASIS - 1);

/// `e`'s decoded parts.
/// `e` decoded; every buffer is wiped when dropped.
struct Decoded<'d, F> {
    z: SecretVec<i64>,
    v: SecretVec<F>,
    v_digits: SecretVec<i64>,
    theta: SecretVec<F>,
    /// Each compressed image as its witness reconstructs it, in layout
    /// order; `t_D` is last.
    images: Vec<SecretVec<F>>,
    witnesses: Vec<&'d [i8]>,
}

impl<F> Decoded<'_, F> {
    /// `t_D`, as its witness reconstructs it.
    fn v_image(&self) -> Result<&[F], WhiteoutError> {
        self.images
            .last()
            .map(|image| &image[..])
            .ok_or(WhiteoutError::InvalidRelation("t_D's compression missing"))
    }
}

/// The row counts of a [`ResponseStatement`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResponseRowShape {
    pub v_image: usize,
    pub linear: usize,
    pub products: usize,
}

/// One value (or one weight) per affine row of the response statement, in
/// this order: Item 2 per coefficient of `t_D`, Item 3 per coefficient of
/// `N`'s outputs, Item 5, then Item 4's three product terms at `α`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResponseRows<F> {
    /// `A_D G⁻¹(V) − t_D`.
    pub v_image: Vec<F>,
    /// `N z − f − c t`.
    pub linear: Vec<F>,
    /// `ct(h)`.
    pub ct_h: F,
    /// `[H_e](α) + (α^64 + 1)θ(α)`.
    pub product_target: F,
    /// `U_i = μ_i [X_i(c, z)](α)`.
    pub product_x: Vec<F>,
    /// `W_i = [Y_i(c, z)](α)`.
    pub product_y: Vec<F>,
}

impl<F: JoltField> ResponseRows<F> {
    pub fn shape(&self) -> ResponseRowShape {
        ResponseRowShape {
            v_image: self.v_image.len(),
            linear: self.linear.len(),
            products: self.product_x.len(),
        }
    }

    /// `Σ_l self_l · other_l`.
    pub fn dot(&self, other: &Self) -> Result<F, WhiteoutError> {
        if self.shape() != other.shape() || self.product_y.len() != other.product_y.len() {
            return Err(WhiteoutError::InvalidRelation("row shapes differ"));
        }
        let dot = |a: &[F], b: &[F]| a.iter().zip(b).fold(F::zero(), |sum, (a, b)| sum + *a * *b);
        Ok(dot(&self.v_image, &other.v_image)
            + dot(&self.linear, &other.linear)
            + self.ct_h * other.ct_h
            + self.product_target * other.product_target
            + dot(&self.product_x, &other.product_x)
            + dot(&self.product_y, &other.product_y))
    }
}

/// `(1, α, …, α^63)` as a challenge-ring element: `[r](α) = ⟨r, powers⟩`.
pub(crate) fn alpha_powers<F: JoltField>(alpha: F) -> ChallengeRing<F> {
    let mut power = F::one();
    CyclotomicRing::from_coefficients(std::array::from_fn(|_| {
        let current = power;
        power *= alpha;
        current
    }))
}

/// `α^64`.
pub(crate) fn alpha_pow_ring_dim<F: JoltField>(alpha: F) -> F {
    (0..CHALLENGE_RING_DIM).fold(F::one(), |power, _| power * alpha)
}

#[cfg(feature = "whiteout-prover")]
impl ResponseWitness {
    /// `e` for the accepted response `z`, the first message `V` (as flat
    /// coefficients), `θ`, and `t_D`'s compression witness; the compressed
    /// targets' witnesses come from their images.
    pub(crate) fn new<F: JoltField, const D: usize>(
        setup: &SigmaSetup<F, D>,
        compression: &dyn ImageCompression<F>,
        z: &[i64],
        v: &[F],
        theta: &[F],
        t_d_witness: &[i8],
    ) -> Result<Self, WhiteoutError> {
        let layout = setup.response_layout(compression)?;
        let mut witnesses = Vec::new();
        for (_, target) in setup.linear().targets() {
            if let BlockTarget::Compressed { payload, image } = target {
                let image: Vec<F> = image
                    .as_ref()
                    .ok_or(WhiteoutError::InvalidRelation(
                        "the prover's setup lacks a compressed target's image",
                    ))?
                    .iter()
                    .flat_map(|ring| ring.coefficients().iter().copied())
                    .collect();
                let compressed = compression.compress(&image)?;
                if compressed.payload != *payload {
                    return Err(WhiteoutError::InvariantViolation(
                        "a compressed target differs from its image",
                    ));
                }
                witnesses.push(compressed.witness);
            }
        }
        witnesses.push(t_d_witness.to_vec());
        let mut witnesses = witnesses.into_iter();
        let mut witness = Self {
            digits: vec![0i8; layout.len()],
        };
        let mut out = witness.digits.as_mut_slice();
        let mut rest = z;
        for segment in layout.segments {
            let (digits, tail) = std::mem::take(&mut out)
                .split_at_mut_checked(segment.num_digits())
                .ok_or(WhiteoutError::InvariantViolation("response layout length"))?;
            out = tail;
            match (segment, segment.planes()) {
                (ResponseSegment::Response { coefficients, .. }, Some(planes)) => {
                    let (values, tail) = rest.split_at(coefficients);
                    rest = tail;
                    planes.integer_digits(values, digits)?;
                }
                (ResponseSegment::V { coefficients }, Some(planes)) => planes.field_digits(
                    v.get(..coefficients)
                        .ok_or(WhiteoutError::InvariantViolation(
                            "V shorter than its segment",
                        ))?,
                    digits,
                )?,
                (ResponseSegment::Theta, Some(planes)) => planes.field_digits(theta, digits)?,
                (ResponseSegment::Compression { .. }, _) => {
                    let witness = witnesses.next().ok_or(WhiteoutError::InvariantViolation(
                        "compression witness missing",
                    ))?;
                    if witness.len() != digits.len() {
                        return Err(WhiteoutError::InvariantViolation(
                            "compression witness length",
                        ));
                    }
                    digits.copy_from_slice(&witness);
                }
                (_, None) => {
                    return Err(WhiteoutError::InvariantViolation(
                        "a planar segment without planes",
                    ))
                }
            }
        }
        Ok(witness)
    }
}
