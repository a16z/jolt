//! The response statement (Whiteout §4.3) and the `ResponseProver` seam.
//!
//! After an accepted attempt, the verifier holds `(c, ρ, t_D)` and the
//! commitment images, but never `z`, `V` or `θ`. The response proof certifies
//! the vector `e = (G_z⁻¹(z), G⁻¹(V), G⁻¹(θ))`:
//!
//! 1. every digit of `e` lies in the balanced radix-16 alphabet;
//! 2. `t_D = A_D G⁻¹(V)`;
//! 3. `N z = f + c t`, on Gaussian and public coordinates alike;
//! 4. `Q_μ(c, z) + L z = E_0 + c E_1 + c² h`, with `θ` the quotient of
//!    `Σ μ_i [X_i(c,z)][Y_i(c,z)]` by `X^64 + 1`;
//! 5. `ct(h) = 0`.
//!
//! The compression digits ξ join `e` with W5's compression. Akita proves the
//! statement (W5); [`ResponseStatement::check_witness`] is its definition.

#[cfg(feature = "whiteout-prover")]
use jolt_algebra::balanced_decompose_coefficients_pow2_i8_into;
#[cfg(feature = "whiteout-prover")]
use jolt_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use jolt_algebra::{CyclotomicRing, JoltField};
use jolt_crypto::challenge::SparseChallenge;
use jolt_crypto::lattice::CHALLENGE_RING_DIM;
use jolt_transcript::{ProverTranscript, Sponge, VerifierTranscript};

use super::challenge::{challenge_ring, challenge_times_ring};
use super::lift::{ChallengeRing, LiftedRelation};
use super::params::RESPONSE_LOG_BASIS;
use super::segments::balanced_range;
use super::sigma::SigmaSetup;
use super::WhiteoutError;
use crate::masking::ZkError;

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
}

impl ResponseSegment {
    fn len(&self) -> usize {
        match *self {
            Self::Response {
                coefficients,
                digits,
                ..
            } => coefficients * digits,
            Self::V { coefficients } => coefficients * FIELD_LEVELS,
            Self::Theta => THETA_COEFFICIENTS * FIELD_LEVELS,
        }
    }
}

/// The logical layout of `e`, from the public setup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResponseLayout {
    pub segments: Vec<ResponseSegment>,
}

impl ResponseLayout {
    pub fn len(&self) -> usize {
        self.segments.iter().map(ResponseSegment::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// `e` in digit form, in [`ResponseLayout`] order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResponseWitness {
    pub digits: Vec<i8>,
}

/// What the response proof certifies.
pub struct ResponseStatement<'a, F: JoltField, const D: usize> {
    pub(crate) setup: &'a SigmaSetup<F, D>,
    pub(crate) relation: &'a LiftedRelation<F>,
    pub(crate) challenge: SparseChallenge,
    /// The aggregation challenges; `μ` follows from the relation's labels.
    pub(crate) rho: Vec<F>,
    pub(crate) v_image: Vec<ChallengeRing<F>>,
}

/// Proves the response statement. Akita implements it (W5).
pub trait ResponseProver<F: JoltField, const D: usize> {
    fn prove_response<H: Sponge>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        witness: &ResponseWitness,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<(), ZkError>;

    fn verify_response<H: Sponge>(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), ZkError>;
}

impl<F: JoltField, const D: usize> ResponseStatement<'_, F, D> {
    pub fn layout(&self) -> ResponseLayout {
        self.setup.response_layout()
    }

    /// The Σ-protocol challenge `c`.
    pub fn challenge(&self) -> &SparseChallenge {
        &self.challenge
    }

    /// The aggregation challenges `ρ`.
    pub fn rho(&self) -> &[F] {
        &self.rho
    }

    /// The commitment `t_D` to `V`.
    pub fn v_image(&self) -> &[ChallengeRing<F>] {
        &self.v_image
    }

    /// Whether `witness` satisfies items 1–5: the statement's definition.
    pub fn check_witness(&self, witness: &ResponseWitness) -> Result<(), WhiteoutError> {
        let layout = self.layout();
        if witness.digits.len() != layout.len() {
            return Err(WhiteoutError::ResponseRejected("response length"));
        }
        // Item 1.
        let (min, max) = balanced_range(RESPONSE_LOG_BASIS, 1)
            .ok_or(WhiteoutError::ResponseRejected("digit alphabet"))?;
        if witness
            .digits
            .iter()
            .any(|&digit| !(min..=max).contains(&i128::from(digit)))
        {
            return Err(WhiteoutError::ResponseRejected(
                "digit outside its alphabet",
            ));
        }
        let table = self.setup.table();
        let mut z = Vec::with_capacity(table.num_coefficients());
        let mut v = Vec::new();
        let mut v_digits: Vec<i64> = Vec::new();
        let mut theta = Vec::new();
        let mut rest = witness.digits.as_slice();
        for segment in &layout.segments {
            let (digits, tail) = rest.split_at(segment.len());
            rest = tail;
            match *segment {
                ResponseSegment::Response {
                    coefficients,
                    digits: levels,
                    ..
                } => z.extend(integers_from_digits(digits, coefficients, levels)),
                ResponseSegment::V { coefficients } => {
                    for ring in digits.chunks_exact(FIELD_LEVELS * CHALLENGE_RING_DIM) {
                        v.extend(field_from_planes::<F>(ring, CHALLENGE_RING_DIM));
                    }
                    v.truncate(coefficients);
                    v_digits = digits.iter().map(|&digit| i64::from(digit)).collect();
                }
                ResponseSegment::Theta => {
                    theta = field_from_planes::<F>(digits, THETA_COEFFICIENTS);
                }
            }
        }

        // Item 2: t_D = A_D G⁻¹(V), A_D applied to e's V digits as given.
        let t_d = self
            .setup
            .v_matrix()
            .apply_centered(&v_digits, max.unsigned_abs().max(min.unsigned_abs()) as u32)?;
        if t_d != self.v_image {
            return Err(WhiteoutError::ResponseRejected("V commitment"));
        }

        // Item 3: N z = f + c t.
        let c = self.challenge.to_dense_ct()?;
        let bound = self.setup.params().response_bound();
        let nz = self.setup.linear().apply(&z, bound)?;
        let f = v.chunks_exact(D).map(CyclotomicRing::<F, D>::from_slice);
        for ((nz, f), t) in nz.iter().zip(f).zip(self.setup.linear().targets()) {
            if *nz != f + challenge_times_ring(&c, t) {
                return Err(WhiteoutError::ResponseRejected("ring-linear rows"));
            }
        }

        // Items 4 and 5.
        let tail = v
            .get(self.setup.linear().num_outputs() * D..)
            .ok_or(WhiteoutError::ResponseRejected("V too short"))?;
        let [e0, e1, h] = [0, 1, 2].map(|index| {
            CyclotomicRing::from_slice(
                tail.get(index * CHALLENGE_RING_DIM..(index + 1) * CHALLENGE_RING_DIM)
                    .unwrap_or_default(),
            )
        });
        if !h.constant_term().is_zero() {
            return Err(WhiteoutError::ResponseRejected("ct(h)"));
        }
        let c_ring: ChallengeRing<F> = challenge_ring(&c);
        let aggregated = self.relation.aggregate(&self.rho)?;
        let tables = self.relation.claim_tables()?;
        let parts = aggregated.linear_parts(&tables, &z, false)?;
        let lhs = aggregated.quadratic(&c_ring, &parts) + self.setup.l_row().apply(&z, bound)?;
        if lhs != e0 + c_ring * e1 + c_ring * c_ring * h {
            return Err(WhiteoutError::ResponseRejected("ring-quadratic identity"));
        }
        let product = aggregated.product_polynomial(&c_ring, &parts);
        if product.get(CHALLENGE_RING_DIM..) != Some(theta.as_slice()) {
            return Err(WhiteoutError::ResponseRejected("product quotient θ"));
        }
        Ok(())
    }
}

/// The balanced radix-16 digits of every integer in `values` with
/// `levels` digits each, digit plane by digit plane.
#[cfg(feature = "whiteout-prover")]
pub(crate) fn integer_digits(values: &[i64], levels: usize) -> Result<Vec<i8>, WhiteoutError> {
    let radix = 1i128 << RESPONSE_LOG_BASIS;
    let (min, top) = balanced_range(RESPONSE_LOG_BASIS, levels)
        .ok_or(WhiteoutError::InvariantViolation("response digit range"))?;
    let offset = -min;
    let mut digits = vec![0i8; values.len() * levels];
    for (index, &value) in values.iter().enumerate() {
        let value = i128::from(value);
        if !(-offset..=top).contains(&value) {
            return Err(WhiteoutError::InvariantViolation(
                "response coefficient outside its digit range",
            ));
        }
        let mut shifted = value + offset;
        for level in 0..levels {
            if let Some(digit) = digits.get_mut(level * values.len() + index) {
                *digit = ((shifted % radix) - radix / 2) as i8;
            }
            shifted /= radix;
        }
    }
    Ok(digits)
}

/// The integers whose `levels` balanced radix-16 digits per coefficient are
/// `digits`, digit plane by digit plane.
fn integers_from_digits(digits: &[i8], coefficients: usize, levels: usize) -> Vec<i64> {
    let mut values = vec![0i64; coefficients];
    for plane in digits.chunks_exact(coefficients.max(1)).take(levels).rev() {
        for (value, &digit) in values.iter_mut().zip(plane) {
            *value = *value * (1 << RESPONSE_LOG_BASIS) + i64::from(digit);
        }
    }
    values
}

/// The field elements whose 32 balanced radix-16 digit planes over
/// `coefficients` values are `planes`.
fn field_from_planes<F: JoltField>(planes: &[i8], coefficients: usize) -> Vec<F> {
    let radix = F::from_u64(1 << RESPONSE_LOG_BASIS);
    let mut values = vec![F::zero(); coefficients];
    for plane in planes
        .chunks_exact(coefficients.max(1))
        .take(FIELD_LEVELS)
        .rev()
    {
        for (value, &digit) in values.iter_mut().zip(plane) {
            *value = *value * radix + F::from_i64(i64::from(digit));
        }
    }
    values
}

/// The 32 balanced radix-16 digit planes of `values`.
#[cfg(feature = "whiteout-prover")]
pub(crate) fn field_digits<F: JoltField>(values: &[F]) -> Vec<i8> {
    let params = BalancedDecomposePow2Params::<F>::new(FIELD_LEVELS, RESPONSE_LOG_BASIS);
    let mut digits = vec![0i8; values.len() * FIELD_LEVELS];
    balanced_decompose_coefficients_pow2_i8_into(values, &mut digits, &params);
    digits
}

#[cfg(feature = "whiteout-prover")]
impl ResponseWitness {
    /// `e` for the accepted response `z`, the first message `V` (as flat
    /// coefficients) and `θ`.
    pub(crate) fn new<F: JoltField, const D: usize>(
        setup: &SigmaSetup<F, D>,
        z: &[i64],
        v: &[F],
        theta: &[F],
    ) -> Result<Self, WhiteoutError> {
        let mut digits = Vec::with_capacity(setup.response_layout().len());
        let mut rest = z;
        for segment in setup.response_layout().segments {
            match segment {
                ResponseSegment::Response {
                    coefficients,
                    digits: levels,
                    ..
                } => {
                    let (values, tail) = rest.split_at(coefficients);
                    rest = tail;
                    digits.extend(integer_digits(values, levels)?);
                }
                ResponseSegment::V { .. } => {
                    for ring in v.chunks_exact(CHALLENGE_RING_DIM) {
                        digits.extend(field_digits(ring));
                    }
                }
                ResponseSegment::Theta => digits.extend(field_digits(theta)),
            }
        }
        Ok(Self { digits })
    }
}
