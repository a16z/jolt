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
use std::ops::Range;

use jolt_algebra::{CyclotomicRing, JoltField};
use jolt_crypto::challenge::SparseChallenge;
use jolt_crypto::lattice::CHALLENGE_RING_DIM;
use jolt_transcript::{ProverTranscript, Sponge, VerifierTranscript};

use super::challenge::{challenge_ring, challenge_times_ring, challenge_times_ring_transpose};
use super::compression::ImageCompression;
use super::lift::{ChallengeRing, LiftedRelation, ProductAdjointWeights};
use super::linear::BlockTarget;
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
    /// The number of digits of `e` in this segment.
    pub fn num_digits(&self) -> usize {
        match *self {
            Self::Response {
                coefficients,
                digits,
                ..
            } => coefficients * digits,
            Self::V { coefficients } => coefficients * FIELD_LEVELS,
            Self::Theta => THETA_COEFFICIENTS * FIELD_LEVELS,
            Self::Compression { digits, .. } => digits,
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
        self.segments.iter().map(ResponseSegment::num_digits).sum()
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
    /// The response prover's compression map.
    pub(crate) compression: &'a dyn ImageCompression<F>,
    pub(crate) challenge: SparseChallenge,
    /// The aggregation challenges; `μ` follows from the relation's labels.
    pub(crate) rho: Vec<F>,
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

/// Proves the response statement. Akita implements it (W5).
pub trait ResponseProver<F: JoltField, const D: usize> {
    /// The compression map of the statement's images.
    fn compression(&self) -> &dyn ImageCompression<F>;

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
            self.compression.check(target.payload, witness)?;
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
        let aggregated = self.relation.aggregate(&self.rho)?;
        let tables = self.relation.claim_tables()?;
        let parts = aggregated.linear_parts(&tables, z, false)?;
        let lhs = aggregated.quadratic(&c_ring, &parts) + self.setup.l_row().apply(z, bound)?;
        if lhs != e0 + c_ring * e1 + c_ring * c_ring * h {
            return Err(WhiteoutError::ResponseRejected("ring-quadratic identity"));
        }
        let product = aggregated.product_polynomial(&c_ring, &parts);
        if product.get(CHALLENGE_RING_DIM..) != Some(decoded.theta.as_slice()) {
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
        let mut linear = flat(self.setup.linear().apply(z, bound)?);
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
        let aggregated = self.relation.aggregate(&self.rho)?;
        let tables = self.relation.claim_tables()?;
        let parts = aggregated.linear_parts(&tables, z, false)?;
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
    /// (Whiteout §4.3, "Evaluating the public weights").
    pub fn weigh(&self, weights: &ResponseRows<F>, alpha: F) -> Result<(Vec<F>, F), WhiteoutError> {
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
        let aggregated = self.relation.aggregate(&self.rho)?;
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
        let radix = F::from_u64(1 << RESPONSE_LOG_BASIS);
        let mut a = Vec::with_capacity(layout.len());
        let mut z_rest = z.as_slice();
        for segment in &layout.segments {
            match *segment {
                ResponseSegment::Response {
                    coefficients,
                    digits,
                    ..
                } => {
                    let (values, tail) = z_rest.split_at(coefficients.min(z_rest.len()));
                    z_rest = tail;
                    push_planes(&mut a, values, digits, radix);
                }
                ResponseSegment::V { coefficients } => {
                    let start = a.len();
                    for ring in v
                        .get(..coefficients)
                        .ok_or(WhiteoutError::InvalidRelation("V weights too short"))?
                        .chunks_exact(CHALLENGE_RING_DIM)
                    {
                        push_planes(&mut a, ring, FIELD_LEVELS, radix);
                    }
                    let columns = a
                        .get_mut(start..)
                        .ok_or(WhiteoutError::InvalidRelation("V weights too short"))?;
                    for (column, dst) in v_digit_columns
                        .iter()
                        .zip(columns.chunks_exact_mut(CHALLENGE_RING_DIM))
                    {
                        for (dst, w) in dst.iter_mut().zip(column.sigma_m1().coefficients()) {
                            *dst += *w;
                        }
                    }
                }
                ResponseSegment::Theta => push_planes(&mut a, &theta, FIELD_LEVELS, radix),
                ResponseSegment::Compression {
                    image,
                    coefficients,
                    ..
                } => a.extend(
                    self.compression.reconstruct_transpose(
                        coefficients,
                        images
                            .get(image)
                            .ok_or(WhiteoutError::InvalidRelation("compressed image missing"))?,
                    )?,
                ),
            }
        }
        if a.len() != layout.len() {
            return Err(WhiteoutError::InvariantViolation("row weights length"));
        }
        let constant = weights.dot(&self.rows(&vec![0; layout.len()], alpha)?)?;
        Ok((a, constant))
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
            z: Vec::with_capacity(self.setup.table().num_coefficients()),
            v: Vec::new(),
            v_digits: Vec::new(),
            theta: Vec::new(),
            images: Vec::new(),
            witnesses: Vec::new(),
        };
        let mut rest = digits;
        for segment in &layout.segments {
            let (digits, tail) = rest.split_at(segment.num_digits());
            rest = tail;
            match *segment {
                ResponseSegment::Response {
                    coefficients,
                    digits: levels,
                    ..
                } => decoded
                    .z
                    .extend(integers_from_digits(digits, coefficients, levels)),
                ResponseSegment::V { coefficients } => {
                    for ring in digits.chunks_exact(FIELD_LEVELS * CHALLENGE_RING_DIM) {
                        decoded
                            .v
                            .extend(field_from_planes::<F>(ring, CHALLENGE_RING_DIM));
                    }
                    decoded.v.truncate(coefficients);
                    decoded.v_digits = digits.iter().map(|&digit| i64::from(digit)).collect();
                }
                ResponseSegment::Theta => {
                    decoded.theta = field_from_planes::<F>(digits, THETA_COEFFICIENTS);
                }
                ResponseSegment::Compression { coefficients, .. } => {
                    decoded
                        .images
                        .push(self.compression.reconstruct(coefficients, digits)?);
                    decoded.witnesses.push(digits);
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
struct Decoded<'d, F> {
    z: Vec<i64>,
    v: Vec<F>,
    v_digits: Vec<i64>,
    theta: Vec<F>,
    /// Each compressed image as its witness reconstructs it, in layout
    /// order; `t_D` is last.
    images: Vec<Vec<F>>,
    witnesses: Vec<&'d [i8]>,
}

impl<F> Decoded<'_, F> {
    /// `t_D`, as its witness reconstructs it.
    fn v_image(&self) -> Result<&[F], WhiteoutError> {
        self.images
            .last()
            .map(Vec::as_slice)
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
fn alpha_powers<F: JoltField>(alpha: F) -> ChallengeRing<F> {
    let mut power = F::one();
    CyclotomicRing::from_coefficients(std::array::from_fn(|_| {
        let current = power;
        power *= alpha;
        current
    }))
}

/// `α^64`.
fn alpha_pow_ring_dim<F: JoltField>(alpha: F) -> F {
    (0..CHALLENGE_RING_DIM).fold(F::one(), |power, _| power * alpha)
}

/// Appends the digit weights of `values` reconstructed from `levels`
/// radix planes, plane-major: plane `j` of value `k` takes `radix^j · w_k`.
pub(crate) fn push_planes<F: JoltField>(out: &mut Vec<F>, values: &[F], levels: usize, radix: F) {
    let mut power = F::one();
    for _ in 0..levels {
        out.extend(values.iter().map(|value| *value * power));
        power *= radix;
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
pub(crate) fn field_from_planes<F: JoltField>(planes: &[i8], coefficients: usize) -> Vec<F> {
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
        let mut digits = Vec::with_capacity(layout.len());
        let mut rest = z;
        for segment in layout.segments {
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
                ResponseSegment::Compression {
                    digits: expected, ..
                } => {
                    let witness = witnesses.next().ok_or(WhiteoutError::InvariantViolation(
                        "compression witness missing",
                    ))?;
                    if witness.len() != expected {
                        return Err(WhiteoutError::InvariantViolation(
                            "compression witness length",
                        ));
                    }
                    digits.extend(witness);
                }
            }
        }
        Ok(Self { digits })
    }
}
