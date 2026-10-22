//! The batched response row at a point, without the dense weights
//! (`specs/whiteout-akita.md` §5.8, P1). For row weights `ω`,
//! [`ResponseStatement::weigh`] gives `a` with `Σ_l ω_l·row_l(e) = ⟨a, e⟩ +
//! b`; the verifier needs only `Σ_j a_j·eq(r, j)`, which is the rows' linear
//! part evaluated at the field vector `x_j = eq(r, j)`. That vector factors
//! as `eq(r_lo, j mod D)·eq(r_hi, j / D)`, so every response segment of `e`
//! decodes to a vector over `s` whose native element `E` is `S_E·Ê`, one
//! scalar times one common ring, and `N`, `L` and the lifted forms apply to
//! it with one column combination and one ring product per row.

use jolt_algebra::{CyclotomicRing, JoltField};
use jolt_crypto::lattice::CHALLENGE_RING_DIM;
use jolt_poly::EqPolynomial;
use rayon::prelude::*;

use super::challenge::{challenge_ring, challenge_times_ring, challenge_times_ring_transpose};
use super::lift::{ChallengeRing, ClaimTable, SValues, WitnessShape};
use super::linear::BlockTarget;
use super::params::RESPONSE_LOG_BASIS;
use super::response::{
    alpha_pow_ring_dim, alpha_powers, PlaneLayout, ResponseRows, ResponseSegment,
    ResponseStatement, THETA_COEFFICIENTS,
};
use super::segments::{SegmentMask, SegmentTable};
use super::WhiteoutError;

/// `eq(point, ·)` over `{0,1}^k` with `point[0]` the least significant bit.
fn eq_table<F: JoltField>(point: &[F]) -> Vec<F> {
    EqPolynomial::new(point.iter().rev().copied().collect()).evaluations()
}

/// `eq(point, j)` split at bit `lo_bits`.
struct EqDual<F> {
    lo: Vec<F>,
    hi: Vec<F>,
    lo_bits: usize,
}

impl<F: JoltField> EqDual<F> {
    fn new(point: &[F], lo_bits: usize) -> Self {
        let (lo, hi) = point.split_at(lo_bits.min(point.len()));
        Self {
            lo: eq_table(lo),
            hi: eq_table(hi),
            lo_bits,
        }
    }

    fn at(&self, j: usize) -> F {
        let lo = self
            .lo
            .get(j & ((1 << self.lo_bits) - 1))
            .copied()
            .unwrap_or_default();
        lo * self.hi.get(j >> self.lo_bits).copied().unwrap_or_default()
    }

    /// `eq(r_hi, j / 2^lo_bits)`.
    fn high(&self, j: usize) -> F {
        self.hi.get(j >> self.lo_bits).copied().unwrap_or_default()
    }

    /// `eq(r_lo, ·)` on `[offset, offset + len)`.
    fn low(&self, offset: usize, len: usize) -> &[F] {
        self.lo.get(offset..offset + len).unwrap_or_default()
    }
}

/// A vector laid out like `s` whose native element `E` is `scales[E]·lo`.
pub(crate) struct FactoredS<F: JoltField, const D: usize> {
    scales: Vec<F>,
    lo: CyclotomicRing<F, D>,
}

impl<F: JoltField, const D: usize> FactoredS<F, D> {
    fn value(&self, position: usize) -> F {
        let scale = self.scales.get(position / D).copied().unwrap_or_default();
        scale
            * self
                .lo
                .coefficients()
                .get(position % D)
                .copied()
                .unwrap_or_default()
    }
}

impl<F: JoltField, const D: usize> SValues<F> for FactoredS<F, D> {
    fn ring(&self, table: &SegmentTable, ring: usize) -> Result<ChallengeRing<F>, WhiteoutError> {
        let (start, stride) = table.ring_position(ring)?;
        Ok(CyclotomicRing::from_coefficients(std::array::from_fn(
            |index| self.value(start + index * stride),
        )))
    }

    fn claim_sum(
        &self,
        witness: &WitnessShape,
        table: &ClaimTable<F>,
        skip_public: bool,
    ) -> Result<[F; CHALLENGE_RING_DIM], WhiteoutError> {
        if witness.native_dim != D {
            return Err(WhiteoutError::InvalidRelation(
                "witness dimension differs from the setup's",
            ));
        }
        let mut outer = F::zero();
        let mut element_offset = 0usize;
        for placed in &witness.segments {
            let elements = placed.segment.num_coefficients() / D;
            if !(skip_public && placed.segment.mask == SegmentMask::Public) {
                let weights = table
                    .element
                    .get(element_offset..element_offset + elements)
                    .ok_or(WhiteoutError::InvalidRelation("claim shorter than w"))?;
                let scales = self
                    .scales
                    .get(placed.coefficient / D..placed.coefficient / D + elements)
                    .ok_or(WhiteoutError::InvalidRelation("witness outside s"))?;
                outer += weights
                    .iter()
                    .zip(scales)
                    .fold(F::zero(), |sum, (w, s)| sum + *w * *s);
            }
            element_offset += elements;
        }
        let stride = D / CHALLENGE_RING_DIM;
        Ok(std::array::from_fn(|i| {
            let inner = table
                .component
                .iter()
                .enumerate()
                .fold(F::zero(), |sum, (k, eq)| {
                    sum + *eq
                        * self
                            .lo
                            .coefficients()
                            .get(stride * i + k)
                            .copied()
                            .unwrap_or_default()
                });
            outer * inner
        }))
    }
}

/// `eq(r, ·)` read through `e`'s layout: `z` factored over `s`, `V` and
/// `θ` as field values, `V`'s digit columns as scales per `Ê` quarter, and
/// each compression witness as its entries.
struct Dual<F: JoltField, const D: usize> {
    z: FactoredS<F, D>,
    v: Vec<F>,
    /// Per `V` digit column (64 digits), its scale, and which 64-entry
    /// block of `eq(r_lo, ·)` it reads.
    v_columns: Vec<(F, usize)>,
    theta: Vec<F>,
    compression: Vec<Vec<F>>,
}

impl<F: JoltField, const D: usize> ResponseStatement<'_, F, D> {
    /// `Σ_{j < |e|} a_j·eq(point, j)` for `(a, _) = weigh(weights, alpha)`,
    /// with `point[0]` the least significant index bit, in `O(|e|/D + |N|)`
    /// field operations. `e`'s response segments must start and end on
    /// native elements, which every layout from a setup does.
    pub fn weight_at(
        &self,
        weights: &ResponseRows<F>,
        alpha: F,
        point: &[F],
    ) -> Result<F, WhiteoutError> {
        if weights.shape() != self.row_shape() {
            return Err(WhiteoutError::InvalidRelation(
                "row weights differ from the statement's rows",
            ));
        }
        let dual = self.dual(point)?;
        let c = self.challenge.to_dense_ct()?;
        let c_ring: ChallengeRing<F> = challenge_ring(&c);
        let n_out = self.setup.linear().num_outputs() * D;
        let dot = |a: &[F], b: &[F]| a.iter().zip(b).fold(F::zero(), |sum, (a, b)| sum + *a * *b);
        let flat = |rings: &[CyclotomicRing<F, D>]| -> Vec<F> {
            rings
                .iter()
                .flat_map(|ring| ring.coefficients().iter().copied())
                .collect()
        };

        // Item 2: A_D G⁻¹(V) − t_D, with t_D from its compression witness.
        let quarters = D / CHALLENGE_RING_DIM;
        let v_cols = self.setup.v_matrix().shape().width;
        let mut a_d = vec![ChallengeRing::<F>::zero(); self.setup.v_matrix().shape().rows];
        for quarter in 0..quarters {
            let scales: Vec<F> = dual
                .v_columns
                .iter()
                .map(|(scale, q)| if *q == quarter { *scale } else { F::zero() })
                .collect();
            if scales.len() != v_cols {
                return Err(WhiteoutError::InvalidRelation("V digit columns"));
            }
            let lo = CyclotomicRing::from_slice(
                dual.z
                    .lo
                    .coefficients()
                    .get(quarter * CHALLENGE_RING_DIM..(quarter + 1) * CHALLENGE_RING_DIM)
                    .unwrap_or_default(),
            );
            for (dst, row) in a_d
                .iter_mut()
                .zip(self.setup.v_matrix().combine_columns(&scales)?)
            {
                *dst += row * lo;
            }
        }
        let mut total = dot(
            &weights.v_image,
            &a_d.iter()
                .flat_map(|ring| ring.coefficients().iter().copied())
                .collect::<Vec<F>>(),
        );
        let mut images = dual.compression.iter();
        let mut targets = self.compressed_targets()?.into_iter();
        let mut compressed = |weights: &[F]| -> Result<F, WhiteoutError> {
            let (x, target) = (
                images
                    .next()
                    .ok_or(WhiteoutError::InvalidRelation("compressed image missing"))?,
                targets
                    .next()
                    .ok_or(WhiteoutError::InvalidRelation("compressed image missing"))?,
            );
            Ok(dot(
                &self
                    .compression
                    .reconstruct_transpose(target.image_len, weights)?,
                x,
            ))
        };

        // Item 3: N z − f − c t, with the compressed t from their witnesses.
        let nz = self
            .setup
            .linear()
            .apply_factored(&dual.z.scales, &dual.z.lo)?;
        let mut linear = flat(&nz);
        for (value, f) in linear.iter_mut().zip(&dual.v) {
            *value -= *f;
        }
        total += dot(&weights.linear, &linear);
        let mut rest = weights.linear.as_slice();
        for (rows, target) in self.setup.linear().targets() {
            let (block, tail) = rest.split_at((rows * D).min(rest.len()));
            rest = tail;
            if let BlockTarget::Compressed { .. } = target {
                let through_c: Vec<F> = block
                    .chunks_exact(D)
                    .flat_map(|ring| {
                        challenge_times_ring_transpose(
                            &c,
                            &CyclotomicRing::<F, D>::from_slice(ring),
                        )
                        .coefficients()
                        .to_vec()
                    })
                    .collect();
                total -= compressed(&through_c)?;
            }
        }
        total -= compressed(&weights.v_image)?;

        // Item 5.
        let tail = |index: usize| -> Result<ChallengeRing<F>, WhiteoutError> {
            dual.v
                .get(n_out + index * CHALLENGE_RING_DIM..n_out + (index + 1) * CHALLENGE_RING_DIM)
                .map(CyclotomicRing::from_slice)
                .ok_or(WhiteoutError::InvalidRelation("V too short"))
        };
        let [e0, e1, h] = [tail(0)?, tail(1)?, tail(2)?];
        total += weights.ct_h * h.constant_term();

        // Item 4 at α.
        let aggregated = self.relation.aggregate(&self.rho)?;
        let tables = self.relation.claim_tables()?;
        let parts = aggregated.linear_parts(&tables, &dual.z, false)?;
        let l_z = self
            .setup
            .l_row()
            .apply_factored(&dual.z.scales, &dual.z.lo)?;
        let h_e = aggregated.target_linear(&c_ring, &parts)? - l_z
            + e0
            + c_ring * e1
            + c_ring * c_ring * h;
        let powers = alpha_powers(alpha);
        let theta = dual
            .theta
            .iter()
            .zip(powers.coefficients())
            .fold(F::zero(), |sum, (theta, power)| sum + *theta * *power);
        total += weights.product_target
            * (h_e.coefficient_inner_product(&powers)
                + (alpha_pow_ring_dim(alpha) + F::one()) * theta);
        let operands = aggregated.product_operands(&ChallengeRing::<F>::zero(), &parts);
        for ((x, y), (wx, wy)) in operands
            .iter()
            .zip(weights.product_x.iter().zip(&weights.product_y))
        {
            total += *wx * x.coefficient_inner_product(&powers)
                + *wy * y.coefficient_inner_product(&powers);
        }
        Ok(total)
    }

    /// `Σ_l ω_l·row_l(0)`: the batched row's constant, from the public
    /// targets and the relation's constants alone.
    pub fn constant(&self, weights: &ResponseRows<F>, alpha: F) -> Result<F, WhiteoutError> {
        if weights.shape() != self.row_shape() {
            return Err(WhiteoutError::InvalidRelation(
                "row weights differ from the statement's rows",
            ));
        }
        let c = self.challenge.to_dense_ct()?;
        let c_ring: ChallengeRing<F> = challenge_ring(&c);
        let mut total = F::zero();
        let mut rest = weights.linear.as_slice();
        for (rows, target) in self.setup.linear().targets() {
            let (block, tail) = rest.split_at((rows * D).min(rest.len()));
            rest = tail;
            if let BlockTarget::Public(image) = target {
                for (weights, t) in block.chunks_exact(D).zip(image) {
                    total -= challenge_times_ring(&c, t)
                        .coefficient_inner_product(&CyclotomicRing::from_slice(weights));
                }
            }
        }
        let aggregated = self.relation.aggregate(&self.rho)?;
        let zero = aggregated.zero_parts();
        let operands = aggregated.product_operands(&c_ring, &zero);
        let reduced = operands
            .iter()
            .fold(ChallengeRing::<F>::zero(), |sum, (x, y)| sum + *x * *y);
        let h_e = reduced - aggregated.quadratic(&c_ring, &zero);
        let powers = alpha_powers(alpha);
        total += weights.product_target * h_e.coefficient_inner_product(&powers);
        for ((x, y), (wx, wy)) in operands
            .iter()
            .zip(weights.product_x.iter().zip(&weights.product_y))
        {
            total += *wx * x.coefficient_inner_product(&powers)
                + *wy * y.coefficient_inner_product(&powers);
        }
        Ok(total)
    }

    /// `eq(point, ·)` read through `e`'s layout.
    fn dual(&self, point: &[F]) -> Result<Dual<F, D>, WhiteoutError> {
        let layout = self.layout()?;
        let lo_bits = D.trailing_zeros() as usize;
        if point.len() < lo_bits
            || u32::try_from(point.len())
                .ok()
                .and_then(|bits| 1usize.checked_shl(bits))
                .is_none_or(|domain| domain < layout.len())
        {
            return Err(WhiteoutError::InvalidRelation(
                "point does not cover the response vector",
            ));
        }
        let eq = EqDual::new(point, lo_bits);
        let radix = F::from_u64(1 << RESPONSE_LOG_BASIS);
        let mut dual = Dual {
            z: FactoredS {
                scales: Vec::with_capacity(self.setup.table().num_coefficients() / D),
                lo: CyclotomicRing::from_slice(eq.low(0, D)),
            },
            v: Vec::new(),
            v_columns: Vec::new(),
            theta: vec![F::zero(); THETA_COEFFICIENTS],
            compression: Vec::new(),
        };
        let mut offset = 0;
        for segment in &layout.segments {
            // `Σ_p radix^p·eq(point, offset + index(k, p))`, through the
            // segment's position law.
            let planar = |planes: PlaneLayout, k: usize, at: &(dyn Fn(usize) -> F + Sync)| {
                let mut power = F::one();
                let mut value = F::zero();
                for plane in 0..planes.planes {
                    value += power * at(offset + planes.index(k, plane));
                    power *= radix;
                }
                value
            };
            match (*segment, segment.planes()) {
                (ResponseSegment::Response { coefficients, .. }, Some(planes)) => {
                    // Each native element's D coefficients are D consecutive
                    // digits of every plane, so its weight is one scale of
                    // eq(r_hi, ·) times eq(r_lo, ·).
                    if offset % D != 0 || coefficients % D != 0 || planes.width % D != 0 {
                        return Err(WhiteoutError::InvalidRelation(
                            "a response segment is not on native elements",
                        ));
                    }
                    dual.z.scales.par_extend(
                        (0..coefficients / D)
                            .into_par_iter()
                            .map(|element| planar(planes, element * D, &|j| eq.high(j))),
                    );
                }
                (ResponseSegment::V { coefficients }, Some(planes)) => {
                    // A_D reads V's digits one 64-digit ring per column, each
                    // inside one quarter of eq(r_lo, ·).
                    if offset % CHALLENGE_RING_DIM != 0 {
                        return Err(WhiteoutError::InvalidRelation(
                            "V's digits do not start on a challenge ring",
                        ));
                    }
                    dual.v.par_extend(
                        (0..coefficients)
                            .into_par_iter()
                            .map(|k| planar(planes, k, &|j| eq.at(j))),
                    );
                    for column in
                        (offset..offset + segment.num_digits()).step_by(CHALLENGE_RING_DIM)
                    {
                        dual.v_columns
                            .push((eq.high(column), (column % D) / CHALLENGE_RING_DIM));
                    }
                }
                (ResponseSegment::Theta, Some(planes)) => {
                    for (k, dst) in dual.theta.iter_mut().enumerate() {
                        *dst = planar(planes, k, &|j| eq.at(j));
                    }
                }
                (ResponseSegment::Compression { digits, .. }, _) => dual.compression.push(
                    (offset..offset + digits)
                        .into_par_iter()
                        .map(|j| eq.at(j))
                        .collect(),
                ),
                (_, None) => {
                    return Err(WhiteoutError::InvariantViolation(
                        "a planar segment without planes",
                    ))
                }
            }
            offset += segment.num_digits();
        }
        Ok(dual)
    }
}
