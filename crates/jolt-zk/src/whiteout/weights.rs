//! The batched response row at a point, without the dense weights
//! (`specs/whiteout-akita.md` §5.8, P1). For row weights `ω`,
//! [`ResponseStatement::weigh`] gives `a` with `Σ_l ω_l·row_l(e) = ⟨a, e⟩ +
//! b`; the verifier needs only `Σ_j a_j·eq(r, j)`, which is the rows' linear
//! part evaluated at the field vector `x_j = eq(r, j)`. That vector factors
//! as `eq(r_lo, j mod D)·eq(r_hi, j / D)`, so every response segment of `e`
//! decodes to a vector over `s` whose native element `E` is `S_E·Ê`, one
//! scalar times one common ring, and `N`, `L` and the lifted forms apply to
//! it with one column combination and one ring product per row.

use std::collections::BTreeMap;

use jolt_algebra::{CyclotomicRing, JoltField};
use jolt_crypto::lattice::CHALLENGE_RING_DIM;
use jolt_poly::EqPolynomial;
use rayon::prelude::*;

use super::challenge::{challenge_ring, challenge_times_ring, challenge_times_ring_transpose};
use super::lift::{ChallengeRing, ClaimTable, DenseTerm, SValues, WitnessShape};
use super::linear::BlockTarget;
use super::linear_factored::{ColumnScales, ScaleWindow};
use super::params::RESPONSE_LOG_BASIS;
use super::response::{
    alpha_pow_ring_dim, alpha_powers, PlaneLayout, ResponseRows, ResponseSegment,
    ResponseStatement, THETA_COEFFICIENTS,
};
use super::segments::{SegmentMask, SegmentTable};
use super::WhiteoutError;

/// `eq(point, ·)` over `{0,1}^k` with `point[0]` the least significant bit.
pub(super) fn eq_table<F: JoltField>(point: &[F]) -> Vec<F> {
    EqPolynomial::new(point.iter().rev().copied().collect()).evaluations()
}

/// `eq(point, j)` split at bit `lo_bits`, with `eq(r_hi, h)` split again at
/// bit `split` as `top[h >> split]·low[h mod 2^split]`, so that no table of
/// `eq(r_hi, ·)` is written out.
struct EqDual<F> {
    lo: Vec<F>,
    low: Vec<F>,
    top: Vec<F>,
    /// `r_hi`, for the narrower splits of [`Self::window_sums`].
    hi_point: Vec<F>,
    lo_bits: usize,
    split: usize,
}

impl<F: JoltField> EqDual<F> {
    fn new(point: &[F], lo_bits: usize, split: usize) -> Self {
        let (lo, hi) = point.split_at(lo_bits.min(point.len()));
        let split = split.min(hi.len());
        let (low, top) = hi.split_at(split);
        Self {
            lo: eq_table(lo),
            low: eq_table(low),
            top: eq_table(top),
            hi_point: hi.to_vec(),
            lo_bits,
            split,
        }
    }

    fn at(&self, j: usize) -> F {
        let lo = self
            .lo
            .get(j & ((1 << self.lo_bits) - 1))
            .copied()
            .unwrap_or_default();
        lo * self.hi(j >> self.lo_bits)
    }

    /// `eq(r_hi, h)`.
    fn hi(&self, h: usize) -> F {
        self.low_at(h) * self.top_at(h >> self.split)
    }

    /// `eq(r_hi, j / 2^lo_bits)`.
    fn high(&self, j: usize) -> F {
        self.hi(j >> self.lo_bits)
    }

    /// `low[h mod 2^split]`.
    fn low_at(&self, h: usize) -> F {
        self.low
            .get(h & ((1 << self.split) - 1))
            .copied()
            .unwrap_or_default()
    }

    fn top_at(&self, index: usize) -> F {
        self.top.get(index).copied().unwrap_or_default()
    }

    /// `eq(r_lo, ·)` on `[offset, offset + len)`.
    fn low(&self, offset: usize, len: usize) -> &[F] {
        self.lo.get(offset..offset + len).unwrap_or_default()
    }

    /// `Σ_w w.weight·eq(r_hi, w.start + l)` for `l < len`, over windows whose
    /// starts agree mod `2^split`: one sum of top entries per run of
    /// `2^split` positions, then one product per position. Below the
    /// tables' own split `s`, `eq(r_hi, ·)` splits at `split` as
    /// `low'[h mod 2^split]·mid[(h >> split) mod 2^(s − split)]·top[h >> s]`,
    /// with `low'` and `mid` the tables of `r_hi`'s bits below `split` and
    /// in `[split, s)`: at most `2^s` entries each.
    fn window_sums(&self, windows: &[ScaleWindow<F>], len: usize, split: usize) -> Vec<F> {
        let split = split.min(self.split);
        let gap = self.split - split;
        let narrow = (gap > 0).then(|| {
            (
                eq_table(self.hi_point.get(..split).unwrap_or_default()),
                eq_table(self.hi_point.get(split..self.split).unwrap_or_default()),
            )
        });
        let (low, mid): (&[F], &[F]) = match &narrow {
            Some((low, mid)) => (low, mid),
            None => (&self.low, &[]),
        };
        let top_at = |index: usize| match narrow {
            Some(_) => {
                mid.get(index & ((1 << gap) - 1))
                    .copied()
                    .unwrap_or_default()
                    * self.top_at(index >> gap)
            }
            None => self.top_at(index),
        };
        let mask = (1 << split) - 1;
        let residue = windows.first().map_or(0, |window| window.start & mask);
        debug_assert!(windows.iter().all(|window| window.start & mask == residue));
        let runs = (residue + len).div_ceil(1 << split);
        (0..runs)
            .into_par_iter()
            .flat_map_iter(|run| {
                let sum = windows.iter().fold(F::zero(), |sum, window| {
                    sum + window.weight * top_at((window.start >> split) + run)
                });
                let positions = (run << split).max(residue) - residue
                    ..(((run + 1) << split) - residue).min(len);
                positions
                    .map(move |l| sum * low.get((residue + l) & mask).copied().unwrap_or_default())
            })
            .collect()
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

    /// A ring inside one native element `E` is `S_E` times the ring
    /// `lo[o + i·stride]`, which depends only on `(o, stride)`. So the sum
    /// takes one scalar multiple per term and one ring product per distinct
    /// `(o, stride)`; a ring that straddles two elements is multiplied
    /// directly.
    fn dense_sum(
        &self,
        table: &SegmentTable,
        dense: &[DenseTerm<F>],
    ) -> Result<ChallengeRing<F>, WhiteoutError> {
        type Groups<F> = (BTreeMap<(usize, usize), ChallengeRing<F>>, ChallengeRing<F>);
        let (groups, direct) = dense
            .par_iter()
            .try_fold(
                || (BTreeMap::new(), CyclotomicRing::zero()),
                |(mut groups, mut direct): Groups<F>, (ring, _, weights)| {
                    let (start, stride) = table.ring_position(*ring)?;
                    let offset = start % D;
                    if offset + (CHALLENGE_RING_DIM - 1) * stride < D {
                        let scale = self.scales.get(start / D).copied().unwrap_or_default();
                        weights.scale_accumulate_into(
                            groups
                                .entry((offset, stride))
                                .or_insert_with(CyclotomicRing::zero),
                            scale,
                        );
                    } else {
                        weights.mul_accumulate_into(&self.ring(table, *ring)?, &mut direct);
                    }
                    Ok::<_, WhiteoutError>((groups, direct))
                },
            )
            .try_reduce(
                || (BTreeMap::new(), CyclotomicRing::zero()),
                |(mut groups, direct), (other, other_direct)| {
                    for (key, ring) in other {
                        *groups.entry(key).or_insert_with(CyclotomicRing::zero) += ring;
                    }
                    Ok((groups, direct + other_direct))
                },
            )?;
        Ok(groups
            .into_iter()
            .fold(direct, |sum, ((offset, stride), weights)| {
                let lo = CyclotomicRing::from_coefficients(std::array::from_fn(|index| {
                    self.lo
                        .coefficients()
                        .get(offset + index * stride)
                        .copied()
                        .unwrap_or_default()
                }));
                sum + weights * lo
            }))
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

/// `eq(r, ·)` read through `e`'s layout: `z` factored over `s`, its scales
/// also as windows of `eq(r_hi, ·)`, `V` and `θ` as field values, `V`'s
/// digit columns as scales per `Ê` quarter, and each compression witness as
/// its entries.
struct Dual<F: JoltField, const D: usize> {
    z: FactoredS<F, D>,
    /// `z.scales` as windows of `eq(r_hi, ·)`, one per digit plane of each
    /// response segment's plane block.
    z_windows: Vec<ScaleWindow<F>>,
    /// `r_hi`: the point past its `log₂ D` low variables.
    hi_point: Vec<F>,
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
        let dual = tracing::info_span!("whiteout_weight_dual").in_scope(|| self.dual(point))?;
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
        let v_span = tracing::info_span!("whiteout_weight_v_image").entered();
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

        drop(v_span);
        // Item 3: N z − f − c t, with the compressed t from their witnesses.
        let nz_span = tracing::info_span!("whiteout_weight_nz").entered();
        let nz = self.setup.linear().apply_factored(
            &ColumnScales {
                dense: &dual.z.scales,
                windows: &dual.z_windows,
                point: &dual.hi_point,
            },
            &dual.z.lo,
        )?;
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
        drop(nz_span);

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
        let _lifted_span = tracing::info_span!("whiteout_weight_lifted_rows").entered();
        let aggregated = &self.aggregated;
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
        let aggregated = &self.aggregated;
        let operands = aggregated.product_operands(&c_ring, &aggregated.zero_parts());
        let h_e = aggregated.target_at_zero(&c_ring)?;
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
        // Split `eq(r_hi, ·)` in halves; a response block whose planes sit
        // `width/D` elements apart is summed at that stride's lowest set bit
        // (`EqDual::window_sums`), so all its planes share one residue.
        let split = (point.len() - lo_bits).div_ceil(2);
        let eq = EqDual::new(point, lo_bits, split);
        let radix = F::from_u64(1 << RESPONSE_LOG_BASIS);
        let mut dual = Dual {
            z: FactoredS {
                scales: Vec::with_capacity(self.setup.table().num_coefficients() / D),
                lo: CyclotomicRing::from_slice(eq.low(0, D)),
            },
            z_windows: Vec::new(),
            hi_point: point.get(lo_bits..).unwrap_or_default().to_vec(),
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
                    // Plane `p` of a plane block starting at coefficient
                    // `k` reads `eq(r_hi, ·)` from digit `offset +
                    // index(k, p)` on, one entry per native element.
                    let first = dual.z.scales.len();
                    for block in (0..coefficients).step_by(planes.width.max(1)) {
                        let elements =
                            first + block / D..first + (block + planes.width).min(coefficients) / D;
                        let windows = dual.z_windows.len();
                        let mut power = F::one();
                        for plane in 0..planes.planes {
                            dual.z_windows.push(ScaleWindow {
                                elements: elements.clone(),
                                start: (offset + planes.index(block, plane)) / D,
                                weight: power,
                            });
                            power *= radix;
                        }
                        dual.z.scales.extend(eq.window_sums(
                            dual.z_windows.get(windows..).unwrap_or_default(),
                            elements.len(),
                            (planes.width / D).trailing_zeros() as usize,
                        ));
                    }
                }
                (ResponseSegment::V { coefficients }, Some(planes)) => {
                    // A_D reads V's digits one 64-digit ring per column, each
                    // inside one quarter of eq(r_lo, ·). So a ring's
                    // coefficient `i` is `Σ_q eq(r_lo, 64q + i)·S_q`, with
                    // `S_q` its planes' radix-weighted eq(r_hi, ·) entries
                    // in quarter `q`.
                    if offset % CHALLENGE_RING_DIM != 0 || planes.width != CHALLENGE_RING_DIM {
                        return Err(WhiteoutError::InvalidRelation(
                            "V's digits do not start on a challenge ring",
                        ));
                    }
                    let quarters = D / CHALLENGE_RING_DIM;
                    dual.v.par_extend(
                        (0..coefficients)
                            .into_par_iter()
                            .step_by(CHALLENGE_RING_DIM)
                            .flat_map_iter(|block| {
                                let mut sums = vec![F::zero(); quarters];
                                let mut power = F::one();
                                for plane in 0..planes.planes {
                                    let digit = offset + planes.index(block, plane);
                                    if let Some(sum) =
                                        sums.get_mut((digit % D) / CHALLENGE_RING_DIM)
                                    {
                                        *sum += power * eq.high(digit);
                                    }
                                    power *= radix;
                                }
                                let eq = &eq;
                                (0..CHALLENGE_RING_DIM.min(coefficients - block)).map(move |i| {
                                    sums.iter().enumerate().fold(F::zero(), |value, (q, sum)| {
                                        value
                                            + *sum
                                                * eq.low(q * CHALLENGE_RING_DIM + i, 1)
                                                    .first()
                                                    .copied()
                                                    .unwrap_or_default()
                                    })
                                })
                            }),
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
                // One eq(r_hi, ·) entry per native element's run of digits.
                (ResponseSegment::Compression { digits, .. }, _) => dual.compression.push(
                    (offset / D..(offset + digits).div_ceil(D))
                        .into_par_iter()
                        .flat_map_iter(|element| {
                            let high = eq.hi(element);
                            let eq = &eq;
                            ((element * D).max(offset)..((element + 1) * D).min(offset + digits))
                                .map(move |j| {
                                    eq.low(j % D, 1).first().copied().unwrap_or_default() * high
                                })
                        })
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

#[cfg(all(test, feature = "whiteout-prover"))]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests should fail loudly"
)]
mod tests {
    use jolt_algebra::{CyclotomicRing, Field, Zero};
    use jolt_crypto::lattice::CHALLENGE_RING_DIM;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use super::super::fixture::F;
    use super::super::lift::{ChallengeRing, DenseTerm, SValues};
    use super::super::linear_factored::ScaleWindow;
    use super::super::segments::{Segment, SegmentMask, SegmentRole, SegmentShape};
    use super::super::SegmentTable;
    use super::{eq_table, EqDual, FactoredS};

    /// `dense_sum` on a factored vector against `Σ_k σ(ℓ̂_k)·x_k` over the
    /// vector written out coefficient by coefficient. With `D = 64`, the
    /// 64-wide segments' rings lie in one native element (the grouped path)
    /// and the 128-wide segment's rings straddle two (the direct path).
    #[test]
    fn factored_dense_sum_matches_the_written_out_vector() {
        const D: usize = 64;
        let mut rng = ChaCha20Rng::seed_from_u64(0x0_3b);
        let segment = |native_dim, len| Segment {
            role: SegmentRole::Witness,
            len,
            native_dim,
            log_basis: 4,
            shape: SegmentShape::Dense,
            mask: SegmentMask::Gaussian,
        };
        let table =
            SegmentTable::new(vec![segment(64, 3), segment(128, 4), segment(64, 2)]).unwrap();
        let len = table.num_coefficients();
        let x = FactoredS::<F, D> {
            scales: (0..len / D).map(|_| F::random(&mut rng)).collect(),
            lo: CyclotomicRing::from_coefficients(std::array::from_fn(|_| F::random(&mut rng))),
        };
        let written: Vec<F> = (0..len)
            .map(|p| x.scales[p / D] * x.lo.coefficients()[p % D])
            .collect();
        let rings = len / CHALLENGE_RING_DIM;
        let straddling = (0..rings)
            .filter(|&ring| {
                let (start, stride) = table.ring_position(ring).unwrap();
                start % D + (CHALLENGE_RING_DIM - 1) * stride >= D
            })
            .count();
        assert_eq!(straddling, 4);

        let dense: Vec<DenseTerm<F>> = (0..rings)
            .map(|ring| {
                let hat: [F; CHALLENGE_RING_DIM] = std::array::from_fn(|_| F::random(&mut rng));
                (ring, hat, CyclotomicRing::from_coefficients(hat).sigma_m1())
            })
            .collect();
        let expected = dense
            .iter()
            .fold(ChallengeRing::<F>::zero(), |sum, (ring, _, sigma)| {
                let (start, stride) = table.ring_position(*ring).unwrap();
                sum + *sigma
                    * CyclotomicRing::from_coefficients(std::array::from_fn(|i| {
                        written[start + i * stride]
                    }))
            });
        assert_eq!(x.dense_sum(&table, &dense).unwrap(), expected);
    }

    /// `window_sums` at every split up to the balanced one against
    /// `Σ_w weight_w·eq(r_hi, start_w + l)` with `eq(r_hi, ·)` written out:
    /// the narrower splits rebuild `top` from `mid` and the balanced `top`.
    #[test]
    fn window_sums_match_the_written_out_eq_at_every_split() {
        let mut rng = ChaCha20Rng::seed_from_u64(0x0_3c);
        let (lo_bits, hi_bits): (usize, usize) = (3, 10);
        let point: Vec<F> = (0..lo_bits + hi_bits)
            .map(|_| F::random(&mut rng))
            .collect();
        let balanced = hi_bits.div_ceil(2);
        let dual = EqDual::new(&point, lo_bits, balanced);
        let eq_hi = eq_table(&point[lo_bits..]);
        for split in 0..=balanced {
            let step = 1 << split;
            let residue = 1 % step;
            let windows: Vec<ScaleWindow<F>> = [1, 3, 7]
                .into_iter()
                .map(|block| ScaleWindow {
                    elements: 0..0,
                    start: block * step + residue,
                    weight: F::random(&mut rng),
                })
                .collect();
            let len = 5 * step + 3;
            let expected: Vec<F> = (0..len)
                .map(|l| {
                    windows.iter().fold(F::zero(), |sum, window| {
                        sum + window.weight * eq_hi[window.start + l]
                    })
                })
                .collect();
            assert_eq!(
                dual.window_sums(&windows, len, split),
                expected,
                "split {split}"
            );
        }
    }
}
