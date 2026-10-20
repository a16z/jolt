//! The constant-term lift of the residual relation (Whiteout
//! Lemma `wo:lem:ct-lift`, Definition `wo:def:residual`).
//!
//! Every field-level row becomes the constant coefficient of an `R`-linear
//! form in `(u, s)` over the challenge ring `R = F[X]/(X^64 + 1)`:
//! a functional `ℓ` with coefficient vector `ℓ̂` lifts to
//! `L(u, s) = Σ σ(ℓ̂_k) s_k + u·b`, whose constant coefficient is `ℓ(s) + u·b`.
//! A product `x·y = z` becomes `X·Y − u·Z` with `X = u·x̂ − P_x(s)` the
//! constant operand. [`AggregatedRelation`] combines every row under the
//! weights `μ` of [`RowLabels`] into the quadratic `Q_μ` of Protocol
//! `wo:prot:sigma`.

use std::collections::BTreeMap;

use jolt_algebra::{CyclotomicRing, JoltField};
use jolt_crypto::lattice::CHALLENGE_RING_DIM;
use jolt_poly::EqPolynomial;

use super::labels::RowLabels;
use super::placement::{
    PadPlace, PadPlacement, OPERAND_LEVELS, OPERAND_LOG_BASIS, PAD_LEVELS, PAD_LOG_BASIS,
};
use super::segments::{SegmentRole, SegmentTable};
use super::WhiteoutError;
use crate::masking::{AffineRow, ResidualRelation, ResidualVar};

/// An element of the challenge ring.
pub type ChallengeRing<F> = CyclotomicRing<F, CHALLENGE_RING_DIM>;

/// The evaluation claims of the residual commitment that
/// `ResidualVar::Witness(j)` names: claim `j` is `w̃(r_j)`, the multilinear
/// extension of `w` (the `Witness` segments of `s`, in table order and native
/// coefficient order, zero-padded to a power of two) at `points[j]`, with the
/// first point coordinate the most significant index bit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluationClaims<F> {
    pub points: Vec<Vec<F>>,
}

/// A field-linear functional on `s` with a constant, as coefficient weights
/// `ℓ̂_k` per challenge-ring element `k`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Weights<F: JoltField> {
    rings: BTreeMap<usize, [F; CHALLENGE_RING_DIM]>,
    constant: F,
}

impl<F: JoltField> Weights<F> {
    fn new(constant: F) -> Self {
        Self {
            rings: BTreeMap::new(),
            constant,
        }
    }

    fn add(&mut self, ring: usize, position: usize, weight: F) -> Result<(), WhiteoutError> {
        let slot = self
            .rings
            .entry(ring)
            .or_insert([F::zero(); CHALLENGE_RING_DIM])
            .get_mut(position)
            .ok_or(WhiteoutError::InvalidRelation(
                "weight position outside a ring",
            ))?;
        *slot += weight;
        Ok(())
    }

    fn add_scaled(&mut self, other: &Self, scale: F) {
        for (&ring, coefficients) in &other.rings {
            let target = self
                .rings
                .entry(ring)
                .or_insert([F::zero(); CHALLENGE_RING_DIM]);
            for (dst, &src) in target.iter_mut().zip(coefficients) {
                *dst += src * scale;
            }
        }
        self.constant += other.constant * scale;
    }

    /// The lift `L(u, s) = Σ σ(ℓ̂_k) s_k + u·b`.
    fn lift(&self) -> Lift<F> {
        Lift {
            terms: self
                .rings
                .iter()
                .map(|(&ring, coefficients)| {
                    (
                        ring,
                        CyclotomicRing::from_coefficients(*coefficients).sigma_m1(),
                    )
                })
                .collect(),
            constant: self.constant,
        }
    }
}

/// `L(u, s) = Σ_k σ(ℓ̂_k)·s_k + u·b`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Lift<F: JoltField> {
    terms: Vec<(usize, ChallengeRing<F>)>,
    constant: F,
}

impl<F: JoltField> Lift<F> {
    /// The `s`-part `Σ σ(ℓ̂_k) s_k` on a challenge-ring view.
    fn linear(&self, view: &[ChallengeRing<F>]) -> Result<ChallengeRing<F>, WhiteoutError> {
        let mut out = CyclotomicRing::zero();
        for (ring, coefficient) in &self.terms {
            let value = view
                .get(*ring)
                .ok_or(WhiteoutError::InvalidRelation("lift reads outside s"))?;
            coefficient.mul_accumulate_into(value, &mut out);
        }
        Ok(out)
    }
}

/// `Σ_j scale_j · s_{k_j}`: an `R`-linear reconstruction, such as `P_x` or g.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Reconstruction<F> {
    terms: Vec<(usize, F)>,
}

impl<F: JoltField> Reconstruction<F> {
    fn digits(first_ring: usize, levels: usize, log_basis: u32) -> Self {
        let radix = F::from_u64(1 << log_basis);
        let mut power = F::one();
        let terms = (0..levels)
            .map(|level| {
                let term = (first_ring + level, power);
                power *= radix;
                term
            })
            .collect();
        Self { terms }
    }

    fn evaluate(&self, view: &[ChallengeRing<F>]) -> Result<ChallengeRing<F>, WhiteoutError> {
        let mut out = CyclotomicRing::zero();
        for (ring, scale) in &self.terms {
            view.get(*ring)
                .ok_or(WhiteoutError::InvalidRelation(
                    "reconstruction reads outside s",
                ))?
                .scale_accumulate_into(&mut out, *scale);
        }
        Ok(out)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LiftedProduct<F: JoltField> {
    masked: F,
    operand: usize,
    y: Lift<F>,
    z: Lift<F>,
}

/// The residual relation lifted to the challenge ring over a segment table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiftedRelation<F: JoltField> {
    labels: RowLabels,
    affine: Vec<Weights<F>>,
    products: Vec<LiftedProduct<F>>,
    /// `P_x` per operand.
    operands: Vec<Reconstruction<F>>,
    /// First ring of each operand's digit polynomials.
    operand_rings: Vec<usize>,
    g: Reconstruction<F>,
}

impl<F: JoltField> LiftedRelation<F> {
    /// Lifts `relation` over `table`, with its pads placed by `placement`
    /// and its witness variables naming `claims`.
    ///
    /// Release checks: every product's operand is an operand pad, the pad
    /// segments are large enough for the placement, and every variable
    /// exists.
    pub fn new(
        relation: &ResidualRelation<F>,
        placement: &PadPlacement,
        table: &SegmentTable,
        claims: &EvaluationClaims<F>,
    ) -> Result<Self, WhiteoutError> {
        placement.check_products(relation)?;
        if relation.num_witness != claims.points.len() {
            return Err(WhiteoutError::InvalidRelation(
                "witness variables differ from the evaluation claims",
            ));
        }
        let starts = table.starts();
        let ring_start = |role| -> Result<(usize, usize), WhiteoutError> {
            let index = table.unique(role).ok_or(WhiteoutError::InvalidRelation(
                "pad segment missing or repeated",
            ))?;
            let (_, ring) = starts
                .get(index)
                .copied()
                .ok_or(WhiteoutError::InvalidRelation("segment start missing"))?;
            let len = table
                .segments()
                .get(index)
                .map(|segment| segment.len)
                .ok_or(WhiteoutError::InvalidRelation("segment missing"))?;
            Ok((ring, len))
        };
        let (pad_ring, pad_len) = ring_start(SegmentRole::PadDigits)?;
        if pad_len < placement.pad_rings() {
            return Err(WhiteoutError::InvalidRelation(
                "pad digit segment too short",
            ));
        }
        let operand_ring = if placement.operand_pads().is_empty() {
            0
        } else {
            let (ring, len) = ring_start(SegmentRole::OperandDigits)?;
            if len < placement.operand_rings() {
                return Err(WhiteoutError::InvalidRelation(
                    "operand digit segment too short",
                ));
            }
            ring
        };
        let witness = WitnessIndex::new(table, &starts)?;
        let context = LiftContext {
            placement,
            pad_ring,
            operand_ring,
            claims: claims
                .points
                .iter()
                .map(|point| witness.weights(point))
                .collect::<Result<Vec<_>, _>>()?,
        };

        let affine = relation
            .affine
            .iter()
            .map(|row| context.weights(row))
            .collect::<Result<Vec<_>, _>>()?;
        let products = relation
            .products
            .iter()
            .map(|row| {
                let PadPlace::Operand(operand) = placement.place(row.x.pad)? else {
                    return Err(WhiteoutError::InvalidRelation(
                        "a product's operand is not an operand pad",
                    ));
                };
                Ok(LiftedProduct {
                    masked: row.x.masked,
                    operand,
                    y: context.weights(&row.y)?.lift(),
                    z: context.weights(&row.z)?.lift(),
                })
            })
            .collect::<Result<Vec<_>, WhiteoutError>>()?;
        let operand_rings: Vec<usize> = (0..placement.operand_pads().len())
            .map(|operand| operand_ring + operand * OPERAND_LEVELS)
            .collect();
        Ok(Self {
            labels: RowLabels::new(
                placement.operand_pads().len(),
                relation.products.len(),
                relation.affine.len(),
            )?,
            affine,
            products,
            operands: operand_rings
                .iter()
                .map(|&ring| Reconstruction::digits(ring, OPERAND_LEVELS, OPERAND_LOG_BASIS))
                .collect(),
            operand_rings,
            g: Reconstruction::digits(
                pad_ring + placement.regular_groups() * PAD_LEVELS,
                PAD_LEVELS,
                PAD_LOG_BASIS,
            ),
        })
    }

    pub fn labels(&self) -> &RowLabels {
        &self.labels
    }

    /// Combines every row under the weights `μ` from the aggregation
    /// challenges `rho`.
    pub fn aggregate(&self, rho: &[F]) -> Result<AggregatedRelation<'_, F>, WhiteoutError> {
        let mut lambda = Weights::new(F::zero());
        for (index, row) in self.affine.iter().enumerate() {
            lambda.add_scaled(row, self.labels.weight(self.labels.affine(index), rho)?);
        }
        // Scalarity rows: coefficient r ∈ [1, 63] of P_x vanishes. Their sum
        // over r puts weight Σ_r μ_{64o+r} 256^j X^r on digit ring j.
        let radix = F::from_u64(1 << OPERAND_LOG_BASIS);
        for (operand, &first) in self.operand_rings.iter().enumerate() {
            let mut power = F::one();
            for digit in 0..OPERAND_LEVELS {
                for r in 1..CHALLENGE_RING_DIM {
                    let weight = self.labels.weight(self.labels.scalarity(operand, r), rho)?;
                    lambda.add(first + digit, r, weight * power)?;
                }
                power *= radix;
            }
        }
        let product_weights = (0..self.products.len())
            .map(|index| self.labels.weight(self.labels.product(index), rho))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(AggregatedRelation {
            relation: self,
            product_weights,
            lambda: lambda.lift(),
        })
    }

    /// The field value of every row on `view` (the challenge-ring view of
    /// `s`), in the order affine rows, scalarity rows, product rows, then
    /// `ct(g)`; the index of the first nonzero one, if any. This is the
    /// prover's per-row check before the aggregation challenges.
    pub fn first_violation(
        &self,
        view: &[ChallengeRing<F>],
    ) -> Result<Option<usize>, WhiteoutError> {
        let mut index = 0;
        for row in &self.affine {
            if !row_constant_ok(row, view)? {
                return Ok(Some(index));
            }
            index += 1;
        }
        for operand in &self.operands {
            let p = operand.evaluate(view)?;
            for r in 1..CHALLENGE_RING_DIM {
                if p.coefficients().get(r).is_some_and(|c| !c.is_zero()) {
                    return Ok(Some(index));
                }
                index += 1;
            }
        }
        for product in &self.products {
            let p = self
                .operand(product.operand)?
                .evaluate(view)?
                .constant_term();
            let x = product.masked - p;
            let y = product.y.linear(view)?.constant_term() + product.y.constant;
            let z = product.z.linear(view)?.constant_term() + product.z.constant;
            if x * y != z {
                return Ok(Some(index));
            }
            index += 1;
        }
        if !self.g.evaluate(view)?.constant_term().is_zero() {
            return Ok(Some(index));
        }
        Ok(None)
    }

    fn operand(&self, index: usize) -> Result<&Reconstruction<F>, WhiteoutError> {
        self.operands
            .get(index)
            .ok_or(WhiteoutError::InvalidRelation(
                "operand outside the relation",
            ))
    }
}

/// Whether an affine row's value, linear part plus constant, is zero.
fn row_constant_ok<F: JoltField>(
    row: &Weights<F>,
    view: &[ChallengeRing<F>],
) -> Result<bool, WhiteoutError> {
    Ok((row.lift().linear(view)?.constant_term() + row.constant).is_zero())
}

/// `Q_μ` and its pieces.
pub struct AggregatedRelation<'a, F: JoltField> {
    relation: &'a LiftedRelation<F>,
    product_weights: Vec<F>,
    lambda: Lift<F>,
}

/// The `s`-linear parts of every form at one view: `u = 0`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearParts<F: JoltField> {
    /// `−P_x` per product.
    x: Vec<ChallengeRing<F>>,
    y: Vec<ChallengeRing<F>>,
    z: Vec<ChallengeRing<F>>,
    lambda: ChallengeRing<F>,
    g: ChallengeRing<F>,
}

impl<F: JoltField> AggregatedRelation<'_, F> {
    pub fn linear_parts(&self, view: &[ChallengeRing<F>]) -> Result<LinearParts<F>, WhiteoutError> {
        let mut parts = LinearParts {
            x: Vec::with_capacity(self.relation.products.len()),
            y: Vec::with_capacity(self.relation.products.len()),
            z: Vec::with_capacity(self.relation.products.len()),
            lambda: self.lambda.linear(view)?,
            g: self.relation.g.evaluate(view)?,
        };
        for product in &self.relation.products {
            parts
                .x
                .push(-self.relation.operand(product.operand)?.evaluate(view)?);
            parts.y.push(product.y.linear(view)?);
            parts.z.push(product.z.linear(view)?);
        }
        Ok(parts)
    }

    /// `Q_μ(u, ·)` from the linear parts of `·`.
    pub fn quadratic(&self, u: &ChallengeRing<F>, parts: &LinearParts<F>) -> ChallengeRing<F> {
        let mut out = CyclotomicRing::zero();
        for ((((product, weight), x), y), z) in self
            .relation
            .products
            .iter()
            .zip(&self.product_weights)
            .zip(&parts.x)
            .zip(&parts.y)
            .zip(&parts.z)
        {
            let x = u.scale(&product.masked) + *x;
            let y = u.scale(&product.y.constant) + *y;
            let z = u.scale(&product.z.constant) + *z;
            (x * y - *u * z).scale_accumulate_into(&mut out, *weight);
        }
        out + *u * (u.scale(&self.lambda.constant) + parts.lambda + parts.g)
    }

    /// `Q_μ(0, y)` from the linear parts of `y`.
    pub fn quadratic_at_zero(&self, y: &LinearParts<F>) -> ChallengeRing<F> {
        let mut out = CyclotomicRing::zero();
        for ((weight, x), yy) in self.product_weights.iter().zip(&y.x).zip(&y.y) {
            (*x * *yy).scale_accumulate_into(&mut out, *weight);
        }
        out
    }

    /// The polar form `B_μ((1, s), (0, y))`.
    pub fn polar(&self, s: &LinearParts<F>, y: &LinearParts<F>) -> ChallengeRing<F> {
        let mut out = y.lambda + y.g;
        for ((((((product, weight), xs), ys), xy), yy), zy) in self
            .relation
            .products
            .iter()
            .zip(&self.product_weights)
            .zip(&s.x)
            .zip(&s.y)
            .zip(&y.x)
            .zip(&y.y)
            .zip(&y.z)
        {
            let x1 = constant_ring(product.masked) + *xs;
            let y1 = constant_ring(product.y.constant) + *ys;
            (x1 * *yy + *xy * y1 - *zy).scale_accumulate_into(&mut out, *weight);
        }
        out
    }

    /// `Σ_i μ_i [X_i(u, ·)]·[Y_i(u, ·)]` as a polynomial of degree at most
    /// 126 over `F`, before reduction modulo `X^64 + 1` (the left side of
    /// Whiteout's product-lift equation (4)).
    pub fn product_polynomial(&self, u: &ChallengeRing<F>, parts: &LinearParts<F>) -> Vec<F> {
        let mut out = vec![F::zero(); 2 * CHALLENGE_RING_DIM - 1];
        for (((product, weight), x), y) in self
            .relation
            .products
            .iter()
            .zip(&self.product_weights)
            .zip(&parts.x)
            .zip(&parts.y)
        {
            let x = u.scale(&product.masked) + *x;
            let y = u.scale(&product.y.constant) + *y;
            for (i, a) in x.coefficients().iter().enumerate() {
                let scaled = *a * *weight;
                for (dst, b) in out.iter_mut().skip(i).zip(y.coefficients()) {
                    *dst += scaled * *b;
                }
            }
        }
        out
    }
}

/// `c` as a constant polynomial.
pub fn constant_ring<F: JoltField>(c: F) -> ChallengeRing<F> {
    let mut ring = CyclotomicRing::zero();
    if let Some(first) = ring.coefficients_mut().first_mut() {
        *first = c;
    }
    ring
}

/// The weights of rows, from their variables' places.
struct LiftContext<'a, F: JoltField> {
    placement: &'a PadPlacement,
    pad_ring: usize,
    operand_ring: usize,
    /// Per evaluation claim: `(ring, position, weight)` over `w`.
    claims: Vec<Vec<(usize, usize, F)>>,
}

impl<F: JoltField> LiftContext<'_, F> {
    fn weights(&self, row: &AffineRow<F>) -> Result<Weights<F>, WhiteoutError> {
        let mut weights = Weights::new(row.constant);
        for &(variable, coefficient) in &row.terms {
            match variable {
                ResidualVar::Pad(pad) => match self.placement.place(pad)? {
                    PadPlace::Regular { group, position } => {
                        let radix = F::from_u64(1 << PAD_LOG_BASIS);
                        let mut power = coefficient;
                        for level in 0..PAD_LEVELS {
                            weights.add(
                                self.pad_ring + group * PAD_LEVELS + level,
                                position,
                                power,
                            )?;
                            power *= radix;
                        }
                    }
                    PadPlace::Operand(operand) => {
                        let radix = F::from_u64(1 << OPERAND_LOG_BASIS);
                        let mut power = coefficient;
                        for level in 0..OPERAND_LEVELS {
                            weights.add(
                                self.operand_ring + operand * OPERAND_LEVELS + level,
                                0,
                                power,
                            )?;
                            power *= radix;
                        }
                    }
                },
                ResidualVar::Witness(claim) => {
                    for &(ring, position, weight) in self
                        .claims
                        .get(claim)
                        .ok_or(WhiteoutError::InvalidRelation("witness claim missing"))?
                    {
                        weights.add(ring, position, weight * coefficient)?;
                    }
                }
            }
        }
        Ok(weights)
    }
}

/// The places of `w`'s coefficients in the challenge-ring view.
struct WitnessIndex {
    /// `(ring, position)` per coefficient of `w`.
    places: Vec<(usize, usize)>,
}

impl WitnessIndex {
    fn new(table: &SegmentTable, starts: &[(usize, usize)]) -> Result<Self, WhiteoutError> {
        let mut places = Vec::new();
        for (segment, &(_, ring_start)) in table.segments().iter().zip(starts) {
            if segment.role != SegmentRole::Witness {
                continue;
            }
            let stride = segment.native_dim / CHALLENGE_RING_DIM;
            for coefficient in 0..segment.num_coefficients() {
                let (element, offset) = (
                    coefficient / segment.native_dim,
                    coefficient % segment.native_dim,
                );
                places.push((
                    ring_start + element * stride + offset % stride,
                    offset / stride,
                ));
            }
        }
        Ok(Self { places })
    }

    fn weights<F: JoltField>(&self, point: &[F]) -> Result<Vec<(usize, usize, F)>, WhiteoutError> {
        if self.places.len() > 1usize.checked_shl(point.len() as u32).unwrap_or(0) {
            return Err(WhiteoutError::InvalidRelation(
                "evaluation point too short for the witness",
            ));
        }
        let eq = EqPolynomial::new(point.to_vec()).evaluations();
        Ok(self
            .places
            .iter()
            .zip(eq)
            .map(|(&(ring, position), weight)| (ring, position, weight))
            .collect())
    }
}
