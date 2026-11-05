//! The constant-term lift of the residual relation (Whiteout
//! Lemma `wo:lem:ct-lift`, Definition `wo:def:residual`).
//!
//! Every field-level row becomes the constant coefficient of an `R`-linear
//! form in `(u, s)` over the challenge ring `R = F[X]/(X^64 + 1)`: a
//! functional `ℓ` with coefficient vector `ℓ̂` lifts to
//! `L(u, s) = Σ σ(ℓ̂_k) s_k + u·b`, whose constant coefficient is
//! `ℓ(s) + u·b`. A product row `Σ_k x_k·y_k = z` becomes
//! `Σ_k X_k·Y_k − u·Z` with `X_k = u·x̂_k − P_{x_k}(s)` the constant operands.
//! Its pairs are flattened into product entries that share the row's weight;
//! only the row's first entry carries `Z`. [`AggregatedRelation`] combines
//! every row under the weights `μ` of [`RowLabels`] into the quadratic `Q_μ`
//! of Protocol `wo:prot:sigma`.
//!
//! An evaluation claim `w̃(r)` stays symbolic in the relation: the verifier
//! holds only its point. Its weights `eq(r, n)` over `w`'s native index
//! `n = E·d′ + m·i + k` factor as `eq(r_E, E)·eq(r_i, i)·eq(r_k, k)`, so its
//! lift is `σ(Ê)·Σ_{E,k} eq(r_E,E) eq(r_k,k) w_{E,k}` with `Ê = eq(r_i, ·)`:
//! one weighted sum over `w` and one ring product. Only the prover (and the
//! statement's direct check) expands it, through [`ClaimTables`].

use std::collections::BTreeMap;

use jolt_algebra::{lift_biased, CyclotomicRing, JoltField, LIFT_BIAS};
use jolt_crypto::lattice::CHALLENGE_RING_DIM;
use jolt_poly::EqPolynomial;
use rayon::prelude::*;

use super::labels::RowLabels;
use super::placement::{
    PadPlace, PadPlacement, OPERAND_LEVELS, OPERAND_LOG_BASIS, PAD_LEVELS, PAD_LOG_BASIS,
};
use super::segments::{Placed, SegmentMask, SegmentRole, SegmentTable};
use super::WhiteoutError;
use crate::masking::{AffineRow, ResidualRelation, ResidualVar};
use crate::secret::SecretVec;

/// An element of the challenge ring.
pub type ChallengeRing<F> = CyclotomicRing<F, CHALLENGE_RING_DIM>;

/// Index bits of the in-ring position `i`.
const RING_BITS: usize = CHALLENGE_RING_DIM.trailing_zeros() as usize;

/// The evaluation claims of the residual commitment that
/// `ResidualVar::Witness(j)` names: claim `j` is `w̃(r_j)`, the multilinear
/// extension of `w` (the `Witness` segments of `s`, in table order and native
/// coefficient order, zero-padded to a power of two) at `points[j]`, with the
/// first point coordinate the most significant index bit. Every point has
/// exactly `⌈log₂ |w|⌉` coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluationClaims<F> {
    pub points: Vec<Vec<F>>,
}

/// A linear form being built: dense weights `ℓ̂_k` per ring, symbolic claim
/// coefficients, and a constant.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Weights<F: JoltField> {
    rings: BTreeMap<usize, [F; CHALLENGE_RING_DIM]>,
    claims: BTreeMap<usize, F>,
    constant: F,
}

impl<F: JoltField> Weights<F> {
    fn new(constant: F) -> Self {
        Self {
            rings: BTreeMap::new(),
            claims: BTreeMap::new(),
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
        for (&claim, &coefficient) in &other.claims {
            *self.claims.entry(claim).or_insert_with(F::zero) += coefficient * scale;
        }
        self.constant += other.constant * scale;
    }

    #[cfg(feature = "whiteout-prover")]
    /// The row's field value on `x`: the constant term of its `s`-part plus
    /// its constant. Dot products only, no ring products.
    fn value(
        &self,
        table: &SegmentTable,
        x: &[i64],
        claim_values: &[F],
    ) -> Result<F, WhiteoutError> {
        let mut value = self.constant;
        for (ring, weights) in &self.rings {
            value += table
                .ring::<F>(x, *ring)?
                .coefficients()
                .iter()
                .zip(weights)
                .map(|(a, b)| *a * *b)
                .sum::<F>();
        }
        for (claim, coefficient) in &self.claims {
            value += *coefficient
                * *claim_values
                    .get(*claim)
                    .ok_or(WhiteoutError::InvalidRelation("claim outside the relation"))?;
        }
        Ok(value)
    }

    fn form(&self) -> Form<F> {
        Form {
            dense: self
                .rings
                .iter()
                .map(|(&ring, coefficients)| {
                    (
                        ring,
                        *coefficients,
                        CyclotomicRing::from_coefficients(*coefficients).sigma_m1(),
                    )
                })
                .collect(),
            claims: self.claims.iter().map(|(&j, &c)| (j, c)).collect(),
            constant: self.constant,
        }
    }
}

/// A form's dense term `(ring, ℓ̂, σ(ℓ̂))`.
pub(crate) type DenseTerm<F> = (usize, [F; CHALLENGE_RING_DIM], ChallengeRing<F>);

/// `L(u, s) = Σ_k σ(ℓ̂_k)·s_k + Σ_j a_j·σ(Ê_j)·S_j(s) + u·b`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Form<F: JoltField> {
    dense: Vec<DenseTerm<F>>,
    claims: Vec<(usize, F)>,
    constant: F,
}

impl<F: JoltField> Form<F> {
    /// The `s`-part on `x`, given the claim terms `T_j = σ(Ê_j)·S_j(x)`.
    fn linear<X: SValues<F> + ?Sized>(
        &self,
        table: &SegmentTable,
        x: &X,
        claims: &[ChallengeRing<F>],
    ) -> Result<ChallengeRing<F>, WhiteoutError> {
        let mut out = x.dense_sum(table, &self.dense)?;
        for (claim, coefficient) in &self.claims {
            claims
                .get(*claim)
                .ok_or(WhiteoutError::InvalidRelation("claim outside the relation"))?
                .scale_accumulate_into(&mut out, *coefficient);
        }
        Ok(out)
    }
}

impl<F: JoltField> Form<F> {
    /// Adds to `out` (laid out like `s`) the weights of `⟨ω, linear(x)⟩`:
    /// the transpose of multiplication by `m` is multiplication by `σ(m)`
    /// under `σ`, so a dense ring `k` takes `ω·ℓ̂_k` and claim `j` takes
    /// `ω·Ê_j` on `S_j`.
    fn adjoint(
        &self,
        table: &SegmentTable,
        tables: &ClaimTables<F>,
        weight: &ChallengeRing<F>,
        out: &mut [F],
    ) -> Result<(), WhiteoutError> {
        for (ring, hat, _) in &self.dense {
            table.scatter_ring(
                out,
                *ring,
                &(*weight * CyclotomicRing::from_coefficients(*hat)),
            )?;
        }
        for (claim, coefficient) in &self.claims {
            tables.adjoint(*claim, &weight.scale(coefficient), out)?;
        }
        Ok(())
    }
}

/// `Σ_j scale_j · s_{k_j}`: an `R`-linear reconstruction, such as `P_x` or g.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Reconstruction<F> {
    terms: Vec<(usize, F)>,
}

impl<F: JoltField> Reconstruction<F> {
    fn digits(rings: impl Iterator<Item = usize>, log_basis: u32) -> Self {
        let radix = F::from_u64(1 << log_basis);
        let mut power = F::one();
        let terms = rings
            .map(|ring| {
                let term = (ring, power);
                power *= radix;
                term
            })
            .collect();
        Self { terms }
    }

    /// Adds to `out` the weights of `⟨ω, evaluate(x)⟩`.
    fn adjoint(
        &self,
        table: &SegmentTable,
        weight: &ChallengeRing<F>,
        out: &mut [F],
    ) -> Result<(), WhiteoutError> {
        for (ring, scale) in &self.terms {
            table.scatter_ring(out, *ring, &weight.scale(scale))?;
        }
        Ok(())
    }

    fn evaluate<X: SValues<F> + ?Sized>(
        &self,
        table: &SegmentTable,
        x: &X,
    ) -> Result<ChallengeRing<F>, WhiteoutError> {
        let mut out = CyclotomicRing::zero();
        for (ring, scale) in &self.terms {
            x.ring(table, *ring)?
                .scale_accumulate_into(&mut out, *scale);
        }
        Ok(out)
    }
}

/// One pair of a product row, with the row's `Z` on its first pair and zero
/// on the others.
#[derive(Clone, Debug, PartialEq, Eq)]
struct LiftedProduct<F: JoltField> {
    /// The product row this pair belongs to.
    row: usize,
    masked: F,
    operand: usize,
    y: Weights<F>,
    z: Weights<F>,
    y_form: Form<F>,
    z_form: Form<F>,
}

/// The residual relation lifted to the challenge ring over one segment
/// table. Built by `SigmaSetup::lift`, which binds it to the setup's table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiftedRelation<F: JoltField> {
    table: SegmentTable,
    labels: RowLabels,
    affine: Vec<Weights<F>>,
    products: Vec<LiftedProduct<F>>,
    /// `P_x` per operand, and its digit rings.
    operands: Vec<Reconstruction<F>>,
    operand_rings: Vec<Vec<usize>>,
    g: Reconstruction<F>,
    claims: EvaluationClaims<F>,
}

impl<F: JoltField> LiftedRelation<F> {
    /// Lifts `relation` over `table`, with its pads placed by `placement`
    /// and its witness variables naming `claims`. Release checks: the
    /// relation has the placement's layout, every product's operand is an
    /// operand pad, every variable exists, and every claim point has
    /// `⌈log₂ |w|⌉` coordinates. Does no work proportional to `w`.
    pub(crate) fn new(
        relation: &ResidualRelation<F>,
        placement: &PadPlacement,
        table: &SegmentTable,
        claims: &EvaluationClaims<F>,
    ) -> Result<Self, WhiteoutError> {
        placement.check_relation(relation)?;
        if relation.num_witness != claims.points.len() {
            return Err(WhiteoutError::InvalidRelation(
                "witness variables differ from the evaluation claims",
            ));
        }
        let witness = WitnessShape::new(table)?;
        if claims
            .points
            .iter()
            .any(|point| point.len() != witness.num_vars)
        {
            return Err(WhiteoutError::InvalidRelation(
                "evaluation point length differs from ⌈log₂ |w|⌉",
            ));
        }
        let pad = table
            .unique(SegmentRole::PadDigits)
            .ok_or(WhiteoutError::InvalidRelation("pad digit segment missing"))?;
        if pad.segment.len < placement.pad_rings() {
            return Err(WhiteoutError::InvalidRelation(
                "pad digit segment too short",
            ));
        }
        let operand_ring =
            if placement.num_operands() == 0 {
                0
            } else {
                let operand = table.unique(SegmentRole::OperandDigits).ok_or(
                    WhiteoutError::InvalidRelation("operand digit segment missing"),
                )?;
                if operand.segment.len < placement.operand_rings() {
                    return Err(WhiteoutError::InvalidRelation(
                        "operand digit segment too short",
                    ));
                }
                operand.ring
            };
        let context = LiftContext {
            placement,
            pad_ring: pad.ring,
            operand_ring,
            num_claims: claims.points.len(),
        };
        let affine = relation
            .affine
            .iter()
            .map(|row| context.weights(row))
            .collect::<Result<Vec<_>, _>>()?;
        let mut products = Vec::new();
        for (index, row) in relation.products.iter().enumerate() {
            let z = context.weights(&row.z)?;
            for (k, pair) in row.pairs().enumerate() {
                let not_an_operand =
                    WhiteoutError::InvalidRelation("a product's operand is not an operand pad");
                let x = pair.x.masked().ok_or(not_an_operand.clone())?;
                let PadPlace::Operand(operand) = placement.place(x.pad)? else {
                    return Err(not_an_operand);
                };
                let y = context.weights(&pair.y)?;
                let z = if k == 0 {
                    z.clone()
                } else {
                    Weights::new(F::zero())
                };
                products.push(LiftedProduct {
                    row: index,
                    masked: x.masked,
                    operand,
                    y_form: y.form(),
                    z_form: z.form(),
                    y,
                    z,
                });
            }
        }
        let operand_rings: Vec<Vec<usize>> = (0..placement.num_operands())
            .map(|operand| {
                (0..OPERAND_LEVELS)
                    .map(|level| operand_ring + PadPlacement::operand_ring(operand, level))
                    .collect()
            })
            .collect();
        Ok(Self {
            table: table.clone(),
            labels: RowLabels::new(
                placement.num_operands(),
                relation.products.len(),
                relation.affine.len(),
            )?,
            affine,
            products,
            operands: operand_rings
                .iter()
                .map(|rings| Reconstruction::digits(rings.iter().copied(), OPERAND_LOG_BASIS))
                .collect(),
            operand_rings,
            g: Reconstruction::digits(
                (0..PAD_LEVELS).map(|level| pad.ring + placement.g_ring(level)),
                PAD_LOG_BASIS,
            ),
            claims: claims.clone(),
        })
    }

    pub(crate) fn num_products(&self) -> usize {
        self.products.len()
    }

    pub(crate) fn labels(&self) -> &RowLabels {
        &self.labels
    }

    pub(crate) fn table(&self) -> &SegmentTable {
        &self.table
    }

    /// The prover's expansion of the evaluation claims: `O(|w|)` work and
    /// memory, never done by the verifier.
    pub(crate) fn claim_tables(&self) -> Result<ClaimTables<F>, WhiteoutError> {
        let witness = WitnessShape::new(&self.table)?;
        let tables = self
            .claims
            .points
            .iter()
            .map(|point| witness.claim(point))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ClaimTables { witness, tables })
    }

    /// Combines every row under the weights `μ` from the aggregation
    /// challenges `rho`.
    pub(crate) fn aggregate(&self, rho: &[F]) -> Result<AggregatedRelation<'_, F>, WhiteoutError> {
        let mut lambda = Weights::new(F::zero());
        for (index, row) in self.affine.iter().enumerate() {
            lambda.add_scaled(row, self.labels.weight(self.labels.affine(index), rho)?);
        }
        // Scalarity rows: coefficient r ∈ [1, 63] of P_x vanishes. Their sum
        // over r puts weight Σ_r μ_{64o+r} 256^j X^r on digit ring j. The
        // labels 64o and r < 64 have disjoint bits, so μ_{64o+r} = μ_{64o}·μ_r.
        let radix = F::from_u64(1 << OPERAND_LOG_BASIS);
        let low = (0..CHALLENGE_RING_DIM)
            .map(|r| self.labels.weight(r, rho))
            .collect::<Result<Vec<_>, _>>()?;
        for (operand, rings) in self.operand_rings.iter().enumerate() {
            let mut power = self.labels.weight(RowLabels::scalarity(operand, 0), rho)?;
            for &ring in rings {
                let slot = lambda
                    .rings
                    .entry(ring)
                    .or_insert([F::zero(); CHALLENGE_RING_DIM]);
                for (dst, weight) in slot.iter_mut().zip(&low).skip(1) {
                    *dst += *weight * power;
                }
                power *= radix;
            }
        }
        let product_weights = self
            .products
            .iter()
            .map(|product| self.labels.weight(self.labels.product(product.row), rho))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(AggregatedRelation {
            relation: self,
            product_weights,
            lambda: lambda.form(),
        })
    }

    #[cfg(feature = "whiteout-prover")]
    /// The field value of every row on `s`, in the order affine rows,
    /// scalarity rows, product rows, then `ct(g)`; the index of the first
    /// nonzero one, if any. The prover's per-row check before the
    /// aggregation challenges: constant terms only.
    pub(crate) fn first_violation(
        &self,
        tables: &ClaimTables<F>,
        s: &[i64],
    ) -> Result<Option<usize>, WhiteoutError> {
        let claims = tables.values(s)?;
        let mut index = 0;
        for row in &self.affine {
            if !row.value(&self.table, s, &claims)?.is_zero() {
                return Ok(Some(index));
            }
            index += 1;
        }
        for operand in &self.operands {
            let p = operand.evaluate(&self.table, s)?;
            for coefficient in p.coefficients().iter().skip(1) {
                if !coefficient.is_zero() {
                    return Ok(Some(index));
                }
                index += 1;
            }
        }
        let mut entries = self.products.iter().peekable();
        while let Some(first) = entries.next() {
            let mut value = F::zero();
            let mut product = first;
            loop {
                let x = product.masked
                    - self
                        .operand(product.operand)?
                        .evaluate(&self.table, s)?
                        .constant_term();
                value += x * product.y.value(&self.table, s, &claims)?
                    - product.z.value(&self.table, s, &claims)?;
                match entries.next_if(|next| next.row == first.row) {
                    Some(next) => product = next,
                    None => break,
                }
            }
            if !value.is_zero() {
                return Ok(Some(index));
            }
            index += 1;
        }
        if !self.g.evaluate(&self.table, s)?.constant_term().is_zero() {
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

/// A vector laid out like `s` that the lifted forms read: the prover's
/// integers, or the verifier's factored `eq` dual (`weights.rs`).
pub(crate) trait SValues<F: JoltField>: Sync {
    /// Challenge-ring element `ring` ([`SegmentTable::ring`]).
    fn ring(&self, table: &SegmentTable, ring: usize) -> Result<ChallengeRing<F>, WhiteoutError>;

    /// `Σ_k σ(ℓ̂_k)·x_k` over a form's dense terms `(k, ℓ̂_k, σ(ℓ̂_k))`.
    ///
    /// WARNING: `mul_accumulate_into` skips zero coefficients of its
    /// receiver, so the public weights `σ(ℓ̂)` must stay on that side: with
    /// the secret ring there, the time would depend on its zeros.
    fn dense_sum(
        &self,
        table: &SegmentTable,
        dense: &[DenseTerm<F>],
    ) -> Result<ChallengeRing<F>, WhiteoutError> {
        dense
            .par_iter()
            .map(|(ring, _, weights)| {
                let mut product = CyclotomicRing::zero();
                weights.mul_accumulate_into(&self.ring(table, *ring)?, &mut product);
                Ok::<_, WhiteoutError>(product)
            })
            .try_reduce(CyclotomicRing::zero, |a, b| Ok(a + b))
    }

    /// `S(x) = Σ_{E,k} eq(r_E,E) eq(r_k,k) x_{E,k}` for one claim, over the
    /// witness segments; public segments are skipped when `skip_public` (a
    /// public decision: the mask is zero there).
    fn claim_sum(
        &self,
        witness: &WitnessShape,
        table: &ClaimTable<F>,
        skip_public: bool,
    ) -> Result<[F; CHALLENGE_RING_DIM], WhiteoutError>;
}

impl<F: JoltField> SValues<F> for [i64] {
    fn ring(&self, table: &SegmentTable, ring: usize) -> Result<ChallengeRing<F>, WhiteoutError> {
        table.ring(self, ring)
    }

    fn claim_sum(
        &self,
        witness: &WitnessShape,
        table: &ClaimTable<F>,
        skip_public: bool,
    ) -> Result<[F; CHALLENGE_RING_DIM], WhiteoutError> {
        let x = self;
        let stride = witness.native_dim / CHALLENGE_RING_DIM;
        let mut element_offset = 0usize;
        let mut sum = [F::zero(); CHALLENGE_RING_DIM];
        for placed in &witness.segments {
            let elements = placed.segment.num_coefficients() / witness.native_dim;
            let skip = skip_public && placed.segment.mask == SegmentMask::Public;
            if !skip {
                let values = x
                    .get(placed.coefficients())
                    .ok_or(WhiteoutError::InvalidRelation("witness outside s"))?;
                let weights = table
                    .element
                    .get(element_offset..element_offset + elements)
                    .ok_or(WhiteoutError::InvalidRelation("claim shorter than w"))?;
                let partial = values
                    .par_chunks_exact(witness.native_dim)
                    .zip(weights.par_iter())
                    .fold(
                        || [F::zero(); CHALLENGE_RING_DIM],
                        |mut acc, (element, &outer)| {
                            for (k, &inner) in table.component.iter().enumerate() {
                                let weight = outer * inner;
                                let bias = weight * F::from_u64(LIFT_BIAS);
                                for (dst, &value) in
                                    acc.iter_mut().zip(element.iter().skip(k).step_by(stride))
                                {
                                    *dst += weight * lift_biased::<F>(value) - bias;
                                }
                            }
                            acc
                        },
                    )
                    .reduce(
                        || [F::zero(); CHALLENGE_RING_DIM],
                        |mut a, b| {
                            for (x, y) in a.iter_mut().zip(b) {
                                *x += y;
                            }
                            a
                        },
                    );
                for (dst, value) in sum.iter_mut().zip(partial) {
                    *dst += value;
                }
            }
            element_offset += elements;
        }
        Ok(sum)
    }
}

/// The witness part of `s`: its segments and the claim index geometry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WitnessShape {
    pub(crate) segments: Vec<Placed>,
    pub(crate) native_dim: usize,
    num_vars: usize,
}

impl WitnessShape {
    fn new(table: &SegmentTable) -> Result<Self, WhiteoutError> {
        let segments: Vec<Placed> = table
            .placed()
            .iter()
            .filter(|placed| placed.segment.role == SegmentRole::Witness)
            .copied()
            .collect();
        let native_dim = segments
            .first()
            .map_or(CHALLENGE_RING_DIM, |p| p.segment.native_dim);
        if segments.iter().any(|p| p.segment.native_dim != native_dim) {
            return Err(WhiteoutError::InvalidRelation(
                "witness segments must share one native dimension",
            ));
        }
        let len: usize = segments.iter().map(|p| p.segment.num_coefficients()).sum();
        let num_vars = (len.max(native_dim)).next_power_of_two().trailing_zeros() as usize;
        Ok(Self {
            segments,
            native_dim,
            num_vars,
        })
    }

    /// The factored weights of the claim at `point`.
    fn claim<F: JoltField>(&self, point: &[F]) -> Result<ClaimTable<F>, WhiteoutError> {
        let stride_bits = (self.native_dim / CHALLENGE_RING_DIM).trailing_zeros() as usize;
        let element_bits = self.num_vars.checked_sub(RING_BITS + stride_bits).ok_or(
            WhiteoutError::InvalidRelation("witness shorter than one native element"),
        )?;
        let (element, rest) = point.split_at(element_bits.min(point.len()));
        let (position, component) = rest.split_at(RING_BITS.min(rest.len()));
        let position: [F; CHALLENGE_RING_DIM] = EqPolynomial::new(position.to_vec())
            .evaluations()
            .try_into()
            .map_err(|_| WhiteoutError::InvalidRelation("claim point split"))?;
        Ok(ClaimTable {
            element: EqPolynomial::new(element.to_vec()).evaluations(),
            component: EqPolynomial::new(component.to_vec()).evaluations(),
            position,
            sigma: CyclotomicRing::from_coefficients(position).sigma_m1(),
        })
    }
}

/// One claim's factored weights.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ClaimTable<F: JoltField> {
    /// `eq(r_E, E)` per native element of `w`.
    pub(crate) element: Vec<F>,
    /// `eq(r_k, k)` per stride component.
    pub(crate) component: Vec<F>,
    /// `Ê = eq(r_i, ·)` and `σ(Ê)`.
    position: [F; CHALLENGE_RING_DIM],
    sigma: ChallengeRing<F>,
}

/// The prover's expanded evaluation claims.
pub(crate) struct ClaimTables<F: JoltField> {
    witness: WitnessShape,
    tables: Vec<ClaimTable<F>>,
}

impl<F: JoltField> ClaimTables<F> {
    /// `S_j(x) = Σ_{E,k} eq(r_E,E) eq(r_k,k) x_{E,k}` for every claim, over the
    /// witness segments of `x`; public segments are skipped when
    /// `skip_public` (a public decision: the mask is zero there).
    fn sums<X: SValues<F> + ?Sized>(
        &self,
        x: &X,
        skip_public: bool,
    ) -> Result<Vec<[F; CHALLENGE_RING_DIM]>, WhiteoutError> {
        self.tables
            .iter()
            .map(|table| x.claim_sum(&self.witness, table, skip_public))
            .collect()
    }

    /// Adds to `out` (laid out like `s`) the weights of `⟨ω, σ(Ê_j)·S_j(x)⟩`
    /// for claim `j`, given `weight = ω·Ê_j`.
    fn adjoint(
        &self,
        claim: usize,
        weight: &ChallengeRing<F>,
        out: &mut [F],
    ) -> Result<(), WhiteoutError> {
        let table = self
            .tables
            .get(claim)
            .ok_or(WhiteoutError::InvalidRelation("claim outside the relation"))?;
        let weight = *weight * CyclotomicRing::from_coefficients(table.position);
        let stride = self.witness.native_dim / CHALLENGE_RING_DIM;
        let mut element_offset = 0usize;
        for placed in &self.witness.segments {
            let elements = placed.segment.num_coefficients() / self.witness.native_dim;
            let targets = out
                .get_mut(placed.coefficients())
                .ok_or(WhiteoutError::InvalidRelation("witness outside s"))?;
            let outer = table
                .element
                .get(element_offset..element_offset + elements)
                .ok_or(WhiteoutError::InvalidRelation("claim shorter than w"))?;
            targets
                .par_chunks_exact_mut(self.witness.native_dim)
                .zip(outer.par_iter())
                .for_each(|(element, &outer)| {
                    for (k, &inner) in table.component.iter().enumerate() {
                        let scale = outer * inner;
                        for (dst, w) in element
                            .iter_mut()
                            .skip(k)
                            .step_by(stride)
                            .zip(weight.coefficients())
                        {
                            *dst += scale * *w;
                        }
                    }
                });
            element_offset += elements;
        }
        Ok(())
    }

    /// The claim terms `T_j = σ(Ê_j)·S_j(x)`.
    fn terms<X: SValues<F> + ?Sized>(
        &self,
        x: &X,
        skip_public: bool,
    ) -> Result<Vec<ChallengeRing<F>>, WhiteoutError> {
        Ok(self
            .sums(x, skip_public)?
            .into_iter()
            .zip(&self.tables)
            .map(|(sum, table)| table.sigma * CyclotomicRing::from_coefficients(sum))
            .collect())
    }

    #[cfg(feature = "whiteout-prover")]
    /// The claim values `w̃(r_j) = ⟨Ê_j, S_j⟩` on `x`.
    pub(crate) fn values(&self, x: &[i64]) -> Result<Vec<F>, WhiteoutError> {
        Ok(self
            .sums(x, false)?
            .into_iter()
            .zip(&self.tables)
            .map(|(sum, table)| sum.iter().zip(&table.position).map(|(a, b)| *a * *b).sum())
            .collect())
    }
}

/// Ring weights for [`AggregatedRelation::product_adjoint`].
pub(crate) struct ProductAdjointWeights<'a, F: JoltField> {
    pub target: ChallengeRing<F>,
    pub x: &'a [ChallengeRing<F>],
    pub y: &'a [ChallengeRing<F>],
}

/// `Q_μ` and its pieces.
pub(crate) struct AggregatedRelation<'a, F: JoltField> {
    relation: &'a LiftedRelation<F>,
    product_weights: Vec<F>,
    lambda: Form<F>,
}

/// The `s`-linear parts of every form at one vector: `u = 0`. As secret as
/// the vector (`s`, a mask or a response), so the per-product parts are
/// zeroed when dropped.
/// A term of the reduced target's `s`-linear part.
#[derive(Clone, Copy)]
enum TargetTerm {
    Z(usize),
    Lambda,
    G,
}

pub(crate) struct LinearParts<F: JoltField> {
    /// `−P_x` per product.
    x: SecretVec<ChallengeRing<F>>,
    y: SecretVec<ChallengeRing<F>>,
    z: SecretVec<ChallengeRing<F>>,
    lambda: ChallengeRing<F>,
    g: ChallengeRing<F>,
}

impl<F: JoltField> AggregatedRelation<'_, F> {
    /// The linear parts at `x` (laid out like `s`). With `skip_public`, the
    /// witness claims skip public segments, where a mask is zero.
    pub(crate) fn linear_parts<X: SValues<F> + ?Sized>(
        &self,
        tables: &ClaimTables<F>,
        x: &X,
        skip_public: bool,
    ) -> Result<LinearParts<F>, WhiteoutError> {
        let table = &self.relation.table;
        let claims = SecretVec::from(tables.terms(x, skip_public)?);
        let products = self.relation.products.len();
        let zeros = || SecretVec::from(vec![CyclotomicRing::zero(); products]);
        let (mut parts_x, mut parts_y, mut parts_z) = (zeros(), zeros(), zeros());
        self.relation
            .products
            .par_iter()
            .zip(parts_x.par_iter_mut())
            .zip(parts_y.par_iter_mut())
            .zip(parts_z.par_iter_mut())
            .try_for_each(|(((product, px), py), pz)| {
                *px = -self.relation.operand(product.operand)?.evaluate(table, x)?;
                *py = product.y_form.linear(table, x, &claims)?;
                *pz = product.z_form.linear(table, x, &claims)?;
                Ok::<_, WhiteoutError>(())
            })?;
        Ok(LinearParts {
            x: parts_x,
            y: parts_y,
            z: parts_z,
            lambda: self.lambda.linear(table, x, &claims)?,
            g: self.relation.g.evaluate(table, x)?,
        })
    }

    /// `Q_μ(u, ·)` from the linear parts of `·`.
    pub(crate) fn quadratic(
        &self,
        u: &ChallengeRing<F>,
        parts: &LinearParts<F>,
    ) -> ChallengeRing<F> {
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

    #[cfg(feature = "whiteout-prover")]
    /// `Q_μ(0, y)` from the linear parts of `y`.
    pub(crate) fn quadratic_at_zero(&self, y: &LinearParts<F>) -> ChallengeRing<F> {
        let mut out = CyclotomicRing::zero();
        for ((weight, x), yy) in self.product_weights.iter().zip(&y.x).zip(&y.y) {
            (*x * *yy).scale_accumulate_into(&mut out, *weight);
        }
        out
    }

    #[cfg(feature = "whiteout-prover")]
    /// The polar form `B_μ((1, s), (0, y))`.
    pub(crate) fn polar(&self, s: &LinearParts<F>, y: &LinearParts<F>) -> ChallengeRing<F> {
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

    /// `(μ_i X_i(u, ·), Y_i(u, ·))` for every product, the factors of
    /// Whiteout's product-lift equation (4).
    pub(crate) fn product_operands(
        &self,
        u: &ChallengeRing<F>,
        parts: &LinearParts<F>,
    ) -> Vec<(ChallengeRing<F>, ChallengeRing<F>)> {
        self.relation
            .products
            .iter()
            .zip(&self.product_weights)
            .zip(&parts.x)
            .zip(&parts.y)
            .map(|(((product, weight), x), y)| {
                (
                    (u.scale(&product.masked) + *x).scale(weight),
                    u.scale(&product.y.constant) + *y,
                )
            })
            .collect()
    }

    /// `Σ_i μ_i [X_i(u, ·)]·[Y_i(u, ·)]` as a polynomial of degree at most
    /// 126 over `F`, before reduction modulo `X^64 + 1` (the left side of
    /// Whiteout's product-lift equation (4)).
    pub(crate) fn product_polynomial(
        &self,
        u: &ChallengeRing<F>,
        parts: &LinearParts<F>,
    ) -> Vec<F> {
        let mut out = vec![F::zero(); 2 * CHALLENGE_RING_DIM - 1];
        for (x, y) in self.product_operands(u, parts) {
            for (i, a) in x.coefficients().iter().enumerate() {
                for (dst, b) in out.iter_mut().skip(i).zip(y.coefficients()) {
                    *dst += *a * *b;
                }
            }
        }
        out
    }

    /// The `s`-linear part of `reduce(Σ μ_i X_i Y_i) − Q_μ(u, ·)` from the
    /// linear parts of `·`: `Σ μ_i u Z_i − u(λ + g)`, whose transpose is
    /// [`Self::product_adjoint`]'s first term.
    pub(crate) fn target_linear(
        &self,
        u: &ChallengeRing<F>,
        parts: &LinearParts<F>,
    ) -> Result<ChallengeRing<F>, WhiteoutError> {
        let mut sum = CyclotomicRing::zero();
        for (term, weight) in self.target_terms() {
            let part = match term {
                TargetTerm::Z(i) => parts
                    .z
                    .get(i)
                    .ok_or(WhiteoutError::InvalidRelation("product parts"))?,
                TargetTerm::Lambda => &parts.lambda,
                TargetTerm::G => &parts.g,
            };
            part.scale_accumulate_into(&mut sum, weight);
        }
        Ok(*u * sum)
    }

    /// `reduce(Σ μ_i X_i Y_i) − Q_μ(u, 0)`, the batched row's constant term.
    ///
    /// [`Self::quadratic`] owns `Q_μ`, and [`Self::product_operands`] the
    /// operands; at the zero vector their forms reduce to the constants at
    /// `u`: `X_i = u·m̂_i`, `Y_i = u·ŷ_i`, `Z_i = u·ẑ_i`, `λ = u·λ̂`, `g = 0`.
    /// The products `X_i·Y_i` appear in both terms and cancel, as
    /// [`Self::target_linear`] states for any vector, so
    /// `h = u·(Σ μ_i·u·ẑ_i − u·λ̂)`: [`Self::target_linear`] over those
    /// constants.
    pub(crate) fn target_at_zero(
        &self,
        u: &ChallengeRing<F>,
    ) -> Result<ChallengeRing<F>, WhiteoutError> {
        let mut at_zero = self.zero_parts();
        for (z, product) in at_zero.z.iter_mut().zip(&self.relation.products) {
            *z = u.scale(&product.z.constant);
        }
        at_zero.lambda = u.scale(&self.lambda.constant);
        self.target_linear(u, &at_zero)
    }

    /// The terms of `Σ μ_i Z_i − λ − g` with their weights, which
    /// [`Self::target_linear`] and [`Self::product_adjoint`] both read:
    /// that sum times `u` is the `s`-linear part of the reduced target.
    fn target_terms(&self) -> impl Iterator<Item = (TargetTerm, F)> + '_ {
        self.product_weights
            .iter()
            .enumerate()
            .map(|(i, mu)| (TargetTerm::Z(i), *mu))
            .chain([(TargetTerm::Lambda, -F::one()), (TargetTerm::G, -F::one())])
    }

    /// The linear parts of the zero vector.
    pub(crate) fn zero_parts(&self) -> LinearParts<F> {
        let products = self.relation.products.len();
        LinearParts {
            x: SecretVec::from(vec![CyclotomicRing::zero(); products]),
            y: SecretVec::from(vec![CyclotomicRing::zero(); products]),
            z: SecretVec::from(vec![CyclotomicRing::zero(); products]),
            lambda: CyclotomicRing::zero(),
            g: CyclotomicRing::zero(),
        }
    }

    /// Adds to `out` (laid out like `s`) the weights of the `s`-linear part
    /// of `⟨ω_H, reduce(Σ μ_i X_i Y_i) − Q_μ(u, ·)⟩ + Σ_i ⟨ω_X,i, μ_i X_i⟩ +
    /// Σ_i ⟨ω_Y,i, Y_i⟩`. The first term is `Σ μ_i u Z_i − u(λ + g)`; the
    /// quadratic parts cancel.
    pub(crate) fn product_adjoint(
        &self,
        tables: &ClaimTables<F>,
        u: &ChallengeRing<F>,
        weights: &ProductAdjointWeights<'_, F>,
        out: &mut [F],
    ) -> Result<(), WhiteoutError> {
        let table = &self.relation.table;
        if weights.x.len() != self.relation.products.len()
            || weights.y.len() != self.relation.products.len()
        {
            return Err(WhiteoutError::InvalidRelation(
                "product weights differ from the products",
            ));
        }
        // ω ↦ σ(σ(ω)·u), the transpose of multiplication by u.
        let times_u = (weights.target.sigma_m1() * *u).sigma_m1();
        for (term, weight) in self.target_terms() {
            let scaled = times_u.scale(&weight);
            match term {
                TargetTerm::Z(i) => self
                    .relation
                    .products
                    .get(i)
                    .ok_or(WhiteoutError::InvalidRelation("product parts"))?
                    .z_form
                    .adjoint(table, tables, &scaled, out)?,
                TargetTerm::Lambda => self.lambda.adjoint(table, tables, &scaled, out)?,
                TargetTerm::G => self.relation.g.adjoint(table, &scaled, out)?,
            }
        }
        for (((product, mu), x), y) in self
            .relation
            .products
            .iter()
            .zip(&self.product_weights)
            .zip(weights.x)
            .zip(weights.y)
        {
            self.relation
                .operand(product.operand)?
                .adjoint(table, &(-x.scale(mu)), out)?;
            product.y_form.adjoint(table, tables, y, out)?;
        }
        Ok(())
    }
}

#[cfg(feature = "whiteout-prover")]
/// `c` as a constant polynomial.
pub(crate) fn constant_ring<F: JoltField>(c: F) -> ChallengeRing<F> {
    let mut ring = CyclotomicRing::zero();
    if let Some(first) = ring.coefficients_mut().first_mut() {
        *first = c;
    }
    ring
}

/// The weights of rows, from their variables' places.
struct LiftContext<'a> {
    placement: &'a PadPlacement,
    pad_ring: usize,
    operand_ring: usize,
    num_claims: usize,
}

impl LiftContext<'_> {
    fn weights<F: JoltField>(&self, row: &AffineRow<F>) -> Result<Weights<F>, WhiteoutError> {
        let mut weights = Weights::new(row.constant);
        for &(variable, coefficient) in &row.terms {
            match variable {
                ResidualVar::Pad(pad) => match self.placement.place(pad)? {
                    PadPlace::Regular { group, position } => {
                        let radix = F::from_u64(1 << PAD_LOG_BASIS);
                        let mut power = coefficient;
                        for level in 0..PAD_LEVELS {
                            weights.add(
                                self.pad_ring + PadPlacement::regular_ring(group, level),
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
                                self.operand_ring + PadPlacement::operand_ring(operand, level),
                                0,
                                power,
                            )?;
                            power *= radix;
                        }
                    }
                },
                ResidualVar::Witness(claim) => {
                    if claim >= self.num_claims {
                        return Err(WhiteoutError::InvalidRelation("witness claim missing"));
                    }
                    *weights.claims.entry(claim).or_insert_with(F::zero) += coefficient;
                }
            }
        }
        Ok(weights)
    }
}
