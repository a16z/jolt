//! Lowering verifier checks on masked messages.
//!
//! [`lower_factored`] turns a factored claim formula into a value of a
//! [`ClaimDomain`], allocating one auxiliary slot per intermediate product.
//! The allocation depends only on the formula's structure, never on values,
//! so the same routine serves three domains:
//!
//! - [`ShapeDomain`] counts auxiliary slots for the pad layout;
//! - [`RowsDomain`] builds the residual rows over (pads, witness);
//! - [`ValuesDomain`] computes the prover's auxiliary values.
//!
//! Auxiliary slots form an acyclic chain: each is defined by exactly one
//! product row or defining affine row whose inputs are earlier slots.

use jolt_claims::{Expr, FactoredExpr, Source};
use jolt_field::Field;
use jolt_sumcheck::{BooleanHypercube, MaskedRound, SumcheckDomain, SumcheckDomainSpec};

use super::residual::{AffineRow, ProductRow, ResidualRelation};
use super::{PadIndex, ZkError};

/// The values a claim formula lowers into.
pub trait ClaimDomain<F> {
    type Value: Clone;

    fn constant(&self, value: F) -> Self::Value;
    fn add(&self, lhs: &Self::Value, rhs: &Self::Value) -> Self::Value;
    fn scale(&self, value: &Self::Value, scale: F) -> Self::Value;
    /// A new auxiliary slot defined by the affine `value`; returns the slot.
    fn define_auxiliary(&mut self, value: Self::Value) -> Result<Self::Value, ZkError>;
    /// A new auxiliary slot holding `x · y`, where `x` is a masked slot.
    fn product(&mut self, x: Self::Value, y: Self::Value) -> Result<Self::Value, ZkError>;
}

/// Counts auxiliary slots; carries no values.
#[derive(Debug, Default)]
pub struct ShapeDomain {
    pub auxiliary: usize,
}

impl<F> ClaimDomain<F> for ShapeDomain {
    type Value = ();

    fn constant(&self, _value: F) {}

    fn add(&self, _lhs: &(), _rhs: &()) {}

    fn scale(&self, _value: &(), _scale: F) {}

    fn define_auxiliary(&mut self, _value: ()) -> Result<(), ZkError> {
        self.bump()
    }

    fn product(&mut self, _x: (), _y: ()) -> Result<(), ZkError> {
        self.bump()
    }
}

impl ShapeDomain {
    fn bump(&mut self) -> Result<(), ZkError> {
        self.auxiliary = self
            .auxiliary
            .checked_add(1)
            .ok_or(ZkError::InvalidLayout {
                reason: "auxiliary slot count overflows usize",
            })?;
        Ok(())
    }
}

/// The prover's decoded values; records each auxiliary value in slot order.
#[derive(Debug, Default)]
pub struct ValuesDomain<F> {
    pub auxiliary: Vec<F>,
}

impl<F: Field> ClaimDomain<F> for ValuesDomain<F> {
    type Value = F;

    fn constant(&self, value: F) -> F {
        value
    }

    fn add(&self, lhs: &F, rhs: &F) -> F {
        *lhs + *rhs
    }

    fn scale(&self, value: &F, scale: F) -> F {
        *value * scale
    }

    fn define_auxiliary(&mut self, value: F) -> Result<F, ZkError> {
        self.auxiliary.push(value);
        Ok(value)
    }

    fn product(&mut self, x: F, y: F) -> Result<F, ZkError> {
        let z = x * y;
        self.auxiliary.push(z);
        Ok(z)
    }
}

/// The verifier's residual rows. Auxiliary slots are taken in order from the
/// layout's auxiliary range, with their masked values from the received
/// auxiliary block.
pub struct RowsDomain<'r, F> {
    relation: &'r mut ResidualRelation<F>,
    masked_auxiliary: &'r [F],
    next: usize,
}

impl<'r, F: Field> RowsDomain<'r, F> {
    pub fn new(relation: &'r mut ResidualRelation<F>, masked_auxiliary: &'r [F]) -> Self {
        Self {
            relation,
            masked_auxiliary,
            next: 0,
        }
    }

    pub fn relation(&mut self) -> &mut ResidualRelation<F> {
        self.relation
    }

    /// Checks that every received auxiliary value has its defining row.
    ///
    /// # Errors
    ///
    /// [`ZkError::PadCountMismatch`] otherwise.
    pub fn finish(&self) -> Result<(), ZkError> {
        if self.next == self.masked_auxiliary.len() {
            Ok(())
        } else {
            Err(ZkError::PadCountMismatch {
                scheduled: self.masked_auxiliary.len(),
                used: self.next,
            })
        }
    }

    fn next_slot(&mut self) -> Result<AffineRow<F>, ZkError> {
        let masked = *self
            .masked_auxiliary
            .get(self.next)
            .ok_or(ZkError::PadsExhausted {
                requested: 1,
                remaining: 0,
            })?;
        let pad = self
            .relation
            .layout
            .auxiliary
            .start
            .checked_add(self.next)
            .filter(|pad| self.relation.layout.auxiliary.contains(pad))
            .ok_or(ZkError::PadsExhausted {
                requested: 1,
                remaining: 0,
            })?;
        self.next += 1;
        Ok(AffineRow::masked(masked, PadIndex(pad)))
    }
}

impl<F: Field> ClaimDomain<F> for RowsDomain<'_, F> {
    type Value = AffineRow<F>;

    fn constant(&self, value: F) -> AffineRow<F> {
        AffineRow::constant(value)
    }

    fn add(&self, lhs: &AffineRow<F>, rhs: &AffineRow<F>) -> AffineRow<F> {
        lhs.add_scaled(rhs, F::one())
    }

    fn scale(&self, value: &AffineRow<F>, scale: F) -> AffineRow<F> {
        value.scale(scale)
    }

    fn define_auxiliary(&mut self, value: AffineRow<F>) -> Result<AffineRow<F>, ZkError> {
        let slot = self.next_slot()?;
        self.relation
            .affine
            .push(slot.add_scaled(&value, -F::one()));
        Ok(slot)
    }

    fn product(&mut self, x: AffineRow<F>, y: AffineRow<F>) -> Result<AffineRow<F>, ZkError> {
        let x = x.as_masked_operand().ok_or(ZkError::InvalidRelation {
            reason: "product operand is not a single masked slot",
        })?;
        let z = self.next_slot()?;
        self.relation
            .products
            .push(ProductRow { x, y, z: z.clone() });
        Ok(z)
    }
}

/// How one factor of a term lowers: a public scalar, one opening scaled by a
/// public scalar (usable as the masked operand of a product), or an affine
/// form in several openings.
enum Factor<'e, O> {
    Public,
    Opening(&'e O),
    Affine,
}

fn classify<F, O, P, C>(factor: &Expr<F, O, P, C>) -> Result<Factor<'_, O>, ZkError> {
    let mut opening_terms = 0usize;
    let mut single = None;
    for term in &factor.terms {
        let mut openings = term.factors.iter().filter_map(|source| match source {
            Source::Opening(id) => Some(id),
            Source::Challenge(_) | Source::Derived(_) => None,
        });
        match (openings.next(), openings.next()) {
            (None, _) => {}
            (Some(id), None) => {
                opening_terms += 1;
                single = Some(id);
            }
            (Some(_), Some(_)) => {
                return Err(ZkError::InvalidRelation {
                    reason: "factor is not affine in its openings",
                })
            }
        }
    }
    Ok(match (opening_terms, single) {
        (0, _) => Factor::Public,
        (1, Some(id)) if factor.terms.len() == 1 => Factor::Opening(id),
        _ => Factor::Affine,
    })
}

/// Lowers `expr` into `domain`. Openings resolve to domain values; challenges
/// and derived ids resolve to public scalars. Auxiliary slots are allocated
/// per term from its structure alone: a term with h ≥ 2 hidden factors takes
/// h − 1 product slots, plus one defining slot when none of its factors is a
/// single opening. No term is skipped, whatever its coefficient.
///
/// # Errors
///
/// Resolver errors, and [`ZkError::InvalidRelation`] for a factor that is
/// not affine in its openings.
pub fn lower_factored<F, O, P, C, D>(
    expr: &FactoredExpr<F, O, P, C>,
    domain: &mut D,
    opening: &mut impl FnMut(&O) -> Result<D::Value, ZkError>,
    public: &mut impl FnMut(&Source<O, P, C>) -> Result<F, ZkError>,
) -> Result<D::Value, ZkError>
where
    F: Field,
    D: ClaimDomain<F>,
{
    let mut sum = domain.constant(F::zero());
    for term in &expr.terms {
        let mut scale = term.coefficient;
        let mut hidden = Vec::new();
        let mut operand = None;
        for factor in &term.factors {
            match classify(factor)? {
                Factor::Public => scale *= evaluate_affine(factor, domain, opening, public)?.1,
                Factor::Opening(id) => {
                    let [inner] = factor.terms.as_slice() else {
                        return Err(ZkError::InvalidRelation {
                            reason: "single-opening factor has several terms",
                        });
                    };
                    scale *= inner.coefficient;
                    for source in &inner.factors {
                        if !matches!(source, Source::Opening(_)) {
                            scale *= public(source)?;
                        }
                    }
                    if operand.is_none() {
                        operand = Some(hidden.len());
                    }
                    hidden.push(opening(id)?);
                }
                Factor::Affine => {
                    hidden.push(evaluate_affine(factor, domain, opening, public)?.0);
                }
            }
        }
        let value = match hidden.len() {
            0 => domain.constant(F::one()),
            1 => hidden.swap_remove(0),
            _ => {
                let mut product = match operand {
                    Some(index) => hidden.remove(index),
                    None => domain.define_auxiliary(hidden.remove(0))?,
                };
                for factor in hidden {
                    product = domain.product(product, factor)?;
                }
                product
            }
        };
        sum = domain.add(&sum, &domain.scale(&value, scale));
    }
    Ok(sum)
}

/// An affine factor's value, and its public scalar when it has no opening.
fn evaluate_affine<F, O, P, C, D>(
    factor: &Expr<F, O, P, C>,
    domain: &D,
    opening: &mut impl FnMut(&O) -> Result<D::Value, ZkError>,
    public: &mut impl FnMut(&Source<O, P, C>) -> Result<F, ZkError>,
) -> Result<(D::Value, F), ZkError>
where
    F: Field,
    D: ClaimDomain<F>,
{
    let mut value = domain.constant(F::zero());
    let mut scalar = F::zero();
    for term in &factor.terms {
        let mut coefficient = term.coefficient;
        let mut hidden = None;
        for source in &term.factors {
            match source {
                Source::Opening(id) => hidden = Some(opening(id)?),
                Source::Challenge(_) | Source::Derived(_) => coefficient *= public(source)?,
            }
        }
        if let Some(hidden) = hidden {
            value = domain.add(&value, &domain.scale(&hidden, coefficient));
        } else {
            scalar += coefficient;
            value = domain.add(&value, &domain.constant(coefficient));
        }
    }
    Ok((value, scalar))
}

/// The wire format of a run of masked rounds.
#[derive(Clone, Copy, Debug)]
pub enum RoundFormat {
    /// The compressed Boolean-hypercube format: the linear coefficient is
    /// omitted and solved from the running claim, so the round check adds no
    /// row.
    Compressed,
    /// Every coefficient sent: the round check is an explicit row.
    Full(SumcheckDomainSpec),
}

/// The running claim after `rounds`, as an affine form over the pads, from
/// `input`. Full rounds push their round-check rows into `relation`.
///
/// # Errors
///
/// [`ZkError::InvalidRelation`] on a malformed round.
pub fn sumcheck_rows<F: Field>(
    relation: &mut ResidualRelation<F>,
    input: AffineRow<F>,
    rounds: &[MaskedRound<F>],
    degree: usize,
    format: RoundFormat,
) -> Result<AffineRow<F>, ZkError> {
    let mut claim = input;
    for round in rounds {
        let sent = round
            .wire
            .iter()
            .enumerate()
            .map(|(offset, &masked)| {
                AffineRow::masked(masked, PadIndex(round.slots.start + offset))
            })
            .collect::<Vec<_>>();
        if sent.len() != round.slots.len {
            return Err(ZkError::InvalidRelation {
                reason: "round wire length differs from its pad slots",
            });
        }
        let coefficients = match format {
            RoundFormat::Compressed => {
                let weights = BooleanHypercube
                    .round_sum_coefficients(degree.max(1))
                    .map_err(|_| ZkError::InvalidRelation {
                        reason: "round degree has no round-sum weights",
                    })?;
                compressed_coefficients(&claim, sent, &weights)?
            }
            RoundFormat::Full(domain) => {
                let weights = domain.round_sum_coefficients(degree).map_err(|_| {
                    ZkError::InvalidRelation {
                        reason: "round degree has no round-sum weights",
                    }
                })?;
                if weights.len() != sent.len() {
                    return Err(ZkError::InvalidRelation {
                        reason: "full round width differs from its weights",
                    });
                }
                let sum = sent
                    .iter()
                    .zip(&weights)
                    .fold(AffineRow::constant(F::zero()), |sum, (c, &w)| {
                        sum.add_scaled(c, w)
                    });
                relation.affine.push(sum.add_scaled(&claim, -F::one()));
                sent
            }
        };
        let mut power = F::one();
        claim = AffineRow::constant(F::zero());
        for coefficient in &coefficients {
            claim = claim.add_scaled(coefficient, power);
            power *= round.challenge;
        }
    }
    Ok(claim)
}

/// Rebuilds `[c0, c1, c2, …]` from the wire `[c0, c2, …]`, solving the
/// omitted c1 from Σ_j w_j c_j = claim.
fn compressed_coefficients<F: Field>(
    claim: &AffineRow<F>,
    sent: Vec<AffineRow<F>>,
    weights: &[F],
) -> Result<Vec<AffineRow<F>>, ZkError> {
    let mut sent = sent.into_iter();
    let c0 = sent.next().ok_or(ZkError::InvalidRelation {
        reason: "compressed round has no constant coefficient",
    })?;
    let rest: Vec<AffineRow<F>> = sent.collect();
    let (&w0, tail) = weights.split_first().ok_or(ZkError::InvalidRelation {
        reason: "round weights are empty",
    })?;
    let (&w1, higher) = tail.split_first().ok_or(ZkError::InvalidRelation {
        reason: "round weights have no linear weight",
    })?;
    let inverse = w1.inverse().ok_or(ZkError::InvalidRelation {
        reason: "linear round weight is zero",
    })?;
    if higher.len() != rest.len() {
        return Err(ZkError::InvalidRelation {
            reason: "compressed round width differs from its weights",
        });
    }
    let mut known = c0.scale(w0);
    for (coefficient, &weight) in rest.iter().zip(higher) {
        known = known.add_scaled(coefficient, weight);
    }
    let c1 = claim.add_scaled(&known, -F::one()).scale(inverse);
    let mut coefficients = Vec::with_capacity(rest.len() + 2);
    coefficients.push(c0);
    coefficients.push(c1);
    coefficients.extend(rest);
    Ok(coefficients)
}
