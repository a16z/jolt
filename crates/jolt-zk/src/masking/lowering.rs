//! Lowering verifier checks on masked messages.
//!
//! [`lower_factored`] turns a factored claim formula into a value of a
//! [`ClaimDomain`] (`jolt-transcript`'s, shared with Akita's stage checks),
//! allocating one auxiliary slot per intermediate product.
//! The allocation depends only on the formula's structure, never on values,
//! so the same routine serves three domains:
//!
//! - [`OperandShapeDomain`] counts auxiliary slots and records the product
//!   operands' pads, giving the pad layout;
//! - [`RowsDomain`] builds the residual rows over (pads, witness);
//! - [`ValuesDomain`] computes the prover's auxiliary values.
//!
//! Auxiliary slots form an acyclic chain: each is defined by exactly one
//! product row or defining affine row whose inputs are earlier slots.

use jolt_algebra::Field;
use jolt_claims::{Expr, FactoredExpr, Source};
use jolt_poly::{Point, LOW_TO_HIGH};
use jolt_sumcheck::{
    BooleanHypercube, DegreeSchedule, EqFactoredRound, MaskedRound, RoundWire, SumcheckDomain,
    SumcheckDomainSpec,
};
use jolt_transcript::{ClaimDomain, ClearDomain};

use super::residual::{AffineRow, ProductRow, ResidualRelation};
use super::{PadIndex, PadLayout, ZkError};

/// The numbering of auxiliary slots: slot `k` of the block is pad
/// `start + k`. [`RowsDomain`] and [`OperandShapeDomain`] both take slots
/// through it, so the shape routine names the pads the rows use.
#[derive(Clone, Copy, Debug)]
struct AuxiliarySlots {
    start: usize,
    next: usize,
}

impl AuxiliarySlots {
    fn new(start: usize) -> Self {
        Self { start, next: 0 }
    }

    /// The next slot, below `end` if the range is bounded.
    fn take(&mut self, end: Option<usize>) -> Result<PadIndex, ZkError> {
        let pad = self
            .start
            .checked_add(self.next)
            .filter(|&pad| end.is_none_or(|end| pad < end))
            .ok_or(ZkError::PadsExhausted {
                requested: 1,
                remaining: 0,
            })?;
        self.next += 1;
        Ok(PadIndex(pad))
    }
}

/// The shape routine: counts the auxiliary slots and records the pads of
/// product operands, before the pads are committed. The lattice back end
/// commits operand pads as constant digit polynomials (Whiteout §3.2).
/// Openings resolve to their masked slots and auxiliary slots are numbered
/// as [`RowsDomain`] numbers them, so the operands are every product row's
/// operand pad, in the order the rows are emitted, each once.
#[derive(Debug)]
pub struct OperandShapeDomain {
    slots: AuxiliarySlots,
    operands: Vec<PadIndex>,
}

impl OperandShapeDomain {
    /// The shape of a schedule with `base` message slots, whose auxiliary
    /// block starts right after them.
    pub fn new(base: usize) -> Self {
        Self {
            slots: AuxiliarySlots::new(base),
            operands: Vec::new(),
        }
    }

    /// The pad layout: `base` message slots, the auxiliary slots counted,
    /// and the operand pads recorded.
    ///
    /// # Errors
    ///
    /// [`ZkError::InvalidLayout`] if the total overflows `usize`.
    pub fn layout(self) -> Result<PadLayout, ZkError> {
        Ok(PadLayout::new(self.slots.start, self.slots.next)?.with_operand_pads(self.operands))
    }

    /// Records the pad of a product's operand `x`, each pad once.
    fn record_operand(&mut self, x: Option<PadIndex>) -> Result<(), ZkError> {
        let x = x.ok_or(ZkError::InvalidRelation {
            reason: "product operand is not a single masked slot",
        })?;
        if !self.operands.contains(&x) {
            self.operands.push(x);
        }
        Ok(())
    }
}

impl<F> ClaimDomain<F> for OperandShapeDomain {
    /// The slot of a value that is exactly one masked slot.
    type Value = Option<PadIndex>;
    type Error = ZkError;

    fn constant(&self, _value: F) -> Option<PadIndex> {
        None
    }

    fn add(&self, _lhs: &Option<PadIndex>, _rhs: &Option<PadIndex>) -> Option<PadIndex> {
        None
    }

    fn scale(&self, _value: &Option<PadIndex>, _scale: F) -> Option<PadIndex> {
        None
    }

    fn define_auxiliary(&mut self, _value: Option<PadIndex>) -> Result<Option<PadIndex>, ZkError> {
        Ok(Some(self.slots.take(None)?))
    }

    fn product(
        &mut self,
        x: Option<PadIndex>,
        _y: Option<PadIndex>,
    ) -> Result<Option<PadIndex>, ZkError> {
        self.record_operand(x)?;
        Ok(Some(self.slots.take(None)?))
    }

    fn product_into(
        &mut self,
        x: Option<PadIndex>,
        _y: Option<PadIndex>,
        _target: Option<PadIndex>,
    ) -> Result<(), ZkError> {
        self.record_operand(x)
    }

    fn check_zero(&mut self, _value: Option<PadIndex>) -> Result<(), ZkError> {
        Ok(())
    }
}

/// The prover's decoded values; records each auxiliary value in slot order.
/// Checks compare as [`ClearDomain`] does.
#[derive(Debug, Default)]
pub struct ValuesDomain<F> {
    pub auxiliary: Vec<F>,
}

impl<F: Field> ClaimDomain<F> for ValuesDomain<F> {
    type Value = F;
    type Error = ZkError;

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
        let z = ClearDomain.product(x, y)?;
        self.auxiliary.push(z);
        Ok(z)
    }

    fn product_into(&mut self, x: F, y: F, target: F) -> Result<(), ZkError> {
        Ok(ClearDomain.product_into(x, y, target)?)
    }

    fn check_zero(&mut self, value: F) -> Result<(), ZkError> {
        Ok(ClaimDomain::<F>::check_zero(&mut ClearDomain, value)?)
    }
}

/// The verifier's residual rows. Auxiliary slots are taken in order from the
/// layout's auxiliary range, with their masked values from the received
/// auxiliary block.
pub struct RowsDomain<'r, F> {
    relation: &'r mut ResidualRelation<F>,
    masked_auxiliary: &'r [F],
    slots: AuxiliarySlots,
}

impl<'r, F: Field> RowsDomain<'r, F> {
    pub fn new(relation: &'r mut ResidualRelation<F>, masked_auxiliary: &'r [F]) -> Self {
        let slots = AuxiliarySlots::new(relation.layout.auxiliary.start);
        Self {
            relation,
            masked_auxiliary,
            slots,
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
        if self.slots.next == self.masked_auxiliary.len() {
            Ok(())
        } else {
            Err(ZkError::PadCountMismatch {
                scheduled: self.masked_auxiliary.len(),
                used: self.slots.next,
            })
        }
    }

    fn next_slot(&mut self) -> Result<AffineRow<F>, ZkError> {
        let masked = *self
            .masked_auxiliary
            .get(self.slots.next)
            .ok_or(ZkError::PadsExhausted {
                requested: 1,
                remaining: 0,
            })?;
        let pad = self.slots.take(Some(self.relation.layout.auxiliary.end))?;
        Ok(AffineRow::masked(masked, pad))
    }
}

impl<F: Field> ClaimDomain<F> for RowsDomain<'_, F> {
    type Value = AffineRow<F>;
    type Error = ZkError;

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
        let z = self.next_slot()?;
        self.product_into(x, y, z.clone())?;
        Ok(z)
    }

    fn product_into(
        &mut self,
        x: AffineRow<F>,
        y: AffineRow<F>,
        target: AffineRow<F>,
    ) -> Result<(), ZkError> {
        let x = x.as_masked_operand().ok_or(ZkError::InvalidRelation {
            reason: "product operand is not a single masked slot",
        })?;
        self.relation.products.push(ProductRow { x, y, z: target });
        Ok(())
    }

    fn check_zero(&mut self, value: AffineRow<F>) -> Result<(), ZkError> {
        self.relation.affine.push(value);
        Ok(())
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
    D: ClaimDomain<F, Error = ZkError>,
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
    D: ClaimDomain<F, Error = ZkError>,
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
#[derive(Clone, Debug)]
pub enum RoundFormat<F> {
    /// The compressed Boolean-hypercube format: the linear coefficient is
    /// omitted and solved from the running claim, so the round check adds no
    /// row.
    Compressed,
    /// `jolt-sumcheck`'s `SingleFormat::EqFactored`: round `i` sends
    /// `[q1, …, qd]` of `s(X) = eq(τ_i, X) · q(X)`, with
    /// `τ_i = equality_point[i]`; `q0` is solved from
    /// `(1 − τ_i) q(0) + τ_i q(1) = claim`, so the round check adds no row, and
    /// the running claim is the normalized `q(r_i)`. Akita's masked stage 1
    /// (Akita `specs/whiteout-akita.md` §4.5).
    EqFactored {
        equality_point: Point<LOW_TO_HIGH, F>,
    },
    /// Every coefficient sent: the round check is an explicit row.
    Full(SumcheckDomainSpec),
}

impl<F> RoundFormat<F> {
    /// Both one-omitted formats send `degree` values per round.
    fn wire(&self) -> RoundWire {
        match self {
            Self::Compressed | Self::EqFactored { .. } => RoundWire::Compressed,
            Self::Full(_) => RoundWire::Full,
        }
    }
}

/// The running claim after `rounds`, as an affine form over the pads, from
/// `input`; round `j` is read at `degrees`' degree `j`. Full rounds push their
/// round-check rows into `relation`.
///
/// # Errors
///
/// [`ZkError::RoundCountMismatch`] if the rounds, the schedule or an
/// equality point differ in count;
/// [`ZkError::RoundWidthMismatch`] if a round's wire length differs from its
/// scheduled width; [`ZkError::InvalidRelation`] on another malformed round.
pub fn sumcheck_rows<F: Field>(
    relation: &mut ResidualRelation<F>,
    input: AffineRow<F>,
    rounds: &[MaskedRound<F>],
    degrees: &DegreeSchedule,
    format: &RoundFormat<F>,
) -> Result<AffineRow<F>, ZkError> {
    if rounds.len() != degrees.num_rounds() {
        return Err(ZkError::RoundCountMismatch {
            expected: degrees.num_rounds(),
            got: rounds.len(),
        });
    }
    if let RoundFormat::EqFactored { equality_point } = format {
        if equality_point.len() != rounds.len() {
            return Err(ZkError::RoundCountMismatch {
                expected: rounds.len(),
                got: equality_point.len(),
            });
        }
    }
    let mut claim = input;
    for (index, (round, &degree)) in rounds.iter().zip(degrees.degrees()).enumerate() {
        let width = format
            .wire()
            .width::<F>(degree)
            .map_err(|_| ZkError::InvalidRelation {
                reason: "round degree has no wire width",
            })?;
        if round.wire.len() != width {
            return Err(ZkError::RoundWidthMismatch {
                round: index,
                expected: width,
                got: round.wire.len(),
            });
        }
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
                omitted_coefficient(&claim, sent, &weights, 1)?
            }
            RoundFormat::EqFactored { equality_point } => {
                let tau =
                    *equality_point
                        .as_slice()
                        .get(index)
                        .ok_or(ZkError::InvalidRelation {
                            reason: "equality point has no coordinate for the round",
                        })?;
                let weights = EqFactoredRound { tau }
                    .round_sum_coefficients(degree)
                    .map_err(|_| ZkError::InvalidRelation {
                        reason: "round degree has no round-sum weights",
                    })?;
                omitted_coefficient(&claim, sent, &weights, 0)?
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

/// Rebuilds every coefficient from a wire that omits the one at `omitted`
/// (`[c0, c2, …]` for 1, `[c1, c2, …]` for 0), solving it from
/// Σ_j w_j c_j = claim.
fn omitted_coefficient<F: Field>(
    claim: &AffineRow<F>,
    sent: Vec<AffineRow<F>>,
    weights: &[F],
    omitted: usize,
) -> Result<Vec<AffineRow<F>>, ZkError> {
    if weights.len() != sent.len() + 1 || omitted > sent.len() {
        return Err(ZkError::InvalidRelation {
            reason: "round width differs from its weights",
        });
    }
    let mut known = AffineRow::constant(F::zero());
    let mut omitted_weight = None;
    let mut sent = sent.into_iter();
    let mut coefficients = Vec::with_capacity(weights.len());
    for (index, &weight) in weights.iter().enumerate() {
        if index == omitted {
            omitted_weight = Some(weight);
            coefficients.push(AffineRow::constant(F::zero()));
        } else {
            let coefficient = sent.next().ok_or(ZkError::InvalidRelation {
                reason: "round width differs from its weights",
            })?;
            known = known.add_scaled(&coefficient, weight);
            coefficients.push(coefficient);
        }
    }
    let inverse =
        omitted_weight
            .and_then(|weight| weight.inverse())
            .ok_or(ZkError::InvalidRelation {
                reason: "omitted round weight is zero",
            })?;
    let solved = claim.add_scaled(&known, -F::one()).scale(inverse);
    let slot = coefficients
        .get_mut(omitted)
        .ok_or(ZkError::InvalidRelation {
            reason: "round width differs from its weights",
        })?;
    *slot = solved;
    Ok(coefficients)
}
