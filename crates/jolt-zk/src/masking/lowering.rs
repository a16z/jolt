//! Lowering verifier checks on masked messages.
//!
//! [`lower_factored`] turns a factored claim formula into a [`ProductSum`]
//! over a [`ClaimDomain`] (`jolt-transcript`'s, shared with Akita's stage
//! checks): an affine value plus each term's last product, left pending so
//! that a check can land it without a slot ([`ClaimDomain::products_into`]).
//! Every other intermediate product takes one auxiliary slot, shared by
//! every term that repeats it ([`ProductMemo`]).
//! The allocation depends only on the formulas' structure, never on values,
//! so the same routine serves three domains:
//!
//! - [`OperandShapeDomain`] counts auxiliary slots and records the product
//!   operands' pads, giving the pad layout;
//! - [`RowsDomain`] builds the residual rows over (pads, witness);
//! - [`ValuesDomain`] computes the prover's auxiliary values.
//!
//! Auxiliary slots form an acyclic chain: each is defined by exactly one
//! product row or defining affine row whose inputs are earlier slots.

use std::collections::btree_map::Entry;
use std::collections::BTreeMap;

use jolt_algebra::Field;
use jolt_claims::{Expr, FactoredExpr, Source};
use jolt_poly::{Point, LOW_TO_HIGH};
use jolt_sumcheck::{
    BooleanHypercube, DegreeSchedule, EqFactoredRound, MaskedRound, RoundWire, SumcheckDomain,
    SumcheckDomainSpec,
};
use jolt_transcript::{ClaimDomain, ClearDomain};

use super::residual::{
    AffineRow, OperandRule, ProductOperand, ProductPair, ProductRow, ResidualRelation,
};
use super::{PadIndex, PadLayout, ZkError};

/// A [`ClaimDomain`] the masking lowering runs in: it states the
/// [`OperandRule`] of its back end, which decides whether
/// [`lower_factored`] gives a product without a single-opening factor a
/// defining slot. Every domain of one lowering must state the same rule, so
/// they allocate the same slots.
pub trait LoweringDomain<F>: ClaimDomain<F> {
    fn operands(&self) -> OperandRule;
}

const NOT_A_MASKED_SLOT: ZkError = ZkError::InvalidRelation {
    reason: "product operand is not a single masked slot",
};

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
    rule: OperandRule,
    operands: Vec<PadIndex>,
}

impl OperandShapeDomain {
    /// The shape of a schedule with `base` message slots, whose auxiliary
    /// block starts right after them.
    pub fn new(base: usize) -> Self {
        Self {
            slots: AuxiliarySlots::new(base),
            rule: OperandRule::MaskedSlot,
            operands: Vec::new(),
        }
    }

    /// The same shape under `rule`. Under [`OperandRule::Affine`] no
    /// operand pad is recorded.
    pub fn with_operands(self, rule: OperandRule) -> Self {
        Self { rule, ..self }
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

    /// The auxiliary slots allocated so far.
    pub fn auxiliary_count(&self) -> usize {
        self.slots.next
    }

    /// Records the pad of a product's operand `x`, each pad once, under
    /// [`OperandRule::MaskedSlot`].
    fn record_operand(&mut self, x: Option<PadIndex>) -> Result<(), ZkError> {
        if self.rule == OperandRule::Affine {
            return Ok(());
        }
        let x = x.ok_or(NOT_A_MASKED_SLOT)?;
        if !self.operands.contains(&x) {
            self.operands.push(x);
        }
        Ok(())
    }
}

impl<F> LoweringDomain<F> for OperandShapeDomain {
    fn operands(&self) -> OperandRule {
        self.rule
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

    fn products_into(
        &mut self,
        first: (Option<PadIndex>, Option<PadIndex>),
        rest: Vec<(Option<PadIndex>, Option<PadIndex>)>,
        _target: Option<PadIndex>,
    ) -> Result<(), ZkError> {
        std::iter::once(first)
            .chain(rest)
            .try_for_each(|(x, _)| self.record_operand(x))
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
    rule: OperandRule,
}

impl<F> ValuesDomain<F> {
    /// An empty domain under `rule`.
    pub fn with_operands(rule: OperandRule) -> Self {
        Self {
            auxiliary: Vec::new(),
            rule,
        }
    }
}

impl<F: Field> LoweringDomain<F> for ValuesDomain<F> {
    fn operands(&self) -> OperandRule {
        self.rule
    }
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

    fn products_into(
        &mut self,
        first: (F, F),
        rest: Vec<(F, F)>,
        target: F,
    ) -> Result<(), ZkError> {
        Ok(ClearDomain.products_into(first, rest, target)?)
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
    rule: OperandRule,
}

impl<'r, F: Field> RowsDomain<'r, F> {
    pub fn new(relation: &'r mut ResidualRelation<F>, masked_auxiliary: &'r [F]) -> Self {
        let first = relation.layout.auxiliary.start;
        Self::from_slot(relation, masked_auxiliary, first)
    }

    /// Rows whose auxiliary values `masked_auxiliary` take the slots from
    /// `first` on: a protocol whose auxiliary block follows another's in the
    /// same relation.
    pub fn from_slot(
        relation: &'r mut ResidualRelation<F>,
        masked_auxiliary: &'r [F],
        first: usize,
    ) -> Self {
        Self {
            relation,
            masked_auxiliary,
            slots: AuxiliarySlots::new(first),
            rule: OperandRule::MaskedSlot,
        }
    }

    /// The same rows under `rule`.
    pub fn with_operands(self, rule: OperandRule) -> Self {
        Self { rule, ..self }
    }

    /// `x` as a product operand under the rule.
    fn operand(&self, x: AffineRow<F>) -> Result<ProductOperand<F>, ZkError> {
        match (x.as_masked_operand(), self.rule) {
            (Some(operand), _) => Ok(ProductOperand::Masked(operand)),
            (None, OperandRule::Affine) => Ok(ProductOperand::Affine(x)),
            (None, OperandRule::MaskedSlot) => Err(NOT_A_MASKED_SLOT),
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

impl<F: Field> LoweringDomain<F> for RowsDomain<'_, F> {
    fn operands(&self) -> OperandRule {
        self.rule
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

    fn products_into(
        &mut self,
        first: (AffineRow<F>, AffineRow<F>),
        rest: Vec<(AffineRow<F>, AffineRow<F>)>,
        target: AffineRow<F>,
    ) -> Result<(), ZkError> {
        let pair = |(x, y)| {
            Ok::<_, ZkError>(ProductPair {
                x: self.operand(x)?,
                y,
            })
        };
        let first = pair(first)?;
        let rest = rest.into_iter().map(pair).collect::<Result<Vec<_>, _>>()?;
        self.relation.products.push(ProductRow {
            first,
            rest,
            z: target,
        });
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

/// A lowered claim: `Σ_k x_k · y_k + affine`, with each term's last product
/// `x_k · y_k` still pending. A check lands the pairs with
/// [`check_equals`](Self::check_equals); a value that must stay a single
/// domain value takes slots for them with [`into_value`](Self::into_value).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProductSum<V> {
    pub pairs: Vec<(V, V)>,
    pub affine: V,
}

impl<V: Clone> ProductSum<V> {
    pub fn zero<F, D: ClaimDomain<F, Value = V>>(domain: &D) -> Self
    where
        F: Field,
    {
        Self {
            pairs: Vec::new(),
            affine: domain.constant(F::zero()),
        }
    }

    pub fn add<F, D: ClaimDomain<F, Value = V>>(mut self, other: Self, domain: &D) -> Self {
        self.pairs.extend(other.pairs);
        self.affine = domain.add(&self.affine, &other.affine);
        self
    }

    /// Scales each pair's `y`, so its `x` stays a single slot.
    pub fn scale<F: Copy, D: ClaimDomain<F, Value = V>>(self, scale: F, domain: &D) -> Self {
        Self {
            pairs: self
                .pairs
                .into_iter()
                .map(|(x, y)| (x, domain.scale(&y, scale)))
                .collect(),
            affine: domain.scale(&self.affine, scale),
        }
    }

    /// The value as one domain value: one auxiliary slot per pending pair.
    ///
    /// # Errors
    ///
    /// As [`ClaimDomain::product`].
    pub fn into_value<F, D>(self, domain: &mut D) -> Result<V, D::Error>
    where
        D: ClaimDomain<F, Value = V>,
    {
        let mut value = self.affine;
        for (x, y) in self.pairs {
            let product = domain.product(x, y)?;
            value = domain.add(&value, &product);
        }
        Ok(value)
    }

    /// The check `self = target`: one multi-pair row
    /// `Σ_k x_k · y_k = target − affine`, or an affine row when no pair is
    /// pending.
    ///
    /// # Errors
    ///
    /// As [`ClaimDomain::products_into`] and [`ClaimDomain::check_zero`].
    pub fn check_equals<F: Field, D>(self, target: V, domain: &mut D) -> Result<(), D::Error>
    where
        D: ClaimDomain<F, Value = V>,
    {
        let rest = domain.add(&target, &domain.scale(&self.affine, -F::one()));
        let mut pairs = self.pairs.into_iter();
        match pairs.next() {
            None => domain.check_zero(rest),
            Some(first) => domain.products_into(first, pairs.collect(), rest),
        }
    }
}

/// The intermediate products a lowering has taken slots for, keyed by their
/// opening factors in chain order. One memo spans a whole relation's
/// lowering, so a product of the same openings that recurs across terms,
/// members or stages takes one slot ([`lower_factored`]). Keys are
/// structural, so every domain hits the same entries.
///
/// Contract: one memo per lowering of one relation in one domain. Its values
/// are that domain's slots, so another domain, or a second pass over the same
/// relation, starts from a fresh memo.
#[derive(Debug)]
pub struct ProductMemo<O, V> {
    products: BTreeMap<Vec<O>, V>,
}

impl<O, V> Default for ProductMemo<O, V> {
    fn default() -> Self {
        Self {
            products: BTreeMap::new(),
        }
    }
}

impl<O: Ord + Clone, V: Clone> ProductMemo<O, V> {
    /// `x · y`, with `key` the openings of `x · y` in chain order: the slot
    /// an earlier term took for the same key, or a new one.
    fn product<F, D>(&mut self, key: Option<&[O]>, x: V, y: V, domain: &mut D) -> Result<V, ZkError>
    where
        D: ClaimDomain<F, Value = V, Error = ZkError>,
    {
        let Some(key) = key else {
            return domain.product(x, y);
        };
        match self.products.entry(key.to_vec()) {
            Entry::Occupied(product) => Ok(product.get().clone()),
            Entry::Vacant(slot) => Ok(slot.insert(domain.product(x, y)?).clone()),
        }
    }
}

/// Lowers `expr` into `domain`. Openings resolve to domain values; challenges
/// and derived ids resolve to public scalars. Auxiliary slots are allocated
/// per term from its structure, the domain's [`OperandRule`] and `products`
/// alone, never from values. A term with h ≥ 2 hidden factors chains its
/// first single-opening factor, then the others in formula order, and
/// leaves the last product pending: h − 2 intermediate products, each of
/// which takes a slot unless an earlier term took one for the same openings
/// in the same order (`products`). Under [`OperandRule::MaskedSlot`] a term
/// with no single-opening factor first takes one defining slot. No term is
/// skipped, whatever its coefficient.
///
/// # Errors
///
/// Resolver errors, and [`ZkError::InvalidRelation`] for a factor that is
/// not affine in its openings.
pub fn lower_factored<F, O, P, C, D>(
    expr: &FactoredExpr<F, O, P, C>,
    domain: &mut D,
    products: &mut ProductMemo<O, D::Value>,
    opening: &mut impl FnMut(&O) -> Result<D::Value, ZkError>,
    public: &mut impl FnMut(&Source<O, P, C>) -> Result<F, ZkError>,
) -> Result<ProductSum<D::Value>, ZkError>
where
    F: Field,
    O: Ord + Clone,
    D: LoweringDomain<F, Error = ZkError>,
{
    let mut sum = ProductSum::zero(domain);
    for term in &expr.terms {
        let mut scale = term.coefficient;
        // Each hidden factor's value, and its opening when it is a single one.
        let mut hidden: Vec<(D::Value, Option<&O>)> = Vec::new();
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
                    hidden.push((opening(id)?, Some(id)));
                }
                Factor::Affine => {
                    hidden.push((evaluate_affine(factor, domain, opening, public)?.0, None));
                }
            }
        }
        let term = match hidden.len() {
            0 => ProductSum {
                pairs: Vec::new(),
                affine: domain.constant(F::one()),
            },
            1 => ProductSum {
                pairs: Vec::new(),
                affine: hidden.swap_remove(0).0,
            },
            _ => {
                let operand = hidden.iter().position(|(_, id)| id.is_some());
                let (first, first_id) = hidden.remove(operand.unwrap_or(0));
                let mut product = match (operand, domain.operands()) {
                    (Some(_), _) | (None, OperandRule::Affine) => first,
                    (None, OperandRule::MaskedSlot) => domain.define_auxiliary(first)?,
                };
                let (last, _) = hidden.pop().ok_or(ZkError::InvalidRelation {
                    reason: "a product term lost its last factor",
                })?;
                // The chain's openings so far, while it is a product of
                // openings alone.
                let mut key = first_id.map(|id| vec![id.clone()]);
                for (factor, id) in hidden {
                    key = key.zip(id).map(|(mut key, id)| {
                        key.push(id.clone());
                        key
                    });
                    product = products.product(key.as_deref(), product, factor, domain)?;
                }
                ProductSum {
                    pairs: vec![(product, last)],
                    affine: domain.constant(F::zero()),
                }
            }
        };
        sum = sum.add(term.scale(scale, domain), domain);
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
