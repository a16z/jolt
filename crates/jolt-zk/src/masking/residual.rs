//! The residual relation: the verifier's checks on masked messages, as affine
//! and product rows over the pads and the residual witness (Whiteout
//! Definition `wo:def:residual`).

use std::cmp::Ordering;
use std::collections::BTreeMap;

use jolt_field::Field;

use super::{PadIndex, PadLayout, ZkError};

/// A variable of the residual relation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResidualVar {
    Pad(PadIndex),
    /// A coordinate of the back end's residual witness.
    Witness(usize),
}

impl ResidualVar {
    /// This variable's value in an assignment of pads and witness.
    ///
    /// # Errors
    ///
    /// [`ZkError::InvalidRelation`] if the variable is outside the assignment.
    pub fn value<F: Copy>(self, pads: &[F], witness: &[F]) -> Result<F, ZkError> {
        match self {
            Self::Pad(PadIndex(index)) => pads.get(index),
            Self::Witness(index) => witness.get(index),
        }
        .copied()
        .ok_or(ZkError::InvalidRelation {
            reason: "row variable outside the assignment",
        })
    }
}

/// `Σ_k a_k · x_k + b`; as a row, the constraint that this is zero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AffineRow<F> {
    pub terms: Vec<(ResidualVar, F)>,
    pub constant: F,
}

impl<F: Field> AffineRow<F> {
    pub fn constant(constant: F) -> Self {
        Self {
            terms: Vec::new(),
            constant,
        }
    }

    /// The masked value m̂ at `pad`, decoded: m̂ − p.
    pub fn masked(masked: F, pad: PadIndex) -> Self {
        Self {
            terms: vec![(ResidualVar::Pad(pad), -F::one())],
            constant: masked,
        }
    }

    pub fn variable(variable: ResidualVar) -> Self {
        Self {
            terms: vec![(variable, F::one())],
            constant: F::zero(),
        }
    }

    /// `self + scale · other`, with like terms merged, in variable order.
    /// Terms whose coefficients cancel are kept, with a zero coefficient.
    pub fn add_scaled(&self, other: &Self, scale: F) -> Self {
        let ordered = |terms: &[(ResidualVar, F)]| terms.is_sorted_by(|a, b| a.0 < b.0);
        if ordered(&self.terms) && ordered(&other.terms) {
            return self.merge_ordered(other, scale);
        }
        let mut merged: BTreeMap<ResidualVar, F> = BTreeMap::new();
        for &(variable, coefficient) in &self.terms {
            *merged.entry(variable).or_insert_with(F::zero) += coefficient;
        }
        for &(variable, coefficient) in &other.terms {
            *merged.entry(variable).or_insert_with(F::zero) += coefficient * scale;
        }
        Self {
            terms: merged.into_iter().collect(),
            constant: self.constant + other.constant * scale,
        }
    }

    /// [`Self::add_scaled`] for terms already strictly ordered by variable,
    /// as every row it builds is: one linear merge, no map.
    fn merge_ordered(&self, other: &Self, scale: F) -> Self {
        let mut terms = Vec::with_capacity(self.terms.len() + other.terms.len());
        let (mut lhs, mut rhs) = (self.terms.iter().peekable(), other.terms.iter().peekable());
        loop {
            let order = match (lhs.peek(), rhs.peek()) {
                (None, None) => break,
                (Some(_), None) => Ordering::Less,
                (None, Some(_)) => Ordering::Greater,
                (Some(a), Some(b)) => a.0.cmp(&b.0),
            };
            terms.extend(match order {
                Ordering::Less => lhs.next().copied(),
                Ordering::Greater => rhs.next().map(|&(b, y)| (b, y * scale)),
                Ordering::Equal => lhs
                    .next()
                    .zip(rhs.next())
                    .map(|(&(a, x), &(_, y))| (a, x + y * scale)),
            });
        }
        Self {
            terms,
            constant: self.constant + other.constant * scale,
        }
    }

    pub fn scale(&self, scale: F) -> Self {
        Self::constant(F::zero()).add_scaled(self, scale)
    }

    /// The value under an assignment.
    ///
    /// # Errors
    ///
    /// As [`ResidualVar::value`].
    pub fn evaluate(&self, pads: &[F], witness: &[F]) -> Result<F, ZkError> {
        self.terms
            .iter()
            .try_fold(self.constant, |sum, &(variable, coefficient)| {
                Ok(sum + coefficient * variable.value(pads, witness)?)
            })
    }

    /// The masked operand this form is, if it is exactly m̂ − p.
    pub fn as_masked_operand(&self) -> Option<MaskedOperand<F>> {
        match self.terms.as_slice() {
            [(ResidualVar::Pad(pad), coefficient)] if *coefficient == -F::one() => {
                Some(MaskedOperand {
                    masked: self.constant,
                    pad: *pad,
                })
            }
            _ => None,
        }
    }
}

/// A product operand that is a single masked value, x = m̂ − p: one constant
/// operand per product (Whiteout §3.2), so a back end under
/// [`OperandRule::MaskedSlot`] can add the scalarity rows of x's pad.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaskedOperand<F> {
    pub masked: F,
    pub pad: PadIndex,
}

impl<F: Field> MaskedOperand<F> {
    pub fn form(self) -> AffineRow<F> {
        AffineRow::masked(self.masked, self.pad)
    }
}

/// What a back end accepts as a product's operand `x`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OperandRule {
    /// A single masked slot, whose pad the back end commits separately
    /// (Whiteout §3.2). A product with no
    /// single-opening factor first gives one factor a defining slot.
    #[default]
    MaskedSlot,
    /// Any affine form: an R1CS constraint's A side is a linear combination,
    /// so a back end that proves R1CS rows (BlindFold) needs no defining
    /// slot.
    Affine,
}

/// A product's operand `x`: a single masked slot, or, under
/// [`OperandRule::Affine`], an affine form that is not one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProductOperand<F> {
    Masked(MaskedOperand<F>),
    Affine(AffineRow<F>),
}

impl<F: Field> ProductOperand<F> {
    pub fn form(&self) -> AffineRow<F> {
        match self {
            Self::Masked(operand) => operand.form(),
            Self::Affine(form) => form.clone(),
        }
    }
}

impl<F: Copy> ProductOperand<F> {
    /// The operand as a single masked slot, if it is one.
    pub fn masked(&self) -> Option<MaskedOperand<F>> {
        match self {
            Self::Masked(operand) => Some(*operand),
            Self::Affine(_) => None,
        }
    }
}

/// One product `x · y` of a [`ProductRow`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProductPair<F> {
    pub x: ProductOperand<F>,
    pub y: AffineRow<F>,
}

/// `Σ_k x_k · y_k = z` over the pairs `first` and `rest`, so a row has at
/// least one pair. With one pair it is the paper's `x · y = z`; several pairs
/// land several products on one claim, such as a stage's running claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProductRow<F> {
    pub first: ProductPair<F>,
    pub rest: Vec<ProductPair<F>>,
    pub z: AffineRow<F>,
}

impl<F> ProductRow<F> {
    /// Every pair, `first` then `rest`.
    pub fn pairs(&self) -> impl Iterator<Item = &ProductPair<F>> {
        std::iter::once(&self.first).chain(&self.rest)
    }
}

impl<F: Field> ProductRow<F> {
    /// `Σ_k x_k · y_k − z` under an assignment.
    ///
    /// # Errors
    ///
    /// As [`ResidualVar::value`].
    pub fn evaluate(&self, pads: &[F], witness: &[F]) -> Result<F, ZkError> {
        self.pairs()
            .try_fold(-self.z.evaluate(pads, witness)?, |sum, pair| {
                Ok(
                    sum + pair.x.form().evaluate(pads, witness)?
                        * pair.y.evaluate(pads, witness)?,
                )
            })
    }
}

/// The rows a masked transcript must satisfy. Rows are never dropped: one
/// whose variables cancel but whose constant is nonzero stays, unsatisfiable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidualRelation<F> {
    pub affine: Vec<AffineRow<F>>,
    pub products: Vec<ProductRow<F>>,
    pub layout: PadLayout,
    pub num_witness: usize,
}

/// One row of either kind, as the relation's consumers iterate them.
#[derive(Clone, Copy, Debug)]
pub enum Row<'a, F> {
    Affine(&'a AffineRow<F>),
    Product(&'a ProductRow<F>),
}

impl<F: Field> ResidualRelation<F> {
    pub fn new(layout: PadLayout, num_witness: usize) -> Self {
        Self {
            affine: Vec::new(),
            products: Vec::new(),
            layout,
            num_witness,
        }
    }

    /// Every row: the affine rows, then the product rows.
    pub fn rows(&self) -> impl Iterator<Item = Row<'_, F>> {
        self.affine
            .iter()
            .map(Row::Affine)
            .chain(self.products.iter().map(Row::Product))
    }

    /// The index of the first row `pads` and `witness` violate, if any.
    ///
    /// # Errors
    ///
    /// [`ZkError::InvalidRelation`] if a row names a variable outside the
    /// assignment.
    pub fn first_violation(&self, pads: &[F], witness: &[F]) -> Result<Option<usize>, ZkError> {
        for (index, row) in self.rows().enumerate() {
            let satisfied = match row {
                Row::Affine(row) => row.evaluate(pads, witness)?.is_zero(),
                Row::Product(row) => row.evaluate(pads, witness)?.is_zero(),
            };
            if !satisfied {
                return Ok(Some(index));
            }
        }
        Ok(None)
    }
}

/// A verifier that contributes rows after replaying its part of the
/// transcript: Jolt's stages (through the verifier replay) and Akita's
/// folds up to the cut point.
pub trait ResidualRows<F> {
    type Error;
    fn append_rows(&self, relation: &mut ResidualRelation<F>) -> Result<(), Self::Error>;
}
