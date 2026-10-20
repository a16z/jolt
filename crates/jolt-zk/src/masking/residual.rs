//! The residual relation: the verifier's checks on masked messages, as affine
//! and product rows over the pads and the residual witness (Whiteout
//! Definition `wo:def:residual`).

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

    /// `self + scale · other`, with like terms merged.
    pub fn add_scaled(&self, other: &Self, scale: F) -> Self {
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
/// operand per product (Whiteout §3.2), so a lattice back end can add the
/// scalarity rows of x's pad.
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

/// x · y = z.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProductRow<F> {
    pub x: MaskedOperand<F>,
    pub y: AffineRow<F>,
    pub z: AffineRow<F>,
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
                Row::Product(row) => {
                    row.x.form().evaluate(pads, witness)? * row.y.evaluate(pads, witness)?
                        == row.z.evaluate(pads, witness)?
                }
            };
            if !satisfied {
                return Ok(Some(index));
            }
        }
        Ok(None)
    }
}

/// A verifier that contributes rows after replaying its part of the
/// transcript: Jolt's stages (through the verifier replay) and, in W3,
/// Akita's folds up to the cut point.
pub trait ResidualRows<F> {
    type Error;
    fn append_rows(&self, relation: &mut ResidualRelation<F>) -> Result<(), Self::Error>;
}
