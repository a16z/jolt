//! Proof-local ownership of the field-register increment column.

use crate::optimized::support::map_indices;
use crate::{KernelError, ProofSession};
use jolt_claims::protocols::field_inline::{
    FieldInlineCommittedPolynomial, FieldInlinePolynomialId,
};
use jolt_field::JoltField;
use jolt_poly::{BindingOrder, Polynomial};
use jolt_witness::field_inline::FieldInlineWitnessOracle;
use std::sync::Arc;

/// One immutable increment column shared by commitment, sumchecks and opening.
/// Empty storage represents an all-zero column with the recorded trace length.
#[derive(Clone)]
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
pub struct FieldIncrementColumn<F: JoltField> {
    values: Arc<Vec<F>>,
    cycles: usize,
}

impl<F: JoltField> FieldIncrementColumn<F> {
    pub fn from_values(mut values: Vec<F>) -> Self {
        let cycles = values.len();
        if values.iter().all(|value| value.is_zero()) {
            values = Vec::new();
        }
        Self {
            values: Arc::new(values),
            cycles,
        }
    }

    /// Reuse the proof's column, or extract it for a standalone kernel invocation.
    pub fn resolve(
        session: &mut ProofSession,
        oracle: &dyn FieldInlineWitnessOracle<F>,
        cycles: usize,
    ) -> Result<Self, KernelError<F>> {
        let column = if let Some(column) = session.state::<Self>() {
            column.clone()
        } else {
            let column = Self::from_values(oracle.oracle_table(
                FieldInlinePolynomialId::Committed(FieldInlineCommittedPolynomial::FieldRdInc),
            )?);
            session.park(column.clone());
            column
        };
        if column.cycles != cycles {
            return Err(KernelError::TableSizeMismatch {
                table: "FieldRdInc".to_owned(),
                expected: cycles,
                got: column.cycles,
            });
        }
        Ok(column)
    }

    pub fn len(&self) -> usize {
        self.cycles
    }
    pub fn is_empty(&self) -> bool {
        self.cycles == 0
    }
    pub fn value(&self, cycle: usize) -> F {
        self.values.get(cycle).copied().unwrap_or_else(F::zero)
    }
    pub fn is_zero(&self) -> bool {
        self.values.is_empty()
    }
    pub fn nonzero_entries(&self) -> impl Iterator<Item = (usize, F)> + '_ {
        self.values
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, value)| !value.is_zero())
    }
}

/// The unbound round reads shared storage; its first bind allocates only T/2
/// values. All-zero columns stay allocation-free throughout the sumcheck.
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
pub(crate) enum IncrementRounds<F: JoltField> {
    Shared(FieldIncrementColumn<F>),
    Bound(Polynomial<F>),
    Zero(usize),
}

impl<F: JoltField> IncrementRounds<F> {
    pub(crate) fn new(column: FieldIncrementColumn<F>) -> Self {
        if column.is_zero() {
            Self::Zero(column.len())
        } else {
            Self::Shared(column)
        }
    }
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Shared(c) => c.len(),
            Self::Bound(p) => p.len(),
            Self::Zero(n) => *n,
        }
    }
    pub(crate) fn value(&self, index: usize) -> F {
        match self {
            Self::Shared(c) => c.value(index),
            Self::Bound(p) => p.evals()[index],
            Self::Zero(_) => F::zero(),
        }
    }
    pub(crate) fn pair(&self, index: usize) -> (F, F) {
        (self.value(2 * index), self.value(2 * index + 1))
    }
    pub(crate) fn bind(&mut self, r: F) {
        match self {
            Self::Shared(column) => {
                let values = map_indices(column.len() / 2, |i| {
                    let lo = column.value(2 * i);
                    let hi = column.value(2 * i + 1);
                    lo + r * (hi - lo)
                });
                *self = Self::Bound(Polynomial::new(values));
            }
            Self::Bound(poly) => poly.bind_with_order(r, BindingOrder::LowToHigh),
            Self::Zero(len) => *len /= 2,
        }
    }
}
