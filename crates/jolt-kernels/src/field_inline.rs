//! Proof-local ownership of the field-register increment column.

use crate::optimized::support::map_indices;
use crate::{KernelError, ProofSession};
use jolt_claims::protocols::field_inline::{
    FieldInlineCommittedPolynomial, FieldInlinePolynomialId,
};
use jolt_field::JoltField;
use jolt_poly::bind_prefix_in_place;
use jolt_witness::field_inline::FieldInlineWitnessOracle;
use std::sync::Arc;

/// One immutable increment column shared by commitment, sumchecks and opening.
/// Storage ends at the last nonzero value; the remaining cycles are zero.
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
    /// Retain an explicit prefix, with zero increments through `cycles`.
    pub fn from_prefix(mut values: Vec<F>, cycles: usize) -> Result<Self, KernelError<F>> {
        if values.len() > cycles {
            return Err(KernelError::TableSizeMismatch {
                table: "FieldRdInc prefix".to_owned(),
                expected: cycles,
                got: values.len(),
            });
        }
        let prefix = values
            .iter()
            .rposition(|value| !value.is_zero())
            .map_or(0, |i| i + 1);
        if prefix == 0 {
            values = Vec::new();
        } else {
            values.truncate(prefix);
            if values.capacity() / 2 >= prefix {
                values.shrink_to_fit();
            }
        }
        Ok(Self {
            values: Arc::new(values),
            cycles,
        })
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
            let values = oracle.oracle_table(FieldInlinePolynomialId::Committed(
                FieldInlineCommittedPolynomial::FieldRdInc,
            ))?;
            if values.len() != cycles {
                return Err(KernelError::TableSizeMismatch {
                    table: "FieldRdInc".to_owned(),
                    expected: cycles,
                    got: values.len(),
                });
            }
            let column = Self::from_prefix(values, cycles)?;
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
    pub(crate) fn prefix_len(&self) -> usize {
        self.values.len()
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

/// The unbound round reads shared storage; binds retain only the nonzero
/// prefix. All-zero columns stay allocation-free throughout the sumcheck.
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
pub(crate) enum IncrementRounds<F: JoltField> {
    Shared(FieldIncrementColumn<F>),
    Bound(Vec<F>),
    Zero,
}

impl<F: JoltField> IncrementRounds<F> {
    pub(crate) fn new(column: FieldIncrementColumn<F>) -> Self {
        if column.is_zero() {
            Self::Zero
        } else {
            Self::Shared(column)
        }
    }
    pub(crate) fn prefix_len(&self) -> usize {
        match self {
            Self::Shared(c) => c.prefix_len(),
            Self::Bound(values) => values.len(),
            Self::Zero => 0,
        }
    }
    pub(crate) fn value(&self, index: usize) -> F {
        match self {
            Self::Shared(c) => c.value(index),
            Self::Bound(values) => values.get(index).copied().unwrap_or_else(F::zero),
            Self::Zero => F::zero(),
        }
    }
    pub(crate) fn pair(&self, index: usize) -> (F, F) {
        (self.value(2 * index), self.value(2 * index + 1))
    }
    pub(crate) fn bind(&mut self, r: F) {
        match self {
            Self::Shared(column) => {
                let values = map_indices(column.prefix_len().div_ceil(2), |i| {
                    let lo = column.value(2 * i);
                    let hi = column.value(2 * i + 1);
                    lo + r * (hi - lo)
                });
                *self = Self::Bound(values);
            }
            Self::Bound(values) => {
                let _ = bind_prefix_in_place(values, r, F::zero());
            }
            Self::Zero => {}
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_field::{Fr, Ring, Zero};
    use jolt_poly::Polynomial;

    #[test]
    fn field_increment_prefix_binding_preserves_evaluation() {
        assert!(FieldIncrementColumn::from_prefix(vec![Fr::zero(); 33], 32).is_err());
        for prefix in [0usize, 1, 2, 3, 15, 16, 17, 31, 32] {
            let mut values = vec![Fr::zero(); 32];
            for (i, value) in values[..prefix].iter_mut().enumerate() {
                *value = Fr::from_u64(i as u64 + 1);
            }
            let poly = Polynomial::new(values.clone());
            let column = FieldIncrementColumn::from_prefix(values[..prefix].to_vec(), 32).unwrap();
            assert_eq!(column.len(), 32);
            let mut rounds = IncrementRounds::new(column);
            let point: Vec<_> = (3..8).map(Fr::from_u64).collect();
            for (round, &challenge) in point.iter().rev().enumerate() {
                rounds.bind(challenge);
                assert_eq!(rounds.prefix_len(), prefix.div_ceil(1 << (round + 1)));
            }
            assert_eq!(rounds.value(0), poly.evaluate(&point));
        }
    }
}
