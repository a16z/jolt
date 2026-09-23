//! Dense reference tables in the configured read/write round order.

use jolt_claims::protocols::jolt::ReadWriteDimensions;
use jolt_field::JoltField;
use jolt_poly::Polynomial;

use crate::KernelError;

/// Canonical big-endian table variables mapped to low-to-high binding rounds.
/// Omitted rounds repeat coefficients, so the naive evaluator independently
/// accounts for the RAM address relations' unused cycle variables and scaling.
pub(super) struct ReadWriteTableLayout {
    rounds: usize,
    variables: Vec<usize>,
}

impl ReadWriteTableLayout {
    pub(super) fn joint<F: JoltField>(
        dimensions: ReadWriteDimensions,
    ) -> Result<Self, KernelError<F>> {
        Self::validate::<F>(dimensions)?;
        let rounds = dimensions.read_write_rounds();
        let variables = dimensions
            .read_write_opening_indices()
            .map_err(|_| KernelError::InvariantViolation {
                reason: "invalid read/write opening geometry",
            })?
            .collect();
        Ok(Self { rounds, variables })
    }

    pub(super) fn address<F: JoltField>(
        dimensions: ReadWriteDimensions,
    ) -> Result<Self, KernelError<F>> {
        Self::validate::<F>(dimensions)?;
        let rounds = dimensions.output_check_rounds();
        let variables = dimensions
            .address_opening_indices()
            .map_err(|_| KernelError::InvariantViolation {
                reason: "invalid RAM address opening geometry",
            })?
            .collect();
        Ok(Self { rounds, variables })
    }

    fn validate<F: JoltField>(dimensions: ReadWriteDimensions) -> Result<(), KernelError<F>> {
        dimensions
            .validate_phase_split()
            .map_err(|_| KernelError::InvariantViolation {
                reason: "read/write phase split exceeds the cycle/address dimensions",
            })?;
        if dimensions
            .log_t()
            .checked_add(dimensions.log_k())
            .is_none_or(|rounds| rounds >= usize::BITS as usize)
        {
            return Err(KernelError::Unsupported {
                reason: "dense read/write reference tables exceed the host index width",
            });
        }
        Ok(())
    }

    pub(super) fn table<F: JoltField>(
        &self,
        values: Vec<F>,
    ) -> Result<Polynomial<F>, KernelError<F>> {
        let expected = 1usize << self.variables.len();
        if values.len() != expected {
            return Err(KernelError::TableSizeMismatch {
                table: "read/write reference table".into(),
                expected,
                got: values.len(),
            });
        }
        if self.variables.iter().copied().eq((0..self.rounds).rev()) {
            return Ok(Polynomial::new(values));
        }
        let reordered = (0..1usize << self.rounds)
            .map(|index| {
                let source = self
                    .variables
                    .iter()
                    .fold(0, |source, &round| (source << 1) | ((index >> round) & 1));
                values[source]
            })
            .collect();
        Ok(Polynomial::new(reordered))
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::ReadWriteDimensions;
    use jolt_field::{Fr as F, Ring, Zero};

    use super::ReadWriteTableLayout;
    use crate::KernelError;

    #[test]
    fn tables_follow_the_configured_variable_order() {
        // Entry k * 4 + t identifies its original address/cycle coordinates.
        // Expected tables are literal fixtures, independent of the geometry API.
        let joint_values: Vec<F> = (0..16).map(F::from_u64).collect();
        let address_values: Vec<F> = (0..4).map(F::from_u64).collect();
        for (dimensions, expected_joint, expected_address) in [
            (
                ReadWriteDimensions::new(2, 2, 2, 2),
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                vec![0, 1, 2, 3],
            ),
            (
                ReadWriteDimensions::new(2, 2, 0, 2),
                [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15],
                vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            ),
            (
                ReadWriteDimensions::new(2, 2, 1, 1),
                [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15],
                vec![0, 1, 0, 1, 2, 3, 2, 3],
            ),
            (
                ReadWriteDimensions::new(2, 2, 0, 0),
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            ),
        ] {
            let layout = ReadWriteTableLayout::joint::<F>(dimensions).unwrap();
            let table = layout.table(joint_values.clone()).unwrap();
            assert_eq!(
                table.evals(),
                &expected_joint.map(F::from_u64),
                "joint layout: {dimensions:?}"
            );

            let layout = ReadWriteTableLayout::address::<F>(dimensions).unwrap();
            let table = layout.table(address_values.clone()).unwrap();
            assert_eq!(
                table.evals(),
                &expected_address
                    .into_iter()
                    .map(F::from_u64)
                    .collect::<Vec<_>>(),
                "address layout: {dimensions:?}"
            );
        }
    }

    #[test]
    fn rejects_invalid_geometry_and_table_sizes() {
        for dimensions in [
            ReadWriteDimensions::new(3, 2, 4, 2),
            ReadWriteDimensions::new(3, 2, 0, 3),
        ] {
            assert!(matches!(
                ReadWriteTableLayout::joint::<F>(dimensions),
                Err(KernelError::InvariantViolation { .. })
            ));
            assert!(matches!(
                ReadWriteTableLayout::address::<F>(dimensions),
                Err(KernelError::InvariantViolation { .. })
            ));
        }
        assert!(matches!(
            ReadWriteTableLayout::joint::<F>(ReadWriteDimensions::new(
                usize::BITS as usize,
                1,
                0,
                1
            )),
            Err(KernelError::Unsupported { .. })
        ));
        let layout =
            ReadWriteTableLayout::joint::<F>(ReadWriteDimensions::new(3, 2, 0, 2)).unwrap();
        assert!(matches!(
            layout.table(vec![F::zero(); 16]),
            Err(KernelError::TableSizeMismatch {
                expected: 32,
                got: 16,
                ..
            })
        ));
    }
}
