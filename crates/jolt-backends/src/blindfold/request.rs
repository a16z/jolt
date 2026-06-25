use std::collections::BTreeSet;

use jolt_field::Field;
use jolt_r1cs::ConstraintMatrices;

use crate::{BackendError, BackendValueSlot};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlindFoldSlot(pub u32);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlindFoldRoundRequest<F: Field> {
    pub slot: BlindFoldSlot,
    pub coefficients: Vec<BackendValueSlot>,
    pub blinding_label: &'static str,
    pub _field: core::marker::PhantomData<F>,
}

impl<F: Field> BlindFoldRoundRequest<F> {
    pub const fn new(
        slot: BlindFoldSlot,
        coefficients: Vec<BackendValueSlot>,
        blinding_label: &'static str,
    ) -> Self {
        Self {
            slot,
            coefficients,
            blinding_label,
            _field: core::marker::PhantomData,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlindFoldRequest<F: Field> {
    pub label: &'static str,
    pub rounds: Vec<BlindFoldRoundRequest<F>>,
    pub output_claims: Vec<BackendValueSlot>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlindFoldRowCommitmentRequest<'a, F: Field> {
    pub label: &'static str,
    pub rows: &'a [Vec<F>],
    pub blindings: &'a [F],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlindFoldRowOpeningRequest<'a, F: Field> {
    pub label: &'static str,
    pub rows: &'a [Vec<F>],
    pub blindings: &'a [F],
    pub row_point: &'a [F],
    pub entry_point: &'a [F],
}

impl<'a, F: Field> BlindFoldRowOpeningRequest<'a, F> {
    pub const fn new(
        label: &'static str,
        rows: &'a [Vec<F>],
        blindings: &'a [F],
        row_point: &'a [F],
        entry_point: &'a [F],
    ) -> Self {
        Self {
            label,
            rows,
            blindings,
            row_point,
            entry_point,
        }
    }

    pub fn validate(&self, backend: &'static str, capacity: usize) -> Result<(), BackendError> {
        validate_row_opening_shape(
            backend,
            self.label,
            self.rows,
            self.blindings,
            self.row_point.len(),
            self.entry_point.len(),
            capacity,
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BlindFoldErrorRowsRequest<'a, F: Field> {
    pub label: &'static str,
    pub r1cs: &'a ConstraintMatrices<F>,
    pub u: F,
    pub witness: &'a [F],
    pub row_count: usize,
    pub row_len: usize,
}

impl<'a, F: Field> BlindFoldErrorRowsRequest<'a, F> {
    pub const fn new(
        label: &'static str,
        r1cs: &'a ConstraintMatrices<F>,
        u: F,
        witness: &'a [F],
        row_count: usize,
        row_len: usize,
    ) -> Self {
        Self {
            label,
            r1cs,
            u,
            witness,
            row_count,
            row_len,
        }
    }

    pub fn validate(&self, backend: &'static str) -> Result<(), BackendError> {
        validate_error_row_shape(
            backend,
            self.label,
            self.r1cs,
            self.row_count,
            self.row_len,
            [self.witness.len() + 1],
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BlindFoldCrossTermErrorRowsRequest<'a, F: Field> {
    pub label: &'static str,
    pub r1cs: &'a ConstraintMatrices<F>,
    pub real_u: F,
    pub real_witness: &'a [F],
    pub random_u: F,
    pub random_witness: &'a [F],
    pub row_count: usize,
    pub row_len: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlindFoldFoldRowsRequest<'a, F: Field> {
    pub label: &'static str,
    pub real: &'a [Vec<F>],
    pub random: &'a [Vec<F>],
    pub challenge: F,
}

impl<'a, F: Field> BlindFoldFoldRowsRequest<'a, F> {
    pub const fn new(
        label: &'static str,
        real: &'a [Vec<F>],
        random: &'a [Vec<F>],
        challenge: F,
    ) -> Self {
        Self {
            label,
            real,
            random,
            challenge,
        }
    }

    pub fn validate(&self, backend: &'static str) -> Result<(), BackendError> {
        validate_binary_rows(backend, self.label, self.real, self.random)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlindFoldFoldScalarsRequest<'a, F: Field> {
    pub label: &'static str,
    pub real: &'a [F],
    pub random: &'a [F],
    pub challenge: F,
}

impl<'a, F: Field> BlindFoldFoldScalarsRequest<'a, F> {
    pub const fn new(label: &'static str, real: &'a [F], random: &'a [F], challenge: F) -> Self {
        Self {
            label,
            real,
            random,
            challenge,
        }
    }

    pub fn validate(&self, backend: &'static str) -> Result<(), BackendError> {
        validate_binary_scalars(backend, self.label, self.real.len(), self.random.len())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlindFoldFoldErrorRowsRequest<'a, F: Field> {
    pub label: &'static str,
    pub real: &'a [Vec<F>],
    pub cross: &'a [Vec<F>],
    pub random: &'a [Vec<F>],
    pub challenge: F,
}

impl<'a, F: Field> BlindFoldFoldErrorRowsRequest<'a, F> {
    pub const fn new(
        label: &'static str,
        real: &'a [Vec<F>],
        cross: &'a [Vec<F>],
        random: &'a [Vec<F>],
        challenge: F,
    ) -> Self {
        Self {
            label,
            real,
            cross,
            random,
            challenge,
        }
    }

    pub fn validate(&self, backend: &'static str) -> Result<(), BackendError> {
        validate_ternary_rows(backend, self.label, self.real, self.cross, self.random)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlindFoldFoldErrorScalarsRequest<'a, F: Field> {
    pub label: &'static str,
    pub real: &'a [F],
    pub cross: &'a [F],
    pub random: &'a [F],
    pub challenge: F,
}

impl<'a, F: Field> BlindFoldFoldErrorScalarsRequest<'a, F> {
    pub const fn new(
        label: &'static str,
        real: &'a [F],
        cross: &'a [F],
        random: &'a [F],
        challenge: F,
    ) -> Self {
        Self {
            label,
            real,
            cross,
            random,
            challenge,
        }
    }

    pub fn validate(&self, backend: &'static str) -> Result<(), BackendError> {
        validate_ternary_scalars(
            backend,
            self.label,
            self.real.len(),
            self.cross.len(),
            self.random.len(),
        )
    }
}

impl<'a, F: Field> BlindFoldCrossTermErrorRowsRequest<'a, F> {
    #[expect(
        clippy::too_many_arguments,
        reason = "cross-term error rows are defined by two relaxed witnesses"
    )]
    pub const fn new(
        label: &'static str,
        r1cs: &'a ConstraintMatrices<F>,
        real_u: F,
        real_witness: &'a [F],
        random_u: F,
        random_witness: &'a [F],
        row_count: usize,
        row_len: usize,
    ) -> Self {
        Self {
            label,
            r1cs,
            real_u,
            real_witness,
            random_u,
            random_witness,
            row_count,
            row_len,
        }
    }

    pub fn validate(&self, backend: &'static str) -> Result<(), BackendError> {
        if self.real_witness.len() != self.random_witness.len() {
            return Err(invalid_error_rows(
                backend,
                format!(
                    "{} real witness has {} values but random witness has {}",
                    self.label,
                    self.real_witness.len(),
                    self.random_witness.len()
                ),
            ));
        }
        validate_error_row_shape(
            backend,
            self.label,
            self.r1cs,
            self.row_count,
            self.row_len,
            [self.real_witness.len() + 1, self.random_witness.len() + 1],
        )
    }
}

impl<'a, F: Field> BlindFoldRowCommitmentRequest<'a, F> {
    pub const fn new(label: &'static str, rows: &'a [Vec<F>], blindings: &'a [F]) -> Self {
        Self {
            label,
            rows,
            blindings,
        }
    }

    pub fn validate(&self, backend: &'static str, capacity: usize) -> Result<(), BackendError> {
        if self.label.trim().is_empty() {
            return Err(invalid_row_commitment(backend, "label must not be empty"));
        }
        if self.rows.len() != self.blindings.len() {
            return Err(invalid_row_commitment(
                backend,
                format!(
                    "{} rows have {} entries but {} blindings",
                    self.label,
                    self.rows.len(),
                    self.blindings.len()
                ),
            ));
        }
        for (index, row) in self.rows.iter().enumerate() {
            if row.len() > capacity {
                return Err(invalid_row_commitment(
                    backend,
                    format!(
                        "{} row {index} has length {}, exceeding capacity {capacity}",
                        self.label,
                        row.len()
                    ),
                ));
            }
        }
        Ok(())
    }
}

impl<F: Field> BlindFoldRequest<F> {
    pub const fn new(
        label: &'static str,
        rounds: Vec<BlindFoldRoundRequest<F>>,
        output_claims: Vec<BackendValueSlot>,
    ) -> Self {
        Self {
            label,
            rounds,
            output_claims,
        }
    }

    pub fn validate(&self, backend: &'static str) -> Result<(), BackendError> {
        if self.label.trim().is_empty() {
            return Err(invalid(backend, "label must not be empty"));
        }
        if self.rounds.is_empty() && self.output_claims.is_empty() {
            return Err(invalid(
                backend,
                "request must include committed rounds or output-claim rows",
            ));
        }

        let mut round_slots = BTreeSet::new();
        for round in &self.rounds {
            if !round_slots.insert(round.slot) {
                return Err(invalid(
                    backend,
                    format!("duplicate BlindFold round slot {:?}", round.slot),
                ));
            }
            if round.coefficients.is_empty() {
                return Err(invalid(
                    backend,
                    format!("BlindFold round {:?} must include coefficients", round.slot),
                ));
            }
            let mut coefficient_slots = BTreeSet::new();
            for slot in &round.coefficients {
                if !coefficient_slots.insert(*slot) {
                    return Err(invalid(
                        backend,
                        format!(
                            "duplicate coefficient slot {slot:?} in round {:?}",
                            round.slot
                        ),
                    ));
                }
            }
        }

        let mut output_claim_slots = BTreeSet::new();
        for slot in &self.output_claims {
            if !output_claim_slots.insert(*slot) {
                return Err(invalid(
                    backend,
                    format!("duplicate output-claim slot {slot:?}"),
                ));
            }
        }

        Ok(())
    }
}

fn invalid(backend: &'static str, reason: impl Into<String>) -> BackendError {
    BackendError::InvalidRequest {
        backend,
        task: "blindfold",
        reason: reason.into(),
    }
}

fn invalid_row_commitment(backend: &'static str, reason: impl Into<String>) -> BackendError {
    BackendError::InvalidRequest {
        backend,
        task: "blindfold row commitments",
        reason: reason.into(),
    }
}

fn invalid_row_opening(backend: &'static str, reason: impl Into<String>) -> BackendError {
    BackendError::InvalidRequest {
        backend,
        task: "blindfold row openings",
        reason: reason.into(),
    }
}

fn invalid_error_rows(backend: &'static str, reason: impl Into<String>) -> BackendError {
    BackendError::InvalidRequest {
        backend,
        task: "blindfold error rows",
        reason: reason.into(),
    }
}

fn invalid_fold(backend: &'static str, reason: impl Into<String>) -> BackendError {
    BackendError::InvalidRequest {
        backend,
        task: "blindfold folding",
        reason: reason.into(),
    }
}

fn validate_nonempty_label(
    backend: &'static str,
    task_label: &'static str,
) -> Result<(), BackendError> {
    if task_label.trim().is_empty() {
        return Err(invalid_fold(backend, "label must not be empty"));
    }
    Ok(())
}

fn validate_binary_scalars(
    backend: &'static str,
    label: &'static str,
    real_len: usize,
    random_len: usize,
) -> Result<(), BackendError> {
    validate_nonempty_label(backend, label)?;
    if real_len != random_len {
        return Err(invalid_fold(
            backend,
            format!("{label} length mismatch: real {real_len}, random {random_len}"),
        ));
    }
    Ok(())
}

fn validate_ternary_scalars(
    backend: &'static str,
    label: &'static str,
    real_len: usize,
    cross_len: usize,
    random_len: usize,
) -> Result<(), BackendError> {
    validate_nonempty_label(backend, label)?;
    if real_len != cross_len || real_len != random_len {
        return Err(invalid_fold(
            backend,
            format!(
                "{label} length mismatch: real {real_len}, cross {cross_len}, random {random_len}"
            ),
        ));
    }
    Ok(())
}

fn validate_binary_rows<F: Field>(
    backend: &'static str,
    label: &'static str,
    real: &[Vec<F>],
    random: &[Vec<F>],
) -> Result<(), BackendError> {
    validate_binary_scalars(backend, label, real.len(), random.len())?;
    for (row_index, (real_row, random_row)) in real.iter().zip(random).enumerate() {
        if real_row.len() != random_row.len() {
            return Err(invalid_fold(
                backend,
                format!(
                    "{label} row {row_index} length mismatch: real {}, random {}",
                    real_row.len(),
                    random_row.len()
                ),
            ));
        }
    }
    Ok(())
}

fn validate_ternary_rows<F: Field>(
    backend: &'static str,
    label: &'static str,
    real: &[Vec<F>],
    cross: &[Vec<F>],
    random: &[Vec<F>],
) -> Result<(), BackendError> {
    validate_ternary_scalars(backend, label, real.len(), cross.len(), random.len())?;
    for (row_index, ((real_row, cross_row), random_row)) in
        real.iter().zip(cross).zip(random).enumerate()
    {
        if real_row.len() != cross_row.len() || real_row.len() != random_row.len() {
            return Err(invalid_fold(
                backend,
                format!(
                    "{label} row {row_index} length mismatch: real {}, cross {}, random {}",
                    real_row.len(),
                    cross_row.len(),
                    random_row.len()
                ),
            ));
        }
    }
    Ok(())
}

fn validate_row_opening_shape<F: Field>(
    backend: &'static str,
    label: &'static str,
    rows: &[Vec<F>],
    blindings: &[F],
    row_point_len: usize,
    entry_point_len: usize,
    capacity: usize,
) -> Result<(), BackendError> {
    if label.trim().is_empty() {
        return Err(invalid_row_opening(backend, "label must not be empty"));
    }
    let row_count = basis_len_from_point_len(backend, "row point", row_point_len)?;
    if rows.len() != row_count {
        return Err(invalid_row_opening(
            backend,
            format!(
                "{label} has {} rows but row point selects {row_count}",
                rows.len()
            ),
        ));
    }
    if blindings.len() != row_count {
        return Err(invalid_row_opening(
            backend,
            format!(
                "{label} has {} blindings but row point selects {row_count}",
                blindings.len()
            ),
        ));
    }
    let row_len = rows.first().map_or(0, Vec::len);
    let expected_row_len = basis_len_from_point_len(backend, "entry point", entry_point_len)?;
    if row_len != expected_row_len {
        return Err(invalid_row_opening(
            backend,
            format!(
                "{label} row length {row_len} does not match entry point length {entry_point_len}"
            ),
        ));
    }
    if row_len > capacity {
        return Err(invalid_row_opening(
            backend,
            format!("{label} row length {row_len} exceeds capacity {capacity}"),
        ));
    }
    for (row_index, row) in rows.iter().enumerate() {
        if row.len() != row_len {
            return Err(invalid_row_opening(
                backend,
                format!(
                    "{label} row {row_index} length mismatch: expected {row_len}, got {}",
                    row.len()
                ),
            ));
        }
    }
    Ok(())
}

fn basis_len_from_point_len(
    backend: &'static str,
    name: &'static str,
    point_len: usize,
) -> Result<usize, BackendError> {
    if point_len >= usize::BITS as usize {
        return Err(invalid_row_opening(
            backend,
            format!("{name} length {point_len} cannot be represented"),
        ));
    }
    Ok(1_usize << point_len)
}

fn validate_error_row_shape<const N: usize, F: Field>(
    backend: &'static str,
    label: &'static str,
    r1cs: &ConstraintMatrices<F>,
    row_count: usize,
    row_len: usize,
    witness_lengths: [usize; N],
) -> Result<(), BackendError> {
    if label.trim().is_empty() {
        return Err(invalid_error_rows(backend, "label must not be empty"));
    }
    if row_count == 0 || !row_count.is_power_of_two() {
        return Err(invalid_error_rows(
            backend,
            format!("{label} row count must be a nonzero power of two, got {row_count}"),
        ));
    }
    if row_len == 0 || !row_len.is_power_of_two() {
        return Err(invalid_error_rows(
            backend,
            format!("{label} row length must be a nonzero power of two, got {row_len}"),
        ));
    }
    let target_len = row_count.checked_mul(row_len).ok_or_else(|| {
        invalid_error_rows(
            backend,
            format!("{label} row dimensions overflow: {row_count} x {row_len}"),
        )
    })?;
    if r1cs.num_constraints > target_len {
        return Err(invalid_error_rows(
            backend,
            format!(
                "{label} has {} constraints but only {target_len} padded error slots",
                r1cs.num_constraints
            ),
        ));
    }
    for (matrix_name, rows) in [("A", &r1cs.a), ("B", &r1cs.b), ("C", &r1cs.c)] {
        if rows.len() < r1cs.num_constraints {
            return Err(invalid_error_rows(
                backend,
                format!(
                    "{label} matrix {matrix_name} has {} rows but {} constraints",
                    rows.len(),
                    r1cs.num_constraints
                ),
            ));
        }
        for (row_index, row) in rows.iter().take(r1cs.num_constraints).enumerate() {
            for &(column, _) in row {
                if witness_lengths.iter().any(|&length| column >= length) {
                    return Err(invalid_error_rows(
                        backend,
                        format!(
                            "{label} matrix {matrix_name} row {row_index} references column {column} outside witness length(s) {witness_lengths:?}"
                        ),
                    ));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_r1cs::ConstraintMatrices;

    use super::*;

    #[test]
    fn blindfold_request_validation_accepts_well_formed_slots() -> Result<(), BackendError> {
        let request = BlindFoldRequest::<Fr>::new(
            "blindfold",
            vec![
                BlindFoldRoundRequest::new(
                    BlindFoldSlot(0),
                    vec![BackendValueSlot(0), BackendValueSlot(1)],
                    "r0",
                ),
                BlindFoldRoundRequest::new(
                    BlindFoldSlot(1),
                    vec![BackendValueSlot(2), BackendValueSlot(3)],
                    "r1",
                ),
            ],
            vec![BackendValueSlot(10), BackendValueSlot(11)],
        );

        request.validate("cpu")?;
        Ok(())
    }

    #[test]
    fn blindfold_request_validation_rejects_empty_or_duplicate_slots() {
        assert!(BlindFoldRequest::<Fr>::new("", Vec::new(), Vec::new())
            .validate("cpu")
            .is_err());
        assert!(
            BlindFoldRequest::<Fr>::new("blindfold", Vec::new(), Vec::new())
                .validate("cpu")
                .is_err()
        );
        assert!(BlindFoldRequest::<Fr>::new(
            "blindfold",
            vec![
                BlindFoldRoundRequest::new(BlindFoldSlot(0), vec![BackendValueSlot(0)], "r0",),
                BlindFoldRoundRequest::new(BlindFoldSlot(0), vec![BackendValueSlot(1)], "r1",),
            ],
            Vec::new(),
        )
        .validate("cpu")
        .is_err());
        assert!(BlindFoldRequest::<Fr>::new(
            "blindfold",
            vec![BlindFoldRoundRequest::new(
                BlindFoldSlot(0),
                vec![BackendValueSlot(0), BackendValueSlot(0)],
                "r0",
            )],
            Vec::new(),
        )
        .validate("cpu")
        .is_err());
        assert!(BlindFoldRequest::<Fr>::new(
            "blindfold",
            Vec::new(),
            vec![BackendValueSlot(0), BackendValueSlot(0)],
        )
        .validate("cpu")
        .is_err());
    }

    #[test]
    fn blindfold_row_commitment_validation_checks_lengths_and_capacity() -> Result<(), BackendError>
    {
        let rows = vec![vec![Fr::from_u64(1), Fr::from_u64(2)]];
        let blindings = vec![Fr::from_u64(3)];
        BlindFoldRowCommitmentRequest::new("rows", &rows, &blindings).validate("cpu", 2)?;

        assert!(BlindFoldRowCommitmentRequest::new("", &rows, &blindings)
            .validate("cpu", 2)
            .is_err());
        assert!(BlindFoldRowCommitmentRequest::new("rows", &rows, &[])
            .validate("cpu", 2)
            .is_err());
        assert!(
            BlindFoldRowCommitmentRequest::new("rows", &rows, &blindings)
                .validate("cpu", 1)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn blindfold_row_opening_validation_checks_shape_and_capacity() -> Result<(), BackendError> {
        let rows = vec![
            vec![Fr::from_u64(1), Fr::from_u64(2)],
            vec![Fr::from_u64(3), Fr::from_u64(4)],
        ];
        let blindings = vec![Fr::from_u64(5), Fr::from_u64(6)];
        let row_point = vec![Fr::from_u64(7)];
        let entry_point = vec![Fr::from_u64(8)];
        BlindFoldRowOpeningRequest::new("rows", &rows, &blindings, &row_point, &entry_point)
            .validate("cpu", 2)?;

        assert!(
            BlindFoldRowOpeningRequest::new("", &rows, &blindings, &row_point, &entry_point)
                .validate("cpu", 2)
                .is_err()
        );
        assert!(BlindFoldRowOpeningRequest::new(
            "rows",
            &rows[..1],
            &blindings,
            &row_point,
            &entry_point
        )
        .validate("cpu", 2)
        .is_err());
        assert!(BlindFoldRowOpeningRequest::new(
            "rows",
            &rows,
            &blindings[..1],
            &row_point,
            &entry_point
        )
        .validate("cpu", 2)
        .is_err());
        assert!(
            BlindFoldRowOpeningRequest::new("rows", &rows, &blindings, &row_point, &[])
                .validate("cpu", 2)
                .is_err()
        );
        assert!(BlindFoldRowOpeningRequest::new(
            "rows",
            &rows,
            &blindings,
            &row_point,
            &entry_point
        )
        .validate("cpu", 1)
        .is_err());
        Ok(())
    }

    #[test]
    fn blindfold_error_row_validation_checks_dimensions_and_witness_columns(
    ) -> Result<(), BackendError> {
        let r1cs = tiny_r1cs();
        let witness = vec![Fr::from_u64(2), Fr::from_u64(3)];
        BlindFoldErrorRowsRequest::new("errors", &r1cs, Fr::from_u64(1), &witness, 1, 1)
            .validate("cpu")?;

        assert!(
            BlindFoldErrorRowsRequest::new("errors", &r1cs, Fr::from_u64(1), &witness, 0, 1)
                .validate("cpu")
                .is_err()
        );
        assert!(
            BlindFoldErrorRowsRequest::new("errors", &r1cs, Fr::from_u64(1), &witness, 1, 3)
                .validate("cpu")
                .is_err()
        );
        assert!(
            BlindFoldErrorRowsRequest::new("errors", &r1cs, Fr::from_u64(1), &[], 1, 1)
                .validate("cpu")
                .is_err()
        );
        assert!(BlindFoldCrossTermErrorRowsRequest::new(
            "cross",
            &r1cs,
            Fr::from_u64(1),
            &witness,
            Fr::from_u64(4),
            &witness[..1],
            1,
            1,
        )
        .validate("cpu")
        .is_err());
        Ok(())
    }

    #[test]
    fn blindfold_fold_validation_checks_lengths_and_row_shapes() -> Result<(), BackendError> {
        let real_rows = vec![vec![Fr::from_u64(1), Fr::from_u64(2)]];
        let random_rows = vec![vec![Fr::from_u64(3), Fr::from_u64(4)]];
        BlindFoldFoldRowsRequest::new("rows", &real_rows, &random_rows, Fr::from_u64(5))
            .validate("cpu")?;

        let short_rows = vec![vec![Fr::from_u64(3)]];
        assert!(
            BlindFoldFoldRowsRequest::new("rows", &real_rows, &short_rows, Fr::from_u64(5))
                .validate("cpu")
                .is_err()
        );
        assert!(
            BlindFoldFoldRowsRequest::new("", &real_rows, &random_rows, Fr::from_u64(5))
                .validate("cpu")
                .is_err()
        );

        let real_scalars = vec![Fr::from_u64(1), Fr::from_u64(2)];
        let random_scalars = vec![Fr::from_u64(3), Fr::from_u64(4)];
        BlindFoldFoldScalarsRequest::new(
            "scalars",
            &real_scalars,
            &random_scalars,
            Fr::from_u64(5),
        )
        .validate("cpu")?;
        assert!(BlindFoldFoldErrorScalarsRequest::new(
            "error scalars",
            &real_scalars,
            &random_scalars[..1],
            &random_scalars,
            Fr::from_u64(5),
        )
        .validate("cpu")
        .is_err());
        Ok(())
    }

    fn tiny_r1cs() -> ConstraintMatrices<Fr> {
        ConstraintMatrices::new(
            1,
            3,
            vec![vec![(1, Fr::from_u64(1))]],
            vec![vec![(2, Fr::from_u64(1))]],
            vec![vec![(0, Fr::from_u64(1))]],
        )
    }
}
