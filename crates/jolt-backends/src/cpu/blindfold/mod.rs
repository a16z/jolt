//! CPU BlindFold private-material compute modules.

use jolt_crypto::VectorCommitment;
use jolt_field::{Field, RingAccumulator, WithAccumulator};
use jolt_r1cs::ConstraintMatrices;
use rayon::prelude::*;

use crate::{
    Backend, BackendError, BlindFoldBackend, BlindFoldCrossTermErrorRowsRequest,
    BlindFoldErrorRowsRequest, BlindFoldErrorRowsResult, BlindFoldFoldErrorRowsRequest,
    BlindFoldFoldErrorScalarsRequest, BlindFoldFoldRowsRequest, BlindFoldFoldRowsResult,
    BlindFoldFoldScalarsRequest, BlindFoldFoldScalarsResult, BlindFoldRowCommitmentRequest,
    BlindFoldRowCommitmentResult, BlindFoldRowOpeningRequest, BlindFoldRowOpeningResult,
};

use super::CpuBackend;

impl<F> BlindFoldBackend<F> for CpuBackend
where
    F: Field,
{
    type Proof = ();

    fn commit_blindfold_rows<VC>(
        &mut self,
        request: BlindFoldRowCommitmentRequest<'_, F>,
        setup: &VC::Setup,
    ) -> Result<BlindFoldRowCommitmentResult<VC::Output>, BackendError>
    where
        VC: VectorCommitment<Field = F>,
    {
        request.validate(self.name(), VC::capacity(setup))?;
        Ok(BlindFoldRowCommitmentResult::new(
            request
                .rows
                .par_iter()
                .zip(request.blindings.par_iter())
                .map(|(row, blinding)| VC::commit(setup, row, blinding))
                .collect(),
        ))
    }

    fn compute_blindfold_error_rows(
        &mut self,
        request: BlindFoldErrorRowsRequest<'_, F>,
    ) -> Result<BlindFoldErrorRowsResult<F>, BackendError> {
        request.validate(self.name())?;
        Ok(BlindFoldErrorRowsResult::new(error_rows_for(
            request.r1cs,
            request.u,
            request.witness,
            request.row_count,
            request.row_len,
        )?))
    }

    fn compute_blindfold_cross_term_error_rows(
        &mut self,
        request: BlindFoldCrossTermErrorRowsRequest<'_, F>,
    ) -> Result<BlindFoldErrorRowsResult<F>, BackendError> {
        request.validate(self.name())?;
        Ok(BlindFoldErrorRowsResult::new(cross_term_error_rows_for(
            request.r1cs,
            request.real_u,
            request.real_witness,
            request.random_u,
            request.random_witness,
            request.row_count,
            request.row_len,
        )?))
    }

    fn fold_blindfold_rows(
        &mut self,
        request: BlindFoldFoldRowsRequest<'_, F>,
    ) -> Result<BlindFoldFoldRowsResult<F>, BackendError> {
        request.validate(self.name())?;
        Ok(BlindFoldFoldRowsResult::new(fold_rows(
            request.real,
            request.random,
            request.challenge,
        )))
    }

    fn fold_blindfold_scalars(
        &mut self,
        request: BlindFoldFoldScalarsRequest<'_, F>,
    ) -> Result<BlindFoldFoldScalarsResult<F>, BackendError> {
        request.validate(self.name())?;
        Ok(BlindFoldFoldScalarsResult::new(fold_scalars(
            request.real,
            request.random,
            request.challenge,
        )))
    }

    fn open_blindfold_rows<VC>(
        &mut self,
        request: BlindFoldRowOpeningRequest<'_, F>,
        setup: &VC::Setup,
    ) -> Result<BlindFoldRowOpeningResult<F>, BackendError>
    where
        VC: VectorCommitment<Field = F>,
        <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
    {
        request.validate(self.name(), VC::capacity(setup))?;
        let row_len = request.rows.first().map_or(0, Vec::len);
        let (opening, evaluation) = VC::open_committed_rows(
            &flatten(request.rows),
            request.blindings,
            row_len,
            request.row_point,
            request.entry_point,
        )
        .map_err(|error| BackendError::InvalidRequest {
            backend: self.name(),
            task: "blindfold row openings",
            reason: error.to_string(),
        })?;
        Ok(BlindFoldRowOpeningResult::new(opening, evaluation))
    }

    fn fold_blindfold_error_rows(
        &mut self,
        request: BlindFoldFoldErrorRowsRequest<'_, F>,
    ) -> Result<BlindFoldFoldRowsResult<F>, BackendError> {
        request.validate(self.name())?;
        Ok(BlindFoldFoldRowsResult::new(fold_error_rows(
            request.real,
            request.cross,
            request.random,
            request.challenge,
        )))
    }

    fn fold_blindfold_error_scalars(
        &mut self,
        request: BlindFoldFoldErrorScalarsRequest<'_, F>,
    ) -> Result<BlindFoldFoldScalarsResult<F>, BackendError> {
        request.validate(self.name())?;
        Ok(BlindFoldFoldScalarsResult::new(fold_error_scalars(
            request.real,
            request.cross,
            request.random,
            request.challenge,
        )))
    }
}

fn fold_rows<F>(real: &[Vec<F>], random: &[Vec<F>], challenge: F) -> Vec<Vec<F>>
where
    F: Field,
{
    real.iter()
        .zip(random)
        .map(|(real_row, random_row)| {
            real_row
                .iter()
                .zip(random_row)
                .map(|(&real, &random)| real + challenge * random)
                .collect()
        })
        .collect()
}

fn fold_scalars<F>(real: &[F], random: &[F], challenge: F) -> Vec<F>
where
    F: Field,
{
    real.iter()
        .zip(random)
        .map(|(&real, &random)| real + challenge * random)
        .collect()
}

fn fold_error_rows<F>(
    real: &[Vec<F>],
    cross: &[Vec<F>],
    random: &[Vec<F>],
    challenge: F,
) -> Vec<Vec<F>>
where
    F: Field,
{
    let challenge_squared = challenge * challenge;
    real.iter()
        .zip(cross)
        .zip(random)
        .map(|((real_row, cross_row), random_row)| {
            real_row
                .iter()
                .zip(cross_row)
                .zip(random_row)
                .map(|((&real, &cross), &random)| {
                    real + challenge * cross + challenge_squared * random
                })
                .collect()
        })
        .collect()
}

fn fold_error_scalars<F>(real: &[F], cross: &[F], random: &[F], challenge: F) -> Vec<F>
where
    F: Field,
{
    let challenge_squared = challenge * challenge;
    real.iter()
        .zip(cross)
        .zip(random)
        .map(|((&real, &cross), &random)| real + challenge * cross + challenge_squared * random)
        .collect()
}

fn error_rows_for<F>(
    r1cs: &ConstraintMatrices<F>,
    u: F,
    witness: &[F],
    row_count: usize,
    row_len: usize,
) -> Result<Vec<Vec<F>>, BackendError>
where
    F: Field,
{
    let target_len = checked_error_len("error values", row_count, row_len)?;
    let z = z_vector(u, witness);
    let mut errors = (0..r1cs.num_constraints)
        .map(|row_index| {
            dot(&r1cs.a[row_index], &z) * dot(&r1cs.b[row_index], &z)
                - u * dot(&r1cs.c[row_index], &z)
        })
        .collect::<Vec<_>>();
    pad_to_len("error values", &mut errors, target_len)?;
    Ok(errors.chunks(row_len).map(<[F]>::to_vec).collect())
}

fn cross_term_error_rows_for<F>(
    r1cs: &ConstraintMatrices<F>,
    real_u: F,
    real_witness: &[F],
    random_u: F,
    random_witness: &[F],
    row_count: usize,
    row_len: usize,
) -> Result<Vec<Vec<F>>, BackendError>
where
    F: Field,
{
    let target_len = checked_error_len("cross-term error values", row_count, row_len)?;
    let real_z = z_vector(real_u, real_witness);
    let random_z = z_vector(random_u, random_witness);
    let mut errors = (0..r1cs.num_constraints)
        .map(|row_index| {
            dot(&r1cs.a[row_index], &real_z) * dot(&r1cs.b[row_index], &random_z)
                + dot(&r1cs.a[row_index], &random_z) * dot(&r1cs.b[row_index], &real_z)
                - real_u * dot(&r1cs.c[row_index], &random_z)
                - random_u * dot(&r1cs.c[row_index], &real_z)
        })
        .collect::<Vec<_>>();
    pad_to_len("cross-term error values", &mut errors, target_len)?;
    Ok(errors.chunks(row_len).map(<[F]>::to_vec).collect())
}

fn checked_error_len(
    name: &'static str,
    row_count: usize,
    row_len: usize,
) -> Result<usize, BackendError> {
    row_count
        .checked_mul(row_len)
        .ok_or_else(|| BackendError::InvalidRequest {
            backend: "cpu",
            task: "blindfold error rows",
            reason: format!("{name} row dimensions overflow: {row_count} x {row_len}"),
        })
}

fn pad_to_len<F>(
    name: &'static str,
    values: &mut Vec<F>,
    target_len: usize,
) -> Result<(), BackendError>
where
    F: Field,
{
    if values.len() > target_len {
        return Err(BackendError::InvalidRequest {
            backend: "cpu",
            task: "blindfold error rows",
            reason: format!(
                "{name} length mismatch: expected at most {target_len}, got {}",
                values.len()
            ),
        });
    }
    values.resize(target_len, F::zero());
    Ok(())
}

fn z_vector<F>(u: F, witness: &[F]) -> Vec<F>
where
    F: Field,
{
    let mut z = Vec::with_capacity(witness.len() + 1);
    z.push(u);
    z.extend_from_slice(witness);
    z
}

fn dot<F>(row: &[(usize, F)], witness: &[F]) -> F
where
    F: Field,
{
    row.iter()
        .map(|&(column, coefficient)| coefficient * witness[column])
        .sum()
}

fn flatten<F>(rows: &[Vec<F>]) -> Vec<F>
where
    F: Field,
{
    rows.iter().flat_map(|row| row.iter().copied()).collect()
}

#[cfg(test)]
mod tests {
    use jolt_crypto::{Bn254G1, JoltGroup, Pedersen, PedersenSetup};
    use jolt_field::{Fr, FromPrimitiveInt};

    use crate::{
        BlindFoldCrossTermErrorRowsRequest, BlindFoldErrorRowsRequest,
        BlindFoldFoldErrorRowsRequest, BlindFoldFoldErrorScalarsRequest, BlindFoldFoldRowsRequest,
        BlindFoldFoldScalarsRequest, BlindFoldRowOpeningRequest,
    };

    use super::*;

    #[test]
    fn cpu_blindfold_folding_kernels_match_formulas() -> Result<(), BackendError> {
        let mut backend = CpuBackend::default();
        let challenge = Fr::from_u64(5);
        let real_rows = vec![vec![Fr::from_u64(1), Fr::from_u64(2)]];
        let random_rows = vec![vec![Fr::from_u64(3), Fr::from_u64(4)]];
        let folded_rows = backend
            .fold_blindfold_rows(BlindFoldFoldRowsRequest::new(
                "rows",
                &real_rows,
                &random_rows,
                challenge,
            ))?
            .rows;
        assert_eq!(folded_rows, vec![vec![Fr::from_u64(16), Fr::from_u64(22)]]);

        let real_scalars = vec![Fr::from_u64(1), Fr::from_u64(2)];
        let random_scalars = vec![Fr::from_u64(3), Fr::from_u64(4)];
        let folded_scalars = backend
            .fold_blindfold_scalars(BlindFoldFoldScalarsRequest::new(
                "scalars",
                &real_scalars,
                &random_scalars,
                challenge,
            ))?
            .scalars;
        assert_eq!(folded_scalars, vec![Fr::from_u64(16), Fr::from_u64(22)]);

        let cross_rows = vec![vec![Fr::from_u64(7), Fr::from_u64(8)]];
        let folded_error_rows = backend
            .fold_blindfold_error_rows(BlindFoldFoldErrorRowsRequest::new(
                "error rows",
                &real_rows,
                &cross_rows,
                &random_rows,
                challenge,
            ))?
            .rows;
        assert_eq!(
            folded_error_rows,
            vec![vec![Fr::from_u64(111), Fr::from_u64(142)]]
        );

        let cross_scalars = vec![Fr::from_u64(7), Fr::from_u64(8)];
        let folded_error_scalars = backend
            .fold_blindfold_error_scalars(BlindFoldFoldErrorScalarsRequest::new(
                "error scalars",
                &real_scalars,
                &cross_scalars,
                &random_scalars,
                challenge,
            ))?
            .scalars;
        assert_eq!(
            folded_error_scalars,
            vec![Fr::from_u64(111), Fr::from_u64(142)]
        );
        Ok(())
    }

    #[test]
    fn cpu_blindfold_row_opening_matches_vector_commitment_reference() -> Result<(), BackendError> {
        let mut backend = CpuBackend::default();
        let setup = PedersenSetup::new(
            vec![
                <Bn254G1 as JoltGroup>::identity(),
                <Bn254G1 as JoltGroup>::identity(),
            ],
            <Bn254G1 as JoltGroup>::identity(),
        );
        let rows = vec![
            vec![Fr::from_u64(1), Fr::from_u64(2)],
            vec![Fr::from_u64(3), Fr::from_u64(4)],
        ];
        let blindings = vec![Fr::from_u64(5), Fr::from_u64(6)];
        let row_point = vec![Fr::from_u64(7)];
        let entry_point = vec![Fr::from_u64(8)];

        let result = backend.open_blindfold_rows::<Pedersen<Bn254G1>>(
            BlindFoldRowOpeningRequest::new("opening", &rows, &blindings, &row_point, &entry_point),
            &setup,
        )?;
        let (expected_opening, expected_evaluation) = Pedersen::<Bn254G1>::open_committed_rows(
            &flatten(&rows),
            &blindings,
            rows[0].len(),
            &row_point,
            &entry_point,
        )
        .map_err(|error| BackendError::InvalidRequest {
            backend: "test",
            task: "blindfold row openings",
            reason: error.to_string(),
        })?;

        assert_eq!(result.opening, expected_opening);
        assert_eq!(result.evaluation, expected_evaluation);
        Ok(())
    }

    #[test]
    fn cpu_blindfold_error_row_kernels_match_relaxed_r1cs() -> Result<(), BackendError> {
        let mut backend = CpuBackend::default();
        let r1cs = tiny_r1cs();
        let real_witness = vec![Fr::from_u64(2), Fr::from_u64(3)];
        let random_witness = vec![Fr::from_u64(5), Fr::from_u64(6)];

        let error_rows = backend
            .compute_blindfold_error_rows(BlindFoldErrorRowsRequest::new(
                "errors",
                &r1cs,
                Fr::from_u64(1),
                &real_witness,
                2,
                1,
            ))?
            .rows;
        assert_eq!(
            error_rows,
            vec![vec![Fr::from_u64(5)], vec![Fr::from_u64(0)]]
        );

        let cross_term_rows = backend
            .compute_blindfold_cross_term_error_rows(BlindFoldCrossTermErrorRowsRequest::new(
                "cross errors",
                &r1cs,
                Fr::from_u64(1),
                &real_witness,
                Fr::from_u64(4),
                &random_witness,
                2,
                1,
            ))?
            .rows;
        assert_eq!(
            cross_term_rows,
            vec![vec![Fr::from_u64(19)], vec![Fr::from_u64(0)]]
        );
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
