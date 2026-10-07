use jolt_crypto::{HomomorphicCommitment, VectorCommitment, VectorCommitmentOpening};
use jolt_field::JoltField;
use jolt_poly::{BindingOrder, EqPolynomial, Polynomial, UnivariatePoly};
use jolt_r1cs::{ConstraintMatrices, ConstraintMatrixEvalError, SparseRow};
use jolt_sumcheck::send_compressed_round;
use jolt_transcript::{Channel, ProverTranscript, Sponge};
use rand_core::RngCore;
use rayon::prelude::*;

use crate::wire::{send_opening, FoldedEvaluations, FoldingCommitments, OuterClaims};
use crate::{BlindFoldProtocol, ProverError, WitnessCoordinate};

pub(crate) const OUTER_SUMCHECK_DEGREE: usize = 3;
pub(crate) const INNER_SUMCHECK_DEGREE: usize = 2;

#[derive(Clone, Copy, Debug)]
pub struct BlindFoldWitness<'a, F: JoltField> {
    pub rows: &'a [Vec<F>],
    pub blindings: &'a [F],
    pub eval_outputs: &'a [F],
    pub eval_blindings: &'a [F],
}

pub trait BlindFoldRowCommitter<F, VC>
where
    F: JoltField,
    VC: VectorCommitment<Field = F>,
{
    fn commit_rows(
        &mut self,
        setup: &VC::Setup,
        rows: &[Vec<F>],
        blindings: &[F],
        name: &'static str,
    ) -> Result<Vec<VC::Output>, ProverError<F>>;

    fn compute_error_rows(
        &mut self,
        r1cs: &ConstraintMatrices<F>,
        u: F,
        witness: &[F],
        row_count: usize,
        row_len: usize,
        name: &'static str,
    ) -> Result<Vec<Vec<F>>, ProverError<F>> {
        let _ = name;
        error_rows_for(r1cs, u, witness, row_count, row_len)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "cross-term error rows are defined by two relaxed witnesses"
    )]
    fn compute_cross_term_error_rows(
        &mut self,
        r1cs: &ConstraintMatrices<F>,
        real_u: F,
        real_witness: &[F],
        random_u: F,
        random_witness: &[F],
        row_count: usize,
        row_len: usize,
        name: &'static str,
    ) -> Result<Vec<Vec<F>>, ProverError<F>> {
        let _ = name;
        cross_term_error_rows_for(
            r1cs,
            real_u,
            real_witness,
            random_u,
            random_witness,
            row_count,
            row_len,
        )
    }

    fn fold_rows(
        &mut self,
        real: &[Vec<F>],
        random: &[Vec<F>],
        challenge: F,
        name: &'static str,
    ) -> Result<Vec<Vec<F>>, ProverError<F>> {
        let _ = name;
        fold_rows(real, random, challenge)
    }

    fn fold_scalars(
        &mut self,
        real: &[F],
        random: &[F],
        challenge: F,
        name: &'static str,
    ) -> Result<Vec<F>, ProverError<F>> {
        fold_scalars(name, real, random, challenge)
    }

    fn fold_error_rows(
        &mut self,
        real: &[Vec<F>],
        cross: &[Vec<F>],
        random: &[Vec<F>],
        challenge: F,
        name: &'static str,
    ) -> Result<Vec<Vec<F>>, ProverError<F>> {
        let _ = name;
        fold_error_rows(real, cross, random, challenge)
    }

    fn fold_error_scalars(
        &mut self,
        real: &[F],
        cross: &[F],
        random: &[F],
        challenge: F,
        name: &'static str,
    ) -> Result<Vec<F>, ProverError<F>> {
        fold_error_scalars(name, real, cross, random, challenge)
    }

    fn open_rows(
        &mut self,
        setup: &VC::Setup,
        rows: &[Vec<F>],
        blindings: &[F],
        row_point: &[F],
        entry_point: &[F],
        name: &'static str,
    ) -> Result<(VectorCommitmentOpening<F>, F), ProverError<F>> {
        open_committed_rows::<F, VC>(setup, rows, blindings, row_point, entry_point, name)
    }
}

#[derive(Debug, Default)]
pub struct DirectBlindFoldRowCommitter;

impl<F, VC> BlindFoldRowCommitter<F, VC> for DirectBlindFoldRowCommitter
where
    F: JoltField,
    VC: VectorCommitment<Field = F>,
{
    fn commit_rows(
        &mut self,
        setup: &VC::Setup,
        rows: &[Vec<F>],
        blindings: &[F],
        name: &'static str,
    ) -> Result<Vec<VC::Output>, ProverError<F>> {
        commit_rows::<F, VC>(setup, rows, blindings, name)
    }
}

/// Proves the BlindFold statement, writing the proof into `transcript`.
pub fn prove<F, VC, H, R>(
    setup: &VC::Setup,
    protocol: &BlindFoldProtocol<F, VC::Output>,
    transcript: &mut ProverTranscript<H>,
    witness: BlindFoldWitness<'_, F>,
    rng: &mut R,
) -> Result<(), ProverError<F>>
where
    F: JoltField,
    VC: VectorCommitment<Field = F>,
    VC::Output: HomomorphicCommitment<F>,
    H: Sponge,
    R: RngCore,
{
    let mut row_committer = DirectBlindFoldRowCommitter;
    prove_with_row_committer::<F, VC, H, R, DirectBlindFoldRowCommitter>(
        setup,
        protocol,
        transcript,
        witness,
        rng,
        &mut row_committer,
    )
}

pub fn prove_with_row_committer<F, VC, H, R, C>(
    setup: &VC::Setup,
    protocol: &BlindFoldProtocol<F, VC::Output>,
    transcript: &mut ProverTranscript<H>,
    witness: BlindFoldWitness<'_, F>,
    rng: &mut R,
    row_committer: &mut C,
) -> Result<(), ProverError<F>>
where
    F: JoltField,
    VC: VectorCommitment<Field = F>,
    VC::Output: HomomorphicCommitment<F>,
    H: Sponge,
    R: RngCore,
    C: BlindFoldRowCommitter<F, VC>,
{
    validate_witness::<F, VC>(setup, protocol, witness)?;

    let auxiliary_range = protocol.dimensions.witness_rows.auxiliary.clone();
    let auxiliary_row_commitments = row_committer.commit_rows(
        setup,
        range_slice(
            "auxiliary witness rows",
            witness.rows,
            auxiliary_range.clone(),
        )?,
        range_slice(
            "auxiliary witness blindings",
            witness.blindings,
            auxiliary_range,
        )?,
        "auxiliary witness rows",
    )?;
    for (index, ((commitment, &output), &blinding)) in protocol
        .eval_commitments
        .iter()
        .zip(witness.eval_outputs)
        .zip(witness.eval_blindings)
        .enumerate()
    {
        if !VC::verify(setup, commitment, &[output], &blinding) {
            return Err(ProverError::EvalCommitmentMismatch { index });
        }
    }

    let random_u = F::random(rng);
    let mut random_witness_rows = random_rows(
        protocol.dimensions.witness.row_count,
        protocol.dimensions.witness.row_len,
        rng,
    );
    let mut random_witness_blindings = (0..protocol.dimensions.witness.row_count)
        .map(|_| F::random(rng))
        .collect::<Vec<_>>();
    let padding_range = protocol.dimensions.witness_rows.padding.clone();
    for row in range_slice_mut(
        "padding witness rows",
        &mut random_witness_rows,
        padding_range.clone(),
    )? {
        row.fill(F::zero());
    }
    for blinding in range_slice_mut(
        "padding witness blindings",
        &mut random_witness_blindings,
        padding_range,
    )? {
        *blinding = F::zero();
    }

    let random_eval_outputs = (0..protocol.eval_commitments.len())
        .map(|_| F::random(rng))
        .collect::<Vec<_>>();
    let random_eval_blindings = (0..protocol.eval_commitments.len())
        .map(|_| F::random(rng))
        .collect::<Vec<_>>();
    let final_coordinates = protocol.final_opening_witness_coordinates()?;
    ensure_len(
        "final opening bindings",
        protocol.eval_commitments.len(),
        final_coordinates.len(),
    )?;
    let mut dedicated_rows = Vec::new();
    for coordinates in &final_coordinates {
        if let Some(coordinate) = coordinates.evaluation {
            dedicated_rows.push(coordinate.row);
        }
        if let Some(coordinate) = coordinates.blinding {
            dedicated_rows.push(coordinate.row);
        }
    }
    dedicated_rows.sort_unstable();
    dedicated_rows.dedup();
    #[expect(
        clippy::indexing_slicing,
        reason = "dedicated rows come from final-opening coordinates, range-checked against the witness grid in witness_coordinate"
    )]
    for row in dedicated_rows {
        // Values only: the folded row must stay dedicated for the verifier's
        // opening check, while the row blinding stays random so the published
        // row commitment hides the final evaluation and its Dory blinding.
        random_witness_rows[row].fill(F::zero());
    }
    #[expect(
        clippy::indexing_slicing,
        reason = "final-opening coordinates are range-checked against the witness grid in witness_coordinate"
    )]
    for (coordinates, (&random_output, &random_blinding)) in final_coordinates
        .iter()
        .zip(random_eval_outputs.iter().zip(&random_eval_blindings))
    {
        if let Some(coordinate) = coordinates.evaluation {
            random_witness_rows[coordinate.row][coordinate.column] = random_output;
        }
        if let Some(coordinate) = coordinates.blinding {
            random_witness_rows[coordinate.row][coordinate.column] = random_blinding;
        }
    }

    let random_error_rows = row_committer.compute_error_rows(
        &protocol.r1cs,
        random_u,
        &flatten(&random_witness_rows),
        protocol.dimensions.error.row_count,
        protocol.dimensions.error.row_len,
        "random error rows",
    )?;
    ensure_len(
        "random error rows",
        protocol.dimensions.error.row_count,
        random_error_rows.len(),
    )?;
    let random_error_blindings = (0..protocol.dimensions.error.row_count)
        .map(|_| F::random(rng))
        .collect::<Vec<_>>();
    let coefficient_range = protocol.dimensions.witness_rows.coefficients.clone();
    let output_claim_range = protocol.dimensions.witness_rows.output_claims.clone();
    let auxiliary_range = protocol.dimensions.witness_rows.auxiliary.clone();
    let random_round_commitments = row_committer.commit_rows(
        setup,
        range_slice(
            "random coefficient rows",
            &random_witness_rows,
            coefficient_range.clone(),
        )?,
        range_slice(
            "random coefficient blindings",
            &random_witness_blindings,
            coefficient_range,
        )?,
        "random coefficient rows",
    )?;
    let random_output_claim_row_commitments = row_committer.commit_rows(
        setup,
        range_slice(
            "random output-claim rows",
            &random_witness_rows,
            output_claim_range.clone(),
        )?,
        range_slice(
            "random output-claim blindings",
            &random_witness_blindings,
            output_claim_range,
        )?,
        "random output-claim rows",
    )?;
    let random_auxiliary_row_commitments = row_committer.commit_rows(
        setup,
        range_slice(
            "random auxiliary rows",
            &random_witness_rows,
            auxiliary_range.clone(),
        )?,
        range_slice(
            "random auxiliary blindings",
            &random_witness_blindings,
            auxiliary_range,
        )?,
        "random auxiliary rows",
    )?;
    let random_error_row_commitments = row_committer.commit_rows(
        setup,
        &random_error_rows,
        &random_error_blindings,
        "random error rows",
    )?;
    let random_eval_rows = random_eval_outputs
        .iter()
        .copied()
        .map(|output| vec![output])
        .collect::<Vec<_>>();
    let random_eval_commitments = row_committer.commit_rows(
        setup,
        &random_eval_rows,
        &random_eval_blindings,
        "random eval rows",
    )?;

    let cross_term_error_rows = row_committer.compute_cross_term_error_rows(
        &protocol.r1cs,
        F::one(),
        &flatten(witness.rows),
        random_u,
        &flatten(&random_witness_rows),
        protocol.dimensions.error.row_count,
        protocol.dimensions.error.row_len,
        "cross-term error rows",
    )?;
    ensure_len(
        "cross-term error rows",
        protocol.dimensions.error.row_count,
        cross_term_error_rows.len(),
    )?;
    let cross_term_error_blindings = (0..protocol.dimensions.error.row_count)
        .map(|_| F::random(rng))
        .collect::<Vec<_>>();
    let cross_term_error_row_commitments = row_committer.commit_rows(
        setup,
        &cross_term_error_rows,
        &cross_term_error_blindings,
        "cross-term error rows",
    )?;

    FoldingCommitments {
        auxiliary_rows: auxiliary_row_commitments,
        random_u,
        random_rounds: random_round_commitments,
        random_output_claim_rows: random_output_claim_row_commitments,
        random_auxiliary_rows: random_auxiliary_row_commitments,
        random_error_rows: random_error_row_commitments,
        random_evals: random_eval_commitments,
        cross_term_error_rows: cross_term_error_row_commitments,
    }
    .send(transcript);
    let folding_challenge: F = transcript.challenge_small();

    let folded_u = F::one() + folding_challenge * random_u;
    let folded_witness_rows = row_committer.fold_rows(
        witness.rows,
        &random_witness_rows,
        folding_challenge,
        "folded witness rows",
    )?;
    let folded_witness_blindings = row_committer.fold_scalars(
        witness.blindings,
        &random_witness_blindings,
        folding_challenge,
        "folded witness blindings",
    )?;
    let folded_error_rows = row_committer.fold_error_rows(
        &zero_rows(
            protocol.dimensions.error.row_count,
            protocol.dimensions.error.row_len,
        ),
        &cross_term_error_rows,
        &random_error_rows,
        folding_challenge,
        "folded error rows",
    )?;
    let zero_error_blindings = vec![F::zero(); protocol.dimensions.error.row_count];
    let folded_error_blindings = row_committer.fold_error_scalars(
        &zero_error_blindings,
        &cross_term_error_blindings,
        &random_error_blindings,
        folding_challenge,
        "folded error blindings",
    )?;
    let folded_eval_outputs = row_committer.fold_scalars(
        witness.eval_outputs,
        &random_eval_outputs,
        folding_challenge,
        "folded eval outputs",
    )?;
    let folded_eval_blindings = row_committer.fold_scalars(
        witness.eval_blindings,
        &random_eval_blindings,
        folding_challenge,
        "folded eval blindings",
    )?;

    let folded_evaluations = FoldedEvaluations {
        outputs: folded_eval_outputs,
        blindings: folded_eval_blindings,
    };
    folded_evaluations.send(transcript);
    for (index, (coordinates, (&folded_output, &folded_blinding))) in final_coordinates
        .iter()
        .zip(
            folded_evaluations
                .outputs
                .iter()
                .zip(&folded_evaluations.blindings),
        )
        .enumerate()
    {
        if let Some(coordinate) = coordinates.evaluation {
            let (opening, opened) = open_witness_coordinate::<F, VC, C>(
                setup,
                row_committer,
                &folded_witness_rows,
                &folded_witness_blindings,
                coordinate,
                "folded eval output opening",
            )?;
            if opened != folded_output {
                return Err(ProverError::EvalWitnessMismatch {
                    kind: "output",
                    index,
                    expected: folded_output,
                    actual: opened,
                });
            }
            send_opening(&opening, transcript);
        }
        if let Some(coordinate) = coordinates.blinding {
            let (opening, opened) = open_witness_coordinate::<F, VC, C>(
                setup,
                row_committer,
                &folded_witness_rows,
                &folded_witness_blindings,
                coordinate,
                "folded eval blinding opening",
            )?;
            if opened != folded_blinding {
                return Err(ProverError::EvalWitnessMismatch {
                    kind: "blinding",
                    index,
                    expected: folded_blinding,
                    actual: opened,
                });
            }
            send_opening(&opening, transcript);
        }
    }

    let outer_num_vars = log2_power_of_two("error row count", protocol.dimensions.error.row_count)?
        + log2_power_of_two("error row length", protocol.dimensions.error.row_len)?;
    if outer_num_vars == 0 {
        return Err(ProverError::DegenerateSumcheck {
            name: "outer folded R1CS sumcheck",
        });
    }
    let tau = transcript.challenges_small(outer_num_vars);
    let flattened_folded_witness = flatten(&folded_witness_rows);
    let flattened_folded_error = flatten(&folded_error_rows);
    let outer_trace = prove_outer_sumcheck(
        &protocol.r1cs,
        folded_u,
        &flattened_folded_witness,
        &flattened_folded_error,
        &tau,
        transcript,
    )?;

    let (az_rx, bz_rx, cz_rx) = abc_at_point(
        &protocol.r1cs,
        folded_u,
        &flattened_folded_witness,
        &outer_trace.point,
    );
    let error_row_vars = log2_power_of_two("error row count", protocol.dimensions.error.row_count)?;
    let (error_row_point, error_entry_point) = outer_trace.point.split_at(error_row_vars);
    let (error_opening, _) = row_committer.open_rows(
        setup,
        &folded_error_rows,
        &folded_error_blindings,
        error_row_point,
        error_entry_point,
        "folded error row opening",
    )?;

    OuterClaims {
        abc: [az_rx, bz_rx, cz_rx],
        error_opening,
    }
    .send(transcript);

    let ra: F = transcript.challenge_small();
    let rb: F = transcript.challenge_small();
    let rc: F = transcript.challenge_small();
    let inner_num_vars =
        log2_power_of_two("witness row count", protocol.dimensions.witness.row_count)?
            + log2_power_of_two("witness row length", protocol.dimensions.witness.row_len)?;
    if inner_num_vars == 0 {
        return Err(ProverError::DegenerateSumcheck {
            name: "inner folded R1CS sumcheck",
        });
    }
    let row_weights = EqPolynomial::<F>::evals(&outer_trace.point, None);
    let public = protocol
        .r1cs
        .public_column_contributions(&row_weights, 0, folded_u)?;
    let inner_claim = ra * (az_rx - public.a) + rb * (bz_rx - public.b) + rc * (cz_rx - public.c);
    let inner_trace = prove_inner_sumcheck(
        &protocol.r1cs,
        &outer_trace.point,
        &folded_witness_rows,
        ra,
        rb,
        rc,
        inner_claim,
        transcript,
    )?;
    let witness_row_vars =
        log2_power_of_two("witness row count", protocol.dimensions.witness.row_count)?;
    let (witness_row_point, witness_entry_point) = inner_trace.point.split_at(witness_row_vars);
    let (witness_opening, _) = row_committer.open_rows(
        setup,
        &folded_witness_rows,
        &folded_witness_blindings,
        witness_row_point,
        witness_entry_point,
        "folded witness row opening",
    )?;

    send_opening(&witness_opening, transcript);
    Ok(())
}

fn validate_witness<F, VC>(
    setup: &VC::Setup,
    protocol: &BlindFoldProtocol<F, VC::Output>,
    witness: BlindFoldWitness<'_, F>,
) -> Result<(), ProverError<F>>
where
    F: JoltField,
    VC: VectorCommitment<Field = F>,
{
    let _ = log2_power_of_two("witness row count", protocol.dimensions.witness.row_count)?;
    let _ = log2_power_of_two("witness row length", protocol.dimensions.witness.row_len)?;
    let _ = log2_power_of_two("error row count", protocol.dimensions.error.row_count)?;
    let _ = log2_power_of_two("error row length", protocol.dimensions.error.row_len)?;
    ensure_row_capacity::<F, VC>(setup, "witness rows", protocol.dimensions.witness.row_len)?;
    ensure_row_capacity::<F, VC>(setup, "error rows", protocol.dimensions.error.row_len)?;
    ensure_row_capacity::<F, VC>(setup, "evaluation rows", 1)?;
    ensure_len(
        "witness rows",
        protocol.dimensions.witness.row_count,
        witness.rows.len(),
    )?;
    ensure_len(
        "witness row blindings",
        protocol.dimensions.witness.row_count,
        witness.blindings.len(),
    )?;
    for (row, values) in witness.rows.iter().enumerate() {
        if values.len() != protocol.dimensions.witness.row_len {
            return Err(ProverError::WitnessRowLengthMismatch {
                row,
                expected: protocol.dimensions.witness.row_len,
                actual: values.len(),
            });
        }
    }
    ensure_len(
        "final opening evaluation values",
        protocol.eval_commitments.len(),
        witness.eval_outputs.len(),
    )?;
    ensure_len(
        "final opening blindings",
        protocol.eval_commitments.len(),
        witness.eval_blindings.len(),
    )?;
    Ok(())
}

fn ensure_row_capacity<F, VC>(
    setup: &VC::Setup,
    name: &'static str,
    row_len: usize,
) -> Result<(), ProverError<F>>
where
    F: JoltField,
    VC: VectorCommitment<Field = F>,
{
    let capacity = VC::capacity(setup);
    if row_len > capacity {
        return Err(ProverError::CommitmentCapacityExceeded {
            name,
            capacity,
            row_len,
        });
    }
    Ok(())
}

fn commit_rows<F, VC>(
    setup: &VC::Setup,
    rows: &[Vec<F>],
    blindings: &[F],
    name: &'static str,
) -> Result<Vec<VC::Output>, ProverError<F>>
where
    F: JoltField,
    VC: VectorCommitment<Field = F>,
{
    ensure_len(name, rows.len(), blindings.len())?;
    let capacity = VC::capacity(setup);
    for row in rows {
        if row.len() > capacity {
            return Err(ProverError::CommitmentCapacityExceeded {
                name,
                capacity,
                row_len: row.len(),
            });
        }
    }
    // At most one job per worker: ark-ec's msm_bigint_wnaf builds a 2-thread pool per
    // chunk (variable_base/mod.rs:853), so more outer jobs than workers oversubscribe.
    let rows_per_job = rows.len().div_ceil(rayon::current_num_threads()).max(1);
    Ok(rows
        .par_chunks(rows_per_job)
        .zip(blindings.par_chunks(rows_per_job))
        .flat_map_iter(|(rows, blindings)| {
            rows.iter()
                .zip(blindings)
                .map(|(row, blinding)| VC::commit(setup, row, blinding))
        })
        .collect())
}

fn open_committed_rows<F, VC>(
    setup: &VC::Setup,
    rows: &[Vec<F>],
    blindings: &[F],
    row_point: &[F],
    entry_point: &[F],
    name: &'static str,
) -> Result<(VectorCommitmentOpening<F>, F), ProverError<F>>
where
    F: JoltField,
    VC: VectorCommitment<Field = F>,
{
    let row_count = basis_len_from_point_len("row point", row_point.len())?;
    ensure_len(name, row_count, rows.len())?;
    ensure_len(name, row_count, blindings.len())?;
    let row_len = rows.first().map_or(0, Vec::len);
    let expected_row_len = basis_len_from_point_len("entry point", entry_point.len())?;
    ensure_len(name, expected_row_len, row_len)?;
    ensure_row_capacity::<F, VC>(setup, name, row_len)?;
    for (row_index, row) in rows.iter().enumerate() {
        if row.len() != row_len {
            return Err(ProverError::WitnessRowLengthMismatch {
                row: row_index,
                expected: row_len,
                actual: row.len(),
            });
        }
    }
    Ok(VC::open_committed_rows(
        &flatten(rows),
        blindings,
        row_len,
        row_point,
        entry_point,
    )?)
}

fn basis_len_from_point_len<F>(
    name: &'static str,
    point_len: usize,
) -> Result<usize, ProverError<F>>
where
    F: JoltField,
{
    if point_len >= usize::BITS as usize {
        return Err(ProverError::DimensionOverflow {
            name,
            value: point_len,
        });
    }
    Ok(1_usize << point_len)
}

#[derive(Clone, Debug)]
struct SumcheckTrace<F: JoltField> {
    point: Vec<F>,
}

fn prove_outer_sumcheck<F, H>(
    r1cs: &ConstraintMatrices<F>,
    u: F,
    witness: &[F],
    error_values: &[F],
    tau: &[F],
    transcript: &mut ProverTranscript<H>,
) -> Result<SumcheckTrace<F>, ProverError<F>>
where
    F: JoltField,
    H: Sponge,
{
    let num_vars = log2_power_of_two("outer folded R1CS sumcheck", error_values.len())?;
    ensure_len("outer challenge vector", num_vars, tau.len())?;

    let z = z_vector(u, witness);
    let mut az = matrix_vector_product(&r1cs.a, &z);
    let mut bz = matrix_vector_product(&r1cs.b, &z);
    let mut cz = matrix_vector_product(&r1cs.c, &z);
    let mut e = error_values.to_vec();
    let padded_len = error_values.len();
    pad_to_len("outer Az values", &mut az, padded_len)?;
    pad_to_len("outer Bz values", &mut bz, padded_len)?;
    pad_to_len("outer Cz values", &mut cz, padded_len)?;

    let mut az = Polynomial::new(az);
    let mut bz = Polynomial::new(bz);
    let mut cz = Polynomial::new(cz);
    let mut e = Polynomial::new(std::mem::take(&mut e));
    let mut eq_tau = Polynomial::new(EqPolynomial::<F>::evals(tau, None));

    let mut running_sum = F::zero();
    let mut point = Vec::with_capacity(num_vars);

    for _round in 0..num_vars {
        let half = az.len() / 2;
        let mut evals = [F::zero(); OUTER_SUMCHECK_DEGREE + 1];
        for i in 0..half {
            let (eq_lo, eq_hi) = eq_tau.sumcheck_eval_pair(i, BindingOrder::HighToLow);
            let (az_lo, az_hi) = az.sumcheck_eval_pair(i, BindingOrder::HighToLow);
            let (bz_lo, bz_hi) = bz.sumcheck_eval_pair(i, BindingOrder::HighToLow);
            let (cz_lo, cz_hi) = cz.sumcheck_eval_pair(i, BindingOrder::HighToLow);
            let (e_lo, e_hi) = e.sumcheck_eval_pair(i, BindingOrder::HighToLow);

            let eq_delta = eq_hi - eq_lo;
            let az_delta = az_hi - az_lo;
            let bz_delta = bz_hi - bz_lo;
            let cz_delta = cz_hi - cz_lo;
            let e_delta = e_hi - e_lo;

            evals[0] += eq_lo * (az_lo * bz_lo - u * cz_lo - e_lo);
            evals[1] += eq_hi * (az_hi * bz_hi - u * cz_hi - e_hi);

            let eq_2 = eq_lo + eq_delta + eq_delta;
            let az_2 = az_lo + az_delta + az_delta;
            let bz_2 = bz_lo + bz_delta + bz_delta;
            let cz_2 = cz_lo + cz_delta + cz_delta;
            let e_2 = e_lo + e_delta + e_delta;
            evals[2] += eq_2 * (az_2 * bz_2 - u * cz_2 - e_2);

            let eq_3 = eq_2 + eq_delta;
            let az_3 = az_2 + az_delta;
            let bz_3 = bz_2 + bz_delta;
            let cz_3 = cz_2 + cz_delta;
            let e_3 = e_2 + e_delta;
            evals[3] += eq_3 * (az_3 * bz_3 - u * cz_3 - e_3);
        }

        let round_poly = UnivariatePoly::from_evals(&evals);
        let coefficients = round_poly.coefficients();
        let round_sum = coefficients.first().copied().unwrap_or_else(F::zero)
            + coefficients.iter().copied().sum::<F>();
        if round_sum != running_sum {
            return Err(ProverError::SumcheckRoundClaimMismatch {
                expected: running_sum,
                actual: round_sum,
            });
        }
        send_compressed_round(&round_poly, OUTER_SUMCHECK_DEGREE, transcript)?;
        let challenge: F = transcript.challenge_small();
        running_sum = round_poly.evaluate(challenge);
        az.bind_with_order(challenge, BindingOrder::HighToLow);
        bz.bind_with_order(challenge, BindingOrder::HighToLow);
        cz.bind_with_order(challenge, BindingOrder::HighToLow);
        e.bind_with_order(challenge, BindingOrder::HighToLow);
        eq_tau.bind_with_order(challenge, BindingOrder::HighToLow);
        point.push(challenge);
    }

    Ok(SumcheckTrace { point })
}

#[expect(
    clippy::too_many_arguments,
    reason = "inner folded R1CS sumcheck is parameterized by three random matrix weights"
)]
fn prove_inner_sumcheck<F, H>(
    r1cs: &ConstraintMatrices<F>,
    outer_point: &[F],
    witness_rows: &[Vec<F>],
    ra: F,
    rb: F,
    rc: F,
    claim: F,
    transcript: &mut ProverTranscript<H>,
) -> Result<SumcheckTrace<F>, ProverError<F>>
where
    F: JoltField,
    H: Sponge,
{
    let witness_values = flatten(witness_rows);
    let num_vars = log2_power_of_two("inner folded R1CS sumcheck", witness_values.len())?;
    let row_weights = EqPolynomial::<F>::evals(outer_point, None);
    let l_w =
        linear_form_project_columns(r1cs, &row_weights, 1, witness_values.len(), [ra, rb, rc])?;

    let mut l_w = Polynomial::new(l_w);
    let mut witness = Polynomial::new(witness_values);
    let mut running_sum = claim;
    let mut point = Vec::with_capacity(num_vars);

    for _round in 0..num_vars {
        let half = l_w.len() / 2;
        let mut evals = [F::zero(); INNER_SUMCHECK_DEGREE + 1];
        for i in 0..half {
            let (lw_lo, lw_hi) = l_w.sumcheck_eval_pair(i, BindingOrder::HighToLow);
            let (w_lo, w_hi) = witness.sumcheck_eval_pair(i, BindingOrder::HighToLow);
            let lw_delta = lw_hi - lw_lo;
            let w_delta = w_hi - w_lo;

            evals[0] += lw_lo * w_lo;
            evals[1] += lw_hi * w_hi;

            let lw_2 = lw_lo + lw_delta + lw_delta;
            let w_2 = w_lo + w_delta + w_delta;
            evals[2] += lw_2 * w_2;
        }

        let round_poly = UnivariatePoly::from_evals(&evals);
        let coefficients = round_poly.coefficients();
        let round_sum = coefficients.first().copied().unwrap_or_else(F::zero)
            + coefficients.iter().copied().sum::<F>();
        if round_sum != running_sum {
            return Err(ProverError::SumcheckRoundClaimMismatch {
                expected: running_sum,
                actual: round_sum,
            });
        }
        send_compressed_round(&round_poly, INNER_SUMCHECK_DEGREE, transcript)?;
        let challenge: F = transcript.challenge_small();
        running_sum = round_poly.evaluate(challenge);
        l_w.bind_with_order(challenge, BindingOrder::HighToLow);
        witness.bind_with_order(challenge, BindingOrder::HighToLow);
        point.push(challenge);
    }

    Ok(SumcheckTrace { point })
}

fn matrix_vector_product<F>(rows: &[SparseRow<F>], vector: &[F]) -> Vec<F>
where
    F: JoltField,
{
    rows.par_iter().map(|row| dot(row, vector)).collect()
}

fn linear_form_project_columns<F>(
    r1cs: &ConstraintMatrices<F>,
    row_weights: &[F],
    start_col: usize,
    col_count: usize,
    weights: [F; 3],
) -> Result<Vec<F>, ProverError<F>>
where
    F: JoltField,
{
    if row_weights.len() < r1cs.num_constraints {
        return Err(ConstraintMatrixEvalError::RowWeightsLengthMismatch {
            expected: r1cs.num_constraints,
            actual: row_weights.len(),
        }
        .into());
    }
    if start_col.checked_add(col_count).is_none() {
        return Err(ConstraintMatrixEvalError::ColumnRangeOverflow {
            start: start_col,
            count: col_count,
        }
        .into());
    }

    let mut projected = vec![F::zero(); col_count];
    project_matrix_columns(&mut projected, &r1cs.a, row_weights, start_col, weights[0]);
    project_matrix_columns(&mut projected, &r1cs.b, row_weights, start_col, weights[1]);
    project_matrix_columns(&mut projected, &r1cs.c, row_weights, start_col, weights[2]);
    Ok(projected)
}

fn project_matrix_columns<F>(
    projected: &mut [F],
    rows: &[SparseRow<F>],
    row_weights: &[F],
    start_col: usize,
    weight: F,
) where
    F: JoltField,
{
    if weight.is_zero() {
        return;
    }
    for (row, &row_weight) in rows.iter().zip(row_weights) {
        let scaled_weight = weight * row_weight;
        for &(column, coefficient) in row {
            if let Some(entry) = column
                .checked_sub(start_col)
                .and_then(|offset| projected.get_mut(offset))
            {
                *entry += scaled_weight * coefficient;
            }
        }
    }
}

fn abc_at_point<F>(r1cs: &ConstraintMatrices<F>, u: F, witness: &[F], point: &[F]) -> (F, F, F)
where
    F: JoltField,
{
    let row_weights = EqPolynomial::<F>::evals(point, None);
    let z = z_vector(u, witness);
    let mut az = F::zero();
    let mut bz = F::zero();
    let mut cz = F::zero();
    for (((&row_weight, a_row), b_row), c_row) in
        row_weights.iter().zip(&r1cs.a).zip(&r1cs.b).zip(&r1cs.c)
    {
        az += row_weight * dot(a_row, &z);
        bz += row_weight * dot(b_row, &z);
        cz += row_weight * dot(c_row, &z);
    }
    (az, bz, cz)
}

fn open_witness_coordinate<F, VC, C>(
    setup: &VC::Setup,
    row_committer: &mut C,
    witness_rows: &[Vec<F>],
    witness_blindings: &[F],
    coordinate: WitnessCoordinate,
    name: &'static str,
) -> Result<(VectorCommitmentOpening<F>, F), ProverError<F>>
where
    F: JoltField,
    VC: VectorCommitment<Field = F>,
    C: BlindFoldRowCommitter<F, VC>,
{
    let row_vars = log2_power_of_two("witness row count", witness_rows.len())?;
    let entry_vars = log2_power_of_two(
        "witness row length",
        witness_rows.first().map_or(0, Vec::len),
    )?;
    row_committer.open_rows(
        setup,
        witness_rows,
        witness_blindings,
        &boolean_point(coordinate.row, row_vars),
        &boolean_point(coordinate.column, entry_vars),
        name,
    )
}

fn random_rows<F, R>(row_count: usize, row_len: usize, rng: &mut R) -> Vec<Vec<F>>
where
    F: JoltField,
    R: RngCore,
{
    (0..row_count)
        .map(|_| (0..row_len).map(|_| F::random(rng)).collect())
        .collect()
}

fn zero_rows<F: JoltField>(row_count: usize, row_len: usize) -> Vec<Vec<F>> {
    vec![vec![F::zero(); row_len]; row_count]
}

fn fold_rows<F>(
    real: &[Vec<F>],
    random: &[Vec<F>],
    challenge: F,
) -> Result<Vec<Vec<F>>, ProverError<F>>
where
    F: JoltField,
{
    ensure_len("random witness rows", real.len(), random.len())?;
    let mut folded = Vec::with_capacity(real.len());
    for (row_index, (real_row, random_row)) in real.iter().zip(random).enumerate() {
        if real_row.len() != random_row.len() {
            return Err(ProverError::WitnessRowLengthMismatch {
                row: row_index,
                expected: real_row.len(),
                actual: random_row.len(),
            });
        }
        folded.push(
            real_row
                .iter()
                .zip(random_row)
                .map(|(&real, &random)| real + challenge * random)
                .collect(),
        );
    }
    Ok(folded)
}

fn fold_scalars<F>(
    name: &'static str,
    real: &[F],
    random: &[F],
    challenge: F,
) -> Result<Vec<F>, ProverError<F>>
where
    F: JoltField,
{
    ensure_len(name, real.len(), random.len())?;
    Ok(real
        .iter()
        .zip(random)
        .map(|(&real, &random)| real + challenge * random)
        .collect())
}

fn fold_error_rows<F>(
    real: &[Vec<F>],
    cross: &[Vec<F>],
    random: &[Vec<F>],
    challenge: F,
) -> Result<Vec<Vec<F>>, ProverError<F>>
where
    F: JoltField,
{
    ensure_len("cross-term error rows", real.len(), cross.len())?;
    ensure_len("random error rows", real.len(), random.len())?;
    let challenge_squared = challenge * challenge;
    let mut folded = Vec::with_capacity(real.len());
    for (row_index, ((real_row, cross_row), random_row)) in
        real.iter().zip(cross).zip(random).enumerate()
    {
        if real_row.len() != cross_row.len() {
            return Err(ProverError::WitnessRowLengthMismatch {
                row: row_index,
                expected: real_row.len(),
                actual: cross_row.len(),
            });
        }
        if real_row.len() != random_row.len() {
            return Err(ProverError::WitnessRowLengthMismatch {
                row: row_index,
                expected: real_row.len(),
                actual: random_row.len(),
            });
        }
        folded.push(
            real_row
                .iter()
                .zip(cross_row)
                .zip(random_row)
                .map(|((&real, &cross), &random)| {
                    real + challenge * cross + challenge_squared * random
                })
                .collect(),
        );
    }
    Ok(folded)
}

fn fold_error_scalars<F>(
    name: &'static str,
    real: &[F],
    cross: &[F],
    random: &[F],
    challenge: F,
) -> Result<Vec<F>, ProverError<F>>
where
    F: JoltField,
{
    ensure_len(name, real.len(), cross.len())?;
    ensure_len(name, real.len(), random.len())?;
    let challenge_squared = challenge * challenge;
    Ok(real
        .iter()
        .zip(cross)
        .zip(random)
        .map(|((&real, &cross), &random)| real + challenge * cross + challenge_squared * random)
        .collect())
}

fn error_rows_for<F>(
    r1cs: &ConstraintMatrices<F>,
    u: F,
    witness: &[F],
    row_count: usize,
    row_len: usize,
) -> Result<Vec<Vec<F>>, ProverError<F>>
where
    F: JoltField,
{
    let _ = log2_power_of_two("error row length", row_len)?;
    let target_len = row_count
        .checked_mul(row_len)
        .ok_or(ProverError::DimensionOverflow {
            name: "error values",
            value: row_count,
        })?;
    let z = z_vector(u, witness);
    let mut errors = r1cs
        .a
        .iter()
        .zip(&r1cs.b)
        .zip(&r1cs.c)
        .map(|((a_row, b_row), c_row)| dot(a_row, &z) * dot(b_row, &z) - u * dot(c_row, &z))
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
) -> Result<Vec<Vec<F>>, ProverError<F>>
where
    F: JoltField,
{
    let _ = log2_power_of_two("error row length", row_len)?;
    let target_len = row_count
        .checked_mul(row_len)
        .ok_or(ProverError::DimensionOverflow {
            name: "cross-term error values",
            value: row_count,
        })?;
    let real_z = z_vector(real_u, real_witness);
    let random_z = z_vector(random_u, random_witness);
    let mut errors = r1cs
        .a
        .iter()
        .zip(&r1cs.b)
        .zip(&r1cs.c)
        .map(|((a_row, b_row), c_row)| {
            dot(a_row, &real_z) * dot(b_row, &random_z)
                + dot(a_row, &random_z) * dot(b_row, &real_z)
                - real_u * dot(c_row, &random_z)
                - random_u * dot(c_row, &real_z)
        })
        .collect::<Vec<_>>();
    pad_to_len("cross-term error values", &mut errors, target_len)?;
    Ok(errors.chunks(row_len).map(<[F]>::to_vec).collect())
}

fn boolean_point<F>(index: usize, num_vars: usize) -> Vec<F>
where
    F: JoltField,
{
    (0..num_vars)
        .map(|bit| {
            let shift = num_vars - bit - 1;
            F::from_u64(((index >> shift) & 1) as u64)
        })
        .collect()
}

fn pad_to_len<F>(
    name: &'static str,
    values: &mut Vec<F>,
    target_len: usize,
) -> Result<(), ProverError<F>>
where
    F: JoltField,
{
    if values.len() > target_len {
        return Err(ProverError::LengthMismatch {
            name,
            expected: target_len,
            actual: values.len(),
        });
    }
    values.resize(target_len, F::zero());
    Ok(())
}

fn z_vector<F>(u: F, witness: &[F]) -> Vec<F>
where
    F: JoltField,
{
    let mut z = Vec::with_capacity(witness.len() + 1);
    z.push(u);
    z.extend_from_slice(witness);
    z
}

#[expect(
    clippy::indexing_slicing,
    reason = "sparse row columns are checked against num_vars at ConstraintMatrices construction and every caller builds z vectors covering at least num_vars entries"
)]
fn dot<F>(row: &[(usize, F)], witness: &[F]) -> F
where
    F: JoltField,
{
    row.iter()
        .map(|&(column, coefficient)| coefficient * witness[column])
        .sum()
}

fn flatten<F>(rows: &[Vec<F>]) -> Vec<F>
where
    F: JoltField,
{
    rows.iter().flat_map(|row| row.iter().copied()).collect()
}

fn range_slice<'a, T, F>(
    name: &'static str,
    values: &'a [T],
    range: std::ops::Range<usize>,
) -> Result<&'a [T], ProverError<F>>
where
    F: JoltField,
{
    let actual = values.len();
    values
        .get(range.clone())
        .ok_or(ProverError::LengthMismatch {
            name,
            expected: range.end,
            actual,
        })
}

fn range_slice_mut<'a, T, F>(
    name: &'static str,
    values: &'a mut [T],
    range: std::ops::Range<usize>,
) -> Result<&'a mut [T], ProverError<F>>
where
    F: JoltField,
{
    let actual = values.len();
    values
        .get_mut(range.clone())
        .ok_or(ProverError::LengthMismatch {
            name,
            expected: range.end,
            actual,
        })
}

fn ensure_len<F>(name: &'static str, expected: usize, actual: usize) -> Result<(), ProverError<F>>
where
    F: JoltField,
{
    if expected != actual {
        return Err(ProverError::LengthMismatch {
            name,
            expected,
            actual,
        });
    }
    Ok(())
}

fn log2_power_of_two<F>(name: &'static str, value: usize) -> Result<usize, ProverError<F>>
where
    F: JoltField,
{
    jolt_utils::checked_log2_power_of_two(value)
        .ok_or(ProverError::InvalidPowerOfTwo { name, value })
}
