use ark_serialize::CanonicalSerialize;
use jolt_crypto::{HomomorphicCommitment, VectorCommitment, VectorCommitmentOpening};
use jolt_field::{Field, RingAccumulator, WithAccumulator};
use jolt_poly::{BindingOrder, EqPolynomial, Polynomial, UnivariatePoly};
use jolt_r1cs::{ConstraintMatrices, ConstraintMatrixEvalError, SparseRow};
use jolt_sumcheck::CompressedSumcheckProof;
use jolt_transcript::{FsAbsorb, FsTranscript};
use rand_core::RngCore;
use rayon::prelude::*;

use crate::{
    transcript_codec::absorb_legacy_field_vec, BlindFoldProof, BlindFoldProtocol, ProverError,
    WitnessCoordinate,
};

const OUTER_SUMCHECK_DEGREE: usize = 3;
const INNER_SUMCHECK_DEGREE: usize = 2;

#[derive(Clone, Copy, Debug)]
pub struct BlindFoldWitness<'a, F: Field> {
    pub rows: &'a [Vec<F>],
    pub blindings: &'a [F],
    pub eval_outputs: &'a [F],
    pub eval_blindings: &'a [F],
}

pub trait BlindFoldRowCommitter<F, VC>
where
    F: Field,
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
    ) -> Result<(VectorCommitmentOpening<F>, F), ProverError<F>>
    where
        <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
    {
        open_committed_rows::<F, VC>(setup, rows, blindings, row_point, entry_point, name)
    }
}

#[derive(Debug, Default)]
pub struct DirectBlindFoldRowCommitter;

impl<F, VC> BlindFoldRowCommitter<F, VC> for DirectBlindFoldRowCommitter
where
    F: Field,
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

pub fn prove<F, VC, T, R>(
    setup: &VC::Setup,
    protocol: &BlindFoldProtocol<F, VC::Output>,
    transcript: &mut T,
    witness: BlindFoldWitness<'_, F>,
    rng: &mut R,
) -> Result<BlindFoldProof<F, VC::Output>, ProverError<F>>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
    VC::Output: HomomorphicCommitment<F>,
    T: FsTranscript<F>,
    R: RngCore,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    let mut row_committer = DirectBlindFoldRowCommitter;
    prove_with_row_committer::<F, VC, T, R, DirectBlindFoldRowCommitter>(
        setup,
        protocol,
        transcript,
        witness,
        rng,
        &mut row_committer,
    )
}

pub fn prove_with_row_committer<F, VC, T, R, C>(
    setup: &VC::Setup,
    protocol: &BlindFoldProtocol<F, VC::Output>,
    transcript: &mut T,
    witness: BlindFoldWitness<'_, F>,
    rng: &mut R,
    row_committer: &mut C,
) -> Result<BlindFoldProof<F, VC::Output>, ProverError<F>>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
    VC::Output: HomomorphicCommitment<F>,
    T: FsTranscript<F>,
    R: RngCore,
    C: BlindFoldRowCommitter<F, VC>,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    validate_witness::<F, VC>(setup, protocol, witness)?;

    let auxiliary_range = protocol.dimensions.witness_rows.auxiliary.clone();
    let auxiliary_row_commitments = row_committer.commit_rows(
        setup,
        &witness.rows[auxiliary_range.clone()],
        &witness.blindings[auxiliary_range],
        "auxiliary witness rows",
    )?;
    let committed = protocol.committed_relaxed_instance(&auxiliary_row_commitments)?;
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
    for row in protocol.dimensions.witness_rows.padding.clone() {
        random_witness_rows[row].fill(F::zero());
        random_witness_blindings[row] = F::zero();
    }

    let random_eval_outputs = (0..protocol.eval_commitments.len())
        .map(|_| F::random(rng))
        .collect::<Vec<_>>();
    let random_eval_blindings = (0..protocol.eval_commitments.len())
        .map(|_| F::random(rng))
        .collect::<Vec<_>>();
    let final_coordinates = protocol.final_opening_witness_coordinates()?;
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
    for row in dedicated_rows {
        random_witness_rows[row].fill(F::zero());
        random_witness_blindings[row] = F::zero();
    }
    for (index, coordinates) in final_coordinates.iter().enumerate() {
        if let Some(coordinate) = coordinates.evaluation {
            random_witness_rows[coordinate.row][coordinate.column] = random_eval_outputs[index];
        }
        if let Some(coordinate) = coordinates.blinding {
            random_witness_rows[coordinate.row][coordinate.column] = random_eval_blindings[index];
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
        &random_witness_rows[coefficient_range.clone()],
        &random_witness_blindings[coefficient_range],
        "random coefficient rows",
    )?;
    let random_output_claim_row_commitments = row_committer.commit_rows(
        setup,
        &random_witness_rows[output_claim_range.clone()],
        &random_witness_blindings[output_claim_range],
        "random output-claim rows",
    )?;
    let random_auxiliary_row_commitments = row_committer.commit_rows(
        setup,
        &random_witness_rows[auxiliary_range.clone()],
        &random_witness_blindings[auxiliary_range],
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
    let random_instance = protocol.random_relaxed_instance(
        &random_round_commitments,
        &random_output_claim_row_commitments,
        &random_auxiliary_row_commitments,
        &random_error_row_commitments,
        &random_eval_commitments,
        random_u,
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

    append_relaxed_instance(
        transcript,
        committed.u,
        &committed.witness_row_commitments,
        &committed.error_row_commitments,
        &committed.eval_commitments,
    );
    append_relaxed_instance(
        transcript,
        random_u,
        &random_instance.witness_row_commitments,
        &random_instance.error_row_commitments,
        &random_instance.eval_commitments,
    );
    transcript.absorb(&cross_term_error_row_commitments);
    let folding_challenge = transcript.challenge();

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

    let mut folded_eval_output_openings = Vec::new();
    let mut folded_eval_blinding_openings = Vec::new();
    for (index, coordinates) in final_coordinates.iter().enumerate() {
        if let Some(coordinate) = coordinates.evaluation {
            let (opening, opened) = open_witness_coordinate::<F, VC, C>(
                setup,
                row_committer,
                &folded_witness_rows,
                &folded_witness_blindings,
                coordinate,
                "folded eval output opening",
            )?;
            if opened != folded_eval_outputs[index] {
                return Err(ProverError::EvalWitnessMismatch {
                    kind: "output",
                    index,
                    expected: folded_eval_outputs[index],
                    actual: opened,
                });
            }
            folded_eval_output_openings.push(opening);
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
            if opened != folded_eval_blindings[index] {
                return Err(ProverError::EvalWitnessMismatch {
                    kind: "blinding",
                    index,
                    expected: folded_eval_blindings[index],
                    actual: opened,
                });
            }
            folded_eval_blinding_openings.push(opening);
        }
    }
    for opening in &folded_eval_output_openings {
        append_vector_opening(transcript, opening);
    }
    for opening in &folded_eval_blinding_openings {
        append_vector_opening(transcript, opening);
    }

    let outer_num_vars = log2_power_of_two("error row count", protocol.dimensions.error.row_count)?
        + log2_power_of_two("error row length", protocol.dimensions.error.row_len)?;
    if outer_num_vars == 0 {
        return Err(ProverError::DegenerateSumcheck {
            name: "outer folded R1CS sumcheck",
        });
    }
    let tau = transcript.challenge_vector(outer_num_vars);
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

    transcript.absorb_field_slice(&[az_rx, bz_rx, cz_rx]);
    append_vector_opening(transcript, &error_opening);

    let ra = transcript.challenge();
    let rb = transcript.challenge();
    let rc = transcript.challenge();
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

    Ok(BlindFoldProof {
        auxiliary_row_commitments,
        random_round_commitments,
        random_output_claim_row_commitments,
        random_auxiliary_row_commitments,
        random_error_row_commitments,
        random_eval_commitments,
        random_u,
        cross_term_error_row_commitments,
        outer_sumcheck: outer_trace.proof,
        az_rx,
        bz_rx,
        cz_rx,
        inner_sumcheck: inner_trace.proof,
        witness_opening,
        error_opening,
        folded_eval_outputs,
        folded_eval_blindings,
        folded_eval_output_openings,
        folded_eval_blinding_openings,
    })
}

fn validate_witness<F, VC>(
    setup: &VC::Setup,
    protocol: &BlindFoldProtocol<F, VC::Output>,
    witness: BlindFoldWitness<'_, F>,
) -> Result<(), ProverError<F>>
where
    F: Field,
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
    F: Field,
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
    F: Field,
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
    Ok(rows
        .par_iter()
        .zip(blindings.par_iter())
        .map(|(row, blinding)| VC::commit(setup, row, blinding))
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
    F: Field,
    VC: VectorCommitment<Field = F>,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
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
    F: Field,
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
struct SumcheckTrace<F: Field> {
    proof: CompressedSumcheckProof<F>,
    point: Vec<F>,
}

fn prove_outer_sumcheck<F, T>(
    r1cs: &ConstraintMatrices<F>,
    u: F,
    witness: &[F],
    error_values: &[F],
    tau: &[F],
    transcript: &mut T,
) -> Result<SumcheckTrace<F>, ProverError<F>>
where
    F: Field,
    T: FsTranscript<F>,
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
    let mut rounds = Vec::with_capacity(num_vars);
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
        let round_sum =
            round_poly.coefficients()[0] + round_poly.coefficients().iter().copied().sum::<F>();
        if round_sum != running_sum {
            return Err(ProverError::SumcheckRoundClaimMismatch {
                expected: running_sum,
                actual: round_sum,
            });
        }
        let compressed = round_poly.compress();
        absorb_legacy_field_vec(transcript, compressed.coeffs_except_linear_term());
        let challenge = transcript.challenge();
        running_sum = round_poly.evaluate(challenge);
        az.bind_with_order(challenge, BindingOrder::HighToLow);
        bz.bind_with_order(challenge, BindingOrder::HighToLow);
        cz.bind_with_order(challenge, BindingOrder::HighToLow);
        e.bind_with_order(challenge, BindingOrder::HighToLow);
        eq_tau.bind_with_order(challenge, BindingOrder::HighToLow);
        point.push(challenge);
        rounds.push(compressed);
    }

    Ok(SumcheckTrace {
        proof: CompressedSumcheckProof {
            round_polynomials: rounds,
        },
        point,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "inner folded R1CS sumcheck is parameterized by three random matrix weights"
)]
fn prove_inner_sumcheck<F, T>(
    r1cs: &ConstraintMatrices<F>,
    outer_point: &[F],
    witness_rows: &[Vec<F>],
    ra: F,
    rb: F,
    rc: F,
    claim: F,
    transcript: &mut T,
) -> Result<SumcheckTrace<F>, ProverError<F>>
where
    F: Field,
    T: FsTranscript<F>,
{
    let witness_values = flatten(witness_rows);
    let num_vars = log2_power_of_two("inner folded R1CS sumcheck", witness_values.len())?;
    let row_weights = EqPolynomial::<F>::evals(outer_point, None);
    let l_w =
        linear_form_project_columns(r1cs, &row_weights, 1, witness_values.len(), [ra, rb, rc])?;

    let mut l_w = Polynomial::new(l_w);
    let mut witness = Polynomial::new(witness_values);
    let mut running_sum = claim;
    let mut rounds = Vec::with_capacity(num_vars);
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
        let round_sum =
            round_poly.coefficients()[0] + round_poly.coefficients().iter().copied().sum::<F>();
        if round_sum != running_sum {
            return Err(ProverError::SumcheckRoundClaimMismatch {
                expected: running_sum,
                actual: round_sum,
            });
        }
        let compressed = round_poly.compress();
        absorb_legacy_field_vec(transcript, compressed.coeffs_except_linear_term());
        let challenge = transcript.challenge();
        running_sum = round_poly.evaluate(challenge);
        l_w.bind_with_order(challenge, BindingOrder::HighToLow);
        witness.bind_with_order(challenge, BindingOrder::HighToLow);
        point.push(challenge);
        rounds.push(compressed);
    }

    Ok(SumcheckTrace {
        proof: CompressedSumcheckProof {
            round_polynomials: rounds,
        },
        point,
    })
}

fn matrix_vector_product<F>(rows: &[SparseRow<F>], vector: &[F]) -> Vec<F>
where
    F: Field,
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
    F: Field,
{
    if row_weights.len() < r1cs.num_constraints {
        return Err(ConstraintMatrixEvalError::RowWeightsLengthMismatch {
            expected: r1cs.num_constraints,
            actual: row_weights.len(),
        }
        .into());
    }
    let end_col =
        start_col
            .checked_add(col_count)
            .ok_or(ConstraintMatrixEvalError::ColumnRangeOverflow {
                start: start_col,
                count: col_count,
            })?;

    let mut projected = vec![F::zero(); col_count];
    project_matrix_columns(
        &mut projected,
        &r1cs.a,
        row_weights,
        start_col,
        end_col,
        weights[0],
    );
    project_matrix_columns(
        &mut projected,
        &r1cs.b,
        row_weights,
        start_col,
        end_col,
        weights[1],
    );
    project_matrix_columns(
        &mut projected,
        &r1cs.c,
        row_weights,
        start_col,
        end_col,
        weights[2],
    );
    Ok(projected)
}

fn project_matrix_columns<F>(
    projected: &mut [F],
    rows: &[SparseRow<F>],
    row_weights: &[F],
    start_col: usize,
    end_col: usize,
    weight: F,
) where
    F: Field,
{
    if weight.is_zero() {
        return;
    }
    for (row, &row_weight) in rows.iter().zip(row_weights) {
        let scaled_weight = weight * row_weight;
        for &(column, coefficient) in row {
            if (start_col..end_col).contains(&column) {
                projected[column - start_col] += scaled_weight * coefficient;
            }
        }
    }
}

fn abc_at_point<F>(r1cs: &ConstraintMatrices<F>, u: F, witness: &[F], point: &[F]) -> (F, F, F)
where
    F: Field,
{
    let row_weights = EqPolynomial::<F>::evals(point, None);
    let z = z_vector(u, witness);
    let mut az = F::zero();
    let mut bz = F::zero();
    let mut cz = F::zero();
    for (row_index, &row_weight) in row_weights.iter().enumerate().take(r1cs.num_constraints) {
        az += row_weight * dot(&r1cs.a[row_index], &z);
        bz += row_weight * dot(&r1cs.b[row_index], &z);
        cz += row_weight * dot(&r1cs.c[row_index], &z);
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
    F: Field,
    VC: VectorCommitment<Field = F>,
    C: BlindFoldRowCommitter<F, VC>,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    let row_vars = log2_power_of_two("witness row count", witness_rows.len())?;
    let entry_vars = log2_power_of_two("witness row length", witness_rows[0].len())?;
    row_committer.open_rows(
        setup,
        witness_rows,
        witness_blindings,
        &boolean_point(coordinate.row, row_vars),
        &boolean_point(coordinate.column, entry_vars),
        name,
    )
}

fn append_relaxed_instance<F, C, T>(
    transcript: &mut T,
    u: F,
    witness_commitments: &Vec<C>,
    error_commitments: &Vec<C>,
    eval_commitments: &Vec<C>,
) where
    F: Field,
    C: CanonicalSerialize,
    T: FsAbsorb,
{
    transcript.absorb_field(&u);
    transcript.absorb(witness_commitments);
    transcript.absorb(error_commitments);
    transcript.absorb(eval_commitments);
}

fn append_vector_opening<F, T>(transcript: &mut T, opening: &VectorCommitmentOpening<F>)
where
    F: Field,
    T: FsAbsorb,
{
    absorb_legacy_field_vec(transcript, &opening.combined_vector);
    transcript.absorb_field(&opening.combined_blinding);
}

fn random_rows<F, R>(row_count: usize, row_len: usize, rng: &mut R) -> Vec<Vec<F>>
where
    F: Field,
    R: RngCore,
{
    (0..row_count)
        .map(|_| (0..row_len).map(|_| F::random(rng)).collect())
        .collect()
}

fn zero_rows<F: Field>(row_count: usize, row_len: usize) -> Vec<Vec<F>> {
    vec![vec![F::zero(); row_len]; row_count]
}

fn fold_rows<F>(
    real: &[Vec<F>],
    random: &[Vec<F>],
    challenge: F,
) -> Result<Vec<Vec<F>>, ProverError<F>>
where
    F: Field,
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
    F: Field,
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
    F: Field,
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
    F: Field,
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
    F: Field,
{
    let _ = log2_power_of_two("error row length", row_len)?;
    let target_len = row_count
        .checked_mul(row_len)
        .ok_or(ProverError::DimensionOverflow {
            name: "error values",
            value: row_count,
        })?;
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
) -> Result<Vec<Vec<F>>, ProverError<F>>
where
    F: Field,
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

fn boolean_point<F>(index: usize, num_vars: usize) -> Vec<F>
where
    F: Field,
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
    F: Field,
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

fn ensure_len<F>(name: &'static str, expected: usize, actual: usize) -> Result<(), ProverError<F>>
where
    F: Field,
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
    F: Field,
{
    if value == 0 || !value.is_power_of_two() {
        return Err(ProverError::InvalidPowerOfTwo { name, value });
    }
    Ok(value.trailing_zeros() as usize)
}
