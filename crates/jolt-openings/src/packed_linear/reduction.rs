use jolt_field::Field;
use jolt_transcript::{AppendToTranscript, LabelWithCount, Transcript};

use crate::{BatchOpeningResult, BatchOpeningStatement, OpeningsError, PhysicalView};

use super::encoding::{decode_round, encode_round, field_bytes, field_from_bytes};
use super::selector::{
    logical_coefficients, native_opening_point, packed_selector_eval, packed_selector_evals,
    reduced_claim, validate_term,
};
use super::transcript::{append_round, bind_packed_statement};
use super::types::{
    PackedLinearLayout, PackedLinearProverReduction, PackedLinearReductionProof,
    PackedLinearVerifierReduction, PackedLinearWitnessSource,
};
use super::util::{checked_domain_size, invalid_batch};

pub fn has_packed_linear_view<F, C, OpeningId, RelationId, Claim>(
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId, Claim>,
) -> bool {
    statement
        .claims
        .iter()
        .any(|claim| matches!(claim.view, PhysicalView::PackedLinear { .. }))
}

pub fn validate_packed_linear_statement<F, C, OpeningId, RelationId, L>(
    layout: &L,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
) -> Result<C, OpeningsError>
where
    F: Field,
    C: Clone + Eq,
    L: PackedLinearLayout,
{
    let digest = layout.digest();
    if statement.claims.is_empty() {
        return Err(invalid_batch(
            "packed linear opening requires at least one claim",
        ));
    }
    if statement.layout_digest != digest {
        return Err(invalid_batch(
            "packed linear statement layout digest does not match setup layout",
        ));
    }
    let commitment = statement.claims[0].commitment.clone();
    for claim in &statement.claims {
        if claim.commitment != commitment {
            return Err(invalid_batch(
                "packed linear opening claims must use one packed commitment",
            ));
        }
        let PhysicalView::PackedLinear {
            layout_digest,
            terms,
        } = &claim.view
        else {
            return Err(invalid_batch(
                "packed linear opening requires PackedLinear physical views",
            ));
        };
        if layout_digest != &digest {
            return Err(invalid_batch(
                "packed linear view layout digest does not match statement layout",
            ));
        }
        if terms.is_empty() {
            return Err(invalid_batch(
                "packed linear view requires at least one term",
            ));
        }
        for term in terms {
            validate_term(layout, term)?;
        }
    }
    Ok(commitment)
}

pub fn prove_packed_linear_reduction<F, C, OpeningId, RelationId, L, T>(
    layout: &L,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    packed_evals: Vec<F>,
    transcript: &mut T,
) -> Result<PackedLinearProverReduction<F>, OpeningsError>
where
    F: Field,
    C: Clone + Eq + AppendToTranscript,
    L: PackedLinearLayout,
    T: Transcript<Challenge = F>,
{
    let _ = validate_packed_linear_statement(layout, statement)?;
    bind_packed_statement(layout, statement, transcript)?;
    let gamma_powers = transcript.challenge_scalar_powers(statement.claims.len());
    let claimed_sum = reduced_claim(statement, &gamma_powers);
    let selector = packed_selector_evals(layout, statement, &gamma_powers)?;
    let (proof, sumcheck_point_lsb, opening_eval) =
        prove_product_sumcheck(selector, packed_evals, claimed_sum, transcript)?;
    Ok(PackedLinearProverReduction {
        proof,
        opening_point: native_opening_point(&sumcheck_point_lsb),
        opening_eval,
    })
}

pub fn prove_sparse_packed_linear_reduction<F, C, OpeningId, RelationId, L, S, T>(
    layout: &L,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    source: &S,
    transcript: &mut T,
) -> Result<PackedLinearProverReduction<F>, OpeningsError>
where
    F: Field,
    C: Clone + Eq + AppendToTranscript,
    L: PackedLinearLayout,
    S: PackedLinearWitnessSource<F>,
    T: Transcript<Challenge = F>,
{
    let _ = validate_packed_linear_statement(layout, statement)?;
    validate_source_layout(layout, source.layout())?;
    bind_packed_statement(layout, statement, transcript)?;
    let gamma_powers = transcript.challenge_scalar_powers(statement.claims.len());
    let claimed_sum = reduced_claim(statement, &gamma_powers);
    let (proof, sumcheck_point_lsb, opening_eval) = prove_sparse_product_sumcheck(
        layout,
        statement,
        &gamma_powers,
        source,
        claimed_sum,
        transcript,
    )?;
    Ok(PackedLinearProverReduction {
        proof,
        opening_point: native_opening_point(&sumcheck_point_lsb),
        opening_eval,
    })
}

pub fn verify_packed_linear_reduction<F, C, OpeningId, RelationId, L, T>(
    layout: &L,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    proof: &PackedLinearReductionProof,
    transcript: &mut T,
) -> Result<PackedLinearVerifierReduction<F, C>, OpeningsError>
where
    F: Field,
    C: Clone + Eq + AppendToTranscript,
    L: PackedLinearLayout,
    T: Transcript<Challenge = F>,
{
    let commitment = validate_packed_linear_statement(layout, statement)?;
    bind_packed_statement(layout, statement, transcript)?;
    let gamma_powers = transcript.challenge_scalar_powers(statement.claims.len());
    let coefficients = logical_coefficients(statement, &gamma_powers);
    let claimed_sum = reduced_claim(statement, &gamma_powers);
    let (sumcheck_point_lsb, final_claim) =
        verify_product_sumcheck::<F, _>(proof, claimed_sum, transcript)?;
    let selector_eval =
        packed_selector_eval(layout, statement, &gamma_powers, &sumcheck_point_lsb)?;
    let opening_eval = field_from_bytes::<F>(&proof.opening_eval)?;
    if final_claim != selector_eval * opening_eval {
        return Err(OpeningsError::VerificationFailed);
    }
    Ok(PackedLinearVerifierReduction {
        opening_point: native_opening_point(&sumcheck_point_lsb),
        opening_eval,
        result: BatchOpeningResult {
            coefficients,
            joint_commitment: commitment,
            reduced_opening: claimed_sum,
        },
    })
}

fn validate_source_layout<L, S>(layout: &L, source_layout: &S) -> Result<(), OpeningsError>
where
    L: PackedLinearLayout,
    S: PackedLinearLayout,
{
    if source_layout.digest() != layout.digest() || source_layout.dimension() != layout.dimension()
    {
        return Err(invalid_batch(
            "packed linear source layout does not match packed statement",
        ));
    }
    Ok(())
}

fn prove_product_sumcheck<F, T>(
    mut left: Vec<F>,
    mut right: Vec<F>,
    claimed_sum: F,
    transcript: &mut T,
) -> Result<(PackedLinearReductionProof, Vec<F>, F), OpeningsError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    if left.len() != right.len() || !left.len().is_power_of_two() {
        return Err(invalid_batch(
            "packed linear sumcheck inputs must have equal power-of-two lengths",
        ));
    }
    let rounds = left.len().trailing_zeros() as usize;
    let mut proof_rounds = Vec::with_capacity(rounds);
    let mut point = Vec::with_capacity(rounds);
    let mut current_claim = claimed_sum;
    transcript.append(&LabelWithCount(b"akpk_sum_rounds", rounds as u64));

    while left.len() > 1 {
        let round = product_round(&left, &right);
        if round[0] + round[1] != current_claim {
            return Err(invalid_batch(
                "packed linear claims do not match packed witness evaluations",
            ));
        }
        append_round(transcript, &round);
        let challenge = transcript.challenge_scalar();
        point.push(challenge);
        fold_product_inputs(&mut left, &mut right, challenge);
        current_claim = eval_quadratic(round, challenge);
        proof_rounds.push(encode_round(round));
    }
    if left[0] * right[0] != current_claim {
        return Err(invalid_batch("packed linear sumcheck final claim mismatch"));
    }
    let opening_eval = right[0];
    opening_eval.append_to_transcript(transcript);
    Ok((
        PackedLinearReductionProof {
            rounds: proof_rounds,
            opening_eval: field_bytes(opening_eval),
        },
        point,
        opening_eval,
    ))
}

fn prove_sparse_product_sumcheck<F, C, OpeningId, RelationId, L, S, T>(
    layout: &L,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    gamma_powers: &[F],
    source: &S,
    claimed_sum: F,
    transcript: &mut T,
) -> Result<(PackedLinearReductionProof, Vec<F>, F), OpeningsError>
where
    F: Field,
    L: PackedLinearLayout,
    S: PackedLinearWitnessSource<F>,
    T: Transcript<Challenge = F>,
{
    let mut right = sparse_product_input(source)?;
    let rounds = layout.dimension();
    let mut proof_rounds = Vec::with_capacity(rounds);
    let mut point = Vec::with_capacity(rounds);
    let mut current_claim = claimed_sum;
    transcript.append(&LabelWithCount(b"akpk_sum_rounds", rounds as u64));

    for _ in 0..rounds {
        let round = sparse_product_round(layout, statement, gamma_powers, &point, &right)?;
        if round[0] + round[1] != current_claim {
            return Err(invalid_batch(
                "packed linear claims do not match sparse packed witness evaluations",
            ));
        }
        append_round(transcript, &round);
        let challenge = transcript.challenge_scalar();
        point.push(challenge);
        fold_sparse_product_input(&mut right, challenge);
        current_claim = eval_quadratic(round, challenge);
        proof_rounds.push(encode_round(round));
    }

    let opening_eval = right.first().map_or_else(F::zero, |(_, eval)| *eval);
    let selector_eval = packed_selector_eval(layout, statement, gamma_powers, &point)?;
    if selector_eval * opening_eval != current_claim {
        return Err(invalid_batch("packed linear sumcheck final claim mismatch"));
    }
    opening_eval.append_to_transcript(transcript);
    Ok((
        PackedLinearReductionProof {
            rounds: proof_rounds,
            opening_eval: field_bytes(opening_eval),
        },
        point,
        opening_eval,
    ))
}

fn sparse_product_input<F, S>(source: &S) -> Result<Vec<(usize, F)>, OpeningsError>
where
    F: Field,
    S: PackedLinearWitnessSource<F>,
{
    let layout = source.layout();
    let domain_size = checked_domain_size(layout.dimension())?;
    if layout.cells() > domain_size {
        return Err(invalid_batch(format!(
            "packed linear witness has {} cells but dimension {} supports {domain_size}",
            layout.cells(),
            layout.dimension()
        )));
    }

    let mut entries = Vec::new();
    let mut error = None;
    source.for_each_nonzero(|rank, value| {
        if error.is_some() {
            return;
        }
        if rank >= layout.cells() {
            error = Some(invalid_batch(format!(
                "packed linear witness source emitted rank {rank} outside {} real cells",
                layout.cells()
            )));
            return;
        }
        if value.is_zero() {
            error = Some(invalid_batch(format!(
                "packed linear witness source emitted zero at rank {rank}"
            )));
            return;
        }
        entries.push((rank, value));
    });
    if let Some(error) = error {
        return Err(error);
    }

    entries.sort_by_key(|(rank, _)| *rank);
    for window in entries.windows(2) {
        if window[0].0 == window[1].0 {
            return Err(invalid_batch(format!(
                "packed linear witness source emitted rank {} more than once",
                window[0].0
            )));
        }
    }
    Ok(entries)
}

fn sparse_product_round<F, C, OpeningId, RelationId, L>(
    layout: &L,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    gamma_powers: &[F],
    fixed_point: &[F],
    right: &[(usize, F)],
) -> Result<[F; 3], OpeningsError>
where
    F: Field,
    L: PackedLinearLayout,
{
    let mut evals = [F::zero(); 3];
    let mut cursor = 0usize;
    while cursor < right.len() {
        let pair_index = right[cursor].0 / 2;
        let mut right_0 = F::zero();
        let mut right_1 = F::zero();
        while cursor < right.len() && right[cursor].0 / 2 == pair_index {
            let (index, value) = right[cursor];
            if index & 1 == 0 {
                right_0 += value;
            } else {
                right_1 += value;
            }
            cursor += 1;
        }

        let left_0 = packed_selector_eval_at_index(
            layout,
            statement,
            gamma_powers,
            fixed_point,
            pair_index * 2,
        )?;
        let left_1 = packed_selector_eval_at_index(
            layout,
            statement,
            gamma_powers,
            fixed_point,
            pair_index * 2 + 1,
        )?;
        evals[0] += left_0 * right_0;
        evals[1] += left_1 * right_1;
        evals[2] += (left_1 + left_1 - left_0) * (right_1 + right_1 - right_0);
    }
    Ok(evals)
}

fn packed_selector_eval_at_index<F, C, OpeningId, RelationId, L>(
    layout: &L,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    gamma_powers: &[F],
    fixed_point: &[F],
    index: usize,
) -> Result<F, OpeningsError>
where
    F: Field,
    L: PackedLinearLayout,
{
    if fixed_point.len() > layout.dimension() {
        return Err(invalid_batch(
            "packed linear selector fixed point exceeds layout dimension",
        ));
    }
    let remaining_bits = layout.dimension() - fixed_point.len();
    if index >= (1usize << remaining_bits) {
        return Err(invalid_batch(
            "packed linear selector index exceeds folded domain",
        ));
    }

    let mut point = Vec::with_capacity(layout.dimension());
    point.extend_from_slice(fixed_point);
    for bit in 0..remaining_bits {
        if (index >> bit) & 1 == 0 {
            point.push(F::zero());
        } else {
            point.push(F::one());
        }
    }
    packed_selector_eval(layout, statement, gamma_powers, &point)
}

fn fold_sparse_product_input<F>(right: &mut Vec<(usize, F)>, r: F)
where
    F: Field,
{
    let mut folded: Vec<(usize, F)> = Vec::with_capacity(right.len());
    for &(index, value) in right.iter() {
        let next_index = index / 2;
        let weight = if index & 1 == 0 { F::one() - r } else { r };
        let folded_value = value * weight;
        if folded_value.is_zero() {
            continue;
        }
        match folded.last_mut() {
            Some((last_index, last_value)) if *last_index == next_index => {
                *last_value += folded_value;
                if last_value.is_zero() {
                    let _ = folded.pop();
                }
            }
            _ => folded.push((next_index, folded_value)),
        }
    }
    *right = folded;
}

fn verify_product_sumcheck<F, T>(
    proof: &PackedLinearReductionProof,
    claimed_sum: F,
    transcript: &mut T,
) -> Result<(Vec<F>, F), OpeningsError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    transcript.append(&LabelWithCount(
        b"akpk_sum_rounds",
        proof.rounds.len() as u64,
    ));
    let mut point = Vec::with_capacity(proof.rounds.len());
    let mut current_claim = claimed_sum;
    for encoded_round in &proof.rounds {
        let round = decode_round(encoded_round)?;
        if round[0] + round[1] != current_claim {
            return Err(OpeningsError::VerificationFailed);
        }
        append_round(transcript, &round);
        let challenge = transcript.challenge_scalar();
        point.push(challenge);
        current_claim = eval_quadratic(round, challenge);
    }
    field_from_bytes::<F>(&proof.opening_eval)?.append_to_transcript(transcript);
    Ok((point, current_claim))
}

fn product_round<F>(left: &[F], right: &[F]) -> [F; 3]
where
    F: Field,
{
    let mut evals = [F::zero(); 3];
    for (left_pair, right_pair) in left.chunks_exact(2).zip(right.chunks_exact(2)) {
        let left_0 = left_pair[0];
        let left_1 = left_pair[1];
        let right_0 = right_pair[0];
        let right_1 = right_pair[1];
        evals[0] += left_0 * right_0;
        evals[1] += left_1 * right_1;
        evals[2] += (left_1 + left_1 - left_0) * (right_1 + right_1 - right_0);
    }
    evals
}

fn fold_product_inputs<F>(left: &mut Vec<F>, right: &mut Vec<F>, r: F)
where
    F: Field,
{
    let half = left.len() / 2;
    for index in 0..half {
        let left_0 = left[2 * index];
        let left_1 = left[2 * index + 1];
        let right_0 = right[2 * index];
        let right_1 = right[2 * index + 1];
        left[index] = left_0 + r * (left_1 - left_0);
        right[index] = right_0 + r * (right_1 - right_0);
    }
    left.truncate(half);
    right.truncate(half);
}

fn eval_quadratic<F>(evals: [F; 3], r: F) -> F
where
    F: Field,
{
    let two_inv = F::from_u64(2).inv_or_zero();
    let l0 = (r - F::one()) * (r - F::from_u64(2)) * two_inv;
    let l1 = F::zero() - r * (r - F::from_u64(2));
    let l2 = r * (r - F::one()) * two_inv;
    evals[0] * l0 + evals[1] * l1 + evals[2] * l2
}
