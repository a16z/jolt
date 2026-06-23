use jolt_field::Field;
use jolt_transcript::{AppendToTranscript, LabelWithCount, Transcript};

use crate::{BatchOpeningResult, BatchOpeningStatement, OpeningsError, PhysicalView};

use super::encoding::{decode_round, encode_round, field_bytes, field_from_bytes};
use super::selector::{
    for_each_packed_selector_sparse_eval, logical_coefficients, native_opening_point,
    packed_selector_eval, packed_selector_evals, reduced_claim, validate_term,
};
use super::transcript::{append_round, bind_packed_statement};
use super::types::{
    PackingLayout, PackingProverReduction, PackingReductionProof, PackingSource,
    PackingVerifierReduction,
};
use super::util::{checked_domain_size, invalid_batch};

pub fn has_packing_view<F, C, OpeningId, RelationId, Claim>(
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId, Claim>,
) -> bool {
    statement
        .claims
        .iter()
        .any(|claim| matches!(claim.view, PhysicalView::Packing { .. }))
}

pub fn validate_packing_statement<F, C, OpeningId, RelationId, L>(
    layout: &L,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
) -> Result<C, OpeningsError>
where
    F: Field,
    C: Clone + Eq,
    L: PackingLayout,
{
    let digest = layout.digest();
    if statement.claims.is_empty() {
        return Err(invalid_batch("packing opening requires at least one claim"));
    }
    if statement.layout_digest != digest {
        return Err(invalid_batch(
            "packing statement layout digest does not match setup layout",
        ));
    }
    let commitment = statement.claims[0].commitment.clone();
    for claim in &statement.claims {
        if claim.commitment != commitment {
            return Err(invalid_batch(
                "packing opening claims must use one packed commitment",
            ));
        }
        let PhysicalView::Packing {
            layout_digest,
            terms,
        } = &claim.view
        else {
            return Err(invalid_batch(
                "packing opening requires Packing physical views",
            ));
        };
        if layout_digest != &digest {
            return Err(invalid_batch(
                "packing view layout digest does not match statement layout",
            ));
        }
        if terms.is_empty() {
            return Err(invalid_batch("packing view requires at least one term"));
        }
        for term in terms {
            validate_term(layout, term)?;
        }
    }
    Ok(commitment)
}

pub fn prove_packing_reduction<F, C, OpeningId, RelationId, L, T>(
    layout: &L,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    packed_evals: Vec<F>,
    transcript: &mut T,
) -> Result<PackingProverReduction<F>, OpeningsError>
where
    F: Field,
    C: Clone + Eq + AppendToTranscript,
    L: PackingLayout,
    T: Transcript<Challenge = F>,
{
    let _ = validate_packing_statement(layout, statement)?;
    bind_packed_statement(layout, statement, transcript)?;
    let gamma_powers = transcript.challenge_scalar_powers(statement.claims.len());
    let claimed_sum = reduced_claim(statement, &gamma_powers);
    let selector = packed_selector_evals(layout, statement, &gamma_powers)?;
    let (proof, sumcheck_point_lsb, opening_eval) =
        prove_product_sumcheck(selector, packed_evals, claimed_sum, transcript)?;
    Ok(PackingProverReduction {
        proof,
        opening_point: native_opening_point(&sumcheck_point_lsb),
        opening_eval,
    })
}

pub fn prove_sparse_packing_reduction<F, C, OpeningId, RelationId, L, S, T>(
    layout: &L,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    source: &S,
    transcript: &mut T,
) -> Result<PackingProverReduction<F>, OpeningsError>
where
    F: Field,
    C: Clone + Eq + AppendToTranscript,
    L: PackingLayout,
    S: PackingSource<F>,
    T: Transcript<Challenge = F>,
{
    let _ = validate_packing_statement(layout, statement)?;
    validate_source_layout(layout, source.layout())?;
    bind_packed_statement(layout, statement, transcript)?;
    let gamma_powers = transcript.challenge_scalar_powers(statement.claims.len());
    let claimed_sum = reduced_claim(statement, &gamma_powers);
    let right = sparse_product_input(source)?;
    let (proof, sumcheck_point_lsb, opening_eval) = prove_sparse_product_sumcheck(
        layout,
        statement,
        &gamma_powers,
        right,
        claimed_sum,
        transcript,
    )?;
    Ok(PackingProverReduction {
        proof,
        opening_point: native_opening_point(&sumcheck_point_lsb),
        opening_eval,
    })
}

pub fn prove_sparse_packing_reduction_from_entries<F, C, OpeningId, RelationId, L, T>(
    layout: &L,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    entries: Vec<(usize, F)>,
    transcript: &mut T,
) -> Result<PackingProverReduction<F>, OpeningsError>
where
    F: Field,
    C: Clone + Eq + AppendToTranscript,
    L: PackingLayout,
    T: Transcript<Challenge = F>,
{
    let _ = validate_packing_statement(layout, statement)?;
    bind_packed_statement(layout, statement, transcript)?;
    let gamma_powers = transcript.challenge_scalar_powers(statement.claims.len());
    let claimed_sum = reduced_claim(statement, &gamma_powers);
    let right = sparse_product_input_entries(layout, entries)?;
    let (proof, sumcheck_point_lsb, opening_eval) = prove_sparse_product_sumcheck(
        layout,
        statement,
        &gamma_powers,
        right,
        claimed_sum,
        transcript,
    )?;
    Ok(PackingProverReduction {
        proof,
        opening_point: native_opening_point(&sumcheck_point_lsb),
        opening_eval,
    })
}

pub fn verify_packing_reduction<F, C, OpeningId, RelationId, L, T>(
    layout: &L,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    proof: &PackingReductionProof,
    transcript: &mut T,
) -> Result<PackingVerifierReduction<F, C>, OpeningsError>
where
    F: Field,
    C: Clone + Eq + AppendToTranscript,
    L: PackingLayout,
    T: Transcript<Challenge = F>,
{
    let commitment = validate_packing_statement(layout, statement)?;
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
    Ok(PackingVerifierReduction {
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
    L: PackingLayout,
    S: PackingLayout,
{
    if source_layout.digest() != layout.digest() || source_layout.dimension() != layout.dimension()
    {
        return Err(invalid_batch(
            "packing source layout does not match packed statement",
        ));
    }
    Ok(())
}

fn prove_product_sumcheck<F, T>(
    mut left: Vec<F>,
    mut right: Vec<F>,
    claimed_sum: F,
    transcript: &mut T,
) -> Result<(PackingReductionProof, Vec<F>, F), OpeningsError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    if left.len() != right.len() || !left.len().is_power_of_two() {
        return Err(invalid_batch(
            "packing sumcheck inputs must have equal power-of-two lengths",
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
                "packing claims do not match packed witness evaluations",
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
        return Err(invalid_batch("packing sumcheck final claim mismatch"));
    }
    let opening_eval = right[0];
    opening_eval.append_to_transcript(transcript);
    Ok((
        PackingReductionProof {
            rounds: proof_rounds,
            opening_eval: field_bytes(opening_eval),
        },
        point,
        opening_eval,
    ))
}

fn prove_sparse_product_sumcheck<F, C, OpeningId, RelationId, L, T>(
    layout: &L,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    gamma_powers: &[F],
    mut right: Vec<(usize, F)>,
    claimed_sum: F,
    transcript: &mut T,
) -> Result<(PackingReductionProof, Vec<F>, F), OpeningsError>
where
    F: Field,
    L: PackingLayout,
    T: Transcript<Challenge = F>,
{
    let rounds = layout.dimension();
    let mut proof_rounds = Vec::with_capacity(rounds);
    let mut point = Vec::with_capacity(rounds);
    let mut current_claim = claimed_sum;
    transcript.append(&LabelWithCount(b"akpk_sum_rounds", rounds as u64));

    let streamed_rounds = streamed_selector_rounds(rounds);
    for _ in 0..streamed_rounds {
        let round = sparse_product_round_streaming_selector(
            layout,
            statement,
            gamma_powers,
            &point,
            &right,
        )?;
        if round[0] + round[1] != current_claim {
            return Err(invalid_batch(
                "packing claims do not match sparse packed witness evaluations",
            ));
        }
        append_round(transcript, &round);
        let challenge = transcript.challenge_scalar();
        point.push(challenge);
        fold_sparse_product_input(&mut right, challenge);
        current_claim = eval_quadratic(round, challenge);
        proof_rounds.push(encode_round(round));
    }

    let mut left = folded_selector_sparse_input(layout, statement, gamma_powers, &point)?;
    for _ in streamed_rounds..rounds {
        let round = sparse_sparse_product_round(&left, &right);
        if round[0] + round[1] != current_claim {
            return Err(invalid_batch(
                "packing claims do not match sparse packed witness evaluations",
            ));
        }
        append_round(transcript, &round);
        let challenge = transcript.challenge_scalar();
        point.push(challenge);
        fold_sparse_product_input(&mut left, challenge);
        fold_sparse_product_input(&mut right, challenge);
        current_claim = eval_quadratic(round, challenge);
        proof_rounds.push(encode_round(round));
    }

    let opening_eval = right.first().map_or_else(F::zero, |(_, eval)| *eval);
    let selector_eval = left.first().map_or_else(F::zero, |(_, eval)| *eval);
    if std::env::var_os("JOLT_DEBUG_PACKING_REDUCTION").is_some() {
        let direct_selector_eval = packed_selector_eval(layout, statement, gamma_powers, &point)?;
        if direct_selector_eval != selector_eval {
            return Err(invalid_batch(
                "packing sparse selector fold disagrees with direct selector evaluation",
            ));
        }
    }
    if selector_eval * opening_eval != current_claim {
        return Err(invalid_batch("packing sumcheck final claim mismatch"));
    }
    opening_eval.append_to_transcript(transcript);
    Ok((
        PackingReductionProof {
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
    S: PackingSource<F>,
{
    let layout = source.layout();
    let domain_size = checked_domain_size(layout.dimension())?;
    if layout.cells() > domain_size {
        return Err(invalid_batch(format!(
            "packing witness has {} cells but dimension {} supports {domain_size}",
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
                "packing witness source emitted rank {rank} outside {} real cells",
                layout.cells()
            )));
            return;
        }
        if value.is_zero() {
            error = Some(invalid_batch(format!(
                "packing witness source emitted zero at rank {rank}"
            )));
            return;
        }
        entries.push((rank, value));
    });
    if let Some(error) = error {
        return Err(error);
    }

    entries.sort_by_key(|(rank, _)| *rank);
    sparse_product_input_entries(layout, entries)
}

fn sparse_product_input_entries<F, L>(
    layout: &L,
    entries: Vec<(usize, F)>,
) -> Result<Vec<(usize, F)>, OpeningsError>
where
    F: Field,
    L: PackingLayout,
{
    let domain_size = checked_domain_size(layout.dimension())?;
    if layout.cells() > domain_size {
        return Err(invalid_batch(format!(
            "packing witness has {} cells but dimension {} supports {domain_size}",
            layout.cells(),
            layout.dimension()
        )));
    }
    for (rank, value) in &entries {
        if *rank >= layout.cells() {
            return Err(invalid_batch(format!(
                "packing witness source emitted rank {rank} outside {} real cells",
                layout.cells()
            )));
        }
        if value.is_zero() {
            return Err(invalid_batch(format!(
                "packing witness source emitted zero at rank {rank}"
            )));
        }
    }
    for window in entries.windows(2) {
        if window[0].0 == window[1].0 {
            return Err(invalid_batch(format!(
                "packing witness source emitted rank {} more than once",
                window[0].0
            )));
        }
        if window[0].0 > window[1].0 {
            return Err(invalid_batch(
                "packing witness source entries are not sorted",
            ));
        }
    }
    Ok(entries)
}

fn streamed_selector_rounds(rounds: usize) -> usize {
    rounds.saturating_sub(24).min(8)
}

fn sparse_sparse_product_round<F>(left: &[(usize, F)], right: &[(usize, F)]) -> [F; 3]
where
    F: Field,
{
    let mut evals = [F::zero(); 3];
    let mut left_cursor = 0usize;
    let mut right_cursor = 0usize;
    while left_cursor < left.len() || right_cursor < right.len() {
        let pair_index = match (
            left.get(left_cursor).map(|(index, _)| index / 2),
            right.get(right_cursor).map(|(index, _)| index / 2),
        ) {
            (Some(left_pair), Some(right_pair)) => left_pair.min(right_pair),
            (Some(left_pair), None) => left_pair,
            (None, Some(right_pair)) => right_pair,
            (None, None) => break,
        };

        let (left_0, left_1) = take_sparse_pair(left, &mut left_cursor, pair_index);
        let (right_0, right_1) = take_sparse_pair(right, &mut right_cursor, pair_index);
        add_product_round_pair(&mut evals, left_0, left_1, right_0, right_1);
    }
    evals
}

fn sparse_product_round_streaming_selector<F, C, OpeningId, RelationId, L>(
    layout: &L,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    gamma_powers: &[F],
    prefix_challenges: &[F],
    right: &[(usize, F)],
) -> Result<[F; 3], OpeningsError>
where
    F: Field,
    L: PackingLayout,
{
    let mut evals = [F::zero(); 3];
    let mut right_cursor = 0usize;
    stream_folded_selector_pairs(
        layout,
        statement,
        gamma_powers,
        prefix_challenges,
        |left_pair, left_0, left_1| {
            while peek_sparse_pair(right, right_cursor)
                .is_some_and(|right_pair| right_pair < left_pair)
            {
                let (_, right_0, right_1) = take_current_sparse_pair(right, &mut right_cursor);
                add_product_round_pair(&mut evals, F::zero(), F::zero(), right_0, right_1);
            }
            let (right_0, right_1) = if peek_sparse_pair(right, right_cursor) == Some(left_pair) {
                let (_, right_0, right_1) = take_current_sparse_pair(right, &mut right_cursor);
                (right_0, right_1)
            } else {
                (F::zero(), F::zero())
            };
            add_product_round_pair(&mut evals, left_0, left_1, right_0, right_1);
            Ok(())
        },
    )?;
    while right_cursor < right.len() {
        let (_, right_0, right_1) = take_current_sparse_pair(right, &mut right_cursor);
        add_product_round_pair(&mut evals, F::zero(), F::zero(), right_0, right_1);
    }
    Ok(evals)
}

fn folded_selector_sparse_input<F, C, OpeningId, RelationId, L>(
    layout: &L,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    gamma_powers: &[F],
    prefix_challenges: &[F],
) -> Result<Vec<(usize, F)>, OpeningsError>
where
    F: Field,
    L: PackingLayout,
{
    let mut entries = Vec::new();
    for_each_packed_selector_sparse_eval(layout, statement, gamma_powers, |rank, value| {
        let value = value * sparse_fold_weight(rank, prefix_challenges);
        if value.is_zero() {
            return Ok(());
        }
        push_merged_sparse_entry(&mut entries, rank >> prefix_challenges.len(), value)
    })?;
    Ok(entries)
}

fn stream_folded_selector_pairs<F, C, OpeningId, RelationId, L>(
    layout: &L,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    gamma_powers: &[F],
    prefix_challenges: &[F],
    mut visit: impl FnMut(usize, F, F) -> Result<(), OpeningsError>,
) -> Result<(), OpeningsError>
where
    F: Field,
    L: PackingLayout,
{
    let mut current_pair = None;
    let mut left_0 = F::zero();
    let mut left_1 = F::zero();
    for_each_packed_selector_sparse_eval(layout, statement, gamma_powers, |rank, value| {
        let value = value * sparse_fold_weight(rank, prefix_challenges);
        if value.is_zero() {
            return Ok(());
        }
        let folded_index = rank >> prefix_challenges.len();
        let pair = folded_index / 2;
        if let Some(current) = current_pair {
            if current != pair {
                visit(current, left_0, left_1)?;
                left_0 = F::zero();
                left_1 = F::zero();
            }
        }
        current_pair = Some(pair);
        if folded_index & 1 == 0 {
            left_0 += value;
        } else {
            left_1 += value;
        }
        Ok(())
    })?;
    if let Some(pair) = current_pair {
        visit(pair, left_0, left_1)?;
    }
    Ok(())
}

fn sparse_fold_weight<F>(rank: usize, challenges: &[F]) -> F
where
    F: Field,
{
    let mut weight = F::one();
    for (round, challenge) in challenges.iter().copied().enumerate() {
        if (rank >> round) & 1 == 0 {
            weight *= F::one() - challenge;
        } else {
            weight *= challenge;
        }
    }
    weight
}

fn push_merged_sparse_entry<F>(
    entries: &mut Vec<(usize, F)>,
    index: usize,
    value: F,
) -> Result<(), OpeningsError>
where
    F: Field,
{
    match entries.last_mut() {
        Some((last_index, last_value)) if *last_index == index => {
            *last_value += value;
            if last_value.is_zero() {
                let _ = entries.pop();
            }
        }
        Some((last_index, _)) if *last_index > index => {
            return Err(invalid_batch(
                "folded packing selector entries are not sorted",
            ));
        }
        _ => entries.push((index, value)),
    }
    Ok(())
}

fn peek_sparse_pair<F>(entries: &[(usize, F)], cursor: usize) -> Option<usize>
where
    F: Field,
{
    entries.get(cursor).map(|(index, _)| index / 2)
}

fn take_current_sparse_pair<F>(entries: &[(usize, F)], cursor: &mut usize) -> (usize, F, F)
where
    F: Field,
{
    let pair_index = entries[*cursor].0 / 2;
    let (value_0, value_1) = take_sparse_pair(entries, cursor, pair_index);
    (pair_index, value_0, value_1)
}

fn add_product_round_pair<F>(evals: &mut [F; 3], left_0: F, left_1: F, right_0: F, right_1: F)
where
    F: Field,
{
    evals[0] += left_0 * right_0;
    evals[1] += left_1 * right_1;
    evals[2] += (left_1 + left_1 - left_0) * (right_1 + right_1 - right_0);
}

fn take_sparse_pair<F>(entries: &[(usize, F)], cursor: &mut usize, pair_index: usize) -> (F, F)
where
    F: Field,
{
    let mut value_0 = F::zero();
    let mut value_1 = F::zero();
    while let Some((index, value)) = entries.get(*cursor).copied() {
        let current_pair = index / 2;
        if current_pair != pair_index {
            break;
        }
        if index & 1 == 0 {
            value_0 += value;
        } else {
            value_1 += value;
        }
        *cursor += 1;
    }
    (value_0, value_1)
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
    proof: &PackingReductionProof,
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
