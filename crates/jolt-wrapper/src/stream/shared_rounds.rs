use jolt_crypto::{Bn254, Bn254G1, PairingGroup};
use jolt_field::{Field, Fr, One, Ring, Zero};
use jolt_hyperkzg::{
    open_variable_batch, verify_variable_batch_observed, HyperKZGProverSetup,
    HyperKZGVerifierSetup, VariableBatchKzgProof, VerifierObserver,
};
use jolt_sumcheck::batch::{BatchMember, BatchPrelude};
use jolt_sumcheck::prover::{prove_batch, ProveRounds, SequentialRounds};
use jolt_sumcheck::recorder::SumcheckRecorder;
use jolt_sumcheck::{append_sumcheck_claim, CompressedSumcheckProof};
use jolt_transcript::Transcript;

use super::{
    CommittedStageProof, KzgBatchRecorder, ScaledRounds, StageMember, StageMemberSpec, StageProof,
    StageResult, StreamError, KZG_ROUND_COMMITMENT_LABEL, KZG_ROUND_NEXT_LABEL,
    KZG_STAGE_SUM_ZERO_LABEL,
};

pub(crate) struct PendingRoundStage {
    polynomials: Vec<Vec<Fr>>,
    commitments: Vec<Bn254G1>,
    challenges: Vec<Fr>,
    round_claims: Vec<Fr>,
    initial_claim: Fr,
    degree: usize,
}

pub(crate) struct VerifiedRoundStage<'a> {
    committed: &'a CommittedStageProof,
    challenges: Vec<Fr>,
    initial_claim: Fr,
    degree: usize,
}

pub(crate) fn prove_batch_rounds<T: Transcript<Challenge = Fr>>(
    members: &mut [StageMember<'_>],
    setup: &HyperKZGProverSetup<Bn254>,
    transcript: &mut T,
) -> Result<(PendingRoundStage, StageResult), StreamError> {
    let max_rounds = members
        .iter()
        .map(|member| member.offset + member.prover.num_rounds())
        .max()
        .ok_or(StreamError::EmptyStage)?;
    let max_degree = members
        .iter()
        .map(|member| member.degree)
        .max()
        .ok_or(StreamError::EmptyStage)?;
    if max_rounds == 0 {
        return Err(StreamError::EmptyStage);
    }
    let input_claims = members
        .iter()
        .map(|member| member.input_claim)
        .collect::<Vec<_>>();
    let mut recorder = KzgBatchRecorder::new(setup, max_degree)?;
    recorder.absorb_input_claims(&input_claims, transcript);
    let coefficients = members
        .iter()
        .map(|_| transcript.challenge())
        .collect::<Vec<_>>();
    let descriptions = members
        .iter()
        .zip(&coefficients)
        .map(|(member, &coefficient)| BatchMember {
            input_claim: member.input_claim,
            coefficient,
            rounds: member.prover.num_rounds(),
            offset: member.offset,
        })
        .collect();
    let prelude = BatchPrelude::new(descriptions, max_rounds, max_degree);
    let mut scaled = members
        .iter_mut()
        .map(|member| {
            let scale =
                Fr::one().mul_pow_2(max_rounds - member.offset - member.prover.num_rounds());
            Ok(ScaledRounds {
                scale,
                scale_inverse: scale.inverse().ok_or(StreamError::StageScale)?,
                inner: &mut *member.prover,
            })
        })
        .collect::<Result<Vec<_>, StreamError>>()?;
    let mut provers = scaled
        .iter_mut()
        .map(|member| member as &mut dyn ProveRounds<Fr>)
        .collect::<Vec<_>>();
    let proved = prove_batch(
        &prelude,
        &mut provers,
        &mut SequentialRounds,
        &mut recorder,
        transcript,
    )?;
    Ok((
        PendingRoundStage {
            polynomials: recorder.polynomials,
            commitments: recorder.commitments,
            challenges: recorder.challenges,
            round_claims: recorder.claims,
            initial_claim: prelude.claimed_sum,
            degree: max_degree,
        },
        StageResult {
            point: proved.challenges,
            coefficients,
            output_claims: proved.member_claims,
            final_claim: proved.final_claim,
        },
    ))
}

pub(crate) fn prove_rounds<T: Transcript<Challenge = Fr>>(
    prover: &mut dyn ProveRounds<Fr>,
    input_claim: Fr,
    degree: usize,
    setup: &HyperKZGProverSetup<Bn254>,
    transcript: &mut T,
) -> Result<(PendingRoundStage, StageResult), StreamError> {
    if !(5..=6).contains(&degree) || prover.num_rounds() == 0 {
        return Err(StreamError::StageEncoding);
    }
    append_sumcheck_claim(transcript, &input_claim);
    let mut polynomials = Vec::with_capacity(prover.num_rounds());
    let mut commitments = Vec::with_capacity(prover.num_rounds());
    let mut round_claims = Vec::with_capacity(prover.num_rounds());
    let mut challenges = Vec::with_capacity(prover.num_rounds());
    let mut previous_bind = None;
    let mut claim = input_claim;
    for round in 0..prover.num_rounds() {
        let polynomial = prover.prove_round(previous_bind, round, claim)?;
        if polynomial.coefficients().len() > degree + 1
            || polynomial.evaluate(Fr::zero()) + polynomial.evaluate(Fr::one()) != claim
        {
            return Err(StreamError::StageOutputClaim);
        }
        let bases = setup
            .g1_powers()
            .get(..polynomial.coefficients().len())
            .ok_or(StreamError::SetupTooSmall {
                required: polynomial.coefficients().len(),
                actual: setup.g1_powers().len(),
            })?;
        let commitment = Bn254::g1_affine_msm(bases, polynomial.coefficients());
        transcript.append_labeled(KZG_ROUND_COMMITMENT_LABEL, &commitment);
        let challenge = transcript.challenge();
        let next_claim = polynomial.evaluate(challenge);
        transcript.append_labeled(KZG_ROUND_NEXT_LABEL, &next_claim);
        challenges.push(challenge);
        round_claims.push(next_claim);
        commitments.push(commitment);
        polynomials.push(polynomial.into_coefficients());
        previous_bind = Some(challenge);
        claim = next_claim;
    }
    prover.finish_rounds(previous_bind.ok_or(StreamError::EmptyStage)?)?;
    Ok((
        PendingRoundStage {
            polynomials,
            commitments,
            challenges: challenges.clone(),
            round_claims,
            initial_claim: input_claim,
            degree,
        },
        StageResult {
            point: challenges,
            coefficients: vec![Fr::one()],
            output_claims: vec![claim],
            final_claim: claim,
        },
    ))
}

pub(crate) fn prove_shared_opening<T: Transcript<Challenge = Fr>>(
    stages: Vec<PendingRoundStage>,
    setup: &HyperKZGProverSetup<Bn254>,
    transcript: &mut T,
) -> Result<(Vec<StageProof>, VariableBatchKzgProof<Bn254>), StreamError> {
    let coefficients = stages
        .iter()
        .map(|stage| {
            stage
                .polynomials
                .iter()
                .map(|_| transcript.challenge())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut opening_polynomials = Vec::new();
    let mut points = Vec::new();
    let mut evaluations = Vec::new();
    let mut proofs = Vec::with_capacity(stages.len());
    for (stage, coefficients) in stages.into_iter().zip(coefficients) {
        if stage.degree > 6 {
            return Err(StreamError::StageEncoding);
        }
        let mut sum = vec![Fr::zero(); 7];
        for (polynomial, coefficient) in stage.polynomials.iter().zip(&coefficients) {
            for (target, &value) in sum.iter_mut().zip(polynomial) {
                *target += *coefficient * value;
            }
        }
        let sum_at_zero = sum[0];
        transcript.append_labeled(KZG_STAGE_SUM_ZERO_LABEL, &sum_at_zero);
        let sum_claim = coefficients
            .iter()
            .zip(std::iter::once(&stage.initial_claim).chain(&stage.round_claims))
            .map(|(&coefficient, &claim)| coefficient * claim)
            .sum::<Fr>();
        opening_polynomials.extend(stage.polynomials);
        opening_polynomials.push(sum);
        points.extend(stage.challenges.iter().map(|&challenge| vec![challenge]));
        points.push(vec![Fr::zero(), Fr::one()]);
        evaluations.extend(stage.round_claims.iter().map(|&claim| vec![claim]));
        evaluations.push(vec![sum_at_zero, sum_claim - sum_at_zero]);
        proofs.push(StageProof {
            round_polynomials: CompressedSumcheckProof {
                round_polynomials: Vec::new(),
            },
            committed_rounds: Some(CommittedStageProof {
                round_commitments: stage.commitments,
                round_claims: stage.round_claims,
                sum_at_zero,
                opening: None,
            }),
        });
    }
    let opening = open_variable_batch(
        &opening_polynomials,
        &points,
        &evaluations,
        6,
        setup,
        transcript,
    )?;
    Ok((proofs, opening))
}

pub(crate) fn verify_batch_rounds<'a, T, O>(
    proof: &'a StageProof,
    members: &[StageMemberSpec],
    input_claims: &[Fr],
    transcript: &mut T,
    observer: &mut O,
) -> Result<(StageResult, VerifiedRoundStage<'a>), StreamError>
where
    T: Transcript<Challenge = Fr>,
    O: VerifierObserver,
{
    if input_claims.len() != members.len() || members.is_empty() {
        return Err(StreamError::StageMemberCount);
    }
    let rounds = members
        .iter()
        .map(|member| member.offset + member.rounds)
        .max()
        .ok_or(StreamError::EmptyStage)?;
    let degree = members
        .iter()
        .map(|member| member.degree)
        .max()
        .ok_or(StreamError::EmptyStage)?;
    for claim in input_claims {
        append_sumcheck_claim(transcript, claim);
    }
    let coefficients = members
        .iter()
        .map(|_| transcript.challenge())
        .collect::<Vec<_>>();
    let descriptions = members
        .iter()
        .zip(input_claims)
        .zip(&coefficients)
        .map(|((member, &input_claim), &coefficient)| BatchMember {
            input_claim,
            coefficient,
            rounds: member.rounds,
            offset: member.offset,
        })
        .collect();
    let prelude = BatchPrelude::new_observed(descriptions, rounds, degree, || {
        observer.record_fr_mul();
    });
    verify_round_transcript(
        proof,
        rounds,
        prelude.claimed_sum,
        degree,
        coefficients,
        transcript,
    )
}

pub(crate) fn verify_rounds<'a, T: Transcript<Challenge = Fr>>(
    proof: &'a StageProof,
    input_claim: Fr,
    rounds: usize,
    degree: usize,
    transcript: &mut T,
) -> Result<(StageResult, VerifiedRoundStage<'a>), StreamError> {
    append_sumcheck_claim(transcript, &input_claim);
    verify_round_transcript(
        proof,
        rounds,
        input_claim,
        degree,
        vec![Fr::one()],
        transcript,
    )
}

fn verify_round_transcript<'a, T: Transcript<Challenge = Fr>>(
    proof: &'a StageProof,
    rounds: usize,
    initial_claim: Fr,
    degree: usize,
    coefficients: Vec<Fr>,
    transcript: &mut T,
) -> Result<(StageResult, VerifiedRoundStage<'a>), StreamError> {
    let committed = proof
        .committed_rounds
        .as_ref()
        .filter(|committed| committed.opening.is_none())
        .ok_or(StreamError::StageEncoding)?;
    if !proof.round_polynomials.round_polynomials.is_empty()
        || committed.round_commitments.len() != rounds
        || committed.round_claims.len() != rounds
    {
        return Err(StreamError::StageCount);
    }
    let mut challenges = Vec::with_capacity(rounds);
    let mut claim = initial_claim;
    for (&commitment, &next_claim) in committed
        .round_commitments
        .iter()
        .zip(&committed.round_claims)
    {
        transcript.append_labeled(KZG_ROUND_COMMITMENT_LABEL, &commitment);
        challenges.push(transcript.challenge());
        transcript.append_labeled(KZG_ROUND_NEXT_LABEL, &next_claim);
        claim = next_claim;
    }
    Ok((
        StageResult {
            point: challenges.clone(),
            coefficients,
            output_claims: Vec::new(),
            final_claim: claim,
        },
        VerifiedRoundStage {
            committed,
            challenges,
            initial_claim,
            degree,
        },
    ))
}

pub(crate) fn verify_shared_opening<T, O>(
    stages: &[VerifiedRoundStage<'_>],
    opening: &VariableBatchKzgProof<Bn254>,
    setup: &HyperKZGVerifierSetup<Bn254>,
    transcript: &mut T,
    observer: &mut O,
) -> Result<(), StreamError>
where
    T: Transcript<Challenge = Fr>,
    O: VerifierObserver,
{
    let coefficients = stages
        .iter()
        .map(|stage| {
            stage
                .committed
                .round_commitments
                .iter()
                .map(|_| transcript.challenge())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut commitments = Vec::new();
    let mut points = Vec::new();
    let mut evaluations = Vec::new();
    for (stage, coefficients) in stages.iter().zip(coefficients) {
        if stage.degree > 6 {
            return Err(StreamError::StageEncoding);
        }
        transcript.append_labeled(KZG_STAGE_SUM_ZERO_LABEL, &stage.committed.sum_at_zero);
        let sum_commitment = Bn254::g1_msm(&stage.committed.round_commitments, &coefficients);
        observer.ec_mul(coefficients.len());
        observer.ec_add(coefficients.len());
        let sum_claim = coefficients
            .iter()
            .zip(std::iter::once(&stage.initial_claim).chain(&stage.committed.round_claims))
            .fold(Fr::zero(), |sum, (&coefficient, &claim)| {
                sum + observer.fr_mul(coefficient, claim)
            });
        commitments.extend_from_slice(&stage.committed.round_commitments);
        commitments.push(sum_commitment);
        points.extend(stage.challenges.iter().map(|&challenge| vec![challenge]));
        points.push(vec![Fr::zero(), Fr::one()]);
        evaluations.extend(
            stage
                .committed
                .round_claims
                .iter()
                .map(|&claim| vec![claim]),
        );
        evaluations.push(vec![
            stage.committed.sum_at_zero,
            sum_claim - stage.committed.sum_at_zero,
        ]);
    }
    verify_variable_batch_observed(
        &commitments,
        &points,
        &evaluations,
        6,
        opening,
        setup,
        transcript,
        observer,
    )?;
    Ok(())
}
