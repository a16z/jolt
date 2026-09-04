//! Shared clear-sumcheck stages and the final packed HyperKZG opening.

use jolt_crypto::{Bn254, Bn254G1, PairingGroup};
use jolt_field::{Field, Fr, One, Ring, Zero};
use jolt_hyperkzg::{
    open_variable_batch, verify_variable_batch_observed, HyperKZGProverSetup,
    HyperKZGVerifierSetup, NoopVerifierObserver, VariableBatchKzgProof, VerifierObserver,
};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::batch::{BatchMember, BatchPrelude};
use jolt_sumcheck::prover::{prove_batch, ProveRounds, SequentialRounds};
use jolt_sumcheck::recorder::{ClearSumcheckRecorder, RecordedSumcheck, SumcheckRecorder};
use jolt_sumcheck::{
    append_sumcheck_claim, BooleanHypercube, ClearProof, CompressedSumcheckProof, SumcheckClaim,
    SumcheckError, SumcheckProof, OPENING_CLAIM_TRANSCRIPT_LABEL, SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::Transcript;

mod types;
pub use types::*;
mod packing;
pub use packing::{
    combine_packed_phases, commit_packed, Column, PackedColumns, PackedPolynomial, PackingLayout,
};
mod term_stage;
pub(crate) use term_stage::multilinear_evaluation_observed;
pub(crate) use term_stage::{
    coefficient_evaluation_with_weights_observed, eq_evaluations_observed, term_reduction,
    term_reduction_with_weights_observed, TermStageProver, WeightedColumnReduction,
};
mod shared_rounds;
pub(crate) use shared_rounds::{
    prove_batch_rounds, prove_rounds, prove_shared_opening, verify_batch_rounds, verify_rounds,
    verify_shared_opening, PendingRoundStage, VerifiedRoundStage,
};
mod protocol;
pub use protocol::{prove_assembly, verify_assembly, verify_assembly_with_cost};
pub(crate) use protocol::{
    prove_spartan_assembly, verify_spartan_assembly_from_transcript, ProverColumnPlan,
    SpartanAssembly, SpartanVerifierAssembly,
};
mod transcript;
pub use transcript::commitment_prefix_challenges;
pub(crate) use transcript::{assembly_transcript, CountingKeccakTranscript};

const STREAM_LABEL: &[u8] = b"jolt-wrapper-v1";
const KZG_ROUND_COMMITMENT_LABEL: &[u8] = b"sumcheck_kzg_commitment";
const KZG_ROUND_NEXT_LABEL: &[u8] = b"sumcheck_kzg_next";
const KZG_STAGE_SUM_ZERO_LABEL: &[u8] = b"sumcheck_kzg_sum_zero";

struct ScaledRounds<'a> {
    inner: &'a mut dyn ProveRounds<Fr>,
    scale: Fr,
    scale_inverse: Fr,
}

impl ProveRounds<Fr> for ScaledRounds<'_> {
    fn num_rounds(&self) -> usize {
        self.inner.num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        let polynomial =
            self.inner
                .prove_round(bind, round, previous_claim * self.scale_inverse)?;
        Ok(UnivariatePoly::new(
            polynomial
                .coefficients()
                .iter()
                .map(|coefficient| *coefficient * self.scale)
                .collect(),
        ))
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.inner.finish_rounds(bind)
    }
}

pub fn prove_stage<T: Transcript<Challenge = Fr>>(
    members: &mut [StageMember<'_>],
    transcript: &mut T,
) -> Result<(StageProof, StageResult), StreamError> {
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
    let input_claims: Vec<Fr> = members.iter().map(|member| member.input_claim).collect();
    let mut recorder = ClearSumcheckRecorder::<Fr, Commitment>::new();
    recorder.absorb_input_claims(&input_claims, transcript);
    let coefficients: Vec<Fr> = members.iter().map(|_| transcript.challenge()).collect();
    let descriptions: Vec<BatchMember<Fr>> = members
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
    let mut scaled: Vec<ScaledRounds<'_>> = members
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
    let mut provers: Vec<&mut dyn ProveRounds<Fr>> = scaled
        .iter_mut()
        .map(|member| member as &mut dyn ProveRounds<Fr>)
        .collect();
    let proved = prove_batch(
        &prelude,
        &mut provers,
        &mut SequentialRounds,
        &mut recorder,
        transcript,
    )?;
    let recorded = recorder.finish(&proved.member_claims, transcript)?;
    let round_polynomials = recorded
        .proof
        .as_clear()
        .and_then(|proof| match proof {
            ClearProof::Compressed(proof) => Some(proof.clone()),
            ClearProof::Full(_) => None,
        })
        .ok_or(StreamError::StageOutputClaim)?;
    let proof = StageProof {
        round_polynomials,
        committed_rounds: None,
    };
    Ok((
        proof,
        StageResult {
            point: proved.challenges,
            coefficients,
            output_claims: proved.member_claims,
            final_claim: proved.final_claim,
        },
    ))
}

fn prove_aggregated_round_opening<T: Transcript<Challenge = Fr>>(
    polynomials: &[Vec<Fr>],
    challenges: &[Fr],
    initial_claim: Fr,
    round_claims: &[Fr],
    degree: usize,
    setup: &HyperKZGProverSetup<Bn254>,
    transcript: &mut T,
) -> Result<(Fr, VariableBatchKzgProof<Bn254>), StreamError> {
    if polynomials.len() != challenges.len() || polynomials.len() != round_claims.len() {
        return Err(StreamError::StageCount);
    }
    let coefficients = (0..polynomials.len())
        .map(|_| transcript.challenge())
        .collect::<Vec<_>>();
    let mut sum = vec![Fr::zero(); degree + 1];
    for (polynomial, &coefficient) in polynomials.iter().zip(&coefficients) {
        for (target, &value) in sum.iter_mut().zip(polynomial) {
            *target += coefficient * value;
        }
    }
    let sum_at_zero = sum[0];
    transcript.append_labeled(KZG_STAGE_SUM_ZERO_LABEL, &sum_at_zero);
    let sum_claim = coefficients
        .iter()
        .zip(std::iter::once(&initial_claim).chain(round_claims.iter()))
        .map(|(&coefficient, &claim)| coefficient * claim)
        .sum::<Fr>();
    let mut opening_polynomials = polynomials.to_vec();
    opening_polynomials.push(sum);
    let mut points = challenges
        .iter()
        .map(|&challenge| vec![challenge])
        .collect::<Vec<_>>();
    points.push(vec![Fr::zero(), Fr::one()]);
    let mut evaluations = round_claims
        .iter()
        .map(|&claim| vec![claim])
        .collect::<Vec<_>>();
    evaluations.push(vec![sum_at_zero, sum_claim - sum_at_zero]);
    let opening = open_variable_batch(
        &opening_polynomials,
        &points,
        &evaluations,
        degree,
        setup,
        transcript,
    )?;
    Ok((sum_at_zero, opening))
}

#[expect(
    clippy::too_many_arguments,
    reason = "the aggregated opening is shared by standalone and batched committed sumchecks"
)]
fn verify_aggregated_round_opening<T, O>(
    commitments: &[Bn254G1],
    challenges: &[Fr],
    initial_claim: Fr,
    round_claims: &[Fr],
    sum_at_zero: Fr,
    degree: usize,
    opening: &VariableBatchKzgProof<Bn254>,
    setup: &HyperKZGVerifierSetup<Bn254>,
    transcript: &mut T,
    observer: &mut O,
) -> Result<(), StreamError>
where
    T: Transcript<Challenge = Fr>,
    O: VerifierObserver,
{
    if commitments.len() != challenges.len() || commitments.len() != round_claims.len() {
        return Err(StreamError::StageCount);
    }
    let coefficients = (0..commitments.len())
        .map(|_| transcript.challenge())
        .collect::<Vec<_>>();
    transcript.append_labeled(KZG_STAGE_SUM_ZERO_LABEL, &sum_at_zero);
    let sum_commitment = Bn254::g1_msm(commitments, &coefficients);
    observer.ec_mul(commitments.len());
    observer.ec_add(commitments.len());
    let mut sum_claim = Fr::zero();
    for (&coefficient, &claim) in coefficients
        .iter()
        .zip(std::iter::once(&initial_claim).chain(round_claims.iter()))
    {
        sum_claim += observer.fr_mul(coefficient, claim);
    }
    let mut opening_commitments = commitments.to_vec();
    opening_commitments.push(sum_commitment);
    let mut points = challenges
        .iter()
        .map(|&challenge| vec![challenge])
        .collect::<Vec<_>>();
    points.push(vec![Fr::zero(), Fr::one()]);
    let mut evaluations = round_claims
        .iter()
        .map(|&claim| vec![claim])
        .collect::<Vec<_>>();
    evaluations.push(vec![sum_at_zero, sum_claim - sum_at_zero]);
    verify_variable_batch_observed(
        &opening_commitments,
        &points,
        &evaluations,
        degree,
        opening,
        setup,
        transcript,
        observer,
    )?;
    Ok(())
}

pub fn prove_kzg_stage<T: Transcript<Challenge = Fr>>(
    prover: &mut dyn ProveRounds<Fr>,
    input_claim: Fr,
    degree: usize,
    setup: &HyperKZGProverSetup<Bn254>,
    transcript: &mut T,
) -> Result<(StageProof, StageResult), StreamError> {
    if !(5..=6).contains(&degree) || prover.num_rounds() == 0 {
        return Err(StreamError::StageEncoding);
    }
    append_sumcheck_claim(transcript, &input_claim);
    let mut polynomials = Vec::with_capacity(prover.num_rounds());
    let mut round_commitments = Vec::with_capacity(prover.num_rounds());
    let mut round_claims = Vec::with_capacity(prover.num_rounds());
    let mut challenges = Vec::with_capacity(prover.num_rounds());
    let mut previous_bind = None;
    let mut claim = input_claim;
    for round in 0..prover.num_rounds() {
        let polynomial = prover.prove_round(previous_bind, round, claim)?;
        if polynomial.coefficients().len() > degree + 1 {
            return Err(StreamError::StageEncoding);
        }
        let at_zero = polynomial.evaluate(Fr::zero());
        let at_one = polynomial.evaluate(Fr::one());
        if at_zero + at_one != claim {
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
        round_commitments.push(commitment);
        polynomials.push(polynomial.into_coefficients());
        previous_bind = Some(challenge);
        claim = next_claim;
    }
    let final_bind = previous_bind.ok_or(StreamError::EmptyStage)?;
    prover.finish_rounds(final_bind)?;
    let (sum_at_zero, opening) = prove_aggregated_round_opening(
        &polynomials,
        &challenges,
        input_claim,
        &round_claims,
        degree,
        setup,
        transcript,
    )?;
    Ok((
        StageProof {
            round_polynomials: CompressedSumcheckProof {
                round_polynomials: Vec::new(),
            },
            committed_rounds: Some(CommittedStageProof {
                round_commitments,
                round_claims,
                sum_at_zero,
                opening: Some(opening),
            }),
        },
        StageResult {
            point: challenges,
            coefficients: vec![Fr::one()],
            output_claims: vec![claim],
            final_claim: claim,
        },
    ))
}

struct KzgBatchRecorder<'a> {
    setup: &'a HyperKZGProverSetup<Bn254>,
    degree: usize,
    polynomials: Vec<Vec<Fr>>,
    commitments: Vec<Bn254G1>,
    claims: Vec<Fr>,
    challenges: Vec<Fr>,
}

impl<'a> KzgBatchRecorder<'a> {
    fn new(setup: &'a HyperKZGProverSetup<Bn254>, degree: usize) -> Result<Self, StreamError> {
        if !(3..=6).contains(&degree) {
            return Err(StreamError::StageEncoding);
        }
        if setup.g1_powers().len() < degree + 1 {
            return Err(StreamError::SetupTooSmall {
                required: degree + 1,
                actual: setup.g1_powers().len(),
            });
        }
        Ok(Self {
            setup,
            degree,
            polynomials: Vec::new(),
            commitments: Vec::new(),
            claims: Vec::new(),
            challenges: Vec::new(),
        })
    }
}

impl SumcheckRecorder<Fr> for KzgBatchRecorder<'_> {
    type Commitment = Commitment;

    fn absorb_input_claims<T>(&mut self, input_claims: &[Fr], transcript: &mut T)
    where
        T: Transcript<Challenge = Fr>,
    {
        for input_claim in input_claims {
            append_sumcheck_claim(transcript, input_claim);
        }
    }

    fn absorb_round<T>(
        &mut self,
        round_poly: &UnivariatePoly<Fr>,
        transcript: &mut T,
    ) -> Result<Fr, SumcheckError<Fr>>
    where
        T: Transcript<Challenge = Fr>,
    {
        if round_poly.coefficients().len() > self.degree + 1 {
            return Err(SumcheckError::DegreeBoundExceeded {
                got: round_poly.coefficients().len().saturating_sub(1),
                max: self.degree,
            });
        }
        let bases = self
            .setup
            .g1_powers()
            .get(..round_poly.coefficients().len())
            .ok_or(SumcheckError::DegreeBoundExceeded {
                got: round_poly.coefficients().len().saturating_sub(1),
                max: self.degree,
            })?;
        let commitment = Bn254::g1_affine_msm(bases, round_poly.coefficients());
        transcript.append_labeled(KZG_ROUND_COMMITMENT_LABEL, &commitment);
        let challenge = transcript.challenge();
        let next_claim = round_poly.evaluate(challenge);
        transcript.append_labeled(KZG_ROUND_NEXT_LABEL, &next_claim);
        self.polynomials.push(round_poly.coefficients().to_vec());
        self.commitments.push(commitment);
        self.claims.push(next_claim);
        self.challenges.push(challenge);
        Ok(challenge)
    }

    fn finish<T>(
        self,
        output_claim_values: &[Fr],
        transcript: &mut T,
    ) -> Result<RecordedSumcheck<Fr, Self::Commitment>, SumcheckError<Fr>>
    where
        T: Transcript<Challenge = Fr>,
    {
        for output_claim in output_claim_values {
            transcript.append_labeled(OPENING_CLAIM_TRANSCRIPT_LABEL, output_claim);
        }
        Ok(RecordedSumcheck {
            proof: SumcheckProof::Clear(ClearProof::Compressed(CompressedSumcheckProof {
                round_polynomials: Vec::new(),
            })),
            committed_witness: None,
        })
    }
}

pub fn prove_kzg_batch_stage<T: Transcript<Challenge = Fr>>(
    members: &mut [StageMember<'_>],
    setup: &HyperKZGProverSetup<Bn254>,
    transcript: &mut T,
) -> Result<(StageProof, StageResult), StreamError> {
    prove_kzg_batch_stage_inner(members, setup, transcript, true)
}

pub fn prove_kzg_batch_stage_deferred<T: Transcript<Challenge = Fr>>(
    members: &mut [StageMember<'_>],
    setup: &HyperKZGProverSetup<Bn254>,
    transcript: &mut T,
) -> Result<(StageProof, StageResult), StreamError> {
    prove_kzg_batch_stage_inner(members, setup, transcript, false)
}

fn prove_kzg_batch_stage_inner<T: Transcript<Challenge = Fr>>(
    members: &mut [StageMember<'_>],
    setup: &HyperKZGProverSetup<Bn254>,
    transcript: &mut T,
    absorb_member_claims: bool,
) -> Result<(StageProof, StageResult), StreamError> {
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
    let input_claims: Vec<Fr> = members.iter().map(|member| member.input_claim).collect();
    let mut recorder = KzgBatchRecorder::new(setup, max_degree)?;
    recorder.absorb_input_claims(&input_claims, transcript);
    let coefficients: Vec<Fr> = members.iter().map(|_| transcript.challenge()).collect();
    let descriptions: Vec<BatchMember<Fr>> = members
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
    let mut scaled: Vec<ScaledRounds<'_>> = members
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
    let mut provers: Vec<&mut dyn ProveRounds<Fr>> = scaled
        .iter_mut()
        .map(|member| member as &mut dyn ProveRounds<Fr>)
        .collect();
    let proved = prove_batch(
        &prelude,
        &mut provers,
        &mut SequentialRounds,
        &mut recorder,
        transcript,
    )?;
    if absorb_member_claims {
        for output_claim in &proved.member_claims {
            transcript.append_labeled(OPENING_CLAIM_TRANSCRIPT_LABEL, output_claim);
        }
    }
    let (sum_at_zero, opening) = prove_aggregated_round_opening(
        &recorder.polynomials,
        &recorder.challenges,
        prelude.claimed_sum,
        &recorder.claims,
        max_degree,
        setup,
        transcript,
    )?;
    Ok((
        StageProof {
            round_polynomials: CompressedSumcheckProof {
                round_polynomials: Vec::new(),
            },
            committed_rounds: Some(CommittedStageProof {
                round_commitments: recorder.commitments,
                round_claims: recorder.claims,
                sum_at_zero,
                opening: Some(opening),
            }),
        },
        StageResult {
            point: proved.challenges,
            coefficients,
            output_claims: proved.member_claims,
            final_claim: proved.final_claim,
        },
    ))
}

pub fn verify_kzg_batch_stage<T, F>(
    proof: &StageProof,
    members: &[StageMemberSpec],
    input_claims: &[Fr],
    setup: &HyperKZGVerifierSetup<Bn254>,
    transcript: &mut T,
    output_claims: F,
) -> Result<StageResult, StreamError>
where
    T: Transcript<Challenge = Fr>,
    F: FnOnce(&StageResult) -> Result<Vec<Fr>, StreamError>,
{
    verify_kzg_batch_stage_observed(
        proof,
        members,
        input_claims,
        setup,
        transcript,
        &mut NoopVerifierObserver,
        |result, _| output_claims(result),
    )
}

pub fn verify_kzg_batch_stage_observed<T, F, O>(
    proof: &StageProof,
    members: &[StageMemberSpec],
    input_claims: &[Fr],
    setup: &HyperKZGVerifierSetup<Bn254>,
    transcript: &mut T,
    observer: &mut O,
    output_claims: F,
) -> Result<StageResult, StreamError>
where
    T: Transcript<Challenge = Fr>,
    F: FnOnce(&StageResult, &mut O) -> Result<Vec<Fr>, StreamError>,
    O: VerifierObserver,
{
    verify_kzg_batch_stage_inner(
        proof,
        members,
        input_claims,
        setup,
        transcript,
        observer,
        |result, observer| output_claims(result, observer).map(Some),
    )
}

pub fn verify_kzg_batch_stage_deferred_observed<T, O>(
    proof: &StageProof,
    members: &[StageMemberSpec],
    input_claims: &[Fr],
    setup: &HyperKZGVerifierSetup<Bn254>,
    transcript: &mut T,
    observer: &mut O,
) -> Result<StageResult, StreamError>
where
    T: Transcript<Challenge = Fr>,
    O: VerifierObserver,
{
    verify_kzg_batch_stage_inner(
        proof,
        members,
        input_claims,
        setup,
        transcript,
        observer,
        |_, _| Ok(None),
    )
}

fn verify_kzg_batch_stage_inner<T, F, O>(
    proof: &StageProof,
    members: &[StageMemberSpec],
    input_claims: &[Fr],
    setup: &HyperKZGVerifierSetup<Bn254>,
    transcript: &mut T,
    observer: &mut O,
    output_claims: F,
) -> Result<StageResult, StreamError>
where
    T: Transcript<Challenge = Fr>,
    F: FnOnce(&StageResult, &mut O) -> Result<Option<Vec<Fr>>, StreamError>,
    O: VerifierObserver,
{
    if proof.committed_rounds.is_none() || !proof.round_polynomials.round_polynomials.is_empty() {
        return Err(StreamError::StageEncoding);
    }
    if input_claims.len() != members.len() || members.is_empty() {
        return Err(StreamError::StageMemberCount);
    }
    let max_rounds = members
        .iter()
        .map(|member| member.offset + member.rounds)
        .max()
        .ok_or(StreamError::EmptyStage)?;
    let max_degree = members
        .iter()
        .map(|member| member.degree)
        .max()
        .ok_or(StreamError::EmptyStage)?;
    if !(5..=6).contains(&max_degree) || max_rounds == 0 {
        return Err(StreamError::StageEncoding);
    }
    for input_claim in input_claims {
        append_sumcheck_claim(transcript, input_claim);
    }
    let coefficients: Vec<Fr> = members.iter().map(|_| transcript.challenge()).collect();
    let descriptions: Vec<BatchMember<Fr>> = members
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
    let prelude = BatchPrelude::new_observed(descriptions, max_rounds, max_degree, || {
        observer.record_fr_mul();
    });
    let committed = proof
        .committed_rounds
        .as_ref()
        .ok_or(StreamError::StageEncoding)?;
    if committed.round_commitments.len() != max_rounds || committed.round_claims.len() != max_rounds
    {
        return Err(StreamError::StageCount);
    }
    let mut claim = prelude.claimed_sum;
    let mut challenges = Vec::with_capacity(max_rounds);
    for (&commitment, &next_claim) in committed
        .round_commitments
        .iter()
        .zip(&committed.round_claims)
    {
        transcript.append_labeled(KZG_ROUND_COMMITMENT_LABEL, &commitment);
        let challenge = transcript.challenge();
        transcript.append_labeled(KZG_ROUND_NEXT_LABEL, &next_claim);
        challenges.push(challenge);
        claim = next_claim;
    }
    let mut result = StageResult {
        point: challenges.clone(),
        coefficients,
        output_claims: Vec::new(),
        final_claim: claim,
    };
    let output_claims = output_claims(&result, observer)?;
    if let Some(output_claims) = &output_claims {
        if output_claims.len() != members.len() {
            return Err(StreamError::StageMemberCount);
        }
        let mut expected = Fr::zero();
        for (&coefficient, &output_claim) in result.coefficients.iter().zip(output_claims) {
            expected += observer.fr_mul(coefficient, output_claim);
        }
        if expected != result.final_claim {
            return Err(StreamError::StageOutputClaim);
        }
        for output_claim in output_claims {
            transcript.append_labeled(OPENING_CLAIM_TRANSCRIPT_LABEL, output_claim);
        }
    }
    verify_aggregated_round_opening(
        &committed.round_commitments,
        &challenges,
        prelude.claimed_sum,
        &committed.round_claims,
        committed.sum_at_zero,
        max_degree,
        committed
            .opening
            .as_ref()
            .ok_or(StreamError::StageEncoding)?,
        setup,
        transcript,
        observer,
    )?;
    result.output_claims = output_claims.unwrap_or_default();
    Ok(result)
}

pub fn verify_kzg_stage<T: Transcript<Challenge = Fr>>(
    proof: &StageProof,
    input_claim: Fr,
    rounds: usize,
    degree: usize,
    setup: &HyperKZGVerifierSetup<Bn254>,
    transcript: &mut T,
) -> Result<StageResult, StreamError> {
    verify_kzg_stage_observed(
        proof,
        input_claim,
        rounds,
        degree,
        setup,
        transcript,
        &mut NoopVerifierObserver,
    )
}

pub fn verify_kzg_stage_observed<T, O>(
    proof: &StageProof,
    input_claim: Fr,
    rounds: usize,
    degree: usize,
    setup: &HyperKZGVerifierSetup<Bn254>,
    transcript: &mut T,
    observer: &mut O,
) -> Result<StageResult, StreamError>
where
    T: Transcript<Challenge = Fr>,
    O: VerifierObserver,
{
    if !(5..=6).contains(&degree) || !proof.round_polynomials.round_polynomials.is_empty() {
        return Err(StreamError::StageEncoding);
    }
    let committed = proof
        .committed_rounds
        .as_ref()
        .ok_or(StreamError::StageEncoding)?;
    if committed.round_commitments.len() != rounds || committed.round_claims.len() != rounds {
        return Err(StreamError::StageCount);
    }
    append_sumcheck_claim(transcript, &input_claim);
    let mut challenges = Vec::with_capacity(rounds);
    let mut claim = input_claim;
    for (&commitment, &next_claim) in committed
        .round_commitments
        .iter()
        .zip(&committed.round_claims)
    {
        transcript.append_labeled(KZG_ROUND_COMMITMENT_LABEL, &commitment);
        let challenge = transcript.challenge();
        transcript.append_labeled(KZG_ROUND_NEXT_LABEL, &next_claim);
        challenges.push(challenge);
        claim = next_claim;
    }
    verify_aggregated_round_opening(
        &committed.round_commitments,
        &challenges,
        input_claim,
        &committed.round_claims,
        committed.sum_at_zero,
        degree,
        committed
            .opening
            .as_ref()
            .ok_or(StreamError::StageEncoding)?,
        setup,
        transcript,
        observer,
    )?;
    Ok(StageResult {
        point: challenges,
        coefficients: vec![Fr::one()],
        output_claims: vec![claim],
        final_claim: claim,
    })
}

pub fn verify_stage_with<T, F>(
    proof: &StageProof,
    members: &[StageMemberSpec],
    input_claims: &[Fr],
    transcript: &mut T,
    output_claims: F,
) -> Result<StageResult, StreamError>
where
    T: Transcript<Challenge = Fr>,
    F: FnOnce(&StageResult) -> Result<Vec<Fr>, StreamError>,
{
    verify_stage_with_observed(
        proof,
        members,
        input_claims,
        transcript,
        &mut NoopVerifierObserver,
        |result, _| output_claims(result),
    )
}

pub fn verify_stage_with_observed<T, F, O>(
    proof: &StageProof,
    members: &[StageMemberSpec],
    input_claims: &[Fr],
    transcript: &mut T,
    observer: &mut O,
    output_claims: F,
) -> Result<StageResult, StreamError>
where
    T: Transcript<Challenge = Fr>,
    F: FnOnce(&StageResult, &mut O) -> Result<Vec<Fr>, StreamError>,
    O: VerifierObserver,
{
    let mut result =
        verify_stage_without_output_observed(proof, members, input_claims, transcript, observer)?;
    let output_claims = output_claims(&result, observer)?;
    if output_claims.len() != members.len() {
        return Err(StreamError::StageMemberCount);
    }
    let mut expected = Fr::zero();
    for (&coefficient, &claim) in result.coefficients.iter().zip(&output_claims) {
        expected += observer.fr_mul(coefficient, claim);
    }
    if result.final_claim != expected {
        return Err(StreamError::StageOutputClaim);
    }
    absorb_output_claims(&output_claims, transcript);
    result.output_claims = output_claims;
    Ok(result)
}

fn absorb_output_claims<T: Transcript>(claims: &[Fr], transcript: &mut T) {
    for claim in claims {
        transcript.append_labeled(OPENING_CLAIM_TRANSCRIPT_LABEL, claim);
    }
}

fn verify_stage_without_output_observed<T, O>(
    proof: &StageProof,
    members: &[StageMemberSpec],
    input_claims: &[Fr],
    transcript: &mut T,
    observer: &mut O,
) -> Result<StageResult, StreamError>
where
    T: Transcript<Challenge = Fr>,
    O: VerifierObserver,
{
    if proof.committed_rounds.is_some() {
        return Err(StreamError::StageEncoding);
    }
    if input_claims.len() != members.len() {
        return Err(StreamError::StageMemberCount);
    }
    let max_rounds = members
        .iter()
        .map(|member| member.offset + member.rounds)
        .max()
        .ok_or(StreamError::EmptyStage)?;
    let max_degree = members
        .iter()
        .map(|member| member.degree)
        .max()
        .ok_or(StreamError::EmptyStage)?;
    let mut recorder = ClearSumcheckRecorder::<Fr, Commitment>::new();
    recorder.absorb_input_claims(input_claims, transcript);
    let coefficients: Vec<Fr> = members.iter().map(|_| transcript.challenge()).collect();
    let descriptions: Vec<BatchMember<Fr>> = members
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
    let prelude = BatchPrelude::new_observed(descriptions, max_rounds, max_degree, || {
        observer.record_fr_mul();
    });
    let evaluation = proof.round_polynomials.verify_observed(
        &SumcheckClaim::new(max_rounds, max_degree, prelude.claimed_sum),
        BooleanHypercube,
        SUMCHECK_ROUND_TRANSCRIPT_LABEL,
        transcript,
        &mut || observer.record_fr_mul(),
    )?;
    Ok(StageResult {
        point: evaluation.point.into_vec(),
        coefficients,
        output_claims: Vec::new(),
        final_claim: evaluation.value,
    })
}
