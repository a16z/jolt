//! PCS-generic clear Spartan proving for an authenticated sparse R1CS.

#![forbid(unsafe_code)]
#![deny(clippy::indexing_slicing, clippy::panic_in_result_fn)]

mod rounds;

use jolt_field::{JoltField, One, Zero};
use jolt_openings::CommitmentScheme;
use jolt_poly::{EqPolynomial, Polynomial};
use jolt_spartan_verifier::{SpartanError, SpartanKey, SpartanProof, INNER_DEGREE, OUTER_DEGREE};
use jolt_sumcheck::{
    prove_batch, BatchMember, BatchPrelude, ClearProof, ClearSumcheckRecorder,
    CompressedSumcheckProof, ProveRounds, SequentialRounds, SumcheckProof, SumcheckRecorder,
};
use jolt_transcript::{AppendToTranscript, Transcript};

use rounds::{InnerRounds, OuterRounds};

/// Proves `A z * B z = C z` for `z = [1, public_inputs, witness]`.
///
/// Both setup and key policy are supplied by the application. This is a clear
/// argument; it reveals sumcheck coefficients and evaluations. The caller must
/// not interpret it as a zero-knowledge wrapper.
#[expect(
    clippy::type_complexity,
    reason = "the PCS determines field, commitment, and opening proof types"
)]
pub fn prove<PCS: CommitmentScheme>(
    key: &SpartanKey<PCS::Field>,
    public_inputs: &[PCS::Field],
    witness: &[PCS::Field],
    pcs_setup: &PCS::ProverSetup,
    transcript: &mut impl Transcript<Challenge = PCS::Field>,
) -> Result<SpartanProof<PCS::Field, PCS::Output, PCS::Proof>, SpartanError<PCS::Field>>
where
    PCS::Field: AppendToTranscript,
    PCS::Output: AppendToTranscript,
{
    key.validate_public_inputs(public_inputs)?;
    if witness.len() != key.witness_len() {
        return Err(SpartanError::WitnessLength);
    }
    let assignment = std::iter::once(PCS::Field::one())
        .chain(public_inputs.iter().copied())
        .chain(witness.iter().copied())
        .collect::<Vec<_>>();
    let mut witness = witness.to_vec();
    witness.resize(key.padded_witness_len(), PCS::Field::zero());
    let witness_poly = Polynomial::new(witness.clone());
    let (witness_commitment, hint) = PCS::commit(&witness_poly, pcs_setup)?;
    let tau = key.begin(public_inputs, &witness_commitment, transcript)?;
    let mut outer_rounds = OuterRounds::new(key, &assignment, &tau)?;
    let (outer, rx, outer_claim) = prove_rounds(
        &mut outer_rounds,
        OUTER_DEGREE,
        PCS::Field::zero(),
        transcript,
    )?;
    let outer_evaluations = outer_rounds.evaluations()?;
    key.check_outer(&tau, &rx, outer_claim, outer_evaluations)?;
    let row_weights = EqPolynomial::new(rx).evaluations();
    let (weights, inner_claim) =
        key.begin_inner(&row_weights, public_inputs, outer_evaluations, transcript)?;
    let mut linear = key.matrices().project_column_range(
        &row_weights,
        key.public_columns(),
        key.witness_len(),
        weights,
    )?;
    linear.resize(key.padded_witness_len(), PCS::Field::zero());
    let mut inner_rounds = InnerRounds::new(linear, witness, key.witness_vars());
    let (inner, ry, final_claim) =
        prove_rounds(&mut inner_rounds, INNER_DEGREE, inner_claim, transcript)?;
    let [linear_evaluation, witness_evaluation] = inner_rounds.evaluations()?;
    if final_claim != linear_evaluation * witness_evaluation {
        return Err(SpartanError::InnerClaim);
    }
    SpartanKey::append_witness_evaluation(witness_evaluation, transcript);
    let opening = PCS::open(
        &witness_poly,
        &ry,
        witness_evaluation,
        pcs_setup,
        Some(hint),
        transcript,
    )?;
    Ok(SpartanProof {
        witness_commitment,
        outer,
        outer_evaluations,
        inner,
        witness_evaluation,
        opening,
    })
}

fn prove_rounds<F: JoltField + AppendToTranscript>(
    member: &mut dyn ProveRounds<F>,
    degree: usize,
    claim: F,
    transcript: &mut impl Transcript<Challenge = F>,
) -> Result<(CompressedSumcheckProof<F>, Vec<F>, F), SpartanError<F>> {
    let prelude = BatchPrelude::try_new(
        vec![BatchMember {
            input_claim: claim,
            coefficient: F::one(),
            rounds: member.num_rounds(),
            offset: 0,
        }],
        member.num_rounds(),
        degree,
    )?;
    let mut recorder = ClearSumcheckRecorder::<F>::new();
    let result = prove_batch(
        &prelude,
        &mut [member],
        &mut SequentialRounds,
        &mut recorder,
        transcript,
    )?;
    let recorded = recorder.finish(&[], transcript)?;
    let SumcheckProof::Clear(ClearProof::Compressed(proof)) = recorded.proof else {
        return Err(SpartanError::InternalShape);
    };
    Ok((proof, result.challenges, result.final_claim))
}

/// Conditional clear v2 SPARK prototype; security and deployment gates remain.
#[cfg(feature = "preprocessed")]
pub mod preprocessed;
