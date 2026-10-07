//! PCS-generic clear Spartan proving for an authenticated sparse R1CS.

#![forbid(unsafe_code)]
#![deny(clippy::indexing_slicing, clippy::panic_in_result_fn)]

mod rounds;

use jolt_field::{CanonicalDecode, JoltField, One, Zero};
use jolt_openings::CommitmentScheme;
use jolt_poly::{EqPolynomial, Polynomial};
use jolt_spartan_verifier::{inner_relation, SpartanError, SpartanKey, INNER_DEGREE, OUTER_DEGREE};
use jolt_sumcheck::{
    prove_batch, BatchMember, BatchPrelude, ClearSumcheckRecorder, ProveRounds, SequentialRounds,
    SumcheckRecorder,
};
use jolt_transcript::{ProverTranscript, Sponge};

use rounds::{InnerRounds, OuterRounds};

/// Proves `A z * B z = C z` for `z = [1, public_inputs, witness]` into
/// `transcript`, in the message order [`SpartanKey::verify`] reads.
///
/// This is a clear argument; it reveals sumcheck coefficients and evaluations.
/// The caller must not interpret it as a zero-knowledge wrapper.
pub fn prove<PCS: CommitmentScheme, H: Sponge>(
    key: &SpartanKey<PCS::Field>,
    public_inputs: &[PCS::Field],
    witness: &[PCS::Field],
    pcs_setup: &PCS::ProverSetup,
    transcript: &mut ProverTranscript<H>,
) -> Result<(), SpartanError<PCS::Field>>
where
    PCS::Field: CanonicalDecode,
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
    key.bind_statement(public_inputs, transcript)?;
    PCS::send_commitment(&witness_commitment, transcript);
    let tau = key.draw_tau(transcript);
    let mut outer_rounds = OuterRounds::new(key, &assignment, &tau)?;
    let (rx, outer_claim) = prove_rounds(
        &mut outer_rounds,
        OUTER_DEGREE,
        PCS::Field::zero(),
        transcript,
    )?;
    let mut outer_evaluations = outer_rounds.evaluations()?;
    key.check_outer(&tau, &rx, outer_claim, outer_evaluations)?;
    let row_weights = EqPolynomial::new(rx).evaluations();
    let (weights, inner_claim) = key.begin_inner(
        &row_weights,
        public_inputs,
        &mut outer_evaluations,
        transcript,
    )?;
    let mut linear = key.matrices().project_column_range(
        &row_weights,
        key.public_columns(),
        key.witness_len(),
        weights,
    )?;
    linear.resize(key.padded_witness_len(), PCS::Field::zero());
    let mut inner_rounds = InnerRounds::new(linear, witness, key.witness_vars());
    let (ry, final_claim) = prove_rounds(&mut inner_rounds, INNER_DEGREE, inner_claim, transcript)?;
    let [linear_evaluation, mut witness_evaluation] = inner_rounds.evaluations()?;
    if final_claim != inner_relation(linear_evaluation, witness_evaluation) {
        return Err(SpartanError::InnerClaim);
    }
    SpartanKey::exchange_witness_evaluation(&mut witness_evaluation, transcript)?;
    PCS::open(
        &witness_poly,
        &ry,
        witness_evaluation,
        pcs_setup,
        Some(hint),
        transcript,
    )?;
    Ok(())
}

/// Sends one compressed sumcheck and returns its point and final claim.
fn prove_rounds<F: JoltField, H: Sponge>(
    member: &mut dyn ProveRounds<F>,
    degree: usize,
    claim: F,
    transcript: &mut ProverTranscript<H>,
) -> Result<(Vec<F>, F), SpartanError<F>> {
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
    recorder.finish(&[], transcript)?;
    Ok((result.challenges, result.final_claim))
}
