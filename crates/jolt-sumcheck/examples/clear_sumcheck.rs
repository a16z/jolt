// One clear batched sumcheck, proved and verified over spongefish transcripts.
//
// Every message here is public: the prover and the verifier absorb the same
// values, so the verifier's NARG string is empty.

use jolt_field::{Prime128Offset275, Ring};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{
    prove_batch, BatchMember, BatchPrelude, BooleanHypercube, ClearSumcheckRecorder, ProveRounds,
    SequentialRounds, SumcheckClaim, SumcheckError, SumcheckRecorder, SumcheckVerifier,
};
use jolt_transcript::{
    prover_transcript, verifier_transcript, FieldTranscript, Keccak, Transcript,
};

type F = Prime128Offset275;
const SESSION: &[u8] = b"external-sumcheck-example";

struct LinearRound;

impl ProveRounds<F> for LinearRound {
    fn num_rounds(&self) -> usize {
        1
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        assert!(bind.is_none());
        assert_eq!(round, 0);
        assert_eq!(previous_claim, F::from_u64(8));
        Ok(UnivariatePoly::new(vec![F::from_u64(3), F::from_u64(2)]))
    }

    fn finish_rounds(&mut self, _bind: F) -> Result<(), SumcheckError<F>> {
        Ok(())
    }
}

fn main() -> Result<(), SumcheckError<F>> {
    let claim = SumcheckClaim::new(1, 1, F::from_u64(8));
    let mut prover_state =
        prover_transcript::<Keccak>(&jolt_transcript::protocol_tag(SESSION), &claim.claimed_sum);
    let mut recorder = ClearSumcheckRecorder::<F>::new();
    recorder.absorb_input_claims(&[claim.claimed_sum], &mut prover_state);
    let coefficient = prover_state.challenge()?;
    let prelude = BatchPrelude::try_new(
        vec![BatchMember {
            input_claim: claim.claimed_sum,
            coefficient,
            rounds: 1,
            offset: 0,
        }],
        1,
        1,
    )?;
    let mut member = LinearRound;
    let mut members: [&mut dyn ProveRounds<F>; 1] = [&mut member];
    let proved = prove_batch(
        &prelude,
        &mut members,
        &mut SequentialRounds,
        &mut recorder,
        &mut prover_state,
    )?;
    let recorded = recorder.finish(&proved.member_claims, &mut prover_state)?;
    let proof = recorded.proof;

    let narg = prover_state.narg_string().to_vec();
    let mut verifier_transcript = verifier_transcript::<Keccak>(
        &jolt_transcript::protocol_tag(SESSION),
        &claim.claimed_sum,
        &narg,
    );
    verifier_transcript.append(&claim.claimed_sum);
    let verifier_coefficient: F = verifier_transcript.challenge()?;
    let combined_claim = SumcheckClaim::new(
        claim.num_vars,
        claim.degree,
        verifier_coefficient * claim.claimed_sum,
    );
    let reduced = SumcheckVerifier::verify_compressed(
        &combined_claim,
        &proof,
        BooleanHypercube,
        &mut verifier_transcript,
    )?;
    for opening_claim in &proved.member_claims {
        verifier_transcript.append(opening_claim);
    }

    let challenge = proved.challenges[0];
    let round = UnivariatePoly::new(vec![F::from_u64(3), F::from_u64(2)]);
    assert_eq!(reduced.point.as_slice(), &[challenge]);
    assert_eq!(proved.member_claims, vec![round.evaluate(challenge)]);
    assert_eq!(reduced.value, coefficient * round.evaluate(challenge));
    assert_eq!(proved.final_claim, reduced.value);
    assert_eq!(
        prover_state.verifier_message::<[u8; 32]>(),
        verifier_transcript.verifier_message::<[u8; 32]>()
    );
    verifier_transcript
        .check_eof()
        .map_err(|_| SumcheckError::TranscriptRejected)?;
    Ok(())
}
