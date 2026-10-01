use jolt_field::{Prime64Offset59, Ring};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{
    prove_batch, BatchMember, BatchPrelude, ClearSumcheckRecorder, ProveRounds, SequentialRounds,
    SumcheckClaim, SumcheckError, SumcheckRecorder, SumcheckVerifier,
};
use jolt_transcript::{Channel, Keccak, ProtocolId, ProverTranscript, VerifierTranscript};

type F = Prime64Offset59;

const PROTOCOL: ProtocolId = ProtocolId::new::<Keccak>("jolt-sumcheck/example");
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
    // g(X) = 3 + 2X, so g(0) + g(1) = 8.
    let claim = SumcheckClaim::new(1, 1, F::from_u64(8));

    let mut prover_transcript = ProverTranscript::<Keccak>::new(&PROTOCOL, SESSION);
    let mut recorder = ClearSumcheckRecorder::<F>::new();
    recorder.absorb_input_claims(&[claim.claimed_sum], &mut prover_transcript);
    let coefficient: F = prover_transcript.challenge_small();
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
        &mut prover_transcript,
    )?;
    recorder.finish(&proved.member_claims, &mut prover_transcript)?;
    let proof = prover_transcript.finish();

    let mut verifier_transcript = VerifierTranscript::<Keccak>::new(&PROTOCOL, SESSION, &proof);
    verifier_transcript.public(&claim.claimed_sum);
    let verifier_coefficient: F = verifier_transcript.challenge_small();
    let combined_claim = SumcheckClaim::new(
        claim.num_vars,
        claim.degree,
        verifier_coefficient * claim.claimed_sum,
    );
    let reduced = SumcheckVerifier::verify_compressed(&combined_claim, &mut verifier_transcript)?;
    let opening_claims: Vec<F> = verifier_transcript.receive_n(1)?;
    verifier_transcript.finish()?;

    let challenge = reduced.point.as_slice()[0];
    let opening = F::from_u64(3) + F::from_u64(2) * challenge;
    assert_eq!(opening_claims, vec![opening]);
    assert_eq!(proved.member_claims, vec![opening]);
    assert_eq!(reduced.value, verifier_coefficient * opening);
    assert_eq!(proved.final_claim, reduced.value);
    Ok(())
}
