use jolt_field::{Prime64Offset59, Ring};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{
    prove_batch, BatchMember, BatchPrelude, BooleanHypercube, ClearProof,
    ClearSumcheckRecorder, ProveRounds, SequentialRounds, SumcheckClaim, SumcheckProof,
    SumcheckError, SumcheckRecorder, SumcheckVerifier, SUMCHECK_CLAIM_TRANSCRIPT_LABEL,
    SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::Transcript;

type F = Prime64Offset59;

#[derive(Default)]
struct FixedTranscript;

impl Transcript for FixedTranscript {
    type Challenge = F;

    fn new(_label: &'static [u8]) -> Self {
        Self
    }

    fn append_bytes(&mut self, _bytes: &[u8]) {}

    fn challenge(&mut self) -> F {
        F::from_u64(7)
    }

    fn state(&self) -> [u8; 32] {
        [0; 32]
    }
}

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

    fn finish_rounds(
        &mut self,
        bind: F,
    ) -> Result<(), SumcheckError<F>> {
        assert_eq!(bind, F::from_u64(7));
        Ok(())
    }
}

fn main() -> Result<(), SumcheckError<F>> {
    let claim = SumcheckClaim::new(1, 1, F::from_u64(8));
    let prelude = BatchPrelude::try_new(
        vec![BatchMember {
            input_claim: claim.claimed_sum,
            coefficient: F::from_u64(1),
            rounds: 1,
            offset: 0,
        }],
        1,
        1,
    )?;
    let mut member = LinearRound;
    let mut members: [&mut dyn ProveRounds<F>; 1] = [&mut member];
    let mut recorder = ClearSumcheckRecorder::<F>::new();
    let mut prover_transcript = FixedTranscript::new(b"external-sumcheck-example");
    recorder.absorb_input_claims(&[claim.claimed_sum], &mut prover_transcript);
    let proved = prove_batch(
        &prelude,
        &mut members,
        &mut SequentialRounds,
        &mut recorder,
        &mut prover_transcript,
    )?;
    let recorded = recorder.finish(&proved.member_claims, &mut prover_transcript)?;
    let SumcheckProof::Clear(ClearProof::Compressed(proof)) = recorded.proof else {
        unreachable!("the clear recorder always returns a compressed clear proof")
    };

    let mut verifier_transcript = FixedTranscript::new(b"external-sumcheck-example");
    verifier_transcript.append_labeled(SUMCHECK_CLAIM_TRANSCRIPT_LABEL, &claim.claimed_sum);
    let reduced = SumcheckVerifier::verify_compressed(
        &claim,
        &proof,
        BooleanHypercube,
        SUMCHECK_ROUND_TRANSCRIPT_LABEL,
        &mut verifier_transcript,
    )?;

    assert_eq!(reduced.point.as_slice(), &[F::from_u64(7)]);
    assert_eq!(reduced.value, F::from_u64(17));
    assert_eq!(proved.final_claim, reduced.value);
    Ok(())
}
