use jolt_field::{Prime64Offset59, Ring};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{
    prove_batch, BatchMember, BatchPrelude, BooleanHypercube, ClearProof, ClearSumcheckRecorder,
    ProveRounds, SequentialRounds, SumcheckClaim, SumcheckError, SumcheckProof, SumcheckRecorder,
    SumcheckVerifier, OPENING_CLAIM_TRANSCRIPT_LABEL, SUMCHECK_CLAIM_TRANSCRIPT_LABEL,
    SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::Transcript;

type F = Prime64Offset59;

#[derive(Default)]
struct FixedTranscript {
    state: [u8; 32],
    cursor: usize,
}

impl Transcript for FixedTranscript {
    type Challenge = F;

    fn new(label: &'static [u8]) -> Self {
        let mut transcript = Self::default();
        transcript.append_bytes(label);
        transcript
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            let index = self.cursor % self.state.len();
            self.state[index] = self.state[index]
                .wrapping_mul(31)
                .wrapping_add(*byte)
                .wrapping_add(1);
            self.cursor += 1;
        }
    }

    fn challenge(&mut self) -> F {
        self.append_bytes(b"challenge");
        F::from_u64(7)
    }

    fn state(&self) -> [u8; 32] {
        self.state
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

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        assert_eq!(bind, F::from_u64(7));
        Ok(())
    }
}

fn main() -> Result<(), SumcheckError<F>> {
    // g(X) = 3 + 2X, so g(0) + g(1) = 8.
    let claim = SumcheckClaim::new(1, 1, F::from_u64(8));
    let mut prover_transcript = FixedTranscript::new(b"external-sumcheck-example");
    let mut recorder = ClearSumcheckRecorder::<F>::new();
    recorder.absorb_input_claims(&[claim.claimed_sum], &mut prover_transcript);
    let coefficient = prover_transcript.challenge_scalar();
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
    let recorded = recorder.finish(&proved.member_claims, &mut prover_transcript)?;
    let SumcheckProof::Clear(ClearProof::Compressed(proof)) = recorded.proof else {
        unreachable!("the clear recorder always returns a compressed clear proof")
    };

    let mut verifier_transcript = FixedTranscript::new(b"external-sumcheck-example");
    verifier_transcript.append_labeled(SUMCHECK_CLAIM_TRANSCRIPT_LABEL, &claim.claimed_sum);
    let verifier_coefficient = verifier_transcript.challenge_scalar();
    let combined_claim = SumcheckClaim::new(
        claim.num_vars,
        claim.degree,
        verifier_coefficient * claim.claimed_sum,
    );
    let reduced = SumcheckVerifier::verify_compressed(
        &combined_claim,
        &proof,
        BooleanHypercube,
        SUMCHECK_ROUND_TRANSCRIPT_LABEL,
        &mut verifier_transcript,
    )?;
    for opening_claim in &proved.member_claims {
        verifier_transcript.append_labeled(OPENING_CLAIM_TRANSCRIPT_LABEL, opening_claim);
    }

    assert_eq!(reduced.point.as_slice(), &[F::from_u64(7)]);
    assert_eq!(proved.member_claims, vec![F::from_u64(17)]);
    assert_eq!(reduced.value, coefficient * F::from_u64(17));
    assert_eq!(proved.final_claim, reduced.value);
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
    Ok(())
}
