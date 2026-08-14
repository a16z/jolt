//! RoundScheduler seam: reorder must leave proof bytes unchanged; skipping an
//! active member must be reported, not folded as padding.

#![expect(
    clippy::unwrap_used,
    reason = "tests use unwrap on fallible prove paths under assertion"
)]

use jolt_crypto::Bn254G1;
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_transcript::{Blake2bTranscript, Transcript};

use crate::batch::{BatchMember, BatchPrelude};
use crate::error::SumcheckError;
use crate::prover::{
    prove_batch, MemberFinish, MemberRound, ProveRounds, RoundScheduler, SequentialRounds,
};
use crate::recorder::{ClearSumcheckRecorder, SumcheckRecorder};
use crate::tests::DenseMember;

type F = Fr;

struct ChaosRounds;

impl RoundScheduler<F> for ChaosRounds {
    fn batch_prove_round(
        &mut self,
        work: &mut [MemberRound<'_, F>],
    ) -> Result<(), SumcheckError<F>> {
        work.reverse();
        for item in work.iter_mut() {
            item.run()?;
        }
        Ok(())
    }

    fn batch_finish_rounds(
        &mut self,
        finishes: &mut [MemberFinish<'_, F>],
    ) -> Result<(), SumcheckError<F>> {
        for item in finishes.iter_mut().rev() {
            item.run()?;
        }
        Ok(())
    }
}

// Redundant for compilation (lib.rs gates the whole module), but the
// jolt-verifier FS audit parses this file standalone and would otherwise
// enroll the absorb below as a production Fiat-Shamir site.
#[cfg(test)]
fn traversal_fixture() -> (
    DenseMember,
    DenseMember,
    BatchPrelude<F>,
    Blake2bTranscript,
    ClearSumcheckRecorder<F, Bn254G1>,
) {
    let sum_long = F::from_u64(4242);
    let sum_short = F::from_u64(99);
    let long = DenseMember::with_sum(3, sum_long, 7);
    let short = DenseMember::with_sum(1, sum_short, 13);

    let mut transcript = Blake2bTranscript::new(b"traversal-seam");
    let mut recorder = ClearSumcheckRecorder::<F, Bn254G1>::new();
    recorder.absorb_input_claims(&[sum_long, sum_short], &mut transcript);
    let coeff_long: F = transcript.challenge_scalar();
    let coeff_short: F = transcript.challenge_scalar();
    let prelude = BatchPrelude::new(
        vec![
            BatchMember {
                input_claim: sum_long,
                coefficient: coeff_long,
                rounds: 3,
                offset: 0,
            },
            BatchMember {
                input_claim: sum_short,
                coefficient: coeff_short,
                rounds: 1,
                offset: 2,
            },
        ],
        3,
        1,
    );
    (long, short, prelude, transcript, recorder)
}

#[test]
fn proof_is_invariant_under_a_reordering_traversal() {
    let prove = |chaos: bool| {
        let (mut long, mut short, prelude, mut transcript, mut recorder) = traversal_fixture();
        let mut members: Vec<&mut dyn ProveRounds<F>> = vec![&mut long, &mut short];
        let mut sequential = SequentialRounds;
        let mut chaotic = ChaosRounds;
        let scheduler: &mut dyn RoundScheduler<F> =
            if chaos { &mut chaotic } else { &mut sequential };
        let proved = prove_batch(
            &prelude,
            &mut members,
            scheduler,
            &mut recorder,
            &mut transcript,
        )
        .unwrap();
        let recorded = recorder
            .finish(&proved.member_claims, &mut transcript)
            .unwrap();
        (proved, recorded.proof, transcript.state())
    };

    let (sequential, sequential_proof, sequential_state) = prove(false);
    let (chaotic, chaotic_proof, chaotic_state) = prove(true);
    assert_eq!(sequential.challenges, chaotic.challenges);
    assert_eq!(sequential.final_claim, chaotic.final_claim);
    assert_eq!(sequential.member_claims, chaotic.member_claims);
    assert_eq!(sequential_proof, chaotic_proof);
    assert_eq!(sequential_state, chaotic_state);
}

#[test]
fn skipping_an_active_member_is_reported_against_that_member() {
    struct SkipsMemberOne;
    impl RoundScheduler<F> for SkipsMemberOne {
        fn batch_prove_round(
            &mut self,
            work: &mut [MemberRound<'_, F>],
        ) -> Result<(), SumcheckError<F>> {
            for item in work.iter_mut().filter(|item| item.index != 1) {
                item.run()?;
            }
            Ok(())
        }

        fn batch_finish_rounds(
            &mut self,
            _finishes: &mut [MemberFinish<'_, F>],
        ) -> Result<(), SumcheckError<F>> {
            Ok(())
        }
    }

    let (mut long, mut short, prelude, mut transcript, mut recorder) = traversal_fixture();
    let mut members: Vec<&mut dyn ProveRounds<F>> = vec![&mut long, &mut short];
    let error = prove_batch(
        &prelude,
        &mut members,
        &mut SkipsMemberOne,
        &mut recorder,
        &mut transcript,
    )
    .unwrap_err();
    assert!(
        matches!(error, SumcheckError::MissingRoundMessage { member: 1 }),
        "expected MissingRoundMessage for member 1, got {error:?}"
    );
}

#[test]
fn a_rewritten_member_index_is_rejected() {
    struct RewritesIndex;
    impl RoundScheduler<F> for RewritesIndex {
        fn batch_prove_round(
            &mut self,
            work: &mut [MemberRound<'_, F>],
        ) -> Result<(), SumcheckError<F>> {
            for item in work.iter_mut() {
                item.run()?;
                item.index = usize::MAX;
            }
            Ok(())
        }

        fn batch_finish_rounds(
            &mut self,
            _finishes: &mut [MemberFinish<'_, F>],
        ) -> Result<(), SumcheckError<F>> {
            Ok(())
        }
    }

    let (mut long, mut short, prelude, mut transcript, mut recorder) = traversal_fixture();
    let mut members: Vec<&mut dyn ProveRounds<F>> = vec![&mut long, &mut short];
    let error = prove_batch(
        &prelude,
        &mut members,
        &mut RewritesIndex,
        &mut recorder,
        &mut transcript,
    )
    .unwrap_err();
    assert!(
        matches!(
            error,
            SumcheckError::RoundMemberIndexOutOfRange {
                member: usize::MAX,
                members: 2
            }
        ),
        "expected RoundMemberIndexOutOfRange, got {error:?}"
    );
}
