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
    prove_batch, MemberFinish, MemberRound, ProveRounds, RoundExecutionDomain, RoundScheduler,
    SequentialRounds, TwoLaneRounds,
};
use crate::recorder::{ClearSumcheckRecorder, SumcheckRecorder};
use crate::tests::DenseMember;

type F = Fr;

struct DomainMember {
    inner: DenseMember,
    domain: RoundExecutionDomain,
}

impl ProveRounds<F> for DomainMember {
    fn num_rounds(&self) -> usize {
        self.inner.num_rounds()
    }

    fn execution_domain(&self) -> RoundExecutionDomain {
        self.domain
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<jolt_poly::UnivariatePoly<F>, SumcheckError<F>> {
        self.inner.prove_round(bind, round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.inner.finish_rounds(bind)
    }
}

#[test]
fn two_lane_scheduler_enters_host_and_accelerator_before_either_finishes() {
    use std::sync::{Arc, Condvar, Mutex};
    use std::time::Duration;

    struct BlockingMember {
        domain: RoundExecutionDomain,
        rendezvous: Arc<(Mutex<usize>, Condvar)>,
    }

    impl ProveRounds<F> for BlockingMember {
        fn num_rounds(&self) -> usize {
            1
        }

        fn execution_domain(&self) -> RoundExecutionDomain {
            self.domain
        }

        fn prove_round(
            &mut self,
            _bind: Option<F>,
            _round: usize,
            _previous_claim: F,
        ) -> Result<jolt_poly::UnivariatePoly<F>, SumcheckError<F>> {
            let (entered, ready) = &*self.rendezvous;
            let mut entered = entered.lock().unwrap();
            *entered += 1;
            ready.notify_all();
            let (entered, wait) = ready
                .wait_timeout_while(entered, Duration::from_secs(1), |entered| *entered < 2)
                .unwrap();
            if wait.timed_out() && *entered < 2 {
                return Err(SumcheckError::ComputeBackend {
                    backend: "two-lane-test",
                    message: "the second lane did not overlap the first".to_owned(),
                });
            }
            Ok(jolt_poly::UnivariatePoly::new(vec![
                F::from_u64(0),
                F::from_u64(0),
            ]))
        }

        fn finish_rounds(&mut self, _bind: F) -> Result<(), SumcheckError<F>> {
            Ok(())
        }
    }

    let rendezvous = Arc::new((Mutex::new(0), Condvar::new()));
    let mut host = BlockingMember {
        domain: RoundExecutionDomain::Host,
        rendezvous: rendezvous.clone(),
    };
    let mut accelerator = BlockingMember {
        domain: RoundExecutionDomain::Accelerator,
        rendezvous,
    };
    let mut work = [
        MemberRound {
            index: 0,
            local_round: 0,
            bind: None,
            claim: F::from_u64(0),
            member: &mut host,
            message: None,
        },
        MemberRound {
            index: 1,
            local_round: 0,
            bind: None,
            claim: F::from_u64(0),
            member: &mut accelerator,
            message: None,
        },
    ];

    assert!(TwoLaneRounds.batch_prove_round(&mut work).is_ok());
}

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
fn proof_is_invariant_under_two_lane_traversal() {
    let prove = |two_lane: bool| {
        let (long, short, prelude, mut transcript, mut recorder) = traversal_fixture();
        let mut long = DomainMember {
            inner: long,
            domain: RoundExecutionDomain::Accelerator,
        };
        let mut short = DomainMember {
            inner: short,
            domain: RoundExecutionDomain::Host,
        };
        let mut members: Vec<&mut dyn ProveRounds<F>> = vec![&mut long, &mut short];
        let mut sequential = SequentialRounds;
        let mut overlapped = TwoLaneRounds;
        let scheduler: &mut dyn RoundScheduler<F> = if two_lane {
            &mut overlapped
        } else {
            &mut sequential
        };
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
    let (overlapped, overlapped_proof, overlapped_state) = prove(true);
    assert_eq!(sequential, overlapped);
    assert_eq!(sequential_proof, overlapped_proof);
    assert_eq!(sequential_state, overlapped_state);
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
