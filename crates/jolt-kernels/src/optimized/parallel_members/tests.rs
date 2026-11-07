#![expect(
    clippy::unwrap_used,
    reason = "valid fixtures and test pool construction"
)]

use std::cell::Cell;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use jolt_field::{Fr, Ring};
use jolt_poly::{BindingOrder, Polynomial, UnivariatePoly};
use jolt_sumcheck::{
    prove_batch, BatchMember, BatchPrelude, ClearSumcheckRecorder, ProveRounds, SendRounds,
    SequentialRounds, SumcheckRecorder,
};
use jolt_transcript::{prover_transcript, Blake2b512, FieldTranscript};

use crate::reference::naive::sample_dense_round;
use rayon::ThreadPoolBuilder;

use super::*;

struct CountingMember<C>(C);

impl<C: FnMut()> ProveRounds<Fr> for CountingMember<C> {
    fn num_rounds(&self) -> usize {
        1
    }

    fn prove_round(
        &mut self,
        _bind: Option<Fr>,
        _round: usize,
        claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        (self.0)();
        Ok(UnivariatePoly::new(vec![Fr::from_u64(0), claim]))
    }

    fn finish_rounds(&mut self, _bind: Fr) -> Result<(), SumcheckError<Fr>> {
        (self.0)();
        Ok(())
    }
}

#[test]
fn mixed_capabilities_execute_every_member_once() {
    // Exercise the capability fallback even on single-core test runners.
    ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap()
        .install(|| {
            let send_visits = AtomicUsize::new(0);
            let local_visits = Rc::new(Cell::new(0));
            let mut send = SendRounds(CountingMember(|| {
                let _ = send_visits.fetch_add(1, Ordering::Relaxed);
            }));
            let mut local = CountingMember(|| local_visits.set(local_visits.get() + 1));
            let mut work = [
                MemberRound {
                    index: 0,
                    local_round: 0,
                    bind: None,
                    claim: Fr::from_u64(1),
                    member: &mut send,
                    message: None,
                },
                MemberRound {
                    index: 1,
                    local_round: 0,
                    bind: None,
                    claim: Fr::from_u64(1),
                    member: &mut local,
                    message: None,
                },
            ];
            assert!(ParallelMembers.batch_prove_round(&mut work).is_ok());
            assert!(work.iter().all(|item| item.message.is_some()));
            let mut finishes = [
                MemberFinish {
                    bind: Fr::from_u64(1),
                    member: &mut send,
                },
                MemberFinish {
                    bind: Fr::from_u64(1),
                    member: &mut local,
                },
            ];
            assert!(ParallelMembers.batch_finish_rounds(&mut finishes).is_ok());
            assert_eq!(send_visits.load(Ordering::Relaxed), 2);
            assert_eq!(local_visits.get(), 2);
        });
}

struct DenseMember {
    polynomial: Polynomial<Fr>,
    rounds: usize,
}

impl ProveRounds<Fr> for DenseMember {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        if let Some(challenge) = bind {
            self.polynomial
                .bind_with_order(challenge, BindingOrder::HighToLow);
        }
        sample_dense_round(
            self.polynomial.len() / 2,
            1,
            round,
            claim,
            |index, point| Ok(self.polynomial.sumcheck_round_eval(index, point)),
        )
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.polynomial
            .bind_with_order(bind, BindingOrder::HighToLow);
        Ok(())
    }
}

#[test]
fn concurrent_members_preserve_batch_proof_and_transcript() {
    ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
        .install(|| {
            let prove = |scheduler: &mut dyn RoundScheduler<Fr>| {
                let mut transcript = prover_transcript::<Blake2b512>(
                    &jolt_transcript::protocol_tag(b"concurrent-members"),
                    b"",
                );
                let mut recorder = ClearSumcheckRecorder::<Fr>::new();
                let tables: Vec<Vec<Fr>> = [3, 1, 2]
                    .into_iter()
                    .map(|rounds| {
                        (0..1u64 << rounds)
                            .map(|i| Fr::from_u64(7 + 31 * i))
                            .collect()
                    })
                    .collect();
                let claims: Vec<Fr> = tables
                    .iter()
                    .map(|values| values.iter().copied().sum())
                    .collect();
                recorder.absorb_input_claims(&claims, &mut transcript);
                let prelude = BatchPrelude::new(
                    [3, 1, 2]
                        .into_iter()
                        .zip(&claims)
                        .map(|(rounds, &input_claim)| BatchMember {
                            input_claim,
                            coefficient: transcript.challenge().unwrap(),
                            rounds,
                            offset: 3 - rounds,
                        })
                        .collect(),
                    3,
                    1,
                );
                let mut provers: Vec<_> = [3, 1, 2]
                    .into_iter()
                    .zip(&tables)
                    .map(|(rounds, values)| {
                        SendRounds(DenseMember {
                            polynomial: Polynomial::from(values.clone()),
                            rounds,
                        })
                    })
                    .collect();
                let mut members: Vec<&mut dyn ProveRounds<Fr>> = provers
                    .iter_mut()
                    .map(|member| member as &mut dyn ProveRounds<Fr>)
                    .collect();
                let proved = prove_batch(
                    &prelude,
                    &mut members,
                    scheduler,
                    &mut recorder,
                    &mut transcript,
                )
                .unwrap();
                for ((member, table), &claim) in prelude
                    .members
                    .iter()
                    .zip(tables)
                    .zip(&proved.member_claims)
                {
                    let expected =
                        Polynomial::from(table).evaluate(&proved.challenges[member.offset..]);
                    assert_eq!(claim, expected);
                }
                let recorded = recorder
                    .finish(&proved.member_claims, &mut transcript)
                    .unwrap();
                let next: Fr = transcript.challenge().unwrap();
                (proved, recorded.proof, next)
            };
            assert_eq!(prove(&mut ParallelMembers), prove(&mut SequentialRounds));
        });
}

struct FailingMember(usize);

impl ProveRounds<Fr> for FailingMember {
    fn num_rounds(&self) -> usize {
        1
    }

    fn prove_round(
        &mut self,
        _bind: Option<Fr>,
        _round: usize,
        _claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        Err(SumcheckError::MissingRoundMessage { member: self.0 })
    }

    fn finish_rounds(&mut self, _bind: Fr) -> Result<(), SumcheckError<Fr>> {
        Err(SumcheckError::MissingRoundMessage { member: self.0 })
    }
}

#[test]
fn concurrent_errors_follow_declaration_order() {
    ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
        .install(|| {
            let mut provers = [SendRounds(FailingMember(7)), SendRounds(FailingMember(2))];
            let mut work: Vec<_> = provers
                .iter_mut()
                .enumerate()
                .map(|(index, member)| MemberRound {
                    index,
                    local_round: 0,
                    bind: None,
                    claim: Fr::from_u64(0),
                    member,
                    message: None,
                })
                .collect();
            assert!(matches!(
                ParallelMembers.batch_prove_round(&mut work),
                Err(SumcheckError::MissingRoundMessage { member: 7 })
            ));
            let mut finishes: Vec<_> = provers
                .iter_mut()
                .map(|member| MemberFinish {
                    bind: Fr::from_u64(1),
                    member,
                })
                .collect();
            assert!(matches!(
                ParallelMembers.batch_finish_rounds(&mut finishes),
                Err(SumcheckError::MissingRoundMessage { member: 7 })
            ));
        });
}
