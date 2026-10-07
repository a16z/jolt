//! Unit tests for round messages, recorders, the batched prover, and the
//! uni-skip provers against the verifier entry points.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic_in_result_fn,
    reason = "tests may panic on assertion failures and index fixture data"
)]

use jolt_crypto::{Bn254, Bn254G1, JoltGroup, Pedersen, PedersenSetup, VectorCommitment};
use jolt_field::{CanonicalBytes, Field, Fr, Ring};
use jolt_poly::UnivariatePoly;
use jolt_transcript::{Blake2b512, Channel, ProtocolId, ProverTranscript, VerifierTranscript};

use crate::batch::{BatchMember, BatchPrelude};
use crate::claim::{SumcheckClaim, SumcheckStatement};
use crate::committed::{
    BatchedCommittedSumcheckConsistency, CommittedSumcheckConsistency, VerifiedCommittedRound,
};
use crate::domain::{CenteredIntegerDomain, SumcheckDomain};
use crate::error::SumcheckError;
use crate::prover::{
    prove_batch, prove_uniskip_clear, prove_uniskip_committed, ProveRounds, ProvedBatch,
    SequentialRounds,
};
use crate::recorder::{ClearSumcheckRecorder, CommittedSumcheckRecorder, SumcheckRecorder};
use crate::round_proof::{
    receive_compressed_round, receive_full_round, send_compressed_round, send_full_round,
};
use crate::verifier::SumcheckVerifier;

type F = Fr;
type H = Blake2b512;
type VC = Pedersen<Bn254G1>;

const PROTOCOL: ProtocolId = ProtocolId::new::<H>("jolt-sumcheck/unit-tests");

pub(crate) fn prover(session: &[u8]) -> ProverTranscript<H> {
    ProverTranscript::new(&PROTOCOL, session)
}

fn verifier<'a>(session: &[u8], narg: &'a [u8]) -> VerifierTranscript<'a, H> {
    VerifierTranscript::new(&PROTOCOL, session, narg)
}

/// The sponge state both roles must agree on after the same message sequence.
pub(crate) fn fingerprint<C: Channel>(channel: &mut C) -> [u8; 32] {
    channel.challenge_bytes::<32>()
}

fn poly(coefficients: &[u64]) -> UnivariatePoly<F> {
    UnivariatePoly::new(coefficients.iter().copied().map(F::from_u64).collect())
}

fn encode(values: &[u64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&value| F::from_u64(value).to_bytes_le_vec())
        .collect()
}

#[test]
fn round_messages_travel_at_the_degree_bound_width() {
    let quadratic = poly(&[2, 3, 5]);

    let mut full = prover(b"wire");
    send_full_round(&quadratic, 3, &mut full).unwrap();
    let full = full.finish();
    assert_eq!(full, encode(&[2, 3, 5, 0]));

    let mut compressed = prover(b"wire");
    send_compressed_round(&quadratic, 3, &mut compressed).unwrap();
    let compressed = compressed.finish();
    assert_eq!(compressed, encode(&[2, 5, 0]));

    let mut full_reader = verifier(b"wire", &full);
    let received = receive_full_round::<F, _>(3, &mut full_reader).unwrap();
    assert_eq!(received, poly(&[2, 3, 5, 0]));
    full_reader.finish().unwrap();

    let mut compressed_reader = verifier(b"wire", &compressed);
    let received = receive_compressed_round::<F, _>(3, &mut compressed_reader).unwrap();
    assert_eq!(
        received.coeffs_except_linear_term(),
        poly(&[2, 5, 0]).coefficients()
    );
    compressed_reader.finish().unwrap();
}

#[test]
fn senders_reject_rounds_above_the_degree_bound_before_writing() {
    let cubic = poly(&[1, 2, 3, 4]);
    let mut transcript = prover(b"over-degree");
    let mut untouched = prover(b"over-degree");

    assert!(matches!(
        send_full_round(&cubic, 2, &mut transcript),
        Err(SumcheckError::DegreeBoundExceeded { got: 3, max: 2 })
    ));
    assert!(matches!(
        send_compressed_round(&cubic, 2, &mut transcript),
        Err(SumcheckError::DegreeBoundExceeded { got: 3, max: 2 })
    ));
    assert!(transcript.narg().is_empty());
    assert_eq!(fingerprint(&mut transcript), fingerprint(&mut untouched));
}

fn verify_single_round(
    round: &UnivariatePoly<F>,
    degree: usize,
    claimed_sum: F,
    domain: CenteredIntegerDomain,
) -> Result<jolt_poly::EvaluationClaim<F>, SumcheckError<F>> {
    let mut transcript = prover(b"integer-domain");
    send_full_round(round, degree, &mut transcript).unwrap();
    let narg = transcript.finish();
    let mut transcript = verifier(b"integer-domain", &narg);
    let reduced = SumcheckVerifier::verify(
        &SumcheckClaim::new(1, degree, claimed_sum),
        domain,
        &mut transcript,
    )?;
    transcript.finish()?;
    Ok(reduced)
}

#[test]
fn centered_integer_domain_verifies_round_sum() {
    // Domain {-1, 0, 1}: s(-1) + s(0) + s(1) = 4 + 2 + 10.
    let round = poly(&[2, 3, 5]);
    let reduced =
        verify_single_round(&round, 2, F::from_u64(16), CenteredIntegerDomain::new(3)).unwrap();
    assert_eq!(reduced.point.len(), 1);
    assert_eq!(reduced.value, round.evaluate(reduced.point[0]));
}

#[test]
fn centered_integer_domain_uses_core_even_window_convention() {
    // Domain {-1, 0, 1, 2}: the identity sums to 2.
    let reduced = verify_single_round(
        &poly(&[0, 1]),
        1,
        F::from_u64(2),
        CenteredIntegerDomain::new(4),
    );
    assert!(reduced.is_ok(), "verification failed: {:?}", reduced.err());
}

#[test]
fn centered_integer_domain_rejects_wrong_sum() {
    let result = verify_single_round(
        &poly(&[0, 1]),
        1,
        F::from_u64(3),
        CenteredIntegerDomain::new(4),
    );
    assert!(matches!(
        result,
        Err(SumcheckError::RoundCheckFailed {
            round: 0,
            expected,
            actual,
        }) if expected == F::from_u64(3) && actual == F::from_u64(2)
    ));
}

#[test]
fn centered_integer_domain_rejects_empty_domain() {
    let result = verify_single_round(
        &poly(&[0, 1]),
        1,
        F::from_u64(0),
        CenteredIntegerDomain::new(0),
    );
    assert!(matches!(
        result,
        Err(SumcheckError::InvalidIntegerDomain { domain_size: 0 })
    ));
}

#[test]
fn centered_integer_domain_exposes_power_sums() {
    let domain = CenteredIntegerDomain::new(4);

    assert_eq!(domain.start().unwrap(), -1);
    assert_eq!(domain.power_sums(4).unwrap(), vec![4, 2, 6, 8]);
    assert_eq!(
        <CenteredIntegerDomain as SumcheckDomain<F>>::round_sum_coefficients(&domain, 3).unwrap(),
        vec![
            F::from_u64(4),
            F::from_u64(2),
            F::from_u64(6),
            F::from_u64(8)
        ]
    );
}

#[test]
fn batched_committed_consistency_accessors() {
    // `BatchedCommittedSumcheckConsistency` is produced by the generated ZK
    // verify driver (jolt-verifier-derive) and read back by BlindFold through
    // these accessors. Exercise the front-loaded suffix arithmetic and its
    // range errors directly on a hand-built instance.
    let consistency = CommittedSumcheckConsistency {
        rounds: vec![
            VerifiedCommittedRound {
                commitment: F::from_u64(11),
                degree: 2,
                challenge: F::from_u64(101),
            },
            VerifiedCommittedRound {
                commitment: F::from_u64(12),
                degree: 1,
                challenge: F::from_u64(102),
            },
            VerifiedCommittedRound {
                commitment: F::from_u64(13),
                degree: 0,
                challenge: F::from_u64(103),
            },
        ],
    };
    let batched = BatchedCommittedSumcheckConsistency {
        consistency,
        batching_coefficients: vec![F::from_u64(7), F::from_u64(9)],
        max_num_vars: 3,
        max_degree: 2,
    };

    let challenges = vec![F::from_u64(101), F::from_u64(102), F::from_u64(103)];
    assert_eq!(batched.challenges(), challenges);
    assert_eq!(batched.try_round_offset(1).unwrap(), 2);
    assert_eq!(
        batched.try_instance_point(1).unwrap(),
        challenges[2..].to_vec()
    );
    assert_eq!(batched.try_instance_point_at(0, 3).unwrap(), challenges);
    assert!(matches!(
        batched.try_instance_point(4),
        Err(SumcheckError::BatchedPointOutOfRange {
            offset: 0,
            num_vars: 4,
            total: 3
        })
    ));
    assert!(matches!(
        batched.try_instance_point_at(usize::MAX, 1),
        Err(SumcheckError::BatchedPointRangeOverflow {
            offset: usize::MAX,
            num_vars: 1
        })
    ));
}

fn batched_consistency_with_challenges(
    challenges: &[F],
    max_num_vars: usize,
) -> BatchedCommittedSumcheckConsistency<F, F> {
    BatchedCommittedSumcheckConsistency {
        consistency: CommittedSumcheckConsistency {
            rounds: challenges
                .iter()
                .enumerate()
                .map(|(index, &challenge)| VerifiedCommittedRound {
                    commitment: F::from_u64(50 + index as u64),
                    degree: 1,
                    challenge,
                })
                .collect(),
        },
        batching_coefficients: vec![F::from_u64(7)],
        max_num_vars,
        max_degree: 1,
    }
}

#[test]
fn batched_try_round_offset_rejects_instance_wider_than_batch() {
    let challenges: Vec<F> = (101..=103).map(F::from_u64).collect();
    let batched = batched_consistency_with_challenges(&challenges, 3);

    assert!(matches!(
        batched.try_round_offset(4),
        Err(SumcheckError::BatchedPointOutOfRange {
            offset: 0,
            num_vars: 4,
            total: 3
        })
    ));
    assert_eq!(batched.try_round_offset(3).unwrap(), 0);
}

#[test]
fn batched_instance_points_are_challenge_suffixes_for_mixed_arities() {
    let challenges: Vec<F> = (201..=204).map(F::from_u64).collect();
    let batched = batched_consistency_with_challenges(&challenges, 4);

    // Tail-aligned members of a mixed-arity batch (4, 2, and 1 rounds) get
    // dummy rounds front-loaded, so each point is a challenge suffix.
    assert_eq!(batched.try_round_offset(4).unwrap(), 0);
    assert_eq!(batched.try_round_offset(2).unwrap(), 2);
    assert_eq!(batched.try_round_offset(1).unwrap(), 3);
    assert_eq!(batched.try_instance_point(4).unwrap(), challenges);
    assert_eq!(
        batched.try_instance_point(2).unwrap(),
        challenges[2..].to_vec()
    );
    assert_eq!(
        batched.try_instance_point(1).unwrap(),
        challenges[3..].to_vec()
    );
    assert_eq!(
        batched.try_instance_point_at(0, 2).unwrap(),
        challenges[..2].to_vec()
    );
    assert_eq!(
        batched.try_instance_point_at(1, 2).unwrap(),
        challenges[1..3].to_vec()
    );
}

#[test]
fn batched_instance_point_at_rejects_windows_past_recorded_challenges() {
    let challenges: Vec<F> = (301..=303).map(F::from_u64).collect();
    let batched = batched_consistency_with_challenges(&challenges, 3);

    assert!(matches!(
        batched.try_instance_point_at(2, 2),
        Err(SumcheckError::BatchedPointOutOfRange {
            offset: 2,
            num_vars: 2,
            total: 3
        })
    ));
    assert!(matches!(
        batched.try_instance_point_at(4, 1),
        Err(SumcheckError::BatchedPointOutOfRange {
            offset: 4,
            num_vars: 1,
            total: 3
        })
    ));
    assert!(matches!(
        batched.try_instance_point_at(usize::MAX - 1, 2),
        Err(SumcheckError::BatchedPointRangeOverflow {
            offset,
            num_vars: 2
        }) if offset == usize::MAX - 1
    ));
    assert_eq!(
        batched.try_instance_point_at(3, 0).unwrap(),
        Vec::<F>::new()
    );
}

#[test]
fn batched_instance_point_rejects_declared_width_exceeding_recorded_rounds() {
    // Adversarial mismatch: the batch declares five rounds but the proof only
    // recorded three. The suffix offset is computed from the declared width,
    // so the lookup must fail against the recorded challenge count.
    let challenges: Vec<F> = (401..=403).map(F::from_u64).collect();
    let batched = batched_consistency_with_challenges(&challenges, 5);

    assert_eq!(batched.try_round_offset(1).unwrap(), 4);
    assert!(matches!(
        batched.try_instance_point(1),
        Err(SumcheckError::BatchedPointOutOfRange {
            offset: 4,
            num_vars: 1,
            total: 3
        })
    ));
}

#[test]
#[should_panic(expected = "degree >= 1")]
fn sumcheck_claim_new_rejects_degree_zero() {
    let _ = SumcheckClaim::<Fr>::new(3, 0, Fr::from_u64(0));
}

#[test]
#[should_panic(expected = "degree >= 1")]
fn sumcheck_statement_new_rejects_degree_zero() {
    let _ = SumcheckStatement::new(3, 0);
}

/// Drive an honest degree-1 sumcheck through the clear recorder: the claim
/// is absorbed publicly, the rounds and the output claim are sent, and the
/// verifier reading them back must land on the prover's bound value.
#[test]
fn clear_recorder_roundtrip_matches_compressed_verifier() {
    let num_vars = 3;
    let evals: Vec<F> = (0..1u64 << num_vars)
        .map(|i| F::from_u64(3 * i + 7))
        .collect();
    let claimed_sum: F = evals.iter().copied().sum();

    let mut prover_transcript = prover(b"recorder");
    let mut recorder = ClearSumcheckRecorder::<F>::new();
    recorder.absorb_input_claims(&[claimed_sum], &mut prover_transcript);

    let mut buf = evals;
    for _round in 0..num_vars {
        let half = buf.len() / 2;
        let eval_0: F = buf[..half].iter().copied().sum();
        let eval_1: F = buf[half..].iter().copied().sum();
        let round_poly = UnivariatePoly::new(vec![eval_0, eval_1 - eval_0]);

        let r = recorder
            .absorb_round(&round_poly, 1, &mut prover_transcript)
            .unwrap();

        for i in 0..half {
            buf[i] = buf[i] + r * (buf[i + half] - buf[i]);
        }
        buf.truncate(half);
    }
    let final_eval = buf[0];
    recorder
        .finish(&[final_eval], &mut prover_transcript)
        .unwrap();
    let prover_state = fingerprint(&mut prover_transcript);
    let narg = prover_transcript.finish();

    let mut verifier_transcript = verifier(b"recorder", &narg);
    verifier_transcript.public(&claimed_sum);
    let reduction = SumcheckVerifier::verify_compressed(
        &SumcheckClaim::new(num_vars, 1, claimed_sum),
        &mut verifier_transcript,
    )
    .unwrap();
    let output_claim: F = verifier_transcript.receive().unwrap();

    assert_eq!(reduction.value, final_eval);
    assert_eq!(output_claim, final_eval);
    assert_eq!(fingerprint(&mut verifier_transcript), prover_state);
    verifier_transcript.finish().unwrap();
}

/// A minimal dense multilinear [`ProveRounds`] member (HighToLow binding),
/// constructible with a prescribed total sum — the engine tests' stand-in for
/// a real kernel-backed batch member.
pub(crate) struct DenseMember {
    evals: Vec<F>,
    num_rounds: usize,
}

impl DenseMember {
    pub(crate) fn with_sum(num_rounds: usize, sum: F, seed: u64) -> Self {
        let size = 1u64 << num_rounds;
        let mut evals: Vec<F> = (0..size).map(|i| F::from_u64(seed + 31 * i + 11)).collect();
        let current: F = evals.iter().copied().sum();
        evals[0] += sum - current;
        Self { evals, num_rounds }
    }

    fn final_eval(&self) -> F {
        self.evals[0]
    }

    fn bind(&mut self, challenge: F) {
        let half = self.evals.len() / 2;
        for i in 0..half {
            self.evals[i] = self.evals[i] + challenge * (self.evals[i + half] - self.evals[i]);
        }
        self.evals.truncate(half);
    }
}

impl ProveRounds<F> for DenseMember {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        let half = self.evals.len() / 2;
        let eval_0: F = self.evals[..half].iter().copied().sum();
        let eval_1: F = self.evals[half..].iter().copied().sum();
        assert_eq!(eval_0 + eval_1, previous_claim);
        Ok(UnivariatePoly::new(vec![eval_0, eval_1 - eval_0]))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

/// A dense member exercising the fused contract for real: on each
/// `prove_round` the pending bind and the round evaluation happen in ONE pass
/// over the table (each pair is bound and immediately accumulated), never
/// leaving a fully bound intermediate table behind.
struct FusedDenseMember {
    evals: Vec<F>,
    num_rounds: usize,
}

impl ProveRounds<F> for FusedDenseMember {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            let half = self.evals.len() / 2;
            let quarter = half / 2;
            let mut eval_0 = F::from_u64(0);
            let mut eval_1 = F::from_u64(0);
            for i in 0..half {
                let bound = self.evals[i] + challenge * (self.evals[i + half] - self.evals[i]);
                self.evals[i] = bound;
                if i < quarter {
                    eval_0 += bound;
                } else {
                    eval_1 += bound;
                }
            }
            self.evals.truncate(half);
            assert_eq!(eval_0 + eval_1, previous_claim);
            return Ok(UnivariatePoly::new(vec![eval_0, eval_1 - eval_0]));
        }
        let half = self.evals.len() / 2;
        let eval_0: F = self.evals[..half].iter().copied().sum();
        let eval_1: F = self.evals[half..].iter().copied().sum();
        assert_eq!(eval_0 + eval_1, previous_claim);
        Ok(UnivariatePoly::new(vec![eval_0, eval_1 - eval_0]))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        let half = self.evals.len() / 2;
        for i in 0..half {
            self.evals[i] = self.evals[i] + bind * (self.evals[i + half] - self.evals[i]);
        }
        self.evals.truncate(half);
        Ok(())
    }
}

/// The `begin_batch` head both roles run: input claims absorbed by the
/// recorder (publicly for the clear one), then one coefficient per member.
fn batch_head<R: SumcheckRecorder<F>, C: Channel>(
    recorder: &mut R,
    input_claims: &[F],
    channel: &mut C,
) -> Vec<F> {
    recorder.absorb_input_claims(input_claims, channel);
    channel.challenges_small(input_claims.len())
}

fn prelude(members: &[(F, F, usize, usize)], max_num_vars: usize) -> BatchPrelude<F> {
    BatchPrelude::new(
        members
            .iter()
            .map(|&(input_claim, coefficient, rounds, offset)| BatchMember {
                input_claim,
                coefficient,
                rounds,
                offset,
            })
            .collect(),
        max_num_vars,
        1,
    )
}

/// A member fusing bind and eval into one table pass must be byte-identical
/// to the reference member that binds and evaluates separately.
#[test]
fn fused_bind_eval_member_byte_matches_separate_passes() {
    let num_rounds = 4;
    let sum = F::from_u64(90210);
    let prove = |member: &mut dyn ProveRounds<F>| {
        let mut transcript = prover(b"fused-vs-separate");
        let mut recorder = ClearSumcheckRecorder::<F>::new();
        let coefficients = batch_head(&mut recorder, &[sum], &mut transcript);
        let prelude = prelude(&[(sum, coefficients[0], num_rounds, 0)], num_rounds);
        let mut members: Vec<&mut dyn ProveRounds<F>> = vec![member];
        let proved = prove_batch(
            &prelude,
            &mut members,
            &mut SequentialRounds,
            &mut recorder,
            &mut transcript,
        )
        .unwrap();
        recorder
            .finish(&proved.member_claims, &mut transcript)
            .unwrap();
        let state = fingerprint(&mut transcript);
        (proved, transcript.finish(), state)
    };

    let mut separate = DenseMember::with_sum(num_rounds, sum, 41);
    let mut fused = FusedDenseMember {
        evals: separate.evals.clone(),
        num_rounds,
    };
    let (separate_proved, separate_narg, separate_state) = prove(&mut separate);
    let (fused_proved, fused_narg, fused_state) = prove(&mut fused);

    assert_eq!(separate_proved, fused_proved);
    assert_eq!(separate_narg, fused_narg);
    assert_eq!(separate.final_eval(), fused.evals[0]);
    assert_eq!(separate_state, fused_state);
}

fn pedersen_setup(capacity: u64) -> PedersenSetup<Bn254G1> {
    let generator = Bn254::g1_generator();
    let generators = (2..2 + capacity)
        .map(|k| generator.scalar_mul(&F::from_u64(k)))
        .collect();
    PedersenSetup::new(generators, generator.scalar_mul(&F::from_u64(99)))
}

/// A batch that declares `max_degree == 0` while having rounds to prove must
/// be rejected with `ZeroBatchDegree`, not proved (a degree-0 round polynomial
/// cannot carry a sumcheck round) and not panic.
#[test]
fn prove_batch_rejects_zero_max_degree() {
    let sum = F::from_u64(42);
    let mut member = DenseMember::with_sum(2, sum, 7);
    let prelude = BatchPrelude {
        members: vec![BatchMember {
            input_claim: sum,
            coefficient: F::from_u64(1),
            rounds: 2,
            offset: 0,
        }],
        claimed_sum: sum,
        max_num_vars: 2,
        max_degree: 0,
    };
    let mut members: Vec<&mut dyn ProveRounds<F>> = vec![&mut member];
    let mut transcript = prover(b"zero-degree-batch");
    let mut recorder = ClearSumcheckRecorder::<F>::new();
    let result = prove_batch(
        &prelude,
        &mut members,
        &mut SequentialRounds,
        &mut recorder,
        &mut transcript,
    );
    assert!(matches!(
        result,
        Err(SumcheckError::ZeroBatchDegree { max_num_vars: 2 })
    ));
}

/// Proves a 3-round and a 1-round member through the clear recorder and
/// replays the verifier's half of the protocol over the proof: head,
/// compressed rounds, then the sent member claims.
fn clear_batch_twin(
    session: &[u8],
    long: &mut DenseMember,
    short: &mut DenseMember,
    input_claims: [F; 2],
    short_offset: usize,
) -> ProvedBatch<F> {
    let mut prover_transcript = prover(session);
    let mut recorder = ClearSumcheckRecorder::<F>::new();
    let coefficients = batch_head(&mut recorder, &input_claims, &mut prover_transcript);
    let prelude = prelude(
        &[
            (input_claims[0], coefficients[0], 3, 0),
            (input_claims[1], coefficients[1], 1, short_offset),
        ],
        3,
    );
    let mut members: Vec<&mut dyn ProveRounds<F>> = vec![long, short];
    let proved = prove_batch(
        &prelude,
        &mut members,
        &mut SequentialRounds,
        &mut recorder,
        &mut prover_transcript,
    )
    .unwrap();
    recorder
        .finish(&proved.member_claims, &mut prover_transcript)
        .unwrap();
    let prover_state = fingerprint(&mut prover_transcript);
    let narg = prover_transcript.finish();

    let mut verifier_transcript = verifier(session, &narg);
    let verifier_coefficients = batch_head(
        &mut ClearSumcheckRecorder::<F>::new(),
        &input_claims,
        &mut verifier_transcript,
    );
    assert_eq!(verifier_coefficients, coefficients);
    // The padded claimed sum is position-independent: the short member is
    // scaled by 2^(3 - 1) whether it is head- or tail-aligned.
    let claimed_sum = verifier_coefficients[0] * input_claims[0]
        + verifier_coefficients[1] * input_claims[1].mul_pow_2(2);
    assert_eq!(claimed_sum, prelude.claimed_sum);
    let reduction = SumcheckVerifier::verify_compressed(
        &SumcheckClaim::new(3, 1, claimed_sum),
        &mut verifier_transcript,
    )
    .unwrap();
    let member_claims: Vec<F> = verifier_transcript.receive_n(2).unwrap();

    assert_eq!(member_claims, proved.member_claims);
    assert_eq!(reduction.value, proved.final_claim);
    assert_eq!(reduction.point.as_slice(), proved.challenges.as_slice());
    assert_eq!(fingerprint(&mut verifier_transcript), prover_state);
    verifier_transcript.finish().unwrap();
    proved
}

#[test]
fn prove_batch_clear_twin_matches_compressed_verifier_with_padding() {
    let input_claims = [F::from_u64(1234), F::from_u64(777)];
    let mut long = DenseMember::with_sum(3, input_claims[0], 5);
    let mut short = DenseMember::with_sum(1, input_claims[1], 91);

    let proved = clear_batch_twin(b"prove-batch-twin", &mut long, &mut short, input_claims, 2);
    assert_eq!(
        proved.member_claims,
        vec![long.final_eval(), short.final_eval()]
    );
}

/// A 1-round member at `offset: 0` is active in the batch's FIRST round and
/// then halves through the trailing dummy rounds. Its kernel must emit at the
/// dummy-round padding scale (the table sums to `2^(max − rounds) ·
/// input_claim`), so its final batch claim is the fully-bound value with the
/// padding halved back out.
#[test]
fn prove_batch_clear_twin_head_aligned_member() {
    let input_claims = [F::from_u64(1234), F::from_u64(777)];
    let mut long = DenseMember::with_sum(3, input_claims[0], 5);
    let mut short = DenseMember::with_sum(1, input_claims[1].mul_pow_2(2), 91);

    let proved = clear_batch_twin(
        b"prove-batch-head-twin",
        &mut long,
        &mut short,
        input_claims,
        0,
    );
    let quarter = F::from_u64(4).inverse().unwrap();
    assert_eq!(
        proved.member_claims,
        vec![long.final_eval(), short.final_eval() * quarter]
    );
}

/// The committed twin of the batched engine: no claim scalar reaches the
/// transcript on either side, the verifier reads back the prover's
/// challenges, and the retained witness opens every commitment it read.
#[test]
fn prove_batch_committed_twin_matches_committed_consistency() {
    let setup = pedersen_setup(4);

    let input_claims = [F::from_u64(4242), F::from_u64(1717)];
    let mut long = DenseMember::with_sum(3, input_claims[0], 23);
    let mut short = DenseMember::with_sum(1, input_claims[1], 57);

    let mut prover_transcript = prover(b"prove-batch-zk-twin");
    let mut recorder =
        CommittedSumcheckRecorder::<F, VC, _>::new(&setup, rand_core::OsRng).unwrap();
    let coefficients = batch_head(&mut recorder, &input_claims, &mut prover_transcript);
    let prelude = prelude(
        &[
            (input_claims[0], coefficients[0], 3, 0),
            (input_claims[1], coefficients[1], 1, 2),
        ],
        3,
    );
    let mut members: Vec<&mut dyn ProveRounds<F>> = vec![&mut long, &mut short];
    let proved = prove_batch(
        &prelude,
        &mut members,
        &mut SequentialRounds,
        &mut recorder,
        &mut prover_transcript,
    )
    .unwrap();
    let witness = recorder
        .finish(&proved.member_claims, &mut prover_transcript)
        .unwrap();
    let prover_state = fingerprint(&mut prover_transcript);
    let narg = prover_transcript.finish();

    let mut verifier_transcript = verifier(b"prove-batch-zk-twin", &narg);
    let verifier_coefficients: Vec<F> = verifier_transcript.challenges_small(2);
    assert_eq!(verifier_coefficients, coefficients);
    let (consistency, output_claims) = SumcheckVerifier::verify_committed::<F, _, Bn254G1>(
        SumcheckStatement::new(3, 1),
        1,
        &mut verifier_transcript,
    )
    .unwrap();

    assert_eq!(consistency.challenges(), proved.challenges);
    assert_eq!(fingerprint(&mut verifier_transcript), prover_state);
    verifier_transcript.finish().unwrap();

    assert_eq!(witness.round_coefficients.len(), 3);
    for ((commitment, coefficients), blinding) in consistency
        .round_commitments()
        .iter()
        .zip(&witness.round_coefficients)
        .zip(&witness.round_blindings)
    {
        assert!(VC::verify(&setup, commitment, coefficients, blinding));
    }
    for ((commitment, row), blinding) in output_claims
        .commitments
        .iter()
        .zip(&witness.output_claim_rows)
        .zip(&witness.output_claim_blindings)
    {
        assert!(VC::verify(&setup, commitment, row, blinding));
    }
    assert_eq!(witness.output_claim_rows.concat(), proved.member_claims);
}

fn uniskip_round() -> (UnivariatePoly<F>, F, usize, usize) {
    let degree = 3;
    let domain_size = 4;
    let round_poly = poly(&[3, 1, 4, 1]);
    let start = CenteredIntegerDomain::new(domain_size).start().unwrap();
    let input_claim = (start..start + domain_size as i64)
        .map(|point| round_poly.evaluate(F::from_i64(point)))
        .sum();
    (round_poly, input_claim, degree, domain_size)
}

/// The clear uni-skip prover against the verify choreography of
/// `jolt-verifier`'s `uniskip::verify_clear`: a full round over the centered
/// domain, its challenge, then the sent output claim.
#[test]
fn prove_uniskip_clear_twin_matches_uniskip_verify() {
    let (round_poly, input_claim, degree, domain_size) = uniskip_round();

    let mut prover_transcript = prover(b"uniskip-twin");
    let proved = prove_uniskip_clear(
        round_poly.clone(),
        input_claim,
        degree,
        domain_size,
        &mut prover_transcript,
    )
    .unwrap();
    let prover_state = fingerprint(&mut prover_transcript);
    let narg = prover_transcript.finish();

    let mut verifier_transcript = verifier(b"uniskip-twin", &narg);
    let reduction = SumcheckVerifier::verify(
        &SumcheckClaim::new(1, degree, input_claim),
        CenteredIntegerDomain::new(domain_size),
        &mut verifier_transcript,
    )
    .unwrap();
    let output_claim: F = verifier_transcript.receive().unwrap();

    assert_eq!(reduction.point.as_slice(), &[proved.challenge]);
    assert_eq!(reduction.value, round_poly.evaluate(proved.challenge));
    assert_eq!(output_claim, reduction.value);
    assert_eq!(proved.output_claim, reduction.value);
    assert_eq!(fingerprint(&mut verifier_transcript), prover_state);
    verifier_transcript.finish().unwrap();
}

#[test]
fn prove_uniskip_rejects_a_round_off_the_domain_sum_before_writing() {
    let (round_poly, input_claim, degree, domain_size) = uniskip_round();
    let mut transcript = prover(b"uniskip-bad-sum");
    let result = prove_uniskip_clear(
        round_poly,
        input_claim + F::from_u64(1),
        degree,
        domain_size,
        &mut transcript,
    );
    assert!(matches!(
        result,
        Err(SumcheckError::RoundCheckFailed { round: 0, .. })
    ));
    assert!(transcript.finish().is_empty());
}

/// The committed uni-skip twin: the verifier reads one round commitment and
/// one output-claim commitment, and the witness opens both.
#[test]
fn prove_uniskip_committed_twin_matches_committed_consistency() {
    let setup = pedersen_setup(4);
    let (round_poly, input_claim, degree, domain_size) = uniskip_round();

    let mut prover_transcript = prover(b"uniskip-zk-twin");
    let proved = prove_uniskip_committed::<F, VC, _, _>(
        round_poly.clone(),
        input_claim,
        degree,
        domain_size,
        &setup,
        rand_core::OsRng,
        &mut prover_transcript,
    )
    .unwrap();
    let prover_state = fingerprint(&mut prover_transcript);
    let narg = prover_transcript.finish();

    let mut verifier_transcript = verifier(b"uniskip-zk-twin", &narg);
    let (consistency, output_claims) = SumcheckVerifier::verify_committed::<F, _, Bn254G1>(
        SumcheckStatement::new(1, degree),
        1,
        &mut verifier_transcript,
    )
    .unwrap();

    assert_eq!(consistency.challenges(), vec![proved.challenge]);
    assert_eq!(fingerprint(&mut verifier_transcript), prover_state);
    verifier_transcript.finish().unwrap();

    let witness = &proved.witness;
    assert_eq!(
        witness.round_coefficients,
        vec![round_poly.coefficients().to_vec()]
    );
    assert!(VC::verify(
        &setup,
        &consistency.round_commitments()[0],
        &witness.round_coefficients[0],
        &witness.round_blindings[0],
    ));
    assert_eq!(witness.output_claim_rows, vec![vec![proved.output_claim]]);
    assert_eq!(proved.output_claim, round_poly.evaluate(proved.challenge));
    assert!(VC::verify(
        &setup,
        &output_claims.commitments[0],
        &witness.output_claim_rows[0],
        &witness.output_claim_blindings[0],
    ));
}
