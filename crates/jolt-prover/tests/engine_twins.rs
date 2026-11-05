//! Twin-transcript engine locks: toy members driven through
//! `jolt_sumcheck::prove_batch` (and `prove_uniskip_clear`) against the
//! GENERATED `verify_clear` / `verify_masked` drivers and the shared uni-skip
//! `verify_clear` core, asserting byte-identical transcript states. This pins
//! the prove-side engine to the verifier independently of any real stage.
//! Protocol-agnostic: the locks exercise the shared engine/driver seam and
//! run under both the Dory and Akita builds.

#![expect(clippy::unwrap_used, reason = "test crate")]

use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafDimensions;
use jolt_claims::protocols::jolt::relations::instruction::InstructionReadRafInputClaims;
use jolt_claims::protocols::jolt::relations::registers::RegistersValEvaluationInputClaims;
use jolt_crypto::Bn254G1;
#[cfg(feature = "zk")]
use jolt_crypto::{Bn254, JoltGroup, Pedersen, PedersenSetup};
use jolt_field::{Fr, JoltField, Ring};
use jolt_poly::{UnivariatePoly, UnivariatePolynomial};
use jolt_sumcheck::{
    prove_batch, prove_uniskip_clear, CenteredIntegerDomain, ClearRound, ClearSumcheckRecorder,
    ProveRounds, SequentialRounds, SumcheckDomain, SumcheckError, SumcheckProof, SumcheckRecorder,
};
use jolt_transcript::{prover_transcript, Keccak, Transcript};
#[cfg(feature = "zk")]
use jolt_transcript::{Shake128, ShapeTranscript};
use jolt_verifier::stages::relations::{ConcreteSumcheck as _, SumcheckBatch};
use jolt_verifier::stages::stage5::{InstructionReadRaf, RegistersValEvaluation};
use jolt_verifier::stages::uniskip::{self, UniskipParams};
#[cfg(feature = "zk")]
use jolt_zk::blindfold::BlindFold;
#[cfg(feature = "zk")]
use jolt_zk::masking::{
    CommittedPadsView, MaskedMessages, MaskedReader, MaskedSession, MaskingRecorder, PadLayout,
    ZkBackend,
};
#[cfg(feature = "zk")]
use rand_core::OsRng;

#[derive(SumcheckBatch)]
struct TwinFixtureSumchecks<F: JoltField> {
    instruction_read_raf: InstructionReadRaf<F>,
    registers_val_evaluation: RegistersValEvaluation<F>,
}

/// Small geometry so a dense toy prover is feasible: the instruction
/// member gets 8 rounds (6 address + 2 cycle), the registers member 3 —
/// so the generated-driver twins also exercise front-loaded padding.
fn fixture() -> TwinFixtureSumchecks<Fr> {
    TwinFixtureSumchecks {
        instruction_read_raf: InstructionReadRaf::new(
            InstructionReadRafDimensions::try_from((2, 6, 2)).unwrap(),
        ),
        registers_val_evaluation: RegistersValEvaluation::new(TraceDimensions::new(3)),
    }
}

fn inputs() -> TwinFixtureInputClaims<Fr> {
    let fr = Fr::from_u64;
    TwinFixtureInputClaims {
        instruction_read_raf: InstructionReadRafInputClaims {
            lookup_output: fr(2),
            left_lookup_operand: fr(3),
            right_lookup_operand: fr(5),
        },
        registers_val_evaluation: RegistersValEvaluationInputClaims {
            registers_val: fr(7),
        },
    }
}

/// A dense multilinear toy batch member with a prescribed total sum
/// (HighToLow binding) — degree 1, which every relation's degree bound
/// admits.
struct DenseMember {
    evals: Vec<Fr>,
    num_rounds: usize,
}

impl DenseMember {
    fn with_sum(num_rounds: usize, sum: Fr, seed: u64) -> Self {
        let size = 1u64 << num_rounds;
        let mut evals: Vec<Fr> = (0..size)
            .map(|i| Fr::from_u64(seed + 31 * i + 11))
            .collect();
        let current: Fr = evals.iter().copied().sum();
        evals[0] += sum - current;
        Self { evals, num_rounds }
    }

    fn bind(&mut self, challenge: Fr) {
        let half = self.evals.len() / 2;
        for i in 0..half {
            self.evals[i] = self.evals[i] + challenge * (self.evals[i + half] - self.evals[i]);
        }
        self.evals.truncate(half);
    }
}

impl ProveRounds<Fr> for DenseMember {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        _round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        let half = self.evals.len() / 2;
        let eval_0: Fr = self.evals[..half].iter().copied().sum();
        let eval_1: Fr = self.evals[half..].iter().copied().sum();
        assert_eq!(eval_0 + eval_1, previous_claim);
        Ok(UnivariatePoly::new(vec![eval_0, eval_1 - eval_0]))
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.bind(bind);
        Ok(())
    }
}

#[cfg(feature = "zk")]
fn pedersen_setup(capacity: u64) -> PedersenSetup<Bn254G1> {
    let generator = Bn254::g1_generator();
    let generators = (2..2 + capacity)
        .map(|k| generator.scalar_mul(&Fr::from_u64(k)))
        .collect();
    PedersenSetup::new(generators, generator.scalar_mul(&Fr::from_u64(99)))
}

fn synthetic_output_values() -> Vec<Fr> {
    vec![Fr::from_u64(11), Fr::from_u64(22), Fr::from_u64(33)]
}

#[test]
fn clear_engine_twin_matches_generated_verify_clear() {
    let sumchecks = fixture();
    let inputs = inputs();
    let mut prover_state =
        prover_transcript::<Keccak>(&jolt_transcript::protocol_tag(b"engine-twin"), b"");
    let prover_challenges = sumchecks.draw_challenges(&mut prover_state).unwrap();

    let instruction_sum = sumchecks
        .instruction_read_raf
        .input_claim(
            &inputs.instruction_read_raf,
            &prover_challenges.instruction_read_raf,
        )
        .unwrap();
    let registers_sum = sumchecks
        .registers_val_evaluation
        .input_claim(
            &inputs.registers_val_evaluation,
            &prover_challenges.registers_val_evaluation,
        )
        .unwrap();
    let mut instruction_member =
        DenseMember::with_sum(sumchecks.instruction_read_raf.rounds(), instruction_sum, 5);
    let mut registers_member = DenseMember::with_sum(
        sumchecks.registers_val_evaluation.rounds(),
        registers_sum,
        91,
    );

    let mut recorder = ClearSumcheckRecorder::<Fr>::new();
    let (batch, prover_coefficients) = sumchecks
        .begin_batch(
            &inputs,
            &prover_challenges,
            &mut recorder,
            &mut prover_state,
        )
        .unwrap();
    let mut members: Vec<&mut dyn ProveRounds<Fr>> =
        vec![&mut instruction_member, &mut registers_member];
    let proved = prove_batch(
        &batch,
        &mut members,
        &mut SequentialRounds,
        &mut recorder,
        &mut prover_state,
    )
    .unwrap();
    let output_values = synthetic_output_values();
    let recorded = recorder.finish(&output_values, &mut prover_state).unwrap();

    // Verifier: draw → begin_batch → verify_compressed_scheduled → output-claim
    // absorbs (the low-level clear path the composed `verify_clear` wraps).
    let mut verifier_transcript =
        prover_transcript::<Keccak>(&jolt_transcript::protocol_tag(b"engine-twin"), b"");
    let verifier_challenges = sumchecks.draw_challenges(&mut verifier_transcript).unwrap();
    let mut verifier_recorder = ClearSumcheckRecorder::<Fr>::new();
    let (verifier_batch, verifier_coefficients) = sumchecks
        .begin_batch(
            &inputs,
            &verifier_challenges,
            &mut verifier_recorder,
            &mut verifier_transcript,
        )
        .unwrap();
    let reduction = SumcheckProof::<Fr, Bn254G1>::from(recorded.proof)
        .verify_compressed_scheduled(
            &verifier_batch.degrees,
            verifier_batch.claimed_sum,
            &mut verifier_transcript,
        )
        .unwrap();
    for value in &output_values {
        verifier_transcript.append(value);
    }

    assert_eq!(reduction.value, proved.final_claim);
    assert_eq!(reduction.point.as_slice(), proved.challenges.as_slice());
    assert_eq!(verifier_coefficients, prover_coefficients);
    assert_eq!(
        prover_state.verifier_message::<[u8; 32]>(),
        verifier_transcript.verifier_message::<[u8; 32]>()
    );
}

/// The masked twin's vector-commitment capacity: the pad-row length.
#[cfg(feature = "zk")]
const ROW_LEN: usize = 8;

/// The masked twin: the prover commits pads on the layout the masked verifier
/// counts (a counting replay over a shape transcript and a zero tape, as
/// `pad_layout` derives it), records the batch through `MaskingRecorder`, and
/// the generated `verify_masked` must read the whole masked tape, use every
/// pad, reach the prover's sponge state, and record values that decode to
/// the prover's.
#[cfg(feature = "zk")]
#[test]
fn masked_engine_twin_matches_generated_verify_masked() {
    type VC = Pedersen<Bn254G1>;
    let setup = pedersen_setup(ROW_LEN as u64);
    let sumchecks = fixture();
    let inputs = inputs();
    // The registers member reads its read-write point: the 7 register
    // address bits then the 3 cycle bits.
    let mut input_points = sumchecks.empty_input_points();
    input_points.registers_val_evaluation.registers_val = (0..10).map(Fr::from_u64).collect();

    let mut counting_transcript = ShapeTranscript::<Shake128>::new();
    let mut counter = MaskedSession::new(CommittedPadsView::counting(), MaskedReader::zeros());
    let _ = sumchecks.draw_challenges(&mut counting_transcript).unwrap();
    let shape = sumchecks
        .verify_masked(
            &input_points,
            &mut counter.channel(&mut counting_transcript),
        )
        .unwrap();
    let layout = PadLayout::new(counter.pads.used(), 0).unwrap();

    let session = jolt_transcript::protocol_tag(b"engine-masked-twin");
    let mut prover_state = prover_transcript::<Keccak>(&session, b"");
    let (mut pads, pad_commitment) =
        BlindFold::<VC>::commit_pads(&setup, &layout, &mut prover_state, &mut OsRng).unwrap();
    let mut tape = MaskedMessages::default();
    let prover_challenges = sumchecks.draw_challenges(&mut prover_state).unwrap();
    let instruction_sum = sumchecks
        .instruction_read_raf
        .input_claim(
            &inputs.instruction_read_raf,
            &prover_challenges.instruction_read_raf,
        )
        .unwrap();
    let registers_sum = sumchecks
        .registers_val_evaluation
        .input_claim(
            &inputs.registers_val_evaluation,
            &prover_challenges.registers_val_evaluation,
        )
        .unwrap();
    let mut instruction_member =
        DenseMember::with_sum(sumchecks.instruction_read_raf.rounds(), instruction_sum, 23);
    let mut registers_member = DenseMember::with_sum(
        sumchecks.registers_val_evaluation.rounds(),
        registers_sum,
        57,
    );
    let mut recorder = MaskingRecorder::new(&mut pads, &mut tape);
    let (batch, prover_coefficients) = sumchecks
        .begin_batch(
            &inputs,
            &prover_challenges,
            &mut recorder,
            &mut prover_state,
        )
        .unwrap();
    let mut members: Vec<&mut dyn ProveRounds<Fr>> =
        vec![&mut instruction_member, &mut registers_member];
    let proved = prove_batch(
        &batch,
        &mut members,
        &mut SequentialRounds,
        &mut recorder,
        &mut prover_state,
    )
    .unwrap();
    let output_values: Vec<Fr> = (0..shape.output_claims.received.len() as u64)
        .map(|k| Fr::from_u64(40 + k))
        .collect();
    let _ = recorder.finish(&output_values, &mut prover_state).unwrap();
    pads.finish().unwrap();

    let mut verifier_transcript = prover_transcript::<Keccak>(&session, b"");
    let view = BlindFold::<VC>::receive_pad_commitment(
        &setup,
        &layout,
        pad_commitment,
        &mut verifier_transcript,
    )
    .unwrap();
    let mut verifier = MaskedSession::new(view, MaskedReader::new(&tape));
    let _ = sumchecks.draw_challenges(&mut verifier_transcript).unwrap();
    let verified = sumchecks
        .verify_masked(
            &input_points,
            &mut verifier.channel(&mut verifier_transcript),
        )
        .unwrap();
    verifier.finish().unwrap();

    let points = sumchecks
        .derive_opening_points(&proved.challenges, &input_points)
        .unwrap();
    assert_eq!(verified.output_points, points);
    assert_eq!(verified.rounds.challenges(), proved.challenges);
    assert_eq!(
        verified.rounds.batching_coefficients,
        vec![
            prover_coefficients.instruction_read_raf,
            prover_coefficients.registers_val_evaluation,
        ],
    );
    assert_eq!(verified.rounds.degrees, batch.degrees);
    let decoded: Vec<Fr> = verified
        .output_claims
        .received
        .iter()
        .map(|(_, operand)| pads.decode(operand.masked, operand.pad).unwrap())
        .collect();
    assert_eq!(decoded, output_values);
    assert_eq!(
        prover_state.verifier_message::<[u8; 32]>(),
        verifier_transcript.verifier_message::<[u8; 32]>()
    );
}

/// Twin-transcript lock for the shared uni-skip verification core: a clear
/// uni-skip round proved through `jolt_sumcheck::prove_uniskip_clear` must be
/// accepted by `jolt-verifier`'s `uniskip::verify_clear` with byte-identical
/// transcript states (round proof, output-claim absorb, reduction challenge).
#[test]
fn uniskip_prover_twin_matches_uniskip_verify_clear() {
    let params = UniskipParams::spartan_outer();
    let degree = params.degree();
    let domain_size = params.domain_size();
    let poly = UnivariatePoly::new(
        (0..=degree as u64)
            .map(|k| Fr::from_u64(3 * k + 2))
            .collect(),
    );
    let coefficients = CenteredIntegerDomain::new(domain_size)
        .round_sum_coefficients(UnivariatePolynomial::degree(&poly))
        .unwrap();
    let input_claim = <UnivariatePoly<Fr> as ClearRound<Fr>>::coefficient_linear_combination(
        &poly,
        &coefficients,
    );

    let mut prover_state =
        prover_transcript::<Keccak>(&jolt_transcript::protocol_tag(b"uniskip-stage-twin"), b"");
    let proved =
        prove_uniskip_clear::<Fr, _>(poly, input_claim, degree, domain_size, &mut prover_state)
            .unwrap();

    let mut verifier_transcript =
        prover_transcript::<Keccak>(&jolt_transcript::protocol_tag(b"uniskip-stage-twin"), b"");
    let challenge = uniskip::verify_clear(
        &SumcheckProof::<Fr, Bn254G1>::from(proved.proof),
        &params,
        input_claim,
        proved.output_claim,
        &mut verifier_transcript,
    )
    .unwrap();

    assert_eq!(challenge, proved.challenge);
    assert_eq!(
        prover_state.verifier_message::<[u8; 32]>(),
        verifier_transcript.verifier_message::<[u8; 32]>()
    );
}
