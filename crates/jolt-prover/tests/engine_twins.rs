//! Twin-transcript engine locks: toy members driven through
//! `jolt_sumcheck::prove_batch` (and `prove_uniskip_clear`) against the
//! GENERATED `verify_clear` / `verify_zk` drivers and the shared uni-skip
//! `verify_clear` core. Each verifier reads the prover's argument string to
//! EOF and must land on the prover's sponge state. This pins
//! the prove-side engine to the verifier independently of any real stage.
//! Protocol-agnostic: the locks exercise the shared engine/driver seam and
//! run under both the Dory and Akita builds.

#![expect(clippy::unwrap_used, reason = "test crate")]

use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafDimensions;
use jolt_claims::protocols::jolt::relations::instruction::InstructionReadRafInputClaims;
use jolt_claims::protocols::jolt::relations::registers::RegistersValEvaluationInputClaims;
use jolt_crypto::{Bn254, Bn254G1, JoltGroup, Pedersen, PedersenSetup};
use jolt_field::{Fr, JoltField, Ring};
use jolt_poly::{UnivariatePoly, UnivariatePolynomial};
use jolt_sumcheck::{
    prove_batch, prove_uniskip_clear, CenteredIntegerDomain, ClearSumcheckRecorder,
    CommittedSumcheckRecorder, ProveRounds, SequentialRounds, SumcheckClaim, SumcheckDomain,
    SumcheckError, SumcheckRecorder, SumcheckVerifier,
};
use jolt_transcript::{
    Blake2b512, Channel as _, ProtocolId, ProverTranscript, Sponge, VerifierTranscript,
};
use jolt_verifier::stages::relations::{ConcreteSumcheck as _, SumcheckBatch};
use jolt_verifier::stages::stage5::{InstructionReadRaf, RegistersValEvaluation};
use jolt_verifier::stages::uniskip::{self, UniskipParams};

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

/// The ZK twin's vector-commitment capacity: one output-claim row.
const ROW_LEN: usize = 8;

fn pedersen_setup(capacity: u64) -> PedersenSetup<Bn254G1> {
    let generator = Bn254::g1_generator();
    let generators = (2..2 + capacity)
        .map(|k| generator.scalar_mul(&Fr::from_u64(k)))
        .collect();
    PedersenSetup::new(generators, generator.scalar_mul(&Fr::from_u64(99)))
}

fn prover_transcript(name: &str) -> ProverTranscript<Blake2b512> {
    ProverTranscript::new(&ProtocolId::new::<Blake2b512>(name), b"")
}

fn verifier_transcript<'a>(name: &str, narg: &'a [u8]) -> VerifierTranscript<'a, Blake2b512> {
    VerifierTranscript::new(&ProtocolId::new::<Blake2b512>(name), b"", narg)
}

/// The verifier consumed the whole argument string and reached the prover's
/// sponge state.
fn assert_twins<H: Sponge>(
    prover: &mut ProverTranscript<H>,
    mut verifier: VerifierTranscript<'_, H>,
) {
    assert_eq!(
        verifier.challenge_bytes::<32>(),
        prover.challenge_bytes::<32>()
    );
    verifier.finish().unwrap();
}

/// Synthetic stand-ins for a stage's flattened output-claim values.
fn synthetic_output_values() -> Vec<Fr> {
    vec![Fr::from_u64(11), Fr::from_u64(22), Fr::from_u64(33)]
}

#[test]
fn clear_engine_twin_matches_generated_verify_clear() {
    // Prover: draw → sums → begin_batch(clear) → prove_batch → finish.
    let sumchecks = fixture();
    let inputs = inputs();
    let mut prover_transcript = prover_transcript("engine-twin");
    let prover_challenges = sumchecks.draw_challenges(&mut prover_transcript).unwrap();

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
            &mut prover_transcript,
        )
        .unwrap();
    let mut members: Vec<&mut dyn ProveRounds<Fr>> =
        vec![&mut instruction_member, &mut registers_member];
    let proved = prove_batch(
        &batch,
        &mut members,
        &mut SequentialRounds,
        &mut recorder,
        &mut prover_transcript,
    )
    .unwrap();
    let output_values = synthetic_output_values();
    recorder
        .finish(&output_values, &mut prover_transcript)
        .unwrap();

    // Verifier: draw → begin_batch → compressed rounds → output claims (the
    // low-level clear path the composed `verify_clear` wraps).
    let narg = prover_transcript.narg().to_vec();
    let mut verifier_transcript = verifier_transcript("engine-twin", &narg);
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
    let reduction = SumcheckVerifier::verify_compressed(
        &SumcheckClaim::new(
            verifier_batch.max_num_vars,
            verifier_batch.max_degree,
            verifier_batch.claimed_sum,
        ),
        &mut verifier_transcript,
    )
    .unwrap();
    let received_values: Vec<Fr> = verifier_transcript.receive_n(output_values.len()).unwrap();

    assert_eq!(reduction.value, proved.final_claim);
    assert_eq!(reduction.point.as_slice(), proved.challenges.as_slice());
    assert_eq!(received_values, output_values);
    assert_eq!(verifier_coefficients, prover_coefficients);
    assert_twins(&mut prover_transcript, verifier_transcript);
}

#[test]
fn committed_engine_twin_matches_generated_verify_zk() {
    type VC = Pedersen<Bn254G1>;
    let setup = pedersen_setup(ROW_LEN as u64);

    // Prover: draw → sums → begin_batch(committed; claim absorbs no-op) →
    // prove_batch → finish (output-claim row commitments absorbed).
    let sumchecks = fixture();
    let inputs = inputs();
    let mut prover_transcript = prover_transcript("engine-zk-twin");
    let prover_challenges = sumchecks.draw_challenges(&mut prover_transcript).unwrap();

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

    let mut recorder =
        CommittedSumcheckRecorder::<Fr, VC, _>::new(&setup, rand_core::OsRng).unwrap();
    let (batch, prover_coefficients) = sumchecks
        .begin_batch(
            &inputs,
            &prover_challenges,
            &mut recorder,
            &mut prover_transcript,
        )
        .unwrap();
    let mut members: Vec<&mut dyn ProveRounds<Fr>> =
        vec![&mut instruction_member, &mut registers_member];
    let proved = prove_batch(
        &batch,
        &mut members,
        &mut SequentialRounds,
        &mut recorder,
        &mut prover_transcript,
    )
    .unwrap();
    // Synthetic values in the shape the verifier commits: one per committed
    // cell of the points derived at the batch point.
    // The registers member reads its read-write point: the 7 register
    // address bits then the 3 cycle bits.
    let mut input_points = sumchecks.empty_input_points();
    input_points.registers_val_evaluation.registers_val = (0..10).map(Fr::from_u64).collect();
    let points = sumchecks
        .derive_opening_points(&proved.challenges, &input_points)
        .unwrap();
    let layout = TwinFixtureSumchecks::committed_claim_layout(
        &points,
        &TwinFixtureSumchecks::claim_routes(&points).unwrap(),
    );
    let output_values: Vec<Fr> = (0..layout.ids.len() as u64).map(Fr::from_u64).collect();
    let witness = recorder
        .finish(&output_values, &mut prover_transcript)
        .unwrap();

    // Verifier: draw → generated verify_zk (coefficient draws, committed
    // rounds, derived points, output-claim row commitments).
    let narg = prover_transcript.narg().to_vec();
    let mut verifier_transcript = verifier_transcript("engine-zk-twin", &narg);
    let _verifier_challenges = sumchecks.draw_challenges(&mut verifier_transcript).unwrap();
    let verified = sumchecks
        .verify_zk::<Bn254G1, _>(ROW_LEN, &input_points, &mut verifier_transcript)
        .unwrap();
    assert_eq!(verified.output_points, points);
    let (consistency, output_commitments) =
        (verified.consistency, verified.output_claims.commitments);

    assert_eq!(consistency.challenges(), proved.challenges);
    assert_eq!(
        consistency.batching_coefficients,
        vec![
            prover_coefficients.instruction_read_raf,
            prover_coefficients.registers_val_evaluation,
        ],
    );
    assert_eq!(consistency.max_num_vars, batch.max_num_vars);
    assert_eq!(consistency.max_degree, batch.max_degree);
    assert_eq!(
        output_commitments.commitments.len(),
        witness.output_claim_rows.len()
    );
    assert_twins(&mut prover_transcript, verifier_transcript);
}

/// Twin-transcript lock for the shared uni-skip verification core: a clear
/// uni-skip round proved through `jolt_sumcheck::prove_uniskip_clear` must be
/// accepted by `jolt-verifier`'s `uniskip::verify_clear`, which reads the
/// round polynomial and output claim and lands on the prover's state.
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
    let coefficients: Vec<Fr> = CenteredIntegerDomain::new(domain_size)
        .round_sum_coefficients(UnivariatePolynomial::degree(&poly))
        .unwrap();
    let input_claim = poly
        .coefficients()
        .iter()
        .zip(&coefficients)
        .fold(Fr::from_u64(0), |sum, (coefficient, weight)| {
            sum + *coefficient * *weight
        });

    let mut prover_transcript = prover_transcript("uniskip-stage-twin");
    let proved = prove_uniskip_clear::<Fr, _>(
        poly,
        input_claim,
        degree,
        domain_size,
        &mut prover_transcript,
    )
    .unwrap();

    let narg = prover_transcript.narg().to_vec();
    let mut verifier_transcript = verifier_transcript("uniskip-stage-twin", &narg);
    let verified = uniskip::verify_clear(&params, input_claim, &mut verifier_transcript).unwrap();

    assert_eq!(verified.challenge, proved.challenge);
    assert_eq!(verified.output_claim, proved.output_claim);
    assert_twins(&mut prover_transcript, verifier_transcript);
}
