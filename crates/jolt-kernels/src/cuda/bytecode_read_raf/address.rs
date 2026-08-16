use jolt_claims::protocols::jolt::geometry::bytecode::{
    read_raf_stage_values, BytecodeReadRafStageValueInputs,
};
use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage6a::bytecode_read_raf::{
    BytecodeReadRafAddressPhase, BytecodeReadRafAddressPhaseInputClaims,
    BytecodeReadRafAddressPhaseOutputClaims,
};
use jolt_witness::{collect_bundles, JoltWitnessPlane};

use super::pushforward::{DeviceBytecodePushforward, PushforwardInputs, STAGES};
use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::one_hot_fold::DeviceOneHotColumns;
use crate::cuda::{require_context, CudaBackend};
use crate::reference::bytecode_read_raf::BytecodeReadRafWitness;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

pub struct BytecodeReadRafAddressKernel<F: Field> {
    context: &'static CudaKernelContext,
    relation: BytecodeReadRafAddressPhase<F>,
    state: DeviceBytecodePushforward,
    last_round_poly: Option<UnivariatePoly<F>>,
    intermediate: Option<F>,
    val_stages: Vec<F>,
    rounds_bound: usize,
}

impl<F: Field> BytecodeReadRafAddressKernel<F> {
    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        let failed = || SumcheckError::MissingEvaluationSource {
            kind: "cuda bytecode read-RAF address-phase bind",
        };
        self.state
            .bind(self.context, challenge)
            .map_err(|_| failed())?;
        self.rounds_bound += 1;
        let final_round = self.rounds_bound == self.relation.symbolic().rounds();
        if let Some(poly) = self.last_round_poly.take() {
            if final_round {
                self.intermediate = Some(poly.evaluate(challenge));
            }
        }
        if final_round && self.relation.committed_program() {
            self.val_stages = self.state.val_claims().map_err(|_| failed())?;
        }
        Ok(())
    }
}

impl<F: Field> ProveRounds<F> for BytecodeReadRafAddressKernel<F> {
    fn num_rounds(&self) -> usize {
        self.relation.symbolic().rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge)?;
        }
        let (at_zero, at_two) = self.state.round_lanes(self.context).map_err(|_| {
            SumcheckError::MissingEvaluationSource {
                kind: "cuda bytecode read-RAF address-phase round",
            }
        })?;
        let poly = UnivariatePoly::from_evals_and_hint(previous_claim, &[at_zero, at_two]);
        self.last_round_poly = Some(poly.clone());
        Ok(poly)
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind)
    }
}

impl<F: Field> SumcheckKernel<F> for BytecodeReadRafAddressKernel<F> {
    type Relation = BytecodeReadRafAddressPhase<F>;

    fn output_claims(
        &mut self,
        _inputs: &BytecodeReadRafAddressPhaseInputClaims<F>,
    ) -> Result<BytecodeReadRafAddressPhaseOutputClaims<F>, SumcheckKernelError<F>> {
        let remaining = self.relation.symbolic().rounds() - self.rounds_bound;
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        if self.state.len() != 1 {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "CUDA bytecode read-RAF address phase counted every round but its tables \
                         are not fully bound",
            });
        }
        let intermediate = self
            .intermediate
            .ok_or(SumcheckKernelError::InvariantViolation {
                reason: "CUDA bytecode read-RAF address phase never staged its intermediate claim",
            })?;
        Ok(BytecodeReadRafAddressPhaseOutputClaims {
            intermediate,
            val_stages: self.val_stages.clone(),
        })
    }
}

impl<F: Field> PrepareKernel<F, BytecodeReadRafAddressPhase<F>> for CudaBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, BytecodeReadRafAddressPhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BytecodeReadRafAddressPhase<F>>>, KernelError<F>>
    {
        let context = require_context()?;
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let addresses = 1usize << dimensions.log_k();
        let cycles = 1usize << dimensions.log_t();
        if relation.register_read_write_point().len() < REGISTER_ADDRESS_BITS
            || relation.register_val_evaluation_point().len() < REGISTER_ADDRESS_BITS
        {
            return Err(KernelError::InvariantViolation {
                reason: "a bytecode read-RAF register point is shorter than its address prefix",
            });
        }
        if relation
            .stage_cycle_points()
            .iter()
            .any(|point| point.len() != dimensions.log_t())
        {
            return Err(KernelError::InvariantViolation {
                reason: "a bytecode read-RAF stage cycle point has the wrong variable count",
            });
        }

        let program = witness.program_preprocessing();
        let stage_gammas = inputs.challenges.stage_gamma_powers();
        let stage_values = read_raf_stage_values(BytecodeReadRafStageValueInputs {
            bytecode: &program.bytecode.bytecode,
            register_read_write_point: &relation.register_read_write_point()
                [..REGISTER_ADDRESS_BITS],
            register_val_evaluation_point: &relation.register_val_evaluation_point()
                [..REGISTER_ADDRESS_BITS],
            stage1_gammas: &stage_gammas[0],
            stage2_gammas: &stage_gammas[1],
            stage3_gammas: &stage_gammas[2],
            stage4_gammas: &stage_gammas[3],
            stage5_gammas: &stage_gammas[4],
        });
        if stage_values.len() != addresses {
            return Err(KernelError::TableSizeMismatch {
                table: "bytecode stage values".to_owned(),
                expected: addresses,
                got: stage_values.len(),
            });
        }

        let rows = collect_bundles::<BytecodeReadRafWitness>(witness, cycles)?;
        let mut pcs = Vec::with_capacity(cycles);
        for row in &rows {
            let pc = u32::try_from(row.bytecode_pc.0).map_err(|_| KernelError::Unsupported {
                reason: "the CUDA bytecode read-RAF address phase packs each PC into one 32-bit \
                         word",
            })?;
            if row.bytecode_pc.0 >= addresses {
                return Err(KernelError::InvariantViolation {
                    reason: "a bytecode PC escapes the padded bytecode domain",
                });
            }
            pcs.push(pc);
        }
        let entry_trace_index = rows
            .first()
            .ok_or(KernelError::InvariantViolation {
                reason: "the bytecode read-RAF address phase needs at least one cycle",
            })?
            .bytecode_pc
            .0;
        drop(rows);

        let columns = DeviceOneHotColumns::new(
            context,
            &[],
            &[],
            &pcs,
            [0, 0, 1],
            dimensions.log_k(),
            cycles,
        )?;
        drop(pcs);

        let state = DeviceBytecodePushforward::new(
            context,
            &columns,
            PushforwardInputs {
                stage_cycle_points: relation.stage_cycle_points(),
                stage_values: &stage_values,
                entry_trace_index,
                entry_expected_index: relation.entry_bytecode_index(),
                gamma: inputs.challenges.gamma,
            },
        )?;
        drop(columns);

        Ok(Box::new(BytecodeReadRafAddressKernel {
            context,
            relation: relation.clone(),
            state,
            last_round_poly: None,
            intermediate: None,
            val_stages: Vec::with_capacity(STAGES),
            rounds_bound: 0,
        }))
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations and fixture errors fail loudly"
)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Arc;

    use ark_bn254::Fr as LegacyFr;
    use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
    use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltRelationId};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_prover_legacy::field::JoltField as LegacyJoltField;
    use jolt_prover_legacy::poly::commitment::dory::DoryCommitmentScheme;
    use jolt_prover_legacy::poly::opening_proof::{
        OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId, BIG_ENDIAN,
    };
    use jolt_prover_legacy::subprotocols::sumcheck_prover::SumcheckInstanceProver;
    use jolt_prover_legacy::transcripts::{Blake2bTranscript, Transcript};
    use jolt_prover_legacy::zkvm::bytecode::read_raf_checking::{
        BytecodeReadRafAddressSumcheckProver, BytecodeReadRafSumcheckParams,
    };
    use jolt_prover_legacy::zkvm::config::OneHotParams;
    use jolt_prover_legacy::zkvm::instruction::{CircuitFlags, InstructionFlags};
    use jolt_prover_legacy::zkvm::lookup_table::NUM_LOOKUP_TABLES;
    use jolt_prover_legacy::zkvm::program::{FullProgramPreprocessing, ProgramPreprocessing};
    use jolt_prover_legacy::zkvm::ram::RAMPreprocessing as LegacyRamPreprocessing;
    use jolt_prover_legacy::zkvm::witness::VirtualPolynomial;
    use jolt_verifier::stages::formula_dimensions_from_parts;
    use jolt_verifier::stages::stage6a::bytecode_read_raf::{
        BytecodeReadRafAddressPhase, BytecodeReadRafAddressPhaseInputClaims, BytecodeStagePoints,
    };
    use jolt_witness::{collect_bundles, JoltWitnessPlane};
    use strum::IntoEnumIterator;

    use crate::cuda::booleanity::address::legacy_fixture::{
        slot_for_cycle, with_legacy_witness, LegacyFixture, SLOTS,
    };
    use crate::cuda::common::context::shared_context;
    use crate::cuda::CudaBackend;
    use crate::reference::bytecode_read_raf::BytecodeReadRafWitness;
    use crate::{PrepareKernel, ProofSession, ProverInputs};

    const LOG_T: usize = 8;

    const RAM_K: usize = 1 << 9;

    const SEED: u64 = 20_260_816;

    const SAMPLE_POINTS: usize = 4;

    const STAGE_COUNT: usize = 5;

    fn one_hot() -> JoltOneHotConfig {
        JoltOneHotConfig {
            log_k_chunk: 4,
            lookups_ra_virtual_log_k_chunk: 16,
        }
    }

    fn to_fr(value: LegacyFr) -> Fr {
        Fr::from(value)
    }

    fn challenge_to_fr(challenge: <LegacyFr as LegacyJoltField>::Challenge) -> Fr {
        to_fr(<LegacyFr as From<_>>::from(challenge))
    }

    type Challenge = <LegacyFr as LegacyJoltField>::Challenge;

    struct LegacyRun {
        messages: Vec<[Fr; SAMPLE_POINTS]>,
        challenges: Vec<Fr>,
        input_claim: Fr,
        output_claim: Fr,
        stage_cycle_points: [Vec<Fr>; STAGE_COUNT],
        register_read_write_point: Vec<Fr>,
        register_val_evaluation_point: Vec<Fr>,
        entry_bytecode_index: usize,
        gammas: [Fr; 6],
        rounds: usize,
    }

    fn synthetic_accumulator(
        transcript: &mut Blake2bTranscript,
    ) -> (
        ProverOpeningAccumulator<LegacyFr>,
        [Vec<Challenge>; STAGE_COUNT],
        Vec<Challenge>,
        Vec<Challenge>,
    ) {
        let mut accumulator = ProverOpeningAccumulator::new(LOG_T);
        let cycle_point = |transcript: &mut Blake2bTranscript| -> Vec<Challenge> {
            transcript.challenge_vector_optimized::<LegacyFr>(LOG_T)
        };
        let register_point = |transcript: &mut Blake2bTranscript| -> Vec<Challenge> {
            transcript.challenge_vector_optimized::<LegacyFr>(REGISTER_ADDRESS_BITS + LOG_T)
        };

        let outer = cycle_point(transcript);
        let product = cycle_point(transcript);
        let shift = cycle_point(transcript);
        let register_read_write = register_point(transcript);
        let register_val_evaluation = register_point(transcript);

        let mut next = 1_000u64;
        let mut claim = move || {
            next += 17;
            <LegacyFr as LegacyJoltField>::from_u64(next)
        };
        let append = |accumulator: &mut ProverOpeningAccumulator<LegacyFr>,
                      polynomial: VirtualPolynomial,
                      sumcheck: SumcheckId,
                      point: &[Challenge],
                      value: LegacyFr| {
            accumulator.append_virtual(
                polynomial,
                sumcheck,
                OpeningPoint::<BIG_ENDIAN, LegacyFr>::new(point.to_vec()),
                value,
            );
        };

        append(
            &mut accumulator,
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanOuter,
            &outer,
            claim(),
        );
        append(
            &mut accumulator,
            VirtualPolynomial::Imm,
            SumcheckId::SpartanOuter,
            &outer,
            claim(),
        );
        for flag in CircuitFlags::iter() {
            append(
                &mut accumulator,
                VirtualPolynomial::OpFlags(flag),
                SumcheckId::SpartanOuter,
                &outer,
                claim(),
            );
        }
        append(
            &mut accumulator,
            VirtualPolynomial::PC,
            SumcheckId::SpartanOuter,
            &outer,
            claim(),
        );

        append(
            &mut accumulator,
            VirtualPolynomial::OpFlags(CircuitFlags::Jump),
            SumcheckId::SpartanProductVirtualization,
            &product,
            claim(),
        );
        append(
            &mut accumulator,
            VirtualPolynomial::InstructionFlags(InstructionFlags::Branch),
            SumcheckId::SpartanProductVirtualization,
            &product,
            claim(),
        );
        append(
            &mut accumulator,
            VirtualPolynomial::OpFlags(CircuitFlags::WriteLookupOutputToRD),
            SumcheckId::SpartanProductVirtualization,
            &product,
            claim(),
        );
        append(
            &mut accumulator,
            VirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
            SumcheckId::SpartanProductVirtualization,
            &product,
            claim(),
        );

        let unexpanded_pc = claim();
        append(
            &mut accumulator,
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
            &shift,
            unexpanded_pc,
        );
        append(
            &mut accumulator,
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::InstructionInputVirtualization,
            &shift,
            unexpanded_pc,
        );
        append(
            &mut accumulator,
            VirtualPolynomial::Imm,
            SumcheckId::InstructionInputVirtualization,
            &shift,
            claim(),
        );
        for flag in [
            InstructionFlags::LeftOperandIsRs1Value,
            InstructionFlags::LeftOperandIsPC,
            InstructionFlags::RightOperandIsRs2Value,
            InstructionFlags::RightOperandIsImm,
        ] {
            append(
                &mut accumulator,
                VirtualPolynomial::InstructionFlags(flag),
                SumcheckId::InstructionInputVirtualization,
                &shift,
                claim(),
            );
        }
        append(
            &mut accumulator,
            VirtualPolynomial::InstructionFlags(InstructionFlags::IsNoop),
            SumcheckId::SpartanShift,
            &shift,
            claim(),
        );
        append(
            &mut accumulator,
            VirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
            SumcheckId::SpartanShift,
            &shift,
            claim(),
        );
        append(
            &mut accumulator,
            VirtualPolynomial::OpFlags(CircuitFlags::IsFirstInSequence),
            SumcheckId::SpartanShift,
            &shift,
            claim(),
        );
        append(
            &mut accumulator,
            VirtualPolynomial::PC,
            SumcheckId::SpartanShift,
            &shift,
            claim(),
        );

        for polynomial in [
            VirtualPolynomial::RdWa,
            VirtualPolynomial::Rs1Ra,
            VirtualPolynomial::Rs2Ra,
        ] {
            append(
                &mut accumulator,
                polynomial,
                SumcheckId::RegistersReadWriteChecking,
                &register_read_write,
                claim(),
            );
        }

        append(
            &mut accumulator,
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
            &register_val_evaluation,
            claim(),
        );
        append(
            &mut accumulator,
            VirtualPolynomial::InstructionRafFlag,
            SumcheckId::InstructionReadRaf,
            &register_val_evaluation,
            claim(),
        );
        for table in 0..NUM_LOOKUP_TABLES {
            append(
                &mut accumulator,
                VirtualPolynomial::LookupTableFlag(table),
                SumcheckId::InstructionReadRaf,
                &register_val_evaluation,
                claim(),
            );
        }

        let register_read_write_cycle = register_read_write[REGISTER_ADDRESS_BITS..].to_vec();
        let register_val_evaluation_cycle =
            register_val_evaluation[REGISTER_ADDRESS_BITS..].to_vec();
        (
            accumulator,
            [
                outer,
                product,
                shift,
                register_read_write_cycle,
                register_val_evaluation_cycle,
            ],
            register_read_write,
            register_val_evaluation,
        )
    }

    fn run_legacy(fixture: &LegacyFixture) -> LegacyRun {
        let one_hot_params = OneHotParams::new(LOG_T, fixture.bytecode.code_size, fixture.ram_k);
        let transcript = &mut Blake2bTranscript::new(&[]);
        let (mut accumulator, stage_cycle_points, register_read_write, register_val_evaluation) =
            synthetic_accumulator(transcript);

        let program =
            ProgramPreprocessing::<DoryCommitmentScheme>::Full(FullProgramPreprocessing {
                bytecode: Arc::new(fixture.bytecode.clone()),
                ram: LegacyRamPreprocessing::preprocess(Vec::new()),
            });
        let params = BytecodeReadRafSumcheckParams::<LegacyFr>::gen::<DoryCommitmentScheme>(
            &program,
            None,
            LOG_T,
            &one_hot_params,
            &accumulator,
            transcript,
        );

        assert_eq!(
            params.r_cycles.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![LOG_T; STAGE_COUNT],
            "legacy derived the wrong stage cycle point widths from the synthetic accumulator",
        );
        for (stage, (legacy, wired)) in params.r_cycles.iter().zip(&stage_cycle_points).enumerate()
        {
            assert_eq!(
                legacy, wired,
                "stage {stage}: legacy read a different cycle point than the one wired into the \
                 modular relation",
            );
        }

        let rounds = params.log_K;
        let input_claim = to_fr(params.input_claim);
        let entry_bytecode_index = params.entry_bytecode_index;
        let gammas = [
            to_fr(params.gamma_powers[1]),
            to_fr(params.stage1_gammas[1]),
            to_fr(params.stage2_gammas[1]),
            to_fr(params.stage3_gammas[1]),
            to_fr(params.stage4_gammas[1]),
            to_fr(params.stage5_gammas[1]),
        ];

        let legacy_challenges: Vec<Challenge> =
            transcript.challenge_vector_optimized::<LegacyFr>(rounds);
        let mut legacy = BytecodeReadRafAddressSumcheckProver::initialize(
            params,
            Arc::new(fixture.trace.clone()),
            Arc::new(fixture.bytecode.clone()),
        );

        let mut claim = <LegacyFr as LegacyJoltField>::from_u64(0);
        let mut messages = Vec::with_capacity(rounds);
        for (round, &r_j) in legacy_challenges.iter().enumerate() {
            let message = SumcheckInstanceProver::<LegacyFr, Blake2bTranscript>::compute_message(
                &mut legacy,
                round,
                claim,
            );
            let mut evals = [Fr::from_u64(0); SAMPLE_POINTS];
            for (point, eval) in evals.iter_mut().enumerate() {
                *eval =
                    to_fr(message.evaluate(&<LegacyFr as LegacyJoltField>::from_u64(point as u64)));
            }
            messages.push(evals);
            claim = message.evaluate(&<LegacyFr as From<_>>::from(r_j));
            SumcheckInstanceProver::<LegacyFr, Blake2bTranscript>::ingest_challenge(
                &mut legacy,
                r_j,
                round,
            );
        }
        SumcheckInstanceProver::<LegacyFr, Blake2bTranscript>::cache_openings(
            &legacy,
            &mut accumulator,
            &legacy_challenges,
        );
        let output_claim = to_fr(
            accumulator
                .get_virtual_polynomial_opening(
                    VirtualPolynomial::BytecodeReadRafAddrClaim,
                    SumcheckId::BytecodeReadRafAddressPhase,
                )
                .1,
        );

        let as_fr = |point: &[Challenge]| -> Vec<Fr> {
            point.iter().map(|value| challenge_to_fr(*value)).collect()
        };
        LegacyRun {
            messages,
            challenges: as_fr(&legacy_challenges),
            input_claim,
            output_claim,
            stage_cycle_points: core::array::from_fn(|stage| as_fr(&stage_cycle_points[stage])),
            register_read_write_point: as_fr(&register_read_write),
            register_val_evaluation_point: as_fr(&register_val_evaluation),
            entry_bytecode_index,
            gammas,
            rounds,
        }
    }

    #[test]
    fn fixture_bytecode_pushforward_is_not_degenerate() {
        with_legacy_witness(LOG_T, RAM_K, one_hot(), SEED, |witness, fixture| {
            let one_hot_params =
                OneHotParams::new(LOG_T, fixture.bytecode.code_size, fixture.ram_k);
            let dimensions = formula_dimensions_from_parts(
                one_hot(),
                LOG_T,
                fixture.bytecode.code_size,
                fixture.ram_k,
                JoltRelationId::BytecodeReadRaf,
            )
            .expect("formula dimensions")
            .bytecode_read_raf;
            assert_eq!(
                (dimensions.log_k(), dimensions.log_t()),
                (one_hot_params.bytecode_len.trailing_zeros() as usize, LOG_T),
                "the two tiers disagree on the bytecode read-RAF domain",
            );

            let rows: Vec<BytecodeReadRafWitness> =
                collect_bundles(witness as &dyn JoltWitnessPlane<Fr>, 1usize << LOG_T)
                    .expect("bytecode read-RAF bundles");
            let addresses = 1usize << dimensions.log_k();
            let mut touched = BTreeSet::new();
            for (cycle, row) in rows.iter().enumerate() {
                assert!(
                    row.bytecode_pc.0 < addresses,
                    "cycle {cycle}: PC {} escapes the padded bytecode domain",
                    row.bytecode_pc.0,
                );
                assert_eq!(
                    row.bytecode_pc.0,
                    slot_for_cycle(cycle) + 1,
                    "cycle {cycle}: the read-RAF pushforward source disagrees with the fixture's \
                     own slot schedule",
                );
                let _ = touched.insert(row.bytecode_pc.0);
            }
            assert_eq!(
                touched.len(),
                SLOTS,
                "the pushforward reaches {} of {SLOTS} mapped bytecode rows, so some Val rows \
                 carry no cycle weight at all",
                touched.len(),
            );
            assert!(
                touched.len() < addresses,
                "every padded bytecode row is touched, so the fixture cannot detect a kernel \
                 that ignores the pushforward's zeros",
            );

            let entry = fixture
                .bytecode
                .entry_bytecode_index()
                .expect("the fixture bytecode has an entry mapping");
            assert_eq!(
                entry, rows[0].bytecode_pc.0,
                "the entry term is degenerate unless cycle 0 lands on the entry index",
            );

            let mut distinct_values = BTreeSet::new();
            for row in &fixture.bytecode.bytecode[..=SLOTS] {
                let _ = distinct_values.insert((
                    format!("{:?}", row.instruction_kind),
                    row.operands.imm,
                    row.operands.rd,
                    row.operands.rs1,
                    row.operands.rs2,
                ));
            }
            assert!(
                distinct_values.len() > 1,
                "every mapped bytecode row decodes identically, so the per-stage Val tables are \
                 constant and cannot detect a wrong Val bind",
            );
        });
    }

    #[test]
    fn bytecode_read_raf_address_matches_legacy_round_for_round() {
        let Some(_) = shared_context() else {
            return;
        };
        with_legacy_witness(LOG_T, RAM_K, one_hot(), SEED, |witness, fixture| {
            let legacy = run_legacy(fixture);
            let dimensions = formula_dimensions_from_parts(
                one_hot(),
                LOG_T,
                fixture.bytecode.code_size,
                fixture.ram_k,
                JoltRelationId::BytecodeReadRaf,
            )
            .expect("formula dimensions")
            .bytecode_read_raf;
            assert_eq!(
                dimensions.log_k(),
                legacy.rounds,
                "the modular address round count disagrees with legacy's log_K",
            );

            let relation = BytecodeReadRafAddressPhase::<Fr>::new(
                dimensions,
                false,
                BytecodeStagePoints {
                    stage_cycle_points: legacy.stage_cycle_points.clone(),
                    register_read_write_point: legacy.register_read_write_point.clone(),
                    register_val_evaluation_point: legacy.register_val_evaluation_point.clone(),
                },
                legacy.entry_bytecode_index,
            );

            let claims = BytecodeReadRafAddressPhaseInputClaims::default();
            let points = BytecodeReadRafAddressPhaseInputClaims::default();
            let challenge_set = jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges {
                gamma: legacy.gammas[0],
                stage1_gamma: legacy.gammas[1],
                stage2_gamma: legacy.gammas[2],
                stage3_gamma: legacy.gammas[3],
                stage4_gamma: legacy.gammas[4],
                stage5_gamma: legacy.gammas[5],
            };
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenge_set,
            };

            let mut got_kernel = CudaBackend
                .prepare(&mut ProofSession::default(), witness, inputs())
                .expect("cuda prepare");

            let mut claim = legacy.input_claim;
            let mut bind = None;
            for (round, &challenge) in legacy.challenges.iter().enumerate() {
                let message = got_kernel
                    .prove_round(bind, round, claim)
                    .expect("cuda prove_round");
                let mut got = [Fr::from_u64(0); SAMPLE_POINTS];
                for (point, eval) in got.iter_mut().enumerate() {
                    *eval = message.evaluate(Fr::from_u64(point as u64));
                }
                assert_eq!(
                    got, legacy.messages[round],
                    "round {round} message diverged"
                );
                claim = message.evaluate(challenge);
                bind = Some(challenge);
            }
            got_kernel
                .finish_rounds(legacy.challenges[legacy.challenges.len() - 1])
                .expect("cuda finish_rounds");
            let got = got_kernel
                .output_claims(&claims)
                .expect("cuda output claims");
            assert_eq!(
                got.intermediate, legacy.output_claim,
                "the staged address-phase claim diverged",
            );
            assert!(
                got.val_stages.is_empty(),
                "full-program mode stages no BytecodeValClaim wires",
            );
        });
    }
}
