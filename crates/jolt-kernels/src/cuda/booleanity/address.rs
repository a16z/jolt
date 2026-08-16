use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::{BindingOrder, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage6a::booleanity::{
    BooleanityAddressPhase, BooleanityAddressPhaseInputClaims, BooleanityAddressPhaseOutputClaims,
};
use jolt_witness::{collect_bundles, JoltWitnessPlane};

use super::masses::DeviceBooleanityMasses;
use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::one_hot_fold::DeviceOneHotColumns;
use crate::cuda::common::one_hot_witness::{packed_columns, OneHotCycleWitness};
use crate::cuda::common::split_eq::DeviceSplitEq;
use crate::cuda::{require_context, CudaBackend};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

pub struct BooleanityAddressKernel<F: Field> {
    context: &'static CudaKernelContext,
    relation: BooleanityAddressPhase<F>,
    masses: DeviceBooleanityMasses,
    eq: DeviceSplitEq<F>,
    last_round_poly: Option<UnivariatePoly<F>>,
    intermediate: Option<F>,
    rounds_bound: usize,
}

impl<F: Field> BooleanityAddressKernel<F> {
    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        self.masses.bind(self.context, challenge).map_err(|_| {
            SumcheckError::MissingEvaluationSource {
                kind: "cuda booleanity address-phase bind",
            }
        })?;
        self.eq.bind(challenge);
        self.rounds_bound += 1;
        if let Some(poly) = self.last_round_poly.take() {
            if self.rounds_bound == self.relation.symbolic().rounds() {
                self.intermediate = Some(poly.evaluate(challenge));
            }
        }
        Ok(())
    }
}

impl<F: Field> ProveRounds<F> for BooleanityAddressKernel<F> {
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
        let (at_zero, leading) = self
            .masses
            .round_lanes(self.context, &self.eq)
            .map_err(|_| SumcheckError::MissingEvaluationSource {
                kind: "cuda booleanity address-phase round",
            })?;
        let mut coefficients = self
            .eq
            .gruen_poly_deg_3(at_zero, leading, previous_claim)
            .into_coefficients();
        coefficients.resize(self.relation.degree() + 1, F::from_u64(0));
        let poly = UnivariatePoly::new(coefficients);
        self.last_round_poly = Some(poly.clone());
        Ok(poly)
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind)
    }
}

impl<F: Field> SumcheckKernel<F> for BooleanityAddressKernel<F> {
    type Relation = BooleanityAddressPhase<F>;

    fn output_claims(
        &mut self,
        _inputs: &BooleanityAddressPhaseInputClaims<F>,
    ) -> Result<BooleanityAddressPhaseOutputClaims<F>, SumcheckKernelError<F>> {
        let remaining = self.relation.symbolic().rounds() - self.rounds_bound;
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        if self.masses.len() != 1 {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "CUDA booleanity address phase counted every round but its mass \
                         tables are not fully bound",
            });
        }
        let intermediate = self
            .intermediate
            .ok_or(SumcheckKernelError::InvariantViolation {
                reason: "CUDA booleanity address phase never staged its intermediate claim",
            })?;
        Ok(BooleanityAddressPhaseOutputClaims { intermediate })
    }
}

impl<F: Field> PrepareKernel<F, BooleanityAddressPhase<F>> for CudaBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, BooleanityAddressPhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BooleanityAddressPhase<F>>>, KernelError<F>>
    {
        let context = require_context()?;
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let reference_cycle = relation.reference_cycle();
        if inputs.challenges.reference_address.len() != dimensions.log_k_chunk
            || reference_cycle.len() != dimensions.log_t
        {
            return Err(KernelError::InvariantViolation {
                reason: "a booleanity address-phase reference point has the wrong variable count",
            });
        }

        let layout = dimensions.layout;
        let cycles = 1usize << dimensions.log_t;
        let rows = collect_bundles::<OneHotCycleWitness>(witness, cycles)?;
        let columns = packed_columns(&rows).map_err(|_| KernelError::Unsupported {
            reason: "the CUDA booleanity address phase packs the bytecode PC and the remapped RAM \
                     word address into one 32-bit word each, reserving the all-ones word for a \
                     cold cycle",
        })?;
        drop(rows);

        let device_columns = DeviceOneHotColumns::new(
            context,
            &columns.lookup,
            &columns.pc,
            &columns.ram,
            [layout.instruction(), layout.bytecode(), layout.ram()],
            dimensions.log_k_chunk,
            cycles,
        )?;
        drop(columns);

        let masses = DeviceBooleanityMasses::new(
            context,
            &device_columns,
            &reference_cycle,
            inputs.challenges.gamma,
        )?;
        drop(device_columns);

        let eq = DeviceSplitEq::new(
            context,
            &inputs.challenges.reference_address,
            BindingOrder::LowToHigh,
        )?;

        Ok(Box::new(BooleanityAddressKernel {
            context,
            relation: relation.clone(),
            masses,
            eq,
            last_round_poly: None,
            intermediate: None,
            rounds_bound: 0,
        }))
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "test scaffolding: fixture and device errors fail loudly"
)]
pub(crate) mod legacy_fixture {
    use common::constants::RAM_START_ADDRESS;
    use common::jolt_device::{MemoryConfig, MemoryLayout};
    use jolt_claims::protocols::jolt::JoltOneHotConfig;
    use jolt_program::execution::{JoltProgram, OwnedTrace, RamAccess, TraceOutput, TraceRow};
    use jolt_program::preprocess::{
        BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing,
    };
    use jolt_riscv::{JoltInstructionRow, RV64IMAC_JOLT};
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use strum::IntoEnumIterator;
    use tracer::instruction::{Cycle, RAMRead, RAMWrite};

    pub(crate) const SLOTS: usize = 31;

    pub(crate) const SLOT_STRIDE: usize = 13;

    const KINDS: [&str; 8] = ["ADD", "SUB", "XOR", "OR", "AND", "SLTU", "LD", "SD"];

    pub(crate) struct LegacyFixture {
        pub(crate) trace: Vec<Cycle>,
        pub(crate) rows: Vec<TraceRow>,
        pub(crate) bytecode: BytecodePreprocessing,
        pub(crate) memory_layout: MemoryLayout,
        pub(crate) ram_k: usize,
    }

    pub(crate) const fn slot_address(slot: usize) -> u64 {
        RAM_START_ADDRESS + 4 * slot as u64
    }

    pub(crate) const fn slot_for_cycle(cycle: usize) -> usize {
        (cycle * SLOT_STRIDE) % SLOTS
    }

    pub(crate) const fn row_ram_address(row: &TraceRow) -> u64 {
        match row.ram_access {
            RamAccess::Read(read) => read.address,
            RamAccess::Write(write) => write.address,
            RamAccess::NoOp => 0,
        }
    }

    fn palette() -> Vec<Cycle> {
        let all: Vec<Cycle> = Cycle::iter().collect();
        KINDS
            .iter()
            .map(|kind| {
                *all.iter()
                    .find(|cycle| {
                        let name: &'static str = (*cycle).into();
                        name == *kind
                    })
                    .expect("the fixture palette names existing Cycle variants")
            })
            .collect()
    }

    fn set_address(cycle: &mut Cycle, address: u64) {
        match cycle {
            Cycle::ADD(inner) => inner.instruction.address = address,
            Cycle::SUB(inner) => inner.instruction.address = address,
            Cycle::XOR(inner) => inner.instruction.address = address,
            Cycle::OR(inner) => inner.instruction.address = address,
            Cycle::AND(inner) => inner.instruction.address = address,
            Cycle::SLTU(inner) => inner.instruction.address = address,
            Cycle::LD(inner) => inner.instruction.address = address,
            Cycle::SD(inner) => inner.instruction.address = address,
            other => panic!("the fixture palette has no address arm for {other:?}"),
        }
    }

    fn set_ram(cycle: &mut Cycle, address: u64, value: u64) {
        match cycle {
            Cycle::LD(inner) => inner.ram_access = RAMRead { address, value },
            Cycle::SD(inner) => {
                inner.ram_access = RAMWrite {
                    address,
                    pre_value: value,
                    post_value: value ^ 0x55,
                }
            }
            _ => {}
        }
    }

    fn jolt_row(cycle: &Cycle) -> JoltInstructionRow {
        cycle
            .instruction()
            .try_jolt_instruction_row()
            .expect("every fixture palette cycle has a final Jolt instruction row")
    }

    pub(crate) fn with_legacy_witness<R>(
        log_t: usize,
        ram_k: usize,
        one_hot: JoltOneHotConfig,
        seed: u64,
        body: impl FnOnce(&TraceBackend<'_, OwnedTrace>, &LegacyFixture) -> R,
    ) -> R {
        let memory_layout = MemoryLayout::new(&MemoryConfig {
            program_size: Some(1 << 12),
            ..MemoryConfig::default()
        });
        let lowest = memory_layout.get_lowest_address();
        let palette = palette();
        let mut rng = StdRng::seed_from_u64(seed);

        let bytecode_rows: Vec<JoltInstructionRow> = (0..SLOTS)
            .map(|slot| {
                let mut cycle = palette[slot % palette.len()].random(&mut rng);
                set_address(&mut cycle, slot_address(slot));
                jolt_row(&cycle)
            })
            .collect();

        let cycles = 1usize << log_t;
        let trace: Vec<Cycle> = (0..cycles)
            .map(|index| {
                let slot = slot_for_cycle(index);
                let mut cycle = palette[slot % palette.len()].random(&mut rng);
                set_address(&mut cycle, slot_address(slot));
                let word = (index as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_shr(11)
                    % ram_k as u64;
                set_ram(&mut cycle, lowest + 8 * word, 900 + index as u64);
                cycle
            })
            .collect();

        let entry_address = bytecode_rows[0].address as u64;
        let bytecode =
            BytecodePreprocessing::preprocess(bytecode_rows, entry_address, RV64IMAC_JOLT)
                .expect("the fixture bytecode preprocesses");
        let rows: Vec<TraceRow> = trace
            .iter()
            .map(|cycle| {
                tracer::trace_row_from_cycle(*cycle)
                    .expect("every fixture cycle converts to a modular trace row")
            })
            .collect();

        let preprocessing = JoltProgramPreprocessing {
            bytecode: bytecode.clone(),
            ram: RAMPreprocessing::default(),
            memory_layout: memory_layout.clone(),
            max_padded_trace_length: cycles,
        };
        let program = JoltProgram::default();
        let output = TraceOutput::new(OwnedTrace::new(rows.clone()), Default::default(), None);
        let backend = TraceBackend::new(
            JoltVmWitnessConfig::new(log_t, ram_k, one_hot),
            JoltVmWitnessInputs::new(&program, &preprocessing, output),
        );

        let fixture = LegacyFixture {
            trace,
            rows,
            bytecode,
            memory_layout,
            ram_k,
        };
        body(&backend, &fixture)
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations and fixture errors fail loudly"
)]
mod tests {
    use std::collections::BTreeSet;

    use ark_bn254::Fr as LegacyFr;
    use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
    use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltOneHotConfig, JoltRelationId};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_prover_legacy::field::JoltField as LegacyJoltField;
    use jolt_prover_legacy::poly::opening_proof::{
        OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId, BIG_ENDIAN,
    };
    use jolt_prover_legacy::poly::shared_ra_polys::RaIndices;
    use jolt_prover_legacy::subprotocols::booleanity::{
        BooleanityAddressSumcheckProver, BooleanitySumcheckParams,
    };
    use jolt_prover_legacy::subprotocols::sumcheck_prover::SumcheckInstanceProver;
    use jolt_prover_legacy::transcripts::{Blake2bTranscript, Transcript};
    use jolt_prover_legacy::zkvm::bytecode::get_pc_for_cycle;
    use jolt_prover_legacy::zkvm::config::OneHotParams;
    use jolt_prover_legacy::zkvm::witness::VirtualPolynomial;
    use jolt_verifier::stages::formula_dimensions_from_parts;
    use jolt_verifier::stages::stage6a::booleanity::{
        BooleanityAddressPhase, BooleanityAddressPhaseChallenges, BooleanityAddressPhaseInputClaims,
    };

    use super::legacy_fixture::{
        row_ram_address, slot_for_cycle, with_legacy_witness, LegacyFixture, SLOTS,
    };
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{fr, hot_addresses};
    use crate::cuda::CudaBackend;
    use crate::reference::ReferenceBackend;
    use crate::{PrepareKernel, ProofSession, ProverInputs};

    const LOG_T: usize = 8;

    const RAM_K: usize = 1 << 9;

    const SEED: u64 = 20_260_816;

    const DEGREE: usize = 3;

    fn one_hot() -> JoltOneHotConfig {
        JoltOneHotConfig {
            log_k_chunk: 4,
            lookups_ra_virtual_log_k_chunk: 16,
        }
    }

    fn legacy_params(fixture: &LegacyFixture) -> OneHotParams {
        OneHotParams::new(LOG_T, fixture.bytecode.code_size, fixture.ram_k)
    }

    fn to_fr(value: LegacyFr) -> Fr {
        Fr::from(value)
    }

    fn challenge_to_fr(challenge: <LegacyFr as LegacyJoltField>::Challenge) -> Fr {
        to_fr(<LegacyFr as From<_>>::from(challenge))
    }

    struct Family {
        name: &'static str,
        base: usize,
        count: usize,
        make: fn(usize) -> JoltCommittedPolynomial,
    }

    struct LegacyRun {
        messages: Vec<[Fr; DEGREE + 1]>,
        challenges: Vec<Fr>,
        output_claim: Fr,
        reference_address: Vec<Fr>,
        instruction_r_address: Vec<Fr>,
        instruction_r_cycle: Vec<Fr>,
        gamma: Fr,
        chunk_bits: usize,
    }

    fn run_legacy(fixture: &LegacyFixture) -> LegacyRun {
        let one_hot_params = legacy_params(fixture);
        let log_k_instruction = one_hot_params.lookups_ra_virtual_log_k_chunk;
        let chunk_bits = one_hot_params.log_k_chunk;

        let transcript = &mut Blake2bTranscript::new(&[]);
        let mut accumulator = ProverOpeningAccumulator::new(LOG_T);
        let stage5_point: Vec<<LegacyFr as LegacyJoltField>::Challenge> =
            transcript.challenge_vector_optimized::<LegacyFr>(log_k_instruction + LOG_T);
        accumulator.append_virtual(
            VirtualPolynomial::InstructionRa(0),
            SumcheckId::InstructionReadRaf,
            OpeningPoint::<BIG_ENDIAN, LegacyFr>::new(stage5_point.clone()),
            <LegacyFr as LegacyJoltField>::from_u64(7),
        );

        let params = BooleanitySumcheckParams::<LegacyFr>::new(
            LOG_T,
            &one_hot_params,
            &accumulator,
            transcript,
        );
        let rounds = params.log_k_chunk;
        assert_eq!(rounds, chunk_bits, "the address phase binds the chunk bits");
        let reference_address: Vec<Fr> = params
            .r_address
            .iter()
            .map(|challenge| challenge_to_fr(*challenge))
            .collect();
        let gamma = challenge_to_fr(params.gamma);

        let legacy_challenges: Vec<<LegacyFr as LegacyJoltField>::Challenge> =
            transcript.challenge_vector_optimized::<LegacyFr>(rounds);

        let mut legacy = BooleanityAddressSumcheckProver::initialize(
            params,
            &fixture.trace,
            &fixture.bytecode,
            &fixture.memory_layout,
        );

        let mut claim = <LegacyFr as LegacyJoltField>::from_u64(0);
        let mut messages = Vec::with_capacity(rounds);
        for (round, &r_j) in legacy_challenges.iter().enumerate() {
            let message = SumcheckInstanceProver::<LegacyFr, Blake2bTranscript>::compute_message(
                &mut legacy,
                round,
                claim,
            );
            let mut evals = [Fr::from_u64(0); DEGREE + 1];
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
                    VirtualPolynomial::BooleanityAddrClaim,
                    SumcheckId::BooleanityAddressPhase,
                )
                .1,
        );

        LegacyRun {
            messages,
            challenges: legacy_challenges
                .iter()
                .map(|challenge| challenge_to_fr(*challenge))
                .collect(),
            output_claim,
            reference_address,
            instruction_r_address: stage5_point[..log_k_instruction]
                .iter()
                .map(|challenge| challenge_to_fr(*challenge))
                .collect(),
            instruction_r_cycle: stage5_point[log_k_instruction..]
                .iter()
                .map(|challenge| challenge_to_fr(*challenge))
                .collect(),
            gamma,
            chunk_bits,
        }
    }

    #[test]
    fn fixture_one_hot_columns_agree_across_tiers() {
        with_legacy_witness(LOG_T, RAM_K, one_hot(), SEED, |witness, fixture| {
            let one_hot_params = legacy_params(fixture);
            let addresses = 1usize << one_hot_params.log_k_chunk;
            let cycles = 1usize << LOG_T;
            let layout = formula_dimensions_from_parts(
                one_hot(),
                LOG_T,
                fixture.bytecode.code_size,
                fixture.ram_k,
                JoltRelationId::Booleanity,
            )
            .expect("formula dimensions")
            .ra_layout;

            assert_eq!(
                (layout.instruction(), layout.bytecode(), layout.ram()),
                (
                    one_hot_params.instruction_d,
                    one_hot_params.bytecode_d,
                    one_hot_params.ram_d
                ),
                "the two tiers disagree on the one-hot chunk layout, so the oracle would \
                 compare different polynomials",
            );

            let legacy_indices: Vec<RaIndices> = fixture
                .trace
                .iter()
                .map(|cycle| {
                    RaIndices::from_cycle(
                        cycle,
                        &fixture.bytecode,
                        &fixture.memory_layout,
                        &one_hot_params,
                    )
                })
                .collect();

            let families: [Family; 3] = [
                Family {
                    name: "instruction",
                    base: 0,
                    count: layout.instruction(),
                    make: JoltCommittedPolynomial::InstructionRa,
                },
                Family {
                    name: "bytecode",
                    base: layout.instruction(),
                    count: layout.bytecode(),
                    make: JoltCommittedPolynomial::BytecodeRa,
                },
                Family {
                    name: "ram",
                    base: layout.instruction() + layout.bytecode(),
                    count: layout.ram(),
                    make: JoltCommittedPolynomial::RamRa,
                },
            ];

            for Family {
                name: family,
                base,
                count,
                make,
            } in families
            {
                assert!(count > 0, "the {family} family is empty in this config");
                for index in 0..count {
                    let hot = hot_addresses(witness, make(index), addresses, cycles);
                    let mut distinct = BTreeSet::new();
                    for (cycle, address) in hot.iter().enumerate() {
                        let legacy = legacy_indices[cycle]
                            .get_index(base + index, &one_hot_params)
                            .map(usize::from);
                        assert_eq!(
                            *address, legacy,
                            "{family} chunk {index} cycle {cycle}: the witness plane and the \
                             legacy RaIndices disagree on the hot address",
                        );
                        if let Some(address) = address {
                            let _ = distinct.insert(*address);
                        }
                    }
                    assert!(
                        distinct.len() > 1,
                        "{family} chunk {index} is hot at a single address across the whole \
                         fixture, so it cannot detect a wrong address bind",
                    );
                }
            }
        });
    }

    #[test]
    fn fixture_is_well_formed_for_both_tiers() {
        with_legacy_witness(LOG_T, RAM_K, one_hot(), SEED, |_witness, fixture| {
            let cycles = 1usize << LOG_T;
            let mut visited = BTreeSet::new();
            let mut ram_hot = 0usize;
            let mut ram_cold = 0usize;
            for (index, (cycle, row)) in fixture.trace.iter().zip(&fixture.rows).enumerate() {
                let mapped = fixture
                    .bytecode
                    .get_pc(&row.instruction)
                    .expect("every fixture row has a bytecode mapping");
                assert_eq!(
                    mapped,
                    get_pc_for_cycle(&fixture.bytecode, cycle),
                    "cycle {index}: the modular row and the legacy cycle map to different PCs",
                );
                assert_eq!(
                    mapped,
                    slot_for_cycle(index) + 1,
                    "cycle {index}: the fixture's own slot schedule disagrees with the \
                     preprocessing",
                );
                let _ = visited.insert(mapped);
                let address = row_ram_address(row);
                assert_eq!(
                    address,
                    cycle.ram_access().address() as u64,
                    "cycle {index}: the modular row and the legacy cycle disagree on the RAM \
                     address",
                );
                match fixture
                    .memory_layout
                    .remap_word_address(address)
                    .expect("no fixture RAM address sits below the lowest mapped address")
                {
                    Some(word) => {
                        assert!(
                            (word as usize) < fixture.ram_k,
                            "cycle {index}: remapped RAM word {word} exceeds ram_K",
                        );
                        ram_hot += 1;
                    }
                    None => ram_cold += 1,
                }
            }
            assert_eq!(
                visited.len(),
                SLOTS,
                "the fixture does not reach every bytecode slot",
            );
            assert!(
                ram_hot > 0 && ram_cold > 0,
                "{ram_hot} hot and {ram_cold} cold RAM cycles of {cycles}: one of the two RAM \
                 paths is unexercised",
            );
        });
    }

    #[test]
    fn fixture_legacy_message_vanishes_at_the_boolean_points() {
        with_legacy_witness(LOG_T, RAM_K, one_hot(), SEED, |_witness, fixture| {
            let legacy = run_legacy(fixture);
            assert_eq!(
                (legacy.messages[0][0], legacy.messages[0][1]),
                (Fr::from_u64(0), Fr::from_u64(0)),
                "round 0 must vanish at X = 0 and X = 1: the booleanity summand is identically \
                 zero on a genuinely one-hot witness, so a non-zero value means the oracle is \
                 reading the wrong trace, bytecode, memory layout or eq points",
            );
            assert_ne!(
                legacy.messages[0][2],
                Fr::from_u64(0),
                "round 0 must NOT vanish at X = 2: the squared leg's extension is quadratic \
                 while the linear leg's is linear, so a zero here means the masses are zero and \
                 the fixture is blind",
            );
        });
    }

    #[test]
    fn booleanity_address_matches_reference_round_for_round() {
        let Some(_) = shared_context() else {
            return;
        };
        with_legacy_witness(LOG_T, RAM_K, one_hot(), SEED, |witness, fixture| {
            let layout = formula_dimensions_from_parts(
                one_hot(),
                LOG_T,
                fixture.bytecode.code_size,
                fixture.ram_k,
                JoltRelationId::Booleanity,
            )
            .expect("formula dimensions")
            .ra_layout;
            let chunk_bits = usize::from(one_hot().log_k_chunk);
            let dimensions = BooleanityDimensions::new(layout, LOG_T, chunk_bits);
            let cycle_point: Vec<Fr> = (0..LOG_T).map(|i| fr(3 * i as u64 + 1)).collect();
            let address_point: Vec<Fr> = (0..chunk_bits).map(|i| fr(7 * i as u64 + 5)).collect();
            let relation =
                BooleanityAddressPhase::<Fr>::new(dimensions, address_point.clone(), cycle_point);

            let claims = BooleanityAddressPhaseInputClaims::default();
            let points = BooleanityAddressPhaseInputClaims::default();
            let challenge_set = BooleanityAddressPhaseChallenges {
                reference_address: address_point,
                gamma: fr(11),
            };
            let make_inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenge_set,
            };

            let mut expected_kernel = ReferenceBackend
                .prepare(&mut ProofSession::default(), witness, make_inputs())
                .expect("reference prepare");
            let mut got_kernel = CudaBackend
                .prepare(&mut ProofSession::default(), witness, make_inputs())
                .expect("cuda prepare");

            let challenges: Vec<Fr> = (0..chunk_bits).map(|i| fr(13 * i as u64 + 2)).collect();
            let mut expected_claim = Fr::from_u64(0);
            let mut got_claim = Fr::from_u64(0);
            let mut bind = None;
            for (round, &challenge) in challenges.iter().enumerate() {
                let expected = expected_kernel
                    .prove_round(bind, round, expected_claim)
                    .expect("reference prove_round");
                let got = got_kernel
                    .prove_round(bind, round, got_claim)
                    .expect("cuda prove_round");
                for point in 0..=DEGREE {
                    let at = Fr::from_u64(point as u64);
                    assert_eq!(
                        got.evaluate(at),
                        expected.evaluate(at),
                        "round {round} message diverged at X = {point}",
                    );
                }
                expected_claim = expected.evaluate(challenge);
                got_claim = got.evaluate(challenge);
                bind = Some(challenge);
            }
            let last = challenges[challenges.len() - 1];
            expected_kernel
                .finish_rounds(last)
                .expect("reference finish_rounds");
            got_kernel.finish_rounds(last).expect("cuda finish_rounds");
            assert_eq!(
                got_kernel
                    .output_claims(&claims)
                    .expect("cuda output claims")
                    .intermediate,
                expected_kernel
                    .output_claims(&claims)
                    .expect("reference output claims")
                    .intermediate,
                "the staged address-phase claim diverged",
            );
        });
    }

    #[test]
    fn booleanity_address_matches_legacy_round_for_round() {
        let Some(_) = shared_context() else {
            return;
        };
        with_legacy_witness(LOG_T, RAM_K, one_hot(), SEED, |witness, fixture| {
            let legacy = run_legacy(fixture);
            let layout = formula_dimensions_from_parts(
                one_hot(),
                LOG_T,
                fixture.bytecode.code_size,
                fixture.ram_k,
                JoltRelationId::Booleanity,
            )
            .expect("formula dimensions")
            .ra_layout;
            let dimensions = BooleanityDimensions::new(layout, LOG_T, legacy.chunk_bits);
            let relation = BooleanityAddressPhase::<Fr>::new(
                dimensions,
                legacy.instruction_r_address.clone(),
                legacy.instruction_r_cycle.clone(),
            );
            assert_eq!(
                relation.reference_cycle(),
                legacy
                    .instruction_r_cycle
                    .iter()
                    .rev()
                    .copied()
                    .collect::<Vec<_>>(),
                "the modular reference cycle must be the reversed stage-5 cycle legacy uses",
            );

            let claims = BooleanityAddressPhaseInputClaims::default();
            let points = BooleanityAddressPhaseInputClaims::default();
            let challenge_set = BooleanityAddressPhaseChallenges {
                reference_address: legacy.reference_address.clone(),
                gamma: legacy.gamma,
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

            let mut claim = Fr::from_u64(0);
            let mut bind = None;
            for (round, &challenge) in legacy.challenges.iter().enumerate() {
                let message = got_kernel
                    .prove_round(bind, round, claim)
                    .expect("cuda prove_round");
                let mut got = [Fr::from_u64(0); DEGREE + 1];
                for (point, eval) in got.iter_mut().enumerate() {
                    *eval = message.evaluate(Fr::from_u64(point as u64));
                }
                let expected = legacy.messages[round];
                assert_eq!(got, expected, "round {round} message diverged");
                claim = message.evaluate(challenge);
                bind = Some(challenge);
            }
            got_kernel
                .finish_rounds(legacy.challenges[legacy.challenges.len() - 1])
                .expect("cuda finish_rounds");
            let got = got_kernel
                .output_claims(&claims)
                .expect("cuda output claims")
                .intermediate;
            assert_eq!(
                got, legacy.output_claim,
                "the staged address-phase claim diverged",
            );
        });
    }
}
