use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::{BindingOrder, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage6a::booleanity::{
    BooleanityAddressPhase, BooleanityAddressPhaseInputClaims, BooleanityAddressPhaseOutputClaims,
};
use jolt_witness::JoltWitnessPlane;

use super::masses::DeviceBooleanityMasses;
use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::device_columns::{device_trace_columns, ANY_SPAN};
use crate::cuda::common::one_hot_fold::DeviceOneHotColumns;
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

#[cfg(feature = "allocative")]
impl<F: Field> allocative::Allocative for BooleanityAddressKernel<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(allocative::Key::new("eq"), self.eq.device_bytes());
        visitor.exit();
    }
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
        session: &mut ProofSession,
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
        let families = [layout.instruction(), layout.bytecode(), layout.ram()];
        let columns =
            device_trace_columns::<F>(context, session, witness, cycles, families, ANY_SPAN)?;
        let device_columns =
            DeviceOneHotColumns::from_device(columns, families, dimensions.log_k_chunk, cycles)?;

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
pub(crate) mod fixture_support {
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
    use std::sync::Arc;
    use strum::IntoEnumIterator;
    use tracer::instruction::{Cycle, RAMRead, RAMWrite};

    pub(crate) const SLOTS: usize = 31;

    pub(crate) const SLOT_STRIDE: usize = 13;

    const KINDS: [&str; 8] = ["ADD", "SUB", "XOR", "OR", "AND", "SLTU", "LD", "SD"];

    pub(crate) struct Fixture {
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

    pub(crate) fn with_witness<R>(
        log_t: usize,
        ram_k: usize,
        one_hot: JoltOneHotConfig,
        seed: u64,
        body: impl FnOnce(&TraceBackend<OwnedTrace>, &Fixture) -> R,
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

        let preprocessing = Arc::new(JoltProgramPreprocessing {
            bytecode: bytecode.clone(),
            ram: RAMPreprocessing::default(),
            memory_layout: memory_layout.clone(),
            max_padded_trace_length: cycles,
        });
        let program = Arc::new(JoltProgram::default());
        let output = TraceOutput::new(
            OwnedTrace::new(rows.clone()),
            Default::default(),
            None,
            None,
        );
        let backend = TraceBackend::new(
            JoltVmWitnessConfig::new(log_t, ram_k, one_hot),
            JoltVmWitnessInputs::new(&program, &preprocessing, output),
        );

        let fixture = Fixture {
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

    use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
    use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltOneHotConfig, JoltRelationId};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_verifier::stages::formula_dimensions_from_parts;
    use jolt_verifier::stages::stage6a::booleanity::{
        BooleanityAddressPhase, BooleanityAddressPhaseChallenges, BooleanityAddressPhaseInputClaims,
    };

    use super::fixture_support::{row_ram_address, slot_for_cycle, with_witness, SLOTS};
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{fr, hot_addresses};
    use crate::cuda::CudaBackend;
    use crate::optimized::booleanity::OptimizedBooleanityAddress;
    use crate::{PrepareKernel, ProofSession, ProverInputs};

    const LOG_T: usize = 8;

    const RAM_K: usize = 1 << 9;

    const SEED: u64 = 20_260_816;

    const DEGREE: usize = 3;

    fn one_hot_with(log_k_chunk: u8) -> JoltOneHotConfig {
        JoltOneHotConfig {
            log_k_chunk,
            lookups_ra_virtual_log_k_chunk: 16,
        }
    }

    fn one_hot() -> JoltOneHotConfig {
        one_hot_with(4)
    }

    struct Family {
        name: &'static str,
        count: usize,
        make: fn(usize) -> JoltCommittedPolynomial,
    }

    #[test]
    fn fixture_one_hot_columns_are_not_blind() {
        with_witness(LOG_T, RAM_K, one_hot(), SEED, |witness, fixture| {
            let addresses = 1usize << one_hot().log_k_chunk;
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

            let families: [Family; 3] = [
                Family {
                    name: "instruction",
                    count: layout.instruction(),
                    make: JoltCommittedPolynomial::InstructionRa,
                },
                Family {
                    name: "bytecode",
                    count: layout.bytecode(),
                    make: JoltCommittedPolynomial::BytecodeRa,
                },
                Family {
                    name: "ram",
                    count: layout.ram(),
                    make: JoltCommittedPolynomial::RamRa,
                },
            ];

            for Family {
                name: family,
                count,
                make,
            } in families
            {
                assert!(count > 0, "the {family} family is empty in this config");
                for index in 0..count {
                    let hot = hot_addresses(witness, make(index), addresses, cycles);
                    let distinct: BTreeSet<usize> = hot.iter().flatten().copied().collect();
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
    fn fixture_is_well_formed() {
        with_witness(LOG_T, RAM_K, one_hot(), SEED, |_witness, fixture| {
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
                    slot_for_cycle(index) + 1,
                    "cycle {index}: the fixture's own slot schedule disagrees with the \
                     preprocessing",
                );
                let _ = visited.insert(mapped);
                let address = row_ram_address(row);
                assert_eq!(
                    address,
                    cycle.ram_access().address() as u64,
                    "cycle {index}: the modular row and the tracer cycle disagree on the RAM \
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

    struct Geometry {
        log_t: usize,
        ram_k: usize,
        log_k_chunk: u8,
    }

    const DEFAULT_GEOMETRY: Geometry = Geometry {
        log_t: LOG_T,
        ram_k: RAM_K,
        log_k_chunk: 4,
    };

    const SINGLE_CYCLE_ROUND: Geometry = Geometry {
        log_t: 1,
        ram_k: RAM_K,
        log_k_chunk: 4,
    };

    const WIDE_CHUNK: Geometry = Geometry {
        log_t: LOG_T,
        ram_k: RAM_K,
        log_k_chunk: 8,
    };

    struct Parity<'a> {
        relation: BooleanityAddressPhase<Fr>,
        claims: BooleanityAddressPhaseInputClaims<Fr>,
        challenges: BooleanityAddressPhaseChallenges<Fr>,
        points: BooleanityAddressPhaseInputClaims<Vec<Fr>>,
        witness: &'a dyn jolt_witness::JoltWitnessPlane<Fr>,
        chunk_bits: usize,
    }

    fn parity_setup<'a>(
        geometry: &Geometry,
        witness: &'a dyn jolt_witness::JoltWitnessPlane<Fr>,
        code_size: usize,
        ram_k: usize,
    ) -> Parity<'a> {
        let one_hot = one_hot_with(geometry.log_k_chunk);
        let layout = formula_dimensions_from_parts(
            one_hot,
            geometry.log_t,
            code_size,
            ram_k,
            JoltRelationId::Booleanity,
        )
        .expect("formula dimensions")
        .ra_layout;
        let chunk_bits = usize::from(geometry.log_k_chunk);
        let dimensions = BooleanityDimensions::new(layout, geometry.log_t, chunk_bits);
        let cycle_point: Vec<Fr> = (0..geometry.log_t).map(|i| fr(3 * i as u64 + 1)).collect();
        let address_point: Vec<Fr> = (0..chunk_bits).map(|i| fr(7 * i as u64 + 5)).collect();
        Parity {
            relation: BooleanityAddressPhase::<Fr>::new(
                dimensions,
                address_point.clone(),
                cycle_point,
            ),
            claims: BooleanityAddressPhaseInputClaims::default(),
            challenges: BooleanityAddressPhaseChallenges {
                reference_address: address_point,
                gamma: fr(11),
            },
            points: BooleanityAddressPhaseInputClaims::default(),
            witness,
            chunk_bits,
        }
    }

    impl Parity<'_> {
        fn inputs(&self) -> ProverInputs<'_, Fr, BooleanityAddressPhase<Fr>> {
            ProverInputs {
                relation: &self.relation,
                claims: &self.claims,
                points: &self.points,
                challenges: &self.challenges,
            }
        }
    }

    fn address_parity(geometry: &Geometry) {
        let Some(_) = shared_context() else {
            return;
        };
        with_witness(
            geometry.log_t,
            geometry.ram_k,
            one_hot_with(geometry.log_k_chunk),
            SEED,
            |witness, fixture| {
                let parity =
                    parity_setup(geometry, witness, fixture.bytecode.code_size, fixture.ram_k);

                let mut expected_kernel = OptimizedBooleanityAddress
                    .prepare(
                        &mut ProofSession::default(),
                        parity.witness,
                        parity.inputs(),
                    )
                    .expect("optimized prepare");
                let mut got_kernel = CudaBackend
                    .prepare(
                        &mut ProofSession::default(),
                        parity.witness,
                        parity.inputs(),
                    )
                    .expect("cuda prepare");

                assert_eq!(
                    expected_kernel.num_rounds(),
                    got_kernel.num_rounds(),
                    "the two tiers disagree on the address-phase round count",
                );
                assert!(
                    parity.chunk_bits > 0,
                    "a zero-round parity run proves nothing",
                );

                let challenges: Vec<Fr> = (0..parity.chunk_bits)
                    .map(|i| fr(13 * i as u64 + 2))
                    .collect();
                let mut expected_claim = Fr::from_u64(0);
                let mut got_claim = Fr::from_u64(0);
                let mut bind = None;
                for (round, &challenge) in challenges.iter().enumerate() {
                    let expected = expected_kernel
                        .prove_round(bind, round, expected_claim)
                        .expect("optimized prove_round");
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
                    if round == 0 {
                        assert_eq!(
                            (
                                expected.evaluate(Fr::from_u64(0)),
                                expected.evaluate(Fr::from_u64(1))
                            ),
                            (Fr::from_u64(0), Fr::from_u64(0)),
                            "round 0 must vanish at X = 0 and X = 1: the booleanity summand is \
                             identically zero on a genuinely one-hot witness, so a non-zero value \
                             means the oracle is reading the wrong trace, bytecode, memory layout \
                             or eq points",
                        );
                        assert_ne!(
                            expected.evaluate(Fr::from_u64(2)),
                            Fr::from_u64(0),
                            "round 0 must NOT vanish at X = 2: the squared leg's extension is \
                             quadratic while the linear leg's is linear, so a zero here means the \
                             masses are zero and the fixture is blind",
                        );
                    }
                    expected_claim = expected.evaluate(challenge);
                    got_claim = got.evaluate(challenge);
                    bind = Some(challenge);
                }
                let last = challenges[challenges.len() - 1];
                expected_kernel
                    .finish_rounds(last)
                    .expect("optimized finish_rounds");
                got_kernel.finish_rounds(last).expect("cuda finish_rounds");
                assert_eq!(
                    got_kernel
                        .output_claims(&parity.claims)
                        .expect("cuda output claims")
                        .intermediate,
                    expected_kernel
                        .output_claims(&parity.claims)
                        .expect("optimized output claims")
                        .intermediate,
                    "the staged address-phase claim diverged",
                );
            },
        );
    }

    #[test]
    fn booleanity_address_matches_optimized_round_for_round() {
        address_parity(&DEFAULT_GEOMETRY);
    }

    #[test]
    fn booleanity_address_matches_optimized_single_cycle_round() {
        address_parity(&SINGLE_CYCLE_ROUND);
    }

    #[test]
    fn booleanity_address_matches_optimized_wide_chunk() {
        address_parity(&WIDE_CHUNK);
    }
}
