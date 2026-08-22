use crate::fixture::{Fixture, Parts};
use jolt_prover::ProverConfig;
use jolt_witness::JoltWitnessPlane;
use std::time::{Duration, Instant};

use common::constants::XLEN as RISCV_XLEN;
use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::{
    COMMITTED_BYTECODE_LANE_CAPACITY, NUM_BYTECODE_VAL_STAGES,
};
use jolt_claims::protocols::jolt::geometry::claim_reductions::hamming_weight::HammingWeightClaimReductionDimensions;
use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::protocols::jolt::geometry::ram::RamRafEvaluationDimensions;
use jolt_claims::protocols::jolt::geometry::ram::RamValCheckInit;
use jolt_claims::protocols::jolt::geometry::spartan::{
    SpartanOuterDimensions, SpartanProductDimensions,
};
use jolt_claims::protocols::jolt::relations::booleanity::{
    BooleanityAddressPhaseChallenges, BooleanityAddressPhaseInputClaims,
};
use jolt_claims::protocols::jolt::relations::bytecode::{
    BytecodeReadRafAddressPhaseChallenges, BytecodeReadRafAddressPhaseInputClaims,
};
use jolt_claims::protocols::jolt::relations::claim_reductions::advice::{
    TrustedAdviceAddressPhaseInputClaims, TrustedAdviceCyclePhaseInputClaims,
    UntrustedAdviceAddressPhaseInputClaims, UntrustedAdviceCyclePhaseInputClaims,
};
use jolt_claims::protocols::jolt::relations::claim_reductions::bytecode::{
    BytecodeReductionAddressPhaseInputClaims, BytecodeReductionCyclePhaseChallenges,
    BytecodeReductionCyclePhaseInputClaims,
};
use jolt_claims::protocols::jolt::relations::claim_reductions::hamming_weight::{
    HammingWeightClaimReductionChallenges, HammingWeightClaimReductionInputClaims,
};
use jolt_claims::protocols::jolt::relations::claim_reductions::increments::{
    IncClaimReductionChallenges, IncClaimReductionInputClaims,
};
use jolt_claims::protocols::jolt::relations::claim_reductions::instruction::{
    InstructionClaimReductionChallenges, InstructionClaimReductionInputClaims,
};
use jolt_claims::protocols::jolt::relations::claim_reductions::program_image::{
    ProgramImageReductionAddressPhaseInputClaims, ProgramImageReductionCyclePhaseInputClaims,
};
use jolt_claims::protocols::jolt::relations::claim_reductions::registers::{
    RegistersClaimReductionChallenges, RegistersClaimReductionInputClaims,
};
use jolt_claims::protocols::jolt::relations::instruction::{
    InstructionInputChallenges, InstructionInputInputClaims, InstructionReadRafChallenges,
    InstructionReadRafInputClaims,
};
use jolt_claims::protocols::jolt::relations::ram::{
    RamHammingBooleanityInputClaims, RamOutputCheckChallenges, RamOutputCheckInputClaims,
    RamRaClaimReductionChallenges, RamRaClaimReductionInputClaims, RamRafEvaluationInputClaims,
    RamValCheckChallenges, RamValCheckInputClaims,
};
use jolt_claims::protocols::jolt::relations::registers::RegistersValEvaluationInputClaims;
use jolt_claims::protocols::jolt::relations::spartan::{
    ProductRemainderInputClaims, SpartanShiftChallenges, SpartanShiftInputClaims,
};
use jolt_claims::protocols::jolt::JoltCommittedPolynomial;
use jolt_claims::protocols::jolt::{
    AdviceClaimReductionLayout, BytecodeClaimReductionLayout, JoltAdviceKind, JoltChallengeId,
    JoltRelationId, PrecommittedClaimReduction, PrecommittedReductionLayout,
    ProgramImageClaimReductionLayout, TraceDimensions,
};
use jolt_claims::NoChallenges;
use jolt_dory::DoryScheme;
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_kernels::{CommitmentGrid, JoltBackend, ProofSession, ProverInputs};
use jolt_program::preprocess::PublicIoMemory;
use jolt_sumcheck::SumcheckError;
use jolt_verifier::stages::formula_dimensions_from_parts;
use jolt_verifier::stages::stage1::outer_remainder::{
    outer_remainder_input_values_from_uniskip_output, OuterRemainder, OuterRemainderInputClaims,
};
use jolt_verifier::stages::stage2::instruction_claim_reduction::InstructionClaimReduction;
use jolt_verifier::stages::stage2::product_remainder::{
    product_remainder_input_values_from_uniskip_output, ProductRemainder,
};
use jolt_verifier::stages::stage2::ram_output_check::RamOutputCheck;
use jolt_verifier::stages::stage2::ram_raf_evaluation::RamRafEvaluation;
use jolt_verifier::stages::stage2::ram_read_write_checking::{
    RamReadWriteChallenges, RamReadWriteChecking, RamReadWriteInputClaims,
};
use jolt_verifier::stages::stage3::outputs::{
    InstructionInput, RegistersClaimReduction, SpartanShift,
};
use jolt_verifier::stages::stage4::registers_read_write_checking::{
    RegistersReadWriteChallenges, RegistersReadWriteChecking, RegistersReadWriteInputClaims,
};
use jolt_verifier::stages::stage5::instruction_read_raf::InstructionReadRaf;
use jolt_verifier::stages::stage5::ram_ra_claim_reduction::RamRaClaimReduction;
use jolt_verifier::stages::stage5::registers_val_evaluation::RegistersValEvaluation;
use jolt_verifier::stages::stage6a::booleanity::BooleanityAddressPhase;
use jolt_verifier::stages::stage6a::bytecode_read_raf::{
    BytecodeReadRafAddressPhase, BytecodeStagePoints,
};
use jolt_verifier::stages::stage6b::booleanity::{
    Booleanity, BooleanityCyclePhaseChallenges, BooleanityInputClaims,
};
use jolt_verifier::stages::stage6b::bytecode_read_raf::{
    BytecodeReadRafCommittedCycleInputs, BytecodeReadRafCycle,
    BytecodeReadRafCyclePhaseCommittedChallenges, BytecodeReadRafInputClaims,
    READ_RAF_CYCLE_STAGES,
};
use jolt_verifier::stages::stage6b::committed_reduction_cycle_phase::{
    BytecodeReductionCyclePhase, ProgramImageReductionCyclePhase, TrustedAdviceCyclePhase,
    UntrustedAdviceCyclePhase,
};
use jolt_verifier::stages::stage6b::inc_claim_reduction::IncClaimReduction;
use jolt_verifier::stages::stage6b::instruction_ra_virtualization::{
    InstructionRaVirtualization, InstructionRaVirtualizationChallenges,
    InstructionRaVirtualizationInputClaims,
};
use jolt_verifier::stages::stage6b::outputs::BytecodeReductionWeights;
use jolt_verifier::stages::stage6b::ram_hamming_booleanity::RamHammingBooleanity;
use jolt_verifier::stages::stage6b::ram_ra_virtualization::{
    RamRaVirtualization, RamRaVirtualizationInputClaims,
};
use jolt_verifier::stages::stage7::advice_address_phase::{
    TrustedAdviceAddressPhase, UntrustedAdviceAddressPhase,
};
use jolt_verifier::stages::stage7::committed_reduction_address_phase::{
    BytecodeReductionAddressPhase, ProgramImageReductionAddressPhase,
};
use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::HammingWeightClaimReduction;
use jolt_verifier::stages::PrecommittedSchedule;

use jolt_prover::profile::BackendKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoundPhase {
    Address,
    Handoff,
    Cycle,
}

#[derive(Clone, Copy, Debug)]
pub struct VerticalTiming {
    pub log_t: usize,
    pub prepare: Duration,
    pub address: Duration,
    pub handoff: Duration,
    pub cycle: Duration,
    pub claims: Duration,
}

impl VerticalTiming {
    pub fn total(&self) -> Duration {
        self.prepare + self.address + self.handoff + self.cycle + self.claims
    }
}

#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "measurement harness: kernel errors fail loudly"
)]
fn drive_rounds<F, R>(
    kernel: &mut dyn jolt_kernels::SumcheckKernel<F, Relation = R>,
    claims: &jolt_verifier::stages::relations::SumcheckInputClaims<F, R>,
    rounds: usize,
    log_t: usize,
    prepare: Duration,
    phase: impl Fn(usize) -> RoundPhase,
) -> VerticalTiming
where
    F: jolt_field::Field + FromPrimitiveInt,
    R: jolt_verifier::stages::relations::ConcreteSumcheck<F>,
    jolt_verifier::stages::relations::SumcheckInputClaims<F, R>: jolt_claims::InputClaims<F>,
    jolt_verifier::stages::relations::SumcheckOutputClaims<F, R>: jolt_claims::OutputClaims<F>,
    jolt_verifier::stages::relations::ConcreteSumcheckChallenges<F, R>:
        jolt_claims::SumcheckChallenges<F, JoltChallengeId>,
{
    let mut claim = F::from_u64(0);
    let mut bind = None;
    let mut address = Duration::ZERO;
    let mut handoff = Duration::ZERO;
    let mut cycle = Duration::ZERO;

    for round in 0..rounds {
        let challenge = F::from_u64(17 + round as u64);
        let start = Instant::now();
        let outcome = kernel.prove_round(bind, round, claim);
        let elapsed = start.elapsed();
        match phase(round) {
            RoundPhase::Address => address += elapsed,
            RoundPhase::Handoff => handoff += elapsed,
            RoundPhase::Cycle => cycle += elapsed,
        }
        claim = match outcome {
            Ok(poly) => poly.evaluate(challenge),
            Err(SumcheckError::RoundCheckFailed { actual, .. }) => actual,
            Err(error) => panic!("vertical round {round} failed: {error:?}"),
        };
        bind = Some(challenge);
    }
    kernel
        .finish_rounds(F::from_u64(17 + rounds as u64))
        .expect("finish the vertical rounds");

    let start = Instant::now();
    let _ = kernel
        .output_claims(claims)
        .expect("vertical output claims");
    let claims_time = start.elapsed();

    VerticalTiming {
        log_t,
        prepare,
        address,
        handoff,
        cycle,
        claims: claims_time,
    }
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
pub fn measure_instruction_read_raf(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    let Parts {
        program_preprocessing,
        config,
        log_t,
        ..
    } = f.parts();

    let dimensions = formula_dimensions_from_parts(
        config.one_hot_config,
        log_t,
        program_preprocessing.bytecode.code_size,
        config.ram_K,
        JoltRelationId::InstructionReadRaf,
    )
    .expect("formula dimensions")
    .instruction_read_raf;
    let relation = InstructionReadRaf::<Fr>::new(dimensions);

    let point = |offset: u64| -> Vec<Fr> {
        (0..log_t)
            .map(|i| Fr::from_u64(offset + 7 * i as u64 + 3))
            .collect()
    };
    let claims = InstructionReadRafInputClaims {
        lookup_output: Fr::from_u64(0),
        left_lookup_operand: Fr::from_u64(0),
        right_lookup_operand: Fr::from_u64(0),
    };
    let points = InstructionReadRafInputClaims {
        lookup_output: point(31),
        left_lookup_operand: point(131),
        right_lookup_operand: point(231),
    };
    let challenges = InstructionReadRafChallenges {
        gamma: Fr::from_u64(101),
    };
    let inputs = || ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        BackendKind::Optimized => JoltBackend::<Fr, DoryScheme>::optimized(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    warm_shared_witness(&mut session, witness, backend, f.log_t);
    let start = Instant::now();
    let mut kernel = selected
        .instruction_read_raf
        .prepare(&mut session, witness, inputs())
        .expect("prepare the stage-5 read-RAF kernel");
    let prepare = start.elapsed();

    let address_bits = 2 * RISCV_XLEN;
    drive_rounds(
        &mut *kernel,
        &claims,
        address_bits + log_t,
        log_t,
        prepare,
        |round| match round.cmp(&address_bits) {
            std::cmp::Ordering::Less => RoundPhase::Address,
            std::cmp::Ordering::Equal => RoundPhase::Handoff,
            std::cmp::Ordering::Greater => RoundPhase::Cycle,
        },
    )
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
pub fn measure_instruction_ra_virtualization(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    let Parts {
        program_preprocessing,
        config,
        log_t,
        ..
    } = f.parts();

    let dimensions = formula_dimensions_from_parts(
        config.one_hot_config,
        log_t,
        program_preprocessing.bytecode.code_size,
        config.ram_K,
        JoltRelationId::InstructionRaVirtualization,
    )
    .expect("formula dimensions")
    .instruction_ra_virtualization;
    let chunk_bits = config.one_hot_config.committed_chunk_bits();
    let instruction_address: Vec<Fr> = (0..dimensions.num_committed_ra_polys() * chunk_bits)
        .map(|i| Fr::from_u64(29 + 5 * i as u64))
        .collect();
    let instruction_read_raf_cycle: Vec<Fr> = (0..log_t)
        .map(|i| Fr::from_u64(37 + 7 * i as u64))
        .collect();
    let relation = InstructionRaVirtualization::<Fr>::new(
        dimensions,
        instruction_address,
        instruction_read_raf_cycle,
        chunk_bits,
    );

    let claims = InstructionRaVirtualizationInputClaims {
        instruction_ra: vec![Fr::from_u64(0); dimensions.num_virtual_ra_polys()],
    };
    let points = InstructionRaVirtualizationInputClaims {
        instruction_ra: (0..dimensions.num_virtual_ra_polys())
            .map(|virtual_index| {
                (0..dimensions.num_committed_per_virtual() * chunk_bits + log_t)
                    .map(|bit| Fr::from_u64(11 + 3 * (virtual_index * 97 + bit) as u64))
                    .collect()
            })
            .collect(),
    };
    let challenges = InstructionRaVirtualizationChallenges {
        gamma: Fr::from_u64(101),
    };
    let inputs = || ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        BackendKind::Optimized => JoltBackend::<Fr, DoryScheme>::optimized(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    warm_shared_witness(&mut session, witness, backend, f.log_t);
    let start = Instant::now();
    let mut kernel = selected
        .instruction_ra_virtualization
        .prepare(&mut session, witness, inputs())
        .expect("prepare the stage-6b RA virtualization kernel");
    let prepare = start.elapsed();

    drive_rounds(&mut *kernel, &claims, log_t, log_t, prepare, |_| {
        RoundPhase::Cycle
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
pub fn measure_booleanity_cycle(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    let Parts {
        program_preprocessing,
        config,
        log_t,
        ..
    } = f.parts();

    let chunk_bits = config.one_hot_config.committed_chunk_bits();
    let layout = formula_dimensions_from_parts(
        config.one_hot_config,
        log_t,
        program_preprocessing.bytecode.code_size,
        config.ram_K,
        JoltRelationId::Booleanity,
    )
    .expect("formula dimensions")
    .ra_layout;
    let dimensions = BooleanityDimensions::new(layout, log_t, chunk_bits);

    let r_address: Vec<Fr> = (0..chunk_bits)
        .map(|i| Fr::from_u64(23 + 3 * i as u64))
        .collect();
    let reference_address: Vec<Fr> = (0..chunk_bits)
        .map(|i| Fr::from_u64(29 + 5 * i as u64))
        .collect();
    let reference_cycle: Vec<Fr> = (0..log_t)
        .map(|i| Fr::from_u64(37 + 7 * i as u64))
        .collect();
    let relation = Booleanity::<Fr>::new(
        dimensions,
        r_address.clone(),
        reference_address,
        reference_cycle.clone(),
    );

    let claims = BooleanityInputClaims {
        address_phase: Fr::from_u64(0),
    };
    let points = BooleanityInputClaims {
        address_phase: [r_address.as_slice(), reference_cycle.as_slice()].concat(),
    };
    let challenges = BooleanityCyclePhaseChallenges {
        gamma: Fr::from_u64(101),
    };
    let inputs = || ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        BackendKind::Optimized => JoltBackend::<Fr, DoryScheme>::optimized(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    warm_shared_witness(&mut session, witness, backend, f.log_t);
    let start = Instant::now();
    let mut kernel = selected
        .booleanity_cycle
        .prepare(&mut session, witness, inputs())
        .expect("prepare the stage-6b booleanity cycle-phase kernel");
    let prepare = start.elapsed();

    drive_rounds(&mut *kernel, &claims, log_t, log_t, prepare, |_| {
        RoundPhase::Cycle
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
pub fn measure_bytecode_read_raf_cycle(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    let Parts {
        program_preprocessing,
        config,
        log_t,
        ..
    } = f.parts();

    let chunk_bits = config.one_hot_config.committed_chunk_bits();
    let dimensions = formula_dimensions_from_parts(
        config.one_hot_config,
        log_t,
        program_preprocessing.bytecode.code_size,
        config.ram_K,
        JoltRelationId::BytecodeReadRaf,
    )
    .expect("formula dimensions")
    .bytecode_read_raf;
    let log_k = dimensions.log_k();

    let r_address: Vec<Fr> = (0..log_k)
        .map(|i| Fr::from_u64(23 + 3 * i as u64))
        .collect();
    let stage_cycle_points: [Vec<Fr>; READ_RAF_CYCLE_STAGES] = core::array::from_fn(|stage| {
        (0..log_t)
            .map(|i| Fr::from_u64(37 + 7 * i as u64 + 101 * stage as u64))
            .collect()
    });
    let relation = BytecodeReadRafCycle::<Fr>::committed(BytecodeReadRafCommittedCycleInputs {
        dimensions,
        r_address: r_address.clone(),
        stage_cycle_points,
        entry_bytecode_index: (1usize << log_k) - 2,
        committed_chunk_bits: chunk_bits,
        val_stages: (0..NUM_BYTECODE_VAL_STAGES)
            .map(|stage| Fr::from_u64(53 + 11 * stage as u64))
            .collect(),
    });

    let claims = BytecodeReadRafInputClaims {
        address_phase: Fr::from_u64(0),
    };
    let points = BytecodeReadRafInputClaims {
        address_phase: r_address,
    };
    let challenges = BytecodeReadRafCyclePhaseCommittedChallenges {
        gamma: Fr::from_u64(101),
    };
    let inputs = || ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        BackendKind::Optimized => JoltBackend::<Fr, DoryScheme>::optimized(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    warm_shared_witness(&mut session, witness, backend, f.log_t);
    let start = Instant::now();
    let mut kernel = selected
        .bytecode_read_raf_cycle
        .prepare(&mut session, witness, inputs())
        .expect("prepare the stage-6b bytecode read-RAF cycle-phase kernel");
    let prepare = start.elapsed();

    drive_rounds(&mut *kernel, &claims, log_t, log_t, prepare, |_| {
        RoundPhase::Cycle
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
pub fn measure_ram_ra_virtualization(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    let Parts {
        program_preprocessing,
        config,
        log_t,
        ..
    } = f.parts();

    let ram_log_k = config.ram_K.ilog2() as usize;
    let dimensions = formula_dimensions_from_parts(
        config.one_hot_config,
        log_t,
        program_preprocessing.bytecode.code_size,
        config.ram_K,
        JoltRelationId::RamRaVirtualization,
    )
    .expect("formula dimensions")
    .ram_ra_virtualization;
    let chunk_bits = config.one_hot_config.committed_chunk_bits();
    let ram_reduced_address: Vec<Fr> = (0..ram_log_k)
        .map(|i| Fr::from_u64(29 + 5 * i as u64))
        .collect();
    let ram_reduced_cycle: Vec<Fr> = (0..log_t)
        .map(|i| Fr::from_u64(37 + 7 * i as u64))
        .collect();
    let relation = RamRaVirtualization::<Fr>::new(
        dimensions,
        ram_reduced_address.clone(),
        ram_reduced_cycle.clone(),
        chunk_bits,
    );

    let claims = RamRaVirtualizationInputClaims {
        ram_ra_reduced: Fr::from_u64(0),
    };
    let points = RamRaVirtualizationInputClaims {
        ram_ra_reduced: [ram_reduced_address.as_slice(), ram_reduced_cycle.as_slice()].concat(),
    };
    let challenges = NoChallenges::<Fr>::default();
    let inputs = || ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        BackendKind::Optimized => JoltBackend::<Fr, DoryScheme>::optimized(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    warm_shared_witness(&mut session, witness, backend, f.log_t);
    let start = Instant::now();
    let mut kernel = selected
        .ram_ra_virtualization
        .prepare(&mut session, witness, inputs())
        .expect("prepare the stage-6b RAM RA virtualization kernel");
    let prepare = start.elapsed();

    drive_rounds(&mut *kernel, &claims, log_t, log_t, prepare, |_| {
        RoundPhase::Cycle
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
pub fn measure_ram_val_check(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    let Parts {
        program_preprocessing: _,
        config,
        log_t,
        ..
    } = f.parts();

    let ram_log_k = config.ram_K.ilog2() as usize;
    let relation = jolt_verifier::stages::stage4::ram_val_check::RamValCheck::<Fr>::new(
        TraceDimensions::new(log_t),
        ram_log_k,
        RamValCheckInit::full(Fr::from_u64(0)),
    );

    let point: Vec<Fr> = (0..ram_log_k + log_t)
        .map(|i| Fr::from_u64(31 + 7 * i as u64 + 3))
        .collect();
    let claims = RamValCheckInputClaims {
        ram_val: Fr::from_u64(0),
        ram_val_final: Fr::from_u64(0),
        untrusted_advice: None,
        trusted_advice: None,
        program_image: None,
    };
    let points = RamValCheckInputClaims {
        ram_val: point.clone(),
        ram_val_final: point,
        untrusted_advice: None,
        trusted_advice: None,
        program_image: None,
    };
    let challenges = RamValCheckChallenges {
        gamma: Fr::from_u64(101),
    };
    let inputs = || ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        BackendKind::Optimized => JoltBackend::<Fr, DoryScheme>::optimized(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    warm_shared_witness(&mut session, witness, backend, f.log_t);
    let start = Instant::now();
    let mut kernel = selected
        .ram_val_check
        .prepare(&mut session, witness, inputs())
        .expect("prepare the stage-4 RAM value-check kernel");
    let prepare = start.elapsed();

    drive_rounds(&mut *kernel, &claims, log_t, log_t, prepare, |_| {
        RoundPhase::Cycle
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
pub fn measure_ram_read_write(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    let Parts {
        program_preprocessing: _program_preprocessing,
        config,
        log_t,
        ..
    } = f.parts();
    let ram_log_k = config.ram_K.ilog2() as usize;

    let ram_dimensions = config.rw_config.ram_dimensions(log_t, ram_log_k);
    let tau_low: Vec<Fr> = (0..log_t)
        .map(|i| Fr::from_u64(53 + 11 * i as u64))
        .collect();
    let relation = RamReadWriteChecking::<Fr>::new(ram_dimensions, ram_log_k, tau_low);

    let point = |offset: u64| -> Vec<Fr> {
        (0..log_t)
            .map(|i| Fr::from_u64(offset + 7 * i as u64 + 3))
            .collect()
    };
    let claims = RamReadWriteInputClaims {
        ram_read_value: Fr::from_u64(0),
        ram_write_value: Fr::from_u64(0),
    };
    let points = RamReadWriteInputClaims {
        ram_read_value: point(41),
        ram_write_value: point(141),
    };
    let challenges = RamReadWriteChallenges {
        gamma: Fr::from_u64(103),
    };
    let inputs = || ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        BackendKind::Optimized => JoltBackend::<Fr, DoryScheme>::optimized(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    warm_shared_witness(&mut session, witness, backend, f.log_t);
    let start = Instant::now();
    let mut kernel = selected
        .ram_read_write
        .prepare(&mut session, witness, inputs())
        .expect("prepare the stage-2 RAM read-write kernel");
    let prepare = start.elapsed();

    let rounds = jolt_claims::SymbolicSumcheck::rounds(
        jolt_verifier::stages::relations::ConcreteSumcheck::symbolic(&relation),
    );
    drive_rounds(&mut *kernel, &claims, rounds, log_t, prepare, |round| {
        if round < log_t {
            RoundPhase::Cycle
        } else {
            RoundPhase::Address
        }
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
pub fn measure_registers_read_write(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    let Parts {
        program_preprocessing: _,
        config,
        log_t,
        ..
    } = f.parts();

    let register_dimensions = config
        .rw_config
        .register_dimensions(log_t, REGISTER_ADDRESS_BITS);
    let relation = RegistersReadWriteChecking::<Fr>::new(register_dimensions);

    let point = |offset: u64| -> Vec<Fr> {
        (0..log_t)
            .map(|i| Fr::from_u64(offset + 7 * i as u64 + 3))
            .collect()
    };
    let claims = RegistersReadWriteInputClaims {
        rd_write_value: Fr::from_u64(0),
        rs1_value: Fr::from_u64(0),
        rs2_value: Fr::from_u64(0),
    };
    let points = RegistersReadWriteInputClaims {
        rd_write_value: point(31),
        rs1_value: point(131),
        rs2_value: point(231),
    };
    let challenges = RegistersReadWriteChallenges {
        gamma: Fr::from_u64(101),
    };
    let inputs = || ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        BackendKind::Optimized => JoltBackend::<Fr, DoryScheme>::optimized(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    warm_shared_witness(&mut session, witness, backend, f.log_t);
    let start = Instant::now();
    let mut kernel = selected
        .registers_read_write
        .prepare(&mut session, witness, inputs())
        .expect("prepare the stage-4 registers read-write kernel");
    let prepare = start.elapsed();

    let rounds = jolt_claims::SymbolicSumcheck::rounds(
        jolt_verifier::stages::relations::ConcreteSumcheck::symbolic(&relation),
    );
    drive_rounds(&mut *kernel, &claims, rounds, log_t, prepare, |round| {
        if round < log_t {
            RoundPhase::Cycle
        } else {
            RoundPhase::Address
        }
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
pub fn measure_spartan_outer(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    let Parts {
        program_preprocessing: _,
        config: _,
        log_t,
        ..
    } = f.parts();

    let tau: Vec<Fr> = (0..log_t + 2)
        .map(|i| Fr::from_u64(37 + 7 * i as u64))
        .collect();
    let uniskip_challenge = Fr::from_u64(101);

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        BackendKind::Optimized => JoltBackend::<Fr, DoryScheme>::optimized(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    warm_shared_witness(&mut session, witness, backend, f.log_t);
    let start = Instant::now();
    selected
        .spartan_outer_uniskip
        .prepare(&mut session, log_t, &tau, witness)
        .expect("prepare the stage-1 Spartan outer uni-skip kernel");
    let prepare = start.elapsed();

    let start = Instant::now();
    let _ = selected
        .spartan_outer_uniskip
        .first_round_poly(&mut session, &[])
        .expect("the stage-1 Spartan outer uni-skip first-round polynomial");
    let uniskip_poly = start.elapsed();

    let relation = OuterRemainder::new(
        SpartanOuterDimensions::rv64(log_t),
        tau.clone(),
        uniskip_challenge,
    );
    let claims = outer_remainder_input_values_from_uniskip_output(Fr::from_u64(0));
    let points = OuterRemainderInputClaims {
        outer_uniskip: Vec::new(),
    };
    let challenges = NoChallenges::default();
    let inputs = ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let start = Instant::now();
    let mut kernel = selected
        .spartan_outer_remainder
        .prepare(&mut session, witness, inputs)
        .expect("prepare the stage-1 Spartan outer remainder kernel");
    let remainder_prepare = start.elapsed();

    let rounds = jolt_claims::SymbolicSumcheck::rounds(
        jolt_verifier::stages::relations::ConcreteSumcheck::symbolic(&relation),
    );
    let mut timing = drive_rounds(&mut *kernel, &claims, rounds, log_t, prepare, |_| {
        RoundPhase::Cycle
    });
    timing.handoff = uniskip_poly;
    timing.address = remainder_prepare;
    timing
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
pub fn measure_spartan_shift(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    let Parts {
        program_preprocessing: _,
        config: _,
        log_t,
        ..
    } = f.parts();

    let product_tau_low: Vec<Fr> = (0..log_t)
        .map(|i| Fr::from_u64(23 + 3 * i as u64))
        .collect();
    let product_remainder_point: Vec<Fr> = (0..log_t)
        .map(|i| Fr::from_u64(37 + 7 * i as u64))
        .collect();
    let relation = SpartanShift::<Fr>::new(
        TraceDimensions::new(log_t),
        product_tau_low,
        product_remainder_point,
    );

    let claims = SpartanShiftInputClaims::default();
    let points = SpartanShiftInputClaims::default();
    let challenges = SpartanShiftChallenges {
        gamma: Fr::from_u64(101),
    };
    let inputs = || ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        BackendKind::Optimized => JoltBackend::<Fr, DoryScheme>::optimized(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    warm_shared_witness(&mut session, witness, backend, f.log_t);
    let start = Instant::now();
    let mut kernel = selected
        .spartan_shift
        .prepare(&mut session, witness, inputs())
        .expect("prepare the stage-3 Spartan shift kernel");
    let prepare = start.elapsed();

    drive_rounds(&mut *kernel, &claims, log_t, log_t, prepare, |_| {
        RoundPhase::Cycle
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
pub fn measure_instruction_input(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    let Parts {
        program_preprocessing: _,
        config: _,
        log_t,
        ..
    } = f.parts();

    let product_remainder_point: Vec<Fr> = (0..log_t)
        .map(|i| Fr::from_u64(37 + 7 * i as u64))
        .collect();
    let relation =
        InstructionInput::<Fr>::new(TraceDimensions::new(log_t), product_remainder_point);

    let claims = InstructionInputInputClaims::default();
    let points = InstructionInputInputClaims::default();
    let challenges = InstructionInputChallenges {
        gamma: Fr::from_u64(101),
    };
    let inputs = || ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        BackendKind::Optimized => JoltBackend::<Fr, DoryScheme>::optimized(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    warm_shared_witness(&mut session, witness, backend, f.log_t);
    let start = Instant::now();
    let mut kernel = selected
        .instruction_input
        .prepare(&mut session, witness, inputs())
        .expect("prepare the stage-3 instruction input-virtualization kernel");
    let prepare = start.elapsed();

    drive_rounds(&mut *kernel, &claims, log_t, log_t, prepare, |_| {
        RoundPhase::Cycle
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
pub fn measure_registers_claim_reduction(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    let Parts {
        program_preprocessing: _,
        config: _,
        log_t,
        ..
    } = f.parts();

    let product_tau_low: Vec<Fr> = (0..log_t)
        .map(|i| Fr::from_u64(23 + 3 * i as u64))
        .collect();
    let relation = RegistersClaimReduction::<Fr>::new(TraceDimensions::new(log_t), product_tau_low);

    let claims = RegistersClaimReductionInputClaims::default();
    let points = RegistersClaimReductionInputClaims::default();
    let challenges = RegistersClaimReductionChallenges {
        gamma: Fr::from_u64(101),
    };
    let inputs = || ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        BackendKind::Optimized => JoltBackend::<Fr, DoryScheme>::optimized(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    warm_shared_witness(&mut session, witness, backend, f.log_t);
    let start = Instant::now();
    let mut kernel = selected
        .registers_claim_reduction
        .prepare(&mut session, witness, inputs())
        .expect("prepare the stage-3 registers claim-reduction kernel");
    let prepare = start.elapsed();

    drive_rounds(&mut *kernel, &claims, log_t, log_t, prepare, |_| {
        RoundPhase::Cycle
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
pub fn measure_instruction_claim_reduction(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    let Parts {
        program_preprocessing: _,
        config: _,
        log_t,
        ..
    } = f.parts();

    let tau_low: Vec<Fr> = (0..log_t)
        .map(|i| Fr::from_u64(23 + 3 * i as u64))
        .collect();
    let relation = InstructionClaimReduction::<Fr>::new(TraceDimensions::new(log_t), tau_low);

    let claims = InstructionClaimReductionInputClaims::default();
    let points = InstructionClaimReductionInputClaims::default();
    let challenges = InstructionClaimReductionChallenges {
        gamma: Fr::from_u64(101),
    };
    let inputs = || ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        BackendKind::Optimized => JoltBackend::<Fr, DoryScheme>::optimized(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    warm_shared_witness(&mut session, witness, backend, f.log_t);
    let start = Instant::now();
    let mut kernel = selected
        .instruction_claim_reduction
        .prepare(&mut session, witness, inputs())
        .expect("prepare the stage-2 instruction claim-reduction kernel");
    let prepare = start.elapsed();

    drive_rounds(&mut *kernel, &claims, log_t, log_t, prepare, |_| {
        RoundPhase::Cycle
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
pub fn measure_inc_claim_reduction(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    let Parts {
        program_preprocessing: _,
        config: _,
        log_t,
        ..
    } = f.parts();

    let cycle_point = |offset: u64| -> Vec<Fr> {
        (0..log_t)
            .map(|i| Fr::from_u64(23 + 3 * i as u64 + offset))
            .collect()
    };
    let relation = IncClaimReduction::<Fr>::new(
        TraceDimensions::new(log_t),
        cycle_point(0),
        cycle_point(101),
        cycle_point(202),
        cycle_point(303),
    );

    let claims = IncClaimReductionInputClaims::default();
    let points = IncClaimReductionInputClaims::default();
    let challenges = IncClaimReductionChallenges {
        gamma: Fr::from_u64(101),
    };
    let inputs = || ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        BackendKind::Optimized => JoltBackend::<Fr, DoryScheme>::optimized(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    warm_shared_witness(&mut session, witness, backend, f.log_t);
    let start = Instant::now();
    let mut kernel = selected
        .inc_claim_reduction
        .prepare(&mut session, witness, inputs())
        .expect("prepare the stage-6b increment claim-reduction kernel");
    let prepare = start.elapsed();

    drive_rounds(&mut *kernel, &claims, log_t, log_t, prepare, |_| {
        RoundPhase::Cycle
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
pub fn measure_ram_hamming_booleanity(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    let Parts {
        program_preprocessing: _,
        config: _,
        log_t,
        ..
    } = f.parts();

    let stage1_cycle_binding: Vec<Fr> = (0..log_t)
        .map(|i| Fr::from_u64(23 + 3 * i as u64))
        .collect();
    let relation =
        RamHammingBooleanity::<Fr>::new(TraceDimensions::new(log_t), stage1_cycle_binding);

    let claims = RamHammingBooleanityInputClaims::default();
    let points = RamHammingBooleanityInputClaims::default();
    let challenges = NoChallenges::default();
    let inputs = || ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        BackendKind::Optimized => JoltBackend::<Fr, DoryScheme>::optimized(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    warm_shared_witness(&mut session, witness, backend, f.log_t);
    let start = Instant::now();
    let mut kernel = selected
        .ram_hamming_booleanity
        .prepare(&mut session, witness, inputs())
        .expect("prepare the stage-6b RAM Hamming-booleanity kernel");
    let prepare = start.elapsed();

    drive_rounds(&mut *kernel, &claims, log_t, log_t, prepare, |_| {
        RoundPhase::Cycle
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
pub fn measure_registers_val_evaluation(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    let Parts {
        program_preprocessing: _,
        config: _,
        log_t,
        ..
    } = f.parts();

    let relation = RegistersValEvaluation::<Fr>::new(TraceDimensions::new(log_t));
    let point: Vec<Fr> = (0..REGISTER_ADDRESS_BITS + log_t)
        .map(|i| Fr::from_u64(31 + 7 * i as u64))
        .collect();

    let claims = RegistersValEvaluationInputClaims {
        registers_val: Fr::from_u64(0),
    };
    let points = RegistersValEvaluationInputClaims {
        registers_val: point,
    };
    let challenges = NoChallenges::default();
    let inputs = || ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        BackendKind::Optimized => JoltBackend::<Fr, DoryScheme>::optimized(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    warm_shared_witness(&mut session, witness, backend, f.log_t);
    let start = Instant::now();
    let mut kernel = selected
        .registers_val_evaluation
        .prepare(&mut session, witness, inputs())
        .expect("prepare the stage-5 registers value-evaluation kernel");
    let prepare = start.elapsed();

    drive_rounds(&mut *kernel, &claims, log_t, log_t, prepare, |_| {
        RoundPhase::Cycle
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
pub fn measure_ram_ra_claim_reduction(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    let Parts {
        program_preprocessing: _,
        config,
        log_t,
        ..
    } = f.parts();

    let ram_log_k = config.ram_K.ilog2() as usize;
    let relation = RamRaClaimReduction::<Fr>::new(TraceDimensions::new(log_t), ram_log_k);
    let point = |offset: u64| -> Vec<Fr> {
        (0..ram_log_k + log_t)
            .map(|i| Fr::from_u64(31 + 7 * i as u64 + offset))
            .collect()
    };

    let claims = RamRaClaimReductionInputClaims {
        raf: Fr::from_u64(0),
        read_write: Fr::from_u64(0),
        val_check: Fr::from_u64(0),
    };
    let points = RamRaClaimReductionInputClaims {
        raf: point(0),
        read_write: point(101),
        val_check: point(202),
    };
    let challenges = RamRaClaimReductionChallenges {
        gamma: Fr::from_u64(101),
    };
    let inputs = || ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        BackendKind::Optimized => JoltBackend::<Fr, DoryScheme>::optimized(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    warm_shared_witness(&mut session, witness, backend, f.log_t);
    let start = Instant::now();
    let mut kernel = selected
        .ram_ra_claim_reduction
        .prepare(&mut session, witness, inputs())
        .expect("prepare the stage-5 RAM ra claim-reduction kernel");
    let prepare = start.elapsed();

    drive_rounds(&mut *kernel, &claims, log_t, log_t, prepare, |_| {
        RoundPhase::Cycle
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
pub fn measure_hamming_weight_claim_reduction(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    let Parts {
        program_preprocessing,
        config,
        log_t,
        ..
    } = f.parts();

    let chunk_bits = config.one_hot_config.committed_chunk_bits();
    let layout = formula_dimensions_from_parts(
        config.one_hot_config,
        log_t,
        program_preprocessing.bytecode.code_size,
        config.ram_K,
        JoltRelationId::HammingWeightClaimReduction,
    )
    .expect("formula dimensions")
    .ra_layout;
    let dimensions = HammingWeightClaimReductionDimensions::new(layout, chunk_bits);

    let r_cycle: Vec<Fr> = (0..log_t)
        .map(|i| Fr::from_u64(37 + 7 * i as u64))
        .collect();
    let r_address: Vec<Fr> = (0..dimensions.log_k_chunk)
        .map(|i| Fr::from_u64(23 + 3 * i as u64))
        .collect();
    let virtualization_points: Vec<Vec<Fr>> = (0..dimensions.layout.total())
        .map(|leg| {
            (0..dimensions.log_k_chunk)
                .map(|i| Fr::from_u64(29 + 5 * i as u64 + 101 * leg as u64))
                .collect()
        })
        .collect();
    let relation = HammingWeightClaimReduction::<Fr>::new(
        dimensions,
        r_cycle,
        r_address,
        virtualization_points,
    );

    let claims = HammingWeightClaimReductionInputClaims::default();
    let points = HammingWeightClaimReductionInputClaims::default();
    let challenges = HammingWeightClaimReductionChallenges {
        gamma: Fr::from_u64(101),
    };
    let inputs = || ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        BackendKind::Optimized => JoltBackend::<Fr, DoryScheme>::optimized(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    warm_shared_witness(&mut session, witness, backend, f.log_t);
    let start = Instant::now();
    let mut kernel = selected
        .hamming_weight_claim_reduction
        .prepare(&mut session, witness, inputs())
        .expect("prepare the stage-7 Hamming-weight claim-reduction kernel");
    let prepare = start.elapsed();

    let rounds = jolt_claims::SymbolicSumcheck::rounds(
        jolt_verifier::stages::relations::ConcreteSumcheck::symbolic(&relation),
    );
    drive_rounds(&mut *kernel, &claims, rounds, log_t, prepare, |_| {
        RoundPhase::Address
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
pub fn measure_ram_raf_evaluation(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    let Parts {
        program_preprocessing: _,
        config,
        log_t,
        ..
    } = f.parts();

    let lowest_address = f.device().memory_layout.get_lowest_address();

    let ram_log_k = config.ram_K.ilog2() as usize;
    let read_write_dimensions = config.rw_config.ram_dimensions(log_t, ram_log_k);
    let raf_dimensions = RamRafEvaluationDimensions::try_from(read_write_dimensions)
        .expect("RAM RAF evaluation dimensions");
    let tau_low: Vec<Fr> = (0..log_t)
        .map(|i| Fr::from_u64(23 + 3 * i as u64))
        .collect();
    let relation = RamRafEvaluation::<Fr>::new(
        read_write_dimensions,
        raf_dimensions,
        ram_log_k,
        lowest_address,
        tau_low,
    );

    let claims = RamRafEvaluationInputClaims {
        ram_address: Fr::from_u64(0),
    };
    let points = RamRafEvaluationInputClaims {
        ram_address: Vec::new(),
    };
    let challenges = NoChallenges::default();
    let inputs = || ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        BackendKind::Optimized => JoltBackend::<Fr, DoryScheme>::optimized(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    warm_shared_witness(&mut session, witness, backend, f.log_t);
    let start = Instant::now();
    let mut kernel = selected
        .ram_raf_evaluation
        .prepare(&mut session, witness, inputs())
        .expect("prepare the stage-2 RAM RAF-evaluation kernel");
    let prepare = start.elapsed();

    let rounds = jolt_claims::SymbolicSumcheck::rounds(
        jolt_verifier::stages::relations::ConcreteSumcheck::symbolic(&relation),
    );
    drive_rounds(&mut *kernel, &claims, rounds, log_t, prepare, |_| {
        RoundPhase::Address
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
pub fn measure_ram_output_check(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    let Parts {
        program_preprocessing: _,
        config,
        log_t,
        ..
    } = f.parts();

    let public_memory = PublicIoMemory::new(f.device()).expect("public IO memory");

    let ram_log_k = config.ram_K.ilog2() as usize;
    let read_write_dimensions = config.rw_config.ram_dimensions(log_t, ram_log_k);
    let relation = RamOutputCheck::<Fr>::new(read_write_dimensions, public_memory);

    let claims = RamOutputCheckInputClaims::default();
    let points = RamOutputCheckInputClaims::default();
    let challenges = RamOutputCheckChallenges {
        output_address: (0..ram_log_k)
            .map(|i| Fr::from_u64(23 + 3 * i as u64))
            .collect(),
    };
    let inputs = || ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        BackendKind::Optimized => JoltBackend::<Fr, DoryScheme>::optimized(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    warm_shared_witness(&mut session, witness, backend, f.log_t);
    let start = Instant::now();
    let mut kernel = selected
        .ram_output_check
        .prepare(&mut session, witness, inputs())
        .expect("prepare the stage-2 RAM output-check kernel");
    let prepare = start.elapsed();

    let rounds = jolt_claims::SymbolicSumcheck::rounds(
        jolt_verifier::stages::relations::ConcreteSumcheck::symbolic(&relation),
    );
    drive_rounds(&mut *kernel, &claims, rounds, log_t, prepare, |_| {
        RoundPhase::Address
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
pub fn measure_booleanity_address(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    let Parts {
        program_preprocessing,
        config,
        log_t,
        ..
    } = f.parts();

    let chunk_bits = config.one_hot_config.committed_chunk_bits();
    let layout = formula_dimensions_from_parts(
        config.one_hot_config,
        log_t,
        program_preprocessing.bytecode.code_size,
        config.ram_K,
        JoltRelationId::Booleanity,
    )
    .expect("formula dimensions")
    .ra_layout;
    let dimensions = BooleanityDimensions::new(layout, log_t, chunk_bits);

    let instruction_r_address: Vec<Fr> = (0..chunk_bits)
        .map(|i| Fr::from_u64(23 + 3 * i as u64))
        .collect();
    let instruction_r_cycle: Vec<Fr> = (0..log_t)
        .map(|i| Fr::from_u64(37 + 7 * i as u64))
        .collect();
    let relation =
        BooleanityAddressPhase::<Fr>::new(dimensions, instruction_r_address, instruction_r_cycle);

    let claims = BooleanityAddressPhaseInputClaims::default();
    let points = BooleanityAddressPhaseInputClaims::default();
    let challenges = BooleanityAddressPhaseChallenges {
        reference_address: (0..chunk_bits)
            .map(|i| Fr::from_u64(29 + 5 * i as u64))
            .collect(),
        gamma: Fr::from_u64(101),
    };
    let inputs = || ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        BackendKind::Optimized => JoltBackend::<Fr, DoryScheme>::optimized(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    warm_shared_witness(&mut session, witness, backend, f.log_t);
    let start = Instant::now();
    let mut kernel = selected
        .booleanity_address
        .prepare(&mut session, witness, inputs())
        .expect("prepare the stage-6a booleanity address-phase kernel");
    let prepare = start.elapsed();

    let rounds = jolt_claims::SymbolicSumcheck::rounds(
        jolt_verifier::stages::relations::ConcreteSumcheck::symbolic(&relation),
    );
    drive_rounds(&mut *kernel, &claims, rounds, log_t, prepare, |_| {
        RoundPhase::Address
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
pub fn measure_bytecode_read_raf_address(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    let Parts {
        program_preprocessing,
        config,
        log_t,
        ..
    } = f.parts();

    let dimensions = formula_dimensions_from_parts(
        config.one_hot_config,
        log_t,
        program_preprocessing.bytecode.code_size,
        config.ram_K,
        JoltRelationId::BytecodeReadRaf,
    )
    .expect("formula dimensions")
    .bytecode_read_raf;
    let log_k = dimensions.log_k();

    let stage_cycle_points: [Vec<Fr>; READ_RAF_CYCLE_STAGES] = core::array::from_fn(|stage| {
        (0..log_t)
            .map(|i| Fr::from_u64(37 + 7 * i as u64 + 101 * stage as u64))
            .collect()
    });
    let register_point = |offset: u64| -> Vec<Fr> {
        (0..REGISTER_ADDRESS_BITS + log_t)
            .map(|i| Fr::from_u64(31 + 7 * i as u64 + offset))
            .collect()
    };
    let relation = BytecodeReadRafAddressPhase::<Fr>::new(
        dimensions,
        false,
        BytecodeStagePoints {
            stage_cycle_points,
            register_read_write_point: register_point(0),
            register_val_evaluation_point: register_point(101),
        },
        (1usize << log_k) - 2,
    );

    let claims = BytecodeReadRafAddressPhaseInputClaims::default();
    let points = BytecodeReadRafAddressPhaseInputClaims::default();
    let challenges = BytecodeReadRafAddressPhaseChallenges {
        gamma: Fr::from_u64(101),
        stage1_gamma: Fr::from_u64(103),
        stage2_gamma: Fr::from_u64(107),
        stage3_gamma: Fr::from_u64(109),
        stage4_gamma: Fr::from_u64(113),
        stage5_gamma: Fr::from_u64(127),
    };
    let inputs = || ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        BackendKind::Optimized => JoltBackend::<Fr, DoryScheme>::optimized(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    warm_shared_witness(&mut session, witness, backend, f.log_t);
    let start = Instant::now();
    let mut kernel = selected
        .bytecode_read_raf_address
        .prepare(&mut session, witness, inputs())
        .expect("prepare the stage-6a bytecode read-RAF address-phase kernel");
    let prepare = start.elapsed();

    let rounds = jolt_claims::SymbolicSumcheck::rounds(
        jolt_verifier::stages::relations::ConcreteSumcheck::symbolic(&relation),
    );
    drive_rounds(&mut *kernel, &claims, rounds, log_t, prepare, |_| {
        RoundPhase::Address
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
pub fn measure_spartan_product(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    let Parts {
        program_preprocessing: _,
        config: _,
        log_t,
        ..
    } = f.parts();

    let tau_low: Vec<Fr> = (0..log_t)
        .map(|i| Fr::from_u64(23 + 3 * i as u64))
        .collect();
    let tau_high = Fr::from_u64(97);
    let uniskip_challenge = Fr::from_u64(101);

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        BackendKind::Optimized => JoltBackend::<Fr, DoryScheme>::optimized(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    warm_shared_witness(&mut session, witness, backend, f.log_t);
    let start = Instant::now();
    selected
        .spartan_product_uniskip
        .prepare(&mut session, log_t, &tau_low, witness)
        .expect("prepare the stage-2 Spartan product uni-skip kernel");
    let prepare = start.elapsed();

    let start = Instant::now();
    let _ = selected
        .spartan_product_uniskip
        .first_round_poly(&mut session, &[tau_high])
        .expect("the stage-2 Spartan product uni-skip first-round polynomial");
    let uniskip_poly = start.elapsed();

    let relation = ProductRemainder::new(
        SpartanProductDimensions::new(log_t),
        uniskip_challenge,
        tau_high,
        tau_low,
    );
    let claims = product_remainder_input_values_from_uniskip_output(Fr::from_u64(0));
    let points = ProductRemainderInputClaims {
        product_uniskip: Vec::new(),
    };
    let challenges = NoChallenges::default();
    let inputs = ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let start = Instant::now();
    let mut kernel = selected
        .spartan_product_remainder
        .prepare(&mut session, witness, inputs)
        .expect("prepare the stage-2 Spartan product remainder kernel");
    let remainder_prepare = start.elapsed();

    let rounds = jolt_claims::SymbolicSumcheck::rounds(
        jolt_verifier::stages::relations::ConcreteSumcheck::symbolic(&relation),
    );
    let mut timing = drive_rounds(&mut *kernel, &claims, rounds, log_t, prepare, |_| {
        RoundPhase::Cycle
    });
    timing.handoff = uniskip_poly;
    timing.address = remainder_prepare;
    timing
}

fn with_precommitted_fixture<T>(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    body: impl FnOnce(&dyn JoltWitnessPlane<Fr>, &PrecommittedSchedule, PrecommittedGeometry) -> T,
) -> T {
    let schedule = f.precommitted_schedule();
    let geometry = PrecommittedGeometry {
        log_t: f.log_t,
        ram_log_k: f.config.ram_K.ilog2() as usize,
        bytecode_chunk_count: f.bytecode_chunk_count,
    };
    body(witness, &schedule, geometry)
}

#[derive(Clone, Copy)]
pub struct PrecommittedGeometry {
    log_t: usize,
    ram_log_k: usize,
    bytecode_chunk_count: usize,
}

fn with_commit_fixture<T>(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    body: impl FnOnce(&dyn JoltWitnessPlane<Fr>, CommitmentGrid, &ProverConfig) -> T,
) -> T {
    body(witness, f.commitment_grid(), &f.config)
}

#[expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "measurement harness: kernel errors fail loudly and geometry is reported to stdout"
)]
pub fn measure_commit(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    with_commit_fixture(f, witness, |witness, grid, _config| {
        let ids: Vec<JoltCommittedPolynomial> =
            jolt_witness::JoltWitnessOracle::<Fr>::committed_order(witness)
                .expect("committed order")
                .into_iter()
                .filter(|id| {
                    !matches!(
                        id,
                        JoltCommittedPolynomial::TrustedAdvice
                            | JoltCommittedPolynomial::UntrustedAdvice
                    )
                })
                .collect();
        let setup = DoryScheme::setup_prover(grid.total_vars);
        let selected = selected_backend(backend);
        let mut session = ProofSession::default();
        warm_shared_witness(&mut session, witness, backend, f.log_t);
        let start = Instant::now();
        let committed = selected
            .commit
            .commit_witness(&mut session, witness, &ids, grid, &setup)
            .expect("commit the witness polynomials");
        let elapsed = start.elapsed();
        println!(
            "         {} committed columns, grid {} vars, {} columns per row",
            committed.len(),
            grid.total_vars,
            grid.num_columns(),
        );
        VerticalTiming {
            log_t: grid.log_t,
            prepare: Duration::ZERO,
            address: Duration::ZERO,
            handoff: Duration::ZERO,
            cycle: elapsed,
            claims: Duration::ZERO,
        }
    })
}

#[expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "measurement harness: kernel errors fail loudly and geometry is reported to stdout"
)]
pub fn measure_joint_opening(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    with_commit_fixture(f, witness, |witness, grid, _config| {
        let order: Vec<JoltCommittedPolynomial> =
            jolt_witness::JoltWitnessOracle::<Fr>::committed_order(witness)
                .expect("committed order")
                .into_iter()
                .filter(|id| {
                    !matches!(
                        id,
                        JoltCommittedPolynomial::TrustedAdvice
                            | JoltCommittedPolynomial::UntrustedAdvice
                    )
                })
                .collect();
        let tables = std::collections::BTreeMap::new();
        let selected = selected_backend(backend);
        let mut session = ProofSession::default();
        warm_shared_witness(&mut session, witness, backend, f.log_t);
        let setup = DoryScheme::setup_prover(grid.total_vars);
        let _ = selected
            .commit
            .commit_witness(&mut session, witness, &order, grid, &setup)
            .expect("warm the session the way stage 0 does before stage 8 runs");

        let start = Instant::now();
        let polynomials = selected
            .joint_opening
            .prepare(&mut session, witness, &order, &tables, grid)
            .expect("prepare the joint-opening polynomials");
        let prepare = start.elapsed();

        let sigma = grid.total_vars.div_ceil(2);
        let left = synthetic_point(1usize << (grid.total_vars - sigma), 17);
        let start = Instant::now();
        for polynomial in &polynomials {
            let folded = polynomial.fold_rows(&left, sigma);
            assert_eq!(folded.len(), 1usize << sigma, "fold width");
        }
        let fold = start.elapsed();

        println!(
            "         {} polynomials, grid {} vars, sigma {}",
            polynomials.len(),
            grid.total_vars,
            sigma,
        );
        VerticalTiming {
            log_t: grid.log_t,
            prepare,
            address: Duration::ZERO,
            handoff: Duration::ZERO,
            cycle: fold,
            claims: Duration::ZERO,
        }
    })
}

#[expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "measurement harness: kernel errors fail loudly and geometry is reported to stdout"
)]
pub fn measure_advice_opening(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
    kind: JoltAdviceKind,
) -> VerticalTiming {
    with_precommitted_fixture(f, witness, |witness, schedule, geometry| {
        let layout = match kind {
            JoltAdviceKind::Trusted => schedule.trusted_advice.as_ref(),
            JoltAdviceKind::Untrusted => schedule.untrusted_advice.as_ref(),
        }
        .expect("advice layout present");
        let vars = layout
            .precommitted()
            .poly_opening_round_permutation_be()
            .len();
        let point = synthetic_point(vars, 29);
        let selected = selected_backend(backend);
        let mut session = ProofSession::default();
        warm_shared_witness(&mut session, witness, backend, f.log_t);
        let start = Instant::now();
        let value = selected
            .advice_opening
            .evaluate(&mut session, kind, &point, witness)
            .expect("evaluate the advice opening");
        let elapsed = start.elapsed();
        println!(
            "         {kind:?} advice, {vars} vars, value nonzero = {}",
            value != Fr::from_u64(0)
        );
        VerticalTiming {
            log_t: geometry.log_t,
            prepare: Duration::ZERO,
            address: Duration::ZERO,
            handoff: Duration::ZERO,
            cycle: elapsed,
            claims: Duration::ZERO,
        }
    })
}

fn selected_backend(backend: BackendKind) -> JoltBackend<Fr, DoryScheme> {
    match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        BackendKind::Optimized => JoltBackend::<Fr, DoryScheme>::optimized(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    }
}

fn synthetic_point(len: usize, seed: u64) -> Vec<Fr> {
    (0..len)
        .map(|i| Fr::from_u64(seed + 7 * i as u64 + 1))
        .collect()
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: schedule errors fail loudly"
)]
fn synthetic_cycle_variables(reduction: &PrecommittedClaimReduction) -> Vec<Fr> {
    let challenges = synthetic_point(reduction.cycle_phase_total_rounds(), 97);
    reduction
        .cycle_phase_variable_challenges(&challenges)
        .expect("cycle-phase variable challenges")
}

fn bytecode_reduction_weights_fixture(
    layout: &BytecodeClaimReductionLayout,
) -> BytecodeReductionWeights<Fr> {
    BytecodeReductionWeights {
        r_bc: synthetic_point(layout.log_bytecode_chunk_size(), 31),
        chunk_rbc_weights: synthetic_point(layout.chunk_count(), 53),
        lane_weights: synthetic_point(COMMITTED_BYTECODE_LANE_CAPACITY, 71),
    }
}

#[expect(
    clippy::print_stdout,
    reason = "measurement harness: reports to stdout like the surrounding arms"
)]
fn absent_address_phase(
    label: &str,
    reduction: &PrecommittedClaimReduction,
    geometry: PrecommittedGeometry,
) -> VerticalTiming {
    println!(
        "  {label}: NO ADDRESS PHASE at log_T={} — the polynomial has {} variables and \
         {}/{} active cycle rounds, so none of its variables land in the schedule's top \
         {} address rounds; the reduction finalizes in the cycle phase and stage 7 has no \
         member for it. Nothing to measure.",
        geometry.log_t,
        reduction.poly_opening_round_permutation_be().len(),
        reduction.cycle_phase_rounds().len(),
        reduction.cycle_phase_total_rounds(),
        reduction.address_phase_total_rounds(),
    );
    VerticalTiming {
        log_t: geometry.log_t,
        prepare: Duration::ZERO,
        address: Duration::ZERO,
        handoff: Duration::ZERO,
        cycle: Duration::ZERO,
        claims: Duration::ZERO,
    }
}

#[expect(
    clippy::print_stdout,
    reason = "measurement harness: reports to stdout like the surrounding arms"
)]
fn report_precommitted_geometry(
    label: &str,
    reduction: &PrecommittedClaimReduction,
    tables: usize,
) {
    let vars = reduction.poly_opening_round_permutation_be().len();
    println!(
        "  {label}: table 2^{vars} = {} coefficients x {tables} tables ({:.2} MiB), \
         cycle {}/{} active, address {}/{} active",
        1usize << vars,
        ((tables << vars) * 32) as f64 / (1024.0 * 1024.0),
        reduction.cycle_phase_rounds().len(),
        reduction.cycle_phase_total_rounds(),
        reduction.address_phase_rounds().len(),
        reduction.address_phase_total_rounds(),
    );
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
pub fn measure_advice_cycle(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
    kind: JoltAdviceKind,
) -> VerticalTiming {
    with_precommitted_fixture(f, witness, |witness, schedule, geometry| {
        let layout: &AdviceClaimReductionLayout =
            schedule.advice(kind).expect("advice layout present");
        let reduction = layout.precommitted();
        let rounds = reduction.cycle_phase_total_rounds();
        let r_val = synthetic_point(reduction.poly_opening_round_permutation_be().len(), 11);
        let selected = selected_backend(backend);
        let challenges = NoChallenges::default();
        let mut session = ProofSession::default();
        warm_shared_witness(&mut session, witness, backend, f.log_t);
        report_precommitted_geometry(
            match kind {
                JoltAdviceKind::Trusted => "trusted-advice-cycle",
                JoltAdviceKind::Untrusted => "untrusted-advice-cycle",
            },
            reduction,
            2,
        );

        match kind {
            JoltAdviceKind::Trusted => {
                let relation = TrustedAdviceCyclePhase::<Fr>::new(layout, Some(r_val));
                let claims = TrustedAdviceCyclePhaseInputClaims::default();
                let points = TrustedAdviceCyclePhaseInputClaims::default();
                let start = Instant::now();
                let mut kernel = selected
                    .trusted_advice_cycle
                    .prepare(
                        &mut session,
                        witness,
                        ProverInputs {
                            relation: &relation,
                            claims: &claims,
                            points: &points,
                            challenges: &challenges,
                        },
                    )
                    .expect("prepare the stage-6b trusted-advice cycle-phase kernel");
                let prepare = start.elapsed();
                drive_rounds(
                    &mut *kernel,
                    &claims,
                    rounds,
                    geometry.log_t,
                    prepare,
                    |_| RoundPhase::Cycle,
                )
            }
            JoltAdviceKind::Untrusted => {
                let relation = UntrustedAdviceCyclePhase::<Fr>::new(layout, Some(r_val));
                let claims = UntrustedAdviceCyclePhaseInputClaims::default();
                let points = UntrustedAdviceCyclePhaseInputClaims::default();
                let start = Instant::now();
                let mut kernel = selected
                    .untrusted_advice_cycle
                    .prepare(
                        &mut session,
                        witness,
                        ProverInputs {
                            relation: &relation,
                            claims: &claims,
                            points: &points,
                            challenges: &challenges,
                        },
                    )
                    .expect("prepare the stage-6b untrusted-advice cycle-phase kernel");
                let prepare = start.elapsed();
                drive_rounds(
                    &mut *kernel,
                    &claims,
                    rounds,
                    geometry.log_t,
                    prepare,
                    |_| RoundPhase::Cycle,
                )
            }
        }
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
pub fn measure_advice_address(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
    kind: JoltAdviceKind,
) -> VerticalTiming {
    with_precommitted_fixture(f, witness, |witness, schedule, geometry| {
        let layout: &AdviceClaimReductionLayout =
            schedule.advice(kind).expect("advice layout present");
        let reduction = layout.precommitted();
        let label = match kind {
            JoltAdviceKind::Trusted => "trusted-advice-address",
            JoltAdviceKind::Untrusted => "untrusted-advice-address",
        };
        if reduction.num_address_phase_rounds() == 0 {
            return absent_address_phase(label, reduction, geometry);
        }
        report_precommitted_geometry(label, reduction, 2);

        let cycle_rounds = reduction.cycle_phase_total_rounds();
        let address_rounds = reduction.address_phase_total_rounds();
        let cycle_variables = synthetic_cycle_variables(reduction);
        let r_val = synthetic_point(reduction.poly_opening_round_permutation_be().len(), 11);
        let selected = selected_backend(backend);
        let challenges = NoChallenges::default();
        let mut session = ProofSession::default();
        warm_shared_witness(&mut session, witness, backend, f.log_t);

        match kind {
            JoltAdviceKind::Trusted => {
                let cycle_relation =
                    TrustedAdviceCyclePhase::<Fr>::new(layout, Some(r_val.clone()));
                let cycle_claims = TrustedAdviceCyclePhaseInputClaims::default();
                let cycle_points = TrustedAdviceCyclePhaseInputClaims::default();
                let cycle_kernel = selected
                    .trusted_advice_cycle
                    .prepare(
                        &mut session,
                        witness,
                        ProverInputs {
                            relation: &cycle_relation,
                            claims: &cycle_claims,
                            points: &cycle_points,
                            challenges: &challenges,
                        },
                    )
                    .expect("prepare the stage-6b trusted-advice cycle-phase kernel");
                let mut cycle_kernel = cycle_kernel;
                let _ = drive_rounds(
                    &mut *cycle_kernel,
                    &cycle_claims,
                    cycle_rounds,
                    geometry.log_t,
                    Duration::ZERO,
                    |_| RoundPhase::Cycle,
                );
                cycle_kernel.park_residue(&mut session);

                let relation =
                    TrustedAdviceAddressPhase::<Fr>::new(layout, Some(r_val), cycle_variables);
                let claims = TrustedAdviceAddressPhaseInputClaims::default();
                let points = TrustedAdviceAddressPhaseInputClaims::default();
                let start = Instant::now();
                let mut kernel = selected
                    .trusted_advice_address
                    .prepare(
                        &mut session,
                        witness,
                        ProverInputs {
                            relation: &relation,
                            claims: &claims,
                            points: &points,
                            challenges: &challenges,
                        },
                    )
                    .expect("prepare the stage-7 trusted-advice address-phase kernel");
                let prepare = start.elapsed();
                drive_rounds(
                    &mut *kernel,
                    &claims,
                    address_rounds,
                    geometry.log_t,
                    prepare,
                    |_| RoundPhase::Address,
                )
            }
            JoltAdviceKind::Untrusted => {
                let cycle_relation =
                    UntrustedAdviceCyclePhase::<Fr>::new(layout, Some(r_val.clone()));
                let cycle_claims = UntrustedAdviceCyclePhaseInputClaims::default();
                let cycle_points = UntrustedAdviceCyclePhaseInputClaims::default();
                let mut cycle_kernel = selected
                    .untrusted_advice_cycle
                    .prepare(
                        &mut session,
                        witness,
                        ProverInputs {
                            relation: &cycle_relation,
                            claims: &cycle_claims,
                            points: &cycle_points,
                            challenges: &challenges,
                        },
                    )
                    .expect("prepare the stage-6b untrusted-advice cycle-phase kernel");
                let _ = drive_rounds(
                    &mut *cycle_kernel,
                    &cycle_claims,
                    cycle_rounds,
                    geometry.log_t,
                    Duration::ZERO,
                    |_| RoundPhase::Cycle,
                );
                cycle_kernel.park_residue(&mut session);

                let relation =
                    UntrustedAdviceAddressPhase::<Fr>::new(layout, Some(r_val), cycle_variables);
                let claims = UntrustedAdviceAddressPhaseInputClaims::default();
                let points = UntrustedAdviceAddressPhaseInputClaims::default();
                let start = Instant::now();
                let mut kernel = selected
                    .untrusted_advice_address
                    .prepare(
                        &mut session,
                        witness,
                        ProverInputs {
                            relation: &relation,
                            claims: &claims,
                            points: &points,
                            challenges: &challenges,
                        },
                    )
                    .expect("prepare the stage-7 untrusted-advice address-phase kernel");
                let prepare = start.elapsed();
                drive_rounds(
                    &mut *kernel,
                    &claims,
                    address_rounds,
                    geometry.log_t,
                    prepare,
                    |_| RoundPhase::Address,
                )
            }
        }
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
pub fn measure_bytecode_reduction_cycle(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    with_precommitted_fixture(f, witness, |witness, schedule, geometry| {
        let layout: &BytecodeClaimReductionLayout = schedule
            .bytecode
            .as_ref()
            .expect("committed bytecode layout present");
        let reduction = layout.precommitted();
        report_precommitted_geometry(
            "bytecode-reduction-cycle",
            reduction,
            2 + geometry.bytecode_chunk_count,
        );

        let relation = BytecodeReductionCyclePhase::<Fr>::new(
            layout,
            bytecode_reduction_weights_fixture(layout),
        );
        let claims = BytecodeReductionCyclePhaseInputClaims::default();
        let points = BytecodeReductionCyclePhaseInputClaims::default();
        let challenges = BytecodeReductionCyclePhaseChallenges {
            eta: Fr::from_u64(101),
        };
        let selected = selected_backend(backend);
        let mut session = ProofSession::default();
        warm_shared_witness(&mut session, witness, backend, f.log_t);
        let start = Instant::now();
        let mut kernel = selected
            .bytecode_reduction_cycle
            .prepare(
                &mut session,
                witness,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenges,
                },
            )
            .expect("prepare the stage-6b committed-bytecode cycle-phase kernel");
        let prepare = start.elapsed();
        drive_rounds(
            &mut *kernel,
            &claims,
            reduction.cycle_phase_total_rounds(),
            geometry.log_t,
            prepare,
            |_| RoundPhase::Cycle,
        )
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
pub fn measure_bytecode_reduction_address(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    with_precommitted_fixture(f, witness, |witness, schedule, geometry| {
        let layout: &BytecodeClaimReductionLayout = schedule
            .bytecode
            .as_ref()
            .expect("committed bytecode layout present");
        let reduction = layout.precommitted();
        if reduction.num_address_phase_rounds() == 0 {
            return absent_address_phase("bytecode-reduction-address", reduction, geometry);
        }
        report_precommitted_geometry(
            "bytecode-reduction-address",
            reduction,
            2 + geometry.bytecode_chunk_count,
        );

        let weights = bytecode_reduction_weights_fixture(layout);
        let cycle_relation = BytecodeReductionCyclePhase::<Fr>::new(layout, weights.clone());
        let cycle_claims = BytecodeReductionCyclePhaseInputClaims::default();
        let cycle_points = BytecodeReductionCyclePhaseInputClaims::default();
        let cycle_challenges = BytecodeReductionCyclePhaseChallenges {
            eta: Fr::from_u64(101),
        };
        let selected = selected_backend(backend);
        let mut session = ProofSession::default();
        warm_shared_witness(&mut session, witness, backend, f.log_t);
        let mut cycle_kernel = selected
            .bytecode_reduction_cycle
            .prepare(
                &mut session,
                witness,
                ProverInputs {
                    relation: &cycle_relation,
                    claims: &cycle_claims,
                    points: &cycle_points,
                    challenges: &cycle_challenges,
                },
            )
            .expect("prepare the stage-6b committed-bytecode cycle-phase kernel");
        let _ = drive_rounds(
            &mut *cycle_kernel,
            &cycle_claims,
            reduction.cycle_phase_total_rounds(),
            geometry.log_t,
            Duration::ZERO,
            |_| RoundPhase::Cycle,
        );
        cycle_kernel.park_residue(&mut session);

        let relation = BytecodeReductionAddressPhase::<Fr>::new(
            layout,
            Some(weights),
            synthetic_cycle_variables(reduction),
        );
        let claims = BytecodeReductionAddressPhaseInputClaims::default();
        let points = BytecodeReductionAddressPhaseInputClaims::default();
        let challenges = NoChallenges::default();
        let start = Instant::now();
        let mut kernel = selected
            .bytecode_reduction_address
            .prepare(
                &mut session,
                witness,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenges,
                },
            )
            .expect("prepare the stage-7 committed-bytecode address-phase kernel");
        let prepare = start.elapsed();
        drive_rounds(
            &mut *kernel,
            &claims,
            reduction.address_phase_total_rounds(),
            geometry.log_t,
            prepare,
            |_| RoundPhase::Address,
        )
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
pub fn measure_program_image_reduction_cycle(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    with_precommitted_fixture(f, witness, |witness, schedule, geometry| {
        let layout: &ProgramImageClaimReductionLayout = schedule
            .program_image
            .as_ref()
            .expect("program image layout present");
        let reduction = layout.precommitted();
        report_precommitted_geometry("program-image-reduction-cycle", reduction, 2);

        let relation = ProgramImageReductionCyclePhase::<Fr>::new(
            layout,
            synthetic_point(geometry.ram_log_k, 13),
        );
        let claims = ProgramImageReductionCyclePhaseInputClaims::default();
        let points = ProgramImageReductionCyclePhaseInputClaims::default();
        let challenges = NoChallenges::default();
        let selected = selected_backend(backend);
        let mut session = ProofSession::default();
        warm_shared_witness(&mut session, witness, backend, f.log_t);
        let start = Instant::now();
        let mut kernel = selected
            .program_image_reduction_cycle
            .prepare(
                &mut session,
                witness,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenges,
                },
            )
            .expect("prepare the stage-6b program-image cycle-phase kernel");
        let prepare = start.elapsed();
        drive_rounds(
            &mut *kernel,
            &claims,
            reduction.cycle_phase_total_rounds(),
            geometry.log_t,
            prepare,
            |_| RoundPhase::Cycle,
        )
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
pub fn measure_program_image_reduction_address(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    with_precommitted_fixture(f, witness, |witness, schedule, geometry| {
        let layout: &ProgramImageClaimReductionLayout = schedule
            .program_image
            .as_ref()
            .expect("program image layout present");
        let reduction = layout.precommitted();
        if reduction.num_address_phase_rounds() == 0 {
            return absent_address_phase("program-image-reduction-address", reduction, geometry);
        }
        report_precommitted_geometry("program-image-reduction-address", reduction, 2);

        let r_addr_rw = synthetic_point(geometry.ram_log_k, 13);
        let cycle_relation = ProgramImageReductionCyclePhase::<Fr>::new(layout, r_addr_rw.clone());
        let cycle_claims = ProgramImageReductionCyclePhaseInputClaims::default();
        let cycle_points = ProgramImageReductionCyclePhaseInputClaims::default();
        let challenges = NoChallenges::default();
        let selected = selected_backend(backend);
        let mut session = ProofSession::default();
        warm_shared_witness(&mut session, witness, backend, f.log_t);
        let mut cycle_kernel = selected
            .program_image_reduction_cycle
            .prepare(
                &mut session,
                witness,
                ProverInputs {
                    relation: &cycle_relation,
                    claims: &cycle_claims,
                    points: &cycle_points,
                    challenges: &challenges,
                },
            )
            .expect("prepare the stage-6b program-image cycle-phase kernel");
        let _ = drive_rounds(
            &mut *cycle_kernel,
            &cycle_claims,
            reduction.cycle_phase_total_rounds(),
            geometry.log_t,
            Duration::ZERO,
            |_| RoundPhase::Cycle,
        );
        cycle_kernel.park_residue(&mut session);

        let relation = ProgramImageReductionAddressPhase::<Fr>::new(
            layout,
            Some(r_addr_rw),
            synthetic_cycle_variables(reduction),
        );
        let claims = ProgramImageReductionAddressPhaseInputClaims::default();
        let points = ProgramImageReductionAddressPhaseInputClaims::default();
        let start = Instant::now();
        let mut kernel = selected
            .program_image_reduction_address
            .prepare(
                &mut session,
                witness,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenges,
                },
            )
            .expect("prepare the stage-7 program-image address-phase kernel");
        let prepare = start.elapsed();
        drive_rounds(
            &mut *kernel,
            &claims,
            reduction.address_phase_total_rounds(),
            geometry.log_t,
            prepare,
            |_| RoundPhase::Address,
        )
    })
}

pub fn advice_opening(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    measure_advice_opening(f, witness, backend, JoltAdviceKind::Trusted)
}

pub fn trusted_advice_cycle(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    measure_advice_cycle(f, witness, backend, JoltAdviceKind::Trusted)
}

pub fn untrusted_advice_cycle(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    measure_advice_cycle(f, witness, backend, JoltAdviceKind::Untrusted)
}

pub fn trusted_advice_address(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    measure_advice_address(f, witness, backend, JoltAdviceKind::Trusted)
}

pub fn untrusted_advice_address(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    measure_advice_address(f, witness, backend, JoltAdviceKind::Untrusted)
}

pub fn warm_shared_witness(
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
    log_t: usize,
) {
    match backend {
        BackendKind::Reference => {}
        BackendKind::Optimized => {
            jolt_kernels::optimized::warm_shared_witness(session, witness, log_t)
                .expect("warm the optimized shared witness state");
        }
        BackendKind::Cuda => {
            jolt_kernels::cuda::warm_shared_witness(session, witness, log_t)
                .expect("warm the cuda shared witness state");
        }
    }
}

pub fn measure_witness_generation(
    f: &Fixture,
    witness: &dyn JoltWitnessPlane<Fr>,
    backend: BackendKind,
) -> VerticalTiming {
    let log_t = f.log_t;
    let mut session = ProofSession::default();
    let start = Instant::now();
    warm_shared_witness(&mut session, witness, backend, log_t);
    VerticalTiming {
        log_t,
        prepare: start.elapsed(),
        address: Duration::ZERO,
        handoff: Duration::ZERO,
        cycle: Duration::ZERO,
        claims: Duration::ZERO,
    }
}
