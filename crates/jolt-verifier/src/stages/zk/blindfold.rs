//! Jolt-specific BlindFold protocol construction.
//!
//! This module lowers the already-verified committed-sumcheck frontier into the
//! generic `jolt-blindfold` statement API. It does not build R1CS matrices
//! directly. Instead, it describes the verifier equations in typed Jolt terms:
//! sumcheck statements, domains, committed round consistency, committed output
//! claim rows, input/output claim expressions, public constants, transcript
//! challenges, and the final PCS opening binding. `jolt-blindfold` then lowers
//! that statement into `jolt-r1cs` constraints and exposes the row layout used
//! by the BlindFold verifier.
//!
//! Each Stage 1-7 entry added here has four pieces:
//!
//! - a `SumcheckStatement` and `SumcheckDomainSpec`, which determine the
//!   generic round-sum and round-evaluation constraints checked for every
//!   committed sumcheck round;
//! - committed consistency data from the stage verifier, which supplies the
//!   committed round polynomial rows, round challenges, degrees, and batching
//!   coefficients already checked against the transcript;
//! - committed output claim rows, whose opening IDs are listed in the exact row
//!   order used by the stage's vector commitments; aliases map duplicate
//!   semantic openings to the same R1CS variable instead of allocating another
//!   hidden value;
//! - `jolt-claims` expressions for the stage input and output claims. These
//!   expressions bind the sumcheck chain endpoints to openings, public values,
//!   and challenges. Public values and challenges are inserted explicitly into
//!   `SourceValues`; missing sources fail construction.
//!
//! Batched stages are represented as one committed sumcheck statement. Their
//! input expression is the batched sum of each component input expression,
//! including the power-of-two scaling required by the batched committed
//! sumcheck. Their output expression is the batching-coefficient-weighted sum
//! of component output expressions. This keeps the hidden claim relation in one
//! place: the same formula metadata used by clear verification is what
//! BlindFold lowers into R1CS.
//!
//! Stage 8 contributes the final opening binding. The verifier has already
//! formed the linear PCS batch, so the BlindFold statement adds one final
//! equation
//!
//! ```text
//! evaluation = sum_i coefficient_i * opening_i
//! ```
//!
//! and carries the hiding evaluation commitment for that evaluation. The
//! generic BlindFold verifier opens the folded witness at the fixed coordinates
//! for the folded evaluation and its blinding, checks that those coordinates
//! equal the folded eval scalars in the proof, checks the opened rows are
//! dedicated to those scalars, and then checks the folded eval commitment. This
//! binds the hidden R1CS witness value used in the Jolt claim relation to the
//! hidden PCS evaluation proved by Stage 8.
//!
//! In ZK mode this is the bridge between the committed stage proofs and the
//! final PCS opening proof: no clear output claim scalars are accepted by the
//! verifier, and every hidden scalar that crosses a stage boundary is either in
//! a committed output-claim row or in the final hiding evaluation commitment.
use jolt_blindfold::{BlindFoldProtocol, BlindFoldProtocolBuilder, OpeningAlias};
use jolt_claims::{
    pow2,
    protocols::jolt::{
        formulas::{
            booleanity::{self, BooleanityDimensions},
            bytecode::{self, BytecodeReadRafEvaluationInputs},
            claim_reductions::{advice, hamming_weight, increments},
            dimensions::{
                AdviceClaimReductionLayout, JoltFormulaDimensions, JoltSumcheckSpec,
                PRODUCT_UNISKIP_DOMAIN_SIZE, REGISTER_ADDRESS_BITS,
            },
            instruction, ram, registers,
            spartan::{
                self, outer_opening, outer_uniskip_opening, product_remainder_output_openings,
                product_uniskip_opening, shift_output_openings, SpartanOuterDimensions,
                SpartanProductDimensions, SpartanProductPublicValues,
            },
        },
        AdviceClaimReductionPublic, BooleanityChallenge, BooleanityPublic,
        BytecodeReadRafChallenge, BytecodeReadRafPublic, HammingWeightClaimReductionChallenge,
        HammingWeightClaimReductionPublic, IncClaimReductionChallenge, IncClaimReductionPublic,
        InstructionClaimReductionChallenge, InstructionInputChallenge,
        InstructionRaVirtualizationChallenge, InstructionReadRafChallenge, JoltAdviceKind,
        JoltChallengeId, JoltOpeningId, JoltPublicId, JoltStageClaims, JoltStageId,
        JoltSumcheckDomain, RamHammingBooleanityPublic, RamOutputCheckPublic,
        RamRaClaimReductionChallenge, RamRaClaimReductionPublic, RamRaVirtualizationPublic,
        RamRafEvaluationPublic, RamReadWriteChallenge, RamValCheckChallenge,
        RegistersClaimReductionChallenge, RegistersReadWriteChallenge,
        RegistersValEvaluationChallenge, SpartanProductVirtualizationPublic, SpartanShiftChallenge,
        SpartanShiftPublic,
    },
    Expr,
};
use jolt_crypto::VectorCommitment;
use jolt_field::{Field, FromPrimitiveInt};
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
use jolt_openings::CommitmentScheme;
use jolt_poly::{
    block_selector_mle_msb,
    lagrange::{centered_lagrange_evals_array, centered_lagrange_kernel},
    range_mask_mle_msb, sparse_segments_mle_msb, try_eq_mle, EqPlusOnePolynomial,
    IdentityPolynomial, LtPolynomial, OperandPolynomial, OperandSide,
};
use jolt_program::preprocess::PublicIoMemory;
use jolt_r1cs::constraints::rv64::Rv64SpartanOuterRemainder;
use jolt_sumcheck::{
    BatchedCommittedSumcheckConsistency, CommittedSumcheckConsistency, SumcheckDomainSpec,
    SumcheckStatement,
};
use num_traits::{One, Zero};

use super::{
    inputs::BlindFoldInputs,
    outputs::{BlindFoldOutput, CommittedOutputClaimOutput},
};
use crate::VerifierError;

type Builder<F, C> = BlindFoldProtocolBuilder<F, JoltOpeningId, C, JoltPublicId, JoltChallengeId>;
type JoltExpr<F> = Expr<F, JoltOpeningId, JoltPublicId, JoltChallengeId>;

#[derive(Default)]
struct SourceValues<F: Field> {
    publics: Vec<(JoltPublicId, F)>,
    challenges: Vec<(JoltChallengeId, F)>,
}

pub(crate) fn build<PCS, VC, ZkProof>(
    input: BlindFoldInputs<'_, PCS, VC, ZkProof>,
) -> Result<BlindFoldOutput<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    VC::Output: Clone,
{
    let mut values = SourceValues::default();
    let mut builder = BlindFoldProtocol::<PCS::Field, VC::Output>::builder::<
        JoltOpeningId,
        JoltPublicId,
        JoltChallengeId,
    >();

    builder = add_stage1(&input, builder, &mut values)?;
    builder = add_stage2(&input, builder, &mut values)?;
    builder = add_stage3(&input, builder, &mut values)?;
    builder = add_stage4(&input, builder, &mut values)?;
    builder = add_stage5(&input, builder, &mut values)?;
    builder = add_stage6(&input, builder, &mut values)?;
    builder = add_stage7(&input, builder, &mut values)?;

    for (id, value) in values.publics {
        builder = builder.public(id, value);
    }
    for (id, value) in values.challenges {
        builder = builder.challenge(id, value);
    }

    let protocol = builder
        .final_opening(
            input.stage8.opening_ids.clone(),
            input.stage8.constraint_coefficients.clone(),
            input.stage8.hiding_evaluation_commitment,
        )
        .build()
        .map_err(blindfold_error)?;

    Ok(BlindFoldOutput { protocol })
}

fn add_stage1<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    mut builder: Builder<PCS::Field, VC::Output>,
    values: &mut SourceValues<PCS::Field>,
) -> Result<Builder<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    VC::Output: Clone,
{
    let log_t = input.checked.trace_length.ilog2() as usize;
    let dimensions = SpartanOuterDimensions::rv64(log_t);
    let uniskip = spartan::outer_uniskip::<PCS::Field>(&dimensions);
    builder = add_single_stage(
        builder,
        "stage1.outer_uniskip",
        &uniskip,
        &input.stage1.uniskip_consistency,
        &input.stage1.uniskip_output_claims,
        values,
        vec![outer_uniskip_opening()],
        Vec::new(),
    )?;

    let remainder = spartan::outer_remainder::<PCS::Field>(&dimensions);
    let remainder_formula = Rv64SpartanOuterRemainder::new(
        &dimensions,
        &input.stage1.public.tau,
        input.stage1.public.uniskip_challenge,
        &input.stage1.public.remainder_challenges,
    )
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltStageId::SpartanOuter,
        reason: error.to_string(),
    })?;
    for (id, value) in remainder_formula
        .public_claims(&dimensions)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltStageId::SpartanOuter,
            reason: error.to_string(),
        })?
    {
        values.public(JoltPublicId::from(id), value)?;
    }
    let remainder_ids = dimensions
        .variables()
        .iter()
        .copied()
        .map(outer_opening)
        .collect::<Vec<_>>();
    add_batched_stage(
        builder,
        "stage1.outer_remainder",
        &[remainder],
        &input.stage1.remainder_consistency,
        &input.stage1.remainder_output_claims,
        values,
        remainder_ids,
        Vec::new(),
    )
}

fn add_stage2<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    mut builder: Builder<PCS::Field, VC::Output>,
    values: &mut SourceValues<PCS::Field>,
) -> Result<Builder<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    VC::Output: Clone,
{
    let log_t = input.checked.trace_length.ilog2() as usize;
    let log_k = input.checked.ram_K.ilog2() as usize;
    let trace_dimensions = jolt_claims::protocols::jolt::TraceDimensions::new(log_t);
    let read_write_dimensions = input.proof.rw_config.ram_dimensions(log_t, log_k);
    let product_dimensions = SpartanProductDimensions::from(log_t);
    let raf_dimensions =
        ram::RamRafEvaluationDimensions::try_from(read_write_dimensions).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltStageId::RamRafEvaluation,
                reason: error.to_string(),
            }
        })?;

    let product_uniskip = spartan::product_uniskip::<PCS::Field>(product_dimensions);
    let product_weights = centered_lagrange_evals_array::<
        PCS::Field,
        { PRODUCT_UNISKIP_DOMAIN_SIZE },
    >(input.stage2.public.product_tau_high)
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltStageId::SpartanProductVirtualization,
        reason: error.to_string(),
    })?;
    for (index, value) in product_weights.into_iter().enumerate() {
        values.public(
            JoltPublicId::from(SpartanProductVirtualizationPublic::LagrangeWeight(index)),
            value,
        )?;
    }
    builder = add_single_stage(
        builder,
        "stage2.product_uniskip",
        &product_uniskip,
        &input.stage2.product_uniskip_consistency,
        &input.stage2.product_uniskip_output_claims,
        values,
        vec![product_uniskip_opening()],
        Vec::new(),
    )?;

    let ram_read_write = ram::read_write_checking::<PCS::Field>(read_write_dimensions);
    let product_remainder = spartan::product_remainder::<PCS::Field>(product_dimensions);
    let instruction_reduction =
        jolt_claims::protocols::jolt::formulas::claim_reductions::instruction::claim_reduction::<
            PCS::Field,
        >(trace_dimensions);
    let ram_raf = ram::raf_evaluation::<PCS::Field>(raf_dimensions);
    let ram_output = ram::output_check::<PCS::Field>(read_write_dimensions);

    let ram_read_write_point = input
        .stage2
        .batch_consistency
        .try_instance_point(ram_read_write.sumcheck.rounds)
        .map_err(|error| stage_sumcheck_error(JoltStageId::RamReadWriteChecking, error))?;
    let ram_read_write_opening = read_write_dimensions
        .read_write_opening_point(&ram_read_write_point)
        .map_err(|error| public_error(JoltStageId::RamReadWriteChecking, error))?;
    let eq_cycle = try_eq_mle(
        &input.stage2.public.product_tau_low,
        &ram_read_write_opening.r_cycle,
    )
    .map_err(|error| public_error(JoltStageId::RamReadWriteChecking, error))?;
    values.challenge(
        JoltChallengeId::from(RamReadWriteChallenge::Gamma),
        input.stage2.public.ram_read_write_gamma,
    )?;
    values.challenge(
        JoltChallengeId::from(RamReadWriteChallenge::EqCycle),
        eq_cycle,
    )?;

    let product_point = input
        .stage2
        .batch_consistency
        .try_instance_point(product_remainder.sumcheck.rounds)
        .map_err(|error| stage_sumcheck_error(JoltStageId::SpartanProductVirtualization, error))?;
    let product_opening_point = product_point.iter().rev().copied().collect::<Vec<_>>();
    let product_tau_high_bound = centered_lagrange_kernel(
        PRODUCT_UNISKIP_DOMAIN_SIZE,
        input.stage2.public.product_tau_high,
        input.stage2.public.product_uniskip_challenge,
    )
    .map_err(|error| public_error(JoltStageId::SpartanProductVirtualization, error))?;
    let product_tau_low_eq =
        try_eq_mle(&input.stage2.public.product_tau_low, &product_opening_point)
            .map_err(|error| public_error(JoltStageId::SpartanProductVirtualization, error))?;
    let product_publics = SpartanProductPublicValues {
        lagrange_weights: centered_lagrange_evals_array::<
            PCS::Field,
            { PRODUCT_UNISKIP_DOMAIN_SIZE },
        >(input.stage2.public.product_uniskip_challenge)
        .map_err(|error| public_error(JoltStageId::SpartanProductVirtualization, error))?,
        tau_kernel: product_tau_high_bound * product_tau_low_eq,
    };
    values.public(
        JoltPublicId::from(SpartanProductVirtualizationPublic::TauKernel),
        product_publics
            .value(SpartanProductVirtualizationPublic::TauKernel)
            .unwrap_or_else(PCS::Field::zero),
    )?;

    let instruction_point = input
        .stage2
        .batch_consistency
        .try_instance_point(instruction_reduction.sumcheck.rounds)
        .map_err(|error| stage_sumcheck_error(JoltStageId::InstructionClaimReduction, error))?;
    let instruction_opening_point = instruction_point.iter().rev().copied().collect::<Vec<_>>();
    let eq_spartan = try_eq_mle(
        &instruction_opening_point,
        &input.stage2.public.product_tau_low,
    )
    .map_err(|error| public_error(JoltStageId::InstructionClaimReduction, error))?;
    values.challenge(
        JoltChallengeId::from(InstructionClaimReductionChallenge::Gamma),
        input.stage2.public.instruction_gamma,
    )?;
    values.challenge(
        JoltChallengeId::from(InstructionClaimReductionChallenge::EqSpartan),
        eq_spartan,
    )?;

    let active_stage2_rounds = log_t + log_k;
    let phase1_offset = input
        .stage2
        .batch_consistency
        .try_round_offset(active_stage2_rounds)
        .map_err(|error| stage_sumcheck_error(JoltStageId::RamRafEvaluation, error))?
        + read_write_dimensions.phase1_num_rounds();
    let ram_raf_point = input
        .stage2
        .batch_consistency
        .try_instance_point_at(phase1_offset, ram_raf.sumcheck.rounds)
        .map_err(|error| stage_sumcheck_error(JoltStageId::RamRafEvaluation, error))?;
    let ram_raf_address = read_write_dimensions
        .address_opening_point(&ram_raf_point)
        .map_err(|error| public_error(JoltStageId::RamRafEvaluation, error))?;
    let ram_raf_unmap_address = IdentityPolynomial::new(log_k).evaluate(&ram_raf_address)
        * PCS::Field::from_u64(8)
        + PCS::Field::from_u64(input.checked.public_io.memory_layout.get_lowest_address());
    values.public(
        JoltPublicId::from(RamRafEvaluationPublic::UnmapAddress),
        ram_raf_unmap_address,
    )?;

    let ram_output_point = input
        .stage2
        .batch_consistency
        .try_instance_point_at(phase1_offset, ram_output.sumcheck.rounds)
        .map_err(|error| stage_sumcheck_error(JoltStageId::RamOutputCheck, error))?;
    let ram_output_address = read_write_dimensions
        .address_opening_point(&ram_output_point)
        .map_err(|error| public_error(JoltStageId::RamOutputCheck, error))?;
    let output_publics = ram_output_publics(
        input,
        &input.stage2.public.output_address_challenges,
        &ram_output_address,
    )?;
    values.public(
        JoltPublicId::from(RamOutputCheckPublic::EqIoMask),
        output_publics.0,
    )?;
    values.public(
        JoltPublicId::from(RamOutputCheckPublic::NegEqIoMaskValIo),
        output_publics.1,
    )?;

    let mut output_ids = Vec::new();
    output_ids.extend(ram::read_write_checking_output_openings());
    output_ids.extend(product_remainder_output_openings());
    let instruction_outputs =
        jolt_claims::protocols::jolt::formulas::claim_reductions::instruction::claim_reduction_output_openings();
    output_ids.push(instruction_outputs[1]);
    output_ids.push(instruction_outputs[2]);
    output_ids.extend(ram::raf_evaluation_output_openings());
    output_ids.extend(ram::output_check_output_openings());
    let aliases = vec![
        OpeningAlias::new(
            instruction_outputs[0],
            product_remainder_output_openings()[4],
        ),
        OpeningAlias::new(
            instruction_outputs[3],
            product_remainder_output_openings()[0],
        ),
        OpeningAlias::new(
            instruction_outputs[4],
            product_remainder_output_openings()[1],
        ),
    ];
    add_batched_stage(
        builder,
        "stage2.batch",
        &[
            ram_read_write,
            product_remainder,
            instruction_reduction,
            ram_raf,
            ram_output,
        ],
        &input.stage2.batch_consistency,
        &input.stage2.batch_output_claims,
        values,
        output_ids,
        aliases,
    )
}

fn add_stage3<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    builder: Builder<PCS::Field, VC::Output>,
    values: &mut SourceValues<PCS::Field>,
) -> Result<Builder<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    VC::Output: Clone,
{
    let log_t = input.checked.trace_length.ilog2() as usize;
    let dimensions = jolt_claims::protocols::jolt::TraceDimensions::new(log_t);
    let shift = spartan::shift::<PCS::Field>(dimensions);
    let instruction_input = instruction::input_virtualization::<PCS::Field>(dimensions);
    let registers_reduction =
        jolt_claims::protocols::jolt::formulas::claim_reductions::registers::claim_reduction::<
            PCS::Field,
        >(dimensions);

    values.challenge(
        JoltChallengeId::from(SpartanShiftChallenge::Gamma),
        input.stage3.public.shift_gamma,
    )?;
    values.challenge(
        JoltChallengeId::from(InstructionInputChallenge::Gamma),
        input.stage3.public.instruction_gamma,
    )?;
    values.challenge(
        JoltChallengeId::from(RegistersClaimReductionChallenge::Gamma),
        input.stage3.public.registers_gamma,
    )?;

    let shift_point = input
        .stage3
        .batch_consistency
        .try_instance_point(shift.sumcheck.rounds)
        .map_err(|error| stage_sumcheck_error(JoltStageId::SpartanShift, error))?;
    let shift_opening_point = shift_point.iter().rev().copied().collect::<Vec<_>>();
    let eq_plus_one_outer = EqPlusOnePolynomial::new(input.stage2.public.product_tau_low.clone())
        .evaluate(&shift_opening_point);
    let product_point = input
        .stage2
        .batch_consistency
        .try_instance_point(
            SpartanProductDimensions::from(log_t)
                .remainder_sumcheck()
                .rounds,
        )
        .map_err(|error| stage_sumcheck_error(JoltStageId::SpartanProductVirtualization, error))?;
    let product_opening_point = product_point.iter().rev().copied().collect::<Vec<_>>();
    let eq_plus_one_product =
        EqPlusOnePolynomial::new(product_opening_point.clone()).evaluate(&shift_opening_point);
    values.public(
        JoltPublicId::from(SpartanShiftPublic::EqPlusOneOuter),
        eq_plus_one_outer,
    )?;
    values.public(
        JoltPublicId::from(SpartanShiftPublic::EqPlusOneProduct),
        eq_plus_one_product,
    )?;

    let instruction_point = input
        .stage3
        .batch_consistency
        .try_instance_point(instruction_input.sumcheck.rounds)
        .map_err(|error| {
            stage_sumcheck_error(JoltStageId::InstructionInputVirtualization, error)
        })?;
    let instruction_opening_point = instruction_point.iter().rev().copied().collect::<Vec<_>>();
    values.challenge(
        JoltChallengeId::from(InstructionInputChallenge::EqProduct),
        try_eq_mle(&instruction_opening_point, &product_opening_point)
            .map_err(|error| public_error(JoltStageId::InstructionInputVirtualization, error))?,
    )?;

    let registers_point = input
        .stage3
        .batch_consistency
        .try_instance_point(registers_reduction.sumcheck.rounds)
        .map_err(|error| stage_sumcheck_error(JoltStageId::RegistersClaimReduction, error))?;
    let registers_opening_point = registers_point.iter().rev().copied().collect::<Vec<_>>();
    values.challenge(
        JoltChallengeId::from(RegistersClaimReductionChallenge::EqSpartan),
        try_eq_mle(
            &registers_opening_point,
            &input.stage2.public.product_tau_low,
        )
        .map_err(|error| public_error(JoltStageId::RegistersClaimReduction, error))?,
    )?;

    let instruction_outputs = instruction::input_virtualization_output_openings();
    let register_outputs =
        jolt_claims::protocols::jolt::formulas::claim_reductions::registers::claim_reduction_output_openings();
    let output_ids = vec![
        shift_output_openings()[0],
        shift_output_openings()[1],
        shift_output_openings()[2],
        shift_output_openings()[3],
        shift_output_openings()[4],
        instruction_outputs[4],
        instruction_outputs[5],
        instruction_outputs[6],
        instruction_outputs[0],
        instruction_outputs[1],
        instruction_outputs[2],
        instruction_outputs[3],
        register_outputs[0],
    ];
    let aliases = vec![
        OpeningAlias::new(instruction_outputs[7], shift_output_openings()[0]),
        OpeningAlias::new(register_outputs[1], instruction_outputs[5]),
        OpeningAlias::new(register_outputs[2], instruction_outputs[1]),
    ];
    add_batched_stage(
        builder,
        "stage3.batch",
        &[shift, instruction_input, registers_reduction],
        &input.stage3.batch_consistency,
        &input.stage3.batch_output_claims,
        values,
        output_ids,
        aliases,
    )
}

fn add_stage4<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    builder: Builder<PCS::Field, VC::Output>,
    values: &mut SourceValues<PCS::Field>,
) -> Result<Builder<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    VC::Output: Clone,
{
    let log_t = input.checked.trace_length.ilog2() as usize;
    let log_k = input.checked.ram_K.ilog2() as usize;
    let trace_dimensions = jolt_claims::protocols::jolt::TraceDimensions::new(log_t);
    let register_dimensions = input
        .proof
        .rw_config
        .register_dimensions(log_t, REGISTER_ADDRESS_BITS);
    let registers_claims = registers::read_write_checking::<PCS::Field>(register_dimensions);
    let ram_init = ram_val_check_init(input)?;
    let ram_val_claims = ram::val_check::<PCS::Field>(trace_dimensions, ram_init);

    values.challenge(
        JoltChallengeId::from(RegistersReadWriteChallenge::Gamma),
        input.stage4.public.registers_gamma,
    )?;
    let registers_point = input
        .stage4
        .batch_consistency
        .try_instance_point(registers_claims.sumcheck.rounds)
        .map_err(|error| stage_sumcheck_error(JoltStageId::RegistersReadWriteChecking, error))?;
    let registers_opening = register_dimensions
        .read_write_opening_point(&registers_point)
        .map_err(|error| public_error(JoltStageId::RegistersReadWriteChecking, error))?;
    let registers_reduction_point = input
        .stage3
        .batch_consistency
        .try_instance_point(
            jolt_claims::protocols::jolt::TraceDimensions::new(log_t)
                .sumcheck(3)
                .rounds,
        )
        .map_err(|error| stage_sumcheck_error(JoltStageId::RegistersClaimReduction, error))?;
    let registers_reduction_opening = registers_reduction_point
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    values.challenge(
        JoltChallengeId::from(RegistersReadWriteChallenge::EqCycle),
        try_eq_mle(&registers_reduction_opening, &registers_opening.r_cycle)
            .map_err(|error| public_error(JoltStageId::RegistersReadWriteChecking, error))?,
    )?;

    values.challenge(
        JoltChallengeId::from(RamValCheckChallenge::Gamma),
        input.stage4.public.ram_val_check_gamma,
    )?;
    let ram_val_point = input
        .stage4
        .batch_consistency
        .try_instance_point(ram_val_claims.sumcheck.rounds)
        .map_err(|error| stage_sumcheck_error(JoltStageId::RamValCheck, error))?;
    let ram_val_cycle = ram_val_point.iter().rev().copied().collect::<Vec<_>>();
    let r_cycle = input
        .stage2
        .ram_val_check_inputs
        .ram_read_write_opening_point
        .get(log_k..)
        .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
            stage: JoltStageId::RamValCheck,
            reason: "RAM read-write opening point is shorter than the RAM address".to_string(),
        })?;
    values.challenge(
        JoltChallengeId::from(RamValCheckChallenge::LtCyclePlusGamma),
        LtPolynomial::evaluate(&ram_val_cycle, r_cycle) + input.stage4.public.ram_val_check_gamma,
    )?;

    let mut output_ids = Vec::new();
    if input.proof.untrusted_advice_commitment.is_some() {
        output_ids.push(ram::val_check_advice_opening(JoltAdviceKind::Untrusted));
    }
    if input.checked.trusted_advice_commitment_present {
        output_ids.push(ram::val_check_advice_opening(JoltAdviceKind::Trusted));
    }
    output_ids.extend(registers::read_write_checking_output_openings());
    output_ids.extend(ram::val_check_output_openings());
    add_batched_stage(
        builder,
        "stage4.batch",
        &[registers_claims, ram_val_claims],
        &input.stage4.batch_consistency,
        &input.stage4.batch_output_claims,
        values,
        output_ids,
        Vec::new(),
    )
}

fn add_stage5<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    builder: Builder<PCS::Field, VC::Output>,
    values: &mut SourceValues<PCS::Field>,
) -> Result<Builder<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    VC::Output: Clone,
{
    let log_t = input.checked.trace_length.ilog2() as usize;
    let log_k = input.checked.ram_K.ilog2() as usize;
    let trace_dimensions = jolt_claims::protocols::jolt::TraceDimensions::new(log_t);
    let formula_dimensions = formula_dimensions(input)?;
    let instruction_claims =
        instruction::read_raf::<RISCV_XLEN, PCS::Field>(formula_dimensions.instruction_read_raf);
    let ram_claims = ram::ra_claim_reduction::<PCS::Field>(trace_dimensions);
    let registers_claims = registers::val_evaluation::<PCS::Field>(trace_dimensions);

    values.challenge(
        JoltChallengeId::from(InstructionReadRafChallenge::Gamma),
        input.stage5.public.instruction_gamma,
    )?;
    let instruction_output_openings = instruction::read_raf_output_openings::<RISCV_XLEN>(
        formula_dimensions.instruction_read_raf,
    );
    let instruction_point = input
        .stage5
        .batch_consistency
        .try_instance_point(instruction_claims.sumcheck.rounds)
        .map_err(|error| stage_sumcheck_error(JoltStageId::InstructionReadRaf, error))?;
    let instruction_opening = formula_dimensions
        .instruction_read_raf
        .opening_point(&instruction_point)
        .map_err(|error| public_error(JoltStageId::InstructionReadRaf, error))?;
    let stage2_instruction_point = input
        .stage2
        .batch_consistency
        .try_instance_point(
            jolt_claims::protocols::jolt::TraceDimensions::new(log_t)
                .sumcheck(2)
                .rounds,
        )
        .map_err(|error| stage_sumcheck_error(JoltStageId::InstructionClaimReduction, error))?;
    let stage2_instruction_opening = stage2_instruction_point
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let eq_reduction = try_eq_mle(&stage2_instruction_opening, &instruction_opening.r_cycle)
        .map_err(|error| public_error(JoltStageId::InstructionReadRaf, error))?;
    let left_operand_eval = OperandPolynomial::new(2 * RISCV_XLEN, OperandSide::Left)
        .evaluate(&instruction_opening.r_address);
    let right_operand_eval = OperandPolynomial::new(2 * RISCV_XLEN, OperandSide::Right)
        .evaluate(&instruction_opening.r_address);
    let identity_eval =
        IdentityPolynomial::new(2 * RISCV_XLEN).evaluate(&instruction_opening.r_address);
    let instruction_gamma_squared =
        input.stage5.public.instruction_gamma * input.stage5.public.instruction_gamma;
    for table in LookupTableKind::<RISCV_XLEN>::iter() {
        values.challenge(
            JoltChallengeId::from(InstructionReadRafChallenge::EqTableValue(table.index())),
            eq_reduction
                * table.evaluate_mle::<PCS::Field, PCS::Field>(&instruction_opening.r_address),
        )?;
    }
    values.challenge(
        JoltChallengeId::from(InstructionReadRafChallenge::EqRafConstant),
        eq_reduction
            * (input.stage5.public.instruction_gamma * left_operand_eval
                + instruction_gamma_squared * right_operand_eval),
    )?;
    values.challenge(
        JoltChallengeId::from(InstructionReadRafChallenge::EqRafFlag),
        eq_reduction
            * (instruction_gamma_squared * identity_eval
                - input.stage5.public.instruction_gamma * left_operand_eval
                - instruction_gamma_squared * right_operand_eval),
    )?;

    values.challenge(
        JoltChallengeId::from(RamRaClaimReductionChallenge::Gamma),
        input.stage5.public.ram_gamma,
    )?;
    let ram_point = input
        .stage5
        .batch_consistency
        .try_instance_point(ram_claims.sumcheck.rounds)
        .map_err(|error| stage_sumcheck_error(JoltStageId::RamRaClaimReduction, error))?;
    let ram_cycle = trace_dimensions
        .cycle_opening_point(&ram_point)
        .map_err(|error| public_error(JoltStageId::RamRaClaimReduction, error))?;
    let ram_raf_cycle = &input
        .stage2
        .ram_ra_claim_reduction_inputs
        .ram_raf_evaluation_opening_point[log_k..];
    let ram_read_write_cycle = &input
        .stage2
        .ram_ra_claim_reduction_inputs
        .ram_read_write_opening_point[log_k..];
    let ram_val_cycle = &input.stage4.ram_val_check_opening_point[log_k..];
    values.public(
        JoltPublicId::from(RamRaClaimReductionPublic::EqCycleRaf),
        try_eq_mle(&ram_cycle, ram_raf_cycle)
            .map_err(|error| public_error(JoltStageId::RamRaClaimReduction, error))?,
    )?;
    values.public(
        JoltPublicId::from(RamRaClaimReductionPublic::EqCycleReadWrite),
        try_eq_mle(&ram_cycle, ram_read_write_cycle)
            .map_err(|error| public_error(JoltStageId::RamRaClaimReduction, error))?,
    )?;
    values.public(
        JoltPublicId::from(RamRaClaimReductionPublic::EqCycleValCheck),
        try_eq_mle(&ram_cycle, ram_val_cycle)
            .map_err(|error| public_error(JoltStageId::RamRaClaimReduction, error))?,
    )?;

    let registers_point = input
        .stage5
        .batch_consistency
        .try_instance_point(registers_claims.sumcheck.rounds)
        .map_err(|error| stage_sumcheck_error(JoltStageId::RegistersValEvaluation, error))?;
    let registers_cycle = trace_dimensions
        .cycle_opening_point(&registers_point)
        .map_err(|error| public_error(JoltStageId::RegistersValEvaluation, error))?;
    let registers_read_write_cycle =
        &input.stage4.registers_read_write_opening_point[REGISTER_ADDRESS_BITS..];
    values.challenge(
        JoltChallengeId::from(RegistersValEvaluationChallenge::LtCycle),
        LtPolynomial::evaluate(&registers_cycle, registers_read_write_cycle),
    )?;

    let mut output_ids = Vec::new();
    output_ids.extend(instruction_output_openings.lookup_table_flags);
    output_ids.extend(instruction_output_openings.instruction_ra);
    output_ids.push(instruction_output_openings.instruction_raf_flag);
    output_ids.extend(ram::ra_claim_reduction_output_openings());
    output_ids.extend(registers::val_evaluation_output_openings());
    add_batched_stage(
        builder,
        "stage5.batch",
        &[instruction_claims, ram_claims, registers_claims],
        &input.stage5.batch_consistency,
        &input.stage5.batch_output_claims,
        values,
        output_ids,
        Vec::new(),
    )
}

fn add_stage6<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    builder: Builder<PCS::Field, VC::Output>,
    values: &mut SourceValues<PCS::Field>,
) -> Result<Builder<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    VC::Output: Clone,
{
    let log_t = input.checked.trace_length.ilog2() as usize;
    let trace_dimensions = jolt_claims::protocols::jolt::TraceDimensions::new(log_t);
    let formula_dimensions = formula_dimensions(input)?;
    let bytecode_claims =
        bytecode::read_raf::<RISCV_XLEN, PCS::Field>(formula_dimensions.bytecode_read_raf);
    let booleanity_dimensions = BooleanityDimensions::from((
        formula_dimensions.ra_layout,
        log_t,
        input.proof.one_hot_config.committed_chunk_bits(),
    ));
    let booleanity_claims = booleanity::booleanity::<PCS::Field>(booleanity_dimensions);
    let ram_hamming_claims = ram::hamming_booleanity::<PCS::Field>(trace_dimensions);
    let ram_ra_claims =
        ram::ra_virtualization::<PCS::Field>(formula_dimensions.ram_ra_virtualization);
    let instruction_ra_claims = instruction::ra_virtualization::<PCS::Field>(
        formula_dimensions.instruction_ra_virtualization,
    );
    let inc_claims = increments::claim_reduction::<PCS::Field>(trace_dimensions);
    let (trusted_layout, trusted_claims) = advice_cycle_claim(input, JoltAdviceKind::Trusted);
    let (untrusted_layout, untrusted_claims) = advice_cycle_claim(input, JoltAdviceKind::Untrusted);

    add_stage6_publics_and_challenges(
        input,
        values,
        &bytecode_claims,
        &booleanity_claims,
        &ram_hamming_claims,
        &ram_ra_claims,
        &instruction_ra_claims,
        &inc_claims,
    )?;
    if let (Some(layout), Some(_claim), Some(public)) = (
        trusted_layout.as_ref(),
        trusted_claims.as_ref(),
        input.stage6.trusted_advice_cycle_phase.as_ref(),
    ) {
        add_advice_cycle_publics(input, values, layout, JoltAdviceKind::Trusted, public)?;
    }
    if let (Some(layout), Some(_claim), Some(public)) = (
        untrusted_layout.as_ref(),
        untrusted_claims.as_ref(),
        input.stage6.untrusted_advice_cycle_phase.as_ref(),
    ) {
        add_advice_cycle_publics(input, values, layout, JoltAdviceKind::Untrusted, public)?;
    }

    let mut claims = vec![
        bytecode_claims,
        booleanity_claims,
        ram_hamming_claims,
        ram_ra_claims,
        instruction_ra_claims,
        inc_claims,
    ];
    if let Some(claim) = trusted_claims {
        claims.push(claim);
    }
    if let Some(claim) = untrusted_claims {
        claims.push(claim);
    }

    let mut output_ids = Vec::new();
    output_ids.extend(
        bytecode::read_raf_output_openings(formula_dimensions.bytecode_read_raf).bytecode_ra,
    );
    output_ids.extend(booleanity::booleanity_output_openings(
        formula_dimensions.ra_layout,
    ));
    output_ids.extend(ram::hamming_booleanity_output_openings());
    output_ids.extend(ram::ra_virtualization_output_openings(
        formula_dimensions.ram_ra_virtualization,
    ));
    output_ids.extend(
        instruction::ra_virtualization_output_openings(
            formula_dimensions.instruction_ra_virtualization,
        )
        .all(),
    );
    output_ids.extend(increments::claim_reduction_output_openings());
    if let Some(layout) = trusted_layout {
        output_ids.extend(advice::cycle_phase_output_openings(
            JoltAdviceKind::Trusted,
            layout.dimensions(),
        ));
    }
    if let Some(layout) = untrusted_layout {
        output_ids.extend(advice::cycle_phase_output_openings(
            JoltAdviceKind::Untrusted,
            layout.dimensions(),
        ));
    }
    add_batched_stage(
        builder,
        "stage6.batch",
        &claims,
        &input.stage6.batch_consistency,
        &input.stage6.batch_output_claims,
        values,
        output_ids,
        Vec::new(),
    )
}

fn add_stage7<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    builder: Builder<PCS::Field, VC::Output>,
    values: &mut SourceValues<PCS::Field>,
) -> Result<Builder<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    VC::Output: Clone,
{
    let formula_dimensions = formula_dimensions(input)?;
    let hamming_dimensions = hamming_weight::HammingWeightClaimReductionDimensions::from((
        formula_dimensions.ra_layout,
        input.proof.one_hot_config.committed_chunk_bits(),
    ));
    let hamming_claims = hamming_weight::claim_reduction::<PCS::Field>(hamming_dimensions);
    let (trusted_layout, trusted_claims) = advice_address_claim(input, JoltAdviceKind::Trusted);
    let (untrusted_layout, untrusted_claims) =
        advice_address_claim(input, JoltAdviceKind::Untrusted);

    values.challenge(
        JoltChallengeId::from(HammingWeightClaimReductionChallenge::Gamma),
        input.stage7.public.hamming_gamma,
    )?;
    let hamming_point = input
        .stage7
        .batch_consistency
        .try_instance_point(hamming_claims.sumcheck.rounds)
        .map_err(|error| stage_sumcheck_error(JoltStageId::HammingWeightClaimReduction, error))?;
    let rho_rev = hamming_point.iter().rev().copied().collect::<Vec<_>>();
    values.public(
        JoltPublicId::from(HammingWeightClaimReductionPublic::EqBooleanity),
        try_eq_mle(&rho_rev, &input.stage6.booleanity.r_address)
            .map_err(|error| public_error(JoltStageId::HammingWeightClaimReduction, error))?,
    )?;
    let virtualization_points = stage6_virtualization_points(input, hamming_dimensions)?;
    for (index, point) in virtualization_points.iter().enumerate() {
        values.public(
            JoltPublicId::from(HammingWeightClaimReductionPublic::EqVirtualization(index)),
            try_eq_mle(&rho_rev, point)
                .map_err(|error| public_error(JoltStageId::HammingWeightClaimReduction, error))?,
        )?;
    }
    if let (Some(layout), Some(_claim), Some(public)) = (
        trusted_layout.as_ref(),
        trusted_claims.as_ref(),
        input.stage7.trusted_advice_address_phase.as_ref(),
    ) {
        add_advice_address_publics(input, values, layout, JoltAdviceKind::Trusted, public)?;
    }
    if let (Some(layout), Some(_claim), Some(public)) = (
        untrusted_layout.as_ref(),
        untrusted_claims.as_ref(),
        input.stage7.untrusted_advice_address_phase.as_ref(),
    ) {
        add_advice_address_publics(input, values, layout, JoltAdviceKind::Untrusted, public)?;
    }

    let mut claims = vec![hamming_claims];
    if let Some(claim) = trusted_claims {
        claims.push(claim);
    }
    if let Some(claim) = untrusted_claims {
        claims.push(claim);
    }
    let output_openings = hamming_weight::claim_reduction_output_openings(hamming_dimensions);
    let mut output_ids = output_openings.all();
    if let Some(layout) = trusted_layout {
        if layout.dimensions().has_address_phase() {
            output_ids.extend(advice::address_phase_output_openings(
                JoltAdviceKind::Trusted,
            ));
        }
    }
    if let Some(layout) = untrusted_layout {
        if layout.dimensions().has_address_phase() {
            output_ids.extend(advice::address_phase_output_openings(
                JoltAdviceKind::Untrusted,
            ));
        }
    }
    add_batched_stage(
        builder,
        "stage7.batch",
        &claims,
        &input.stage7.batch_consistency,
        &input.stage7.batch_output_claims,
        values,
        output_ids,
        Vec::new(),
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "BlindFold stages are deliberately explicit."
)]
fn add_single_stage<F, C>(
    builder: Builder<F, C>,
    name: &'static str,
    claim: &JoltStageClaims<F>,
    consistency: &CommittedSumcheckConsistency<F, C>,
    output_claims: &CommittedOutputClaimOutput<C>,
    values: &SourceValues<F>,
    opening_ids: Vec<JoltOpeningId>,
    aliases: Vec<OpeningAlias<JoltOpeningId>>,
) -> Result<Builder<F, C>, VerifierError>
where
    F: Field,
    C: Clone,
{
    let statement = SumcheckStatement::new(claim.sumcheck.rounds, claim.sumcheck.degree);
    add_stage(
        builder,
        name,
        statement,
        domain_spec(claim.sumcheck),
        consistency.clone(),
        output_claims,
        values,
        opening_ids,
        aliases,
        claim.input.expression.clone(),
        claim.output.expression.clone(),
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "BlindFold stages are deliberately explicit."
)]
fn add_batched_stage<F, C>(
    builder: Builder<F, C>,
    name: &'static str,
    claims: &[JoltStageClaims<F>],
    consistency: &BatchedCommittedSumcheckConsistency<F, C>,
    output_claims: &CommittedOutputClaimOutput<C>,
    values: &SourceValues<F>,
    opening_ids: Vec<JoltOpeningId>,
    aliases: Vec<OpeningAlias<JoltOpeningId>>,
) -> Result<Builder<F, C>, VerifierError>
where
    F: Field,
    C: Clone,
{
    let Some(first) = claims.first() else {
        return Err(VerifierError::BlindFoldConstructionFailed {
            reason: format!("{name}: empty batched claims"),
        });
    };
    let domain = domain_spec(first.sumcheck);
    let input_claim = batched_input_expr(claims, consistency);
    let output_claim = batched_output_expr(claims, consistency);
    add_stage(
        builder,
        name,
        SumcheckStatement::new(consistency.max_num_vars, consistency.max_degree),
        domain,
        consistency.consistency.clone(),
        output_claims,
        values,
        opening_ids,
        aliases,
        input_claim,
        output_claim,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "BlindFold stages are deliberately explicit."
)]
fn add_stage<F, C>(
    builder: Builder<F, C>,
    name: &'static str,
    statement: SumcheckStatement,
    domain: SumcheckDomainSpec,
    consistency: CommittedSumcheckConsistency<F, C>,
    output_claims: &CommittedOutputClaimOutput<C>,
    values: &SourceValues<F>,
    opening_ids: Vec<JoltOpeningId>,
    aliases: Vec<OpeningAlias<JoltOpeningId>>,
    input_claim: JoltExpr<F>,
    output_claim: JoltExpr<F>,
) -> Result<Builder<F, C>, VerifierError>
where
    F: Field,
    C: Clone,
{
    require_expr_sources(name, "input claim", &input_claim, values)?;
    require_expr_sources(name, "output claim", &output_claim, values)?;
    if opening_ids.len() != output_claims.shape.output_claim_count {
        return Err(VerifierError::BlindFoldConstructionFailed {
            reason: format!(
                "{name}: output opening id count mismatch: expected {}, got {}",
                output_claims.shape.output_claim_count,
                opening_ids.len()
            ),
        });
    }
    builder
        .stage(name)
        .sumcheck(statement)
        .domain(domain)
        .consistency(consistency)
        .output_claim_rows(
            opening_ids,
            output_claims.shape.row_len,
            output_claims.commitments.clone(),
        )
        .output_claim_aliases(aliases)
        .input_claim(input_claim)
        .output_claim(output_claim)
        .finish_stage()
        .map_err(blindfold_error)
}

fn batched_input_expr<F, C>(
    claims: &[JoltStageClaims<F>],
    consistency: &BatchedCommittedSumcheckConsistency<F, C>,
) -> JoltExpr<F>
where
    F: Field,
{
    claims.iter().zip(&consistency.batching_coefficients).fold(
        JoltExpr::zero(),
        |acc, (claim, coefficient)| {
            let scale = *coefficient * pow2::<F>(consistency.max_num_vars - claim.sumcheck.rounds);
            acc + scale_expr(claim.input.expression.clone(), scale)
        },
    )
}

fn batched_output_expr<F, C>(
    claims: &[JoltStageClaims<F>],
    consistency: &BatchedCommittedSumcheckConsistency<F, C>,
) -> JoltExpr<F>
where
    F: Field,
{
    claims
        .iter()
        .zip(&consistency.batching_coefficients)
        .fold(JoltExpr::zero(), |acc, (claim, coefficient)| {
            acc + scale_expr(claim.output.expression.clone(), *coefficient)
        })
}

fn scale_expr<F: Field>(mut expr: JoltExpr<F>, scale: F) -> JoltExpr<F> {
    if scale.is_zero() {
        return JoltExpr::zero();
    }
    for term in &mut expr.terms {
        term.coefficient *= scale;
    }
    expr
}

fn require_expr_sources<F: Field>(
    stage: &'static str,
    expression: &'static str,
    expr: &JoltExpr<F>,
    values: &SourceValues<F>,
) -> Result<(), VerifierError> {
    for id in expr.required_publics() {
        if !values.has_public(id) {
            return Err(VerifierError::BlindFoldConstructionFailed {
                reason: format!("{stage} {expression} is missing public source {id:?}"),
            });
        }
    }
    for id in expr.required_challenges() {
        if !values.has_challenge(id) {
            return Err(VerifierError::BlindFoldConstructionFailed {
                reason: format!("{stage} {expression} is missing challenge source {id:?}"),
            });
        }
    }
    Ok(())
}

fn domain_spec(spec: JoltSumcheckSpec) -> SumcheckDomainSpec {
    match spec.domain {
        JoltSumcheckDomain::BooleanHypercube => SumcheckDomainSpec::BooleanHypercube,
        JoltSumcheckDomain::CenteredInteger { domain_size } => {
            SumcheckDomainSpec::CenteredInteger { domain_size }
        }
    }
}

fn formula_dimensions<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
) -> Result<JoltFormulaDimensions, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let log_t = input.checked.trace_length.ilog2() as usize;
    JoltFormulaDimensions::try_from(input.proof.one_hot_config.dimensions(
        log_t,
        2 * RISCV_XLEN,
        input.preprocessing.program.bytecode.code_size,
        input.checked.ram_K,
    ))
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltStageId::BytecodeReadRaf,
        reason: error.to_string(),
    })
}

fn ram_output_publics<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    output_address_challenges: &[PCS::Field],
    ram_output_address: &[PCS::Field],
) -> Result<(PCS::Field, PCS::Field), VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let public_memory = PublicIoMemory::new(&input.checked.public_io).map_err(|error| {
        VerifierError::StageClaimPublicInputFailed {
            stage: JoltStageId::RamOutputCheck,
            reason: error.to_string(),
        }
    })?;
    let output_eq = try_eq_mle(output_address_challenges, ram_output_address)
        .map_err(|error| public_error(JoltStageId::RamOutputCheck, error))?;
    let output_mask = range_mask_mle_msb(
        public_memory.io_mask_start,
        public_memory.io_mask_end,
        ram_output_address,
    )
    .map_err(|error| public_error(JoltStageId::RamOutputCheck, error))?;
    let io_num_vars = public_memory.io_num_vars();
    let (r_hi, r_lo) = ram_output_address.split_at(
        ram_output_address
            .len()
            .checked_sub(io_num_vars)
            .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
                stage: JoltStageId::RamOutputCheck,
                reason: format!(
                    "RAM output address has {} variables but public IO needs {io_num_vars}",
                    ram_output_address.len()
                ),
            })?,
    );
    let hi_scale = r_hi.iter().fold(PCS::Field::one(), |acc, challenge| {
        acc * (PCS::Field::one() - *challenge)
    });
    let val_io = hi_scale
        * sparse_segments_mle_msb(
            public_memory
                .segments
                .iter()
                .map(|segment| (segment.start_index, segment.words.as_slice())),
            r_lo,
        );
    let eq_io_mask = output_eq * output_mask;
    Ok((eq_io_mask, -eq_io_mask * val_io))
}

fn ram_val_check_init<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
) -> Result<ram::RamValCheckInit<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let r_address = ram_val_check_address(input)?;
    let mut advice_contributions = Vec::new();
    if input.proof.untrusted_advice_commitment.is_some() {
        let selector = advice_selector(input, JoltAdviceKind::Untrusted, &r_address)?;
        advice_contributions.push(ram::RamValCheckAdviceContribution::untrusted(-selector.0));
    }
    if input.checked.trusted_advice_commitment_present {
        let selector = advice_selector(input, JoltAdviceKind::Trusted, &r_address)?;
        advice_contributions.push(ram::RamValCheckAdviceContribution::trusted(-selector.0));
    }
    Ok(ram::RamValCheckInit::decomposed(
        input.stage4.ram_val_check_public_eval,
        advice_contributions,
    ))
}

fn ram_val_check_address<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
) -> Result<Vec<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let log_k = input.checked.ram_K.ilog2() as usize;
    input
        .stage2
        .ram_val_check_inputs
        .ram_read_write_opening_point
        .get(..log_k)
        .map(<[PCS::Field]>::to_vec)
        .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
            stage: JoltStageId::RamValCheck,
            reason: "RAM read-write opening point is shorter than the RAM address".to_string(),
        })
}

fn advice_selector<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    kind: JoltAdviceKind,
    r_address: &[PCS::Field],
) -> Result<(PCS::Field, Vec<PCS::Field>), VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let layout = &input.checked.public_io.memory_layout;
    let (start_address, max_size) = match kind {
        JoltAdviceKind::Trusted => (layout.trusted_advice_start, layout.max_trusted_advice_size),
        JoltAdviceKind::Untrusted => (
            layout.untrusted_advice_start,
            layout.max_untrusted_advice_size,
        ),
    };
    let start_index = layout
        .remapped_word_address(start_address)
        .map_err(|error| public_error(JoltStageId::RamValCheck, error))?
        as usize;
    let advice_num_vars = ((max_size as usize) / 8).next_power_of_two().ilog2() as usize;
    let selector = block_selector_mle_msb(start_index, advice_num_vars, r_address)
        .map_err(|error| public_error(JoltStageId::RamValCheck, error))?;
    let opening_point = r_address
        .get(r_address.len().checked_sub(advice_num_vars).ok_or_else(|| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltStageId::RamValCheck,
                reason: format!(
                    "{kind:?} advice point needs {advice_num_vars} variables but RAM address has {}",
                    r_address.len()
                ),
            }
        })?..)
        .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
            stage: JoltStageId::RamValCheck,
            reason: "advice opening point is out of range".to_string(),
        })?
        .to_vec();
    Ok((selector, opening_point))
}

fn advice_source_point<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    kind: JoltAdviceKind,
) -> Result<Vec<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let r_address = ram_val_check_address(input)?;
    advice_selector(input, kind, &r_address).map(|(_, point)| point)
}

fn advice_cycle_claim<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    kind: JoltAdviceKind,
) -> (
    Option<AdviceClaimReductionLayout>,
    Option<JoltStageClaims<PCS::Field>>,
)
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let layout = advice_layout(input, kind);
    let claim = layout
        .as_ref()
        .map(|layout| advice::cycle_phase::<PCS::Field>(kind, layout.dimensions()));
    (layout, claim)
}

fn advice_address_claim<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    kind: JoltAdviceKind,
) -> (
    Option<AdviceClaimReductionLayout>,
    Option<JoltStageClaims<PCS::Field>>,
)
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let layout = advice_layout(input, kind);
    let claim = layout.as_ref().and_then(|layout| {
        layout
            .dimensions()
            .has_address_phase()
            .then(|| advice::address_phase::<PCS::Field>(kind, layout.dimensions()))
    });
    (layout, claim)
}

fn advice_layout<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    kind: JoltAdviceKind,
) -> Option<AdviceClaimReductionLayout>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let present = match kind {
        JoltAdviceKind::Trusted => input.checked.trusted_advice_commitment_present,
        JoltAdviceKind::Untrusted => input.proof.untrusted_advice_commitment.is_some(),
    };
    present.then(|| {
        let log_t = input.checked.trace_length.ilog2() as usize;
        let max_size = match kind {
            JoltAdviceKind::Trusted => {
                input
                    .checked
                    .public_io
                    .memory_layout
                    .max_trusted_advice_size as usize
            }
            JoltAdviceKind::Untrusted => {
                input
                    .checked
                    .public_io
                    .memory_layout
                    .max_untrusted_advice_size as usize
            }
        };
        AdviceClaimReductionLayout::balanced(
            input.proof.trace_polynomial_order,
            log_t,
            input.proof.one_hot_config.committed_chunk_bits(),
            max_size,
        )
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 6 has several protocol components."
)]
fn add_stage6_publics_and_challenges<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    values: &mut SourceValues<PCS::Field>,
    bytecode_claims: &JoltStageClaims<PCS::Field>,
    booleanity_claims: &JoltStageClaims<PCS::Field>,
    ram_hamming_claims: &JoltStageClaims<PCS::Field>,
    ram_ra_claims: &JoltStageClaims<PCS::Field>,
    instruction_ra_claims: &JoltStageClaims<PCS::Field>,
    inc_claims: &JoltStageClaims<PCS::Field>,
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let log_t = input.checked.trace_length.ilog2() as usize;
    let log_k = input.checked.ram_K.ilog2() as usize;
    let trace_dimensions = jolt_claims::protocols::jolt::TraceDimensions::new(log_t);
    let formula_dimensions = formula_dimensions(input)?;

    values.challenge(
        JoltChallengeId::from(BytecodeReadRafChallenge::Gamma),
        input.stage6.public.bytecode_gamma_powers[1],
    )?;
    values.challenge(
        JoltChallengeId::from(BytecodeReadRafChallenge::Stage1Gamma),
        input.stage6.public.stage1_gammas[1],
    )?;
    values.challenge(
        JoltChallengeId::from(BytecodeReadRafChallenge::Stage2Gamma),
        input.stage6.public.stage2_gammas[1],
    )?;
    values.challenge(
        JoltChallengeId::from(BytecodeReadRafChallenge::Stage3Gamma),
        input.stage6.public.stage3_gammas[1],
    )?;
    values.challenge(
        JoltChallengeId::from(BytecodeReadRafChallenge::Stage4Gamma),
        input.stage6.public.stage4_gammas[1],
    )?;
    values.challenge(
        JoltChallengeId::from(BytecodeReadRafChallenge::Stage5Gamma),
        input.stage6.public.stage5_gammas[1],
    )?;
    values.challenge(
        JoltChallengeId::from(BooleanityChallenge::Gamma),
        input.stage6.public.booleanity_gamma,
    )?;
    values.challenge(
        JoltChallengeId::from(InstructionRaVirtualizationChallenge::Gamma),
        input
            .stage6
            .public
            .instruction_ra_gamma_powers
            .get(1)
            .copied()
            .unwrap_or_else(PCS::Field::one),
    )?;
    values.challenge(
        JoltChallengeId::from(IncClaimReductionChallenge::Gamma),
        input.stage6.public.inc_gamma,
    )?;

    let bytecode_point = input
        .stage6
        .batch_consistency
        .try_instance_point(bytecode_claims.sumcheck.rounds)
        .map_err(|error| stage_sumcheck_error(JoltStageId::BytecodeReadRaf, error))?;
    let bytecode_opening = formula_dimensions
        .bytecode_read_raf
        .opening_point(&bytecode_point)
        .map_err(|error| public_error(JoltStageId::BytecodeReadRaf, error))?;
    let stage1_cycle = input.stage1.public.remainder_challenges[1..]
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let stage2_product_point = input
        .stage2
        .batch_consistency
        .try_instance_point(
            SpartanProductDimensions::from(log_t)
                .remainder_sumcheck()
                .rounds,
        )
        .map_err(|error| stage_sumcheck_error(JoltStageId::SpartanProductVirtualization, error))?;
    let stage2_cycle = stage2_product_point
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let stage3_shift_point = input
        .stage3
        .batch_consistency
        .try_instance_point(
            jolt_claims::protocols::jolt::TraceDimensions::new(log_t)
                .sumcheck(2)
                .rounds,
        )
        .map_err(|error| stage_sumcheck_error(JoltStageId::SpartanShift, error))?;
    let stage3_cycle = stage3_shift_point.iter().rev().copied().collect::<Vec<_>>();
    let stage4_cycle = &input.stage4.registers_read_write_opening_point[REGISTER_ADDRESS_BITS..];
    let stage5_cycle =
        &input.stage5.registers_val_evaluation.opening_point[REGISTER_ADDRESS_BITS..];
    let entry_bytecode_index = input
        .preprocessing
        .program
        .bytecode
        .entry_bytecode_index()
        .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
            stage: JoltStageId::BytecodeReadRaf,
            reason: "entry address was not found in bytecode preprocessing".to_string(),
        })?;
    let bytecode_public_values = bytecode::read_raf_public_values::<RISCV_XLEN, PCS::Field>(
        BytecodeReadRafEvaluationInputs {
            bytecode: &input.preprocessing.program.bytecode.bytecode,
            r_address: &bytecode_opening.r_address,
            r_cycle: &bytecode_opening.r_cycle,
            stage_cycle_points: [
                &stage1_cycle,
                &stage2_cycle,
                &stage3_cycle,
                stage4_cycle,
                stage5_cycle,
            ],
            register_read_write_point: &input.stage4.registers_read_write_opening_point
                [..REGISTER_ADDRESS_BITS],
            register_val_evaluation_point: &input.stage5.registers_val_evaluation.opening_point
                [..REGISTER_ADDRESS_BITS],
            entry_bytecode_index,
            stage1_gammas: &input.stage6.public.stage1_gammas,
            stage2_gammas: &input.stage6.public.stage2_gammas,
            stage3_gammas: &input.stage6.public.stage3_gammas,
            stage4_gammas: &input.stage6.public.stage4_gammas,
            stage5_gammas: &input.stage6.public.stage5_gammas,
        },
    )
    .map_err(|error| public_error(JoltStageId::BytecodeReadRaf, error))?;
    for index in 0..5 {
        values.public(
            JoltPublicId::from(BytecodeReadRafPublic::StageValue(index)),
            bytecode_public_values.value(BytecodeReadRafPublic::StageValue(index)),
        )?;
    }
    values.public(
        JoltPublicId::from(BytecodeReadRafPublic::SpartanOuterRaf),
        bytecode_public_values.value(BytecodeReadRafPublic::SpartanOuterRaf),
    )?;
    values.public(
        JoltPublicId::from(BytecodeReadRafPublic::SpartanShiftRaf),
        bytecode_public_values.value(BytecodeReadRafPublic::SpartanShiftRaf),
    )?;
    values.public(
        JoltPublicId::from(BytecodeReadRafPublic::Entry),
        bytecode_public_values.value(BytecodeReadRafPublic::Entry),
    )?;

    let booleanity_point = input
        .stage6
        .batch_consistency
        .try_instance_point(booleanity_claims.sumcheck.rounds)
        .map_err(|error| stage_sumcheck_error(JoltStageId::Booleanity, error))?;
    let reference_eq_point = input
        .stage6
        .public
        .booleanity_reference_address
        .iter()
        .rev()
        .chain(input.stage6.public.booleanity_reference_cycle.iter().rev())
        .copied()
        .collect::<Vec<_>>();
    values.public(
        JoltPublicId::from(BooleanityPublic::EqAddressCycle),
        try_eq_mle(&booleanity_point, &reference_eq_point)
            .map_err(|error| public_error(JoltStageId::Booleanity, error))?,
    )?;

    let ram_hamming_point = input
        .stage6
        .batch_consistency
        .try_instance_point(ram_hamming_claims.sumcheck.rounds)
        .map_err(|error| stage_sumcheck_error(JoltStageId::RamHammingBooleanity, error))?;
    let stage1_cycle_binding = &input.stage1.public.remainder_challenges[1..];
    values.public(
        JoltPublicId::from(RamHammingBooleanityPublic::EqCycle),
        try_eq_mle(&ram_hamming_point, stage1_cycle_binding)
            .map_err(|error| public_error(JoltStageId::RamHammingBooleanity, error))?,
    )?;

    let ram_ra_point = input
        .stage6
        .batch_consistency
        .try_instance_point(ram_ra_claims.sumcheck.rounds)
        .map_err(|error| stage_sumcheck_error(JoltStageId::RamRaVirtualization, error))?;
    let ram_ra_cycle = trace_dimensions
        .cycle_opening_point(&ram_ra_point)
        .map_err(|error| public_error(JoltStageId::RamRaVirtualization, error))?;
    let ram_reduced_cycle = &input.stage5.ram_ra_claim_reduction.opening_point[log_k..];
    values.public(
        JoltPublicId::from(RamRaVirtualizationPublic::EqCycle),
        try_eq_mle(ram_reduced_cycle, &ram_ra_cycle)
            .map_err(|error| public_error(JoltStageId::RamRaVirtualization, error))?,
    )?;

    let instruction_ra_point = input
        .stage6
        .batch_consistency
        .try_instance_point(instruction_ra_claims.sumcheck.rounds)
        .map_err(|error| stage_sumcheck_error(JoltStageId::InstructionRaVirtualization, error))?;
    let instruction_ra_cycle = trace_dimensions
        .cycle_opening_point(&instruction_ra_point)
        .map_err(|error| public_error(JoltStageId::InstructionRaVirtualization, error))?;
    values.challenge(
        JoltChallengeId::from(InstructionRaVirtualizationChallenge::EqCycle),
        try_eq_mle(
            &input.stage5.instruction_read_raf.r_cycle,
            &instruction_ra_cycle,
        )
        .map_err(|error| public_error(JoltStageId::InstructionRaVirtualization, error))?,
    )?;

    let inc_point = input
        .stage6
        .batch_consistency
        .try_instance_point(inc_claims.sumcheck.rounds)
        .map_err(|error| stage_sumcheck_error(JoltStageId::IncClaimReduction, error))?;
    let inc_opening_point = trace_dimensions
        .cycle_opening_point(&inc_point)
        .map_err(|error| public_error(JoltStageId::IncClaimReduction, error))?;
    values.public(
        JoltPublicId::from(IncClaimReductionPublic::EqRamReadWrite),
        try_eq_mle(
            &inc_opening_point,
            &input
                .stage2
                .ram_val_check_inputs
                .ram_read_write_opening_point[log_k..],
        )
        .map_err(|error| public_error(JoltStageId::IncClaimReduction, error))?,
    )?;
    values.public(
        JoltPublicId::from(IncClaimReductionPublic::EqRamValCheck),
        try_eq_mle(
            &inc_opening_point,
            &input.stage4.ram_val_check_opening_point[log_k..],
        )
        .map_err(|error| public_error(JoltStageId::IncClaimReduction, error))?,
    )?;
    values.public(
        JoltPublicId::from(IncClaimReductionPublic::EqRegistersReadWrite),
        try_eq_mle(
            &inc_opening_point,
            &input.stage4.registers_read_write_opening_point[REGISTER_ADDRESS_BITS..],
        )
        .map_err(|error| public_error(JoltStageId::IncClaimReduction, error))?,
    )?;
    values.public(
        JoltPublicId::from(IncClaimReductionPublic::EqRegistersValEvaluation),
        try_eq_mle(
            &inc_opening_point,
            &input.stage5.registers_val_evaluation.opening_point[REGISTER_ADDRESS_BITS..],
        )
        .map_err(|error| public_error(JoltStageId::IncClaimReduction, error))?,
    )?;

    Ok(())
}

fn add_advice_cycle_publics<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    values: &mut SourceValues<PCS::Field>,
    layout: &AdviceClaimReductionLayout,
    kind: JoltAdviceKind,
    public: &crate::stages::stage6::outputs::AdviceCyclePhasePublicOutput<PCS::Field>,
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let source_point = advice_source_point(input, kind)?;
    let scale = layout
        .cycle_phase_final_output_scale(&source_point, &public.sumcheck_point)
        .map_err(|error| public_error(JoltStageId::AdviceClaimReductionCyclePhase, error))?;
    values.public(
        JoltPublicId::from(AdviceClaimReductionPublic::FinalScale(kind)),
        scale,
    )
}

fn add_advice_address_publics<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    values: &mut SourceValues<PCS::Field>,
    layout: &AdviceClaimReductionLayout,
    kind: JoltAdviceKind,
    public: &crate::stages::stage7::outputs::AdviceAddressPhasePublicOutput<PCS::Field>,
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let source_point = advice_source_point(input, kind)?;
    let cycle_phase = match kind {
        JoltAdviceKind::Trusted => input.stage6.trusted_advice_cycle_phase.as_ref(),
        JoltAdviceKind::Untrusted => input.stage6.untrusted_advice_cycle_phase.as_ref(),
    }
    .ok_or_else(|| VerifierError::MissingOpeningClaim {
        id: advice::cycle_phase_advice_opening(kind),
    })?;
    let scale = layout
        .address_phase_final_output_scale(
            &source_point,
            &cycle_phase.cycle_phase_variables,
            &public.sumcheck_point,
        )
        .map_err(|error| public_error(JoltStageId::AdviceClaimReduction, error))?;
    values.public(
        JoltPublicId::from(AdviceClaimReductionPublic::FinalScale(kind)),
        scale,
    )
}

fn stage6_virtualization_points<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    dimensions: hamming_weight::HammingWeightClaimReductionDimensions,
) -> Result<Vec<Vec<PCS::Field>>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let mut points = Vec::with_capacity(dimensions.layout.total());
    for point in &input
        .stage6
        .instruction_ra_virtualization
        .instruction_ra_opening_points
    {
        points.push(hamming_virtualization_address_point(
            dimensions.log_k_chunk,
            point,
        )?);
    }
    for point in &input.stage6.bytecode_read_raf.bytecode_ra_opening_points {
        points.push(hamming_virtualization_address_point(
            dimensions.log_k_chunk,
            point,
        )?);
    }
    for point in &input.stage6.ram_ra_virtualization.ram_ra_opening_points {
        points.push(hamming_virtualization_address_point(
            dimensions.log_k_chunk,
            point,
        )?);
    }
    Ok(points)
}

fn hamming_virtualization_address_point<F: Field>(
    log_k_chunk: usize,
    point: &[F],
) -> Result<Vec<F>, VerifierError> {
    point.get(..log_k_chunk)
        .map(<[F]>::to_vec)
        .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
            stage: JoltStageId::HammingWeightClaimReduction,
            reason: format!(
                "Stage 6 RA opening point is too short for HammingWeight address chunk: expected at least {log_k_chunk}, got {}",
                point.len()
            ),
        })
}

impl<F: Field> SourceValues<F> {
    fn public(&mut self, id: JoltPublicId, value: F) -> Result<(), VerifierError> {
        push_unique(&mut self.publics, id, value, "public")
    }

    fn challenge(&mut self, id: JoltChallengeId, value: F) -> Result<(), VerifierError> {
        push_unique(&mut self.challenges, id, value, "challenge")
    }

    fn has_public(&self, id: JoltPublicId) -> bool {
        self.publics.iter().any(|(candidate, _)| *candidate == id)
    }

    fn has_challenge(&self, id: JoltChallengeId) -> bool {
        self.challenges
            .iter()
            .any(|(candidate, _)| *candidate == id)
    }
}

fn push_unique<Id, F>(
    values: &mut Vec<(Id, F)>,
    id: Id,
    value: F,
    kind: &'static str,
) -> Result<(), VerifierError>
where
    Id: Copy + PartialEq + core::fmt::Debug,
    F: Field,
{
    if let Some((_, existing)) = values.iter().find(|(candidate, _)| *candidate == id) {
        if *existing != value {
            return Err(VerifierError::BlindFoldConstructionFailed {
                reason: format!("{kind} source {id:?} was assigned inconsistent values"),
            });
        }
        return Ok(());
    }
    values.push((id, value));
    Ok(())
}

fn stage_sumcheck_error<F: Field>(
    stage: JoltStageId,
    error: jolt_sumcheck::SumcheckError<F>,
) -> VerifierError {
    VerifierError::StageClaimSumcheckFailed {
        stage,
        reason: error.to_string(),
    }
}

fn public_error(stage: JoltStageId, error: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage,
        reason: error.to_string(),
    }
}

fn blindfold_error(error: impl ToString) -> VerifierError {
    VerifierError::BlindFoldConstructionFailed {
        reason: error.to_string(),
    }
}
