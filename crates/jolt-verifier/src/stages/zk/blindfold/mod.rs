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
//! In ZK mode this is the link between the committed stage proofs and the
//! final PCS opening proof: no clear output claim scalars are accepted by the
//! verifier, and every hidden scalar that crosses a stage boundary is either in
//! a committed output-claim row or in the final hiding evaluation commitment.
use jolt_blindfold::{BlindFoldProtocol, BlindFoldProtocolBuilder, OpeningAlias};
use jolt_claims::protocols::jolt::relations;
use jolt_claims::{
    derived, opening,
    protocols::jolt::{
        geometry::{
            booleanity::{self, BooleanityDimensions},
            bytecode::{
                self, BytecodeReadRafCommittedEvaluationInputs, BytecodeReadRafEvaluationInputs,
            },
            claim_reductions::{
                advice,
                bytecode::{self as bytecode_reduction, BytecodeOutputWeightInputs},
                hamming_weight, program_image, registers as registers_claim_reduction,
            },
            dimensions::{JoltFormulaDimensions, REGISTER_ADDRESS_BITS},
            instruction, ram,
            spartan::{
                self, branch_flag_product, jump_flag_product, left_instruction_input_product,
                lookup_output_product, next_is_noop_product, outer_opening, outer_uniskip_opening,
                product_outer_opening, product_should_branch_outer_opening,
                product_should_jump_outer_opening, product_uniskip_opening,
                right_instruction_input_product, SpartanOuterDimensions, SpartanProductDimensions,
            },
        },
        AdviceClaimReductionLayout, AdviceClaimReductionPublic, BooleanityChallenge,
        BooleanityPublic, BytecodeClaimReductionChallenge, BytecodeClaimReductionLayout,
        BytecodeClaimReductionPublic, BytecodeReadRafChallenge, BytecodeReadRafPublic,
        HammingWeightClaimReductionChallenge, HammingWeightClaimReductionPublic,
        IncClaimReductionChallenge, IncClaimReductionPublic, InstructionClaimReductionChallenge,
        InstructionClaimReductionPublic, InstructionInputChallenge, InstructionInputPublic,
        InstructionRaVirtualizationChallenge, InstructionRaVirtualizationPublic,
        InstructionReadRafChallenge, InstructionReadRafPublic, JoltAdviceKind, JoltChallengeId,
        JoltCommittedPolynomial, JoltDerivedId, JoltExpr, JoltOpeningId, JoltPolynomialId,
        JoltRelationId, JoltSumcheckDomain, JoltVirtualPolynomial, PrecommittedReductionLayout,
        ProgramImageClaimReductionLayout, ProgramImageClaimReductionPublic,
        RamHammingBooleanityPublic, RamOutputCheckPublic, RamRaClaimReductionChallenge,
        RamRaClaimReductionPublic, RamRaVirtualizationPublic, RamRafEvaluationPublic,
        RamReadWriteChallenge, RamReadWritePublic, RamValCheckChallenge, RamValCheckPublic,
        RegistersClaimReductionChallenge, RegistersClaimReductionPublic,
        RegistersReadWriteChallenge, RegistersReadWritePublic, RegistersValEvaluationPublic,
        SpartanShiftChallenge, SpartanShiftPublic,
    },
    Expr, OutputClaims, Source, SymbolicSumcheck, Term,
};
use jolt_crypto::VectorCommitment;
use jolt_field::{Field, FromPrimitiveInt, RingCore};
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
use jolt_openings::CommitmentScheme;
use jolt_poly::{
    block_selector_mle_msb,
    lagrange::{centered_lagrange_evals, centered_lagrange_kernel},
    range_mask_mle_msb, sparse_segments_mle_msb, try_eq_mle, EqPlusOnePolynomial,
    IdentityPolynomial, LtPolynomial, MultilinearEvaluation, OperandPolynomial, OperandSide,
};
use jolt_program::preprocess::PublicIoMemory;
use jolt_r1cs::constraints::jolt::{
    JoltSpartanOuterPublic, JoltSpartanOuterRemainder, JoltSpartanOuterRemainderChallenges,
    SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE,
    SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
};
use jolt_sumcheck::{
    BatchedCommittedSumcheckConsistency, CommittedSumcheckConsistency, SumcheckDomainSpec,
    SumcheckStatement,
};
use num_traits::{One, Zero};

use super::{
    inputs::BlindFoldInputs,
    outputs::{BlindFoldOutput, CommittedOutputClaimOutput},
};
use crate::stages::{
    stage6::{outputs::BytecodeReductionWeights, verify},
    stage8::outputs::Stage8OpeningId,
};
use crate::VerifierError;

mod stage1;
mod stage2;
mod stage3;
mod stage4;
mod stage5;
mod stage6;
mod stage7;

type Builder<F, C> = BlindFoldProtocolBuilder<F, VerifierOpeningId, C, VerifierPublicId>;
type VerifierExpr<F> = Expr<F, VerifierOpeningId, VerifierPublicId>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VerifierOpeningId {
    Jolt(JoltOpeningId),
}

impl From<JoltOpeningId> for VerifierOpeningId {
    fn from(id: JoltOpeningId) -> Self {
        Self::Jolt(id)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VerifierPublicId {
    Jolt(JoltDerivedId),
    SpartanOuter(JoltSpartanOuterPublic),
    /// Gamma values that remain as `JoltChallengeId` variants (not moved to Public) but are
    /// treated as public inputs in the BlindFold R1CS wiring.
    Challenge(JoltChallengeId),
}

impl From<JoltDerivedId> for VerifierPublicId {
    fn from(id: JoltDerivedId) -> Self {
        Self::Jolt(id)
    }
}

#[derive(Default)]
struct SourceValues<F: Field> {
    publics: Vec<(VerifierPublicId, F)>,
}

pub fn build<PCS, VC, ZkProof>(
    input: BlindFoldInputs<'_, PCS, VC, ZkProof>,
) -> Result<BlindFoldOutput<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    VC::Output: Clone,
{
    let mut values = SourceValues::default();
    let mut builder = BlindFoldProtocol::<PCS::Field, VC::Output>::builder::<
        VerifierOpeningId,
        VerifierPublicId,
        usize,
    >();

    builder = stage1::add_stage1(&input, builder, &mut values)?;
    builder = stage2::add_stage2(&input, builder, &mut values)?;
    builder = stage3::add_stage3(&input, builder, &mut values)?;
    builder = stage4::add_stage4(&input, builder, &mut values)?;
    builder = stage5::add_stage5(&input, builder, &mut values)?;
    builder = stage6::add_stage6(&input, builder, &mut values)?;
    builder = stage7::add_stage7(&input, builder, &mut values)?;

    for (id, value) in values.publics {
        builder = builder.public(id, value);
    }

    let protocol = builder
        .final_opening(
            map_stage8_opening_ids(input.stage8.opening_ids.clone()),
            input.stage8.constraint_coefficients.clone(),
            input.stage8.hiding_evaluation_commitment,
        )
        .build()
        .map_err(blindfold_error)?;

    Ok(BlindFoldOutput { protocol })
}

#[expect(
    clippy::too_many_arguments,
    reason = "BlindFold stages are deliberately explicit."
)]
fn add_batched_stage<F, C>(
    builder: Builder<F, C>,
    name: &'static str,
    batch_domain: JoltSumcheckDomain,
    rounds: &[usize],
    inputs: &[JoltExpr<F>],
    outputs: &[JoltExpr<F>],
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
    if rounds.is_empty() {
        return Err(VerifierError::BlindFoldConstructionFailed {
            reason: format!("{name}: empty batched claims"),
        });
    }
    let domain = domain_spec(batch_domain);
    let input_claim = batched_input_expr(rounds, inputs, consistency);
    let output_claim = batched_output_expr(outputs, consistency);
    add_stage(
        builder,
        name,
        SumcheckStatement::new(consistency.max_num_vars, consistency.max_degree),
        domain,
        consistency.consistency.clone(),
        output_claims,
        values,
        map_jolt_opening_ids(opening_ids),
        map_jolt_aliases(aliases),
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
    opening_ids: Vec<VerifierOpeningId>,
    aliases: Vec<OpeningAlias<VerifierOpeningId>>,
    input_claim: VerifierExpr<F>,
    output_claim: VerifierExpr<F>,
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
    rounds: &[usize],
    inputs: &[JoltExpr<F>],
    consistency: &BatchedCommittedSumcheckConsistency<F, C>,
) -> VerifierExpr<F>
where
    F: Field,
{
    inputs
        .iter()
        .zip(rounds)
        .zip(&consistency.batching_coefficients)
        .fold(
            VerifierExpr::zero(),
            |acc, ((input, instance_rounds), coefficient)| {
                let scale = *coefficient * F::pow2(consistency.max_num_vars - *instance_rounds);
                acc + scale_expr(map_jolt_expr(input.clone()), scale)
            },
        )
}

fn batched_output_expr<F, C>(
    outputs: &[JoltExpr<F>],
    consistency: &BatchedCommittedSumcheckConsistency<F, C>,
) -> VerifierExpr<F>
where
    F: Field,
{
    outputs
        .iter()
        .zip(&consistency.batching_coefficients)
        .fold(VerifierExpr::zero(), |acc, (output, coefficient)| {
            acc + scale_expr(map_jolt_expr(output.clone()), *coefficient)
        })
}

fn scale_expr<F: Field>(mut expr: VerifierExpr<F>, scale: F) -> VerifierExpr<F> {
    if scale.is_zero() {
        return VerifierExpr::zero();
    }
    for term in &mut expr.terms {
        term.coefficient *= scale;
    }
    expr
}

fn map_jolt_opening_ids(opening_ids: Vec<JoltOpeningId>) -> Vec<VerifierOpeningId> {
    opening_ids
        .into_iter()
        .map(VerifierOpeningId::from)
        .collect()
}

fn map_stage8_opening_ids(opening_ids: Vec<Stage8OpeningId>) -> Vec<VerifierOpeningId> {
    opening_ids
        .into_iter()
        .map(|id| match id {
            Stage8OpeningId::Jolt(id) => VerifierOpeningId::Jolt(id),
        })
        .collect()
}

fn map_jolt_aliases(
    aliases: Vec<OpeningAlias<JoltOpeningId>>,
) -> Vec<OpeningAlias<VerifierOpeningId>> {
    aliases
        .into_iter()
        .map(|alias| OpeningAlias::new(alias.alias.into(), alias.source.into()))
        .collect()
}

fn map_jolt_expr<F: Field>(expr: JoltExpr<F>) -> VerifierExpr<F> {
    Expr {
        terms: expr
            .terms
            .into_iter()
            .map(|term| Term {
                coefficient: term.coefficient,
                factors: term
                    .factors
                    .into_iter()
                    .map(|source| match source {
                        Source::Opening(id) => Source::Opening(id.into()),
                        Source::Derived(id) => Source::Derived(VerifierPublicId::Jolt(id)),
                        Source::Challenge(id) => Source::Derived(VerifierPublicId::Challenge(id)),
                    })
                    .collect(),
            })
            .collect(),
    }
}

fn require_expr_sources<F: Field>(
    stage: &'static str,
    expression: &'static str,
    expr: &VerifierExpr<F>,
    values: &SourceValues<F>,
) -> Result<(), VerifierError> {
    for factor in expr.terms.iter().flat_map(|term| term.factors.iter()) {
        if let Source::Derived(id) = factor {
            if !values.has_public(*id) {
                return Err(VerifierError::BlindFoldConstructionFailed {
                    reason: format!("{stage} {expression} is missing public source {id:?}"),
                });
            }
        }
    }
    Ok(())
}

fn domain_spec(domain: JoltSumcheckDomain) -> SumcheckDomainSpec {
    match domain {
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
        input.preprocessing.program.bytecode_len(),
        input.checked.ram_K,
    ))
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::BytecodeReadRaf,
        reason: error.to_string(),
    })
}

/// Recompute stage 1's remainder cycle point — the low half of the Spartan outer
/// remainder sumcheck point — from the singleton remainder batch's committed
/// challenges. The stage-2 carrier stores the same value as `product_tau_low` (for
/// downstream relation construction), but BlindFold reconstructs it here so the
/// BakedPublicInputs derivation stays independent. Orientation matches
/// `stage2/verify.rs::verify_product_uniskip`: drop the leading challenge, then
/// reverse (`reverse(challenges()[1..])`). Used as `product_tau_low` by stages 2 and
/// 3 and as the stage-1 cycle binding within `add_stage6_publics_and_challenges`.
fn stage1_remainder_cycle<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
) -> Vec<PCS::Field>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    input.stage1.remainder_consistency.challenges()[1..]
        .iter()
        .rev()
        .copied()
        .collect()
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
            stage: JoltRelationId::RamOutputCheck,
            reason: error.to_string(),
        }
    })?;
    let output_eq = try_eq_mle(output_address_challenges, ram_output_address)
        .map_err(|error| public_error(JoltRelationId::RamOutputCheck, error))?;
    let output_mask = range_mask_mle_msb(
        public_memory.io_mask_start,
        public_memory.io_mask_end,
        ram_output_address,
    )
    .map_err(|error| public_error(JoltRelationId::RamOutputCheck, error))?;
    let io_num_vars = public_memory.io_num_vars();
    let (r_hi, r_lo) = ram_output_address.split_at(
        ram_output_address
            .len()
            .checked_sub(io_num_vars)
            .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamOutputCheck,
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
    // WARNING: contribution order and selectors must stay in lockstep with the
    // clear-path decomposition in stage4/verify.rs.
    let mut contributions = Vec::new();
    if input.checked.precommitted.program_image.is_some() {
        contributions.push(ram::RamValCheckInitContribution::program_image(
            -PCS::Field::one(),
        ));
    }
    if input.proof.untrusted_advice_commitment.is_some() {
        let selector = advice_selector(input, JoltAdviceKind::Untrusted, &r_address)?;
        contributions.push(ram::RamValCheckInitContribution::untrusted(-selector.0));
    }
    if input.checked.trusted_advice_commitment_present {
        let selector = advice_selector(input, JoltAdviceKind::Trusted, &r_address)?;
        contributions.push(ram::RamValCheckInitContribution::trusted(-selector.0));
    }
    Ok(ram::RamValCheckInit::decomposed(
        input.stage4.ram_val_check_public_eval,
        contributions,
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
        .output_points
        .ram_read_write_point()
        .get(..log_k)
        .map(<[PCS::Field]>::to_vec)
        .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamValCheck,
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
        .map_err(|error| public_error(JoltRelationId::RamValCheck, error))?
        as u128;
    let advice_num_vars = ((max_size as usize) / 8).next_power_of_two().ilog2() as usize;
    let selector = block_selector_mle_msb(start_index, advice_num_vars, r_address)
        .map_err(|error| public_error(JoltRelationId::RamValCheck, error))?;
    let opening_point = r_address
        .get(r_address.len().checked_sub(advice_num_vars).ok_or_else(|| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamValCheck,
                reason: format!(
                    "{kind:?} advice point needs {advice_num_vars} variables but RAM address has {}",
                    r_address.len()
                ),
            }
        })?..)
        .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamValCheck,
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
    Option<relations::claim_reductions::advice::CyclePhase>,
)
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let layout = advice_layout(input, kind);
    let claim = layout.as_ref().map(|layout| {
        relations::claim_reductions::advice::CyclePhase::new((kind, layout.dimensions()))
    });
    (layout, claim)
}

fn advice_address_claim<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    kind: JoltAdviceKind,
) -> (
    Option<AdviceClaimReductionLayout>,
    Option<relations::claim_reductions::advice::AddressPhase>,
)
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let layout = advice_layout(input, kind);
    let claim = layout.as_ref().and_then(|layout| {
        layout.dimensions().has_address_phase().then(|| {
            relations::claim_reductions::advice::AddressPhase::new((kind, layout.dimensions()))
        })
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
    input.checked.precommitted.advice(kind).cloned()
}

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 6 has several protocol components."
)]
fn add_stage6_publics_and_challenges<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    values: &mut SourceValues<PCS::Field>,
    bytecode_address_rounds: usize,
    bytecode_rounds: usize,
    booleanity_address_rounds: usize,
    booleanity_rounds: usize,
    ram_hamming_rounds: usize,
    ram_ra_rounds: usize,
    instruction_ra_rounds: usize,
    inc_rounds: usize,
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let log_t = input.checked.trace_length.ilog2() as usize;
    let log_k = input.checked.ram_K.ilog2() as usize;
    let trace_dimensions = jolt_claims::protocols::jolt::TraceDimensions::new(log_t);

    values.public(
        VerifierPublicId::Challenge(JoltChallengeId::from(BytecodeReadRafChallenge::Gamma)),
        input.stage6.challenges.bytecode_gamma_powers[1],
    )?;
    values.public(
        VerifierPublicId::Challenge(JoltChallengeId::from(BytecodeReadRafChallenge::Stage1Gamma)),
        input.stage6.challenges.stage1_gammas[1],
    )?;
    values.public(
        VerifierPublicId::Challenge(JoltChallengeId::from(BytecodeReadRafChallenge::Stage2Gamma)),
        input.stage6.challenges.stage2_gammas[1],
    )?;
    values.public(
        VerifierPublicId::Challenge(JoltChallengeId::from(BytecodeReadRafChallenge::Stage3Gamma)),
        input.stage6.challenges.stage3_gammas[1],
    )?;
    values.public(
        VerifierPublicId::Challenge(JoltChallengeId::from(BytecodeReadRafChallenge::Stage4Gamma)),
        input.stage6.challenges.stage4_gammas[1],
    )?;
    values.public(
        VerifierPublicId::Challenge(JoltChallengeId::from(BytecodeReadRafChallenge::Stage5Gamma)),
        input.stage6.challenges.stage5_gammas[1],
    )?;
    values.public(
        VerifierPublicId::Challenge(JoltChallengeId::from(BooleanityChallenge::Gamma)),
        input.stage6.challenges.booleanity_gamma,
    )?;
    values.public(
        VerifierPublicId::Challenge(JoltChallengeId::from(
            InstructionRaVirtualizationChallenge::Gamma,
        )),
        input
            .stage6
            .challenges
            .instruction_ra_gamma_powers
            .get(1)
            .copied()
            .unwrap_or_else(PCS::Field::one),
    )?;
    values.public(
        VerifierPublicId::Challenge(JoltChallengeId::from(IncClaimReductionChallenge::Gamma)),
        input.stage6.challenges.inc_gamma,
    )?;

    let bytecode_address_point = input
        .stage6
        .address_phase_consistency
        .try_instance_point(bytecode_address_rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::BytecodeReadRaf, error))?;
    let bytecode_r_address = bytecode_address_point
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let bytecode_point = input
        .stage6
        .batch_consistency
        .try_instance_point(bytecode_rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::BytecodeReadRaf, error))?;
    let bytecode_r_cycle = bytecode_point.iter().rev().copied().collect::<Vec<_>>();
    let stage1_remainder_challenges = input.stage1.remainder_consistency.challenges();
    let stage1_cycle = stage1_remainder_cycle(input);
    let stage2_product_point = input
        .stage2
        .batch_consistency
        .try_instance_point(log_t)
        .map_err(|error| {
            stage_sumcheck_error(JoltRelationId::SpartanProductVirtualization, error)
        })?;
    let stage2_cycle = stage2_product_point
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let stage3_shift_point = input
        .stage3
        .batch_consistency
        .try_instance_point(log_t)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::SpartanShift, error))?;
    let stage3_cycle = stage3_shift_point.iter().rev().copied().collect::<Vec<_>>();
    let stage4_cycle =
        &input.stage4.output_points.registers_read_write_point()[REGISTER_ADDRESS_BITS..];
    let stage5_cycle =
        &input.stage5.output_points.registers_opening_point()[REGISTER_ADDRESS_BITS..];
    let entry_bytecode_index = input
        .preprocessing
        .program
        .entry_bytecode_index()
        .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: "entry address was not found in bytecode preprocessing".to_string(),
        })?;
    if input.checked.precommitted.bytecode.is_some() {
        let committed_public_values = bytecode::read_raf_committed_public_values::<PCS::Field>(
            BytecodeReadRafCommittedEvaluationInputs {
                r_address: &bytecode_r_address,
                r_cycle: &bytecode_r_cycle,
                stage_cycle_points: [
                    &stage1_cycle,
                    &stage2_cycle,
                    &stage3_cycle,
                    stage4_cycle,
                    stage5_cycle,
                ],
                entry_bytecode_index,
            },
        );
        for (index, stage_cycle_eq) in committed_public_values.stage_cycle_eqs.iter().enumerate() {
            values.public(
                JoltDerivedId::from(BytecodeReadRafPublic::StageCycleEq(index)),
                *stage_cycle_eq,
            )?;
        }
        values.public(
            JoltDerivedId::from(BytecodeReadRafPublic::SpartanOuterRaf),
            committed_public_values.spartan_outer_raf,
        )?;
        values.public(
            JoltDerivedId::from(BytecodeReadRafPublic::SpartanShiftRaf),
            committed_public_values.spartan_shift_raf,
        )?;
        values.public(
            JoltDerivedId::from(BytecodeReadRafPublic::Entry),
            committed_public_values.entry,
        )?;
    } else {
        let full_program = input.preprocessing.program.as_full().ok_or_else(|| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::BytecodeReadRaf,
                reason: "full bytecode table is unavailable".to_string(),
            }
        })?;
        let bytecode_public_values =
            bytecode::read_raf_public_values::<PCS::Field>(BytecodeReadRafEvaluationInputs {
                bytecode: &full_program.bytecode.bytecode,
                r_address: &bytecode_r_address,
                r_cycle: &bytecode_r_cycle,
                stage_cycle_points: [
                    &stage1_cycle,
                    &stage2_cycle,
                    &stage3_cycle,
                    stage4_cycle,
                    stage5_cycle,
                ],
                register_read_write_point: &input.stage4.output_points.registers_read_write_point()
                    [..REGISTER_ADDRESS_BITS],
                register_val_evaluation_point: &input
                    .stage5
                    .output_points
                    .registers_opening_point()[..REGISTER_ADDRESS_BITS],
                entry_bytecode_index,
                stage1_gammas: &input.stage6.challenges.stage1_gammas,
                stage2_gammas: &input.stage6.challenges.stage2_gammas,
                stage3_gammas: &input.stage6.challenges.stage3_gammas,
                stage4_gammas: &input.stage6.challenges.stage4_gammas,
                stage5_gammas: &input.stage6.challenges.stage5_gammas,
            })
            .map_err(|error| public_error(JoltRelationId::BytecodeReadRaf, error))?;
        for (index, stage_value) in bytecode_public_values.stage_values.iter().enumerate() {
            values.public(
                JoltDerivedId::from(BytecodeReadRafPublic::StageValue(index)),
                *stage_value,
            )?;
        }
        values.public(
            JoltDerivedId::from(BytecodeReadRafPublic::SpartanOuterRaf),
            bytecode_public_values.spartan_outer_raf,
        )?;
        values.public(
            JoltDerivedId::from(BytecodeReadRafPublic::SpartanShiftRaf),
            bytecode_public_values.spartan_shift_raf,
        )?;
        values.public(
            JoltDerivedId::from(BytecodeReadRafPublic::Entry),
            bytecode_public_values.entry,
        )?;
    }

    let booleanity_address_point = input
        .stage6
        .address_phase_consistency
        .try_instance_point(booleanity_address_rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::Booleanity, error))?;
    let booleanity_point = input
        .stage6
        .batch_consistency
        .try_instance_point(booleanity_rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::Booleanity, error))?;
    let reference_eq_point = input
        .stage6
        .challenges
        .booleanity_reference_address
        .iter()
        .rev()
        .chain(
            input
                .stage6
                .challenges
                .booleanity_reference_cycle
                .iter()
                .rev(),
        )
        .copied()
        .collect::<Vec<_>>();
    let booleanity_full_point = [
        booleanity_address_point.as_slice(),
        booleanity_point.as_slice(),
    ]
    .concat();
    values.public(
        JoltDerivedId::from(BooleanityPublic::EqAddressCycle),
        try_eq_mle(&booleanity_full_point, &reference_eq_point)
            .map_err(|error| public_error(JoltRelationId::Booleanity, error))?,
    )?;

    let ram_hamming_point = input
        .stage6
        .batch_consistency
        .try_instance_point(ram_hamming_rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RamHammingBooleanity, error))?;
    let stage1_cycle_binding = &stage1_remainder_challenges[1..];
    values.public(
        JoltDerivedId::from(RamHammingBooleanityPublic::EqCycle),
        try_eq_mle(&ram_hamming_point, stage1_cycle_binding)
            .map_err(|error| public_error(JoltRelationId::RamHammingBooleanity, error))?,
    )?;

    let ram_ra_point = input
        .stage6
        .batch_consistency
        .try_instance_point(ram_ra_rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RamRaVirtualization, error))?;
    let ram_ra_cycle = trace_dimensions
        .cycle_opening_point(&ram_ra_point)
        .map_err(|error| public_error(JoltRelationId::RamRaVirtualization, error))?;
    let ram_reduced_cycle = &input.stage5.output_points.ram_reduced_opening_point()[log_k..];
    values.public(
        JoltDerivedId::from(RamRaVirtualizationPublic::EqCycle),
        try_eq_mle(ram_reduced_cycle, &ram_ra_cycle)
            .map_err(|error| public_error(JoltRelationId::RamRaVirtualization, error))?,
    )?;

    let instruction_ra_point = input
        .stage6
        .batch_consistency
        .try_instance_point(instruction_ra_rounds)
        .map_err(|error| {
            stage_sumcheck_error(JoltRelationId::InstructionRaVirtualization, error)
        })?;
    let instruction_ra_cycle = trace_dimensions
        .cycle_opening_point(&instruction_ra_point)
        .map_err(|error| public_error(JoltRelationId::InstructionRaVirtualization, error))?;
    values.public(
        JoltDerivedId::from(InstructionRaVirtualizationPublic::EqCycle),
        try_eq_mle(
            input.stage5.output_points.instruction_r_cycle(),
            &instruction_ra_cycle,
        )
        .map_err(|error| public_error(JoltRelationId::InstructionRaVirtualization, error))?,
    )?;

    let inc_point = input
        .stage6
        .batch_consistency
        .try_instance_point(inc_rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::IncClaimReduction, error))?;
    let inc_opening_point = trace_dimensions
        .cycle_opening_point(&inc_point)
        .map_err(|error| public_error(JoltRelationId::IncClaimReduction, error))?;
    values.public(
        JoltDerivedId::from(IncClaimReductionPublic::EqRamReadWrite),
        try_eq_mle(
            &inc_opening_point,
            &input.stage2.output_points.ram_read_write_point()[log_k..],
        )
        .map_err(|error| public_error(JoltRelationId::IncClaimReduction, error))?,
    )?;
    values.public(
        JoltDerivedId::from(IncClaimReductionPublic::EqRamValCheck),
        try_eq_mle(
            &inc_opening_point,
            &input.stage4.output_points.ram_val_check_point()[log_k..],
        )
        .map_err(|error| public_error(JoltRelationId::IncClaimReduction, error))?,
    )?;
    values.public(
        JoltDerivedId::from(IncClaimReductionPublic::EqRegistersReadWrite),
        try_eq_mle(
            &inc_opening_point,
            &input.stage4.output_points.registers_read_write_point()[REGISTER_ADDRESS_BITS..],
        )
        .map_err(|error| public_error(JoltRelationId::IncClaimReduction, error))?,
    )?;
    values.public(
        JoltDerivedId::from(IncClaimReductionPublic::EqRegistersValEvaluation),
        try_eq_mle(
            &inc_opening_point,
            &input.stage5.output_points.registers_opening_point()[REGISTER_ADDRESS_BITS..],
        )
        .map_err(|error| public_error(JoltRelationId::IncClaimReduction, error))?,
    )?;

    Ok(())
}

fn bytecode_reduction_weights<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    layout: &BytecodeClaimReductionLayout,
) -> Result<BytecodeReductionWeights<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let eta = input
        .stage6
        .challenges
        .bytecode_reduction_eta
        .ok_or_else(|| VerifierError::MissingStageClaimChallenge {
            id: JoltChallengeId::from(BytecodeClaimReductionChallenge::Eta),
        })?;
    verify::bytecode_reduction_weights(
        layout,
        verify::BytecodeReductionWeightInputs {
            eta,
            stage1_gammas: &input.stage6.challenges.stage1_gammas,
            stage2_gammas: &input.stage6.challenges.stage2_gammas,
            stage3_gammas: &input.stage6.challenges.stage3_gammas,
            stage4_gammas: &input.stage6.challenges.stage4_gammas,
            stage5_gammas: &input.stage6.challenges.stage5_gammas,
            register_read_write_point: &input.stage4.output_points.registers_read_write_point()
                [..REGISTER_ADDRESS_BITS],
            register_val_evaluation_point: &input.stage5.output_points.registers_opening_point()
                [..REGISTER_ADDRESS_BITS],
            bytecode_r_address: &input
                .stage6
                .output_points
                .address_phase
                .bytecode_read_raf
                .intermediate,
        },
    )
}

fn add_bytecode_chunk_weight_publics<F: Field>(
    values: &mut SourceValues<F>,
    chunk_weights: Vec<F>,
) -> Result<(), VerifierError> {
    for (chunk_idx, weight) in chunk_weights.into_iter().enumerate() {
        values.public(
            JoltDerivedId::from(BytecodeClaimReductionPublic::ChunkOutputWeight(chunk_idx)),
            weight,
        )?;
    }
    Ok(())
}

fn add_bytecode_reduction_cycle_publics<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    values: &mut SourceValues<PCS::Field>,
    layout: &BytecodeClaimReductionLayout,
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    if layout.dimensions().has_address_phase() {
        return Ok(());
    }
    let opening_point = input
        .stage6
        .output_points
        .bytecode_reduction_opening_point()
        .ok_or_else(|| VerifierError::MissingOpeningClaim {
            id: bytecode_reduction::cycle_phase_intermediate_opening(),
        })?;
    let weights = bytecode_reduction_weights(input, layout)?;
    // The cycle scale recovered from the produced opening point equals the
    // sumcheck-point form (`cycle_phase_permuted_*` agree — unit-tested in
    // `claim_reductions::precommitted`), matching the clear relation's path.
    let chunk_weights = layout
        .cycle_phase_final_output_weights_at_opening_point(
            BytecodeOutputWeightInputs {
                r_bc: &weights.r_bc,
                chunk_rbc_weights: &weights.chunk_rbc_weights,
                lane_weights: &weights.lane_weights,
            },
            opening_point,
        )
        .map_err(|error| public_error(JoltRelationId::BytecodeClaimReductionCyclePhase, error))?;
    add_bytecode_chunk_weight_publics(values, chunk_weights)
}

fn add_bytecode_reduction_address_publics<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    values: &mut SourceValues<PCS::Field>,
    layout: &BytecodeClaimReductionLayout,
    sumcheck_point: &[PCS::Field],
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let cycle_phase_variables = input
        .stage6
        .output_points
        .bytecode_cycle_phase_variables()
        .ok_or_else(|| VerifierError::MissingOpeningClaim {
            id: bytecode_reduction::cycle_phase_intermediate_opening(),
        })?;
    let weights = bytecode_reduction_weights(input, layout)?;
    let chunk_weights = layout
        .address_phase_final_output_weights(
            BytecodeOutputWeightInputs {
                r_bc: &weights.r_bc,
                chunk_rbc_weights: &weights.chunk_rbc_weights,
                lane_weights: &weights.lane_weights,
            },
            &cycle_phase_variables,
            sumcheck_point,
        )
        .map_err(|error| public_error(JoltRelationId::BytecodeClaimReduction, error))?;
    add_bytecode_chunk_weight_publics(values, chunk_weights)
}

fn add_program_image_reduction_cycle_publics<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    values: &mut SourceValues<PCS::Field>,
    layout: &ProgramImageClaimReductionLayout,
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    if layout.dimensions().has_address_phase() {
        return Ok(());
    }
    let opening_point = input
        .stage6
        .output_points
        .program_image_opening_point()
        .ok_or_else(|| VerifierError::MissingOpeningClaim {
            id: program_image::cycle_phase_program_image_opening(),
        })?;
    let r_addr_rw = ram_val_check_address(input)?;
    let scale = layout
        .cycle_phase_scale_at_opening_point(&r_addr_rw, opening_point)
        .map_err(|error| {
            public_error(JoltRelationId::ProgramImageClaimReductionCyclePhase, error)
        })?;
    values.public(
        JoltDerivedId::from(ProgramImageClaimReductionPublic::FinalScale),
        scale,
    )
}

fn add_program_image_reduction_address_publics<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    values: &mut SourceValues<PCS::Field>,
    layout: &ProgramImageClaimReductionLayout,
    sumcheck_point: &[PCS::Field],
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let cycle_phase_variables = input
        .stage6
        .output_points
        .program_image_cycle_phase_variables()
        .ok_or_else(|| VerifierError::MissingOpeningClaim {
            id: program_image::cycle_phase_program_image_opening(),
        })?;
    let r_addr_rw = ram_val_check_address(input)?;
    let scale = layout
        .address_phase_final_output_scale(&r_addr_rw, &cycle_phase_variables, sumcheck_point)
        .map_err(|error| public_error(JoltRelationId::ProgramImageClaimReduction, error))?;
    values.public(
        JoltDerivedId::from(ProgramImageClaimReductionPublic::FinalScale),
        scale,
    )
}

fn add_advice_cycle_publics<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    values: &mut SourceValues<PCS::Field>,
    layout: &AdviceClaimReductionLayout,
    kind: JoltAdviceKind,
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    // The cycle-phase relation references `FinalScale` only when the reduction
    // finalizes at the cycle-phase handoff; otherwise the stage 7 address
    // phase supplies it.
    if layout.dimensions().has_address_phase() {
        return Ok(());
    }
    let opening_point = input
        .stage6
        .output_points
        .advice_cycle_phase_opening_point(kind)
        .ok_or_else(|| VerifierError::MissingOpeningClaim {
            id: advice::cycle_phase_advice_opening(kind),
        })?;
    let source_point = advice_source_point(input, kind)?;
    let scale = layout
        .cycle_phase_scale_at_opening_point(&source_point, opening_point)
        .map_err(|error| public_error(JoltRelationId::AdviceClaimReductionCyclePhase, error))?;
    values.public(
        JoltDerivedId::from(AdviceClaimReductionPublic::FinalScale(kind)),
        scale,
    )
}

fn add_advice_address_publics<PCS, VC, ZkProof>(
    input: &BlindFoldInputs<'_, PCS, VC, ZkProof>,
    values: &mut SourceValues<PCS::Field>,
    layout: &AdviceClaimReductionLayout,
    kind: JoltAdviceKind,
    sumcheck_point: &[PCS::Field],
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let source_point = advice_source_point(input, kind)?;
    let cycle_phase_variables = input
        .stage6
        .output_points
        .advice_cycle_phase_variables(kind)
        .ok_or_else(|| VerifierError::MissingOpeningClaim {
            id: advice::cycle_phase_advice_opening(kind),
        })?;
    let scale = layout
        .address_phase_final_output_scale(&source_point, &cycle_phase_variables, sumcheck_point)
        .map_err(|error| public_error(JoltRelationId::AdviceClaimReduction, error))?;
    values.public(
        JoltDerivedId::from(AdviceClaimReductionPublic::FinalScale(kind)),
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
    let output_points = &input.stage6.output_points;
    let mut points = Vec::with_capacity(dimensions.layout.total());
    for point in &output_points
        .cycle_phase
        .instruction_ra_virtualization
        .committed_instruction_ra
    {
        points.push(hamming_virtualization_address_point(
            dimensions.log_k_chunk,
            point,
        )?);
    }
    for point in &output_points.cycle_phase.bytecode_read_raf.bytecode_ra {
        points.push(hamming_virtualization_address_point(
            dimensions.log_k_chunk,
            point,
        )?);
    }
    for point in &output_points.cycle_phase.ram_ra_virtualization.ram_ra {
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
            stage: JoltRelationId::HammingWeightClaimReduction,
            reason: format!(
                "Stage 6 RA opening point is too short for HammingWeight address chunk: expected at least {log_k_chunk}, got {}",
                point.len()
            ),
        })
}

impl<F: Field> SourceValues<F> {
    fn public(&mut self, id: impl Into<VerifierPublicId>, value: F) -> Result<(), VerifierError> {
        push_unique(&mut self.publics, id.into(), value, "public")
    }

    fn has_public(&self, id: VerifierPublicId) -> bool {
        self.publics.iter().any(|(candidate, _)| *candidate == id)
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
    stage: JoltRelationId,
    error: jolt_sumcheck::SumcheckError<F>,
) -> VerifierError {
    VerifierError::StageClaimSumcheckFailed {
        stage,
        reason: error.to_string(),
    }
}

fn public_error(stage: JoltRelationId, error: impl ToString) -> VerifierError {
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
