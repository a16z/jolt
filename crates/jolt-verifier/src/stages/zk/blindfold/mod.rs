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
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::{
    formulas::{
        bytecode as field_bytecode,
        claim_reductions::{
            increments as field_increments, registers as field_registers_claim_reduction,
        },
        product as field_product, registers as field_registers, spartan as field_spartan,
    },
    FieldInlineChallengeId, FieldInlineOpeningId, FieldInlinePublicId,
    FieldInlineVirtualPolynomial, FieldRegistersClaimReductionChallenge,
    FieldRegistersIncClaimReductionChallenge, FieldRegistersIncClaimReductionPublic,
    FieldRegistersReadWriteChallenge, FieldRegistersTraceDimensions,
    FieldRegistersValEvaluationChallenge,
};
use jolt_claims::{
    opening,
    protocols::jolt::{
        formulas::{
            booleanity::{self, BooleanityDimensions},
            bytecode::{self, BytecodeReadRafEvaluationInputs},
            claim_reductions::{advice, hamming_weight, increments},
            dimensions::{JoltFormulaDimensions, JoltSumcheckSpec, REGISTER_ADDRESS_BITS},
            instruction, ram, registers,
            spartan::{
                self, outer_opening, outer_uniskip_opening, product_outer_opening,
                product_remainder_output_openings, product_should_branch_outer_opening,
                product_should_jump_outer_opening, product_uniskip_opening, shift_output_openings,
                SpartanOuterDimensions, SpartanProductDimensions,
            },
        },
        AdviceClaimReductionLayout, AdviceClaimReductionPublic, BooleanityChallenge,
        BooleanityPublic, BytecodeReadRafChallenge, BytecodeReadRafPublic,
        HammingWeightClaimReductionChallenge, HammingWeightClaimReductionPublic,
        IncClaimReductionChallenge, IncClaimReductionPublic, InstructionClaimReductionChallenge,
        InstructionInputChallenge, InstructionRaVirtualizationChallenge,
        InstructionReadRafChallenge, JoltAdviceKind, JoltChallengeId, JoltOpeningId, JoltPublicId,
        JoltRelationClaims, JoltRelationId, JoltSumcheckDomain, RamHammingBooleanityChallenge,
        RamOutputCheckPublic, RamRaClaimReductionChallenge, RamRaClaimReductionPublic,
        RamRaVirtualizationChallenge, RamRafEvaluationPublic, RamReadWriteChallenge,
        RamValCheckChallenge, RegistersClaimReductionChallenge, RegistersReadWriteChallenge,
        RegistersValEvaluationChallenge, SpartanShiftChallenge, SpartanShiftPublic,
    },
    public, Expr, Source, Term,
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
    SPARTAN_OUTER_REMAINDER_DEGREE, SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE,
    SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE, SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
    SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
};
use jolt_sumcheck::{
    BatchedCommittedSumcheckConsistency, CommittedSumcheckConsistency, SumcheckDomainSpec,
    SumcheckStatement,
};
use num_traits::One;

use super::{
    inputs::BlindFoldInputs,
    outputs::{BlindFoldOutput, CommittedOutputClaimOutput},
};
use crate::stages::{
    stage1::inputs::{spartan_outer_opening_order, Stage1SpartanOuterOpening},
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

type Builder<F, C> =
    BlindFoldProtocolBuilder<F, VerifierOpeningId, C, VerifierPublicId, VerifierChallengeId>;
type VerifierExpr<F> = Expr<F, VerifierOpeningId, VerifierPublicId, VerifierChallengeId>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VerifierOpeningId {
    Jolt(JoltOpeningId),
    #[cfg(feature = "field-inline")]
    FieldInline(FieldInlineOpeningId),
}

impl From<JoltOpeningId> for VerifierOpeningId {
    fn from(id: JoltOpeningId) -> Self {
        Self::Jolt(id)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VerifierPublicId {
    Jolt(JoltPublicId),
    SpartanOuter(JoltSpartanOuterPublic),
    #[cfg(feature = "field-inline")]
    FieldInline(FieldInlinePublicId),
}

impl From<JoltPublicId> for VerifierPublicId {
    fn from(id: JoltPublicId) -> Self {
        Self::Jolt(id)
    }
}

#[cfg(feature = "field-inline")]
impl From<FieldInlinePublicId> for VerifierPublicId {
    fn from(id: FieldInlinePublicId) -> Self {
        Self::FieldInline(id)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VerifierChallengeId {
    Jolt(JoltChallengeId),
    #[cfg(feature = "field-inline")]
    FieldInline(FieldInlineChallengeId),
}

impl From<JoltChallengeId> for VerifierChallengeId {
    fn from(id: JoltChallengeId) -> Self {
        Self::Jolt(id)
    }
}

#[cfg(feature = "field-inline")]
impl From<FieldInlineChallengeId> for VerifierChallengeId {
    fn from(id: FieldInlineChallengeId) -> Self {
        Self::FieldInline(id)
    }
}

#[derive(Default)]
struct SourceValues<F: Field> {
    publics: Vec<(VerifierPublicId, F)>,
    challenges: Vec<(VerifierChallengeId, F)>,
}

pub fn build<PCS, VC>(
    input: BlindFoldInputs<'_, PCS, VC>,
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
        VerifierChallengeId,
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
    for (id, value) in values.challenges {
        builder = builder.challenge(id, value);
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
    claims: &[JoltRelationClaims<F>],
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
    claims: &[JoltRelationClaims<F>],
    consistency: &BatchedCommittedSumcheckConsistency<F, C>,
) -> VerifierExpr<F>
where
    F: Field,
{
    claims.iter().zip(&consistency.batching_coefficients).fold(
        VerifierExpr::zero(),
        |acc, (claim, coefficient)| {
            let scale = *coefficient * F::pow2(consistency.max_num_vars - claim.sumcheck.rounds);
            acc + scale_expr(map_jolt_expr(claim.input.expression().clone()), scale)
        },
    )
}

fn batched_output_expr<F, C>(
    claims: &[JoltRelationClaims<F>],
    consistency: &BatchedCommittedSumcheckConsistency<F, C>,
) -> VerifierExpr<F>
where
    F: Field,
{
    claims.iter().zip(&consistency.batching_coefficients).fold(
        VerifierExpr::zero(),
        |acc, (claim, coefficient)| {
            acc + scale_expr(
                map_jolt_expr(claim.output.expression().clone()),
                *coefficient,
            )
        },
    )
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
            #[cfg(feature = "field-inline")]
            Stage8OpeningId::FieldInline(id) => VerifierOpeningId::FieldInline(id),
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

#[cfg(feature = "field-inline")]
fn map_field_inline_opening_ids(opening_ids: Vec<FieldInlineOpeningId>) -> Vec<VerifierOpeningId> {
    opening_ids
        .into_iter()
        .map(VerifierOpeningId::FieldInline)
        .collect()
}

#[cfg(feature = "field-inline")]
fn map_field_inline_expr<F: Field>(
    expr: Expr<F, FieldInlineOpeningId, FieldInlinePublicId, FieldInlineChallengeId>,
) -> VerifierExpr<F> {
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
                        Source::Opening(id) => Source::Opening(VerifierOpeningId::FieldInline(id)),
                        Source::Public(id) => Source::Public(VerifierPublicId::FieldInline(id)),
                        Source::Challenge(id) => {
                            Source::Challenge(VerifierChallengeId::FieldInline(id))
                        }
                    })
                    .collect(),
            })
            .collect(),
    }
}

#[cfg(feature = "field-inline")]
fn map_field_inline_bytecode_expr<F: Field>(
    expr: field_bytecode::FieldInlineBytecodeExpr<F>,
) -> VerifierExpr<F> {
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
                        Source::Opening(id) => Source::Opening(VerifierOpeningId::FieldInline(id)),
                        Source::Public(()) => unreachable!("field bytecode has no public sources"),
                        Source::Challenge(id) => Source::Challenge(VerifierChallengeId::Jolt(id)),
                    })
                    .collect(),
            })
            .collect(),
    }
}

fn map_jolt_expr<F: Field>(
    expr: Expr<F, JoltOpeningId, JoltPublicId, JoltChallengeId>,
) -> VerifierExpr<F> {
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
                        Source::Public(id) => Source::Public(id.into()),
                        Source::Challenge(id) => Source::Challenge(id.into()),
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

fn formula_dimensions<PCS, VC>(
    input: &BlindFoldInputs<'_, PCS, VC>,
) -> Result<JoltFormulaDimensions, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let log_t = input.checked.trace_length.ilog2() as usize;
    JoltFormulaDimensions::try_from(input.context.one_hot_config.dimensions(
        log_t,
        2 * RISCV_XLEN,
        input.preprocessing.program.bytecode.code_size,
        input.checked.ram_K,
    ))
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::BytecodeReadRaf,
        reason: error.to_string(),
    })
}

fn ram_output_publics<PCS, VC>(
    input: &BlindFoldInputs<'_, PCS, VC>,
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

fn ram_val_check_init<PCS, VC>(
    input: &BlindFoldInputs<'_, PCS, VC>,
) -> Result<ram::RamValCheckInit<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let r_address = ram_val_check_address(input)?;
    let mut advice_contributions = Vec::new();
    if input.context.untrusted_advice_commitment_present {
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

fn ram_val_check_address<PCS, VC>(
    input: &BlindFoldInputs<'_, PCS, VC>,
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
            stage: JoltRelationId::RamValCheck,
            reason: "RAM read-write opening point is shorter than the RAM address".to_string(),
        })
}

fn advice_selector<PCS, VC>(
    input: &BlindFoldInputs<'_, PCS, VC>,
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
        as usize;
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

fn advice_source_point<PCS, VC>(
    input: &BlindFoldInputs<'_, PCS, VC>,
    kind: JoltAdviceKind,
) -> Result<Vec<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let r_address = ram_val_check_address(input)?;
    advice_selector(input, kind, &r_address).map(|(_, point)| point)
}

fn advice_cycle_claim<PCS, VC>(
    input: &BlindFoldInputs<'_, PCS, VC>,
    kind: JoltAdviceKind,
) -> (
    Option<AdviceClaimReductionLayout>,
    Option<JoltRelationClaims<PCS::Field>>,
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

fn advice_address_claim<PCS, VC>(
    input: &BlindFoldInputs<'_, PCS, VC>,
    kind: JoltAdviceKind,
) -> (
    Option<AdviceClaimReductionLayout>,
    Option<JoltRelationClaims<PCS::Field>>,
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

fn advice_layout<PCS, VC>(
    input: &BlindFoldInputs<'_, PCS, VC>,
    kind: JoltAdviceKind,
) -> Option<AdviceClaimReductionLayout>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let present = match kind {
        JoltAdviceKind::Trusted => input.checked.trusted_advice_commitment_present,
        JoltAdviceKind::Untrusted => input.context.untrusted_advice_commitment_present,
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
            input.context.trace_polynomial_order,
            log_t,
            input.context.one_hot_config.committed_chunk_bits(),
            max_size,
        )
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 6 has several protocol components."
)]
fn add_stage6_publics_and_challenges<PCS, VC>(
    input: &BlindFoldInputs<'_, PCS, VC>,
    values: &mut SourceValues<PCS::Field>,
    bytecode_claims: &JoltRelationClaims<PCS::Field>,
    booleanity_claims: &JoltRelationClaims<PCS::Field>,
    ram_hamming_claims: &JoltRelationClaims<PCS::Field>,
    ram_ra_claims: &JoltRelationClaims<PCS::Field>,
    instruction_ra_claims: &JoltRelationClaims<PCS::Field>,
    inc_claims: &JoltRelationClaims<PCS::Field>,
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
        .map_err(|error| stage_sumcheck_error(JoltRelationId::BytecodeReadRaf, error))?;
    let bytecode_opening = formula_dimensions
        .bytecode_read_raf
        .opening_point(&bytecode_point)
        .map_err(|error| public_error(JoltRelationId::BytecodeReadRaf, error))?;
    let stage1_cycle = input.stage1.public.remainder_challenges[1..]
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let stage2_product_point = input
        .stage2
        .batch_consistency
        .try_instance_point(
            SpartanProductDimensions::new(log_t)
                .remainder_sumcheck()
                .rounds,
        )
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
        .try_instance_point(
            jolt_claims::protocols::jolt::TraceDimensions::new(log_t)
                .sumcheck(2)
                .rounds,
        )
        .map_err(|error| stage_sumcheck_error(JoltRelationId::SpartanShift, error))?;
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
            stage: JoltRelationId::BytecodeReadRaf,
            reason: "entry address was not found in bytecode preprocessing".to_string(),
        })?;
    #[cfg(feature = "field-inline")]
    let base_bytecode_rows = input
        .preprocessing
        .program
        .bytecode
        .bytecode
        .iter()
        .map(field_bytecode::base_jolt_bytecode_row)
        .collect::<Vec<_>>();
    #[cfg(feature = "field-inline")]
    let bytecode_rows = base_bytecode_rows.as_slice();
    #[cfg(not(feature = "field-inline"))]
    let bytecode_rows = input.preprocessing.program.bytecode.bytecode.as_slice();

    let bytecode_public_values =
        bytecode::read_raf_public_values::<PCS::Field>(BytecodeReadRafEvaluationInputs {
            bytecode: bytecode_rows,
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
        })
        .map_err(|error| public_error(JoltRelationId::BytecodeReadRaf, error))?;
    #[cfg(feature = "field-inline")]
    let bytecode_public_values = {
        let mut bytecode_public_values = bytecode_public_values;
        let field_inline_bytecode = input
            .preprocessing
            .field_inline_bytecode
            .as_deref()
            .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::BytecodeReadRaf,
                reason: "field-inline bytecode metadata is missing".to_string(),
            })?;
        let field_log_k = input.context.protocol.field_inline.field_register_log_k;
        let field_read_write_opening = &input.stage4.field_registers_read_write_opening_point;
        let field_val_evaluation_opening = &input
            .stage5
            .field_inline
            .field_registers_val_evaluation
            .opening_point;
        if field_read_write_opening.len() != field_log_k + log_t {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::BytecodeReadRaf,
                reason: format!(
                    "field-register read-write opening point length mismatch: expected {}, got {}",
                    field_log_k + log_t,
                    field_read_write_opening.len()
                ),
            });
        }
        if field_val_evaluation_opening.len() != field_log_k + log_t {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::BytecodeReadRaf,
                reason: format!(
                    "field-register val-evaluation opening point length mismatch: expected {}, got {}",
                    field_log_k + log_t,
                    field_val_evaluation_opening.len()
                ),
            });
        }
        let (field_read_write_address, field_read_write_cycle) =
            field_read_write_opening.split_at(field_log_k);
        let (field_val_evaluation_address, field_val_evaluation_cycle) =
            field_val_evaluation_opening.split_at(field_log_k);
        let field_values = field_bytecode::read_raf_public_values(
            field_bytecode::FieldInlineBytecodeReadRafEvaluationInputs {
                bytecode: field_inline_bytecode,
                field_register_log_k: field_log_k,
                r_address: &bytecode_opening.r_address,
                r_cycle: &bytecode_opening.r_cycle,
                stage1_cycle_point: &stage1_cycle,
                field_register_read_write_point: field_read_write_address,
                field_register_read_write_cycle_point: field_read_write_cycle,
                field_register_val_evaluation_point: field_val_evaluation_address,
                field_register_val_evaluation_cycle_point: field_val_evaluation_cycle,
                stage1_gammas: &input.stage6.public.stage1_gammas,
                stage4_gammas: &input.stage6.public.stage4_gammas,
                stage5_gammas: &input.stage6.public.stage5_gammas,
            },
        )
        .map_err(|error| public_error(JoltRelationId::BytecodeReadRaf, error))?;
        for (stage_value, field_value) in bytecode_public_values
            .stage_values
            .iter_mut()
            .zip(field_values.stage_values)
        {
            *stage_value += field_value;
        }
        bytecode_public_values
    };
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
        .map_err(|error| stage_sumcheck_error(JoltRelationId::Booleanity, error))?;
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
            .map_err(|error| public_error(JoltRelationId::Booleanity, error))?,
    )?;

    let ram_hamming_point = input
        .stage6
        .batch_consistency
        .try_instance_point(ram_hamming_claims.sumcheck.rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RamHammingBooleanity, error))?;
    let stage1_cycle_binding = &input.stage1.public.remainder_challenges[1..];
    values.challenge(
        JoltChallengeId::from(RamHammingBooleanityChallenge::EqCycle),
        try_eq_mle(&ram_hamming_point, stage1_cycle_binding)
            .map_err(|error| public_error(JoltRelationId::RamHammingBooleanity, error))?,
    )?;

    let ram_ra_point = input
        .stage6
        .batch_consistency
        .try_instance_point(ram_ra_claims.sumcheck.rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RamRaVirtualization, error))?;
    let ram_ra_cycle = trace_dimensions
        .cycle_opening_point(&ram_ra_point)
        .map_err(|error| public_error(JoltRelationId::RamRaVirtualization, error))?;
    let ram_reduced_cycle = &input.stage5.ram_ra_claim_reduction.opening_point[log_k..];
    values.challenge(
        JoltChallengeId::from(RamRaVirtualizationChallenge::EqCycle),
        try_eq_mle(ram_reduced_cycle, &ram_ra_cycle)
            .map_err(|error| public_error(JoltRelationId::RamRaVirtualization, error))?,
    )?;

    let instruction_ra_point = input
        .stage6
        .batch_consistency
        .try_instance_point(instruction_ra_claims.sumcheck.rounds)
        .map_err(|error| {
            stage_sumcheck_error(JoltRelationId::InstructionRaVirtualization, error)
        })?;
    let instruction_ra_cycle = trace_dimensions
        .cycle_opening_point(&instruction_ra_point)
        .map_err(|error| public_error(JoltRelationId::InstructionRaVirtualization, error))?;
    values.challenge(
        JoltChallengeId::from(InstructionRaVirtualizationChallenge::EqCycle),
        try_eq_mle(
            &input.stage5.instruction_read_raf.r_cycle,
            &instruction_ra_cycle,
        )
        .map_err(|error| public_error(JoltRelationId::InstructionRaVirtualization, error))?,
    )?;

    let inc_point = input
        .stage6
        .batch_consistency
        .try_instance_point(inc_claims.sumcheck.rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::IncClaimReduction, error))?;
    let inc_opening_point = trace_dimensions
        .cycle_opening_point(&inc_point)
        .map_err(|error| public_error(JoltRelationId::IncClaimReduction, error))?;
    values.public(
        JoltPublicId::from(IncClaimReductionPublic::EqRamReadWrite),
        try_eq_mle(
            &inc_opening_point,
            &input
                .stage2
                .ram_val_check_inputs
                .ram_read_write_opening_point[log_k..],
        )
        .map_err(|error| public_error(JoltRelationId::IncClaimReduction, error))?,
    )?;
    values.public(
        JoltPublicId::from(IncClaimReductionPublic::EqRamValCheck),
        try_eq_mle(
            &inc_opening_point,
            &input.stage4.ram_val_check_opening_point[log_k..],
        )
        .map_err(|error| public_error(JoltRelationId::IncClaimReduction, error))?,
    )?;
    values.public(
        JoltPublicId::from(IncClaimReductionPublic::EqRegistersReadWrite),
        try_eq_mle(
            &inc_opening_point,
            &input.stage4.registers_read_write_opening_point[REGISTER_ADDRESS_BITS..],
        )
        .map_err(|error| public_error(JoltRelationId::IncClaimReduction, error))?,
    )?;
    values.public(
        JoltPublicId::from(IncClaimReductionPublic::EqRegistersValEvaluation),
        try_eq_mle(
            &inc_opening_point,
            &input.stage5.registers_val_evaluation.opening_point[REGISTER_ADDRESS_BITS..],
        )
        .map_err(|error| public_error(JoltRelationId::IncClaimReduction, error))?,
    )?;

    Ok(())
}

fn add_advice_cycle_publics<PCS, VC>(
    input: &BlindFoldInputs<'_, PCS, VC>,
    values: &mut SourceValues<PCS::Field>,
    layout: &AdviceClaimReductionLayout,
    kind: JoltAdviceKind,
    public: &crate::stages::stage6::outputs::AdviceCyclePhasePublicOutput<PCS::Field>,
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    if layout.dimensions().has_address_phase() {
        return Ok(());
    }

    let source_point = advice_source_point(input, kind)?;
    let scale = layout
        .cycle_phase_final_output_scale(&source_point, &public.sumcheck_point)
        .map_err(|error| public_error(JoltRelationId::AdviceClaimReductionCyclePhase, error))?;
    values.public(
        JoltPublicId::from(AdviceClaimReductionPublic::FinalScale(kind)),
        scale,
    )
}

fn add_advice_address_publics<PCS, VC>(
    input: &BlindFoldInputs<'_, PCS, VC>,
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
        .map_err(|error| public_error(JoltRelationId::AdviceClaimReduction, error))?;
    values.public(
        JoltPublicId::from(AdviceClaimReductionPublic::FinalScale(kind)),
        scale,
    )
}

fn stage6_virtualization_points<PCS, VC>(
    input: &BlindFoldInputs<'_, PCS, VC>,
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

    fn challenge(
        &mut self,
        id: impl Into<VerifierChallengeId>,
        value: F,
    ) -> Result<(), VerifierError> {
        push_unique(&mut self.challenges, id.into(), value, "challenge")
    }

    fn has_public(&self, id: VerifierPublicId) -> bool {
        self.publics.iter().any(|(candidate, _)| *candidate == id)
    }

    fn has_challenge(&self, id: VerifierChallengeId) -> bool {
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
