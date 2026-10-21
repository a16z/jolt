//! The residual relation of Jolt's stages over a masked transcript.
//!
//! The stage verifiers record each masked message with its pad; this module
//! states every check they would have run on the hidden values as rows over
//! the pads (`specs/whiteout-masking.md`):
//!
//! - each stage's sumcheck rounds reduce its batched input claim to a running
//!   claim, an affine form in the pads ([`sumcheck_rows`]; uni-skip rounds
//!   add their explicit round-sum row);
//! - one row equates that running claim with the batched output claim;
//! - claim formulas come from `jolt-claims` in factored form and lower through
//!   [`lower_factored`], one auxiliary slot per intermediate product;
//! - runtime claim aliases, which a masked proof sends, become equality rows.
//!
//! Each stage file computes the stage's public values (eq/lt evaluations,
//! challenges) and registers its members' factored formulas with
//! [`ResidualBuilder`]. The claim lowering is one routine run in three value
//! domains: [`OperandShapeDomain`] counts auxiliary slots and records the
//! product operands for the pad layout,
//! [`ValuesDomain`] computes the prover's auxiliary values, and
//! [`RowsDomain`] builds the rows. Stage 8's row, which ties the PCS
//! evaluation to the masked claims, belongs to the back end's PCS integration
//! and is built from [`JoltResidual::opening_form`].

use jolt_algebra::JoltField;
use jolt_claims::protocols::composed::geometry::{
    SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
};
use jolt_claims::protocols::composed::r1cs::{
    JoltSpartanOuterPublic, JoltSpartanOuterRemainder, JoltSpartanOuterRemainderChallenges,
    SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE,
};
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::{FieldInlineChallengeId, FieldInlineDerivedId};
#[cfg(not(feature = "field-inline"))]
use jolt_claims::protocols::jolt::geometry::bytecode::BytecodeReadRafCommittedEvaluationInputs;
use jolt_claims::protocols::jolt::relations;
#[cfg(feature = "fuzzing")]
use jolt_claims::protocols::jolt::{JoltExpr, JoltOpeningId};
use jolt_claims::SumcheckDomain;
use jolt_claims::{
    protocols::jolt::{
        geometry::{
            booleanity::BooleanityDimensions,
            bytecode::{self, BytecodeReadRafEvaluationInputs},
            claim_reductions::{
                advice,
                bytecode::{self as bytecode_reduction},
                hamming_weight, program_image,
            },
            dimensions::{JoltFormulaDimensions, REGISTER_ADDRESS_BITS},
            ram,
            spartan::{SpartanOuterDimensions, SpartanProductDimensions},
        },
        AdviceClaimReductionLayout, AdviceClaimReductionPublic, BooleanityChallenge,
        BooleanityPublic, BytecodeClaimReductionChallenge, BytecodeClaimReductionLayout,
        BytecodeClaimReductionPublic, BytecodeReadRafChallenge, BytecodeReadRafPublic,
        HammingWeightClaimReductionChallenge, HammingWeightClaimReductionPublic,
        IncClaimReductionChallenge, IncClaimReductionPublic, InstructionClaimReductionChallenge,
        InstructionClaimReductionPublic, InstructionInputChallenge, InstructionInputPublic,
        InstructionRaVirtualizationChallenge, InstructionRaVirtualizationPublic,
        InstructionReadRafChallenge, InstructionReadRafPublic, JoltAdviceKind, JoltChallengeId,
        JoltDerivedId, JoltRelationId, PrecommittedReductionLayout,
        ProgramImageClaimReductionLayout, ProgramImageClaimReductionPublic,
        RamHammingBooleanityPublic, RamOutputCheckPublic, RamRaClaimReductionChallenge,
        RamRaClaimReductionPublic, RamRaVirtualizationPublic, RamRafEvaluationPublic,
        RamReadWriteChallenge, RamReadWritePublic, RamValCheckChallenge, RamValCheckPublic,
        RegistersClaimReductionChallenge, RegistersClaimReductionPublic,
        RegistersReadWriteChallenge, RegistersReadWritePublic, RegistersValEvaluationPublic,
        SpartanOuterPublic, SpartanShiftChallenge, SpartanShiftPublic,
    },
    FactoredExpr, Source, SymbolicSumcheck,
};
#[cfg(any(feature = "fuzzing", all(test, feature = "field-inline")))]
use jolt_claims::{Expr, Term};
use jolt_crypto::VectorCommitment;
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
use jolt_openings::CommitmentScheme;
use jolt_poly::{
    block_selector_mle_msb,
    lagrange::{centered_lagrange_evals, centered_lagrange_kernel},
    try_eq_mle, EqPlusOnePolynomial, IdentityPolynomial, LtPolynomial, MultilinearEvaluation,
    OperandPolynomial, OperandSide,
};
use jolt_program::preprocess::PublicIoMemory;
use jolt_sumcheck::{BatchedMaskedRounds, DegreeSchedule, MaskedRound, SumcheckDomainSpec};
use jolt_zk::masking::{
    lower_factored, sumcheck_rows, AffineRow, ClaimDomain, MaskedOperand, OperandShapeDomain,
    PadLayout, ResidualRelation, ResidualRows, RoundFormat, RowsDomain, ValuesDomain, ZkError,
};
use std::collections::BTreeMap;

use crate::preprocessing::JoltVerifierPreprocessing;
use crate::stages::relations::MaskedClaims;
use crate::stages::{
    stage1::Stage1ZkOutput, stage2::Stage2ZkOutput, stage3::Stage3ZkOutput, stage4::Stage4ZkOutput,
    stage5::Stage5ZkOutput, stage6a::Stage6aZkOutput, stage6b::Stage6bZkOutput,
    stage7::Stage7ZkOutput,
};
use crate::stages::{
    stage2::ram_output_check::ram_output_check_publics,
    stage6b::{committed_reduction_cycle_phase, outputs::BytecodeReductionWeights},
};
use crate::verifier::CheckedInputs;
use crate::VerifierError;

#[cfg(feature = "field-inline")]
mod field_inline;
mod stage1;
mod stage2;
mod stage3;
mod stage4;
mod stage5;
mod stage6a;
mod stage6b;
mod stage7;

/// The lowering's opening-id type: the composite [`ComposedOpeningId`], so hidden values
/// from either protocol family (jolt, field-inline) live in one claim-source namespace.
use jolt_claims::protocols::composed::ComposedOpeningId;

#[cfg(any(feature = "fuzzing", all(test, feature = "field-inline")))]
type VerifierExpr<F> = Expr<F, ComposedOpeningId, VerifierPublicId>;
type VerifierFactored<F> = FactoredExpr<F, ComposedOpeningId, VerifierPublicId>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VerifierPublicId {
    Jolt(JoltDerivedId),
    SpartanOuter(JoltSpartanOuterPublic),
    /// Gamma values that remain as `JoltChallengeId` variants (not moved to Public) but are
    /// treated as public inputs of the residual relation.
    Challenge(JoltChallengeId),
    #[cfg(feature = "field-inline")]
    FieldInline(FieldInlineDerivedId),
    /// Field-inline challenges, treated as public inputs exactly like the
    /// jolt `Challenge` arm.
    #[cfg(feature = "field-inline")]
    FieldInlineChallenge(FieldInlineChallengeId),
}

impl From<JoltDerivedId> for VerifierPublicId {
    fn from(id: JoltDerivedId) -> Self {
        if let JoltDerivedId::SpartanOuter(public) = id {
            return Self::SpartanOuter(match public {
                SpartanOuterPublic::TauKernel => JoltSpartanOuterPublic::TauKernel,
                SpartanOuterPublic::AzWeight(i) => JoltSpartanOuterPublic::AzWeight(i),
                SpartanOuterPublic::BzWeight(i) => JoltSpartanOuterPublic::BzWeight(i),
                SpartanOuterPublic::AzConstant => JoltSpartanOuterPublic::AzConstant,
                SpartanOuterPublic::BzConstant => JoltSpartanOuterPublic::BzConstant,
            });
        }
        Self::Jolt(id)
    }
}

impl From<JoltChallengeId> for VerifierPublicId {
    fn from(id: JoltChallengeId) -> Self {
        Self::Challenge(id)
    }
}

#[cfg(feature = "field-inline")]
impl From<FieldInlineDerivedId> for VerifierPublicId {
    fn from(id: FieldInlineDerivedId) -> Self {
        Self::FieldInline(id)
    }
}

#[cfg(feature = "field-inline")]
impl From<FieldInlineChallengeId> for VerifierPublicId {
    fn from(id: FieldInlineChallengeId) -> Self {
        Self::FieldInlineChallenge(id)
    }
}

#[derive(Default)]
struct SourceValues<F: JoltField> {
    publics: Vec<(VerifierPublicId, F)>,
}

/// The masked stage outputs the residual relation is built from.
pub struct ResidualInputs<'a, PCS, VC>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    pub checked: &'a CheckedInputs,
    pub preprocessing: &'a JoltVerifierPreprocessing<PCS, VC>,
    pub stage1: &'a Stage1ZkOutput<PCS::Field>,
    pub stage2: &'a Stage2ZkOutput<PCS::Field>,
    pub stage3: &'a Stage3ZkOutput<PCS::Field>,
    pub stage4: &'a Stage4ZkOutput<PCS::Field>,
    pub stage5: &'a Stage5ZkOutput<PCS::Field>,
    pub stage6a: &'a Stage6aZkOutput<PCS::Field>,
    pub stage6b: &'a Stage6bZkOutput<PCS::Field>,
    pub stage7: &'a Stage7ZkOutput<PCS::Field>,
}

/// One batch member: its round count and its factored input and output
/// formulas, with its batching coefficient.
struct Member<F: JoltField> {
    rounds: usize,
    input: VerifierFactored<F>,
    output: VerifierFactored<F>,
    coefficient: F,
}

/// One masked sumcheck of the residual relation.
struct Stage<F: JoltField> {
    name: &'static str,
    format: RoundFormat,
    degrees: DegreeSchedule,
    rounds: Vec<MaskedRound<F>>,
    members: Vec<Member<F>>,
}

/// The stages a residual relation is built from, accumulated by the stage
/// files in protocol order, with every masked claim they received.
#[derive(Default)]
struct ResidualBuilder<F: JoltField> {
    stages: Vec<Stage<F>>,
    openings: BTreeMap<ComposedOpeningId, MaskedOperand<F>>,
    aliases: BTreeMap<ComposedOpeningId, ComposedOpeningId>,
    equalities: Vec<(ComposedOpeningId, ComposedOpeningId)>,
}

impl<F: JoltField> ResidualBuilder<F> {
    fn record_claims(&mut self, claims: &MaskedClaims<F>) -> Result<(), VerifierError> {
        for (id, value) in &claims.received {
            if self.openings.insert(*id, *value).is_some() {
                return Err(residual_error(format!("opening {id:?} was received twice")));
            }
        }
        for (alias, source) in &claims.aliases {
            if self.aliases.insert(*alias, *source).is_some() {
                return Err(residual_error(format!(
                    "opening {alias:?} is aliased twice"
                )));
            }
        }
        self.equalities.extend_from_slice(&claims.equalities);
        Ok(())
    }
}

/// The residual relation of Jolt's stages 1–7 over a masked transcript.
pub struct JoltResidual<F: JoltField> {
    stages: Vec<Stage<F>>,
    openings: BTreeMap<ComposedOpeningId, MaskedOperand<F>>,
    aliases: BTreeMap<ComposedOpeningId, ComposedOpeningId>,
    equalities: Vec<(ComposedOpeningId, ComposedOpeningId)>,
    publics: SourceValues<F>,
}

impl<F: JoltField> JoltResidual<F> {
    /// Collects the stages' residual structure from the masked stage outputs.
    ///
    /// # Errors
    ///
    /// [`VerifierError`] if a stage's public values cannot be derived or its
    /// masked records are malformed.
    pub fn build<PCS, VC>(input: ResidualInputs<'_, PCS, VC>) -> Result<Self, VerifierError>
    where
        PCS: CommitmentScheme<Field = F>,
        VC: VectorCommitment<Field = F>,
    {
        let mut values = SourceValues::default();
        let mut builder = ResidualBuilder::default();

        builder = stage1::add_stage1(&input, builder, &mut values)?;
        builder = stage2::add_stage2(&input, builder, &mut values)?;
        builder = stage3::add_stage3(&input, builder, &mut values)?;
        builder = stage4::add_stage4(&input, builder, &mut values)?;
        builder = stage5::add_stage5(&input, builder, &mut values)?;
        builder = stage6a::add_stage6a(&input, builder, &mut values)?;
        builder = stage6b::add_stage6b(&input, builder, &mut values)?;
        builder = stage7::add_stage7(&input, builder, &mut values)?;

        Ok(Self {
            stages: builder.stages,
            openings: builder.openings,
            aliases: builder.aliases,
            equalities: builder.equalities,
            publics: values,
        })
    }

    /// The masked value a proof received for `id`, through its alias.
    ///
    /// # Errors
    ///
    /// [`VerifierError::MissingOpeningClaim`] if no stage received it.
    pub fn opening(&self, id: &ComposedOpeningId) -> Result<MaskedOperand<F>, VerifierError> {
        let source = self.aliases.get(id).unwrap_or(id);
        self.openings
            .get(source)
            .copied()
            .ok_or(VerifierError::MissingOpeningClaim { id: *source })
    }

    /// The decoded value of `id`, m̂ − p, as an affine form over the pads.
    ///
    /// # Errors
    ///
    /// As [`opening`](Self::opening).
    pub fn opening_form(&self, id: &ComposedOpeningId) -> Result<AffineRow<F>, VerifierError> {
        Ok(self.opening(id)?.form())
    }

    /// The pad layout of a schedule with `base` message slots: the
    /// auxiliary slots the relation's products take, and their operand pads.
    ///
    /// # Errors
    ///
    /// [`VerifierError`] if a formula does not lower.
    pub fn pad_layout(&self, base: usize) -> Result<PadLayout, VerifierError> {
        let mut shape = OperandShapeDomain::new(base);
        let _ = self.lower_claims(&mut shape, &mut |id| {
            Ok(Some(
                self.opening(id)
                    .map_err(|_| ZkError::InvalidRelation {
                        reason: "a claim formula names an opening no stage received",
                    })?
                    .pad,
            ))
        })?;
        Ok(shape.layout()?)
    }

    /// The prover's auxiliary values, in slot order, from the decoded
    /// openings.
    ///
    /// # Errors
    ///
    /// [`VerifierError`] if a formula does not lower or `decode` fails.
    pub fn auxiliary_values(
        &self,
        decode: impl Fn(MaskedOperand<F>) -> Result<F, ZkError>,
    ) -> Result<Vec<F>, VerifierError> {
        let mut values = ValuesDomain::default();
        let _ = self.lower_claims(&mut values, &mut |id| {
            decode(self.opening(id).map_err(|_| ZkError::InvalidRelation {
                reason: "a claim formula names an opening no stage received",
            })?)
        })?;
        Ok(values.auxiliary)
    }

    /// The rows over `masked_auxiliary`, the received auxiliary block.
    pub fn rows<'r>(&'r self, masked_auxiliary: &'r [F]) -> JoltResidualRows<'r, F> {
        JoltResidualRows {
            residual: self,
            masked_auxiliary,
        }
    }

    /// Each stage's batched input and output claims, in one fixed order
    /// (stage by stage; inputs, then outputs, member by member) so every
    /// domain allocates the same auxiliary slots.
    fn lower_claims<D: ClaimDomain<F>>(
        &self,
        domain: &mut D,
        opening: &mut impl FnMut(&ComposedOpeningId) -> Result<D::Value, ZkError>,
    ) -> Result<Vec<StageClaims<D::Value>>, VerifierError> {
        let mut public = |source: &Source<ComposedOpeningId, VerifierPublicId, usize>| {
            match source {
                Source::Derived(id) => self.publics.value(*id),
                Source::Opening(_) | Source::Challenge(_) => None,
            }
            .ok_or(ZkError::InvalidRelation {
                reason: "a claim formula names a public value no stage supplied",
            })
        };
        let mut claims = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            let mut input = domain.constant(F::zero());
            let max_num_vars = stage.degrees.num_rounds();
            for member in &stage.members {
                let extra = max_num_vars.checked_sub(member.rounds).ok_or_else(|| {
                    residual_error(format!(
                        "{}: member rounds {} exceed the batch's {max_num_vars}",
                        stage.name, member.rounds
                    ))
                })?;
                let value = lower_factored(&member.input, domain, opening, &mut public)?;
                input = domain.add(
                    &input,
                    &domain.scale(&value, member.coefficient * F::pow2(extra)),
                );
            }
            let mut output = domain.constant(F::zero());
            for member in &stage.members {
                let value = lower_factored(&member.output, domain, opening, &mut public)?;
                output = domain.add(&output, &domain.scale(&value, member.coefficient));
            }
            claims.push(StageClaims { input, output });
        }
        Ok(claims)
    }
}

/// One stage's batched input and output claims in a lowering domain.
struct StageClaims<V> {
    input: V,
    output: V,
}

/// [`JoltResidual`]'s rows over a received auxiliary block.
pub struct JoltResidualRows<'r, F: JoltField> {
    residual: &'r JoltResidual<F>,
    masked_auxiliary: &'r [F],
}

impl<F: JoltField> ResidualRows<F> for JoltResidualRows<'_, F> {
    type Error = VerifierError;

    fn append_rows(&self, relation: &mut ResidualRelation<F>) -> Result<(), VerifierError> {
        let residual = self.residual;
        let mut rows = RowsDomain::new(relation, self.masked_auxiliary);
        let claims = residual.lower_claims(&mut rows, &mut |id| {
            residual
                .opening_form(id)
                .map_err(|_| ZkError::InvalidRelation {
                    reason: "a claim formula names an opening no stage received",
                })
        })?;
        rows.finish()?;
        let relation = rows.relation();
        for (stage, StageClaims { input, output }) in residual.stages.iter().zip(claims) {
            let running =
                sumcheck_rows(relation, input, &stage.rounds, &stage.degrees, stage.format)?;
            relation.affine.push(running.add_scaled(&output, -F::one()));
        }
        for (cell, source) in &residual.equalities {
            relation.affine.push(
                residual
                    .opening_form(cell)?
                    .add_scaled(&residual.opening_form(source)?, -F::one()),
            );
        }
        Ok(())
    }
}

/// Registers a batched sumcheck: the members' factored claims (declaration
/// order, one per batching coefficient), its masked rounds and its masked
/// output claims.
fn add_batched_stage<F: JoltField>(
    mut builder: ResidualBuilder<F>,
    name: &'static str,
    batch_domain: SumcheckDomain,
    claims: &[MemberClaim<F>],
    rounds: &BatchedMaskedRounds<F>,
    output_claims: &MaskedClaims<F>,
    values: &SourceValues<F>,
) -> Result<ResidualBuilder<F>, VerifierError> {
    if claims.is_empty() {
        return Err(residual_error(format!("{name}: empty batched claims")));
    }
    if claims.len() != rounds.batching_coefficients.len() {
        return Err(residual_error(format!(
            "{name}: expected {} batching coefficients, got {}",
            claims.len(),
            rounds.batching_coefficients.len()
        )));
    }
    let members = claims
        .iter()
        .zip(&rounds.batching_coefficients)
        .map(|(claim, &coefficient)| {
            require_expr_sources(name, "input claim", &claim.input, values)?;
            require_expr_sources(name, "output claim", &claim.output, values)?;
            Ok(Member {
                rounds: claim.rounds,
                input: claim.input.clone(),
                output: claim.output.clone(),
                coefficient,
            })
        })
        .collect::<Result<Vec<_>, VerifierError>>()?;
    builder.record_claims(output_claims)?;
    builder.stages.push(Stage {
        name,
        format: round_format(batch_domain, false),
        degrees: rounds.degrees.clone(),
        rounds: rounds.rounds.clone(),
        members,
    });
    Ok(builder)
}

/// A uni-skip first round as the masked verifier received it.
struct UniskipRound<'a, F: JoltField> {
    degree: usize,
    domain: SumcheckDomain,
    rounds: &'a [MaskedRound<F>],
    output_claims: &'a MaskedClaims<F>,
}

/// Registers a uni-skip first round: one member over the centered domain,
/// every coefficient sent.
fn add_uniskip_stage<F: JoltField>(
    mut builder: ResidualBuilder<F>,
    name: &'static str,
    round: UniskipRound<'_, F>,
    values: &SourceValues<F>,
    claim: MemberClaim<F>,
) -> Result<ResidualBuilder<F>, VerifierError> {
    let UniskipRound {
        degree,
        domain,
        rounds,
        output_claims,
    } = round;
    require_expr_sources(name, "input claim", &claim.input, values)?;
    require_expr_sources(name, "output claim", &claim.output, values)?;
    builder.record_claims(output_claims)?;
    builder.stages.push(Stage {
        name,
        format: round_format(domain, true),
        degrees: DegreeSchedule::uniform(claim.rounds, degree),
        rounds: rounds.to_vec(),
        members: vec![Member {
            rounds: claim.rounds,
            input: claim.input,
            output: claim.output,
            coefficient: F::one(),
        }],
    });
    Ok(builder)
}

/// The round wire of a stage: uni-skip rounds send every coefficient, batched
/// rounds the compressed Boolean-hypercube form.
fn round_format(domain: SumcheckDomain, full: bool) -> RoundFormat {
    if full {
        RoundFormat::Full(domain_spec(domain))
    } else {
        RoundFormat::Compressed
    }
}

/// One relation's `(rounds, input, output)` in factored form.
struct MemberClaim<F: JoltField> {
    rounds: usize,
    input: VerifierFactored<F>,
    output: VerifierFactored<F>,
}

/// Lower one symbolic relation into its member claim. Generic over the
/// relation's id family: jolt and field-inline relations both map into the
/// composite-id [`VerifierFactored`], so the hidden claim algebra is
/// single-sourced from each family's symbolic formulas.
fn relation_claim<F, S>(relation: &S) -> MemberClaim<F>
where
    F: JoltField,
    S: SymbolicSumcheck,
    S::OpeningId: Into<ComposedOpeningId>,
    S::DerivedId: Into<VerifierPublicId>,
    S::ChallengeId: Into<VerifierPublicId>,
{
    MemberClaim {
        rounds: relation.rounds(),
        input: relation.input_factored::<F>().map_sources(map_source),
        output: relation.output_factored::<F>().map_sources(map_source),
    }
}

/// Map a protocol-family source into the composite ids: openings become
/// hidden values named by the composite id; derived values and challenges
/// both become publics.
fn map_source<O, P, C>(
    source: Source<O, P, C>,
) -> Source<ComposedOpeningId, VerifierPublicId, usize>
where
    O: Into<ComposedOpeningId>,
    P: Into<VerifierPublicId>,
    C: Into<VerifierPublicId>,
{
    match source {
        Source::Opening(id) => Source::Opening(id.into()),
        Source::Derived(id) => Source::Derived(id.into()),
        Source::Challenge(id) => Source::Derived(id.into()),
    }
}

/// Map a protocol-family expression into the composite-id [`VerifierExpr`].
#[cfg(any(feature = "fuzzing", all(test, feature = "field-inline")))]
fn map_expr<F, O, P, C>(expr: Expr<F, O, P, C>) -> VerifierExpr<F>
where
    F: JoltField,
    O: Into<ComposedOpeningId>,
    P: Into<VerifierPublicId>,
    C: Into<VerifierPublicId>,
{
    Expr {
        terms: expr
            .terms
            .into_iter()
            .map(|term| Term {
                coefficient: term.coefficient,
                factors: term.factors.into_iter().map(map_source).collect(),
            })
            .collect(),
    }
}

/// Evaluates the residual form of a Jolt claim expression with the same
/// source values used by the clear verifier.
///
/// This is exposed only for differential fuzzing of the expression boundary.
#[cfg(feature = "fuzzing")]
#[expect(
    clippy::unreachable,
    reason = "mapping a Jolt expression cannot produce field-inline IDs or retain challenges"
)]
pub fn evaluate_mapped_expression<F, Opening, Challenge, Derived>(
    expr: JoltExpr<F>,
    mut opening: Opening,
    mut challenge: Challenge,
    mut derived: Derived,
) -> F
where
    F: JoltField,
    Opening: FnMut(&JoltOpeningId) -> F,
    Challenge: FnMut(&JoltChallengeId) -> F,
    Derived: FnMut(&JoltDerivedId) -> F,
{
    map_expr(expr).evaluate(
        |id| match id {
            ComposedOpeningId::Jolt(id) => opening(id),
            ComposedOpeningId::FieldInline(_) => {
                unreachable!("Jolt expressions do not contain field-inline openings")
            }
        },
        |_| unreachable!("Jolt challenges map to residual public inputs"),
        |id| match id {
            VerifierPublicId::Jolt(id) => derived(id),
            VerifierPublicId::Challenge(id) => challenge(id),
            VerifierPublicId::SpartanOuter(id) => derived(&JoltDerivedId::SpartanOuter(match id {
                JoltSpartanOuterPublic::TauKernel => SpartanOuterPublic::TauKernel,
                JoltSpartanOuterPublic::AzWeight(i) => SpartanOuterPublic::AzWeight(*i),
                JoltSpartanOuterPublic::BzWeight(i) => SpartanOuterPublic::BzWeight(*i),
                JoltSpartanOuterPublic::AzConstant => SpartanOuterPublic::AzConstant,
                JoltSpartanOuterPublic::BzConstant => SpartanOuterPublic::BzConstant,
            })),
            #[cfg(feature = "field-inline")]
            VerifierPublicId::FieldInline(_) | VerifierPublicId::FieldInlineChallenge(_) => {
                unreachable!("Jolt expressions do not contain field-inline publics")
            }
        },
    )
}

fn require_expr_sources<F: JoltField>(
    stage: &'static str,
    expression: &'static str,
    expr: &VerifierFactored<F>,
    values: &SourceValues<F>,
) -> Result<(), VerifierError> {
    let sources = expr
        .terms
        .iter()
        .flat_map(|term| &term.factors)
        .flat_map(|factor| &factor.terms)
        .flat_map(|term| &term.factors);
    for source in sources {
        if let Source::Derived(id) = source {
            if values.value(*id).is_none() {
                return Err(residual_error(format!(
                    "{stage} {expression} is missing public source {id:?}"
                )));
            }
        }
    }
    Ok(())
}

fn domain_spec(domain: SumcheckDomain) -> SumcheckDomainSpec {
    match domain {
        SumcheckDomain::BooleanHypercube => SumcheckDomainSpec::BooleanHypercube,
        SumcheckDomain::CenteredInteger { domain_size } => {
            SumcheckDomainSpec::CenteredInteger { domain_size }
        }
    }
}

fn formula_dimensions<PCS, VC>(
    input: &ResidualInputs<'_, PCS, VC>,
) -> Result<JoltFormulaDimensions, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    crate::stages::build_formula_dimensions(
        input.preprocessing,
        input.checked,
        crate::num::ilog2(input.checked.trace_length),
        JoltRelationId::BytecodeReadRaf,
    )
}

#[expect(
    clippy::type_complexity,
    reason = "the three RAM output-check publics (eq, mask, val_io)"
)]
fn ram_output_publics<PCS, VC>(
    input: &ResidualInputs<'_, PCS, VC>,
    output_address_challenges: &[PCS::Field],
    ram_output_address: &[PCS::Field],
) -> Result<(PCS::Field, PCS::Field, PCS::Field), VerifierError>
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
    ram_output_check_publics(
        &public_memory,
        output_address_challenges,
        ram_output_address,
    )
}

// Binding the scalar field to a bare `F` parameter (rather than spelling
// `PCS::Field`) lets clippy.toml's `arithmetic-side-effects-allowed = ["F"]`
// recognize the side-effect-free field negations in the body.
fn ram_val_check_init<F, PCS, VC>(
    input: &ResidualInputs<'_, PCS, VC>,
) -> Result<ram::RamValCheckInit<F>, VerifierError>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
{
    let r_address = ram_val_check_address(input)?;
    // WARNING: contribution order and selectors must stay in lockstep with the
    // clear-path decomposition in stage4/verify.rs.
    let mut contributions = Vec::new();
    if input.checked.precommitted.program_image.is_some() {
        contributions.push(ram::RamValCheckInitContribution::program_image(-F::one()));
    }
    if input.checked.untrusted_advice_commitment_present {
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

fn ram_val_check_address<PCS, VC>(
    input: &ResidualInputs<'_, PCS, VC>,
) -> Result<Vec<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let log_k = crate::num::ilog2(input.checked.ram_K);
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

fn advice_selector<PCS, VC>(
    input: &ResidualInputs<'_, PCS, VC>,
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
    let start_index = u128::from(
        layout
            .remapped_word_address(start_address)
            .map_err(|error| public_error(JoltRelationId::RamValCheck, error))?,
    );
    let max_size =
        usize::try_from(max_size).map_err(|_| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamValCheck,
            reason: format!("{kind:?} advice size {max_size} exceeds usize"),
        })?;
    // Floor division mirrors the shared advice geometry in jolt-claims
    // (`geometry/dimensions.rs`): the advice block holds `max_size / 8` words.
    #[expect(
        clippy::integer_division,
        reason = "floor division bytes -> words matches the prover-shared advice geometry"
    )]
    let advice_num_vars = crate::num::ilog2((max_size / 8).next_power_of_two());
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
    input: &ResidualInputs<'_, PCS, VC>,
    kind: JoltAdviceKind,
) -> Result<Vec<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let r_address = ram_val_check_address(input)?;
    advice_selector(input, kind, &r_address).map(|(_, point)| point)
}

fn advice_layout<PCS, VC>(
    input: &ResidualInputs<'_, PCS, VC>,
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
fn add_stage6_publics_and_challenges<PCS, VC>(
    input: &ResidualInputs<'_, PCS, VC>,
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
    let log_t = crate::num::ilog2(input.checked.trace_length);
    let log_k = crate::num::ilog2(input.checked.ram_K);
    let trace_dimensions = jolt_claims::protocols::jolt::TraceDimensions::new(log_t);

    let bytecode_challenges = &input.stage6a.challenges.bytecode_read_raf;
    let challenge_publics: [(JoltChallengeId, PCS::Field); 9] = [
        (
            BytecodeReadRafChallenge::Gamma.into(),
            bytecode_challenges.gamma,
        ),
        (
            BytecodeReadRafChallenge::Stage1Gamma.into(),
            bytecode_challenges.stage1_gamma,
        ),
        (
            BytecodeReadRafChallenge::Stage2Gamma.into(),
            bytecode_challenges.stage2_gamma,
        ),
        (
            BytecodeReadRafChallenge::Stage3Gamma.into(),
            bytecode_challenges.stage3_gamma,
        ),
        (
            BytecodeReadRafChallenge::Stage4Gamma.into(),
            bytecode_challenges.stage4_gamma,
        ),
        (
            BytecodeReadRafChallenge::Stage5Gamma.into(),
            bytecode_challenges.stage5_gamma,
        ),
        (
            BooleanityChallenge::Gamma.into(),
            input.stage6a.challenges.booleanity.gamma,
        ),
        (
            InstructionRaVirtualizationChallenge::Gamma.into(),
            input.stage6b.challenges.instruction_ra_gamma,
        ),
        (
            IncClaimReductionChallenge::Gamma.into(),
            input.stage6b.challenges.inc_gamma,
        ),
    ];
    for (id, value) in challenge_publics {
        values.public(VerifierPublicId::Challenge(id), value)?;
    }

    let bytecode_address_point = input
        .stage6a
        .rounds
        .try_instance_point(bytecode_address_rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::BytecodeReadRaf, error))?;
    let bytecode_r_address = bytecode_address_point
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let bytecode_point = input
        .stage6b
        .batch_rounds
        .try_instance_point(bytecode_rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::BytecodeReadRaf, error))?;
    let bytecode_r_cycle = bytecode_point.iter().rev().copied().collect::<Vec<_>>();
    let stage1_remainder_challenges = input.stage1.remainder_rounds.challenges();
    let stage1_cycle = input.stage2.product_tau_low.clone();
    let stage2_product_point = input
        .stage2
        .batch_rounds
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
        .batch_rounds
        .try_instance_point(log_t)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::SpartanShift, error))?;
    let stage3_cycle = stage3_shift_point.iter().rev().copied().collect::<Vec<_>>();
    let stage4_cycle = point_suffix(
        input.stage4.output_points.registers_read_write_point(),
        REGISTER_ADDRESS_BITS,
        JoltRelationId::BytecodeReadRaf,
    )?;
    let stage5_cycle = point_suffix(
        input.stage5.output_points.registers_opening_point(),
        REGISTER_ADDRESS_BITS,
        JoltRelationId::BytecodeReadRaf,
    )?;
    let entry_bytecode_index = input
        .preprocessing
        .program
        .entry_bytecode_index()
        .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: "entry address was not found in bytecode preprocessing".to_string(),
        })?;
    let (spartan_outer_raf, spartan_shift_raf, entry) =
        if input.checked.precommitted.bytecode.is_some() {
            // The field-inline extension anchors the field access selectors through the public
            // public bytecode, which committed-program mode cannot supply; the stage-6b batch build
            // already rejected this combination, so this arm is reachable only when
            // field-inline is disabled.
            #[cfg(feature = "field-inline")]
            return Err(crate::stages::stage6b::field_inline::committed_program_rejection());
            #[cfg(not(feature = "field-inline"))]
            {
                let v = bytecode::read_raf_committed_public_values::<PCS::Field>(
                    BytecodeReadRafCommittedEvaluationInputs {
                        r_address: &bytecode_r_address,
                        r_cycle: &bytecode_r_cycle,
                        stage_cycle_points: [
                            stage1_cycle.as_slice(),
                            stage2_cycle.as_slice(),
                            stage3_cycle.as_slice(),
                            stage4_cycle,
                            stage5_cycle,
                        ],
                        entry_bytecode_index,
                    },
                );
                for (index, stage_cycle_eq) in v.stage_cycle_eqs.iter().enumerate() {
                    values.public(
                        JoltDerivedId::from(BytecodeReadRafPublic::StageCycleEq(index)),
                        *stage_cycle_eq,
                    )?;
                }
                (v.spartan_outer_raf, v.spartan_shift_raf, v.entry)
            }
        } else {
            let full_program = input.preprocessing.program.as_full().ok_or_else(|| {
                VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::BytecodeReadRaf,
                    reason: "full bytecode table is unavailable".to_string(),
                }
            })?;
            let stage_gamma_powers = bytecode_challenges.stage_gamma_powers();
            let bytecode_rows = &full_program.bytecode.bytecode;
            #[cfg_attr(not(feature = "field-inline"), expect(unused_mut))]
            let mut v =
                bytecode::read_raf_public_values::<PCS::Field>(BytecodeReadRafEvaluationInputs {
                    bytecode: bytecode_rows,
                    r_address: &bytecode_r_address,
                    r_cycle: &bytecode_r_cycle,
                    stage_cycle_points: [
                        &stage1_cycle,
                        &stage2_cycle,
                        &stage3_cycle,
                        stage4_cycle,
                        stage5_cycle,
                    ],
                    register_read_write_point: point_prefix(
                        input.stage4.output_points.registers_read_write_point(),
                        REGISTER_ADDRESS_BITS,
                        JoltRelationId::BytecodeReadRaf,
                    )?,
                    register_val_evaluation_point: point_prefix(
                        input.stage5.output_points.registers_opening_point(),
                        REGISTER_ADDRESS_BITS,
                        JoltRelationId::BytecodeReadRaf,
                    )?,
                    entry_bytecode_index,
                    stage1_gammas: &stage_gamma_powers[0],
                    stage2_gammas: &stage_gamma_powers[1],
                    stage3_gammas: &stage_gamma_powers[2],
                    stage4_gammas: &stage_gamma_powers[3],
                    stage5_gammas: &stage_gamma_powers[4],
                })
                .map_err(|error| public_error(JoltRelationId::BytecodeReadRaf, error))?;
            // The composed publics: the field-register access contributions (already
            // cycle-weighted per stage) add onto the ordinary staged publics BEFORE they bake,
            // so the same `StageValue(i)` publics the symbolic output expression references
            // carry both families — exactly the clear composed relation's public composition.
            #[cfg(feature = "field-inline")]
            field_inline::extend_bytecode_stage_values(
                &mut v.stage_values,
                &input.preprocessing.program,
                &bytecode_r_address,
                &bytecode_r_cycle,
                input
                    .stage4
                    .output_points
                    .field_registers_read_write_point(),
                input
                    .stage5
                    .output_points
                    .field_registers_val_evaluation_point(),
                bytecode_challenges,
            )?;
            for (index, stage_value) in v.stage_values.iter().enumerate() {
                values.public(
                    JoltDerivedId::from(BytecodeReadRafPublic::StageValue(index)),
                    *stage_value,
                )?;
            }
            (v.spartan_outer_raf, v.spartan_shift_raf, v.entry)
        };
    values.public(
        JoltDerivedId::from(BytecodeReadRafPublic::SpartanOuterRaf),
        spartan_outer_raf,
    )?;
    values.public(
        JoltDerivedId::from(BytecodeReadRafPublic::SpartanShiftRaf),
        spartan_shift_raf,
    )?;
    values.public(JoltDerivedId::from(BytecodeReadRafPublic::Entry), entry)?;

    let booleanity_address_point = input
        .stage6a
        .rounds
        .try_instance_point(booleanity_address_rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::Booleanity, error))?;
    let booleanity_point = input
        .stage6b
        .batch_rounds
        .try_instance_point(booleanity_rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::Booleanity, error))?;
    // The (little-endian) reference cycle is the reversed stage-5 instruction
    // cycle, so its big-endian form here is the stage-5 point itself.
    let reference_eq_point = input
        .stage6a
        .challenges
        .booleanity
        .reference_address
        .iter()
        .rev()
        .chain(input.stage5.output_points.instruction_r_cycle().iter())
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
        .stage6b
        .batch_rounds
        .try_instance_point(ram_hamming_rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RamHammingBooleanity, error))?;
    let stage1_cycle_binding = point_suffix(
        &stage1_remainder_challenges,
        1,
        JoltRelationId::RamHammingBooleanity,
    )?;
    values.public(
        JoltDerivedId::from(RamHammingBooleanityPublic::EqCycle),
        try_eq_mle(&ram_hamming_point, stage1_cycle_binding)
            .map_err(|error| public_error(JoltRelationId::RamHammingBooleanity, error))?,
    )?;

    let ram_ra_point = input
        .stage6b
        .batch_rounds
        .try_instance_point(ram_ra_rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::RamRaVirtualization, error))?;
    let ram_ra_cycle = trace_dimensions
        .cycle_opening_point(&ram_ra_point)
        .map_err(|error| public_error(JoltRelationId::RamRaVirtualization, error))?;
    let ram_reduced_cycle = point_suffix(
        input.stage5.output_points.ram_reduced_opening_point(),
        log_k,
        JoltRelationId::RamRaVirtualization,
    )?;
    values.public(
        JoltDerivedId::from(RamRaVirtualizationPublic::EqCycle),
        try_eq_mle(ram_reduced_cycle, &ram_ra_cycle)
            .map_err(|error| public_error(JoltRelationId::RamRaVirtualization, error))?,
    )?;

    let instruction_ra_point = input
        .stage6b
        .batch_rounds
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
        .stage6b
        .batch_rounds
        .try_instance_point(inc_rounds)
        .map_err(|error| stage_sumcheck_error(JoltRelationId::IncClaimReduction, error))?;
    let inc_opening_point = trace_dimensions
        .cycle_opening_point(&inc_point)
        .map_err(|error| public_error(JoltRelationId::IncClaimReduction, error))?;
    values.public(
        JoltDerivedId::from(IncClaimReductionPublic::EqRamReadWrite),
        try_eq_mle(
            &inc_opening_point,
            point_suffix(
                input.stage2.output_points.ram_read_write_point(),
                log_k,
                JoltRelationId::IncClaimReduction,
            )?,
        )
        .map_err(|error| public_error(JoltRelationId::IncClaimReduction, error))?,
    )?;
    values.public(
        JoltDerivedId::from(IncClaimReductionPublic::EqRamValCheck),
        try_eq_mle(
            &inc_opening_point,
            point_suffix(
                input.stage4.output_points.ram_val_check_point(),
                log_k,
                JoltRelationId::IncClaimReduction,
            )?,
        )
        .map_err(|error| public_error(JoltRelationId::IncClaimReduction, error))?,
    )?;
    values.public(
        JoltDerivedId::from(IncClaimReductionPublic::EqRegistersReadWrite),
        try_eq_mle(
            &inc_opening_point,
            point_suffix(
                input.stage4.output_points.registers_read_write_point(),
                REGISTER_ADDRESS_BITS,
                JoltRelationId::IncClaimReduction,
            )?,
        )
        .map_err(|error| public_error(JoltRelationId::IncClaimReduction, error))?,
    )?;
    values.public(
        JoltDerivedId::from(IncClaimReductionPublic::EqRegistersValEvaluation),
        try_eq_mle(
            &inc_opening_point,
            point_suffix(
                input.stage5.output_points.registers_opening_point(),
                REGISTER_ADDRESS_BITS,
                JoltRelationId::IncClaimReduction,
            )?,
        )
        .map_err(|error| public_error(JoltRelationId::IncClaimReduction, error))?,
    )?;

    #[cfg(feature = "field-inline")]
    field_inline::stage6b_inc_publics(
        values,
        &inc_opening_point,
        input.stage6b.challenges.field_registers_inc_gamma,
        input
            .stage4
            .output_points
            .field_registers_read_write_point(),
        input
            .stage5
            .output_points
            .field_registers_val_evaluation_point(),
    )?;

    Ok(())
}

fn bytecode_reduction_weights<PCS, VC>(
    input: &ResidualInputs<'_, PCS, VC>,
    layout: &BytecodeClaimReductionLayout,
) -> Result<BytecodeReductionWeights<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let eta = input
        .stage6b
        .challenges
        .bytecode_reduction_eta
        .ok_or_else(|| VerifierError::MissingStageClaimChallenge {
            id: JoltChallengeId::from(BytecodeClaimReductionChallenge::Eta).into(),
        })?;
    let stage_gamma_powers = input
        .stage6a
        .challenges
        .bytecode_read_raf
        .stage_gamma_powers();
    committed_reduction_cycle_phase::bytecode_reduction_weights(
        layout,
        bytecode_reduction::BytecodeLaneWeightInputs {
            eta,
            stage1_gammas: &stage_gamma_powers[0],
            stage2_gammas: &stage_gamma_powers[1],
            stage3_gammas: &stage_gamma_powers[2],
            stage4_gammas: &stage_gamma_powers[3],
            stage5_gammas: &stage_gamma_powers[4],
            register_read_write_point: point_prefix(
                input.stage4.output_points.registers_read_write_point(),
                REGISTER_ADDRESS_BITS,
                JoltRelationId::BytecodeClaimReduction,
            )?,
            register_val_evaluation_point: point_prefix(
                input.stage5.output_points.registers_opening_point(),
                REGISTER_ADDRESS_BITS,
                JoltRelationId::BytecodeClaimReduction,
            )?,
        },
        &input.stage6a.output_points.bytecode_read_raf.intermediate,
    )
}

fn add_bytecode_reduction_cycle_publics<PCS, VC>(
    input: &ResidualInputs<'_, PCS, VC>,
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
        .stage6b
        .output_points
        .bytecode_reduction_opening_point()
        .ok_or_else(|| VerifierError::MissingOpeningClaim {
            id: bytecode_reduction::cycle_phase_intermediate_opening().into(),
        })?;
    let weights = bytecode_reduction_weights(input, layout)?;
    // The cycle scale recovered from the produced opening point equals the
    // sumcheck-point form (`cycle_phase_permuted_*` agree — unit-tested in
    // `claim_reductions::precommitted`), matching the clear relation's path.
    let weight = layout
        .cycle_phase_final_output_weight_at_opening_point(weights.as_inputs(), opening_point)
        .map_err(|error| public_error(JoltRelationId::BytecodeClaimReductionCyclePhase, error))?;
    values.public(
        JoltDerivedId::from(BytecodeClaimReductionPublic::OutputWeight),
        weight,
    )
}

fn add_bytecode_reduction_address_publics<PCS, VC>(
    input: &ResidualInputs<'_, PCS, VC>,
    values: &mut SourceValues<PCS::Field>,
    layout: &BytecodeClaimReductionLayout,
    sumcheck_point: &[PCS::Field],
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let cycle_phase_variables = input
        .stage6b
        .output_points
        .bytecode_cycle_phase_variables()
        .ok_or_else(|| VerifierError::MissingOpeningClaim {
            id: bytecode_reduction::cycle_phase_intermediate_opening().into(),
        })?;
    let weights = bytecode_reduction_weights(input, layout)?;
    let weight = layout
        .address_phase_final_output_weight(
            weights.as_inputs(),
            &cycle_phase_variables,
            sumcheck_point,
        )
        .map_err(|error| public_error(JoltRelationId::BytecodeClaimReduction, error))?;
    values.public(
        JoltDerivedId::from(BytecodeClaimReductionPublic::OutputWeight),
        weight,
    )
}

fn add_program_image_reduction_cycle_publics<PCS, VC>(
    input: &ResidualInputs<'_, PCS, VC>,
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
        .stage6b
        .output_points
        .program_image_opening_point()
        .ok_or_else(|| VerifierError::MissingOpeningClaim {
            id: program_image::cycle_phase_program_image_opening().into(),
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

fn add_program_image_reduction_address_publics<PCS, VC>(
    input: &ResidualInputs<'_, PCS, VC>,
    values: &mut SourceValues<PCS::Field>,
    layout: &ProgramImageClaimReductionLayout,
    sumcheck_point: &[PCS::Field],
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let cycle_phase_variables = input
        .stage6b
        .output_points
        .program_image_cycle_phase_variables()
        .ok_or_else(|| VerifierError::MissingOpeningClaim {
            id: program_image::cycle_phase_program_image_opening().into(),
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

fn add_advice_cycle_publics<PCS, VC>(
    input: &ResidualInputs<'_, PCS, VC>,
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
        .stage6b
        .output_points
        .advice_cycle_phase_opening_point(kind)
        .ok_or_else(|| VerifierError::MissingOpeningClaim {
            id: advice::cycle_phase_advice_opening(kind).into(),
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

fn add_advice_address_publics<PCS, VC>(
    input: &ResidualInputs<'_, PCS, VC>,
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
        .stage6b
        .output_points
        .advice_cycle_phase_variables(kind)
        .ok_or_else(|| VerifierError::MissingOpeningClaim {
            id: advice::cycle_phase_advice_opening(kind).into(),
        })?;
    let scale = layout
        .address_phase_final_output_scale(&source_point, &cycle_phase_variables, sumcheck_point)
        .map_err(|error| public_error(JoltRelationId::AdviceClaimReduction, error))?;
    values.public(
        JoltDerivedId::from(AdviceClaimReductionPublic::FinalScale(kind)),
        scale,
    )
}

fn stage6_virtualization_points<PCS, VC>(
    input: &ResidualInputs<'_, PCS, VC>,
    dimensions: hamming_weight::HammingWeightClaimReductionDimensions,
) -> Result<Vec<Vec<PCS::Field>>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let output_points = &input.stage6b.output_points;
    output_points
        .instruction_ra_virtualization
        .committed_instruction_ra
        .iter()
        .chain(&output_points.bytecode_read_raf.bytecode_ra)
        .chain(&output_points.ram_ra_virtualization.ram_ra)
        .map(|point| hamming_virtualization_address_point(dimensions.log_k_chunk, point))
        .collect()
}

fn hamming_virtualization_address_point<F: JoltField>(
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

impl<F: JoltField> SourceValues<F> {
    fn public(&mut self, id: impl Into<VerifierPublicId>, value: F) -> Result<(), VerifierError> {
        let id = id.into();
        if let Some((_, existing)) = self.publics.iter().find(|(candidate, _)| *candidate == id) {
            if *existing != value {
                return Err(VerifierError::ResidualConstructionFailed {
                    reason: format!("public source {id:?} was assigned inconsistent values"),
                });
            }
            return Ok(());
        }
        self.publics.push((id, value));
        Ok(())
    }

    fn value(&self, id: VerifierPublicId) -> Option<F> {
        self.publics
            .iter()
            .find(|(candidate, _)| *candidate == id)
            .map(|(_, value)| *value)
    }
}

fn stage_sumcheck_error<F: JoltField>(
    stage: JoltRelationId,
    error: jolt_sumcheck::SumcheckError<F>,
) -> VerifierError {
    VerifierError::StageClaimSumcheckFailed {
        stage: format!("{stage:?}"),
        reason: error.to_string(),
    }
}

fn public_error(stage: JoltRelationId, error: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage,
        reason: error.to_string(),
    }
}

fn point_prefix<F: JoltField>(
    point: &[F],
    prefix_len: usize,
    stage: JoltRelationId,
) -> Result<&[F], VerifierError> {
    point.get(..prefix_len).ok_or_else(|| {
        public_error(
            stage,
            format!(
                "opening point is too short: expected at least {prefix_len} variables, got {}",
                point.len()
            ),
        )
    })
}

fn point_suffix<F: JoltField>(
    point: &[F],
    prefix_len: usize,
    stage: JoltRelationId,
) -> Result<&[F], VerifierError> {
    point.get(prefix_len..).ok_or_else(|| {
        public_error(
            stage,
            format!(
                "opening point is too short: expected at least {prefix_len} variables, got {}",
                point.len()
            ),
        )
    })
}

fn residual_error(reason: impl ToString) -> VerifierError {
    VerifierError::ResidualConstructionFailed {
        reason: reason.to_string(),
    }
}

/// Value-parity locks for every FieldInline-family batch member lowered
/// through the generic [`relation_claim`] / [`map_expr`]: the lowered
/// input/output expressions, evaluated against the composite claim sources
/// (hidden rows by composite id, challenges and deriveds as
/// [`VerifierPublicId`] publics), reproduce the clear path's
/// `ConcreteSumcheck::input_claim` / `expected_output` on synthetic values.
#[cfg(all(test, feature = "field-inline"))]
#[expect(clippy::unwrap_used)]
#[expect(
    clippy::as_conversions,
    clippy::arithmetic_side_effects,
    reason = "tests use plain arithmetic on fixture data"
)]
#[expect(
    clippy::panic,
    reason = "unexpected expression sources must fail parity tests"
)]
mod field_inline_relation_parity {
    use super::*;
    use crate::stages::relations::{
        ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
        SumcheckOutputClaims, SumcheckOutputPoints,
    };
    use jolt_algebra::{Fr, Ring};
    use jolt_claims::protocols::field_inline::{
        FieldInlineChallengeId, FieldInlineDerivedId, FieldInlineOpeningId,
        FieldRegistersTraceDimensions,
    };
    use jolt_claims::{InputClaims, OutputClaims, SumcheckChallenges};
    use jolt_transcript::PadSlots;

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn point(start: u64, len: usize) -> Vec<Fr> {
        (0..len as u64).map(|i| fr(start + i)).collect()
    }

    /// Assert the lowered `(input, output)` expressions of `relation` evaluate
    /// to the clear claims using the production ZK public-value assembly.
    fn assert_relation_parity<S>(
        relation: &S,
        inputs: &SumcheckInputClaims<Fr, S>,
        input_points: &SumcheckInputPoints<Fr, S>,
        outputs: &SumcheckOutputClaims<Fr, S>,
        sumcheck_point: &[Fr],
        challenges: &ConcreteSumcheckChallenges<Fr, S>,
        assemble_publics: impl FnOnce(&SumcheckOutputPoints<Fr, S>) -> SourceValues<Fr>,
    ) where
        S: ConcreteSumcheck<Fr>,
        S::Symbolic: SymbolicSumcheck<
            OpeningId = FieldInlineOpeningId,
            DerivedId = FieldInlineDerivedId,
            ChallengeId = FieldInlineChallengeId,
        >,
        SumcheckInputClaims<Fr, S>: InputClaims<Fr, FieldInlineOpeningId>,
        SumcheckOutputClaims<Fr, S>: OutputClaims<Fr, FieldInlineOpeningId>,
        ConcreteSumcheckChallenges<Fr, S>: SumcheckChallenges<Fr, FieldInlineChallengeId>,
    {
        let output_points: SumcheckOutputPoints<Fr, S> = relation
            .derive_opening_points(sumcheck_point, input_points)
            .unwrap();

        let clear_input = relation.input_claim(inputs, challenges).unwrap();
        let claim = relation_claim::<Fr, S::Symbolic>(relation.symbolic());
        let lowered_input_expr = claim.input.expand();
        let lowered_output_expr = claim.output.expand();
        let publics = assemble_publics(&output_points);
        let resolve_challenge = |id: &VerifierPublicId| {
            publics
                .publics
                .iter()
                .find(|(candidate, _)| candidate == id)
                .unwrap()
                .1
        };
        let lowered_input = lowered_input_expr.evaluate(
            |id| match id {
                ComposedOpeningId::FieldInline(id) => inputs.resolve_input(id).unwrap(),
                ComposedOpeningId::Jolt(_) => panic!("unexpected Jolt opening"),
            },
            |_| panic!("lowered expression retained a challenge"),
            resolve_challenge,
        );
        assert_eq!(lowered_input, clear_input, "input claim parity");

        let clear_output = relation
            .expected_output(input_points, outputs, &output_points, challenges)
            .unwrap();
        let lowered_output = lowered_output_expr.evaluate(
            |id| match id {
                ComposedOpeningId::FieldInline(id) => outputs.resolve_output(id).unwrap(),
                ComposedOpeningId::Jolt(_) => panic!("unexpected Jolt opening"),
            },
            |_| panic!("lowered expression retained a challenge"),
            resolve_challenge,
        );
        assert_eq!(lowered_output, clear_output, "output claim parity");
    }

    #[test]
    fn field_registers_claim_reduction_lowering_matches_the_clear_relation() {
        use crate::stages::stage2::field_registers_claim_reduction::{
            FieldRegistersClaimReduction, FieldRegistersClaimReductionChallenges,
            FieldRegistersClaimReductionInputClaims, FieldRegistersClaimReductionOutputClaims,
        };

        let log_t = 4usize;
        let relation = FieldRegistersClaimReduction::<Fr>::new(
            FieldRegistersTraceDimensions::new(log_t),
            point(50, log_t),
        );
        assert_relation_parity(
            &relation,
            &FieldRegistersClaimReductionInputClaims {
                rd_value: fr(3),
                rs1_value: fr(5),
                rs2_value: fr(7),
            },
            &FieldRegistersClaimReductionInputClaims::default(),
            &FieldRegistersClaimReductionOutputClaims {
                rd_value: fr(11),
                rs1_value: fr(13),
                rs2_value: fr(17),
            },
            &point(60, log_t),
            &FieldRegistersClaimReductionChallenges { gamma: fr(19) },
            |_| {
                let mut publics = SourceValues::default();
                let rounds = point(900, 2)
                    .into_iter()
                    .chain(point(60, log_t))
                    .map(|challenge| MaskedRound {
                        wire: Vec::new(),
                        slots: PadSlots::default(),
                        challenge,
                    })
                    .collect();
                let batch = BatchedMaskedRounds {
                    rounds,
                    batching_coefficients: vec![fr(1)],
                    degrees: DegreeSchedule::uniform(log_t + 2, 3),
                };
                let _ = field_inline::stage2_claim_reduction(
                    &mut publics,
                    log_t,
                    &batch,
                    fr(19),
                    relation.tau_low(),
                )
                .unwrap();
                publics
            },
        );
    }

    #[test]
    fn field_registers_read_write_lowering_matches_the_clear_relation() {
        use crate::stages::stage4::field_registers_read_write_checking::{
            FieldRegistersReadWriteChallenges, FieldRegistersReadWriteChecking,
            FieldRegistersReadWriteInputClaims, FieldRegistersReadWriteOutputClaims,
        };
        use jolt_claims::protocols::field_inline::FieldInlineConfig;

        let log_t = 4usize;
        let dimensions = FieldInlineConfig::enabled().read_write_dimensions(log_t);
        let relation = FieldRegistersReadWriteChecking::<Fr>::new(dimensions);
        let fixed_cycle = point(70, log_t);
        assert_relation_parity(
            &relation,
            &FieldRegistersReadWriteInputClaims {
                rd_value: fr(3),
                rs1_value: fr(5),
                rs2_value: fr(7),
            },
            &FieldRegistersReadWriteInputClaims {
                rd_value: fixed_cycle.clone(),
                rs1_value: fixed_cycle.clone(),
                rs2_value: fixed_cycle.clone(),
            },
            &FieldRegistersReadWriteOutputClaims {
                registers_val: fr(11),
                rs1_ra: fr(13),
                rs2_ra: fr(17),
                rd_wa: fr(19),
                rd_inc: fr(23),
            },
            &point(80, relation.rounds()),
            &FieldRegistersReadWriteChallenges { gamma: fr(29) },
            |points| {
                let mut publics = SourceValues::default();
                let _ = field_inline::stage4_read_write(
                    &mut publics,
                    log_t,
                    fr(29),
                    &fixed_cycle,
                    points.registers_val(),
                )
                .unwrap();
                publics
            },
        );
    }

    #[test]
    fn field_registers_val_evaluation_lowering_matches_the_clear_relation() {
        use crate::stages::stage5::field_registers_val_evaluation::{
            FieldRegistersValEvaluation, FieldRegistersValEvaluationInputClaims,
            FieldRegistersValEvaluationOutputClaims,
        };
        use jolt_claims::protocols::field_inline::FIELD_REGISTERS_LOG_K;
        use jolt_claims::NoChallenges;

        let log_t = 4usize;
        let relation =
            FieldRegistersValEvaluation::<Fr>::new(FieldRegistersTraceDimensions::new(log_t));
        assert_relation_parity(
            &relation,
            &FieldRegistersValEvaluationInputClaims {
                registers_val: fr(3),
            },
            &FieldRegistersValEvaluationInputClaims {
                registers_val: point(90, FIELD_REGISTERS_LOG_K + log_t),
            },
            &FieldRegistersValEvaluationOutputClaims {
                rd_inc: fr(5),
                rd_wa: fr(7),
            },
            &point(100, log_t),
            &NoChallenges::default(),
            |points| {
                let mut publics = SourceValues::default();
                let _ = field_inline::stage5_val_evaluation(
                    &mut publics,
                    log_t,
                    points.rd_inc(),
                    &point(90, FIELD_REGISTERS_LOG_K + log_t),
                )
                .unwrap();
                publics
            },
        );
    }

    #[test]
    fn field_registers_inc_claim_reduction_lowering_matches_the_clear_relation() {
        use crate::stages::stage6b::field_registers_inc_claim_reduction::{
            FieldRegistersIncClaimReduction, FieldRegistersIncClaimReductionChallenges,
            FieldRegistersIncClaimReductionInputClaims,
            FieldRegistersIncClaimReductionOutputClaims,
        };

        let log_t = 4usize;
        let relation = FieldRegistersIncClaimReduction::<Fr>::new(
            FieldRegistersTraceDimensions::new(log_t),
            point(110, log_t),
            point(120, log_t),
        );
        assert_relation_parity(
            &relation,
            &FieldRegistersIncClaimReductionInputClaims {
                rd_inc_read_write: fr(3),
                rd_inc_val_evaluation: fr(5),
            },
            &FieldRegistersIncClaimReductionInputClaims::default(),
            &FieldRegistersIncClaimReductionOutputClaims { rd_inc: fr(7) },
            &point(130, log_t),
            &FieldRegistersIncClaimReductionChallenges { gamma: fr(11) },
            |points| {
                use jolt_claims::protocols::field_inline::FIELD_REGISTERS_LOG_K;
                let mut publics = SourceValues::default();
                let address = point(500, FIELD_REGISTERS_LOG_K);
                field_inline::stage6b_inc_publics(
                    &mut publics,
                    points.rd_inc(),
                    fr(11),
                    &[address.as_slice(), &point(110, log_t)].concat(),
                    &[address.as_slice(), &point(120, log_t)].concat(),
                )
                .unwrap();
                publics
            },
        );
    }
}
