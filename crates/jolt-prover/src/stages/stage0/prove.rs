use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Debug,
};

use jolt_backends::{
    CommitmentBackend, CommitmentMode, CommitmentRequest, CommitmentRequestItem, CommitmentResult,
    CommitmentSlot, TracePolynomialEmbedding,
};
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::FieldInlineCommittedPolynomial;
use jolt_claims::protocols::jolt::{
    formulas::{dimensions::TracePolynomialOrder, ra::JoltRaPolynomialLayout},
    JoltCommittedPolynomial,
};
use jolt_openings::CommitmentScheme;
use jolt_verifier::config::{JoltProtocolConfig, ZkConfig};
#[cfg(feature = "field-inline")]
use jolt_verifier::proof::{FieldInlineCommitments, FieldRegistersCommitments};
use jolt_verifier::proof::{JoltCommitments, JoltRaCommitments};
use jolt_verifier::stages::stage8::{stage8_final_opening_order, Stage8FinalOpening};
#[cfg(feature = "field-inline")]
use jolt_witness::protocols::jolt_vm::field_inline::FieldInlineNamespace;
use jolt_witness::{
    protocols::jolt_vm::JoltVmNamespace, CommittedWitnessProvider, MaterializationPolicy,
    OracleKind, OracleRef, RetentionHint, ViewRequirement, WitnessNamespace,
};

use crate::ProverError;

#[cfg(feature = "field-inline")]
pub type FieldInlineCommitmentWitness<'a, F> =
    dyn CommittedWitnessProvider<F, FieldInlineNamespace> + Sync + 'a;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitmentStageConfig {
    pub ra_layout: JoltRaPolynomialLayout,
    pub include_trusted_advice: bool,
    pub include_untrusted_advice: bool,
    pub trace_polynomial_order: TracePolynomialOrder,
    pub final_opening_trace_rows: Option<usize>,
    pub final_opening_address_columns: Option<usize>,
}

impl CommitmentStageConfig {
    pub const fn new(
        ra_layout: JoltRaPolynomialLayout,
        include_trusted_advice: bool,
        include_untrusted_advice: bool,
    ) -> Self {
        Self {
            ra_layout,
            include_trusted_advice,
            include_untrusted_advice,
            trace_polynomial_order: TracePolynomialOrder::CycleMajor,
            final_opening_trace_rows: None,
            final_opening_address_columns: None,
        }
    }

    pub fn with_final_opening_trace_embedding(
        mut self,
        log_t: usize,
        committed_chunk_bits: usize,
        trace_polynomial_order: TracePolynomialOrder,
    ) -> Self {
        self.trace_polynomial_order = trace_polynomial_order;
        self.final_opening_trace_rows = Some(1usize << log_t);
        self.final_opening_address_columns = Some(1usize << committed_chunk_bits);
        self
    }
}

#[derive(Clone)]
pub struct CommitmentStageInput<'a, W, PCS: CommitmentScheme> {
    pub witness: &'a W,
    pub setup: &'a PCS::ProverSetup,
    pub config: CommitmentStageConfig,
    pub protocol: JoltProtocolConfig,
    #[cfg(feature = "field-inline")]
    pub field_inline_witness: &'a FieldInlineCommitmentWitness<'a, PCS::Field>,
}

impl<'a, W, PCS: CommitmentScheme> CommitmentStageInput<'a, W, PCS> {
    pub fn new(
        witness: &'a W,
        setup: &'a PCS::ProverSetup,
        config: CommitmentStageConfig,
        protocol: JoltProtocolConfig,
        #[cfg(feature = "field-inline")] field_inline_witness: &'a FieldInlineCommitmentWitness<
            'a,
            PCS::Field,
        >,
    ) -> Self {
        Self {
            witness,
            setup,
            config,
            protocol,
            #[cfg(feature = "field-inline")]
            field_inline_witness,
        }
    }
}

#[derive(Clone)]
pub struct CommitmentComponent<PCS: CommitmentScheme> {
    pub commitments: JoltCommitments<PCS::Output>,
    pub trusted_advice_commitment: Option<PCS::Output>,
    pub untrusted_advice_commitment: Option<PCS::Output>,
    pub(crate) prover_state: CommitmentProverState<PCS::OpeningHint>,
}

#[derive(Clone)]
pub(crate) struct CommitmentProverState<OpeningHint> {
    pub opening_hints: BTreeMap<JoltCommittedPolynomial, OpeningHint>,
    #[cfg(feature = "field-inline")]
    pub field_inline_opening_hints: BTreeMap<FieldInlineCommittedPolynomial, OpeningHint>,
}

pub type Stage8OpeningInputs<PCS> = (
    Vec<<PCS as jolt_crypto::Commitment>::Output>,
    Vec<<PCS as CommitmentScheme>::OpeningHint>,
);

impl<PCS: CommitmentScheme> CommitmentComponent<PCS> {
    pub fn from_backend_result(
        result: CommitmentResult<JoltVmNamespace, PCS>,
        #[cfg(feature = "field-inline")] field_inline_result: CommitmentResult<
            FieldInlineNamespace,
            PCS,
        >,
        config: CommitmentStageConfig,
    ) -> Result<Self, ProverError> {
        let mut jolt_outputs =
            CommitmentOutputMap::<JoltCommittedPolynomial, PCS>::from_backend_result(
                "Jolt", result,
            )?;
        let jolt = take_jolt_commitments(&mut jolt_outputs, config)?;
        let opening_hints = jolt_outputs.finish()?;

        #[cfg(feature = "field-inline")]
        let mut field_inline_outputs =
            CommitmentOutputMap::<FieldInlineCommittedPolynomial, PCS>::from_backend_result(
                "field-inline",
                field_inline_result,
            )?;
        #[cfg(feature = "field-inline")]
        let field_inline = take_field_inline_commitments(&mut field_inline_outputs)?;
        #[cfg(feature = "field-inline")]
        let field_inline_opening_hints = field_inline_outputs.finish()?;

        #[cfg(not(feature = "field-inline"))]
        let commitments = JoltCommitments::new(jolt.rd_inc, jolt.ram_inc, jolt.ra);
        #[cfg(feature = "field-inline")]
        let commitments = JoltCommitments::new(jolt.rd_inc, jolt.ram_inc, jolt.ra, field_inline);

        Ok(Self {
            commitments,
            trusted_advice_commitment: jolt.trusted_advice_commitment,
            untrusted_advice_commitment: jolt.untrusted_advice_commitment,
            prover_state: CommitmentProverState {
                opening_hints,
                #[cfg(feature = "field-inline")]
                field_inline_opening_hints,
            },
        })
    }

    pub fn stage8_opening_inputs(
        &self,
        layout: JoltRaPolynomialLayout,
    ) -> Result<Stage8OpeningInputs<PCS>, ProverError>
    where
        PCS::Output: Clone,
        PCS::OpeningHint: Clone,
    {
        let final_openings = stage8_final_opening_order(
            layout,
            self.trusted_advice_commitment.is_some(),
            self.untrusted_advice_commitment.is_some(),
        );
        let mut ordered_commitments = Vec::with_capacity(final_openings.len());
        let mut opening_hints = Vec::with_capacity(ordered_commitments.capacity());

        for opening in final_openings {
            match opening {
                Stage8FinalOpening::Jolt(polynomial) => {
                    let commitment = self.jolt_stage8_commitment(polynomial)?;
                    push_jolt_stage8_opening(
                        &mut ordered_commitments,
                        &mut opening_hints,
                        commitment,
                        &self.prover_state,
                        polynomial,
                    )?;
                }
                #[cfg(feature = "field-inline")]
                Stage8FinalOpening::FieldInline(polynomial) => {
                    let commitment = self.field_inline_stage8_commitment(polynomial);
                    push_field_inline_stage8_opening(
                        &mut ordered_commitments,
                        &mut opening_hints,
                        commitment,
                        &self.prover_state,
                        polynomial,
                    )?;
                }
            }
        }

        Ok((ordered_commitments, opening_hints))
    }

    fn jolt_stage8_commitment(
        &self,
        polynomial: JoltCommittedPolynomial,
    ) -> Result<&PCS::Output, ProverError> {
        match polynomial {
            JoltCommittedPolynomial::RdInc => Ok(&self.commitments.rd_inc),
            JoltCommittedPolynomial::RamInc => Ok(&self.commitments.ram_inc),
            JoltCommittedPolynomial::InstructionRa(index) => self
                .commitments
                .ra
                .instruction
                .get(index)
                .ok_or_else(|| missing_stage8_commitment(polynomial)),
            JoltCommittedPolynomial::BytecodeRa(index) => self
                .commitments
                .ra
                .bytecode
                .get(index)
                .ok_or_else(|| missing_stage8_commitment(polynomial)),
            JoltCommittedPolynomial::RamRa(index) => self
                .commitments
                .ra
                .ram
                .get(index)
                .ok_or_else(|| missing_stage8_commitment(polynomial)),
            JoltCommittedPolynomial::TrustedAdvice => self
                .trusted_advice_commitment
                .as_ref()
                .ok_or_else(|| missing_stage8_commitment(polynomial)),
            JoltCommittedPolynomial::UntrustedAdvice => self
                .untrusted_advice_commitment
                .as_ref()
                .ok_or_else(|| missing_stage8_commitment(polynomial)),
        }
    }

    #[cfg(feature = "field-inline")]
    fn field_inline_stage8_commitment(
        &self,
        polynomial: FieldInlineCommittedPolynomial,
    ) -> &PCS::Output {
        match polynomial {
            FieldInlineCommittedPolynomial::FieldRdInc => {
                &self.commitments.field_inline.field_registers.rd_inc
            }
        }
    }
}

fn missing_stage8_commitment(polynomial: JoltCommittedPolynomial) -> ProverError {
    ProverError::InvalidStageRequest {
        reason: format!("Stage 8 opening commitment is missing for {polynomial:?}"),
    }
}

fn push_jolt_stage8_opening<C, H>(
    commitments: &mut Vec<C>,
    hints: &mut Vec<H>,
    commitment: &C,
    prover_state: &CommitmentProverState<H>,
    polynomial: JoltCommittedPolynomial,
) -> Result<(), ProverError>
where
    C: Clone,
    H: Clone,
{
    commitments.push(commitment.clone());
    let hint = prover_state
        .opening_hints
        .get(&polynomial)
        .cloned()
        .ok_or_else(|| ProverError::InvalidStageRequest {
            reason: format!("Stage 8 opening hint is missing for {polynomial:?}"),
        })?;
    hints.push(hint);
    Ok(())
}

#[cfg(feature = "field-inline")]
fn push_field_inline_stage8_opening<C, H>(
    commitments: &mut Vec<C>,
    hints: &mut Vec<H>,
    commitment: &C,
    prover_state: &CommitmentProverState<H>,
    polynomial: FieldInlineCommittedPolynomial,
) -> Result<(), ProverError>
where
    C: Clone,
    H: Clone,
{
    commitments.push(commitment.clone());
    let hint = prover_state
        .field_inline_opening_hints
        .get(&polynomial)
        .cloned()
        .ok_or_else(|| ProverError::InvalidStageRequest {
            reason: format!("Stage 8 field-inline opening hint is missing for {polynomial:?}"),
        })?;
    hints.push(hint);
    Ok(())
}

pub fn prove<F, W, B, PCS>(
    input: CommitmentStageInput<'_, W, PCS>,
    backend: &mut B,
) -> Result<CommitmentComponent<PCS>, ProverError>
where
    PCS: CommitmentScheme<Field = F>,
    W: CommittedWitnessProvider<F, JoltVmNamespace> + Sync,
    B: CommitmentStageBackend<F, PCS>,
{
    let mode = match input.protocol.zk {
        ZkConfig::Transparent => CommitmentMode::Transparent,
        ZkConfig::BlindFold => CommitmentMode::Zk,
    };
    let trace_embedding = final_opening_trace_embedding(&input.config)?;
    let result = commit::<F, JoltVmNamespace, W, B, PCS>(
        input.witness,
        backend,
        input.setup,
        mode,
        input.config.trace_polynomial_order,
        trace_embedding,
    )?;
    #[cfg(feature = "field-inline")]
    let field_inline_result = backend.commit_field_inline(
        input.field_inline_witness,
        input.setup,
        mode,
        input.config.trace_polynomial_order,
        trace_embedding,
    )?;

    CommitmentComponent::from_backend_result(
        result,
        #[cfg(feature = "field-inline")]
        field_inline_result,
        input.config,
    )
}

pub trait CommitmentStageBackend<F, PCS>: CommitmentBackend<F, JoltVmNamespace, PCS>
where
    PCS: CommitmentScheme<Field = F>,
{
    #[cfg(feature = "field-inline")]
    fn commit_field_inline(
        &mut self,
        witness: &FieldInlineCommitmentWitness<'_, F>,
        setup: &PCS::ProverSetup,
        mode: CommitmentMode,
        trace_polynomial_order: TracePolynomialOrder,
        trace_embedding: Option<TracePolynomialEmbedding>,
    ) -> Result<CommitmentResult<FieldInlineNamespace, PCS>, ProverError>;
}

#[cfg(not(feature = "field-inline"))]
impl<F, PCS, B> CommitmentStageBackend<F, PCS> for B
where
    PCS: CommitmentScheme<Field = F>,
    B: CommitmentBackend<F, JoltVmNamespace, PCS>,
{
}

#[cfg(feature = "field-inline")]
impl<F, PCS, B> CommitmentStageBackend<F, PCS> for B
where
    PCS: CommitmentScheme<Field = F>,
    B: CommitmentBackend<F, JoltVmNamespace, PCS> + CommitmentBackend<F, FieldInlineNamespace, PCS>,
{
    fn commit_field_inline(
        &mut self,
        witness: &FieldInlineCommitmentWitness<'_, F>,
        setup: &PCS::ProverSetup,
        mode: CommitmentMode,
        trace_polynomial_order: TracePolynomialOrder,
        trace_embedding: Option<TracePolynomialEmbedding>,
    ) -> Result<CommitmentResult<FieldInlineNamespace, PCS>, ProverError> {
        commit::<F, FieldInlineNamespace, FieldInlineCommitmentWitness<'_, F>, Self, PCS>(
            witness,
            self,
            setup,
            mode,
            trace_polynomial_order,
            trace_embedding,
        )
    }
}

pub(super) fn build_commitment_request<F, N, W>(
    witness: &W,
    mode: CommitmentMode,
    trace_polynomial_order: TracePolynomialOrder,
    trace_embedding: Option<TracePolynomialEmbedding>,
) -> Result<CommitmentRequest<N>, ProverError>
where
    N: WitnessNamespace,
    W: CommittedWitnessProvider<F, N> + ?Sized,
{
    let mut items = Vec::new();
    for (index, committed) in witness.committed_oracle_order()?.into_iter().enumerate() {
        let oracle = OracleRef::committed(committed);
        let descriptor = witness.describe_oracle(oracle)?;
        let retention = witness
            .view_requirements(oracle)?
            .first()
            .map_or(RetentionHint::ThroughStage8, |requirement| {
                requirement.retention
            });

        items.push(
            CommitmentRequestItem::with_mode(
                CommitmentSlot(index as u32),
                ViewRequirement::new(
                    oracle,
                    descriptor.encoding,
                    MaterializationPolicy::Streaming,
                    retention,
                ),
                mode,
            )
            .with_trace_polynomial_order(trace_polynomial_order)
            .with_trace_embedding(trace_embedding),
        );
    }

    Ok(CommitmentRequest::new(items))
}

pub(super) fn commit<F, N, W, B, PCS>(
    witness: &W,
    backend: &mut B,
    setup: &PCS::ProverSetup,
    mode: CommitmentMode,
    trace_polynomial_order: TracePolynomialOrder,
    trace_embedding: Option<TracePolynomialEmbedding>,
) -> Result<CommitmentResult<N, PCS>, ProverError>
where
    N: WitnessNamespace,
    PCS: CommitmentScheme<Field = F>,
    W: CommittedWitnessProvider<F, N> + Sync + ?Sized,
    B: CommitmentBackend<F, N, PCS>,
{
    let request = build_commitment_request::<F, N, W>(
        witness,
        mode,
        trace_polynomial_order,
        trace_embedding,
    )?;
    Ok(backend.commit(&request, witness, setup)?)
}

struct CommitmentOutputMap<Id, PCS>
where
    Id: Copy + Ord + Debug,
    PCS: CommitmentScheme,
{
    label: &'static str,
    pending: BTreeMap<Id, (PCS::Output, PCS::OpeningHint)>,
    opening_hints: BTreeMap<Id, PCS::OpeningHint>,
}

impl<Id, PCS> CommitmentOutputMap<Id, PCS>
where
    Id: Copy + Ord + Debug,
    PCS: CommitmentScheme,
{
    fn from_backend_result<N>(
        label: &'static str,
        result: CommitmentResult<N, PCS>,
    ) -> Result<Self, ProverError>
    where
        N: WitnessNamespace<CommittedId = Id>,
    {
        let mut pending = BTreeMap::new();
        let mut slots = BTreeSet::new();

        for output in result.commitments {
            if !slots.insert(output.slot.0) {
                return Err(ProverError::InvalidCommitmentOutput {
                    reason: format!("duplicate {label} commitment output slot {:?}", output.slot),
                });
            }
            let OracleKind::Committed(polynomial) = output.oracle.kind else {
                return Err(ProverError::InvalidCommitmentOutput {
                    reason: format!("{label} commitment backend emitted a virtual oracle output"),
                });
            };
            if output.rows == 0 {
                return Err(ProverError::InvalidCommitmentOutput {
                    reason: format!("{label} commitment output for {polynomial:?} has zero rows"),
                });
            }
            if pending
                .insert(polynomial, (output.commitment, output.opening_hint))
                .is_some()
            {
                return Err(ProverError::InvalidCommitmentOutput {
                    reason: format!("duplicate {label} commitment output for {polynomial:?}"),
                });
            }
        }

        Ok(Self {
            label,
            pending,
            opening_hints: BTreeMap::new(),
        })
    }

    fn take(&mut self, polynomial: Id) -> Result<PCS::Output, ProverError> {
        let (commitment, opening_hint) = self.pending.remove(&polynomial).ok_or_else(|| {
            ProverError::InvalidCommitmentOutput {
                reason: format!(
                    "missing {} commitment output for {polynomial:?}",
                    self.label
                ),
            }
        })?;
        let _ = self.opening_hints.insert(polynomial, opening_hint);
        Ok(commitment)
    }

    fn take_vec(
        &mut self,
        polynomial_at: impl Fn(usize) -> Id,
        count: usize,
    ) -> Result<Vec<PCS::Output>, ProverError> {
        let mut commitments = Vec::with_capacity(count);
        for index in 0..count {
            commitments.push(self.take(polynomial_at(index))?);
        }
        Ok(commitments)
    }

    fn take_optional(
        &mut self,
        polynomial: Id,
        expected: bool,
    ) -> Result<Option<PCS::Output>, ProverError> {
        expected.then(|| self.take(polynomial)).transpose()
    }

    fn finish(self) -> Result<BTreeMap<Id, PCS::OpeningHint>, ProverError> {
        if let Some(unexpected) = self.pending.keys().next().copied() {
            return Err(ProverError::InvalidCommitmentOutput {
                reason: format!(
                    "unexpected {} commitment output for {unexpected:?}",
                    self.label
                ),
            });
        }
        Ok(self.opening_hints)
    }
}

struct JoltCommitmentParts<C> {
    rd_inc: C,
    ram_inc: C,
    ra: JoltRaCommitments<C>,
    trusted_advice_commitment: Option<C>,
    untrusted_advice_commitment: Option<C>,
}

fn take_jolt_commitments<PCS>(
    outputs: &mut CommitmentOutputMap<JoltCommittedPolynomial, PCS>,
    config: CommitmentStageConfig,
) -> Result<JoltCommitmentParts<PCS::Output>, ProverError>
where
    PCS: CommitmentScheme,
{
    let rd_inc = outputs.take(JoltCommittedPolynomial::RdInc)?;
    let ram_inc = outputs.take(JoltCommittedPolynomial::RamInc)?;
    let instruction = outputs.take_vec(
        JoltCommittedPolynomial::InstructionRa,
        config.ra_layout.instruction(),
    )?;
    let ram = outputs.take_vec(JoltCommittedPolynomial::RamRa, config.ra_layout.ram())?;
    let bytecode = outputs.take_vec(
        JoltCommittedPolynomial::BytecodeRa,
        config.ra_layout.bytecode(),
    )?;

    Ok(JoltCommitmentParts {
        rd_inc,
        ram_inc,
        ra: JoltRaCommitments::new(instruction, ram, bytecode),
        trusted_advice_commitment: outputs.take_optional(
            JoltCommittedPolynomial::TrustedAdvice,
            config.include_trusted_advice,
        )?,
        untrusted_advice_commitment: outputs.take_optional(
            JoltCommittedPolynomial::UntrustedAdvice,
            config.include_untrusted_advice,
        )?,
    })
}

#[cfg(feature = "field-inline")]
fn take_field_inline_commitments<PCS>(
    outputs: &mut CommitmentOutputMap<FieldInlineCommittedPolynomial, PCS>,
) -> Result<FieldInlineCommitments<PCS::Output>, ProverError>
where
    PCS: CommitmentScheme,
{
    let field_rd_inc = outputs.take(FieldInlineCommittedPolynomial::FieldRdInc)?;
    Ok(FieldInlineCommitments::new(FieldRegistersCommitments::new(
        field_rd_inc,
    )))
}

fn final_opening_trace_embedding(
    config: &CommitmentStageConfig,
) -> Result<Option<TracePolynomialEmbedding>, ProverError> {
    match (
        config.final_opening_trace_rows,
        config.final_opening_address_columns,
    ) {
        (Some(trace_rows), Some(address_columns)) => Ok(Some(TracePolynomialEmbedding::new(
            trace_rows,
            address_columns,
            config.trace_polynomial_order,
        ))),
        (None, None) => Ok(None),
        _ => Err(ProverError::InvalidStageRequest {
            reason:
                "Stage 0 final-opening trace embedding requires both trace rows and address columns"
                    .to_owned(),
        }),
    }
}
