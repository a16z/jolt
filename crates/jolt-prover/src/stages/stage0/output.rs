use std::collections::BTreeMap;

use jolt_backends::CommitmentResult;
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::FieldInlineCommittedPolynomial;
use jolt_claims::protocols::jolt::JoltCommittedPolynomial;
use jolt_openings::CommitmentScheme;
#[cfg(feature = "field-inline")]
use jolt_verifier::proof::{FieldInlineCommitments, FieldRegistersCommitments};
use jolt_verifier::proof::{JoltCommitments, JoltRaCommitments};
#[cfg(feature = "field-inline")]
use jolt_witness::protocols::jolt_vm::field_inline::FieldInlineNamespace;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;

use crate::builder::{VerifierCommitmentBuilder, VerifierComponentSpec};
use crate::ProverError;

use super::input::CommitmentStageConfig;

#[derive(Clone)]
pub struct CommitmentStageOutput<PCS: CommitmentScheme> {
    pub commitments: JoltCommitments<PCS::Output>,
    pub trusted_advice_commitment: Option<PCS::Output>,
    pub untrusted_advice_commitment: Option<PCS::Output>,
    pub prover_state: CommitmentStageProverState<PCS::OpeningHint>,
}

#[derive(Clone)]
pub struct CommitmentStageProverState<OpeningHint> {
    pub opening_hints: BTreeMap<JoltCommittedPolynomial, OpeningHint>,
    #[cfg(feature = "field-inline")]
    pub field_inline_opening_hints: BTreeMap<FieldInlineCommittedPolynomial, OpeningHint>,
}

impl<PCS: CommitmentScheme> CommitmentStageOutput<PCS> {
    pub fn from_backend_result(
        result: CommitmentResult<JoltVmNamespace, PCS>,
        #[cfg(feature = "field-inline")] field_inline_result: CommitmentResult<
            FieldInlineNamespace,
            PCS,
        >,
        config: CommitmentStageConfig,
    ) -> Result<Self, ProverError> {
        let jolt = VerifierCommitmentBuilder::<JoltCommittedPolynomial, PCS>::from_backend_result(
            "Jolt", result,
        )?
        .build(JoltCommitmentsSpec { config })?;

        #[cfg(feature = "field-inline")]
        let field_inline =
            VerifierCommitmentBuilder::<FieldInlineCommittedPolynomial, PCS>::from_backend_result(
                "field-inline",
                field_inline_result,
            )?
            .build(FieldInlineCommitmentsSpec)?;

        #[cfg(not(feature = "field-inline"))]
        let commitments = JoltCommitments::new(
            jolt.verifier.rd_inc,
            jolt.verifier.ram_inc,
            jolt.verifier.ra,
        );
        #[cfg(feature = "field-inline")]
        let commitments = JoltCommitments::new(
            jolt.verifier.rd_inc,
            jolt.verifier.ram_inc,
            jolt.verifier.ra,
            field_inline.verifier,
        );

        Ok(Self {
            commitments,
            trusted_advice_commitment: jolt.verifier.trusted_advice_commitment,
            untrusted_advice_commitment: jolt.verifier.untrusted_advice_commitment,
            prover_state: CommitmentStageProverState {
                opening_hints: jolt.prover_state,
                #[cfg(feature = "field-inline")]
                field_inline_opening_hints: field_inline.prover_state,
            },
        })
    }
}

struct JoltCommitmentsSpec {
    config: CommitmentStageConfig,
}

struct JoltCommitmentParts<C> {
    rd_inc: C,
    ram_inc: C,
    ra: JoltRaCommitments<C>,
    trusted_advice_commitment: Option<C>,
    untrusted_advice_commitment: Option<C>,
}

impl<PCS> VerifierComponentSpec<VerifierCommitmentBuilder<JoltCommittedPolynomial, PCS>>
    for JoltCommitmentsSpec
where
    PCS: CommitmentScheme,
{
    type VerifierComponent = JoltCommitmentParts<PCS::Output>;

    fn assemble(
        self,
        builder: &mut VerifierCommitmentBuilder<JoltCommittedPolynomial, PCS>,
    ) -> Result<Self::VerifierComponent, ProverError> {
        let rd_inc = builder.take(JoltCommittedPolynomial::RdInc)?;
        let ram_inc = builder.take(JoltCommittedPolynomial::RamInc)?;
        let instruction = builder.take_vec(
            JoltCommittedPolynomial::InstructionRa,
            self.config.ra_layout.instruction(),
        )?;
        let ram = builder.take_vec(JoltCommittedPolynomial::RamRa, self.config.ra_layout.ram())?;
        let bytecode = builder.take_vec(
            JoltCommittedPolynomial::BytecodeRa,
            self.config.ra_layout.bytecode(),
        )?;

        Ok(JoltCommitmentParts {
            rd_inc,
            ram_inc,
            ra: JoltRaCommitments::new(instruction, ram, bytecode),
            trusted_advice_commitment: builder.take_optional(
                JoltCommittedPolynomial::TrustedAdvice,
                self.config.include_trusted_advice,
            )?,
            untrusted_advice_commitment: builder.take_optional(
                JoltCommittedPolynomial::UntrustedAdvice,
                self.config.include_untrusted_advice,
            )?,
        })
    }
}

#[cfg(feature = "field-inline")]
struct FieldInlineCommitmentsSpec;

#[cfg(feature = "field-inline")]
impl<PCS> VerifierComponentSpec<VerifierCommitmentBuilder<FieldInlineCommittedPolynomial, PCS>>
    for FieldInlineCommitmentsSpec
where
    PCS: CommitmentScheme,
{
    type VerifierComponent = FieldInlineCommitments<PCS::Output>;

    fn assemble(
        self,
        builder: &mut VerifierCommitmentBuilder<FieldInlineCommittedPolynomial, PCS>,
    ) -> Result<Self::VerifierComponent, ProverError> {
        let field_rd_inc = builder.take(FieldInlineCommittedPolynomial::FieldRdInc)?;
        Ok(FieldInlineCommitments::new(FieldRegistersCommitments::new(
            field_rd_inc,
        )))
    }
}
