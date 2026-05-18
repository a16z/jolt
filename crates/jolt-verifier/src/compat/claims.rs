//! Compatibility opening-claim conversion.

#[cfg(any(feature = "jolt-core-compat", test))]
use std::collections::BTreeMap;

#[cfg(any(feature = "jolt-core-compat", test))]
use crate::compat::ids as legacy;
use crate::proof::JoltProof;
use jolt_claims::protocols::jolt as native;
use jolt_crypto::VectorCommitment;
#[cfg(any(feature = "jolt-core-compat", test))]
use jolt_field::Field;
use jolt_openings::CommitmentScheme;

#[cfg(any(feature = "jolt-core-compat", test))]
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct LegacyOpeningClaims<F: Field>(pub BTreeMap<legacy::OpeningId, F>);

#[cfg(any(feature = "jolt-core-compat", test))]
pub(crate) fn native_opening_claims_from_legacy<F: Field>(
    claims: LegacyOpeningClaims<F>,
) -> Vec<(native::JoltOpeningId, F)> {
    claims
        .0
        .into_iter()
        .map(|(id, opening_claim)| (opening_id(id), opening_claim))
        .collect()
}

#[doc(hidden)]
pub fn attach_opening_claims<PCS, VC, ZkProof>(
    proof: &mut JoltProof<PCS, VC, ZkProof>,
    claims: impl IntoIterator<Item = (native::JoltOpeningId, PCS::Field)>,
) where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    proof.opening_claims = Some(claims.into_iter().collect());
}

#[doc(hidden)]
pub fn clear_opening_claims<PCS, VC, ZkProof>(proof: &mut JoltProof<PCS, VC, ZkProof>)
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    proof.opening_claims = None;
}

#[doc(hidden)]
pub fn attach_empty_opening_claims<PCS, VC, ZkProof>(proof: &mut JoltProof<PCS, VC, ZkProof>)
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    proof.opening_claims = Some(Vec::new());
}

#[cfg(any(feature = "jolt-core-compat", test))]
fn opening_id(id: legacy::OpeningId) -> native::JoltOpeningId {
    match id {
        legacy::OpeningId::Polynomial(polynomial, stage) => {
            native::JoltOpeningId::polynomial(polynomial_id(polynomial), stage_id(stage))
        }
        legacy::OpeningId::UntrustedAdvice(stage) => {
            native::JoltOpeningId::untrusted_advice(stage_id(stage))
        }
        legacy::OpeningId::TrustedAdvice(stage) => {
            native::JoltOpeningId::trusted_advice(stage_id(stage))
        }
    }
}

#[cfg(any(feature = "jolt-core-compat", test))]
fn polynomial_id(id: legacy::PolynomialId) -> native::JoltPolynomialId {
    match id {
        legacy::PolynomialId::Committed(polynomial) => {
            native::JoltPolynomialId::Committed(committed_polynomial(polynomial))
        }
        legacy::PolynomialId::Virtual(polynomial) => {
            native::JoltPolynomialId::Virtual(virtual_polynomial(polynomial))
        }
    }
}

#[cfg(any(feature = "jolt-core-compat", test))]
fn committed_polynomial(
    polynomial: legacy::CommittedPolynomial,
) -> native::JoltCommittedPolynomial {
    match polynomial {
        legacy::CommittedPolynomial::RdInc => native::JoltCommittedPolynomial::RdInc,
        legacy::CommittedPolynomial::RamInc => native::JoltCommittedPolynomial::RamInc,
        legacy::CommittedPolynomial::InstructionRa(index) => {
            native::JoltCommittedPolynomial::InstructionRa(index)
        }
        legacy::CommittedPolynomial::BytecodeRa(index) => {
            native::JoltCommittedPolynomial::BytecodeRa(index)
        }
        legacy::CommittedPolynomial::RamRa(index) => native::JoltCommittedPolynomial::RamRa(index),
        legacy::CommittedPolynomial::TrustedAdvice => {
            native::JoltCommittedPolynomial::TrustedAdvice
        }
        legacy::CommittedPolynomial::UntrustedAdvice => {
            native::JoltCommittedPolynomial::UntrustedAdvice
        }
    }
}

#[cfg(any(feature = "jolt-core-compat", test))]
fn virtual_polynomial(polynomial: legacy::VirtualPolynomial) -> native::JoltVirtualPolynomial {
    match polynomial {
        legacy::VirtualPolynomial::PC => native::JoltVirtualPolynomial::PC,
        legacy::VirtualPolynomial::UnexpandedPC => native::JoltVirtualPolynomial::UnexpandedPC,
        legacy::VirtualPolynomial::NextPC => native::JoltVirtualPolynomial::NextPC,
        legacy::VirtualPolynomial::NextUnexpandedPC => {
            native::JoltVirtualPolynomial::NextUnexpandedPC
        }
        legacy::VirtualPolynomial::NextIsNoop => native::JoltVirtualPolynomial::NextIsNoop,
        legacy::VirtualPolynomial::NextIsVirtual => native::JoltVirtualPolynomial::NextIsVirtual,
        legacy::VirtualPolynomial::NextIsFirstInSequence => {
            native::JoltVirtualPolynomial::NextIsFirstInSequence
        }
        legacy::VirtualPolynomial::LeftLookupOperand => {
            native::JoltVirtualPolynomial::LeftLookupOperand
        }
        legacy::VirtualPolynomial::RightLookupOperand => {
            native::JoltVirtualPolynomial::RightLookupOperand
        }
        legacy::VirtualPolynomial::LeftInstructionInput => {
            native::JoltVirtualPolynomial::LeftInstructionInput
        }
        legacy::VirtualPolynomial::RightInstructionInput => {
            native::JoltVirtualPolynomial::RightInstructionInput
        }
        legacy::VirtualPolynomial::Product => native::JoltVirtualPolynomial::Product,
        legacy::VirtualPolynomial::ShouldJump => native::JoltVirtualPolynomial::ShouldJump,
        legacy::VirtualPolynomial::ShouldBranch => native::JoltVirtualPolynomial::ShouldBranch,
        legacy::VirtualPolynomial::Rd => native::JoltVirtualPolynomial::Rd,
        legacy::VirtualPolynomial::Imm => native::JoltVirtualPolynomial::Imm,
        legacy::VirtualPolynomial::Rs1Value => native::JoltVirtualPolynomial::Rs1Value,
        legacy::VirtualPolynomial::Rs2Value => native::JoltVirtualPolynomial::Rs2Value,
        legacy::VirtualPolynomial::RdWriteValue => native::JoltVirtualPolynomial::RdWriteValue,
        legacy::VirtualPolynomial::Rs1Ra => native::JoltVirtualPolynomial::Rs1Ra,
        legacy::VirtualPolynomial::Rs2Ra => native::JoltVirtualPolynomial::Rs2Ra,
        legacy::VirtualPolynomial::RdWa => native::JoltVirtualPolynomial::RdWa,
        legacy::VirtualPolynomial::LookupOutput => native::JoltVirtualPolynomial::LookupOutput,
        legacy::VirtualPolynomial::InstructionRaf => native::JoltVirtualPolynomial::InstructionRaf,
        legacy::VirtualPolynomial::InstructionRafFlag => {
            native::JoltVirtualPolynomial::InstructionRafFlag
        }
        legacy::VirtualPolynomial::InstructionRa(index) => {
            native::JoltVirtualPolynomial::InstructionRa(index)
        }
        legacy::VirtualPolynomial::RegistersVal => native::JoltVirtualPolynomial::RegistersVal,
        legacy::VirtualPolynomial::RamAddress => native::JoltVirtualPolynomial::RamAddress,
        legacy::VirtualPolynomial::RamRa => native::JoltVirtualPolynomial::RamRa,
        legacy::VirtualPolynomial::RamReadValue => native::JoltVirtualPolynomial::RamReadValue,
        legacy::VirtualPolynomial::RamWriteValue => native::JoltVirtualPolynomial::RamWriteValue,
        legacy::VirtualPolynomial::RamVal => native::JoltVirtualPolynomial::RamVal,
        legacy::VirtualPolynomial::RamValInit => native::JoltVirtualPolynomial::RamValInit,
        legacy::VirtualPolynomial::RamValFinal => native::JoltVirtualPolynomial::RamValFinal,
        legacy::VirtualPolynomial::RamHammingWeight => {
            native::JoltVirtualPolynomial::RamHammingWeight
        }
        legacy::VirtualPolynomial::UnivariateSkip => native::JoltVirtualPolynomial::UnivariateSkip,
        legacy::VirtualPolynomial::OpFlags(flag) => native::JoltVirtualPolynomial::OpFlags(flag),
        legacy::VirtualPolynomial::InstructionFlags(flag) => {
            native::JoltVirtualPolynomial::InstructionFlags(flag)
        }
        legacy::VirtualPolynomial::LookupTableFlag(index) => {
            native::JoltVirtualPolynomial::LookupTableFlag(index)
        }
    }
}

#[cfg(any(feature = "jolt-core-compat", test))]
fn stage_id(id: legacy::SumcheckId) -> native::JoltStageId {
    match id {
        legacy::SumcheckId::SpartanOuter => native::JoltStageId::SpartanOuter,
        legacy::SumcheckId::SpartanProductVirtualization => {
            native::JoltStageId::SpartanProductVirtualization
        }
        legacy::SumcheckId::SpartanShift => native::JoltStageId::SpartanShift,
        legacy::SumcheckId::InstructionClaimReduction => {
            native::JoltStageId::InstructionClaimReduction
        }
        legacy::SumcheckId::InstructionInputVirtualization => {
            native::JoltStageId::InstructionInputVirtualization
        }
        legacy::SumcheckId::InstructionReadRaf => native::JoltStageId::InstructionReadRaf,
        legacy::SumcheckId::InstructionRaVirtualization => {
            native::JoltStageId::InstructionRaVirtualization
        }
        legacy::SumcheckId::RamReadWriteChecking => native::JoltStageId::RamReadWriteChecking,
        legacy::SumcheckId::RamRafEvaluation => native::JoltStageId::RamRafEvaluation,
        legacy::SumcheckId::RamOutputCheck => native::JoltStageId::RamOutputCheck,
        legacy::SumcheckId::RamValCheck => native::JoltStageId::RamValCheck,
        legacy::SumcheckId::RamRaClaimReduction => native::JoltStageId::RamRaClaimReduction,
        legacy::SumcheckId::RamHammingBooleanity => native::JoltStageId::RamHammingBooleanity,
        legacy::SumcheckId::RamRaVirtualization => native::JoltStageId::RamRaVirtualization,
        legacy::SumcheckId::RegistersClaimReduction => native::JoltStageId::RegistersClaimReduction,
        legacy::SumcheckId::RegistersReadWriteChecking => {
            native::JoltStageId::RegistersReadWriteChecking
        }
        legacy::SumcheckId::RegistersValEvaluation => native::JoltStageId::RegistersValEvaluation,
        legacy::SumcheckId::BytecodeReadRaf => native::JoltStageId::BytecodeReadRaf,
        legacy::SumcheckId::Booleanity => native::JoltStageId::Booleanity,
        legacy::SumcheckId::AdviceClaimReductionCyclePhase => {
            native::JoltStageId::AdviceClaimReductionCyclePhase
        }
        legacy::SumcheckId::AdviceClaimReduction => native::JoltStageId::AdviceClaimReduction,
        legacy::SumcheckId::IncClaimReduction => native::JoltStageId::IncClaimReduction,
        legacy::SumcheckId::HammingWeightClaimReduction => {
            native::JoltStageId::HammingWeightClaimReduction
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    #[test]
    fn converts_legacy_opening_ids_to_native_opening_ids() -> Result<(), crate::VerifierError> {
        let legacy_claims = LegacyOpeningClaims(BTreeMap::from([
            (
                legacy::OpeningId::committed(
                    legacy::CommittedPolynomial::RamInc,
                    legacy::SumcheckId::RamReadWriteChecking,
                ),
                Fr::from_u64(3),
            ),
            (
                legacy::OpeningId::virt(
                    legacy::VirtualPolynomial::RamVal,
                    legacy::SumcheckId::RamReadWriteChecking,
                ),
                Fr::from_u64(5),
            ),
            (
                legacy::OpeningId::TrustedAdvice(legacy::SumcheckId::AdviceClaimReduction),
                Fr::from_u64(7),
            ),
            (
                legacy::OpeningId::UntrustedAdvice(legacy::SumcheckId::AdviceClaimReduction),
                Fr::from_u64(11),
            ),
        ]));

        let native_claims = native_opening_claims_from_legacy(legacy_claims);

        assert_eq!(
            opening_claim(
                &native_claims,
                native::JoltOpeningId::committed(
                    native::JoltCommittedPolynomial::RamInc,
                    native::JoltStageId::RamReadWriteChecking,
                )
            ),
            Some(Fr::from_u64(3))
        );
        assert_eq!(
            opening_claim(
                &native_claims,
                native::JoltOpeningId::virtual_polynomial(
                    native::JoltVirtualPolynomial::RamVal,
                    native::JoltStageId::RamReadWriteChecking,
                )
            ),
            Some(Fr::from_u64(5))
        );
        assert_eq!(
            opening_claim(
                &native_claims,
                native::JoltOpeningId::trusted_advice(native::JoltStageId::AdviceClaimReduction,)
            ),
            Some(Fr::from_u64(7))
        );
        assert_eq!(
            opening_claim(
                &native_claims,
                native::JoltOpeningId::untrusted_advice(native::JoltStageId::AdviceClaimReduction,)
            ),
            Some(Fr::from_u64(11))
        );
        Ok(())
    }

    fn opening_claim(
        claims: &[(native::JoltOpeningId, Fr)],
        id: native::JoltOpeningId,
    ) -> Option<Fr> {
        claims
            .iter()
            .find_map(|&(claim_id, opening_claim)| (claim_id == id).then_some(opening_claim))
    }
}
