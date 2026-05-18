//! Verifier-owned proof model types.

use crate::compat::config::{OneHotConfig, ReadWriteConfig};
use crate::VerifierError;
use jolt_sumcheck::SumcheckProof as ModularSumcheckProof;

#[expect(non_snake_case, reason = "Matches current jolt-core proof field name.")]
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct JoltProof<
    Commitment,
    UniSkipProof,
    SumcheckProof,
    OpeningProof,
    OpeningClaims = (),
    BlindFoldProof = (),
> {
    pub commitments: Vec<Commitment>,
    pub stages: JoltStageProofs<UniSkipProof, SumcheckProof>,
    pub joint_opening_proof: OpeningProof,
    pub untrusted_advice_commitment: Option<Commitment>,
    pub opening_claims: Option<OpeningClaims>,
    pub blindfold_proof: Option<BlindFoldProof>,
    pub trace_length: usize,
    pub ram_K: usize,
    pub rw_config: ReadWriteConfig,
    pub one_hot_config: OneHotConfig,
    pub dory_layout: DoryLayout,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct JoltStageProofs<UniSkipProof, SumcheckProof> {
    pub stage1_uni_skip_first_round_proof: UniSkipProof,
    pub stage1_sumcheck_proof: SumcheckProof,
    pub stage2_uni_skip_first_round_proof: UniSkipProof,
    pub stage2_sumcheck_proof: SumcheckProof,
    pub stage3_sumcheck_proof: SumcheckProof,
    pub stage4_sumcheck_proof: SumcheckProof,
    pub stage5_sumcheck_proof: SumcheckProof,
    pub stage6_sumcheck_proof: SumcheckProof,
    pub stage7_sumcheck_proof: SumcheckProof,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DoryLayout {
    #[default]
    CycleMajor,
    AddressMajor,
}

impl DoryLayout {
    pub fn address_cycle_to_index(
        self,
        address: usize,
        cycle: usize,
        num_addresses: usize,
        num_cycles: usize,
    ) -> usize {
        match self {
            Self::CycleMajor => address * num_cycles + cycle,
            Self::AddressMajor => cycle * num_addresses + address,
        }
    }

    pub fn index_to_address_cycle(
        self,
        index: usize,
        num_addresses: usize,
        num_cycles: usize,
    ) -> (usize, usize) {
        match self {
            Self::CycleMajor => (index / num_cycles, index % num_cycles),
            Self::AddressMajor => (index % num_addresses, index / num_addresses),
        }
    }
}

impl From<DoryLayout> for u8 {
    fn from(layout: DoryLayout) -> Self {
        match layout {
            DoryLayout::CycleMajor => 0,
            DoryLayout::AddressMajor => 1,
        }
    }
}

impl TryFrom<u8> for DoryLayout {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::CycleMajor),
            1 => Ok(Self::AddressMajor),
            _ => Err(()),
        }
    }
}

impl<Commitment, F, RoundCommitment, OpeningProof, OpeningClaims, BlindFoldProof>
    JoltProof<
        Commitment,
        ModularSumcheckProof<F, RoundCommitment>,
        ModularSumcheckProof<F, RoundCommitment>,
        OpeningProof,
        OpeningClaims,
        BlindFoldProof,
    >
where
    F: jolt_field::Field,
{
    pub fn validate_zk_flag(&self, zk: bool) -> Result<(), VerifierError> {
        if zk {
            self.validate_committed_proof()
        } else {
            self.validate_clear_proof()
        }
    }

    pub fn validate_clear_proof(&self) -> Result<(), VerifierError> {
        self.stages.validate_clear_sumcheck_proofs()?;
        if self.opening_claims.is_none() {
            return Err(VerifierError::MissingOpeningClaims);
        }
        if self.blindfold_proof.is_some() {
            return Err(VerifierError::UnexpectedBlindFoldProof);
        }
        Ok(())
    }

    pub fn validate_committed_proof(&self) -> Result<(), VerifierError> {
        self.stages.validate_committed_sumcheck_proofs()?;
        if self.opening_claims.is_some() {
            return Err(VerifierError::UnexpectedOpeningClaims);
        }
        if self.blindfold_proof.is_none() {
            return Err(VerifierError::MissingBlindFoldProof);
        }
        Ok(())
    }
}

impl<F, RoundCommitment>
    JoltStageProofs<
        ModularSumcheckProof<F, RoundCommitment>,
        ModularSumcheckProof<F, RoundCommitment>,
    >
where
    F: jolt_field::Field,
{
    pub fn validate_clear_sumcheck_proofs(&self) -> Result<(), VerifierError> {
        ensure_clear(
            &self.stage1_uni_skip_first_round_proof,
            "stage1_uni_skip_first_round_proof",
        )?;
        ensure_clear(&self.stage1_sumcheck_proof, "stage1_sumcheck_proof")?;
        ensure_clear(
            &self.stage2_uni_skip_first_round_proof,
            "stage2_uni_skip_first_round_proof",
        )?;
        ensure_clear(&self.stage2_sumcheck_proof, "stage2_sumcheck_proof")?;
        ensure_clear(&self.stage3_sumcheck_proof, "stage3_sumcheck_proof")?;
        ensure_clear(&self.stage4_sumcheck_proof, "stage4_sumcheck_proof")?;
        ensure_clear(&self.stage5_sumcheck_proof, "stage5_sumcheck_proof")?;
        ensure_clear(&self.stage6_sumcheck_proof, "stage6_sumcheck_proof")?;
        ensure_clear(&self.stage7_sumcheck_proof, "stage7_sumcheck_proof")?;
        Ok(())
    }

    pub fn validate_committed_sumcheck_proofs(&self) -> Result<(), VerifierError> {
        ensure_committed(
            &self.stage1_uni_skip_first_round_proof,
            "stage1_uni_skip_first_round_proof",
        )?;
        ensure_committed(&self.stage1_sumcheck_proof, "stage1_sumcheck_proof")?;
        ensure_committed(
            &self.stage2_uni_skip_first_round_proof,
            "stage2_uni_skip_first_round_proof",
        )?;
        ensure_committed(&self.stage2_sumcheck_proof, "stage2_sumcheck_proof")?;
        ensure_committed(&self.stage3_sumcheck_proof, "stage3_sumcheck_proof")?;
        ensure_committed(&self.stage4_sumcheck_proof, "stage4_sumcheck_proof")?;
        ensure_committed(&self.stage5_sumcheck_proof, "stage5_sumcheck_proof")?;
        ensure_committed(&self.stage6_sumcheck_proof, "stage6_sumcheck_proof")?;
        ensure_committed(&self.stage7_sumcheck_proof, "stage7_sumcheck_proof")?;
        Ok(())
    }
}

fn ensure_clear<F, C>(
    proof: &ModularSumcheckProof<F, C>,
    field: &'static str,
) -> Result<(), VerifierError>
where
    F: jolt_field::Field,
{
    match proof {
        ModularSumcheckProof::Clear(_) => Ok(()),
        ModularSumcheckProof::Committed(_) => Err(VerifierError::ExpectedClearProof { field }),
    }
}

fn ensure_committed<F, C>(
    proof: &ModularSumcheckProof<F, C>,
    field: &'static str,
) -> Result<(), VerifierError>
where
    F: jolt_field::Field,
{
    match proof {
        ModularSumcheckProof::Clear(_) => Err(VerifierError::ExpectedCommittedProof { field }),
        ModularSumcheckProof::Committed(_) => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use jolt_sumcheck::{CommittedSumcheckProof, CompressedSumcheckProof};

    type StageProof = ModularSumcheckProof<Fr, ()>;

    #[test]
    fn stage_validation_accepts_expected_form() {
        assert!(stage_proofs(clear_proof())
            .validate_clear_sumcheck_proofs()
            .is_ok());
        assert!(stage_proofs(committed_proof())
            .validate_committed_sumcheck_proofs()
            .is_ok());
    }

    #[test]
    fn stage_validation_rejects_unexpected_form() {
        let mut stages = stage_proofs(clear_proof());
        stages.stage5_sumcheck_proof = committed_proof();

        assert!(matches!(
            stages.validate_clear_sumcheck_proofs(),
            Err(VerifierError::ExpectedClearProof {
                field: "stage5_sumcheck_proof",
            })
        ));
    }

    #[test]
    fn proof_validation_checks_clear_fields() {
        let proof = proof_with_fields(clear_proof(), Some(()), None);

        assert!(proof.validate_zk_flag(false).is_ok());
    }

    #[test]
    fn proof_validation_checks_committed_fields() {
        let proof = proof_with_fields(committed_proof(), None, Some(()));

        assert!(proof.validate_zk_flag(true).is_ok());
    }

    #[test]
    fn clear_proof_requires_opening_claims() {
        let proof = proof_with_fields(clear_proof(), None, None);

        assert!(matches!(
            proof.validate_zk_flag(false),
            Err(VerifierError::MissingOpeningClaims)
        ));
    }

    #[test]
    fn committed_proof_requires_blindfold_proof() {
        let proof = proof_with_fields(committed_proof(), None, None);

        assert!(matches!(
            proof.validate_zk_flag(true),
            Err(VerifierError::MissingBlindFoldProof)
        ));
    }

    fn proof_with_fields(
        proof_form: StageProof,
        opening_claims: Option<()>,
        blindfold_proof: Option<()>,
    ) -> JoltProof<(), StageProof, StageProof, (), (), ()> {
        JoltProof {
            commitments: Vec::new(),
            stages: stage_proofs(proof_form),
            joint_opening_proof: (),
            untrusted_advice_commitment: None,
            opening_claims,
            blindfold_proof,
            trace_length: 0,
            ram_K: 1,
            rw_config: ReadWriteConfig {
                ram_rw_phase1_num_rounds: 0,
                ram_rw_phase2_num_rounds: 0,
                registers_rw_phase1_num_rounds: 0,
                registers_rw_phase2_num_rounds: 0,
            },
            one_hot_config: OneHotConfig {
                log_k_chunk: 0,
                lookups_ra_virtual_log_k_chunk: 0,
            },
            dory_layout: DoryLayout::CycleMajor,
        }
    }

    fn stage_proofs(proof_form: StageProof) -> JoltStageProofs<StageProof, StageProof> {
        JoltStageProofs {
            stage1_uni_skip_first_round_proof: proof_form.clone(),
            stage1_sumcheck_proof: proof_form.clone(),
            stage2_uni_skip_first_round_proof: proof_form.clone(),
            stage2_sumcheck_proof: proof_form.clone(),
            stage3_sumcheck_proof: proof_form.clone(),
            stage4_sumcheck_proof: proof_form.clone(),
            stage5_sumcheck_proof: proof_form.clone(),
            stage6_sumcheck_proof: proof_form.clone(),
            stage7_sumcheck_proof: proof_form,
        }
    }

    fn clear_proof() -> StageProof {
        ModularSumcheckProof::Clear(CompressedSumcheckProof::default())
    }

    fn committed_proof() -> StageProof {
        ModularSumcheckProof::Committed(CommittedSumcheckProof::default())
    }
}
