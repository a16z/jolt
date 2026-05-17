//! ZK-mode proof helpers.

use crate::{
    proof::{JoltProof, JoltStageProofs, ProofPayload},
    VerifierError,
};

pub trait ProofMode {
    fn is_zk(&self) -> bool;
}

impl<T> ProofMode for &T
where
    T: ProofMode + ?Sized,
{
    fn is_zk(&self) -> bool {
        (*self).is_zk()
    }
}

impl<Commitment, UniSkipProof, SumcheckProof, OpeningProof, OpeningClaims, BlindFoldProof>
    JoltProof<Commitment, UniSkipProof, SumcheckProof, OpeningProof, OpeningClaims, BlindFoldProof>
where
    UniSkipProof: ProofMode,
    SumcheckProof: ProofMode,
{
    pub fn verify_zk_consistency(&self) -> Result<bool, VerifierError> {
        self.stages.verify_zk_consistency()
    }
}

impl<UniSkipProof, SumcheckProof> JoltStageProofs<UniSkipProof, SumcheckProof>
where
    UniSkipProof: ProofMode,
    SumcheckProof: ProofMode,
{
    pub fn verify_zk_consistency(&self) -> Result<bool, VerifierError> {
        let zk_mode = self.stage1_sumcheck_proof.is_zk();

        let consistent = self.stage1_uni_skip_first_round_proof.is_zk() == zk_mode
            && self.stage2_uni_skip_first_round_proof.is_zk() == zk_mode
            && self.stage2_sumcheck_proof.is_zk() == zk_mode
            && self.stage3_sumcheck_proof.is_zk() == zk_mode
            && self.stage4_sumcheck_proof.is_zk() == zk_mode
            && self.stage5_sumcheck_proof.is_zk() == zk_mode
            && self.stage6_sumcheck_proof.is_zk() == zk_mode
            && self.stage7_sumcheck_proof.is_zk() == zk_mode;

        if consistent {
            Ok(zk_mode)
        } else {
            Err(VerifierError::InconsistentProofZkMode)
        }
    }
}

impl<OpeningClaims, BlindFoldProof> ProofPayload<OpeningClaims, BlindFoldProof> {
    pub fn is_zk(&self) -> bool {
        matches!(self, Self::Zk { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct ModeProof(bool);

    impl ProofMode for ModeProof {
        fn is_zk(&self) -> bool {
            self.0
        }
    }

    #[test]
    fn stage_consistency_returns_runtime_mode() {
        assert_eq!(
            stage_proofs(false).verify_zk_consistency().ok(),
            Some(false)
        );
        assert_eq!(stage_proofs(true).verify_zk_consistency().ok(), Some(true));
    }

    #[test]
    fn stage_consistency_rejects_mixed_modes() {
        let mut stages = stage_proofs(false);
        stages.stage5_sumcheck_proof = ModeProof(true);

        assert!(matches!(
            stages.verify_zk_consistency(),
            Err(VerifierError::InconsistentProofZkMode)
        ));
    }

    fn stage_proofs(zk: bool) -> JoltStageProofs<ModeProof, ModeProof> {
        JoltStageProofs {
            stage1_uni_skip_first_round_proof: ModeProof(zk),
            stage1_sumcheck_proof: ModeProof(zk),
            stage2_uni_skip_first_round_proof: ModeProof(zk),
            stage2_sumcheck_proof: ModeProof(zk),
            stage3_sumcheck_proof: ModeProof(zk),
            stage4_sumcheck_proof: ModeProof(zk),
            stage5_sumcheck_proof: ModeProof(zk),
            stage6_sumcheck_proof: ModeProof(zk),
            stage7_sumcheck_proof: ModeProof(zk),
        }
    }
}
