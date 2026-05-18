//! Top-level verifier entry point.

use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::SumcheckProof;

use crate::{proof::JoltProof, VerifierError};

pub fn validate_proof_consistency<PCS, VC, OpeningClaims, ZkProof>(
    proof: &JoltProof<PCS, VC, OpeningClaims, ZkProof>,
    zk: bool,
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    validate_sumcheck_representation(
        &proof.stages.stage1_uni_skip_first_round_proof,
        "stage1_uni_skip_first_round_proof",
        zk,
    )?;
    validate_sumcheck_representation(
        &proof.stages.stage1_sumcheck_proof,
        "stage1_sumcheck_proof",
        zk,
    )?;
    validate_sumcheck_representation(
        &proof.stages.stage2_uni_skip_first_round_proof,
        "stage2_uni_skip_first_round_proof",
        zk,
    )?;
    validate_sumcheck_representation(
        &proof.stages.stage2_sumcheck_proof,
        "stage2_sumcheck_proof",
        zk,
    )?;
    validate_sumcheck_representation(
        &proof.stages.stage3_sumcheck_proof,
        "stage3_sumcheck_proof",
        zk,
    )?;
    validate_sumcheck_representation(
        &proof.stages.stage4_sumcheck_proof,
        "stage4_sumcheck_proof",
        zk,
    )?;
    validate_sumcheck_representation(
        &proof.stages.stage5_sumcheck_proof,
        "stage5_sumcheck_proof",
        zk,
    )?;
    validate_sumcheck_representation(
        &proof.stages.stage6_sumcheck_proof,
        "stage6_sumcheck_proof",
        zk,
    )?;
    validate_sumcheck_representation(
        &proof.stages.stage7_sumcheck_proof,
        "stage7_sumcheck_proof",
        zk,
    )?;

    if zk {
        if proof.opening_claims.is_some() {
            return Err(VerifierError::UnexpectedOpeningClaims);
        }
        if proof.blindfold_proof.is_none() {
            return Err(VerifierError::MissingBlindFoldProof);
        }
    } else {
        if proof.opening_claims.is_none() {
            return Err(VerifierError::MissingOpeningClaims);
        }
        if proof.blindfold_proof.is_some() {
            return Err(VerifierError::UnexpectedBlindFoldProof);
        }
    }
    Ok(())
}

fn validate_sumcheck_representation<F, RoundCommitment>(
    proof: &SumcheckProof<F, RoundCommitment>,
    field: &'static str,
    zk: bool,
) -> Result<(), VerifierError>
where
    F: Field,
{
    if proof.is_committed() == zk {
        return Ok(());
    }

    if zk {
        Err(VerifierError::ExpectedCommittedProof { field })
    } else {
        Err(VerifierError::ExpectedClearProof { field })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::JoltStageProofs;
    use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig};
    use jolt_crypto::{Commitment, VectorCommitment};
    use jolt_field::Fr;
    use jolt_openings::{CommitmentScheme, OpeningsError};
    use jolt_poly::{MultilinearPoly, Polynomial};
    use jolt_sumcheck::{
        ClearProof, ClearSumcheckProof, CommittedSumcheckProof, CompressedSumcheckProof,
    };
    use jolt_transcript::Transcript;

    #[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    struct TestPcs;

    #[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    struct TestVectorCommitment;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    struct TestCommitment;

    impl Commitment for TestPcs {
        type Output = TestCommitment;
    }

    impl CommitmentScheme for TestPcs {
        type Field = Fr;
        type Proof = ();
        type ProverSetup = ();
        type VerifierSetup = ();
        type Polynomial = Polynomial<Fr>;
        type OpeningHint = ();
        type SetupParams = ();

        fn setup(_params: Self::SetupParams) -> (Self::ProverSetup, Self::VerifierSetup) {
            ((), ())
        }

        fn verifier_setup(_prover_setup: &Self::ProverSetup) -> Self::VerifierSetup {}

        fn commit<P: MultilinearPoly<Self::Field> + ?Sized>(
            _poly: &P,
            _setup: &Self::ProverSetup,
        ) -> (Self::Output, Self::OpeningHint) {
            (TestCommitment, ())
        }

        fn open(
            _poly: &Self::Polynomial,
            _point: &[Self::Field],
            _eval: Self::Field,
            _setup: &Self::ProverSetup,
            _hint: Option<Self::OpeningHint>,
            _transcript: &mut impl Transcript<Challenge = Self::Field>,
        ) -> Self::Proof {
        }

        fn verify(
            _commitment: &Self::Output,
            _point: &[Self::Field],
            _eval: Self::Field,
            _proof: &Self::Proof,
            _setup: &Self::VerifierSetup,
            _transcript: &mut impl Transcript<Challenge = Self::Field>,
        ) -> Result<(), OpeningsError> {
            Ok(())
        }

        fn bind_opening_inputs(
            _transcript: &mut impl Transcript<Challenge = Self::Field>,
            _point: &[Self::Field],
            _eval: &Self::Field,
        ) {
        }
    }

    impl Commitment for TestVectorCommitment {
        type Output = TestCommitment;
    }

    impl VectorCommitment for TestVectorCommitment {
        type Field = Fr;
        type Setup = ();

        fn capacity(_setup: &Self::Setup) -> usize {
            usize::MAX
        }

        fn commit(
            _setup: &Self::Setup,
            _values: &[Self::Field],
            _blinding: &Self::Field,
        ) -> Self::Output {
            TestCommitment
        }

        fn verify(
            _setup: &Self::Setup,
            _commitment: &Self::Output,
            _values: &[Self::Field],
            _blinding: &Self::Field,
        ) -> bool {
            true
        }
    }

    impl jolt_transcript::AppendToTranscript for TestCommitment {
        fn append_to_transcript<T: Transcript>(&self, _transcript: &mut T) {}
    }

    type TestProof = JoltProof<TestPcs, TestVectorCommitment, (), ()>;

    #[test]
    fn proof_wrapper_uses_modular_trait_bounds() {
        fn assert_proof_traits<T>()
        where
            T: Clone
                + std::fmt::Debug
                + PartialEq
                + Eq
                + Send
                + Sync
                + 'static
                + serde::Serialize
                + serde::de::DeserializeOwned,
        {
        }

        assert_proof_traits::<TestProof>();
    }

    #[test]
    fn accepts_standard_proof_consistency() {
        let proof = proof_with_zk(false, Some(()), None);

        assert!(validate_proof_consistency(&proof, false).is_ok());
    }

    #[test]
    fn accepts_zk_proof_consistency() {
        let proof = proof_with_zk(true, None, Some(()));

        assert!(validate_proof_consistency(&proof, true).is_ok());
    }

    #[test]
    fn rejects_wrong_stage_representation() {
        let mut proof = proof_with_zk(false, Some(()), None);
        proof.stages.stage5_sumcheck_proof =
            SumcheckProof::Committed(CommittedSumcheckProof::default());

        assert!(matches!(
            validate_proof_consistency(&proof, false),
            Err(VerifierError::ExpectedClearProof {
                field: "stage5_sumcheck_proof",
            })
        ));
    }

    #[test]
    fn rejects_wrong_verifier_zk_flag() {
        let proof = proof_with_zk(false, Some(()), None);

        assert!(matches!(
            validate_proof_consistency(&proof, true),
            Err(VerifierError::ExpectedCommittedProof {
                field: "stage1_uni_skip_first_round_proof",
            })
        ));
    }

    #[test]
    fn checks_payload_for_selected_zk_flag() {
        assert!(matches!(
            validate_proof_consistency(&proof_with_zk(false, None, None), false),
            Err(VerifierError::MissingOpeningClaims)
        ));
        assert!(matches!(
            validate_proof_consistency(&proof_with_zk(false, Some(()), Some(())), false),
            Err(VerifierError::UnexpectedBlindFoldProof)
        ));
        assert!(matches!(
            validate_proof_consistency(&proof_with_zk(true, None, None), true),
            Err(VerifierError::MissingBlindFoldProof)
        ));
        assert!(matches!(
            validate_proof_consistency(&proof_with_zk(true, Some(()), Some(())), true),
            Err(VerifierError::UnexpectedOpeningClaims)
        ));
    }

    fn proof_with_zk(
        is_zk: bool,
        opening_claims: Option<()>,
        blindfold_proof: Option<()>,
    ) -> TestProof {
        JoltProof {
            commitments: Vec::new(),
            stages: stage_proofs(is_zk),
            joint_opening_proof: (),
            untrusted_advice_commitment: None,
            opening_claims,
            blindfold_proof,
            trace_length: 0,
            ram_K: 1,
            rw_config: JoltReadWriteConfig {
                ram_rw_phase1_num_rounds: 0,
                ram_rw_phase2_num_rounds: 0,
                registers_rw_phase1_num_rounds: 0,
                registers_rw_phase2_num_rounds: 0,
            },
            one_hot_config: JoltOneHotConfig {
                log_k_chunk: 0,
                lookups_ra_virtual_log_k_chunk: 0,
            },
        }
    }

    fn stage_proofs(is_zk: bool) -> JoltStageProofs<Fr, TestVectorCommitment> {
        JoltStageProofs {
            stage1_uni_skip_first_round_proof: uniskip_proof(is_zk),
            stage1_sumcheck_proof: sumcheck_proof(is_zk),
            stage2_uni_skip_first_round_proof: uniskip_proof(is_zk),
            stage2_sumcheck_proof: sumcheck_proof(is_zk),
            stage3_sumcheck_proof: sumcheck_proof(is_zk),
            stage4_sumcheck_proof: sumcheck_proof(is_zk),
            stage5_sumcheck_proof: sumcheck_proof(is_zk),
            stage6_sumcheck_proof: sumcheck_proof(is_zk),
            stage7_sumcheck_proof: sumcheck_proof(is_zk),
        }
    }

    fn uniskip_proof(is_zk: bool) -> SumcheckProof<Fr, TestCommitment> {
        if is_zk {
            SumcheckProof::Committed(CommittedSumcheckProof::default())
        } else {
            SumcheckProof::Clear(ClearProof::Full(ClearSumcheckProof::default()))
        }
    }

    fn sumcheck_proof(is_zk: bool) -> SumcheckProof<Fr, TestCommitment> {
        if is_zk {
            SumcheckProof::Committed(CommittedSumcheckProof::default())
        } else {
            SumcheckProof::Clear(ClearProof::Compressed(CompressedSumcheckProof::default()))
        }
    }
}
