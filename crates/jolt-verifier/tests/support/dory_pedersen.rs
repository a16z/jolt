use common::jolt_device::{JoltDevice, MemoryLayout};
use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig};
use jolt_crypto::{Bn254G1, DeriveSetup, Pedersen, PedersenSetup, VectorCommitment};
use jolt_dory::{DoryCommitment, DoryProof, DoryScheme, DoryVerifierSetup};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_openings::{CommitmentScheme, ZkOpeningScheme};
use jolt_poly::Polynomial;
use jolt_program::preprocess::{BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing};
use jolt_sumcheck::{
    ClearProof, ClearSumcheckProof, CommittedOutputClaims, CommittedRound, CommittedSumcheckProof,
    CompressedSumcheckProof, SumcheckProof,
};
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_verifier::{
    proof::JoltStageProofs, verify, JoltProof, JoltVerifierPreprocessing, VerifierError,
};

const DORY_NUM_VARS: usize = 4;
const VC_CAPACITY: usize = 4;

pub type TestVectorCommitment = Pedersen<Bn254G1>;
pub type TestProof = JoltProof<DoryScheme, TestVectorCommitment, (), ()>;
pub type TestPreprocessing = JoltVerifierPreprocessing<DoryScheme, TestVectorCommitment>;

pub struct DoryPedersenVerifierCase {
    pub preprocessing: TestPreprocessing,
    pub public_io: JoltDevice,
    pub proof: TestProof,
    pub zk: bool,
    pub trusted_advice_commitment: Option<DoryCommitment>,
}

impl DoryPedersenVerifierCase {
    pub fn verify(&self) -> Result<(), VerifierError> {
        verify::<Fr, DoryScheme, TestVectorCommitment, Blake2bTranscript, (), ()>(
            &self.preprocessing,
            &self.public_io,
            &self.proof,
            self.trusted_advice_commitment.as_ref(),
            self.zk,
        )
    }
}

pub fn standard_case() -> DoryPedersenVerifierCase {
    let artifacts = dory_artifacts(false);
    DoryPedersenVerifierCase {
        preprocessing: preprocessing(artifacts.pcs_setup.clone(), None),
        public_io: public_io(),
        proof: proof_with_payload(false, Some(()), None, &artifacts),
        zk: false,
        trusted_advice_commitment: None,
    }
}

pub fn zk_case() -> DoryPedersenVerifierCase {
    let artifacts = dory_artifacts(true);
    DoryPedersenVerifierCase {
        preprocessing: preprocessing(
            artifacts.pcs_setup.clone(),
            Some(artifacts.vc_setup.clone()),
        ),
        public_io: public_io(),
        proof: proof_with_payload(true, None, Some(()), &artifacts),
        zk: true,
        trusted_advice_commitment: None,
    }
}

pub fn preprocessing(
    pcs_setup: DoryVerifierSetup,
    vc_setup: Option<PedersenSetup<Bn254G1>>,
) -> TestPreprocessing {
    let memory_layout = memory_layout();
    JoltVerifierPreprocessing::new(
        JoltProgramPreprocessing {
            bytecode: BytecodePreprocessing::default(),
            ram: RAMPreprocessing::default(),
            memory_layout,
            max_padded_trace_length: 16,
        },
        [7; 32],
        pcs_setup,
        vc_setup,
    )
}

pub fn public_io() -> JoltDevice {
    JoltDevice {
        memory_layout: memory_layout(),
        inputs: vec![1, 2, 3],
        outputs: vec![4],
        ..JoltDevice::default()
    }
}

pub fn proof_with_payload(
    is_zk: bool,
    opening_claims: Option<()>,
    blindfold_proof: Option<()>,
    artifacts: &DoryArtifacts,
) -> TestProof {
    JoltProof {
        commitments: vec![artifacts.commitment.clone()],
        stages: stage_proofs(is_zk, &artifacts.vc_setup),
        joint_opening_proof: artifacts.opening_proof.clone(),
        untrusted_advice_commitment: None,
        opening_claims,
        blindfold_proof,
        trace_length: 1,
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

#[derive(Clone)]
pub struct DoryArtifacts {
    pub pcs_setup: DoryVerifierSetup,
    pub vc_setup: PedersenSetup<Bn254G1>,
    pub commitment: DoryCommitment,
    pub opening_proof: DoryProof,
}

fn dory_artifacts(is_zk: bool) -> DoryArtifacts {
    let prover_setup = DoryScheme::setup_prover(DORY_NUM_VARS);
    let pcs_setup = DoryScheme::verifier_setup(&prover_setup);
    let vc_setup = PedersenSetup::<Bn254G1>::derive(&prover_setup, VC_CAPACITY);
    let poly = Polynomial::new(vec![Fr::from_u64(2), Fr::from_u64(7)]);
    let point = vec![Fr::from_u64(3)];
    let eval = poly.evaluate(&point);
    let mut transcript = Blake2bTranscript::new(b"jolt-verifier-dory-pedersen-test");

    let (commitment, opening_proof) = if is_zk {
        let (commitment, hint) = DoryScheme::commit_zk(poly.evaluations(), &prover_setup);
        let (proof, _, _) =
            DoryScheme::open_zk(&poly, &point, eval, &prover_setup, hint, &mut transcript);
        (commitment, proof)
    } else {
        let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup);
        let proof = DoryScheme::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut transcript,
        );
        (commitment, proof)
    };

    DoryArtifacts {
        pcs_setup,
        vc_setup,
        commitment,
        opening_proof,
    }
}

fn memory_layout() -> MemoryLayout {
    MemoryLayout {
        max_input_size: 8,
        max_output_size: 8,
        heap_size: 8,
        ..MemoryLayout::default()
    }
}

fn stage_proofs(
    is_zk: bool,
    vc_setup: &PedersenSetup<Bn254G1>,
) -> JoltStageProofs<Fr, TestVectorCommitment> {
    JoltStageProofs {
        stage1_uni_skip_first_round_proof: uniskip_proof(is_zk, vc_setup),
        stage1_sumcheck_proof: sumcheck_proof(is_zk, vc_setup),
        stage2_uni_skip_first_round_proof: uniskip_proof(is_zk, vc_setup),
        stage2_sumcheck_proof: sumcheck_proof(is_zk, vc_setup),
        stage3_sumcheck_proof: sumcheck_proof(is_zk, vc_setup),
        stage4_sumcheck_proof: sumcheck_proof(is_zk, vc_setup),
        stage5_sumcheck_proof: sumcheck_proof(is_zk, vc_setup),
        stage6_sumcheck_proof: sumcheck_proof(is_zk, vc_setup),
        stage7_sumcheck_proof: sumcheck_proof(is_zk, vc_setup),
    }
}

fn uniskip_proof(is_zk: bool, vc_setup: &PedersenSetup<Bn254G1>) -> SumcheckProof<Fr, Bn254G1> {
    if is_zk {
        SumcheckProof::Committed(committed_sumcheck_proof(vc_setup))
    } else {
        SumcheckProof::Clear(ClearProof::Full(ClearSumcheckProof::default()))
    }
}

fn sumcheck_proof(is_zk: bool, vc_setup: &PedersenSetup<Bn254G1>) -> SumcheckProof<Fr, Bn254G1> {
    if is_zk {
        SumcheckProof::Committed(committed_sumcheck_proof(vc_setup))
    } else {
        SumcheckProof::Clear(ClearProof::Compressed(CompressedSumcheckProof::default()))
    }
}

fn committed_sumcheck_proof(vc_setup: &PedersenSetup<Bn254G1>) -> CommittedSumcheckProof<Bn254G1> {
    CommittedSumcheckProof {
        rounds: vec![CommittedRound {
            commitment: pedersen_commit(vc_setup, &[1, 2, 3], 4),
            degree: 2,
        }],
        output_claims: CommittedOutputClaims {
            commitments: vec![pedersen_commit(vc_setup, &[5, 6], 7)],
        },
    }
}

fn pedersen_commit(setup: &PedersenSetup<Bn254G1>, values: &[u64], blinding: u64) -> Bn254G1 {
    let values: Vec<_> = values.iter().copied().map(Fr::from_u64).collect();
    Pedersen::commit(setup, &values, &Fr::from_u64(blinding))
}
