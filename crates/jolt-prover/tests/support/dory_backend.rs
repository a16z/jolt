use common::jolt_device::JoltDevice;
use jolt_crypto::{Bn254G1, Pedersen};
use jolt_dory::DoryScheme;
use jolt_field::Fr;
#[cfg(feature = "cuda")]
use jolt_kernels::cuda::CudaDoryScheme;
use jolt_prover::dory::stages::stage0::TrustedAdviceCommitment;
use jolt_prover::dory::{self, DoryProverPreprocessing};
use jolt_prover::{JoltBackend, ProverConfig, ProverError};
use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;
use jolt_verifier::proof::JoltProof;
use jolt_witness::TraceBackend;

pub type Proof = JoltProof<DoryScheme, Pedersen<Bn254G1>>;

pub trait DoryBackend: 'static {
    fn prove(
        self,
        witness: &TraceBackend,
        preprocessing: &DoryProverPreprocessing,
        config: &ProverConfig,
        trusted_advice: Option<&TrustedAdviceCommitment<DoryScheme>>,
        public_io: &JoltDevice,
    ) -> Result<Proof, ProverError<Fr>>;
}

impl DoryBackend for JoltBackend<Fr, DoryScheme> {
    fn prove(
        self,
        witness: &TraceBackend,
        preprocessing: &DoryProverPreprocessing,
        config: &ProverConfig,
        trusted_advice: Option<&TrustedAdviceCommitment<DoryScheme>>,
        public_io: &JoltDevice,
    ) -> Result<Proof, ProverError<Fr>> {
        dory::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
            self.with_witness(witness),
            preprocessing,
            config,
            trusted_advice,
            public_io,
        )
    }
}

#[cfg(feature = "cuda")]
impl DoryBackend for JoltBackend<Fr, CudaDoryScheme> {
    #[expect(clippy::expect_used, reason = "backend fixtures fail loudly")]
    fn prove(
        self,
        witness: &TraceBackend,
        preprocessing: &DoryProverPreprocessing,
        config: &ProverConfig,
        trusted_advice: Option<&TrustedAdviceCommitment<DoryScheme>>,
        public_io: &JoltDevice,
    ) -> Result<Proof, ProverError<Fr>> {
        use jolt_prover::JoltProverPreprocessing;

        assert!(
            jolt_kernels::cuda::device_available(),
            "CUDA backend tests require a usable CUDA device"
        );
        assert!(preprocessing.committed_program.is_none());
        let cuda_preprocessing = JoltProverPreprocessing {
            verifier: CudaDoryScheme::adopt_verifier_preprocessing(preprocessing.verifier.clone())
                .expect("CUDA verifier preprocessing"),
            pcs_setup: preprocessing.pcs_setup.clone(),
            committed_program: None,
        };
        let cuda_advice = trusted_advice.map(|advice| TrustedAdviceCommitment {
            commitment: advice.commitment.clone(),
            hint: advice.hint.clone(),
        });
        let proof = dory::prove::<Fr, CudaDoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
            self.with_witness(witness),
            &cuda_preprocessing,
            config,
            cuda_advice.as_ref(),
            public_io,
        )?;
        // Decode the CUDA wire output as a CPU proof so the existing verifier
        // and equality checks also enforce interoperability between PCS types.
        let bytes = bincode::serde::encode_to_vec(&proof, bincode::config::standard())
            .expect("serialize CUDA proof");
        let (proof, consumed) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                .expect("decode CUDA proof as a CPU proof");
        assert_eq!(consumed, bytes.len());
        Ok(proof)
    }
}
