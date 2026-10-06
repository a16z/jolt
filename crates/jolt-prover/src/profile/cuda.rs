use std::sync::Arc;
use std::time::Duration;

use common::jolt_device::JoltDevice;
use jolt_crypto::{Bn254G1, Pedersen};
use jolt_dory::DoryScheme;
use jolt_field::Fr;
use jolt_kernels::cuda::CudaDoryScheme;
use jolt_verifier::JoltVerifierPreprocessing;

use super::measure_prove;
use crate::{JoltBackend, JoltProverPreprocessing, ProverConfig};

pub(super) fn prove_measured<W>(
    verifier_preprocessing: JoltVerifierPreprocessing<DoryScheme, Pedersen<Bn254G1>>,
    total_vars: usize,
    config: &ProverConfig,
    witness: Arc<W>,
    public_io: &JoltDevice,
) -> (Duration, usize)
where
    W: jolt_witness::JoltWitnessPlane<Fr> + 'static,
{
    let prover_preprocessing = JoltProverPreprocessing::<CudaDoryScheme, Pedersen<Bn254G1>> {
        verifier: CudaDoryScheme::adopt_verifier_preprocessing(verifier_preprocessing)
            .expect("the CUDA scheme adopts the verifier preprocessing"),
        pcs_setup: CudaDoryScheme::setup_prover(total_vars),
        committed_program: None,
    };
    measure_prove(
        &JoltBackend::<Fr, CudaDoryScheme>::cuda(),
        &prover_preprocessing,
        config,
        witness,
        public_io,
    )
}
