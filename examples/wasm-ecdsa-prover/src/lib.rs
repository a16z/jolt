use ark_bn254::Fr;
use ark_serialize::CanonicalDeserialize;
use common::jolt_device::{JoltDevice, MemoryConfig};
use jolt_core::{
    poly::commitment::dory::DoryCommitmentScheme,
    zkvm::{prover::JoltProverPreprocessing, verifier::JoltVerifierPreprocessing, Serializable},
};
use wasm_bindgen::prelude::*;

pub use wasm_bindgen_rayon::init_thread_pool;

#[no_mangle]
#[cfg(not(target_arch = "wasm32"))]
pub static mut _HEAP_PTR: u8 = 0;

type ProverPreprocessing = JoltProverPreprocessing<Fr, DoryCommitmentScheme>;
type VerifierPreprocessing = JoltVerifierPreprocessing<Fr, DoryCommitmentScheme>;

#[wasm_bindgen(start)]
pub fn wasm_main() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub fn init_inlines() -> Result<(), JsValue> {
    jolt_inlines_secp256k1::init_inlines().map_err(|e| JsValue::from_str(&e))
}

#[wasm_bindgen]
pub struct WasmProver {
    preprocessing: ProverPreprocessing,
    elf_bytes: Vec<u8>,
}

#[wasm_bindgen]
impl WasmProver {
    #[wasm_bindgen(constructor)]
    pub fn new(preprocessing_bytes: &[u8], elf_bytes: &[u8]) -> Result<WasmProver, JsValue> {
        use jolt_core::poly::commitment::dory::ArkworksProverSetup;
        use jolt_core::zkvm::verifier::JoltSharedPreprocessing;
        use std::io::Cursor;

        let mut cursor = Cursor::new(preprocessing_bytes);

        let generators = ArkworksProverSetup::deserialize_with_mode(
            &mut cursor,
            ark_serialize::Compress::No,
            ark_serialize::Validate::No,
        )
        .map_err(|e| JsValue::from_str(&format!("ProverSetup deserialize error: {e}")))?;

        let shared = JoltSharedPreprocessing::deserialize_with_mode(
            &mut cursor,
            ark_serialize::Compress::No,
            ark_serialize::Validate::No,
        )
        .map_err(|e| JsValue::from_str(&format!("SharedPreprocessing deserialize error: {e}")))?;

        let preprocessing = ProverPreprocessing { generators, shared };

        Ok(Self {
            preprocessing,
            elf_bytes: elf_bytes.to_vec(),
        })
    }

    pub fn prove_ecdsa(
        &self,
        z: &[u64],
        r: &[u64],
        s: &[u64],
        q: &[u64],
    ) -> Result<ProveResult, JsValue> {
        use jolt_core::zkvm::prover::JoltCpuProver;

        // Convert slices to fixed arrays (postcard serializes slices with length prefix, arrays without)
        let z: [u64; 4] = z.try_into().map_err(|_| JsValue::from_str("z must have 4 elements"))?;
        let r: [u64; 4] = r.try_into().map_err(|_| JsValue::from_str("r must have 4 elements"))?;
        let s: [u64; 4] = s.try_into().map_err(|_| JsValue::from_str("s must have 4 elements"))?;
        let q: [u64; 8] = q.try_into().map_err(|_| JsValue::from_str("q must have 8 elements"))?;

        let inputs = postcard::to_allocvec(&(z, r, s, q))
            .map_err(|e| JsValue::from_str(&format!("Input serialization error: {e}")))?;

        let memory_config = MemoryConfig {
            max_untrusted_advice_size: self
                .preprocessing
                .shared
                .memory_layout
                .max_untrusted_advice_size,
            max_trusted_advice_size: self
                .preprocessing
                .shared
                .memory_layout
                .max_trusted_advice_size,
            max_input_size: self.preprocessing.shared.memory_layout.max_input_size,
            max_output_size: self.preprocessing.shared.memory_layout.max_output_size,
            stack_size: self.preprocessing.shared.memory_layout.stack_size,
            heap_size: self.preprocessing.shared.memory_layout.heap_size,
            program_size: Some(self.preprocessing.shared.memory_layout.program_size),
        };

        let (lazy_trace, trace, final_memory, program_io) =
            jolt_core::guest::program::trace(&self.elf_bytes, None, &inputs, &[], &[], &memory_config);

        let prover: JoltCpuProver<'_, Fr, DoryCommitmentScheme, _> = JoltCpuProver::gen_from_trace(
            &self.preprocessing,
            lazy_trace,
            trace,
            program_io.clone(),
            None,
            None,
            final_memory,
        );

        let (proof, _) = prover.prove();

        let proof_bytes = proof
            .serialize_to_bytes()
            .map_err(|e| JsValue::from_str(&format!("Proof serialization error: {e}")))?;

        let program_io_bytes = program_io
            .serialize_to_bytes()
            .map_err(|e| JsValue::from_str(&format!("Program IO serialization error: {e}")))?;

        Ok(ProveResult {
            proof_bytes,
            program_io_bytes,
        })
    }
}

#[wasm_bindgen]
pub struct ProveResult {
    proof_bytes: Vec<u8>,
    program_io_bytes: Vec<u8>,
}

#[wasm_bindgen]
impl ProveResult {
    #[wasm_bindgen(getter)]
    pub fn proof(&self) -> Vec<u8> {
        self.proof_bytes.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn program_io(&self) -> Vec<u8> {
        self.program_io_bytes.clone()
    }
}

#[wasm_bindgen]
pub struct WasmVerifier {
    preprocessing: VerifierPreprocessing,
}

#[wasm_bindgen]
impl WasmVerifier {
    #[wasm_bindgen(constructor)]
    pub fn new(preprocessing_bytes: &[u8]) -> Result<WasmVerifier, JsValue> {
        use jolt_core::poly::commitment::dory::ArkworksVerifierSetup;
        use jolt_core::zkvm::verifier::JoltSharedPreprocessing;
        use std::io::Cursor;

        let mut cursor = Cursor::new(preprocessing_bytes);

        let generators = ArkworksVerifierSetup::deserialize_with_mode(
            &mut cursor,
            ark_serialize::Compress::No,
            ark_serialize::Validate::No,
        )
        .map_err(|e| JsValue::from_str(&format!("VerifierSetup deserialize error: {e}")))?;

        let shared = JoltSharedPreprocessing::deserialize_with_mode(
            &mut cursor,
            ark_serialize::Compress::No,
            ark_serialize::Validate::No,
        )
        .map_err(|e| JsValue::from_str(&format!("SharedPreprocessing deserialize error: {e}")))?;

        let preprocessing = VerifierPreprocessing { generators, shared };

        Ok(Self { preprocessing })
    }

    pub fn verify(&self, proof_bytes: &[u8], program_io_bytes: &[u8]) -> Result<bool, JsValue> {
        use jolt_core::zkvm::{proof_serialization::JoltProof, RV64IMACVerifier};

        let proof: JoltProof<Fr, DoryCommitmentScheme, _> =
            JoltProof::deserialize_from_bytes(proof_bytes)
                .map_err(|e| JsValue::from_str(&format!("Proof deserialize error: {e}")))?;

        let program_io: JoltDevice = JoltDevice::deserialize_from_bytes(program_io_bytes)
            .map_err(|e| JsValue::from_str(&format!("Program IO deserialize error: {e}")))?;

        let verifier = RV64IMACVerifier::new(&self.preprocessing, proof, program_io, None, None)
            .map_err(|e| JsValue::from_str(&format!("Verifier init error: {e}")))?;

        verifier
            .verify()
            .map(|_| true)
            .map_err(|e| JsValue::from_str(&format!("Verification failed: {e}")))
    }
}
