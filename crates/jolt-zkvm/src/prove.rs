//! Top-level prover API.
//!
//! Two entry points:
//! - [`prove`] — highest-level: takes a RISC-V execution trace via
//!   [`TraceData`] and builds witness + R1CS internally before proving
//! - [`prove_with_buffers`] — lower-level: takes a pre-built
//!   [`BufferProvider`] for non-RISC-V or custom protocols

use common::jolt_device::MemoryLayout;
use jolt_compiler::PolynomialSpec;
use jolt_compute::{BufferProvider, ComputeBackend, Executable};
use jolt_field::Field;
use jolt_host::{extract_trace, BytecodePreprocessing, CycleRow};
use jolt_openings::AdditivelyHomomorphic;
use jolt_r1cs::{constraints::rv64, R1csKey, R1csProvider};
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::ProverConfig;
use jolt_witness::{PolynomialConfig, PolynomialId, Polynomials};

use crate::buffers::ProverBuffers;
use crate::runtime::execute;

/// Total instruction lookup index bit width (RV64: XLEN × 2 = 128).
const LOG_K_INSTRUCTION: usize = 128;

/// Execution trace data needed to build witness polynomials and R1CS.
pub struct TraceData<'a, C> {
    /// RISC-V execution trace (one entry per cycle).
    pub trace: &'a [C],
    /// Preprocessed bytecode table (for PC mapping).
    pub bytecode: &'a BytecodePreprocessing,
    /// Memory address layout (for RAM address remapping).
    pub memory_layout: &'a MemoryLayout,
}

/// Execute the full proving pipeline from a RISC-V execution trace.
///
/// Internally converts the trace into witness polynomials and R1CS data,
/// then deterministically executes the compiled schedule — compute ops,
/// PCS ops, and orchestration ops — returning a complete
/// [`JoltProof`](jolt_verifier::JoltProof).
///
/// # Panics
///
/// Panics if `config` is invalid (non-power-of-two trace length, etc.).
pub fn prove<C, B, F, T, PCS>(
    executable: &Executable<PolynomialId, B, F>,
    data: &TraceData<'_, C>,
    backend: &B,
    pcs_setup: &PCS::ProverSetup,
    transcript: &mut T,
    config: ProverConfig,
) -> jolt_verifier::JoltProof<F, PCS>
where
    C: CycleRow,
    B: ComputeBackend,
    F: Field,
    T: Transcript<Challenge = F>,
    PCS: AdditivelyHomomorphic<Field = F>,
    PCS::Output: AppendToTranscript,
{
    if let Err(e) = config.validate() {
        panic!("invalid ProverConfig: {e}");
    }

    let size = config.trace_length;
    let one_hot = config.one_hot_params_from_config();

    let poly_config = PolynomialConfig::new(
        one_hot.log_k_chunk,
        LOG_K_INSTRUCTION,
        config.bytecode_k.next_power_of_two().trailing_zeros() as usize,
        config.ram_k.trailing_zeros() as usize,
    );

    // -- 1. Single-pass extraction: trace → witness inputs + R1CS witness --
    let matrices = rv64::rv64_constraints::<F>();
    let r1cs_key = R1csKey::new(matrices, size);

    let (inputs, r1cs_witness) = extract_trace::<C, F>(
        data.trace,
        size,
        data.bytecode,
        data.memory_layout,
        r1cs_key.num_vars_padded,
    );

    // -- 2. Build witness polynomials --
    let mut polys = Polynomials::<F>::new(poly_config);
    polys.push(&inputs);
    polys.finish();

    // Advice polynomials (zero-filled when no advice tape is provided).
    let _ = polys.insert(PolynomialId::UntrustedAdvice, vec![F::zero(); size]);
    let _ = polys.insert(PolynomialId::TrustedAdvice, vec![F::zero(); size]);

    // -- 3. Assemble buffers and execute --
    let r1cs = R1csProvider::new(&r1cs_key, &r1cs_witness);
    let mut provider = ProverBuffers::new(&mut polys, r1cs);

    execute::<PolynomialId, B, F, T, PCS>(
        executable,
        &mut provider,
        backend,
        pcs_setup,
        transcript,
        config,
    )
}

/// Execute the proving pipeline with a pre-built buffer provider.
///
/// Lower-level entry point for non-RISC-V protocols or testing.
/// The caller is responsible for building witness polynomials and
/// R1CS data; this function just validates config and executes the schedule.
pub fn prove_with_buffers<P, B, F, T, PCS>(
    executable: &Executable<P, B, F>,
    provider: &mut impl BufferProvider<P, B, F>,
    backend: &B,
    pcs_setup: &PCS::ProverSetup,
    transcript: &mut T,
    config: ProverConfig,
) -> jolt_verifier::JoltProof<F, PCS>
where
    P: PolynomialSpec,
    B: ComputeBackend,
    F: Field,
    T: Transcript<Challenge = F>,
    PCS: AdditivelyHomomorphic<Field = F>,
    PCS::Output: AppendToTranscript,
{
    if let Err(e) = config.validate() {
        panic!("invalid ProverConfig: {e}");
    }
    execute::<P, B, F, T, PCS>(executable, provider, backend, pcs_setup, transcript, config)
}
