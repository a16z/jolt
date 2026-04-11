//! End-to-end prove → verify integration test.
//!
//! Exercises the full pipeline with a toy protocol:
//!   Protocol IR → compile → link → prove → verify
//!
//! The test validates that the four-step API works end-to-end:
//! 1. Compile and link the protocol
//! 2. Build witness data
//! 3. `prove()` — one call, handles everything
//! 4. `verify()` — checks the proof
#![allow(unused_results)]

use jolt_compiler::{
    compile, CompileParams, Density, Expr, Objective, PolyKind, Protocol, SolverConfig,
};
use jolt_compute::link;
use jolt_cpu::CpuBackend;
use jolt_field::{Field, Fr};
use jolt_openings::mock::MockCommitmentScheme;
use jolt_r1cs::{ConstraintMatrices, R1csKey, R1csSource};
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_verifier::{
    verify, JoltVerifyingKey, OneHotConfig, ProverConfig, ReadWriteConfig, TRANSCRIPT_LABEL,
};
use jolt_witness::derived::DerivedSource;
use jolt_witness::preprocessed::PreprocessedSource;
use jolt_witness::provider::ProverData;
use jolt_witness::{CycleInput, PolynomialConfig, PolynomialId, Polynomials};
use jolt_zkvm::prove::prove;
use num_traits::Zero;

type MockPCS = MockCommitmentScheme<Fr>;

/// Build a toy protocol: single sumcheck proving Σ a(x)·b(x) = 0 (zero-check).
///
/// Two committed polynomials `a` and `b`, one sumcheck vertex with
/// composition `a * b`, zero input sum.
fn build_protocol() -> Protocol {
    let mut p = Protocol::new();
    let log_n = p.dim("log_n");

    let a = p.poly("a", &[log_n], PolyKind::Committed);
    let b = p.poly("b", &[log_n], PolyKind::Committed);

    // Sumcheck: Σ_x a(x) · b(x) = 0
    let claims = p.sumcheck(
        Expr::from(a) * Expr::from(b),
        Expr::from(0i64),
        &[log_n],
        Density::Dense,
    );

    // Evaluate both polys at the sumcheck challenge point
    p.evaluate(a, claims[0]);
    p.evaluate(b, claims[1]);

    p
}

/// Minimal R1CS for routing completeness (not exercised by this protocol).
fn dummy_r1cs(size: usize) -> (R1csKey<Fr>, Vec<Fr>) {
    let matrices = ConstraintMatrices::new(
        1,
        2,
        vec![vec![(0, Fr::from_u64(1))]],
        vec![vec![(0, Fr::from_u64(1))]],
        vec![vec![(1, Fr::from_u64(1))]],
    );
    let key = R1csKey::new(matrices, size);
    let witness = vec![Fr::zero(); size * key.num_vars_padded];
    (key, witness)
}

/// Full prove → verify round-trip.
///
/// 1. Compile + link the protocol
/// 2. Build witness (all-zero cycles satisfy Σ a·b = 0)
/// 3. `prove()` — single call, owns commitment + execution + opening + assembly
/// 4. `verify()` — checks the proof
#[test]
fn prove_verify_roundtrip() {
    let num_vars = 2;
    let size = 1usize << num_vars;

    // -- 1. Compile + link --
    let protocol = build_protocol();
    let params = CompileParams {
        dim_sizes: vec![num_vars as u64],
        field_size_bytes: 32,
        pcs_proof_size: 1,
    };
    let config = SolverConfig {
        proof_size: Objective::Minimize,
        peak_memory: Objective::Ignore,
        prover_time: Objective::Ignore,
    };
    let poly_ids = [PolynomialId::RdInc, PolynomialId::RamInc];
    let module =
        compile(&protocol, &params, &config, &poly_ids).expect("compilation should succeed");

    let backend = CpuBackend;
    let executable = link(module, &backend);

    // -- 2. Build witness --
    let poly_config = PolynomialConfig::new(4, 8, 4, 4);
    let mut polys = Polynomials::<Fr>::new(poly_config);
    polys.push(&[CycleInput::PADDING; 4]);
    polys.finish();

    let (r1cs_key, r1cs_witness) = dummy_r1cs(size);
    let r1cs = R1csSource::new(&r1cs_key, &r1cs_witness);
    let derived = DerivedSource::new(&r1cs_witness, size, r1cs_key.num_vars_padded);
    let preprocessed = PreprocessedSource::new();
    let mut provider = ProverData::new(&mut polys, r1cs, derived, preprocessed);

    let ram_k = size; // Must be power-of-two with log₂(ram_k) ≥ rw_config.ram_rw_phase2_num_rounds
    let ram_log_k = ram_k.trailing_zeros() as usize;
    let prover_config = ProverConfig {
        trace_length: size,
        ram_k,
        bytecode_k: size,
        one_hot_config: OneHotConfig::new(num_vars),
        rw_config: ReadWriteConfig::new(num_vars, ram_log_k),
        memory_start: 0x8000_0000,
        memory_end: 0x8001_0000,
        entry_address: 0x8000_0000,
        io_hash: [0u8; 32],
        max_input_size: 0,
        max_output_size: 0,
        heap_size: 0,
        inputs: Vec::new(),
        outputs: Vec::new(),
        panic: false,
        ram_lowest_address: 0x7FFF_0000,
        input_word_offset: 0,
        output_word_offset: 0,
        panic_word_offset: 0,
        termination_word_offset: 0,
    };

    // -- 3. Prove --
    let mut transcript = Blake2bTranscript::<Fr>::new(TRANSCRIPT_LABEL);
    let proof = prove::<_, _, _, MockPCS>(
        &executable,
        &mut provider,
        &backend,
        &(),
        &mut transcript,
        prover_config,
        None,
        None,
    );

    // -- 4. Verify --
    let vk = JoltVerifyingKey::<Fr, MockPCS>::new(&executable.module, (), r1cs_key);
    verify(&vk, &proof, &[0u8; 32]).expect("proof should verify");
}
