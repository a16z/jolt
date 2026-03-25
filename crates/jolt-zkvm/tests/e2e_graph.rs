//! End-to-end test: compile guest → trace → witness → prove → verify
//! using the graph-driven pipeline.
//!
//! Uses MockCommitmentScheme for fast iteration. Virtual polynomial data
//! is computed on-the-fly from the trace via `CycleRow` — no pre-materialized tables.

use std::collections::HashMap;
use std::sync::Arc;

use jolt_field::{Field, Fr};
use jolt_ir::protocol::{build_jolt_protocol, Symbol};
use jolt_openings::mock::MockCommitmentScheme;
use jolt_verifier::verify::build_symbol_table;
use jolt_zkvm::preprocessing::{interleave_witnesses, preprocess, JoltConfig};
use jolt_zkvm::prover::GraphProverInput;
use jolt_zkvm::witness::generate::generate_witnesses;

type MockPCS = MockCommitmentScheme<Fr>;

fn symbols(
    config: &jolt_verifier::ProverConfig,
    spartan_key: &jolt_spartan::UniformSpartanKey<Fr>,
) -> HashMap<Symbol, usize> {
    let params = config.one_hot_params_from_config();
    build_symbol_table(
        config.log_trace_length(),
        params.log_k_chunk,
        spartan_key.num_row_vars(),
        spartan_key.num_col_vars(),
        params.instruction_d,
        params.bytecode_d,
        params.ram_d,
    )
}

#[test]
#[ignore] // Requires guest ELF compilation
fn graph_driven_muldiv_mock_pcs() {
    // 1. Compile and trace
    let mut program = jolt_host::Program::new("muldiv-guest");
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
    let (_lazy, trace, _memory, _device) = program.trace(&inputs, &[], &[]);

    // 2. Generate committed polynomial witnesses
    let witness_output = generate_witnesses::<Fr>(&trace);
    let config = &witness_output.config;

    // 3. Preprocessing
    let jolt_config = JoltConfig {
        num_cycles: config.trace_length,
    };
    let proving_key = preprocess::<Fr, MockPCS>(&jolt_config, |_| ((), ()));

    // 4. R1CS witness
    let flat_witness =
        interleave_witnesses(&proving_key.spartan_key, &witness_output.cycle_witnesses);

    // 5. Protocol graph
    let graph = build_jolt_protocol(config.to_protocol_config());

    // 6. Symbol table
    let syms = symbols(config, &proving_key.spartan_key);

    // 7. External values
    // raf_scale = 2^{phase3_cycle_rounds} where phase3 = log_T - phase1
    let phase1_rounds = config.rw_config.ram_rw_phase1_num_rounds as usize;
    let log_t = config.log_trace_length();
    let phase3_rounds = log_t.saturating_sub(phase1_rounds);
    let raf_scale = Fr::from_u64(1u64 << phase3_rounds);
    let mut external: HashMap<&str, Fr> = HashMap::new();
    let _ = external.insert("raf_scale", raf_scale);
    let _ = external.insert("neg_init", Fr::from_u64(0)); // no initial RAM state for muldiv

    // 8. Prove — trace provides virtual poly data on-the-fly via TracePolynomials
    let expanded_pcs: Vec<u32> = trace
        .iter()
        .map(|c| witness_output.bytecode.get_pc_for(c) as u32)
        .collect();
    let trace_polys = jolt_witness::TracePolynomials::new(&trace).with_expanded_pcs(expanded_pcs);
    let backend = Arc::new(jolt_cpu::CpuBackend);
    let input = GraphProverInput {
        graph: &graph,
        trace_polys: &trace_polys,
        committed_store: &witness_output.witness_store,
        symbols: &syms,
        external: &external,
        spartan_key: &proving_key.spartan_key,
        flat_witness: &flat_witness,
        pcs_setup: &proving_key.pcs_prover_setup,
        pcs_verifier_setup: proving_key.pcs_verifier_setup.clone(),
        config: config.clone(),
        backend,
    };
    eprintln!("=== prove_from_graph ===");
    let (proof, vk) = jolt_zkvm::prover::prove_from_graph(input).expect("proving should succeed");
    eprintln!("=== proved, {} stage proofs ===", proof.stage_proofs.len());

    // 9. Verify
    let graph2 = build_jolt_protocol(config.to_protocol_config());
    let syms2 = symbols(config, &vk.spartan_key);
    jolt_verifier::verify::verify_from_graph::<Fr, MockPCS>(
        &graph2, &proof, &vk, &syms2, &external,
    )
    .expect("verification should succeed");
}
