//! End-to-end test: compile guest → trace → witness → prove → verify
//! using the graph-driven pipeline.
//!
//! Uses MockCommitmentScheme for fast iteration. Can be switched to Dory
//! for production soundness testing.

use std::collections::HashMap;
use std::sync::Arc;

use jolt_field::Fr;
use jolt_ir::protocol::{build_jolt_protocol, ProtocolConfig, Symbol};
use jolt_openings::mock::MockCommitmentScheme;
use jolt_verifier::verify::build_symbol_table;
use jolt_zkvm::preprocessing::{interleave_witnesses, preprocess, JoltConfig};
use jolt_zkvm::prover::{prove_from_graph, GraphProverInput};
use jolt_zkvm::tables::PolynomialTables;
use jolt_zkvm::witness::generate::generate_witnesses;

type MockPCS = MockCommitmentScheme<Fr>;

/// Build a ProtocolConfig from a ProverConfig.
fn protocol_config(config: &jolt_verifier::ProverConfig) -> ProtocolConfig {
    config.to_protocol_config()
}

/// Build the symbol table from config + Spartan key dimensions.
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
fn graph_driven_muldiv_mock_pcs() {
    // 1. Compile and trace the muldiv guest program
    let mut program = jolt_host::Program::new("muldiv-guest");
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
    let (_lazy, trace, _memory, _device) = program.trace(&inputs, &[], &[]);

    // 2. Generate witnesses + polynomial tables
    let witness_output = generate_witnesses::<Fr>(&trace);
    let config = &witness_output.config;
    let tables = PolynomialTables::from_witness(
        &witness_output.witness_store,
        &witness_output.cycle_witnesses,
        &trace,
        config,
    );

    // 3. Preprocessing: Spartan key + PCS setup
    let jolt_config = JoltConfig {
        num_cycles: config.trace_length,
    };
    let proving_key = preprocess::<Fr, MockPCS>(&jolt_config, |_vars| ((), ()));

    // 4. Build interleaved R1CS witness
    let flat_witness =
        interleave_witnesses(&proving_key.spartan_key, &witness_output.cycle_witnesses);

    // 5. Build protocol graph
    let proto_config = protocol_config(config);
    let graph = build_jolt_protocol(proto_config);

    // 6. Build symbol table
    let syms = symbols(config, &proving_key.spartan_key);

    // 7. External values (none needed for muldiv — no RAM init, no advice)
    let external: HashMap<&str, Fr> = HashMap::new();

    // 8. Prove
    let backend = Arc::new(jolt_cpu::CpuBackend);
    let input = GraphProverInput {
        graph: &graph,
        tables: &tables,
        symbols: &syms,
        external: &external,
        spartan_key: &proving_key.spartan_key,
        flat_witness: &flat_witness,
        pcs_setup: &proving_key.pcs_prover_setup,
        pcs_verifier_setup: proving_key.pcs_verifier_setup.clone(),
        config: config.clone(),
        backend,
    };
    eprintln!("=== Starting prove_from_graph (graph has {} stages, {} claims) ===",
        graph.staging.stages.len(), graph.claim_graph.claims.len());
    let (proof, vk) = prove_from_graph::<Fr, MockPCS, _>(input).expect("proving should succeed");
    eprintln!(
        "=== prove_from_graph succeeded, {} stage proofs ===",
        proof.stage_proofs.len()
    );

    // 9. Verify
    let graph2 = build_jolt_protocol(protocol_config(config));
    let syms2 = symbols(config, &vk.spartan_key);
    jolt_verifier::verify::verify_from_graph::<Fr, MockPCS>(
        &graph2, &proof, &vk, &syms2, &external,
    )
    .expect("verification should succeed");
}
