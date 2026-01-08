//! Transpile Jolt verifier stages 1-6 to Gnark circuit
//!
//! Uses TranspilableVerifier with symbolic proof and MleOpeningAccumulator
//! to generate a Gnark circuit for stages 1-6 of the Jolt verifier.

use ark_serialize::CanonicalDeserialize;
use gnark_transpiler::{
    symbolize_proof, AstCommitmentScheme, MleOpeningAccumulator,
    PoseidonAstTranscript, MemoizedCodeGen, sanitize_go_name,
};
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::transpilable_verifier::{JoltVerifierPreprocessing, TranspilableVerifier};
use jolt_core::zkvm::RV64IMACProof;
use common::jolt_device::JoltDevice;
use zklean_extractor::mle_ast::{enable_constraint_mode, take_constraints as take_assertions, MleAst};
use std::collections::HashMap;

fn main() {
    println!("=== Transpiling Jolt Verifier Stages 1-6 to Gnark ===\n");

    // Load proof
    let proof_path = "/tmp/fib_proof.bin";
    println!("Loading proof from: {}", proof_path);
    let proof_bytes = std::fs::read(proof_path).expect("Failed to read proof file");
    let real_proof: RV64IMACProof =
        CanonicalDeserialize::deserialize_compressed(&proof_bytes[..])
            .expect("Failed to deserialize proof");
    println!("  trace_length: {}", real_proof.trace_length);
    println!("  commitments: {}", real_proof.commitments.len());

    // Load io_device
    let io_device_path = "/tmp/fib_io_device.bin";
    println!("\nLoading io_device from: {}", io_device_path);
    let io_device_bytes = std::fs::read(io_device_path).expect("Failed to read io_device file");
    let io_device: JoltDevice = CanonicalDeserialize::deserialize_compressed(&io_device_bytes[..])
        .expect("Failed to deserialize io_device");
    println!("  inputs: {} bytes", io_device.inputs.len());
    println!("  outputs: {} bytes", io_device.outputs.len());

    // Load preprocessing (Dory version - matches jolt-sdk)
    let preprocessing_path = "/tmp/jolt_verifier_preprocessing.dat";
    println!("\nLoading preprocessing from: {}", preprocessing_path);
    let preprocessing_bytes =
        std::fs::read(preprocessing_path).expect("Failed to read preprocessing file");
    let real_preprocessing: JoltVerifierPreprocessing<ark_bn254::Fr, DoryCommitmentScheme> =
        CanonicalDeserialize::deserialize_compressed(&preprocessing_bytes[..])
            .expect("Failed to deserialize preprocessing");
    println!("  memory_layout: {:?}", real_preprocessing.memory_layout);

    // Convert preprocessing to AstCommitmentScheme version
    // (only generators change, bytecode/ram/memory_layout stay the same)
    let symbolic_preprocessing: JoltVerifierPreprocessing<MleAst, AstCommitmentScheme> =
        JoltVerifierPreprocessing {
            generators: gnark_transpiler::ast_commitment_scheme::AstVerifierSetup,
            bytecode: real_preprocessing.bytecode,
            ram: real_preprocessing.ram,
            memory_layout: real_preprocessing.memory_layout,
        };

    // Symbolize the proof
    println!("\n=== Symbolizing Proof ===");
    let (symbolic_proof, mut accumulator, var_alloc) = symbolize_proof(&real_proof);
    println!("  Total symbolic variables: {}", var_alloc.next_idx());

    // Create transcript
    let transcript: PoseidonAstTranscript = Transcript::new(b"Jolt");

    // Create TranspilableVerifier with symbolic types
    println!("\n=== Creating TranspilableVerifier ===");
    let verifier = TranspilableVerifier::<
        MleAst,
        AstCommitmentScheme,
        PoseidonAstTranscript,
        MleOpeningAccumulator,
    >::new_with_accumulator(
        &symbolic_preprocessing,
        symbolic_proof,
        io_device,
        None, // trusted_advice_commitment
        transcript,
        accumulator,
    );

    // Enable assertion mode so MleAst comparisons register equality checks
    enable_constraint_mode();

    // Run verification (stages 1-6)
    println!("\n=== Running Symbolic Verification (Stages 1-6) ===");
    match verifier.verify() {
        Ok(()) => println!("  Verification completed successfully"),
        Err(e) => {
            println!("  Verification error: {:?}", e);
            return;
        }
    }

    // Collect accumulated assertions (equality checks that become api.AssertIsEqual calls)
    let assertions = take_assertions();
    println!("\n=== Accumulated Assertions ===");
    println!("  Total assertions: {}", assertions.len());

    // Debug: analyze each assertion to find problematic ones (constant vs constant)
    use zklean_extractor::mle_ast::{get_node, Node, Atom, Edge};

    fn is_constant(edge: &Edge) -> bool {
        match edge {
            Edge::Atom(Atom::Scalar(_)) => true,
            Edge::Atom(Atom::Var(_)) => false,
            Edge::Atom(Atom::NamedVar(_)) => false,
            Edge::NodeRef(id) => is_node_constant(*id),
        }
    }

    fn is_node_constant(node_id: usize) -> bool {
        let node = get_node(node_id);
        match node {
            Node::Atom(Atom::Scalar(_)) => true,
            Node::Atom(Atom::Var(_)) => false,
            Node::Atom(Atom::NamedVar(_)) => false,
            Node::Neg(e) => is_constant(&e),
            Node::Inv(e) => is_constant(&e),
            Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) | Node::Div(a, b) => {
                is_constant(&a) && is_constant(&b)
            }
            Node::Poseidon(a, b, c) => is_constant(&a) && is_constant(&b) && is_constant(&c),
            Node::Keccak256(e) | Node::ByteReverse(e) | Node::Truncate128Reverse(e)
            | Node::Truncate128(e) | Node::MulTwoPow192(e) => is_constant(&e),
        }
    }

    fn describe_node(node_id: usize, depth: usize) -> String {
        if depth > 5 {
            return "...".to_string();
        }
        let node = get_node(node_id);
        let indent = "  ".repeat(depth);
        match node {
            Node::Atom(Atom::Scalar(limbs)) => {
                if limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0 {
                    format!("Scalar({})", limbs[0])
                } else {
                    format!("Scalar(large)")
                }
            }
            Node::Atom(Atom::Var(idx)) => format!("Var({})", idx),
            Node::Atom(Atom::NamedVar(idx)) => format!("NamedVar({})", idx),
            Node::Neg(e) => format!("Neg({})", describe_edge(&e, depth + 1)),
            Node::Inv(e) => format!("Inv({})", describe_edge(&e, depth + 1)),
            Node::Add(a, b) => format!("Add({}, {})", describe_edge(&a, depth + 1), describe_edge(&b, depth + 1)),
            Node::Sub(a, b) => format!("Sub({}, {})", describe_edge(&a, depth + 1), describe_edge(&b, depth + 1)),
            Node::Mul(a, b) => format!("Mul({}, {})", describe_edge(&a, depth + 1), describe_edge(&b, depth + 1)),
            Node::Div(a, b) => format!("Div({}, {})", describe_edge(&a, depth + 1), describe_edge(&b, depth + 1)),
            Node::Poseidon(a, b, c) => format!("Poseidon(...)"),
            Node::Keccak256(e) => format!("Keccak256({})", describe_edge(&e, depth + 1)),
            Node::ByteReverse(e) => format!("ByteReverse({})", describe_edge(&e, depth + 1)),
            Node::Truncate128Reverse(e) => format!("Truncate128Reverse({})", describe_edge(&e, depth + 1)),
            Node::Truncate128(e) => format!("Truncate128({})", describe_edge(&e, depth + 1)),
            Node::MulTwoPow192(e) => format!("MulTwoPow192({})", describe_edge(&e, depth + 1)),
        }
    }

    fn describe_edge(edge: &Edge, depth: usize) -> String {
        match edge {
            Edge::Atom(Atom::Scalar(limbs)) => {
                if limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0 {
                    format!("{}", limbs[0])
                } else {
                    "large".to_string()
                }
            }
            Edge::Atom(Atom::Var(idx)) => format!("Var({})", idx),
            Edge::Atom(Atom::NamedVar(idx)) => format!("NamedVar({})", idx),
            Edge::NodeRef(id) => describe_node(*id, depth),
        }
    }

    // Check each assertion for constant-vs-constant issues
    println!("\n=== Analyzing Assertions for Constant Issues ===");
    let mut problematic_count = 0;
    for (i, assertion) in assertions.iter().enumerate() {
        let root_id = assertion.root();
        if is_node_constant(root_id) {
            problematic_count += 1;
            if problematic_count <= 10 {
                println!("  [PROBLEMATIC] Assertion {}: entirely constant!", i);
                println!("    Structure: {}", describe_node(root_id, 0));
            }
        }
    }
    if problematic_count > 10 {
        println!("  ... and {} more problematic assertions", problematic_count - 10);
    }
    println!("  Total problematic assertions: {}", problematic_count);

    // Count multiplications by constant 0 in each assertion
    fn count_mul_by_zero(node_id: usize) -> usize {
        let node = get_node(node_id);
        match node {
            Node::Atom(_) => 0,
            Node::Neg(e) | Node::Inv(e) | Node::Keccak256(e) | Node::ByteReverse(e)
            | Node::Truncate128Reverse(e) | Node::Truncate128(e) | Node::MulTwoPow192(e) => {
                count_mul_by_zero_edge(&e)
            }
            Node::Add(a, b) | Node::Sub(a, b) | Node::Div(a, b) => {
                count_mul_by_zero_edge(&a) + count_mul_by_zero_edge(&b)
            }
            Node::Mul(a, b) => {
                let is_zero_mul = match (&a, &b) {
                    (_, Edge::Atom(Atom::Scalar(limbs))) | (Edge::Atom(Atom::Scalar(limbs)), _) => {
                        limbs[0] == 0 && limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0
                    }
                    _ => false,
                };
                let count = if is_zero_mul { 1 } else { 0 };
                count + count_mul_by_zero_edge(&a) + count_mul_by_zero_edge(&b)
            }
            Node::Poseidon(a, b, c) => {
                count_mul_by_zero_edge(&a) + count_mul_by_zero_edge(&b) + count_mul_by_zero_edge(&c)
            }
        }
    }

    fn count_mul_by_zero_edge(edge: &Edge) -> usize {
        match edge {
            Edge::Atom(_) => 0,
            Edge::NodeRef(id) => count_mul_by_zero(*id),
        }
    }

    println!("\n=== Checking for Mul-by-Zero Pattern ===");
    let mut mul_zero_assertions = Vec::new();
    for (i, assertion) in assertions.iter().enumerate() {
        let count = count_mul_by_zero(assertion.root());
        if count > 0 {
            mul_zero_assertions.push((i, count));
        }
    }
    if !mul_zero_assertions.is_empty() {
        println!("  Found {} assertions with mul-by-zero:", mul_zero_assertions.len());
        for (i, count) in mul_zero_assertions.iter().take(20) {
            println!("    Assertion {}: {} mul-by-zero operations", i, count);
        }
        if mul_zero_assertions.len() > 20 {
            println!("    ... and {} more", mul_zero_assertions.len() - 20);
        }
    } else {
        println!("  No mul-by-zero operations found");
    }

    // Build variable name mapping from VarAllocator
    let var_names: HashMap<u16, String> = var_alloc
        .descriptions()
        .iter()
        .map(|(idx, name)| (*idx, name.clone()))
        .collect();

    // Generate Gnark circuit
    println!("\n=== Generating Gnark Circuit ===");
    let circuit_code = generate_stages_circuit(&assertions, &var_names, "JoltStages16Circuit");

    // Write to file
    let output_path = "/Users/mariogalante/DEV/wonderjolt/jolt/gnark-transpiler/go/stages16_circuit.go";
    std::fs::write(output_path, &circuit_code).expect("Failed to write circuit file");
    println!("  Circuit written to: {}", output_path);
    println!("  Circuit size: {} bytes", circuit_code.len());

    println!("\n=== SUCCESS ===");
    println!("TranspilableVerifier stages 1-6 transpiled to Gnark circuit.");
}

/// Generate Gnark circuit code from accumulated assertions
fn generate_stages_circuit(
    assertions: &[MleAst],
    var_names: &HashMap<u16, String>,
    circuit_name: &str,
) -> String {
    let mut codegen = MemoizedCodeGen::with_var_names(var_names.clone());

    // First pass: count references to all assertion roots
    for assertion in assertions {
        codegen.count_refs(assertion.root());
    }

    // Second pass: generate expressions for each assertion
    let assertion_exprs: Vec<String> = assertions
        .iter()
        .map(|a| codegen.generate_expr(a.root()))
        .collect();

    let bindings_code = codegen.bindings_code();
    let vars = codegen.vars();

    let mut output = String::new();

    // Package and imports
    output.push_str("package jolt_verifier\n\n");
    output.push_str("import (\n");
    output.push_str("\t\"github.com/consensys/gnark/frontend\"\n");
    if bindings_code.contains("poseidon.Hash")
        || assertion_exprs.iter().any(|e| e.contains("poseidon.Hash"))
    {
        output.push_str("\t\"jolt_verifier/poseidon\"\n");
    }
    output.push_str(")\n\n");

    // Note: bigInt helper is defined in helpers.go

    // Circuit struct
    output.push_str(&format!("type {} struct {{\n", circuit_name));

    // Add input variables - use sanitized names
    for var_idx in vars.iter() {
        let name = var_names
            .get(var_idx)
            .map(|n| sanitize_go_name(n))
            .unwrap_or_else(|| format!("X{}", var_idx));
        output.push_str(&format!(
            "\t{} frontend.Variable `gnark:\",public\"`\n",
            name
        ));
    }

    output.push_str("}\n\n");

    // Define method
    output.push_str(&format!(
        "func (circuit *{}) Define(api frontend.API) error {{\n",
        circuit_name
    ));

    // CSE bindings
    if !bindings_code.is_empty() {
        output.push_str("\t// Memoized subexpressions (CSE)\n");
        output.push_str(&bindings_code);
        output.push_str("\n");
    }

    // Generate assertions - each expression must equal zero
    output.push_str("\t// Verification assertions (each must equal 0)\n");
    for (i, expr) in assertion_exprs.iter().enumerate() {
        output.push_str(&format!("\ta{} := {}\n", i, expr));
        output.push_str(&format!("\tapi.AssertIsEqual(a{}, 0)\n", i));
        if (i + 1) % 100 == 0 {
            output.push_str(&format!("\t// ... assertion {} of {}\n", i + 1, assertion_exprs.len()));
        }
    }

    output.push_str("\n\treturn nil\n");
    output.push_str("}\n");

    output
}

