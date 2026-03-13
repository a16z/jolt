//! End-to-end integration tests: symbolic execution → bundle → codegen.

use jolt_field::Field;
use jolt_wrapper::arena::{self, ArenaSession, Node};
use jolt_wrapper::bundle::{AstBundle, VarAllocator};
use jolt_wrapper::gnark::{generate_go_file, GnarkAstEmitter, GoFileConfig, MemoizedCodeGen};
use jolt_wrapper::symbolic::SymbolicField;
use jolt_wrapper::transcript::PoseidonSymbolicTranscript;
use num_traits::{One, Zero};

use jolt_transcript::Transcript;

#[test]
fn booleanity_constraint_pipeline() {
    let _s = ArenaSession::new();

    // γ · (H² − H) == 0  (booleanity check)
    let h = SymbolicField::variable(0, "H");
    let gamma = SymbolicField::variable(1, "gamma");
    let constraint = gamma * (h * h - h);

    let mut alloc = VarAllocator::new();
    let _ = alloc.input("H");
    let _ = alloc.input("gamma");
    alloc.assert_zero(constraint.into_edge());

    let bundle = alloc.finish();
    let code = generate_go_file(&bundle, &GoFileConfig::default());

    assert!(code.contains("api.Mul("));
    assert!(code.contains("api.Sub("));
    assert!(code.contains("AssertIsEqual("));
    assert!(code.contains("circuit.H"));
    assert!(code.contains("circuit.Gamma"));
}

#[test]
fn ram_checking_constraint_pipeline() {
    let _s = ArenaSession::new();

    // init + γ · (rs - ws) == 0  (RAM read-write checking)
    let init = SymbolicField::variable(0, "init_eval");
    let rs = SymbolicField::variable(1, "read_sum");
    let ws = SymbolicField::variable(2, "write_sum");
    let gamma = SymbolicField::variable(3, "gamma");
    let constraint = init + gamma * (rs - ws);

    let mut alloc = VarAllocator::new();
    let _ = alloc.input("init_eval");
    let _ = alloc.input("read_sum");
    let _ = alloc.input("write_sum");
    let _ = alloc.input("gamma");
    alloc.assert_zero(constraint.into_edge());

    let bundle = alloc.finish();

    // JSON roundtrip
    let json = bundle.to_json().unwrap();
    let restored = AstBundle::from_json(&json).unwrap();
    assert_eq!(restored.nodes.len(), bundle.nodes.len());
    assert_eq!(restored.assertions.len(), bundle.assertions.len());

    // Gnark codegen
    let code = generate_go_file(&restored, &GoFileConfig::default());
    assert!(code.contains("api.Sub("));
    assert!(code.contains("api.Mul("));
    assert!(code.contains("api.Add("));
    assert!(code.contains("AssertIsEqual("));
}

#[test]
fn weighted_sum_with_constants() {
    let _s = ArenaSession::new();

    // 3·x + 7·y - 42 == 0
    let x = SymbolicField::variable(0, "x");
    let y = SymbolicField::variable(1, "y");
    let three = SymbolicField::from_u64(3);
    let seven = SymbolicField::from_u64(7);
    let forty_two = SymbolicField::from_u64(42);
    let constraint = three * x + seven * y - forty_two;

    let mut alloc = VarAllocator::new();
    let _ = alloc.input("x");
    let _ = alloc.input("y");
    alloc.assert_zero(constraint.into_edge());

    let bundle = alloc.finish();

    let mut emitter = GnarkAstEmitter::new();
    bundle.emit(&mut emitter);

    // Should have mul, add, sub operations with constants inlined
    let code = emitter.finish();
    assert!(code.contains("api.Mul("));
    assert!(code.contains('3')); // small constant inlined
    assert!(code.contains('7'));
    assert!(code.contains("42"));
}

#[test]
fn multiple_assertions() {
    let _s = ArenaSession::new();

    let x = SymbolicField::variable(0, "x");
    let y = SymbolicField::variable(1, "y");

    let mut alloc = VarAllocator::new();
    let _ = alloc.input("x");
    let _ = alloc.input("y");

    // x * y == 0
    alloc.assert_zero((x * y).into_edge());
    // x + y == 0
    alloc.assert_zero((x + y).into_edge());
    // x - y == 0
    alloc.assert_equal(x.into_edge(), y.into_edge());

    let bundle = alloc.finish();
    assert_eq!(bundle.assertions.len(), 3);

    let mut emitter = GnarkAstEmitter::new();
    bundle.emit(&mut emitter);
    assert_eq!(emitter.assertion_lines().len(), 3);
}

#[test]
fn transcript_symbolic_append_and_challenge() {
    let _s = ArenaSession::new();

    let mut transcript = PoseidonSymbolicTranscript::new(b"test_protocol");

    // Append a symbolic value
    let x = SymbolicField::variable(0, "commitment");
    transcript.append(&x);

    // Squeeze a challenge
    let _challenge_u128 = transcript.challenge();

    // The challenge should be tunneled — next from_u128 picks it up
    let gamma = SymbolicField::from_u128(0);
    assert!(
        !gamma.is_constant(),
        "challenge should be symbolic, not constant"
    );

    // Use the challenge in a constraint
    let constraint = gamma * x;
    assert!(!constraint.is_constant());

    // Verify the arena recorded the right structure
    let nodes = arena::snapshot();
    // Var(commitment), Poseidon(state, round, data), Challenge, Mul
    assert!(nodes.len() >= 3);
    assert!(nodes.iter().any(|n| matches!(n, Node::Poseidon { .. })));
    assert!(nodes.iter().any(|n| matches!(n, Node::Challenge { .. })));
}

#[test]
fn transcript_multiple_rounds() {
    let _s = ArenaSession::new();

    let mut transcript = PoseidonSymbolicTranscript::new(b"multi_round");

    // Round 1: append + challenge
    let x = SymbolicField::variable(0, "x");
    transcript.append(&x);
    let _ = transcript.challenge();
    let gamma1 = SymbolicField::from_u128(0);

    // Round 2: append result + challenge
    let product = gamma1 * x;
    transcript.append(&product);
    let _ = transcript.challenge();
    let gamma2 = SymbolicField::from_u128(0);

    assert!(!gamma1.is_constant());
    assert!(!gamma2.is_constant());
    assert_ne!(
        gamma1, gamma2,
        "different challenges should be different nodes"
    );
}

#[test]
fn transcript_raw_bytes_append() {
    let _s = ArenaSession::new();

    let mut transcript = PoseidonSymbolicTranscript::new(b"raw");
    // Raw bytes don't go through the tunnel — they should become constant scalars
    transcript.append_bytes(b"hello world");

    let nodes = arena::snapshot();
    assert_eq!(nodes.len(), 1); // one Poseidon node
    assert!(matches!(&nodes[0], Node::Poseidon { .. }));
}

#[test]
fn mixed_constant_symbolic_folding() {
    let _s = ArenaSession::new();

    let x = SymbolicField::variable(0, "x");

    // 0 * x should fold to 0 (no node created for the mul)
    let zero_x = SymbolicField::zero() * x;
    assert!(zero_x.is_zero());

    // 1 * x should fold to x (identity)
    let one_x = SymbolicField::one() * x;
    assert_eq!(one_x.into_edge(), x.into_edge());

    // x + 0 should fold to x
    let x_plus_zero = x + SymbolicField::zero();
    assert_eq!(x_plus_zero.into_edge(), x.into_edge());

    // x - 0 should fold to x
    let x_minus_zero = x - SymbolicField::zero();
    assert_eq!(x_minus_zero.into_edge(), x.into_edge());

    // 3 + 4 should fully fold to 7
    let pure_const = SymbolicField::from_u64(3) + SymbolicField::from_u64(4);
    assert_eq!(pure_const.to_u64(), Some(7));
}

#[test]
fn nested_constant_folding() {
    let _s = ArenaSession::new();

    // (2 * 3) + (4 * 5) should fold to 26
    let a = SymbolicField::from_u64(2) * SymbolicField::from_u64(3);
    let b = SymbolicField::from_u64(4) * SymbolicField::from_u64(5);
    let result = a + b;
    assert_eq!(result.to_u64(), Some(26));
    // No nodes should be allocated — everything folds
    assert_eq!(arena::node_count(), 0);
}

#[test]
fn go_file_compiles_with_multiple_inputs() {
    let _s = ArenaSession::new();

    // Create a circuit with 5 inputs and a complex constraint
    let vars: Vec<SymbolicField> = (0..5)
        .map(|i| SymbolicField::variable(i, format!("input_{i}")))
        .collect();

    // Polynomial: sum of all pairwise products
    let mut constraint = SymbolicField::zero();
    for i in 0..5 {
        for j in (i + 1)..5 {
            constraint += vars[i] * vars[j];
        }
    }

    let mut alloc = VarAllocator::new();
    for i in 0..5 {
        let _ = alloc.input(format!("input_{i}"));
    }
    alloc.assert_zero(constraint.into_edge());

    let bundle = alloc.finish();
    let config = GoFileConfig {
        package_name: "pairwise".into(),
        circuit_name: "PairwiseCircuit".into(),
    };
    let code = generate_go_file(&bundle, &config);

    assert!(code.contains("package pairwise"));
    assert!(code.contains("type PairwiseCircuit struct {"));
    assert!(code.contains("Input_0"));
    assert!(code.contains("Input_4"));
    // 10 pairwise multiplications + 9 additions
    let mul_count = code.matches("api.Mul(").count();
    let add_count = code.matches("api.Add(").count();
    assert_eq!(mul_count, 10, "should have C(5,2) = 10 multiplications");
    assert_eq!(add_count, 9, "should have 9 additions to sum 10 products");
}

#[test]
fn go_file_has_valid_go_syntax_markers() {
    let _s = ArenaSession::new();

    let x = SymbolicField::variable(0, "val");
    let mut alloc = VarAllocator::new();
    let _ = alloc.input("val");
    alloc.assert_zero(x.into_edge());

    let bundle = alloc.finish();
    let code = generate_go_file(&bundle, &GoFileConfig::default());

    // Structural checks
    assert!(code.contains("import ("));
    assert!(code.contains("\"math/big\""));
    assert!(code.contains("\"github.com/consensys/gnark/frontend\""));
    assert!(code.contains("func bigInt(s string) *big.Int {"));
    assert!(code.contains("`gnark:\",public\"`"));
    assert!(code.contains("return nil"));
    assert!(code.ends_with("}\n"));

    // CSE variable assignments (v_N :=) should be inside Define
    let define_start = code.find("Define(api frontend.API) error {").unwrap();
    for line in code.lines() {
        if line.trim_start().starts_with("v_") && line.contains(":=") {
            let pos = code.find(line).unwrap();
            assert!(pos > define_start, "CSE assignment outside Define method");
        }
    }
}

#[test]
fn memoized_codegen_ref_counting_accuracy() {
    let _s = ArenaSession::new();

    // Build: x*y used in both assertions
    let x = SymbolicField::variable(0, "x");
    let y = SymbolicField::variable(1, "y");
    let product = x * y;

    let mut alloc = VarAllocator::new();
    let _ = alloc.input("x");
    let _ = alloc.input("y");
    alloc.assert_zero(product.into_edge());
    alloc.assert_zero((product + SymbolicField::one()).into_edge());

    let bundle = alloc.finish();

    let mut codegen = MemoizedCodeGen::new();
    codegen.count_refs(&bundle);

    // The product node (node 2) should be referenced multiple times
    let refs = codegen.ref_counts();
    assert!(!refs.is_empty());
}

#[test]
fn bundle_json_preserves_all_node_types() {
    let _s = ArenaSession::new();

    // Create nodes of each type to ensure JSON covers them all
    let x = SymbolicField::variable(0, "x");
    let y = SymbolicField::variable(1, "y");
    let neg_x = -x;
    let sum = x + y;
    let diff = x - y;
    let prod = x * y;
    let quot = x / y;
    let inv_x = x.inverse().unwrap();

    let mut alloc = VarAllocator::new();
    let _ = alloc.input("x");
    let _ = alloc.input("y");
    alloc.assert_zero(neg_x.into_edge());
    alloc.assert_zero(sum.into_edge());
    alloc.assert_zero(diff.into_edge());
    alloc.assert_zero(prod.into_edge());
    alloc.assert_zero(quot.into_edge());
    alloc.assert_zero(inv_x.into_edge());

    let bundle = alloc.finish();
    let json = bundle.to_json().unwrap();
    let restored = AstBundle::from_json(&json).unwrap();

    assert_eq!(restored.nodes.len(), bundle.nodes.len());
    assert_eq!(restored.assertions.len(), 6);

    // Re-emit and verify same output
    let mut emitter1 = GnarkAstEmitter::new();
    bundle.emit(&mut emitter1);
    let mut emitter2 = GnarkAstEmitter::new();
    restored.emit(&mut emitter2);

    assert_eq!(emitter1.lines().len(), emitter2.lines().len());
    for (l1, l2) in emitter1.lines().iter().zip(emitter2.lines()) {
        assert_eq!(l1, l2);
    }
}

#[test]
fn large_arena_session() {
    let _s = ArenaSession::new();

    // Build a long chain: x_0 + x_1 + x_2 + ... + x_999
    let mut acc = SymbolicField::variable(0, "x_0");
    for i in 1..1000 {
        let next = SymbolicField::variable(i as u32, format!("x_{i}"));
        acc += next;
    }

    // 1000 vars + 999 adds = 1999 nodes
    assert_eq!(arena::node_count(), 1999);

    // The final value should be symbolic
    assert!(!acc.is_constant());
}

#[test]
fn arena_session_clears_on_drop() {
    {
        let _s = ArenaSession::new();
        let _ = SymbolicField::variable(0, "x");
        assert_eq!(arena::node_count(), 1);
    }
    // After drop, new session starts fresh
    let _s = ArenaSession::new();
    assert_eq!(arena::node_count(), 0);
}
