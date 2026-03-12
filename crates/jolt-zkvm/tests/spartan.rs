//! Spartan-only integration tests.
//!
//! These test the uniform Spartan R1CS prover/verifier in isolation
//! (no sumcheck stages, no opening proofs). Exercises the Jolt R1CS
//! constraint system with hand-crafted and real-program witnesses.

mod common;

use common::*;
use jolt_spartan::{UniformSpartanProver, UniformSpartanVerifier};
use jolt_zkvm::preprocessing::interleave_witnesses;
use jolt_zkvm::stages::s1_spartan::UniformSpartanStage;

#[test]
fn jolt_r1cs_key_dimensions() {
    let config = JoltConfig { num_cycles: 4 };
    let key = preprocess::<Fr, MockPCS>(&config, |_| ((), ()));

    assert_eq!(
        key.spartan_key.num_constraints,
        r1cs::NUM_CONSTRAINTS_PER_CYCLE
    );
    assert_eq!(key.spartan_key.num_vars, r1cs::NUM_VARS_PER_CYCLE);
    assert_eq!(key.spartan_key.num_cycles, 4);
}

#[test]
fn uniform_spartan_nop_only() {
    let config = JoltConfig { num_cycles: 2 };
    let key = preprocess::<Fr, MockPCS>(&config, |_| ((), ())).spartan_key;

    let witnesses = vec![nop_cycle_witness(0, 0), nop_cycle_witness(4, 1)];
    let flat = interleave_witnesses(&key, &witnesses);

    let mut pt = Blake2bTranscript::new(b"nop-only");
    commit_and_append::<MockPCS>(&flat, &(), &mut pt);
    let result = UniformSpartanStage::prove(&key, &flat, &flat, &mut pt)
        .expect("NOP-only proving should succeed");

    let mut vt = Blake2bTranscript::new(b"nop-only");
    commit_and_append::<MockPCS>(&flat, &(), &mut vt);
    let _ = UniformSpartanStage::verify(&key, &result.proof, &mut vt)
        .expect("NOP-only verification should succeed");
}

#[test]
fn uniform_spartan_mixed_cycles() {
    let config = JoltConfig { num_cycles: 4 };
    let key = preprocess::<Fr, MockPCS>(&config, |_| ((), ())).spartan_key;

    let witnesses = vec![
        nop_cycle_witness(0, 0),
        add_cycle_witness(4, 1, 7, 3),
        nop_cycle_witness(8, 2),
        add_cycle_witness(12, 3, 10, 5),
    ];
    let flat = interleave_witnesses(&key, &witnesses);

    let mut pt = Blake2bTranscript::new(b"mixed-cycles");
    commit_and_append::<MockPCS>(&flat, &(), &mut pt);
    let result = UniformSpartanStage::prove(&key, &flat, &flat, &mut pt)
        .expect("mixed-cycle proving should succeed");

    let mut vt = Blake2bTranscript::new(b"mixed-cycles");
    commit_and_append::<MockPCS>(&flat, &(), &mut vt);
    let _ = UniformSpartanStage::verify(&key, &result.proof, &mut vt)
        .expect("mixed-cycle verification should succeed");
}

#[test]
fn uniform_spartan_load_store() {
    let config = JoltConfig { num_cycles: 4 };
    let key = preprocess::<Fr, MockPCS>(&config, |_| ((), ())).spartan_key;

    let witnesses = vec![
        load_cycle_witness(0, 0, 100, 20, 42),
        store_cycle_witness(4, 1, 200, 50, 42),
        load_cycle_witness(8, 2, 300, 0, 77),
        store_cycle_witness(12, 3, 400, 10, 77),
    ];
    let flat = interleave_witnesses(&key, &witnesses);

    let mut pt = Blake2bTranscript::new(b"load-store");
    commit_and_append::<MockPCS>(&flat, &(), &mut pt);
    let result = UniformSpartanStage::prove(&key, &flat, &flat, &mut pt)
        .expect("LOAD/STORE proving should succeed");

    let mut vt = Blake2bTranscript::new(b"load-store");
    commit_and_append::<MockPCS>(&flat, &(), &mut vt);
    let _ = UniformSpartanStage::verify(&key, &result.proof, &mut vt)
        .expect("LOAD/STORE verification should succeed");
}

#[test]
fn s1_challenge_vector_dimensions() {
    let config = JoltConfig { num_cycles: 4 };
    let key = preprocess::<Fr, MockPCS>(&config, |_| ((), ())).spartan_key;

    let witnesses = vec![
        nop_cycle_witness(0, 0),
        nop_cycle_witness(4, 1),
        nop_cycle_witness(8, 2),
        nop_cycle_witness(12, 3),
    ];
    let flat = interleave_witnesses(&key, &witnesses);

    let mut pt = Blake2bTranscript::new(b"s1-dims");
    commit_and_append::<MockPCS>(&flat, &(), &mut pt);
    let result =
        UniformSpartanStage::prove(&key, &flat, &flat, &mut pt).expect("proving should succeed");

    assert_eq!(result.r_x.len(), key.num_row_vars());
    assert_eq!(result.r_y.len(), key.num_col_vars());
    assert_eq!(result.witness_opening_claim.point, result.r_y);
    assert_eq!(result.witness_opening_claim.eval, result.proof.witness_eval);
}

mod real_program_spartan {
    use super::*;
    use jolt_host::Program;
    use jolt_openings::mock::MockCommitmentScheme;
    use jolt_zkvm::witness::generate_witnesses;

    #[test]
    fn muldiv_spartan_only() {
        let inputs = postcard::to_stdvec(&(9u32, 5u32, 3u32)).unwrap();
        let mut program = Program::new("muldiv-guest");
        let (_, trace, _, _) = program.trace(&inputs, &[], &[]);
        let output = generate_witnesses::<Fr>(&trace);

        let key = r1cs::build_jolt_spartan_key::<Fr>(output.cycle_witnesses.len());

        let total_cols_padded = key.total_cols().next_power_of_two();
        let mut flat = vec![Fr::from_u64(0); total_cols_padded];
        for (c, w) in output.cycle_witnesses.iter().enumerate() {
            let base = c * key.num_vars_padded;
            for (v, &val) in w.iter().enumerate().take(key.num_vars) {
                flat[base + v] = val;
            }
        }

        let (commitment, ()) = MockCommitmentScheme::<Fr>::commit(&flat, &());

        let mut pt = Blake2bTranscript::new(b"spartan-only");
        pt.append_bytes(format!("{commitment:?}").as_bytes());
        let proof = UniformSpartanProver::prove_dense(&key, &flat, &mut pt)
            .expect("proving should succeed");

        let mut vt = Blake2bTranscript::new(b"spartan-only");
        vt.append_bytes(format!("{commitment:?}").as_bytes());
        UniformSpartanVerifier::verify(&key, &proof, &mut vt).expect("verification should succeed");
    }

    #[test]
    fn muldiv_r1cs_satisfaction() {
        let inputs = postcard::to_stdvec(&(9u32, 5u32, 3u32)).unwrap();
        let mut program = Program::new("muldiv-guest");
        let (_, trace, _, _) = program.trace(&inputs, &[], &[]);
        let output = generate_witnesses::<Fr>(&trace);

        let key = r1cs::build_jolt_spartan_key::<Fr>(output.cycle_witnesses.len());
        let mut violations = Vec::new();

        for (cycle_idx, w) in output.cycle_witnesses.iter().enumerate() {
            for k in 0..r1cs::NUM_CONSTRAINTS_PER_CYCLE {
                let a_val: Fr = key.a_sparse[k]
                    .iter()
                    .map(|&(idx, coeff)| coeff * w[idx])
                    .sum();
                let b_val: Fr = key.b_sparse[k]
                    .iter()
                    .map(|&(idx, coeff)| coeff * w[idx])
                    .sum();
                let c_val: Fr = key.c_sparse[k]
                    .iter()
                    .map(|&(idx, coeff)| coeff * w[idx])
                    .sum();
                if a_val * b_val != c_val {
                    violations.push((cycle_idx, k));
                }
            }
        }

        assert!(
            violations.is_empty(),
            "{} R1CS violations (first: cycle {} constraint {})",
            violations.len(),
            violations[0].0,
            violations[0].1
        );
    }

    /// Verifies NOP-only Spartan at many sizes to catch size-dependent bugs.
    #[test]
    fn nop_spartan_size_sweep() {
        use jolt_zkvm::witness::bytecode::BytecodePreprocessing;
        use jolt_zkvm::witness::r1cs_inputs;

        for &n in &[2, 4, 8, 16, 32, 64, 128, 256, 512] {
            let trace: Vec<tracer::instruction::Cycle> = vec![tracer::instruction::Cycle::NoOp; n];
            let bytecode = BytecodePreprocessing::new(&trace);
            let witnesses: Vec<Vec<Fr>> = trace
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let next = trace.get(i + 1);
                    r1cs_inputs::cycle_to_witness(c, next, &bytecode)
                })
                .collect();

            let key = r1cs::build_jolt_spartan_key::<Fr>(n);
            let total_cols_padded = key.total_cols().next_power_of_two();
            let mut flat = vec![Fr::from_u64(0); total_cols_padded];
            for (c, w) in witnesses.iter().enumerate() {
                let base = c * key.num_vars_padded;
                for (v, &val) in w.iter().enumerate().take(key.num_vars) {
                    flat[base + v] = val;
                }
            }

            let (commitment, ()) = MockCommitmentScheme::<Fr>::commit(&flat, &());

            let mut pt = Blake2bTranscript::new(b"nop-sweep");
            pt.append_bytes(format!("{commitment:?}").as_bytes());
            let proof = UniformSpartanProver::prove_dense(&key, &flat, &mut pt)
                .unwrap_or_else(|_| panic!("proving should succeed at n={n}"));

            let mut vt = Blake2bTranscript::new(b"nop-sweep");
            vt.append_bytes(format!("{commitment:?}").as_bytes());
            UniformSpartanVerifier::verify(&key, &proof, &mut vt)
                .unwrap_or_else(|_| panic!("verification should succeed at n={n}"));
        }
    }
}
