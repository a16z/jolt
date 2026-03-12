//! Soundness and negative tests.
//!
//! Verifies that the proving system correctly rejects:
//! - Bad witnesses (violated R1CS constraints)
//! - Tampered proofs (modified Spartan evaluations)

mod common;

use common::*;

/// Bad witness (violated constraint) causes prover to fail.
#[test]
fn bad_witness_rejected() {
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let config = JoltConfig { num_cycles: 4 };
    let key = preprocess::<Fr, MockPCS>(&config, |_| ((), ()));

    let mut bad_add = add_cycle_witness(4, 1, 7, 3);
    bad_add[r1cs::V_PRODUCT] = Fr::from_u64(999);

    let witnesses = vec![
        nop_cycle_witness(0, 0),
        bad_add,
        nop_cycle_witness(8, 2),
        nop_cycle_witness(12, 3),
    ];

    let num_col_vars = key.spartan_key.num_col_vars();
    let reduction_data = build_reduction_data(num_col_vars, &mut rng);
    let (com_a, ()) = MockPCS::commit(&reduction_data.poly_a, &());
    let (com_b, ()) = MockPCS::commit(&reduction_data.poly_b, &());

    let mut pt = Blake2bTranscript::new(b"jolt-e2e");

    let result = prove::<MockPCS, Blake2bTranscript>(
        &key,
        &witnesses,
        vec![com_a, com_b],
        |_r_x, r_y, _t| {
            let stage = build_prover_stage(&reduction_data, r_y.to_vec());
            vec![Box::new(stage)]
        },
        &mut pt,
    );

    assert!(result.is_err(), "proving with bad witness should fail");
}

/// Tampered proof is rejected by the verifier.
#[test]
fn tampered_proof_rejected() {
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let config = JoltConfig { num_cycles: 4 };
    let key = preprocess::<Fr, MockPCS>(&config, |_| ((), ()));

    let witnesses = vec![
        nop_cycle_witness(0, 0),
        add_cycle_witness(4, 1, 7, 3),
        nop_cycle_witness(8, 2),
        nop_cycle_witness(12, 3),
    ];

    let num_col_vars = key.spartan_key.num_col_vars();
    let reduction_data = build_reduction_data(num_col_vars, &mut rng);
    let (com_a, ()) = MockPCS::commit(&reduction_data.poly_a, &());
    let (com_b, ()) = MockPCS::commit(&reduction_data.poly_b, &());

    let mut pt = Blake2bTranscript::new(b"jolt-e2e");

    let mut proof = prove::<MockPCS, Blake2bTranscript>(
        &key,
        &witnesses,
        vec![com_a, com_b],
        |_r_x, r_y, _t| {
            let stage = build_prover_stage(&reduction_data, r_y.to_vec());
            vec![Box::new(stage)]
        },
        &mut pt,
    )
    .expect("proving should succeed");

    proof.spartan_proof.witness_eval += Fr::from_u64(1);

    let vk = jolt_verifier::JoltVerifyingKey {
        spartan_key: key.spartan_key.clone(),
        pcs_setup: (),
    };

    let mut vt = Blake2bTranscript::new(b"jolt-e2e");

    let result = jolt_verifier::verify::<MockPCS, Blake2bTranscript>(
        &proof,
        &vk,
        |_r_x, r_y, _t| vec![build_verifier_descriptor(&reduction_data, r_y, 0)],
        &mut vt,
    );

    assert!(result.is_err(), "tampered proof should fail verification");
}

/// Wrong constant (V_CONST=0) — either prover or verifier rejects.
/// Currently underconstrained; enable once constant-check sumcheck is added.
#[test]
#[ignore = "underconstrained: enable once constant-check sumcheck is added"]
fn bad_constant_rejected() {
    let mut rng = ChaCha20Rng::seed_from_u64(77);
    let config = JoltConfig { num_cycles: 4 };
    let key = preprocess::<Fr, MockPCS>(&config, |_| ((), ()));

    let mut bad_nop = nop_cycle_witness(0, 0);
    bad_nop[r1cs::V_CONST] = Fr::from_u64(0);

    let witnesses = vec![
        bad_nop,
        nop_cycle_witness(4, 1),
        nop_cycle_witness(8, 2),
        nop_cycle_witness(12, 3),
    ];

    let num_col_vars = key.spartan_key.num_col_vars();
    let reduction_data = build_reduction_data(num_col_vars, &mut rng);
    let (com_a, ()) = MockPCS::commit(&reduction_data.poly_a, &());
    let (com_b, ()) = MockPCS::commit(&reduction_data.poly_b, &());

    let mut pt = Blake2bTranscript::new(b"bad-const");

    let prove_result = prove::<MockPCS, Blake2bTranscript>(
        &key,
        &witnesses,
        vec![com_a, com_b],
        |_r_x, r_y, _t| {
            let stage = build_prover_stage(&reduction_data, r_y.to_vec());
            vec![Box::new(stage)]
        },
        &mut pt,
    );

    // Spartan is probabilistic: prover may or may not detect the violation.
    // If it produces a proof, the verifier must reject it.
    if let Ok(proof) = prove_result {
        let vk = jolt_verifier::JoltVerifyingKey {
            spartan_key: key.spartan_key.clone(),
            pcs_setup: (),
        };
        let mut vt = Blake2bTranscript::new(b"bad-const");
        let verify_result = jolt_verifier::verify::<MockPCS, Blake2bTranscript>(
            &proof,
            &vk,
            |_r_x, r_y, _t| vec![build_verifier_descriptor(&reduction_data, r_y, 0)],
            &mut vt,
        );
        assert!(
            verify_result.is_err(),
            "verifier should reject bad-constant proof"
        );
    }
}

/// Inconsistent PC update (NextPC=999) — either prover or verifier rejects.
/// Currently underconstrained; enable once PC-consistency sumcheck is added.
#[test]
#[ignore = "underconstrained: enable once PC-consistency sumcheck is added"]
fn bad_pc_update_rejected() {
    let mut rng = ChaCha20Rng::seed_from_u64(88);
    let config = JoltConfig { num_cycles: 4 };
    let key = preprocess::<Fr, MockPCS>(&config, |_| ((), ()));

    let mut bad_pc = nop_cycle_witness(0, 0);
    bad_pc[r1cs::V_NEXT_PC] = Fr::from_u64(999);

    let witnesses = vec![
        bad_pc,
        nop_cycle_witness(4, 1),
        nop_cycle_witness(8, 2),
        nop_cycle_witness(12, 3),
    ];

    let num_col_vars = key.spartan_key.num_col_vars();
    let reduction_data = build_reduction_data(num_col_vars, &mut rng);
    let (com_a, ()) = MockPCS::commit(&reduction_data.poly_a, &());
    let (com_b, ()) = MockPCS::commit(&reduction_data.poly_b, &());

    let mut pt = Blake2bTranscript::new(b"bad-pc");

    let prove_result = prove::<MockPCS, Blake2bTranscript>(
        &key,
        &witnesses,
        vec![com_a, com_b],
        |_r_x, r_y, _t| {
            let stage = build_prover_stage(&reduction_data, r_y.to_vec());
            vec![Box::new(stage)]
        },
        &mut pt,
    );

    if let Ok(proof) = prove_result {
        let vk = jolt_verifier::JoltVerifyingKey {
            spartan_key: key.spartan_key.clone(),
            pcs_setup: (),
        };
        let mut vt = Blake2bTranscript::new(b"bad-pc");
        let verify_result = jolt_verifier::verify::<MockPCS, Blake2bTranscript>(
            &proof,
            &vk,
            |_r_x, r_y, _t| vec![build_verifier_descriptor(&reduction_data, r_y, 0)],
            &mut vt,
        );
        assert!(
            verify_result.is_err(),
            "verifier should reject bad-PC proof"
        );
    }
}

/// Wrong ADD sum (lookup output doesn't match left+right) should fail.
#[test]
fn bad_add_output_rejected() {
    let mut rng = ChaCha20Rng::seed_from_u64(99);
    let config = JoltConfig { num_cycles: 4 };
    let key = preprocess::<Fr, MockPCS>(&config, |_| ((), ()));

    let mut bad_add = add_cycle_witness(4, 1, 7, 3);
    bad_add[r1cs::V_LOOKUP_OUTPUT] = Fr::from_u64(999);

    let witnesses = vec![
        nop_cycle_witness(0, 0),
        bad_add,
        nop_cycle_witness(8, 2),
        nop_cycle_witness(12, 3),
    ];

    let num_col_vars = key.spartan_key.num_col_vars();
    let reduction_data = build_reduction_data(num_col_vars, &mut rng);
    let (com_a, ()) = MockPCS::commit(&reduction_data.poly_a, &());
    let (com_b, ()) = MockPCS::commit(&reduction_data.poly_b, &());

    let mut pt = Blake2bTranscript::new(b"bad-add");

    let result = prove::<MockPCS, Blake2bTranscript>(
        &key,
        &witnesses,
        vec![com_a, com_b],
        |_r_x, r_y, _t| {
            let stage = build_prover_stage(&reduction_data, r_y.to_vec());
            vec![Box::new(stage)]
        },
        &mut pt,
    );

    assert!(result.is_err(), "wrong ADD output should break constraints");
}

/// LOAD with wrong RAM address (RamAddr ≠ Rs1+Imm) should fail.
#[test]
fn bad_load_address_rejected() {
    let mut rng = ChaCha20Rng::seed_from_u64(101);
    let config = JoltConfig { num_cycles: 4 };
    let key = preprocess::<Fr, MockPCS>(&config, |_| ((), ()));

    let mut bad_load = load_cycle_witness(0, 0, 100, 20, 42);
    bad_load[r1cs::V_RAM_ADDRESS] = Fr::from_u64(999);

    let witnesses = vec![
        bad_load,
        nop_cycle_witness(4, 1),
        nop_cycle_witness(8, 2),
        nop_cycle_witness(12, 3),
    ];

    let num_col_vars = key.spartan_key.num_col_vars();
    let reduction_data = build_reduction_data(num_col_vars, &mut rng);
    let (com_a, ()) = MockPCS::commit(&reduction_data.poly_a, &());
    let (com_b, ()) = MockPCS::commit(&reduction_data.poly_b, &());

    let mut pt = Blake2bTranscript::new(b"bad-load");

    let result = prove::<MockPCS, Blake2bTranscript>(
        &key,
        &witnesses,
        vec![com_a, com_b],
        |_r_x, r_y, _t| {
            let stage = build_prover_stage(&reduction_data, r_y.to_vec());
            vec![Box::new(stage)]
        },
        &mut pt,
    );

    assert!(
        result.is_err(),
        "wrong RAM address should break constraints"
    );
}
