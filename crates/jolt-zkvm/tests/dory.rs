//! Integration tests using the real Dory commitment scheme.
//!
//! These are slower than MockPCS tests but exercise the full cryptographic
//! pipeline including real polynomial commitments and opening proofs.

mod common;

use common::*;
use jolt_dory::DoryScheme;

#[test]
fn dory_mixed_instructions() {
    run_e2e::<DoryScheme>(
        &[
            nop_cycle_witness(0, 0),
            add_cycle_witness(4, 1, 7, 3),
            nop_cycle_witness(8, 2),
            nop_cycle_witness(12, 3),
        ],
        |num_vars| {
            let prover_setup = DoryScheme::setup_prover(num_vars);
            let verifier_setup = DoryScheme::setup_verifier(num_vars);
            (prover_setup, verifier_setup)
        },
    );
}

#[test]
fn dory_load_store() {
    run_e2e::<DoryScheme>(
        &[
            load_cycle_witness(0, 0, 100, 20, 42),
            add_cycle_witness(4, 1, 42, 8),
            store_cycle_witness(8, 2, 180, 20, 50),
            nop_cycle_witness(12, 3),
        ],
        |num_vars| {
            let prover_setup = DoryScheme::setup_prover(num_vars);
            let verifier_setup = DoryScheme::setup_verifier(num_vars);
            (prover_setup, verifier_setup)
        },
    );
}

#[test]
fn dory_all_nops() {
    run_e2e::<DoryScheme>(
        &[
            nop_cycle_witness(0, 0),
            nop_cycle_witness(4, 1),
            nop_cycle_witness(8, 2),
            nop_cycle_witness(12, 3),
        ],
        |num_vars| {
            let prover_setup = DoryScheme::setup_prover(num_vars);
            let verifier_setup = DoryScheme::setup_verifier(num_vars);
            (prover_setup, verifier_setup)
        },
    );
}

#[test]
fn dory_all_adds() {
    run_e2e::<DoryScheme>(
        &[
            add_cycle_witness(0, 0, 1, 2),
            add_cycle_witness(4, 1, 100, 200),
            add_cycle_witness(8, 2, 0, 0),
            add_cycle_witness(12, 3, 999, 1),
        ],
        |num_vars| {
            let prover_setup = DoryScheme::setup_prover(num_vars);
            let verifier_setup = DoryScheme::setup_verifier(num_vars);
            (prover_setup, verifier_setup)
        },
    );
}

#[test]
fn dory_single_cycle() {
    run_e2e::<DoryScheme>(&[nop_cycle_witness(0, 0)], |num_vars| {
        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryScheme::setup_verifier(num_vars);
        (prover_setup, verifier_setup)
    });
}

#[test]
fn dory_eight_cycles() {
    run_e2e::<DoryScheme>(
        &[
            nop_cycle_witness(0, 0),
            load_cycle_witness(4, 1, 0, 8, 100),
            add_cycle_witness(8, 2, 100, 50),
            store_cycle_witness(12, 3, 0, 16, 150),
            load_cycle_witness(16, 4, 0, 24, 200),
            add_cycle_witness(20, 5, 200, 150),
            store_cycle_witness(24, 6, 0, 32, 350),
            nop_cycle_witness(28, 7),
        ],
        |num_vars| {
            let prover_setup = DoryScheme::setup_prover(num_vars);
            let verifier_setup = DoryScheme::setup_verifier(num_vars);
            (prover_setup, verifier_setup)
        },
    );
}
