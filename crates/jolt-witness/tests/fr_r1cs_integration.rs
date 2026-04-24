//! End-to-end: FieldRegEvent stream → replayed per-cycle snapshots →
//! R1CS witness → ConstraintMatrices::check_witness PASSES.
//!
//! Mirrors what the Phase 4 prover pipeline will do at the Az/Bz/Cz
//! materialization step: the FR Twist sumcheck will prove that the three
//! virtual operand slots (V_FIELD_RS1/RS2/RD_VALUE) at the Spartan random
//! cycle point match the FR register state implied by FieldRegInc + Ra.
//! Here we stay per-cycle (no MLE / sumcheck), just verify the witness
//! satisfies every R1CS row.

#![allow(non_snake_case)]

use jolt_field::Fr;
use jolt_r1cs::constraints::rv64::*;
use jolt_witness::{replay_field_regs, FieldRegEvent, FrCycleBytecode, FrCycleData};
use num_traits::{One, Zero};

fn empty_cycle_witness() -> Vec<Fr> {
    vec![Fr::zero(); NUM_VARS_PER_CYCLE]
}

fn limbs_to_fr(limbs: [u64; 4]) -> Fr {
    use jolt_field::Field;
    let lo = Fr::from_u128((limbs[0] as u128) | ((limbs[1] as u128) << 64));
    let hi = Fr::from_u128((limbs[2] as u128) | ((limbs[3] as u128) << 64));
    lo + hi * Fr::one().mul_pow_2(128)
}

/// Apply a single FR snapshot's three operand values onto a per-cycle R1CS
/// witness. Mirrors what r1cs_cycle_witness will do in Phase 4 once the
/// replay output is threaded through.
fn apply_fr_snapshot(w: &mut [Fr], snap: &FrCycleData) {
    w[V_FIELD_RS1_VALUE] = limbs_to_fr(snap.rs1_val);
    w[V_FIELD_RS2_VALUE] = limbs_to_fr(snap.rs2_val);
    w[V_FIELD_RD_VALUE] = limbs_to_fr(snap.rd_val);
}

fn base_witness() -> Vec<Fr> {
    let mut w = empty_cycle_witness();
    w[V_CONST] = Fr::one();
    // No-op RV defaults: keep PC etc. at 0 and gate DoNotUpdateUnexpandedPC
    // so constraint 16 (NextUnexpPCUpdateOtherwise) is satisfied.
    w[V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC] = Fr::one();
    w
}

/// Honest 3-cycle trace:
///   cycle 0: FieldMov x5 → f1   (x5 = 10)
///   cycle 1: FieldMov x6 → f2   (x6 = 20)
///   cycle 2: FieldAdd f1, f2 → f3
/// The replay stream produces the correct operand values at every cycle,
/// and the resulting R1CS witnesses satisfy all 35 constraints per cycle.
#[test]
fn field_mov_then_add_satisfies_r1cs_on_every_cycle() {
    let matrices = rv64_constraints::<Fr>();

    let bytecode = vec![
        // cycle 0: FieldMov — bridge op, doesn't read frs2 or frs1 on the
        // FR side (Rs1Value comes from integer registers).
        FrCycleBytecode {
            frs1: 0,
            frs2: 0,
            reads_frs2: false,
        },
        // cycle 1: FieldMov — same
        FrCycleBytecode {
            frs1: 0,
            frs2: 0,
            reads_frs2: false,
        },
        // cycle 2: FieldAdd — reads FReg[1] and FReg[2]
        FrCycleBytecode {
            frs1: 1,
            frs2: 2,
            reads_frs2: true,
        },
    ];
    let events = vec![
        FieldRegEvent {
            cycle: 0,
            slot: 1,
            old: [0; 4],
            new: [10, 0, 0, 0],
        },
        FieldRegEvent {
            cycle: 1,
            slot: 2,
            old: [0; 4],
            new: [20, 0, 0, 0],
        },
        FieldRegEvent {
            cycle: 2,
            slot: 3,
            old: [0; 4],
            new: [30, 0, 0, 0],
        },
    ];

    let snaps = replay_field_regs(3, &bytecode, &events);

    // Build + check cycle 0 (FieldMov, x5 = 10). FieldMov sets
    // LeftOperandIsRs1Value so V_LEFT_INSTRUCTION_INPUT = V_RS1_VALUE; the
    // default lookup-operand routing (rows 5-10) mirrors the integer case.
    for (cycle_idx, xreg_val) in [(0usize, 10u64), (1usize, 20u64)] {
        let mut w = base_witness();
        w[V_FLAG_IS_FIELD_MOV] = Fr::one();
        w[V_RS1_VALUE] = Fr::from(xreg_val);
        w[V_LEFT_INSTRUCTION_INPUT] = Fr::from(xreg_val);
        w[V_LEFT_LOOKUP_OPERAND] = Fr::from(xreg_val);
        // V_PRODUCT = V_LEFT_INSTRUCTION_INPUT · V_RIGHT_INSTRUCTION_INPUT
        // = xreg_val · 0 = 0; leave default.
        apply_fr_snapshot(&mut w, &snaps[cycle_idx]);
        matrices
            .check_witness(&w)
            .unwrap_or_else(|row| panic!("cycle {cycle_idx} (FieldMov) failed row {row}"));
    }

    // Cycle 2 (FieldAdd f1+f2→f3). FR rs1 = 10, rs2 = 20, rd = 30. The
    // lookup operands need to match instruction inputs (rows 6/10 guard
    // with 1-Add-Sub-Mul(-Advice); FieldAdd doesn't set any of those).
    {
        let mut w = base_witness();
        w[V_FLAG_IS_FIELD_ADD] = Fr::one();
        apply_fr_snapshot(&mut w, &snaps[2]);
        matrices
            .check_witness(&w)
            .unwrap_or_else(|row| panic!("cycle 2 (FieldAdd) failed row {row}"));
    }
}

/// If the replayed FR state disagrees with the prover's R1CS operand
/// slots, the R1CS check catches it. Here we forge `rd_val` on the
/// FieldAdd cycle; the constraint `Rs1V + Rs2V − RdV = 0` fails.
///
/// Demonstrates that the R1CS alone (before FR Twist) catches intra-row
/// operand tampering. Cross-cycle state tampering is what Phase 4's FR
/// Twist catches.
#[test]
fn tampered_rd_val_breaks_field_add_row() {
    let matrices = rv64_constraints::<Fr>();
    let bytecode = vec![
        FrCycleBytecode::default(),
        FrCycleBytecode::default(),
        FrCycleBytecode {
            frs1: 1,
            frs2: 2,
            reads_frs2: true,
        },
    ];
    let events = vec![
        FieldRegEvent {
            cycle: 0,
            slot: 1,
            old: [0; 4],
            new: [10, 0, 0, 0],
        },
        FieldRegEvent {
            cycle: 1,
            slot: 2,
            old: [0; 4],
            new: [20, 0, 0, 0],
        },
        FieldRegEvent {
            cycle: 2,
            slot: 3,
            old: [0; 4],
            new: [30, 0, 0, 0],
        },
    ];
    let mut snaps = replay_field_regs(3, &bytecode, &events);
    // Tamper cycle 2: claim rd_val = 31 instead of 30.
    snaps[2].rd_val = [31, 0, 0, 0];

    let mut w = base_witness();
    w[V_FLAG_IS_FIELD_ADD] = Fr::one();
    apply_fr_snapshot(&mut w, &snaps[2]);
    assert!(
        matrices.check_witness(&w).is_err(),
        "tampered rd_val must fail FieldAdd R1CS row (guard · (rs1+rs2−rd) = 0)"
    );
}
