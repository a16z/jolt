//! Debug test: check per-cycle R1CS satisfaction of the jolt-r1cs witness.
//!
//! Computes Az[k]*Bz[k] for each cycle c and constraint k, reporting any
//! (cycle, constraint) pairs that violate Az*Bz=Cz. This diagnoses whether
//! the witness extraction or constraint matrices are the source of nonzero
//! base evaluations in the outer uniskip sumcheck.
#![allow(non_snake_case, clippy::print_stderr)]

use jolt_field::{Field, Fr};
use jolt_host::{extract_trace, BytecodePreprocessing, Program};
use jolt_r1cs::constraint::ConstraintMatrices;
use jolt_r1cs::constraints::rv64::{self, *};
use jolt_r1cs::{R1csKey, R1csSource};
use num_traits::Zero;

fn setup() -> (ConstraintMatrices<Fr>, Vec<Fr>, usize, usize) {
    let mut program = Program::new("muldiv-guest");
    let (bytecode_raw, _init_mem, _program_size, entry_address) = program.decode();
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
    let (_, trace, _, io_device) = program.trace(&inputs, &[], &[]);

    let bytecode = BytecodePreprocessing::preprocess(bytecode_raw, entry_address);

    let trace_length = trace.len().next_power_of_two();
    let matrices = rv64::rv64_constraints::<Fr>();
    let r1cs_key = R1csKey::new(matrices.clone(), trace_length);

    let (_, r1cs_witness, _) = extract_trace::<_, Fr>(
        &trace,
        trace_length,
        &bytecode,
        &io_device.memory_layout,
        r1cs_key.num_vars_padded,
    );

    let num_cycles = trace_length;
    let num_vars_padded = r1cs_key.num_vars_padded;

    (matrices, r1cs_witness, num_cycles, num_vars_padded)
}

/// Per-cycle check: Az[k] * Bz[k] == Cz[k] for all 22 constraints.
#[test]
fn r1cs_per_cycle_satisfaction() {
    let (matrices, witness, num_cycles, v_pad) = setup();

    let constraint_names = [
        "0:RamAddrEqRs1PlusImmIfLoadStore",
        "1:RamAddrEqZeroIfNotLoadStore",
        "2:RamReadEqRamWriteIfLoad",
        "3:RamReadEqRdWriteIfLoad",
        "4:Rs2EqRamWriteIfStore",
        "5:LeftLookupZeroUnlessAddSubMul",
        "6:LeftLookupEqLeftInputOtherwise",
        "7:RightLookupAdd",
        "8:RightLookupSub",
        "9:RightLookupEqProductIfMul",
        "10:RightLookupEqRightInputOtherwise",
        "11:AssertLookupOne",
        "12:RdWriteEqLookupIfWriteLookupToRd",
        "13:RdWriteEqPCPlusConstIfWritePCtoRD",
        "14:NextUnexpPCEqLookupIfShouldJump",
        "15:NextUnexpPCEqPCPlusImmIfShouldBranch",
        "16:NextUnexpPCUpdateOtherwise",
        "17:NextPCEqPCPlusOneIfInline",
        "18:MustStartSequenceFromBeginning",
        "19:Product",
        "20:ShouldBranch",
        "21:ShouldJump",
    ];

    let mut violations: Vec<(usize, usize, Fr, Fr, Fr)> = Vec::new();

    for c in 0..num_cycles {
        let w = &witness[c * v_pad..(c + 1) * v_pad];

        for k in 0..matrices.num_constraints {
            let az: Fr = matrices.a[k].iter().map(|&(j, coeff)| coeff * w[j]).sum();
            let bz: Fr = matrices.b[k].iter().map(|&(j, coeff)| coeff * w[j]).sum();
            let cz: Fr = matrices.c[k].iter().map(|&(j, coeff)| coeff * w[j]).sum();

            if az * bz != cz {
                violations.push((c, k, az, bz, cz));
            }
        }
    }

    if violations.is_empty() {
        eprintln!(
            "All {num_cycles} cycles × {} constraints satisfied.",
            matrices.num_constraints
        );
        return;
    }

    eprintln!("=== R1CS VIOLATIONS ({} total) ===", violations.len());

    // Print first 20 violations with details
    for &(c, k, az, bz, cz) in violations.iter().take(20) {
        let name = constraint_names.get(k).copied().unwrap_or("?");
        eprintln!(
            "  cycle={c}, constraint={name}: Az={az}, Bz={bz}, Cz={cz}, Az*Bz={}, diff={}",
            az * bz,
            az * bz - cz
        );

        // Print the witness values for this cycle
        let w = &witness[c * v_pad..(c + 1) * v_pad];
        eprintln!("    witness[CONST]={}", w[V_CONST]);

        // Print variables referenced by this constraint
        for &(j, coeff) in &matrices.a[k] {
            let var_name = var_name(j);
            eprintln!("    A: {var_name} (idx={j}) = {}, coeff={coeff}", w[j]);
        }
        for &(j, coeff) in &matrices.b[k] {
            let var_name = var_name(j);
            eprintln!("    B: {var_name} (idx={j}) = {}, coeff={coeff}", w[j]);
        }
        for &(j, coeff) in &matrices.c[k] {
            let var_name = var_name(j);
            eprintln!("    C: {var_name} (idx={j}) = {}, coeff={coeff}", w[j]);
        }
    }

    // Summarize which constraints are failing
    let mut by_constraint: std::collections::BTreeMap<usize, usize> = Default::default();
    for &(_, k, _, _, _) in &violations {
        *by_constraint.entry(k).or_default() += 1;
    }
    eprintln!("\n=== Summary by constraint ===");
    for (k, count) in &by_constraint {
        let name = constraint_names.get(*k).copied().unwrap_or("?");
        eprintln!("  {name}: {count} violations");
    }

    panic!(
        "{} R1CS violations across {} constraints",
        violations.len(),
        by_constraint.len()
    );
}

/// Check which of the 4 nonzero-base-eval constraints (3, 11, 14, 17) have violations.
/// These are the constraints identified by the outer uniskip diagnostic as having
/// nonzero Az*Bz at base domain points.
#[test]
fn r1cs_check_suspect_constraints() {
    let (matrices, witness, num_cycles, v_pad) = setup();

    let suspects = [
        (3, "RamReadEqRdWriteIfLoad"),
        (11, "AssertLookupOne"),
        (14, "NextUnexpPCEqLookupIfShouldJump"),
        (17, "NextPCEqPCPlusOneIfInline"),
    ];

    for (k, name) in suspects {
        let mut violation_cycles: Vec<usize> = Vec::new();

        for c in 0..num_cycles {
            let w = &witness[c * v_pad..(c + 1) * v_pad];
            let az: Fr = matrices.a[k].iter().map(|&(j, coeff)| coeff * w[j]).sum();
            let bz: Fr = matrices.b[k].iter().map(|&(j, coeff)| coeff * w[j]).sum();
            let cz: Fr = matrices.c[k].iter().map(|&(j, coeff)| coeff * w[j]).sum();

            if az * bz != cz {
                violation_cycles.push(c);
            }
        }

        if violation_cycles.is_empty() {
            eprintln!("Constraint {k} ({name}): OK (all {num_cycles} cycles)");
        } else {
            eprintln!(
                "Constraint {k} ({name}): {} violations at cycles {:?}",
                violation_cycles.len(),
                &violation_cycles[..violation_cycles.len().min(10)]
            );

            // Print details of first violation
            let c = violation_cycles[0];
            let w = &witness[c * v_pad..(c + 1) * v_pad];
            let az: Fr = matrices.a[k].iter().map(|&(j, coeff)| coeff * w[j]).sum();
            let bz: Fr = matrices.b[k].iter().map(|&(j, coeff)| coeff * w[j]).sum();
            eprintln!("  First violation (cycle {c}): Az={az}, Bz={bz}");
            for &(j, coeff) in &matrices.a[k] {
                eprintln!("    A: {} (idx={j}) = {}, coeff={coeff}", var_name(j), w[j]);
            }
            for &(j, coeff) in &matrices.b[k] {
                eprintln!("    B: {} (idx={j}) = {}, coeff={coeff}", var_name(j), w[j]);
            }
        }
    }
}

/// Compare initial/final RAM states between jolt-core and jolt-host construction.
#[test]
fn compare_ram_states() {
    use jolt_core::host;
    use jolt_core::poly::commitment::dory::DoryGlobals;
    use jolt_core::zkvm::ram::{compute_min_ram_K, gen_ram_memory_states, RAMPreprocessing};

    type CoreFr = ark_bn254::Fr;

    DoryGlobals::reset();

    let mut core_program = host::Program::new("muldiv-guest");
    let (bytecode, init_memory_state, _, _e_entry) = core_program.decode();
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
    let (_, _, core_memory, io_device) = core_program.trace(&inputs, &[], &[]);

    let ram_pp = RAMPreprocessing::preprocess(init_memory_state.clone());
    let ram_k = compute_min_ram_K(&ram_pp, &io_device.memory_layout);
    eprintln!("ram_k = {ram_k}");

    let (core_initial, core_final) =
        gen_ram_memory_states::<CoreFr>(ram_k, &ram_pp, &io_device, &core_memory);

    let mut host_program = Program::new("muldiv-guest");
    let (_, host_init_mem, _, _) = host_program.decode();
    let (_, _, host_memory, host_io) = host_program.trace(&inputs, &[], &[]);

    let (host_initial, host_final) =
        jolt_host::ram::build_ram_states(&host_init_mem, &host_memory, &host_io, ram_k);

    // Compare initial
    let mut init_diffs = 0;
    for k in 0..ram_k {
        if core_initial[k] != host_initial[k] {
            if init_diffs < 20 {
                eprintln!(
                    "  initial[{k}]: core={:#018x} host={:#018x}",
                    core_initial[k], host_initial[k]
                );
            }
            init_diffs += 1;
        }
    }
    let core_nz = core_initial.iter().filter(|&&v| v != 0).count();
    let host_nz = host_initial.iter().filter(|&&v| v != 0).count();
    eprintln!("Initial: {init_diffs}/{ram_k} diffs, nonzero: core={core_nz} host={host_nz}");

    // Compare final
    let mut final_diffs = 0;
    for k in 0..ram_k {
        if core_final[k] != host_final[k] {
            if final_diffs < 20 {
                eprintln!(
                    "  final[{k}]: core={:#018x} host={:#018x}",
                    core_final[k], host_final[k]
                );
            }
            final_diffs += 1;
        }
    }
    eprintln!("Final: {final_diffs}/{ram_k} diffs");

    assert_eq!(init_diffs, 0, "initial RAM state mismatch");
    assert_eq!(final_diffs, 0, "final RAM state mismatch");
}

fn var_name(idx: usize) -> &'static str {
    match idx {
        0 => "CONST",
        1 => "LeftInstructionInput",
        2 => "RightInstructionInput",
        3 => "Product",
        4 => "ShouldBranch",
        5 => "PC",
        6 => "UnexpandedPC",
        7 => "Imm",
        8 => "RamAddress",
        9 => "Rs1Value",
        10 => "Rs2Value",
        11 => "RdWriteValue",
        12 => "RamReadValue",
        13 => "RamWriteValue",
        14 => "LeftLookupOperand",
        15 => "RightLookupOperand",
        16 => "NextUnexpandedPC",
        17 => "NextPC",
        18 => "NextIsVirtual",
        19 => "NextIsFirstInSequence",
        20 => "LookupOutput",
        21 => "ShouldJump",
        22 => "FLAG_AddOperands",
        23 => "FLAG_SubtractOperands",
        24 => "FLAG_MultiplyOperands",
        25 => "FLAG_Load",
        26 => "FLAG_Store",
        27 => "FLAG_Jump",
        28 => "FLAG_WriteLookupOutputToRD",
        29 => "FLAG_VirtualInstruction",
        30 => "FLAG_Assert",
        31 => "FLAG_DoNotUpdateUnexpandedPC",
        32 => "FLAG_Advice",
        33 => "FLAG_IsCompressed",
        34 => "FLAG_IsFirstInSequence",
        35 => "FLAG_IsLastInSequence",
        36 => "Branch",
        37 => "NextIsNoop",
        _ => "???",
    }
}
