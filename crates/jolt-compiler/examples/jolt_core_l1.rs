//! Compile the full jolt-core protocol and dump the result.
//!
//! Builds a protocol with all sumcheck instances matching jolt-core
//! (including claim reductions), then runs the compiler to produce
//! a ProverSchedule and VerifierScript.
//!
//! Usage:
//!   cargo run --example jolt_core_l1 -p jolt-compiler              # human-readable
//!   cargo run --example jolt_core_l1 -p jolt-compiler -- --json    # JSON for L2

#![allow(non_snake_case, unused_variables, dead_code, clippy::print_stderr, clippy::print_stdout)]

use jolt_compiler::{
    compile, ChallengeSource, ClaimId, CompileParams, CompilerOutput, Expr, Objective, Poly,
    PolyKind, Protocol, ProverStep, PublicPoly, SolverConfig, Vertex,
};
use PolyKind::{Committed, Public, Virtual};

fn main() {
    let (protocol, _vertices) = build_protocol();

    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.iter().any(|a| a == "--dot") {
        let dot = jolt_compiler::dot::protocol_to_dot(&protocol);
        print!("{dot}");
        return;
    }

    let params = CompileParams {
        dim_sizes: vec![20, 8, 5, 8, 10, 4],
        field_size_bytes: 32,
        pcs_proof_size: 1600,
    };

    let config = SolverConfig {
        proof_size: Objective::Minimize,
        peak_memory: Objective::Ignore,
        prover_time: Objective::Ignore,
    };

    let info = jolt_compiler::analyze(&protocol).expect("validation failed");
    let output = compile(&protocol, &params, &config).expect("compilation failed");

    if args.iter().any(|a| a == "--json") {
        let json = serde_json::to_string_pretty(&output).expect("json serialize");
        println!("{json}");
        return;
    }

    print_human_readable(&protocol, &info, &output);
}

fn print_human_readable(
    protocol: &Protocol,
    info: &jolt_compiler::IRInfo,
    output: &CompilerOutput,
) {
    eprintln!("=== Protocol ===");
    eprintln!(
        "  vertices: {} ({} sumchecks, {} uniskips, {} evals)",
        protocol.vertices.len(),
        protocol.vertices.iter().filter(|v| matches!(v, Vertex::Sumcheck { domain_size: None, .. })).count(),
        protocol.vertices.iter().filter(|v| matches!(v, Vertex::Sumcheck { domain_size: Some(_), .. })).count(),
        protocol.vertices.iter().filter(|v| matches!(v, Vertex::Evaluate { .. })).count(),
    );
    eprintln!(
        "  polys: {} ({} committed, {} virtual, {} public)",
        protocol.polynomials.len(),
        protocol.polynomials.iter().filter(|p| matches!(p.kind, Committed)).count(),
        protocol.polynomials.iter().filter(|p| matches!(p.kind, Virtual)).count(),
        protocol.polynomials.iter().filter(|p| matches!(p.kind, Public(_))).count(),
    );
    eprintln!("  claims: {}", protocol.claims.len());
    eprintln!("  dims: {:?}", protocol.dim_names);
    eprintln!("  critical path: {}", info.critical_path);

    let polys = &output.polys;
    let challenges = &output.challenges;
    let sched = &output.schedule;

    eprintln!("\n=== Prover Schedule ===");
    eprintln!("  steps: {}", sched.steps.len());
    eprintln!("  kernels: {}", sched.kernels.len());
    eprintln!("  polys: {}", polys.len());
    eprintln!("  challenges: {}", challenges.len());
    eprintln!("  FS steps: {}, compute steps: {}", sched.fs_step_count(), sched.compute_step_count());

    let round_challenges = challenges
        .iter()
        .filter(|c| matches!(c.source, ChallengeSource::SumcheckRound { .. }))
        .count();
    eprintln!("  round challenges: {round_challenges}");

    for (i, k) in sched.kernels.iter().enumerate() {
        let input_names: Vec<&str> = k.inputs.iter().map(|&pi| polys[pi].name.as_str()).collect();
        eprintln!(
            "  kernel {i}: degree={}, rounds={}, {} inputs [{}]",
            k.degree, k.num_rounds, k.inputs.len(), input_names.join(", ")
        );
    }

    let script = &output.script;
    eprintln!("\n=== Verifier Script ===");
    eprintln!("  stages: {}", script.stages.len());
    for (i, stage) in script.stages.iter().enumerate() {
        eprintln!(
            "  stage {i}: rounds={}, degree={}, {} evals, {} post-squeeze",
            stage.num_rounds, stage.degree, stage.evaluations.len(), stage.post_squeeze.len(),
        );
    }

    eprintln!("\n=== Steps ({}) ===", sched.steps.len());
    for (i, step) in sched.steps.iter().enumerate() {
        match step {
            ProverStep::AppendCommitments { polys: committed } => {
                let names: Vec<&str> = committed.iter().map(|&p| polys[p].name.as_str()).collect();
                eprintln!("  [{i:>4}] FS  AppendCommitments({} polys): {}", committed.len(), names.join(", "));
            }
            ProverStep::AppendRoundPoly { kernel, num_coeffs } => {
                eprintln!("  [{i:>4}] FS  AppendRoundPoly(kernel={kernel}, {num_coeffs} coeffs)");
            }
            ProverStep::AppendScalars { evals } => {
                eprintln!("  [{i:>4}] FS  AppendScalars({} evals)", evals.len());
            }
            ProverStep::Squeeze { challenge } => {
                let ch = &challenges[*challenge];
                eprintln!("  [{i:>4}] FS  Squeeze({}: {})", challenge, ch.name);
            }
            ProverStep::Materialize { poly } => {
                eprintln!("  [{i:>4}]     Materialize({})", polys[*poly].name);
            }
            ProverStep::SumcheckRound { kernel, round, num_vars_remaining } => {
                eprintln!("  [{i:>4}]     SumcheckRound(k{kernel}r{round}, vars={num_vars_remaining})");
            }
            ProverStep::Bind { polys: bound, challenge, order } => {
                eprintln!("  [{i:>4}]     Bind({} polys, ch={}, {:?})", bound.len(), challenge, order);
            }
            ProverStep::Evaluate { poly } => {
                eprintln!("  [{i:>4}]     Evaluate({})", polys[*poly].name);
            }
        }
    }
}

/// Vertex indices tracked during protocol construction.
struct Vertices {
    outer_uniskip: usize,
    outer_remaining: usize,
    product_uniskip: usize,
    product_remaining: usize,
    ram_rw: usize,
    instruction_claim_reduction: usize,
    raf_eval: usize,
    output_check: usize,
    shift: usize,
    inst_input: usize,
    registers_claim_reduction: usize,
    reg_rw: usize,
    ram_val_check: usize,
    inst_read_raf: usize,
    ram_ra_reduction: usize,
    reg_val_eval: usize,
    bytecode_raf: Vec<usize>,
    booleanity: usize,
    hamming_booleanity: usize,
    ram_ra_virtual: usize,
    inst_ra_virtual: usize,
    inc_reduction: usize,
    advice_trusted_cycle: usize,
    advice_untrusted_cycle: usize,
    hamming_weight_reduction: usize,
    advice_trusted_addr: usize,
    advice_untrusted_addr: usize,
    evals_after_outer: Vec<usize>,
    evals_after_product: Vec<usize>,
    evals_after_shift: Vec<usize>,
    evals_after_inst_input: Vec<usize>,
    evals_after_ram_rw: Vec<usize>,
    evals_after_reg_rw: Vec<usize>,
    evals_after_inst_read_raf: Vec<usize>,
    evals_after_ram_val: Vec<usize>,
    evals_after_reg_val: Vec<usize>,
}

/// Build the L0 protocol with ALL sumcheck instances matching jolt-core.
/// Includes claim reductions that are absent from jolt_l0.rs.
fn build_protocol() -> (Protocol, Vertices) {
    let mut p = Protocol::new();
    const D: usize = 4; // one-hot chunk count (matches L1 example for compact output)

    let log_T = p.dim("log_T");
    let log_K = p.dim("log_K");
    let gamma = p.challenge("gamma");

    // ========== Polynomial declarations ==========

    // Public
    let eq_T = p.poly("eq_T", &[log_T], Public(PublicPoly::Eq(None)));
    let eq_K = p.poly("eq_K", &[log_K], Public(PublicPoly::Eq(None)));
    let lt_T = p.poly("lt_T", &[log_T], Public(PublicPoly::Lt(None)));

    // Public — preprocessing (known to both prover and verifier)
    let io_mask = p.poly("IoMask", &[log_K], Public(PublicPoly::Preprocessed));
    let val_io = p.poly("ValIo", &[log_K], Public(PublicPoly::Preprocessed));

    // Committed
    let ram_inc = p.poly("RamInc", &[log_T], Committed);
    let rd_inc = p.poly("RdInc", &[log_T], Committed);
    let trusted = p.poly("TrustedAdvice", &[log_T], Committed);
    let untrusted = p.poly("UntrustedAdvice", &[log_T], Committed);
    let inst_ra: Vec<Poly> = (0..D)
        .map(|d| p.poly(&format!("InstRa_{d}"), &[log_K, log_T], Committed))
        .collect();
    let bytecode_ra: Vec<Poly> = (0..D)
        .map(|d| p.poly(&format!("BytecodeRa_{d}"), &[log_K, log_T], Committed))
        .collect();
    let ram_ra: Vec<Poly> = (0..D)
        .map(|d| p.poly(&format!("RamRa_{d}"), &[log_K, log_T], Committed))
        .collect();

    // Virtual — Spartan R1CS
    let az = p.poly("Az", &[log_T], Virtual);
    let bz = p.poly("Bz", &[log_T], Virtual);
    let left = p.poly("ProductLeft", &[log_T], Virtual);
    let right = p.poly("ProductRight", &[log_T], Virtual);

    // Virtual — trace polys
    let pc = p.poly("PC", &[log_T], Virtual);
    let unexpanded_pc = p.poly("UnexpandedPC", &[log_T], Virtual);
    let is_noop = p.poly("IsNoop", &[log_T], Virtual);
    let is_virtual = p.poly("IsVirtual", &[log_T], Virtual);
    let is_first = p.poly("IsFirstInSeq", &[log_T], Virtual);
    let is_rs1 = p.poly("IsRs1", &[log_T], Virtual);
    let is_rs2 = p.poly("IsRs2", &[log_T], Virtual);
    let is_pc_flag = p.poly("IsPcFlag", &[log_T], Virtual);
    let is_imm_flag = p.poly("IsImmFlag", &[log_T], Virtual);
    let rs1_val = p.poly("Rs1Val", &[log_T], Virtual);
    let rs2_val = p.poly("Rs2Val", &[log_T], Virtual);
    let imm_val = p.poly("Imm", &[log_T], Virtual);

    // Virtual — derived
    let hamming = p.poly("HammingWeight", &[log_T], Virtual);

    // Virtual — RAM
    let ram_combined_ra = p.poly("RamCombinedRa", &[log_K, log_T], Virtual);
    let ram_val = p.poly("RamVal", &[log_K, log_T], Virtual);
    let ram_val_final = p.poly("RamValFinal", &[log_K], Virtual);
    let ram_wa = p.poly("RamWA", &[log_T], Virtual);
    let ram_unmap = p.poly("RamUnmap", &[log_K], Public(PublicPoly::Preprocessed));
    let ram_rv = p.poly("RamReadValue", &[log_T], Virtual);
    let ram_wv = p.poly("RamWriteValue", &[log_T], Virtual);
    let ram_init = p.poly("RamInit", &[log_K], Public(PublicPoly::Preprocessed));
    let ram_address = p.poly("RamAddress", &[log_T], Virtual);

    // Virtual — registers
    let reg_wa = p.poly("RegWA", &[log_K, log_T], Virtual);
    let reg_ra_rs1 = p.poly("RegRaRs1", &[log_K, log_T], Virtual);
    let reg_ra_rs2 = p.poly("RegRaRs2", &[log_K, log_T], Virtual);
    let reg_val = p.poly("RegVal", &[log_K, log_T], Virtual);
    let rd_wa = p.poly("RdWA", &[log_T], Virtual);
    let rd_wv = p.poly("RdWriteValue", &[log_T], Virtual);
    let rs1_rv = p.poly("Rs1ReadValue", &[log_T], Virtual);
    let rs2_rv = p.poly("Rs2ReadValue", &[log_T], Virtual);

    // Virtual — lookups
    let lookup_table = p.poly("LookupTable", &[log_K], Public(PublicPoly::Preprocessed));
    let lookup_output = p.poly("LookupOutput", &[log_T], Virtual);
    let left_op = p.poly("LeftLookupOp", &[log_T], Virtual);
    let right_op = p.poly("RightLookupOp", &[log_T], Virtual);
    let left_inst_input = p.poly("LeftInstructionInput", &[log_T], Virtual);
    let right_inst_input = p.poly("RightInstructionInput", &[log_T], Virtual);

    // Virtual — RAF
    let ram_raf_ra = p.poly("RamRafRa", &[log_K], Virtual);
    let inst_raf_ra = p.poly("InstRafRa", &[log_K, log_T], Virtual);
    let bytecode_raf_ra = p.poly("BytecodeRafRa", &[log_K, log_T], Virtual);
    let bc_table: Vec<Poly> = (0..5)
        .map(|s| p.poly(&format!("BcTable_{s}"), &[log_K], Public(PublicPoly::Preprocessed)))
        .collect();
    let bc_rv: Vec<Poly> = (0..5)
        .map(|s| p.poly(&format!("BcReadVal_{s}"), &[log_T], Virtual))
        .collect();

    // Virtual — product factors (from ALL_R1CS_INPUTS)
    let product_poly = p.poly("Product", &[log_T], Virtual);
    let should_branch = p.poly("ShouldBranch", &[log_T], Virtual);
    let should_jump = p.poly("ShouldJump", &[log_T], Virtual);

    // Virtual — OpFlags (circuit flags, 12 not already declared as trace polys)
    // is_virtual = OpFlags(VirtualInstruction), is_first = OpFlags(IsFirstInSequence)
    let op_add = p.poly("OpAddOperands", &[log_T], Virtual);
    let op_sub = p.poly("OpSubtractOperands", &[log_T], Virtual);
    let op_mul = p.poly("OpMultiplyOperands", &[log_T], Virtual);
    let op_load = p.poly("OpLoad", &[log_T], Virtual);
    let op_store = p.poly("OpStore", &[log_T], Virtual);
    let op_jump = p.poly("OpJump", &[log_T], Virtual);
    let op_write_rd = p.poly("OpWriteLookupOutputToRD", &[log_T], Virtual);
    let op_assert = p.poly("OpAssert", &[log_T], Virtual);
    let op_no_update_pc = p.poly("OpDoNotUpdatePC", &[log_T], Virtual);
    let op_advice = p.poly("OpAdvice", &[log_T], Virtual);
    let op_is_compressed = p.poly("OpIsCompressed", &[log_T], Virtual);
    let op_is_last_in_seq = p.poly("OpIsLastInSequence", &[log_T], Virtual);

    // All 14 OpFlags for bulk evaluation at cache_openings points
    let all_op_flags: [Poly; 14] = [
        op_add, op_sub, op_mul, op_load, op_store, op_jump, op_write_rd,
        is_virtual, op_assert, op_no_update_pc, op_advice, op_is_compressed,
        is_first, op_is_last_in_seq,
    ];

    // Virtual — InstructionFlags (not already declared as trace polys)
    let inst_flag_branch = p.poly("InstFlagBranch", &[log_T], Virtual);

    // Virtual — RAF flags (cached by InstructionReadRaf, consumed by BytecodeReadRaf)
    let inst_raf_flag = p.poly("InstructionRafFlag", &[log_T], Virtual);
    const N_TABLES: usize = 8;
    let lookup_table_flags: Vec<Poly> = (0..N_TABLES)
        .map(|i| p.poly(&format!("LookupTableFlag_{i}"), &[log_T], Virtual))
        .collect();

    // Virtual — HW reduction pushforward polys: G_i(k) = Σ_j eq(r_cycle,j) · ra_i(k,j)
    let hw_G: Vec<Poly> = (0..(3 * D))
        .map(|i| p.poly(&format!("G_{i}"), &[log_K], Virtual))
        .collect();

    // ========== STAGE 1: Outer Spartan ==========
    const OUTER_UNISKIP_DOMAIN: usize = 10;
    const PRODUCT_UNISKIP_DOMAIN: usize = 3;

    let outer_uniskip_v = p.vertices.len();
    let outer_mid = p.uniskip_round(eq_T * az * bz, 0, &[log_T], OUTER_UNISKIP_DOMAIN);

    let outer_remaining_v = p.vertices.len();
    let outer = p.sumcheck(eq_T * az * bz, outer_mid, &[log_T]);

    // ========== Evals after outer (available to Stage 2) ==========
    // jolt-core caches ALL R1CS input polynomial evaluations at r_outer.
    // We model the ones consumed by downstream sumchecks.
    let mut evals_after_outer = Vec::new();

    // For InstructionClaimReduction input
    let lookup_output_at_outer = p.evaluate(lookup_output, outer[0]);
    evals_after_outer.push(p.vertices.len() - 1);
    let left_op_at_outer = p.evaluate(left_op, outer[0]);
    evals_after_outer.push(p.vertices.len() - 1);
    let right_op_at_outer = p.evaluate(right_op, outer[0]);
    evals_after_outer.push(p.vertices.len() - 1);
    let left_inst_at_outer = p.evaluate(left_inst_input, outer[0]);
    evals_after_outer.push(p.vertices.len() - 1);
    let right_inst_at_outer = p.evaluate(right_inst_input, outer[0]);
    evals_after_outer.push(p.vertices.len() - 1);

    // For RamRW input
    let ram_rv_at_outer = p.evaluate(ram_rv, outer[0]);
    evals_after_outer.push(p.vertices.len() - 1);
    let ram_wv_at_outer = p.evaluate(ram_wv, outer[0]);
    evals_after_outer.push(p.vertices.len() - 1);

    // For RegistersClaimReduction input
    let rd_wv_at_outer = p.evaluate(rd_wv, outer[0]);
    evals_after_outer.push(p.vertices.len() - 1);
    let rs1_rv_at_outer = p.evaluate(rs1_rv, outer[0]);
    evals_after_outer.push(p.vertices.len() - 1);
    let rs2_rv_at_outer = p.evaluate(rs2_rv, outer[0]);
    evals_after_outer.push(p.vertices.len() - 1);

    // For RafEvaluation input
    let ram_address_at_outer = p.evaluate(ram_address, outer[0]);
    evals_after_outer.push(p.vertices.len() - 1);

    // For product uniskip input (Lagrange-weighted sum of 3 product evals)
    let product_at_outer = p.evaluate(product_poly, outer[0]);
    evals_after_outer.push(p.vertices.len() - 1);
    let should_branch_at_outer = p.evaluate(should_branch, outer[0]);
    evals_after_outer.push(p.vertices.len() - 1);
    let should_jump_at_outer = p.evaluate(should_jump, outer[0]);
    evals_after_outer.push(p.vertices.len() - 1);

    // For BytecodeReadRaf rv_claim_1: Imm + 14 OpFlags at r_outer
    // (is_virtual and is_first already go through evals_after_product for shift)
    let imm_at_outer = p.evaluate(imm_val, outer[0]);
    evals_after_outer.push(p.vertices.len() - 1);
    for &flag in &all_op_flags {
        let _ = p.evaluate(flag, outer[0]);
        evals_after_outer.push(p.vertices.len() - 1);
    }

    // For BytecodeReadRaf raf_claim: PC at r_outer
    let pc_at_outer = p.evaluate(pc, outer[0]);
    evals_after_outer.push(p.vertices.len() - 1);

    // ========== STAGE 2: Product + RamRW + InstructionClaimReduction + RafEval + OutputCheck ==========

    // Product uni-skip: input = Σ_i L_i(τ_high) · base_evals[i]
    // where base_evals = [Product(r_outer), ShouldBranch(r_outer), ShouldJump(r_outer)]
    // Lagrange coefficients are challenge-derived; modeled as γ-weighted sum at L0.
    let product_uniskip_v = p.vertices.len();
    let product_mid = p.uniskip_round(
        eq_T * left * right,
        product_at_outer + gamma * should_branch_at_outer + gamma.pow(2) * should_jump_at_outer,
        &[log_T],
        PRODUCT_UNISKIP_DOMAIN,
    );

    // Product remaining
    let product_remaining_v = p.vertices.len();
    let product = p.sumcheck(eq_T * left * right, product_mid, &[log_T]);

    // RamReadWriteChecking: eq · ra · (Val + γ·(Val + inc)) = rv + γ·wv
    let ram_rw_v = p.vertices.len();
    let ram_rw = p.sumcheck(
        eq_T * ram_combined_ra * (ram_val + gamma * (ram_val + ram_inc)),
        ram_rv_at_outer + gamma * ram_wv_at_outer,
        &[log_K, log_T],
    );

    // InstructionLookupsClaimReduction: eq(r_outer) · (output + γ·left + γ²·right + γ³·left_inst + γ⁴·right_inst)
    let eq_outer = p.poly(
        "eq_outer",
        &[log_T],
        Public(PublicPoly::Eq(Some(outer[0]))),
    );
    let inst_claim_reduction_v = p.vertices.len();
    let inst_claim_reduction = p.sumcheck(
        eq_outer
            * (lookup_output
                + gamma * left_op
                + gamma.pow(2) * right_op
                + gamma.pow(3) * left_inst_input
                + gamma.pow(4) * right_inst_input),
        lookup_output_at_outer
            + gamma * left_op_at_outer
            + gamma.pow(2) * right_op_at_outer
            + gamma.pow(3) * left_inst_at_outer
            + gamma.pow(4) * right_inst_at_outer,
        &[log_T],
    );

    // RafEvaluation: eq · ra · unmap = RamAddress(r_outer)
    let raf_eval_v = p.vertices.len();
    let _raf_eval = p.sumcheck(
        eq_K * ram_raf_ra * ram_unmap,
        ram_address_at_outer,
        &[log_K],
    );

    // OutputCheck: eq · io_mask · (val_final − val_io) = 0
    let output_check_v = p.vertices.len();
    let _output_check =
        p.sumcheck(eq_K * io_mask * (ram_val_final - val_io), 0, &[log_K]);

    // ========== Evals after product (available to Stage 3) ==========
    let mut evals_after_product = Vec::new();

    // For Shift input (4 at outer, 1 at product)
    let next_unexpanded_pc = p.evaluate(unexpanded_pc, outer[0]);
    evals_after_product.push(p.vertices.len() - 1);
    let next_pc = p.evaluate(pc, outer[0]);
    evals_after_product.push(p.vertices.len() - 1);
    let next_is_virtual = p.evaluate(is_virtual, outer[0]);
    evals_after_product.push(p.vertices.len() - 1);
    let next_is_first = p.evaluate(is_first, outer[0]);
    evals_after_product.push(p.vertices.len() - 1);
    let next_is_noop = p.evaluate(is_noop, product[0]);
    evals_after_product.push(p.vertices.len() - 1);

    // For InstructionInput
    let left_inst_eval = p.evaluate(left_inst_input, product[0]);
    evals_after_product.push(p.vertices.len() - 1);
    let right_inst_eval = p.evaluate(right_inst_input, product[0]);
    evals_after_product.push(p.vertices.len() - 1);

    // From PRODUCT_UNIQUE_FACTOR_VIRTUALS cache_openings (for BytecodeReadRaf rv_claim_2)
    let _op_jump_at_product = p.evaluate(op_jump, product[0]);
    evals_after_product.push(p.vertices.len() - 1);
    let _op_write_rd_at_product = p.evaluate(op_write_rd, product[0]);
    evals_after_product.push(p.vertices.len() - 1);
    let _lookup_output_at_product = p.evaluate(lookup_output, product[0]);
    evals_after_product.push(p.vertices.len() - 1);
    let _inst_flag_branch_at_product = p.evaluate(inst_flag_branch, product[0]);
    evals_after_product.push(p.vertices.len() - 1);
    let _is_virtual_at_product = p.evaluate(is_virtual, product[0]);
    evals_after_product.push(p.vertices.len() - 1);

    // ========== STAGE 3: Shift + InstructionInput + RegistersClaimReduction ==========

    let eq1_outer = p.poly(
        "eq1_outer",
        &[log_T],
        Public(PublicPoly::EqPlusOne(Some(outer[0]))),
    );
    let eq1_product = p.poly(
        "eq1_product",
        &[log_T],
        Public(PublicPoly::EqPlusOne(Some(product[0]))),
    );

    let shift_v = p.vertices.len();
    let shift = p.sumcheck(
        eq1_outer
            * (unexpanded_pc
                + gamma * pc
                + gamma.pow(2) * is_virtual
                + gamma.pow(3) * is_first)
            + eq1_product * (gamma.pow(4) - gamma.pow(4) * is_noop),
        next_unexpanded_pc
            + gamma * next_pc
            + gamma.pow(2) * next_is_virtual
            + gamma.pow(3) * next_is_first
            + gamma.pow(4)
            - gamma.pow(4) * next_is_noop,
        &[log_T],
    );

    let inst_input_v = p.vertices.len();
    let inst_input = p.sumcheck(
        eq_T
            * (is_rs2 * rs2_val
                + is_imm_flag * imm_val
                + gamma * (is_rs1 * rs1_val + is_pc_flag * unexpanded_pc)),
        right_inst_eval + gamma * left_inst_eval,
        &[log_T],
    );

    // RegistersClaimReduction: eq(r_outer) · (rd_wv + γ·rs1_rv + γ²·rs2_rv)
    let reg_claim_reduction_v = p.vertices.len();
    let _reg_claim_reduction = p.sumcheck(
        eq_outer * (rd_wv + gamma * rs1_rv + gamma.pow(2) * rs2_rv),
        rd_wv_at_outer + gamma * rs1_rv_at_outer + gamma.pow(2) * rs2_rv_at_outer,
        &[log_T],
    );

    // ========== Evals after shift (Shift cache_openings, available to Stage 4+) ==========
    // Shift sumcheck claims already cover: unexpanded_pc, pc, is_virtual, is_first, is_noop.
    // These are terminal claims for PCS opening. The shift→bytecodeRaf dependency is captured
    // through the bc_rv evaluate vertices and eq poly anchors.
    let evals_after_shift = Vec::new();

    // ========== Evals after inst_input (available to Stage 4) ==========
    let mut evals_after_inst_input = Vec::new();

    // RegistersRW needs rd_wv, rs1_rv, rs2_rv at the registers_claim_reduction point
    let rd_wv_eval = p.evaluate(rd_wv, _reg_claim_reduction[0]);
    evals_after_inst_input.push(p.vertices.len() - 1);
    let rs1_rv_eval = p.evaluate(rs1_rv, _reg_claim_reduction[0]);
    evals_after_inst_input.push(p.vertices.len() - 1);
    let rs2_rv_eval = p.evaluate(rs2_rv, _reg_claim_reduction[0]);
    evals_after_inst_input.push(p.vertices.len() - 1);

    // ========== STAGE 4: RegistersRW + RamValCheck ==========

    let reg_rw_v = p.vertices.len();
    let reg_rw = p.sumcheck(
        eq_T
            * (reg_wa * (rd_inc + reg_val)
                + gamma * reg_ra_rs1 * reg_val
                + gamma.pow(2) * reg_ra_rs2 * reg_val),
        rd_wv_eval + gamma * (rs1_rv_eval + gamma * rs2_rv_eval),
        &[log_K, log_T],
    );

    // RamValCheck input: (val − init) + γ·(final − init), from ram_rw evals
    let ram_val_eval = p.evaluate(ram_val, ram_rw[0]);
    let ram_init_eval = p.evaluate(ram_init, ram_rw[0]);
    let ram_val_final_eval = p.evaluate(ram_val_final, ram_rw[0]);

    let ram_val_check_v = p.vertices.len();
    let ram_val_sc = p.sumcheck(
        ram_inc * ram_wa * (lt_T + gamma),
        (ram_val_eval - ram_init_eval) + gamma * (ram_val_final_eval - ram_init_eval),
        &[log_T],
    );

    // ========== Evals after ram_rw and reg_rw (available to Stage 5) ==========
    let mut evals_after_ram_rw = Vec::new();

    let ram_ra_evals: Vec<ClaimId> = ram_ra
        .iter()
        .map(|&ra| {
            let c = p.evaluate(ra, ram_rw[0]);
            evals_after_ram_rw.push(p.vertices.len() - 1);
            c
        })
        .collect();

    let mut evals_after_reg_rw = Vec::new();
    let reg_val_eval = p.evaluate(reg_val, reg_rw[0]);
    evals_after_reg_rw.push(p.vertices.len() - 1);

    // BytecodeReadRaf rv_claim_4 needs RdWa at the reg_rw point (1D at cycle portion)
    let _rd_wa_at_reg_rw = p.evaluate(rd_wa, reg_rw[0]);
    evals_after_reg_rw.push(p.vertices.len() - 1);

    // ========== STAGE 5: InstructionReadRaf + RamRaReduction + RegistersValEval ==========

    // InstructionReadRaf: eq · ra · table = output + γ·left + γ²·right
    // Input from InstructionClaimReduction evals
    let lookup_output_reduced = p.evaluate(lookup_output, inst_claim_reduction[0]);
    let left_op_reduced = p.evaluate(left_op, inst_claim_reduction[0]);
    let right_op_reduced = p.evaluate(right_op, inst_claim_reduction[0]);

    let inst_read_raf_v = p.vertices.len();
    let inst_read_raf = p.sumcheck(
        eq_K * inst_raf_ra * lookup_table,
        lookup_output_reduced + gamma * left_op_reduced + gamma.pow(2) * right_op_reduced,
        &[log_K, log_T],
    );

    // RamRaClaimReduction: (eq_raf + γ·eq_rw + γ²·eq_val) · RamRa
    // Reduces RamRa from 3 upstream points (raf_eval, ram_rw, ram_val) to one point
    // Input: claim_raf + γ·claim_rw + γ²·claim_val
    let ram_ra_at_raf = p.evaluate(ram_ra[0], _raf_eval[0]);
    let ram_ra_at_val = p.evaluate(ram_ra[0], ram_val_sc[0]);

    let eq_raf_cycle = p.poly(
        "eq_raf_cycle",
        &[log_T],
        Public(PublicPoly::Eq(Some(_raf_eval[0]))),
    );
    let eq_rw_cycle = p.poly(
        "eq_rw_cycle",
        &[log_T],
        Public(PublicPoly::Eq(Some(ram_rw[0]))),
    );
    let eq_val_cycle = p.poly(
        "eq_val_cycle",
        &[log_T],
        Public(PublicPoly::Eq(Some(ram_val_sc[0]))),
    );

    let ram_ra_reduction_v = p.vertices.len();
    let ram_ra_reduction = p.sumcheck(
        (eq_raf_cycle + gamma * eq_rw_cycle + gamma.pow(2) * eq_val_cycle) * ram_ra[0],
        ram_ra_at_raf + gamma * ram_ra_evals[0] + gamma.pow(2) * ram_ra_at_val,
        &[log_T],
    );

    // RegistersValEvaluation: inc · wa · LT = registers_val_claim
    let reg_val_eval_v = p.vertices.len();
    let reg_val_sc = p.sumcheck(lt_T * rd_inc * rd_wa, reg_val_eval, &[log_T]);

    // ========== Evals after inst_read_raf, ram_val, reg_val (available to Stage 6) ==========
    let mut evals_after_inst_read_raf = Vec::new();

    let inst_ra_evals: Vec<ClaimId> = inst_ra
        .iter()
        .map(|&ra| {
            let c = p.evaluate(ra, inst_read_raf[0]);
            evals_after_inst_read_raf.push(p.vertices.len() - 1);
            c
        })
        .collect();

    // BytecodeReadRaf rv_claim_5: InstructionRafFlag + LookupTableFlags at inst_read_raf point
    let _inst_raf_flag_at_read_raf = p.evaluate(inst_raf_flag, inst_read_raf[0]);
    evals_after_inst_read_raf.push(p.vertices.len() - 1);
    for &flag in &lookup_table_flags {
        let _ = p.evaluate(flag, inst_read_raf[0]);
        evals_after_inst_read_raf.push(p.vertices.len() - 1);
    }

    let evals_after_ram_val = Vec::new(); // ram_val evals already captured above

    // BytecodeReadRaf rv_claim_5: RdWa at reg_val point
    let mut evals_after_reg_val = Vec::new();
    let _rd_wa_at_reg_val = p.evaluate(rd_wa, reg_val_sc[0]);
    evals_after_reg_val.push(p.vertices.len() - 1);

    // ========== STAGE 6: BytecodeRaf + Booleanity + HammingBool + RamRaVirt +
    //                     InstRaVirt + IncReduction + Advice ==========

    // BytecodeRaf: 5 sumchecks (fused in jolt-core)
    // Capture first claim for bytecode RA virt evals used by HW reduction
    let bc_upstream = [outer[0], product[0], shift[0], reg_rw[0], reg_val_sc[0]];
    let mut bytecode_raf_vs = Vec::new();
    let mut bytecode_raf_first_claim: Option<ClaimId> = None;
    for s in 0..5 {
        let eq_s = p.poly(
            &format!("BcEq_{s}"),
            &[log_T],
            Public(PublicPoly::Eq(Some(bc_upstream[s]))),
        );
        let rv_eval = p.evaluate(bc_rv[s], bc_upstream[s]);
        let v_idx = p.vertices.len();
        let bc_claims = p.sumcheck(
            eq_s * bytecode_raf_ra * bc_table[s],
            rv_eval,
            &[log_K, log_T],
        );
        bytecode_raf_vs.push(v_idx);
        if s == 0 {
            bytecode_raf_first_claim = Some(bc_claims[0]);
        }
    }
    let bc_raf_claim = bytecode_raf_first_claim.unwrap();

    // Bytecode RA virt evals: each bytecode_ra[d] at the bytecode RAF point
    // (for HW reduction virtualization claims)
    let bc_ra_virt_claims: Vec<ClaimId> = bytecode_ra
        .iter()
        .map(|&ra| p.evaluate(ra, bc_raf_claim))
        .collect();

    // Booleanity: γ-batched eq · (ra² − ra) = 0
    // RA polys are 2D [log_K, log_T]; booleanity binds both dimensions.
    let eq_bool = p.poly("eq_bool", &[log_K, log_T], Public(PublicPoly::Eq(None)));
    let all_ra: Vec<Poly> = inst_ra
        .iter()
        .chain(bytecode_ra.iter())
        .chain(ram_ra.iter())
        .copied()
        .collect();
    let mut bool_comp = eq_bool * (all_ra[0] * all_ra[0] - all_ra[0]);
    for (i, &ra) in all_ra.iter().enumerate().skip(1) {
        bool_comp = bool_comp + gamma.pow(i as u32) * eq_bool * (ra * ra - ra);
    }
    let booleanity_v = p.vertices.len();
    let booleanity_claims = p.sumcheck(bool_comp, 0, &[log_K, log_T]);

    // HammingBooleanity: eq · (H² − H) = 0
    let hamming_booleanity_v = p.vertices.len();
    let hamming_bool_claims = p.sumcheck(eq_T * (hamming * hamming - hamming), 0, &[log_T]);

    // RamRaVirtual: eq · Π ra_d = Π ra_d(r_reduced)
    let ram_ra_product = ram_ra
        .iter()
        .skip(1)
        .fold(Expr::from(eq_T) * Expr::from(ram_ra[0]), |acc, &ra| {
            acc * ra
        });
    let ram_ra_at_reduction: Vec<ClaimId> = ram_ra
        .iter()
        .map(|&ra| p.evaluate(ra, ram_ra_reduction[0]))
        .collect();
    let mut ram_ra_virt_input: Expr = ram_ra_at_reduction[0].into();
    for &eval in ram_ra_at_reduction.iter().skip(1) {
        ram_ra_virt_input = ram_ra_virt_input * eval;
    }
    let ram_ra_virtual_v = p.vertices.len();
    let ram_ra_virt_claims = p.sumcheck(ram_ra_product, ram_ra_virt_input, &[log_T]);

    // InstructionRaVirtual: eq · Σ γⁱ · ra_i = Σ γⁱ · ra_i(r)
    let mut ra_comp = eq_T * inst_ra[0];
    for (d, &ra) in inst_ra.iter().enumerate().skip(1) {
        ra_comp = ra_comp + eq_T * gamma.pow(d as u32) * ra;
    }
    let mut ra_input: Expr = inst_ra_evals[0].into();
    for (d, &eval) in inst_ra_evals.iter().enumerate().skip(1) {
        ra_input = ra_input + gamma.pow(d as u32) * eval;
    }
    let inst_ra_virtual_v = p.vertices.len();
    let inst_ra_virt_claims = p.sumcheck(ra_comp, ra_input, &[log_T]);

    // IncClaimReduction: reduces RamInc + RdInc from 4 upstream points
    // RamInc from: ram_rw (Stage 2) and ram_val_check (Stage 4)
    // RdInc from: reg_rw (Stage 4) and reg_val_eval (Stage 5)
    let eq_inc_rw = p.poly(
        "eq_inc_rw",
        &[log_T],
        Public(PublicPoly::Eq(Some(ram_rw[0]))),
    );
    let eq_inc_val = p.poly(
        "eq_inc_val",
        &[log_T],
        Public(PublicPoly::Eq(Some(ram_val_sc[0]))),
    );
    let eq_inc_reg_rw = p.poly(
        "eq_inc_reg_rw",
        &[log_T],
        Public(PublicPoly::Eq(Some(reg_rw[0]))),
    );
    let eq_inc_reg_val = p.poly(
        "eq_inc_reg_val",
        &[log_T],
        Public(PublicPoly::Eq(Some(reg_val_sc[0]))),
    );

    let ram_inc_at_rw = p.evaluate(ram_inc, ram_rw[0]);
    let ram_inc_at_val = p.evaluate(ram_inc, ram_val_sc[0]);
    let rd_inc_at_reg_rw = p.evaluate(rd_inc, reg_rw[0]);
    let rd_inc_at_reg_val = p.evaluate(rd_inc, reg_val_sc[0]);

    let inc_reduction_v = p.vertices.len();
    let _inc_reduction = p.sumcheck(
        ram_inc * (eq_inc_rw + gamma * eq_inc_val)
            + gamma.pow(2) * rd_inc * (eq_inc_reg_rw + gamma * eq_inc_reg_val),
        ram_inc_at_rw
            + gamma * ram_inc_at_val
            + gamma.pow(2) * rd_inc_at_reg_rw
            + gamma.pow(3) * rd_inc_at_reg_val,
        &[log_T],
    );

    // Advice claim reductions — two-phase: cycle (Stage 6) → address (Stage 7)
    // Phase 1 (cycle): eq(r_cycle_val) · advice → intermediate claim
    // Phase 2 (address): eq(r_addr) · advice → final opening claim
    let trusted_eval = p.evaluate(trusted, ram_val_sc[0]);
    let untrusted_eval = p.evaluate(untrusted, ram_val_sc[0]);
    let eq_advice_cycle = p.poly(
        "eq_advice_cycle",
        &[log_T],
        Public(PublicPoly::Eq(Some(ram_val_sc[0]))),
    );
    let advice_trusted_cycle_v = p.vertices.len();
    let trusted_mid = p.sumcheck(eq_advice_cycle * trusted, trusted_eval, &[log_T]);
    let advice_untrusted_cycle_v = p.vertices.len();
    let untrusted_mid = p.sumcheck(eq_advice_cycle * untrusted, untrusted_eval, &[log_T]);

    // ========== STAGE 7: HammingWeightClaimReduction + Advice address phases ==========
    // Fused reduction: HW + Booleanity + Virtualization for all RA polys
    // Operates over address variables only (log_K) using pushforward polys G_i.
    //
    // Composition: Σ_i G_i(k) · [γ^{3i} + γ^{3i+1}·eq_bool(k) + γ^{3i+2}·eq_virt(k)]
    // Input: Σ_i [γ^{3i}·hw_i + γ^{3i+1}·bool_i + γ^{3i+2}·virt_i]
    //   hw_i = 1 for inst/bytecode RA, hamming(r_booleanity) for RAM RA
    //   bool_i = booleanity claim for RA poly i
    //   virt_i = virtualization claim for RA poly i (group-dependent)

    // Eq polys for the three claim types, anchored at upstream reduction points
    let eq_hw_bool = p.poly(
        "eq_hw_bool",
        &[log_K],
        Public(PublicPoly::Eq(Some(booleanity_claims[0]))),
    );
    let eq_hw_virt_inst = p.poly(
        "eq_hw_virt_inst",
        &[log_K],
        Public(PublicPoly::Eq(Some(inst_ra_virt_claims[0]))),
    );
    let eq_hw_virt_bc = p.poly(
        "eq_hw_virt_bc",
        &[log_K],
        Public(PublicPoly::Eq(Some(bc_raf_claim))),
    );
    let eq_hw_virt_ram = p.poly(
        "eq_hw_virt_ram",
        &[log_K],
        Public(PublicPoly::Eq(Some(ram_ra_virt_claims[0]))),
    );

    // Build composition: Σ_i G_i · [γ^{3i} + γ^{3i+1}·eq_bool + γ^{3i+2}·eq_virt_group]
    let mut hw_comp: Expr = 0.into();
    for (i, &g) in hw_G.iter().enumerate() {
        let base = (3 * i) as u32;
        let eq_virt = if i < D {
            eq_hw_virt_inst
        } else if i < 2 * D {
            eq_hw_virt_bc
        } else {
            eq_hw_virt_ram
        };
        hw_comp = hw_comp
            + gamma.pow(base) * g
            + gamma.pow(base + 1) * g * eq_hw_bool
            + gamma.pow(base + 2) * g * eq_virt;
    }

    // Build input sum: Σ_i [γ^{3i}·hw_i + γ^{3i+1}·bool_i + γ^{3i+2}·virt_i]
    let mut hw_input: Expr = 0.into();
    // Instruction RA (0..D): hw_i = 1
    for (i, &virt_claim) in inst_ra_virt_claims.iter().enumerate() {
        let base = (3 * i) as u32;
        hw_input = hw_input
            + gamma.pow(base) // hw = 1
            + gamma.pow(base + 1) * booleanity_claims[i]
            + gamma.pow(base + 2) * virt_claim;
    }
    // Bytecode RA (D..2D): hw_i = 1
    for (i, &virt_claim) in bc_ra_virt_claims.iter().enumerate() {
        let j = D + i;
        let base = (3 * j) as u32;
        hw_input = hw_input
            + gamma.pow(base) // hw = 1
            + gamma.pow(base + 1) * booleanity_claims[j]
            + gamma.pow(base + 2) * virt_claim;
    }
    // RAM RA (2D..3D): hw_i = hamming(r_booleanity)
    for (i, &virt_claim) in ram_ra_virt_claims.iter().enumerate() {
        let j = 2 * D + i;
        let base = (3 * j) as u32;
        hw_input = hw_input
            + gamma.pow(base) * hamming_bool_claims[0]
            + gamma.pow(base + 1) * booleanity_claims[j]
            + gamma.pow(base + 2) * virt_claim;
    }

    let hamming_weight_reduction_v = p.vertices.len();
    let _hw_reduction = p.sumcheck(hw_comp, hw_input, &[log_K]);

    // Advice address phases (Stage 7, alongside HW reduction)
    // These operate on pushforward polys (cycle→address projection), not the
    // original committed advice. Declare address-phase polys over [log_K].
    let trusted_addr_poly = p.poly("TrustedAdviceAddr", &[log_K], Virtual);
    let untrusted_addr_poly = p.poly("UntrustedAdviceAddr", &[log_K], Virtual);
    let eq_advice_addr = p.poly(
        "eq_advice_addr",
        &[log_K],
        Public(PublicPoly::Eq(None)),
    );
    let advice_trusted_addr_v = p.vertices.len();
    let _trusted_addr = p.sumcheck(eq_advice_addr * trusted_addr_poly, trusted_mid[0], &[log_K]);
    let advice_untrusted_addr_v = p.vertices.len();
    let _untrusted_addr = p.sumcheck(eq_advice_addr * untrusted_addr_poly, untrusted_mid[0], &[log_K]);

    // ========== Return protocol + vertex map ==========

    let vertices = Vertices {
        outer_uniskip: outer_uniskip_v,
        outer_remaining: outer_remaining_v,
        product_uniskip: product_uniskip_v,
        product_remaining: product_remaining_v,
        ram_rw: ram_rw_v,
        instruction_claim_reduction: inst_claim_reduction_v,
        raf_eval: raf_eval_v,
        output_check: output_check_v,
        shift: shift_v,
        inst_input: inst_input_v,
        registers_claim_reduction: reg_claim_reduction_v,
        reg_rw: reg_rw_v,
        ram_val_check: ram_val_check_v,
        inst_read_raf: inst_read_raf_v,
        ram_ra_reduction: ram_ra_reduction_v,
        reg_val_eval: reg_val_eval_v,
        bytecode_raf: bytecode_raf_vs,
        booleanity: booleanity_v,
        hamming_booleanity: hamming_booleanity_v,
        ram_ra_virtual: ram_ra_virtual_v,
        inst_ra_virtual: inst_ra_virtual_v,
        inc_reduction: inc_reduction_v,
        advice_trusted_cycle: advice_trusted_cycle_v,
        advice_untrusted_cycle: advice_untrusted_cycle_v,
        hamming_weight_reduction: hamming_weight_reduction_v,
        advice_trusted_addr: advice_trusted_addr_v,
        advice_untrusted_addr: advice_untrusted_addr_v,
        evals_after_outer,
        evals_after_product,
        evals_after_shift,
        evals_after_inst_input,
        evals_after_ram_rw,
        evals_after_reg_rw,
        evals_after_inst_read_raf,
        evals_after_ram_val,
        evals_after_reg_val,
    };

    (p, vertices)
}
