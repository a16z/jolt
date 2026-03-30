//! Compile the Jolt protocol and dump the result.
//!
//! ```bash
//! cargo run --example jolt_ir -p jolt-compiler              # human-readable
//! cargo run --example jolt_ir -p jolt-compiler -- --json     # JSON serialization
//! cargo run --example jolt_ir -p jolt-compiler -- --dot      # Graphviz DOT (protocol only)
//! ```

#![allow(non_snake_case, unused_variables, clippy::print_stderr, clippy::print_stdout)]

use jolt_compiler::{
    compile, ChallengeSource, ClaimId, CompileParams, Expr, Objective, Poly, PolyKind, Protocol,
    ProverStep, PublicPoly, SolverConfig, Vertex,
};
use PolyKind::{Committed, Public, Virtual};

fn main() {
    let protocol = build_jolt_protocol();

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

    // Protocol stats
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

    // Prover schedule stats
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

    // Verifier script stats
    let script = &output.script;
    eprintln!("\n=== Verifier Script ===");
    eprintln!("  stages: {}", script.stages.len());
    for (i, stage) in script.stages.iter().enumerate() {
        eprintln!(
            "  stage {i}: rounds={}, degree={}, {} evals, {} post-squeeze",
            stage.num_rounds, stage.degree, stage.evaluations.len(), stage.post_squeeze.len(),
        );
    }

    // Dump full schedule
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
            ProverStep::Bind { polys: bound_polys, challenge, order } => {
                eprintln!("  [{i:>4}]     Bind({} polys, ch={}, {:?})", bound_polys.len(), challenge, order);
            }
            ProverStep::Evaluate { poly } => {
                eprintln!("  [{i:>4}]     Evaluate({})", polys[*poly].name);
            }
        }
    }
}

fn build_jolt_protocol() -> Protocol {
    let mut p = Protocol::new();

    // Dimensions
    let log_T = p.dim("log_T");
    let log_K_inst = p.dim("log_K_inst");
    let log_K_reg = p.dim("log_K_reg");
    let log_K_ram = p.dim("log_K_ram");
    let log_K_bc = p.dim("log_K_bc");
    let log_K_chunk = p.dim("log_K_chunk");

    const D_INST: usize = 16;
    const D_BC: usize = 8;
    const D_RAM: usize = 8;
    const M_INST: usize = 4;

    let gamma = p.challenge("gamma");

    // Committed polynomials
    let ram_inc = p.poly("RamInc", &[log_T], Committed);
    let rd_inc = p.poly("RdInc", &[log_T], Committed);
    let trusted = p.poly("TrustedAdvice", &[log_T], Committed);
    let untrusted = p.poly("UntrustedAdvice", &[log_T], Committed);

    let inst_ra: Vec<Poly> = (0..D_INST)
        .map(|d| p.poly(&format!("InstRa_{d}"), &[log_T], Committed))
        .collect();
    let bytecode_ra: Vec<Poly> = (0..D_BC)
        .map(|d| p.poly(&format!("BytecodeRa_{d}"), &[log_T], Committed))
        .collect();
    let ram_ra: Vec<Poly> = (0..D_RAM)
        .map(|d| p.poly(&format!("RamRa_{d}"), &[log_T], Committed))
        .collect();

    // Public polynomials (preprocessed from program)
    let ram_unmap = p.poly("RamUnmap", &[log_K_ram], Public(PublicPoly::Preprocessed));
    let ram_init = p.poly("RamInit", &[log_K_ram], Public(PublicPoly::Preprocessed));
    let io_mask = p.poly("IoMask", &[log_K_ram], Public(PublicPoly::Preprocessed));
    let val_io = p.poly("ValIo", &[log_K_ram], Public(PublicPoly::Preprocessed));
    let lookup_table = p.poly("LookupTable", &[log_K_inst], Public(PublicPoly::Preprocessed));
    let bc_table: Vec<Poly> = (0..5)
        .map(|s| {
            p.poly(
                &format!("BcTable_{s}"),
                &[log_K_bc],
                Public(PublicPoly::Preprocessed),
            )
        })
        .collect();
    let f_entry_trace = p.poly("FEntryTrace", &[log_K_bc], Public(PublicPoly::Preprocessed));
    let f_entry_expected = p.poly("FEntryExpected", &[log_K_bc], Public(PublicPoly::Preprocessed));

    // Virtual polynomials
    let az = p.poly("Az", &[log_T], Virtual);
    let bz = p.poly("Bz", &[log_T], Virtual);
    let product_virt = p.poly("Product", &[log_T], Virtual);
    let should_branch = p.poly("ShouldBranch", &[log_T], Virtual);
    let should_jump = p.poly("ShouldJump", &[log_T], Virtual);
    let left = p.poly("ProductLeft", &[log_T], Virtual);
    let right = p.poly("ProductRight", &[log_T], Virtual);

    let pc = p.poly("PC", &[log_T], Virtual);
    let unexpanded_pc = p.poly("UnexpandedPC", &[log_T], Virtual);
    let is_noop = p.poly("IsNoop", &[log_T], Virtual);
    let is_virtual = p.poly("IsVirtual", &[log_T], Virtual);
    let is_first = p.poly("IsFirstInSeq", &[log_T], Virtual);
    let is_rs1 = p.poly("IsRs1", &[log_T], Virtual);
    let is_rs2 = p.poly("IsRs2", &[log_T], Virtual);
    let is_pc_flag = p.poly("IsPcFlag", &[log_T], Virtual);
    let is_imm_flag = p.poly("IsImmFlag", &[log_T], Virtual);
    let _is_branch = p.poly("IsBranch", &[log_T], Virtual);
    let _jump_flag = p.poly("JumpFlag", &[log_T], Virtual);

    let rs1_val = p.poly("Rs1Val", &[log_T], Virtual);
    let rs2_val = p.poly("Rs2Val", &[log_T], Virtual);
    let imm_val = p.poly("Imm", &[log_T], Virtual);
    let left_inst_input = p.poly("LeftInstructionInput", &[log_T], Virtual);
    let right_inst_input = p.poly("RightInstructionInput", &[log_T], Virtual);
    let lookup_output = p.poly("LookupOutput", &[log_T], Virtual);
    let left_op = p.poly("LeftLookupOp", &[log_T], Virtual);
    let right_op = p.poly("RightLookupOp", &[log_T], Virtual);

    let ram_combined_ra = p.poly("RamCombinedRa", &[log_K_ram, log_T], Virtual);
    let ram_val = p.poly("RamVal", &[log_K_ram, log_T], Virtual);
    let ram_val_final = p.poly("RamValFinal", &[log_K_ram], Virtual);
    let ram_wa = p.poly("RamWA", &[log_T], Virtual);
    let ram_rv = p.poly("RamReadValue", &[log_T], Virtual);
    let ram_wv = p.poly("RamWriteValue", &[log_T], Virtual);
    let ram_address = p.poly("RamAddress", &[log_T], Virtual);
    let hamming = p.poly("HammingWeight", &[log_T], Virtual);

    let reg_wa = p.poly("RegWA", &[log_K_reg, log_T], Virtual);
    let reg_ra_rs1 = p.poly("RegRaRs1", &[log_K_reg, log_T], Virtual);
    let reg_ra_rs2 = p.poly("RegRaRs2", &[log_K_reg, log_T], Virtual);
    let reg_val = p.poly("RegVal", &[log_K_reg, log_T], Virtual);
    let rd_wa = p.poly("RdWA", &[log_T], Virtual);
    let rd_wv = p.poly("RdWriteValue", &[log_T], Virtual);
    let rs1_rv = p.poly("Rs1ReadValue", &[log_T], Virtual);
    let rs2_rv = p.poly("Rs2ReadValue", &[log_T], Virtual);

    let ram_raf_ra = p.poly("RamRafRa", &[log_K_ram], Virtual);
    let inst_raf_ra = p.poly("InstRafRa", &[log_K_inst, log_T], Virtual);
    let inst_raf_val = p.poly("InstRafVal", &[log_K_inst, log_T], Virtual);
    let bytecode_raf_ra = p.poly("BytecodeRafRa", &[log_K_bc, log_T], Virtual);
    let eq_zero = p.poly("EqZero", &[log_T], Virtual);
    let bc_rv: Vec<Poly> = (0..5)
        .map(|s| p.poly(&format!("BcReadVal_{s}"), &[log_T], Virtual))
        .collect();

    let hw_G: Vec<Poly> = (0..D_INST + D_BC + D_RAM)
        .map(|i| p.poly(&format!("G_{i}"), &[log_K_chunk], Virtual))
        .collect();

    // Public eq / lt polynomials
    let eq_T = p.poly("eq_T", &[log_T], Public(PublicPoly::Eq(None)));
    let lt_T = p.poly("lt_T", &[log_T], Public(PublicPoly::Lt(None)));
    let eq_K_ram = p.poly("eq_K_ram", &[log_K_ram], Public(PublicPoly::Eq(None)));

    // S1: Spartan
    const OUTER_UNISKIP_DOMAIN: usize = 10;
    const PRODUCT_UNISKIP_DOMAIN: usize = 3;

    let outer_mid = p.uniskip_round(eq_T * az * bz, 0, &[log_T], OUTER_UNISKIP_DOMAIN);
    let outer = p.sumcheck(eq_T * az * bz, outer_mid, &[log_T]);

    let product_upstream = p.evaluate(product_virt, outer[0]);
    let branch_upstream = p.evaluate(should_branch, outer[0]);
    let jump_upstream = p.evaluate(should_jump, outer[0]);
    let product_mid = p.uniskip_round(
        eq_T * left * right,
        product_upstream + gamma * branch_upstream + gamma.pow(2) * jump_upstream,
        &[log_T],
        PRODUCT_UNISKIP_DOMAIN,
    );
    let product = p.sumcheck(eq_T * left * right, product_mid, &[log_T]);

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
    let next_unexpanded_pc = p.evaluate(unexpanded_pc, outer[0]);
    let next_pc = p.evaluate(pc, outer[0]);
    let next_is_virtual = p.evaluate(is_virtual, outer[0]);
    let next_is_first = p.evaluate(is_first, outer[0]);
    let next_is_noop = p.evaluate(is_noop, product[0]);

    let shift = p.sumcheck(
        eq1_outer
            * (unexpanded_pc + gamma * pc + gamma.pow(2) * is_virtual + gamma.pow(3) * is_first)
            + eq1_product * (gamma.pow(4) - gamma.pow(4) * is_noop),
        next_unexpanded_pc
            + gamma * next_pc
            + gamma.pow(2) * next_is_virtual
            + gamma.pow(3) * next_is_first
            + gamma.pow(4)
            - gamma.pow(4) * next_is_noop,
        &[log_T],
    );

    let left_inst_eval = p.evaluate(left_inst_input, product[0]);
    let right_inst_eval = p.evaluate(right_inst_input, product[0]);
    let _inst_input = p.sumcheck(
        eq_T * (is_rs2 * rs2_val + is_imm_flag * imm_val
            + gamma * (is_rs1 * rs1_val + is_pc_flag * unexpanded_pc)),
        right_inst_eval + gamma * left_inst_eval,
        &[log_T],
    );

    let ram_rv_eval = p.evaluate(ram_rv, outer[0]);
    let ram_wv_eval = p.evaluate(ram_wv, outer[0]);
    let rd_wv_eval = p.evaluate(rd_wv, outer[0]);
    let rs1_rv_eval = p.evaluate(rs1_rv, outer[0]);
    let rs2_rv_eval = p.evaluate(rs2_rv, outer[0]);
    let lookup_output_eval = p.evaluate(lookup_output, outer[0]);
    let left_op_eval = p.evaluate(left_op, outer[0]);
    let right_op_eval = p.evaluate(right_op, outer[0]);
    let left_inst_outer_eval = p.evaluate(left_inst_input, outer[0]);
    let right_inst_outer_eval = p.evaluate(right_inst_input, outer[0]);

    // S2: Claim reductions + RAM RW + RAM RAF + output
    let eq_inst_cr_anchor = p.poly(
        "eq_inst_cr",
        &[log_T],
        Public(PublicPoly::Eq(Some(outer[0]))),
    );
    let inst_cr = p.sumcheck(
        eq_inst_cr_anchor
            * (lookup_output
                + gamma * left_op
                + gamma.pow(2) * right_op
                + gamma.pow(3) * left_inst_input
                + gamma.pow(4) * right_inst_input),
        lookup_output_eval
            + gamma * left_op_eval
            + gamma.pow(2) * right_op_eval
            + gamma.pow(3) * left_inst_outer_eval
            + gamma.pow(4) * right_inst_outer_eval,
        &[log_T],
    );

    let eq_reg_cr_anchor = p.poly(
        "eq_reg_cr",
        &[log_T],
        Public(PublicPoly::Eq(Some(outer[0]))),
    );
    let reg_cr = p.sumcheck(
        eq_reg_cr_anchor * (rd_wv + gamma * rs1_rv + gamma.pow(2) * rs2_rv),
        rd_wv_eval + gamma * rs1_rv_eval + gamma.pow(2) * rs2_rv_eval,
        &[log_T],
    );

    let ram_rw = p.sumcheck(
        eq_T * ram_combined_ra * (ram_val + gamma * (ram_val + ram_inc)),
        ram_rv_eval + gamma * ram_wv_eval,
        &[log_K_ram, log_T],
    );

    let ram_address_eval = p.evaluate(ram_address, outer[0]);
    let _ram_raf = p.sumcheck(ram_raf_ra * ram_unmap, ram_address_eval, &[log_K_ram]);

    let _ram_output = p.sumcheck(
        eq_K_ram * io_mask * (ram_val_final - val_io),
        0,
        &[log_K_ram],
    );

    // S3: Registers RW + RAM val check
    let reg_cr_rd_wv = p.evaluate(rd_wv, reg_cr[0]);
    let reg_cr_rs1_rv = p.evaluate(rs1_rv, reg_cr[0]);
    let reg_cr_rs2_rv = p.evaluate(rs2_rv, reg_cr[0]);
    let reg_rw = p.sumcheck(
        eq_T * (reg_wa * (rd_inc + reg_val)
            + gamma * reg_ra_rs1 * reg_val
            + gamma.pow(2) * reg_ra_rs2 * reg_val),
        reg_cr_rd_wv + gamma * (reg_cr_rs1_rv + gamma * reg_cr_rs2_rv),
        &[log_K_reg, log_T],
    );

    let ram_val_eval = p.evaluate(ram_val, ram_rw[0]);
    let ram_init_eval = p.evaluate(ram_init, ram_rw[0]);
    let ram_val_final_eval = p.evaluate(ram_val_final, ram_rw[0]);
    let ram_val_sc = p.sumcheck(
        ram_inc * ram_wa * (lt_T + gamma),
        (ram_val_eval - ram_init_eval) + gamma * (ram_val_final_eval - ram_init_eval),
        &[log_T],
    );

    // S4: Instruction ReadRaf + RAM RA CR + Registers val eval
    let eq_inst_raf = p.poly(
        "eq_inst_raf",
        &[log_T],
        Public(PublicPoly::Eq(Some(inst_cr[0]))),
    );
    let inst_cr_output_eval = p.evaluate(lookup_output, inst_cr[0]);
    let inst_cr_left_eval = p.evaluate(left_op, inst_cr[0]);
    let inst_cr_right_eval = p.evaluate(right_op, inst_cr[0]);
    let inst_read_raf = p.sumcheck(
        eq_inst_raf * inst_raf_ra * (lookup_table + gamma * inst_raf_val),
        inst_cr_output_eval + gamma * inst_cr_left_eval + gamma.pow(2) * inst_cr_right_eval,
        &[log_K_inst, log_T],
    );

    let reg_val_eval = p.evaluate(reg_val, reg_rw[0]);
    let reg_val_sc = p.sumcheck(lt_T * rd_inc * rd_wa, reg_val_eval, &[log_T]);

    // S5: Bytecode + booleanity + virtualization + Inc CR + advice cycle
    let bc_upstream = [outer[0], product[0], shift[0], reg_rw[0], reg_val_sc[0]];
    let bc_eq: Vec<Poly> = (0..5)
        .map(|s| {
            p.poly(
                &format!("BcEq_{s}"),
                &[log_T],
                Public(PublicPoly::Eq(Some(bc_upstream[s]))),
            )
        })
        .collect();
    let bc_rv_evals: Vec<ClaimId> = (0..5)
        .map(|s| p.evaluate(bc_rv[s], bc_upstream[s]))
        .collect();
    let raf_pc_outer = p.evaluate(pc, outer[0]);
    let raf_pc_shift = p.evaluate(pc, shift[0]);

    let mut bc_comp: Expr = Expr::from(bc_eq[0]) * bytecode_raf_ra * bc_table[0];
    for s in 1..5 {
        bc_comp = bc_comp + gamma.pow(s as u32) * bc_eq[s] * bytecode_raf_ra * bc_table[s];
    }
    bc_comp = bc_comp + gamma.pow(7) * eq_zero * f_entry_trace * f_entry_expected;
    let mut bc_input: Expr = bc_rv_evals[0].into();
    for (s, &eval) in bc_rv_evals.iter().enumerate().skip(1) {
        bc_input = bc_input + gamma.pow(s as u32) * eval;
    }
    bc_input = bc_input + gamma.pow(5) * raf_pc_outer + gamma.pow(6) * raf_pc_shift + gamma.pow(7);
    let _bytecode_raf = p.sumcheck(bc_comp, bc_input, &[log_K_bc, log_T]);

    let all_ra: Vec<Poly> = inst_ra
        .iter()
        .chain(bytecode_ra.iter())
        .chain(ram_ra.iter())
        .copied()
        .collect();
    let mut bool_comp = eq_T * (all_ra[0] * all_ra[0] - all_ra[0]);
    for (i, &ra) in all_ra.iter().enumerate().skip(1) {
        bool_comp = bool_comp + gamma.pow(2 * i as u32) * eq_T * (ra * ra - ra);
    }
    let _booleanity = p.sumcheck(bool_comp, 0, &[log_K_chunk, log_T]);

    let _hamming_bool = p.sumcheck(eq_T * (hamming * hamming - hamming), 0, &[log_T]);

    let inst_ra_evals: Vec<ClaimId> = inst_ra
        .iter()
        .map(|&ra| p.evaluate(ra, inst_read_raf[0]))
        .collect();
    let mut ra_comp = Expr::from(0i64);
    for (i, chunk) in inst_ra.chunks(M_INST).enumerate() {
        let seed: Expr = if i == 0 {
            Expr::from(eq_T)
        } else {
            gamma.pow(i as u32) * eq_T
        };
        ra_comp = ra_comp + chunk.iter().fold(seed, |acc, &ra| acc * ra);
    }
    let mut ra_input = Expr::from(0i64);
    for (i, chunk) in inst_ra_evals.chunks(M_INST).enumerate() {
        let first: Expr = if i == 0 {
            chunk[0].into()
        } else {
            gamma.pow(i as u32) * chunk[0]
        };
        ra_input = ra_input + chunk[1..].iter().fold(first, |acc, &eval| acc * eval);
    }
    let _inst_ra_virt = p.sumcheck(ra_comp, ra_input, &[log_T]);

    let ram_ra_evals: Vec<ClaimId> = ram_ra
        .iter()
        .map(|&ra| p.evaluate(ra, ram_rw[0]))
        .collect();
    let ram_ra_product = ram_ra
        .iter()
        .skip(1)
        .fold(Expr::from(eq_T) * Expr::from(ram_ra[0]), |acc, &ra| {
            acc * ra
        });
    let mut ram_ra_input: Expr = ram_ra_evals[0].into();
    for &eval in &ram_ra_evals[1..] {
        ram_ra_input = ram_ra_input * eval;
    }
    let _ram_ra_virt = p.sumcheck(ram_ra_product, ram_ra_input, &[log_T]);

    let ram_inc_rw_eval = p.evaluate(ram_inc, ram_rw[0]);
    let ram_inc_val_eval = p.evaluate(ram_inc, ram_val_sc[0]);
    let rd_inc_reg_rw_eval = p.evaluate(rd_inc, reg_rw[0]);
    let rd_inc_reg_val_eval = p.evaluate(rd_inc, reg_val_sc[0]);
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
    let _inc_cr = p.sumcheck(
        ram_inc * (eq_inc_rw + gamma * eq_inc_val)
            + gamma.pow(2) * rd_inc * (eq_inc_reg_rw + gamma * eq_inc_reg_val),
        ram_inc_rw_eval
            + gamma * ram_inc_val_eval
            + gamma.pow(2) * rd_inc_reg_rw_eval
            + gamma.pow(3) * rd_inc_reg_val_eval,
        &[log_T],
    );

    let trusted_eval = p.evaluate(trusted, ram_val_sc[0]);
    let untrusted_eval = p.evaluate(untrusted, ram_val_sc[0]);
    let eq_advice_cycle = p.poly(
        "eq_advice_cycle",
        &[log_T],
        Public(PublicPoly::Eq(Some(ram_val_sc[0]))),
    );
    let advice_trusted_cycle =
        p.sumcheck(eq_advice_cycle * trusted, trusted_eval, &[log_T]);
    let advice_untrusted_cycle =
        p.sumcheck(eq_advice_cycle * untrusted, untrusted_eval, &[log_T]);

    // S6: Fused HammingWeight + Address Reduction
    let eq_hw_bool = p.poly(
        "eq_hw_bool",
        &[log_K_chunk],
        Public(PublicPoly::Eq(Some(_booleanity[0]))),
    );
    let eq_hw_virt_inst = p.poly(
        "eq_hw_virt_inst",
        &[log_K_chunk],
        Public(PublicPoly::Eq(Some(_inst_ra_virt[0]))),
    );
    let eq_hw_virt_bc = p.poly(
        "eq_hw_virt_bc",
        &[log_K_chunk],
        Public(PublicPoly::Eq(Some(_bytecode_raf[0]))),
    );
    let eq_hw_virt_ram = p.poly(
        "eq_hw_virt_ram",
        &[log_K_chunk],
        Public(PublicPoly::Eq(Some(_ram_ra_virt[0]))),
    );

    let eq_virt_for = |i: usize| -> Poly {
        if i < D_INST {
            eq_hw_virt_inst
        } else if i < D_INST + D_BC {
            eq_hw_virt_bc
        } else {
            eq_hw_virt_ram
        }
    };

    let mut hw_comp = Expr::from(0i64);
    for (i, &g) in hw_G.iter().enumerate() {
        let b = (3 * i) as u32;
        hw_comp = hw_comp
            + g * (gamma.pow(b)
                + gamma.pow(b + 1) * eq_hw_bool
                + gamma.pow(b + 2) * eq_virt_for(i));
    }

    let ram_hw_eval = p.evaluate(hamming, _hamming_bool[0]);
    let hw_G_bool: Vec<ClaimId> = hw_G
        .iter()
        .map(|&g| p.evaluate(g, _booleanity[0]))
        .collect();
    let hw_G_virt: Vec<ClaimId> = hw_G
        .iter()
        .enumerate()
        .map(|(i, &g)| {
            if i < D_INST {
                p.evaluate(g, _inst_ra_virt[0])
            } else if i < D_INST + D_BC {
                p.evaluate(g, _bytecode_raf[0])
            } else {
                p.evaluate(g, _ram_ra_virt[0])
            }
        })
        .collect();

    let mut hw_input = Expr::from(0i64);
    for i in 0..D_INST + D_BC + D_RAM {
        let b = (3 * i) as u32;
        let hw_term: Expr = if i < D_INST + D_BC {
            gamma.pow(b)
        } else {
            gamma.pow(b) * ram_hw_eval
        };
        hw_input = hw_input
            + hw_term
            + gamma.pow(b + 1) * hw_G_bool[i]
            + gamma.pow(b + 2) * hw_G_virt[i];
    }

    let _hw_cr = p.sumcheck(hw_comp, hw_input, &[log_K_chunk]);

    let eq_advice_addr_t = p.poly(
        "eq_advice_addr_t",
        &[log_T],
        Public(PublicPoly::Eq(Some(advice_trusted_cycle[0]))),
    );
    let eq_advice_addr_u = p.poly(
        "eq_advice_addr_u",
        &[log_T],
        Public(PublicPoly::Eq(Some(advice_untrusted_cycle[0]))),
    );
    let trusted_cycle_mid = p.evaluate(trusted, advice_trusted_cycle[0]);
    let untrusted_cycle_mid = p.evaluate(untrusted, advice_untrusted_cycle[0]);
    let _advice_trusted_addr =
        p.sumcheck(eq_advice_addr_t * trusted, trusted_cycle_mid, &[log_T]);
    let _advice_untrusted_addr =
        p.sumcheck(eq_advice_addr_u * untrusted, untrusted_cycle_mid, &[log_T]);

    p
}
