//! Hand-written ground truth Module for the jolt-core prover.
//!
//! This builds a `Module` directly (bypassing the compiler) that exactly
//! matches jolt-core's prover execution: commitment ordering, stage
//! boundaries, kernel compositions, evaluation placement, and transcript
//! barriers. It is the reference target for compiler auto-derivation and
//! for runtime transcript-parity testing.
//!
//! Usage:
//!   cargo run --example jolt_core_module -p jolt-compiler                          # human-readable stats (default params)
//!   cargo run --example jolt_core_module -p jolt-compiler -- --emit protocol.jolt  # binary module
//!   cargo run --example jolt_core_module -p jolt-compiler -- --log-t 10 --log-k-bytecode 8 --log-k-ram 12 --emit out.jolt

#![allow(
    non_snake_case,
    unused_variables,
    dead_code,
    clippy::print_stderr,
    clippy::print_stdout
)]

use jolt_compiler::formula::{BindingOrder, Factor, Formula, ProductTerm};
use jolt_compiler::ir::PolyKind;
use jolt_compiler::kernel_spec::Iteration;
use jolt_compiler::module::{
    BatchedInstance, BatchedSumcheckDef, ChallengeDecl, ChallengeSource, ClaimFactor, ClaimFormula,
    ClaimTerm, DomainSeparator, Evaluation, InputBinding, InstancePhase, KernelDef, Module, Op,
    PointNormalization, PolyDecl, R1CSMatrix, ScalarCapture, Schedule, SegmentedConfig,
    SumcheckInstance, VerifierOp, VerifierSchedule, VerifierStageIndex,
};
use jolt_compiler::params::{ModuleParams, LOG_K_REG, NUM_LOOKUP_TABLES, NUM_R1CS_INPUTS};
use jolt_compiler::KernelSpec;
use jolt_compiler::PolynomialId;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let log_t = parse_arg(&args, "--log-t").unwrap_or(25);
    let log_k_bytecode = parse_arg(&args, "--log-k-bytecode").unwrap_or(16);
    let log_k_ram = parse_arg(&args, "--log-k-ram").unwrap_or(20);

    let p = ModuleParams::new(log_t, log_k_bytecode, log_k_ram);
    let module = build_module(&p);

    if let Some(pos) = args.iter().position(|a| a == "--emit") {
        let path = args.get(pos + 1).expect("--emit requires a path argument");
        let bytes = module.to_bytes();
        std::fs::write(path, &bytes).expect("failed to write protocol binary");
        eprintln!("wrote {} bytes to {path}", bytes.len());
        return;
    }

    print_stats(&module, &p);
}

fn parse_arg(args: &[String], flag: &str) -> Option<usize> {
    args.iter().position(|a| a == flag).map(|pos| {
        args.get(pos + 1)
            .unwrap_or_else(|| panic!("{flag} requires a value"))
            .parse()
            .unwrap_or_else(|_| panic!("{flag} value must be a positive integer"))
    })
}

// ---------------------------------------------------------------------------
// Polynomial index table.
//
// Every polynomial referenced by the Module is registered here with a
// stable index. The ordering within committed polys matches jolt-core's
// `all_committed_polynomials()` iteration order, which determines the
// Fiat-Shamir transcript commitment sequence.
// ---------------------------------------------------------------------------

struct PolyTable {
    polys: Vec<PolyDecl>,
}

impl PolyTable {
    fn new() -> Self {
        Self { polys: Vec::new() }
    }

    fn add(
        &mut self,
        id: PolynomialId,
        name: &str,
        kind: PolyKind,
        num_vars: usize,
    ) -> PolynomialId {
        self.polys.push(PolyDecl {
            name: name.to_string(),
            kind,
            num_elements: 1 << num_vars,
        });
        id
    }

    fn into_vec(self) -> Vec<PolyDecl> {
        self.polys
    }
}

/// All polynomial identifiers, grouped by commitment / virtual / public.
struct Polys {
    // === Committed (transcript order matches jolt-core) ===
    rd_inc: PolynomialId,
    ram_inc: PolynomialId,
    instruction_ra: Vec<PolynomialId>, // [0..p.instruction_d)
    ram_ra: Vec<PolynomialId>,         // [0..p.ram_d)
    bytecode_ra: Vec<PolynomialId>,    // [0..p.bytecode_d)
    untrusted_advice: PolynomialId,
    trusted_advice: PolynomialId,

    // === Virtual — Spartan internal ===
    az: PolynomialId,
    bz: PolynomialId,
    spartan_eq: PolynomialId,
    product_left: PolynomialId,
    product_right: PolynomialId,

    // === Virtual — trace-derived (R1CS inputs, 35 entries in ALL_R1CS_INPUTS order) ===
    left_instruction_input: PolynomialId,
    right_instruction_input: PolynomialId,
    product: PolynomialId,
    should_branch: PolynomialId,
    pc: PolynomialId,
    unexpanded_pc: PolynomialId,
    imm: PolynomialId,
    ram_address: PolynomialId,
    rs1_val: PolynomialId,
    rs2_val: PolynomialId,
    rd_write_value: PolynomialId,
    ram_read_value: PolynomialId,
    ram_write_value: PolynomialId,
    left_lookup_operand: PolynomialId,
    right_lookup_operand: PolynomialId,
    next_unexpanded_pc: PolynomialId,
    next_pc: PolynomialId,
    next_is_virtual: PolynomialId,
    next_is_first: PolynomialId,
    lookup_output: PolynomialId,
    should_jump: PolynomialId,
    op_flags: Vec<PolynomialId>, // [0..NUM_CIRCUIT_FLAGS)

    // === Virtual — non-R1CS-input trace values ===
    next_is_noop: PolynomialId,
    rd: PolynomialId,

    // === Virtual — InstructionFlags (6 total) ===
    inst_flag_left_is_pc: PolynomialId,
    inst_flag_right_is_imm: PolynomialId,
    inst_flag_left_is_rs1: PolynomialId,
    inst_flag_right_is_rs2: PolynomialId,
    inst_flag_branch: PolynomialId,
    inst_flag_is_noop: PolynomialId,

    // === Virtual — registers ===
    reg_wa: PolynomialId,
    reg_ra_rs1: PolynomialId,
    reg_ra_rs2: PolynomialId,
    reg_val: PolynomialId,

    // === Virtual — RAM ===
    ram_combined_ra: PolynomialId,
    ram_val: PolynomialId,
    ram_val_final: PolynomialId,
    ram_wa: PolynomialId,
    hamming_weight: PolynomialId,

    // === Virtual — RamRW eq tables (segmented phase 1/2) ===
    ram_eq_cycle: PolynomialId,
    ram_eq_addr: PolynomialId,

    // === Virtual — RAF ===
    ram_ra_indicator: PolynomialId,
    ram_raf_ra: PolynomialId,
    inst_raf_ra: PolynomialId,
    bytecode_raf_ra: PolynomialId,
    inst_raf_flag: PolynomialId,
    lookup_table_flags: Vec<PolynomialId>, // [0..NUM_LOOKUP_TABLES)

    // === Virtual — bytecode read values ===
    bc_read_val: Vec<PolynomialId>, // [0..5)

    // === Virtual — HW reduction pushforward ===
    hw_g: Vec<PolynomialId>, // [0..total_d)

    // === Virtual — Spartan uniskip evaluations ===
    outer_uniskip_eval: PolynomialId,
    product_uniskip_eval: PolynomialId,

    // === Virtual — advice address phase ===
    trusted_advice_addr: PolynomialId,
    untrusted_advice_addr: PolynomialId,

    // === Public — preprocessed ===
    io_mask: PolynomialId,
    val_io: PolynomialId,
    ram_unmap: PolynomialId,
    ram_init: PolynomialId,
    lookup_table: PolynomialId,
    bc_table: Vec<PolynomialId>, // [0..5)
}

fn register_polys(pt: &mut PolyTable, p: &ModuleParams) -> Polys {
    use PolyKind::{Committed, Virtual};
    use PolynomialId::{
        Az, BranchFlag, BytecodeRa, BytecodeReadRafVal, Bz, ExpandedPc, HammingG, HammingWeight,
        Imm, InstructionRa, InstructionRafFlag, IoMask, LeftInstructionInput, LeftIsPc, LeftIsRs1,
        LeftLookupOperand, LookupOutput, LookupTableFlag, NextIsFirstInSequence, NextIsNoop,
        NextIsVirtual, NextPc, NextUnexpandedPc, NoopFlag, OpFlag, OuterUniskipEval, Product,
        ProductLeft, ProductRight, ProductUniskipEval, RamAddress, RamCombinedRa, RamInc, RamRa,
        RamRafRa, RamReadValue, RamVal, RamValFinal, RamWriteValue,
        Rd, RdInc, RdWa, RdWriteValue, RegistersVal, RightInstructionInput, RightIsImm,
        RightIsRs2, RightLookupOperand, Rs1Ra,
        Rs1Value, Rs2Ra, Rs2Value, ShouldBranch, ShouldJump, SpartanEq, TrustedAdvice,
        UnexpandedPc, UntrustedAdvice, ValIo,
    };

    // --- Committed (jolt-core transcript order) ---
    let rd_inc = pt.add(RdInc, "RdInc", Committed, p.log_t);
    let ram_inc = pt.add(RamInc, "RamInc", Committed, p.log_t);
    let instruction_ra: Vec<_> = (0..p.instruction_d)
        .map(|d| {
            pt.add(
                InstructionRa(d),
                &format!("InstructionRa_{d}"),
                Committed,
                p.log_k_chunk + p.log_t,
            )
        })
        .collect();
    let ram_ra: Vec<_> = (0..p.ram_d)
        .map(|d| {
            pt.add(
                RamRa(d),
                &format!("RamRa_{d}"),
                Committed,
                p.log_k_chunk + p.log_t,
            )
        })
        .collect();
    let bytecode_ra: Vec<_> = (0..p.bytecode_d)
        .map(|d| {
            pt.add(
                BytecodeRa(d),
                &format!("BytecodeRa_{d}"),
                Committed,
                p.log_k_chunk + p.log_t,
            )
        })
        .collect();
    let untrusted_advice = pt.add(UntrustedAdvice, "UntrustedAdvice", Committed, p.log_t);
    let trusted_advice = pt.add(TrustedAdvice, "TrustedAdvice", Committed, p.log_t);

    // --- Virtual — Spartan internal ---
    let az = pt.add(Az, "Az", Virtual, p.log_t + 1);
    let bz = pt.add(Bz, "Bz", Virtual, p.log_t + 1);
    let spartan_eq = pt.add(SpartanEq, "SpartanEqTable", Virtual, p.log_t + 1);
    // Domain-indexed: T cycles × stride 4 (3 product constraints, padded to next power of 2).
    let product_stride_log = (p.product_uniskip_domain as u64)
        .next_power_of_two()
        .trailing_zeros() as usize;
    let product_left = pt.add(
        ProductLeft,
        "ProductLeft",
        Virtual,
        p.log_t + product_stride_log,
    );
    let product_right = pt.add(
        ProductRight,
        "ProductRight",
        Virtual,
        p.log_t + product_stride_log,
    );

    // --- Virtual — R1CS inputs (35 entries, ALL_R1CS_INPUTS order) ---
    let left_instruction_input = pt.add(
        LeftInstructionInput,
        "LeftInstructionInput",
        Virtual,
        p.log_t,
    );
    let right_instruction_input = pt.add(
        RightInstructionInput,
        "RightInstructionInput",
        Virtual,
        p.log_t,
    );
    let product = pt.add(Product, "Product", Virtual, p.log_t);
    let should_branch = pt.add(ShouldBranch, "ShouldBranch", Virtual, p.log_t);
    let pc = pt.add(ExpandedPc, "PC", Virtual, p.log_t);
    let unexpanded_pc = pt.add(UnexpandedPc, "UnexpandedPC", Virtual, p.log_t);
    let imm = pt.add(Imm, "Imm", Virtual, p.log_t);
    let ram_address = pt.add(RamAddress, "RamAddress", Virtual, p.log_t);
    let rs1_val = pt.add(Rs1Value, "Rs1Value", Virtual, p.log_t);
    let rs2_val = pt.add(Rs2Value, "Rs2Value", Virtual, p.log_t);
    let rd_write_value = pt.add(RdWriteValue, "RdWriteValue", Virtual, p.log_t);
    let ram_read_value = pt.add(RamReadValue, "RamReadValue", Virtual, p.log_t);
    let ram_write_value = pt.add(RamWriteValue, "RamWriteValue", Virtual, p.log_t);
    let left_lookup_operand = pt.add(LeftLookupOperand, "LeftLookupOperand", Virtual, p.log_t);
    let right_lookup_operand = pt.add(RightLookupOperand, "RightLookupOperand", Virtual, p.log_t);
    let next_unexpanded_pc = pt.add(NextUnexpandedPc, "NextUnexpandedPC", Virtual, p.log_t);
    let next_pc = pt.add(NextPc, "NextPC", Virtual, p.log_t);
    let next_is_virtual = pt.add(NextIsVirtual, "NextIsVirtual", Virtual, p.log_t);
    let next_is_first = pt.add(
        NextIsFirstInSequence,
        "NextIsFirstInSequence",
        Virtual,
        p.log_t,
    );
    let lookup_output = pt.add(LookupOutput, "LookupOutput", Virtual, p.log_t);
    let should_jump = pt.add(ShouldJump, "ShouldJump", Virtual, p.log_t);
    let op_flags: Vec<_> = (0..14)
        .map(|i| pt.add(OpFlag(i), &format!("OpFlag_{i}"), Virtual, p.log_t))
        .collect();

    // --- Virtual — non-R1CS-input trace values ---
    let next_is_noop = pt.add(NextIsNoop, "NextIsNoop", Virtual, p.log_t);
    let rd = pt.add(Rd, "Rd", Virtual, p.log_t);

    // --- Virtual — InstructionFlags (6 variants) ---
    let inst_flag_left_is_pc = pt.add(LeftIsPc, "InstFlag_LeftOperandIsPC", Virtual, p.log_t);
    let inst_flag_right_is_imm = pt.add(RightIsImm, "InstFlag_RightOperandIsImm", Virtual, p.log_t);
    let inst_flag_left_is_rs1 = pt.add(
        LeftIsRs1,
        "InstFlag_LeftOperandIsRs1Value",
        Virtual,
        p.log_t,
    );
    let inst_flag_right_is_rs2 = pt.add(
        RightIsRs2,
        "InstFlag_RightOperandIsRs2Value",
        Virtual,
        p.log_t,
    );
    let inst_flag_branch = pt.add(BranchFlag, "InstFlag_Branch", Virtual, p.log_t);
    let inst_flag_is_noop = pt.add(NoopFlag, "InstFlag_IsNoop", Virtual, p.log_t);

    // --- Virtual — registers ---
    let reg_wa = pt.add(RdWa, "RdWA", Virtual, p.log_t);
    let reg_ra_rs1 = pt.add(Rs1Ra, "RegRaRs1", Virtual, LOG_K_REG + p.log_t);
    let reg_ra_rs2 = pt.add(Rs2Ra, "RegRaRs2", Virtual, LOG_K_REG + p.log_t);
    let reg_val = pt.add(RegistersVal, "RegVal", Virtual, LOG_K_REG + p.log_t);

    // --- Virtual — RAM ---
    let ram_combined_ra = pt.add(
        RamCombinedRa,
        "RamCombinedRa",
        Virtual,
        p.log_k_ram + p.log_t,
    );
    let ram_val = pt.add(RamVal, "RamVal", Virtual, p.log_k_ram + p.log_t);
    let ram_val_final = pt.add(RamValFinal, "RamValFinal", Virtual, p.log_k_ram);
    let ram_wa = pt.add(PolynomialId::RamWa, "RamWA", Virtual, p.log_t);
    let hamming_weight = pt.add(HammingWeight, "HammingWeight", Virtual, p.log_t);

    // --- Virtual — RamRW segmented eq tables ---
    let ram_eq_cycle = pt.add(
        PolynomialId::RamEqCycle,
        "RamEqCycle",
        Virtual,
        p.log_t,
    );
    let ram_eq_addr = pt.add(
        PolynomialId::RamEqAddr,
        "RamEqAddr",
        Virtual,
        p.log_k_ram,
    );

    // --- Virtual — RAF ---
    let ram_ra_indicator = pt.add(
        PolynomialId::RamRaIndicator,
        "RamRaIndicator",
        Virtual,
        p.log_k_ram + p.log_t,
    );
    let ram_raf_ra = pt.add(RamRafRa, "RamRafRa", Virtual, p.log_k_ram);
    let inst_raf_ra = pt.add(
        PolynomialId::InstructionRafRa,
        "InstRafRa",
        Virtual,
        p.log_k_chunk + p.log_t,
    );
    let bytecode_raf_ra = pt.add(
        PolynomialId::BytecodeRafRa,
        "BytecodeRafRa",
        Virtual,
        p.log_k_chunk + p.log_t,
    );
    let inst_raf_flag = pt.add(InstructionRafFlag, "InstructionRafFlag", Virtual, p.log_t);
    let lookup_table_flags: Vec<_> = (0..NUM_LOOKUP_TABLES)
        .map(|i| {
            pt.add(
                LookupTableFlag(i),
                &format!("LookupTableFlag_{i}"),
                Virtual,
                p.log_t,
            )
        })
        .collect();

    // --- Virtual — bytecode read values ---
    let bc_read_val: Vec<_> = (0..5)
        .map(|s| {
            pt.add(
                BytecodeReadRafVal(s),
                &format!("BcReadVal_{s}"),
                Virtual,
                p.log_t,
            )
        })
        .collect();

    // --- Virtual — HW reduction pushforward ---
    let total_d = p.instruction_d + p.bytecode_d + p.ram_d;
    let hw_g: Vec<_> = (0..total_d)
        .map(|i| pt.add(HammingG(i), &format!("G_{i}"), Virtual, p.log_k_chunk))
        .collect();

    // --- Virtual — Spartan uniskip evaluations ---
    let outer_uniskip_eval = pt.add(OuterUniskipEval, "OuterUniskipEval", Virtual, 0);
    let product_uniskip_eval = pt.add(ProductUniskipEval, "ProductUniskipEval", Virtual, 0);

    // --- Virtual — advice address phase ---
    let trusted_advice_addr = pt.add(
        PolynomialId::TrustedAdviceAddr,
        "TrustedAdviceAddr",
        Virtual,
        p.log_k_ram,
    );
    let untrusted_advice_addr = pt.add(
        PolynomialId::UntrustedAdviceAddr,
        "UntrustedAdviceAddr",
        Virtual,
        p.log_k_ram,
    );

    // --- Public — preprocessed ---
    let pp = PolyKind::Public(jolt_compiler::PublicPoly::Preprocessed);
    let io_mask = pt.add(IoMask, "IoMask", pp.clone(), p.log_k_ram);
    let val_io = pt.add(ValIo, "ValIo", pp.clone(), p.log_k_ram);
    let ram_unmap = pt.add(PolynomialId::RamUnmap, "RamUnmap", pp.clone(), p.log_k_ram);
    let ram_init = pt.add(PolynomialId::RamInit, "RamInit", pp.clone(), p.log_k_ram);
    let lookup_table = pt.add(
        PolynomialId::LookupTable,
        "LookupTable",
        pp.clone(),
        p.log_k_chunk,
    );
    let bc_table: Vec<_> = (0..5)
        .map(|s| {
            pt.add(
                PolynomialId::BytecodeTable(s),
                &format!("BcTable_{s}"),
                pp.clone(),
                p.log_k_bytecode,
            )
        })
        .collect();

    Polys {
        rd_inc,
        ram_inc,
        instruction_ra,
        ram_ra,
        bytecode_ra,
        untrusted_advice,
        trusted_advice,
        az,
        bz,
        spartan_eq,
        product_left,
        product_right,
        left_instruction_input,
        right_instruction_input,
        product,
        should_branch,
        pc,
        unexpanded_pc,
        imm,
        ram_address,
        rs1_val,
        rs2_val,
        rd_write_value,
        ram_read_value,
        ram_write_value,
        left_lookup_operand,
        right_lookup_operand,
        next_unexpanded_pc,
        next_pc,
        next_is_virtual,
        next_is_first,
        lookup_output,
        should_jump,
        op_flags,
        next_is_noop,
        rd,
        inst_flag_left_is_pc,
        inst_flag_right_is_imm,
        inst_flag_left_is_rs1,
        inst_flag_right_is_rs2,
        inst_flag_branch,
        inst_flag_is_noop,
        reg_wa,
        reg_ra_rs1,
        reg_ra_rs2,
        reg_val,
        ram_combined_ra,
        ram_val,
        ram_val_final,
        ram_wa,
        hamming_weight,
        ram_eq_cycle,
        ram_eq_addr,
        ram_ra_indicator,
        ram_raf_ra,
        inst_raf_ra,
        bytecode_raf_ra,
        inst_raf_flag,
        lookup_table_flags,
        bc_read_val,
        hw_g,
        outer_uniskip_eval,
        product_uniskip_eval,
        trusted_advice_addr,
        untrusted_advice_addr,
        io_mask,
        val_io,
        ram_unmap,
        ram_init,
        lookup_table,
        bc_table,
    }
}

// ---------------------------------------------------------------------------
// Module construction — one function per phase/stage.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Challenge table helper.
// ---------------------------------------------------------------------------

struct ChallengeTable {
    decls: Vec<ChallengeDecl>,
}

impl ChallengeTable {
    fn new() -> Self {
        Self { decls: Vec::new() }
    }

    fn add(&mut self, name: &str, source: ChallengeSource) -> usize {
        let idx = self.decls.len();
        self.decls.push(ChallengeDecl {
            name: name.to_string(),
            source,
        });
        idx
    }

    fn into_vec(self) -> Vec<ChallengeDecl> {
        self.decls
    }
}

// ---------------------------------------------------------------------------
// R1CS input polynomial indices (35 entries, matching jolt-core's
// `ALL_R1CS_INPUTS` ordering — determines transcript flush order).
// ---------------------------------------------------------------------------

/// R1CS input poly IDs in ALL_R1CS_INPUTS order.
/// This is the EXACT order jolt-core's outer sumcheck flushes evaluations.
fn r1cs_input_polys(p: &Polys) -> [PolynomialId; NUM_R1CS_INPUTS] {
    [
        p.left_instruction_input,  //  0: LeftInstructionInput
        p.right_instruction_input, //  1: RightInstructionInput
        p.product,                 //  2: Product
        p.should_branch,           //  3: ShouldBranch
        p.pc,                      //  4: PC
        p.unexpanded_pc,           //  5: UnexpandedPC
        p.imm,                     //  6: Imm
        p.ram_address,             //  7: RamAddress
        p.rs1_val,                 //  8: Rs1Value
        p.rs2_val,                 //  9: Rs2Value
        p.rd_write_value,          // 10: RdWriteValue
        p.ram_read_value,          // 11: RamReadValue
        p.ram_write_value,         // 12: RamWriteValue
        p.left_lookup_operand,     // 13: LeftLookupOperand
        p.right_lookup_operand,    // 14: RightLookupOperand
        p.next_unexpanded_pc,      // 15: NextUnexpandedPC
        p.next_pc,                 // 16: NextPC
        p.next_is_virtual,         // 17: NextIsVirtual
        p.next_is_first,           // 18: NextIsFirstInSequence
        p.lookup_output,           // 19: LookupOutput
        p.should_jump,             // 20: ShouldJump
        p.op_flags[0],             // 21: OpFlags(AddOperands)
        p.op_flags[1],             // 22: OpFlags(SubtractOperands)
        p.op_flags[2],             // 23: OpFlags(MultiplyOperands)
        p.op_flags[3],             // 24: OpFlags(Load)
        p.op_flags[4],             // 25: OpFlags(Store)
        p.op_flags[5],             // 26: OpFlags(Jump)
        p.op_flags[6],             // 27: OpFlags(WriteLookupOutputToRD)
        p.op_flags[7],             // 28: OpFlags(VirtualInstruction)
        p.op_flags[8],             // 29: OpFlags(Assert)
        p.op_flags[9],             // 30: OpFlags(DoNotUpdateUnexpandedPC)
        p.op_flags[10],            // 31: OpFlags(Advice)
        p.op_flags[11],            // 32: OpFlags(IsCompressed)
        p.op_flags[12],            // 33: OpFlags(IsFirstInSequence)
        p.op_flags[13],            // 34: OpFlags(IsLastInSequence)
    ]
}

fn build_module(params: &ModuleParams) -> Module {
    let mut pt = PolyTable::new();
    let p = register_polys(&mut pt, params);

    let mut ops: Vec<Op> = Vec::new();
    let mut kernels: Vec<KernelDef> = Vec::new();
    let mut ch = ChallengeTable::new();

    // Phase 0: Preamble + Commitment
    ops.push(Op::Preamble);
    build_commitment_phase(&p, params, &mut ops);

    // Stage 1: Outer Spartan (uniskip + remaining)
    ops.push(Op::BeginStage { index: 0 });
    build_stage1(&p, params, &mut ops, &mut kernels, &mut ch);

    // Stage 2: Product + RamRW + InstructionClaimReduction + RafEval + OutputCheck
    ops.push(Op::BeginStage { index: 1 });
    let mut batched_sumchecks = Vec::new();
    build_stage2(
        &p,
        params,
        &mut ops,
        &mut kernels,
        &mut ch,
        &mut batched_sumchecks,
    );

    // TODO: Stage 3 — Shift + InstructionInput + RegistersClaimReduction
    // TODO: Stage 4 — RegistersRW + RamValCheck
    // TODO: Stage 5 — InstructionReadRaf + RamRaReduction + RegistersValEval
    // TODO: Stage 6 — BytecodeRaf + Booleanity + HammingBool + RamRaVirt + InstRaVirt + IncReduction + Advice(Cycle)
    // TODO: Stage 7 — HammingWeightReduction + Advice(Address)
    // TODO: Stage 8 — Dory batch opening proof

    // Build verifier schedule
    let mut verifier_ops = vec![VerifierOp::Preamble];
    verifier_ops.extend(build_verifier_stage1_ops(&p, params, &ch));
    verifier_ops.extend(build_verifier_stage2_ops(&p, params, &ch));
    verifier_ops.push(VerifierOp::VerifyOpenings);

    let polys = pt.into_vec();
    let challenges = ch.into_vec();
    let num_polys = polys.len();
    let num_challenges = challenges.len();

    Module {
        polys,
        challenges,
        prover: Schedule {
            ops,
            kernels,
            batched_sumchecks,
        },
        verifier: VerifierSchedule {
            ops: verifier_ops,
            num_challenges,
            num_polys,
            num_stages: 2,
        },
    }
}

/// Phase 0: Commitment — emit polynomial commitments to the transcript.
///
/// jolt-core commits in three separate transcript barriers:
///   1. Main witness: RdInc, RamInc, InstructionRa[..], RamRa[..], BytecodeRa[..]
///   2. UntrustedAdvice (separate tag: b"untrusted_advice")
///   3. TrustedAdvice (separate tag: b"trusted_advice")
fn build_commitment_phase(p: &Polys, params: &ModuleParams, ops: &mut Vec<Op>) {
    // Barrier 1: main witness polynomials
    let mut main_witness = Vec::with_capacity(params.num_committed);
    main_witness.push(p.rd_inc);
    main_witness.push(p.ram_inc);
    main_witness.extend_from_slice(&p.instruction_ra);
    main_witness.extend_from_slice(&p.ram_ra);
    main_witness.extend_from_slice(&p.bytecode_ra);
    // Main witness grid: K_chunk × T (matches DoryGlobals::initialize_context(K, T, Main))
    let main_num_vars = params.log_k_chunk + params.log_t;
    ops.push(Op::Commit {
        polys: main_witness,
        tag: DomainSeparator::Commitment,
        num_vars: main_num_vars,
    });

    // Barrier 2: untrusted advice (trace-length grid)
    ops.push(Op::Commit {
        polys: vec![p.untrusted_advice],
        tag: DomainSeparator::UntrustedAdvice,
        num_vars: params.log_t,
    });

    // Barrier 3: trusted advice (trace-length grid)
    ops.push(Op::Commit {
        polys: vec![p.trusted_advice],
        tag: DomainSeparator::TrustedAdvice,
        num_vars: params.log_t,
    });
}

/// Stage 1: Outer Spartan — univariate skip + streaming remaining rounds.
///
/// Transcript sequence (must match jolt-core exactly):
///   1. Squeeze τ (params.num_tau = 27 challenges)
///   2. Outer uniskip: emit s1(Y) coefficients → squeeze r0 → flush 1 opening claim
///   3. Outer remaining (batched, 1 instance):
///      a. Emit 1 input claim (= s1(r0), tag "sumcheck_claim")
///      b. Squeeze 1 batching coefficient
///      c. 26 rounds: emit compressed round poly → squeeze r_j
///      d. Flush 35 R1CS input opening claims (tag "opening_claim")
///
/// The outer uniskip evaluates the Spartan R1CS identity:
///   s1(Y) = L(τ_high, Y) · Σ_{x_out, x_in} E(τ_low, x_out∥x_in) · Az(x, Y) · Bz(x, Y)
///
/// The outer remaining binds cycle variables with the composition:
///   eq(τ_low, x) · Az(x) · Bz(x)   →  degree 3 (cubic round polys)
///
/// Produces 35 virtual polynomial openings at r_cycle (the R1CS input evaluations).
fn build_stage1(
    p: &Polys,
    params: &ModuleParams,
    ops: &mut Vec<Op>,
    kernels: &mut Vec<KernelDef>,
    ch: &mut ChallengeTable,
) {
    // ---------------------------------------------------------------
    // 1. Squeeze τ = [τ_low ∥ τ_high] (27 challenges)
    //
    // τ_high (last element) is the Lagrange kernel argument for uniskip.
    // τ_low (first 26 elements) seeds the eq table for the remaining rounds.
    // ---------------------------------------------------------------
    let tau_base = ch.decls.len();
    for i in 0..params.num_tau {
        let idx = ch.add(
            &format!("tau_{i}"),
            ChallengeSource::FiatShamir { after_stage: 0 },
        );
        ops.push(Op::Squeeze { challenge: idx });
    }

    // ---------------------------------------------------------------
    // 2. Outer Uniskip: 1 round producing a degree-27 polynomial.
    //
    // The inner composition is the same as the remaining rounds:
    //   eq(τ_low, x) · Az(x) · Bz(x)
    //
    // The uniskip evaluates this sum at each of the 19 constraint
    // domain points, interpolates t1(Y) (degree 18), and multiplies
    // by the Lagrange kernel L(τ_high, Y) (degree 9) to produce
    // s1(Y) of degree 27. The degree-27 comes from protocol-level
    // Lagrange multiplication, not from the composition itself.
    // ---------------------------------------------------------------
    let spartan_formula = Formula::from_terms(vec![ProductTerm {
        coefficient: 1,
        factors: vec![Factor::Input(0), Factor::Input(1), Factor::Input(2)],
    }]);
    let spartan_inputs = vec![
        // Input(0): eq table from τ_low
        InputBinding::EqTable {
            poly: p.spartan_eq,
            challenges: (tau_base..tau_base + params.num_tau - 1).collect(),
        },
        // Input(1): Az — R1CS A-matrix × SpartanWitness
        InputBinding::Provided { poly: p.az },
        // Input(2): Bz — R1CS B-matrix × SpartanWitness
        InputBinding::Provided { poly: p.bz },
    ];

    // τ_high is the last tau challenge — the Lagrange kernel argument.
    let tau_high_idx = tau_base + params.num_tau - 1;

    let outer_uniskip_kernel = kernels.len();
    kernels.push(KernelDef {
        spec: KernelSpec {
            formula: spartan_formula.clone(),
            num_evals: 2 * params.outer_uniskip_domain - 1, // 2K-1 = 37
            iteration: Iteration::Domain {
                domain_size: params.outer_uniskip_domain, // K = 19
                stride: params.num_constraints_padded,    // K_pad = 32
                domain_start: 0,
                domain_indexed: vec![false, true, true], // eq=cycle, Az=domain, Bz=domain
                tau_challenge: tau_high_idx,
            },
            binding_order: BindingOrder::LowToHigh,
        },
        inputs: spartan_inputs.clone(),
        num_rounds: 1,
    });

    ops.push(Op::SumcheckRound {
        kernel: outer_uniskip_kernel,
        round: 0,
        bind_challenge: None,
    });
    ops.push(Op::AbsorbRoundPoly {
        kernel: outer_uniskip_kernel,
        num_coeffs: params.outer_uniskip_num_coeffs,
        tag: DomainSeparator::UniskipPoly,
    });

    // Squeeze r0 (uniskip challenge)
    let ch_r0 = ch.add(
        "outer_uniskip_r0",
        ChallengeSource::SumcheckRound {
            stage: VerifierStageIndex(0),
            round: 0,
        },
    );
    ops.push(Op::Squeeze { challenge: ch_r0 });

    // ---------------------------------------------------------------
    // 2b. Lagrange projection: collapse Az/Bz constraint dimension.
    //
    // After the uniskip, Az/Bz have `num_cycles × K_pad` entries.
    // The Lagrange projection evaluates the constraint-domain polynomial
    // at r0, reducing to `num_cycles` entries per group.
    // ---------------------------------------------------------------
    ops.push(Op::LagrangeProject {
        polys: vec![p.az, p.bz],
        challenge: ch_r0,
        domain_size: params.outer_uniskip_domain,
        domain_start: 0,
        stride: params.num_constraints_padded,
        group_offsets: vec![0],
        kernel_tau: Some(tau_high_idx),
    });

    // Flush uniskip opening claim: s1(r0) appended via flush_to_transcript.
    ops.push(Op::Evaluate {
        poly: p.outer_uniskip_eval,
    });
    ops.push(Op::RecordEvals {
        polys: vec![p.outer_uniskip_eval],
    });
    ops.push(Op::AbsorbEvals {
        polys: vec![p.outer_uniskip_eval],
        tag: DomainSeparator::OpeningClaim,
    });

    // ---------------------------------------------------------------
    // 3. Outer Remaining: 26-round streaming sumcheck.
    //
    // Same composition as uniskip — eq(τ_low, x) · Az(x) · Bz(x) —
    // but evaluated via standard hypercube bind-and-reduce (degree 3).
    // ---------------------------------------------------------------
    let outer_remaining_kernel = kernels.len();
    kernels.push(KernelDef {
        spec: KernelSpec {
            formula: spartan_formula,
            num_evals: params.outer_remaining_degree + 1,
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
        },
        inputs: spartan_inputs,
        num_rounds: params.outer_remaining_rounds,
    });

    // Batched sumcheck protocol header:
    //   1. Emit input_claim for each instance → tag "sumcheck_claim"
    //      (for outer remaining, this is s1(r0) retrieved from the accumulator)
    //   2. Squeeze N batching coefficients (N=1 for this single-instance batch)
    ops.push(Op::AbsorbEvals {
        polys: vec![p.outer_uniskip_eval],
        tag: DomainSeparator::SumcheckClaim,
    });
    let ch_batch = ch.add(
        "outer_remaining_batch",
        ChallengeSource::FiatShamir { after_stage: 0 },
    );
    ops.push(Op::Squeeze {
        challenge: ch_batch,
    });

    // Remaining rounds: log_t binary rounds over cycle variables.
    // Round 0 has no bind (initial T-entry buffers from LagrangeProject).
    // Rounds 1+ bind at the previous round's squeezed challenge.
    for round in 0..params.outer_remaining_rounds {
        let bind = if round > 0 {
            Some(ch_batch + round) // previous round's challenge
        } else {
            None
        };
        ops.push(Op::SumcheckRound {
            kernel: outer_remaining_kernel,
            round,
            bind_challenge: bind,
        });

        let ch_r = ch.add(
            &format!("outer_r_{round}"),
            ChallengeSource::SumcheckRound {
                stage: VerifierStageIndex(0),
                round: round + 1,
            },
        );
        ops.push(Op::AbsorbRoundPoly {
            kernel: outer_remaining_kernel,
            num_coeffs: params.outer_remaining_degree + 1,
            tag: DomainSeparator::SumcheckPoly,
        });
        ops.push(Op::Squeeze { challenge: ch_r });
    }

    // ---------------------------------------------------------------
    // 4. Bind R1CS witness polynomials at the sumcheck challenge point,
    //    then evaluate + flush.
    //
    // The remaining sumcheck only bound the kernel inputs (eq, Az, Bz).
    // The 35 R1CS witness polynomials (T-length buffers) must be bound
    // at each of the 9 sumcheck challenges to produce scalar evaluations.
    // ---------------------------------------------------------------
    let r1cs_polys = r1cs_input_polys(p);
    for round in 0..params.outer_remaining_rounds {
        // challenge index = ch_batch + 1 + round (first sumcheck challenge
        // is at ch_batch+1 because ch_batch is the batching coefficient)
        let ch_idx = ch_batch + 1 + round;
        ops.push(Op::Bind {
            polys: r1cs_polys.to_vec(),
            challenge: ch_idx,
            order: BindingOrder::LowToHigh,
        });
    }
    for &poly in &r1cs_polys {
        ops.push(Op::Evaluate { poly });
    }
    ops.push(Op::RecordEvals {
        polys: r1cs_polys.to_vec(),
    });
    ops.push(Op::AbsorbEvals {
        polys: r1cs_polys.to_vec(),
        tag: DomainSeparator::OpeningClaim,
    });
}

/// Verifier schedule for Stage 1: Outer Spartan.
///
/// Single verifier stage covering the full Stage 1 protocol:
///   1. Absorb 25 commitments (3 barriers)
///   2. Squeeze 27 τ challenges
///   3. Verify uniskip: absorb 28-coeff polynomial, squeeze r0, record s1(r0)
///   4. Verify remaining: input claim = s1(r0), 26 rounds of degree-3 sumcheck
///   5. Record 35 R1CS input evaluations
///   6. Check output: eq(τ_low, r) × L(τ_high, r0) × Az(r0, evals) × Bz(r0, evals)
fn build_verifier_stage1_ops(
    p: &Polys,
    params: &ModuleParams,
    ch: &ChallengeTable,
) -> Vec<VerifierOp> {
    // Commitment list: per-barrier (poly, tag) pairs matching the prover's
    // commitment phase. Tags must be identical for Fiat-Shamir consistency.
    let mut commit_pairs: Vec<(PolynomialId, DomainSeparator)> = Vec::with_capacity(25);
    // Barrier 1: main witness → tag "commitment"
    for &poly in [p.rd_inc, p.ram_inc]
        .iter()
        .chain(p.instruction_ra.iter())
        .chain(p.ram_ra.iter())
        .chain(p.bytecode_ra.iter())
    {
        commit_pairs.push((poly, DomainSeparator::Commitment));
    }
    // Barrier 2: untrusted advice
    commit_pairs.push((p.untrusted_advice, DomainSeparator::UntrustedAdvice));
    // Barrier 3: trusted advice
    commit_pairs.push((p.trusted_advice, DomainSeparator::TrustedAdvice));

    // τ challenges: indices 0..27
    let tau_challenges: Vec<usize> = (0..params.num_tau).collect();

    // Evaluations: 35 R1CS inputs produced by the remaining sumcheck
    let r1cs_polys = r1cs_input_polys(p);
    let evaluations: Vec<_> = r1cs_polys
        .iter()
        .map(|&poly| Evaluation {
            poly,
            at_stage: VerifierStageIndex(0), // outer Spartan
        })
        .collect();

    // Input claim: s1(r0) — the uniskip output evaluation
    let input_claim = ClaimFormula {
        terms: vec![ClaimTerm {
            coeff: 1,
            factors: vec![ClaimFactor::Eval(p.outer_uniskip_eval)],
        }],
    };

    // Output check: the verifier evaluates the composition at the final sumcheck point.
    //
    //   eq(τ_cycle, r_remaining) × L(τ_high, r0) × Az(r0, evals) × Bz(r0, evals)
    //
    // The uniskip handled the constraint dimension via the Lagrange kernel
    // L(τ_high, ·). The remaining sumcheck bound all log_t cycle variables.
    // The eq table has num_tau-1 = log_t cycle variables (τ_0..τ_{log_t-1}).
    let tau_high_idx = params.num_tau - 1;
    let r0_idx = params.num_tau; // the uniskip challenge
    let tau_cycle: Vec<usize> = (0..params.num_tau - 1).collect();
    let output_check = ClaimFormula {
        terms: vec![ClaimTerm {
            coeff: 1,
            factors: vec![
                ClaimFactor::EqEval {
                    challenges: tau_cycle,
                    at_stage: VerifierStageIndex(0),
                },
                ClaimFactor::LagrangeKernelDomain {
                    tau_challenge: tau_high_idx,
                    at_challenge: r0_idx,
                    domain_size: params.outer_uniskip_domain,
                },
                ClaimFactor::UniformR1CSEval {
                    matrix: R1CSMatrix::A,
                    eval_polys: r1cs_polys.to_vec(),
                    at_challenge: r0_idx,
                    num_constraints: params.outer_uniskip_domain,
                },
                ClaimFactor::UniformR1CSEval {
                    matrix: R1CSMatrix::B,
                    eval_polys: r1cs_polys.to_vec(),
                    at_challenge: r0_idx,
                    num_constraints: params.outer_uniskip_domain,
                },
            ],
        }],
    };

    let instances = vec![SumcheckInstance {
        input_claim,
        output_check,
        num_rounds: params.outer_remaining_rounds,
        degree: params.outer_remaining_degree,
        // LowToHigh binding fixes variables LSB-first, so the sumcheck
        // point is reversed relative to the eq table's challenge ordering.
        normalize: Some(PointNormalization::Reverse),
    }];

    let eval_polys: Vec<_> = evaluations.iter().map(|e| e.poly).collect();

    // Uniskip evaluation descriptor — prover emits this BEFORE remaining rounds.
    let uniskip_eval = Evaluation {
        poly: p.outer_uniskip_eval,
        at_stage: VerifierStageIndex(0),
    };

    let mut ops = Vec::new();
    ops.push(VerifierOp::BeginStage);
    for &(poly, ref tag) in &commit_pairs {
        ops.push(VerifierOp::AbsorbCommitment {
            poly,
            tag: tag.clone(),
        });
    }
    for &c in &tau_challenges {
        ops.push(VerifierOp::Squeeze { challenge: c });
    }
    // --- Uniskip protocol: absorb poly → squeeze r0 → record/absorb eval ---
    // TODO: full uniskip verification (check s(0)+s(1), constraint domain evaluation).
    ops.push(VerifierOp::AbsorbRoundPoly {
        num_coeffs: params.outer_uniskip_num_coeffs,
        tag: DomainSeparator::UniskipPoly,
    });
    ops.push(VerifierOp::Squeeze { challenge: r0_idx });
    ops.push(VerifierOp::RecordEvals {
        evals: vec![uniskip_eval],
    });
    // Prover flushes uniskip eval twice: first as OpeningClaim, then as SumcheckClaim.
    ops.push(VerifierOp::AbsorbEvals {
        polys: vec![p.outer_uniskip_eval],
        tag: DomainSeparator::OpeningClaim,
    });
    ops.push(VerifierOp::AbsorbEvals {
        polys: vec![p.outer_uniskip_eval],
        tag: DomainSeparator::SumcheckClaim,
    });
    // Batch coefficient for the (single-instance) remaining sumcheck.
    let ch_batch = r0_idx + 1;
    ops.push(VerifierOp::Squeeze {
        challenge: ch_batch,
    });
    ops.push(VerifierOp::VerifySumcheck {
        instances: instances.clone(),
        stage: 0,
        batch_challenges: Vec::new(),
        claim_tag: None,
    });
    ops.push(VerifierOp::RecordEvals { evals: evaluations });
    ops.push(VerifierOp::AbsorbEvals {
        polys: eval_polys,
        tag: DomainSeparator::OpeningClaim,
    });
    ops.push(VerifierOp::CheckOutput {
        instances,
        stage: 0,
        batch_challenges: Vec::new(),
    });
    ops
}

/// Stage 2: Product virtualization + batched sumcheck.
///
/// Transcript sequence (matches jolt-core exactly):
///   1. Squeeze τ_high (1 challenge for product uniskip Lagrange kernel)
///   2. Product uniskip: emit s2(Y) coefficients (7) → squeeze r0 → flush 1 opening
///   3. Squeeze γ_rw (1 RamReadWriteChecking challenge)
///   4. Squeeze γ_instruction (1 InstructionLookupsClaimReduction challenge)
///   5. Squeeze r_address (params.log_k_ram = 20 OutputCheck address challenges)
///   6. Batched sumcheck (5 instances):
///      a. Emit 5 input claims (tag "sumcheck_claim")
///      b. Squeeze 5 batching coefficients
///      c. 45 rounds: emit compressed round poly → squeeze r_j
///      d. Flush 18 evaluation opening claims (tag "opening_claim")
///
/// Instance ordering (matches jolt-core prover.rs:971):
///   [0] RamReadWriteChecking       — 45 rounds, degree 3
///   [1] ProductVirtualRemainder    — 25 rounds, degree 3
///   [2] InstructionClaimReduction  — 25 rounds, degree 2
///   [3] RafEvaluation              — 20 rounds, degree 2
///   [4] OutputCheck                — 20 rounds, degree 3
fn build_stage2(
    p: &Polys,
    params: &ModuleParams,
    ops: &mut Vec<Op>,
    kernels: &mut Vec<KernelDef>,
    ch: &mut ChallengeTable,
    batched_sumchecks: &mut Vec<BatchedSumcheckDef>,
) {
    // Stage 1 cycle challenge indices: the outer remaining round challenges,
    // reversed to BIG_ENDIAN order (matching jolt-core's r_cycle convention).
    // These are used as τ_low for both the product uniskip and ProductRemainder.
    let stage1_round_base = params.num_tau + 2; // τ_count + r0 + batch
    let stage1_cycle_challenges: Vec<usize> =
        (stage1_round_base..stage1_round_base + params.outer_remaining_rounds)
            .rev()
            .collect();

    // ---------------------------------------------------------------
    // 1. Squeeze τ_high for product uniskip
    //    (τ_low is r_cycle from Stage 1 outer remaining round challenges)
    // ---------------------------------------------------------------
    let ch_product_tau_high = ch.add(
        "product_tau_high",
        ChallengeSource::FiatShamir { after_stage: 1 },
    );
    ops.push(Op::Squeeze {
        challenge: ch_product_tau_high,
    });

    // ---------------------------------------------------------------
    // 2. Product uniskip: 1 round producing a degree-6 polynomial.
    //
    //   s2(Y) = L(τ_high, Y) · Σ_x eq(r_cycle, x) · Left(x, Y) · Right(x, Y)
    //
    // τ_low = r_cycle (Stage 1 outer remaining round challenges), NOT
    // the original Spartan tau. In jolt-core, ProductVirtualUniSkipParams
    // retrieves r_cycle from the opening accumulator.
    // ---------------------------------------------------------------
    let product_uniskip_formula = Formula::from_terms(vec![ProductTerm {
        coefficient: 1,
        factors: vec![Factor::Input(0), Factor::Input(1), Factor::Input(2)],
    }]);
    let product_uniskip_inputs = vec![
        InputBinding::EqTable {
            poly: PolynomialId::BatchEq(0),
            challenges: stage1_cycle_challenges.clone(),
        },
        InputBinding::Provided {
            poly: p.product_left,
        },
        InputBinding::Provided {
            poly: p.product_right,
        },
    ];

    let product_stride = (params.product_uniskip_domain).next_power_of_two();
    let product_uniskip_kernel = kernels.len();
    kernels.push(KernelDef {
        spec: KernelSpec {
            formula: product_uniskip_formula,
            num_evals: 2 * params.product_uniskip_domain - 1,
            iteration: Iteration::Domain {
                domain_size: params.product_uniskip_domain,
                stride: product_stride,
                domain_start: 0,
                domain_indexed: vec![false, true, true],
                tau_challenge: ch_product_tau_high,
            },
            binding_order: BindingOrder::LowToHigh,
        },
        inputs: product_uniskip_inputs,
        num_rounds: 1,
    });

    ops.push(Op::SumcheckRound {
        kernel: product_uniskip_kernel,
        round: 0,
        bind_challenge: None,
    });
    ops.push(Op::AbsorbRoundPoly {
        kernel: product_uniskip_kernel,
        num_coeffs: params.product_uniskip_num_coeffs,
        tag: DomainSeparator::UniskipPoly,
    });

    // Squeeze r0 (product uniskip challenge)
    let ch_product_r0 = ch.add(
        "product_uniskip_r0",
        ChallengeSource::SumcheckRound {
            stage: VerifierStageIndex(1),
            round: 0,
        },
    );
    ops.push(Op::Squeeze {
        challenge: ch_product_r0,
    });

    // Flush product uniskip opening claim: s2(r0)
    ops.push(Op::Evaluate {
        poly: p.product_uniskip_eval,
    });
    ops.push(Op::RecordEvals {
        polys: vec![p.product_uniskip_eval],
    });
    ops.push(Op::AbsorbEvals {
        polys: vec![p.product_uniskip_eval],
        tag: DomainSeparator::OpeningClaim,
    });

    // ---------------------------------------------------------------
    // 2b. Lagrange projection: collapse product_left/right domain dim.
    //
    // After the product uniskip, product_left/right are T × stride
    // domain-indexed buffers. The Lagrange projection evaluates at r0,
    // reducing to T-element vectors. kernel_tau absorbs L(τ_high, r0)
    // into the first poly, so the ProductRemainder sum equals s2(r0).
    // ---------------------------------------------------------------
    ops.push(Op::LagrangeProject {
        polys: vec![p.product_left, p.product_right],
        challenge: ch_product_r0,
        domain_size: params.product_uniskip_domain,
        domain_start: 0,
        stride: product_stride,
        group_offsets: vec![0],
        kernel_tau: Some(ch_product_tau_high),
    });

    // ---------------------------------------------------------------
    // 3. Pre-squeeze challenges for batched instances
    // ---------------------------------------------------------------

    // γ_rw (RamReadWriteChecking mixing challenge)
    let ch_gamma_rw = ch.add("gamma_rw", ChallengeSource::FiatShamir { after_stage: 1 });
    ops.push(Op::Squeeze {
        challenge: ch_gamma_rw,
    });

    // γ_instruction (InstructionLookupsClaimReduction mixing challenge)
    let ch_gamma_instruction = ch.add(
        "gamma_instruction",
        ChallengeSource::FiatShamir { after_stage: 1 },
    );
    ops.push(Op::Squeeze {
        challenge: ch_gamma_instruction,
    });

    // r_address (OutputCheck address challenges, params.log_k_ram = 20)
    let ch_r_address_base = ch.decls.len();
    for i in 0..params.log_k_ram {
        let idx = ch.add(
            &format!("output_r_address_{i}"),
            ChallengeSource::FiatShamir { after_stage: 1 },
        );
        ops.push(Op::Squeeze { challenge: idx });
    }

    // ---------------------------------------------------------------
    // 4. Batched sumcheck (5 instances)
    //
    // Each instance's input_claim is computed from prior evaluations
    // and appended to transcript with "sumcheck_claim" tag.
    // Then 5 batching coefficients are squeezed.
    // ---------------------------------------------------------------

    // Emit 5 input claims. In jolt-core, each instance's input_claim()
    // is computed from the opening accumulator. For the Module, we model
    // these as evaluations of "claim polys" that the runtime computes.
    //
    // The actual claim values are:
    //   [0] RamRW: rv_claim + gamma_rw * wv_claim
    //   [1] ProductRemainder: s2(r0) (product uniskip eval)
    //   [2] InstructionCR: lookup + gamma*left + gamma^2*right + ...
    //   [3] RafEval: ram_address_claim * 2^(phase3_cycle_rounds)
    //   [4] Output: 0 (zero-check)
    //
    // For transcript parity, we emit them in instance order.
    // NOTE: The runtime needs a ComputeClaim-style op to evaluate these
    // formulas. For now, we model the transcript structure directly.

    // ---------------------------------------------------------------
    // 4a. Emit 5 input claims via AbsorbInputClaim.
    //
    // Each ClaimFormula is evaluated by the runtime against current
    // state.evaluations and state.challenges, then absorbed into the
    // transcript with "sumcheck_claim" tag. The batch/instance indices
    // allow the runtime to initialize per-instance claims for inactive
    // round contributions.
    // ---------------------------------------------------------------
    let batch_idx = batched_sumchecks.len();

    // [0] RamReadWriteChecking: ram_read_value + γ_rw * ram_write_value
    let rw_input_claim = ClaimFormula {
        terms: vec![
            ClaimTerm {
                coeff: 1,
                factors: vec![ClaimFactor::Eval(p.ram_read_value)],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_gamma_rw),
                    ClaimFactor::Eval(p.ram_write_value),
                ],
            },
        ],
    };
    ops.push(Op::AbsorbInputClaim {
        formula: rw_input_claim.clone(),
        tag: DomainSeparator::SumcheckClaim,
        batch: batch_idx,
        instance: 0,
    });

    // [1] ProductVirtualRemainder: product_uniskip_eval (= s2(r0))
    let prod_input_claim = ClaimFormula {
        terms: vec![ClaimTerm {
            coeff: 1,
            factors: vec![ClaimFactor::Eval(p.product_uniskip_eval)],
        }],
    };
    ops.push(Op::AbsorbInputClaim {
        formula: prod_input_claim.clone(),
        tag: DomainSeparator::SumcheckClaim,
        batch: batch_idx,
        instance: 1,
    });

    // [2] InstructionClaimReduction: lookup + γ*left_op + γ²*right_op + γ³*left_inst + γ⁴*right_inst
    let inst_cr_input_claim = ClaimFormula {
        terms: vec![
            ClaimTerm {
                coeff: 1,
                factors: vec![ClaimFactor::Eval(p.lookup_output)],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_gamma_instruction),
                    ClaimFactor::Eval(p.left_lookup_operand),
                ],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_gamma_instruction),
                    ClaimFactor::Challenge(ch_gamma_instruction),
                    ClaimFactor::Eval(p.right_lookup_operand),
                ],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_gamma_instruction),
                    ClaimFactor::Challenge(ch_gamma_instruction),
                    ClaimFactor::Challenge(ch_gamma_instruction),
                    ClaimFactor::Eval(p.left_instruction_input),
                ],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_gamma_instruction),
                    ClaimFactor::Challenge(ch_gamma_instruction),
                    ClaimFactor::Challenge(ch_gamma_instruction),
                    ClaimFactor::Challenge(ch_gamma_instruction),
                    ClaimFactor::Eval(p.right_instruction_input),
                ],
            },
        ],
    };
    ops.push(Op::AbsorbInputClaim {
        formula: inst_cr_input_claim.clone(),
        tag: DomainSeparator::SumcheckClaim,
        batch: batch_idx,
        instance: 2,
    });

    // [3] RafEvaluation: ram_address
    let raf_input_claim = ClaimFormula {
        terms: vec![ClaimTerm {
            coeff: 1,
            factors: vec![ClaimFactor::Eval(p.ram_address)],
        }],
    };
    ops.push(Op::AbsorbInputClaim {
        formula: raf_input_claim.clone(),
        tag: DomainSeparator::SumcheckClaim,
        batch: batch_idx,
        instance: 3,
    });

    // [4] OutputCheck: 0 (zero-check)
    let output_check_input_claim = ClaimFormula::zero();
    ops.push(Op::AbsorbInputClaim {
        formula: output_check_input_claim.clone(),
        tag: DomainSeparator::SumcheckClaim,
        batch: batch_idx,
        instance: 4,
    });

    // Squeeze 5 batching coefficients
    let ch_batch_base = ch.decls.len();
    for i in 0..params.stage2_num_instances {
        let idx = ch.add(
            &format!("s2_batch_{i}"),
            ChallengeSource::FiatShamir { after_stage: 1 },
        );
        ops.push(Op::Squeeze { challenge: idx });
    }

    // ---------------------------------------------------------------
    // 5. Build per-instance kernels for the batched sumcheck.
    // ---------------------------------------------------------------

    // [0] RamReadWriteChecking — 2-phase: log_t cycle + log_k_ram address.
    //
    // The RamRW polynomial is eq_cycle(t) * ra(k,t) * ((1+γ)*val(k,t) + γ*inc(t)).
    // NO eq_addr — the address dimension is an unweighted sum over K.
    //
    // Phase 1 (cycle, log_t rounds, degree 3, segmented):
    //   eq_cycle * ra * val + γ_rw * eq_cycle * ra * val + γ_rw * eq_cycle * ra * inc
    //   Segmented with UNIFORM outer weights (no eq_addr).
    //   Mixed inputs: ra(T×K addr-major), val(T×K addr-major). Inner: eq_cycle(T), inc(T).
    //
    // Phase 2 (address, log_k_ram rounds, degree 2):
    //   After phase 1, eq_cycle/inc are fully bound (scalars via ScalarCapture).
    //   ra, val are K-element (carried from phase 1 binding).
    //   Formula: eq_bound*(1+γ)*ra*val + γ*eq_bound*inc_bound*ra
    //   Dense over K-element buffers, no eq_addr.

    // Allocate challenge slots for scalar captures at the phase boundary.
    // ra is T×K mixed — it reduces to K elements after phase 1, not a scalar.
    let ch_rw_eq_bound = ch.add("rw_eq_cycle_bound", ChallengeSource::External);
    let ch_rw_inc_bound = ch.add("rw_inc_bound", ChallengeSource::External);

    // RamRW cycle challenge indices for the eq_cycle table:
    // After Stage 1 Reverse normalization, the cycle point comes from
    // the Stage 1 remaining round challenges reversed.
    let rw_cycle_eq_challenges: Vec<usize> = stage1_cycle_challenges.clone();

    // Address challenge indices for the outer eq table:
    let rw_addr_eq_challenges: Vec<usize> =
        (ch_r_address_base..ch_r_address_base + params.log_k_ram).collect();

    // Phase 1 kernel: cycle binding (Dense inner kernel, segmented reduce).
    let rw_phase1_kernel_idx = kernels.len();
    kernels.push(KernelDef {
        spec: KernelSpec {
            formula: Formula::from_terms(vec![
                ProductTerm {
                    coefficient: 1,
                    factors: vec![Factor::Input(0), Factor::Input(1), Factor::Input(2)],
                },
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(ch_gamma_rw as u32),
                        Factor::Input(0),
                        Factor::Input(1),
                        Factor::Input(2),
                    ],
                },
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(ch_gamma_rw as u32),
                        Factor::Input(0),
                        Factor::Input(1),
                        Factor::Input(3),
                    ],
                },
            ]),
            num_evals: params.rw_checking_degree + 1,
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
        },
        inputs: vec![
            InputBinding::EqTable {
                poly: p.ram_eq_cycle,
                challenges: rw_cycle_eq_challenges.clone(),
            },
            InputBinding::Provided {
                poly: p.ram_combined_ra,
            },
            InputBinding::Provided { poly: p.ram_val },
            InputBinding::Provided { poly: p.ram_inc },
        ],
        num_rounds: params.log_t,
    });

    // Phase 2 kernel: address binding (Dense, K-element inputs).
    //
    // After phase 1, eq_cycle → ch_rw_eq_bound, inc → ch_rw_inc_bound (scalars).
    // ra → K elements (carried from phase 1 binding).
    // val → K elements (carried from phase 1 binding).
    // No eq_addr — the RamRW polynomial has no eq_addr factor.
    // Formula terms:
    //   ch_eq * ra * val + γ * ch_eq * ra * val + γ * ch_eq * ch_inc * ra
    // Degree: max 2 inputs (ra × val).
    let rw_phase2_kernel_idx = kernels.len();
    kernels.push(KernelDef {
        spec: KernelSpec {
            formula: Formula::from_terms(vec![
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(ch_rw_eq_bound as u32),
                        Factor::Input(0),
                        Factor::Input(1),
                    ],
                },
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(ch_gamma_rw as u32),
                        Factor::Challenge(ch_rw_eq_bound as u32),
                        Factor::Input(0),
                        Factor::Input(1),
                    ],
                },
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(ch_gamma_rw as u32),
                        Factor::Challenge(ch_rw_eq_bound as u32),
                        Factor::Challenge(ch_rw_inc_bound as u32),
                        Factor::Input(0),
                    ],
                },
            ]),
            // Degree 2: max input factor count is 2 (ra × val)
            num_evals: 3,
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
        },
        inputs: vec![
            InputBinding::Provided {
                poly: p.ram_combined_ra,
            },
            InputBinding::Provided { poly: p.ram_val },
        ],
        num_rounds: params.log_k_ram,
    });

    // [1] ProductVirtualRemainder — 25 rounds, degree 3.
    //
    // Formula: eq(τ_cycle, x) * left(x) * right(x)
    let prod_kernel_idx = kernels.len();
    kernels.push(KernelDef {
        spec: KernelSpec {
            formula: Formula::from_terms(vec![ProductTerm {
                coefficient: 1,
                factors: vec![Factor::Input(0), Factor::Input(1), Factor::Input(2)],
            }]),
            num_evals: params.product_remainder_degree + 1,
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
        },
        inputs: vec![
            InputBinding::EqTable {
                poly: PolynomialId::BatchEq(1),
                challenges: stage1_cycle_challenges.clone(),
            },
            InputBinding::Provided { poly: p.product_left },
            InputBinding::Provided { poly: p.product_right },
        ],
        num_rounds: params.product_remainder_rounds,
    });

    // [2] InstructionClaimReduction — 25 rounds, degree 2.
    //
    // Formula: eq * (lookup + γ·left_op + γ²·right_op + γ³·left_inst + γ⁴·right_inst)
    let gamma = ch_gamma_instruction as u32;
    let inst_cr_kernel_idx = kernels.len();
    kernels.push(KernelDef {
        spec: KernelSpec {
            formula: Formula::from_terms(vec![
                ProductTerm {
                    coefficient: 1,
                    factors: vec![Factor::Input(0), Factor::Input(1)],
                },
                ProductTerm {
                    coefficient: 1,
                    factors: vec![Factor::Challenge(gamma), Factor::Input(0), Factor::Input(2)],
                },
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(gamma),
                        Factor::Challenge(gamma),
                        Factor::Input(0),
                        Factor::Input(3),
                    ],
                },
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(gamma),
                        Factor::Challenge(gamma),
                        Factor::Challenge(gamma),
                        Factor::Input(0),
                        Factor::Input(4),
                    ],
                },
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(gamma),
                        Factor::Challenge(gamma),
                        Factor::Challenge(gamma),
                        Factor::Challenge(gamma),
                        Factor::Input(0),
                        Factor::Input(5),
                    ],
                },
            ]),
            num_evals: params.instruction_claim_reduction_degree + 1,
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
        },
        inputs: vec![
            InputBinding::EqTable {
                poly: PolynomialId::BatchEq(2),
                challenges: stage1_cycle_challenges,
            },
            InputBinding::Provided { poly: p.lookup_output },
            InputBinding::Provided {
                poly: p.left_lookup_operand,
            },
            InputBinding::Provided {
                poly: p.right_lookup_operand,
            },
            InputBinding::Provided {
                poly: p.left_instruction_input,
            },
            InputBinding::Provided {
                poly: p.right_instruction_input,
            },
        ],
        num_rounds: params.instruction_claim_reduction_rounds,
    });

    // [3] RafEvaluation — 20 rounds, degree 2.
    //
    // Formula: unmap(x) * ra(x)
    let raf_kernel_idx = kernels.len();
    kernels.push(KernelDef {
        spec: KernelSpec {
            formula: Formula::from_terms(vec![ProductTerm {
                coefficient: 1,
                factors: vec![Factor::Input(0), Factor::Input(1)],
            }]),
            num_evals: params.raf_evaluation_degree + 1,
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
        },
        inputs: vec![
            InputBinding::Provided { poly: p.ram_unmap },
            InputBinding::EqProject {
                poly: p.ram_raf_ra,
                source: p.ram_ra_indicator,
                challenges: Vec::new(), // filled after round loop
                inner_size: 1 << params.log_t,
                outer_size: 1 << params.log_k_ram,
            },
        ],
        num_rounds: params.raf_evaluation_rounds,
    });

    // [4] OutputCheck — 20 rounds, degree 3.
    //
    // Formula: eq(r_address, x) * mask(x) * val_final(x) - eq * mask * val_io(x)
    let output_kernel_idx = kernels.len();
    kernels.push(KernelDef {
        spec: KernelSpec {
            formula: Formula::from_terms(vec![
                ProductTerm {
                    coefficient: 1,
                    factors: vec![Factor::Input(0), Factor::Input(1), Factor::Input(2)],
                },
                ProductTerm {
                    coefficient: -1,
                    factors: vec![Factor::Input(0), Factor::Input(1), Factor::Input(3)],
                },
            ]),
            num_evals: params.output_check_degree + 1,
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
        },
        inputs: vec![
            InputBinding::EqTable {
                poly: PolynomialId::BatchEq(4),
                challenges: (ch_r_address_base..ch_r_address_base + params.log_k_ram).collect(),
            },
            InputBinding::Provided { poly: p.io_mask },
            InputBinding::Provided {
                poly: p.ram_val_final,
            },
            InputBinding::Provided { poly: p.val_io },
        ],
        num_rounds: params.output_check_rounds,
    });

    // ---------------------------------------------------------------
    // 5b. Build BatchedSumcheckDef with all 5 instances.
    // ---------------------------------------------------------------
    let batch_idx = batched_sumchecks.len();
    batched_sumchecks.push(BatchedSumcheckDef {
        instances: vec![
            // [0] RamRW — 2 phases: cycle (log_t) + address (log_k_ram).
            BatchedInstance {
                phases: vec![
                    InstancePhase {
                        kernel: rw_phase1_kernel_idx,
                        num_rounds: params.log_t,
                        scalar_captures: vec![],
                        segmented: Some(SegmentedConfig {
                            inner_num_vars: params.log_t,
                            outer_num_vars: params.log_k_ram,
                            // [eq_cycle: inner, ra: mixed(T×K), val: mixed(T×K), inc: inner]
                            inner_only: vec![true, false, false, true],
                            // Empty = uniform outer weights (no eq_addr in RamRW formula).
                            outer_eq_challenges: vec![],
                        }),
                    },
                    InstancePhase {
                        kernel: rw_phase2_kernel_idx,
                        num_rounds: params.log_k_ram,
                        // Capture bound scalars from phase 1 into challenge slots.
                        scalar_captures: vec![
                            ScalarCapture {
                                poly: p.ram_eq_cycle,
                                challenge: ch_rw_eq_bound,
                            },
                            ScalarCapture {
                                poly: p.ram_inc,
                                challenge: ch_rw_inc_bound,
                            },
                        ],
                        segmented: None,
                    },
                ],
                batch_coeff: ch_batch_base,
                first_active_round: 0,
            },
            // [1] ProductRemainder — single phase.
            BatchedInstance {
                phases: vec![InstancePhase {
                    kernel: prod_kernel_idx,
                    num_rounds: params.product_remainder_rounds,
                    scalar_captures: vec![],
                    segmented: None,
                }],
                batch_coeff: ch_batch_base + 1,
                first_active_round: params.stage2_max_rounds - params.product_remainder_rounds,
            },
            // [2] InstructionClaimReduction — single phase.
            BatchedInstance {
                phases: vec![InstancePhase {
                    kernel: inst_cr_kernel_idx,
                    num_rounds: params.instruction_claim_reduction_rounds,
                    scalar_captures: vec![],
                    segmented: None,
                }],
                batch_coeff: ch_batch_base + 2,
                first_active_round: params.stage2_max_rounds
                    - params.instruction_claim_reduction_rounds,
            },
            // [3] RafEvaluation — single phase.
            BatchedInstance {
                phases: vec![InstancePhase {
                    kernel: raf_kernel_idx,
                    num_rounds: params.raf_evaluation_rounds,
                    scalar_captures: vec![],
                    segmented: None,
                }],
                batch_coeff: ch_batch_base + 3,
                first_active_round: params.stage2_max_rounds - params.raf_evaluation_rounds,
            },
            // [4] OutputCheck — single phase.
            BatchedInstance {
                phases: vec![InstancePhase {
                    kernel: output_kernel_idx,
                    num_rounds: params.output_check_rounds,
                    scalar_captures: vec![],
                    segmented: None,
                }],
                batch_coeff: ch_batch_base + 4,
                first_active_round: params.stage2_max_rounds - params.output_check_rounds,
            },
        ],
        input_claims: vec![
            rw_input_claim,
            prod_input_claim,
            inst_cr_input_claim,
            raf_input_claim,
            output_check_input_claim,
        ],
        max_rounds: params.stage2_max_rounds,
        max_degree: params.rw_checking_degree,
    });

    // ---------------------------------------------------------------
    // 5c. 45 rounds of BatchedSumcheckRound.
    //
    // Each round: the runtime dispatches to active instance kernels,
    // combines with batching coefficients, produces one round poly.
    // ---------------------------------------------------------------
    let mut round_challenge_indices = Vec::with_capacity(params.stage2_max_rounds);
    for round in 0..params.stage2_max_rounds {
        let bind = if round > 0 {
            Some(round_challenge_indices[round - 1])
        } else {
            None
        };
        ops.push(Op::BatchedSumcheckRound {
            batch: batch_idx,
            round,
            bind_challenge: bind,
        });

        let ch_r = ch.add(
            &format!("s2_r_{round}"),
            ChallengeSource::SumcheckRound {
                stage: VerifierStageIndex(1),
                round: round + 1,
            },
        );
        round_challenge_indices.push(ch_r);
        ops.push(Op::AbsorbRoundPoly {
            kernel: rw_phase1_kernel_idx,
            num_coeffs: params.rw_checking_degree + 1,
            tag: DomainSeparator::SumcheckPoly,
        });
        ops.push(Op::Squeeze { challenge: ch_r });
    }

    // Patch the RAF kernel's EqProject challenges.
    // The EqProject marginalizes the cycle dimension of the RA indicator using
    // the Stage 1 cycle point (where ram_address was evaluated), NOT the Stage 2
    // round challenges. This ensures Σ_k unmap(k) * eq_project(indicator, r) = eval(ram_address, r).
    {
        if let InputBinding::EqProject {
            challenges: ref mut chs,
            ..
        } = kernels[raf_kernel_idx].inputs[1]
        {
            chs.clone_from(&rw_cycle_eq_challenges);
        }
    }

    // ---------------------------------------------------------------
    // 6. Bind polynomials opened by Instance 1 (ProductRemainder)
    //    that are NOT kernel inputs of any batched instance.
    //
    //    Instances 1 and 2 share the same active rounds (13..21),
    //    so their sumcheck points coincide. Instance 2's kernel
    //    already binds: lookup_output, left/right_lookup_operand,
    //    left/right_instruction_input. The remaining 5 polys need
    //    explicit binding at the same (Instance 1/2) challenge point.
    // ---------------------------------------------------------------
    let prod_open_extra = vec![
        p.op_flags[5], // Jump
        p.op_flags[6], // WriteLookupOutputToRD
        p.inst_flag_branch,
        p.next_is_noop,
        p.op_flags[7], // VirtualInstruction
    ];
    // Force-reload these polys from the provider: they may have stale
    // (partially bound) buffers left over from Stage 1.
    for &poly in &prod_open_extra {
        ops.push(Op::ReleaseDevice { poly });
    }
    let inst_first_active = params.stage2_max_rounds - params.product_remainder_rounds;
    for round in 0..params.product_remainder_rounds {
        let ch_idx = round_challenge_indices[inst_first_active + round];
        ops.push(Op::Bind {
            polys: prod_open_extra.clone(),
            challenge: ch_idx,
            order: BindingOrder::LowToHigh,
        });
    }

    // ---------------------------------------------------------------
    // 7. Flush 18 evaluation opening claims.
    //
    // Order matches jolt-core's cache_openings call order:
    //   RamRW (3): RamVal, RamRa, RamInc
    //   ProductRemainder (8): LeftInstructionInput, RightInstructionInput,
    //     OpFlags(Jump), OpFlags(WriteLookupOutputToRD), LookupOutput,
    //     InstructionFlags(Branch), NextIsNoop, OpFlags(VirtualInstruction)
    //   InstructionCR (5): LookupOutput, LeftLookupOperand,
    //     RightLookupOperand, LeftInstructionInput, RightInstructionInput
    //   RafEval (1): RamRafRa
    //   OutputCheck (1): RamValFinal
    // ---------------------------------------------------------------
    let stage2_eval_polys = vec![
        // RamReadWriteChecking (3)
        p.ram_val,
        p.ram_combined_ra,
        p.ram_inc,
        // ProductVirtualRemainder (8)
        p.left_instruction_input,
        p.right_instruction_input,
        p.op_flags[5], // Jump
        p.op_flags[6], // WriteLookupOutputToRD
        p.lookup_output,
        p.inst_flag_branch,
        p.next_is_noop,
        p.op_flags[7], // VirtualInstruction
        // InstructionLookupsClaimReduction (5)
        p.lookup_output,
        p.left_lookup_operand,
        p.right_lookup_operand,
        p.left_instruction_input,
        p.right_instruction_input,
        // RafEvaluation (1)
        p.ram_raf_ra,
        // OutputCheck (1)
        p.ram_val_final,
    ];

    for &poly in &stage2_eval_polys {
        ops.push(Op::Evaluate { poly });
    }
    ops.push(Op::RecordEvals {
        polys: stage2_eval_polys.clone(),
    });
    ops.push(Op::AbsorbEvals {
        polys: stage2_eval_polys,
        tag: DomainSeparator::OpeningClaim,
    });
}

/// Verifier schedule for Stage 2: Product + batched sumcheck.
///
/// Verifier stage with 5 batched instances:
///   [0] RamReadWriteChecking       — 45 rounds, degree 3
///   [1] ProductVirtualRemainder    — 25 rounds, degree 3
///   [2] InstructionClaimReduction  — 25 rounds, degree 2
///   [3] RafEvaluation              — 20 rounds, degree 2
///   [4] OutputCheck                — 20 rounds, degree 3
fn build_verifier_stage2_ops(
    p: &Polys,
    params: &ModuleParams,
    ch: &ChallengeTable,
) -> Vec<VerifierOp> {
    // Pre-squeeze challenges squeezed before the batched sumcheck:
    // τ_high (1) + γ_rw (1) + γ_instruction (1) + r_address (20) = 23
    // These are appended after the uniskip completes, in the order
    // they appear in the challenge table for stage 2.
    //
    // Find the challenge indices: they start after Stage 1's challenges.
    // Stage 1 used: num_tau + 1 (r0) + 1 (batch) + outer_remaining_rounds
    let s2_ch_base = params.num_tau + 1 + 1 + params.outer_remaining_rounds;

    // τ_high is at s2_ch_base, r0 at s2_ch_base+1, then γ_rw, γ_instruction, r_address...
    let ch_tau_high = s2_ch_base; // 55
    let ch_product_r0 = s2_ch_base + 1; // 56
    let ch_gamma_rw = s2_ch_base + 2; // 57
    let ch_gamma_instruction = s2_ch_base + 3; // 58
    let ch_r_address_base = s2_ch_base + 4; // 59..78

    // Evaluations: 18 openings (same order as prover flush)
    let stage2_eval_polys = [
        // RamReadWriteChecking (3)
        p.ram_val,
        p.ram_combined_ra,
        p.ram_inc,
        // ProductVirtualRemainder (8)
        p.left_instruction_input,
        p.right_instruction_input,
        p.op_flags[5], // Jump
        p.op_flags[6], // WriteLookupOutputToRD
        p.lookup_output,
        p.inst_flag_branch,
        p.next_is_noop,
        p.op_flags[7], // VirtualInstruction
        // InstructionLookupsClaimReduction (5)
        p.lookup_output,
        p.left_lookup_operand,
        p.right_lookup_operand,
        p.left_instruction_input,
        p.right_instruction_input,
        // RafEvaluation (1)
        p.ram_raf_ra,
        // OutputCheck (1)
        p.ram_val_final,
    ];
    let evaluations: Vec<_> = stage2_eval_polys
        .iter()
        .map(|&poly| Evaluation {
            poly,
            at_stage: VerifierStageIndex(1), // Stage 2 (0-indexed)
        })
        .collect();

    // ---------------------------------------------------------------
    // Stage 1 cycle challenge indices used for cross-stage eq evals.
    // Stage 1 outer remaining produces `outer_remaining_rounds` challenges
    // starting at index `num_tau + 2` (after τ, r0, and batch). After
    // Reverse normalization, the cycle point is these indices reversed.
    // ---------------------------------------------------------------
    let stage1_round_base = params.num_tau + 2; // τ + r0 + batch
    let stage1_cycle_challenges: Vec<usize> =
        (stage1_round_base..stage1_round_base + params.outer_remaining_rounds)
            .rev()
            .collect();

    // ---------------------------------------------------------------
    // Evaluation list positions (18 entries in flush order).
    // Used by StageEval in output_check formulas.
    // ---------------------------------------------------------------
    // sp.evals[0] = product_uniskip_eval (from RecordEvals before the batch).
    // Batched sumcheck evals start at index 1.
    const SE_BASE: usize = 1;
    // RamReadWriteChecking (3):
    const SE_RAM_VAL: usize = SE_BASE;
    const SE_RAM_RA: usize = SE_BASE + 1;
    const SE_RAM_INC: usize = SE_BASE + 2;
    // ProductVirtualRemainder (8):
    const SE_L_INST: usize = SE_BASE + 3;
    const SE_R_INST: usize = SE_BASE + 4;
    const SE_JUMP: usize = SE_BASE + 5;
    const _SE_WRITE_LO_TO_RD: usize = SE_BASE + 6; // opened but not used in output formula
    const SE_LOOKUP_OUT_PROD: usize = SE_BASE + 7;
    const SE_BRANCH: usize = SE_BASE + 8;
    const SE_NOOP: usize = SE_BASE + 9;
    const _SE_VIRTUAL_INST: usize = SE_BASE + 10; // opened but not used in output formula
    // InstructionClaimReduction (5):
    const SE_LOOKUP_OUT_INST: usize = SE_BASE + 11;
    const SE_LEFT_LOOKUP_OP: usize = SE_BASE + 12;
    const SE_RIGHT_LOOKUP_OP: usize = SE_BASE + 13;
    const SE_L_INST_CR: usize = SE_BASE + 14;
    const SE_R_INST_CR: usize = SE_BASE + 15;
    // RafEvaluation (1):
    const SE_RAM_RAF_RA: usize = SE_BASE + 16;
    // OutputCheck (1):
    const SE_VAL_FINAL: usize = SE_BASE + 17;

    // ---------------------------------------------------------------
    // Instance 0: RamReadWriteChecking
    //
    // input_claim = RamReadValue + γ_rw * RamWriteValue
    //   (evaluations from Stage 1 outer remaining, no ambiguity)
    //
    // output_check = eq(r_cycle_s1, r_cycle) * ra * (val + γ*(val + inc))
    //   Expanded: eq*ra*val + γ*eq*ra*val + γ*eq*ra*inc  (3 terms)
    //
    // normalize: Segments { sizes: [log_t, log_k_ram], output_order: [1, 0] }
    //   Raw 45 challenges: [phase1_cycle(25), phase2_addr(20)]
    //   After: [rev(addr)(20), rev(cycle)(25)] = [r_address, r_cycle]
    // ---------------------------------------------------------------
    let rw_eq = ClaimFactor::EqEvalSlice {
        challenges: stage1_cycle_challenges.clone(),
        at_stage: VerifierStageIndex(1),
        offset: params.log_k_ram, // cycle starts after address portion
    };
    let rw_input_claim = ClaimFormula {
        terms: vec![
            ClaimTerm {
                coeff: 1,
                factors: vec![ClaimFactor::Eval(p.ram_read_value)],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_gamma_rw),
                    ClaimFactor::Eval(p.ram_write_value),
                ],
            },
        ],
    };
    let rw_output_check = ClaimFormula {
        terms: vec![
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    rw_eq.clone(),
                    ClaimFactor::StageEval(SE_RAM_RA),
                    ClaimFactor::StageEval(SE_RAM_VAL),
                ],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_gamma_rw),
                    rw_eq.clone(),
                    ClaimFactor::StageEval(SE_RAM_RA),
                    ClaimFactor::StageEval(SE_RAM_VAL),
                ],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_gamma_rw),
                    rw_eq,
                    ClaimFactor::StageEval(SE_RAM_RA),
                    ClaimFactor::StageEval(SE_RAM_INC),
                ],
            },
        ],
    };

    // ---------------------------------------------------------------
    // Instance 1: ProductVirtualRemainder
    //
    // input_claim = product_uniskip_eval (= s2(r0))
    //
    // output_check = L(τ_high, r0) * eq(τ_low, r_rev) * fused_left * fused_right
    //   where:
    //     fused_left  = w[0]*l_inst + w[1]*lookup + w[2]*jump
    //     fused_right = w[0]*r_inst + w[1]*branch + w[2]*(1 - noop)
    //     w[k] = L_k(r0) over domain {0, 1, 2}
    //     τ_low = r_cycle from Stage 1 (25 elements)
    //     τ_high = product_tau_high challenge
    //     r0 = product_uniskip_r0 challenge
    //
    //   Expanding fused_left × fused_right with (1-noop) split: 12 terms.
    //
    // normalize: Reverse (25 rounds → big-endian cycle point)
    // ---------------------------------------------------------------
    let prod_lk = ClaimFactor::LagrangeKernelDomain {
        tau_challenge: ch_tau_high,
        at_challenge: ch_product_r0,
        domain_size: params.product_uniskip_domain,
    };
    let prod_eq = ClaimFactor::EqEval {
        challenges: stage1_cycle_challenges.clone(),
        at_stage: VerifierStageIndex(1),
    };
    let w = |k: usize| ClaimFactor::LagrangeWeight {
        challenge: ch_product_r0,
        domain_size: params.product_uniskip_domain,
        basis_index: k,
    };
    let prod_input_claim = ClaimFormula {
        terms: vec![ClaimTerm {
            coeff: 1,
            factors: vec![ClaimFactor::Eval(p.product_uniskip_eval)],
        }],
    };
    // 12-term expansion of L * eq * fused_left * fused_right
    let prod_output_check = ClaimFormula {
        terms: vec![
            // w[0]*l_inst × w[0]*r_inst
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    prod_lk.clone(),
                    prod_eq.clone(),
                    w(0),
                    w(0),
                    ClaimFactor::StageEval(SE_L_INST),
                    ClaimFactor::StageEval(SE_R_INST),
                ],
            },
            // w[0]*l_inst × w[1]*branch
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    prod_lk.clone(),
                    prod_eq.clone(),
                    w(0),
                    w(1),
                    ClaimFactor::StageEval(SE_L_INST),
                    ClaimFactor::StageEval(SE_BRANCH),
                ],
            },
            // w[0]*l_inst × w[2]*1
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    prod_lk.clone(),
                    prod_eq.clone(),
                    w(0),
                    w(2),
                    ClaimFactor::StageEval(SE_L_INST),
                ],
            },
            // -w[0]*l_inst × w[2]*noop
            ClaimTerm {
                coeff: -1,
                factors: vec![
                    prod_lk.clone(),
                    prod_eq.clone(),
                    w(0),
                    w(2),
                    ClaimFactor::StageEval(SE_L_INST),
                    ClaimFactor::StageEval(SE_NOOP),
                ],
            },
            // w[1]*lookup × w[0]*r_inst
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    prod_lk.clone(),
                    prod_eq.clone(),
                    w(1),
                    w(0),
                    ClaimFactor::StageEval(SE_LOOKUP_OUT_PROD),
                    ClaimFactor::StageEval(SE_R_INST),
                ],
            },
            // w[1]*lookup × w[1]*branch
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    prod_lk.clone(),
                    prod_eq.clone(),
                    w(1),
                    w(1),
                    ClaimFactor::StageEval(SE_LOOKUP_OUT_PROD),
                    ClaimFactor::StageEval(SE_BRANCH),
                ],
            },
            // w[1]*lookup × w[2]*1
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    prod_lk.clone(),
                    prod_eq.clone(),
                    w(1),
                    w(2),
                    ClaimFactor::StageEval(SE_LOOKUP_OUT_PROD),
                ],
            },
            // -w[1]*lookup × w[2]*noop
            ClaimTerm {
                coeff: -1,
                factors: vec![
                    prod_lk.clone(),
                    prod_eq.clone(),
                    w(1),
                    w(2),
                    ClaimFactor::StageEval(SE_LOOKUP_OUT_PROD),
                    ClaimFactor::StageEval(SE_NOOP),
                ],
            },
            // w[2]*jump × w[0]*r_inst
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    prod_lk.clone(),
                    prod_eq.clone(),
                    w(2),
                    w(0),
                    ClaimFactor::StageEval(SE_JUMP),
                    ClaimFactor::StageEval(SE_R_INST),
                ],
            },
            // w[2]*jump × w[1]*branch
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    prod_lk.clone(),
                    prod_eq.clone(),
                    w(2),
                    w(1),
                    ClaimFactor::StageEval(SE_JUMP),
                    ClaimFactor::StageEval(SE_BRANCH),
                ],
            },
            // w[2]*jump × w[2]*1
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    prod_lk.clone(),
                    prod_eq.clone(),
                    w(2),
                    w(2),
                    ClaimFactor::StageEval(SE_JUMP),
                ],
            },
            // -w[2]*jump × w[2]*noop
            ClaimTerm {
                coeff: -1,
                factors: vec![
                    prod_lk,
                    prod_eq,
                    w(2),
                    w(2),
                    ClaimFactor::StageEval(SE_JUMP),
                    ClaimFactor::StageEval(SE_NOOP),
                ],
            },
        ],
    };

    // ---------------------------------------------------------------
    // Instance 2: InstructionLookupsClaimReduction
    //
    // input_claim = LookupOutput + γ*LeftLookupOperand + γ²*RightLookupOperand
    //             + γ³*LeftInstructionInput + γ⁴*RightInstructionInput
    //   (evaluations from Stage 1, no ambiguity)
    //
    // output_check = eq(r_spartan, r) * (same weighted sum at InstructionCR point)
    //   where r_spartan = Stage 1 cycle point (25 elements)
    //
    // normalize: Reverse (25 rounds → big-endian)
    // ---------------------------------------------------------------
    let gamma_inst = ch_gamma_instruction;
    let inst_cr_input_claim = ClaimFormula {
        terms: vec![
            ClaimTerm {
                coeff: 1,
                factors: vec![ClaimFactor::Eval(p.lookup_output)],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(gamma_inst),
                    ClaimFactor::Eval(p.left_lookup_operand),
                ],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(gamma_inst),
                    ClaimFactor::Challenge(gamma_inst),
                    ClaimFactor::Eval(p.right_lookup_operand),
                ],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(gamma_inst),
                    ClaimFactor::Challenge(gamma_inst),
                    ClaimFactor::Challenge(gamma_inst),
                    ClaimFactor::Eval(p.left_instruction_input),
                ],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(gamma_inst),
                    ClaimFactor::Challenge(gamma_inst),
                    ClaimFactor::Challenge(gamma_inst),
                    ClaimFactor::Challenge(gamma_inst),
                    ClaimFactor::Eval(p.right_instruction_input),
                ],
            },
        ],
    };
    let inst_cr_eq = ClaimFactor::EqEval {
        challenges: stage1_cycle_challenges,
        at_stage: VerifierStageIndex(1),
    };
    let inst_cr_output_check = ClaimFormula {
        terms: vec![
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    inst_cr_eq.clone(),
                    ClaimFactor::StageEval(SE_LOOKUP_OUT_INST),
                ],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    inst_cr_eq.clone(),
                    ClaimFactor::Challenge(gamma_inst),
                    ClaimFactor::StageEval(SE_LEFT_LOOKUP_OP),
                ],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    inst_cr_eq.clone(),
                    ClaimFactor::Challenge(gamma_inst),
                    ClaimFactor::Challenge(gamma_inst),
                    ClaimFactor::StageEval(SE_RIGHT_LOOKUP_OP),
                ],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    inst_cr_eq.clone(),
                    ClaimFactor::Challenge(gamma_inst),
                    ClaimFactor::Challenge(gamma_inst),
                    ClaimFactor::Challenge(gamma_inst),
                    ClaimFactor::StageEval(SE_L_INST_CR),
                ],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    inst_cr_eq,
                    ClaimFactor::Challenge(gamma_inst),
                    ClaimFactor::Challenge(gamma_inst),
                    ClaimFactor::Challenge(gamma_inst),
                    ClaimFactor::Challenge(gamma_inst),
                    ClaimFactor::StageEval(SE_R_INST_CR),
                ],
            },
        ],
    };

    // ---------------------------------------------------------------
    // Instance 3: RafEvaluation
    //
    // input_claim = RamAddress * 2^(phase3_cycle_rounds)
    //   With default config: phase3_cycle_rounds = log_T - phase1_num_rounds = 0
    //   So input_claim = RamAddress (evaluation from Stage 1)
    //
    // output_check = unmap(r) * ra(r)
    //   where unmap is a preprocessed polynomial and ra is from Stage 2 evals
    //
    // normalize: Reverse (20 rounds → big-endian address point)
    // ---------------------------------------------------------------
    let raf_input_claim = ClaimFormula {
        terms: vec![ClaimTerm {
            coeff: 1,
            factors: vec![ClaimFactor::Eval(p.ram_address)],
        }],
    };
    let raf_output_check = ClaimFormula {
        terms: vec![ClaimTerm {
            coeff: 1,
            factors: vec![
                ClaimFactor::PreprocessedPolyEval {
                    poly: p.ram_unmap,
                    at_stage: VerifierStageIndex(1),
                },
                ClaimFactor::StageEval(SE_RAM_RAF_RA),
            ],
        }],
    };

    // ---------------------------------------------------------------
    // Instance 4: OutputCheck
    //
    // input_claim = 0 (zero-check sumcheck)
    //
    // output_check = eq(r_address, r') * io_mask(r') * (val_final(r') - val_io(r'))
    //   where:
    //     r_address = 20 challenges squeezed before the batched sumcheck
    //     r' = normalized sumcheck point (reversed 20 challenges)
    //
    //   Expanded: 2 terms (val_final positive, val_io negative)
    //
    // normalize: Reverse (20 rounds → big-endian address point)
    // ---------------------------------------------------------------
    let output_eq = ClaimFactor::EqEval {
        challenges: (ch_r_address_base..ch_r_address_base + params.log_k_ram).collect(),
        at_stage: VerifierStageIndex(1),
    };
    let output_check_input_claim = ClaimFormula::zero();
    let output_check_output = ClaimFormula {
        terms: vec![
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    output_eq.clone(),
                    ClaimFactor::PreprocessedPolyEval {
                        poly: p.io_mask,
                        at_stage: VerifierStageIndex(1),
                    },
                    ClaimFactor::StageEval(SE_VAL_FINAL),
                ],
            },
            ClaimTerm {
                coeff: -1,
                factors: vec![
                    output_eq,
                    ClaimFactor::PreprocessedPolyEval {
                        poly: p.io_mask,
                        at_stage: VerifierStageIndex(1),
                    },
                    ClaimFactor::PreprocessedPolyEval {
                        poly: p.val_io,
                        at_stage: VerifierStageIndex(1),
                    },
                ],
            },
        ],
    };

    // ---------------------------------------------------------------
    // Assemble all 5 instances
    // ---------------------------------------------------------------
    let instances = vec![
        // [0] RamReadWriteChecking — 45 rounds, degree 3
        SumcheckInstance {
            input_claim: rw_input_claim,
            output_check: rw_output_check,
            num_rounds: params.rw_checking_rounds,
            degree: params.rw_checking_degree,
            normalize: Some(PointNormalization::Segments {
                sizes: vec![params.log_t, params.log_k_ram],
                output_order: vec![1, 0],
            }),
        },
        // [1] ProductVirtualRemainder — 25 rounds, degree 3
        SumcheckInstance {
            input_claim: prod_input_claim,
            output_check: prod_output_check,
            num_rounds: params.product_remainder_rounds,
            degree: params.product_remainder_degree,
            normalize: Some(PointNormalization::Reverse),
        },
        // [2] InstructionLookupsClaimReduction — 25 rounds, degree 2
        SumcheckInstance {
            input_claim: inst_cr_input_claim,
            output_check: inst_cr_output_check,
            num_rounds: params.instruction_claim_reduction_rounds,
            degree: params.instruction_claim_reduction_degree,
            normalize: Some(PointNormalization::Reverse),
        },
        // [3] RafEvaluation — 20 rounds, degree 2
        SumcheckInstance {
            input_claim: raf_input_claim,
            output_check: raf_output_check,
            num_rounds: params.raf_evaluation_rounds,
            degree: params.raf_evaluation_degree,
            normalize: Some(PointNormalization::Reverse),
        },
        // [4] OutputCheck — 20 rounds, degree 3
        SumcheckInstance {
            input_claim: output_check_input_claim,
            output_check: output_check_output,
            num_rounds: params.output_check_rounds,
            degree: params.output_check_degree,
            normalize: Some(PointNormalization::Reverse),
        },
    ];

    let eval_polys: Vec<_> = evaluations.iter().map(|e| e.poly).collect();

    let mut ops = vec![
        VerifierOp::BeginStage,
        VerifierOp::Squeeze {
            challenge: ch_tau_high,
        },
    ];
    // Product uniskip: absorb round poly, squeeze r0, record+absorb eval.
    ops.push(VerifierOp::AbsorbRoundPoly {
        num_coeffs: params.product_uniskip_num_coeffs,
        tag: DomainSeparator::UniskipPoly,
    });
    ops.push(VerifierOp::Squeeze {
        challenge: ch_product_r0,
    });
    ops.push(VerifierOp::RecordEvals {
        evals: vec![Evaluation {
            poly: p.product_uniskip_eval,
            at_stage: VerifierStageIndex(1),
        }],
    });
    ops.push(VerifierOp::AbsorbEvals {
        polys: vec![p.product_uniskip_eval],
        tag: DomainSeparator::OpeningClaim,
    });
    // Remaining pre-squeeze: γ_rw, γ_instruction, r_address
    ops.push(VerifierOp::Squeeze {
        challenge: ch_gamma_rw,
    });
    ops.push(VerifierOp::Squeeze {
        challenge: ch_gamma_instruction,
    });
    for c in ch_r_address_base..ch_r_address_base + params.log_k_ram {
        ops.push(VerifierOp::Squeeze { challenge: c });
    }
    // Batch coefficient challenge indices for the 5 instances.
    let ch_batch_base = s2_ch_base + 2 + 1 + 1 + params.log_k_ram; // after τ_high, r0, γ_rw, γ_instruction, r_address
    let batch_challenges: Vec<usize> = (0..params.stage2_num_instances)
        .map(|i| ch_batch_base + i)
        .collect();
    ops.push(VerifierOp::VerifySumcheck {
        instances: instances.clone(),
        stage: 1,
        batch_challenges: batch_challenges.clone(),
        claim_tag: Some(DomainSeparator::SumcheckClaim),
    });
    ops.push(VerifierOp::RecordEvals { evals: evaluations });
    ops.push(VerifierOp::AbsorbEvals {
        polys: eval_polys,
        tag: DomainSeparator::OpeningClaim,
    });
    ops.push(VerifierOp::CheckOutput {
        instances,
        stage: 1,
        batch_challenges,
    });
    ops
}

// ---------------------------------------------------------------------------
// Stats printing
// ---------------------------------------------------------------------------

fn print_stats(module: &Module, params: &ModuleParams) {
    let polys = &module.polys;
    let committed = polys
        .iter()
        .filter(|p| matches!(p.kind, PolyKind::Committed))
        .count();
    let virtual_count = polys
        .iter()
        .filter(|p| matches!(p.kind, PolyKind::Virtual))
        .count();
    let public = polys
        .iter()
        .filter(|p| matches!(p.kind, PolyKind::Public(_)))
        .count();

    let log_t = params.log_t;
    let T = 1usize << log_t;

    eprintln!("=== Jolt-Core Ground Truth Module ===");
    eprintln!(
        "  polys: {} ({committed} committed, {virtual_count} virtual, {public} public)",
        polys.len()
    );
    eprintln!("  kernels: {}", module.prover.kernels.len());
    eprintln!("  ops: {}", module.prover.ops.len());
    eprintln!("  challenges: {}", module.challenges.len());
    eprintln!("  verifier ops: {}", module.verifier.ops.len());
    eprintln!("  verifier stages: {}", module.verifier.num_stages);

    eprintln!("\n=== Configuration ===");
    eprintln!("  log_t = {log_t}, T = {T}");
    eprintln!(
        "  log_k_chunk = {}, k_chunk = {}",
        params.log_k_chunk, params.k_chunk
    );
    eprintln!("  instruction_d = {}", params.instruction_d);
    eprintln!("  bytecode_d = {}", params.bytecode_d);
    eprintln!("  ram_d = {}", params.ram_d);
    eprintln!("  LOG_K_REG = {LOG_K_REG}");
    eprintln!("  Total committed (main witness): {}", params.num_committed);

    // Commitment emission order
    eprintln!("\n=== Commitment Phase (3 barriers) ===");
    let mut barrier = 0;
    for op in &module.prover.ops {
        if let Op::Commit {
            polys: committed,
            tag,
            ..
        } = op
        {
            barrier += 1;
            eprintln!("  barrier {barrier} ({:?}): {} polys", tag, committed.len());
            for id in committed {
                eprintln!("    {:?}", id);
            }
        }
    }

    // Op-type counts
    let mut counts = [0usize; 16];
    for op in &module.prover.ops {
        match op {
            Op::SumcheckRound { .. } => counts[0] += 1,
            Op::BatchedSumcheckRound { .. } => counts[14] += 1,
            Op::Evaluate { .. } => counts[1] += 1,
            Op::Bind { .. } => counts[2] += 1,
            Op::LagrangeProject { .. } => counts[13] += 1,
            Op::Commit { .. } | Op::CommitStreaming { .. } => counts[3] += 1,
            Op::ReduceOpenings => counts[4] += 1,
            Op::Open => counts[5] += 1,
            Op::Preamble => counts[6] += 1,
            Op::BeginStage { .. } => counts[12] += 1,
            Op::AbsorbRoundPoly { .. } => counts[7] += 1,
            Op::RecordEvals { .. } | Op::AbsorbEvals { .. } => counts[8] += 1,
            Op::AbsorbInputClaim { .. } => counts[15] += 1,
            Op::Squeeze { .. } => counts[9] += 1,
            Op::CollectOpeningClaim { .. } => counts[10] += 1,
            Op::ReleaseDevice { .. } => counts[11] += 1,
            Op::ReleaseHost { .. } => counts[11] += 1,
        }
    }
    eprintln!("\n=== Op Counts ===");
    eprintln!("  -- Compute --");
    eprintln!("  SumcheckRound:         {}", counts[0]);
    eprintln!("  BatchedSumcheckRound:  {}", counts[14]);
    eprintln!("  Evaluate:              {}", counts[1]);
    eprintln!("  Bind:                  {}", counts[2]);
    eprintln!("  LagrangeProject:       {}", counts[13]);
    eprintln!("  -- PCS --");
    eprintln!("  Commit:                {}", counts[3]);
    eprintln!("  ReduceOpenings:        {}", counts[4]);
    eprintln!("  Open:                  {}", counts[5]);
    eprintln!("  -- Orchestration --");
    eprintln!("  Preamble:              {}", counts[6]);
    eprintln!("  BeginStage:            {}", counts[12]);
    eprintln!("  AbsorbRoundPoly:       {}", counts[7]);
    eprintln!("  AbsorbEvals:           {}", counts[8]);
    eprintln!("  AbsorbInputClaim:      {}", counts[15]);
    eprintln!("  Squeeze:               {}", counts[9]);
    eprintln!("  CollectOpeningClaim:   {}", counts[10]);
    eprintln!("  Release (dev+host):    {}", counts[11]);

    // Kernel details
    if !module.prover.kernels.is_empty() {
        eprintln!("\n=== Kernels ===");
        for (i, k) in module.prover.kernels.iter().enumerate() {
            eprintln!(
                "  [{i}] rounds={}, num_evals={}, formula_terms={}, inputs={}, binding={:?}",
                k.num_rounds,
                k.spec.num_evals,
                k.spec.formula.terms.len(),
                k.inputs.len(),
                k.spec.binding_order,
            );
        }
    }

    // Batched sumchecks
    if !module.prover.batched_sumchecks.is_empty() {
        eprintln!("\n=== Batched Sumchecks ===");
        for (i, bdef) in module.prover.batched_sumchecks.iter().enumerate() {
            eprintln!(
                "  [{i}] max_rounds={}, max_degree={}, instances={}",
                bdef.max_rounds, bdef.max_degree, bdef.instances.len()
            );
            for (j, inst) in bdef.instances.iter().enumerate() {
                let phase_descs: Vec<String> = inst
                    .phases
                    .iter()
                    .map(|ph| {
                        let seg = if ph.segmented.is_some() {
                            " [segmented]"
                        } else {
                            ""
                        };
                        format!("k{}×{}r{}", ph.kernel, ph.num_rounds, seg)
                    })
                    .collect();
                eprintln!(
                    "    [{j}] phases=[{}], batch_coeff={}, first_active={}, total_rounds={}",
                    phase_descs.join(", "),
                    inst.batch_coeff,
                    inst.first_active_round,
                    inst.num_rounds(),
                );
            }
        }
    }

    // Verifier schedule
    if !module.verifier.ops.is_empty() {
        eprintln!("\n=== Verifier Schedule ===");
        eprintln!(
            "  ops={}, stages={}, num_challenges={}, num_polys={}",
            module.verifier.ops.len(),
            module.verifier.num_stages,
            module.verifier.num_challenges,
            module.verifier.num_polys,
        );
    }

    // Challenge summary
    if !module.challenges.is_empty() {
        eprintln!("\n=== Challenges ({}) ===", module.challenges.len());
        let mut fs = 0;
        let mut sc = 0;
        let mut pw = 0;
        let mut ext = 0;
        for c in &module.challenges {
            match c.source {
                ChallengeSource::FiatShamir { .. } => fs += 1,
                ChallengeSource::SumcheckRound { .. } => sc += 1,
                ChallengeSource::Power { .. } => pw += 1,
                ChallengeSource::External => ext += 1,
            }
        }
        eprintln!("  FiatShamir: {fs}, SumcheckRound: {sc}, Power: {pw}, External: {ext}");
    }
}
