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

use jolt_compiler::checkpoint_lowering::lower_checkpoint_rule;
use jolt_compiler::formula::{BindingOrder, Factor, Formula, ProductTerm};
use jolt_compiler::ir::PolyKind;
use jolt_compiler::kernel_spec::{GruenHint, Iteration, LinComboQ};
use jolt_compiler::module::{
    BatchIdx, BatchedInstance, BatchedSumcheckDef, BilinearExpr, ChallengeDecl, ChallengeIdx,
    ChallengeSource, CheckpointAction, CheckpointEvalAction, CheckpointRule, ClaimFactor,
    ClaimFormula, ClaimTerm, CombineEntry, Comparison, DefaultVal, DomainSeparator, EvalMode,
    Evaluation, InputBinding, InstanceConfig, InstanceIdx, InstancePhase, IntBitOp, KernelDef,
    Module, Op, PointNormalization, PolyDecl, PrefixMleFormula, PrefixMleRule, R1CSMatrix,
    RemainingTest, RoundGuard, RoundPolyEncoding, ScalarCapture, Schedule, SegmentedConfig,
    SuffixOp, SumcheckInstance, VerifierOp, VerifierSchedule, VerifierStageIndex, WeightFn,
};
use jolt_compiler::params::{
    ModuleParams, LOG_K_FR, LOG_K_INSTRUCTION, LOG_K_REG, NUM_CIRCUIT_FLAGS, NUM_LOOKUP_TABLES,
    NUM_R1CS_INPUTS,
};
use jolt_compiler::KernelSpec;
use jolt_compiler::PolynomialId;
use jolt_instructions::tables::prefixes::{Prefixes, ALL_PREFIXES, NUM_PREFIXES};
use jolt_instructions::tables::suffixes::Suffixes;
use jolt_instructions::{LookupBits, LookupTableKind, LookupTables};

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

// Polynomial index table.
//
// Every polynomial referenced by the Module is registered here with a
// stable index. The ordering within committed polys matches jolt-core's
// `all_committed_polynomials()` iteration order, which determines the
// Fiat-Shamir transcript commitment sequence.

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
            committed_num_vars: None,
        });
        id
    }

    fn add_committed(
        &mut self,
        id: PolynomialId,
        name: &str,
        kind: PolyKind,
        num_vars: usize,
        committed_num_vars: usize,
    ) -> PolynomialId {
        self.polys.push(PolyDecl {
            name: name.to_string(),
            kind,
            num_elements: 1 << num_vars,
            committed_num_vars: Some(committed_num_vars),
        });
        id
    }

    fn into_vec(self) -> Vec<PolyDecl> {
        self.polys
    }
}

/// All polynomial identifiers, grouped by commitment / virtual / public.
struct Polys {
    // Committed (transcript order matches jolt-core)
    rd_inc: PolynomialId,
    ram_inc: PolynomialId,
    instruction_ra: Vec<PolynomialId>, // [0..p.instruction_d)
    ram_ra: Vec<PolynomialId>,         // [0..p.ram_d)
    bytecode_ra: Vec<PolynomialId>,    // [0..p.bytecode_d)
    untrusted_advice: PolynomialId,
    trusted_advice: PolynomialId,

    // Virtual — Spartan internal
    az: PolynomialId,
    bz: PolynomialId,
    spartan_eq: PolynomialId,
    product_left: PolynomialId,
    product_right: PolynomialId,

    // Virtual — trace-derived (R1CS inputs, 35 entries in ALL_R1CS_INPUTS order)
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

    // Virtual — BN254 Fr coprocessor operand columns (R1CS slots 45/46/47).
    // Phase 4 FR Twist will bind these via γ-batched Val·Ra openings.
    field_rs1_value: PolynomialId,
    field_rs2_value: PolynomialId,
    field_rd_value: PolynomialId,

    // Virtual — non-R1CS-input trace values
    next_is_noop: PolynomialId,
    rd: PolynomialId,

    // Virtual — InstructionFlags (6 total)
    inst_flag_left_is_pc: PolynomialId,
    inst_flag_right_is_imm: PolynomialId,
    inst_flag_left_is_rs1: PolynomialId,
    inst_flag_right_is_rs2: PolynomialId,
    inst_flag_branch: PolynomialId,
    inst_flag_is_noop: PolynomialId,

    // Virtual — registers
    reg_wa: PolynomialId,
    reg_ra_rs1: PolynomialId,
    reg_ra_rs2: PolynomialId,
    reg_val: PolynomialId,

    // Committed — BN254 Fr coprocessor register subsystem
    field_reg_inc: PolynomialId,
    field_reg_ra: Vec<PolynomialId>, // d ∈ 0..p.field_reg_d (typically 1)

    // Virtual — BN254 Fr coprocessor derived polys
    field_reg_wa: PolynomialId,
    field_reg_ra_rs1: PolynomialId,
    field_reg_ra_rs2: PolynomialId,
    field_reg_val: PolynomialId,

    // Virtual — RAM
    ram_combined_ra: PolynomialId,
    ram_val: PolynomialId,
    ram_val_final: PolynomialId,
    ram_wa: PolynomialId,
    hamming_weight: PolynomialId,

    // Virtual — RamRW eq tables (segmented phase 1/2)
    ram_eq_cycle: PolynomialId,
    ram_eq_addr: PolynomialId,

    // Virtual — RAF
    ram_ra_indicator: PolynomialId,
    ram_raf_ra: PolynomialId,
    inst_raf_ra: PolynomialId,
    bytecode_raf_ra: PolynomialId,
    inst_raf_flag: PolynomialId,
    lookup_table_flags: Vec<PolynomialId>, // [0..NUM_LOOKUP_TABLES)

    // Virtual — bytecode read values
    bc_read_val: Vec<PolynomialId>, // [0..5)

    // Virtual — HW reduction pushforward
    hw_g: Vec<PolynomialId>, // [0..total_d)

    // Virtual — Spartan uniskip evaluations
    outer_uniskip_eval: PolynomialId,
    product_uniskip_eval: PolynomialId,

    // Virtual — advice address phase
    trusted_advice_addr: PolynomialId,
    untrusted_advice_addr: PolynomialId,

    // Public — preprocessed
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
        Az, BranchFlag, BytecodeRa, BytecodeReadRafVal, Bz, ExpandedPc, FieldRdValue,
        FieldRegInc, FieldRegRa, FieldRegRaRs1, FieldRegRaRs2, FieldRegVal, FieldRegWa,
        FieldRs1Value, FieldRs2Value, HammingG, HammingWeight, Imm, InstructionRa,
        InstructionRafFlag, IoMask, LeftInstructionInput, LeftIsPc, LeftIsRs1, LeftLookupOperand,
        LookupOutput, LookupTableFlag, NextIsFirstInSequence, NextIsNoop, NextIsVirtual, NextPc,
        NextUnexpandedPc, NoopFlag, OpFlag, OuterUniskipEval, Product, ProductLeft, ProductRight,
        ProductUniskipEval, RamAddress, RamCombinedRa, RamInc, RamRa, RamRafRa, RamReadValue,
        RamVal, RamValFinal, RamWriteValue, Rd, RdInc, RdWa, RdWriteValue, RegistersVal,
        RightInstructionInput, RightIsImm, RightIsRs2, RightLookupOperand, Rs1Ra, Rs1Value, Rs2Ra,
        Rs2Value, ShouldBranch, ShouldJump, SpartanEq, TrustedAdvice, UnexpandedPc,
        UntrustedAdvice, ValIo,
    };

    // Committed (jolt-core transcript order)
    let rd_inc = pt.add_committed(RdInc, "RdInc", Committed, p.log_t, p.log_k_chunk + p.log_t);
    let ram_inc = pt.add_committed(
        RamInc,
        "RamInc",
        Committed,
        p.log_t,
        p.log_k_chunk + p.log_t,
    );
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

    // Virtual — Spartan internal
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

    // Virtual — R1CS inputs (35 entries, ALL_R1CS_INPUTS order)
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
    let op_flags: Vec<_> = (0..NUM_CIRCUIT_FLAGS)
        .map(|i| pt.add(OpFlag(i), &format!("OpFlag_{i}"), Virtual, p.log_t))
        .collect();

    // BN254 Fr virtual operand columns. These live at R1CS slots 45/46/47;
    // their per-cycle values will be bound by the FR Twist at Phase 4, but
    // the polynomial IDs must exist at Stage-1 for Spartan Az/Bz/Cz matrix
    // construction to reference the FR R1CS rows (rv64 rows 19-31).
    let field_rs1_value = pt.add(FieldRs1Value, "FieldRs1Value", Virtual, p.log_t);
    let field_rs2_value = pt.add(FieldRs2Value, "FieldRs2Value", Virtual, p.log_t);
    let field_rd_value = pt.add(FieldRdValue, "FieldRdValue", Virtual, p.log_t);

    // Virtual — non-R1CS-input trace values
    let next_is_noop = pt.add(NextIsNoop, "NextIsNoop", Virtual, p.log_t);
    let rd = pt.add(Rd, "Rd", Virtual, p.log_t);

    // Virtual — InstructionFlags (6 variants)
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

    // Virtual — registers
    let reg_wa = pt.add(RdWa, "RdWA", Virtual, LOG_K_REG + p.log_t);
    let reg_ra_rs1 = pt.add(Rs1Ra, "RegRaRs1", Virtual, LOG_K_REG + p.log_t);
    let reg_ra_rs2 = pt.add(Rs2Ra, "RegRaRs2", Virtual, LOG_K_REG + p.log_t);
    let reg_val = pt.add(RegistersVal, "RegVal", Virtual, LOG_K_REG + p.log_t);

    // Committed — BN254 Fr coprocessor register-file (mirrors integer Rd*).
    // FieldRegInc: committed dense, padded from log_t to log_k_chunk + log_t
    // on the main grid (identical to RdInc shape). FieldRegRa(d): committed
    // one-hot chunks, d ∈ 0..p.field_reg_d (p.field_reg_d = 1 when
    // log_k_chunk ≥ LOG_K_FR = 4, per the mirror plan).
    let field_reg_inc = pt.add_committed(
        FieldRegInc,
        "FieldRegInc",
        Committed,
        p.log_t,
        p.log_k_chunk + p.log_t,
    );
    let field_reg_ra: Vec<_> = (0..p.field_reg_d)
        .map(|d| {
            pt.add(
                FieldRegRa(d),
                &format!("FieldRegRa_{d}"),
                Committed,
                p.log_k_chunk + p.log_t,
            )
        })
        .collect();

    // Virtual — BN254 Fr coprocessor derived polys. Identical shape to
    // integer `reg_{wa,ra_rs1,ra_rs2,val}`, with LOG_K_REG → LOG_K_FR.
    let field_reg_wa = pt.add(FieldRegWa, "FieldRegWa", Virtual, LOG_K_FR + p.log_t);
    let field_reg_ra_rs1 = pt.add(
        FieldRegRaRs1,
        "FieldRegRaRs1",
        Virtual,
        LOG_K_FR + p.log_t,
    );
    let field_reg_ra_rs2 = pt.add(
        FieldRegRaRs2,
        "FieldRegRaRs2",
        Virtual,
        LOG_K_FR + p.log_t,
    );
    let field_reg_val = pt.add(FieldRegVal, "FieldRegVal", Virtual, LOG_K_FR + p.log_t);

    // Virtual — RAM
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

    // Virtual — RamRW segmented eq tables
    let ram_eq_cycle = pt.add(PolynomialId::RamEqCycle, "RamEqCycle", Virtual, p.log_t);
    let ram_eq_addr = pt.add(PolynomialId::RamEqAddr, "RamEqAddr", Virtual, p.log_k_ram);

    // Virtual — RAF
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

    // Virtual — bytecode read values
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

    // Virtual — HW reduction pushforward
    let total_d = p.instruction_d + p.bytecode_d + p.ram_d;
    let hw_g: Vec<_> = (0..total_d)
        .map(|i| pt.add(HammingG(i), &format!("G_{i}"), Virtual, p.log_k_chunk))
        .collect();

    // Virtual — Spartan uniskip evaluations
    let outer_uniskip_eval = pt.add(OuterUniskipEval, "OuterUniskipEval", Virtual, 0);
    let product_uniskip_eval = pt.add(ProductUniskipEval, "ProductUniskipEval", Virtual, 0);

    // Virtual — advice address phase
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

    // Public — preprocessed
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
        field_rs1_value,
        field_rs2_value,
        field_rd_value,
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
        field_reg_inc,
        field_reg_ra,
        field_reg_wa,
        field_reg_ra_rs1,
        field_reg_ra_rs2,
        field_reg_val,
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

// Module construction — one function per phase/stage.

// Challenge table helper.

struct ChallengeTable {
    decls: Vec<ChallengeDecl>,
}

impl ChallengeTable {
    fn new() -> Self {
        Self { decls: Vec::new() }
    }

    fn add(&mut self, name: &str, source: ChallengeSource) -> ChallengeIdx {
        let idx = ChallengeIdx(self.decls.len());
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

/// Emit the unrolled granular ops for a batched sumcheck.
///
/// Emits unrolled per-instance, per-round granular ops for a batched sumcheck.
/// Build the per-slot update list for an `Op::CheckpointEvalBatch` at the
/// given `round` of the sumcheck: lowers each checkpoint rule with the current
/// `(r_x, r_y)` challenges and filters out passthrough slots.
fn build_checkpoint_batch(
    ic: &InstanceConfig,
    r_x: ChallengeIdx,
    r_y: ChallengeIdx,
    round: usize,
) -> Vec<(usize, CheckpointEvalAction)> {
    let suffix_len = ic.total_address_bits - (round / ic.chunk_bits + 1) * ic.chunk_bits;
    ic.checkpoint_rules
        .iter()
        .enumerate()
        .filter_map(|(i, rule)| {
            lower_checkpoint_rule(rule, i, r_x, r_y, round, suffix_len).map(|a| (i, a))
        })
        .collect()
}

fn emit_scatter_ops(ops: &mut Vec<Op>, kernel: usize, phase: usize, chunk_bits: usize) {
    ops.push(Op::SuffixScatter { kernel, phase });
    ops.push(Op::QBufferScatter { kernel, phase });
    ops.push(Op::MaterializePBuffers { kernel });
    ops.push(Op::InitExpandingTable {
        table: PolynomialId::ExpandingTable(phase),
        size: 1 << chunk_bits,
    });
}

/// Reads the BatchedSumcheckDef and kernel definitions to emit per-instance,
/// per-round ops that the runtime can execute as a flat dispatch loop.
#[allow(clippy::too_many_arguments)]
fn emit_unrolled_batched_rounds(
    ops: &mut Vec<Op>,
    kernels: &[KernelDef],
    batched_sumchecks: &[BatchedSumcheckDef],
    ch: &mut ChallengeTable,
    batch_idx: BatchIdx,
    max_rounds: usize,
    num_coeffs_fn: impl Fn(usize) -> usize,
    stage: VerifierStageIndex,
    challenge_prefix: &str,
    round_offset: usize,
    pre_allocated_challenges: Option<&[ChallengeIdx]>,
) -> Vec<ChallengeIdx> {
    let bdef = &batched_sumchecks[batch_idx.0];
    let max_evals = bdef.max_degree + 1;
    let mut indices = Vec::with_capacity(max_rounds);

    for round in 0..max_rounds {
        let bind = if round > 0 {
            Some(indices[round - 1])
        } else {
            None
        };

        ops.push(Op::BatchRoundBegin {
            batch: batch_idx,
            round,
            max_evals,
            bind_challenge: bind,
        });

        for (inst_idx_raw, inst) in bdef.instances.iter().enumerate() {
            let inst_idx = InstanceIdx(inst_idx_raw);
            if round < inst.first_active_round {
                ops.push(Op::BatchInactiveContribution {
                    batch: batch_idx,
                    instance: inst_idx,
                });
                continue;
            }

            let instance_round = round - inst.first_active_round;
            let (phase_idx, phase_start) = inst.phase_for_round(instance_round);
            let phase = &inst.phases[phase_idx];
            let kernel = phase.kernel;
            let kdef = &kernels[kernel];
            let is_ps = kdef.instance_config.is_some();

            if is_ps {
                // Address-decomposition instance: stateless ops.
                let ic = kdef.instance_config.as_ref().unwrap();
                let chunk_bits = ic.chunk_bits;
                let sub_phase = instance_round / chunk_bits;
                let round_in_sub = instance_round % chunk_bits;

                if instance_round == 0 {
                    ops.push(Op::InitInstanceWeights {
                        r_reduction: ic.r_reduction.clone(),
                        num_prefixes: ic.num_prefixes,
                    });
                    emit_scatter_ops(ops, kernel, 0, chunk_bits);
                } else if round_in_sub == 0 {
                    let ch = bind.unwrap();
                    ops.push(Op::Bind {
                        polys: ic.bindable_polys(),
                        challenge: ch,
                        order: BindingOrder::HighToLow,
                    });
                    ops.push(Op::ExpandingTableUpdate {
                        table: PolynomialId::ExpandingTable(sub_phase - 1),
                        challenge: ch,
                        current_len: 1 << (chunk_bits - 1),
                    });
                    if instance_round % 2 == 0 {
                        let updates = build_checkpoint_batch(
                            ic,
                            indices[round - 2],
                            indices[round - 1],
                            instance_round - 1,
                        );
                        if !updates.is_empty() {
                            ops.push(Op::CheckpointEvalBatch { updates });
                        }
                    }
                    ops.push(Op::CaptureScalar {
                        poly: PolynomialId::InstanceP(1, 0),
                        challenge: ic.registry_checkpoint_slots[0],
                    });
                    ops.push(Op::CaptureScalar {
                        poly: PolynomialId::InstanceP(0, 0),
                        challenge: ic.registry_checkpoint_slots[1],
                    });
                    ops.push(Op::CaptureScalar {
                        poly: PolynomialId::InstanceP(2, 0),
                        challenge: ic.registry_checkpoint_slots[2],
                    });
                    ops.push(Op::UpdateInstanceWeights {
                        expanding_table: PolynomialId::ExpandingTable(sub_phase - 1),
                        chunk_bits,
                        num_phases: ic.num_phases,
                        phase: sub_phase,
                    });
                    emit_scatter_ops(ops, kernel, sub_phase, chunk_bits);
                } else {
                    let ch = bind.unwrap();
                    ops.push(Op::Bind {
                        polys: ic.bindable_polys(),
                        challenge: ch,
                        order: BindingOrder::HighToLow,
                    });
                    ops.push(Op::ExpandingTableUpdate {
                        table: PolynomialId::ExpandingTable(sub_phase),
                        challenge: ch,
                        current_len: 1 << (round_in_sub - 1),
                    });
                    if instance_round >= 2 && instance_round % 2 == 0 {
                        let updates = build_checkpoint_batch(
                            ic,
                            indices[round - 2],
                            indices[round - 1],
                            instance_round - 1,
                        );
                        if !updates.is_empty() {
                            ops.push(Op::CheckpointEvalBatch { updates });
                        }
                    }
                }

                let r_x_challenge = if instance_round % 2 == 1 {
                    Some(indices[round - 1])
                } else {
                    None
                };
                ops.push(Op::ReadCheckingReduce {
                    kernel,
                    round: instance_round,
                    r_x_challenge,
                });
                ops.push(Op::RafReduce {
                    batch: batch_idx,
                    instance: inst_idx,
                    kernel,
                });
            } else if instance_round == 0 || instance_round == phase_start {
                // Phase boundary (non-decomp instance).
                if instance_round > 0 {
                    let prev_phase = &inst.phases[phase_idx - 1];
                    let prev_kernel = prev_phase.kernel;
                    let prev_kdef = &kernels[prev_kernel];
                    let prev_is_decomp =
                        matches!(prev_kdef.instance_config, Some(InstanceConfig { .. }));

                    if prev_is_decomp {
                        let ch = bind.unwrap();
                        let ic = prev_kdef.instance_config.as_ref().unwrap();
                        let chunk_bits = ic.chunk_bits;
                        let last_sub_phase = ic.num_phases - 1;
                        ops.push(Op::Bind {
                            polys: ic.bindable_polys(),
                            challenge: ch,
                            order: BindingOrder::HighToLow,
                        });
                        ops.push(Op::ExpandingTableUpdate {
                            table: PolynomialId::ExpandingTable(last_sub_phase),
                            challenge: ch,
                            current_len: 1 << (chunk_bits - 1),
                        });
                        let prev_instance_round = instance_round - 1;
                        if prev_instance_round >= 1 && (prev_instance_round + 1) % 2 == 0 {
                            let updates = build_checkpoint_batch(
                                ic,
                                indices[round - 2],
                                indices[round - 1],
                                prev_instance_round,
                            );
                            if !updates.is_empty() {
                                ops.push(Op::CheckpointEvalBatch { updates });
                            }
                        }
                        ops.push(Op::CaptureScalar {
                            poly: PolynomialId::InstanceP(1, 0),
                            challenge: ic.registry_checkpoint_slots[0],
                        });
                        ops.push(Op::CaptureScalar {
                            poly: PolynomialId::InstanceP(0, 0),
                            challenge: ic.registry_checkpoint_slots[1],
                        });
                        ops.push(Op::CaptureScalar {
                            poly: PolynomialId::InstanceP(2, 0),
                            challenge: ic.registry_checkpoint_slots[2],
                        });
                        ops.push(Op::MaterializeRA {
                            kernel: prev_kernel,
                        });
                        ops.push(Op::MaterializeCombinedVal {
                            kernel: prev_kernel,
                        });
                    } else if let Some(ch_val) = bind {
                        ops.push(Op::InstanceBindPreviousPhase {
                            batch: batch_idx,
                            instance: inst_idx,
                            kernel: prev_kernel,
                            challenge: ch_val,
                        });
                        let prev_carry_polys: Vec<_> =
                            prev_phase.carry_bindings.iter().map(|b| b.poly()).collect();
                        if !prev_carry_polys.is_empty() {
                            ops.push(Op::BindCarryBuffers {
                                polys: prev_carry_polys,
                                challenge: ch_val,
                                order: prev_kdef.spec.binding_order,
                            });
                        }
                    }
                }

                for cap in &phase.scalar_captures {
                    ops.push(Op::CaptureScalar {
                        poly: cap.poly,
                        challenge: cap.challenge,
                    });
                }

                if let Some(seg) = &phase.segmented {
                    ops.push(Op::MaterializeSegmentedOuterEq {
                        batch: batch_idx,
                        instance: inst_idx,
                        segmented: seg.clone(),
                    });
                }
                let is_activation = instance_round == 0;
                if is_activation {
                    for pre_op in &phase.pre_activation_ops {
                        ops.push(pre_op.clone());
                    }
                    for binding in &phase.carry_bindings {
                        ops.push(Op::Materialize {
                            binding: binding.clone(),
                        });
                    }
                    let expected_size = 1usize << kdef.num_rounds;
                    for binding in &kdef.inputs {
                        match binding {
                            InputBinding::Provided { .. } => {
                                ops.push(Op::MaterializeUnlessFresh {
                                    binding: binding.clone(),
                                    expected_size,
                                });
                            }
                            _ => {
                                ops.push(Op::Materialize {
                                    binding: binding.clone(),
                                });
                            }
                        }
                    }
                } else {
                    for binding in &kdef.inputs {
                        ops.push(Op::MaterializeIfAbsent {
                            binding: binding.clone(),
                        });
                    }
                }
            } else {
                // Mid-phase: bind (non-decomp instance).
                if let Some(ch_val) = bind {
                    ops.push(Op::InstanceBind {
                        batch: batch_idx,
                        instance: inst_idx,
                        kernel,
                        challenge: ch_val,
                    });
                    let carry_polys: Vec<_> =
                        phase.carry_bindings.iter().map(|b| b.poly()).collect();
                    if !carry_polys.is_empty() {
                        ops.push(Op::BindCarryBuffers {
                            polys: carry_polys,
                            challenge: ch_val,
                            order: kdef.spec.binding_order,
                        });
                    }
                }
            }

            // Reduce (non-decomp cases — decomp reduce is emitted above).
            if !is_ps {
                if phase.segmented.is_some() {
                    ops.push(Op::InstanceSegmentedReduce {
                        batch: batch_idx,
                        instance: inst_idx,
                        kernel,
                        round_within_phase: instance_round - phase_start,
                        segmented: phase.segmented.clone().unwrap(),
                    });
                } else {
                    ops.push(Op::InstanceReduce {
                        batch: batch_idx,
                        instance: inst_idx,
                        kernel,
                    });
                }
            }

            ops.push(Op::BatchAccumulateInstance {
                batch: batch_idx,
                instance: inst_idx,
                max_evals,
                num_evals: kdef.spec.num_evals,
            });
        }

        ops.push(Op::BatchRoundFinalize { batch: batch_idx });

        let ch_r = if let Some(pre) = pre_allocated_challenges {
            pre[round]
        } else {
            ch.add(
                &format!("{challenge_prefix}_{round}"),
                ChallengeSource::SumcheckRound {
                    stage,
                    round: round_offset + round,
                },
            )
        };
        ops.push(Op::AbsorbRoundPoly {
            num_coeffs: num_coeffs_fn(round),
            tag: DomainSeparator::SumcheckPoly,
            encoding: RoundPolyEncoding::Compressed,
        });
        ops.push(Op::Squeeze { challenge: ch_r });
        indices.push(ch_r);
    }
    indices
}

// R1CS input polynomial indices (35 entries, matching jolt-core's
// `ALL_R1CS_INPUTS` ordering — determines transcript flush order).

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
        // BN254 Fr coprocessor circuit flags (jolt-r1cs slots 36..=44).
        p.op_flags[14],            // 35: OpFlags(IsFieldMul)
        p.op_flags[15],            // 36: OpFlags(IsFieldAdd)
        p.op_flags[16],            // 37: OpFlags(IsFieldSub)
        p.op_flags[17],            // 38: OpFlags(IsFieldInv)
        p.op_flags[18],            // 39: OpFlags(IsFieldAssertEq)
        p.op_flags[19],            // 40: OpFlags(IsFieldMov)
        p.op_flags[20],            // 41: OpFlags(IsFieldSLL64)
        p.op_flags[21],            // 42: OpFlags(IsFieldSLL128)
        p.op_flags[22],            // 43: OpFlags(IsFieldSLL192)
        // BN254 Fr virtual operand columns (jolt-r1cs slots 45..=47).
        p.field_rs1_value,         // 44: FieldRs1Value
        p.field_rs2_value,         // 45: FieldRs2Value
        p.field_rd_value,          // 46: FieldRdValue
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
    let s2 = build_stage2(
        &p,
        params,
        &mut ops,
        &mut kernels,
        &mut ch,
        &mut batched_sumchecks,
    );

    // Stage 3: Shift + InstructionInput + RegistersClaimReduction
    ops.push(Op::BeginStage { index: 2 });
    let s3 = build_stage3(
        &p,
        params,
        &mut ops,
        &mut kernels,
        &mut ch,
        &mut batched_sumchecks,
        &s2,
    );

    // Stage 4: RegistersReadWriteChecking + RamValCheck
    ops.push(Op::BeginStage { index: 3 });
    let s4 = build_stage4(
        &p,
        params,
        &mut ops,
        &mut kernels,
        &mut ch,
        &mut batched_sumchecks,
        &s2,
        &s3,
    );

    // Stage 5: InstructionReadRaf + RamRaReduction + RegistersValEval
    ops.push(Op::BeginStage { index: 4 });
    let s5 = build_stage5(
        &p,
        params,
        &mut ops,
        &mut kernels,
        &mut ch,
        &mut batched_sumchecks,
        &s2,
        &s4,
    );

    // Stage 6: BytecodeRaf + Booleanity + HammingBool + RamRaVirt + InstRaVirt + IncReduction
    ops.push(Op::BeginStage { index: 5 });
    let s6 = build_stage6(
        &p,
        params,
        &mut ops,
        &mut kernels,
        &mut ch,
        &mut batched_sumchecks,
        &s2,
        &s3,
        &s4,
        &s5,
    );

    // Stage 7: HammingWeightReduction
    ops.push(Op::BeginStage { index: 6 });
    let s7 = build_stage7(
        &p,
        params,
        &mut ops,
        &mut kernels,
        &mut ch,
        &mut batched_sumchecks,
        &s6,
    );

    // Stage 8: Dory batch opening proof
    build_stage8(&p, params, &mut ops, &s7);

    // Build verifier schedule
    let mut verifier_ops = vec![VerifierOp::Preamble];
    verifier_ops.extend(build_verifier_stage1_ops(&p, params, &ch));
    verifier_ops.extend(build_verifier_stage2_ops(&p, params, &ch));
    verifier_ops.extend(build_verifier_stage3_ops(&p, params, &ch));
    verifier_ops.extend(build_verifier_stage4_ops(&p, params, &ch));
    verifier_ops.extend(build_verifier_stage5_ops(&p, params, &ch));
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
            num_stages: 5,
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
    main_witness.push(p.field_reg_inc);
    main_witness.extend_from_slice(&p.instruction_ra);
    main_witness.extend_from_slice(&p.ram_ra);
    main_witness.extend_from_slice(&p.bytecode_ra);
    main_witness.extend_from_slice(&p.field_reg_ra);
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
///   1. Squeeze τ (params.num_tau = log_t + 2 challenges)
///   2. Outer uniskip: emit s1(Y) coefficients → squeeze r0 → flush 1 opening claim
///   3. Outer remaining (batched, 1 instance):
///      a. Emit 1 input claim (= s1(r0), tag "sumcheck_claim")
///      b. Squeeze 1 batching coefficient
///      c. log_t + 1 rounds (1 streaming + log_t linear): emit round poly → squeeze r_j
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
    // 1. Squeeze τ = [τ_low ∥ τ_high] (num_tau = log_t + 2 challenges)
    //
    // τ_high (last element) is the Lagrange kernel argument for uniskip.
    // τ_low (first log_t + 1 elements) seeds the eq table for the remaining
    // rounds (1 streaming + log_t linear).
    let tau_base = ch.decls.len();
    for i in 0..params.num_tau {
        let idx = ch.add(
            &format!("tau_{i}"),
            ChallengeSource::FiatShamir { after_stage: 0 },
        );
        ops.push(Op::Squeeze { challenge: idx });
    }

    // 2. Outer Uniskip: group-split, 1 round producing a degree-45 polynomial.
    //
    // 32 R1CS eq constraints (19 RV + 13 BN254 Fr) split into 2 groups of 16.
    // The uniskip domain is 16. The group bit is encoded as the LOWEST
    // variable of the eq table (interleaved with cycle positions),
    // doubling the eq table to 2T entries.
    //
    //   s1(Y) = L(τ_high, Y) · Σ_{x} eq(τ_low, x) · Az_grouped(x, Y) · Bz_grouped(x, Y)
    //
    // x ranges over 2T entries (T cycles × 2 groups, interleaved); Y
    // ranges over the 16-point uniskip domain. L(τ_high, Y) has degree 15,
    // t1(Y) has degree 30, so s1(Y) has degree 45 with 46 coefficients.
    let spartan_formula = Formula::from_terms(vec![ProductTerm {
        coefficient: 1,
        factors: vec![Factor::Input(0), Factor::Input(1), Factor::Input(2)],
    }]);

    // τ_high is the last tau challenge — the Lagrange kernel argument.
    let tau_high_idx = ChallengeIdx(tau_base + params.num_tau - 1);

    // Group-split constraint indices (extends jolt-core's R1CS_CONSTRAINTS_FIRST_GROUP_LABELS
    // with the 13 BN254 Fr coprocessor rows from `rv64::FR_ROWS`).
    // Group 0 (16 constraints): boolean guards / small-Bz + FR equality/bridge rows.
    // Group 1 (16 constraints): arithmetic / large-Bz + FR mul/inv/SLL rows.
    // All 32 eq rows must appear in some group; rows outside `group_indices`
    // are zero-padded by RegroupConstraints and silently unenforced.
    let group0_indices: Vec<usize> =
        vec![1, 2, 3, 4, 5, 6, 11, 14, 17, 18, 19, 20, 21, 24, 27, 28];
    let group1_indices: Vec<usize> =
        vec![0, 7, 8, 9, 10, 12, 13, 15, 16, 22, 23, 25, 26, 29, 30, 31];
    let regrouped_stride = params.outer_uniskip_domain.next_power_of_two(); // 16

    // Regroup Az/Bz from flat T×32 to interleaved 2T×16 layout.
    // After regrouping, buf[(2c + g) * 16 + k] holds group g's k-th constraint for cycle c.
    ops.push(Op::RegroupConstraints {
        polys: vec![p.az, p.bz],
        group_indices: vec![group0_indices, group1_indices],
        old_stride: params.num_constraints_padded,
        new_stride: regrouped_stride,
        num_cycles: 1 << params.log_t,
    });

    // Eq table: log_t+1 challenges (all tau except τ_high), size 2T.
    // In the interleaved eq table construction, bit 0 of the index
    // corresponds to the LAST challenge (τ_{log_t}), which is the group
    // variable in jolt-core's convention (x_in & 1 selects the group).
    // This same eq table serves both the uniskip and the remaining sumcheck.
    let eq_challenges: Vec<ChallengeIdx> = (tau_base..=tau_base + params.log_t)
        .map(ChallengeIdx)
        .collect();

    let uniskip_inputs = vec![
        InputBinding::EqTable {
            poly: p.spartan_eq,
            challenges: eq_challenges.clone(),
        },
        InputBinding::Provided { poly: p.az },
        InputBinding::Provided { poly: p.bz },
    ];

    let remaining_inputs = vec![
        InputBinding::EqTable {
            poly: p.spartan_eq,
            challenges: eq_challenges,
        },
        InputBinding::Provided { poly: p.az },
        InputBinding::Provided { poly: p.bz },
    ];

    let outer_uniskip_kernel = kernels.len();
    kernels.push(KernelDef {
        spec: KernelSpec {
            formula: spartan_formula.clone(),
            num_evals: 2 * params.outer_uniskip_domain - 1, // 2×16−1 = 31
            iteration: Iteration::Domain {
                domain_size: params.outer_uniskip_domain, // 16
                stride: regrouped_stride,                 // 16
                domain_start: -((params.outer_uniskip_domain as i64 - 1) / 2),
                domain_indexed: vec![false, true, true], // eq=cycle, Az=domain, Bz=domain
                tau_challenge: tau_high_idx,
                zero_base: true, // R1CS: Az*Bz = Cz vanishes at base points
            },
            binding_order: BindingOrder::LowToHigh,
            gruen_hint: None,
        },
        inputs: uniskip_inputs,
        num_rounds: 1,
        instance_config: None,
    });

    for binding in &kernels[outer_uniskip_kernel].inputs {
        match binding {
            InputBinding::Provided { .. } => {
                // Az/Bz are already regrouped by RegroupConstraints — don't overwrite.
                ops.push(Op::MaterializeIfAbsent {
                    binding: binding.clone(),
                });
            }
            _ => {
                ops.push(Op::Materialize {
                    binding: binding.clone(),
                });
            }
        }
    }
    ops.push(Op::SumcheckRound {
        kernel: outer_uniskip_kernel,
        round: 0,
        bind_challenge: None,
    });
    ops.push(Op::AbsorbRoundPoly {
        num_coeffs: params.outer_uniskip_num_coeffs,
        tag: DomainSeparator::UniskipPoly,
        encoding: RoundPolyEncoding::Uniskip {
            domain_size: params.outer_uniskip_domain,
            domain_start: -((params.outer_uniskip_domain as i64 - 1) / 2),
            tau_challenge: tau_high_idx,
            zero_base: true,
        },
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

    // 2b. Lagrange projection: collapse constraint dimension.
    //
    // After regrouping, Az/Bz have 2T × regrouped_stride entries.
    // Projection evaluates the domain polynomial at r0, collapsing
    // to 2T scalar entries (one per cycle×group pair).
    ops.push(Op::LagrangeProject {
        polys: vec![p.az, p.bz],
        challenge: ch_r0,
        domain_size: params.outer_uniskip_domain,
        domain_start: -((params.outer_uniskip_domain as i64 - 1) / 2),
        stride: regrouped_stride,
        group_offsets: vec![0],
        kernel_tau: Some(tau_high_idx),
    });

    // Flush uniskip opening claim: s1(r0) appended via flush_to_transcript.
    ops.push(Op::Evaluate {
        poly: p.outer_uniskip_eval,
        mode: EvalMode::RoundPoly,
    });
    ops.push(Op::RecordEvals {
        polys: vec![p.outer_uniskip_eval],
    });
    ops.push(Op::AbsorbEvals {
        polys: vec![p.outer_uniskip_eval],
        tag: DomainSeparator::OpeningClaim,
    });

    // 3. Outer Remaining: log_t+1 rounds of degree-3 sumcheck.
    //
    // Same composition: eq(τ_low, x) · Az(x) · Bz(x), degree 3.
    // Az/Bz are already 2T entries from the group-split regrouping
    // + Lagrange projection. The eq table (2T entries) is shared with
    // the uniskip — no rebuild needed.
    //
    // Round 0 binds the group bit (τ_{log_t}, mapped to bit 0 of the eq table index).
    // Rounds 1..log_t bind cycle variables.

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

    // Remaining kernel: same composition scaled by the batching coefficient.
    // jolt-core's BatchedSumcheck::prove multiplies each instance's round
    // polynomial by its batching coefficient before sending to the transcript.
    let remaining_formula = Formula::from_terms(vec![ProductTerm {
        coefficient: 1,
        factors: vec![
            Factor::Challenge(ch_batch.0 as u32),
            Factor::Input(0),
            Factor::Input(1),
            Factor::Input(2),
        ],
    }]);

    let outer_remaining_kernel = kernels.len();
    kernels.push(KernelDef {
        spec: KernelSpec {
            formula: remaining_formula,
            num_evals: params.outer_remaining_degree + 1,
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
            gruen_hint: None,
        },
        inputs: remaining_inputs,
        num_rounds: params.outer_remaining_rounds,
        instance_config: None,
    });

    // Remaining kernel inputs: EqTable reuses the uniskip's eq buffer;
    // Az/Bz already have projected data from LagrangeProject.
    // MaterializeIfAbsent skips existing buffers.
    for binding in &kernels[outer_remaining_kernel].inputs {
        ops.push(Op::MaterializeIfAbsent {
            binding: binding.clone(),
        });
    }

    // Remaining rounds: log_t + 1 binary rounds binding all eq variables.
    // Round 0 has no bind (initial 2T-entry buffers from group-split projection).
    // Rounds 1+ bind at the previous round's squeezed challenge.
    for round in 0..params.outer_remaining_rounds {
        let bind = if round > 0 {
            Some(ChallengeIdx(ch_batch.0 + round)) // previous round's challenge
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
            num_coeffs: params.outer_remaining_degree + 1,
            tag: DomainSeparator::SumcheckPoly,
            encoding: RoundPolyEncoding::Compressed,
        });
        ops.push(Op::Squeeze { challenge: ch_r });
    }

    // 4. Bind R1CS witness polynomials at the sumcheck challenge point,
    //    then evaluate + flush.
    //
    // The remaining sumcheck only bound the kernel inputs (eq, Az, Bz).
    // The 35 R1CS witness polynomials (T-length buffers) must be bound
    // at each of the log_t cycle challenges to produce scalar evaluations.
    //
    // Round 0 binds the group bit — R1CS witnesses don't depend on the
    // group variable so we skip it (witnesses stay at size T).
    let r1cs_polys = r1cs_input_polys(p);
    for round in 1..params.outer_remaining_rounds {
        // challenge index = ch_batch + 1 + round (first sumcheck challenge
        // is at ch_batch+1 because ch_batch is the batching coefficient)
        let ch_idx = ChallengeIdx(ch_batch.0 + 1 + round);
        ops.push(Op::Bind {
            polys: r1cs_polys.to_vec(),
            challenge: ch_idx,
            order: BindingOrder::LowToHigh,
        });
    }
    for &poly in &r1cs_polys {
        ops.push(Op::Evaluate {
            poly,
            mode: EvalMode::FinalBind,
        });
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
///   1. Absorb commitments (3 barriers) — count = `params.num_committed + 2`
///   2. Squeeze τ challenges
///   3. Verify uniskip: absorb round polynomial, squeeze r0, record s1(r0)
///   4. Verify remaining: input claim = s1(r0), log_t+1 rounds of degree-3 sumcheck
///   5. Record R1CS input evaluations
///   6. Check output: eq(τ_low, r) × L(τ_high, r0) × Az(r0, evals) × Bz(r0, evals)
fn build_verifier_stage1_ops(
    p: &Polys,
    params: &ModuleParams,
    ch: &ChallengeTable,
) -> Vec<VerifierOp> {
    // Commitment list: per-barrier (poly, tag) pairs matching the prover's
    // commitment phase. ORDER MUST MATCH `build_commitment_phase` exactly —
    // every absorption is sequenced into Fiat-Shamir, so any divergence
    // (including a missing poly) misaligns every downstream challenge.
    let mut commit_pairs: Vec<(PolynomialId, DomainSeparator)> =
        Vec::with_capacity(params.num_committed + 2);
    // Barrier 1: main witness → tag "commitment"
    for &poly in [p.rd_inc, p.ram_inc, p.field_reg_inc]
        .iter()
        .chain(p.instruction_ra.iter())
        .chain(p.ram_ra.iter())
        .chain(p.bytecode_ra.iter())
        .chain(p.field_reg_ra.iter())
    {
        commit_pairs.push((poly, DomainSeparator::Commitment));
    }
    // Barrier 2: untrusted advice
    commit_pairs.push((p.untrusted_advice, DomainSeparator::UntrustedAdvice));
    // Barrier 3: trusted advice
    commit_pairs.push((p.trusted_advice, DomainSeparator::TrustedAdvice));

    // τ challenges: indices 0..27
    let tau_challenges: Vec<ChallengeIdx> = (0..params.num_tau).map(ChallengeIdx).collect();

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
    let tau_high_idx = ChallengeIdx(params.num_tau - 1);
    let r0_idx = ChallengeIdx(params.num_tau); // the uniskip challenge
                                               // First sumcheck round challenge = r_stream (group bit).
                                               // Challenge layout: τ (0..num_tau) | r0 (num_tau) | ch_batch (num_tau+1) | r_stream (num_tau+2) | ...
    let r_group_idx = ChallengeIdx(params.num_tau + 2);
    // tau_cycle = τ_low (log_t entries): τ_0..τ_{log_t-1}.
    let tau_cycle: Vec<ChallengeIdx> = (0..params.num_tau - 1).map(ChallengeIdx).collect();
    // Group split: 16 constraints per group (32 eq rows = 19 RV + 13 BN254 Fr).
    // MUST match the prover-side `group0_indices` / `group1_indices` above —
    // mismatch here lets the prover skip enforcement on rows the verifier
    // believes are covered.
    let group0: Vec<usize> = vec![1, 2, 3, 4, 5, 6, 11, 14, 17, 18, 19, 20, 21, 24, 27, 28];
    let group1: Vec<usize> = vec![0, 7, 8, 9, 10, 12, 13, 15, 16, 22, 23, 25, 26, 29, 30, 31];
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
                    domain_start: -((params.outer_uniskip_domain as i64 - 1) / 2),
                },
                ClaimFactor::GroupSplitR1CSEval {
                    matrix: R1CSMatrix::A,
                    eval_polys: r1cs_polys.to_vec(),
                    at_r0: r0_idx,
                    at_r_group: r_group_idx,
                    group0_indices: group0.clone(),
                    group1_indices: group1.clone(),
                    domain_size: params.outer_uniskip_domain,
                    domain_start: -((params.outer_uniskip_domain as i64 - 1) / 2),
                },
                ClaimFactor::GroupSplitR1CSEval {
                    matrix: R1CSMatrix::B,
                    eval_polys: r1cs_polys.to_vec(),
                    at_r0: r0_idx,
                    at_r_group: r_group_idx,
                    group0_indices: group0,
                    group1_indices: group1,
                    domain_size: params.outer_uniskip_domain,
                    domain_start: -((params.outer_uniskip_domain as i64 - 1) / 2),
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
    // Uniskip protocol: absorb poly → squeeze r0 → record/absorb eval
    // Uniskip verification is partial: absorb + squeeze only (no s(0)+s(1) check).
    ops.push(VerifierOp::AbsorbRoundPoly {
        num_coeffs: params.outer_uniskip_num_coeffs,
        tag: DomainSeparator::UniskipPoly,
    });
    ops.push(VerifierOp::Squeeze { challenge: r0_idx });
    ops.push(VerifierOp::RecordEvals {
        evals: vec![uniskip_eval],
    });
    // Prover flushes uniskip eval as OpeningClaim (the SumcheckClaim absorption
    // + ch_batch squeeze are handled internally by VerifySumcheck below).
    ops.push(VerifierOp::AbsorbEvals {
        polys: vec![p.outer_uniskip_eval],
        tag: DomainSeparator::OpeningClaim,
    });
    // Batch coefficient for the (single-instance) remaining sumcheck.
    let ch_batch = ChallengeIdx(r0_idx.0 + 1);
    // Stage 0 sumcheck rounds populate challenges[num_tau+2 .. num_tau+2+outer_remaining_rounds].
    // Slot num_tau+2 is the first round (r_group/r_stream — the group bit).
    let stage0_round_base = params.num_tau + 2;
    let stage0_round_slots: Vec<ChallengeIdx> = (stage0_round_base
        ..stage0_round_base + params.outer_remaining_rounds)
        .map(ChallengeIdx)
        .collect();
    ops.push(VerifierOp::VerifySumcheck {
        instances: instances.clone(),
        stage: 0,
        batch_challenges: vec![ch_batch],
        claim_tag: Some(DomainSeparator::SumcheckClaim),
        sumcheck_challenge_slots: stage0_round_slots,
    });
    ops.push(VerifierOp::RecordEvals { evals: evaluations });
    ops.push(VerifierOp::AbsorbEvals {
        polys: eval_polys,
        tag: DomainSeparator::OpeningClaim,
    });
    ops.push(VerifierOp::CheckOutput {
        instances,
        stage: 0,
        batch_challenges: vec![ch_batch],
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
/// Challenge indices produced by Stage 2 that downstream stages need.
struct Stage2Challenges {
    /// Stage 1 outer remaining cycle challenges (big-endian, log_t entries).
    stage1_cycle: Vec<ChallengeIdx>,
    /// Stage 2 batched sumcheck round challenge indices (all max_rounds).
    round_challenges: Vec<ChallengeIdx>,
}

/// Challenge indices produced by Stage 3 that downstream stages need.
struct Stage3Challenges {
    /// Stage 3 batched sumcheck round challenge indices (log_t entries).
    round_challenges: Vec<ChallengeIdx>,
}

/// Challenge indices produced by Stage 4 that downstream stages need.
struct Stage4Challenges {
    /// Order: [cycle_0, ..., cycle_{log_t-1}, addr_0, ..., addr_{log_k_reg-1}]
    /// (LowToHigh binding order within each phase).
    round_challenges: Vec<ChallengeIdx>,
}

/// Challenge indices produced by Stage 5 that downstream stages need.
struct Stage5Challenges {
    /// Stage 5 batched sumcheck round challenge indices (all max_rounds = LOG_K_INSTRUCTION + log_t).
    round_challenges: Vec<ChallengeIdx>,
}

/// Data produced by Stage 6 that Stage 7 needs.
struct Stage6Challenges {
    round_challenges: Vec<ChallengeIdx>,
    /// Per-chunk address challenge indices for BytecodeRa virtualization opening.
    bc_addr_chunks: Vec<Vec<ChallengeIdx>>,
    /// Per-chunk address challenge indices for InstructionRa virtualization opening.
    inst_addr_chunks: Vec<Vec<ChallengeIdx>>,
    /// Per-chunk address challenge indices for RamRa virtualization opening.
    ram_addr_chunks: Vec<Vec<ChallengeIdx>>,
    /// BatchEq projection poly IDs for BytecodeRa from BytecodeReadRaf.
    bc_raf_proj_ids: Vec<PolynomialId>,
    /// BatchEq projection poly IDs for InstructionRa from InstructionRaVirtual.
    inst_ra_proj_ids: Vec<PolynomialId>,
    /// BatchEq projection poly IDs for RamRa from RamRaVirtual.
    ram_ra_proj_ids: Vec<PolynomialId>,
}

fn build_stage2(
    p: &Polys,
    params: &ModuleParams,
    ops: &mut Vec<Op>,
    kernels: &mut Vec<KernelDef>,
    ch: &mut ChallengeTable,
    batched_sumchecks: &mut Vec<BatchedSumcheckDef>,
) -> Stage2Challenges {
    // Stage 1 cycle challenge indices: the outer remaining round challenges
    // EXCLUDING the first (group interleave variable), reversed to BIG_ENDIAN
    // order (matching jolt-core's r_cycle convention).
    //
    // The outer remaining sumcheck has log_t + 1 rounds: challenge[0] is the
    // group variable (interleave bit), challenges[1..] are cycle variables.
    // jolt-core's normalize_opening_point drops challenge[0], so r_cycle
    // has log_t entries.
    let stage1_round_base = params.num_tau + 2; // τ_count + r0 + batch
    let stage1_cycle_challenges: Vec<ChallengeIdx> = (stage1_round_base + 1
        ..stage1_round_base + params.outer_remaining_rounds)
        .rev()
        .map(ChallengeIdx)
        .collect();

    // 1. Squeeze τ_high for product uniskip
    //    (τ_low is r_cycle from Stage 1 outer remaining round challenges)
    let ch_product_tau_high = ch.add(
        "product_tau_high",
        ChallengeSource::FiatShamir { after_stage: 1 },
    );
    ops.push(Op::Squeeze {
        challenge: ch_product_tau_high,
    });

    // 2. Product uniskip: 1 round producing a degree-6 polynomial.
    //
    //   s2(Y) = L(τ_high, Y) · Σ_x eq(r_cycle, x) · Left(x, Y) · Right(x, Y)
    //
    // τ_low = r_cycle (Stage 1 outer remaining round challenges), NOT
    // the original Spartan tau. In jolt-core, ProductVirtualUniSkipParams
    // retrieves r_cycle from the opening accumulator.
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
                domain_start: -((params.product_uniskip_domain as i64 - 1) / 2),
                domain_indexed: vec![false, true, true],
                tau_challenge: ch_product_tau_high,
                zero_base: false, // product: base evals are non-zero
            },
            binding_order: BindingOrder::LowToHigh,
            gruen_hint: None,
        },
        inputs: product_uniskip_inputs,
        num_rounds: 1,
        instance_config: None,
    });

    for binding in &kernels[product_uniskip_kernel].inputs {
        ops.push(Op::Materialize {
            binding: binding.clone(),
        });
    }
    ops.push(Op::SumcheckRound {
        kernel: product_uniskip_kernel,
        round: 0,
        bind_challenge: None,
    });
    ops.push(Op::AbsorbRoundPoly {
        num_coeffs: params.product_uniskip_num_coeffs,
        tag: DomainSeparator::UniskipPoly,
        encoding: RoundPolyEncoding::Uniskip {
            domain_size: params.product_uniskip_domain,
            domain_start: -((params.product_uniskip_domain as i64 - 1) / 2),
            tau_challenge: ch_product_tau_high,
            zero_base: false,
        },
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
        mode: EvalMode::RoundPoly,
    });
    ops.push(Op::RecordEvals {
        polys: vec![p.product_uniskip_eval],
    });
    ops.push(Op::AbsorbEvals {
        polys: vec![p.product_uniskip_eval],
        tag: DomainSeparator::OpeningClaim,
    });

    // 2b. Lagrange projection: collapse product_left/right domain dim.
    //
    // After the product uniskip, product_left/right are T × stride
    // domain-indexed buffers. The Lagrange projection evaluates at r0,
    // reducing to T-element vectors. kernel_tau absorbs L(τ_high, r0)
    // into the first poly, so the ProductRemainder sum equals s2(r0).
    ops.push(Op::LagrangeProject {
        polys: vec![p.product_left, p.product_right],
        challenge: ch_product_r0,
        domain_size: params.product_uniskip_domain,
        domain_start: -((params.product_uniskip_domain as i64 - 1) / 2),
        stride: product_stride,
        group_offsets: vec![0],
        kernel_tau: Some(ch_product_tau_high),
    });

    // 3. Pre-squeeze challenges for batched instances

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
    let ch_r_address_base = ChallengeIdx(ch.decls.len());
    for i in 0..params.log_k_ram {
        let idx = ch.add(
            &format!("output_r_address_{i}"),
            ChallengeSource::FiatShamir { after_stage: 1 },
        );
        ops.push(Op::Squeeze { challenge: idx });
    }

    // 4. Batched sumcheck (5 instances)
    //
    // Each instance's input_claim is computed from prior evaluations
    // and appended to transcript with "sumcheck_claim" tag.
    // Then 5 batching coefficients are squeezed.

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

    // 4a. Emit 5 input claims via AbsorbInputClaim.
    //
    // Each ClaimFormula is evaluated by the runtime against current
    // state.evaluations and state.challenges, then absorbed into the
    // transcript with "sumcheck_claim" tag. The batch/instance indices
    // allow the runtime to initialize per-instance claims for inactive
    // round contributions.
    let batch_idx = BatchIdx(batched_sumchecks.len());

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
        instance: InstanceIdx(0),
        inactive_scale_bits: 0,
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
        instance: InstanceIdx(1),
        inactive_scale_bits: params.stage2_max_rounds - params.product_remainder_rounds,
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
        instance: InstanceIdx(2),
        inactive_scale_bits: params.stage2_max_rounds - params.instruction_claim_reduction_rounds,
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
        instance: InstanceIdx(3),
        inactive_scale_bits: params.stage2_max_rounds - params.raf_evaluation_rounds,
    });

    // [4] OutputCheck: 0 (zero-check)
    let output_check_input_claim = ClaimFormula::zero();
    ops.push(Op::AbsorbInputClaim {
        formula: output_check_input_claim.clone(),
        tag: DomainSeparator::SumcheckClaim,
        batch: batch_idx,
        instance: InstanceIdx(4),
        inactive_scale_bits: params.stage2_max_rounds - params.output_check_rounds,
    });

    // Squeeze 5 batching coefficients
    let ch_batch_base = ChallengeIdx(ch.decls.len());
    for i in 0..params.stage2_num_instances {
        let idx = ch.add(
            &format!("s2_batch_{i}"),
            ChallengeSource::FiatShamir { after_stage: 1 },
        );
        ops.push(Op::Squeeze { challenge: idx });
    }

    // 5. Build per-instance kernels for the batched sumcheck.

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
    let rw_cycle_eq_challenges: Vec<ChallengeIdx> = stage1_cycle_challenges.clone();

    // Address challenge indices for the outer eq table:
    let rw_addr_eq_challenges: Vec<ChallengeIdx> = (ch_r_address_base.0
        ..ch_r_address_base.0 + params.log_k_ram)
        .map(ChallengeIdx)
        .collect();

    // Phase 1 kernel: cycle binding (Dense inner kernel, segmented reduce).
    //
    // Formula factors as eq_cycle(x) · ra(x) · ((1 + γ_rw)·val(x) + γ_rw·inc(x)),
    // matching the LinComboQ shape: q(x) = a(x) · ((1+γ)·b(x) + γ·c(x))
    // with a=ra (Input 1), b=val (Input 2), c=inc (Input 3).
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
                        Factor::Challenge(ch_gamma_rw.0 as u32),
                        Factor::Input(0),
                        Factor::Input(1),
                        Factor::Input(2),
                    ],
                },
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(ch_gamma_rw.0 as u32),
                        Factor::Input(0),
                        Factor::Input(1),
                        Factor::Input(3),
                    ],
                },
            ]),
            num_evals: params.rw_checking_degree + 1,
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
            gruen_hint: Some(GruenHint {
                eq_input: 0,
                eq_challenges: rw_cycle_eq_challenges.clone(),
                q_lincombo: LinComboQ {
                    a_input: 1,
                    b_input: 2,
                    c_input: 3,
                    gamma_challenge: ch_gamma_rw,
                },
            }),
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
        instance_config: None,
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
                        Factor::Challenge(ch_rw_eq_bound.0 as u32),
                        Factor::Input(0),
                        Factor::Input(1),
                    ],
                },
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(ch_gamma_rw.0 as u32),
                        Factor::Challenge(ch_rw_eq_bound.0 as u32),
                        Factor::Input(0),
                        Factor::Input(1),
                    ],
                },
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(ch_gamma_rw.0 as u32),
                        Factor::Challenge(ch_rw_eq_bound.0 as u32),
                        Factor::Challenge(ch_rw_inc_bound.0 as u32),
                        Factor::Input(0),
                    ],
                },
            ]),
            // Degree 2: max input factor count is 2 (ra × val)
            num_evals: 3,
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
            gruen_hint: None,
        },
        inputs: vec![
            InputBinding::Provided {
                poly: p.ram_combined_ra,
            },
            InputBinding::Provided { poly: p.ram_val },
        ],
        num_rounds: params.log_k_ram,
        instance_config: None,
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
            gruen_hint: None,
        },
        inputs: vec![
            InputBinding::EqTable {
                poly: PolynomialId::BatchEq(1),
                challenges: stage1_cycle_challenges.clone(),
            },
            InputBinding::Provided {
                poly: p.product_left,
            },
            InputBinding::Provided {
                poly: p.product_right,
            },
        ],
        num_rounds: params.product_remainder_rounds,
        instance_config: None,
    });

    // [2] InstructionClaimReduction — 25 rounds, degree 2.
    //
    // Formula: eq * (lookup + γ·left_op + γ²·right_op + γ³·left_inst + γ⁴·right_inst)
    let gamma = ch_gamma_instruction.0 as u32;
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
            gruen_hint: None,
        },
        inputs: vec![
            InputBinding::EqTable {
                poly: PolynomialId::BatchEq(2),
                challenges: stage1_cycle_challenges.clone(),
            },
            InputBinding::Provided {
                poly: p.lookup_output,
            },
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
        instance_config: None,
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
            gruen_hint: None,
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
        instance_config: None,
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
            gruen_hint: None,
        },
        inputs: vec![
            InputBinding::EqTable {
                poly: PolynomialId::BatchEq(4),
                challenges: (ch_r_address_base.0..ch_r_address_base.0 + params.log_k_ram)
                    .map(ChallengeIdx)
                    .collect(),
            },
            InputBinding::Provided { poly: p.io_mask },
            InputBinding::Provided {
                poly: p.ram_val_final,
            },
            InputBinding::Provided { poly: p.val_io },
        ],
        num_rounds: params.output_check_rounds,
        instance_config: None,
    });

    // 5b. Build BatchedSumcheckDef with all 5 instances.
    let batch_idx = BatchIdx(batched_sumchecks.len());
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
                        carry_bindings: vec![],
                        pre_activation_ops: vec![],
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
                        carry_bindings: vec![],
                        pre_activation_ops: vec![],
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
                    carry_bindings: vec![],
                    pre_activation_ops: vec![],
                }],
                batch_coeff: ChallengeIdx(ch_batch_base.0 + 1),
                first_active_round: params.stage2_max_rounds - params.product_remainder_rounds,
            },
            // [2] InstructionClaimReduction — single phase.
            BatchedInstance {
                phases: vec![InstancePhase {
                    kernel: inst_cr_kernel_idx,
                    num_rounds: params.instruction_claim_reduction_rounds,
                    scalar_captures: vec![],
                    segmented: None,
                    carry_bindings: vec![],
                    pre_activation_ops: vec![],
                }],
                batch_coeff: ChallengeIdx(ch_batch_base.0 + 2),
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
                    carry_bindings: vec![],
                    pre_activation_ops: vec![],
                }],
                batch_coeff: ChallengeIdx(ch_batch_base.0 + 3),
                first_active_round: params.stage2_max_rounds - params.raf_evaluation_rounds,
            },
            // [4] OutputCheck — single phase.
            BatchedInstance {
                phases: vec![InstancePhase {
                    kernel: output_kernel_idx,
                    num_rounds: params.output_check_rounds,
                    scalar_captures: vec![],
                    segmented: None,
                    carry_bindings: vec![],
                    pre_activation_ops: vec![],
                }],
                batch_coeff: ChallengeIdx(ch_batch_base.0 + 4),
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

    // Patch the RAF kernel's EqProject challenges BEFORE emitting rounds,
    // so the Materialize ops clone the binding with the correct challenges.
    // The EqProject marginalizes the cycle dimension of the RA indicator using
    // the Stage 1 cycle point (where ram_address was evaluated), NOT the Stage 2
    // round challenges.
    {
        if let InputBinding::EqProject {
            challenges: ref mut chs,
            ..
        } = kernels[raf_kernel_idx].inputs[1]
        {
            chs.clone_from(&rw_cycle_eq_challenges);
        }
    }

    // 5c. Unrolled batched sumcheck rounds.
    let rw_coeffs = params.rw_checking_degree + 1;
    let round_challenge_indices = emit_unrolled_batched_rounds(
        ops,
        kernels,
        batched_sumchecks,
        ch,
        batch_idx,
        params.stage2_max_rounds,
        |_| rw_coeffs,
        VerifierStageIndex(1),
        "s2_r",
        1,
        None,
    );

    // 6. Bind polynomials opened by Instance 1 (ProductRemainder)
    //    that are NOT kernel inputs of any batched instance.
    //
    //    Instances 1 and 2 share the same active rounds (13..21),
    //    so their sumcheck points coincide. Instance 2's kernel
    //    already binds: lookup_output, left/right_lookup_operand,
    //    left/right_instruction_input. The remaining 5 polys need
    //    explicit binding at the same (Instance 1/2) challenge point.
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
    // 15 unique evals (jolt-core aliases duplicates at the same opening point).
    // ProductRemainder and InstructionCR share LookupOutput,
    // LeftInstructionInput, and RightInstructionInput at the same cycle point,
    // so only the first occurrence is flushed to the transcript.
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
        // InstructionLookupsClaimReduction (2 new; LookupOutput,
        // LeftInstructionInput, RightInstructionInput aliased above)
        p.left_lookup_operand,
        p.right_lookup_operand,
        // RafEvaluation (1)
        p.ram_raf_ra,
        // OutputCheck (1)
        p.ram_val_final,
    ];

    for &poly in &stage2_eval_polys {
        ops.push(Op::Evaluate {
            poly,
            mode: EvalMode::FinalBind,
        });
    }
    ops.push(Op::RecordEvals {
        polys: stage2_eval_polys.clone(),
    });
    ops.push(Op::AbsorbEvals {
        polys: stage2_eval_polys,
        tag: DomainSeparator::OpeningClaim,
    });

    Stage2Challenges {
        stage1_cycle: stage1_cycle_challenges.clone(),
        round_challenges: round_challenge_indices,
    }
}

/// Stage 3: Shift + InstructionInput + RegistersClaimReduction.
///
/// Three batched sumcheck instances, all log_T rounds:
///   [0] Shift                        — degree 2
///   [1] InstructionInput             — degree 3
///   [2] RegistersClaimReduction      — degree 2
///
/// Pre-sumcheck transcript operations:
///   1. challenge_scalar_powers(5) → γ_shift (1 squeeze)
///   2. challenge_scalar()         → γ_instruction_input (1 squeeze)
///   3. challenge_scalar()         → γ_registers (1 squeeze)
fn build_stage3(
    p: &Polys,
    params: &ModuleParams,
    ops: &mut Vec<Op>,
    kernels: &mut Vec<KernelDef>,
    ch: &mut ChallengeTable,
    batched_sumchecks: &mut Vec<BatchedSumcheckDef>,
    s2: &Stage2Challenges,
) -> Stage3Challenges {
    // γ_shift (ShiftSumcheckParams::new → challenge_scalar_powers(5) = 1 squeeze)
    let ch_shift_gamma = ch.add(
        "shift_gamma",
        ChallengeSource::FiatShamir { after_stage: 2 },
    );
    ops.push(Op::Squeeze {
        challenge: ch_shift_gamma,
    });

    // γ_instruction_input (InstructionInputParams::new → challenge_scalar() = 1 squeeze)
    let ch_inst_input_gamma = ch.add(
        "instruction_input_gamma",
        ChallengeSource::FiatShamir { after_stage: 2 },
    );
    ops.push(Op::Squeeze {
        challenge: ch_inst_input_gamma,
    });

    // γ_registers (RegistersClaimReductionSumcheckParams::new → challenge_scalar() = 1 squeeze)
    let ch_registers_gamma = ch.add(
        "registers_gamma",
        ChallengeSource::FiatShamir { after_stage: 2 },
    );
    ops.push(Op::Squeeze {
        challenge: ch_registers_gamma,
    });

    let ch_fr_cr_gamma = ch.add(
        "fr_cr_gamma",
        ChallengeSource::FiatShamir { after_stage: 2 },
    );
    ops.push(Op::Squeeze {
        challenge: ch_fr_cr_gamma,
    });

    // Input claims (absorbed before sumcheck rounds begin).
    let batch_idx = BatchIdx(batched_sumchecks.len());

    // [0] Shift: unexpanded_pc + γ*next_pc + γ²*next_is_virtual
    //          + γ³*next_is_first + γ⁴*(1 - next_is_noop)
    let shift_input_claim = ClaimFormula {
        terms: vec![
            ClaimTerm {
                coeff: 1,
                factors: vec![ClaimFactor::Eval(p.next_unexpanded_pc)],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_shift_gamma),
                    ClaimFactor::Eval(p.next_pc),
                ],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_shift_gamma),
                    ClaimFactor::Challenge(ch_shift_gamma),
                    ClaimFactor::Eval(p.next_is_virtual),
                ],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_shift_gamma),
                    ClaimFactor::Challenge(ch_shift_gamma),
                    ClaimFactor::Challenge(ch_shift_gamma),
                    ClaimFactor::Eval(p.next_is_first),
                ],
            },
            // + γ⁴
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_shift_gamma),
                    ClaimFactor::Challenge(ch_shift_gamma),
                    ClaimFactor::Challenge(ch_shift_gamma),
                    ClaimFactor::Challenge(ch_shift_gamma),
                ],
            },
            // - γ⁴ * next_is_noop
            ClaimTerm {
                coeff: -1,
                factors: vec![
                    ClaimFactor::Challenge(ch_shift_gamma),
                    ClaimFactor::Challenge(ch_shift_gamma),
                    ClaimFactor::Challenge(ch_shift_gamma),
                    ClaimFactor::Challenge(ch_shift_gamma),
                    ClaimFactor::Eval(p.next_is_noop),
                ],
            },
        ],
    };
    ops.push(Op::AbsorbInputClaim {
        formula: shift_input_claim.clone(),
        tag: DomainSeparator::SumcheckClaim,
        batch: batch_idx,
        instance: InstanceIdx(0),
        inactive_scale_bits: 0,
    });

    // [1] InstructionInput: right_instruction_input + γ * left_instruction_input
    let inst_input_claim = ClaimFormula {
        terms: vec![
            ClaimTerm {
                coeff: 1,
                factors: vec![ClaimFactor::Eval(p.right_instruction_input)],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_inst_input_gamma),
                    ClaimFactor::Eval(p.left_instruction_input),
                ],
            },
        ],
    };
    ops.push(Op::AbsorbInputClaim {
        formula: inst_input_claim.clone(),
        tag: DomainSeparator::SumcheckClaim,
        batch: batch_idx,
        instance: InstanceIdx(1),
        inactive_scale_bits: 0,
    });

    // [2] RegistersCR: rd_write_value + γ * rs1_val + γ² * rs2_val
    let registers_input_claim = ClaimFormula {
        terms: vec![
            ClaimTerm {
                coeff: 1,
                factors: vec![ClaimFactor::Eval(p.rd_write_value)],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_registers_gamma),
                    ClaimFactor::Eval(p.rs1_val),
                ],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_registers_gamma),
                    ClaimFactor::Challenge(ch_registers_gamma),
                    ClaimFactor::Eval(p.rs2_val),
                ],
            },
        ],
    };
    ops.push(Op::AbsorbInputClaim {
        formula: registers_input_claim.clone(),
        tag: DomainSeparator::SumcheckClaim,
        batch: batch_idx,
        instance: InstanceIdx(2),
        inactive_scale_bits: 0,
    });

    // [3] FieldRegCR input claim: field_rd_value + γ·field_rs1_value + γ²·field_rs2_value
    let fr_cr_input_claim = ClaimFormula {
        terms: vec![
            ClaimTerm {
                coeff: 1,
                factors: vec![ClaimFactor::Eval(p.field_rd_value)],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_fr_cr_gamma),
                    ClaimFactor::Eval(p.field_rs1_value),
                ],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_fr_cr_gamma),
                    ClaimFactor::Challenge(ch_fr_cr_gamma),
                    ClaimFactor::Eval(p.field_rs2_value),
                ],
            },
        ],
    };
    ops.push(Op::AbsorbInputClaim {
        formula: fr_cr_input_claim.clone(),
        tag: DomainSeparator::SumcheckClaim,
        batch: batch_idx,
        instance: InstanceIdx(3),
        inactive_scale_bits: 0,
    });

    // Squeeze 4 batching coefficients (one per instance).
    let ch_batch_base = ChallengeIdx(ch.decls.len());
    for i in 0..4 {
        let idx = ch.add(
            &format!("s3_batch_{i}"),
            ChallengeSource::FiatShamir { after_stage: 2 },
        );
        ops.push(Op::Squeeze { challenge: idx });
    }

    // Eq table challenge points.
    //
    // r_outer / r_spartan = Stage 1 outer remaining cycle point (big-endian).
    // r_product / r_cycle_stage2 = Stage 2 ProductRemainder cycle point (big-endian).
    //
    // ProductRemainder was active for rounds [log_k_ram, log_k_ram+log_t).
    // Its cycle point reversed to big-endian = round_challenges[last..first].
    let eq_outer_challenges: Vec<ChallengeIdx> = s2.stage1_cycle.clone();

    let prod_first_active = params.stage2_max_rounds - params.product_remainder_rounds;
    let eq_prod_challenges: Vec<ChallengeIdx> = (prod_first_active
        ..prod_first_active + params.log_t)
        .rev()
        .map(|r| s2.round_challenges[r])
        .collect();

    // Kernel definitions for each instance.

    // [0] Shift kernel — degree 2.
    //
    // eq_outer(x) * unexpanded_pc(x)
    // + γ * eq_outer(x) * pc(x)
    // + γ² * eq_outer(x) * is_virtual(x)
    // + γ³ * eq_outer(x) * is_first(x)
    // + γ⁴ * eq_prod(x)
    // - γ⁴ * eq_prod(x) * is_noop(x)
    let g = ch_shift_gamma.0 as u32;
    let shift_kernel_idx = kernels.len();
    kernels.push(KernelDef {
        spec: KernelSpec {
            formula: Formula::from_terms(vec![
                // eq_outer * unexpanded_pc
                ProductTerm {
                    coefficient: 1,
                    factors: vec![Factor::Input(0), Factor::Input(1)],
                },
                // γ * eq_outer * pc
                ProductTerm {
                    coefficient: 1,
                    factors: vec![Factor::Challenge(g), Factor::Input(0), Factor::Input(2)],
                },
                // γ² * eq_outer * is_virtual
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(g),
                        Factor::Challenge(g),
                        Factor::Input(0),
                        Factor::Input(3),
                    ],
                },
                // γ³ * eq_outer * is_first
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(g),
                        Factor::Challenge(g),
                        Factor::Challenge(g),
                        Factor::Input(0),
                        Factor::Input(4),
                    ],
                },
                // γ⁴ * eq_prod
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(g),
                        Factor::Challenge(g),
                        Factor::Challenge(g),
                        Factor::Challenge(g),
                        Factor::Input(5),
                    ],
                },
                // -γ⁴ * eq_prod * is_noop
                ProductTerm {
                    coefficient: -1,
                    factors: vec![
                        Factor::Challenge(g),
                        Factor::Challenge(g),
                        Factor::Challenge(g),
                        Factor::Challenge(g),
                        Factor::Input(5),
                        Factor::Input(6),
                    ],
                },
            ]),
            num_evals: 3, // degree 2 → 3 eval points
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
            gruen_hint: None,
        },
        inputs: vec![
            InputBinding::EqPlusOneTable {
                poly: PolynomialId::BatchEq(5),
                challenges: eq_outer_challenges.clone(),
            },
            InputBinding::Provided {
                poly: p.unexpanded_pc,
            },
            InputBinding::Provided { poly: p.pc },
            InputBinding::Provided {
                poly: p.op_flags[7], // VirtualInstruction
            },
            InputBinding::Provided {
                poly: p.op_flags[12], // IsFirstInSequence
            },
            InputBinding::EqPlusOneTable {
                poly: PolynomialId::BatchEq(6),
                challenges: eq_prod_challenges.clone(),
            },
            InputBinding::Provided {
                poly: p.inst_flag_is_noop,
            },
        ],
        num_rounds: params.log_t,
        instance_config: None,
    });

    // [1] InstructionInput kernel — degree 3.
    //
    // eq(r_cycle_stage2, x) * right_is_rs2(x) * rs2_value(x)
    // + eq * right_is_imm(x) * imm(x)
    // + γ * eq * left_is_rs1(x) * rs1_value(x)
    // + γ * eq * left_is_pc(x) * unexpanded_pc(x)
    let gi = ch_inst_input_gamma.0 as u32;
    let inst_input_kernel_idx = kernels.len();
    kernels.push(KernelDef {
        spec: KernelSpec {
            formula: Formula::from_terms(vec![
                // eq * right_is_rs2 * rs2_value
                ProductTerm {
                    coefficient: 1,
                    factors: vec![Factor::Input(0), Factor::Input(1), Factor::Input(2)],
                },
                // eq * right_is_imm * imm
                ProductTerm {
                    coefficient: 1,
                    factors: vec![Factor::Input(0), Factor::Input(3), Factor::Input(4)],
                },
                // γ * eq * left_is_rs1 * rs1_value
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(gi),
                        Factor::Input(0),
                        Factor::Input(5),
                        Factor::Input(6),
                    ],
                },
                // γ * eq * left_is_pc * unexpanded_pc
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(gi),
                        Factor::Input(0),
                        Factor::Input(7),
                        Factor::Input(8),
                    ],
                },
            ]),
            num_evals: 4, // degree 3 → 4 eval points
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
            gruen_hint: None,
        },
        inputs: vec![
            InputBinding::EqTable {
                poly: PolynomialId::BatchEq(7),
                challenges: eq_prod_challenges.clone(),
            },
            InputBinding::Provided {
                poly: p.inst_flag_right_is_rs2,
            },
            InputBinding::Provided { poly: p.rs2_val },
            InputBinding::Provided {
                poly: p.inst_flag_right_is_imm,
            },
            InputBinding::Provided { poly: p.imm },
            InputBinding::Provided {
                poly: p.inst_flag_left_is_rs1,
            },
            InputBinding::Provided { poly: p.rs1_val },
            InputBinding::Provided {
                poly: p.inst_flag_left_is_pc,
            },
            InputBinding::Provided {
                poly: p.unexpanded_pc,
            },
        ],
        num_rounds: params.log_t,
        instance_config: None,
    });

    // [2] RegistersClaimReduction kernel — degree 2.
    //
    // eq(r_spartan, x) * rd_write_value(x)
    // + γ * eq * rs1_value(x)
    // + γ² * eq * rs2_value(x)
    let gr = ch_registers_gamma.0 as u32;
    let registers_kernel_idx = kernels.len();
    kernels.push(KernelDef {
        spec: KernelSpec {
            formula: Formula::from_terms(vec![
                // eq * rd_write_value
                ProductTerm {
                    coefficient: 1,
                    factors: vec![Factor::Input(0), Factor::Input(1)],
                },
                // γ * eq * rs1_value
                ProductTerm {
                    coefficient: 1,
                    factors: vec![Factor::Challenge(gr), Factor::Input(0), Factor::Input(2)],
                },
                // γ² * eq * rs2_value
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(gr),
                        Factor::Challenge(gr),
                        Factor::Input(0),
                        Factor::Input(3),
                    ],
                },
            ]),
            num_evals: 3, // degree 2 → 3 eval points
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
            gruen_hint: None,
        },
        inputs: vec![
            InputBinding::EqTable {
                poly: PolynomialId::BatchEq(8),
                challenges: eq_outer_challenges.clone(),
            },
            InputBinding::Provided {
                poly: p.rd_write_value,
            },
            InputBinding::Provided { poly: p.rs1_val },
            InputBinding::Provided { poly: p.rs2_val },
        ],
        num_rounds: params.log_t,
        instance_config: None,
    });

    // [3] FieldRegClaimReduction kernel — structural mirror of registers_kernel.
    let g_fr_cr = ch_fr_cr_gamma.0 as u32;
    let fr_cr_kernel_idx = kernels.len();
    kernels.push(KernelDef {
        spec: KernelSpec {
            formula: Formula::from_terms(vec![
                ProductTerm {
                    coefficient: 1,
                    factors: vec![Factor::Input(0), Factor::Input(1)],
                },
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(g_fr_cr),
                        Factor::Input(0),
                        Factor::Input(2),
                    ],
                },
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(g_fr_cr),
                        Factor::Challenge(g_fr_cr),
                        Factor::Input(0),
                        Factor::Input(3),
                    ],
                },
            ]),
            num_evals: 3, // degree 2
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
            gruen_hint: None,
        },
        inputs: vec![
            // BatchEq(200..=203) are reserved for the FR Twist (Stage 3 CR, Stage 4
            // RW cycle, Stage 5 EqGather, Stage 5 LtTable). Stage 6's
            // `next_eq` counter starts at 19 and walks upward, so we stay
            // well clear by using the 200 block.
            InputBinding::EqTable {
                poly: PolynomialId::BatchEq(200),
                challenges: eq_outer_challenges.clone(),
            },
            InputBinding::Provided {
                poly: p.field_rd_value,
            },
            InputBinding::Provided {
                poly: p.field_rs1_value,
            },
            InputBinding::Provided {
                poly: p.field_rs2_value,
            },
        ],
        num_rounds: params.log_t,
        instance_config: None,
    });

    // BatchedSumcheckDef with real kernel references.
    batched_sumchecks.push(BatchedSumcheckDef {
        instances: vec![
            BatchedInstance {
                phases: vec![InstancePhase {
                    kernel: shift_kernel_idx,
                    num_rounds: params.log_t,
                    scalar_captures: vec![],
                    segmented: None,
                    carry_bindings: vec![],
                    pre_activation_ops: vec![],
                }],
                batch_coeff: ch_batch_base,
                first_active_round: 0,
            },
            BatchedInstance {
                phases: vec![InstancePhase {
                    kernel: inst_input_kernel_idx,
                    num_rounds: params.log_t,
                    scalar_captures: vec![],
                    segmented: None,
                    carry_bindings: vec![],
                    pre_activation_ops: vec![],
                }],
                batch_coeff: ChallengeIdx(ch_batch_base.0 + 1),
                first_active_round: 0,
            },
            BatchedInstance {
                phases: vec![InstancePhase {
                    kernel: registers_kernel_idx,
                    num_rounds: params.log_t,
                    scalar_captures: vec![],
                    segmented: None,
                    carry_bindings: vec![],
                    pre_activation_ops: vec![],
                }],
                batch_coeff: ChallengeIdx(ch_batch_base.0 + 2),
                first_active_round: 0,
            },
            BatchedInstance {
                phases: vec![InstancePhase {
                    kernel: fr_cr_kernel_idx,
                    num_rounds: params.log_t,
                    scalar_captures: vec![],
                    segmented: None,
                    carry_bindings: vec![],
                    pre_activation_ops: vec![],
                }],
                batch_coeff: ChallengeIdx(ch_batch_base.0 + 3),
                first_active_round: 0,
            },
        ],
        input_claims: vec![
            shift_input_claim,
            inst_input_claim,
            registers_input_claim,
            fr_cr_input_claim,
        ],
        max_rounds: params.log_t,
        max_degree: 3,
    });

    // Unrolled batched sumcheck rounds (log_T rounds).
    let round_challenge_indices = emit_unrolled_batched_rounds(
        ops,
        kernels,
        batched_sumchecks,
        ch,
        batch_idx,
        params.log_t,
        |_| 4, // degree 3 → 4 coefficients
        VerifierStageIndex(2),
        "s3_r",
        1,
        None,
    );

    // Flush 16 unique evaluation opening claims.
    //
    // All 4 instances open at the same cycle point (all log_T rounds,
    // same challenges). Aliasing deduplicates shared polynomials:
    //   - UnexpandedPC: Shift aliases with InstructionInput
    //   - Rs1Value: InstructionInput aliases with RegistersCR
    //   - Rs2Value: InstructionInput aliases with RegistersCR
    //
    // FieldRegCR's three polys (FieldRs1/Rs2/RdValue) are unique — they
    // don't appear in any earlier Stage-3 instance.
    //
    // Order = Shift(5) + InstructionInput(7 new) + RegistersCR(1 new)
    //       + FieldRegCR(3 new) = 16 unique evals.
    let stage3_eval_polys = vec![
        // Shift (5)
        p.unexpanded_pc,
        p.pc,
        p.op_flags[7],  // VirtualInstruction
        p.op_flags[12], // IsFirstInSequence
        p.inst_flag_is_noop,
        // InstructionInput (7 new; UnexpandedPC aliased above)
        p.inst_flag_left_is_rs1,
        p.rs1_val,
        p.inst_flag_left_is_pc,
        // UnexpandedPC aliased with Shift
        p.inst_flag_right_is_rs2,
        p.rs2_val,
        p.inst_flag_right_is_imm,
        p.imm,
        // RegistersCR (1 new; Rs1Value, Rs2Value aliased above)
        p.rd_write_value,
        // FieldRegCR (3 new — FR operand columns are unique to this instance).
        p.field_rd_value,
        p.field_rs1_value,
        p.field_rs2_value,
    ];

    for &poly in &stage3_eval_polys {
        ops.push(Op::Evaluate {
            poly,
            mode: EvalMode::FinalBind,
        });
    }
    ops.push(Op::AbsorbEvals {
        polys: stage3_eval_polys,
        tag: DomainSeparator::OpeningClaim,
    });

    Stage3Challenges {
        round_challenges: round_challenge_indices,
    }
}

// Stage 4: RegistersReadWriteChecking + RamValCheck
//
// Two sumcheck instances batched together:
//   [0] RegistersReadWriteChecking — log_K + log_T = 14 rounds, degree 3
//   [1] RamValCheck — log_T = 7 rounds, degree 3, first_active_round = 7
//
// Pre-sumcheck transcript:
//   1. Squeeze γ_registers_rw
//   2. AppendDomainSeparator("ram_val_check_gamma")
//   3. Squeeze γ_ram_val_check

#[allow(unused_variables, clippy::too_many_arguments)]
fn build_stage4(
    p: &Polys,
    params: &ModuleParams,
    ops: &mut Vec<Op>,
    kernels: &mut Vec<KernelDef>,
    ch: &mut ChallengeTable,
    batched_sumchecks: &mut Vec<BatchedSumcheckDef>,
    s2: &Stage2Challenges,
    s3: &Stage3Challenges,
) -> Stage4Challenges {
    // γ_registers_rw (RegistersReadWriteCheckingParams::new → challenge_scalar)
    let ch_gamma_reg_rw = ch.add(
        "gamma_registers_rw",
        ChallengeSource::FiatShamir { after_stage: 3 },
    );
    ops.push(Op::Squeeze {
        challenge: ch_gamma_reg_rw,
    });

    // γ_fr_rw — BN254 Fr coprocessor Read-Write γ. Mirrors integer
    // gamma_registers_rw. Consumed by the FieldRegReadWriteChecking
    // instance [2] batched alongside RegistersRW + RamValCheck.
    let ch_gamma_fr_rw = ch.add(
        "gamma_fr_rw",
        ChallengeSource::FiatShamir { after_stage: 3 },
    );
    ops.push(Op::Squeeze {
        challenge: ch_gamma_fr_rw,
    });

    // Domain separator before RAM val check gamma
    ops.push(Op::AppendDomainSeparator {
        tag: DomainSeparator::RamValCheckGamma,
    });

    // γ_ram_val_check
    let ch_gamma_ram_vc = ch.add(
        "gamma_ram_val_check",
        ChallengeSource::FiatShamir { after_stage: 3 },
    );
    ops.push(Op::Squeeze {
        challenge: ch_gamma_ram_vc,
    });

    // Compute r_address for init_eval: the address portion of Stage 2's
    // RamRW opening point. Raw Stage 2 rounds [log_t..log_t+log_k_ram]
    // reversed to big-endian (matching Segments normalization).
    let r_address_challenges: Vec<ChallengeIdx> = s2.round_challenges
        [params.log_t..params.log_t + params.log_k_ram]
        .iter()
        .rev()
        .copied()
        .collect();

    // Evaluate Val_init MLE at r_address (preprocessed polynomial).
    // Stores result as state.evaluations[RamInit].
    ops.push(Op::EvaluatePreprocessed {
        source: PolynomialId::RamInit,
        at_challenges: r_address_challenges.clone(),
        store_as: PolynomialId::RamInit,
    });

    // Batched sumcheck definition (must exist before AbsorbInputClaim)
    let batch_idx = BatchIdx(batched_sumchecks.len());
    let log_k_reg = 7usize; // log2(REGISTER_COUNT=128)
    let reg_rw_rounds = log_k_reg + params.log_t; // 16 for log_t=9
    let ram_vc_rounds = params.log_t; // 9 for log_t=9
    let fr_rw_rounds = LOG_K_FR + params.log_t; // 4 + log_t — shorter than RegistersRW
    let stage4_max_rounds = reg_rw_rounds; // RegistersRW is the longest

    batched_sumchecks.push(BatchedSumcheckDef {
        instances: vec![
            // [0] RegistersRW — active from round 0
            BatchedInstance {
                phases: vec![],
                batch_coeff: ChallengeIdx(0),
                first_active_round: 0,
            },
            // [1] RamValCheck — active from round (max_rounds - log_t)
            BatchedInstance {
                phases: vec![],
                batch_coeff: ChallengeIdx(0), // placeholder, set below after squeezing
                first_active_round: stage4_max_rounds - ram_vc_rounds,
            },
            // [2] FieldRegReadWriteChecking — active from round
            // (max_rounds - (LOG_K_FR + log_t)) = log_k_reg - LOG_K_FR = 3.
            // Shorter than RegistersRW because FR uses 16 slots vs 128.
            BatchedInstance {
                phases: vec![],
                batch_coeff: ChallengeIdx(0), // placeholder, set below
                first_active_round: stage4_max_rounds - fr_rw_rounds,
            },
        ],
        input_claims: vec![], // populated below
        max_rounds: stage4_max_rounds,
        max_degree: 3,
    });

    // Input claims

    // [0] RegistersRW: rd_wv + γ * rs1_rv + γ² * rs2_rv
    let reg_rw_input_claim = ClaimFormula {
        terms: vec![
            ClaimTerm {
                coeff: 1,
                factors: vec![ClaimFactor::Eval(p.rd_write_value)],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_gamma_reg_rw),
                    ClaimFactor::Eval(p.rs1_val),
                ],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_gamma_reg_rw),
                    ClaimFactor::Challenge(ch_gamma_reg_rw),
                    ClaimFactor::Eval(p.rs2_val),
                ],
            },
        ],
    };
    ops.push(Op::AbsorbInputClaim {
        formula: reg_rw_input_claim.clone(),
        tag: DomainSeparator::SumcheckClaim,
        batch: batch_idx,
        instance: InstanceIdx(0),
        inactive_scale_bits: 0,
    });

    // [1] RamValCheck: (val_rw - init_eval) + γ * (val_final - init_eval)
    //                = val_rw + γ * val_final - (1 + γ) * init_eval
    let ram_vc_input_claim = ClaimFormula {
        terms: vec![
            ClaimTerm {
                coeff: 1,
                factors: vec![ClaimFactor::Eval(p.ram_val)],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_gamma_ram_vc),
                    ClaimFactor::Eval(p.ram_val_final),
                ],
            },
            ClaimTerm {
                coeff: -1,
                factors: vec![ClaimFactor::Eval(PolynomialId::RamInit)],
            },
            ClaimTerm {
                coeff: -1,
                factors: vec![
                    ClaimFactor::Challenge(ch_gamma_ram_vc),
                    ClaimFactor::Eval(PolynomialId::RamInit),
                ],
            },
        ],
    };
    ops.push(Op::AbsorbInputClaim {
        formula: ram_vc_input_claim.clone(),
        tag: DomainSeparator::SumcheckClaim,
        batch: batch_idx,
        instance: InstanceIdx(1),
        inactive_scale_bits: stage4_max_rounds - ram_vc_rounds,
    });

    // [2] FieldRegReadWriteChecking input claim: rebind of Stage 3 FR CR's
    // γ-batched claim, evaluated at the FR CR final sumcheck point. Same
    // three-term shape as integer RegistersRW's input claim.
    let fr_rw_input_claim = ClaimFormula {
        terms: vec![
            ClaimTerm {
                coeff: 1,
                factors: vec![ClaimFactor::Eval(p.field_rd_value)],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_gamma_fr_rw),
                    ClaimFactor::Eval(p.field_rs1_value),
                ],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_gamma_fr_rw),
                    ClaimFactor::Challenge(ch_gamma_fr_rw),
                    ClaimFactor::Eval(p.field_rs2_value),
                ],
            },
        ],
    };
    ops.push(Op::AbsorbInputClaim {
        formula: fr_rw_input_claim.clone(),
        tag: DomainSeparator::SumcheckClaim,
        batch: batch_idx,
        instance: InstanceIdx(2),
        inactive_scale_bits: stage4_max_rounds - fr_rw_rounds,
    });

    // Batching coefficients (3 instances)
    let ch_batch0 = ch.add(
        "stage4_batch0",
        ChallengeSource::FiatShamir { after_stage: 3 },
    );
    let ch_batch1 = ch.add(
        "stage4_batch1",
        ChallengeSource::FiatShamir { after_stage: 3 },
    );
    let ch_batch2 = ch.add(
        "stage4_batch2",
        ChallengeSource::FiatShamir { after_stage: 3 },
    );
    ops.push(Op::Squeeze {
        challenge: ch_batch0,
    });
    ops.push(Op::Squeeze {
        challenge: ch_batch1,
    });
    ops.push(Op::Squeeze {
        challenge: ch_batch2,
    });

    // Wire batching coefficients into the BatchedSumcheckDef
    batched_sumchecks[batch_idx.0].instances[0].batch_coeff = ch_batch0;
    batched_sumchecks[batch_idx.0].instances[1].batch_coeff = ch_batch1;
    batched_sumchecks[batch_idx.0].instances[2].batch_coeff = ch_batch2;

    // Eq table challenge points for RegistersRW.
    //
    // r_cycle comes from RegistersClaimReduction's opening point, which
    // is the Stage 3 sumcheck challenges reversed to big-endian.
    // (RegistersClaimReduction: normalize_opening_point = LE→BE reverse)
    let reg_rw_cycle_eq: Vec<ChallengeIdx> = s3.round_challenges.iter().rev().copied().collect();

    // Kernel definitions for RegistersRW (2-phase).
    let g = ch_gamma_reg_rw.0 as u32;

    // Scalar capture challenge slots for phase boundary.
    let ch_reg_eq_bound = ch.add("reg_eq_cycle_bound", ChallengeSource::External);
    let ch_reg_inc_bound = ch.add("reg_inc_bound", ChallengeSource::External);

    // Phase 1 kernel: cycle binding (segmented, log_T rounds, degree 3).
    //
    // Formula: eq * rd_wa * val + eq * rd_wa * inc
    //        + γ * eq * rs1_ra * val + γ² * eq * rs2_ra * val
    let reg_rw_p1_kernel_idx = kernels.len();
    kernels.push(KernelDef {
        spec: KernelSpec {
            formula: Formula::from_terms(vec![
                // eq * rd_wa * val
                ProductTerm {
                    coefficient: 1,
                    factors: vec![Factor::Input(0), Factor::Input(3), Factor::Input(4)],
                },
                // eq * rd_wa * inc
                ProductTerm {
                    coefficient: 1,
                    factors: vec![Factor::Input(0), Factor::Input(3), Factor::Input(5)],
                },
                // γ * eq * rs1_ra * val
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(g),
                        Factor::Input(0),
                        Factor::Input(1),
                        Factor::Input(4),
                    ],
                },
                // γ² * eq * rs2_ra * val
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(g),
                        Factor::Challenge(g),
                        Factor::Input(0),
                        Factor::Input(2),
                        Factor::Input(4),
                    ],
                },
            ]),
            num_evals: 4, // degree 3
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
            gruen_hint: None,
        },
        inputs: vec![
            InputBinding::EqTable {
                poly: PolynomialId::BatchEq(9),
                challenges: reg_rw_cycle_eq.clone(),
            },
            InputBinding::Provided { poly: p.reg_ra_rs1 },
            InputBinding::Provided { poly: p.reg_ra_rs2 },
            InputBinding::Provided { poly: p.reg_wa },
            InputBinding::Provided { poly: p.reg_val },
            InputBinding::Provided { poly: p.rd_inc },
        ],
        num_rounds: params.log_t,
        instance_config: None,
    });

    // Phase 2 kernel: address binding (dense, log_K rounds, degree 2).
    //
    // After phase 1, eq_cycle and inc are fully bound (scalars via capture).
    // rs1_ra, rs2_ra, rd_wa, val are K-element.
    // Formula: ch_eq * rd_wa * val + ch_eq * ch_inc * rd_wa
    //        + γ * ch_eq * rs1_ra * val + γ² * ch_eq * rs2_ra * val
    let reg_rw_p2_kernel_idx = kernels.len();
    kernels.push(KernelDef {
        spec: KernelSpec {
            formula: Formula::from_terms(vec![
                // ch_eq * rd_wa * val
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(ch_reg_eq_bound.0 as u32),
                        Factor::Input(0),
                        Factor::Input(1),
                    ],
                },
                // ch_eq * ch_inc * rd_wa
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(ch_reg_eq_bound.0 as u32),
                        Factor::Challenge(ch_reg_inc_bound.0 as u32),
                        Factor::Input(0),
                    ],
                },
                // γ * ch_eq * rs1_ra * val
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(g),
                        Factor::Challenge(ch_reg_eq_bound.0 as u32),
                        Factor::Input(2),
                        Factor::Input(1),
                    ],
                },
                // γ² * ch_eq * rs2_ra * val
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(g),
                        Factor::Challenge(g),
                        Factor::Challenge(ch_reg_eq_bound.0 as u32),
                        Factor::Input(3),
                        Factor::Input(1),
                    ],
                },
            ]),
            num_evals: 3, // degree 2
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
            gruen_hint: None,
        },
        inputs: vec![
            InputBinding::Provided { poly: p.reg_wa },
            InputBinding::Provided { poly: p.reg_val },
            InputBinding::Provided { poly: p.reg_ra_rs1 },
            InputBinding::Provided { poly: p.reg_ra_rs2 },
        ],
        num_rounds: log_k_reg,
        instance_config: None,
    });

    // Kernel definition for RamValCheck (single phase, log_T rounds, degree 3).
    //
    // Formula: inc * wa * lt + γ * inc * wa
    let g_vc = ch_gamma_ram_vc.0 as u32;

    // r_cycle for the LT polynomial: Stage 2 RamRW cycle portion (big-endian).
    let rw_cycle_challenges: Vec<ChallengeIdx> = s2.round_challenges[0..params.log_t]
        .iter()
        .rev()
        .copied()
        .collect();

    let ram_vc_kernel_idx = kernels.len();
    kernels.push(KernelDef {
        spec: KernelSpec {
            formula: Formula::from_terms(vec![
                // inc * wa * lt
                ProductTerm {
                    coefficient: 1,
                    factors: vec![Factor::Input(0), Factor::Input(1), Factor::Input(2)],
                },
                // γ * inc * wa
                ProductTerm {
                    coefficient: 1,
                    factors: vec![Factor::Challenge(g_vc), Factor::Input(0), Factor::Input(1)],
                },
            ]),
            num_evals: 4, // degree 3
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
            gruen_hint: None,
        },
        inputs: vec![
            InputBinding::Provided { poly: p.ram_inc },
            InputBinding::EqProject {
                poly: PolynomialId::BatchEq(10),
                source: p.ram_ra_indicator,
                challenges: r_address_challenges.clone(),
                inner_size: 1 << params.log_t,
                outer_size: 1 << params.log_k_ram,
            },
            InputBinding::LtTable {
                poly: PolynomialId::BatchEq(11),
                challenges: rw_cycle_challenges,
            },
        ],
        num_rounds: ram_vc_rounds,
        instance_config: None,
    });

    // FieldRegReadWriteChecking 2-phase kernels. Structural mirror of the
    // integer RegistersRW with LOG_K_REG → LOG_K_FR = 4 address rounds.
    // r_cycle for FR RW comes from the Stage 3 FieldRegCR final point,
    // which is the same cycle challenges as integer RegistersCR (all
    // Stage-3 instances share the same log_T rounds → same r_cycle).
    let fr_rw_cycle_eq: Vec<ChallengeIdx> = s3.round_challenges.iter().rev().copied().collect();
    let g_fr_rw = ch_gamma_fr_rw.0 as u32;

    // Scalar-capture challenge slots for the phase-1 → phase-2 boundary.
    let ch_fr_eq_bound = ch.add("fr_eq_cycle_bound", ChallengeSource::External);
    let ch_fr_inc_bound = ch.add("fr_inc_bound", ChallengeSource::External);

    // FR Phase 1 kernel: cycle binding (segmented, log_T rounds, degree 3).
    //
    // Formula: eq * field_reg_wa * field_reg_val
    //        + eq * field_reg_wa * field_reg_inc
    //        + γ * eq * field_reg_ra_rs1 * field_reg_val
    //        + γ² * eq * field_reg_ra_rs2 * field_reg_val
    let fr_rw_p1_kernel_idx = kernels.len();
    kernels.push(KernelDef {
        spec: KernelSpec {
            formula: Formula::from_terms(vec![
                // eq * wa * val
                ProductTerm {
                    coefficient: 1,
                    factors: vec![Factor::Input(0), Factor::Input(3), Factor::Input(4)],
                },
                // eq * wa * inc
                ProductTerm {
                    coefficient: 1,
                    factors: vec![Factor::Input(0), Factor::Input(3), Factor::Input(5)],
                },
                // γ * eq * rs1_ra * val
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(g_fr_rw),
                        Factor::Input(0),
                        Factor::Input(1),
                        Factor::Input(4),
                    ],
                },
                // γ² * eq * rs2_ra * val
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(g_fr_rw),
                        Factor::Challenge(g_fr_rw),
                        Factor::Input(0),
                        Factor::Input(2),
                        Factor::Input(4),
                    ],
                },
            ]),
            num_evals: 4, // degree 3
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
            gruen_hint: None,
        },
        inputs: vec![
            // BatchEq(201): fresh slot for FR RW cycle-eq (see 4c notes on
            // BatchEq(200..=203) reservation block).
            InputBinding::EqTable {
                poly: PolynomialId::BatchEq(201),
                challenges: fr_rw_cycle_eq.clone(),
            },
            InputBinding::Provided {
                poly: p.field_reg_ra_rs1,
            },
            InputBinding::Provided {
                poly: p.field_reg_ra_rs2,
            },
            InputBinding::Provided {
                poly: p.field_reg_wa,
            },
            InputBinding::Provided {
                poly: p.field_reg_val,
            },
            InputBinding::Provided {
                poly: p.field_reg_inc,
            },
        ],
        num_rounds: params.log_t,
        instance_config: None,
    });

    // FR Phase 2 kernel: address binding (dense, LOG_K_FR=4 rounds, degree 2).
    //
    // Formula: ch_eq * wa * val + ch_eq * ch_inc * wa
    //        + γ * ch_eq * rs1_ra * val + γ² * ch_eq * rs2_ra * val
    let fr_rw_p2_kernel_idx = kernels.len();
    kernels.push(KernelDef {
        spec: KernelSpec {
            formula: Formula::from_terms(vec![
                // ch_eq * wa * val
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(ch_fr_eq_bound.0 as u32),
                        Factor::Input(0),
                        Factor::Input(1),
                    ],
                },
                // ch_eq * ch_inc * wa
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(ch_fr_eq_bound.0 as u32),
                        Factor::Challenge(ch_fr_inc_bound.0 as u32),
                        Factor::Input(0),
                    ],
                },
                // γ * ch_eq * rs1_ra * val
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(g_fr_rw),
                        Factor::Challenge(ch_fr_eq_bound.0 as u32),
                        Factor::Input(2),
                        Factor::Input(1),
                    ],
                },
                // γ² * ch_eq * rs2_ra * val
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(g_fr_rw),
                        Factor::Challenge(g_fr_rw),
                        Factor::Challenge(ch_fr_eq_bound.0 as u32),
                        Factor::Input(3),
                        Factor::Input(1),
                    ],
                },
            ]),
            num_evals: 3, // degree 2
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
            gruen_hint: None,
        },
        inputs: vec![
            InputBinding::Provided {
                poly: p.field_reg_wa,
            },
            InputBinding::Provided {
                poly: p.field_reg_val,
            },
            InputBinding::Provided {
                poly: p.field_reg_ra_rs1,
            },
            InputBinding::Provided {
                poly: p.field_reg_ra_rs2,
            },
        ],
        num_rounds: LOG_K_FR,
        instance_config: None,
    });

    // Update BatchedSumcheckDef with kernel phases.
    batched_sumchecks[batch_idx.0].instances[0] = BatchedInstance {
        phases: vec![
            InstancePhase {
                kernel: reg_rw_p1_kernel_idx,
                num_rounds: params.log_t,
                scalar_captures: vec![],
                segmented: Some(SegmentedConfig {
                    inner_num_vars: params.log_t,
                    outer_num_vars: log_k_reg,
                    // [eq: inner, rs1_ra: mixed, rs2_ra: mixed, rd_wa: mixed, val: mixed, inc: inner]
                    inner_only: vec![true, false, false, false, false, true],
                    outer_eq_challenges: vec![],
                }),
                carry_bindings: vec![],
                pre_activation_ops: vec![],
            },
            InstancePhase {
                kernel: reg_rw_p2_kernel_idx,
                num_rounds: log_k_reg,
                scalar_captures: vec![
                    ScalarCapture {
                        poly: PolynomialId::BatchEq(9),
                        challenge: ch_reg_eq_bound,
                    },
                    ScalarCapture {
                        poly: p.rd_inc,
                        challenge: ch_reg_inc_bound,
                    },
                ],
                segmented: None,
                carry_bindings: vec![],
                pre_activation_ops: vec![],
            },
        ],
        batch_coeff: ch_batch0,
        first_active_round: 0,
    };
    batched_sumchecks[batch_idx.0].instances[1] = BatchedInstance {
        phases: vec![InstancePhase {
            kernel: ram_vc_kernel_idx,
            num_rounds: ram_vc_rounds,
            scalar_captures: vec![],
            segmented: None,
            carry_bindings: vec![],
            pre_activation_ops: vec![],
        }],
        batch_coeff: ch_batch1,
        first_active_round: stage4_max_rounds - ram_vc_rounds,
    };
    // [2] FieldRegReadWriteChecking — 2-phase (log_T cycle + LOG_K_FR address).
    batched_sumchecks[batch_idx.0].instances[2] = BatchedInstance {
        phases: vec![
            InstancePhase {
                kernel: fr_rw_p1_kernel_idx,
                num_rounds: params.log_t,
                scalar_captures: vec![],
                segmented: Some(SegmentedConfig {
                    inner_num_vars: params.log_t,
                    outer_num_vars: LOG_K_FR,
                    // [eq: inner, rs1_ra: mixed, rs2_ra: mixed, wa: mixed,
                    //  val: mixed, inc: inner]
                    inner_only: vec![true, false, false, false, false, true],
                    outer_eq_challenges: vec![],
                }),
                carry_bindings: vec![],
                pre_activation_ops: vec![],
            },
            InstancePhase {
                kernel: fr_rw_p2_kernel_idx,
                num_rounds: LOG_K_FR,
                scalar_captures: vec![
                    ScalarCapture {
                        poly: PolynomialId::BatchEq(201),
                        challenge: ch_fr_eq_bound,
                    },
                    ScalarCapture {
                        poly: p.field_reg_inc,
                        challenge: ch_fr_inc_bound,
                    },
                ],
                segmented: None,
                carry_bindings: vec![],
                pre_activation_ops: vec![],
            },
        ],
        batch_coeff: ch_batch2,
        first_active_round: stage4_max_rounds - fr_rw_rounds,
    };
    batched_sumchecks[batch_idx.0].input_claims =
        vec![reg_rw_input_claim, ram_vc_input_claim, fr_rw_input_claim];

    // Unrolled batched sumcheck rounds (stage4_max_rounds rounds).
    let round_challenge_indices = emit_unrolled_batched_rounds(
        ops,
        kernels,
        batched_sumchecks,
        ch,
        batch_idx,
        stage4_max_rounds,
        |_| 4, // degree 3 → 4 coefficients
        VerifierStageIndex(3),
        "s4_r",
        1,
        None,
    );

    // Eval flush: open polynomials at the final sumcheck point.
    //
    // Stage-4 opens 12 evals in order:
    //   RegistersRW: RegistersVal, Rs1Ra, Rs2Ra, RdWa, RdInc
    //   RamValCheck: RamRa (wa projection), RamInc
    //   FieldRegRW: FieldRegVal, FieldRegRaRs1, FieldRegRaRs2, FieldRegWa,
    //               FieldRegInc
    let ram_ra_wa = PolynomialId::BatchEq(10); // wa EqProject buffer from RamValCheck
    let stage4_eval_polys = vec![
        p.reg_val,
        p.reg_ra_rs1,
        p.reg_ra_rs2,
        p.reg_wa,
        p.rd_inc,
        ram_ra_wa,
        p.ram_inc,
        p.field_reg_val,
        p.field_reg_ra_rs1,
        p.field_reg_ra_rs2,
        p.field_reg_wa,
        p.field_reg_inc,
    ];
    for &poly in &stage4_eval_polys {
        ops.push(Op::Evaluate {
            poly,
            mode: EvalMode::FinalBind,
        });
    }
    ops.push(Op::AbsorbEvals {
        polys: stage4_eval_polys,
        tag: DomainSeparator::OpeningClaim,
    });

    Stage4Challenges {
        round_challenges: round_challenge_indices,
    }
}

// Stage 5: InstructionReadRaf + RamRaClaimReduction + RegistersValEval
//
// Three batched sumcheck instances:
//   [0] InstructionReadRaf — LOG_K_INSTRUCTION + log_T = 128+9 = 137 rounds
//   [1] RamRaClaimReduction — log_T = 9 rounds, first_active=128
//   [2] RegistersValEvaluation — log_T = 9 rounds, first_active=128
#[allow(clippy::too_many_arguments)]
fn build_stage5(
    p: &Polys,
    params: &ModuleParams,
    ops: &mut Vec<Op>,
    kernels: &mut Vec<KernelDef>,
    ch: &mut ChallengeTable,
    batched_sumchecks: &mut Vec<BatchedSumcheckDef>,
    s2: &Stage2Challenges,
    s4: &Stage4Challenges,
) -> Stage5Challenges {
    let inst_read_raf_rounds = LOG_K_INSTRUCTION + params.log_t; // 137 for muldiv
    let ram_ra_reduction_rounds = params.log_t; // 9
    let reg_val_eval_rounds = params.log_t; // 9
    let fr_val_eval_rounds = params.log_t; // log_T — same as RegistersValEvaluation
    let stage5_max_rounds = inst_read_raf_rounds; // 137

    // Pre-sumcheck challenge squeezes.
    //
    // InstructionReadRafSumcheckParams::new → challenge_scalar → gamma
    // RaReductionParams::new → challenge_scalar → gamma
    // RegistersValEvaluationSumcheckParams::new → no transcript ops
    let ch_gamma_inst_raf = ch.add(
        "gamma_instruction_read_raf",
        ChallengeSource::FiatShamir { after_stage: 4 },
    );
    ops.push(Op::Squeeze {
        challenge: ch_gamma_inst_raf,
    });

    let ch_gamma_ram_ra = ch.add(
        "gamma_ram_ra_reduction",
        ChallengeSource::FiatShamir { after_stage: 4 },
    );
    ops.push(Op::Squeeze {
        challenge: ch_gamma_ram_ra,
    });

    // Challenge index vectors for downstream kernels.
    //
    // RegistersValEvaluation uses r_address/r_cycle from Stage 4's
    // RegistersRW sumcheck. RamRaClaimReduction uses r_address from
    // Stage 2's RamRW, plus three separate r_cycle points:
    //   r_cycle_raf — Stage 1 cycle point (used in EqProject for RAF)
    //   r_cycle_rw  — Stage 2 RamRW cycle phase
    //   r_cycle_val — Stage 4 RamValCheck round challenges
    // All reversed to BIG_ENDIAN (MSB-first) for eq table construction.
    let log_k_reg = LOG_K_REG;

    // RegistersValEvaluation: from Stage 4 RegistersRW opening point
    let reg_val_r_address: Vec<ChallengeIdx> = s4.round_challenges
        [params.log_t..params.log_t + log_k_reg]
        .iter()
        .rev()
        .copied()
        .collect();
    let reg_val_r_cycle: Vec<ChallengeIdx> = s4.round_challenges[0..params.log_t]
        .iter()
        .rev()
        .copied()
        .collect();

    // FieldRegValEvaluation: from Stage 4 FieldRegReadWriteChecking opening
    // point. Stage 4's max_rounds = log_k_reg + log_t (RegistersRW is the
    // longest); FR RW starts at `first_active = log_k_reg - LOG_K_FR = 3`
    // and runs `LOG_K_FR + log_t` rounds. Split into cycle (first log_t) +
    // address (last LOG_K_FR), both reversed to BIG_ENDIAN.
    let fr_rw_first_active = log_k_reg - LOG_K_FR;
    let fr_val_r_cycle: Vec<ChallengeIdx> = s4.round_challenges
        [fr_rw_first_active..fr_rw_first_active + params.log_t]
        .iter()
        .rev()
        .copied()
        .collect();
    let fr_val_r_address: Vec<ChallengeIdx> = s4.round_challenges
        [fr_rw_first_active + params.log_t..fr_rw_first_active + params.log_t + LOG_K_FR]
        .iter()
        .rev()
        .copied()
        .collect();

    // RamRaClaimReduction: unified r_address from Stage 2 RamRW address phase
    let ram_ra_r_address: Vec<ChallengeIdx> = s2.round_challenges
        [params.log_t..params.log_t + params.log_k_ram]
        .iter()
        .rev()
        .copied()
        .collect();
    // Three r_cycle points for the three eq tables
    let ram_ra_r_cycle_raf: Vec<ChallengeIdx> = s2.stage1_cycle.clone(); // already BIG_ENDIAN
    let ram_ra_r_cycle_rw: Vec<ChallengeIdx> = s2.round_challenges[0..params.log_t]
        .iter()
        .rev()
        .copied()
        .collect();
    let ram_ra_r_cycle_val: Vec<ChallengeIdx> = s4.round_challenges[log_k_reg..]
        .iter()
        .rev()
        .copied()
        .collect();

    // Kernel: RegistersValEvaluation — degree 3, log_T rounds
    //
    // Formula: inc(j) × wa(j) × lt(j)
    //   inc = committed RdInc
    //   wa  = eq(r_address, rd[j]) via EqGather
    //   lt  = LT(r_cycle, j) strict less-than polynomial
    let reg_val_kernel_idx = kernels.len();
    kernels.push(KernelDef {
        spec: KernelSpec {
            formula: Formula::from_terms(vec![ProductTerm {
                coefficient: 1,
                factors: vec![Factor::Input(0), Factor::Input(1), Factor::Input(2)],
            }]),
            num_evals: 4, // degree 3
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
            gruen_hint: None,
        },
        inputs: vec![
            InputBinding::Provided { poly: p.rd_inc },
            InputBinding::EqGather {
                poly: PolynomialId::BatchEq(12),
                eq_challenges: reg_val_r_address,
                indices: PolynomialId::RdGatherIndex,
            },
            InputBinding::LtTable {
                poly: PolynomialId::BatchEq(13),
                challenges: reg_val_r_cycle,
            },
        ],
        num_rounds: reg_val_eval_rounds,
        instance_config: None,
    });

    // Kernel: FieldRegValEvaluation — degree 3, log_T rounds.
    // Structural mirror of RegistersValEvaluation with RdInc → FieldRegInc,
    // RdGatherIndex → FrdGatherIndex, LOG_K_REG → LOG_K_FR.
    //
    // Formula: field_reg_inc(j) × eq_gather(r_addr_fr, frd[j]) × LT(r_cycle_fr, j)
    let fr_val_kernel_idx = kernels.len();
    kernels.push(KernelDef {
        spec: KernelSpec {
            formula: Formula::from_terms(vec![ProductTerm {
                coefficient: 1,
                factors: vec![Factor::Input(0), Factor::Input(1), Factor::Input(2)],
            }]),
            num_evals: 4, // degree 3
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
            gruen_hint: None,
        },
        inputs: vec![
            InputBinding::Provided {
                poly: p.field_reg_inc,
            },
            // BatchEq(202): FR val eval gather (reserved block).
            InputBinding::EqGather {
                poly: PolynomialId::BatchEq(202),
                eq_challenges: fr_val_r_address,
                indices: PolynomialId::FrdGatherIndex,
            },
            // BatchEq(203): FR val eval LT (reserved block).
            InputBinding::LtTable {
                poly: PolynomialId::BatchEq(203),
                challenges: fr_val_r_cycle,
            },
        ],
        num_rounds: fr_val_eval_rounds,
        instance_config: None,
    });

    // Kernel: RamRaClaimReduction — degree 2, log_T rounds
    //
    // Formula: eq_raf × ra + γ × eq_rw × ra + γ² × eq_val × ra
    //   eq_raf = eq(r_cycle_raf, j)  (Stage 1 cycle point)
    //   eq_rw  = eq(r_cycle_rw,  j)  (Stage 2 RamRW cycle)
    //   eq_val = eq(r_cycle_val, j)  (Stage 4 RamValCheck cycle)
    //   ra     = eq(r_address, addr[j]) via EqGather (RAM gather index)
    let g_ram_ra = ch_gamma_ram_ra.0 as u32;
    let ram_ra_kernel_idx = kernels.len();
    kernels.push(KernelDef {
        spec: KernelSpec {
            formula: Formula::from_terms(vec![
                ProductTerm {
                    coefficient: 1,
                    factors: vec![Factor::Input(0), Factor::Input(3)],
                },
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(g_ram_ra),
                        Factor::Input(1),
                        Factor::Input(3),
                    ],
                },
                ProductTerm {
                    coefficient: 1,
                    factors: vec![
                        Factor::Challenge(g_ram_ra),
                        Factor::Challenge(g_ram_ra),
                        Factor::Input(2),
                        Factor::Input(3),
                    ],
                },
            ]),
            num_evals: 3, // degree 2
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
            gruen_hint: None,
        },
        inputs: vec![
            InputBinding::EqTable {
                poly: PolynomialId::BatchEq(14),
                challenges: ram_ra_r_cycle_raf,
            },
            InputBinding::EqTable {
                poly: PolynomialId::BatchEq(15),
                challenges: ram_ra_r_cycle_rw,
            },
            InputBinding::EqTable {
                poly: PolynomialId::BatchEq(16),
                challenges: ram_ra_r_cycle_val,
            },
            InputBinding::EqGather {
                poly: PolynomialId::BatchEq(17),
                eq_challenges: ram_ra_r_address,
                indices: PolynomialId::RamGatherIndex,
            },
        ],
        num_rounds: ram_ra_reduction_rounds,
        instance_config: None,
    });

    // Kernel: InstructionReadRaf — 2-phase (address + cycle)
    //
    // Phase 1 (address, LOG_K rounds, degree 2):
    //   PrefixSuffix iteration decomposes the 128-bit address sum into
    //   multi-phase prefix/suffix MLE evaluation. The runtime handles
    //   the expanding tables, suffix recomputation, and phase transitions.
    //
    // Phase 2 (cycle, log_T rounds, degree n_virtual_ra_polys + 2):
    //   Dense product: eq(r_reduction, j) × combined_val(j) × Π ra_i(j)
    //   Inputs are materialized by the address phase at the transition.

    // r_reduction: InstructionClaimReduction opening point (BIG_ENDIAN)
    let inst_cr_first_active = params.stage2_max_rounds - params.instruction_claim_reduction_rounds;
    let r_reduction_challenges: Vec<ChallengeIdx> = s2.round_challenges
        [inst_cr_first_active..inst_cr_first_active + params.log_t]
        .iter()
        .rev()
        .copied()
        .collect();

    let n_vra = params.n_virtual_ra_polys;
    let output_ra_polys: Vec<PolynomialId> = (0..n_vra).map(PolynomialId::InstructionRa).collect();

    // Registry checkpoint slots (p_right, p_left, p_identity).
    let registry_checkpoint_slots: [ChallengeIdx; 3] = [
        ch.add("ps_registry_p_right", ChallengeSource::External),
        ch.add("ps_registry_p_left", ChallengeSource::External),
        ch.add("ps_registry_p_identity", ChallengeSource::External),
    ];

    // Compute prefix-suffix protocol data from jolt-instructions.
    let (
        ps_combine_entries,
        ps_suffix_at_empty,
        ps_suffixes_per_table,
        ps_suffix_ops,
        ps_checkpoint_rules,
        ps_prefix_mle_rules,
    ) = compute_prefix_suffix_data();

    // Phase 1: address rounds (PrefixSuffix iteration)
    let inst_raf_addr_kernel_idx = kernels.len();
    let ps_prefix_lowered = jolt_compiler::prefix_mle_lowering::build_prefix_lowered_rounds(
        &ps_prefix_mle_rules,
        params.instruction_chunk_bits,
        params.instruction_phases,
    );
    kernels.push(KernelDef {
        spec: KernelSpec {
            formula: Formula::from_terms(vec![ProductTerm {
                coefficient: 1,
                factors: vec![Factor::Input(0)],
            }]),
            num_evals: 3, // degree 2 during address rounds
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
            gruen_hint: None,
        },
        inputs: vec![],
        num_rounds: LOG_K_INSTRUCTION,
        instance_config: Some(InstanceConfig {
            kernel: inst_raf_addr_kernel_idx,
            total_address_bits: LOG_K_INSTRUCTION,
            chunk_bits: params.instruction_chunk_bits,
            num_phases: params.instruction_phases,
            ra_virtual_log_k_chunk: params.ra_virtual_log_k_chunk,
            gamma: ch_gamma_inst_raf,
            r_reduction: r_reduction_challenges.clone(),
            output_ra_polys: output_ra_polys.clone(),
            output_combined_val: PolynomialId::InstructionCombinedVal,
            num_tables: NUM_LOOKUP_TABLES,
            suffixes_per_table: ps_suffixes_per_table,
            combine_entries: ps_combine_entries,
            suffix_at_empty: ps_suffix_at_empty,
            suffix_ops: ps_suffix_ops,
            num_prefixes: NUM_PREFIXES,
            checkpoint_rules: ps_checkpoint_rules,
            prefix_mle_rules: ps_prefix_mle_rules,
            prefix_lowered: ps_prefix_lowered,
            registry_checkpoint_slots,
        }),
    });

    // Phase 2: cycle rounds (Dense product)
    let inst_raf_cycle_kernel_idx = kernels.len();
    {
        let cycle_factors: Vec<Factor> = (0..n_vra + 2).map(|i| Factor::Input(i as u32)).collect();
        let mut cycle_inputs = vec![
            InputBinding::EqTable {
                poly: PolynomialId::BatchEq(18),
                challenges: r_reduction_challenges,
            },
            InputBinding::Provided {
                poly: PolynomialId::InstructionCombinedVal,
            },
        ];
        for &ra_id in &output_ra_polys {
            cycle_inputs.push(InputBinding::Provided { poly: ra_id });
        }
        kernels.push(KernelDef {
            spec: KernelSpec {
                formula: Formula::from_terms(vec![ProductTerm {
                    coefficient: 1,
                    factors: cycle_factors,
                }]),
                num_evals: n_vra + 3, // degree = n_vra + 2
                iteration: Iteration::Dense,
                binding_order: BindingOrder::LowToHigh,
                gruen_hint: None,
            },
            inputs: cycle_inputs,
            num_rounds: params.log_t,
            instance_config: None,
        });
    }

    // Batched sumcheck definition
    let batch_idx = BatchIdx(batched_sumchecks.len());
    let inst_raf_cycle_degree = n_vra + 2;

    batched_sumchecks.push(BatchedSumcheckDef {
        instances: vec![
            // [0] InstructionReadRaf — active from round 0, 2-phase
            BatchedInstance {
                phases: vec![
                    InstancePhase {
                        kernel: inst_raf_addr_kernel_idx,
                        num_rounds: LOG_K_INSTRUCTION,
                        scalar_captures: vec![],
                        segmented: None,
                        carry_bindings: vec![],
                        pre_activation_ops: vec![],
                    },
                    InstancePhase {
                        kernel: inst_raf_cycle_kernel_idx,
                        num_rounds: params.log_t,
                        scalar_captures: vec![],
                        segmented: None,
                        carry_bindings: vec![],
                        pre_activation_ops: vec![],
                    },
                ],
                batch_coeff: ChallengeIdx(0),
                first_active_round: 0,
            },
            // [1] RamRaClaimReduction — active from round LOG_K_INSTRUCTION
            BatchedInstance {
                phases: vec![InstancePhase {
                    kernel: ram_ra_kernel_idx,
                    num_rounds: ram_ra_reduction_rounds,
                    scalar_captures: vec![],
                    segmented: None,
                    carry_bindings: vec![],
                    pre_activation_ops: vec![],
                }],
                batch_coeff: ChallengeIdx(0),
                first_active_round: stage5_max_rounds - ram_ra_reduction_rounds,
            },
            // [2] RegistersValEvaluation — active from round LOG_K_INSTRUCTION
            BatchedInstance {
                phases: vec![InstancePhase {
                    kernel: reg_val_kernel_idx,
                    num_rounds: reg_val_eval_rounds,
                    scalar_captures: vec![],
                    segmented: None,
                    carry_bindings: vec![],
                    pre_activation_ops: vec![],
                }],
                batch_coeff: ChallengeIdx(0),
                first_active_round: stage5_max_rounds - reg_val_eval_rounds,
            },
            // [3] FieldRegValEvaluation — active from round LOG_K_INSTRUCTION
            BatchedInstance {
                phases: vec![InstancePhase {
                    kernel: fr_val_kernel_idx,
                    num_rounds: fr_val_eval_rounds,
                    scalar_captures: vec![],
                    segmented: None,
                    carry_bindings: vec![],
                    pre_activation_ops: vec![],
                }],
                batch_coeff: ChallengeIdx(0),
                first_active_round: stage5_max_rounds - fr_val_eval_rounds,
            },
        ],
        input_claims: vec![],
        max_rounds: stage5_max_rounds,
        max_degree: inst_raf_cycle_degree,
    });

    // Input claims

    // [0] InstructionReadRaf: rv_claim + γ * left_operand_claim + γ² * right_operand_claim
    let ram_ra_wa = PolynomialId::BatchEq(10);

    let inst_raf_input_claim = ClaimFormula {
        terms: vec![
            ClaimTerm {
                coeff: 1,
                factors: vec![ClaimFactor::Eval(p.lookup_output)],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_gamma_inst_raf),
                    ClaimFactor::Eval(p.left_lookup_operand),
                ],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_gamma_inst_raf),
                    ClaimFactor::Challenge(ch_gamma_inst_raf),
                    ClaimFactor::Eval(p.right_lookup_operand),
                ],
            },
        ],
    };
    ops.push(Op::AbsorbInputClaim {
        formula: inst_raf_input_claim.clone(),
        tag: DomainSeparator::SumcheckClaim,
        batch: batch_idx,
        instance: InstanceIdx(0),
        inactive_scale_bits: 0,
    });

    // [1] RamRaClaimReduction: claim_raf + γ * claim_rw + γ² * claim_val
    let ram_ra_input_claim = ClaimFormula {
        terms: vec![
            ClaimTerm {
                coeff: 1,
                factors: vec![ClaimFactor::Eval(p.ram_raf_ra)],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_gamma_ram_ra),
                    ClaimFactor::Eval(p.ram_combined_ra),
                ],
            },
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_gamma_ram_ra),
                    ClaimFactor::Challenge(ch_gamma_ram_ra),
                    ClaimFactor::Eval(ram_ra_wa),
                ],
            },
        ],
    };
    ops.push(Op::AbsorbInputClaim {
        formula: ram_ra_input_claim.clone(),
        tag: DomainSeparator::SumcheckClaim,
        batch: batch_idx,
        instance: InstanceIdx(1),
        inactive_scale_bits: stage5_max_rounds - ram_ra_reduction_rounds,
    });

    // [2] RegistersValEvaluation: RegistersVal claim (single term)
    let reg_val_input_claim = ClaimFormula {
        terms: vec![ClaimTerm {
            coeff: 1,
            factors: vec![ClaimFactor::Eval(p.reg_val)],
        }],
    };
    ops.push(Op::AbsorbInputClaim {
        formula: reg_val_input_claim.clone(),
        tag: DomainSeparator::SumcheckClaim,
        batch: batch_idx,
        instance: InstanceIdx(2),
        inactive_scale_bits: stage5_max_rounds - reg_val_eval_rounds,
    });

    // [3] FieldRegValEvaluation input claim: single term Eval(field_reg_val).
    // Mirrors integer RegistersValEvaluation's claim.
    let fr_val_input_claim = ClaimFormula {
        terms: vec![ClaimTerm {
            coeff: 1,
            factors: vec![ClaimFactor::Eval(p.field_reg_val)],
        }],
    };
    ops.push(Op::AbsorbInputClaim {
        formula: fr_val_input_claim.clone(),
        tag: DomainSeparator::SumcheckClaim,
        batch: batch_idx,
        instance: InstanceIdx(3),
        inactive_scale_bits: stage5_max_rounds - fr_val_eval_rounds,
    });

    // Batching coefficients (4 instances)
    let ch_batch0 = ch.add(
        "stage5_batch0",
        ChallengeSource::FiatShamir { after_stage: 4 },
    );
    let ch_batch1 = ch.add(
        "stage5_batch1",
        ChallengeSource::FiatShamir { after_stage: 4 },
    );
    let ch_batch2 = ch.add(
        "stage5_batch2",
        ChallengeSource::FiatShamir { after_stage: 4 },
    );
    let ch_batch3 = ch.add(
        "stage5_batch3",
        ChallengeSource::FiatShamir { after_stage: 4 },
    );
    ops.push(Op::Squeeze {
        challenge: ch_batch0,
    });
    ops.push(Op::Squeeze {
        challenge: ch_batch1,
    });
    ops.push(Op::Squeeze {
        challenge: ch_batch2,
    });
    ops.push(Op::Squeeze {
        challenge: ch_batch3,
    });

    batched_sumchecks[batch_idx.0].instances[0].batch_coeff = ch_batch0;
    batched_sumchecks[batch_idx.0].instances[1].batch_coeff = ch_batch1;
    batched_sumchecks[batch_idx.0].instances[2].batch_coeff = ch_batch2;
    batched_sumchecks[batch_idx.0].instances[3].batch_coeff = ch_batch3;
    batched_sumchecks[batch_idx.0].input_claims = vec![
        inst_raf_input_claim,
        ram_ra_input_claim,
        reg_val_input_claim,
        fr_val_input_claim,
    ];

    // Sumcheck rounds: 137 rounds (128 address + 9 cycle)
    //
    // Address rounds (0..LOG_K_INSTRUCTION-1):
    //   Only InstructionReadRaf is active. The other two instances are
    //   inactive and contribute constant claim/2 to each eval slot.
    //   InstructionReadRaf produces a degree-2 polynomial per round
    //   (prefix-suffix decomposition).
    //
    // Cycle rounds (LOG_K_INSTRUCTION..LOG_K_INSTRUCTION+log_T-1):
    //   All three instances are active.
    //   InstructionReadRaf produces a degree-(instruction_d+2) polynomial
    //   (product of 1 eq + instruction_d ra_polys + 1 combined_val).
    //   RamRaClaimReduction produces degree-2.
    //   RegistersValEvaluation produces degree-3.
    //   Combined degree = instruction_d + 2.
    let inst_raf_addr_degree = 2;
    let round_challenge_indices = emit_unrolled_batched_rounds(
        ops,
        kernels,
        batched_sumchecks,
        ch,
        batch_idx,
        stage5_max_rounds,
        |round| {
            if round < LOG_K_INSTRUCTION {
                inst_raf_addr_degree + 1 // 3
            } else {
                inst_raf_cycle_degree + 1 // instruction_d + 3
            }
        },
        VerifierStageIndex(4),
        "s5_r",
        1,
        None,
    );

    // Post-sumcheck eval flush.
    //
    // After all rounds, each instance caches opening claims. The
    // opening accumulator flushes them to the transcript.
    //
    // Order: InstructionReadRaf, RamRaClaimReduction, RegistersValEvaluation
    //
    // LookupTableFlag and InstructionRafFlag are virtual polynomials not
    // used as kernel inputs during the sumcheck. We bind them at the
    // cycle challenges to evaluate at r_cycle.

    // Bind flag polys at cycle challenges (LowToHigh, same as cycle sumcheck).
    let flag_polys: Vec<PolynomialId> = (0..NUM_LOOKUP_TABLES)
        .map(PolynomialId::LookupTableFlag)
        .chain(std::iter::once(PolynomialId::InstructionRafFlag))
        .collect();
    for cycle_round in 0..params.log_t {
        ops.push(Op::Bind {
            polys: flag_polys.clone(),
            challenge: round_challenge_indices[LOG_K_INSTRUCTION + cycle_round],
            order: BindingOrder::LowToHigh,
        });
    }

    // Evaluate + absorb all opening claims in cache_openings order.
    let ram_ra_gather = PolynomialId::BatchEq(17);
    let rd_wa_gather = PolynomialId::BatchEq(12);
    let mut stage5_eval_polys = Vec::new();

    // [0] InstructionReadRaf: 41 LookupTableFlag + n_vra InstructionRa + 1 RafFlag
    for i in 0..NUM_LOOKUP_TABLES {
        stage5_eval_polys.push(PolynomialId::LookupTableFlag(i));
    }
    for i in 0..n_vra {
        stage5_eval_polys.push(PolynomialId::InstructionRa(i));
    }
    stage5_eval_polys.push(PolynomialId::InstructionRafFlag);

    // [1] RamRaClaimReduction: 1 RamRa gather
    stage5_eval_polys.push(ram_ra_gather);

    // [2] RegistersValEvaluation: RdInc + RdWa
    stage5_eval_polys.push(p.rd_inc);
    stage5_eval_polys.push(rd_wa_gather);

    // [3] FieldRegValEvaluation: FieldRegInc + FieldRegWa (gather).
    // FieldRegInc is a unique commit (not aliased with RdInc/RamInc).
    // FieldRegWa gather = BatchEq(202) (the gather buffer our kernel
    // populates when it fires).
    let fr_wa_gather = PolynomialId::BatchEq(202);
    stage5_eval_polys.push(p.field_reg_inc);
    stage5_eval_polys.push(fr_wa_gather);

    for &poly in &stage5_eval_polys {
        ops.push(Op::Evaluate {
            poly,
            mode: EvalMode::FinalBind,
        });
    }
    ops.push(Op::AbsorbEvals {
        polys: stage5_eval_polys,
        tag: DomainSeparator::OpeningClaim,
    });

    Stage5Challenges {
        round_challenges: round_challenge_indices,
    }
}

// Stage 6: BytecodeReadRaf + Booleanity + HammingBooleanity +
//           RamRaVirtual + InstructionRaVirtual + IncClaimReduction
//
// Six batched sumcheck instances:
//   [0] BytecodeReadRaf           — log_k_bytecode + log_t rounds, degree bytecode_d+1
//   [1] Booleanity                — log_k_chunk + log_t rounds, degree 3
//   [2] HammingBooleanity         — log_t rounds, degree 3
//   [3] RamRaVirtual              — log_t rounds, degree ram_d+1
//   [4] InstructionRaVirtual      — log_t rounds, degree 5
//   [5] IncClaimReduction         — log_t rounds, degree 2
//
// max_rounds = log_k_bytecode + log_t (BytecodeReadRaf is the longest).
#[allow(clippy::too_many_arguments)]
fn build_stage6(
    p: &Polys,
    params: &ModuleParams,
    ops: &mut Vec<Op>,
    kernels: &mut Vec<KernelDef>,
    ch: &mut ChallengeTable,
    batched_sumchecks: &mut Vec<BatchedSumcheckDef>,
    s2: &Stage2Challenges,
    s3: &Stage3Challenges,
    s4: &Stage4Challenges,
    s5: &Stage5Challenges,
) -> Stage6Challenges {
    let stage6_max_rounds = params.log_k_bytecode + params.log_t;
    let booleanity_rounds = params.log_k_chunk + params.log_t;
    let n_committed_per_virtual = params.ra_virtual_log_k_chunk / params.log_k_chunk;

    // Pre-sumcheck challenge squeezes.
    //
    // BytecodeReadRafSumcheckParams::gen → 6 challenge_scalar_powers
    // HammingBooleanitySumcheckParams::new → no transcript ops
    // BooleanitySumcheckParams::new → 1 challenge_scalar_optimized
    // RamRaVirtualParams::new → no transcript ops
    // InstructionRaSumcheckParams::new → 1 challenge_scalar_powers
    // IncClaimReductionSumcheckParams::new → 1 challenge_scalar

    // BytecodeReadRaf: 6 gamma squeezes
    let ch_bc_gamma = ch.add(
        "bc_raf_gamma",
        ChallengeSource::FiatShamir { after_stage: 5 },
    );
    ops.push(Op::Squeeze {
        challenge: ch_bc_gamma,
    });
    let ch_bc_stage1_gamma = ch.add(
        "bc_raf_stage1_gamma",
        ChallengeSource::FiatShamir { after_stage: 5 },
    );
    ops.push(Op::Squeeze {
        challenge: ch_bc_stage1_gamma,
    });
    let ch_bc_stage2_gamma = ch.add(
        "bc_raf_stage2_gamma",
        ChallengeSource::FiatShamir { after_stage: 5 },
    );
    ops.push(Op::Squeeze {
        challenge: ch_bc_stage2_gamma,
    });
    let ch_bc_stage3_gamma = ch.add(
        "bc_raf_stage3_gamma",
        ChallengeSource::FiatShamir { after_stage: 5 },
    );
    ops.push(Op::Squeeze {
        challenge: ch_bc_stage3_gamma,
    });
    let ch_bc_stage4_gamma = ch.add(
        "bc_raf_stage4_gamma",
        ChallengeSource::FiatShamir { after_stage: 5 },
    );
    ops.push(Op::Squeeze {
        challenge: ch_bc_stage4_gamma,
    });
    let ch_bc_stage5_gamma = ch.add(
        "bc_raf_stage5_gamma",
        ChallengeSource::FiatShamir { after_stage: 5 },
    );
    ops.push(Op::Squeeze {
        challenge: ch_bc_stage5_gamma,
    });

    // Booleanity: 1 challenge_scalar_optimized (same as challenge_scalar)
    let ch_booleanity_gamma = ch.add(
        "booleanity_gamma",
        ChallengeSource::FiatShamir { after_stage: 5 },
    );
    ops.push(Op::Squeeze {
        challenge: ch_booleanity_gamma,
    });

    // Booleanity γ^{2d} power challenges for each RA polynomial dimension.
    // Includes FieldRegRa(d) so the BN254 Fr coprocessor's committed write
    // one-hot is also constrained to {0, 1} per cell. Without this, the
    // prover could commit a non-Boolean FieldRegRa and FR Twist soundness
    // would rest entirely on the (separately-checked) cross-binding to
    // FieldRegInc.
    let total_d = params.instruction_d + params.bytecode_d + params.ram_d + params.field_reg_d;
    let ch_booleanity_gamma_sq: Vec<ChallengeIdx> = (0..total_d)
        .map(|d| {
            let idx = ch.add(
                &format!("booleanity_gamma_sq_{d}"),
                ChallengeSource::Power {
                    base: ch_booleanity_gamma,
                    exponent: 2 * d,
                },
            );
            ops.push(Op::ComputePower {
                target: idx,
                base: ch_booleanity_gamma,
                exponent: 2 * d as u64,
            });
            idx
        })
        .collect();

    // Booleanity γ^d power challenges (kept for challenge index stability)
    let _ch_booleanity_gamma_pow: Vec<ChallengeIdx> = (0..total_d)
        .map(|d| {
            let idx = ch.add(
                &format!("booleanity_gamma_pow_{d}"),
                ChallengeSource::Power {
                    base: ch_booleanity_gamma,
                    exponent: d,
                },
            );
            ops.push(Op::ComputePower {
                target: idx,
                base: ch_booleanity_gamma,
                exponent: d as u64,
            });
            idx
        })
        .collect();

    // eq(r_addr, r'_addr) scalar captured at booleanity phase transition
    let ch_bool_eq_r_r = ch.add("booleanity_eq_r_r", ChallengeSource::External);

    // InstructionRaVirtual: 1 challenge_scalar_powers(n_virtual_ra_polys)
    let ch_inst_ra_gamma = ch.add(
        "inst_ra_virtual_gamma",
        ChallengeSource::FiatShamir { after_stage: 5 },
    );
    ops.push(Op::Squeeze {
        challenge: ch_inst_ra_gamma,
    });

    // IncClaimReduction: 1 challenge_scalar
    let ch_inc_gamma = ch.add(
        "inc_reduction_gamma",
        ChallengeSource::FiatShamir { after_stage: 5 },
    );
    ops.push(Op::Squeeze {
        challenge: ch_inc_gamma,
    });

    // Challenge point vectors
    let log_k_reg = LOG_K_REG;
    let mut next_eq = 19usize;

    // HammingBooleanity: r_cycle from Stage 1 (s2.stage1_cycle is BIG_ENDIAN)
    let hamming_r_cycle = s2.stage1_cycle.clone();

    // RamRaVirtual & InstructionRaVirtual: r_cycle from Stage 5 cycle phase
    let ra_virtual_r_cycle: Vec<ChallengeIdx> = s5.round_challenges
        [LOG_K_INSTRUCTION..LOG_K_INSTRUCTION + params.log_t]
        .iter()
        .rev()
        .copied()
        .collect();

    // RamRaVirtual: address chunks from Stage 2 (BIG_ENDIAN)
    let ram_ra_full_addr: Vec<ChallengeIdx> = s2.round_challenges
        [params.log_t..params.log_t + params.log_k_ram]
        .iter()
        .rev()
        .copied()
        .collect();
    // Pad with zero challenges at beginning (matching compute_r_address_chunks),
    // then sequential chunks: chunk[0]=MSB (padded), chunk[D-1]=LSB
    let ram_pad_len = if params.log_k_ram.is_multiple_of(params.log_k_chunk) {
        0
    } else {
        params.log_k_chunk - (params.log_k_ram % params.log_k_chunk)
    };
    let ram_pad_chs: Vec<ChallengeIdx> = (0..ram_pad_len)
        .map(|i| ch.add(&format!("ram_ra_pad_{i}"), ChallengeSource::External))
        .collect();
    let ram_ra_padded: Vec<ChallengeIdx> =
        ram_pad_chs.into_iter().chain(ram_ra_full_addr).collect();
    let ram_addr_chunks: Vec<Vec<ChallengeIdx>> = (0..params.ram_d)
        .map(|i| {
            let c = params.log_k_chunk;
            ram_ra_padded[i * c..(i + 1) * c].to_vec()
        })
        .collect();

    // InstructionRaVirtual: address chunks from Stage 5 address phase
    // InstructionReadRaf's normalize_opening_point keeps address in LE (squeezed) order,
    // so we do NOT reverse here. compute_r_address_chunks splits LE left-to-right.
    let inst_ra_full_addr: Vec<ChallengeIdx> = s5.round_challenges[0..LOG_K_INSTRUCTION].to_vec();
    let inst_addr_chunks: Vec<Vec<ChallengeIdx>> = (0..params.instruction_d)
        .map(|i| {
            let c = params.log_k_chunk;
            inst_ra_full_addr[i * c..(i + 1) * c].to_vec()
        })
        .collect();

    // IncClaimReduction: 4 cycle eq points from 3 distinct stages
    let inc_r_ram_s2: Vec<ChallengeIdx> = s2.round_challenges[0..params.log_t]
        .iter()
        .rev()
        .copied()
        .collect();
    let inc_r_ram_s4: Vec<ChallengeIdx> = s4.round_challenges[log_k_reg..log_k_reg + params.log_t]
        .iter()
        .rev()
        .copied()
        .collect();
    let inc_r_rd_s4: Vec<ChallengeIdx> = s4.round_challenges[0..params.log_t]
        .iter()
        .rev()
        .copied()
        .collect();
    let inc_r_rd_s5: Vec<ChallengeIdx> = s5.round_challenges
        [LOG_K_INSTRUCTION..LOG_K_INSTRUCTION + params.log_t]
        .iter()
        .rev()
        .copied()
        .collect();

    // BytecodeReadRaf challenge vectors (BIG_ENDIAN, log_t entries each)
    let bc_r_cycle_1 = s2.stage1_cycle.clone(); // SpartanOuter
    let bc_r_cycle_2: Vec<ChallengeIdx> =
        s2.round_challenges // SpartanProductVirtualization
        [params.log_k_ram..params.log_k_ram + params.log_t]
            .iter()
            .rev()
            .copied()
            .collect();
    let bc_r_cycle_3: Vec<ChallengeIdx> = s3.round_challenges // SpartanShift
        .iter().rev().copied().collect();
    let bc_r_cycle_4: Vec<ChallengeIdx> = s4.round_challenges // RegistersReadWriteChecking cycle
        [0..params.log_t]
        .iter()
        .rev()
        .copied()
        .collect();
    let bc_r_cycle_5: Vec<ChallengeIdx> =
        s5.round_challenges // RegistersValEvaluation cycle
        [LOG_K_INSTRUCTION..LOG_K_INSTRUCTION + params.log_t]
            .iter()
            .rev()
            .copied()
            .collect();
    // Register eq challenges for Val stages 3 and 4 (BIG_ENDIAN)
    // Both use r_address from RegistersReadWriteChecking (s4), since
    // RegistersValEvaluation inherits the same r_address.
    let bc_r_register_4: Vec<ChallengeIdx> = s4.round_challenges
        [params.log_t..params.log_t + log_k_reg]
        .iter()
        .rev()
        .copied()
        .collect();
    let bc_r_register_5: Vec<ChallengeIdx> = bc_r_register_4.clone();

    // Pre-allocate Stage 6 round challenge indices (needed by Phase 2 EqProject).
    // The actual Squeeze ops are emitted in the sumcheck loop below.
    let s6_round_ch: Vec<ChallengeIdx> = (0..stage6_max_rounds)
        .map(|r| {
            ch.add(
                &format!("s6_r_{r}"),
                ChallengeSource::SumcheckRound {
                    stage: VerifierStageIndex(5),
                    round: r + 1,
                },
            )
        })
        .collect();

    // Kernel definitions
    let n_vra = params.n_virtual_ra_polys;
    let max_degree = [
        params.bytecode_d + 1,
        3,
        3,
        params.ram_d + 1,
        n_committed_per_virtual + 1,
        2,
    ]
    .into_iter()
    .max()
    .unwrap();

    // [0] BytecodeReadRaf — address phase (degree 2)
    //
    // 12 inputs: F[0..5] (EqPushforward), GammaVal[0..5] (BytecodeVal),
    //            f_entry_trace (Provided), entry_weighted (ScaleByChallenge)
    //
    // Formula: Σ_{s=0..4} Input(s) × Input(5+s) + Input(10) × Input(11)
    let bc_raf_addr_kernel_idx = kernels.len();
    let bc_raf_addr_pre_ops: Vec<Op>;
    {
        use jolt_compiler::polynomial_id::bytecode_field as bf;

        let bytecode_k = 1usize << params.log_k_bytecode;
        let bc_r_cycles = [
            &bc_r_cycle_1,
            &bc_r_cycle_2,
            &bc_r_cycle_3,
            &bc_r_cycle_4,
            &bc_r_cycle_5,
        ];
        let stage_gamma_chs: [ChallengeIdx; 5] = [
            ch_bc_stage1_gamma,
            ch_bc_stage2_gamma,
            ch_bc_stage3_gamma,
            ch_bc_stage4_gamma,
            ch_bc_stage5_gamma,
        ];

        let mut addr_inputs = Vec::with_capacity(12);

        // Inputs 0-4: F[s] — EqPushforward of eq(r_cycle_s, ·) through PC index mapping
        for (s, r_cycle) in bc_r_cycles.iter().enumerate() {
            let poly = PolynomialId::BytecodeReadRafF(s);
            addr_inputs.push(InputBinding::EqPushforward {
                poly,
                eq_challenges: (*r_cycle).clone(),
                indices: PolynomialId::BytecodePcIndex,
                output_size: bytecode_k,
            });
        }

        // Inputs 5-9: GammaVal[s] — produced by WeightedSum ops at activation
        for s in 0..5 {
            addr_inputs.push(InputBinding::Provided {
                poly: PolynomialId::BytecodeReadRafGammaVal(s),
            });
        }

        // Input 10: f_entry_trace (preprocessed one-hot at PC[0]'s bytecode index)
        addr_inputs.push(InputBinding::Provided {
            poly: PolynomialId::BytecodeEntryTrace,
        });

        // Input 11: entry_gamma × f_entry_expected (gamma^7 × one-hot at entry index)
        addr_inputs.push(InputBinding::ScaleByChallenge {
            poly: PolynomialId::BytecodeEntryWeighted,
            source: PolynomialId::BytecodeEntryExpected,
            challenge: ch_bc_gamma,
            power: 7,
        });

        // Formula: Σ_{s=0..4} Input(s) × Input(5+s) + Input(10) × Input(11)
        let mut terms = Vec::with_capacity(6);
        for s in 0..5 {
            terms.push(ProductTerm {
                coefficient: 1,
                factors: vec![Factor::Input(s), Factor::Input(5 + s)],
            });
        }
        terms.push(ProductTerm {
            coefficient: 1,
            factors: vec![Factor::Input(10), Factor::Input(11)],
        });

        kernels.push(KernelDef {
            spec: KernelSpec {
                formula: Formula::from_terms(terms),
                num_evals: 3, // degree 2
                iteration: Iteration::Dense,
                binding_order: BindingOrder::LowToHigh,
                gruen_hint: None,
            },
            inputs: addr_inputs,
            num_rounds: params.log_k_bytecode,
            instance_config: None,
        });

        // Build pre-activation WeightedSum ops for GammaVal[0..4].
        // These produce BytecodeReadRafGammaVal(s) device buffers from
        // BytecodeField preprocessed polys + EqGather register lookups.
        let sg = stage_gamma_chs;
        let g = ch_bc_gamma;
        // EqGather for stages 3/4: eq(rd/rs1/rs2, r_register) per bytecode entry
        // Stage 3 uses bc_r_register_4, stage 4 uses bc_r_register_5
        let mut pre_ops = vec![
            Op::Materialize {
                binding: InputBinding::EqGather {
                    poly: PolynomialId::BytecodeRegEq(0),
                    eq_challenges: bc_r_register_4.clone(),
                    indices: PolynomialId::BytecodeField(bf::RD_INDEX),
                },
            },
            Op::Materialize {
                binding: InputBinding::EqGather {
                    poly: PolynomialId::BytecodeRegEq(1),
                    eq_challenges: bc_r_register_4.clone(),
                    indices: PolynomialId::BytecodeField(bf::RS1_INDEX),
                },
            },
            Op::Materialize {
                binding: InputBinding::EqGather {
                    poly: PolynomialId::BytecodeRegEq(2),
                    eq_challenges: bc_r_register_4.clone(),
                    indices: PolynomialId::BytecodeField(bf::RS2_INDEX),
                },
            },
            Op::Materialize {
                binding: InputBinding::EqGather {
                    poly: PolynomialId::BytecodeRegEq(3),
                    eq_challenges: bc_r_register_5.clone(),
                    indices: PolynomialId::BytecodeField(bf::RD_INDEX),
                },
            },
        ];

        // Stage 0: address + γ₀·imm + Σ γ₀^{2+f}·flag[f] + γ^5·k
        {
            let mut t: Vec<(PolynomialId, ChallengeIdx, u8)> = Vec::new();
            t.push((PolynomialId::BytecodeField(bf::ADDRESS), sg[0], 0));
            t.push((PolynomialId::BytecodeField(bf::IMM), sg[0], 1));
            for f in 0..NUM_CIRCUIT_FLAGS {
                t.push((
                    PolynomialId::BytecodeField(bf::CIRCUIT_FLAG_BASE + f),
                    sg[0],
                    (2 + f) as u8,
                ));
            }
            pre_ops.push(Op::WeightedSum {
                output: PolynomialId::BytecodeReadRafGammaVal(0),
                terms: t,
                identity_term: Some((g, 5)),
                overall_scale: None,
            });
        }

        // Stage 1: γ₁⁰·jump + γ₁¹·branch + γ₁²·wlotord + γ₁³·virtual × γ
        pre_ops.push(Op::WeightedSum {
            output: PolynomialId::BytecodeReadRafGammaVal(1),
            terms: vec![
                (
                    PolynomialId::BytecodeField(bf::CIRCUIT_FLAG_BASE + 5),
                    sg[1],
                    0,
                ), // Jump
                (PolynomialId::BytecodeField(bf::IS_BRANCH), sg[1], 1),
                (
                    PolynomialId::BytecodeField(bf::CIRCUIT_FLAG_BASE + 6),
                    sg[1],
                    2,
                ), // WriteLookupOutputToRD
                (
                    PolynomialId::BytecodeField(bf::CIRCUIT_FLAG_BASE + 7),
                    sg[1],
                    3,
                ), // VirtualInstruction
            ],
            identity_term: None,
            overall_scale: Some((g, 1)),
        });

        // Stage 2: imm + γ₂·addr + γ₂²·left_is_rs1 + ... + γ₂⁸·is_first_in_seq + γ^4·k × γ²
        pre_ops.push(Op::WeightedSum {
            output: PolynomialId::BytecodeReadRafGammaVal(2),
            terms: vec![
                (PolynomialId::BytecodeField(bf::IMM), sg[2], 0),
                (PolynomialId::BytecodeField(bf::ADDRESS), sg[2], 1),
                (PolynomialId::BytecodeField(bf::LEFT_IS_RS1), sg[2], 2),
                (PolynomialId::BytecodeField(bf::LEFT_IS_PC), sg[2], 3),
                (PolynomialId::BytecodeField(bf::RIGHT_IS_RS2), sg[2], 4),
                (PolynomialId::BytecodeField(bf::RIGHT_IS_IMM), sg[2], 5),
                (PolynomialId::BytecodeField(bf::IS_NOOP), sg[2], 6),
                (
                    PolynomialId::BytecodeField(bf::CIRCUIT_FLAG_BASE + 7),
                    sg[2],
                    7,
                ), // VirtualInstruction
                (
                    PolynomialId::BytecodeField(bf::CIRCUIT_FLAG_BASE + 12),
                    sg[2],
                    8,
                ), // IsFirstInSequence
            ],
            identity_term: Some((g, 4)),
            overall_scale: Some((g, 2)),
        });

        // Stage 3: γ₃⁰·eq(rd,r4) + γ₃¹·eq(rs1,r4) + γ₃²·eq(rs2,r4) × γ³
        pre_ops.push(Op::WeightedSum {
            output: PolynomialId::BytecodeReadRafGammaVal(3),
            terms: vec![
                (PolynomialId::BytecodeRegEq(0), sg[3], 0),
                (PolynomialId::BytecodeRegEq(1), sg[3], 1),
                (PolynomialId::BytecodeRegEq(2), sg[3], 2),
            ],
            identity_term: None,
            overall_scale: Some((g, 3)),
        });

        // Stage 4: eq(rd,r5) + γ₄¹·raf + Σ γ₄^{2+t}·table_flag[t] × γ⁴
        {
            let mut t: Vec<(PolynomialId, ChallengeIdx, u8)> = Vec::new();
            t.push((PolynomialId::BytecodeRegEq(3), sg[4], 0));
            t.push((PolynomialId::BytecodeField(bf::RAF_FLAG), sg[4], 1));
            for ti in 0..NUM_LOOKUP_TABLES {
                t.push((
                    PolynomialId::BytecodeField(bf::TABLE_FLAG_BASE + ti),
                    sg[4],
                    (2 + ti) as u8,
                ));
            }
            pre_ops.push(Op::WeightedSum {
                output: PolynomialId::BytecodeReadRafGammaVal(4),
                terms: t,
                identity_term: None,
                overall_scale: Some((g, 4)),
            });
        }

        bc_raf_addr_pre_ops = pre_ops;
    }

    // [0] BytecodeReadRaf — cycle phase (degree bytecode_d+1)
    //
    // After the address phase binds all k variables:
    //   - GammaVal[s] → scalar = γ^s × (Val_s(r_addr) + raf)
    //   - entry_weighted → scalar = γ^7 × f_entry_expected(r_addr)
    //   - BytecodeRa(i) → projected via EqProject through address chunks
    //
    // Formula: Σ_{s=0..4} weight_s × eq_s(j) × ∏_i ra_proj_i(j)
    //        + weight_entry × eq_entry(j) × ∏_i ra_proj_i(j)
    //
    // Degree: bytecode_d (from RA product) + 1 (from eq) = bytecode_d + 1
    let bc_raf_cycle_kernel_idx = kernels.len();
    // Allocate challenge slots for ScalarCaptures at the phase boundary.
    let ch_bc_cycle_weight: Vec<ChallengeIdx> = (0..5)
        .map(|s| ch.add(&format!("bc_cycle_weight_{s}"), ChallengeSource::External))
        .collect();
    let ch_bc_entry_weight = ch.add("bc_cycle_entry_weight", ChallengeSource::External);
    let bc_raf_proj_ids: Vec<PolynomialId>;
    let bc_addr_chunks: Vec<Vec<ChallengeIdx>>;
    {
        // Bytecode address chunks from stage 6 round challenges 0..log_k_bytecode.
        // These are the LE address challenges from the BytecodeReadRaf address phase.
        // Reverse to BIG_ENDIAN, pad with zero challenges, then sequential chunks.
        let bc_addr_be: Vec<ChallengeIdx> = s6_round_ch[0..params.log_k_bytecode]
            .iter()
            .rev()
            .copied()
            .collect();
        let bc_pad_len = if params.log_k_bytecode.is_multiple_of(params.log_k_chunk) {
            0
        } else {
            params.log_k_chunk - (params.log_k_bytecode % params.log_k_chunk)
        };
        let bc_pad_chs: Vec<ChallengeIdx> = (0..bc_pad_len)
            .map(|i| ch.add(&format!("bc_ra_pad_{i}"), ChallengeSource::External))
            .collect();
        let bc_addr_padded: Vec<ChallengeIdx> = bc_pad_chs.into_iter().chain(bc_addr_be).collect();
        bc_addr_chunks = (0..params.bytecode_d)
            .map(|i| {
                let c = params.log_k_chunk;
                bc_addr_padded[i * c..(i + 1) * c].to_vec()
            })
            .collect();

        // Entry eq point: all-zero challenges → eq(0, j) = ∏(1 - j_i)
        let entry_eq_chs: Vec<ChallengeIdx> = (0..params.log_t)
            .map(|i| ch.add(&format!("bc_entry_eq_{i}"), ChallengeSource::External))
            .collect();

        let mut cycle_inputs = Vec::with_capacity(params.bytecode_d + 6);

        // Inputs 0..d-1: EqProject of BytecodeRa(i) through address chunks
        let mut proj_ids = Vec::with_capacity(params.bytecode_d);
        for (i, chunk) in bc_addr_chunks.iter().enumerate() {
            let proj_id = PolynomialId::BatchEq(next_eq);
            next_eq += 1;
            proj_ids.push(proj_id);
            let chunk_k = 1usize << chunk.len();
            cycle_inputs.push(InputBinding::EqProject {
                poly: proj_id,
                source: PolynomialId::BytecodeRa(i),
                challenges: chunk.clone(),
                inner_size: chunk_k,
                outer_size: 1 << params.log_t,
            });
        }
        bc_raf_proj_ids = proj_ids;

        // Inputs d..d+4: per-stage eq tables (stage 0..4 cycle challenges)
        let bc_cycle_chs = [
            &bc_r_cycle_1,
            &bc_r_cycle_2,
            &bc_r_cycle_3,
            &bc_r_cycle_4,
            &bc_r_cycle_5,
        ];
        for (s, chs) in bc_cycle_chs.iter().enumerate() {
            let eq_id = PolynomialId::BatchEq(next_eq);
            next_eq += 1;
            cycle_inputs.push(InputBinding::EqTable {
                poly: eq_id,
                challenges: (*chs).clone(),
            });
        }

        // Input d+5: entry eq table (all zeros)
        let entry_eq_id = PolynomialId::BatchEq(next_eq);
        next_eq += 1;
        cycle_inputs.push(InputBinding::EqTable {
            poly: entry_eq_id,
            challenges: entry_eq_chs,
        });

        // Formula: 6 terms, each = Challenge(weight) × Input(eq) × ∏ Input(ra)
        let d = params.bytecode_d;
        let mut terms = Vec::with_capacity(6);
        let ra_factors: Vec<Factor> = (0..d).map(|i| Factor::Input(i as u32)).collect();

        for (s, &weight) in ch_bc_cycle_weight.iter().enumerate() {
            let mut factors = vec![
                Factor::Challenge(weight.0 as u32),
                Factor::Input((d + s) as u32),
            ];
            factors.extend(ra_factors.iter().copied());
            terms.push(ProductTerm {
                coefficient: 1,
                factors,
            });
        }
        // Entry term
        {
            let mut factors = vec![
                Factor::Challenge(ch_bc_entry_weight.0 as u32),
                Factor::Input((d + 5) as u32),
            ];
            factors.extend(ra_factors.iter().copied());
            terms.push(ProductTerm {
                coefficient: 1,
                factors,
            });
        }

        kernels.push(KernelDef {
            spec: KernelSpec {
                formula: Formula::from_terms(terms),
                num_evals: d + 2, // degree d+1
                iteration: Iteration::Dense,
                binding_order: BindingOrder::LowToHigh,
                gruen_hint: None,
            },
            inputs: cycle_inputs,
            num_rounds: params.log_t,
            instance_config: None,
        });
    }

    // [1] Booleanity — single kernel over full (K_chunk × T) ra_d polynomials.
    //
    // The Booleanity sumcheck verifies:
    //   Σ_{addr,cycle} eq(r_addr, addr) × eq(r_cycle, cycle) × Σ_d γ^{2d} × (ra_d² - ra_d) = 0
    //
    // Using projected G_d values is WRONG because (Σ eq×ra)² ≠ Σ eq×ra².
    // We operate on the original ra_d polynomial (K_chunk × T elements) transposed
    // to cycle-major layout: ra_d_cm[cycle * K_chunk + addr].
    //
    // With LowToHigh binding on cycle-major layout:
    //   - Low bits = addr (log_k_chunk rounds) → matches jolt-core Phase 1
    //   - High bits = cycle (log_t rounds)     → matches jolt-core Phase 2
    //
    // Booleanity r_address: derived from InstructionRa(0)'s opening point.
    //
    // After Stage 5's PrefixSuffix sumcheck, each virtual RA polynomial
    // (InstructionRa(i)) gets a ra_virtual_log_k_chunk-sized address chunk.
    // InstructionRa(0) gets r_address_prime[0..16] = rounds 0..15 (the first
    // two PrefixSuffix phases, which handle the MSB bits).
    //
    // jolt-core stores this 16-element point as BE, then reverses to LE and
    // takes the last log_k_chunk entries = the first log_k_chunk rounds reversed.
    let bool_addr_ch: Vec<ChallengeIdx> = s5.round_challenges[0..params.log_k_chunk]
        .iter()
        .rev()
        .copied()
        .collect();
    let bool_cycle_ch: Vec<ChallengeIdx> =
        s5.round_challenges[LOG_K_INSTRUCTION..LOG_K_INSTRUCTION + params.log_t].to_vec();
    let ra_poly_ids: Vec<PolynomialId> = (0..params.instruction_d)
        .map(PolynomialId::InstructionRa)
        .chain((0..params.bytecode_d).map(PolynomialId::BytecodeRa))
        .chain((0..params.ram_d).map(PolynomialId::RamRa))
        .chain((0..params.field_reg_d).map(PolynomialId::FieldRegRa))
        .collect();

    // Booleanity: single-phase dense kernel over transposed RA polynomials.
    //
    // Formula: Σ_d γ²_d × eq_tensor × (ra_d² - ra_d)
    //   eq_tensor = eq([r_cycle_rev, r_addr_rev], [cycle, addr])  (BIG ENDIAN)
    //   ra_d = Transpose(ra_d_committed, rows=k_chunk, cols=t_size) → cycle-major
    //
    // LowToHigh binding processes address (low bits) then cycle (high bits):
    //   13 total rounds = log_k_chunk + log_t
    let k_chunk = 1usize << params.log_k_chunk;
    let t_size = 1usize << params.log_t;
    let booleanity_rounds = params.log_k_chunk + params.log_t;
    let bool_first_active = params.log_k_bytecode - params.log_k_chunk;

    // Combined eq table: BIG ENDIAN point = [cycle_ch reversed, addr_ch]
    // This ensures LowToHigh ch_i ↔ r_addr[i] for i < log_k_chunk,
    // ch_{log_k_chunk+j} ↔ r_cycle[j] for j < log_t.
    let bool_eq_id = PolynomialId::BatchEq(next_eq);
    next_eq += 1;
    // bool_addr_ch is r_address in LE order [ch_3, ch_2, ch_1, ch_0].
    // BIG ENDIAN eq point needs r_addr reversed → [ch_0, ch_1, ch_2, ch_3].
    let bool_eq_challenges: Vec<ChallengeIdx> = bool_cycle_ch
        .iter()
        .rev()
        .chain(bool_addr_ch.iter().rev())
        .copied()
        .collect();

    let mut bool_terms = Vec::with_capacity(2 * total_d);
    for (d, gamma_sq) in ch_booleanity_gamma_sq.iter().enumerate() {
        let gamma_sq_idx = gamma_sq.0 as u32;
        let ra_input = (d + 1) as u32;
        bool_terms.push(ProductTerm {
            coefficient: 1,
            factors: vec![
                Factor::Challenge(gamma_sq_idx),
                Factor::Input(0),
                Factor::Input(ra_input),
                Factor::Input(ra_input),
            ],
        });
        bool_terms.push(ProductTerm {
            coefficient: -1,
            factors: vec![
                Factor::Challenge(gamma_sq_idx),
                Factor::Input(0),
                Factor::Input(ra_input),
            ],
        });
    }

    let bool_g_ids: Vec<PolynomialId> = (0..total_d).map(PolynomialId::BooleanityG).collect();
    let mut bool_inputs = Vec::with_capacity(1 + total_d);
    bool_inputs.push(InputBinding::EqTable {
        poly: bool_eq_id,
        challenges: bool_eq_challenges,
    });
    for (d, &ra_pid) in ra_poly_ids.iter().enumerate() {
        bool_inputs.push(InputBinding::Transpose {
            poly: bool_g_ids[d],
            source: ra_pid,
            rows: k_chunk,
            cols: t_size,
        });
    }

    let bool_kernel_idx = kernels.len();
    kernels.push(KernelDef {
        spec: KernelSpec {
            formula: Formula::from_terms(bool_terms),
            num_evals: 4,
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
            gruen_hint: None,
        },
        inputs: bool_inputs,
        num_rounds: booleanity_rounds,
        instance_config: None,
    });

    // [2] HammingBooleanity — Dense, degree 3: eq × H² − eq × H
    let hamming_kernel_idx = kernels.len();
    {
        let eq_id = PolynomialId::BatchEq(next_eq);
        next_eq += 1;
        kernels.push(KernelDef {
            spec: KernelSpec {
                formula: Formula::from_terms(vec![
                    ProductTerm {
                        coefficient: 1,
                        factors: vec![Factor::Input(0), Factor::Input(1), Factor::Input(1)],
                    },
                    ProductTerm {
                        coefficient: -1,
                        factors: vec![Factor::Input(0), Factor::Input(1)],
                    },
                ]),
                num_evals: 4,
                iteration: Iteration::Dense,
                binding_order: BindingOrder::LowToHigh,
                gruen_hint: None,
            },
            inputs: vec![
                InputBinding::EqTable {
                    poly: eq_id,
                    challenges: hamming_r_cycle,
                },
                InputBinding::Provided {
                    poly: p.hamming_weight,
                },
            ],
            num_rounds: params.log_t,
            instance_config: None,
        });
    }

    // [3] RamRaVirtual — Dense, degree ram_d+1: eq_cycle × Π EqProject(RamRa(i))
    let ram_ra_virtual_kernel_idx = kernels.len();
    let ram_ra_proj_ids: Vec<PolynomialId>;
    {
        let mut factors: Vec<Factor> = vec![Factor::Input(0)];
        for i in 0..params.ram_d {
            factors.push(Factor::Input((i + 1) as u32));
        }
        let eq_id = PolynomialId::BatchEq(next_eq);
        next_eq += 1;
        let mut inputs = vec![InputBinding::EqTable {
            poly: eq_id,
            challenges: ra_virtual_r_cycle.clone(),
        }];
        let mut proj_ids = Vec::with_capacity(params.ram_d);
        for (i, chunk) in ram_addr_chunks.iter().enumerate() {
            let proj_id = PolynomialId::BatchEq(next_eq);
            next_eq += 1;
            proj_ids.push(proj_id);
            let chunk_k = 1usize << chunk.len(); // actual K for this chunk
            inputs.push(InputBinding::EqProject {
                poly: proj_id,
                source: PolynomialId::RamRa(i),
                challenges: chunk.clone(),
                inner_size: chunk_k,
                outer_size: 1 << params.log_t,
            });
        }
        ram_ra_proj_ids = proj_ids;
        kernels.push(KernelDef {
            spec: KernelSpec {
                formula: Formula::from_terms(vec![ProductTerm {
                    coefficient: 1,
                    factors,
                }]),
                num_evals: params.ram_d + 2,
                iteration: Iteration::Dense,
                binding_order: BindingOrder::LowToHigh,
                gruen_hint: None,
            },
            inputs,
            num_rounds: params.log_t,
            instance_config: None,
        });
    }

    // [4] InstructionRaVirtual — Dense, degree M+1
    //     Σ_{b=0..N-1} γ^b × eq_cycle × Π_{k=0..M-1} ra_{b*M+k}
    let inst_ra_virtual_kernel_idx = kernels.len();
    let inst_ra_proj_ids: Vec<PolynomialId>;
    {
        let m = n_committed_per_virtual;
        let eq_id = PolynomialId::BatchEq(next_eq);
        next_eq += 1;
        let mut inputs = vec![InputBinding::EqTable {
            poly: eq_id,
            challenges: ra_virtual_r_cycle,
        }];
        let mut proj_ids = Vec::with_capacity(params.instruction_d);
        for (i, chunk) in inst_addr_chunks.iter().enumerate() {
            let proj_id = PolynomialId::BatchEq(next_eq);
            next_eq += 1;
            proj_ids.push(proj_id);
            let chunk_k = 1usize << chunk.len();
            inputs.push(InputBinding::EqProject {
                poly: proj_id,
                source: PolynomialId::InstructionRa(i),
                challenges: chunk.clone(),
                inner_size: chunk_k,
                outer_size: 1 << params.log_t,
            });
        }
        inst_ra_proj_ids = proj_ids;
        let g_inst = ch_inst_ra_gamma.0 as u32;
        let mut terms = Vec::with_capacity(n_vra);
        for b in 0..n_vra {
            let mut factors = Vec::new();
            for _ in 0..b {
                factors.push(Factor::Challenge(g_inst));
            }
            factors.push(Factor::Input(0));
            for k in 0..m {
                factors.push(Factor::Input((b * m + k + 1) as u32));
            }
            terms.push(ProductTerm {
                coefficient: 1,
                factors,
            });
        }
        kernels.push(KernelDef {
            spec: KernelSpec {
                formula: Formula::from_terms(terms),
                num_evals: m + 2,
                iteration: Iteration::Dense,
                binding_order: BindingOrder::LowToHigh,
                gruen_hint: None,
            },
            inputs,
            num_rounds: params.log_t,
            instance_config: None,
        });
    }

    // [5] IncClaimReduction — Dense, degree 2
    //     RamInc×eq_s2 + γ×RamInc×eq_s4 + γ²×RdInc×eq_s4_rd + γ³×RdInc×eq_s5
    let inc_kernel_idx = kernels.len();
    {
        let g_inc = ch_inc_gamma.0 as u32;
        let eq_s2_id = PolynomialId::BatchEq(next_eq);
        next_eq += 1;
        let eq_s4_ram_id = PolynomialId::BatchEq(next_eq);
        next_eq += 1;
        let eq_s4_rd_id = PolynomialId::BatchEq(next_eq);
        next_eq += 1;
        let eq_s5_id = PolynomialId::BatchEq(next_eq);
        #[allow(unused_assignments)]
        {
            next_eq += 1;
        }
        kernels.push(KernelDef {
            spec: KernelSpec {
                formula: Formula::from_terms(vec![
                    ProductTerm {
                        coefficient: 1,
                        factors: vec![Factor::Input(0), Factor::Input(2)],
                    },
                    ProductTerm {
                        coefficient: 1,
                        factors: vec![Factor::Challenge(g_inc), Factor::Input(0), Factor::Input(3)],
                    },
                    ProductTerm {
                        coefficient: 1,
                        factors: vec![
                            Factor::Challenge(g_inc),
                            Factor::Challenge(g_inc),
                            Factor::Input(1),
                            Factor::Input(4),
                        ],
                    },
                    ProductTerm {
                        coefficient: 1,
                        factors: vec![
                            Factor::Challenge(g_inc),
                            Factor::Challenge(g_inc),
                            Factor::Challenge(g_inc),
                            Factor::Input(1),
                            Factor::Input(5),
                        ],
                    },
                ]),
                num_evals: 3,
                iteration: Iteration::Dense,
                binding_order: BindingOrder::LowToHigh,
                gruen_hint: None,
            },
            inputs: vec![
                InputBinding::Provided { poly: p.ram_inc },
                InputBinding::Provided { poly: p.rd_inc },
                InputBinding::EqTable {
                    poly: eq_s2_id,
                    challenges: inc_r_ram_s2,
                },
                InputBinding::EqTable {
                    poly: eq_s4_ram_id,
                    challenges: inc_r_ram_s4,
                },
                InputBinding::EqTable {
                    poly: eq_s4_rd_id,
                    challenges: inc_r_rd_s4,
                },
                InputBinding::EqTable {
                    poly: eq_s5_id,
                    challenges: inc_r_rd_s5,
                },
            ],
            num_rounds: params.log_t,
            instance_config: None,
        });
    }

    // Batched sumcheck definition
    let batch_idx = BatchIdx(batched_sumchecks.len());

    batched_sumchecks.push(BatchedSumcheckDef {
        instances: vec![
            // [0] BytecodeReadRaf — 2 phases (address then cycle)
            BatchedInstance {
                phases: vec![
                    InstancePhase {
                        kernel: bc_raf_addr_kernel_idx,
                        num_rounds: params.log_k_bytecode,
                        scalar_captures: vec![],
                        segmented: None,
                        carry_bindings: vec![],
                        pre_activation_ops: bc_raf_addr_pre_ops,
                    },
                    InstancePhase {
                        kernel: bc_raf_cycle_kernel_idx,
                        num_rounds: params.log_t,
                        // Capture bound scalars from address phase into challenge slots:
                        // GammaVal[s] → γ^s × (Val_s(r_addr) + raf), entry_weighted → γ^7 × f_entry_expected(r_addr)
                        scalar_captures: (0..5)
                            .map(|s| ScalarCapture {
                                poly: PolynomialId::BytecodeReadRafGammaVal(s),
                                challenge: ch_bc_cycle_weight[s],
                            })
                            .chain(std::iter::once(ScalarCapture {
                                poly: PolynomialId::BytecodeEntryWeighted,
                                challenge: ch_bc_entry_weight,
                            }))
                            .collect(),
                        segmented: None,
                        carry_bindings: vec![],
                        pre_activation_ops: vec![],
                    },
                ],
                batch_coeff: ChallengeIdx(0),
                first_active_round: 0,
            },
            // [1] Booleanity — single-phase over transposed RA polys
            BatchedInstance {
                phases: vec![InstancePhase {
                    kernel: bool_kernel_idx,
                    num_rounds: booleanity_rounds,
                    scalar_captures: vec![],
                    segmented: None,
                    carry_bindings: vec![],
                    pre_activation_ops: vec![],
                }],
                batch_coeff: ChallengeIdx(0),
                first_active_round: bool_first_active,
            },
            // [2] HammingBooleanity
            BatchedInstance {
                phases: vec![InstancePhase {
                    kernel: hamming_kernel_idx,
                    num_rounds: params.log_t,
                    scalar_captures: vec![],
                    segmented: None,
                    carry_bindings: vec![],
                    pre_activation_ops: vec![],
                }],
                batch_coeff: ChallengeIdx(0),
                first_active_round: params.log_k_bytecode,
            },
            // [3] RamRaVirtual
            BatchedInstance {
                phases: vec![InstancePhase {
                    kernel: ram_ra_virtual_kernel_idx,
                    num_rounds: params.log_t,
                    scalar_captures: vec![],
                    segmented: None,
                    carry_bindings: vec![],
                    pre_activation_ops: vec![],
                }],
                batch_coeff: ChallengeIdx(0),
                first_active_round: params.log_k_bytecode,
            },
            // [4] InstructionRaVirtual
            BatchedInstance {
                phases: vec![InstancePhase {
                    kernel: inst_ra_virtual_kernel_idx,
                    num_rounds: params.log_t,
                    scalar_captures: vec![],
                    segmented: None,
                    carry_bindings: vec![],
                    pre_activation_ops: vec![],
                }],
                batch_coeff: ChallengeIdx(0),
                first_active_round: params.log_k_bytecode,
            },
            // [5] IncClaimReduction
            BatchedInstance {
                phases: vec![InstancePhase {
                    kernel: inc_kernel_idx,
                    num_rounds: params.log_t,
                    scalar_captures: vec![],
                    segmented: None,
                    carry_bindings: vec![],
                    pre_activation_ops: vec![],
                }],
                batch_coeff: ChallengeIdx(0),
                first_active_round: params.log_k_bytecode,
            },
        ],
        input_claims: vec![],
        max_rounds: stage6_max_rounds,
        max_degree,
    });

    // Input claims

    // [0] BytecodeReadRaf: complex multi-gamma formula across Stages 1-5.
    //
    // input_claim = Σ_{s=0..4} gamma^s * rv_claim_{s+1}
    //             + gamma^5 * raf_claim + gamma^6 * raf_shift_claim + gamma^7
    //
    // Uses StagedEval for evaluations overwritten by later stages.
    let bc_raf_input_claim = build_bytecode_read_raf_claim(
        p,
        ch_bc_gamma,
        ch_bc_stage1_gamma,
        ch_bc_stage2_gamma,
        ch_bc_stage3_gamma,
        ch_bc_stage4_gamma,
        ch_bc_stage5_gamma,
    );
    ops.push(Op::AbsorbInputClaim {
        formula: bc_raf_input_claim.clone(),
        tag: DomainSeparator::SumcheckClaim,
        batch: batch_idx,
        instance: InstanceIdx(0),
        inactive_scale_bits: 0,
    });

    // [1] Booleanity: input_claim = 0
    let booleanity_input_claim = ClaimFormula::zero();
    ops.push(Op::AbsorbInputClaim {
        formula: booleanity_input_claim.clone(),
        tag: DomainSeparator::SumcheckClaim,
        batch: batch_idx,
        instance: InstanceIdx(1),
        inactive_scale_bits: params.log_k_bytecode - params.log_k_chunk,
    });

    // [2] HammingBooleanity: input_claim = 0
    let hamming_input_claim = ClaimFormula::zero();
    ops.push(Op::AbsorbInputClaim {
        formula: hamming_input_claim.clone(),
        tag: DomainSeparator::SumcheckClaim,
        batch: batch_idx,
        instance: InstanceIdx(2),
        inactive_scale_bits: params.log_k_bytecode,
    });

    // [3] RamRaVirtual: input_claim = ram_combined_ra (from Stage 5)
    // The RamRa claim was evaluated and flushed in Stage 5 via BatchEq(17).
    let ram_ra_virtual_input_claim = ClaimFormula {
        terms: vec![ClaimTerm {
            coeff: 1,
            factors: vec![ClaimFactor::Eval(PolynomialId::BatchEq(17))],
        }],
    };
    ops.push(Op::AbsorbInputClaim {
        formula: ram_ra_virtual_input_claim.clone(),
        tag: DomainSeparator::SumcheckClaim,
        batch: batch_idx,
        instance: InstanceIdx(3),
        inactive_scale_bits: params.log_k_bytecode,
    });

    // [4] InstructionRaVirtual: input_claim = Σ gamma^i * InstructionRa(i) eval
    // InstructionRa evals from Stage 5 are in state.evaluations[InstructionRa(i)].
    let mut inst_ra_terms = Vec::with_capacity(n_vra);
    for i in 0..n_vra {
        let mut factors: Vec<ClaimFactor> = (0..i)
            .map(|_| ClaimFactor::Challenge(ch_inst_ra_gamma))
            .collect();
        factors.push(ClaimFactor::Eval(PolynomialId::InstructionRa(i)));
        inst_ra_terms.push(ClaimTerm { coeff: 1, factors });
    }
    let inst_ra_virtual_input_claim = ClaimFormula {
        terms: inst_ra_terms,
    };
    ops.push(Op::AbsorbInputClaim {
        formula: inst_ra_virtual_input_claim.clone(),
        tag: DomainSeparator::SumcheckClaim,
        batch: batch_idx,
        instance: InstanceIdx(4),
        inactive_scale_bits: params.log_k_bytecode,
    });

    // [5] IncClaimReduction: v_1 + γ*v_2 + γ²*w_1 + γ³*w_2
    // v_1 = RamInc @ RamRW (Stage 2 eval), v_2 = RamInc @ RamValCheck (Stage 4 eval)
    // w_1 = RdInc @ RegistersRW (Stage 4 eval), w_2 = RdInc @ RegistersValEval (Stage 5 eval)
    //
    // By Stage 6: state.evaluations[RamInc] = Stage 4 value (v_2),
    //             state.evaluations[RdInc] = Stage 5 value (w_2).
    // v_1 was overwritten (Stage 2 → Stage 4), w_1 was overwritten (Stage 4 → Stage 5).
    // Need snapshots for v_1 and w_1.
    //
    // Stage indices: Stage 2 = index 1, Stage 4 = index 3, Stage 5 = index 4.
    // RamInc is evaluated at both Stage 2 (v_1) and Stage 4 (v_2);
    // RdInc is evaluated at both Stage 4 (w_1) and Stage 5 (w_2).
    let g_inc = ch_inc_gamma;
    let inc_input_claim = ClaimFormula {
        terms: vec![
            // v_1: RamInc @ Stage 2
            ClaimTerm {
                coeff: 1,
                factors: vec![ClaimFactor::StagedEval {
                    poly: p.ram_inc,
                    stage: 1,
                }],
            },
            // γ * v_2: RamInc @ Stage 4 (current)
            ClaimTerm {
                coeff: 1,
                factors: vec![ClaimFactor::Challenge(g_inc), ClaimFactor::Eval(p.ram_inc)],
            },
            // γ² * w_1: RdInc @ Stage 4
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(g_inc),
                    ClaimFactor::Challenge(g_inc),
                    ClaimFactor::StagedEval {
                        poly: p.rd_inc,
                        stage: 3,
                    },
                ],
            },
            // γ³ * w_2: RdInc @ Stage 5 (current)
            ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(g_inc),
                    ClaimFactor::Challenge(g_inc),
                    ClaimFactor::Challenge(g_inc),
                    ClaimFactor::Eval(p.rd_inc),
                ],
            },
        ],
    };
    ops.push(Op::AbsorbInputClaim {
        formula: inc_input_claim.clone(),
        tag: DomainSeparator::SumcheckClaim,
        batch: batch_idx,
        instance: InstanceIdx(5),
        inactive_scale_bits: params.log_k_bytecode,
    });

    // Batching coefficients (6 instances)
    let batch_challenges: Vec<ChallengeIdx> = (0..6)
        .map(|i| {
            let c = ch.add(
                &format!("stage6_batch{i}"),
                ChallengeSource::FiatShamir { after_stage: 5 },
            );
            ops.push(Op::Squeeze { challenge: c });
            c
        })
        .collect();

    for (i, &bc) in batch_challenges.iter().enumerate() {
        batched_sumchecks[batch_idx.0].instances[i].batch_coeff = bc;
    }
    batched_sumchecks[batch_idx.0].input_claims = vec![
        bc_raf_input_claim,
        booleanity_input_claim,
        hamming_input_claim,
        ram_ra_virtual_input_claim,
        inst_ra_virtual_input_claim,
        inc_input_claim,
    ];

    // Sumcheck rounds: log_k_bytecode + log_t rounds
    // Per-instance degree per phase — used to compute per-round num_coeffs.
    let instance_phase_degrees: Vec<Vec<(usize, usize)>> = batched_sumchecks[batch_idx.0]
        .instances
        .iter()
        .map(|inst| {
            inst.phases
                .iter()
                .map(|ph| {
                    let d = kernels[ph.kernel].spec.num_evals; // degree + 1
                    (ph.num_rounds, d)
                })
                .collect()
        })
        .collect();

    let bdef_instances = &batched_sumchecks[batch_idx.0].instances;
    let _round_challenge_indices = emit_unrolled_batched_rounds(
        ops,
        kernels,
        batched_sumchecks,
        ch,
        batch_idx,
        stage6_max_rounds,
        |round| {
            let mut round_degree = 0usize;
            for (inst_idx, inst) in bdef_instances.iter().enumerate() {
                if round >= inst.first_active_round {
                    let inst_round = round - inst.first_active_round;
                    let mut accum = 0;
                    for &(ph_rounds, ph_evals) in &instance_phase_degrees[inst_idx] {
                        if inst_round < accum + ph_rounds {
                            round_degree = round_degree.max(ph_evals - 1);
                            break;
                        }
                        accum += ph_rounds;
                    }
                }
            }
            round_degree + 1
        },
        VerifierStageIndex(5),
        "s6_r",
        1,
        Some(&s6_round_ch),
    );

    // Post-sumcheck eval flush.
    //
    // Each cache_openings group is flushed separately because the same
    // PolynomialId can be opened at different points by different instances,
    // producing different evaluation values. We compute each group's evaluations
    // from the correct source before absorbing.
    //
    // Order: BytecodeReadRaf, Booleanity, HammingBooleanity,
    //        RamRaVirtual, InstructionRaVirtual, IncClaimReduction

    // [0] BytecodeReadRaf: read from EqProject (BatchEq) buffers in the cycle kernel
    for &proj_id in &bc_raf_proj_ids {
        ops.push(Op::Evaluate {
            poly: proj_id,
            mode: EvalMode::FinalBind,
        });
    }
    ops.push(Op::AbsorbEvals {
        polys: bc_raf_proj_ids.clone(),
        tag: DomainSeparator::OpeningClaim,
    });

    // [1] Booleanity: apply final bind on single-phase kernel, evaluate transposed RA buffers,
    // alias to original RA poly IDs.
    ops.push(Op::InstanceBind {
        batch: batch_idx,
        instance: InstanceIdx(1),
        kernel: bool_kernel_idx,
        challenge: s6_round_ch[stage6_max_rounds - 1],
    });
    // MUST match the `ra_poly_ids` order at the Booleanity kernel input
    // construction — the AbsorbEvals flush is paired with the kernel's
    // FullyBound G_d evaluations via AliasEval.
    let bool_ra_poly_ids: Vec<PolynomialId> = (0..params.instruction_d)
        .map(PolynomialId::InstructionRa)
        .chain((0..params.bytecode_d).map(PolynomialId::BytecodeRa))
        .chain((0..params.ram_d).map(PolynomialId::RamRa))
        .chain((0..params.field_reg_d).map(PolynomialId::FieldRegRa))
        .collect();
    for (d, &ra_pid) in bool_ra_poly_ids.iter().enumerate() {
        ops.push(Op::Evaluate {
            poly: bool_g_ids[d],
            mode: EvalMode::FullyBound,
        });
        ops.push(Op::AliasEval {
            from: bool_g_ids[d],
            to: ra_pid,
        });
    }
    ops.push(Op::AbsorbEvals {
        polys: bool_ra_poly_ids,
        tag: DomainSeparator::OpeningClaim,
    });

    // [2] HammingBooleanity: device buffer is valid (bound during sumcheck)
    ops.push(Op::Evaluate {
        poly: p.hamming_weight,
        mode: EvalMode::FinalBind,
    });
    ops.push(Op::AbsorbEvals {
        polys: vec![p.hamming_weight],
        tag: DomainSeparator::OpeningClaim,
    });

    // [3] RamRaVirtual: read from EqProject (BatchEq) buffers, not stale RamRa device buffers
    for &proj_id in &ram_ra_proj_ids {
        ops.push(Op::Evaluate {
            poly: proj_id,
            mode: EvalMode::FinalBind,
        });
    }
    ops.push(Op::AbsorbEvals {
        polys: ram_ra_proj_ids.clone(),
        tag: DomainSeparator::OpeningClaim,
    });

    // [4] InstructionRaVirtual: read from EqProject (BatchEq) buffers
    for &proj_id in &inst_ra_proj_ids {
        ops.push(Op::Evaluate {
            poly: proj_id,
            mode: EvalMode::FinalBind,
        });
    }
    ops.push(Op::AbsorbEvals {
        polys: inst_ra_proj_ids.clone(),
        tag: DomainSeparator::OpeningClaim,
    });

    // [5] IncClaimReduction: device buffers are valid (bound during sumcheck)
    ops.push(Op::Evaluate {
        poly: p.ram_inc,
        mode: EvalMode::FinalBind,
    });
    ops.push(Op::Evaluate {
        poly: p.rd_inc,
        mode: EvalMode::FinalBind,
    });
    ops.push(Op::AbsorbEvals {
        polys: vec![p.ram_inc, p.rd_inc],
        tag: DomainSeparator::OpeningClaim,
    });

    Stage6Challenges {
        round_challenges: s6_round_ch,
        bc_addr_chunks,
        inst_addr_chunks,
        ram_addr_chunks,
        bc_raf_proj_ids,
        inst_ra_proj_ids,
        ram_ra_proj_ids,
    }
}

// Stage 7: HammingWeight + Address Reduction
// 1 instance: HammingWeightClaimReduction (log_k_chunk rounds, degree 2)
//
// Fuses HammingWeight, Booleanity reduction, and Virtualization reduction
// into a single sumcheck over the address chunk dimension.
struct Stage7Challenges {
    /// HW reduction round challenge indices (LE order, as squeezed).
    round_challenges: Vec<ChallengeIdx>,
    /// Booleanity cycle challenge indices (BE).
    cycle_challenges_be: Vec<ChallengeIdx>,
}

fn build_stage7(
    p: &Polys,
    params: &ModuleParams,
    ops: &mut Vec<Op>,
    kernels: &mut Vec<KernelDef>,
    ch: &mut ChallengeTable,
    batched_sumchecks: &mut Vec<BatchedSumcheckDef>,
    s6: &Stage6Challenges,
) -> Stage7Challenges {
    let total_d = params.instruction_d + params.bytecode_d + params.ram_d;
    let hw_rounds = params.log_k_chunk;

    // Pre-sumcheck: squeeze γ challenge for batching 3N claims.
    // HammingWeightClaimReductionParams::new → 1 challenge_scalar
    let ch_hw_gamma = ch.add("hw_gamma", ChallengeSource::FiatShamir { after_stage: 6 });
    ops.push(Op::Squeeze {
        challenge: ch_hw_gamma,
    });

    // γ^0, γ^1, ..., γ^{3N-1}
    let ch_hw_gamma_pow: Vec<ChallengeIdx> = (0..3 * total_d)
        .map(|d| {
            let idx = ch.add(
                &format!("hw_gamma_pow_{d}"),
                ChallengeSource::Power {
                    base: ch_hw_gamma,
                    exponent: d,
                },
            );
            ops.push(Op::ComputePower {
                target: idx,
                base: ch_hw_gamma,
                exponent: d as u64,
            });
            idx
        })
        .collect();

    // RA poly IDs in jolt-core order: Instruction, Bytecode, Ram
    let ra_poly_ids: Vec<PolynomialId> = (0..params.instruction_d)
        .map(PolynomialId::InstructionRa)
        .chain((0..params.bytecode_d).map(PolynomialId::BytecodeRa))
        .chain((0..params.ram_d).map(PolynomialId::RamRa))
        .collect();

    // Challenge points for eq table construction.
    //
    // r_cycle (BE): booleanity's cycle challenges from stage 6, reversed.
    // Booleanity instance activates at round (log_k_bytecode - log_k_chunk),
    // its cycle portion spans rounds [log_k_bytecode, log_k_bytecode+log_t).
    let cycle_challenges_be: Vec<ChallengeIdx> = s6.round_challenges
        [params.log_k_bytecode..params.log_k_bytecode + params.log_t]
        .iter()
        .rev()
        .copied()
        .collect();

    // r_addr_bool (BE): booleanity's address challenges from stage 6, reversed.
    // Booleanity's address portion spans rounds [log_k_bytecode - log_k_chunk, log_k_bytecode).
    let addr_bool_challenges_be: Vec<ChallengeIdx> = s6.round_challenges
        [params.log_k_bytecode - params.log_k_chunk..params.log_k_bytecode]
        .iter()
        .rev()
        .copied()
        .collect();

    // Per-RA addr_virt: address portion of each RA's virtualization opening.
    // These use the same challenge indices as the EqProject projections in stage 6.
    let mut addr_virt_challenges_be = Vec::with_capacity(total_d);
    for d in 0..params.instruction_d {
        addr_virt_challenges_be.push(s6.inst_addr_chunks[d].clone());
    }
    for d in 0..params.bytecode_d {
        addr_virt_challenges_be.push(s6.bc_addr_chunks[d].clone());
    }
    for d in 0..params.ram_d {
        addr_virt_challenges_be.push(s6.ram_addr_chunks[d].clone());
    }

    // HW Reduction kernel: fully compiled as Formula + InputBindings.
    //
    // Formula: Σ_d (γ^{3d}·G_d + γ^{3d+1}·G_d·eq_bool + γ^{3d+2}·G_d·eq_virt_d)
    // Input layout: [G_0, ..., G_{N-1}, eq_bool, eq_virt_0, ..., eq_virt_{N-1}]
    // Challenge layout: [γ^0, γ^1, ..., γ^{3N-1}] (mapped via ChallengeIdx)
    let cycle_challenges_be_ret = cycle_challenges_be.clone();

    // Build formula terms
    let mut hw_formula_terms = Vec::with_capacity(3 * total_d);
    let eq_bool_input_idx = total_d as u32; // eq_bool is after all G_d inputs
    for d in 0..total_d {
        let g_input = d as u32;
        let eq_virt_input = (total_d + 1 + d) as u32;
        let gamma_hw = ch_hw_gamma_pow[3 * d].0 as u32;
        let gamma_bool = ch_hw_gamma_pow[3 * d + 1].0 as u32;
        let gamma_virt = ch_hw_gamma_pow[3 * d + 2].0 as u32;

        // γ^{3d} · G_d
        hw_formula_terms.push(ProductTerm {
            coefficient: 1,
            factors: vec![Factor::Challenge(gamma_hw), Factor::Input(g_input)],
        });
        // γ^{3d+1} · G_d · eq_bool
        hw_formula_terms.push(ProductTerm {
            coefficient: 1,
            factors: vec![
                Factor::Challenge(gamma_bool),
                Factor::Input(g_input),
                Factor::Input(eq_bool_input_idx),
            ],
        });
        // γ^{3d+2} · G_d · eq_virt_d
        hw_formula_terms.push(ProductTerm {
            coefficient: 1,
            factors: vec![
                Factor::Challenge(gamma_virt),
                Factor::Input(g_input),
                Factor::Input(eq_virt_input),
            ],
        });
    }

    // Build input bindings: G_d via EqProject, eq_bool/eq_virt via EqTable
    let k_chunk = 1usize << params.log_k_chunk;
    let t_size = 1usize << params.log_t;
    let eq_bool_poly = PolynomialId::BatchEq(100);

    let mut hw_inputs = Vec::with_capacity(2 * total_d + 1);
    // G_0, ..., G_{N-1}: project RA data over cycle dimension
    for (g_poly, ra_poly) in p.hw_g.iter().zip(&ra_poly_ids) {
        hw_inputs.push(InputBinding::EqProject {
            poly: *g_poly,
            source: *ra_poly,
            challenges: cycle_challenges_be.clone(),
            inner_size: k_chunk,
            outer_size: t_size,
        });
    }
    // eq_bool: shared eq table over addr_bool challenges
    hw_inputs.push(InputBinding::EqTable {
        poly: eq_bool_poly,
        challenges: addr_bool_challenges_be.clone(),
    });
    // eq_virt_0, ..., eq_virt_{N-1}: per-dimension eq table over addr_virt challenges
    for (d, addr_virt_ch) in addr_virt_challenges_be.iter().enumerate() {
        hw_inputs.push(InputBinding::EqTable {
            poly: PolynomialId::BatchEq(101 + d),
            challenges: addr_virt_ch.clone(),
        });
    }

    let hw_kernel_idx = kernels.len();
    kernels.push(KernelDef {
        spec: KernelSpec {
            formula: Formula::from_terms(hw_formula_terms),
            num_evals: 3, // degree 2 → evals at {0, 1, 2}
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
            gruen_hint: None,
        },
        inputs: hw_inputs,
        num_rounds: hw_rounds,
        instance_config: None,
    });

    // Batched sumcheck definition (1 instance)
    let batch_idx = BatchIdx(batched_sumchecks.len());
    batched_sumchecks.push(BatchedSumcheckDef {
        instances: vec![BatchedInstance {
            phases: vec![InstancePhase {
                kernel: hw_kernel_idx,
                num_rounds: hw_rounds,
                scalar_captures: vec![],
                segmented: None,
                carry_bindings: vec![],
                pre_activation_ops: vec![],
            }],
            batch_coeff: ChallengeIdx(0),
            first_active_round: 0,
        }],
        input_claims: vec![],
        max_rounds: hw_rounds,
        max_degree: 2,
    });

    // Input claim: Σ_i (γ^{3i}·H_i + γ^{3i+1}·bool_i + γ^{3i+2}·virt_i)
    //
    // After stage 6:
    //   state.evaluations[ra_pid] = booleanity eval (from BooleanityCacheOpenings)
    //   Virt evals stored under BatchEq projection IDs
    let mut hw_input_terms = Vec::new();
    for i in 0..total_d {
        let ra_pid = ra_poly_ids[i];
        let is_ram = i >= params.instruction_d + params.bytecode_d;

        // γ^{3i} · H_i
        if is_ram {
            // H_i = HammingWeight evaluation (from stage 6)
            hw_input_terms.push(ClaimTerm {
                coeff: 1,
                factors: vec![
                    ClaimFactor::Challenge(ch_hw_gamma_pow[3 * i]),
                    ClaimFactor::Eval(p.hamming_weight),
                ],
            });
        } else {
            // H_i = 1 for instruction and bytecode
            hw_input_terms.push(ClaimTerm {
                coeff: 1,
                factors: vec![ClaimFactor::Challenge(ch_hw_gamma_pow[3 * i])],
            });
        }

        // γ^{3i+1} · bool_i = ra_d eval from booleanity (in state.evaluations[ra_pid])
        hw_input_terms.push(ClaimTerm {
            coeff: 1,
            factors: vec![
                ClaimFactor::Challenge(ch_hw_gamma_pow[3 * i + 1]),
                ClaimFactor::Eval(ra_pid),
            ],
        });

        // γ^{3i+2} · virt_i = ra_d eval from virtualization (in BatchEq projection)
        let virt_proj_id = if i < params.instruction_d {
            s6.inst_ra_proj_ids[i]
        } else if i < params.instruction_d + params.bytecode_d {
            s6.bc_raf_proj_ids[i - params.instruction_d]
        } else {
            s6.ram_ra_proj_ids[i - params.instruction_d - params.bytecode_d]
        };
        hw_input_terms.push(ClaimTerm {
            coeff: 1,
            factors: vec![
                ClaimFactor::Challenge(ch_hw_gamma_pow[3 * i + 2]),
                ClaimFactor::Eval(virt_proj_id),
            ],
        });
    }

    let hw_input_claim = ClaimFormula {
        terms: hw_input_terms,
    };
    ops.push(Op::AbsorbInputClaim {
        formula: hw_input_claim.clone(),
        tag: DomainSeparator::SumcheckClaim,
        batch: batch_idx,
        instance: InstanceIdx(0),
        inactive_scale_bits: 0,
    });

    // Batching coefficient (1 instance → 1 coefficient)
    let ch_batch = ch.add(
        "stage7_batch0",
        ChallengeSource::FiatShamir { after_stage: 6 },
    );
    ops.push(Op::Squeeze {
        challenge: ch_batch,
    });
    batched_sumchecks[batch_idx.0].instances[0].batch_coeff = ch_batch;
    batched_sumchecks[batch_idx.0].input_claims = vec![hw_input_claim];

    // Batched sumcheck rounds (log_k_chunk rounds, degree 2)
    let num_coeffs_fn = |_round: usize| -> usize { 3 }; // degree 2 → 3 evals (degree + 1)
    let s7_round_ch = emit_unrolled_batched_rounds(
        ops,
        kernels,
        batched_sumchecks,
        ch,
        batch_idx,
        hw_rounds,
        num_coeffs_fn,
        VerifierStageIndex(6),
        "s7_r",
        0,
        None,
    );

    // Post-sumcheck: final bind + extract G evaluations.
    // The last round's challenge was squeezed but never bound into the kernel inputs.
    // Apply it now to finish reducing all buffers to 1 element each.
    ops.push(Op::InstanceBind {
        batch: batch_idx,
        instance: InstanceIdx(0),
        kernel: hw_kernel_idx,
        challenge: s7_round_ch[hw_rounds - 1],
    });

    // Extract G_i evaluations (buffers are now 1-element after all binds).
    let g_poly_ids: Vec<PolynomialId> = (0..total_d).map(|i| p.hw_g[i]).collect();
    for &g_pid in &g_poly_ids {
        ops.push(Op::Evaluate {
            poly: g_pid,
            mode: EvalMode::FullyBound,
        });
    }

    // Flush G evaluations to transcript (one per RA polynomial).
    ops.push(Op::AbsorbEvals {
        polys: g_poly_ids,
        tag: DomainSeparator::OpeningClaim,
    });

    Stage7Challenges {
        round_challenges: s7_round_ch,
        cycle_challenges_be: cycle_challenges_be_ret,
    }
}

fn build_stage8(p: &Polys, params: &ModuleParams, ops: &mut Vec<Op>, s7: &Stage7Challenges) {
    ops.push(Op::BeginStage { index: 7 });

    // Opening point = [r_address_BE, r_cycle_BE].
    // r_address_BE: stage 7 round challenges reversed (LE→BE).
    // r_cycle_BE: booleanity cycle challenges (already BE from stage 7 construction).
    let r_addr_be: Vec<ChallengeIdx> = s7.round_challenges.iter().rev().copied().collect();
    let opening_point: Vec<ChallengeIdx> = r_addr_be
        .iter()
        .chain(s7.cycle_challenges_be.iter())
        .copied()
        .collect();
    let committed_num_vars = params.log_k_chunk + params.log_t;

    // Claim order matches jolt-core prove_stage8:
    //   RamInc, RdInc (dense, scaled by eq(r_addr, 0)),
    //   InstructionRa[0..d], BytecodeRa[0..d], RamRa[0..d] (sparse, scale=1)

    // Dense polys: scale eval by ∏(1 − r_addr_i)
    ops.push(Op::ScaleEval {
        poly: p.ram_inc,
        factor_challenges: r_addr_be.clone(),
    });
    ops.push(Op::ScaleEval {
        poly: p.rd_inc,
        factor_challenges: r_addr_be.clone(),
    });

    // Collect all claims with explicit opening point
    ops.push(Op::CollectOpeningClaimAt {
        poly: p.ram_inc,
        point_challenges: opening_point.clone(),
        committed_num_vars: Some(committed_num_vars),
    });
    ops.push(Op::CollectOpeningClaimAt {
        poly: p.rd_inc,
        point_challenges: opening_point.clone(),
        committed_num_vars: Some(committed_num_vars),
    });

    // Copy G_i(rho) evaluations from HammingG slots to RA poly slots.
    // HwReductionCacheOpenings stored G_i values as HammingG(i), but
    // CollectOpeningClaimAt reads from the RA poly ID (e.g. InstructionRa(d)).
    // Without this copy, state.evaluations[InstructionRa(d)] still holds
    // the stale Booleanity evaluation from stage 6.
    let mut g_idx = 0;
    for d in 0..params.instruction_d {
        ops.push(Op::AliasEval {
            from: p.hw_g[g_idx],
            to: PolynomialId::InstructionRa(d),
        });
        g_idx += 1;
    }
    for d in 0..params.bytecode_d {
        ops.push(Op::AliasEval {
            from: p.hw_g[g_idx],
            to: PolynomialId::BytecodeRa(d),
        });
        g_idx += 1;
    }
    for d in 0..params.ram_d {
        ops.push(Op::AliasEval {
            from: p.hw_g[g_idx],
            to: PolynomialId::RamRa(d),
        });
        g_idx += 1;
    }

    for d in 0..params.instruction_d {
        ops.push(Op::CollectOpeningClaimAt {
            poly: PolynomialId::InstructionRa(d),
            point_challenges: opening_point.clone(),
            committed_num_vars: None,
        });
    }
    for d in 0..params.bytecode_d {
        ops.push(Op::CollectOpeningClaimAt {
            poly: PolynomialId::BytecodeRa(d),
            point_challenges: opening_point.clone(),
            committed_num_vars: None,
        });
    }
    for d in 0..params.ram_d {
        ops.push(Op::CollectOpeningClaimAt {
            poly: PolynomialId::RamRa(d),
            point_challenges: opening_point.clone(),
            committed_num_vars: None,
        });
    }

    // RLC reduction + PCS opening proof + post-proof binding
    ops.push(Op::ReduceOpenings);
    ops.push(Op::Open);
    ops.push(Op::BindOpeningInputs {
        point_challenges: opening_point,
    });
}

/// Build the BytecodeReadRaf input_claim formula.
///
/// This is the most complex input claim in the protocol: it references
/// evaluations from ALL five previous stages, weighted by 6 sets of
/// gamma challenges. Uses StagedEval for evaluations that were
/// overwritten by later stages.
fn build_bytecode_read_raf_claim(
    p: &Polys,
    ch_gamma: ChallengeIdx,    // gamma_powers(8) base
    ch_s1_gamma: ChallengeIdx, // stage1_gammas(16) base
    ch_s2_gamma: ChallengeIdx, // stage2_gammas(4) base
    ch_s3_gamma: ChallengeIdx, // stage3_gammas(9) base
    ch_s4_gamma: ChallengeIdx, // stage4_gammas(3) base
    ch_s5_gamma: ChallengeIdx, // stage5_gammas(43) base
) -> ClaimFormula {
    let mut terms = Vec::new();

    // Helper: build factors for gamma^s * stage_gamma^j * eval(poly)
    let term = |gamma_pow: usize,
                stage_gamma: ChallengeIdx,
                stage_gamma_pow: usize,
                poly: PolynomialId|
     -> ClaimTerm {
        let mut factors = Vec::new();
        for _ in 0..gamma_pow {
            factors.push(ClaimFactor::Challenge(ch_gamma));
        }
        for _ in 0..stage_gamma_pow {
            factors.push(ClaimFactor::Challenge(stage_gamma));
        }
        factors.push(ClaimFactor::Eval(poly));
        ClaimTerm { coeff: 1, factors }
    };

    // Helper: build factors for gamma^rv_idx * stage_gamma^j * staged_eval(poly, stage)
    let term_staged = |rv_idx: usize,
                       stage_gamma: ChallengeIdx,
                       gamma_pow: usize,
                       poly: PolynomialId,
                       stage: usize|
     -> ClaimTerm {
        let mut factors = Vec::new();
        for _ in 0..rv_idx {
            factors.push(ClaimFactor::Challenge(ch_gamma));
        }
        for _ in 0..gamma_pow {
            factors.push(ClaimFactor::Challenge(stage_gamma));
        }
        factors.push(ClaimFactor::StagedEval { poly, stage });
        ClaimTerm { coeff: 1, factors }
    };

    // rv_claim_1 (gamma^0): stage1_gamma[0]*UnexpandedPC + stage1_gamma[1]*Imm
    //   + stage1_gamma[2+i]*OpFlag(i) for i in 0..14
    // Evaluations from Stage 1 (stage index 0). Some polys are overwritten
    // by later stages, so we use StagedEval to reference the Stage 1 value.
    terms.push(term_staged(0, ch_s1_gamma, 0, p.unexpanded_pc, 0)); // UnexpandedPC@Stage1
    terms.push(term_staged(0, ch_s1_gamma, 1, p.imm, 0)); // Imm@Stage1
                                                          // OpFlag(0..4): not overwritten, still hold Stage 1 values
    for i in 0..5 {
        terms.push(term(0, ch_s1_gamma, 2 + i, p.op_flags[i]));
    }
    terms.push(term_staged(0, ch_s1_gamma, 7, p.op_flags[5], 0)); // Jump@Stage1
    terms.push(term_staged(0, ch_s1_gamma, 8, p.op_flags[6], 0)); // WriteLookupOutputToRD@Stage1
    terms.push(term_staged(0, ch_s1_gamma, 9, p.op_flags[7], 0)); // VirtualInstruction@Stage1
                                                                  // OpFlag(8..11): not overwritten
    for i in 8..12 {
        terms.push(term(0, ch_s1_gamma, 2 + i, p.op_flags[i]));
    }
    terms.push(term_staged(0, ch_s1_gamma, 14, p.op_flags[12], 0)); // IsFirstInSequence@Stage1
    terms.push(term(0, ch_s1_gamma, 15, p.op_flags[13])); // IsLastInSequence (not overwritten)

    // rv_claim_2 (gamma^1): stage2_gamma[0]*Jump + stage2_gamma[1]*Branch
    //   + stage2_gamma[2]*WriteLookupOutputToRD + stage2_gamma[3]*VirtualInstruction
    // OpFlag(5) Jump: Stage 2 value is current (not overwritten by Stage 3+)
    // OpFlag(6) WriteLookupOutputToRD: Stage 2 value is current
    // OpFlag(7) VirtualInstruction: Stage 2 value was overwritten by Stage 3
    terms.push(term(1, ch_s2_gamma, 0, p.op_flags[5])); // Jump@S2 (current)
    terms.push(term(1, ch_s2_gamma, 1, p.inst_flag_branch)); // Branch@S2 (current)
    terms.push(term(1, ch_s2_gamma, 2, p.op_flags[6])); // WriteLookupOutputToRD@S2
    terms.push(term_staged(1, ch_s2_gamma, 3, p.op_flags[7], 1)); // VirtualInstruction@Stage2

    // rv_claim_3 (gamma^2): 9 terms from Stage 3
    // All Stage 3 evaluations are current (Stage 3 is the last writer).
    terms.push(term(2, ch_s3_gamma, 0, p.imm)); // Imm@S3 (current)
    terms.push(term(2, ch_s3_gamma, 1, p.unexpanded_pc)); // UnexpandedPC@S3 (current)
    terms.push(term(2, ch_s3_gamma, 2, p.inst_flag_left_is_rs1)); // LeftIsRs1@S3
    terms.push(term(2, ch_s3_gamma, 3, p.inst_flag_left_is_pc)); // LeftIsPC@S3
    terms.push(term(2, ch_s3_gamma, 4, p.inst_flag_right_is_rs2)); // RightIsRs2@S3
    terms.push(term(2, ch_s3_gamma, 5, p.inst_flag_right_is_imm)); // RightIsImm@S3
    terms.push(term(2, ch_s3_gamma, 6, p.inst_flag_is_noop)); // IsNoop@S3
    terms.push(term(2, ch_s3_gamma, 7, p.op_flags[7])); // VirtualInstruction@S3 (current)
    terms.push(term(2, ch_s3_gamma, 8, p.op_flags[12])); // IsFirstInSequence@S3 (current)

    // rv_claim_4 (gamma^3): 3 terms from Stage 4 RegistersRW
    terms.push(term(3, ch_s4_gamma, 0, p.reg_wa)); // RdWa@S4
    terms.push(term(3, ch_s4_gamma, 1, p.reg_ra_rs1)); // Rs1Ra@S4
    terms.push(term(3, ch_s4_gamma, 2, p.reg_ra_rs2)); // Rs2Ra@S4

    // rv_claim_5 (gamma^4): 2 + NUM_LOOKUP_TABLES terms from Stage 5
    // RdWa @ RegistersValEvaluation (Stage 5) — stored as BatchEq(12) in the module builder
    let rd_wa_val_eval = PolynomialId::BatchEq(12);
    terms.push(term(4, ch_s5_gamma, 0, rd_wa_val_eval));
    // InstructionRafFlag @ InstructionReadRaf (Stage 5)
    terms.push(term(4, ch_s5_gamma, 1, PolynomialId::InstructionRafFlag));
    // LookupTableFlag(0..41) @ InstructionReadRaf (Stage 5)
    for i in 0..NUM_LOOKUP_TABLES {
        terms.push(term(
            4,
            ch_s5_gamma,
            2 + i,
            PolynomialId::LookupTableFlag(i),
        ));
    }

    // gamma^5 * raf_claim: PC @ SpartanOuter (Stage 1, stage index 0)
    {
        let mut factors = Vec::new();
        for _ in 0..5 {
            factors.push(ClaimFactor::Challenge(ch_gamma));
        }
        factors.push(ClaimFactor::StagedEval {
            poly: p.pc,
            stage: 0,
        }); // PC@Stage1
        terms.push(ClaimTerm { coeff: 1, factors });
    }

    // gamma^6 * raf_shift_claim: PC @ SpartanShift (Stage 3, current value)
    {
        let mut factors = Vec::new();
        for _ in 0..6 {
            factors.push(ClaimFactor::Challenge(ch_gamma));
        }
        factors.push(ClaimFactor::Eval(p.pc)); // PC@S3 (current)
        terms.push(ClaimTerm { coeff: 1, factors });
    }

    // gamma^7 (constant entry_gamma term, no eval)
    {
        let factors: Vec<ClaimFactor> = (0..7).map(|_| ClaimFactor::Challenge(ch_gamma)).collect();
        terms.push(ClaimTerm { coeff: 1, factors });
    }

    ClaimFormula { terms }
}

/// Verifier schedule for Stage 4.
fn build_verifier_stage4_ops(
    _p: &Polys,
    params: &ModuleParams,
    ch: &ChallengeTable,
) -> Vec<VerifierOp> {
    let mut ops = vec![VerifierOp::BeginStage];

    // Stage 4 pre-sumcheck challenges: 3 gammas + 3 batching = 6.
    // Order MUST mirror prover (lines 3382, 3393, 3402-separator, 3407, 3574, 3578, 3582):
    //   gamma_registers_rw, gamma_fr_rw, [domain separator], gamma_ram_val_check,
    //   stage4_batch0, stage4_batch1, stage4_batch2.
    let num_s4_challenges = 6;
    let s4_ch_base = ch.decls.len() - num_s4_challenges;

    // γ_registers_rw
    ops.push(VerifierOp::Squeeze {
        challenge: ChallengeIdx(s4_ch_base),
    });

    // γ_fr_rw — BN254 Fr coprocessor Read-Write gamma. Mirrors integer
    // gamma_registers_rw; consumed by FieldRegReadWriteChecking instance.
    ops.push(VerifierOp::Squeeze {
        challenge: ChallengeIdx(s4_ch_base + 1),
    });

    // Domain separator before RAM val check gamma
    ops.push(VerifierOp::AppendDomainSeparator {
        tag: DomainSeparator::RamValCheckGamma,
    });

    // γ_ram_val_check
    ops.push(VerifierOp::Squeeze {
        challenge: ChallengeIdx(s4_ch_base + 2),
    });

    // 3 batching coefficients (RegistersRW, RamValCheck, FieldRegRW)
    ops.push(VerifierOp::Squeeze {
        challenge: ChallengeIdx(s4_ch_base + 3),
    });
    ops.push(VerifierOp::Squeeze {
        challenge: ChallengeIdx(s4_ch_base + 4),
    });
    ops.push(VerifierOp::Squeeze {
        challenge: ChallengeIdx(s4_ch_base + 5),
    });

    ops
}

/// Verifier schedule for Stage 5.
///
/// **PLACEHOLDER STUB.** Mirrors the existing Stage 3/4 stub pattern: emits
/// `BeginStage` plus per-challenge `Squeeze` ops to keep the post-stage
/// challenge-table indices in sync, but does NOT emit `AbsorbRoundPoly`,
/// `VerifySumcheck`, `RecordEvals`, `AbsorbEvals`, or `CheckOutput` — so the
/// Fiat-Shamir transcript is misaligned across the round-poly absorptions and
/// no actual sumcheck verification happens.
///
/// The Stage 3/4 stubs share this limitation; full verifier implementations
/// for stages 3-5 are a separate workstream (see `specs/fr-v2-audit.md`
/// findings C2/C3/C4 — pre-existing on refactor-crates).
///
/// Stage 5 prover squeezes (counted from `build_stage5`):
///   - `gamma_instruction_read_raf` + `gamma_ram_ra_reduction` = 2 gammas
///   - `stage5_batch0..3` = 4 batching coefficients
///   - `stage5_max_rounds = LOG_K_INSTRUCTION + log_t` round challenges
fn build_verifier_stage5_ops(
    _p: &Polys,
    params: &ModuleParams,
    ch: &ChallengeTable,
) -> Vec<VerifierOp> {
    let mut ops = vec![VerifierOp::BeginStage];

    // 2 gammas + 4 batching + (LOG_K_INSTRUCTION + log_t) round challenges.
    // MUST match the prover's stage-5 squeezes exactly — any miscount shifts
    // `s5_ch_base` and misaligns the absorbed challenges.
    let num_s5_challenges = 2 + 4 + LOG_K_INSTRUCTION + params.log_t;
    let s5_ch_base = ch.decls.len() - num_s5_challenges;

    for i in 0..num_s5_challenges {
        ops.push(VerifierOp::Squeeze {
            challenge: ChallengeIdx(s5_ch_base + i),
        });
    }

    ops
}

/// Verifier schedule for Stage 3.
fn build_verifier_stage3_ops(
    _p: &Polys,
    params: &ModuleParams,
    ch: &ChallengeTable,
) -> Vec<VerifierOp> {
    let mut ops = vec![VerifierOp::BeginStage];

    // Stage 3 challenges: 4 gammas + 4 batching + log_t round = 8 + log_t total.
    // Gammas: shift, instruction_input, registers, fr_cr.
    // Batching: one per instance (Shift, InstructionInput, RegistersCR, FieldRegCR).
    // MUST match the prover's stage-3 squeezes exactly — any miscount shifts
    // `s3_ch_base` and misaligns every absorbed challenge.
    let num_s3_challenges = 4 + 4 + params.log_t;
    let s3_ch_base = ch.decls.len() - num_s3_challenges;

    for i in 0..num_s3_challenges {
        ops.push(VerifierOp::Squeeze {
            challenge: ChallengeIdx(s3_ch_base + i),
        });
    }

    ops
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
    let ch_tau_high = ChallengeIdx(s2_ch_base); // 55
    let ch_product_r0 = ChallengeIdx(s2_ch_base + 1); // 56
    let ch_gamma_rw = ChallengeIdx(s2_ch_base + 2); // 57
    let ch_gamma_instruction = ChallengeIdx(s2_ch_base + 3); // 58
    let ch_r_address_base = ChallengeIdx(s2_ch_base + 4); // 59..78

    // 15 unique openings (duplicates aliased — see prover side comment).
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
        // InstructionLookupsClaimReduction (2 new)
        p.left_lookup_operand,
        p.right_lookup_operand,
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

    // Stage 1 cycle challenge indices (excluding group variable).
    // The outer remaining has log_t+1 rounds: [0] = group, [1..] = cycle.
    // r_cycle is challenges[1..] reversed (big-endian), with log_t entries.
    let stage1_round_base = params.num_tau + 2; // τ + r0 + batch
    let stage1_cycle_challenges: Vec<ChallengeIdx> = (stage1_round_base + 1
        ..stage1_round_base + params.outer_remaining_rounds)
        .rev()
        .map(ChallengeIdx)
        .collect();

    // Evaluation list positions in the deduplicated 15-entry flush layout.
    // Used by StageEval in output_check formulas.
    // sp.evals[0] = product_uniskip_eval (recorded before the batch).
    // sp.evals[1..16] = the 15 unique batched-sumcheck final-bind evals.
    //
    // Aliased entries (LookupOutput, LeftInstructionInput, RightInstructionInput)
    // point to the canonical position shared with an earlier instance — this
    // matches jolt-core's `cache_openings` deduplication.
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
                                                  // InstructionClaimReduction (aliased to earlier slots + 2 new):
    const SE_LOOKUP_OUT_INST: usize = SE_LOOKUP_OUT_PROD; // aliased
    const SE_LEFT_LOOKUP_OP: usize = SE_BASE + 11;
    const SE_RIGHT_LOOKUP_OP: usize = SE_BASE + 12;
    const SE_L_INST_CR: usize = SE_L_INST; // aliased
    const SE_R_INST_CR: usize = SE_R_INST; // aliased
                                           // RafEvaluation (1):
    const SE_RAM_RAF_RA: usize = SE_BASE + 13;
    // OutputCheck (1):
    const SE_VAL_FINAL: usize = SE_BASE + 14;

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
    let prod_lk = ClaimFactor::LagrangeKernelDomain {
        tau_challenge: ch_tau_high,
        at_challenge: ch_product_r0,
        domain_size: params.product_uniskip_domain,
        domain_start: -((params.product_uniskip_domain as i64 - 1) / 2),
    };
    let prod_eq = ClaimFactor::EqEval {
        challenges: stage1_cycle_challenges.clone(),
        at_stage: VerifierStageIndex(1),
    };
    let w = |k: usize| ClaimFactor::LagrangeWeight {
        challenge: ch_product_r0,
        domain_size: params.product_uniskip_domain,
        domain_start: -((params.product_uniskip_domain as i64 - 1) / 2),
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
    let output_eq = ClaimFactor::EqEval {
        challenges: (ch_r_address_base.0..ch_r_address_base.0 + params.log_k_ram)
            .map(ChallengeIdx)
            .collect(),
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

    // Assemble all 5 instances
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
    for c in ch_r_address_base.0..ch_r_address_base.0 + params.log_k_ram {
        ops.push(VerifierOp::Squeeze {
            challenge: ChallengeIdx(c),
        });
    }
    // Batch coefficient challenge indices for the 5 instances.
    let ch_batch_base = s2_ch_base + 2 + 1 + 1 + params.log_k_ram; // after τ_high, r0, γ_rw, γ_instruction, r_address
    let batch_challenges: Vec<ChallengeIdx> = (0..params.stage2_num_instances)
        .map(|i| ChallengeIdx(ch_batch_base + i))
        .collect();
    // Stage 2 sumcheck round challenges: after batch coefficients.
    let stage2_round_base = ch_batch_base + params.stage2_num_instances;
    let stage2_max_rounds = instances.iter().map(|i| i.num_rounds).max().unwrap_or(0);
    let stage2_round_slots: Vec<ChallengeIdx> = (stage2_round_base
        ..stage2_round_base + stage2_max_rounds)
        .map(ChallengeIdx)
        .collect();
    ops.push(VerifierOp::VerifySumcheck {
        instances: instances.clone(),
        stage: 1,
        batch_challenges: batch_challenges.clone(),
        claim_tag: Some(DomainSeparator::SumcheckClaim),
        sumcheck_challenge_slots: stage2_round_slots,
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

// Stats printing

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
            Op::Evaluate { .. } => counts[1] += 1,
            Op::Bind { .. } => counts[2] += 1,
            Op::LagrangeProject { .. }
            | Op::DuplicateInterleave { .. }
            | Op::RegroupConstraints { .. } => counts[13] += 1,
            Op::Commit { .. } | Op::CommitStreaming { .. } => counts[3] += 1,
            Op::ReduceOpenings => counts[4] += 1,
            Op::Open | Op::BindOpeningInputs { .. } => counts[5] += 1,
            Op::Preamble => counts[6] += 1,
            Op::BeginStage { .. } => counts[12] += 1,
            Op::AbsorbRoundPoly { .. } => counts[7] += 1,
            Op::RecordEvals { .. } | Op::AbsorbEvals { .. } => counts[8] += 1,
            Op::AbsorbInputClaim { .. } => counts[15] += 1,
            Op::Squeeze { .. } | Op::ComputePower { .. } => counts[9] += 1,
            Op::CollectOpeningClaim { .. }
            | Op::CollectOpeningClaimAt { .. }
            | Op::ScaleEval { .. } => counts[10] += 1,
            Op::ReleaseDevice { .. } => counts[11] += 1,
            Op::ReleaseHost { .. } => counts[11] += 1,
            Op::AppendDomainSeparator { .. } => counts[9] += 1,
            Op::EvaluatePreprocessed { .. } => counts[9] += 1,
            Op::AliasEval { .. } => counts[9] += 1,
            Op::BatchRoundBegin { .. }
            | Op::BatchInactiveContribution { .. }
            | Op::Materialize { .. }
            | Op::MaterializeUnlessFresh { .. }
            | Op::MaterializeIfAbsent { .. }
            | Op::MaterializeSegmentedOuterEq { .. }
            | Op::InstanceBindPreviousPhase { .. }
            | Op::CaptureScalar { .. }
            | Op::InstanceReduce { .. }
            | Op::InstanceSegmentedReduce { .. }
            | Op::InstanceBind { .. }
            | Op::BindCarryBuffers { .. }
            | Op::BatchAccumulateInstance { .. }
            | Op::BatchRoundFinalize { .. }
            | Op::ExpandingTableUpdate { .. }
            | Op::CheckpointEvalBatch { .. }
            | Op::InitInstanceWeights { .. }
            | Op::UpdateInstanceWeights { .. }
            | Op::SuffixScatter { .. }
            | Op::QBufferScatter { .. }
            | Op::MaterializePBuffers { .. }
            | Op::InitExpandingTable { .. }
            | Op::ReadCheckingReduce { .. }
            | Op::RafReduce { .. }
            | Op::MaterializeRA { .. }
            | Op::MaterializeCombinedVal { .. }
            | Op::WeightedSum { .. } => counts[14] += 1,
        }
    }
    eprintln!("\n=== Op Counts ===");
    eprintln!("  -- Compute --");
    eprintln!("  SumcheckRound:         {}", counts[0]);
    eprintln!("  GranularBatchOps:     {}", counts[14]);
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
                bdef.max_rounds,
                bdef.max_degree,
                bdef.instances.len()
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
                    "    [{j}] phases=[{}], batch_coeff={:?}, first_active={}, total_rounds={}",
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

/// Compute prefix-suffix protocol data from jolt-instructions.
///
/// Returns (combine_entries, suffix_at_empty, suffixes_per_table, suffix_ops, checkpoint_rules).
#[allow(clippy::type_complexity)]
fn compute_prefix_suffix_data() -> (
    Vec<CombineEntry>,
    Vec<Vec<u64>>,
    Vec<usize>,
    Vec<Vec<SuffixOp>>,
    Vec<CheckpointRule>,
    Vec<PrefixMleRule>,
) {
    const XLEN: usize = 64;
    let all_kinds = all_table_kinds();
    let mut combine_entries = Vec::new();
    let mut suffix_at_empty = Vec::with_capacity(all_kinds.len());
    let mut suffixes_per_table = Vec::with_capacity(all_kinds.len());
    let mut suffix_ops = Vec::with_capacity(all_kinds.len());

    for (t_idx, &kind) in all_kinds.iter().enumerate() {
        let table = LookupTables::<XLEN>::from(kind);
        let suffixes = table.suffixes();
        let empty = LookupBits::new(0, 0);
        suffix_at_empty.push(
            suffixes
                .iter()
                .map(|s| s.suffix_mle::<XLEN>(empty))
                .collect(),
        );
        suffixes_per_table.push(suffixes.len());
        suffix_ops.push(suffixes.iter().map(suffix_to_op::<XLEN>).collect());
        for entry in table.combine_entries() {
            combine_entries.push(CombineEntry {
                table_idx: t_idx,
                prefix_idx: entry.prefix.map(|p| p as usize),
                suffix_local_idx: entry.suffix_idx,
                coefficient: entry.coefficient,
            });
        }
    }

    let checkpoint_rules: Vec<CheckpointRule> =
        ALL_PREFIXES.iter().map(prefix_to_rule::<XLEN>).collect();

    let prefix_mle_rules: Vec<PrefixMleRule> = ALL_PREFIXES
        .iter()
        .map(prefix_to_mle_rule::<XLEN>)
        .collect();

    (
        combine_entries,
        suffix_at_empty,
        suffixes_per_table,
        suffix_ops,
        checkpoint_rules,
        prefix_mle_rules,
    )
}

fn prefix_to_rule<const XLEN: usize>(prefix: &Prefixes) -> CheckpointRule {
    use BilinearExpr::{
        AntiXY, AntiYX, EqBit, NorBit, OneMinusX, OneMinusY, OnePlusY, OrBit, Product, XorBit, X, Y,
    };
    use CheckpointAction::{
        AddTwoTerm, AddWeighted, Const, DepAdd, DepAddWeighted, Hybrid, Mul, Null, Passthrough,
        Pow2DoubleMul, Pow2Init, Rev8WAdd, Set, SetScaled, SignExtAccum,
    };
    use DefaultVal::{Custom, One, Zero};
    use RoundGuard::{JEq, JGe, JGt, JLt, SuffixLenNonZero};
    use WeightFn::{LinearJ, LinearJMinusOne, Positional};

    let pos = |rot: u32, wb: u32, off: usize| Positional {
        rotation: rot,
        word_bits: wb,
        j_offset: off,
    };

    match prefix {
        // === Simple multiplicative: cp *= expr ===
        Prefixes::Eq => CheckpointRule {
            default: One,
            cases: vec![],
            fallback: Mul(EqBit),
        },
        Prefixes::LeftOperandIsZero => CheckpointRule {
            default: One,
            cases: vec![],
            fallback: Mul(OneMinusX),
        },
        Prefixes::RightOperandIsZero => CheckpointRule {
            default: One,
            cases: vec![],
            fallback: Mul(OneMinusY),
        },
        Prefixes::DivByZero => CheckpointRule {
            default: One,
            cases: vec![],
            fallback: Mul(AntiXY),
        },
        Prefixes::LeftShiftHelper => CheckpointRule {
            default: One,
            cases: vec![],
            fallback: Mul(OnePlusY),
        },

        // === Additive with positional shift: cp += 2^shift * expr ===
        Prefixes::And => CheckpointRule {
            default: Zero,
            cases: vec![],
            fallback: AddWeighted {
                weight: pos(0, XLEN as u32, 0),
                expr: Product,
            },
        },
        Prefixes::Andn => CheckpointRule {
            default: Zero,
            cases: vec![],
            fallback: AddWeighted {
                weight: pos(0, XLEN as u32, 0),
                expr: AntiYX,
            },
        },
        Prefixes::Or => CheckpointRule {
            default: Zero,
            cases: vec![],
            fallback: AddWeighted {
                weight: pos(0, XLEN as u32, 0),
                expr: OrBit,
            },
        },
        Prefixes::Xor => CheckpointRule {
            default: Zero,
            cases: vec![],
            fallback: AddWeighted {
                weight: pos(0, XLEN as u32, 0),
                expr: XorBit,
            },
        },
        Prefixes::RightOperand => CheckpointRule {
            default: Zero,
            cases: vec![],
            fallback: AddWeighted {
                weight: pos(0, XLEN as u32, 0),
                expr: Y,
            },
        },

        // XorRot variants (64-bit word)
        Prefixes::XorRot16 => CheckpointRule {
            default: Zero,
            cases: vec![],
            fallback: AddWeighted {
                weight: pos(16, XLEN as u32, 0),
                expr: XorBit,
            },
        },
        Prefixes::XorRot24 => CheckpointRule {
            default: Zero,
            cases: vec![],
            fallback: AddWeighted {
                weight: pos(24, XLEN as u32, 0),
                expr: XorBit,
            },
        },
        Prefixes::XorRot32 => CheckpointRule {
            default: Zero,
            cases: vec![],
            fallback: AddWeighted {
                weight: pos(32, XLEN as u32, 0),
                expr: XorBit,
            },
        },
        Prefixes::XorRot63 => CheckpointRule {
            default: Zero,
            cases: vec![],
            fallback: AddWeighted {
                weight: pos(63, XLEN as u32, 0),
                expr: XorBit,
            },
        },

        // XorRotW variants (32-bit word, j >= XLEN guard)
        Prefixes::XorRotW7 => CheckpointRule {
            default: Zero,
            cases: vec![(JLt(XLEN), Const(Zero))],
            fallback: AddWeighted {
                weight: pos(7, 32, XLEN),
                expr: XorBit,
            },
        },
        Prefixes::XorRotW8 => CheckpointRule {
            default: Zero,
            cases: vec![(JLt(XLEN), Const(Zero))],
            fallback: AddWeighted {
                weight: pos(8, 32, XLEN),
                expr: XorBit,
            },
        },
        Prefixes::XorRotW12 => CheckpointRule {
            default: Zero,
            cases: vec![(JLt(XLEN), Const(Zero))],
            fallback: AddWeighted {
                weight: pos(12, 32, XLEN),
                expr: XorBit,
            },
        },
        Prefixes::XorRotW16 => CheckpointRule {
            default: Zero,
            cases: vec![(JLt(XLEN), Const(Zero))],
            fallback: AddWeighted {
                weight: pos(16, 32, XLEN),
                expr: XorBit,
            },
        },

        // === Two-term additive: cp += 2^a*rx + 2^b*ry ===
        Prefixes::UpperWord => CheckpointRule {
            default: Zero,
            cases: vec![(JGe(XLEN), Passthrough)],
            fallback: AddTwoTerm {
                x_weight: LinearJ { base: XLEN },
                y_weight: LinearJMinusOne { base: XLEN },
            },
        },
        Prefixes::LowerWord => CheckpointRule {
            default: Zero,
            cases: vec![(JLt(XLEN), Null)],
            fallback: AddTwoTerm {
                x_weight: LinearJ { base: 2 * XLEN },
                y_weight: LinearJMinusOne { base: 2 * XLEN },
            },
        },
        Prefixes::LowerHalfWord => CheckpointRule {
            default: Zero,
            cases: vec![(JLt(XLEN + XLEN / 2), Null)],
            fallback: AddTwoTerm {
                x_weight: LinearJ { base: 2 * XLEN },
                y_weight: LinearJMinusOne { base: 2 * XLEN },
            },
        },

        // === Dependent additive: cp += dep_cp * expr ===
        Prefixes::LessThan => CheckpointRule {
            default: Zero,
            cases: vec![],
            fallback: DepAdd {
                dep: Prefixes::Eq as usize,
                dep_default: One,
                expr: AntiXY,
            },
        },

        // === Dependent additive with shift: cp += dep_cp * 2^shift * expr ===
        Prefixes::LeftShift => CheckpointRule {
            default: Zero,
            cases: vec![],
            fallback: DepAddWeighted {
                dep: Prefixes::LeftShiftHelper as usize,
                dep_default: One,
                weight: pos(0, XLEN as u32, 0),
                expr: AntiYX,
            },
        },

        // === Hybrid: cp = cp*mul + add ===
        Prefixes::RightShift => CheckpointRule {
            default: Zero,
            cases: vec![],
            fallback: Hybrid {
                mul: OnePlusY,
                add: Product,
            },
        },

        // === Round-specific set, else passthrough ===
        Prefixes::LeftOperandMsb => CheckpointRule {
            default: Zero,
            cases: vec![(JEq(1), Set(X))],
            fallback: Passthrough,
        },
        Prefixes::RightOperandMsb => CheckpointRule {
            default: Zero,
            cases: vec![(JEq(1), Set(Y))],
            fallback: Passthrough,
        },
        Prefixes::TwoLsb => CheckpointRule {
            default: One,
            cases: vec![(JEq(2 * XLEN - 1), Set(NorBit))],
            fallback: Passthrough,
        },
        Prefixes::SignExtensionUpperHalf => CheckpointRule {
            default: Zero,
            cases: vec![(
                JEq(XLEN + XLEN / 2 + 1),
                SetScaled {
                    coeff: (((1i128 << (XLEN / 2)) - 1) << (XLEN / 2)),
                    expr: X,
                },
            )],
            fallback: Passthrough,
        },
        Prefixes::SignExtensionRightOperand => CheckpointRule {
            default: Zero,
            cases: vec![(
                JEq(XLEN + 1),
                SetScaled {
                    coeff: (1i128 << XLEN) - (1i128 << (XLEN / 2)),
                    expr: Y,
                },
            )],
            fallback: Passthrough,
        },

        // === Round-specific set, else constant ===
        Prefixes::Lsb => CheckpointRule {
            default: One,
            cases: vec![(JEq(2 * XLEN - 1), Set(Y))],
            fallback: Const(One),
        },

        // === Multiplicative with j==1 init override ===
        Prefixes::PositiveRemainderEqualsDivisor => CheckpointRule {
            default: One,
            cases: vec![(JEq(1), Set(NorBit))],
            fallback: Mul(EqBit),
        },
        Prefixes::NegativeDivisorEqualsRemainder => CheckpointRule {
            default: One,
            cases: vec![(JEq(1), Set(Product))],
            fallback: Mul(EqBit),
        },

        // === NegDivZeroRem: j==1 → set, else → cp*(1-rx) ===
        Prefixes::NegativeDivisorZeroRemainder => CheckpointRule {
            default: Zero,
            cases: vec![(JEq(1), Set(AntiXY))],
            fallback: Mul(OneMinusX),
        },

        // === Dependent additive with j==1 and j==3 overrides ===
        Prefixes::PositiveRemainderLessThanDivisor => CheckpointRule {
            default: Zero,
            cases: vec![(JEq(1), Set(NorBit)), (JEq(3), Mul(AntiXY))],
            fallback: DepAdd {
                dep: Prefixes::PositiveRemainderEqualsDivisor as usize,
                dep_default: One,
                expr: AntiXY,
            },
        },
        Prefixes::NegativeDivisorGreaterThanRemainder => CheckpointRule {
            default: Zero,
            cases: vec![(JEq(1), Set(Product)), (JEq(3), Mul(AntiYX))],
            fallback: DepAdd {
                dep: Prefixes::NegativeDivisorEqualsRemainder as usize,
                dep_default: One,
                expr: AntiYX,
            },
        },

        // === ChangeDivisor: j==1 → cp*rx*ry, else → cp*(1-rx)*ry ===
        Prefixes::ChangeDivisor => CheckpointRule {
            default: Custom(2 - (1i128 << XLEN)),
            cases: vec![(JEq(1), Mul(Product))],
            fallback: Mul(AntiXY),
        },

        // === ChangeDivisorW: j<XLEN→0, j==XLEN+1→init, else→cp*(1-rx)*ry ===
        Prefixes::ChangeDivisorW => CheckpointRule {
            default: Zero,
            cases: vec![
                (JLt(XLEN), Const(Zero)),
                (
                    JEq(XLEN + 1),
                    SetScaled {
                        coeff: 2 - (1i128 << XLEN),
                        expr: Product,
                    },
                ),
            ],
            fallback: Mul(AntiXY),
        },

        // === SignExtension ===
        Prefixes::SignExtension => CheckpointRule {
            default: Zero,
            cases: vec![(JEq(1), Null)],
            fallback: SignExtAccum {
                dep: Prefixes::LeftOperandMsb as usize,
                final_j: 2 * XLEN - 1,
            },
        },

        // === Pow2 ===
        Prefixes::Pow2 => {
            let trail = XLEN.trailing_zeros() as usize;
            CheckpointRule {
                default: One,
                cases: vec![
                    (SuffixLenNonZero, Const(One)),
                    (
                        JEq(2 * XLEN - trail),
                        Pow2Init {
                            half_pow: (XLEN / 2) as u32,
                        },
                    ),
                    (JGt(2 * XLEN - trail), Pow2DoubleMul { xlen: XLEN }),
                ],
                fallback: Const(One),
            }
        }
        Prefixes::Pow2W => CheckpointRule {
            default: One,
            cases: vec![
                (SuffixLenNonZero, Const(One)),
                (JEq(2 * XLEN - 5), Pow2Init { half_pow: 16 }),
                (JGt(2 * XLEN - 5), Pow2DoubleMul { xlen: XLEN }),
            ],
            fallback: Const(One),
        },

        // === Rev8W ===
        Prefixes::Rev8W => CheckpointRule {
            default: Zero,
            cases: vec![],
            fallback: Rev8WAdd { xlen: 64 },
        },

        // === W-guarded variants ===
        Prefixes::RightShiftW => CheckpointRule {
            default: Zero,
            cases: vec![(JLt(XLEN), Const(Zero))],
            fallback: Hybrid {
                mul: OnePlusY,
                add: Product,
            },
        },
        Prefixes::LeftShiftWHelper => CheckpointRule {
            default: One,
            cases: vec![(JLt(XLEN), Const(One))],
            fallback: Mul(OnePlusY),
        },
        Prefixes::LeftShiftW => CheckpointRule {
            default: Zero,
            cases: vec![(JLt(XLEN), Const(Zero))],
            fallback: DepAddWeighted {
                dep: Prefixes::LeftShiftWHelper as usize,
                dep_default: One,
                weight: pos(0, XLEN as u32, 0),
                expr: AntiYX,
            },
        },
        Prefixes::RightOperandW => CheckpointRule {
            default: Zero,
            cases: vec![(JLt(XLEN + 1), Passthrough)],
            fallback: AddWeighted {
                weight: pos(0, XLEN as u32, 0),
                expr: Y,
            },
        },
        Prefixes::OverflowBitsZero => CheckpointRule {
            default: One,
            cases: vec![(JGe(128 - XLEN), Passthrough)],
            fallback: Mul(NorBit),
        },
    }
}

fn prefix_to_mle_rule<const XLEN: usize>(prefix: &Prefixes) -> PrefixMleRule {
    use BilinearExpr::{
        AntiXY, AntiYX, EqBit, NorBit, OneMinusX, OneMinusY, OrBit, Product, XorBit,
    };

    let total_bits = 2 * XLEN;
    let idx = *prefix as usize;

    let rule = |default, formula| PrefixMleRule {
        checkpoint_idx: idx,
        default,
        formula,
    };

    match prefix {
        Prefixes::Eq => rule(
            DefaultVal::One,
            PrefixMleFormula::Multiplicative {
                pair: EqBit,
                remaining: RemainingTest::Equality,
            },
        ),
        Prefixes::LeftOperandIsZero => rule(
            DefaultVal::One,
            PrefixMleFormula::Multiplicative {
                pair: OneMinusX,
                remaining: RemainingTest::LeftZero,
            },
        ),
        Prefixes::RightOperandIsZero => rule(
            DefaultVal::One,
            PrefixMleFormula::Multiplicative {
                pair: OneMinusY,
                remaining: RemainingTest::RightZero,
            },
        ),
        Prefixes::DivByZero => rule(
            DefaultVal::One,
            PrefixMleFormula::Multiplicative {
                pair: AntiXY,
                remaining: RemainingTest::LeftZeroRightAllOnes,
            },
        ),
        Prefixes::And => rule(
            DefaultVal::Zero,
            PrefixMleFormula::BitwiseAdditive {
                pair: Product,
                weight: WeightFn::Positional {
                    rotation: 0,
                    word_bits: XLEN as u32,
                    j_offset: 0,
                },
                op: IntBitOp::And,
                total_bits,
                word_bits: XLEN,
                rotation: 0,
            },
        ),
        Prefixes::Andn => rule(
            DefaultVal::Zero,
            PrefixMleFormula::BitwiseAdditive {
                pair: AntiYX,
                weight: WeightFn::Positional {
                    rotation: 0,
                    word_bits: XLEN as u32,
                    j_offset: 0,
                },
                op: IntBitOp::AndNot,
                total_bits,
                word_bits: XLEN,
                rotation: 0,
            },
        ),
        Prefixes::Or => rule(
            DefaultVal::Zero,
            PrefixMleFormula::BitwiseAdditive {
                pair: OrBit,
                weight: WeightFn::Positional {
                    rotation: 0,
                    word_bits: XLEN as u32,
                    j_offset: 0,
                },
                op: IntBitOp::Or,
                total_bits,
                word_bits: XLEN,
                rotation: 0,
            },
        ),
        Prefixes::Xor => rule(
            DefaultVal::Zero,
            PrefixMleFormula::BitwiseAdditive {
                pair: XorBit,
                weight: WeightFn::Positional {
                    rotation: 0,
                    word_bits: XLEN as u32,
                    j_offset: 0,
                },
                op: IntBitOp::Xor,
                total_bits,
                word_bits: XLEN,
                rotation: 0,
            },
        ),
        Prefixes::RightOperand => rule(
            DefaultVal::Zero,
            PrefixMleFormula::RightOperandExtract {
                word_bits: XLEN,
                start_round: 0,
                total_bits,
                suffix_guard: total_bits,
            },
        ),
        Prefixes::XorRot16 => rule(
            DefaultVal::Zero,
            PrefixMleFormula::BitwiseAdditive {
                pair: XorBit,
                weight: WeightFn::Positional {
                    rotation: 16,
                    word_bits: XLEN as u32,
                    j_offset: 0,
                },
                op: IntBitOp::Xor,
                total_bits,
                word_bits: XLEN,
                rotation: 16,
            },
        ),
        Prefixes::XorRot24 => rule(
            DefaultVal::Zero,
            PrefixMleFormula::BitwiseAdditive {
                pair: XorBit,
                weight: WeightFn::Positional {
                    rotation: 24,
                    word_bits: XLEN as u32,
                    j_offset: 0,
                },
                op: IntBitOp::Xor,
                total_bits,
                word_bits: XLEN,
                rotation: 24,
            },
        ),
        Prefixes::XorRot32 => rule(
            DefaultVal::Zero,
            PrefixMleFormula::BitwiseAdditive {
                pair: XorBit,
                weight: WeightFn::Positional {
                    rotation: 32,
                    word_bits: XLEN as u32,
                    j_offset: 0,
                },
                op: IntBitOp::Xor,
                total_bits,
                word_bits: XLEN,
                rotation: 32,
            },
        ),
        Prefixes::XorRot63 => rule(
            DefaultVal::Zero,
            PrefixMleFormula::BitwiseAdditive {
                pair: XorBit,
                weight: WeightFn::Positional {
                    rotation: 63,
                    word_bits: XLEN as u32,
                    j_offset: 0,
                },
                op: IntBitOp::Xor,
                total_bits,
                word_bits: XLEN,
                rotation: 63,
            },
        ),
        Prefixes::XorRotW7 => rule(
            DefaultVal::Zero,
            PrefixMleFormula::BitwiseAdditive {
                pair: XorBit,
                weight: WeightFn::Positional {
                    rotation: 7,
                    word_bits: 32,
                    j_offset: XLEN,
                },
                op: IntBitOp::Xor,
                total_bits,
                word_bits: 32,
                rotation: 7,
            },
        ),
        Prefixes::XorRotW8 => rule(
            DefaultVal::Zero,
            PrefixMleFormula::BitwiseAdditive {
                pair: XorBit,
                weight: WeightFn::Positional {
                    rotation: 8,
                    word_bits: 32,
                    j_offset: XLEN,
                },
                op: IntBitOp::Xor,
                total_bits,
                word_bits: 32,
                rotation: 8,
            },
        ),
        Prefixes::XorRotW12 => rule(
            DefaultVal::Zero,
            PrefixMleFormula::BitwiseAdditive {
                pair: XorBit,
                weight: WeightFn::Positional {
                    rotation: 12,
                    word_bits: 32,
                    j_offset: XLEN,
                },
                op: IntBitOp::Xor,
                total_bits,
                word_bits: 32,
                rotation: 12,
            },
        ),
        Prefixes::XorRotW16 => rule(
            DefaultVal::Zero,
            PrefixMleFormula::BitwiseAdditive {
                pair: XorBit,
                weight: WeightFn::Positional {
                    rotation: 16,
                    word_bits: 32,
                    j_offset: XLEN,
                },
                op: IntBitOp::Xor,
                total_bits,
                word_bits: 32,
                rotation: 16,
            },
        ),
        Prefixes::UpperWord => rule(
            DefaultVal::Zero,
            PrefixMleFormula::OperandExtract {
                base_shift: XLEN,
                has_x: true,
                has_y: true,
                word_bits: XLEN,
                total_bits,
                active_when: Some(RoundGuard::JLt(XLEN)),
                passthrough_when_inactive: true,
                is_upper: true,
            },
        ),
        Prefixes::LowerWord => rule(
            DefaultVal::Zero,
            PrefixMleFormula::OperandExtract {
                base_shift: 2 * XLEN,
                has_x: true,
                has_y: true,
                word_bits: XLEN,
                total_bits,
                active_when: Some(RoundGuard::JGe(XLEN)),
                passthrough_when_inactive: false,
                is_upper: false,
            },
        ),
        Prefixes::LowerHalfWord => rule(
            DefaultVal::Zero,
            PrefixMleFormula::OperandExtract {
                base_shift: 2 * XLEN,
                has_x: true,
                has_y: true,
                word_bits: XLEN / 2,
                total_bits,
                active_when: Some(RoundGuard::JGe(XLEN + XLEN / 2)),
                passthrough_when_inactive: false,
                is_upper: false,
            },
        ),
        Prefixes::RightOperandW => rule(
            DefaultVal::Zero,
            PrefixMleFormula::RightOperandExtract {
                word_bits: XLEN,
                start_round: XLEN,
                total_bits,
                suffix_guard: XLEN,
            },
        ),
        Prefixes::LessThan => rule(
            DefaultVal::Zero,
            PrefixMleFormula::DependentComparison {
                eq_idx: Prefixes::Eq as usize,
                eq_default: DefaultVal::One,
                cmp: Comparison::LessThan,
            },
        ),
        Prefixes::PositiveRemainderLessThanDivisor => rule(
            DefaultVal::Zero,
            PrefixMleFormula::ComplexComparison {
                eq_idx: Prefixes::PositiveRemainderEqualsDivisor as usize,
                cmp: Comparison::LessThan,
                sign_pair_j0: NorBit,
                init_cmp: Comparison::LessThan,
                mul_anti: AntiXY,
                start_round: 0,
            },
        ),
        Prefixes::NegativeDivisorGreaterThanRemainder => rule(
            DefaultVal::Zero,
            PrefixMleFormula::ComplexComparison {
                eq_idx: Prefixes::NegativeDivisorEqualsRemainder as usize,
                cmp: Comparison::GreaterThan,
                sign_pair_j0: Product,
                init_cmp: Comparison::GreaterThan,
                mul_anti: AntiYX,
                start_round: 0,
            },
        ),
        Prefixes::PositiveRemainderEqualsDivisor => rule(
            DefaultVal::One,
            PrefixMleFormula::SignGatedMultiplicative {
                sign_pair: NorBit,
                base_pair: EqBit,
                remaining: RemainingTest::Equality,
                start_round: 0,
            },
        ),
        Prefixes::NegativeDivisorEqualsRemainder => rule(
            DefaultVal::One,
            PrefixMleFormula::SignGatedMultiplicative {
                sign_pair: Product,
                base_pair: EqBit,
                remaining: RemainingTest::Equality,
                start_round: 0,
            },
        ),
        Prefixes::LeftShift => rule(
            DefaultVal::Zero,
            PrefixMleFormula::LeftShift {
                helper_idx: Prefixes::LeftShiftHelper as usize,
                helper_default: DefaultVal::One,
                word_bits: XLEN,
                start_round: 0,
            },
        ),
        Prefixes::LeftShiftHelper => rule(
            DefaultVal::One,
            PrefixMleFormula::LeftShiftHelper {
                word_bits: XLEN,
                start_round: 0,
            },
        ),
        Prefixes::LeftShiftW => rule(
            DefaultVal::Zero,
            PrefixMleFormula::LeftShift {
                helper_idx: Prefixes::LeftShiftWHelper as usize,
                helper_default: DefaultVal::One,
                word_bits: XLEN,
                start_round: XLEN,
            },
        ),
        Prefixes::LeftShiftWHelper => rule(
            DefaultVal::One,
            PrefixMleFormula::LeftShiftHelper {
                word_bits: XLEN,
                start_round: XLEN,
            },
        ),
        Prefixes::RightShift => rule(
            DefaultVal::Zero,
            PrefixMleFormula::RightShift {
                word_bits: XLEN,
                start_round: 0,
            },
        ),
        Prefixes::RightShiftW => rule(
            DefaultVal::Zero,
            PrefixMleFormula::RightShift {
                word_bits: XLEN,
                start_round: XLEN,
            },
        ),
        Prefixes::LeftOperandMsb => rule(
            DefaultVal::Zero,
            PrefixMleFormula::Msb {
                msb_idx: Prefixes::LeftOperandMsb as usize,
                start_round: 0,
                is_left: true,
            },
        ),
        Prefixes::RightOperandMsb => rule(
            DefaultVal::Zero,
            PrefixMleFormula::Msb {
                msb_idx: Prefixes::RightOperandMsb as usize,
                start_round: 0,
                is_left: false,
            },
        ),
        Prefixes::Lsb => rule(DefaultVal::One, PrefixMleFormula::Lsb { total_bits }),
        Prefixes::TwoLsb => rule(DefaultVal::One, PrefixMleFormula::TwoLsb { total_bits }),
        Prefixes::SignExtension => rule(
            DefaultVal::Zero,
            PrefixMleFormula::SignExtension {
                msb_idx: Prefixes::LeftOperandMsb as usize,
                word_bits: XLEN,
            },
        ),
        Prefixes::SignExtensionUpperHalf => rule(
            DefaultVal::Zero,
            PrefixMleFormula::SignExtUpperHalf { word_bits: XLEN },
        ),
        Prefixes::SignExtensionRightOperand => rule(
            DefaultVal::Zero,
            PrefixMleFormula::SignExtRightOp { word_bits: XLEN },
        ),
        Prefixes::Pow2 => rule(
            DefaultVal::One,
            PrefixMleFormula::Pow2 {
                word_mask: XLEN - 1,
                log_word_bits: XLEN.trailing_zeros(),
                total_bits,
            },
        ),
        Prefixes::Pow2W => rule(
            DefaultVal::One,
            PrefixMleFormula::Pow2 {
                word_mask: 31,
                log_word_bits: 5,
                total_bits,
            },
        ),
        Prefixes::Rev8W => rule(DefaultVal::Zero, PrefixMleFormula::Rev8W { xlen: 64 }),
        Prefixes::ChangeDivisor => rule(
            DefaultVal::Custom(2 - (1i128 << XLEN)),
            PrefixMleFormula::ChangeDivisor {
                word_bits: XLEN,
                start_round: 0,
            },
        ),
        Prefixes::ChangeDivisorW => rule(
            DefaultVal::Zero,
            PrefixMleFormula::ChangeDivisor {
                word_bits: XLEN,
                start_round: XLEN,
            },
        ),
        Prefixes::NegativeDivisorZeroRemainder => rule(
            DefaultVal::Zero,
            PrefixMleFormula::NegDivZeroRem {
                word_bits: XLEN,
                start_round: 0,
            },
        ),
        Prefixes::OverflowBitsZero => rule(
            DefaultVal::One,
            PrefixMleFormula::OverflowBitsZero {
                pair: NorBit,
                word_bits: XLEN,
                total_bits,
            },
        ),
    }
}

fn suffix_to_op<const XLEN: usize>(suffix: &Suffixes) -> SuffixOp {
    match suffix {
        Suffixes::One => SuffixOp::One,
        Suffixes::And => SuffixOp::And,
        Suffixes::NotAnd => SuffixOp::Andn,
        Suffixes::Or => SuffixOp::Or,
        Suffixes::Xor => SuffixOp::Xor,
        Suffixes::Eq => SuffixOp::Eq,
        Suffixes::LessThan => SuffixOp::Lt,
        Suffixes::GreaterThan => SuffixOp::Gt,
        Suffixes::RightOperand => SuffixOp::RightOperand,
        Suffixes::RightOperandW => SuffixOp::RightOperandW,
        Suffixes::LeftOperandIsZero => SuffixOp::XIsZero,
        Suffixes::RightOperandIsZero => SuffixOp::YIsZero,
        Suffixes::Lsb => SuffixOp::Lsb,
        Suffixes::TwoLsb => SuffixOp::TwoLsbZero,
        Suffixes::DivByZero => SuffixOp::DivByZero,
        Suffixes::ChangeDivisor => SuffixOp::ChangeDivisor,
        Suffixes::ChangeDivisorW => SuffixOp::ChangeDivisorW,
        Suffixes::LeftShift => SuffixOp::LeftShift,
        Suffixes::LeftShiftW => SuffixOp::LeftShiftW {
            half_xlen: XLEN / 2,
        },
        Suffixes::LeftShiftWHelper => SuffixOp::LeftShiftWHelper,
        Suffixes::RightShift => SuffixOp::RightShift,
        Suffixes::RightShiftW => SuffixOp::RightShiftW {
            half_xlen: XLEN / 2,
        },
        Suffixes::RightShiftHelper => SuffixOp::RightShiftHelper,
        Suffixes::RightShiftWHelper => SuffixOp::RightShiftWHelper {
            half_xlen: XLEN / 2,
        },
        Suffixes::SignExtension => SuffixOp::SignExtension { xlen: XLEN },
        Suffixes::SignExtensionUpperHalf => SuffixOp::SignExtensionUpperHalf {
            half_xlen: XLEN / 2,
        },
        Suffixes::SignExtensionRightOperand => SuffixOp::SignExtensionRightOperand { xlen: XLEN },
        Suffixes::Pow2 => SuffixOp::Pow2 {
            split_bits: XLEN.trailing_zeros() as usize,
        },
        Suffixes::Pow2W => SuffixOp::Pow2 { split_bits: 5 },
        Suffixes::RightShiftPadding => SuffixOp::RightShiftPaddingMask { xlen: XLEN },
        Suffixes::UpperWord => SuffixOp::UpperWord { xlen: XLEN },
        Suffixes::LowerWord => SuffixOp::LowerWord { xlen: XLEN },
        Suffixes::LowerHalfWord => SuffixOp::LowerHalfWord {
            half_xlen: XLEN / 2,
        },
        Suffixes::Rev8W => SuffixOp::ByteReverseW,
        Suffixes::OverflowBitsZero => SuffixOp::OverflowBitsZero { xlen: XLEN },
        Suffixes::XorRot16 => SuffixOp::XorRotate {
            rotation: 16,
            word_bits: 64,
        },
        Suffixes::XorRot24 => SuffixOp::XorRotate {
            rotation: 24,
            word_bits: 64,
        },
        Suffixes::XorRot32 => SuffixOp::XorRotate {
            rotation: 32,
            word_bits: 64,
        },
        Suffixes::XorRot63 => SuffixOp::XorRotate {
            rotation: 63,
            word_bits: 64,
        },
        Suffixes::XorRotW7 => SuffixOp::XorRotate {
            rotation: 7,
            word_bits: 32,
        },
        Suffixes::XorRotW8 => SuffixOp::XorRotate {
            rotation: 8,
            word_bits: 32,
        },
        Suffixes::XorRotW12 => SuffixOp::XorRotate {
            rotation: 12,
            word_bits: 32,
        },
        Suffixes::XorRotW16 => SuffixOp::XorRotate {
            rotation: 16,
            word_bits: 32,
        },
    }
}

fn all_table_kinds() -> [LookupTableKind; LookupTableKind::COUNT] {
    [
        LookupTableKind::RangeCheck,
        LookupTableKind::RangeCheckAligned,
        LookupTableKind::And,
        LookupTableKind::Andn,
        LookupTableKind::Or,
        LookupTableKind::Xor,
        LookupTableKind::Equal,
        LookupTableKind::NotEqual,
        LookupTableKind::SignedLessThan,
        LookupTableKind::UnsignedLessThan,
        LookupTableKind::SignedGreaterThanEqual,
        LookupTableKind::UnsignedGreaterThanEqual,
        LookupTableKind::UnsignedLessThanEqual,
        LookupTableKind::UpperWord,
        LookupTableKind::LowerHalfWord,
        LookupTableKind::SignExtendHalfWord,
        LookupTableKind::Movsign,
        LookupTableKind::Pow2,
        LookupTableKind::Pow2W,
        LookupTableKind::ShiftRightBitmask,
        LookupTableKind::VirtualSRL,
        LookupTableKind::VirtualSRA,
        LookupTableKind::VirtualROTR,
        LookupTableKind::VirtualROTRW,
        LookupTableKind::ValidDiv0,
        LookupTableKind::ValidUnsignedRemainder,
        LookupTableKind::ValidSignedRemainder,
        LookupTableKind::VirtualChangeDivisor,
        LookupTableKind::VirtualChangeDivisorW,
        LookupTableKind::HalfwordAlignment,
        LookupTableKind::WordAlignment,
        LookupTableKind::MulUNoOverflow,
        LookupTableKind::VirtualRev8W,
        LookupTableKind::VirtualXORROT32,
        LookupTableKind::VirtualXORROT24,
        LookupTableKind::VirtualXORROT16,
        LookupTableKind::VirtualXORROT63,
        LookupTableKind::VirtualXORROTW16,
        LookupTableKind::VirtualXORROTW12,
        LookupTableKind::VirtualXORROTW8,
        LookupTableKind::VirtualXORROTW7,
    ]
}
