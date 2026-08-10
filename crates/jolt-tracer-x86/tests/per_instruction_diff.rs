//! Per-instruction differential tests: for every implemented row kind, run
//! ≥1000 random operand/state instances through the AOT backend and the
//! reference interpreter `Cpu`, comparing full register state, PC, scratch
//! memory, and the advice tape (modeled on the tracer's execute-vs-trace
//! harness).
//!
//! Coverage is exhaustive by construction: `classify` matches every
//! `JoltInstructionKind` variant without a wildcard, so a new kind in
//! jolt-riscv fails to compile here until it is classified, and
//! `classification_matches_compiler` asserts the classification agrees with
//! what the transpiler actually accepts.

#![cfg(all(target_arch = "x86_64", target_os = "linux"))]
#![expect(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use common::constants::REGISTER_COUNT;
use common::jolt_device::JoltDevice;
use jolt_platform::JOLT_ADVICE_WRITE_CALL_ID;
use jolt_riscv::{JoltInstructionKind, JoltInstructionRow, NormalizedOperands};
use jolt_tracer_x86::harness::{
    self, run_program, single_row_program, MEM_CAPACITY, SCRATCH_DWORDS, SCRATCH_START, TEST_ADDR,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tracer::emulator::cpu::Cpu;
use tracer::emulator::terminal::DummyTerminal;
use tracer::instruction::Instruction;

const N: usize = 1000;
const REGS: usize = REGISTER_COUNT as usize;

// ── Kind classification (compile-time exhaustive) ───────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    Supported,
    /// Implemented, but only legal inside a source group that computes
    /// advice; a bare row must be a compile-time error (never a stale
    /// advice-slot read), so the single-row harness cannot drive it.
    RequiresAdviceGroup,
    NotYetSupported,
}

/// Marker names of the kinds the transpiler implements (slice 2: base ISA).
const SUPPORTED: &[&str] = &[
    "Add",
    "Addi",
    "And",
    "AndI",
    "AssertHalfwordAlignment",
    "AssertLte",
    "AssertWordAlignment",
    "Auipc",
    "Beq",
    "Bge",
    "BgeU",
    "Blt",
    "BltU",
    "Bne",
    "Fence",
    "Jal",
    "Jalr",
    "Ld",
    "Lui",
    "Mul",
    "MulHU",
    "MulI",
    "Or",
    "OrI",
    "Pow2",
    "Sd",
    "SltI",
    "SltIU",
    "SltU",
    "Sub",
    "VirtualHostIO",
    "VirtualShiftRightBitmask",
    "VirtualSignExtendWord",
    "VirtualSrai",
    "VirtualSrl",
    "VirtualSrli",
    "VirtualZeroExtendWord",
    "Xor",
    "XorI",
    // Slice 3 (fast-pass coverage):
    "Slt",
    "Andn",
    "Pow2I",
    "Pow2W",
    "Pow2IW",
    "MovSign",
    "VirtualRev8W",
    "VirtualRotri",
    "VirtualRotriw",
    "VirtualShiftRightBitmaski",
    "VirtualSra",
    "VirtualXorRot32",
    "VirtualXorRot24",
    "VirtualXorRot16",
    "VirtualXorRot63",
    "VirtualXorRotW16",
    "VirtualXorRotW12",
    "VirtualXorRotW8",
    "VirtualXorRotW7",
    "AssertEq",
    "AssertValidDiv0",
    "AssertValidUnsignedRemainder",
    "AssertMulUNoOverflow",
    "VirtualChangeDivisor",
    "VirtualChangeDivisorW",
    "VirtualAdviceLen",
    "VirtualAdviceLoad",
];

fn class_by_marker(marker: &str) -> Class {
    // Slice 3b: VirtualAdvice reads the group's advice slots, filled by the
    // group's advice computation; whole-guest gates cover it (muldiv's
    // DIV/REM groups, the inline guests).
    if marker == "VirtualAdvice" {
        Class::RequiresAdviceGroup
    } else if SUPPORTED.contains(&marker) {
        Class::Supported
    } else {
        Class::NotYetSupported
    }
}

macro_rules! classify_kinds {
    (
        instructions: [$($(#[$meta:meta])* $instr:ident => $marker:ident => ($tag:expr, $name:expr)),* $(,)?]
    ) => {
        /// Exhaustive (no wildcard): a new `JoltInstructionKind` fails to
        /// compile until classified.
        fn classify(kind: &JoltInstructionKind) -> Class {
            match kind {
                JoltInstructionKind::Noop(_) => Class::NotYetSupported,
                $(
                    $(#[$meta])*
                    JoltInstructionKind::$marker(_) => class_by_marker(stringify!($marker)),
                )*
            }
        }
    };
}
jolt_riscv::for_each_jolt_instruction_kind!(classify_kinds);

fn default_row(kind: JoltInstructionKind) -> JoltInstructionRow {
    JoltInstructionRow {
        instruction_kind: kind,
        address: TEST_ADDR as usize,
        operands: NormalizedOperands {
            rs1: None,
            rs2: None,
            rd: None,
            imm: 0,
        },
        virtual_sequence_remaining: None,
        is_first_in_sequence: true,
        is_compressed: false,
    }
}

/// The classification must agree with what the transpiler accepts: every
/// supported kind compiles as a bare row, everything else fails fast
/// (unimplemented kinds and advice-group-only kinds alike).
#[test]
fn classification_matches_compiler() {
    for &kind in JoltInstructionKind::ALL {
        let program = single_row_program(default_row(kind));
        let compiles = harness::compile_only(&program).is_ok();
        let classified = classify(&kind) == Class::Supported;
        assert_eq!(
            compiles,
            classified,
            "kind {} ({:?}): compiler acceptance disagrees with classification",
            kind.name(),
            kind
        );
    }
}

/// Regression: a `VirtualAdvice` row in a group without an advice
/// computation must be refused at compile time with the advice-specific
/// error, not compiled into a stale-slot read (and not rejected as merely
/// unsupported).
#[test]
fn bare_virtual_advice_is_a_compile_error() {
    let mut row = default_row(kind_by_name("VirtualAdvice"));
    row.operands.rd = Some(5);
    let error = harness::compile_only(&single_row_program(row))
        .expect_err("bare VirtualAdvice must not compile");
    let message = format!("{error:?}");
    assert!(
        message.contains("without an advice computation"),
        "expected the advice-group error, got: {message}"
    );
}

// ── Random row/state generation ─────────────────────────────────────

struct Instance {
    row: JoltInstructionRow,
    pre_regs: [u64; REGS],
    scratch: Vec<u64>,
    advice: Vec<u8>,
}

fn base_instance(rng: &mut StdRng, kind: JoltInstructionKind) -> Instance {
    let mut pre_regs = [0u64; REGS];
    for reg in pre_regs.iter_mut().skip(1) {
        *reg = rng.gen();
    }
    let scratch: Vec<u64> = (0..SCRATCH_DWORDS).map(|_| rng.gen()).collect();
    Instance {
        row: default_row(kind),
        pre_regs,
        scratch,
        advice: Vec::new(),
    }
}

fn reg(rng: &mut StdRng) -> u8 {
    rng.gen_range(0..REGS as u8)
}

fn rd(rng: &mut StdRng) -> u8 {
    rng.gen_range(1..REGS as u8)
}

fn imm12(rng: &mut StdRng) -> i128 {
    rng.gen_range(-2048i64..2048) as i128
}

/// Pin `x[reg]` so that `x[reg] + imm == target` (wrapping).
fn pin_base(instance: &mut Instance, register: u8, imm: i128, target: u64) {
    assert_ne!(register, 0, "cannot pin x0");
    instance.pre_regs[register as usize] = target.wrapping_sub(imm as i64 as u64);
}

fn alu_rr(rng: &mut StdRng, kind: JoltInstructionKind) -> Instance {
    let mut i = base_instance(rng, kind);
    i.row.operands.rs1 = Some(reg(rng));
    i.row.operands.rs2 = Some(reg(rng));
    i.row.operands.rd = Some(rd(rng));
    i
}

fn alu_ri(rng: &mut StdRng, kind: JoltInstructionKind, wide_imm: bool) -> Instance {
    let mut i = base_instance(rng, kind);
    i.row.operands.rs1 = Some(reg(rng));
    i.row.operands.rd = Some(rd(rng));
    i.row.operands.imm = if wide_imm {
        rng.gen::<i64>() as i128
    } else {
        imm12(rng)
    };
    i
}

fn upper_imm(rng: &mut StdRng, kind: JoltInstructionKind) -> Instance {
    let mut i = base_instance(rng, kind);
    i.row.operands.rd = Some(rd(rng));
    // 20-bit immediate << 12, sign-extended (the U-format decode invariant).
    i.row.operands.imm = ((rng.gen_range(-(1i64 << 19)..(1i64 << 19))) << 12) as i128;
    i
}

fn unary(rng: &mut StdRng, kind: JoltInstructionKind) -> Instance {
    let mut i = base_instance(rng, kind);
    i.row.operands.rs1 = Some(reg(rng));
    i.row.operands.rd = Some(rd(rng));
    i
}

fn shift_imm(rng: &mut StdRng, kind: JoltInstructionKind) -> Instance {
    let mut i = base_instance(rng, kind);
    i.row.operands.rs1 = Some(reg(rng));
    i.row.operands.rd = Some(rd(rng));
    // The immediate is a bitmask; shift = imm.trailing_zeros(). One-hot
    // values are the real invariant; 0 exercises the trailing_zeros(0)=64
    // edge on both sides.
    i.row.operands.imm = if rng.gen_ratio(1, 20) {
        0
    } else {
        (1u64 << rng.gen_range(0..64)) as i128
    };
    i
}

fn shift_reg(rng: &mut StdRng, kind: JoltInstructionKind) -> Instance {
    let mut i = alu_rr(rng, kind);
    // rs2 carries a bitmask; mix one-hot, zero, and arbitrary values.
    let rs2 = i.row.operands.rs2.unwrap();
    if rs2 != 0 {
        i.pre_regs[rs2 as usize] = match rng.gen_range(0..3) {
            0 => 1u64 << rng.gen_range(0..64),
            1 => 0,
            _ => rng.gen(),
        };
    }
    i
}

fn branch(rng: &mut StdRng, kind: JoltInstructionKind) -> Instance {
    let mut i = base_instance(rng, kind);
    let rs1 = reg(rng);
    let rs2 = reg(rng);
    i.row.operands.rs1 = Some(rs1);
    i.row.operands.rs2 = Some(rs2);
    i.row.operands.imm = if rng.gen() { 8 } else { -8 };
    // Half the time force equality so both outcomes are exercised.
    if rng.gen() && rs1 != 0 && rs2 != 0 {
        i.pre_regs[rs2 as usize] = i.pre_regs[rs1 as usize];
    }
    i
}

fn jal(rng: &mut StdRng) -> Instance {
    let mut i = base_instance(rng, JoltInstructionKind::JAL);
    i.row.operands.rd = Some(rd(rng));
    i.row.operands.imm = if rng.gen() { 8 } else { -8 };
    i
}

fn jalr(rng: &mut StdRng) -> Instance {
    let mut i = base_instance(rng, JoltInstructionKind::JALR);
    let rs1 = rd(rng); // nonzero so the base can be pinned
    i.row.operands.rs1 = Some(rs1);
    i.row.operands.rd = Some(rd(rng)); // may alias rs1 (ordering test)
    let imm = (rng.gen_range(-8i64..8) * 2) as i128;
    i.row.operands.imm = imm;
    let target = if rng.gen() {
        TEST_ADDR + 8
    } else {
        TEST_ADDR - 8
    };
    pin_base(&mut i, rs1, imm, target);
    i
}

fn scratch_slot(rng: &mut StdRng) -> u64 {
    SCRATCH_START + 8 * rng.gen_range(0..SCRATCH_DWORDS as u64)
}

fn load(rng: &mut StdRng) -> Instance {
    let mut i = base_instance(rng, JoltInstructionKind::LD);
    let rs1 = rd(rng);
    i.row.operands.rs1 = Some(rs1);
    i.row.operands.rd = Some(rd(rng));
    let imm = imm12(rng);
    i.row.operands.imm = imm;
    pin_base(&mut i, rs1, imm, scratch_slot(rng));
    i
}

fn store(rng: &mut StdRng) -> Instance {
    let mut i = base_instance(rng, JoltInstructionKind::SD);
    let rs1 = rd(rng);
    i.row.operands.rs1 = Some(rs1);
    i.row.operands.rs2 = Some(reg(rng));
    let imm = imm12(rng);
    i.row.operands.imm = imm;
    pin_base(&mut i, rs1, imm, scratch_slot(rng));
    i
}

fn assert_align(rng: &mut StdRng, kind: JoltInstructionKind, align: u64) -> Instance {
    let mut i = base_instance(rng, kind);
    let rs1 = rd(rng);
    i.row.operands.rs1 = Some(rs1);
    // FormatAssert immediates are wrapped-to-u64 offsets; keep them small
    // and pin the base so the (passing) alignment holds.
    let imm = (rng.gen_range(0..16u64) * align) as i128;
    i.row.operands.imm = imm;
    let target = SCRATCH_START + align * rng.gen_range(0..64u64);
    pin_base(&mut i, rs1, imm, target);
    i
}

fn assert_lte(rng: &mut StdRng) -> Instance {
    let mut i = base_instance(rng, JoltInstructionKind::VirtualAssertLTE);
    let rs1 = rd(rng);
    let rs2 = rd(rng);
    i.row.operands.rs1 = Some(rs1);
    i.row.operands.rs2 = Some(rs2);
    // Ensure the (passing) invariant x[rs1] <= x[rs2].
    let a = i.pre_regs[rs1 as usize];
    let b = i.pre_regs[rs2 as usize];
    if a > b {
        i.pre_regs[rs1 as usize] = b;
        i.pre_regs[rs2 as usize] = a;
    }
    if rs1 == rs2 || rng.gen_ratio(1, 10) {
        i.pre_regs[rs2 as usize] = i.pre_regs[rs1 as usize];
    }
    i
}

fn fence(rng: &mut StdRng) -> Instance {
    base_instance(rng, JoltInstructionKind::FENCE)
}

fn host_io(rng: &mut StdRng) -> Instance {
    let mut i = base_instance(rng, kind_by_name("VirtualHostIO"));
    i.row.operands.rs1 = Some(reg(rng));
    // FormatI unwraps rd; the exec never writes it (parity holds either way).
    i.row.operands.rd = Some(rd(rng));
    // a0 selects the call: unknown ids are no-ops on both sides; advice
    // writes append guest bytes to the tape on both sides.
    if rng.gen() {
        i.pre_regs[10] = JOLT_ADVICE_WRITE_CALL_ID as u64;
        i.pre_regs[11] = scratch_slot(rng);
        i.pre_regs[12] = rng.gen_range(0..48);
    } else {
        i.pre_regs[10] = rng.gen::<u32>() as u64 | 1 << 33; // unknown id
    }
    i
}

fn imm_j_u64(rng: &mut StdRng, kind: JoltInstructionKind) -> Instance {
    // FormatJ: rd + a raw u64 immediate (stored bit-preserving in i128).
    let mut i = base_instance(rng, kind);
    i.row.operands.rd = Some(rd(rng));
    i.row.operands.imm = (rng.gen::<u64>() as i64) as i128;
    i
}

fn assert_eq_gen(rng: &mut StdRng) -> Instance {
    let mut i = base_instance(rng, K::VirtualAssertEQ);
    let rs1 = rd(rng);
    let rs2 = rd(rng);
    i.row.operands.rs1 = Some(rs1);
    i.row.operands.rs2 = Some(rs2);
    if rng.gen_ratio(1, 10) {
        // Spoil mode: warn-and-continue on both sides.
        i.row.operands.imm = 1;
    } else {
        i.row.operands.imm = 0;
        i.pre_regs[rs2 as usize] = i.pre_regs[rs1 as usize];
    }
    i
}

fn assert_valid_div0(rng: &mut StdRng) -> Instance {
    let mut i = base_instance(rng, K::VirtualAssertValidDiv0);
    let rs1 = rd(rng);
    let rs2 = rd(rng);
    i.row.operands.rs1 = Some(rs1);
    i.row.operands.rs2 = Some(rs2);
    if rng.gen() {
        // divisor == 0 requires quotient == u64::MAX.
        i.pre_regs[rs1 as usize] = 0;
        i.pre_regs[rs2 as usize] = u64::MAX;
        if rs1 == rs2 {
            i.pre_regs[rs1 as usize] = u64::MAX; // MAX != 0, valid either way
        }
    } else if i.pre_regs[rs1 as usize] == 0 {
        i.pre_regs[rs1 as usize] = 1;
    }
    i
}

fn assert_valid_unsigned_remainder(rng: &mut StdRng) -> Instance {
    let mut i = base_instance(rng, K::VirtualAssertValidUnsignedRemainder);
    let rs1 = rd(rng);
    let rs2 = rd(rng);
    i.row.operands.rs1 = Some(rs1);
    i.row.operands.rs2 = Some(rs2);
    if rs1 == rs2 {
        // remainder == divisor is only valid when both are zero.
        i.pre_regs[rs1 as usize] = 0;
    } else if rng.gen_ratio(1, 5) {
        i.pre_regs[rs2 as usize] = 0; // divisor 0: anything passes
    } else {
        let divisor = i.pre_regs[rs2 as usize].max(1);
        i.pre_regs[rs2 as usize] = divisor;
        i.pre_regs[rs1 as usize] %= divisor;
    }
    i
}

fn assert_mulu_no_overflow(rng: &mut StdRng) -> Instance {
    let mut i = base_instance(rng, K::VirtualAssertMulUNoOverflow);
    let rs1 = rd(rng);
    let rs2 = rd(rng);
    i.row.operands.rs1 = Some(rs1);
    i.row.operands.rs2 = Some(rs2);
    // Keep both factors below 2^32 so the product cannot overflow.
    i.pre_regs[rs1 as usize] &= 0xFFFF_FFFF;
    i.pre_regs[rs2 as usize] &= 0xFFFF_FFFF;
    i
}

fn change_divisor(rng: &mut StdRng, name: &str, wide: bool) -> Instance {
    let mut i = alu_rr(rng, kind_by_name(name));
    if rng.gen_ratio(1, 4) {
        // Exercise the MIN/-1 edge.
        let rs1 = i.row.operands.rs1.unwrap();
        let rs2 = i.row.operands.rs2.unwrap();
        if rs1 != 0 && rs2 != 0 && rs1 != rs2 {
            i.pre_regs[rs1 as usize] = if wide {
                i64::MIN as u64
            } else {
                (i32::MIN as i64) as u64
            };
            i.pre_regs[rs2 as usize] = u64::MAX; // -1
        }
    }
    i
}

fn advice_len(rng: &mut StdRng) -> Instance {
    let mut i = base_instance(rng, kind_by_name("VirtualAdviceLen"));
    i.row.operands.rs1 = Some(reg(rng));
    i.row.operands.rd = Some(rd(rng));
    i.advice = (0..rng.gen_range(0..64)).map(|_| rng.gen()).collect();
    i
}

fn advice_load(rng: &mut StdRng) -> Instance {
    let mut i = base_instance(rng, kind_by_name("VirtualAdviceLoad"));
    i.row.operands.rd = Some(rd(rng));
    i.row.operands.imm = *[1i128, 2, 4, 8].get(rng.gen_range(0..4)).unwrap();
    i.advice = (0..8 + rng.gen_range(0..32)).map(|_| rng.gen()).collect();
    i
}

// ── Reference execution ─────────────────────────────────────────────

struct RefOutcome {
    regs: [u64; REGS],
    pc: u64,
    scratch: Vec<u64>,
    advice_tape: Vec<u8>,
}

fn reference_run(instance: &Instance) -> RefOutcome {
    let mut cpu = Cpu::new(Box::new(DummyTerminal {}));
    cpu.get_mut_mmu().jolt_device = Some(JoltDevice::new(&harness::memory_config()));
    cpu.get_mut_mmu().init_memory(MEM_CAPACITY);
    for (slot, &dword) in instance.scratch.iter().enumerate() {
        let _ = cpu
            .mmu
            .store_doubleword(SCRATCH_START + 8 * slot as u64, dword)
            .unwrap();
    }
    for (index, &value) in instance.pre_regs.iter().enumerate().skip(1) {
        cpu.x[index] = value as i64;
    }
    cpu.advice_tape = tracer::AdviceTape::from_bytes(instance.advice.clone());
    // The interpreter pre-increments PC before exec (tick_operate); link
    // values and fall-through PCs depend on it.
    cpu.update_pc(TEST_ADDR + 4);

    let instruction =
        Instruction::try_from_jolt_instruction_row(instance.row).expect("row not executable");
    instruction.trace(&mut cpu, None);

    let mut regs = [0u64; REGS];
    for (slot, &value) in cpu.x.iter().enumerate() {
        regs[slot] = value as u64;
    }
    let scratch = (0..SCRATCH_DWORDS)
        .map(|slot| {
            cpu.mmu
                .load_doubleword(SCRATCH_START + 8 * slot as u64)
                .unwrap()
                .0
        })
        .collect();
    RefOutcome {
        regs,
        pc: cpu.read_pc(),
        scratch,
        advice_tape: cpu.advice_tape.clone().into_bytes(),
    }
}

// ── The differential loop ───────────────────────────────────────────

fn run_difftest(name: &str, generate: fn(&mut StdRng) -> Instance) {
    let mut rng = StdRng::seed_from_u64(0x1717_5EED);
    for iteration in 0..N {
        let instance = generate(&mut rng);
        let reference = reference_run(&instance);

        let program = single_row_program(instance.row);
        let native = run_program(
            &program,
            &instance.pre_regs,
            &instance.scratch,
            &instance.advice,
        )
        .unwrap_or_else(|e| panic!("{name}[{iteration}]: native run failed: {e:?}"));

        assert_eq!(
            native.exit, 1,
            "{name}[{iteration}]: expected termination, got exit {} ({:?}) row {:?}",
            native.exit, native.helper_error, instance.row
        );
        assert_eq!(
            native.trace_len, 2,
            "{name}[{iteration}]: row + terminal expected"
        );
        assert_eq!(
            native.pc, reference.pc,
            "{name}[{iteration}]: pc diverged, row {:?}",
            instance.row
        );
        for register in 0..REGS {
            assert_eq!(
                native.regs[register], reference.regs[register],
                "{name}[{iteration}]: x{register} diverged, row {:?}",
                instance.row
            );
        }
        assert_eq!(
            native.scratch, reference.scratch,
            "{name}[{iteration}]: scratch memory diverged, row {:?}",
            instance.row
        );
        assert_eq!(
            native.advice_tape, reference.advice_tape,
            "{name}[{iteration}]: advice tape diverged, row {:?}",
            instance.row
        );
    }
}

macro_rules! difftests {
    ($($test:ident => $gen:expr;)*) => {
        $(
            #[test]
            fn $test() {
                run_difftest(stringify!($test), $gen);
            }
        )*
    };
}

use JoltInstructionKind as K;

/// For kinds whose associated-const name collides with the enum variant name
/// (the variant constructor wins path resolution), resolve by name instead.
fn kind_by_name(name: &str) -> JoltInstructionKind {
    JoltInstructionKind::from_name(name).expect("unknown kind name")
}

difftests! {
    diff_add => |r| alu_rr(r, K::ADD);
    diff_sub => |r| alu_rr(r, K::SUB);
    diff_and => |r| alu_rr(r, K::AND);
    diff_or => |r| alu_rr(r, K::OR);
    diff_xor => |r| alu_rr(r, K::XOR);
    diff_mul => |r| alu_rr(r, K::MUL);
    diff_mulhu => |r| alu_rr(r, K::MULHU);
    diff_sltu => |r| alu_rr(r, K::SLTU);
    diff_addi => |r| alu_ri(r, K::ADDI, false);
    diff_andi => |r| alu_ri(r, K::ANDI, false);
    diff_ori => |r| alu_ri(r, K::ORI, false);
    diff_xori => |r| alu_ri(r, K::XORI, false);
    diff_slti => |r| alu_ri(r, K::SLTI, false);
    diff_sltiu => |r| alu_ri(r, K::SLTIU, false);
    diff_muli => |r| alu_ri(r, K::VirtualMULI, true);
    diff_lui => |r| upper_imm(r, K::LUI);
    diff_auipc => |r| upper_imm(r, K::AUIPC);
    diff_pow2 => |r| unary(r, K::VirtualPow2);
    diff_shift_right_bitmask => |r| unary(r, kind_by_name("VirtualShiftRightBitmask"));
    diff_sign_extend_word => |r| unary(r, kind_by_name("VirtualSignExtendWord"));
    diff_zero_extend_word => |r| unary(r, kind_by_name("VirtualZeroExtendWord"));
    diff_srai => |r| shift_imm(r, K::VirtualSRAI);
    diff_srli => |r| shift_imm(r, K::VirtualSRLI);
    diff_srl => |r| shift_reg(r, K::VirtualSRL);
    diff_beq => |r| branch(r, K::BEQ);
    diff_bne => |r| branch(r, K::BNE);
    diff_blt => |r| branch(r, K::BLT);
    diff_bge => |r| branch(r, K::BGE);
    diff_bltu => |r| branch(r, K::BLTU);
    diff_bgeu => |r| branch(r, K::BGEU);
    diff_jal => jal;
    diff_jalr => jalr;
    diff_ld => load;
    diff_sd => store;
    diff_assert_halfword_alignment => |r| assert_align(r, K::VirtualAssertHalfwordAlignment, 2);
    diff_assert_word_alignment => |r| assert_align(r, K::VirtualAssertWordAlignment, 4);
    diff_assert_lte => assert_lte;
    diff_fence => fence;
    diff_host_io => host_io;
    diff_slt => |r| alu_rr(r, K::SLT);
    diff_andn => |r| alu_rr(r, K::ANDN);
    diff_pow2i => |r| imm_j_u64(r, K::VirtualPow2I);
    diff_pow2w => |r| unary(r, K::VirtualPow2W);
    diff_pow2iw => |r| imm_j_u64(r, K::VirtualPow2IW);
    diff_movsign => |r| unary(r, K::VirtualMovsign);
    diff_rev8w => |r| unary(r, kind_by_name("VirtualRev8W"));
    diff_rotri => |r| shift_imm(r, K::VirtualROTRI);
    diff_rotriw => |r| shift_imm(r, K::VirtualROTRIW);
    diff_shift_right_bitmaski => |r| imm_j_u64(r, K::VirtualShiftRightBitmaskI);
    diff_sra_reg => |r| shift_reg(r, K::VirtualSRA);
    diff_xorrot32 => |r| alu_rr(r, K::VirtualXORROT32);
    diff_xorrot24 => |r| alu_rr(r, K::VirtualXORROT24);
    diff_xorrot16 => |r| alu_rr(r, K::VirtualXORROT16);
    diff_xorrot63 => |r| alu_rr(r, K::VirtualXORROT63);
    diff_xorrotw16 => |r| alu_rr(r, K::VirtualXORROTW16);
    diff_xorrotw12 => |r| alu_rr(r, K::VirtualXORROTW12);
    diff_xorrotw8 => |r| alu_rr(r, K::VirtualXORROTW8);
    diff_xorrotw7 => |r| alu_rr(r, K::VirtualXORROTW7);
    diff_assert_eq => assert_eq_gen;
    diff_assert_valid_div0 => assert_valid_div0;
    diff_assert_valid_unsigned_remainder => assert_valid_unsigned_remainder;
    diff_assert_mulu_no_overflow => assert_mulu_no_overflow;
    diff_change_divisor => |r| change_divisor(r, "VirtualChangeDivisor", true);
    diff_change_divisor_w => |r| change_divisor(r, "VirtualChangeDivisorW", false);
    diff_advice_len => advice_len;
    diff_advice_load => advice_load;
}

/// Every supported kind has a differential test above; this pins the count
/// so growing SUPPORTED without adding a test fails. (`VirtualAdvice` is
/// classified `RequiresAdviceGroup`, not listed here: a bare row is a
/// compile error, and the whole-guest gates cover its semantics.)
#[test]
fn supported_kinds_all_have_difftests() {
    assert_eq!(SUPPORTED.len(), 66);
}
