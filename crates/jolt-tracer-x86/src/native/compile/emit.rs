//! Per-row-kind x86-64 templates.
//!
//! Semantics source of truth: the tracer's exec implementations
//! (`tracer/src/instruction/*.rs`); every template must reproduce them
//! bit-for-bit. Key interpreter facts baked in here:
//! - registers are semantically `i64`, stored as raw 64-bit values;
//! - the interpreter pre-increments PC before exec, so link values are
//!   `address + (2 if compressed else 4)` and straight-line code never
//!   maintains PC;
//! - branch/`Jal` targets are `address + imm` (imm is a relative offset);
//! - `Jalr` targets are `(x[rs1] + imm) & !1`;
//! - the virtual right shifts take their shift amount from
//!   `imm.trailing_zeros()` (the immediate is a bitmask, not a shift);
//! - guest termination is the PC-stall convention: a jump/taken branch whose
//!   target equals its own source address executes once, then execution
//!   stops.

use dynasm::dynasm;
use dynasmrt::{AssemblyOffset, DynasmApi, DynasmLabelApi};
use jolt_program::execution::TraceError;
use jolt_riscv::{JoltInstructionKind, JoltInstructionRow};

use super::super::helpers;
use super::super::state::{
    advice_slot_offset, reg_offset, ExitReason, OBSERVATION_SIZE, OBS_RAM_ADDRESS, OBS_RAM_POST,
    OBS_RAM_PRE, OBS_RD_POST, OBS_RD_PRE, OBS_ROW_INDEX, OBS_RS1, OBS_RS2, OFF_EXIT,
    OFF_FAULT_ADDR, OFF_MEM_BASE, OFF_MEM_SIZE, OFF_OBS_CURSOR, OFF_OBS_END, OFF_PC, OFF_ROW_LIMIT,
    OFF_TRACE_LEN,
};
use super::emitter::{EmitOutcome, RowEmitter};
use super::{EmitMode, Emitter};

/// The dynasm-template emitter: the production implementor of the
/// [`RowEmitter`] seam, covering every final-bytecode row kind.
pub struct DynasmEmitter;

impl RowEmitter for DynasmEmitter {
    fn emit_row(
        &self,
        cx: &mut Emitter,
        row: &JoltInstructionRow,
    ) -> Result<EmitOutcome, TraceError> {
        emit_row_template(cx, row)
    }

    fn emit_advice_compute(&self, cx: &mut Emitter, job_index: usize) {
        advice_compute(cx, job_index);
    }
}

const RAX: u8 = 0;
const RCX: u8 = 1;
const RDX: u8 = 2;

const RAM_START: u64 = common::constants::RAM_START_ADDRESS;

fn load_reg(e: &mut Emitter, gpr: u8, reg: Option<u8>) {
    match reg {
        None | Some(0) => dynasm!(e.ops ; .arch x64 ; xor Rq(gpr), Rq(gpr)),
        Some(r) => dynasm!(e.ops ; .arch x64 ; mov Rq(gpr), QWORD [r12 + reg_offset(r)]),
    }
}

fn store_rd(e: &mut Emitter, gpr: u8, rd: Option<u8>) {
    if let Some(r) = rd {
        if r != 0 {
            dynasm!(e.ops ; .arch x64 ; mov QWORD [r12 + reg_offset(r)], Rq(gpr));
        }
    }
}

/// Apply a binary op with the guest register `reg` as the second operand,
/// reading it straight from the state plane (`op rax, [state+off]`) instead
/// of loading it into a scratch register first. Saves one instruction and one
/// register per ALU row, the most frequent shape in the bytecode.
fn alu_reg_operand(e: &mut Emitter, op: AluRR, dst: u8, reg: Option<u8>) {
    let Some(r) = reg.filter(|r| *r != 0) else {
        // x0: fold the identity/annihilator rather than touching memory.
        match op {
            AluRR::Add | AluRR::Sub | AluRR::Or | AluRR::Xor => {}
            AluRR::And | AluRR::Mul => dynasm!(e.ops ; .arch x64 ; xor Rq(dst), Rq(dst)),
        }
        return;
    };
    let offset = reg_offset(r);
    match op {
        AluRR::Add => dynasm!(e.ops ; .arch x64 ; add Rq(dst), QWORD [r12 + offset]),
        AluRR::Sub => dynasm!(e.ops ; .arch x64 ; sub Rq(dst), QWORD [r12 + offset]),
        AluRR::And => dynasm!(e.ops ; .arch x64 ; and Rq(dst), QWORD [r12 + offset]),
        AluRR::Or => dynasm!(e.ops ; .arch x64 ; or Rq(dst), QWORD [r12 + offset]),
        AluRR::Xor => dynasm!(e.ops ; .arch x64 ; xor Rq(dst), QWORD [r12 + offset]),
        AluRR::Mul => dynasm!(e.ops ; .arch x64 ; imul Rq(dst), QWORD [r12 + offset]),
    }
}

/// Compare against a guest register straight from the state plane.
fn cmp_reg_operand(e: &mut Emitter, dst: u8, reg: Option<u8>) {
    if let Some(r) = reg.filter(|r| *r != 0) {
        dynasm!(e.ops ; .arch x64 ; cmp Rq(dst), QWORD [r12 + reg_offset(r)]);
    } else {
        dynasm!(e.ops ; .arch x64 ; cmp Rq(dst), 0);
    }
}

fn load_imm(e: &mut Emitter, gpr: u8, value: i64) {
    if let Ok(v) = i32::try_from(value) {
        dynasm!(e.ops ; .arch x64 ; mov Rq(gpr), v);
    } else {
        dynasm!(e.ops ; .arch x64 ; mov Rq(gpr), QWORD value);
    }
}

/// Call a sysv64 helper: rdi = state, rsi/rdx = args (already set by caller
/// emission). Syncs trace_len before the call and checks the exit flag after.
fn call_helper(e: &mut Emitter, f: usize) {
    dynasm!(e.ops
        ; .arch x64
        ; mov QWORD [r12 + OFF_TRACE_LEN], r14
        ; mov rdi, r12
        ; mov rax, QWORD f as i64
        ; call rax
        ; cmp QWORD [r12 + OFF_EXIT], 0
        ; jne ->exit
    );
}

/// Terminal (PC-stall) sequence: record this row already happened (r14 was
/// incremented at row start), publish pc/exit, leave.
fn terminal(e: &mut Emitter, address: u64) {
    dynasm!(e.ops
        ; .arch x64
        ; mov rcx, QWORD address as i64
        ; mov QWORD [r12 + OFF_PC], rcx
        ; mov QWORD [r12 + OFF_EXIT], ExitReason::Terminated as u64 as i32
        ; jmp ->exit
    );
}

/// Indirect dispatch on the guest address in rax (clobbers rcx, rdx).
fn dispatch(e: &mut Emitter) {
    let span = e.text_span as i32;
    dynasm!(e.ops
        ; .arch x64
        ; mov rcx, rax
        ; mov rdx, QWORD e.text_base as i64
        ; sub rcx, rdx
        ; cmp rcx, span
        ; jae ->bad_jump
        ; mov rdx, QWORD [r15 + rcx * 4]
        ; jmp rdx
    );
}

/// Function prologue: pin registers, dispatch to `state.pc`.
pub fn prologue(e: &mut Emitter) -> AssemblyOffset {
    let entry = e.ops.offset();
    dynasm!(e.ops
        ; .arch x64
        ; push r12
        ; push r13
        ; push r14
        ; push r15
        ; sub rsp, 8
        ; mov r12, rdi
        ; mov r13, QWORD [r12 + OFF_MEM_BASE]
        ; mov r14, QWORD [r12 + OFF_TRACE_LEN]
        ; mov r15, rsi
        ; mov rax, QWORD [r12 + OFF_PC]
    );
    dispatch(e);
    entry
}

/// Emitted at each group start: if the row budget is spent, publish this
/// group's (statically known) address as the resume PC and leave. Rows are
/// only ever emitted whole, so a resumed run repeats no row.
pub fn group_pause_check(e: &mut Emitter, address: u64) {
    dynasm!(e.ops
        ; .arch x64
        ; cmp r14, QWORD [r12 + OFF_ROW_LIMIT]
        ; jb >go
        ; mov rax, QWORD address as i64
        ; mov QWORD [r12 + OFF_PC], rax
        ; mov QWORD [r12 + OFF_EXIT], ExitReason::Paused as u64 as i32
        ; jmp ->exit
        ; go:
    );
}

pub struct Stubs {
    pub bad_jump: AssemblyOffset,
}

/// Falling off the end of the compiled program is a bad jump (rax holds pc).
pub fn jump_to_bad_jump(e: &mut Emitter) {
    dynasm!(e.ops ; .arch x64 ; jmp ->bad_jump);
}

pub fn stubs(e: &mut Emitter) -> Stubs {
    let bad_jump = e.ops.offset();
    dynasm!(e.ops
        ; .arch x64
        ; ->obs_overflow:
        ; mov QWORD [r12 + OFF_EXIT], ExitReason::FaultObservationOverflow as u64 as i32
        ; jmp ->exit
        ; ->bad_jump:
        ; mov QWORD [r12 + OFF_FAULT_ADDR], rax
        ; mov QWORD [r12 + OFF_EXIT], ExitReason::FaultBadJumpTarget as u64 as i32
        ; ->exit:
        ; mov QWORD [r12 + OFF_TRACE_LEN], r14
        ; add rsp, 8
        ; pop r15
        ; pop r14
        ; pop r13
        ; pop r12
        ; ret
    );
    Stubs { bad_jump }
}

enum AluRR {
    Add,
    Sub,
    And,
    Or,
    Xor,
    Mul,
}

fn alu_rr(e: &mut Emitter, row: &JoltInstructionRow, op: AluRR) {
    load_reg(e, RAX, row.operands.rs1);
    alu_reg_operand(e, op, RAX, row.operands.rs2);
    store_rd(e, RAX, row.operands.rd);
}

fn alu_ri(e: &mut Emitter, row: &JoltInstructionRow, op: AluRR) {
    load_reg(e, RAX, row.operands.rs1);
    load_imm(e, RCX, row.operands.imm as i64);
    match op {
        AluRR::Add => dynasm!(e.ops ; .arch x64 ; add rax, rcx),
        AluRR::And => dynasm!(e.ops ; .arch x64 ; and rax, rcx),
        AluRR::Or => dynasm!(e.ops ; .arch x64 ; or rax, rcx),
        AluRR::Xor => dynasm!(e.ops ; .arch x64 ; xor rax, rcx),
        AluRR::Mul => dynasm!(e.ops ; .arch x64 ; imul rax, rcx),
        AluRR::Sub => unreachable!("no reg-imm subtract in the row set"),
    }
    store_rd(e, RAX, row.operands.rd);
}

enum Cc {
    Eq,
    Ne,
    LtSigned,
    GeSigned,
    LtUnsigned,
    GeUnsigned,
}

fn branch(e: &mut Emitter, row: &JoltInstructionRow, cc: Cc) {
    let address = row.address as u64;
    let target = (row.address as i64).wrapping_add(row.operands.imm as i64) as u64;
    load_reg(e, RAX, row.operands.rs1);
    cmp_reg_operand(e, RAX, row.operands.rs2);
    if target == address {
        // Taken branch to itself terminates (PC-stall). Invert: skip the
        // terminal sequence when not taken.
        match cc {
            Cc::Eq => dynasm!(e.ops ; .arch x64 ; jne >fall),
            Cc::Ne => dynasm!(e.ops ; .arch x64 ; je >fall),
            Cc::LtSigned => dynasm!(e.ops ; .arch x64 ; jge >fall),
            Cc::GeSigned => dynasm!(e.ops ; .arch x64 ; jl >fall),
            Cc::LtUnsigned => dynasm!(e.ops ; .arch x64 ; jae >fall),
            Cc::GeUnsigned => dynasm!(e.ops ; .arch x64 ; jb >fall),
        }
        terminal(e, address);
        dynasm!(e.ops ; .arch x64 ; fall:);
    } else {
        let label = e.label_for(target);
        match cc {
            Cc::Eq => dynasm!(e.ops ; .arch x64 ; je =>label),
            Cc::Ne => dynasm!(e.ops ; .arch x64 ; jne =>label),
            Cc::LtSigned => dynasm!(e.ops ; .arch x64 ; jl =>label),
            Cc::GeSigned => dynasm!(e.ops ; .arch x64 ; jge =>label),
            Cc::LtUnsigned => dynasm!(e.ops ; .arch x64 ; jb =>label),
            Cc::GeUnsigned => dynasm!(e.ops ; .arch x64 ; jae =>label),
        }
    }
}

fn set_cc_less(e: &mut Emitter, signed: bool, rd: Option<u8>) {
    if signed {
        dynasm!(e.ops ; .arch x64 ; setl al);
    } else {
        dynasm!(e.ops ; .arch x64 ; setb al);
    }
    dynasm!(e.ops ; .arch x64 ; movzx rax, al);
    store_rd(e, RAX, rd);
}

fn link_value(row: &JoltInstructionRow) -> i64 {
    row.address as i64 + if row.is_compressed { 2 } else { 4 }
}

/// `Ld`: fast path hits the RAM plane; device/unaligned/OOB funnel to the
/// slow helper. Clobbers rax/rcx/rdx (+ helper scratch on the cold path).
fn load_doubleword(e: &mut Emitter, row: &JoltInstructionRow) {
    // EA = (x[rs1] as u64).wrapping_add(imm as i32 as u64) — exact cast chain.
    let offset = row.operands.imm as i32;
    load_reg(e, RAX, row.operands.rs1);
    dynasm!(e.ops ; .arch x64 ; add rax, offset);
    if e.mode == EmitMode::Record {
        // The effective address is 8-aligned on the success path, so the
        // reference's word-aligned floor equals it.
        obs_reload(e);
        dynasm!(e.ops ; .arch x64 ; mov QWORD [r10 + OBS_RAM_ADDRESS], rax);
    }
    dynasm!(e.ops
        ; .arch x64
        ; mov rcx, rax
        ; mov rdx, QWORD RAM_START as i64
        ; sub rcx, rdx
        ; mov rdx, QWORD [r12 + OFF_MEM_SIZE]
        ; sub rdx, 7
        ; cmp rcx, rdx
        ; jae >slow
        ; test al, 7
        ; jnz >slow
        ; mov rax, QWORD [r13 + rcx]
        ; jmp >done
        ; slow:
        ; mov rsi, rax
    );
    call_helper(e, helpers::slow_load_doubleword as *const () as usize);
    dynasm!(e.ops ; .arch x64 ; done:);
    if e.mode == EmitMode::Record {
        // The reference records the whole doubleword read, whichever path
        // produced it.
        obs_reload(e);
        dynasm!(e.ops ; .arch x64 ; mov QWORD [r10 + OBS_RAM_PRE], rax);
    }
    store_rd(e, RAX, row.operands.rd);
}

/// `Sd`: mirror of `Ld` with the store value in rsi's place.
fn store_doubleword(e: &mut Emitter, row: &JoltInstructionRow) {
    // EA = x[rs1].wrapping_add(imm) as u64 — imm used as full i64 here.
    load_reg(e, RAX, row.operands.rs1);
    load_imm(e, RCX, row.operands.imm as i64);
    load_reg(e, RDX, row.operands.rs2);
    dynasm!(e.ops ; .arch x64 ; add rax, rcx);
    if e.mode == EmitMode::Record {
        // The reference records the raw effective address and the stored
        // value; the pre-value is captured on whichever path performs the
        // write (fast path below, helper for the device region).
        obs_reload(e);
        dynasm!(e.ops
            ; .arch x64
            ; mov QWORD [r10 + OBS_RAM_ADDRESS], rax
            ; mov QWORD [r10 + OBS_RAM_POST], rdx
        );
    }
    dynasm!(e.ops
        ; .arch x64
        ; mov rcx, rax
        ; mov rsi, QWORD RAM_START as i64
        ; sub rcx, rsi
        ; mov rsi, QWORD [r12 + OFF_MEM_SIZE]
        ; sub rsi, 7
        ; cmp rcx, rsi
        ; jae >slow
        ; test al, 7
        ; jnz >slow
    );
    if e.mode == EmitMode::Record {
        obs_reload(e);
        dynasm!(e.ops
            ; .arch x64
            ; mov r8, QWORD [r13 + rcx]
            ; mov QWORD [r10 + OBS_RAM_PRE], r8
        );
    }
    dynasm!(e.ops
        ; .arch x64
        ; mov QWORD [r13 + rcx], rdx
        ; jmp >done
        ; slow:
        ; mov rsi, rax
        // value already in rdx (helper arg 3)
    );
    call_helper(e, helpers::slow_store_doubleword as *const () as usize);
    dynasm!(e.ops ; .arch x64 ; done:);
}

/// Alignment asserts: compute EA, test low bits, call the fatal helper on
/// failure. `mask` is 1 (halfword) or 3 (word); `code` selects the message.
fn assert_alignment(e: &mut Emitter, row: &JoltInstructionRow, mask: i8, code: u64) {
    load_reg(e, RAX, row.operands.rs1);
    load_imm(e, RCX, row.operands.imm as i64);
    dynasm!(e.ops
        ; .arch x64
        ; add rax, rcx
        ; test al, mask
        ; jz >ok
        ; mov rsi, code as i32
        ; mov rdx, rax
    );
    call_helper(e, helpers::assert_failed as *const () as usize);
    // assert_failed always sets the exit flag, so call_helper's check exits.
    dynasm!(e.ops ; .arch x64 ; ok:);
}

/// Load the low 32 bits of a guest register, zero-extended (32-bit mov).
fn load_reg32(e: &mut Emitter, gpr: u8, reg: Option<u8>) {
    match reg {
        None | Some(0) => dynasm!(e.ops ; .arch x64 ; xor Rd(gpr), Rd(gpr)),
        Some(r) => dynasm!(e.ops ; .arch x64 ; mov Rd(gpr), DWORD [r12 + reg_offset(r)]),
    }
}

/// Load the low 32 bits of a guest register, sign-extended to 64.
fn load_reg32_sext(e: &mut Emitter, gpr: u8, reg: Option<u8>) {
    match reg {
        None | Some(0) => dynasm!(e.ops ; .arch x64 ; xor Rq(gpr), Rq(gpr)),
        Some(r) => dynasm!(e.ops ; .arch x64 ; movsxd Rq(gpr), DWORD [r12 + reg_offset(r)]),
    }
}

/// `(x[rs1] ^ x[rs2]).rotate_right(n)`, 64-bit.
fn xor_rot(e: &mut Emitter, row: &JoltInstructionRow, n: i8) {
    load_reg(e, RAX, row.operands.rs1);
    load_reg(e, RCX, row.operands.rs2);
    dynasm!(e.ops ; .arch x64 ; xor rax, rcx ; ror rax, n);
    store_rd(e, RAX, row.operands.rd);
}

/// `((x[rs1] as u32) ^ (x[rs2] as u32)).rotate_right(n)`, zero-extended.
fn xor_rotw(e: &mut Emitter, row: &JoltInstructionRow, n: i8) {
    load_reg32(e, RAX, row.operands.rs1);
    load_reg32(e, RCX, row.operands.rs2);
    dynasm!(e.ops ; .arch x64 ; xor eax, ecx ; ror eax, n);
    store_rd(e, RAX, row.operands.rd);
}

/// Emit a group's advice computation (job index in rsi), before its rows.
pub fn advice_compute(e: &mut Emitter, job_index: usize) {
    dynasm!(e.ops ; .arch x64 ; mov rsi, job_index as i32);
    call_helper(e, helpers::advice_compute as *const () as usize);
}

/// Record mode: load the observation cursor into r10 and bounds-check it.
/// Kept live only within one row's emission (helper calls clobber r10).
fn obs_open(e: &mut Emitter, row_index: usize) {
    dynasm!(e.ops
        ; .arch x64
        ; mov r10, QWORD [r12 + OFF_OBS_CURSOR]
        ; cmp r10, QWORD [r12 + OFF_OBS_END]
        ; jae ->obs_overflow
        ; mov rax, QWORD row_index as i64
        ; mov QWORD [r10 + OBS_ROW_INDEX], rax
    );
}

/// Reload the cursor into r10. Every capture site does this: a row whose
/// template calls a helper has had r10 clobbered (caller-saved), and the
/// cursor is only advanced by `obs_close`, so the reload always lands on the
/// same slot.
fn obs_reload(e: &mut Emitter) {
    dynasm!(e.ops ; .arch x64 ; mov r10, QWORD [r12 + OFF_OBS_CURSOR]);
}

/// Record mode: advance the cursor past this row's slot.
fn obs_close(e: &mut Emitter) {
    dynasm!(e.ops
        ; .arch x64
        ; mov r10, QWORD [r12 + OFF_OBS_CURSOR]
        ; add r10, OBSERVATION_SIZE
        ; mov QWORD [r12 + OFF_OBS_CURSOR], r10
    );
}

/// Record mode: capture the register values this row reads and the
/// destination's pre-value, before the row's own template runs.
fn obs_registers_pre(e: &mut Emitter, row: &JoltInstructionRow) {
    obs_reload(e);
    for (slot, register) in [
        (OBS_RS1, row.operands.rs1),
        (OBS_RS2, row.operands.rs2),
        (OBS_RD_PRE, row.operands.rd),
    ] {
        match register {
            // x0 reads as zero, matching normalize_register_value.
            None | Some(0) => dynasm!(e.ops
                ; .arch x64
                ; xor rax, rax
                ; mov QWORD [r10 + slot], rax
            ),
            Some(r) => dynasm!(e.ops
                ; .arch x64
                ; mov rax, QWORD [r12 + reg_offset(r)]
                ; mov QWORD [r10 + slot], rax
            ),
        }
    }
}

/// Record mode: store a statically known rd post-value (control-flow rows
/// write their observation before transferring, so their post-value must be
/// known without executing the template; for Jal/Jalr it is the link).
fn obs_rd_post_static(e: &mut Emitter, row: &JoltInstructionRow, value: i64) {
    obs_reload(e);
    let value = match row.operands.rd {
        None | Some(0) => 0,
        Some(_) => value,
    };
    dynasm!(e.ops
        ; .arch x64
        ; mov rax, QWORD value
        ; mov QWORD [r10 + OBS_RD_POST], rax
    );
}

/// Record mode: capture the destination's post-value, after the template ran.
fn obs_rd_post(e: &mut Emitter, row: &JoltInstructionRow) {
    obs_reload(e);
    match row.operands.rd {
        None | Some(0) => dynasm!(e.ops
            ; .arch x64
            ; xor rax, rax
            ; mov QWORD [r10 + OBS_RD_POST], rax
        ),
        Some(r) => dynasm!(e.ops
            ; .arch x64
            ; mov rax, QWORD [r12 + reg_offset(r)]
            ; mov QWORD [r10 + OBS_RD_POST], rax
        ),
    }
}

fn emit_row_template(e: &mut Emitter, row: &JoltInstructionRow) -> Result<EmitOutcome, TraceError> {
    use JoltInstructionKind as K;

    // Every row is one trace row.
    dynasm!(e.ops ; .arch x64 ; inc r14);

    // Record mode brackets each row's template with value capture. Rows that
    // transfer control never reach code emitted after their template, so
    // theirs is written up front: branches have no destination, and Jal/Jalr
    // post-values are the statically known link.
    let record = e.mode == EmitMode::Record;
    let transfers_control = matches!(
        row.instruction_kind,
        K::Beq(_)
            | K::Bne(_)
            | K::Blt(_)
            | K::Bge(_)
            | K::BltU(_)
            | K::BgeU(_)
            | K::Jal(_)
            | K::Jalr(_)
    );
    if record {
        obs_open(e, e.row_index);
        obs_registers_pre(e, row);
        if transfers_control {
            let link = match row.instruction_kind {
                K::Jal(_) | K::Jalr(_) => link_value(row),
                _ => 0,
            };
            obs_rd_post_static(e, row, link);
            obs_close(e);
        }
    }

    match &row.instruction_kind {
        K::Add(_) => alu_rr(e, row, AluRR::Add),
        K::Sub(_) => alu_rr(e, row, AluRR::Sub),
        K::And(_) => alu_rr(e, row, AluRR::And),
        K::Or(_) => alu_rr(e, row, AluRR::Or),
        K::Xor(_) => alu_rr(e, row, AluRR::Xor),
        K::Mul(_) => alu_rr(e, row, AluRR::Mul),
        K::Addi(_) => alu_ri(e, row, AluRR::Add),
        K::AndI(_) => alu_ri(e, row, AluRR::And),
        K::OrI(_) => alu_ri(e, row, AluRR::Or),
        K::XorI(_) => alu_ri(e, row, AluRR::Xor),
        K::MulI(_) => alu_ri(e, row, AluRR::Mul),

        K::MulHU(_) => {
            // rd = high 64 bits of unsigned x[rs1] * x[rs2].
            load_reg(e, RAX, row.operands.rs1);
            load_reg(e, RCX, row.operands.rs2);
            dynasm!(e.ops ; .arch x64 ; mul rcx);
            store_rd(e, RDX, row.operands.rd);
        }

        K::SltU(_) => {
            load_reg(e, RAX, row.operands.rs1);
            cmp_reg_operand(e, RAX, row.operands.rs2);
            set_cc_less(e, false, row.operands.rd);
        }
        K::SltI(_) => {
            load_reg(e, RAX, row.operands.rs1);
            load_imm(e, RCX, row.operands.imm as i64);
            dynasm!(e.ops ; .arch x64 ; cmp rax, rcx);
            set_cc_less(e, true, row.operands.rd);
        }
        K::SltIU(_) => {
            load_reg(e, RAX, row.operands.rs1);
            load_imm(e, RCX, row.operands.imm as i64);
            dynasm!(e.ops ; .arch x64 ; cmp rax, rcx);
            set_cc_less(e, false, row.operands.rd);
        }

        K::Lui(_) => {
            load_imm(e, RAX, row.operands.imm as i64);
            store_rd(e, RAX, row.operands.rd);
        }
        K::Auipc(_) => {
            // rd = address + imm, fully static.
            let value = (row.address as i64).wrapping_add(row.operands.imm as i64);
            load_imm(e, RAX, value);
            store_rd(e, RAX, row.operands.rd);
        }

        K::Pow2(_) => {
            // rd = 1 << (x[rs1] % 64); shl's cl masking is exactly mod 64.
            load_reg(e, RCX, row.operands.rs1);
            dynasm!(e.ops ; .arch x64 ; mov eax, 1 ; shl rax, cl);
            store_rd(e, RAX, row.operands.rd);
        }
        K::VirtualShiftRightBitmask(_) => {
            // rd = u64::MAX << (x[rs1] & 63) — bits [63:shift] set.
            load_reg(e, RCX, row.operands.rs1);
            dynasm!(e.ops ; .arch x64 ; mov rax, -1 ; shl rax, cl);
            store_rd(e, RAX, row.operands.rd);
        }
        K::VirtualSignExtendWord(_) => {
            load_reg(e, RAX, row.operands.rs1);
            dynasm!(e.ops ; .arch x64 ; movsxd rax, eax);
            store_rd(e, RAX, row.operands.rd);
        }
        K::VirtualZeroExtendWord(_) => {
            load_reg(e, RAX, row.operands.rs1);
            dynasm!(e.ops ; .arch x64 ; mov eax, eax);
            store_rd(e, RAX, row.operands.rd);
        }
        K::VirtualSrai(_) => {
            // Shift amount = imm.trailing_zeros() (imm is a bitmask);
            // wrapping_shr semantics = mod 64.
            let shift = ((row.operands.imm as u64).trailing_zeros() % 64) as i8;
            load_reg(e, RAX, row.operands.rs1);
            dynasm!(e.ops ; .arch x64 ; sar rax, shift);
            store_rd(e, RAX, row.operands.rd);
        }
        K::VirtualSrli(_) => {
            let shift = ((row.operands.imm as u64).trailing_zeros() % 64) as i8;
            load_reg(e, RAX, row.operands.rs1);
            dynasm!(e.ops ; .arch x64 ; shr rax, shift);
            store_rd(e, RAX, row.operands.rd);
        }
        K::VirtualSrl(_) => {
            // Shift = x[rs2].trailing_zeros(); tzcnt(0) = 64 → shr masks to
            // 0, matching wrapping_shr(64). Requires BMI1 (checked once).
            load_reg(e, RCX, row.operands.rs2);
            load_reg(e, RAX, row.operands.rs1);
            dynasm!(e.ops ; .arch x64 ; tzcnt rcx, rcx ; shr rax, cl);
            store_rd(e, RAX, row.operands.rd);
        }

        K::Beq(_) => branch(e, row, Cc::Eq),
        K::Bne(_) => branch(e, row, Cc::Ne),
        K::Blt(_) => branch(e, row, Cc::LtSigned),
        K::Bge(_) => branch(e, row, Cc::GeSigned),
        K::BltU(_) => branch(e, row, Cc::LtUnsigned),
        K::BgeU(_) => branch(e, row, Cc::GeUnsigned),

        K::Jal(_) => {
            load_imm(e, RAX, link_value(row));
            store_rd(e, RAX, row.operands.rd);
            let target = (row.address as i64).wrapping_add(row.operands.imm as i64) as u64;
            if target == row.address as u64 {
                terminal(e, target);
            } else {
                let label = e.label_for(target);
                dynasm!(e.ops ; .arch x64 ; jmp =>label);
            }
        }
        K::Jalr(_) => {
            // target = (x[rs1] + imm) & !1, computed before the link write
            // (rs1 may alias rd).
            load_reg(e, RAX, row.operands.rs1);
            load_imm(e, RCX, row.operands.imm as i64);
            dynasm!(e.ops ; .arch x64 ; add rax, rcx ; and rax, -2);
            load_imm(e, RCX, link_value(row));
            store_rd(e, RCX, row.operands.rd);
            // PC-stall check against this row's own source address.
            dynasm!(e.ops ; .arch x64 ; mov rcx, QWORD row.address as i64 ; cmp rax, rcx ; jne >go);
            terminal(e, row.address as u64);
            dynasm!(e.ops ; .arch x64 ; go:);
            dispatch(e);
        }

        K::Ld(_) => load_doubleword(e, row),
        K::Sd(_) => store_doubleword(e, row),

        K::AssertHalfwordAlignment(_) => assert_alignment(e, row, 1, 0),
        K::AssertWordAlignment(_) => assert_alignment(e, row, 3, 1),
        K::AssertLte(_) => {
            // assert!(x[rs1] as u64 <= x[rs2] as u64) — unsigned.
            load_reg(e, RAX, row.operands.rs1);
            load_reg(e, RCX, row.operands.rs2);
            dynasm!(e.ops
                ; .arch x64
                ; cmp rax, rcx
                ; jbe >ok
                ; mov rsi, 2
                ; mov rdx, rax
            );
            call_helper(e, helpers::assert_failed as *const () as usize);
            dynasm!(e.ops ; .arch x64 ; ok:);
        }

        K::Slt(_) => {
            load_reg(e, RAX, row.operands.rs1);
            cmp_reg_operand(e, RAX, row.operands.rs2);
            set_cc_less(e, true, row.operands.rd);
        }
        K::Andn(_) => {
            // rd = x[rs1] & !x[rs2]
            load_reg(e, RAX, row.operands.rs1);
            load_reg(e, RCX, row.operands.rs2);
            dynasm!(e.ops ; .arch x64 ; not rcx ; and rax, rcx);
            store_rd(e, RAX, row.operands.rd);
        }
        K::Pow2I(_) => {
            // Static: rd = 1 << (imm % 64).
            let value = 1i64 << ((row.operands.imm as u64) % 64);
            load_imm(e, RAX, value);
            store_rd(e, RAX, row.operands.rd);
        }
        K::Pow2IW(_) => {
            let value = 1i64 << ((row.operands.imm as u64) % 32);
            load_imm(e, RAX, value);
            store_rd(e, RAX, row.operands.rd);
        }
        K::Pow2W(_) => {
            // rd = 1 << ((x[rs1] as u64) % 32)
            load_reg(e, RCX, row.operands.rs1);
            dynasm!(e.ops ; .arch x64 ; and ecx, 31 ; mov eax, 1 ; shl rax, cl);
            store_rd(e, RAX, row.operands.rd);
        }
        K::MovSign(_) => {
            // rd = -1 if the sign bit is set else 0.
            load_reg(e, RAX, row.operands.rs1);
            dynasm!(e.ops ; .arch x64 ; sar rax, 63);
            store_rd(e, RAX, row.operands.rd);
        }
        K::VirtualRev8W(_) => {
            // Byte-reverse each 32-bit half independently: bswap swaps all 8
            // bytes (including the halves); ror 32 swaps the halves back.
            load_reg(e, RAX, row.operands.rs1);
            dynasm!(e.ops ; .arch x64 ; bswap rax ; ror rax, 32);
            store_rd(e, RAX, row.operands.rd);
        }
        K::VirtualRotri(_) => {
            // Shift amount = imm.trailing_zeros() (bitmask encoding), mod 64.
            let shift = (((row.operands.imm as u64).trailing_zeros()) % 64) as i8;
            load_reg(e, RAX, row.operands.rs1);
            if shift != 0 {
                dynasm!(e.ops ; .arch x64 ; ror rax, shift);
            }
            store_rd(e, RAX, row.operands.rd);
        }
        K::VirtualRotriw(_) => {
            // 32-bit rotate, result zero-extended (NOT sign-extended);
            // shift = tz(imm).min(32), and a u32 rotate by 32 is the identity.
            let shift = ((row.operands.imm as u64).trailing_zeros().min(32) & 31) as i8;
            load_reg32(e, RAX, row.operands.rs1);
            if shift != 0 {
                dynasm!(e.ops ; .arch x64 ; ror eax, shift);
            }
            store_rd(e, RAX, row.operands.rd);
        }
        K::VirtualShiftRightBitmaski(_) => {
            // Static: bits [63:shift] set (all-ones when shift == 0).
            let shift = (row.operands.imm as u64) % 64;
            let value = (((1u128 << (64 - shift)) - 1) << shift) as u64 as i64;
            load_imm(e, RAX, value);
            store_rd(e, RAX, row.operands.rd);
        }
        K::VirtualSra(_) => {
            // Arithmetic sibling of VirtualSrl: shift = tz(x[rs2]), tzcnt(0)=64
            // -> sar masks to 0, matching wrapping_shr(64).
            load_reg(e, RCX, row.operands.rs2);
            load_reg(e, RAX, row.operands.rs1);
            dynasm!(e.ops ; .arch x64 ; tzcnt rcx, rcx ; sar rax, cl);
            store_rd(e, RAX, row.operands.rd);
        }
        K::VirtualXorRot32(_) => xor_rot(e, row, 32),
        K::VirtualXorRot24(_) => xor_rot(e, row, 24),
        K::VirtualXorRot16(_) => xor_rot(e, row, 16),
        K::VirtualXorRot63(_) => xor_rot(e, row, 63),
        K::VirtualXorRotW16(_) => xor_rotw(e, row, 16),
        K::VirtualXorRotW12(_) => xor_rotw(e, row, 12),
        K::VirtualXorRotW8(_) => xor_rotw(e, row, 8),
        K::VirtualXorRotW7(_) => xor_rotw(e, row, 7),
        K::AssertEq(_) => {
            // imm == 0: hard assert. imm != 0: "spoil" mode, warn-and-continue
            // in the interpreter; a no-op here (registers unaffected).
            if row.operands.imm == 0 {
                load_reg(e, RAX, row.operands.rs1);
                load_reg(e, RCX, row.operands.rs2);
                dynasm!(e.ops
                    ; .arch x64
                    ; cmp rax, rcx
                    ; je >ok
                    ; mov rsi, 3
                    ; mov rdx, rax
                );
                call_helper(e, helpers::assert_failed as *const () as usize);
                dynasm!(e.ops ; .arch x64 ; ok:);
            }
        }
        K::AssertValidDiv0(_) => {
            // divisor == 0 implies quotient == u64::MAX.
            load_reg(e, RAX, row.operands.rs1);
            load_reg(e, RCX, row.operands.rs2);
            dynasm!(e.ops
                ; .arch x64
                ; test rax, rax
                ; jnz >ok
                ; cmp rcx, -1
                ; je >ok
                ; mov rsi, 4
                ; mov rdx, rcx
            );
            call_helper(e, helpers::assert_failed as *const () as usize);
            dynasm!(e.ops ; .arch x64 ; ok:);
        }
        K::AssertValidUnsignedRemainder(_) => {
            // divisor == 0 || remainder < divisor (unsigned).
            load_reg(e, RAX, row.operands.rs1);
            load_reg(e, RCX, row.operands.rs2);
            dynasm!(e.ops
                ; .arch x64
                ; test rcx, rcx
                ; jz >ok
                ; cmp rax, rcx
                ; jb >ok
                ; mov rsi, 5
                ; mov rdx, rax
            );
            call_helper(e, helpers::assert_failed as *const () as usize);
            dynasm!(e.ops ; .arch x64 ; ok:);
        }
        K::AssertMulUNoOverflow(_) => {
            // (x[rs1] as u64) * (x[rs2] as u64) must not overflow: unsigned
            // mul leaves the high half in rdx.
            load_reg(e, RAX, row.operands.rs1);
            load_reg(e, RCX, row.operands.rs2);
            dynasm!(e.ops
                ; .arch x64
                ; mul rcx
                ; test rdx, rdx
                ; jz >ok
                ; mov rsi, 6
                ; mov rdx, rax
            );
            call_helper(e, helpers::assert_failed as *const () as usize);
            dynasm!(e.ops ; .arch x64 ; ok:);
        }
        K::VirtualChangeDivisor(_) => {
            // rd = 1 if (dividend, divisor) == (i64::MIN, -1) else divisor.
            load_reg(e, RCX, row.operands.rs1);
            load_reg(e, RAX, row.operands.rs2);
            dynasm!(e.ops
                ; .arch x64
                ; mov rdx, QWORD i64::MIN
                ; cmp rcx, rdx
                ; jne >done
                ; cmp rax, -1
                ; jne >done
                ; mov eax, 1
                ; done:
            );
            store_rd(e, RAX, row.operands.rd);
        }
        K::VirtualChangeDivisorW(_) => {
            // 32-bit variant; the else branch sign-extends the low 32 bits of
            // x[rs2] (upper bits discarded).
            load_reg32_sext(e, RCX, row.operands.rs1);
            load_reg32_sext(e, RAX, row.operands.rs2);
            dynasm!(e.ops
                ; .arch x64
                ; cmp ecx, 0x8000_0000u32 as i32
                ; jne >done
                ; cmp eax, -1
                ; jne >done
                ; mov eax, 1
                ; done:
            );
            store_rd(e, RAX, row.operands.rd);
        }
        K::VirtualAdviceLen(_) => {
            call_helper(e, helpers::advice_remaining as *const () as usize);
            store_rd(e, RAX, row.operands.rd);
        }
        K::VirtualAdviceLoad(_) => {
            // num_bytes is a static row immediate (1, 2, 4 or 8); the helper
            // errors on tape exhaustion (interpreter panics there).
            let num_bytes = (row.operands.imm as u64) as i32;
            dynasm!(e.ops ; .arch x64 ; mov rsi, num_bytes);
            call_helper(e, helpers::advice_read as *const () as usize);
            store_rd(e, RAX, row.operands.rd);
        }
        K::VirtualAdvice(_) => {
            // The value comes from this group's advice slots, filled before
            // the group's first row ran; slots are consumed in row order.
            let slot = e.advice_slot;
            if slot >= super::super::state::ADVICE_SLOTS {
                return Err(TraceError::Backend(
                    "too many VirtualAdvice rows in one group",
                ));
            }
            e.advice_slot += 1;
            dynasm!(e.ops ; .arch x64 ; mov rax, QWORD [r12 + advice_slot_offset(slot)]);
            store_rd(e, RAX, row.operands.rd);
        }

        K::Fence(_) => {}

        K::VirtualHostIO(_) => {
            call_helper(e, helpers::host_io as *const () as usize);
        }

        // Every other final kind is implemented above; `Noop` never appears
        // in executable bytecode. Reporting `Unsupported` (rather than
        // erroring) lets another emitter in the set claim the kind; the set
        // raises the fail-fast error when none does.
        other @ K::Noop(_) => {
            let _ = other;
            return Ok(EmitOutcome::Unsupported);
        }
    }
    if record && !transfers_control {
        obs_rd_post(e, row);
        obs_close(e);
    }
    Ok(EmitOutcome::Emitted)
}
