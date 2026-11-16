//! Guest programs that observe the machine-mode trap CSRs and LR/SC.
//!
//! Each test stores a short RV64 program at `DRAM_BASE`, runs it through
//! `Cpu::tick` in trace mode (the path the prover sees: every source
//! instruction executes as its expanded Jolt rows), and checks the registers
//! the program wrote. Expected values are those of the RISC-V privileged
//! specification for an M-mode-only hart with no F/D/V and IALIGN = 16, which
//! is also what the Sail model computes for that configuration.

use common::jolt_device::{JoltDevice, MemoryConfig};

use crate::emulator::cpu::{Cpu, HaltKind};
use crate::emulator::terminal::DummyTerminal;
use crate::instruction::Cycle;

use super::test::{DRAM_BASE, TEST_MEMORY_CAPACITY};

/// Program words start here; the trap handler, if any, at `HANDLER`.
const CODE: u64 = DRAM_BASE;
const HANDLER: u64 = DRAM_BASE + 0x100;
/// Data for LR/SC, inside the 1 KiB program region (writable, not executed).
const DATA: u64 = DRAM_BASE + 0x200;
const MAX_TICKS: usize = 200;

const MSTATUS: u32 = 0x300;
const MTVEC: u32 = 0x305;
const MSCRATCH: u32 = 0x340;
const MEPC: u32 = 0x341;
const MCAUSE: u32 = 0x342;
const MTVAL: u32 = 0x343;

// Register numbers.
const T0: u8 = 5;
const T1: u8 = 6;
const T2: u8 = 7;
const A0: u8 = 10;
const A1: u8 = 11;
const A2: u8 = 12;
const A3: u8 = 13;
const A4: u8 = 14;
const A5: u8 = 15;

fn i_type(imm: u32, rs1: u8, funct3: u32, rd: u8, opcode: u32) -> u32 {
    ((imm & 0xfff) << 20) | ((rs1 as u32) << 15) | (funct3 << 12) | ((rd as u32) << 7) | opcode
}

fn addi(rd: u8, rs1: u8, imm: i32) -> u32 {
    i_type(imm as u32, rs1, 0, rd, 0x13)
}

fn csrrw(rd: u8, csr: u32, rs1: u8) -> u32 {
    i_type(csr, rs1, 1, rd, 0x73)
}

fn csrrs(rd: u8, csr: u32, rs1: u8) -> u32 {
    i_type(csr, rs1, 2, rd, 0x73)
}

fn csrrc(rd: u8, csr: u32, rs1: u8) -> u32 {
    i_type(csr, rs1, 3, rd, 0x73)
}

/// The immediate forms carry the 5-bit `uimm` in the `rs1` field.
fn csrrwi(rd: u8, csr: u32, uimm: u8) -> u32 {
    i_type(csr, uimm, 5, rd, 0x73)
}

fn csrrsi(rd: u8, csr: u32, uimm: u8) -> u32 {
    i_type(csr, uimm, 6, rd, 0x73)
}

fn csrrci(rd: u8, csr: u32, uimm: u8) -> u32 {
    i_type(csr, uimm, 7, rd, 0x73)
}

fn csrr(rd: u8, csr: u32) -> u32 {
    csrrs(rd, csr, 0)
}

fn csrw(csr: u32, rs1: u8) -> u32 {
    csrrw(0, csr, rs1)
}

const ECALL: u32 = 0x0000_0073;
const EBREAK: u32 = 0x0010_0073;
const MRET: u32 = 0x3020_0073;

fn amo(funct5: u32, rd: u8, rs1: u8, rs2: u8, funct3: u32) -> u32 {
    (funct5 << 27)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | ((rd as u32) << 7)
        | 0x2f
}

fn lr_w(rd: u8, rs1: u8) -> u32 {
    amo(0b00010, rd, rs1, 0, 0b010)
}

fn sc_w(rd: u8, rs1: u8, rs2: u8) -> u32 {
    amo(0b00011, rd, rs1, rs2, 0b010)
}

fn lr_d(rd: u8, rs1: u8) -> u32 {
    amo(0b00010, rd, rs1, 0, 0b011)
}

fn sc_d(rd: u8, rs1: u8, rs2: u8) -> u32 {
    amo(0b00011, rd, rs1, rs2, 0b011)
}

/// A hart with 1 MiB of RAM, `program` at `CODE`, `handler` at `HANDLER`,
/// and the given registers preset.
fn hart(program: &[u32], handler: &[u32], registers: &[(u8, u64)]) -> Cpu {
    let memory_config = MemoryConfig {
        heap_size: TEST_MEMORY_CAPACITY,
        program_size: Some(1024),
        ..Default::default()
    };
    let mut cpu = Cpu::new(Box::new(DummyTerminal::default()));
    cpu.get_mut_mmu().jolt_device = Some(JoltDevice::new(&memory_config));
    cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
    for (base, words) in [(CODE, program), (HANDLER, handler)] {
        for (i, word) in words.iter().enumerate() {
            cpu.get_mut_mmu()
                .store_word(base + 4 * i as u64, *word)
                .expect("program store");
        }
    }
    for &(register, value) in registers {
        cpu.x[register as usize] = value as i64;
    }
    cpu.update_pc(CODE);
    cpu
}

/// Runs until the hart halts at `EBREAK`.
fn run(cpu: &mut Cpu) {
    let mut trace: Vec<Cycle> = Vec::new();
    for _ in 0..MAX_TICKS {
        cpu.tick(Some(&mut trace));
        if let Some(halt) = cpu.halted() {
            assert_eq!(halt, HaltKind::Panic);
            return;
        }
    }
    panic!("program did not reach EBREAK within {MAX_TICKS} instructions");
}

fn x(cpu: &Cpu, register: u8) -> u64 {
    cpu.x[register as usize] as u64
}

/// The handler returns past the ECALL: `mepc += 4; mret`.
fn returning_handler(observe: &[u32]) -> Vec<u32> {
    let mut handler = observe.to_vec();
    handler.extend([csrr(T2, MEPC), addi(T2, T2, 4), csrw(MEPC, T2), MRET]);
    handler
}

#[test]
fn mstatus_reads_mpp_machine_at_reset() {
    // An M-only hart's MPP can hold only M, so mstatus reads 0x1800 from reset.
    let mut cpu = hart(&[csrr(A0, MSTATUS), EBREAK], &[], &[]);
    run(&mut cpu);
    assert_eq!(x(&cpu, A0), 0x1800);
}

#[test]
fn ecall_stacks_mie_into_mpie_and_mret_restores_it() {
    //     csrw  mtvec, t0        ; t0 = HANDLER
    //     csrs  mstatus, t1      ; t1 = 8: MIE := 1
    //     ecall
    //     csrr  a1, mstatus      ; after mret
    //     ebreak
    // HANDLER:
    //     csrr  a0, mstatus      ; inside the trap
    //     csrr  a2, mcause
    //     csrr  a3, mtval
    //     csrr  a4, mepc
    //     (mepc += 4; mret)
    let program = [
        csrw(MTVEC, T0),
        csrrs(0, MSTATUS, T1),
        ECALL,
        csrr(A1, MSTATUS),
        EBREAK,
    ];
    let handler = returning_handler(&[
        csrr(A0, MSTATUS),
        csrr(A2, MCAUSE),
        csrr(A3, MTVAL),
        csrr(A4, MEPC),
    ]);
    let mut cpu = hart(&program, &handler, &[(T0, HANDLER), (T1, 8)]);
    run(&mut cpu);
    assert_eq!(
        x(&cpu, A0),
        0x1880,
        "in handler: MPIE = old MIE = 1, MIE = 0"
    );
    assert_eq!(x(&cpu, A1), 0x1888, "after mret: MIE = MPIE = 1, MPIE = 1");
    assert_eq!(x(&cpu, A2), 11, "mcause = environment call from M-mode");
    assert_eq!(x(&cpu, A3), 0, "mtval = 0");
    assert_eq!(x(&cpu, A4), CODE + 8, "mepc = address of the ecall");
}

#[test]
fn ecall_with_interrupts_disabled_clears_mpie() {
    // MIE = 0 and MPIE = 1 before the trap, so the trap replaces MPIE with
    // MIE = 0, and MRET then sets MIE := 0 and MPIE := 1.
    //     csrw  mtvec, t0        ; t0 = HANDLER
    //     csrw  mstatus, t1      ; t1 = 0x80: MPIE = 1, MIE = 0
    //     ecall
    //     csrr  a1, mstatus      ; after mret
    //     ebreak
    // HANDLER:
    //     csrr  a0, mstatus      ; inside the trap
    //     (mepc += 4; mret)
    let program = [
        csrw(MTVEC, T0),
        csrw(MSTATUS, T1),
        ECALL,
        csrr(A1, MSTATUS),
        EBREAK,
    ];
    let handler = returning_handler(&[csrr(A0, MSTATUS)]);
    let mut cpu = hart(&program, &handler, &[(T0, HANDLER), (T1, 0x80)]);
    run(&mut cpu);
    assert_eq!(x(&cpu, A0), 0x1800);
    assert_eq!(x(&cpu, A1), 0x1880);
}

#[test]
fn mret_twice_sets_mpie_then_moves_it_into_mie() {
    //     csrw  mepc, t0         ; t0 = CODE + 12
    //     mret                   ; MIE := MPIE = 0, MPIE := 1
    //     ebreak                 ; (skipped)
    //     csrr  a0, mstatus      ; CODE + 12
    //     csrw  mepc, t1         ; t1 = CODE + 28
    //     mret                   ; MIE := MPIE = 1, MPIE := 1
    //     ebreak                 ; (skipped)
    //     csrr  a1, mstatus      ; CODE + 28
    //     ebreak
    let program = [
        csrw(MEPC, T0),
        MRET,
        EBREAK,
        csrr(A0, MSTATUS),
        csrw(MEPC, T1),
        MRET,
        EBREAK,
        csrr(A1, MSTATUS),
        EBREAK,
    ];
    let mut cpu = hart(&program, &[], &[(T0, CODE + 12), (T1, CODE + 28)]);
    run(&mut cpu);
    assert_eq!(x(&cpu, A0), 0x1880);
    assert_eq!(x(&cpu, A1), 0x1888);
    assert_eq!(cpu.read_pc(), CODE + 32, "halted on the final ebreak");
}

#[test]
fn mstatus_write_keeps_only_mie_and_mpie() {
    //     csrw  mstatus, t0      ; t0 = all ones
    //     csrr  a0, mstatus
    //     csrw  mstatus, x0
    //     csrr  a1, mstatus      ; MPP stays M
    //     csrrw a2, mstatus, t1  ; t1 = 0x8 (MIE); a2 = old value
    //     csrr  a3, mstatus
    let program = [
        csrw(MSTATUS, T0),
        csrr(A0, MSTATUS),
        csrw(MSTATUS, 0),
        csrr(A1, MSTATUS),
        csrrw(A2, MSTATUS, T1),
        csrr(A3, MSTATUS),
        EBREAK,
    ];
    let mut cpu = hart(&program, &[], &[(T0, u64::MAX), (T1, 8)]);
    run(&mut cpu);
    assert_eq!(x(&cpu, A0), 0x1888);
    assert_eq!(x(&cpu, A1), 0x1800);
    assert_eq!(x(&cpu, A2), 0x1800);
    assert_eq!(x(&cpu, A3), 0x1808);
}

#[test]
fn mstatus_set_bits_and_swap_with_rd_equal_rs1() {
    //     csrrs t0, mstatus, t0  ; t0 = 0xffff: rd = rs1
    //     csrr  a0, mstatus
    //     csrrw t1, mstatus, t1  ; t1 = 0: rd = rs1
    //     csrr  a1, mstatus
    let program = [
        csrrs(T0, MSTATUS, T0),
        csrr(A0, MSTATUS),
        csrrw(T1, MSTATUS, T1),
        csrr(A1, MSTATUS),
        EBREAK,
    ];
    let mut cpu = hart(&program, &[], &[(T0, 0xffff), (T1, 0)]);
    run(&mut cpu);
    assert_eq!(x(&cpu, T0), 0x1800, "old value");
    assert_eq!(x(&cpu, A0), 0x1888);
    assert_eq!(x(&cpu, T1), 0x1888, "old value");
    assert_eq!(x(&cpu, A1), 0x1800);
}

#[test]
fn mtvec_reserved_mode_keeps_previous_mode() {
    //     csrw  mtvec, t0        ; t0 = HANDLER | 1 (Vectored)
    //     csrr  a0, mtvec
    //     csrw  mtvec, t1        ; t1 = HANDLER | 2 (reserved)
    //     csrr  a1, mtvec        ; BASE written, MODE stays Vectored
    //     csrw  mtvec, t2        ; t2 = (HANDLER + 0x40) | 3 (reserved)
    //     csrr  a2, mtvec
    //     csrw  mtvec, x0        ; Direct, BASE 0
    //     csrrs a3, mtvec, a5    ; a5 = HANDLER | 2: set-bits form
    //     csrr  a4, mtvec        ; MODE stays Direct
    let program = [
        csrw(MTVEC, T0),
        csrr(A0, MTVEC),
        csrw(MTVEC, T1),
        csrr(A1, MTVEC),
        csrw(MTVEC, T2),
        csrr(A2, MTVEC),
        csrw(MTVEC, 0),
        csrrs(A3, MTVEC, A5),
        csrr(A4, MTVEC),
        EBREAK,
    ];
    let mut cpu = hart(
        &program,
        &[],
        &[
            (T0, HANDLER | 1),
            (T1, HANDLER | 2),
            (T2, (HANDLER + 0x40) | 3),
            (A5, HANDLER | 2),
        ],
    );
    run(&mut cpu);
    assert_eq!(x(&cpu, A0), HANDLER | 1);
    assert_eq!(x(&cpu, A1), HANDLER | 1);
    assert_eq!(x(&cpu, A2), (HANDLER + 0x40) | 1);
    assert_eq!(x(&cpu, A3), 0, "old value");
    assert_eq!(x(&cpu, A4), HANDLER);
}

#[test]
fn ecall_with_reserved_mtvec_mode_traps_to_base() {
    //     csrw  mtvec, t0        ; t0 = HANDLER | 2 (reserved MODE)
    //     ecall                  ; traps to BASE = HANDLER
    //     ebreak
    // HANDLER:
    //     addi  a0, x0, 1
    //     (mepc += 4; mret)
    let program = [csrw(MTVEC, T0), ECALL, EBREAK];
    let handler = returning_handler(&[addi(A0, 0, 1)]);
    let mut cpu = hart(&program, &handler, &[(T0, HANDLER | 2)]);
    run(&mut cpu);
    assert_eq!(x(&cpu, A0), 1, "handler at BASE ran");
    assert_eq!(cpu.read_pc(), CODE + 8);
}

#[test]
fn mepc_write_clears_bit_zero() {
    //     csrw  mepc, t0         ; t0 = CODE + 13 (odd)
    //     csrr  a0, mepc
    //     csrrs a1, mepc, t1     ; t1 = 1: set bit 0
    //     csrr  a2, mepc
    let program = [
        csrw(MEPC, T0),
        csrr(A0, MEPC),
        csrrs(A1, MEPC, T1),
        csrr(A2, MEPC),
        EBREAK,
    ];
    let mut cpu = hart(&program, &[], &[(T0, CODE + 13), (T1, 1)]);
    run(&mut cpu);
    assert_eq!(x(&cpu, A0), CODE + 12);
    assert_eq!(x(&cpu, A1), CODE + 12);
    assert_eq!(x(&cpu, A2), CODE + 12);
}

#[test]
fn mret_to_odd_mepc_write_lands_on_even_address() {
    //     csrw  mepc, t0         ; t0 = CODE + 13
    //     mret                   ; pc := mepc = CODE + 12
    //     ebreak                 ; (skipped)
    //     addi  a0, x0, 1        ; CODE + 12
    //     ebreak
    let program = [csrw(MEPC, T0), MRET, EBREAK, addi(A0, 0, 1), EBREAK];
    let mut cpu = hart(&program, &[], &[(T0, CODE + 13)]);
    run(&mut cpu);
    assert_eq!(x(&cpu, A0), 1);
}

#[test]
fn raw_csrs_store_any_value() {
    let program = [
        csrw(MSCRATCH, T0),
        csrw(MCAUSE, T0),
        csrw(MTVAL, T0),
        csrr(A0, MSCRATCH),
        csrr(A1, MCAUSE),
        csrr(A2, MTVAL),
        EBREAK,
    ];
    let mut cpu = hart(&program, &[], &[(T0, 0xdead_beef_0123_4567)]);
    run(&mut cpu);
    for register in [A0, A1, A2] {
        assert_eq!(x(&cpu, register), 0xdead_beef_0123_4567);
    }
}

#[test]
fn mstatus_clear_and_immediate_forms() {
    // The `riscv` crate's critical section is `csrrci rd, mstatus, 8`.
    //     csrsi  mstatus, 8      ; MIE := 1
    //     csrs   mstatus, t1     ; t1 = 0x80: MPIE := 1
    //     csrrci a0, mstatus, 8  ; a0 = old value; MIE := 0
    //     csrr   a1, mstatus
    //     csrrwi a2, mstatus, 31 ; writes 0b11111: only MIE sticks
    //     csrr   a3, mstatus
    //     csrrc  a4, mstatus, t0 ; t0 = all ones: MPP stays M
    //     csrr   a5, mstatus
    let program = [
        csrrsi(0, MSTATUS, 8),
        csrrs(0, MSTATUS, T1),
        csrrci(A0, MSTATUS, 8),
        csrr(A1, MSTATUS),
        csrrwi(A2, MSTATUS, 31),
        csrr(A3, MSTATUS),
        csrrc(A4, MSTATUS, T0),
        csrr(A5, MSTATUS),
        EBREAK,
    ];
    let mut cpu = hart(&program, &[], &[(T0, u64::MAX), (T1, 0x80)]);
    run(&mut cpu);
    assert_eq!(x(&cpu, A0), 0x1888, "old value: MIE = MPIE = 1");
    assert_eq!(x(&cpu, A1), 0x1880, "MIE cleared, MPIE kept");
    assert_eq!(x(&cpu, A2), 0x1880, "old value");
    assert_eq!(x(&cpu, A3), 0x1808);
    assert_eq!(x(&cpu, A4), 0x1808, "old value");
    assert_eq!(x(&cpu, A5), 0x1800);
}

#[test]
fn csrrc_clears_bits_with_rd_equal_rs1_and_x0_forms() {
    //     csrw  mscratch, t0     ; t0 = 0xff
    //     csrrc t1, mscratch, t1 ; t1 = 0x0f: rd = rs1
    //     csrr  a0, mscratch
    //     csrrc a1, mscratch, x0 ; rs1 = x0: read only
    //     csrc  mscratch, t2     ; t2 = 0x30: rd = x0
    //     csrr  a2, mscratch
    let program = [
        csrw(MSCRATCH, T0),
        csrrc(T1, MSCRATCH, T1),
        csrr(A0, MSCRATCH),
        csrrc(A1, MSCRATCH, 0),
        csrrc(0, MSCRATCH, T2),
        csrr(A2, MSCRATCH),
        EBREAK,
    ];
    let mut cpu = hart(&program, &[], &[(T0, 0xff), (T1, 0x0f), (T2, 0x30)]);
    run(&mut cpu);
    assert_eq!(x(&cpu, T1), 0xff, "old value");
    assert_eq!(x(&cpu, A0), 0xf0);
    assert_eq!(x(&cpu, A1), 0xf0);
    assert_eq!(x(&cpu, A2), 0xc0);
}

#[test]
fn csr_immediates_are_zero_extended_and_zero_writes_only_for_csrrwi() {
    //     csrrwi x0, mscratch, 31  ; uimm is zero-extended: 31, not -1
    //     csrr   a0, mscratch
    //     csrrsi a1, mscratch, 0   ; uimm = 0: read only
    //     csrrci a2, mscratch, 0   ; uimm = 0: read only
    //     csrrsi x0, mscratch, 0   ; no effect
    //     csrr   a3, mscratch
    //     csrrci a4, mscratch, 5   ; 31 & !5
    //     csrrwi a5, mscratch, 0   ; CSRRWI writes even uimm = 0
    //     csrr   t2, mscratch
    let program = [
        csrrwi(0, MSCRATCH, 31),
        csrr(A0, MSCRATCH),
        csrrsi(A1, MSCRATCH, 0),
        csrrci(A2, MSCRATCH, 0),
        csrrsi(0, MSCRATCH, 0),
        csrr(A3, MSCRATCH),
        csrrci(A4, MSCRATCH, 5),
        csrrwi(A5, MSCRATCH, 0),
        csrr(T2, MSCRATCH),
        EBREAK,
    ];
    let mut cpu = hart(&program, &[], &[]);
    run(&mut cpu);
    assert_eq!(x(&cpu, A0), 31);
    assert_eq!(x(&cpu, A1), 31);
    assert_eq!(x(&cpu, A2), 31);
    assert_eq!(x(&cpu, A3), 31);
    assert_eq!(x(&cpu, A4), 31, "old value");
    assert_eq!(x(&cpu, A5), 26, "old value: 31 & !5");
    assert_eq!(x(&cpu, T2), 0);
}

#[test]
fn mtvec_and_mepc_immediate_and_clear_forms_are_legalized() {
    //     csrw   mtvec, t0       ; t0 = HANDLER | 1 (Vectored)
    //     csrrci a0, mtvec, 1    ; MODE := Direct
    //     csrrsi a1, mtvec, 2    ; writes MODE 2 (reserved): MODE stays Direct
    //     csrrwi a2, mtvec, 1    ; BASE 0, Vectored
    //     csrrwi a3, mtvec, 6    ; BASE 4, MODE 2 (reserved): MODE stays Vectored
    //     csrr   a4, mtvec
    //     csrrwi x0, mepc, 31    ; bit 0 cleared
    //     csrrci a5, mepc, 2     ; a5 = old value
    //     csrr   t2, mepc
    let program = [
        csrw(MTVEC, T0),
        csrrci(A0, MTVEC, 1),
        csrrsi(A1, MTVEC, 2),
        csrrwi(A2, MTVEC, 1),
        csrrwi(A3, MTVEC, 6),
        csrr(A4, MTVEC),
        csrrwi(0, MEPC, 31),
        csrrci(A5, MEPC, 2),
        csrr(T2, MEPC),
        EBREAK,
    ];
    let mut cpu = hart(&program, &[], &[(T0, HANDLER | 1)]);
    run(&mut cpu);
    assert_eq!(x(&cpu, A0), HANDLER | 1, "old value");
    assert_eq!(x(&cpu, A1), HANDLER, "old value");
    assert_eq!(x(&cpu, A2), HANDLER, "old value");
    assert_eq!(x(&cpu, A3), 1, "old value");
    assert_eq!(x(&cpu, A4), 4 | 1);
    assert_eq!(x(&cpu, A5), 30, "old value");
    assert_eq!(x(&cpu, T2), 28);
}

/// Runs `program` with `a0 = DATA`, `a1 = new value`, and the doubleword at
/// `DATA` holding `OLD`; returns the hart.
fn lr_sc(program: &[u32], handler: &[u32]) -> Cpu {
    let mut cpu = hart(
        program,
        handler,
        &[(A0, DATA), (A1, NEW), (T0, HANDLER), (A2, DATA + 4)],
    );
    cpu.get_mut_mmu()
        .store_doubleword(DATA, OLD)
        .expect("data store");
    run(&mut cpu);
    cpu
}

const OLD: u64 = 0x1111_2222_3333_4444;
const NEW: u64 = 0x5555_6666_7777_8888;

fn memory(cpu: &mut Cpu) -> u64 {
    cpu.get_mut_mmu().load_doubleword(DATA).expect("load").0
}

#[test]
fn lr_sc_word_succeeds() {
    let mut cpu = lr_sc(&[lr_w(A3, A0), sc_w(A4, A0, A1), EBREAK], &[]);
    assert_eq!(x(&cpu, A3), 0x3333_4444);
    assert_eq!(x(&cpu, A4), 0, "success");
    assert_eq!(memory(&mut cpu), 0x1111_2222_7777_8888);
}

#[test]
fn sc_without_reservation_fails_and_leaves_memory() {
    let mut cpu = lr_sc(&[sc_w(A4, A0, A1), sc_d(A5, A0, A1), EBREAK], &[]);
    assert_eq!(x(&cpu, A4), 1);
    assert_eq!(x(&cpu, A5), 1);
    assert_eq!(memory(&mut cpu), OLD);
}

#[test]
fn second_sc_fails_because_sc_invalidates_the_reservation() {
    let mut cpu = lr_sc(
        &[lr_d(A3, A0), sc_d(A4, A0, A1), sc_d(A5, A0, 0), EBREAK],
        &[],
    );
    assert_eq!(x(&cpu, A4), 0);
    assert_eq!(x(&cpu, A5), 1);
    assert_eq!(memory(&mut cpu), NEW);
}

#[test]
fn doubleword_reservation_covers_word_sc_at_same_address() {
    let mut cpu = lr_sc(&[lr_d(A3, A0), sc_w(A4, A0, A1), EBREAK], &[]);
    assert_eq!(x(&cpu, A4), 0);
    assert_eq!(memory(&mut cpu), 0x1111_2222_7777_8888);
}

#[test]
fn sc_elsewhere_in_reservation_set_fails_legally() {
    // LR.D reserves DATA..DATA+8; SC.W at DATA+4 is inside the reservation
    // set but not at the LR address. The specification permits failure here
    // (it is not a constrained loop), and Jolt fails it.
    let mut cpu = lr_sc(&[lr_d(A3, A0), sc_w(A4, A2, A1), EBREAK], &[]);
    assert_eq!(x(&cpu, A4), 1);
    assert_eq!(memory(&mut cpu), OLD);
}

#[test]
fn word_reservation_does_not_cover_doubleword_sc() {
    let mut cpu = lr_sc(&[lr_w(A3, A0), sc_d(A4, A0, A1), EBREAK], &[]);
    assert_eq!(x(&cpu, A4), 1);
    assert_eq!(memory(&mut cpu), OLD);
}

#[test]
fn sc_pairs_with_most_recent_lr() {
    // LR.D a0; LR.W a2; SC.D a0: the most recent LR is the word at a2.
    let mut cpu = lr_sc(&[lr_d(A3, A0), lr_w(A3, A2), sc_d(A4, A0, A1), EBREAK], &[]);
    assert_eq!(x(&cpu, A4), 1);
    assert_eq!(memory(&mut cpu), OLD);
}

#[test]
fn reservation_survives_trap_and_mret() {
    // Neither trap entry nor xRET is required to clear the reservation.
    let program = [
        csrw(MTVEC, T0),
        lr_w(A3, A0),
        ECALL,
        sc_w(A4, A0, A1),
        EBREAK,
    ];
    let handler = returning_handler(&[]);
    let mut cpu = lr_sc(&program, &handler);
    assert_eq!(x(&cpu, A4), 0);
    assert_eq!(memory(&mut cpu), 0x1111_2222_7777_8888);
}

fn reference_csr_write(csr: u32, old: u64, value: u64) -> u64 {
    match csr {
        MSTATUS => 0x1800 | (value & 0x88),
        MTVEC if value & 2 != 0 => (value & !3) | (old & 3),
        MEPC => value & !1,
        _ => value,
    }
}

#[test]
fn system_expansions_refine_architectural_formulas() {
    use crate::instruction::Instruction;

    let mut cpu = Cpu::new(Box::new(DummyTerminal::default()));
    let operands = [
        0,
        1,
        2,
        3,
        4,
        5,
        8,
        0x80,
        0x88,
        0x1800,
        u64::MAX,
        u64::MAX - 1,
        u64::MAX - 2,
        u64::MAX - 3,
        1 << 63,
        (1 << 63) | 3,
        0x1234_5678_9abc_def0,
    ];
    let states: &[(u32, usize, &[u64])] = &[
        (MSTATUS, 39, &[0x1800, 0x1808, 0x1880, 0x1888]),
        (
            MTVEC,
            34,
            &[0, 1, 4, 5, HANDLER, HANDLER | 1, u64::MAX - 3, u64::MAX - 2],
        ),
        (
            MEPC,
            36,
            &[0, 2, 4, 6, CODE, CODE + 2, u64::MAX - 1, u64::MAX - 3],
        ),
        (MSCRATCH, 35, &operands),
        (MCAUSE, 37, &operands),
        (MTVAL, 38, &operands),
    ];
    for &(csr, virtual_reg, old_values) in states {
        for funct3 in [1, 2, 3, 5, 6, 7] {
            let cases: Vec<_> = if funct3 < 5 {
                [(0, 0), (A0, 0), (0, T0), (T0, T0), (A0, T0)]
                    .into_iter()
                    .flat_map(|(rd, rs1)| operands.map(|value| (rd, rs1, value)))
                    .collect()
            } else {
                [0, T0]
                    .into_iter()
                    .flat_map(|rd| (0..32).map(move |uimm| (rd, uimm, 0)))
                    .collect()
            };
            for (rd, source, register_value) in cases {
                let word = i_type(csr, source, funct3, rd, 0x73);
                let decoded = Instruction::decode(word, CODE, false).expect("system decode");
                let expanded = decoded.inline_sequence(&cpu.vr_allocator);
                for &old in old_values {
                    cpu.x.fill(0x1234_5678_9abc_def0);
                    cpu.x[0] = 0;
                    for (reg, value) in [
                        (34, HANDLER | 1),
                        (35, u64::MAX),
                        (36, DATA + 2),
                        (37, 11),
                        (38, 1 << 63),
                        (39, 0x88),
                    ] {
                        cpu.x[reg] = value as i64;
                    }
                    cpu.x[virtual_reg] = (if csr == MSTATUS { old & 0x88 } else { old }) as i64;
                    if funct3 < 5 && source != 0 {
                        cpu.x[source as usize] = register_value as i64;
                    }
                    let mut expected = cpu.x;
                    let operand = if funct3 >= 5 {
                        u64::from(source)
                    } else if source == 0 {
                        0
                    } else {
                        register_value
                    };
                    // These are the architectural Zicsr formulas, including
                    // write suppression by operand field rather than value.
                    let written = match funct3 {
                        1 | 5 => Some(operand),
                        2 | 6 if source != 0 => Some(old | operand),
                        3 | 7 if source != 0 => Some(old & !operand),
                        _ => None,
                    };
                    let after = written.map_or(old, |value| reference_csr_write(csr, old, value));
                    if rd != 0 {
                        expected[rd as usize] = old as i64;
                    }
                    expected[virtual_reg] =
                        (if csr == MSTATUS { after & 0x88 } else { after }) as i64;
                    cpu.update_pc(CODE + 4);
                    for row in &expanded {
                        row.trace(&mut cpu, None);
                    }
                    assert_eq!(
                        &cpu.x[..40],
                        &expected[..40],
                        "word={word:#x} old={old:#x} operand={operand:#x}"
                    );
                    let architectural =
                        (cpu.x[virtual_reg] as u64) | if csr == MSTATUS { 0x1800 } else { 0 };
                    assert_eq!(architectural, after, "CSR abstraction for {word:#x}");
                    assert_eq!(cpu.x[39] & !0x88, 0);
                    assert_eq!(cpu.x[34] & 2, 0);
                    assert_eq!(cpu.x[36] & 1, 0);
                    assert_eq!(cpu.read_pc(), CODE + 4);
                }
            }
        }
    }

    for pc in [CODE, CODE + 2] {
        for word in [ECALL, MRET] {
            let decoded = Instruction::decode(word, pc, false).expect("trap decode");
            let expanded = decoded.inline_sequence(&cpu.vr_allocator);
            for status in [0, 8, 0x80, 0x88] {
                for mtvec in [HANDLER, HANDLER | 1, HANDLER + 4, (HANDLER + 4) | 1] {
                    for mepc in [CODE + 0x40, CODE + 0x42] {
                        cpu.x.fill(0x1234_5678_9abc_def0);
                        cpu.x[0] = 0;
                        cpu.x[34] = mtvec as i64;
                        cpu.x[36] = mepc as i64;
                        cpu.x[39] = status;
                        let mut expected = cpu.x;
                        let next_pc = if word == ECALL {
                            expected[36] = pc as i64;
                            expected[37] = 11;
                            expected[38] = 0;
                            expected[39] = (status & 8) << 4;
                            mtvec & !3
                        } else {
                            expected[39] = 0x80 | ((status >> 4) & 8);
                            mepc
                        };
                        cpu.update_pc(pc + 4);
                        for row in &expanded {
                            row.trace(&mut cpu, None);
                        }
                        assert_eq!(
                            &cpu.x[..40],
                            &expected[..40],
                            "word={word:#x} mstatus={status:#x}"
                        );
                        assert_eq!(cpu.x[39] as u64 | 0x1800, expected[39] as u64 | 0x1800);
                        assert_eq!(cpu.x[39] & !0x88, 0);
                        assert_eq!(cpu.x[34] & 2, 0);
                        assert_eq!(cpu.x[36] & 1, 0);
                        assert_eq!(cpu.read_pc(), next_pc);
                    }
                }
            }
        }
    }
}
