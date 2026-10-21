//! Where the interpreter and the native backend end a guest at a JALR that
//! jumps to its own address. One that overwrites its own base with the link
//! leaves on its second run, as RISC-V executes it; one whose link leads back
//! to itself is a fixed point and ends the guest after one run.

#![cfg(all(target_arch = "x86_64", target_os = "linux"))]
#![expect(clippy::unwrap_used)]

use common::constants::{RAM_START_ADDRESS, REGISTER_COUNT};
use common::jolt_device::JoltDevice;
use jolt_program::execution::JoltProgram;
use jolt_riscv::{JoltInstructionKind, JoltInstructionRow, NormalizedOperands};
use jolt_tracer_x86::harness::{self, run_program, terminal_row, MEM_CAPACITY, TEST_ADDR};
use tracer::emulator::cpu::Cpu;
use tracer::emulator::terminal::DummyTerminal;

/// `beq x0, x0, 0`, placed after the jump on both sides.
const BEQ_SELF: u32 = 0x0000_0063;

struct SelfJump {
    /// The jump's encoding, placed at `TEST_ADDR`.
    encoding: u32,
    is_compressed: bool,
    /// Both `rd` and `rs1`.
    register: u8,
    imm: i128,
    /// Initial base register value; makes the first run target `TEST_ADDR`.
    base: u64,
}

/// Where a guest ended: rows traced, final PC, and the jump's register.
#[derive(Debug, PartialEq, Eq)]
struct End {
    rows: u64,
    pc: u64,
    register: u64,
}

impl SelfJump {
    fn len(&self) -> u64 {
        if self.is_compressed {
            2
        } else {
            4
        }
    }

    fn interpreter_end(&self) -> End {
        let mut cpu = Cpu::new(Box::new(DummyTerminal {}));
        cpu.get_mut_mmu().jolt_device = Some(JoltDevice::new(&harness::memory_config()));
        cpu.get_mut_mmu().init_memory(MEM_CAPACITY);
        let code = (u64::from(self.encoding) & ((1 << (8 * self.len())) - 1))
            | (u64::from(BEQ_SELF) << (8 * self.len()));
        for halfword in 0..4 {
            let _ = cpu
                .mmu
                .store_halfword(TEST_ADDR + 2 * halfword, (code >> (16 * halfword)) as u16)
                .unwrap();
        }
        cpu.x[self.register as usize] = self.base as i64;
        cpu.update_pc(TEST_ADDR);

        let mut rows = Vec::new();
        for _ in 0..4 {
            if cpu.has_stalled() {
                break;
            }
            cpu.tick(Some(&mut rows));
        }
        assert!(cpu.has_stalled(), "interpreter did not stall");
        End {
            rows: rows.len() as u64,
            pc: cpu.read_pc(),
            register: cpu.x[self.register as usize] as u64,
        }
    }

    fn native_end(&self) -> End {
        let jump = JoltInstructionRow {
            instruction_kind: JoltInstructionKind::JALR,
            address: TEST_ADDR as usize,
            operands: NormalizedOperands {
                rs1: Some(self.register),
                rs2: None,
                rd: Some(self.register),
                imm: self.imm,
            },
            virtual_sequence_remaining: None,
            is_first_in_sequence: true,
            is_compressed: self.is_compressed,
        };
        let program = JoltProgram::from_parts(
            Vec::new(),
            vec![jump, terminal_row(TEST_ADDR + self.len())],
            Vec::new(),
            RAM_START_ADDRESS + 1024,
            TEST_ADDR,
        );
        let mut regs = [0u64; REGISTER_COUNT as usize];
        regs[self.register as usize] = self.base;
        let outcome = run_program(&program, &regs, &[], &[]).unwrap();
        assert_eq!(outcome.exit, 1, "native run did not terminate");
        End {
            rows: outcome.trace_len,
            pc: outcome.pc,
            register: outcome.regs[self.register as usize],
        }
    }

    fn assert_ends_at(&self, expected: End) {
        assert_eq!(self.interpreter_end(), expected, "interpreter");
        assert_eq!(self.native_end(), expected, "native");
    }
}

#[test]
fn jalr_overwriting_its_base_runs_again() {
    // jalr x5, 0(x5): the second run targets the link, TEST_ADDR + 4.
    SelfJump {
        encoding: 0x0002_82e7,
        is_compressed: false,
        register: 5,
        imm: 0,
        base: TEST_ADDR,
    }
    .assert_ends_at(End {
        rows: 3,
        pc: TEST_ADDR + 4,
        register: TEST_ADDR + 4,
    });

    // c.jalr ra: the second run targets the link, TEST_ADDR + 2.
    SelfJump {
        encoding: 0x9082,
        is_compressed: true,
        register: 1,
        imm: 0,
        base: TEST_ADDR,
    }
    .assert_ends_at(End {
        rows: 3,
        pc: TEST_ADDR + 2,
        register: TEST_ADDR + 2,
    });
}

#[test]
fn jalr_at_its_fixed_point_stops_after_one_run() {
    // jalr x5, -4(x5) with x5 = TEST_ADDR + 4: the link it writes leads back
    // to TEST_ADDR.
    SelfJump {
        encoding: 0xffc2_82e7,
        is_compressed: false,
        register: 5,
        imm: -4,
        base: TEST_ADDR + 4,
    }
    .assert_ends_at(End {
        rows: 1,
        pc: TEST_ADDR,
        register: TEST_ADDR + 4,
    });
}
