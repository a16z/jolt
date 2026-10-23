//! Where the native backend ends a guest at a JALR that jumps to its own
//! address. One that overwrites its own base with the link leaves on its
//! second run, as RISC-V executes it; one whose link leads back to itself is
//! a fixed point and ends the guest after one run. The interpreter's
//! counterparts are in `tracer`'s `Cpu` tests.

#![cfg(all(target_arch = "x86_64", target_os = "linux"))]
#![expect(clippy::unwrap_used)]

use common::constants::{RAM_START_ADDRESS, REGISTER_COUNT};
use jolt_program::execution::JoltProgram;
use jolt_riscv::{JoltInstructionKind, JoltInstructionRow, NormalizedOperands};
use jolt_tracer_x86::harness::{run_program, terminal_row, TEST_ADDR};

struct SelfJump {
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
    /// Runs the jump at `TEST_ADDR`, followed by a terminal self-branch.
    fn end(&self) -> End {
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
        let len = if self.is_compressed { 2 } else { 4 };
        let program = JoltProgram::from_parts(
            Vec::new(),
            vec![jump, terminal_row(TEST_ADDR + len)],
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
}

#[test]
fn jalr_overwriting_its_base_runs_again() {
    // jalr x5, 0(x5): the second run targets the link, TEST_ADDR + 4.
    let jalr = SelfJump {
        is_compressed: false,
        register: 5,
        imm: 0,
        base: TEST_ADDR,
    };
    assert_eq!(
        jalr.end(),
        End {
            rows: 3,
            pc: TEST_ADDR + 4,
            register: TEST_ADDR + 4,
        }
    );

    // c.jalr ra: the second run targets the link, TEST_ADDR + 2.
    let c_jalr = SelfJump {
        is_compressed: true,
        register: 1,
        imm: 0,
        base: TEST_ADDR,
    };
    assert_eq!(
        c_jalr.end(),
        End {
            rows: 3,
            pc: TEST_ADDR + 2,
            register: TEST_ADDR + 2,
        }
    );
}

#[test]
fn jalr_at_its_fixed_point_stops_after_one_run() {
    // jalr x5, -4(x5) with x5 = TEST_ADDR + 4: the link it writes leads back
    // to TEST_ADDR.
    let jalr = SelfJump {
        is_compressed: false,
        register: 5,
        imm: -4,
        base: TEST_ADDR + 4,
    };
    assert_eq!(
        jalr.end(),
        End {
            rows: 1,
            pc: TEST_ADDR,
            register: TEST_ADDR + 4,
        }
    );
}
