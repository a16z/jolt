//! Abstract cycle interface for the proving pipeline.
//!
//! [`CycleRow`] is the boundary between the tracer (which produces concrete
//! `Cycle` values) and the proving system (which consumes per-cycle data to
//! build witnesses). All ISA-specific logic (instruction dispatch, flag
//! computation, operand routing) is pushed into the `CycleRow` implementation,
//! so the prover sees only scalars and boolean arrays.

use jolt_instructions::flags::{NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS};

/// Abstract interface for one execution cycle of a RISC-V trace.
///
/// jolt-zkvm's witness layer is generic over `CycleRow`. The concrete
/// implementation for `tracer::Cycle` lives in this crate (`jolt-host`).
pub trait CycleRow: Copy {
    /// A no-op (padding) cycle.
    fn noop() -> Self;

    /// True if this cycle is a no-op (padding).
    fn is_noop(&self) -> bool;

    /// The unexpanded (pre-virtual-expansion) program counter.
    fn unexpanded_pc(&self) -> u64;

    /// Remaining steps in a virtual instruction sequence, or `None` if
    /// this is a real (non-virtual) instruction.
    fn virtual_sequence_remaining(&self) -> Option<u16>;

    /// True if this is the first instruction in a virtual sequence.
    fn is_first_in_sequence(&self) -> bool;

    /// True if this is a virtual (expanded) instruction.
    fn is_virtual(&self) -> bool;

    /// RS1 register read: `(register_index, value)`, or `None` if unused.
    fn rs1_read(&self) -> Option<(u8, u64)>;

    /// RS2 register read: `(register_index, value)`, or `None` if unused.
    fn rs2_read(&self) -> Option<(u8, u64)>;

    /// RD register write: `(register_index, pre_value, post_value)`, or `None`.
    fn rd_write(&self) -> Option<(u8, u64, u64)>;

    /// The static `rd` operand from the instruction encoding.
    fn rd_operand(&self) -> Option<u8>;

    /// RAM access address, or `None` if no RAM access this cycle.
    fn ram_access_address(&self) -> Option<u64>;

    /// RAM read value (pre-access value). `None` if no RAM access.
    fn ram_read_value(&self) -> Option<u64>;

    /// RAM write value (post-access value). `None` if no RAM access.
    fn ram_write_value(&self) -> Option<u64>;

    /// The immediate operand, sign-extended.
    fn imm(&self) -> i128;

    /// R1CS circuit flags (14 booleans, indexed by `CircuitFlags`).
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS];

    /// Non-R1CS instruction flags (7 booleans, indexed by `InstructionFlags`).
    fn instruction_flags(&self) -> [bool; NUM_INSTRUCTION_FLAGS];

    /// Combined lookup index for RA polynomial construction (128-bit).
    fn lookup_index(&self) -> u128;
}
