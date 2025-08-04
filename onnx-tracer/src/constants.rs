/// Offset constant for calculating the [ONNXInstr] address.
/// The zkVM prepends a no-op instruction to the program code,
/// so all instruction addresses must account for this offset.
pub const BYTECODE_PREPEND_NOOP: usize = 1;

/// The maximum number of elements allowed in a tensor within the constraint system.
/// This constant is used to determine the upper bound on tensor reads and writes
/// performed by the constraint system.
pub const MAX_TENSOR_SIZE: usize = 128;

// TODO(Forpee): Determine actual virtual tensor count
const VIRTUAL_TENSOR_COUNT: usize = 32; //  see Section 6.1 of Jolt paper

pub const fn virtual_tensor_index(index: usize) -> usize {
    index + VIRTUAL_TENSOR_COUNT
}

/// 3 registers (td, ts1, ts2)
pub const MEMORY_OPS_PER_INSTRUCTION: usize = 3;
