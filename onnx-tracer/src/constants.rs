/// Offset constant for calculating the [ONNXInstr] address.
/// The zkVM prepends a no-op instruction to the program code,
/// so all instruction addresses must account for this offset.
pub const BYTECODE_PREPEND_NOOP: usize = 1;

/// The maximum number of elements allowed in a tensor within the constraint system.
/// This constant is used to determine the upper bound on tensor reads and writes
/// performed by the constraint system.
pub const MAX_TENSOR_SIZE: usize = 4;

/// Similar to register count, but for tensors.
/// For example the ONNX memory model can be viewed as registers that store tensors instead of scalars.
///
/// # NOTE: This value is purely used for testing purposes, for production the ONNX memory model requires dynamic amount of tensor slots.
///         However for now we simplify the zkVM to use a fixed number of tensor slots.
///         i.e. : This is a simplification and may not capture all aspects of the ONNX memory model.
pub const TEST_TENSOR_REGISTER_COUNT: u64 = 32;
pub const VIRTUAL_TENSOR_REGISTER_COUNT: u64 = 32; //  see Section 6.1 of Jolt paper
pub const TENSOR_REGISTER_COUNT: u64 = TEST_TENSOR_REGISTER_COUNT + VIRTUAL_TENSOR_REGISTER_COUNT;

/// Helper function used in virtual instructions to calculate the index of a virtual tensor register.
/// This function is used to map a virtual tensor index to its corresponding index in the execution trace.
/// The virtual tensor registers are indexed starting from `TEST_TENSOR_REGISTER_COUNT`.
/// This is used to ensure that virtual tensor registers do not conflict with the actual tensor registers.
/// The virtual tensor registers are used to store intermediate results of virtual instructions.
pub const fn virtual_tensor_index(index: usize) -> usize {
    index + TEST_TENSOR_REGISTER_COUNT as usize
}

/// 3 registers (td, ts1, ts2)
pub const MEMORY_OPS_PER_INSTRUCTION: usize = 3;

/// Used to calculate the zkVM address's from the execution trace.
/// Since the 0 address is reserved for the zero register, we prepend a 1 to the address's in the execution trace.
pub const ZERO_ADDR_PREPEND: usize = 1;
