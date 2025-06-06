use common::{constants, rv_trace::MemoryConfig};
use jolt_core::jolt::vm::rv32i_vm;

/// Groups the constants used for a specific instruction set / decomposition strategy / memory
/// layout. Jolt currently just has one of these, but we abstract over them here for future
/// compatibility.
pub trait JoltParameterSet {
    /// The number of chunks used when decomposing instructions
    const C: usize;
    /// Size of a materialized subtable
    const M: usize;
    /// The number of bits in a subtable entry; this should always be `log2(M)`
    const LOG_M: usize = Self::M.ilog2() as usize;
    /// The architecture size; although this isn't exposed in the Jolt codebase, it can be inferred
    /// from `C` and `LOG_M`, since those constants need to be able to decompose a pair of
    /// `WORD_SIZE` registers
    const WORD_SIZE: usize = (Self::LOG_M * Self::C) / 2;
    /// The memory config to use
    const MEMORY_CONFIG: MemoryConfig;
}

/// The parameters used by Jolt for 32-bit risc-v
#[derive(Clone)]
pub struct RV32IParameterSet;

impl JoltParameterSet for RV32IParameterSet {
    const C: usize = rv32i_vm::C;
    const M: usize = rv32i_vm::M;
    const MEMORY_CONFIG: MemoryConfig = MemoryConfig {
        max_input_size: constants::DEFAULT_MAX_INPUT_SIZE,
        max_output_size: constants::DEFAULT_MAX_OUTPUT_SIZE,
        stack_size: constants::DEFAULT_STACK_SIZE,
        memory_size: constants::DEFAULT_MEMORY_SIZE,
    };
}
