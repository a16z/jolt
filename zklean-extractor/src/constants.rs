use common::{constants, jolt_device::MemoryConfig};

/// Groups the constants used for a specific instruction set / decomposition strategy / memory
/// layout. Jolt currently just has one of these, but we abstract over them here for future
/// compatibility.
pub trait JoltParameterSet {
    /// The architecture size.
    const WORD_SIZE: usize;
    /// The memory config to use
    const MEMORY_CONFIG: MemoryConfig;
}

/// The parameters used by Jolt for 32-bit risc-v
#[derive(Clone)]
pub struct RV32IParameterSet;

impl JoltParameterSet for RV32IParameterSet {
    const WORD_SIZE: usize = 32;
    const MEMORY_CONFIG: MemoryConfig = MemoryConfig {
        max_input_size: constants::DEFAULT_MAX_INPUT_SIZE,
        max_output_size: constants::DEFAULT_MAX_OUTPUT_SIZE,
        stack_size: constants::DEFAULT_STACK_SIZE,
        memory_size: constants::DEFAULT_MEMORY_SIZE,
        program_size: None,
    };
}
