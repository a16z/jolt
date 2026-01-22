use common::{constants, jolt_device::MemoryConfig};

/// Groups the constants used for a specific instruction set / decomposition strategy / memory
/// layout. Jolt currently just has one of these, but we abstract over them here for future
/// compatibility.
pub trait JoltParameterSet {
    /// The architecture size.
    const XLEN: usize;
    /// The memory config to use
    const MEMORY_CONFIG: MemoryConfig;
}

/// The parameters used by Jolt for 32-bit risc-v
#[derive(Clone)]
pub struct RV64IParameterSet;

impl JoltParameterSet for RV64IParameterSet {
    const XLEN: usize = 64;
    const MEMORY_CONFIG: MemoryConfig = MemoryConfig {
        max_input_size: constants::DEFAULT_MAX_INPUT_SIZE,
        max_output_size: constants::DEFAULT_MAX_OUTPUT_SIZE,
        stack_size: constants::DEFAULT_STACK_SIZE,
        memory_size: constants::DEFAULT_MEMORY_SIZE,
        program_size: None,
        max_trusted_advice_size: constants::DEFAULT_MAX_TRUSTED_ADVICE_SIZE,
        max_untrusted_advice_size: constants::DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE,
    };
}
