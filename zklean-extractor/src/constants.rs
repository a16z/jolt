/// Groups the constants used for a specific instruction set / decomposition strategy / memory
/// layout. Jolt currently just has one of these, but we abstract over them here for future
/// compatibility.
pub trait JoltParameterSet {
    /// The architecture size.
    const XLEN: usize;
}

/// The parameters used by Jolt for 32-bit risc-v
#[derive(Clone)]
pub struct RV64IParameterSet;

impl JoltParameterSet for RV64IParameterSet {
    const XLEN: usize = 64;
}
