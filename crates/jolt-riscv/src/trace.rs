//! Tracer-free runtime views of executed Jolt instructions.
//!
//! [`JoltCycle`] pairs the static [`JoltInstructionRowData`](crate::JoltInstructionRowData)
//! vocabulary with register and RAM values captured during execution, so lookup
//! table code can operate on cycle data without depending on tracer's concrete
//! cycle types.

use crate::JoltInstructionRowData;

/// Dynamic cycle view: a populated instruction plus runtime register state.
pub trait JoltCycle {
    type Instruction: JoltInstructionRowData;

    /// The instruction executed during this cycle.
    fn instruction(&self) -> Self::Instruction;

    /// Value held in rs1 at the start of the cycle, or `None` if unused.
    fn rs1_val(&self) -> Option<u64>;

    /// Value held in rs2 at the start of the cycle, or `None` if unused.
    fn rs2_val(&self) -> Option<u64>;

    /// Value held in rd before and after the cycle executes, or `None` if unused.
    fn rd_vals(&self) -> Option<(u64, u64)>;

    /// RAM access address, or `None` if no RAM access this cycle.
    fn ram_access_address(&self) -> Option<u64>;

    /// RAM read value (pre-access value). `None` if no RAM access.
    fn ram_read_value(&self) -> Option<u64>;

    /// RAM write value (post-access value). `None` if no RAM access.
    fn ram_write_value(&self) -> Option<u64>;

    /// Generate a random cycle. Useful for fuzz testing.
    ///
    /// `where Self: Sized` keeps the trait dyn-compatible.
    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self;
}
