//! `JoltCycle`: dynamic, runtime view of a single executed instruction.
//!
//! Pairs the static [`JoltInstruction`] (defined in `jolt-riscv`) with
//! register and RAM state captured during tracing, so lookup-table code can
//! operate on any concrete cycle representation without depending on tracer's
//! types directly.

use jolt_riscv::JoltInstruction;

/// Dynamic cycle view: a populated instruction plus runtime register state.
pub trait JoltCycle {
    type Instruction: JoltInstruction;

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
