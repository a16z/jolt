//! Abstract instruction and cycle views used by the Jolt proof system.
//!
//! `JoltInstruction` exposes static encoding-time data (address, immediate,
//! register operand indices). `JoltCycle` extends it with dynamic register
//! state. Together they let `LookupQuery` impls in `jolt-lookup-tables`
//! operate on any concrete cycle representation without depending on tracer's
//! types directly.

/// Static instruction view: encoding-time data.
pub trait JoltInstruction {
    /// True if this cycle is a no-op (padding).
    fn is_noop(&self) -> bool;

    /// Program-counter address where this instruction lives.
    fn address(&self) -> u64;

    /// Sign-extended immediate, or `0` if the instruction has none.
    fn imm(&self) -> i128;

    /// rs1 register index, or `None` if unused.
    fn rs1(&self) -> Option<u8>;

    /// rs2 register index, or `None` if unused.
    fn rs2(&self) -> Option<u8>;

    /// rd register index, or `None` if unused.
    fn rd(&self) -> Option<u8>;

    /// Remaining steps in a virtual instruction sequence, or `None` if
    /// this is a real (non-virtual) instruction.
    fn virtual_sequence_remaining(&self) -> Option<u16>;

    /// True if this is the first instruction in a virtual sequence.
    fn is_first_in_sequence(&self) -> bool;

    /// True if this is a virtual (expanded) instruction.
    fn is_virtual(&self) -> bool;
}

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
