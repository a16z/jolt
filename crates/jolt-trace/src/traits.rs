//! Abstract instruction and cycle views used by lookup queries.
//!
//! `JoltInstruction` exposes static encoding-time data (address, immediate,
//! register operand indices). `JoltCycle` extends it with dynamic register
//! state. Together they let `LookupQuery` impls in `jolt-lookup-tables`
//! operate on any concrete cycle representation without depending on tracer's
//! types directly.

/// Static instruction view: encoding-time data.
pub trait JoltInstruction {
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
}

/// Dynamic cycle view: a populated instruction plus runtime register state.
pub trait JoltCycle: JoltInstruction {
    /// Value held in rs1 at the start of the cycle, or `None` if unused.
    fn rs1_val(&self) -> Option<u64>;
    /// Value held in rs2 at the start of the cycle, or `None` if unused.
    fn rs2_val(&self) -> Option<u64>;
    /// Value held in rd before and after the cycle executes, or `None` if unused.
    fn rd_vals(&self) -> Option<(u64, u64)>;
}
