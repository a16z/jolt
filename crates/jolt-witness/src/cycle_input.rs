//! Per-cycle input for polynomial generation.
//!
//! [`CycleInput`] is a flat, protocol-level representation of one execution
//! cycle. The caller extracts these values from whatever trace backend is
//! in use (e.g. `tracer::Cycle`) before pushing them into [`Polynomials`].
//!
//! [`Polynomials`]: crate::Polynomials

/// Per-cycle input data for polynomial buffer generation.
///
/// Each field maps to one or more committed polynomial buffers.
/// [`Polynomials::push`](crate::Polynomials::push) decomposes these values
/// into dense and one-hot evaluation entries.
#[derive(Clone, Copy, Debug, Default)]
pub struct CycleInput {
    /// Register write increment: `post_value - pre_value`.
    /// Zero for cycles with no register write. Maps to `RdInc`.
    pub rd_inc: i128,

    /// RAM write increment: `post_value - pre_value`.
    /// Zero for cycles with no RAM write. Maps to `RamInc`.
    pub ram_inc: i128,

    /// Instruction lookup index (128-bit interleaved operand encoding).
    /// Decomposed into `instruction_d` one-hot chunks for `InstructionRa`.
    pub lookup_index: u128,

    /// Virtual bytecode PC index.
    /// Decomposed into `bytecode_d` one-hot chunks for `BytecodeRa`.
    pub pc_index: u32,

    /// Remapped RAM address, or `None` for cycles with no RAM access.
    /// Decomposed into `ram_d` one-hot chunks for `RamRa`.
    pub ram_address: Option<u64>,
}

impl CycleInput {
    /// A padding cycle: zero increments, zero indices, no RAM access.
    pub const PADDING: Self = Self {
        rd_inc: 0,
        ram_inc: 0,
        lookup_index: 0,
        pc_index: 0,
        ram_address: None,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_is_default() {
        let pad = CycleInput::PADDING;
        let def = CycleInput::default();
        assert_eq!(pad.rd_inc, def.rd_inc);
        assert_eq!(pad.ram_inc, def.ram_inc);
        assert_eq!(pad.lookup_index, def.lookup_index);
        assert_eq!(pad.pc_index, def.pc_index);
        assert_eq!(pad.ram_address, def.ram_address);
    }
}
