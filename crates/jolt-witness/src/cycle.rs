//! Per-cycle witness data.
//!
//! [`CycleData`] is a pre-extracted, protocol-level representation of one
//! execution cycle. The caller (jolt-zkvm) converts backend-specific trace
//! rows (e.g. `tracer::Cycle`) into this flat struct before passing to the
//! [`WitnessBuilder`](crate::WitnessBuilder).
//!
//! This keeps jolt-witness free from tracer and instruction dependencies.

/// Pre-extracted per-cycle data for witness polynomial generation.
///
/// Each field corresponds to one or more committed witness polynomials.
/// The caller is responsible for correctly extracting these values from
/// the execution trace — the builder trusts them and applies only the
/// one-hot chunking decomposition.
///
/// # Padding
///
/// For padding (no-op) cycles, use [`CycleData::PADDING`]. The caller
/// is responsible for padding the trace to the required length (typically
/// next power of two) before passing to the builder.
#[derive(Clone, Copy, Debug, Default)]
pub struct CycleData {
    /// Register write increment: `post_value - pre_value`.
    ///
    /// Zero for cycles with no register write. Maps to the `RdInc` polynomial.
    pub rd_inc: i128,

    /// RAM write increment: `post_value - pre_value`.
    ///
    /// Zero for cycles with no RAM write or read-only access. Maps to `RamInc`.
    pub ram_inc: i128,

    /// Instruction lookup index (128-bit interleaved operand encoding).
    ///
    /// Pre-computed by the caller via `LookupQuery::to_lookup_index()`.
    /// Decomposed into `instruction_d` one-hot chunks for `InstructionRa` polynomials.
    pub lookup_index: u128,

    /// Virtual bytecode PC index.
    ///
    /// Pre-computed by the caller via `BytecodePreprocessing::get_pc()`.
    /// Decomposed into `bytecode_d` one-hot chunks for `BytecodeRa` polynomials.
    pub pc_index: u32,

    /// Remapped RAM address, or `None` for cycles with no valid RAM access.
    ///
    /// Pre-computed by the caller via `remap_address()`. `None` maps to
    /// all-zero one-hot vectors in `RamRa` polynomials.
    pub ram_address: Option<u64>,
}

impl CycleData {
    /// A padding cycle (all zeros, no RAM access).
    ///
    /// Equivalent to a no-op: zero increments, lookup index 0, PC index 0,
    /// no RAM address.
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
        let pad = CycleData::PADDING;
        let def = CycleData::default();
        assert_eq!(pad.rd_inc, def.rd_inc);
        assert_eq!(pad.ram_inc, def.ram_inc);
        assert_eq!(pad.lookup_index, def.lookup_index);
        assert_eq!(pad.pc_index, def.pc_index);
        assert_eq!(pad.ram_address, def.ram_address);
    }
}
