//! Per-cycle input for polynomial generation.
//!
//! [`CycleInput`] is a flat, protocol-level representation of one execution
//! cycle. The caller extracts these values from whatever trace backend is
//! in use (e.g. `tracer::Cycle`) before pushing them into [`Polynomials`].
//!
//! Fields are indexed arrays so [`Polynomials::push`] can generically route
//! data to polynomial buffers via [`WitnessSlot`] descriptors — no hardcoded
//! polynomial identity matching.
//!
//! [`Polynomials`]: crate::Polynomials
//! [`WitnessSlot`]: jolt_compiler::WitnessSlot

use jolt_compiler::WitnessSlot;

/// Per-cycle input data for polynomial buffer generation.
///
/// Dense slots (indexed by [`WitnessSlot::Dense(i)`]):
/// - `0` ([`RD_INC`](WitnessSlot::RD_INC)) — register write increment (rd post − rd pre)
/// - `1` ([`RAM_INC`](WitnessSlot::RAM_INC)) — RAM write increment (ram post − ram pre)
///
/// One-hot sources (indexed by [`WitnessSlot::OneHotChunk { source, .. }`]):
/// - `0` ([`INSTRUCTION`](WitnessSlot::INSTRUCTION)) — instruction lookup index (128-bit)
/// - `1` ([`BYTECODE`](WitnessSlot::BYTECODE)) — bytecode PC index
/// - `2` ([`RAM`](WitnessSlot::RAM)) — remapped RAM address (`None` = no access)
#[derive(Clone, Copy, Debug)]
pub struct CycleInput {
    /// Dense witness values, indexed by `WitnessSlot::Dense(i)`.
    pub dense: [i128; WitnessSlot::NUM_DENSE],
    /// One-hot source values, indexed by `WitnessSlot::OneHotChunk { source, .. }`.
    /// `None` means no access for that source on this cycle.
    pub one_hot: [Option<u128>; WitnessSlot::NUM_ONE_HOT],
}

impl CycleInput {
    /// A padding cycle: zero increments, zero indices, no RAM access, no FR
    /// access (sentinel `None` for RAM and FR slots indicates no write).
    pub const PADDING: Self = Self {
        dense: [0; WitnessSlot::NUM_DENSE],
        one_hot: [Some(0), Some(0), None, None],
    };
}

impl Default for CycleInput {
    fn default() -> Self {
        Self::PADDING
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_is_default() {
        let pad = CycleInput::PADDING;
        let def = CycleInput::default();
        assert_eq!(pad.dense, def.dense);
        assert_eq!(pad.one_hot, def.one_hot);
    }
}
