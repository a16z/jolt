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
}

impl<T: JoltCycle> JoltCycle for &T {
    type Instruction = T::Instruction;

    #[inline]
    fn instruction(&self) -> Self::Instruction {
        (**self).instruction()
    }

    #[inline]
    fn rs1_val(&self) -> Option<u64> {
        (**self).rs1_val()
    }

    #[inline]
    fn rs2_val(&self) -> Option<u64> {
        (**self).rs2_val()
    }

    #[inline]
    fn rd_vals(&self) -> Option<(u64, u64)> {
        (**self).rd_vals()
    }

    #[inline]
    fn ram_access_address(&self) -> Option<u64> {
        (**self).ram_access_address()
    }

    #[inline]
    fn ram_read_value(&self) -> Option<u64> {
        (**self).ram_read_value()
    }

    #[inline]
    fn ram_write_value(&self) -> Option<u64> {
        (**self).ram_write_value()
    }
}
