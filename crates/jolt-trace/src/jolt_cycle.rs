//! `JoltCycle`: dynamic, runtime view of a single executed instruction.
//!
//! Pairs the static [`JoltInstruction`] (defined in `jolt-riscv`) with
//! register and RAM state captured during tracing, so `LookupQuery` impls in
//! `jolt-lookup-tables` can operate on any concrete cycle representation
//! without depending on tracer's types directly.
//!
//! The blanket impl below adapts tracer's `RISCVCycle<T>` to this trait. The
//! matching `JoltInstruction` blanket impl lives in `jolt-riscv`.

use jolt_riscv::JoltInstruction;

#[cfg(any(feature = "test-utils", test))]
use tracer::instruction::format::InstructionFormat;

use tracer::instruction::{
    format::InstructionRegisterState, RAMAccess, RISCVCycle, RISCVInstruction,
};

#[cfg(any(feature = "test-utils", test))]
use tracer::instruction::NormalizedInstruction;

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

impl<T: RISCVInstruction + JoltInstruction> JoltCycle for RISCVCycle<T> {
    type Instruction = T;

    fn instruction(&self) -> T {
        self.instruction
    }

    fn rs1_val(&self) -> Option<u64> {
        self.register_state.rs1_value()
    }

    fn rs2_val(&self) -> Option<u64> {
        self.register_state.rs2_value()
    }

    fn rd_vals(&self) -> Option<(u64, u64)> {
        self.register_state.rd_values()
    }

    fn ram_access_address(&self) -> Option<u64> {
        let ram_access: RAMAccess = self.ram_access.into();
        match ram_access {
            RAMAccess::Read(r) => Some(r.address),
            RAMAccess::Write(w) => Some(w.address),
            RAMAccess::NoOp => None,
        }
    }

    fn ram_read_value(&self) -> Option<u64> {
        let ram_access: RAMAccess = self.ram_access.into();
        match ram_access {
            RAMAccess::Read(r) => Some(r.value),
            RAMAccess::Write(w) => Some(w.pre_value),
            RAMAccess::NoOp => None,
        }
    }

    fn ram_write_value(&self) -> Option<u64> {
        let ram_access: RAMAccess = self.ram_access.into();
        match ram_access {
            RAMAccess::Read(r) => Some(r.value),
            RAMAccess::Write(w) => Some(w.post_value),
            RAMAccess::NoOp => None,
        }
    }

    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self {
        let instruction = T::random(rng);
        let normalized: NormalizedInstruction = instruction.into();
        let register_state =
            <<T::Format as InstructionFormat>::RegisterState as InstructionRegisterState>::random(
                rng,
                &normalized.operands,
            );
        // RAM access is left at the default (no-op) state. Coverage gap:
        // any `LookupQuery` that depends on `ram_access_address` /
        // `ram_read_value` / `ram_write_value` will be fuzzed against
        // all-zero RAM. None of the current `jolt-lookup-tables` impls do —
        // loads/stores have `lookup_table = None` and trivial `LookupQuery`,
        // and AMOs don't implement `LookupQuery` at all — so this is safe
        // today. Adding RAM-dependent lookup logic will require a
        // `RAMAccess::random` helper in tracer (or a per-instruction gate
        // that opts out of this generator).
        Self {
            instruction,
            register_state,
            ram_access: T::RAMAccess::default(),
        }
    }
}
