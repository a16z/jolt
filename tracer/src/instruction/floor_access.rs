//! Execution shared by the floor-addressed memory rows (`VirtualLoad*` and
//! `VirtualStore*`), which access the doubleword containing `ea = rs1 + imm`.

use super::format::{format_load::FormatLoad, format_s::FormatS};
use super::{RamRead, RamWrite};
use crate::emulator::cpu::Cpu;

pub(super) use misaligned::aligned_sequences_forced;
#[cfg(any(feature = "test-utils", test))]
pub use misaligned::{force_aligned_sequences, ForcedAlignedSequences};

/// Alignment class of a floor-addressed row.
#[derive(Clone, Copy, Debug)]
pub(super) enum AlignmentClass {
    Byte,
    Halfword,
    Word,
}

impl AlignmentClass {
    /// The offset bits the row's `AddressOffset` lookup returns: `ea & mask`.
    const fn offset_mask(self) -> u64 {
        match self {
            Self::Byte => 7,
            Self::Halfword => 6,
            Self::Word => 4,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Byte => "byte",
            Self::Halfword => "halfword",
            Self::Word => "word",
        }
    }
}

/// Loads the doubleword containing `rs1 + imm` into `rd`.
pub(super) fn load(cpu: &mut Cpu, operands: &FormatLoad, class: AlignmentClass) -> RamRead {
    let (address, doubleword) = addresses(cpu, operands.rs1, operands.imm, class);
    let (value, _) = cpu
        .get_mut_mmu()
        .load_doubleword(doubleword)
        .unwrap_or_else(|_| panic!("MMU load error"));
    cpu.write_register(operands.rd as usize, value as i64);
    RamRead { address, value }
}

/// Stores `rs2` to the doubleword containing `rs1 + imm`.
pub(super) fn store(cpu: &mut Cpu, operands: &FormatS, class: AlignmentClass) -> RamWrite {
    let (address, doubleword) = addresses(cpu, operands.rs1, operands.imm, class);
    let write = cpu
        .mmu
        .store_doubleword(doubleword, cpu.x[operands.rs2 as usize] as u64)
        .unwrap_or_else(|_| panic!("MMU store error"));
    RamWrite { address, ..write }
}

/// The doubleword a native `LD`/`SD` at `ea` accesses: `ea` itself, whose
/// alignment the MMU asserts. Under [`force_aligned_sequences`] a misaligned
/// one accesses the containing doubleword while its row still records `ea`.
pub(super) fn native_doubleword(ea: u64) -> u64 {
    if aligned_sequences_forced() {
        ea & !7
    } else {
        ea
    }
}

/// Returns the RAM address the row records and the doubleword it accesses.
///
/// The recorded address is `ea - AddressOffset(ea)`, the value R1CS
/// constraint 0 assigns to `RamAddress`. It is the accessed doubleword's
/// address exactly when `ea` is aligned to the row's class; otherwise RAM RAF
/// evaluation rejects it, so a misaligned access panics here instead of
/// producing an unprovable trace.
fn addresses(cpu: &Cpu, rs1: u8, imm: i64, class: AlignmentClass) -> (u64, u64) {
    let ea = (cpu.x[rs1 as usize] as u64).wrapping_add(imm as u64);
    let address = ea - (ea & class.offset_mask());
    assert!(
        address.is_multiple_of(8) || aligned_sequences_forced(),
        "RAM access is not {} aligned: {ea:#x}",
        class.name()
    );
    (address, ea & !7)
}

#[cfg(not(any(feature = "test-utils", test)))]
mod misaligned {
    pub(in crate::instruction) const fn aligned_sequences_forced() -> bool {
        false
    }
}

/// Aligned sequences at misaligned addresses, for soundness tests.
#[cfg(any(feature = "test-utils", test))]
mod misaligned {
    use std::cell::Cell;
    use std::marker::PhantomData;

    thread_local! {
        static FORCED: Cell<bool> = const { Cell::new(false) };
    }

    /// Makes dual loads and stores on this thread run their aligned sequence
    /// at any address, and lets floor-addressed rows and native `LD`/`SD`
    /// rows trace an access misaligned for their class instead of panicking.
    /// Such a row still accesses the containing doubleword but records the
    /// R1CS-consistent RAM address, `ea - AddressOffset(ea)` or `ea`, which is
    /// not doubleword-aligned, so only RAM RAF evaluation can reject the
    /// trace. The returned guard restores the honest behavior.
    ///
    /// Only serial tracing sees the setting; two-pass parallel tracing
    /// replays chunks on other threads.
    pub fn force_aligned_sequences() -> ForcedAlignedSequences {
        assert!(
            crate::parallel_config_from_env().is_none(),
            "forced aligned sequences need serial tracing; unset TRACER_PARALLEL"
        );
        FORCED.with(|forced| {
            assert!(
                !forced.replace(true),
                "aligned sequences are already forced on this thread"
            );
        });
        ForcedAlignedSequences {
            _thread: PhantomData,
        }
    }

    /// The scope of a [`force_aligned_sequences`] call on this thread.
    pub struct ForcedAlignedSequences {
        _thread: PhantomData<*const ()>,
    }

    impl Drop for ForcedAlignedSequences {
        fn drop(&mut self) {
            FORCED.with(|forced| forced.set(false));
        }
    }

    /// Whether dual loads and stores on this thread run their aligned
    /// sequence whatever their address (see [`force_aligned_sequences`]).
    pub(in crate::instruction) fn aligned_sequences_forced() -> bool {
        FORCED.with(Cell::get)
    }
}
