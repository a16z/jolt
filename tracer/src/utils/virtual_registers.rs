use common::constants::{
    RISCV_REGISTER_COUNT, VIRTUAL_INSTRUCTION_RESERVED_REGISTER_COUNT, VIRTUAL_REGISTER_COUNT,
};
use std::ops::Deref;
use std::sync::{Arc, Mutex};

const NUM_VIRTUAL_REGISTERS: usize = VIRTUAL_REGISTER_COUNT as usize;
const NUM_VIRTUAL_INSTRUCTION_REGISTERS: usize =
    VIRTUAL_INSTRUCTION_RESERVED_REGISTER_COUNT as usize;
const RISCV_REGISTER_BASE: u8 = RISCV_REGISTER_COUNT;

#[derive(Debug, Clone)]
pub struct VirtualRegisterAllocator {
    allocated: Arc<Mutex<[bool; NUM_VIRTUAL_REGISTERS]>>,
}

impl VirtualRegisterAllocator {
    pub fn new() -> Self {
        Self {
            allocated: Arc::new(Mutex::new([false; NUM_VIRTUAL_REGISTERS])),
        }
    }

    /// Allocate virtual register that can be used in the inline sequence of
    /// an instruction
    pub(crate) fn allocate(&self) -> VirtualRegisterGuard {
        for (i, allocated) in self
            .allocated
            .lock()
            .expect("Failed to lock virtual register allocator")
            .iter_mut()
            .enumerate()
            .take(NUM_VIRTUAL_INSTRUCTION_REGISTERS)
        {
            if !*allocated {
                *allocated = true;
                return VirtualRegisterGuard {
                    index: i as u8 + RISCV_REGISTER_BASE,
                    allocator: self.clone(),
                };
            }
        }
        panic!("Failed to allocate virtual register for instruction: No registers left");
    }

    /// Allocate virtual register that can be used in an inline
    pub fn allocate_for_inline(&self) -> VirtualRegisterGuard {
        for (i, allocated) in self
            .allocated
            .lock()
            .expect("Failed to lock virtual register allocator")
            .iter_mut()
            .enumerate()
            .skip(NUM_VIRTUAL_INSTRUCTION_REGISTERS)
        {
            if !*allocated {
                *allocated = true;
                return VirtualRegisterGuard {
                    index: i as u8 + RISCV_REGISTER_BASE,
                    allocator: self.clone(),
                };
            }
        }
        panic!("Failed to allocate virtual register for inline: No registers left");
    }

    fn deallocate(&self, index: u8) {
        let virtual_index = (index - RISCV_REGISTER_BASE) as usize;
        if virtual_index < NUM_VIRTUAL_REGISTERS {
            self.allocated
                .lock()
                .expect("Failed to lock virtual register allocator")[virtual_index] = false;
        }
    }
}

impl Default for VirtualRegisterAllocator {
    fn default() -> Self {
        Self::new()
    }
}

pub struct VirtualRegisterGuard {
    index: u8,
    allocator: VirtualRegisterAllocator,
}

impl Deref for VirtualRegisterGuard {
    type Target = u8;

    fn deref(&self) -> &Self::Target {
        &self.index
    }
}

impl Drop for VirtualRegisterGuard {
    fn drop(&mut self) {
        self.allocator.deallocate(self.index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_deallocate() {
        let allocator = VirtualRegisterAllocator::new();
        {
            let guard1 = allocator.allocate();
            assert_eq!(*guard1, RISCV_REGISTER_BASE);

            let guard2 = allocator.allocate();
            assert_eq!(*guard2, RISCV_REGISTER_BASE + 1);
        }

        let guard3 = allocator.allocate();
        assert_eq!(*guard3, RISCV_REGISTER_BASE);
    }

    #[test]
    fn test_deref() {
        let allocator = VirtualRegisterAllocator::new();
        let guard = allocator.allocate();
        let index: u8 = *guard;
        assert_eq!(index, RISCV_REGISTER_BASE);
    }

    #[test]
    #[should_panic(expected = "Failed to allocate virtual register")]
    fn test_exhaustion_panic() {
        let allocator = VirtualRegisterAllocator::new();
        let mut guards = Vec::new();

        for i in 0..NUM_VIRTUAL_INSTRUCTION_REGISTERS {
            let guard = allocator.allocate();
            assert_eq!(*guard, RISCV_REGISTER_BASE + i as u8);
            guards.push(guard);
        }

        // This should panic
        let _guard = allocator.allocate();
    }

    #[test]
    fn test_allocate_deallocate_inline() {
        let allocator = VirtualRegisterAllocator::new();
        {
            let guard1 = allocator.allocate_for_inline();
            assert_eq!(
                *guard1,
                RISCV_REGISTER_BASE + NUM_VIRTUAL_INSTRUCTION_REGISTERS as u8
            );

            let guard2 = allocator.allocate_for_inline();
            assert_eq!(
                *guard2,
                RISCV_REGISTER_BASE + NUM_VIRTUAL_INSTRUCTION_REGISTERS as u8 + 1
            );
        }

        let guard3 = allocator.allocate_for_inline();
        assert_eq!(
            *guard3,
            RISCV_REGISTER_BASE + NUM_VIRTUAL_INSTRUCTION_REGISTERS as u8
        );
    }

    #[test]
    fn test_deref_inline() {
        let allocator = VirtualRegisterAllocator::new();
        let guard = allocator.allocate_for_inline();
        let index: u8 = *guard;
        assert_eq!(
            index,
            RISCV_REGISTER_BASE + NUM_VIRTUAL_INSTRUCTION_REGISTERS as u8
        );
    }

    #[test]
    #[should_panic(expected = "Failed to allocate virtual register")]
    fn test_exhaustion_panic_inline() {
        let allocator = VirtualRegisterAllocator::new();
        let mut guards = Vec::new();

        let num_inline_registers = NUM_VIRTUAL_REGISTERS - NUM_VIRTUAL_INSTRUCTION_REGISTERS;
        for i in 0..num_inline_registers {
            let guard = allocator.allocate_for_inline();
            assert_eq!(
                *guard,
                RISCV_REGISTER_BASE + NUM_VIRTUAL_INSTRUCTION_REGISTERS as u8 + i as u8
            );
            guards.push(guard);
        }

        // This should panic
        let _guard = allocator.allocate_for_inline();
    }

    #[test]
    fn test_combined_allocate_and_inline() {
        let allocator = VirtualRegisterAllocator::new();
        // Allocate some regular registers
        let guard1 = allocator.allocate();
        assert_eq!(*guard1, RISCV_REGISTER_BASE);

        let guard2 = allocator.allocate();
        assert_eq!(*guard2, RISCV_REGISTER_BASE + 1);

        // Allocate some inline registers
        let inline_guard1 = allocator.allocate_for_inline();
        assert_eq!(
            *inline_guard1,
            RISCV_REGISTER_BASE + NUM_VIRTUAL_INSTRUCTION_REGISTERS as u8
        );

        let inline_guard2 = allocator.allocate_for_inline();
        assert_eq!(
            *inline_guard2,
            RISCV_REGISTER_BASE + NUM_VIRTUAL_INSTRUCTION_REGISTERS as u8 + 1
        );

        // Allocate more regular registers
        let guard3 = allocator.allocate();
        assert_eq!(*guard3, RISCV_REGISTER_BASE + 2);

        // Drop some guards and reallocate
        drop(guard2);
        drop(inline_guard1);

        // Should reuse the freed slots
        let guard4 = allocator.allocate();
        assert_eq!(*guard4, RISCV_REGISTER_BASE + 1);

        let inline_guard3 = allocator.allocate_for_inline();
        assert_eq!(
            *inline_guard3,
            RISCV_REGISTER_BASE + NUM_VIRTUAL_INSTRUCTION_REGISTERS as u8
        );
    }
}
