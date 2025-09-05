use common::constants::{
    RISCV_REGISTER_COUNT, VIRTUAL_INSTRUCTION_RESERVED_REGISTER_COUNT, VIRTUAL_REGISTER_COUNT,
};
use core::cell::RefCell;
use std::ops::Deref;

const NUM_VIRTUAL_REGISTERS: usize = VIRTUAL_REGISTER_COUNT as usize;
const NUM_VIRTUAL_INSTRUCTION_REGISTERS: usize =
    VIRTUAL_INSTRUCTION_RESERVED_REGISTER_COUNT as usize;
const RISCV_REGISTER_BASE: u8 = RISCV_REGISTER_COUNT;

thread_local! {
    static ALLOCATOR: RefCell<VirtualRegisterAllocator> = const { RefCell::new(VirtualRegisterAllocator::new()) };
}

struct VirtualRegisterAllocator {
    allocated: [bool; NUM_VIRTUAL_REGISTERS],
}

impl VirtualRegisterAllocator {
    const fn new() -> Self {
        Self {
            allocated: [false; NUM_VIRTUAL_REGISTERS],
        }
    }

    fn allocate(&mut self) -> Option<u8> {
        for (i, allocated) in self
            .allocated
            .iter_mut()
            .enumerate()
            .take(NUM_VIRTUAL_INSTRUCTION_REGISTERS)
        {
            if !*allocated {
                *allocated = true;
                return Some(i as u8 + RISCV_REGISTER_BASE);
            }
        }
        None
    }

    fn allocate_for_inline(&mut self) -> Option<u8> {
        for (i, allocated) in self
            .allocated
            .iter_mut()
            .enumerate()
            .skip(NUM_VIRTUAL_INSTRUCTION_REGISTERS)
        {
            if !*allocated {
                *allocated = true;
                return Some(i as u8 + RISCV_REGISTER_BASE);
            }
        }
        None
    }

    fn deallocate(&mut self, index: u8) {
        let virtual_index = (index - RISCV_REGISTER_BASE) as usize;
        if virtual_index < NUM_VIRTUAL_REGISTERS {
            self.allocated[virtual_index] = false;
        }
    }
}

pub struct VirtualRegisterGuard {
    index: u8,
}

impl Deref for VirtualRegisterGuard {
    type Target = u8;

    fn deref(&self) -> &Self::Target {
        &self.index
    }
}

impl Drop for VirtualRegisterGuard {
    fn drop(&mut self) {
        ALLOCATOR.with(|allocator| {
            allocator.borrow_mut().deallocate(self.index);
        });
    }
}

pub(crate) fn allocate_virtual_register() -> VirtualRegisterGuard {
    ALLOCATOR.with(|allocator| {
        if let Some(index) = allocator.borrow_mut().allocate() {
            VirtualRegisterGuard { index }
        } else {
            panic!("Failed to allocate virtual register: all registers in use");
        }
    })
}

pub fn allocate_virtual_register_for_inline() -> VirtualRegisterGuard {
    ALLOCATOR.with(|allocator| {
        if let Some(index) = allocator.borrow_mut().allocate_for_inline() {
            VirtualRegisterGuard { index }
        } else {
            panic!("Failed to allocate virtual register: all registers in use");
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_deallocate() {
        {
            let guard1 = allocate_virtual_register();
            assert_eq!(*guard1, RISCV_REGISTER_BASE);

            let guard2 = allocate_virtual_register();
            assert_eq!(*guard2, RISCV_REGISTER_BASE + 1);
        }

        let guard3 = allocate_virtual_register();
        assert_eq!(*guard3, RISCV_REGISTER_BASE);
    }

    #[test]
    fn test_deref() {
        let guard = allocate_virtual_register();
        let index: u8 = *guard;
        assert_eq!(index, RISCV_REGISTER_BASE);
    }

    #[test]
    #[should_panic(expected = "Failed to allocate virtual register")]
    fn test_exhaustion_panic() {
        let mut guards = Vec::new();

        for i in 0..NUM_VIRTUAL_INSTRUCTION_REGISTERS {
            let guard = allocate_virtual_register();
            assert_eq!(*guard, RISCV_REGISTER_BASE + i as u8);
            guards.push(guard);
        }

        // This should panic
        let _guard = allocate_virtual_register();
    }

    #[test]
    fn test_allocate_deallocate_inline() {
        {
            let guard1 = allocate_virtual_register_for_inline();
            assert_eq!(
                *guard1,
                RISCV_REGISTER_BASE + NUM_VIRTUAL_INSTRUCTION_REGISTERS as u8
            );

            let guard2 = allocate_virtual_register_for_inline();
            assert_eq!(
                *guard2,
                RISCV_REGISTER_BASE + NUM_VIRTUAL_INSTRUCTION_REGISTERS as u8 + 1
            );
        }

        let guard3 = allocate_virtual_register_for_inline();
        assert_eq!(
            *guard3,
            RISCV_REGISTER_BASE + NUM_VIRTUAL_INSTRUCTION_REGISTERS as u8
        );
    }

    #[test]
    fn test_deref_inline() {
        let guard = allocate_virtual_register_for_inline();
        let index: u8 = *guard;
        assert_eq!(
            index,
            RISCV_REGISTER_BASE + NUM_VIRTUAL_INSTRUCTION_REGISTERS as u8
        );
    }

    #[test]
    #[should_panic(expected = "Failed to allocate virtual register")]
    fn test_exhaustion_panic_inline() {
        let mut guards = Vec::new();

        let num_inline_registers = NUM_VIRTUAL_REGISTERS - NUM_VIRTUAL_INSTRUCTION_REGISTERS;
        for i in 0..num_inline_registers {
            let guard = allocate_virtual_register_for_inline();
            assert_eq!(
                *guard,
                RISCV_REGISTER_BASE + NUM_VIRTUAL_INSTRUCTION_REGISTERS as u8 + i as u8
            );
            guards.push(guard);
        }

        // This should panic
        let _guard = allocate_virtual_register_for_inline();
    }

    #[test]
    fn test_combined_allocate_and_inline() {
        // Allocate some regular registers
        let guard1 = allocate_virtual_register();
        assert_eq!(*guard1, RISCV_REGISTER_BASE);

        let guard2 = allocate_virtual_register();
        assert_eq!(*guard2, RISCV_REGISTER_BASE + 1);

        // Allocate some inline registers
        let inline_guard1 = allocate_virtual_register_for_inline();
        assert_eq!(
            *inline_guard1,
            RISCV_REGISTER_BASE + NUM_VIRTUAL_INSTRUCTION_REGISTERS as u8
        );

        let inline_guard2 = allocate_virtual_register_for_inline();
        assert_eq!(
            *inline_guard2,
            RISCV_REGISTER_BASE + NUM_VIRTUAL_INSTRUCTION_REGISTERS as u8 + 1
        );

        // Allocate more regular registers
        let guard3 = allocate_virtual_register();
        assert_eq!(*guard3, RISCV_REGISTER_BASE + 2);

        // Drop some guards and reallocate
        drop(guard2);
        drop(inline_guard1);

        // Should reuse the freed slots
        let guard4 = allocate_virtual_register();
        assert_eq!(*guard4, RISCV_REGISTER_BASE + 1);

        let inline_guard3 = allocate_virtual_register_for_inline();
        assert_eq!(
            *inline_guard3,
            RISCV_REGISTER_BASE + NUM_VIRTUAL_INSTRUCTION_REGISTERS as u8
        );
    }
}
