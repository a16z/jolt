use common::constants::{RISCV_REGISTER_COUNT, VIRTUAL_REGISTER_COUNT};
use std::ops::Deref;
use std::sync::Mutex;

const NUM_VIRTUAL_REGISTERS: usize = VIRTUAL_REGISTER_COUNT as usize;
const RISCV_REGISTER_BASE: u8 = RISCV_REGISTER_COUNT;

static ALLOCATOR: Mutex<VirtualRegisterAllocator> = Mutex::new(VirtualRegisterAllocator::new());

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
        for (i, allocated) in self.allocated.iter_mut().enumerate() {
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
        ALLOCATOR.lock().unwrap().deallocate(self.index);
    }
}

pub fn allocate_virtual_register() -> VirtualRegisterGuard {
    let mut allocator = ALLOCATOR.lock().unwrap();
    if let Some(index) = allocator
        .allocate()
        .map(|index| VirtualRegisterGuard { index })
    {
        index
    } else {
        drop(allocator);
        panic!("Failed to allocate virtual register: all registers in use");
    }
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

        for i in 0..NUM_VIRTUAL_REGISTERS {
            let guard = allocate_virtual_register();
            assert_eq!(*guard, RISCV_REGISTER_BASE + i as u8);
            guards.push(guard);
        }

        // This should panic
        let _guard = allocate_virtual_register();
    }
}
