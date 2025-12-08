/// DRAM base address. Offset from this base address
/// is the address in main memory.
pub const DRAM_BASE: u64 = RAM_START_ADDRESS;

use crate::instruction::{RAMRead, RAMWrite};
use common::constants::{RAM_START_ADDRESS, STACK_CANARY_SIZE};
use common::jolt_device::JoltDevice;

use super::cpu::{get_privilege_mode, PrivilegeMode, Trap, TrapType, Xlen};
use super::memory::Memory;
use super::terminal::Terminal;

/// Emulates Memory Management Unit. It holds the Main memory and peripheral
/// devices, maps address to them, and accesses them depending on address.
/// It also manages virtual-physical address translation and memory protection.
/// It may also be said Bus.
/// @TODO: Memory protection is not implemented yet. We should support.
#[derive(Clone, Debug)]
pub struct Mmu {
    clock: u64,
    xlen: Xlen,
    ppn: u64,
    addressing_mode: AddressingMode,
    privilege_mode: PrivilegeMode,
    pub memory: MemoryWrapper,

    pub jolt_device: Option<JoltDevice>,

    /// Address translation can be affected `mstatus` (MPRV, MPP in machine mode)
    /// then `Mmu` has copy of it.
    mstatus: u64,
}

#[derive(Clone, Debug)]
pub enum AddressingMode {
    None,
    SV32,
    SV39,
    SV48, // @TODO: Implement
}

enum MemoryAccessType {
    Execute,
    Read,
    Write,
}

fn _get_addressing_mode_name(mode: &AddressingMode) -> &'static str {
    match mode {
        AddressingMode::None => "None",
        AddressingMode::SV32 => "SV32",
        AddressingMode::SV39 => "SV39",
        AddressingMode::SV48 => "SV48",
    }
}

impl Mmu {
    /// Creates a new `Mmu`.
    ///
    /// # Arguments
    /// * `xlen`
    /// * `terminal`
    /// * `tracer`
    pub fn new(xlen: Xlen, _terminal: Box<dyn Terminal>) -> Self {
        Mmu {
            clock: 0,
            xlen,
            ppn: 0,
            addressing_mode: AddressingMode::None,
            privilege_mode: PrivilegeMode::Machine,
            memory: MemoryWrapper::new(),
            jolt_device: None,
            mstatus: 0,
        }
    }

    /// Updates XLEN, 32-bit or 64-bit
    ///
    /// # Arguments
    /// * `xlen`
    pub fn update_xlen(&mut self, xlen: Xlen) {
        self.xlen = xlen;
    }

    /// Initializes Main memory. This method is expected to be called only once.
    ///
    /// # Arguments
    /// * `capacity`
    pub fn init_memory(&mut self, capacity: u64) {
        self.memory.init(capacity);
    }

    /// Runs one cycle of MMU and peripheral devices.
    pub fn tick(&mut self) {
        self.clock = self.clock.wrapping_add(1);
    }

    /// Updates addressing mode
    ///
    /// # Arguments
    /// * `new_addressing_mode`
    pub fn update_addressing_mode(&mut self, new_addressing_mode: AddressingMode) {
        self.addressing_mode = new_addressing_mode;
    }

    /// Updates privilege mode
    ///
    /// # Arguments
    /// * `mode`
    pub fn update_privilege_mode(&mut self, mode: PrivilegeMode) {
        self.privilege_mode = mode;
    }

    /// Updates mstatus copy. `CPU` needs to call this method whenever
    /// `mstatus` is updated.
    ///
    /// # Arguments
    /// * `mstatus`
    pub fn update_mstatus(&mut self, mstatus: u64) {
        self.mstatus = mstatus;
    }

    /// Updates PPN used for address translation
    ///
    /// # Arguments
    /// * `ppn`
    pub fn update_ppn(&mut self, ppn: u64) {
        self.ppn = ppn;
    }

    fn get_effective_address(&self, address: u64) -> u64 {
        match self.xlen {
            Xlen::Bit32 => address & 0xffffffff,
            Xlen::Bit64 => address,
        }
    }

    #[inline]
    fn assert_effective_store_address(&self, effective_address: u64) {
        self.assert_effective_address(effective_address, true)
    }

    #[inline]
    fn assert_effective_load_address(&self, effective_address: u64) {
        self.assert_effective_address(effective_address, false)
    }

    /// Asserts the validity of an effective memory address.
    /// Panics if the address is invalid.
    ///
    /// # Arguments
    /// * `effective_address` Effective memory address to validate
    #[inline]
    fn assert_effective_address(&self, ea: u64, is_write: bool) {
        if self.jolt_device.is_none() {
            return;
        }

        let jolt_device = self.jolt_device.as_ref().unwrap();
        let layout = &jolt_device.memory_layout;
        // helper strings
        let (action, verb) = if is_write {
            ("Store", "write to")
        } else {
            ("Load", "read from")
        };

        if ea < DRAM_BASE {
            // below DRAM_BASE => I/O
            // bounds‐check against the termination (top) of the I/O region
            assert!(
                ea <= layout.io_end,
                "I/O overflow: Attempted to {verb} 0x{ea:X}. Out of bounds.\n{layout:#?}",
            );
            assert!(
                ea >= layout.get_lowest_address(),
                "I/O underflow: Attempted to {verb} 0x{ea:X}. Out of bounds.\n{layout:#?}",
            );

            // then check for device I/O pages
            let ok = if is_write {
                // stores only to output/panic/termination
                jolt_device.is_output(ea)
                    || jolt_device.is_panic(ea)
                    || jolt_device.is_termination(ea)
            } else {
                // loads also from input
                jolt_device.is_input(ea)
                    || jolt_device.is_trusted_advice(ea)
                    || jolt_device.is_untrusted_advice(ea)
                    || jolt_device.is_output(ea)
                    || jolt_device.is_panic(ea)
                    || jolt_device.is_termination(ea)
            };
            assert!(
                ok,
                "Illegal device {}: Unknown memory mapping: 0x{ea:X}\n{layout:#?}",
                action.to_lowercase(),
            );
        } else {
            // check within RAM
            if is_write {
                // These errors aren't necessarily correct as there's no way to distinguish between an
                // attempt to write to the stack vs heap, but they're trying their best
                assert!(
                    ea <= layout.stack_end || ea > layout.stack_end + STACK_CANARY_SIZE,
                    "Stack overflow: Triggered Stack Canary. Attempted to {verb} 0x{ea:X}.\n{layout:#?}",
                );
                assert!(
                    ea < layout.memory_end,
                    "Heap overflow: Attempted to {verb} 0x{ea:X}. Heap too small.\n{layout:#?}",
                );
            } else {
                // allow reads across the whole designated memory region as long as the address is valid
                assert!(
                    ea < layout.memory_end,
                    "Illegal Memory Access: Attempted to {verb} 0x{ea:X}.\n{layout:#?}",
                );
            }
        }
    }

    /// Fetches an instruction byte. This method takes virtual address
    /// and translates into physical address inside.
    ///
    /// # Arguments
    /// * `v_address` Virtual address
    fn fetch(&mut self, v_address: u64) -> Result<u8, Trap> {
        match self.translate_address(v_address, &MemoryAccessType::Execute) {
            Ok(p_address) => Ok(self.load_raw(p_address)),
            Err(()) => Err(Trap {
                trap_type: TrapType::InstructionPageFault,
                value: v_address,
            }),
        }
    }

    /// Fetches instruction four bytes. This method takes virtual address
    /// and translates into physical address inside.
    ///
    /// # Arguments
    /// * `v_address` Virtual address
    pub fn fetch_word(&mut self, v_address: u64) -> Result<u32, Trap> {
        let width = 4;
        match (v_address & 0xfff) <= (0x1000 - width) {
            true => {
                // Fast path. All bytes fetched are in the same page so
                // translating an address only once.
                let effective_address = self.get_effective_address(v_address);
                match self.translate_address(effective_address, &MemoryAccessType::Execute) {
                    Ok(p_address) => Ok(self.load_word_raw(p_address)),
                    Err(()) => Err(Trap {
                        trap_type: TrapType::InstructionPageFault,
                        value: effective_address,
                    }),
                }
            }
            false => {
                let mut data = 0_u32;
                for i in 0..width {
                    match self.fetch(v_address.wrapping_add(i)) {
                        Ok(byte) => data |= (byte as u32) << (i * 8),
                        Err(e) => return Err(e),
                    };
                }
                Ok(data)
            }
        }
    }

    /// Loads a byte. This method takes virtual address and translates
    /// into physical address inside.
    ///
    /// # Arguments
    /// * `v_address` Virtual address
    pub fn load(&mut self, v_address: u64) -> Result<(u8, RAMRead), Trap> {
        let effective_address = self.get_effective_address(v_address);
        let memory_read = self.trace_load(effective_address);
        match self.translate_address(effective_address, &MemoryAccessType::Read) {
            Ok(p_address) => Ok((self.load_raw(p_address), memory_read)),
            Err(()) => Err(Trap {
                trap_type: TrapType::LoadPageFault,
                value: v_address,
            }),
        }
    }

    /// Loads multiple bytes. This method takes virtual address and translates
    /// into physical address inside.
    ///
    /// # Arguments
    /// * `v_address` Virtual address
    /// * `width` Must be 1, 2, 4, or 8
    fn load_bytes(&mut self, v_address: u64, width: u64) -> Result<u64, Trap> {
        debug_assert!(
            width == 1 || width == 2 || width == 4 || width == 8,
            "Width must be 1, 2, 4, or 8. {width:X}"
        );
        match (v_address & 0xfff) <= (0x1000 - width) {
            true => match self.translate_address(v_address, &MemoryAccessType::Read) {
                Ok(p_address) => {
                    // Fast path. All bytes fetched are in the same page so
                    // translating an address only once.
                    match width {
                        1 => Ok(self.load_raw(p_address) as u64),
                        2 => Ok(self.load_halfword_raw(p_address) as u64),
                        4 => Ok(self.load_word_raw(p_address) as u64),
                        8 => Ok(self.load_doubleword_raw(p_address)),
                        _ => panic!("Width must be 1, 2, 4, or 8. {width:X}"),
                    }
                }
                Err(()) => Err(Trap {
                    trap_type: TrapType::LoadPageFault,
                    value: v_address,
                }),
            },
            false => {
                let mut data = 0_u64;
                for i in 0..width {
                    match self.load(v_address.wrapping_add(i)) {
                        Ok((byte, _)) => data |= (byte as u64) << (i * 8),
                        Err(e) => return Err(e),
                    };
                }
                Ok(data)
            }
        }
    }

    /// Loads two bytes. This method takes virtual address and translates
    /// into physical address inside.
    ///
    /// # Arguments
    /// * `v_address` Virtual address
    pub fn load_halfword(&mut self, v_address: u64) -> Result<(u16, RAMRead), Trap> {
        let effective_address = self.get_effective_address(v_address);
        assert!(
            effective_address.is_multiple_of(2),
            "Unaligned load_halfword"
        );
        let memory_read = self.trace_load(effective_address);
        match self.load_bytes(v_address, 2) {
            Ok(data) => Ok((data as u16, memory_read)),
            Err(e) => Err(e),
        }
    }

    /// Loads four bytes. This method takes virtual address and translates
    /// into physical address inside.
    ///
    /// # Arguments
    /// * `v_address` Virtual address
    pub fn load_word(&mut self, v_address: u64) -> Result<(u32, RAMRead), Trap> {
        let effective_address = self.get_effective_address(v_address);
        assert_eq!(effective_address % 4, 0, "Unaligned load_word");
        let memory_read = self.trace_load(effective_address);
        match self.load_bytes(v_address, 4) {
            Ok(data) => Ok((data as u32, memory_read)),
            Err(e) => Err(e),
        }
    }

    /// Loads eight bytes. This method takes virtual address and translates
    /// into physical address inside.
    ///
    /// # Arguments
    /// * `v_address` Virtual address
    pub fn load_doubleword(&mut self, v_address: u64) -> Result<(u64, RAMRead), Trap> {
        let effective_address = self.get_effective_address(v_address);
        assert_eq!(effective_address % 8, 0, "Unaligned load_doubleword");
        let memory_read = self.trace_load(effective_address);
        match self.load_bytes(v_address, 8) {
            Ok(data) => Ok((data, memory_read)),
            Err(e) => Err(e),
        }
    }

    /// Store an byte. This method takes virtual address and translates
    /// into physical address inside.
    ///
    /// # Arguments
    /// * `v_address` Virtual address
    /// * `value`
    pub fn store(&mut self, v_address: u64, value: u8) -> Result<RAMWrite, Trap> {
        let effective_address = self.get_effective_address(v_address);
        let memory_write = self.trace_store_byte(effective_address, value as u64);
        match self.translate_address(v_address, &MemoryAccessType::Write) {
            Ok(p_address) => {
                self.store_raw(p_address, value);
                Ok(memory_write)
            }
            Err(()) => Err(Trap {
                trap_type: TrapType::StorePageFault,
                value: v_address,
            }),
        }
    }

    /// Stores multiple bytes. This method takes virtual address and translates
    /// into physical address inside.
    ///
    /// # Arguments
    /// * `v_address` Virtual address
    /// * `value` data written
    /// * `width` Must be 1, 2, 4, or 8
    fn store_bytes(&mut self, v_address: u64, value: u64, width: u64) -> Result<(), Trap> {
        debug_assert!(
            width == 1 || width == 2 || width == 4 || width == 8,
            "Width must be 1, 2, 4, or 8. {width:X}"
        );
        match (v_address & 0xfff) <= (0x1000 - width) {
            true => match self.translate_address(v_address, &MemoryAccessType::Write) {
                Ok(p_address) => {
                    // Fast path. All bytes fetched are in the same page so
                    // translating an address only once.
                    match width {
                        1 => self.store_raw(p_address, value as u8),
                        2 => self.store_halfword_raw(p_address, value as u16),
                        4 => self.store_word_raw(p_address, value as u32),
                        8 => self.store_doubleword_raw(p_address, value),
                        _ => panic!("Width must be 1, 2, 4, or 8. {width:X}"),
                    }
                    Ok(())
                }
                Err(()) => Err(Trap {
                    trap_type: TrapType::StorePageFault,
                    value: v_address,
                }),
            },
            false => {
                for i in 0..width {
                    match self.store(v_address.wrapping_add(i), ((value >> (i * 8)) & 0xff) as u8) {
                        Ok(_) => {}
                        Err(e) => return Err(e),
                    }
                }
                Ok(())
            }
        }
    }

    /// Stores two bytes. This method takes virtual address and translates
    /// into physical address inside.
    ///
    /// # Arguments
    /// * `v_address` Virtual address
    /// * `value` data written
    pub fn store_halfword(&mut self, v_address: u64, value: u16) -> Result<RAMWrite, Trap> {
        let effective_address = self.get_effective_address(v_address);
        assert_eq!(effective_address % 2, 0, "Unaligned store_halfword");
        let memory_write = self.trace_store_halfword(effective_address, value as u64);
        self.store_bytes(v_address, value as u64, 2)?;
        Ok(memory_write)
    }

    /// Stores four bytes. This method takes virtual address and translates
    /// into physical address inside.
    ///
    /// # Arguments
    /// * `v_address` Virtual address
    /// * `value` data written
    pub fn store_word(&mut self, v_address: u64, value: u32) -> Result<RAMWrite, Trap> {
        let effective_address = self.get_effective_address(v_address);
        assert_eq!(effective_address % 4, 0, "Unaligned store_word");
        let memory_write = self.trace_store(effective_address, value as u64);
        self.store_bytes(v_address, value as u64, 4)?;
        Ok(memory_write)
    }

    /// Stores eight bytes. This method takes virtual address and translates
    /// into physical address inside.
    ///
    /// # Arguments
    /// * `v_address` Virtual address
    /// * `value` data written
    pub fn store_doubleword(&mut self, v_address: u64, value: u64) -> Result<RAMWrite, Trap> {
        let effective_address = self.get_effective_address(v_address);
        assert_eq!(effective_address % 8, 0, "Unaligned store_doubleword");
        let memory_write = self.trace_store(effective_address, value);
        self.store_bytes(v_address, value, 8)?;
        Ok(memory_write)
    }

    /// Loads a byte from main memory or peripheral devices depending on
    /// physical address.
    ///
    /// # Arguments
    /// * `p_address` Physical address
    pub fn load_raw(&mut self, p_address: u64) -> u8 {
        let effective_address = self.get_effective_address(p_address);
        self.assert_effective_load_address(effective_address);
        // @TODO: Mapping should be configurable with dtb
        match effective_address >= DRAM_BASE {
            true => self.memory.read_byte(effective_address),
            false => match effective_address {
                // I don't know why but dtb data seems to be stored from 0x1020 on Linux.
                // It might be from self.x[0xb] initialization?
                // And DTB size is arbitrary.
                0x00001020..=0x00001fff => panic!("load_raw:dtb is unsupported."),
                0x02000000..=0x0200ffff => panic!("load_raw:clint is unsupported."),
                0x0C000000..=0x0fffffff => panic!("load_raw:plic is unsupported."),
                0x10000000..=0x100000ff => panic!("load_raw:UART is unsupported."),
                0x10001000..=0x10001FFF => panic!("load_raw:disk is unsupported."),
                _ => {
                    if let Some(jolt_device) = self.jolt_device.as_ref() {
                        if jolt_device.is_input(effective_address)
                            || jolt_device.is_trusted_advice(effective_address)
                            || jolt_device.is_untrusted_advice(effective_address)
                            || jolt_device.is_output(effective_address)
                            || jolt_device.is_panic(effective_address)
                            || jolt_device.is_termination(effective_address)
                        {
                            return jolt_device.load(effective_address);
                        }
                    }
                    panic!("Load Failed: Unknown memory mapping {effective_address:X}.");
                }
            },
        }
    }

    /// Records the memory word being accessed by a load instruction. The memory
    /// state is used in Jolt to construct the witnesses in `read_write_memory.rs`.
    fn trace_load(&self, effective_address: u64) -> RAMRead {
        let word_address = (effective_address >> 2) << 2;
        let bytes = match self.xlen {
            Xlen::Bit32 => 4,
            Xlen::Bit64 => 8,
        };
        if word_address < DRAM_BASE {
            let mut value_bytes = [0u8; 8];
            for i in 0..bytes {
                value_bytes[i as usize] = self
                    .jolt_device
                    .as_ref()
                    .expect("JoltDevice not set")
                    .load(word_address + i);
            }
            RAMRead {
                address: word_address,
                value: u64::from_le_bytes(value_bytes),
            }
        } else {
            let mut value_bytes = [0u8; 8];
            for i in 0..bytes {
                value_bytes[i as usize] = self.memory.read_byte(word_address + i);
            }
            RAMRead {
                address: word_address,
                value: u64::from_le_bytes(value_bytes),
            }
        }
    }

    /// Records the state of the memory word containing the accessed byte
    /// before and after the store instruction. The memory state is used in Jolt to
    /// construct the witnesses in `read_write_memory.rs`.
    fn trace_store_byte(&mut self, effective_address: u64, value: u64) -> RAMWrite {
        self.assert_effective_store_address(effective_address);
        let bytes = match self.xlen {
            Xlen::Bit32 => 4,
            Xlen::Bit64 => 8,
        };
        let word_address = (effective_address >> 2) << 2;

        let pre_value = if effective_address < DRAM_BASE {
            let mut pre_value_bytes = [0u8; 8];
            for i in 0..bytes {
                pre_value_bytes[i as usize] = self
                    .jolt_device
                    .as_ref()
                    .expect("JoltDevice not set")
                    .load(word_address + i);
            }
            u64::from_le_bytes(pre_value_bytes)
        } else {
            let mut pre_value_bytes = [0u8; 8];
            for i in 0..bytes {
                pre_value_bytes[i as usize] = self.memory.read_byte(word_address + i);
            }
            u64::from_le_bytes(pre_value_bytes)
        };

        // Mask the value into the word
        let post_value = match effective_address % 4 {
            0 => value | (pre_value & 0xffffff00),
            1 => (value << 8) | (pre_value & 0xffff00ff),
            2 => (value << 16) | (pre_value & 0xff00ffff),
            3 => (value << 24) | (pre_value & 0x00ffffff),
            _ => unreachable!(),
        };

        RAMWrite {
            address: word_address,
            pre_value,
            post_value,
        }
    }

    /// Records the state of the memory word containing the accessed halfword
    /// before and after the store instruction. The memory state is used in Jolt to
    /// construct the witnesses in `read_write_memory.rs`.
    fn trace_store_halfword(&mut self, effective_address: u64, value: u64) -> RAMWrite {
        self.assert_effective_store_address(effective_address);
        let bytes = match self.xlen {
            Xlen::Bit32 => 4,
            Xlen::Bit64 => 8,
        };
        let word_address = (effective_address >> 2) << 2;

        let pre_value = if effective_address < DRAM_BASE {
            let mut pre_value_bytes = [0u8; 8];
            for i in 0..bytes {
                pre_value_bytes[i as usize] = self
                    .jolt_device
                    .as_ref()
                    .expect("JoltDevice not set")
                    .load(word_address + i);
            }
            u64::from_le_bytes(pre_value_bytes)
        } else {
            let mut pre_value_bytes = [0u8; 8];
            for i in 0..bytes {
                pre_value_bytes[i as usize] = self.memory.read_byte(word_address + i);
            }
            u64::from_le_bytes(pre_value_bytes)
        };

        // Mask the value into the word
        let post_value = if effective_address % 4 == 2 {
            (value << 16) | (pre_value & 0xffff)
        } else if effective_address.is_multiple_of(4) {
            value | (pre_value & 0xffff0000)
        } else {
            panic!("Unaligned store {effective_address:x}");
        };

        RAMWrite {
            address: word_address,
            pre_value,
            post_value,
        }
    }

    /// Records the state of the accessed memory word before and after the store
    /// instruction. The memory state is used in Jolt to construct the witnesses
    /// in `read_write_memory.rs`.
    fn trace_store(&mut self, effective_address: u64, value: u64) -> RAMWrite {
        self.assert_effective_store_address(effective_address);
        let bytes = match self.xlen {
            Xlen::Bit32 => 4,
            Xlen::Bit64 => 8,
        };

        if effective_address < DRAM_BASE {
            let mut pre_value_bytes = [0u8; 8];
            for i in 0..bytes {
                pre_value_bytes[i as usize] = self
                    .jolt_device
                    .as_ref()
                    .expect("JoltDevice not set")
                    .load(effective_address + i);
            }
            let pre_value = u64::from_le_bytes(pre_value_bytes);
            RAMWrite {
                address: effective_address,
                pre_value,
                post_value: value,
            }
        } else {
            let mut pre_value_bytes = [0u8; 8];
            for i in 0..bytes {
                pre_value_bytes[i as usize] = self.memory.read_byte(effective_address + i);
            }
            let pre_value = u64::from_le_bytes(pre_value_bytes);
            RAMWrite {
                address: effective_address,
                pre_value,
                post_value: value,
            }
        }
    }

    /// Loads two bytes from main memory or peripheral devices depending on
    /// physical address.
    ///
    /// # Arguments
    /// * `p_address` Physical address
    fn load_halfword_raw(&mut self, p_address: u64) -> u16 {
        let effective_address = self.get_effective_address(p_address);
        match effective_address >= DRAM_BASE
            && effective_address.wrapping_add(1) > effective_address
        {
            // Fast path. Directly load main memory at a time.
            true => {
                self.assert_effective_load_address(effective_address);
                self.memory.read_halfword(effective_address)
            }
            false => {
                let mut data = 0_u16;
                for i in 0..2 {
                    data |= (self.load_raw(effective_address.wrapping_add(i)) as u16) << (i * 8)
                }
                data
            }
        }
    }

    /// Loads four bytes from main memory or peripheral devices depending on
    /// physical address.
    ///
    /// # Arguments
    /// * `p_address` Physical address
    pub fn load_word_raw(&mut self, p_address: u64) -> u32 {
        let effective_address = self.get_effective_address(p_address);
        match effective_address >= DRAM_BASE
            && effective_address.wrapping_add(3) > effective_address
        {
            // Fast path. Directly load main memory at a time.
            true => {
                self.assert_effective_load_address(effective_address);
                self.memory.read_word(effective_address)
            }
            false => {
                let mut data = 0_u32;
                for i in 0..4 {
                    data |= (self.load_raw(effective_address.wrapping_add(i)) as u32) << (i * 8)
                }
                data
            }
        }
    }

    /// Loads eight bytes from main memory or peripheral devices depending on
    /// physical address.
    ///
    /// # Arguments
    /// * `p_address` Physical address
    pub fn load_doubleword_raw(&mut self, p_address: u64) -> u64 {
        let effective_address = self.get_effective_address(p_address);
        match effective_address >= DRAM_BASE
            && effective_address.wrapping_add(7) > effective_address
        {
            // Fast path. Directly load main memory at a time.
            true => {
                self.assert_effective_load_address(effective_address);
                self.memory.read_doubleword(effective_address)
            }
            false => {
                let mut data = 0_u64;
                for i in 0..8 {
                    data |= (self.load_raw(effective_address.wrapping_add(i)) as u64) << (i * 8)
                }
                data
            }
        }
    }

    /// Stores a byte to main memory or peripheral devices depending on
    /// physical address.
    ///
    /// # Arguments
    /// * `p_address` Physical address
    /// * `value` data written
    pub fn store_raw(&mut self, p_address: u64, value: u8) {
        let effective_address = self.get_effective_address(p_address);
        // @TODO: Mapping should be configurable with dtb
        match effective_address >= DRAM_BASE {
            true => {
                self.assert_effective_store_address(effective_address);
                self.memory.write_byte(effective_address, value)
            }
            false => match effective_address {
                0x02000000..=0x0200ffff => panic!("store_raw:clint is unsupported."),
                0x0c000000..=0x0fffffff => panic!("store_raw:plic is unsupported."),
                0x10000000..=0x100000ff => panic!("store_raw:UART is unsupported."),
                0x10001000..=0x10001FFF => panic!("store_raw:disk is unsupported."),
                _ => {
                    self.assert_effective_store_address(effective_address);
                    if let Some(jolt_device) = self.jolt_device.as_mut() {
                        return jolt_device.store(effective_address, value);
                    };

                    panic!("Store Failed: Unknown memory mapping {effective_address:X}.");
                }
            },
        };
    }

    pub fn setup_bytecode(&mut self, p_address: u64, value: u8) {
        let effective_address = self.get_effective_address(p_address);

        assert!(
            effective_address >= DRAM_BASE,
            "setup_bytecode: Effective address must be >= DRAM_BASE, got {effective_address:X}."
        );

        if let Some(jolt_device) = self.jolt_device.as_ref() {
            assert!(
                effective_address <= jolt_device.memory_layout.stack_end,
                "setup_bytecode: Effective address must be < stack_end, got {effective_address:X}."
            );
        }

        self.memory.write_byte(effective_address, value)
    }

    /// Stores two bytes to main memory or peripheral devices depending on
    /// physical address.
    ///
    /// # Arguments
    /// * `p_address` Physical address
    /// * `value` data written
    fn store_halfword_raw(&mut self, p_address: u64, value: u16) {
        let effective_address = self.get_effective_address(p_address);
        match effective_address >= DRAM_BASE
            && effective_address.wrapping_add(1) > effective_address
        {
            // Fast path. Directly store to main memory at a time.
            true => {
                self.assert_effective_store_address(effective_address);
                self.memory.write_halfword(effective_address, value)
            }
            false => {
                for i in 0..2 {
                    self.store_raw(
                        effective_address.wrapping_add(i),
                        ((value >> (i * 8)) & 0xff) as u8,
                    );
                }
            }
        }
    }

    /// Stores four bytes to main memory or peripheral devices depending on
    /// physical address.
    ///
    /// # Arguments
    /// * `p_address` Physical address
    /// * `value` data written
    fn store_word_raw(&mut self, p_address: u64, value: u32) {
        let effective_address = self.get_effective_address(p_address);
        match effective_address >= DRAM_BASE
            && effective_address.wrapping_add(3) > effective_address
        {
            // Fast path. Directly store to main memory at a time.
            true => {
                self.assert_effective_store_address(effective_address);
                self.memory.write_word(effective_address, value)
            }
            false => {
                for i in 0..4 {
                    self.store_raw(
                        effective_address.wrapping_add(i),
                        ((value >> (i * 8)) & 0xff) as u8,
                    );
                }
            }
        }
    }

    /// Stores eight bytes to main memory or peripheral devices depending on
    /// physical address.
    ///
    /// # Arguments
    /// * `p_address` Physical address
    /// * `value` data written
    fn store_doubleword_raw(&mut self, p_address: u64, value: u64) {
        let effective_address = self.get_effective_address(p_address);
        match effective_address >= DRAM_BASE
            && effective_address.wrapping_add(7) > effective_address
        {
            // Fast path. Directly store to main memory at a time.
            true => {
                self.assert_effective_store_address(effective_address);
                self.memory.write_doubleword(effective_address, value)
            }
            false => {
                for i in 0..8 {
                    self.store_raw(
                        effective_address.wrapping_add(i),
                        ((value >> (i * 8)) & 0xff) as u8,
                    );
                }
            }
        }
    }

    fn translate_address(
        &mut self,
        v_address: u64,
        access_type: &MemoryAccessType,
    ) -> Result<u64, ()> {
        let address = self.get_effective_address(v_address);
        let p_address = match self.addressing_mode {
            AddressingMode::None => Ok(address),
            AddressingMode::SV32 => match self.privilege_mode {
                // @TODO: Optimize
                PrivilegeMode::Machine => match access_type {
                    MemoryAccessType::Execute => Ok(address),
                    // @TODO: Remove magic number
                    _ => match (self.mstatus >> 17) & 1 {
                        0 => Ok(address),
                        _ => {
                            let privilege_mode = get_privilege_mode((self.mstatus >> 11) & 3);
                            match privilege_mode {
                                PrivilegeMode::Machine => Ok(address),
                                _ => {
                                    let current_privilege_mode = self.privilege_mode.clone();
                                    self.update_privilege_mode(privilege_mode);
                                    let result = self.translate_address(v_address, access_type);
                                    self.update_privilege_mode(current_privilege_mode);
                                    result
                                }
                            }
                        }
                    },
                },
                PrivilegeMode::User | PrivilegeMode::Supervisor => {
                    let vpns = [(address >> 12) & 0x3ff, (address >> 22) & 0x3ff];
                    self.traverse_page(address, 2 - 1, self.ppn, &vpns, access_type)
                }
                _ => Ok(address),
            },
            AddressingMode::SV39 => match self.privilege_mode {
                // @TODO: Optimize
                // @TODO: Remove duplicated code with SV32
                PrivilegeMode::Machine => match access_type {
                    MemoryAccessType::Execute => Ok(address),
                    // @TODO: Remove magic number
                    _ => match (self.mstatus >> 17) & 1 {
                        0 => Ok(address),
                        _ => {
                            let privilege_mode = get_privilege_mode((self.mstatus >> 11) & 3);
                            match privilege_mode {
                                PrivilegeMode::Machine => Ok(address),
                                _ => {
                                    let current_privilege_mode = self.privilege_mode.clone();
                                    self.update_privilege_mode(privilege_mode);
                                    let result = self.translate_address(v_address, access_type);
                                    self.update_privilege_mode(current_privilege_mode);
                                    result
                                }
                            }
                        }
                    },
                },
                PrivilegeMode::User | PrivilegeMode::Supervisor => {
                    let vpns = [
                        (address >> 12) & 0x1ff,
                        (address >> 21) & 0x1ff,
                        (address >> 30) & 0x1ff,
                    ];
                    self.traverse_page(address, 3 - 1, self.ppn, &vpns, access_type)
                }
                _ => Ok(address),
            },
            AddressingMode::SV48 => {
                panic!("AddressingMode SV48 is not supported yet.");
            }
        };
        p_address
    }

    fn traverse_page(
        &mut self,
        v_address: u64,
        level: u8,
        parent_ppn: u64,
        vpns: &[u64],
        access_type: &MemoryAccessType,
    ) -> Result<u64, ()> {
        let pagesize = 4096;
        let ptesize = match self.addressing_mode {
            AddressingMode::SV32 => 4,
            _ => 8,
        };
        let pte_address = parent_ppn * pagesize + vpns[level as usize] * ptesize;
        let pte = match self.addressing_mode {
            AddressingMode::SV32 => self.load_word_raw(pte_address) as u64,
            _ => self.load_doubleword_raw(pte_address),
        };
        let ppn = match self.addressing_mode {
            AddressingMode::SV32 => (pte >> 10) & 0x3fffff,
            _ => (pte >> 10) & 0xfffffffffff,
        };
        let ppns = match self.addressing_mode {
            AddressingMode::SV32 => [(pte >> 10) & 0x3ff, (pte >> 20) & 0xfff, 0 /*dummy*/],
            AddressingMode::SV39 => [
                (pte >> 10) & 0x1ff,
                (pte >> 19) & 0x1ff,
                (pte >> 28) & 0x3ffffff,
            ],
            _ => panic!(), // Shouldn't happen
        };
        let _rsw = (pte >> 8) & 0x3;
        let d = (pte >> 7) & 1;
        let a = (pte >> 6) & 1;
        let _g = (pte >> 5) & 1;
        let _u = (pte >> 4) & 1;
        let x = (pte >> 3) & 1;
        let w = (pte >> 2) & 1;
        let r = (pte >> 1) & 1;
        let v = pte & 1;

        // println!("VA:{:X} Level:{:X} PTE_AD:{:X} PTE:{:X} PPPN:{:X} PPN:{:X} PPN1:{:X} PPN0:{:X}", v_address, level, pte_address, pte, parent_ppn, ppn, ppns[1], ppns[0]);

        if v == 0 || (r == 0 && w == 1) {
            return Err(());
        }

        if r == 0 && x == 0 {
            return match level {
                0 => Err(()),
                _ => self.traverse_page(v_address, level - 1, ppn, vpns, access_type),
            };
        }

        // Leaf page found

        if a == 0
            || (match access_type {
                MemoryAccessType::Write => d == 0,
                _ => false,
            })
        {
            let new_pte = pte
                | (1 << 6)
                | (match access_type {
                    MemoryAccessType::Write => 1 << 7,
                    _ => 0,
                });
            match self.addressing_mode {
                AddressingMode::SV32 => self.store_word_raw(pte_address, new_pte as u32),
                _ => self.store_doubleword_raw(pte_address, new_pte),
            };
        }

        match access_type {
            MemoryAccessType::Execute => {
                if x == 0 {
                    return Err(());
                }
            }
            MemoryAccessType::Read => {
                if r == 0 {
                    return Err(());
                }
            }
            MemoryAccessType::Write => {
                if w == 0 {
                    return Err(());
                }
            }
        };

        let offset = v_address & 0xfff; // [11:0]
                                        // @TODO: Optimize
        let p_address = match self.addressing_mode {
            AddressingMode::SV32 => match level {
                1 => {
                    if ppns[0] != 0 {
                        return Err(());
                    }
                    (ppns[1] << 22) | (vpns[0] << 12) | offset
                }
                0 => (ppn << 12) | offset,
                _ => panic!(), // Shouldn't happen
            },
            _ => match level {
                2 => {
                    if ppns[1] != 0 || ppns[0] != 0 {
                        return Err(());
                    }
                    (ppns[2] << 30) | (vpns[1] << 21) | (vpns[0] << 12) | offset
                }
                1 => {
                    if ppns[0] != 0 {
                        return Err(());
                    }
                    (ppns[2] << 30) | (ppns[1] << 21) | (vpns[0] << 12) | offset
                }
                0 => (ppn << 12) | offset,
                _ => panic!(), // Shouldn't happen
            },
        };

        // println!("PA:{:X}", p_address);
        Ok(p_address)
    }
}

/// [`Memory`](../memory/struct.Memory.html) wrapper. Converts physical address to the one in memory
/// using [`DRAM_BASE`](constant.DRAM_BASE.html) and accesses [`Memory`](../memory/struct.Memory.html).
#[derive(Clone, Debug)]
pub struct MemoryWrapper {
    pub memory: Memory,
}

impl MemoryWrapper {
    fn new() -> Self {
        MemoryWrapper {
            memory: Memory::default(),
        }
    }

    fn init(&mut self, capacity: u64) {
        self.memory.init(capacity);
    }

    pub fn read_byte(&self, p_address: u64) -> u8 {
        debug_assert!(
            p_address >= DRAM_BASE,
            "Memory address must equals to or bigger than DRAM_BASE. {p_address:X}"
        );

        self.memory.read_byte(p_address - DRAM_BASE)
    }

    pub fn read_halfword(&mut self, p_address: u64) -> u16 {
        debug_assert!(
            p_address >= DRAM_BASE && p_address.wrapping_add(1) >= DRAM_BASE,
            "Memory address must equals to or bigger than DRAM_BASE. {p_address:X}"
        );

        self.memory.read_halfword(p_address - DRAM_BASE)
    }

    pub fn read_word(&mut self, p_address: u64) -> u32 {
        debug_assert!(
            p_address >= DRAM_BASE && p_address.wrapping_add(3) >= DRAM_BASE,
            "Memory address must equals to or bigger than DRAM_BASE. {p_address:X}"
        );

        self.memory.read_word(p_address - DRAM_BASE)
    }

    pub fn read_doubleword(&mut self, p_address: u64) -> u64 {
        debug_assert!(
            p_address >= DRAM_BASE && p_address.wrapping_add(7) >= DRAM_BASE,
            "Memory address must equals to or bigger than DRAM_BASE. {p_address:X}"
        );

        self.memory.read_doubleword(p_address - DRAM_BASE)
    }

    pub fn write_byte(&mut self, p_address: u64, value: u8) {
        debug_assert!(
            p_address >= DRAM_BASE,
            "Memory address must equals to or bigger than DRAM_BASE. {p_address:X}"
        );

        self.memory.write_byte(p_address - DRAM_BASE, value);
    }

    pub fn write_halfword(&mut self, p_address: u64, value: u16) {
        debug_assert!(
            p_address >= DRAM_BASE && p_address.wrapping_add(1) >= DRAM_BASE,
            "Memory address must equals to or bigger than DRAM_BASE. {p_address:X}"
        );

        self.memory.write_halfword(p_address - DRAM_BASE, value);
    }

    pub fn write_word(&mut self, p_address: u64, value: u32) {
        debug_assert!(
            p_address >= DRAM_BASE && p_address.wrapping_add(3) >= DRAM_BASE,
            "Memory address must equals to or bigger than DRAM_BASE. {p_address:X}"
        );

        self.memory.write_word(p_address - DRAM_BASE, value);
    }

    pub fn write_doubleword(&mut self, p_address: u64, value: u64) {
        debug_assert!(
            p_address >= DRAM_BASE && p_address.wrapping_add(7) >= DRAM_BASE,
            "Memory address must equals to or bigger than DRAM_BASE. {p_address:X}"
        );

        self.memory.write_doubleword(p_address - DRAM_BASE, value);
    }

    pub fn validate_address(&self, address: u64) -> bool {
        self.memory.validate_address(address - DRAM_BASE)
    }
}

#[cfg(test)]
mod test_mmu {
    use super::*;
    use crate::emulator::terminal::DummyTerminal;
    use common::constants::DEFAULT_MEMORY_SIZE;
    use common::jolt_device::MemoryConfig;

    fn setup_mmu() -> Mmu {
        let terminal = Box::new(DummyTerminal::default());
        let mut mmu = Mmu::new(Xlen::Bit64, terminal);
        let memory_config = MemoryConfig {
            program_size: Some(1024),
            ..Default::default()
        };
        mmu.jolt_device = Some(JoltDevice::new(&memory_config));
        mmu.init_memory(DEFAULT_MEMORY_SIZE);

        mmu
    }

    #[test]
    #[should_panic(expected = "Heap overflow")]
    fn test_heap_overflow() {
        let mut mmu = setup_mmu();

        // Try to write beyond the allocated memory
        let overflow_address = mmu.jolt_device.as_ref().unwrap().memory_layout.memory_end + 1;
        mmu.trace_store(overflow_address, 0xc50513);
    }

    #[test]
    #[should_panic(expected = "Stack Canary")]
    fn test_stack_overflow() {
        let mut mmu = setup_mmu();

        let invalid_address = mmu.jolt_device.as_ref().unwrap().memory_layout.stack_end + 1;
        mmu.trace_store(invalid_address, 0xc50513);
    }

    #[test]
    #[should_panic(expected = "I/O underflow")]
    fn test_io_underflow() {
        let mut mmu = setup_mmu();
        let trusted_advice_size = mmu
            .jolt_device
            .as_ref()
            .unwrap()
            .memory_layout
            .max_trusted_advice_size;
        let untrusted_advice_size = mmu
            .jolt_device
            .as_ref()
            .unwrap()
            .memory_layout
            .max_untrusted_advice_size;
        let invalid_addr = mmu.jolt_device.as_ref().unwrap().memory_layout.input_start
            - 1
            - trusted_advice_size
            - untrusted_advice_size;
        // illegal write to inputs
        mmu.store_bytes(invalid_addr, 0xc50513, 2).unwrap();
    }

    #[test]
    #[should_panic(expected = "I/O overflow")]
    fn test_io_overflow() {
        let mut mmu = setup_mmu();
        let invalid_addr = mmu.jolt_device.as_ref().unwrap().memory_layout.io_end + 1;
        // illegal write to inputs
        mmu.store_bytes(invalid_addr, 0xc50513, 2).unwrap();
    }

    #[test]
    fn test_mprv_uses_mpp_machine_fast_path() {
        let mut mmu = setup_mmu();

        mmu.update_addressing_mode(AddressingMode::SV39);
        mmu.update_privilege_mode(PrivilegeMode::Machine);

        let mprv_bit: u64 = 1 << 17;
        let mpp_machine: u64 = (get_privilege_mode(3) as u64) << 11;
        mmu.update_mstatus(mprv_bit | mpp_machine);

        let v_address = DRAM_BASE;
        let result = mmu.translate_address(v_address, &MemoryAccessType::Read);

        assert_eq!(result, Ok(v_address));
    }
}
