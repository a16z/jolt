const TEST_MEMORY_CAPACITY: u64 = DEFAULT_HEAP_SIZE;

const PROGRAM_MEMORY_CAPACITY: u64 = DEFAULT_HEAP_SIZE;

extern crate fnv;

use crate::instruction::Cycle;

#[cfg(feature = "std")]
use self::fnv::FnvHashMap;
#[cfg(not(feature = "std"))]
use alloc::collections::btree_map::BTreeMap as FnvHashMap;

#[cfg(not(feature = "std"))]
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

pub mod cpu;
pub mod decode_cache;
pub mod default_terminal;
pub mod elf_analyzer;
pub mod memory;
pub mod mmu;
pub mod terminal;

use self::cpu::Cpu;
use self::elf_analyzer::ElfAnalyzer;
use self::terminal::Terminal;

use common::constants::{DEFAULT_HEAP_SIZE, RAM_START_ADDRESS};
use std::io::Write;
use std::path::Path;

/// RISC-V emulator. It emulates RISC-V CPU and peripheral devices.
///
/// Sample code to run the emulator.
/// ```ignore
/// // Creates an emulator with arbitrary terminal
/// let mut emulator = Emulator::new(Box::new(DefaultTerminal::new()));
/// // Set up program content binary
/// emulator.setup_program(program_content);
/// // Set up Filesystem content binary
/// emulator.setup_filesystem(fs_content);
/// // Go!
/// emulator.run();
/// ```
#[derive(Clone, Debug)]
pub struct Emulator {
    /// addr2line instance for symbol lookups
    pub elf_path: Option<std::path::PathBuf>,

    cpu: Cpu,

    /// Stores mapping from symbol to virtual address
    symbol_map: FnvHashMap<String, u64>,

    /// [`riscv-tests`](https://github.com/riscv/riscv-tests) program specific
    /// properties. Whether the program set by `setup_program()` is
    /// [`riscv-tests`](https://github.com/riscv/riscv-tests) program.
    is_test: bool,

    /// [`riscv-tests`](https://github.com/riscv/riscv-tests) specific properties.
    /// The address where data will be sent to terminal
    pub tohost_addr: u64,

    /// In RISC-V testing, signatures are memory-stored execution results. They're
    /// used to compare a processor's behavior against a trusted reference model
    /// (like SAIL or Spike) to ensure correct and compliant operation.
    /// The address where the signature region begins
    pub begin_signature_addr: u64,

    /// The address where the signature region ends
    pub end_signature_addr: u64,
}

// type alias EmulatorState to Emulator for now
pub type EmulatorState = Emulator;

// Create a new Emulator from a saved state.
pub fn get_mut_emulator(state: &mut EmulatorState) -> &mut Emulator {
    state
}

impl Emulator {
    /// Creates a new `Emulator`. [`Terminal`](terminal/trait.Terminal.html)
    /// is internally used for transferring input/output data to/from `Emulator`.
    ///
    /// # Arguments
    /// * `terminal`
    pub fn new(terminal: Box<dyn Terminal>) -> Self {
        Self {
            cpu: Cpu::new(terminal),

            symbol_map: FnvHashMap::default(),
            elf_path: None,

            // These can be updated in setup_program()
            is_test: false,
            tohost_addr: 0, // assuming tohost_addr is non-zero if exists
            begin_signature_addr: 0,
            end_signature_addr: 0,
        }
    }

    /// Set the advice tape for this emulator
    pub fn set_advice_tape(&mut self, tape: cpu::AdviceTape) {
        self.cpu.advice_tape = tape;
    }

    /// Get a reference to the advice tape
    pub fn get_advice_tape(&self) -> &cpu::AdviceTape {
        &self.cpu.advice_tape
    }

    /// Get a mutable reference to the advice tape
    pub fn get_mut_advice_tape(&mut self) -> &mut cpu::AdviceTape {
        &mut self.cpu.advice_tape
    }

    /// Take ownership of the advice tape, replacing it with an empty one
    pub fn take_advice_tape(&mut self) -> cpu::AdviceTape {
        std::mem::take(&mut self.cpu.advice_tape)
    }

    /// Method for running [`riscv-tests`](https://github.com/riscv/riscv-tests) program.
    /// The differences from `run_program()` are
    /// * Disassembles every instruction and dumps to terminal
    /// * The emulator stops when the test finishes
    /// * Displays the result message (pass/fail) to terminal
    ///
    /// Returns the HTIF termination code extracted from the `tohost` write:
    /// * `0` — clean exit (RVMODEL_HALT_PASS, or PC-stall termination used by
    ///   Jolt guests that call `jolt_exit()`)
    /// * non-zero — `tohost payload >> 1` from RVMODEL_HALT_FAIL (gp-derived,
    ///   ACT4 uses this for signature-mismatch failures)
    ///
    /// Callers typically collapse this to 0/1 for the OS exit status; see
    /// `tracer/src/main.rs`.
    pub fn run_test(&mut self, trace: bool, disassemble: bool) -> u64 {
        // @TODO: Send this message to terminal?
        #[cfg(feature = "std")]
        tracing::info!("This elf file seems like a riscv-tests elf file. Running in test mode.");
        let mut cycle_count = 0;
        let mut prev_pc: u64 = 0;
        loop {
            // Disassemble and print each instruction if requested (like spike -d)
            if disassemble {
                let disas = self.cpu.disassemble_next_instruction();
                println!("core   0: {disas}");
            }

            // Check for infinite loop termination (PC stall detection)
            // This is used by Jolt guests that terminate via `j .` instruction.
            // The trap handler (in guest_std_boot.rs or guest_no_std_boot.rs) calls
            // jolt_exit() which enters an infinite loop for clean termination.
            let pc = self.cpu.read_pc();
            if prev_pc == pc {
                tracing::info!("Program exited successfully (code 0) after {cycle_count} cycles");
                return 0;
            }
            prev_pc = pc;

            let mut traces = if trace { Some(Vec::new()) } else { None };
            self.tick(traces.as_mut());
            cycle_count += 1;

            // Check if tohost has been written to
            let tohost_value = self.cpu.get_mut_mmu().load_doubleword_raw(self.tohost_addr);
            if tohost_value != 0 {
                // Extract device, cmd and payload from tohost value
                // Format matches sail-riscv's htif_cmd bitfield:
                // device  : 63 .. 56
                // cmd     : 55 .. 48
                // payload : 47 .. 0
                let device = (tohost_value >> 56) & 0xFF;
                let _cmd = (tohost_value >> 48) & 0xFF;
                let payload = tohost_value & 0xFFFFFFFFFFFF;

                // Check if this is a syscall-proxy command (device 0x00)
                // and if the LSB of payload is set (indicating program done)
                if device == 0x00 && (payload & 1) == 1 {
                    // Extract exit code by shifting payload right by 1
                    let endcode = payload >> 1;
                    match endcode {
                        0 => tracing::info!("Test Passed with {endcode:X}\n"),
                        _ => tracing::error!("Test Failed with {endcode:X}\n"),
                    };
                    return endcode;
                }
            }
        }
    }

    /// Runs CPU one cycle
    pub fn tick(&mut self, trace: Option<&mut Vec<Cycle>>) {
        self.cpu.tick(trace)
    }

    /// This enables usage of addr2line to find debug info embedded in the binary
    pub fn set_elf_path(&mut self, elf_path: &Path) {
        if elf_path.exists() {
            self.elf_path = Some(elf_path.to_path_buf());
        }
    }

    /// Sets up program run by the program. This method analyzes the passed content
    /// and configure CPU properly. If the passed contend doesn't seem ELF file,
    /// it panics. This method is expected to be called only once.
    ///
    /// # Arguments
    /// * `data` Program binary
    // @TODO: Make ElfAnalyzer and move the core logic there.
    // @TODO: Returns `Err` if the passed contend doesn't seem ELF file
    pub fn setup_program(&mut self, data: &[u8]) {
        let analyzer = ElfAnalyzer::new(data);

        if !analyzer.validate() {
            panic!("This file does not seem ELF file");
        }

        let header = analyzer.read_header();
        let section_headers = analyzer.read_section_headers(&header);

        let mut program_data_section_headers = vec![];

        for header in &section_headers {
            match header.sh_type {
                // SHT_PROGBITS (1): .text, .data, .rodata, .got, etc.
                // SHT_INIT_ARRAY (14): .init_array - constructor function pointers
                // SHT_FINI_ARRAY (15): .fini_array - destructor function pointers
                // SHT_PREINIT_ARRAY (16): .preinit_array - early constructor pointers
                1 | 14 | 15 | 16 => program_data_section_headers.push(header),
                _ => {}
            };
        }

        // Creates symbol - virtual address mapping
        self.symbol_map
            .extend(analyzer.read_symbol_map(&header, &section_headers));

        // Find tohost, begin_signature, and end_signature addresses from symbol map since they are all global labels
        self.tohost_addr = self.symbol_map.get("tohost").copied().unwrap_or(0);
        self.begin_signature_addr = self.symbol_map.get("begin_signature").copied().unwrap_or(0);
        self.end_signature_addr = self.symbol_map.get("end_signature").copied().unwrap_or(0);

        // Detected whether the elf file is riscv-tests.
        // Setting up CPU and Memory depending on it.

        assert_eq!(header.e_width, 64, "tracer only supports RV64 ELF inputs");

        if self.tohost_addr != 0 {
            // WARNING: a `tohost` symbol is how riscv-tests ELFs are
            // recognized; a Jolt guest built by a foreign toolchain that
            // defines one silently loses the layout-derived memory sizing
            // configured below.
            #[cfg(feature = "std")]
            if self.cpu.get_mut_mmu().jolt_device.is_some() {
                tracing::warn!(
                    "ELF defines a `tohost` symbol, so the tracer is entering riscv-tests mode: \
                    emulator memory is sized to the riscv-tests test capacity instead of the \
                    Jolt memory layout. If this is a Jolt guest, do not emit a `tohost` symbol."
                );
            }
            self.is_test = true;
            self.cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
        } else {
            self.is_test = false;
            let memory_capacity =
                if let Some(jolt_device) = self.cpu.get_mut_mmu().jolt_device.as_ref() {
                    jolt_device.memory_layout.get_total_memory_size()
                } else {
                    PROGRAM_MEMORY_CAPACITY
                };
            self.cpu.get_mut_mmu().init_memory(memory_capacity);
        }

        // Copy program data sections to CPU memory.
        for header in &program_data_section_headers {
            let sh_addr = header.sh_addr;
            let sh_offset = header.sh_offset as usize;
            let sh_size = header.sh_size as usize;
            if sh_addr >= RAM_START_ADDRESS && sh_offset > 0 && sh_size > 0 {
                for j in 0..sh_size {
                    self.cpu
                        .get_mut_mmu()
                        .setup_bytecode(sh_addr + j as u64, analyzer.read_byte(sh_offset + j));
                }
            }
        }

        // Cover the executable sections with the pre-decoded instruction
        // cache. (Initialized after the section copy so the setup stores don't
        // walk the invalidation path.)
        const SHF_EXECINSTR: u64 = 0x4;
        let mut text_base = u64::MAX;
        let mut text_end = 0;
        for header in &program_data_section_headers {
            if header.sh_flags & SHF_EXECINSTR != 0
                && header.sh_addr >= RAM_START_ADDRESS
                && header.sh_size > 0
            {
                text_base = text_base.min(header.sh_addr);
                text_end = text_end.max(header.sh_addr + header.sh_size);
            }
        }
        if text_base < text_end {
            self.cpu
                .get_mut_mmu()
                .init_decode_cache(text_base, text_end);
        }

        self.cpu.update_pc(header.e_entry);
    }

    /// Returns immutable reference to `self.cpu`.
    pub fn get_cpu(&self) -> &Cpu {
        &self.cpu
    }

    /// Returns mutable reference to `self.cpu`.
    pub fn get_mut_cpu(&mut self) -> &mut Cpu {
        &mut self.cpu
    }

    /// Returns a virtual address corresponding to symbol strings
    ///
    /// # Arguments
    /// * `s` Symbol strings
    pub fn get_address_of_symbol(&self, s: &String) -> Option<u64> {
        self.symbol_map.get(s).copied()
    }

    /// Writes the signature region to a writer with specified granularity.
    /// Each word of the signature is written as a hexadecimal string representation.
    ///
    /// # Arguments
    /// * `writer` - Any type that implements Write trait
    /// * `granularity` - Number of bytes to write per line (must be a power of 2)
    ///
    /// # Returns
    /// * `Result<(), std::io::Error>` - Ok if successful, Err if write operations fail
    pub fn write_signature<W: Write>(
        &mut self,
        writer: &mut W,
        granularity: usize,
    ) -> std::io::Result<()> {
        if self.begin_signature_addr == 0 || self.end_signature_addr == 0 {
            return Ok(());
        }

        let sig_len = (self.end_signature_addr - self.begin_signature_addr) as usize;

        for i in (0..sig_len).step_by(granularity) {
            // Write bytes in big-endian order
            for j in (0..granularity).rev() {
                let byte = if i + j < sig_len {
                    self.cpu
                        .get_mut_mmu()
                        .load_raw(self.begin_signature_addr + (i + j) as u64)
                } else {
                    0
                };
                write!(writer, "{byte:02x}")?;
            }
            writeln!(writer)?;
        }

        Ok(())
    }
}

impl Emulator {
    pub fn save_state_with_empty_memory(&self) -> Emulator {
        Emulator {
            elf_path: self.elf_path.clone(),
            cpu: self.cpu.save_state_with_empty_memory(),
            symbol_map: self.symbol_map.clone(),
            is_test: self.is_test,
            tohost_addr: self.tohost_addr,
            begin_signature_addr: self.begin_signature_addr,
            end_signature_addr: self.end_signature_addr,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emulator::terminal::DummyTerminal;

    const TOHOST: u64 = RAM_START_ADDRESS + 0x1000;

    /// Minimal RV64 ELF whose `.shstrtab` precedes `.strtab`, the section
    /// order LLD emits (GNU ld emits `.strtab` first). The symbol table's
    /// `sh_link` names `.strtab`, which is where `tohost` lives.
    fn elf_with_shstrtab_before_strtab() -> Vec<u8> {
        const EHDR: usize = 64;
        const SHDR: usize = 64;
        let shstrtab: &[u8] = b"\0.tohost\0.symtab\0.shstrtab\0.strtab\0";
        let strtab: &[u8] = b"\0tohost\0";
        let mut symtab = vec![0u8; 24]; // null symbol
        let mut sym = [0u8; 24];
        sym[0..4].copy_from_slice(&1u32.to_le_bytes()); // st_name: "tohost"
        sym[4] = 0x10; // STB_GLOBAL | STT_NOTYPE, like an assembler label
        sym[6..8].copy_from_slice(&1u16.to_le_bytes()); // st_shndx: .tohost
        sym[8..16].copy_from_slice(&TOHOST.to_le_bytes()); // st_value
        symtab.extend_from_slice(&sym);
        let tohost_data = [0u8; 8];

        let mut data = Vec::new();
        let mut place = |bytes: &[u8]| {
            let offset = EHDR + data.len();
            data.extend_from_slice(bytes);
            offset as u64
        };
        let tohost_off = place(&tohost_data);
        let symtab_off = place(&symtab);
        let shstrtab_off = place(shstrtab);
        let strtab_off = place(strtab);
        let shoff = EHDR + data.len();

        let mut elf = vec![0u8; EHDR];
        elf[0..4].copy_from_slice(b"\x7fELF");
        elf[4] = 2; // ELFCLASS64
        elf[5] = 1; // little endian
        elf[6] = 1; // EV_CURRENT
        elf[16..18].copy_from_slice(&2u16.to_le_bytes()); // ET_EXEC
        elf[18..20].copy_from_slice(&243u16.to_le_bytes()); // EM_RISCV
        elf[20..24].copy_from_slice(&1u32.to_le_bytes());
        elf[24..32].copy_from_slice(&RAM_START_ADDRESS.to_le_bytes()); // e_entry
        elf[40..48].copy_from_slice(&(shoff as u64).to_le_bytes()); // e_shoff
        elf[52..54].copy_from_slice(&(EHDR as u16).to_le_bytes()); // e_ehsize
        elf[58..60].copy_from_slice(&(SHDR as u16).to_le_bytes()); // e_shentsize
        elf[60..62].copy_from_slice(&5u16.to_le_bytes()); // e_shnum
        elf[62..64].copy_from_slice(&3u16.to_le_bytes()); // e_shstrndx
        elf.extend_from_slice(&data);

        let mut section = |name: u32, ty: u32, addr: u64, offset: u64, size: usize, link: u32| {
            let mut shdr = [0u8; SHDR];
            shdr[0..4].copy_from_slice(&name.to_le_bytes());
            shdr[4..8].copy_from_slice(&ty.to_le_bytes());
            shdr[16..24].copy_from_slice(&addr.to_le_bytes());
            shdr[24..32].copy_from_slice(&offset.to_le_bytes());
            shdr[32..40].copy_from_slice(&(size as u64).to_le_bytes());
            shdr[40..44].copy_from_slice(&link.to_le_bytes());
            elf.extend_from_slice(&shdr);
        };
        section(0, 0, 0, 0, 0, 0); // SHN_UNDEF
        section(1, 1, TOHOST, tohost_off, tohost_data.len(), 0); // [1] .tohost
        section(9, 2, 0, symtab_off, symtab.len(), 4); // [2] .symtab, sh_link -> [4]
        section(17, 3, 0, shstrtab_off, shstrtab.len(), 0); // [3] .shstrtab
        section(27, 3, 0, strtab_off, strtab.len(), 0); // [4] .strtab
        elf
    }

    /// Symbol names must be resolved through the symbol table's own string
    /// table (`sh_link`), not whichever `SHT_STRTAB` section comes first.
    #[test]
    fn resolves_symbols_through_symtab_sh_link() {
        let mut emulator = Emulator::new(Box::new(DummyTerminal::default()));
        emulator.setup_program(&elf_with_shstrtab_before_strtab());
        assert_eq!(
            emulator.get_address_of_symbol(&"tohost".to_string()),
            Some(TOHOST)
        );
        assert_eq!(emulator.tohost_addr, TOHOST);
    }
}
