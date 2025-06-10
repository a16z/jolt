// @TODO: temporal
const TEST_MEMORY_CAPACITY: u64 = 1024 * 512;
const PROGRAM_MEMORY_CAPACITY: u64 = 1024 * 1024 * 128; // big enough to run Linux and xv6

extern crate fnv;

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
pub mod default_terminal;
pub mod device;
pub mod elf_analyzer;
pub mod memory;
pub mod mmu;
pub mod terminal;

use self::cpu::{Cpu, Xlen};
use self::elf_analyzer::ElfAnalyzer;
use self::terminal::Terminal;

use std::io::Write;

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
pub struct Emulator {
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

impl Emulator {
    /// Creates a new `Emulator`. [`Terminal`](terminal/trait.Terminal.html)
    /// is internally used for transferring input/output data to/from `Emulator`.
    ///
    /// # Arguments
    /// * `terminal`
    pub fn new(terminal: Box<dyn Terminal>) -> Self {
        Emulator {
            cpu: Cpu::new(terminal),

            symbol_map: FnvHashMap::default(),

            // These can be updated in setup_program()
            is_test: false,
            tohost_addr: 0, // assuming tohost_addr is non-zero if exists
            begin_signature_addr: 0,
            end_signature_addr: 0,
        }
    }

    /// Runs program set by `setup_program()`. Calls `run_test()` if the program
    /// is [`riscv-tests`](https://github.com/riscv/riscv-tests).
    /// Otherwise calls `run_program()`.
    pub fn run(&mut self) {
        match self.is_test {
            true => self.run_test(),
            false => self.run_program(),
        };
    }

    /// Runs program set by `setup_program()`. The emulator won't stop forever.
    pub fn run_program(&mut self) {
        loop {
            self.tick();
        }
    }

    /// Method for running [`riscv-tests`](https://github.com/riscv/riscv-tests) program.
    /// The differences from `run_program()` are
    /// * Disassembles every instruction and dumps to terminal
    /// * The emulator stops when the test finishes
    /// * Displays the result message (pass/fail) to terminal
    pub fn run_test(&mut self) {
        // @TODO: Send this message to terminal?
        #[cfg(feature = "std")]
        println!("This elf file seems like a riscv-tests elf file. Running in test mode.");
        loop {
            let disas = self.cpu.disassemble_next_instruction();
            self.put_bytes_to_terminal(disas.as_bytes());
            self.put_bytes_to_terminal(&[10]); // new line

            self.tick();

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
                    let exit_code = payload >> 1;
                    if exit_code == 0 {
                        self.put_bytes_to_terminal(b"SUCCESS\n");
                    } else {
                        self.put_bytes_to_terminal(format!("FAILURE: {exit_code}\n").as_bytes());
                    }
                    break;
                }
            }
        }
    }

    /// Helper method. Sends ascii code bytes to terminal.
    ///
    /// # Arguments
    /// * `bytes`
    fn put_bytes_to_terminal(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.cpu.get_mut_terminal().put_byte(*byte);
        }
    }

    /// Runs CPU one cycle
    pub fn tick(&mut self) {
        self.cpu.tick();
    }

    /// Sets up program run by the program. This method analyzes the passed content
    /// and configure CPU properly. If the passed contend doesn't seem ELF file,
    /// it panics. This method is expected to be called only once.
    ///
    /// # Arguments
    /// * `data` Program binary
    // @TODO: Make ElfAnalyzer and move the core logic there.
    // @TODO: Returns `Err` if the passed contend doesn't seem ELF file
    pub fn setup_program(&mut self, data: Vec<u8>) {
        let analyzer = ElfAnalyzer::new(data);

        if !analyzer.validate() {
            panic!("This file does not seem ELF file");
        }

        let header = analyzer.read_header();
        let section_headers = analyzer.read_section_headers(&header);

        let mut program_data_section_headers = vec![];
        let mut symbol_table_section_headers = vec![];
        let mut string_table_section_headers = vec![];

        for header in &section_headers {
            match header.sh_type {
                1 => program_data_section_headers.push(header),
                2 => symbol_table_section_headers.push(header),
                3 => string_table_section_headers.push(header),
                _ => {}
            };
        }

        // Creates symbol - virtual address mapping
        if !string_table_section_headers.is_empty() {
            let entries = analyzer.read_symbol_entries(&header, &symbol_table_section_headers);
            // Assuming symbols are in the first string table section.
            // @TODO: What if symbol can be in the second or later string table sections?
            let map = analyzer.create_symbol_map(&entries, string_table_section_headers[0]);
            for key in map.keys() {
                self.symbol_map
                    .insert(key.to_string(), *map.get(key).unwrap());
            }
        }

        // Find tohost, begin_signature, and end_signature addresses from symbol map since they are all global labels
        self.tohost_addr = self.symbol_map.get("tohost").copied().unwrap_or(0);
        self.begin_signature_addr = self.symbol_map.get("begin_signature").copied().unwrap_or(0);
        self.end_signature_addr = self.symbol_map.get("end_signature").copied().unwrap_or(0);

        // Detected whether the elf file is riscv-tests.
        // Setting up CPU and Memory depending on it.

        self.cpu.update_xlen(match header.e_width {
            32 => Xlen::Bit32,
            64 => Xlen::Bit64,
            _ => panic!("No happen"),
        });

        if self.tohost_addr != 0 {
            self.is_test = true;
            self.cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
        } else {
            self.is_test = false;
            self.cpu.get_mut_mmu().init_memory(PROGRAM_MEMORY_CAPACITY);
        }

        for header in &program_data_section_headers {
            let sh_addr = header.sh_addr;
            let sh_offset = header.sh_offset as usize;
            let sh_size = header.sh_size as usize;
            if sh_addr >= 0x80000000 && sh_offset > 0 && sh_size > 0 {
                for j in 0..sh_size {
                    self.cpu
                        .get_mut_mmu()
                        .store_raw(sh_addr + j as u64, analyzer.read_byte(sh_offset + j));
                }
            }
        }

        self.cpu.update_pc(header.e_entry);
    }

    /// Loads symbols of program and adds them to `symbol_map`.
    ///
    /// # Arguments
    /// * `content` Program binary
    pub fn load_program_for_symbols(&mut self, content: Vec<u8>) {
        let analyzer = ElfAnalyzer::new(content);

        if !analyzer.validate() {
            panic!("This file does not seem ELF file");
        }

        let header = analyzer.read_header();
        let section_headers = analyzer.read_section_headers(&header);

        let mut program_data_section_headers = vec![];
        let mut symbol_table_section_headers = vec![];
        let mut string_table_section_headers = vec![];

        for header in &section_headers {
            match header.sh_type {
                1 => program_data_section_headers.push(header),
                2 => symbol_table_section_headers.push(header),
                3 => string_table_section_headers.push(header),
                _ => {}
            };
        }

        // Creates symbol - virtual address mapping
        if !string_table_section_headers.is_empty() {
            let entries = analyzer.read_symbol_entries(&header, &symbol_table_section_headers);
            // Assuming symbols are in the first string table section.
            // @TODO: What if symbol can be in the second or later string table sections?
            let map = analyzer.create_symbol_map(&entries, string_table_section_headers[0]);
            for key in map.keys() {
                self.symbol_map
                    .insert(key.to_string(), *map.get(key).unwrap());
            }
        }
    }

    /// Sets up filesystem. Use this method if program (e.g. Linux) uses
    /// filesystem. This method is expected to be called up to only once.
    ///
    /// # Arguments
    /// * `content` File system content binary
    pub fn setup_filesystem(&mut self, content: Vec<u8>) {
        self.cpu.get_mut_mmu().init_disk(content);
    }

    /// Sets up device tree. The emulator has default device tree configuration.
    /// If you want to override it, use this method. This method is expected to
    /// to be called up to only once.
    ///
    /// # Arguments
    /// * `content` DTB content binary
    pub fn setup_dtb(&mut self, content: Vec<u8>) {
        self.cpu.get_mut_mmu().init_dtb(content);
    }

    /// Updates XLEN (the width of an integer register in bits) in CPU.
    ///
    /// # Arguments
    /// * `xlen`
    pub fn update_xlen(&mut self, xlen: Xlen) {
        self.cpu.update_xlen(xlen);
    }

    /// Enables or disables page cache optimization.
    /// Page cache optimization is experimental feature.
    /// See [`Mmu`](./mmu/struct.Mmu.html) for the detail.
    ///
    /// # Arguments
    /// * `enabled`
    pub fn enable_page_cache(&mut self, enabled: bool) {
        self.cpu.get_mut_mmu().enable_page_cache(enabled);
    }

    /// Returns mutable reference to `Terminal`.
    pub fn get_mut_terminal(&mut self) -> &mut Box<dyn Terminal> {
        self.cpu.get_mut_terminal()
    }

    /// Returns immutable reference to `Cpu`.
    pub fn get_cpu(&self) -> &Cpu {
        &self.cpu
    }

    /// Returns mutable reference to `Cpu`.
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
    /// The signature is written in little-endian byte order.
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

        for addr in (self.begin_signature_addr..self.end_signature_addr).step_by(granularity) {
            // Load word and write in big-endian order
            let word = self.cpu.get_mut_mmu().load_word_raw(addr);
            write!(writer, "{word:08x}")?;
            writeln!(writer)?;
        }

        Ok(())
    }
}
