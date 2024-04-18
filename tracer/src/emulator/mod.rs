// @TODO: temporal
const TEST_MEMORY_CAPACITY: u64 = 1024 * 512;
const PROGRAM_MEMORY_CAPACITY: u64 = 1024 * 1024 * 128; // big enough to run Linux and xv6

extern crate fnv;

use self::fnv::FnvHashMap;

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

/// RISC-V emulator. It emulates RISC-V CPU and peripheral devices.
///
/// Sample code to run the emulator.
/// ```ignore
/// // Creates an emulator with arbitary terminal
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
    tohost_addr: u64,
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
        println!("This elf file seems riscv-tests elf file. Running in test mode.");
        loop {
            let disas = self.cpu.disassemble_next_instruction();
            self.put_bytes_to_terminal(disas.as_bytes());
            self.put_bytes_to_terminal(&[10]); // new line

            self.tick();

            // It seems in riscv-tests ends with end code
            // written to a certain physical memory address
            // (0x80001000 in mose test cases) so checking
            // the data in the address and terminating the test
            // if non-zero data is written.
            // End code 1 seems to mean pass.
            let endcode = self.cpu.get_mut_mmu().load_word_raw(self.tohost_addr);
            if endcode != 0 {
                match endcode {
                    1 => self.put_bytes_to_terminal(
                        format!("Test Passed with {:X}\n", endcode).as_bytes(),
                    ),
                    _ => self.put_bytes_to_terminal(
                        format!("Test Failed with {:X}\n", endcode).as_bytes(),
                    ),
                };
                break;
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
        //let program_headers = analyzer._read_program_headers(&header);
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

        // Find program data section named .tohost to detect if the elf file is riscv-tests
        self.tohost_addr = analyzer
            .find_tohost_addr(&program_data_section_headers, &string_table_section_headers)
            .unwrap_or(0);

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
}
