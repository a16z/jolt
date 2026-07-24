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
        let mut symbol_table_section_headers = vec![];
        let mut string_table_section_headers = vec![];

        for header in &section_headers {
            match header.sh_type {
                // SHT_PROGBITS (1): .text, .data, .rodata, .got, etc.
                // SHT_INIT_ARRAY (14): .init_array - constructor function pointers
                // SHT_FINI_ARRAY (15): .fini_array - destructor function pointers
                // SHT_PREINIT_ARRAY (16): .preinit_array - early constructor pointers
                1 | 14 | 15 | 16 => program_data_section_headers.push(header),
                2 => symbol_table_section_headers.push(header),
                3 => string_table_section_headers.push(header),
                _ => {}
            };
        }

        // AZ: It seems that string and symbol tables are not being used. I expected them to be loaded
        // in the CPU memory just like the program data sections.

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

        assert_eq!(header.e_width, 64, "tracer only supports RV64 ELF inputs");

        if self.tohost_addr != 0 {
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
#[expect(clippy::unwrap_used, reason = "test-only assertions")]
mod tests {
    use super::elf_analyzer::test_elf::{build_elf64, TestSymbol};
    use super::*;
    use crate::emulator::default_terminal::DefaultTerminal;

    const TOHOST_ADDR: u64 = 0x8000_3000;

    fn tohost_symbols() -> Vec<TestSymbol> {
        vec![
            TestSymbol {
                name: "_start",
                value: 0x8000_0000,
                info: 0x12,
                size: 24,
            },
            TestSymbol {
                name: "tohost",
                value: TOHOST_ADDR,
                info: 0x10,
                size: 8,
            },
        ]
    }

    /// A guest that writes `value` to `tohost` and then spins:
    ///     addi x5, x0, <value> ; lui/slli/srli builds x6 = 0x80003000 ;
    ///     sd x5, 0(x6) ; j .
    fn tohost_program(value: u32) -> Vec<u32> {
        assert!(value < 2048);
        vec![
            (value << 20) | (5 << 7) | 0x13, // addi x5, x0, value
            0x8000_3337,                     // lui  x6, 0x80003
            0x0203_1313,                     // slli x6, x6, 32
            0x0203_5313,                     // srli x6, x6, 32
            0x0053_3023,                     // sd   x5, 0(x6)
            0x0000_006f,                     // jal  x0, 0
        ]
    }

    fn emulator_with(text: &[u32], symbols: &[TestSymbol]) -> Emulator {
        let elf = build_elf64(text, symbols);
        let mut emulator = Emulator::new(Box::new(DefaultTerminal::default()));
        emulator.setup_program(&elf);
        emulator
    }

    #[test]
    fn setup_program_loads_text_finds_symbols_and_sets_the_entry_point() {
        let emulator = emulator_with(&tohost_program(5), &tohost_symbols());
        assert_eq!(emulator.get_cpu().read_pc(), 0x8000_0000);
        assert_eq!(emulator.tohost_addr, TOHOST_ADDR);
        assert_eq!(
            emulator.get_address_of_symbol(&"_start".to_string()),
            Some(0x8000_0000)
        );
        assert_eq!(
            emulator.get_address_of_symbol(&"nonexistent".to_string()),
            None
        );
    }

    #[test]
    #[should_panic(expected = "does not seem ELF")]
    fn setup_program_rejects_non_elf_content() {
        let mut emulator = Emulator::new(Box::new(DefaultTerminal::default()));
        emulator.setup_program(b"definitely not an elf");
    }

    #[test]
    fn run_test_extracts_the_htif_endcode_from_the_tohost_write() {
        // HTIF: payload LSB set means done, endcode = payload >> 1.
        for (tohost_value, endcode) in [(1_u32, 0_u64), (5, 2)] {
            let mut emulator = emulator_with(&tohost_program(tohost_value), &tohost_symbols());
            assert_eq!(
                emulator.run_test(false, false),
                endcode,
                "tohost={tohost_value}"
            );
        }
    }

    #[test]
    fn run_test_treats_a_pc_stall_as_clean_exit() {
        // addi x1, x0, 1 ; j .  — never writes tohost
        let mut emulator = emulator_with(&[0x0010_0093, 0x0000_006f], &tohost_symbols());
        assert_eq!(emulator.run_test(true, false), 0);
        assert_eq!(emulator.get_cpu().read_register(1), 1);
    }

    #[test]
    fn write_signature_emits_big_endian_lines_padded_to_granularity() {
        let mut symbols = tohost_symbols();
        symbols.push(TestSymbol {
            name: "begin_signature",
            value: 0x8000_2000,
            info: 0x10,
            size: 0,
        });
        symbols.push(TestSymbol {
            name: "end_signature",
            value: 0x8000_200c,
            info: 0x10,
            size: 0,
        });
        let mut emulator = emulator_with(&[0x0000_006f], &symbols);

        let signature: [u8; 12] = [
            0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc,
        ];
        for (i, byte) in signature.iter().enumerate() {
            emulator
                .get_mut_cpu()
                .get_mut_mmu()
                .store_raw(0x8000_2000 + i as u64, *byte);
        }

        let mut output = Vec::new();
        emulator.write_signature(&mut output, 4).unwrap();
        assert_eq!(
            String::from_utf8(output).unwrap(),
            "44332211\n88776655\nccbbaa99\n"
        );

        // The 12-byte region does not divide the granularity: the tail line
        // is zero-padded beyond end_signature.
        let mut output = Vec::new();
        emulator.write_signature(&mut output, 8).unwrap();
        assert_eq!(
            String::from_utf8(output).unwrap(),
            "8877665544332211\n00000000ccbbaa99\n"
        );
    }

    #[test]
    fn write_signature_is_a_no_op_without_signature_symbols() {
        let mut emulator = emulator_with(&[0x0000_006f], &tohost_symbols());
        let mut output = Vec::new();
        emulator.write_signature(&mut output, 4).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn advice_tape_accessors_move_the_tape_in_and_out() {
        let mut emulator = Emulator::new(Box::new(DefaultTerminal::default()));
        let mut tape = cpu::AdviceTape::new();
        tape.write(&[1, 2, 3]);
        emulator.set_advice_tape(tape);
        assert_eq!(emulator.get_advice_tape().len(), 3);
        emulator.get_mut_advice_tape().write(&[4]);
        assert_eq!(emulator.get_advice_tape().remaining(), 4);

        let taken = emulator.take_advice_tape();
        assert_eq!(taken.len(), 4);
        assert!(emulator.get_advice_tape().is_empty());
    }

    #[test]
    fn saved_state_keeps_symbols_but_drops_memory() {
        let mut emulator = emulator_with(&tohost_program(1), &tohost_symbols());
        let saved = emulator.save_state_with_empty_memory();
        assert_eq!(saved.tohost_addr, TOHOST_ADDR);
        assert_eq!(
            saved.get_cpu().mmu.memory.memory.data.get_num_doublewords(),
            0
        );
        // get_mut_emulator is an identity accessor over the state alias
        let emulator_ref = get_mut_emulator(&mut emulator);
        assert_eq!(emulator_ref.tohost_addr, TOHOST_ADDR);
    }
}
