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
pub mod host_io;
pub mod memory;
pub mod mmu;
pub mod terminal;

use self::cpu::{Cpu, HaltKind};
use self::elf_analyzer::ElfAnalyzer;
use self::terminal::Terminal;

use common::constants::{DEFAULT_HEAP_SIZE, RAM_START_ADDRESS};
use jolt_program::execution::TraceError;
use jolt_program::preprocess::ram::code_word_intervals;
#[cfg(not(feature = "field-inline"))]
use jolt_riscv::RV64IMAC_JOLT_ALL_INLINES;
#[cfg(feature = "field-inline")]
use jolt_riscv::RV64IMAC_JOLT_FIELD_INLINE;
use std::io::Write;
use std::path::Path;

/// The endcode [`Emulator::run_test`] returns when a test guest stalls after
/// writing a value to `tohost` that is not an HTIF exit command.
const UNRECOGNIZED_TOHOST_ENDCODE: u64 = 1;

/// The endcode [`Emulator::run_test`] returns for a run that halts at
/// `ebreak`, Jolt's failure halt.
const PANIC_HALT_ENDCODE: u64 = 1;

/// The endcode [`Emulator::run_test`] returns when an ELF without a `tohost`
/// symbol (a Jolt guest) stops at a PC stall instead of a halt instruction.
const ENDED_WITHOUT_HALT_ENDCODE: u64 = 1;

/// Returns `payload >> 1` only for an HTIF syscall-proxy exit command:
/// device (bits 63..56) and command (bits 55..48) must both be zero, and
/// bit 0 of the 48-bit payload must be set. Other words have no endcode.
pub fn decode_htif_exit(value: u64) -> Option<u64> {
    let device = (value >> 56) & 0xFF;
    let command = (value >> 48) & 0xFF;
    let payload = value & 0xFFFF_FFFF_FFFF;
    (device == 0 && command == 0 && payload & 1 == 1).then_some(payload >> 1)
}

#[derive(Clone, Debug)]
pub struct Emulator {
    pub elf_path: Option<std::path::PathBuf>,

    cpu: Cpu,

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
        Self {
            cpu: Cpu::new(terminal),

            symbol_map: FnvHashMap::default(),
            elf_path: None,

            is_test: false,
            tohost_addr: 0, // assuming tohost_addr is non-zero if exists
            begin_signature_addr: 0,
            end_signature_addr: 0,
        }
    }

    pub fn set_advice_tape(&mut self, tape: cpu::AdviceTape) {
        self.cpu.advice_tape = tape;
    }

    /// Take ownership of the advice tape, replacing it with an empty one
    pub fn take_advice_tape(&mut self) -> cpu::AdviceTape {
        std::mem::take(&mut self.cpu.advice_tape)
    }

    /// Method for running [`riscv-tests`](https://github.com/riscv/riscv-tests) program.
    ///
    /// Returns the termination code:
    /// * `0` for a Jolt guest that ends at `jolt.success`;
    /// * `PANIC_HALT_ENDCODE` (1) for a guest that halts at `ebreak`, Jolt's
    ///   failure halt;
    /// * `payload >> 1` for an HTIF exit command (device 0, command 0,
    ///   payload bit 0 set): `0` for RVMODEL_HALT_PASS, `1` for ACT4's
    ///   RVMODEL_HALT_FAIL, the failing test number for riscv-tests;
    /// * `0` for a PC stall in a test ELF (one that defines `tohost`) with
    ///   `tohost` untouched;
    /// * `UNRECOGNIZED_TOHOST_ENDCODE` (1) for a PC stall after any other
    ///   non-zero `tohost` write, such as a non-zero command or device byte;
    /// * `ENDED_WITHOUT_HALT_ENDCODE` (1) for a PC stall in an ELF without a
    ///   `tohost` symbol: such an ELF is a Jolt guest, whose run must end at a
    ///   halt (`TraceError::EndedWithoutHalt`).
    ///
    /// Callers typically collapse this to 0/1 for the OS exit status; see
    /// `tracer/src/main.rs`.
    pub fn run_test(&mut self, trace: bool, disassemble: bool) -> u64 {
        #[cfg(feature = "std")]
        tracing::info!("This elf file seems like a riscv-tests elf file. Running in test mode.");
        let mut cycle_count = 0;
        loop {
            if let Some(halt) = self.cpu.halted() {
                return match halt {
                    HaltKind::Success => {
                        tracing::info!(
                            "Program exited successfully (code 0) after {cycle_count} cycles"
                        );
                        0
                    }
                    HaltKind::Panic => {
                        tracing::error!(
                            "Program halted at ebreak (abnormal end) after {cycle_count} cycles"
                        );
                        PANIC_HALT_ENDCODE
                    }
                };
            }

            if disassemble {
                let disas = self.cpu.disassemble_next_instruction();
                println!("core   0: {disas}");
            }

            // A test ELF halts by writing an HTIF exit command to `tohost`,
            // which the check after each tick below catches; it may also spin
            // with `tohost` untouched. A test ELF that spins after writing any
            // other non-zero value there tried to halt with a command this
            // emulator does not recognize, so its outcome is unknown and must
            // not read as a pass. An ELF without `tohost` is a Jolt guest,
            // which must end at a halt instead.
            if self.cpu.has_stalled() {
                if self.tohost_addr == 0 {
                    let pc = self.cpu.read_pc();
                    tracing::error!("{}", TraceError::EndedWithoutHalt { pc });
                    return ENDED_WITHOUT_HALT_ENDCODE;
                }
                let tohost_value = self.cpu.get_mut_mmu().load_doubleword_raw(self.tohost_addr);
                if tohost_value != 0 {
                    tracing::error!(
                        "Program stalled after writing {tohost_value:#x} to tohost, \
                         which is not an HTIF exit command"
                    );
                    return UNRECOGNIZED_TOHOST_ENDCODE;
                }
                tracing::info!("Program exited successfully (code 0) after {cycle_count} cycles");
                return 0;
            }

            let mut traces = if trace { Some(Vec::new()) } else { None };
            self.tick(traces.as_mut());
            cycle_count += 1;

            if self.tohost_addr != 0 {
                let tohost_value = self.cpu.get_mut_mmu().load_doubleword_raw(self.tohost_addr);
                if let Some(endcode) = decode_htif_exit(tohost_value) {
                    match endcode {
                        0 => tracing::info!("Test Passed with {endcode:X}\n"),
                        _ => tracing::error!("Test Failed with {endcode:X}\n"),
                    };
                    return endcode;
                }
            }
        }
    }

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

        self.symbol_map
            .extend(analyzer.read_symbol_map(&header, &section_headers));

        // Find tohost, begin_signature, and end_signature addresses from symbol map since they are all global labels
        self.tohost_addr = self.symbol_map.get("tohost").copied().unwrap_or(0);
        self.begin_signature_addr = self.symbol_map.get("begin_signature").copied().unwrap_or(0);
        self.end_signature_addr = self.symbol_map.get("end_signature").copied().unwrap_or(0);

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

    /// Write-protects every doubleword holding a source instruction of
    /// `elf_contents`: a store that changes one faults. Install the
    /// `JoltDevice` first, since its `lowest_address` anchors the word indices.
    #[expect(
        clippy::expect_used,
        reason = "no error path, like create_emulator on over-long I/O: it panics without a JoltDevice or on a program image jolt-program rejects"
    )]
    pub fn protect_code_words(&mut self, elf_contents: &[u8]) {
        #[cfg(not(feature = "field-inline"))]
        let profile = RV64IMAC_JOLT_ALL_INLINES;
        #[cfg(feature = "field-inline")]
        let profile = RV64IMAC_JOLT_FIELD_INLINE;
        let image = jolt_program::image::decode_elf(elf_contents, profile)
            .expect("program ELF decoding failed");
        let mmu = self.cpu.get_mut_mmu();
        let jolt_device = mmu.jolt_device.as_ref().expect("JoltDevice not set");
        mmu.code_word_intervals = code_word_intervals(
            image.instructions.iter().map(|instruction| {
                let row = instruction.row();
                (row.address as u64, row.is_compressed)
            }),
            jolt_device.memory_layout.get_lowest_address(),
        )
        .expect("decoded code-word intervals are valid")
        .into();
    }

    pub fn get_cpu(&self) -> &Cpu {
        &self.cpu
    }

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
    use super::elf_analyzer::test_elf::{build_elf64, StrtabOrder, TestSymbol};
    use super::*;
    use crate::emulator::default_terminal::DefaultTerminal;
    use jolt_riscv::JOLT_SUCCESS_WORD;

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

    fn emulator_with(text: &[u32], symbols: &[TestSymbol], order: StrtabOrder) -> Emulator {
        let elf = build_elf64(text, symbols, order);
        let mut emulator = Emulator::new(Box::new(DefaultTerminal::default()));
        emulator.setup_program(&elf);
        emulator
    }

    #[test]
    fn setup_program_loads_text_finds_symbols_and_sets_the_entry_point() {
        let emulator = emulator_with(&tohost_program(5), &tohost_symbols(), StrtabOrder::GnuLd);
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

    /// LLD emits `.shstrtab` before `.strtab`. Symbol names must still be
    /// resolved through the symbol table's own `sh_link`, not through the
    /// first `SHT_STRTAB` section.
    #[test]
    fn setup_program_resolves_symbols_through_symtab_sh_link_under_lld_order() {
        let emulator = emulator_with(&tohost_program(5), &tohost_symbols(), StrtabOrder::Lld);
        assert_eq!(emulator.tohost_addr, TOHOST_ADDR);
        assert_eq!(
            emulator.get_address_of_symbol(&"_start".to_string()),
            Some(0x8000_0000)
        );
    }

    fn run(text: &[u32], gp: i64) -> (u64, u64) {
        let mut emulator = emulator_with(text, &tohost_symbols(), StrtabOrder::GnuLd);
        emulator.get_mut_cpu().x[3] = gp;
        let endcode = emulator.run_test(false, false);
        let tohost = emulator
            .get_mut_cpu()
            .get_mut_mmu()
            .load_doubleword_raw(TOHOST_ADDR);
        (endcode, tohost)
    }

    #[test]
    fn run_test_returns_the_htif_exit_code() {
        assert_eq!(run(&tohost_program(1), 0), (0, 1));
        assert_eq!(run(&tohost_program(3), 0), (1, 3));
        assert_eq!(run(&tohost_program(5), 0), (2, 5));
    }

    #[test]
    fn run_test_reports_a_spin_without_a_tohost_write_as_success() {
        assert_eq!(run(&[0x0000_006f], 0), (0, 0)); // jal x0, 0
    }

    #[test]
    fn run_test_reports_halts() {
        assert_eq!(run(&[JOLT_SUCCESS_WORD], 0), (0, 0));
        assert_eq!(run(&[0x0010_0073], 0), (PANIC_HALT_ENDCODE, 0)); // ebreak
        assert_ne!(PANIC_HALT_ENDCODE, 0);
    }

    /// An ELF without `tohost` is a Jolt guest: a spin is not a valid end.
    #[test]
    fn run_test_fails_a_stall_in_an_elf_without_tohost() {
        let symbols = tohost_symbols();
        let without_tohost = &symbols[..1];
        let mut emulator = emulator_with(&[0x0000_006f], without_tohost, StrtabOrder::GnuLd); // j .
        assert_eq!(emulator.tohost_addr, 0);
        assert_eq!(emulator.run_test(false, false), ENDED_WITHOUT_HALT_ENDCODE);
        assert_ne!(ENDED_WITHOUT_HALT_ENDCODE, 0);
    }

    /// Encoding `(gp << 1) | 1` with gp = 0x1cadd5ae9e2ac9c5 sets the HTIF
    /// device byte; with gp = 2^47 it sets command 1 with payload 1. Neither
    /// is an exit command, and a subsequent spin must fail.
    #[test]
    fn run_test_fails_a_spin_after_a_tohost_write_that_is_not_an_exit_command() {
        let halt_with_wide_code = [
            0x0011_9293, // slli x5, x3, 1
            0x0012_e293, // ori  x5, x5, 1
            0x8000_3337, // lui  x6, 0x80003
            0x0203_1313, // slli x6, x6, 32
            0x0203_5313, // srli x6, x6, 32
            0x0053_3023, // sd   x5, 0(x6)
            0x0000_006f, // jal  x0, 0
        ];
        assert_eq!(
            run(&halt_with_wide_code, 0x1cad_d5ae_9e2a_c9c5),
            (UNRECOGNIZED_TOHOST_ENDCODE, 0x395b_ab5d_3c55_938b)
        );
        assert_eq!(
            run(&halt_with_wide_code, 1 << 47),
            (UNRECOGNIZED_TOHOST_ENDCODE, 0x0001_0000_0000_0001)
        );
        assert_ne!(UNRECOGNIZED_TOHOST_ENDCODE, 0);
    }

    #[test]
    #[should_panic(expected = "does not seem ELF")]
    fn setup_program_rejects_non_elf_content() {
        let mut emulator = Emulator::new(Box::new(DefaultTerminal::default()));
        emulator.setup_program(b"definitely not an elf");
    }
}
