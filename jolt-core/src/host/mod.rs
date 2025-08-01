#![allow(clippy::type_complexity)]

use std::path::PathBuf;

#[cfg(feature = "prover")]
pub mod analyze;
#[cfg(feature = "prover")]
pub mod program;
#[cfg(feature = "prover")]
pub mod toolchain;

pub const DEFAULT_TARGET_DIR: &str = "/tmp/jolt-guest-targets";

#[derive(Clone)]
pub struct Program {
    guest: String,
    func: Option<String>,
    memory_size: u64,
    stack_size: u64,
    max_input_size: u64,
    max_output_size: u64,
    std: bool,
    pub elf: Option<PathBuf>,
}

const LINKER_SCRIPT_TEMPLATE: &str = r#"
MEMORY {
  program (rwx) : ORIGIN = 0x80000000, LENGTH = {EMULATOR_MEMORY}
}

SECTIONS {
  .text.boot : {
    *(.text.boot)
  } > program

  .text : {
    *(.text)
  } > program

  .data : {
    *(.data)
  } > program

  .rodata : {
    *(.rodata)
  } > program

  .sbss : {
    *(.sbss)
  } > program

  .bss : {
    *(.bss)
  } > program

  . = ALIGN(8);
  _STACK_END = .;
  . = . + {STACK_CANARY};
  . = . + {STACK_SIZE};
  _STACK_PTR = .;

  . = ALIGN(8);
  _HEAP_PTR = .;
}
"#;
