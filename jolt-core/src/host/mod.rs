#![allow(clippy::type_complexity)]

use std::path::PathBuf;

#[cfg(feature = "host")]
pub mod analyze;
#[cfg(feature = "host")]
pub mod program;
#[cfg(all(feature = "host", not(target_arch = "wasm32")))]
pub mod toolchain;

pub const TOOLCHAIN_VERSION: &str = "1.89.0";

#[derive(Clone)]
pub struct Program {
    guest: String,
    func: Option<String>,
    memory_size: u64,
    stack_size: u64,
    max_input_size: u64,
    max_untrusted_advice_size: u64,
    max_trusted_advice_size: u64,
    max_output_size: u64,
    std: bool,
    guest_features: Vec<String>,
    pub elf: Option<PathBuf>,
}

pub const DEFAULT_TARGET_DIR: &str = "/tmp/jolt-guest-targets";

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

  .bss (NOLOAD) : {
    __bss_start = .;
    *(.bss .bss.*)
    *(COMMON)
    . = ALIGN(8);
    __bss_end = .;
  } > program

  /DISCARD/ : {
    *(.eh_frame*)
    *(.eh_frame_hdr*)
  }

  . = ALIGN(8);
  _STACK_END = .;
  . = . + {STACK_CANARY};
  . = . + {STACK_SIZE};
  . = ALIGN(8);
  _STACK_PTR = .;

  . = ALIGN(8);
  _HEAP_PTR = .;
}
"#;
