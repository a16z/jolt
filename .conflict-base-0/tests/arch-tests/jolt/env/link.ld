OUTPUT_ARCH( "riscv" )
ENTRY(rvtest_entry_point)

MEMORY {
  program (rwx) : ORIGIN = 0x80000000, LENGTH = 0xA00000  /* 10MB of memory (DEFAULT_MEMORY_SIZE) */
}

SECTIONS {
  .text.boot : {
    *(.text.boot)
  } > program

  .text.init : {
    *(.text.init)
  } > program

  .tohost : {
    KEEP(*(.tohost))  /* Needed by RISCOF pass/fail detection */
  } > program

  .text : {
    *(.text)
  } > program

  .data : {
    *(.data)
  } > program

  .data.string : {
    *(.data.string)
  } > program

  .bss : {
    *(.bss)
  } > program

  . = ALIGN(8);
  /* Reserve 4 KiB for stack */
  . = . + 4096;
  _STACK_PTR = .;

  . = ALIGN(8);
  /* Reserve 4 KiB for heap */
  _HEAP_PTR = .;
  _end = .;
}
