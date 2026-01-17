/* Generated linker script for Jolt platform */

OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY {
  RAM : ORIGIN = 0x80000000, LENGTH = 0x7a12000
}

PHDRS {
  text PT_LOAD FLAGS(5);        /* R-X */
  rodata PT_LOAD FLAGS(4);      /* R-- */
  data PT_LOAD FLAGS(6);        /* RW- (TLS sections will be in this LOAD) */
  tls PT_TLS;                   /* TLS metadata (references sections in data LOAD) */
}

SECTIONS {
  . = 0x80000000;
  __ehdr_start = .;

  .text : {
    *(.text.boot)
    *(.text._start)
    *(.text .text.*)
  } > RAM : text

  .rodata : {
    *(.rodata .rodata.*)
  } > RAM : rodata

  /* Constructor/destructor arrays (used by musl's __libc_start_init)
   * GNU convention: these come BEFORE .data section
   * Even if empty, these sections must exist for ELF System V ABI compliance.
   * Musl iterates __init_array_start to __init_array_end before calling main.
   * Populated by: C++ constructors, __attribute__((constructor)), #[ctor] crate */
  .init_array : {
    PROVIDE_HIDDEN(__init_array_start = .);
    KEEP(*(SORT_BY_INIT_PRIORITY(.init_array.*)))
    KEEP(*(.init_array))
    PROVIDE_HIDDEN(__init_array_end = .);
  } > RAM : data

  .fini_array : {
    PROVIDE_HIDDEN(__fini_array_start = .);
    KEEP(*(SORT_BY_INIT_PRIORITY(.fini_array.*)))
    KEEP(*(.fini_array))
    PROVIDE_HIDDEN(__fini_array_end = .);
  } > RAM : data

  /* Start data LOAD segment - TLS sections will be included here */
  .data : {
    *(.data .data.*)
    *(.sdata .sdata.*)
    . = ALIGN(8);
  } > RAM : data

  /* TLS sections - assigned to both data LOAD (for loading) and tls (for PT_TLS) */
  .tdata : ALIGN(16) {
    PROVIDE_HIDDEN(__tdata_start = .);
    *(.tdata .tdata.*)
    /* Force non-empty TLS section with 8-byte dummy data */
    . = . + 8;
    PROVIDE_HIDDEN(__tdata_end = .);
  } > RAM : data : tls

  .tbss : ALIGN(16) {
    PROVIDE_HIDDEN(__tbss_start = .);
    *(.tbss .tbss.*)
    *(.tcommon)
    /* Force non-empty TLS section with 8-byte dummy space */
    . = . + 8;
    PROVIDE_HIDDEN(__tbss_end = .);
  } > RAM : data : tls

  .bss : {
    PROVIDE(__bss_start = .);
    *(.bss .bss.*)
    *(COMMON)
    *(.sbss .sbss.*)
    . = ALIGN(8);
    PROVIDE(__bss_end = .);

    /* HTIF sections must be in writable memory */
    KEEP(*(.tohost));
    . = ALIGN(8);
    PROVIDE_HIDDEN(tohost = . - 8);
    KEEP(*(.fromhost));
    . = ALIGN(8);
    PROVIDE_HIDDEN(fromhost = . - 8);
  } > RAM : data

  /* Heap section - must be PROGBITS not NOLOAD so it's loaded */
  .heap : ALIGN(4096) {
    PROVIDE(__heap_start = .);
    /* _HEAP_PTR is used by jolt-platform's bump allocator */
    PROVIDE(_HEAP_PTR = .);
    . = . + 0x3d09000;
    PROVIDE(__heap_end = .);
  } > RAM : data

  /* Stack section - must be PROGBITS not NOLOAD so it's loaded */
  .stack : ALIGN(16) {
    PROVIDE(__stack_bottom = .);
    . = . + 0x1d09000;
    PROVIDE(__stack_top = .);
    /* _STACK_PTR is used by jolt-sdk's no-std boot code */
    PROVIDE(_STACK_PTR = .);
  } > RAM : data

  /DISCARD/ : {
    *(.eh_frame)
    *(.comment)
  }
}
