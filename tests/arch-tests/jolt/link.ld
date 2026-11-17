/* ACT4 linker script for Jolt's RV64IMAC DUT.
 *
 * The `tohost` symbol need only be reachable by the emulator's symbol lookup
 * (see tracer/src/emulator/mod.rs::setup_program); the pass/fail signal comes
 * from the HTIF write executed inside RVMODEL_HALT_PASS/FAIL.
 *
 * Memory base (0x80000000) and size (10 MiB) match the Jolt emulator's
 * TEST_MEMORY_CAPACITY / RAM_START_ADDRESS.
 */

OUTPUT_ARCH( "riscv" )
ENTRY(rvtest_entry_point)

MEMORY {
  program (rwx) : ORIGIN = 0x80000000, LENGTH = 0xA00000  /* 10 MiB */
}

SECTIONS {
  /* ACT4 places the test entry point inside .text.init. */
  .text.init : {
    *(.text.init)
  } > program

  /* ACT4 test body — one section per test case. */
  .text.rvtest : {
    *(.text.rvtest)
  } > program

  /* Legacy .text.boot (no-op under ACT4) kept for compatibility with any
   * upstream test that still emits it. */
  .text.boot : {
    *(.text.boot)
  } > program

  /* Standard .text goes after the ACT4-required sections so rvtest_entry_point
   * in .text.init remains the ELF entry. */
  .text : {
    *(.text)
  } > program

  /* ACT4's RVMODEL_* routines live in .text.rvmodel. The Sail signature build
   * expands them from sail_macros.h and the self-checking build from
   * rvmodel_macros.h, so the section's size differs between the two builds,
   * while the expected values baked into the second assume .data sits where it
   * did in the first. Upstream places .text.rvmodel after .data; Jolt keeps all
   * code ahead of the writable sections, so the section instead fills a fixed
   * 4 KiB slot that pins every later address. */
  .text.rvmodel : ALIGN(0x1000) {
    *(.text.rvmodel)
    . = ALIGN(0x1000);
  } > program
  ASSERT(SIZEOF(.text.rvmodel) <= 0x1000, ".text.rvmodel outgrew its 4 KiB slot")

  /* tohost/fromhost live in .tohost (see RVMODEL_DATA_SECTION). The emulator
   * locates the `tohost` symbol via the ELF symbol table, so the section name
   * itself is not load-bearing — but we keep it separate so it's easy to
   * inspect in objdump. It follows the last code section: Jolt protects the
   * code region, every doubleword from the first instruction through the last,
   * and ELF admission rejects a writable section inside it. */
  .tohost : {
    KEEP(*(.tohost))
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
