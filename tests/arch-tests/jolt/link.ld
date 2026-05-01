/* ACT4 linker script for Jolt's RV64IMAC DUT.
 *
 * Adapted from the prior RISCOF-era tests/arch-tests/jolt/env/link.ld.
 * Differences:
 *   - Adds ACT4's required sections (.text.rvtest, .text.rvmodel).
 *   - Drops the .tohost-specific ordering that was needed by RISCOF's
 *     pass/fail detection; with ACT4 the pass/fail signal comes from the
 *     HTIF write executed inside RVMODEL_HALT_PASS/FAIL, and the `tohost`
 *     symbol need only be reachable by the emulator's symbol lookup (see
 *     tracer/src/emulator/mod.rs::setup_program).
 *   - Memory base (0x80000000) and size (10 MiB) match the Jolt emulator's
 *     TEST_MEMORY_CAPACITY / RAM_START_ADDRESS.
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

  /* tohost/fromhost live in .tohost (see RVMODEL_DATA_SECTION). The emulator
   * locates the `tohost` symbol via the ELF symbol table, so the section name
   * itself is not load-bearing — but we keep it separate so it's easy to
   * inspect in objdump. */
  .tohost : {
    KEEP(*(.tohost))
  } > program

  /* ACT4's RVMODEL_HALT_PASS / RVMODEL_HALT_FAIL expansion lives in
   * .text.rvmodel. */
  .text.rvmodel : {
    *(.text.rvmodel)
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
