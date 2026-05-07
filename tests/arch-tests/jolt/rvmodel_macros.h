#ifndef _RVMODEL_MACROS_H
#define _RVMODEL_MACROS_H

// ACT4 model macros for Jolt.
//
// ACT4's self-checking ELFs compare their observed signature against a
// Sail-computed expected signature baked in at build time. On comparison
// result the ELF invokes either RVMODEL_HALT_PASS (gp == 1) or
// RVMODEL_HALT_FAIL (gp != 1). Both map onto jolt-emu's existing HTIF
// termination primitive: a doubleword write to `tohost` where
//   device  = 0x00 (syscall-proxy)
//   payload = (exit_code << 1) | 1
// jolt-emu recognizes this pattern (see tracer/src/emulator/mod.rs::run_test),
// extracts the endcode, and the main binary propagates it as the process
// exit status. The ACT4 shell runner (tests/arch-tests/run.sh) treats exit
// code 0 as pass and non-zero as fail.

// ---------------------------------------------------------------------------
// Data section — declares tohost/fromhost and the signature region.
// ---------------------------------------------------------------------------
#define RVMODEL_DATA_SECTION                                                   \
    .pushsection .tohost, "aw", @progbits;                                     \
    .align 8; .global tohost; tohost: .dword 0;                                \
    .align 8; .global fromhost; fromhost: .dword 0;                            \
    .popsection;                                                               \
    .align 8; .global begin_regstate; begin_regstate:                          \
    .word 128;                                                                 \
    .align 8; .global end_regstate; end_regstate:                              \
    .word 4;

// ---------------------------------------------------------------------------
// Halt with pass — gp == 1.
// Encodes endcode 0 in the payload (0 << 1 | 1 == 1), then spins waiting for
// HTIF to observe the write. jolt-emu detects the write and exits 0.
// ---------------------------------------------------------------------------
#define RVMODEL_HALT_PASS                                                      \
    li t0, 1;                                                                  \
    la t1, tohost;                                                             \
    sd t0, 0(t1);                                                              \
1:  j 1b;

// ---------------------------------------------------------------------------
// Halt with fail — gp holds the failing test number (non-zero).
// Encodes endcode = gp in the payload ((gp << 1) | 1), spins. jolt-emu
// detects the write and exits with gp (non-zero).
// ---------------------------------------------------------------------------
#define RVMODEL_HALT_FAIL                                                      \
    slli t0, gp, 1;                                                            \
    ori  t0, t0, 1;                                                            \
    la   t1, tohost;                                                           \
    sd   t0, 0(t1);                                                            \
1:  j 1b;

// ACT4's generic HALT macro: branch to HALT_PASS if gp == 1, else HALT_FAIL.
#define RVMODEL_HALT                                                           \
    li   t2, 1;                                                                \
    beq  gp, t2, 10f;                                                          \
    RVMODEL_HALT_FAIL;                                                         \
10: RVMODEL_HALT_PASS;

// Boot — no-op on Jolt. The linker script places .text.init first and the
// ELF entry point is rvtest_entry_point, which ACT4 provides.
#define RVMODEL_BOOT

// ---------------------------------------------------------------------------
// Signature region markers.
// ---------------------------------------------------------------------------
#define RVMODEL_DATA_BEGIN                                                     \
    RVMODEL_DATA_SECTION                                                       \
    .align 4;                                                                  \
    .global begin_signature;                                                   \
    begin_signature:

#define RVMODEL_DATA_END                                                       \
    .align 4;                                                                  \
    .global end_signature;                                                     \
    end_signature:

// ---------------------------------------------------------------------------
// I/O — unused on Jolt (signature-based self-check is the sole mechanism).
// ---------------------------------------------------------------------------
#define RVMODEL_IO_INIT(_T1, _T2, _T3)
#define RVMODEL_IO_WRITE_STR(_T1, _T2, _T3, _REG)
#define RVMODEL_IO_CHECK()
#define RVMODEL_IO_ASSERT_GPR_EQ(_S, _R, _I)
#define RVMODEL_IO_ASSERT_SFPR_EQ(_F, _R, _I)
#define RVMODEL_IO_ASSERT_DFPR_EQ(_D, _R, _I)

// ---------------------------------------------------------------------------
// Interrupt & timer control — unused on Jolt (no CLINT/PLIC, unprivileged-
// only). Every *_INT expansion below is a no-op; ACT4 tests that actually
// exercise interrupts are filtered out by `include_priv_tests: false` in
// test_config.yaml, so these macros are never invoked in generated test
// bodies. They exist solely to satisfy tests/env/check_defines.h.
//
// INTERRUPT_LATENCY / TIMER_INT_SOON_DELAY are integer cycle counts used in
// arithmetic inside ACT4 test templates. Pick any small non-zero value.
// ---------------------------------------------------------------------------
#define RVMODEL_INTERRUPT_LATENCY      100
#define RVMODEL_TIMER_INT_SOON_DELAY   100

// Machine-mode interrupt controls (external + software).
#define RVMODEL_SET_MEXT_INT
#define RVMODEL_CLR_MEXT_INT
#define RVMODEL_SET_MSW_INT
#define RVMODEL_CLR_MSW_INT

// Supervisor-mode interrupt controls (external + software).
#define RVMODEL_SET_SEXT_INT
#define RVMODEL_CLR_SEXT_INT
#define RVMODEL_SET_SSW_INT
#define RVMODEL_CLR_SSW_INT

#endif // _RVMODEL_MACROS_H
