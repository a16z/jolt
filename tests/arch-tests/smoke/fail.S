# Deliberate-failure smoke test for the ACT4 harness (spec Evaluation →
# Testing Strategy).
#
# The program writes a non-zero HTIF endcode to `tohost` via the same
# mechanism RVMODEL_HALT_FAIL uses. jolt-emu's run_test should detect the
# write and exit with status 2. The shell harness (tests/arch-tests/run.sh)
# should then treat the test as FAIL and exit non-zero itself.
#
# If either jolt-emu exits 0 on this input or the harness reports PASS, the
# pass/fail plumbing is broken and every real test would silently pass.

    .section .text.init, "ax"
    .global rvtest_entry_point
rvtest_entry_point:
    # Write ((2 << 1) | 1) = 5 to `tohost`: device=0, payload=0x5 → endcode 2.
    li    t0, 5
    la    t1, tohost
    sd    t0, 0(t1)
1:  j     1b

    .section .tohost, "aw", @progbits
    .align 8
    .global tohost
tohost:
    .dword 0
    .align 8
    .global fromhost
fromhost:
    .dword 0
