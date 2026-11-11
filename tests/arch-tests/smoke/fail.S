# Regression for ACT4's failure macro and the smoke target's exit-status check.
#
# Invoke the real RVMODEL_HALT_FAIL with gp = 2^63: encoding (gp << 1) | 1
# would drop the top bit and write the success word 1. The failure macro must
# make jolt-emu exit non-zero; make arch-tests-smoke rejects a zero exit.

#include "../jolt/rvmodel_macros.h"

    .section .text.init, "ax"
    .global rvtest_entry_point
rvtest_entry_point:
    li    gp, 0x8000000000000000
    RVMODEL_HALT_FAIL

    .section .tohost, "aw", @progbits
    .align 8
    .global tohost
tohost:
    .dword 0
    .align 8
    .global fromhost
fromhost:
    .dword 0
