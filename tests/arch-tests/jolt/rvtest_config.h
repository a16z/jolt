#ifndef _RVTEST_CONFIG_H
#define _RVTEST_CONFIG_H

// ACT4 per-DUT C header. Jolt's DUT is RV64IMAC, unprivileged-only, with
// signature self-check and HTIF-based termination. No per-test overrides are
// needed; the compile-time defines below are consumed by ACT4's test
// scaffolding where applicable.

#define RVTEST_XLEN        64
#define RVTEST_ISA_RV64IMAC 1

#define RVTEST_HAS_M 1
#define RVTEST_HAS_A 1
#define RVTEST_HAS_C 1
#define RVTEST_HAS_F 0
#define RVTEST_HAS_D 0

// Privileged modes — disabled; privileged tests are excluded via skip.txt.
#define RVTEST_HAS_MACHINE_MODE    0
#define RVTEST_HAS_SUPERVISOR_MODE 0
#define RVTEST_HAS_USER_MODE       0

#endif // _RVTEST_CONFIG_H
