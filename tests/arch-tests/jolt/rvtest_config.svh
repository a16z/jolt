// ACT4 per-DUT SystemVerilog config header.
//
// Jolt is a zkVM, not an RTL simulator, so this file exists only to satisfy
// ACT4's expectation of a `rvtest_config.svh` companion to rvtest_config.h.
// No SystemVerilog code is ever compiled from it.

`ifndef _RVTEST_CONFIG_SVH
`define _RVTEST_CONFIG_SVH

`define RVTEST_XLEN 64
`define RVTEST_ISA_RV64IMAC 1

`define RVTEST_HAS_M 1
`define RVTEST_HAS_A 1
`define RVTEST_HAS_C 1
`define RVTEST_HAS_F 0
`define RVTEST_HAS_D 0

`define RVTEST_HAS_MACHINE_MODE 0
`define RVTEST_HAS_SUPERVISOR_MODE 0
`define RVTEST_HAS_USER_MODE 0

`endif // _RVTEST_CONFIG_SVH
