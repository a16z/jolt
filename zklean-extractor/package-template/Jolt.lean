-- This module serves as the root of the `Jolt` library.
-- Import modules here that should be built as part of the library.

import Jolt.MemOps
import Jolt.Util

-- Extracted modules
import Jolt.R1CS
import Jolt.LookupTables
import Jolt.LookupTableFlags
import Jolt.Instructions
import Jolt.Sumchecks

/--
  All the constraints a single Jolt step enforces.
-/
def jolt_step [ZKField f]
  (inputs: SumcheckVars f)
  (mem_reg mem_ram mem_elfaddress mem_bitflags mem_rs1 mem_rs2 mem_rd mem_imm: RAM f)
  : ZKBuilder f PUnit :=
  do
  uniform_claims inputs
  uniform_jolt_constraints inputs.JoltR1CSInputs

  lookup_step inputs

  memory_step inputs mem_reg mem_ram mem_elfaddress mem_bitflags mem_rs1 mem_rs2 mem_rd mem_imm

/--
  Build constraints for a new Jolt step and constrain them with the previous step's state.
-/
def jolt_next_step [ZKField f]
  (cycle_inputs: SumcheckVars f)
  (next_cycle_inputs: SumcheckVars f)
  (mem_reg mem_ram mem_elfaddress mem_bitflags mem_rs1 mem_rs2 mem_rd mem_imm: RAM f)
  : ZKBuilder f PUnit :=
  do
  jolt_step cycle_inputs mem_reg mem_ram mem_elfaddress mem_bitflags mem_rs1 mem_rs2 mem_rd mem_imm
  non_uniform_claims cycle_inputs next_cycle_inputs
