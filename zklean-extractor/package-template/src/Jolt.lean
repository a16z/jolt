-- This module serves as the root of the `Jolt` library.
-- Import modules here that should be built as part of the library.

import Jolt.R1CS
import Jolt.Subtables
import Jolt.Instructions
import Jolt.InstructionFlags
import Jolt.MemOps

/--
  All the constraints a single Jolt step enforces.
-/
def jolt_step [JoltField f]
  (mem_reg mem_ram mem_elfaddress mem_bitflags mem_rs1 mem_rs2 mem_rd mem_imm: RAM f)
  : ZKBuilder f (JoltR1CSInputs f) :=
  do
  let inputs: JoltR1CSInputs f <- Witnessable.witness;

  uniform_jolt_constraints inputs

  lookup_step inputs

  memory_step inputs mem_reg mem_ram mem_elfaddress mem_bitflags mem_rs1 mem_rs2 mem_rd mem_imm

  pure inputs

/--
  Build constraints for a new Jolt step and constrain them with the previous step's state.
-/
def jolt_next_step [JoltField f]
  (mem_reg mem_ram mem_elfaddress mem_bitflags mem_rs1 mem_rs2 mem_rd mem_imm: RAM f)
  (prev_state: JoltR1CSInputs f):
  ZKBuilder f (JoltR1CSInputs f) :=
  do
  let new_state <- jolt_step mem_reg mem_ram mem_elfaddress mem_bitflags mem_rs1 mem_rs2 mem_rd mem_imm
  non_uniform_jolt_constraints prev_state new_state
  pure new_state
