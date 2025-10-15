import ZkLean
import Jolt.R1CS

/--
  The in-circuit memory operations for a single Jolt step.
  Note: This is NOT automatically extracted, but currently manually written based on the current Jolt implementation.
-/
def memory_step [ZKField f]
  (inputs : JoltR1CSInputs f)
  (mem_reg mem_ram mem_elfaddress mem_bitflags mem_rs1 mem_rs2 mem_rd mem_imm: RAM f) /- We don't currently support storing tuples inside RAM so we've split them into separate ROMs -/
  : ZKBuilder f PUnit := do
  panic! "TODO: rewrite or extract memory ops for Twist & Shout"
