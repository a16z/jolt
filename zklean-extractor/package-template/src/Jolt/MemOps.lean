import ZkLean
import Jolt.InstructionFlags

/--
  The in-circuit memory operations for a single Jolt step.
  Note: This is NOT automatically extracted, but currently manually written based on the current Jolt implementation.
-/
def memory_step [JoltField f]
  (inputs : JoltR1CSInputs f)
  (mem_reg mem_ram mem_elfaddress mem_bitflags mem_rs1 mem_rs2 mem_rd mem_imm: RAM f) /- We don't currently support storing tuples inside RAM so we've split them into separate ROMs -/
  : ZKBuilder f PUnit := do
  let v <- ram_read mem_elfaddress inputs.Bytecode_A
  constrainEq v inputs.Bytecode_ELFAddress

  let v <- ram_read mem_bitflags inputs.Bytecode_A
  constrainEq v inputs.Bytecode_Bitflags

  let v <- ram_read mem_rs1 inputs.Bytecode_A
  constrainEq v inputs.Bytecode_RS1

  let v <- ram_read mem_rs2 inputs.Bytecode_A
  constrainEq v inputs.Bytecode_RS2

  let v <- ram_read mem_rd inputs.Bytecode_A
  constrainEq v inputs.Bytecode_RD

  let v <- ram_read mem_imm inputs.Bytecode_A
  constrainEq v inputs.Bytecode_Imm

  -- Read RS1
  let rs1 <- ram_read mem_reg inputs.Bytecode_RS1
  constrainEq rs1 inputs.RS1_Read

  -- Read RS2
  let rs2 <- ram_read mem_reg inputs.Bytecode_RS2
  constrainEq rs2 inputs.RS2_Read

  -- Read RD, followed by Write RD
  let rd <- ram_read mem_reg inputs.Bytecode_RD
  constrainEq rd inputs.RD_Read
  ram_write mem_reg inputs.Bytecode_RD inputs.RD_Write

  -- Read RAM, followed by Write RAM
  let ram_v <- ram_read mem_ram inputs.RAM_Address
  constrainEq ram_v inputs.RAM_Read
  ram_write mem_ram inputs.RAM_Address inputs.RAM_Write

