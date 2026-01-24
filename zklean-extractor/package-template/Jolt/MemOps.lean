import ZkLean
import Jolt.R1CS
import Jolt.Sumchecks

-- We set this option because, although we currently aren't using the parameters
-- of `memory_step`, we want to preserve its signature.
set_option linter.unusedVariables false

/--
  The in-circuit memory operations for a single Jolt step.
  Note: This is NOT automatically extracted, but currently manually written based on the current Jolt implementation.
-/
def memory_step [ZKField f]
  (inputs: SumcheckVars f)
  (mem_reg mem_ram mem_elfaddress mem_bitflags mem_rs1 mem_rs2 mem_rd mem_imm: RAM f) /- We don't currently support storing tuples inside RAM so we've split them into separate ROMs -/
  : ZKBuilder f PUnit := do
    -- Ensure that the fields in the bytecode match the corresponding R1CS inputs.
    --
    -- These are captured by the bytecode read_raf_checking sumcheck, but we
    -- don't currently extract that one.
    --
    -- XXX There are a number of other checks in read_raf_checking. Do we need
    -- to axiomatize those?
    let v <- ZKBuilder.ramRead mem_elfaddress inputs.JoltR1CSInputs.PC
    ZKBuilder.constrainEq v inputs.JoltR1CSInputs.UnexpandedPC

    let v <- ZKBuilder.ramRead mem_rs1 inputs.JoltR1CSInputs.PC
    ZKBuilder.constrainEq v inputs.RegistersReadWriteChecking_Vars.Rs1Ra

    let v <- ZKBuilder.ramRead mem_rs2 inputs.JoltR1CSInputs.PC
    ZKBuilder.constrainEq v inputs.RegistersReadWriteChecking_Vars.Rs2Ra

    let v <- ZKBuilder.ramRead mem_rd inputs.JoltR1CSInputs.PC
    ZKBuilder.constrainEq v inputs.RegistersReadWriteChecking_Vars.RdWa

    let v <- ZKBuilder.ramRead mem_imm inputs.JoltR1CSInputs.PC
    ZKBuilder.constrainEq v inputs.RegistersReadWriteChecking_Vars.RdInc

    -- Ensure that the values in the register file match the RS1/RS2/RD inputs.
    --
    -- These are captured by the register read-write-checking sumcheck.
    -- See RegistersReadWriteChecking_Vars.uniform_claims in Sumchecks.lean
    --
    -- -- Read RS1
    -- let rs1 <- ram_read mem_reg inputs.Bytecode_RS1
    -- constrainEq rs1 inputs.RS1_Read
    --
    -- -- Read RS2
    -- let rs2 <- ram_read mem_reg inputs.Bytecode_RS2
    -- constrainEq rs2 inputs.RS2_Read
    --
    -- -- Read RD, followed by Write RD
    -- let rd <- ram_read mem_reg inputs.Bytecode_RD
    -- constrainEq rd inputs.RD_Read
    -- ram_write mem_reg inputs.Bytecode_RD inputs.RD_Write

    -- Ensure that the values in RAM match the RAM read/write inputs.
    --
    -- These are captured by the ram read-write-checking sumcheck.
    -- See RamReadWriteChecking_Vars.uniform_claims in Sumchecks.lean
    --
    -- -- Read RAM, followed by Write RAM
    -- let ram_v <- ram_read mem_ram inputs.RAM_Address
    -- constrainEq ram_v inputs.RAM_Read
    -- ram_write mem_ram inputs.RAM_Address inputs.RAM_Write
