/-!
# Instruction syntax (trusted)

The abstract syntax covers exactly the RV64I instructions (and, after
expansion, the RVC instructions) that occur in the copy runtime. The decoder
rejects every other encoding, and the semantics treats a rejected encoding as a
fault, so a fault-free execution only ever runs instructions listed here.

Immediates are stored already sign-extended (or zero-extended where the ISA
says so) to 64 bits, and branch/jump offsets are relative to the address of
the instruction, as in the ISA manual.
-/

namespace CopyRuntime

/-- An integer register number `x0` ... `x31`. -/
abbrev Reg := Fin 32

/-- Access width of a load or store. -/
inductive Width where
  | byte | half | word | double
  deriving DecidableEq, Repr

/-- Number of bytes accessed. -/
def Width.bytes : Width → Nat
  | .byte => 1
  | .half => 2
  | .word => 4
  | .double => 8

inductive BranchCond where
  | eq | ne | ltu | geu
  deriving DecidableEq, Repr

inductive AluImmOp where
  | addi | andi | slli | srli
  deriving DecidableEq, Repr

inductive AluOp where
  | add | sub | or | xor | sll | srl
  deriving DecidableEq, Repr

inductive Instr where
  /-- `auipc rd, imm20`; `imm` is `sext32(imm20 << 12)`. -/
  | auipc (rd : Reg) (imm : BitVec 64)
  /-- `jal rd, offset`. -/
  | jal (rd : Reg) (offset : BitVec 64)
  /-- `jalr rd, offset(rs1)`. -/
  | jalr (rd rs1 : Reg) (offset : BitVec 64)
  /-- `beq`/`bne`/`bltu`/`bgeu rs1, rs2, offset`. -/
  | branch (cond : BranchCond) (rs1 rs2 : Reg) (offset : BitVec 64)
  /-- Zero-extending loads `lbu`/`lhu`/`lwu` and `ld`: `rd := mem[rs1 + offset]`.
  (The sign-extending `lb`/`lh`/`lw` do not occur and are not decoded.) -/
  | load (width : Width) (rd rs1 : Reg) (offset : BitVec 64)
  /-- `sb`/`sh`/`sw`/`sd rs2, offset(rs1)`: `mem[rs1 + offset] := rs2`. -/
  | store (width : Width) (rs2 rs1 : Reg) (offset : BitVec 64)
  /-- `addi`/`andi`/`slli`/`srli rd, rs1, imm` (for shifts `imm` is the shamt). -/
  | aluImm (op : AluImmOp) (rd rs1 : Reg) (imm : BitVec 64)
  /-- `add`/`sub`/`or`/`xor`/`sll`/`srl rd, rs1, rs2`. -/
  | alu (op : AluOp) (rd rs1 rs2 : Reg)
  deriving DecidableEq, Repr

/-- Sign-extended immediate from an integer literal (used by generated tables). -/
def imm (i : Int) : BitVec 64 := BitVec.ofInt 64 i

end CopyRuntime
