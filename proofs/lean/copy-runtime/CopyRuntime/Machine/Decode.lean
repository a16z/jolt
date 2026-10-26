import CopyRuntime.Machine.Instr

/-!
# Instruction decoder (trusted; cross-checked against `llvm-objdump`)

Decodes the RV64I and RVC encodings that occur in the copy runtime, following
"The RISC-V Instruction Set Manual, Volume I: Unprivileged ISA" (base formats
in chapter "RV32I Base Integer Instruction Set", RV64I additions, and the
"C" Standard Extension for Compressed Instructions). Every other encoding,
including all reserved and unsupported ones, decodes to `none`.

`CopyRuntime.DecoderCheck` proves, by kernel evaluation, that on every
instruction of the three extracted sections this decoder agrees with the
listing produced by `llvm-objdump -M no-aliases -M numeric`.
-/

namespace CopyRuntime.Decode

/-- Bits `hi .. lo` (inclusive) of `w`. -/
def bits (w hi lo : Nat) : Nat := (w >>> lo) % 2 ^ (hi + 1 - lo)

/-- Sign-extend the low `width` bits of `v` to 64 bits. -/
def sext (width v : Nat) : BitVec 64 := (BitVec.ofNat width v).signExtend 64

/-- Zero-extend `v` to 64 bits. -/
def zext (v : Nat) : BitVec 64 := BitVec.ofNat 64 v

/-- Register number from a register field. -/
def reg (v : Nat) : Reg := ⟨v % 32, Nat.mod_lt _ (by decide)⟩

/-- Decode a 32-bit instruction word. -/
def decode32 (w : Nat) : Option Instr :=
  let rd := reg (bits w 11 7)
  let rs1 := reg (bits w 19 15)
  let rs2 := reg (bits w 24 20)
  let funct3 := bits w 14 12
  let funct7 := bits w 31 25
  let immI := sext 12 (bits w 31 20)
  let immS := sext 12 ((bits w 31 25 <<< 5) ||| bits w 11 7)
  let immB := sext 13 ((bits w 31 31 <<< 12) ||| (bits w 7 7 <<< 11) |||
    (bits w 30 25 <<< 5) ||| (bits w 11 8 <<< 1))
  let immU := sext 32 (bits w 31 12 <<< 12)
  let immJ := sext 21 ((bits w 31 31 <<< 20) ||| (bits w 19 12 <<< 12) |||
    (bits w 20 20 <<< 11) ||| (bits w 30 21 <<< 1))
  -- RV64 shifts: shamt = imm[5:0], imm[11:6] selects the operation.
  let shamt := zext (bits w 25 20)
  match bits w 6 0 with
  | 0b0010111 => some (.auipc rd immU)
  | 0b1101111 => some (.jal rd immJ)
  | 0b1100111 => if funct3 = 0b000 then some (.jalr rd rs1 immI) else none
  | 0b1100011 =>
    match funct3 with
    | 0b000 => some (.branch .eq rs1 rs2 immB)
    | 0b001 => some (.branch .ne rs1 rs2 immB)
    | 0b110 => some (.branch .ltu rs1 rs2 immB)
    | 0b111 => some (.branch .geu rs1 rs2 immB)
    | _ => none
  | 0b0000011 =>
    match funct3 with
    | 0b011 => some (.load .double rd rs1 immI)  -- ld
    | 0b100 => some (.load .byte rd rs1 immI)    -- lbu
    | 0b101 => some (.load .half rd rs1 immI)    -- lhu
    | 0b110 => some (.load .word rd rs1 immI)    -- lwu
    | _ => none                                  -- lb, lh, lw, reserved
  | 0b0100011 =>
    match funct3 with
    | 0b000 => some (.store .byte rs2 rs1 immS)
    | 0b001 => some (.store .half rs2 rs1 immS)
    | 0b010 => some (.store .word rs2 rs1 immS)
    | 0b011 => some (.store .double rs2 rs1 immS)
    | _ => none
  | 0b0010011 =>
    match funct3 with
    | 0b000 => some (.aluImm .addi rd rs1 immI)
    | 0b111 => some (.aluImm .andi rd rs1 immI)
    | 0b001 => if bits w 31 26 = 0 then some (.aluImm .slli rd rs1 shamt) else none
    | 0b101 => if bits w 31 26 = 0 then some (.aluImm .srli rd rs1 shamt) else none
    | _ => none
  | 0b0110011 =>
    match funct7, funct3 with
    | 0b0000000, 0b000 => some (.alu .add rd rs1 rs2)
    | 0b0100000, 0b000 => some (.alu .sub rd rs1 rs2)
    | 0b0000000, 0b001 => some (.alu .sll rd rs1 rs2)
    | 0b0000000, 0b100 => some (.alu .xor rd rs1 rs2)
    | 0b0000000, 0b101 => some (.alu .srl rd rs1 rs2)
    | 0b0000000, 0b110 => some (.alu .or rd rs1 rs2)
    | _, _ => none
  | _ => none

/-- Decode a 16-bit RVC parcel into the equivalent base instruction, using the
expansions given in the C-extension chapter. HINT encodings (for example
`c.addi x0, imm` or `c.mv x0, rs2`) expand to the base instruction with
`rd = x0`, whose architectural effect is none; this matches the manual's
statement that HINTs execute as no-ops on implementations that ignore them. -/
def decode16 (h : Nat) : Option Instr :=
  let funct3 := bits h 15 13
  let rFull := reg (bits h 11 7)       -- rd / rs1 in CI and CR formats
  let rs2Full := reg (bits h 6 2)      -- rs2 in CR format
  let rHi := reg (8 + bits h 9 7)      -- rd' / rs1' in CL, CB, CA formats
  let rLo := reg (8 + bits h 4 2)      -- rd' in CL, rs2' in CA format
  let immCI := sext 6 ((bits h 12 12 <<< 5) ||| bits h 6 2)
  let shamtCI := zext ((bits h 12 12 <<< 5) ||| bits h 6 2)
  let immCJ := sext 12 ((bits h 12 12 <<< 11) ||| (bits h 11 11 <<< 4) |||
    (bits h 10 9 <<< 8) ||| (bits h 8 8 <<< 10) ||| (bits h 7 7 <<< 6) |||
    (bits h 6 6 <<< 7) ||| (bits h 5 3 <<< 1) ||| (bits h 2 2 <<< 5))
  let immCB := sext 9 ((bits h 12 12 <<< 8) ||| (bits h 11 10 <<< 3) |||
    (bits h 6 5 <<< 6) ||| (bits h 4 3 <<< 1) ||| (bits h 2 2 <<< 5))
  let uimmCLD := zext ((bits h 12 10 <<< 3) ||| (bits h 6 5 <<< 6))
  match bits h 1 0, funct3 with
  -- C.LD rd', uimm(rs1')  =>  ld rd', uimm(rs1')
  | 0b00, 0b011 => some (.load .double rLo rHi uimmCLD)
  -- C.ADDI rd, imm  =>  addi rd, rd, imm  (C.NOP and HINTs included)
  | 0b01, 0b000 => some (.aluImm .addi rFull rFull immCI)
  -- C.LI rd, imm  =>  addi rd, x0, imm
  | 0b01, 0b010 => some (.aluImm .addi rFull 0 immCI)
  | 0b01, 0b100 =>
    match bits h 11 10 with
    -- C.SRLI rd', shamt  =>  srli rd', rd', shamt
    | 0b00 => some (.aluImm .srli rHi rHi shamtCI)
    -- C.ANDI rd', imm  =>  andi rd', rd', imm
    | 0b10 => some (.aluImm .andi rHi rHi immCI)
    -- C.SUB rd', rs2'  =>  sub rd', rd', rs2'   (bit 12 = 0, funct2 = 00)
    | 0b11 =>
      if bits h 12 12 = 0 ∧ bits h 6 5 = 0b00 then some (.alu .sub rHi rHi rLo) else none
    | _ => none
  -- C.J offset  =>  jal x0, offset
  | 0b01, 0b101 => some (.jal 0 immCJ)
  -- C.BEQZ rs1', offset  =>  beq rs1', x0, offset
  | 0b01, 0b110 => some (.branch .eq rHi 0 immCB)
  -- C.BNEZ rs1', offset  =>  bne rs1', x0, offset
  | 0b01, 0b111 => some (.branch .ne rHi 0 immCB)
  -- C.SLLI rd, shamt  =>  slli rd, rd, shamt
  | 0b10, 0b000 => some (.aluImm .slli rFull rFull shamtCI)
  | 0b10, 0b100 =>
    if bits h 12 12 = 0 then
      if bits h 6 2 = 0 then
        -- C.JR rs1  =>  jalr x0, 0(rs1); rs1 = x0 is reserved
        if bits h 11 7 = 0 then none else some (.jalr 0 rFull 0)
      else
        -- C.MV rd, rs2  =>  add rd, x0, rs2
        some (.alu .add rFull 0 rs2Full)
    else
      -- C.EBREAK and C.JALR are not supported.
      if bits h 6 2 = 0 then none
      -- C.ADD rd, rs2  =>  add rd, rd, rs2
      else some (.alu .add rFull rFull rs2Full)
  | _, _ => none

/-- Decode the instruction that starts at the head of `bs`, returning it with
its length in bytes. A parcel whose low two bits are not `11` is a 16-bit
instruction; `11` with bits `[4:2] ≠ 111` is a 32-bit instruction; longer
encodings are not supported. Truncated instructions decode to `none`. -/
def decodeBytes : List UInt8 → Option (Instr × Nat)
  | b0 :: b1 :: rest =>
    let lo := b0.toNat ||| (b1.toNat <<< 8)
    if lo % 4 ≠ 3 then
      (decode16 lo).map (·, 2)
    else if bits lo 4 2 = 0b111 then
      none
    else
      match rest with
      | b2 :: b3 :: _ =>
        (decode32 (lo ||| (b2.toNat <<< 16) ||| (b3.toNat <<< 24))).map (·, 4)
      | _ => none
  | _ => none

/-- Decode the instruction at byte offset `off` of a section. -/
def decodeAt (bytes : List UInt8) (off : Nat) : Option (Instr × Nat) :=
  decodeBytes (bytes.drop off)

end CopyRuntime.Decode
