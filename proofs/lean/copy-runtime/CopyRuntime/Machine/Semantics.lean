import CopyRuntime.Machine.Image
import CopyRuntime.Machine.Decode

/-!
# Sandboxed RV64 execution (trusted)

A deterministic small-step semantics for the instructions of
`CopyRuntime.Machine.Instr`, with a sandbox that turns every behaviour outside
the copy contract into a fault:

* instructions are fetched from immutable, extracted section images loaded at
  symbolic base addresses; a fetch outside every section, at an odd address, of
  an undecodable or truncated encoding, or of a relocation site the linker
  would reject, faults;
* a load faults unless it is naturally aligned and every byte it reads is in
  `Config.canLoad`; a store likewise with `Config.canStore`;
* a control transfer to an odd address faults.

Faulting is modelled by `step` returning `none`. A proof that execution reaches
an exit therefore also proves that no access left the permitted ranges and that
every access was naturally aligned.

Fetching from immutable images instead of from `State.mem` is faithful as long
as stores never hit code; the top-level theorems assume that the code ranges
are disjoint from the writable range.
-/

namespace CopyRuntime

open Decode

/-- Byte-addressed memory. -/
abbrev Mem := BitVec 64 → BitVec 8

structure State where
  pc : BitVec 64
  /-- Raw register file; read it through `State.reg`, which hardwires `x0`. -/
  regs : Reg → BitVec 64
  mem : Mem

/-- Register read; `x0` always reads as zero. -/
def State.reg (s : State) (r : Reg) : BitVec 64 :=
  if r = 0 then 0 else s.regs r

/-- Pointwise register-file update. -/
def writeReg (regs : Reg → BitVec 64) (r : Reg) (v : BitVec 64) : Reg → BitVec 64 :=
  fun i => if i = r then v else regs i

/-- Register write; writes to `x0` are discarded. -/
def State.setReg (s : State) (r : Reg) (v : BitVec 64) : State :=
  if r = 0 then s else { s with regs := writeReg s.regs r v }

/-! ## Memory -/

/-- Little-endian load of `n` bytes at `a`, zero-extended to 64 bits: the byte
at `a` is the least significant byte, the byte at `a + 1` the next, and so on. -/
def loadLE (m : Mem) (a : BitVec 64) : Nat → BitVec 64
  | 0 => 0
  | n + 1 => (m a).setWidth 64 ||| (loadLE m (a + 1) n <<< 8)

/-- Little-endian store of the low `n` bytes of `v` at `a`: the byte at `a + i`
(for `i < n`) receives bits `8i .. 8i+7` of `v`; every other byte is unchanged. -/
def storeLE (m : Mem) (a v : BitVec 64) (n : Nat) : Mem :=
  fun x => if (x - a).toNat < n then (v >>> (8 * (x - a).toNat)).setWidth 8 else m x

/-- An access of `n` bytes at `a` is permitted iff it is naturally aligned and
`perm` allows every accessed byte. -/
def accessOk (perm : BitVec 64 → Bool) (a : BitVec 64) (n : Nat) : Bool :=
  a.toNat % n == 0 && (List.range n).all fun i => perm (a + BitVec.ofNat 64 i)

/-! ## Instruction semantics -/

def BranchCond.eval : BranchCond → BitVec 64 → BitVec 64 → Bool
  | .eq, a, b => a == b
  | .ne, a, b => a != b
  | .ltu, a, b => decide (a.toNat < b.toNat)
  | .geu, a, b => decide (b.toNat ≤ a.toNat)

/-- Shifts use the low six bits of the shift amount (RV64). -/
def AluImmOp.eval : AluImmOp → BitVec 64 → BitVec 64 → BitVec 64
  | .addi, a, i => a + i
  | .andi, a, i => a &&& i
  | .slli, a, i => a <<< (i.toNat % 64)
  | .srli, a, i => a >>> (i.toNat % 64)

/-- Shifts use the low six bits of `rs2` (RV64). -/
def AluOp.eval : AluOp → BitVec 64 → BitVec 64 → BitVec 64
  | .add, a, b => a + b
  | .sub, a, b => a - b
  | .or, a, b => a ||| b
  | .xor, a, b => a ^^^ b
  | .sll, a, b => a <<< (b.toNat % 64)
  | .srl, a, b => a >>> (b.toNat % 64)

/-! ## Code images and linking -/

/-- A section image placed at a load address. -/
structure LoadedSection where
  base : BitVec 64
  sec : Section

def LoadedSection.contains (ls : LoadedSection) (pc : BitVec 64) : Bool :=
  (pc - ls.base).toNat < ls.sec.bytes.length

structure Config where
  /-- The loaded code. -/
  sections : List LoadedSection
  /-- Final symbol addresses chosen by the linker. -/
  resolve : String → BitVec 64
  /-- Bytes the code may read. -/
  canLoad : BitVec 64 → Bool
  /-- Bytes the code may write. -/
  canStore : BitVec 64 → Bool

def Config.sectionAt (cfg : Config) (pc : BitVec 64) : Option LoadedSection :=
  cfg.sections.find? (·.contains pc)

/-- The low 12 bits of a PC-relative value, sign-extended (the `jalr` part of
`R_RISCV_CALL_PLT`). -/
def lo12 (d : BitVec 64) : BitVec 64 := (d.setWidth 12).signExtend 64

/-- lld accepts an `R_RISCV_JAL` value iff it is even and fits in 21 signed bits. -/
def jalInRange (d : BitVec 64) : Bool :=
  (d + 0x100000).toNat < 0x200000 && d.toNat % 2 == 0

/-- lld accepts an `R_RISCV_CALL_PLT` value iff `d + 0x800` fits in 32 signed bits. -/
def callInRange (d : BitVec 64) : Bool :=
  (d + 0x800 + 0x80000000).toNat < 0x100000000

/-- The value `S + A - P` of a relocation. -/
def relocValue (ls : LoadedSection) (resolve : String → BitVec 64) (r : Reloc) : BitVec 64 :=
  resolve r.symbol + BitVec.ofInt 64 r.addend - (ls.base + BitVec.ofNat 64 r.offset)

/-- Does relocation `r` patch the instruction at section offset `off`? -/
def Reloc.covers (r : Reloc) (off : Nat) : Bool :=
  r.offset == off || (r.kind == .callPlt && r.offset + 4 == off)

/-- The instruction at section offset `off` after linking: the raw encoding is
decoded, and if a relocation patches it, its immediate is replaced by the value
the linker writes. For an in-range `R_RISCV_CALL_PLT` value `d`, the `auipc`
adds `d - lo12 d` (that is, `sext32(hi20 << 12)`) and the `jalr` adds `lo12 d`.
An out-of-range value (a link error) or an unexpected opcode at a relocation
site faults. -/
def linkedInstr (ls : LoadedSection) (resolve : String → BitVec 64) (off : Nat) :
    Option (Instr × Nat) := do
  let raw ← decodeAt ls.sec.bytes off
  match ls.sec.relocs.filter (·.covers off) with
  | [] => pure raw
  | [r] =>
    let d := relocValue ls resolve r
    match r.kind, raw with
    | .jal, (.jal rd _, 4) =>
      if r.offset = off ∧ jalInRange d then some (.jal rd d, 4) else none
    | .callPlt, (.auipc rd _, 4) =>
      if r.offset = off ∧ callInRange d then some (.auipc rd (d - lo12 d), 4) else none
    | .callPlt, (.jalr rd rs1 _, 4) =>
      if r.offset + 4 = off ∧ callInRange d then some (.jalr rd rs1 (lo12 d), 4) else none
    | _, _ => none
  | _ => none

def Config.fetch (cfg : Config) (pc : BitVec 64) : Option (Instr × Nat) := do
  let ls ← cfg.sectionAt pc
  if pc.toNat % 2 ≠ 0 then none
  linkedInstr ls cfg.resolve (pc - ls.base).toNat

/-! ## Steps -/

/-- Transfer control; an odd target is an instruction-address-misaligned fault
(with the C extension, IALIGN = 16). -/
def jumpTo (s : State) (target : BitVec 64) : Option State :=
  if target.toNat % 2 = 0 then some { s with pc := target } else none

/-- Execute instruction `i` of length `len` (in bytes) at `s.pc`. `none` is a fault. -/
def execute (cfg : Config) (s : State) (i : Instr) (len : Nat) : Option State :=
  let next := s.pc + BitVec.ofNat 64 len
  match i with
  | .auipc rd imm => some { s.setReg rd (s.pc + imm) with pc := next }
  | .jal rd off => jumpTo (s.setReg rd next) (s.pc + off)
  | .jalr rd rs1 off => jumpTo (s.setReg rd next) ((s.reg rs1 + off) &&& ~~~1)
  | .branch c rs1 rs2 off =>
    if c.eval (s.reg rs1) (s.reg rs2) then jumpTo s (s.pc + off) else some { s with pc := next }
  | .load w rd rs1 off =>
    let a := s.reg rs1 + off
    if accessOk cfg.canLoad a w.bytes then
      some { s.setReg rd (loadLE s.mem a w.bytes) with pc := next }
    else none
  | .store w rs2 rs1 off =>
    let a := s.reg rs1 + off
    if accessOk cfg.canStore a w.bytes then
      some { s with pc := next, mem := storeLE s.mem a (s.reg rs2) w.bytes }
    else none
  | .aluImm op rd rs1 imm => some { s.setReg rd (op.eval (s.reg rs1) imm) with pc := next }
  | .alu op rd rs1 rs2 => some { s.setReg rd (op.eval (s.reg rs1) (s.reg rs2)) with pc := next }

/-- Fetch and execute one instruction. `none` is a fault. -/
def step (cfg : Config) (s : State) : Option State := do
  let (i, len) ← cfg.fetch s.pc
  execute cfg s i len

/-- `Exits cfg region post s`: starting from `s`, the machine executes
instructions only while `pc ∈ region`, never faults, and reaches (after
finitely many steps) the first state whose `pc ∉ region`; that state
satisfies `post`. Because `step` is a function, this exit state is unique
(`Exits.unique`). -/
inductive Exits (cfg : Config) (region : BitVec 64 → Prop) (post : State → Prop) :
    State → Prop where
  | exit {s : State} : ¬ region s.pc → post s → Exits cfg region post s
  | step {s s' : State} : region s.pc → step cfg s = some s' →
      Exits cfg region post s' → Exits cfg region post s

/-- The address range of a section loaded at `base`. -/
def inSection (base : BitVec 64) (sec : Section) (pc : BitVec 64) : Prop :=
  (pc - base).toNat < sec.bytes.length

end CopyRuntime
