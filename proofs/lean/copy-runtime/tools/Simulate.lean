import CopyRuntime.Spec

/-!
# Differential runs of the executable model (not part of the proofs)

Runs the extracted `__wrap_memmove` (linked with `__jolt_copy_misaligned`)
under `CopyRuntime.step` on concrete inputs and compares the result with the
specification. It cross-checks the model and the extracted bytes; it proves
nothing.

    lake build && lake env lean tools/Simulate.lean     # about 30 seconds
-/

open CopyRuntime

def wrapperBase : BitVec 64 := 0x10000
def leafBase : BitVec 64 := 0x20000
def fallbackAddr : BitVec 64 := 0x30000
def returnAddr : BitVec 64 := 0x40000

/-- Run until the pc leaves both sections. `.inl s`: `step` faulted in `s`. -/
def run (cfg : Config) : Nat → State → Sum State State
  | 0, s => .inr s
  | k + 1, s =>
    if (s.pc - wrapperBase).toNat < 1412 ∨ (s.pc - leafBase).toNat < 988 then
      match step cfg s with
      | none => .inl s
      | some s' => run cfg k s'
    else .inr s

def initMem : Mem := fun a => (a.setWidth 8 ^^^ 0x5a#8) + (a >>> 8).setWidth 8

def initRegs (dst src n : BitVec 64) : Reg → BitVec 64 := fun r =>
  if r = 10 then dst else if r = 11 then src else if r = 12 then n
  else if r = 1 then returnAddr else BitVec.ofNat 64 (0x1122330000 + r.val)

/-- `none` if the run matches the specification, else a description. -/
def check (dst src n : BitVec 64) : Option String :=
  let cfg : Config :=
    { sections := [⟨wrapperBase, Generated.wrapMemmove⟩, ⟨leafBase, Generated.joltCopyMisaligned⟩]
      resolve := fun s => if s == "__jolt_copy_misaligned" then leafBase else fallbackAddr
      canLoad := rangePerm src n, canStore := rangePerm dst n }
  let tag := s!"dst={dst.toHex} src={src.toHex} n={n.toNat}"
  let toFallback := (dst - src).toNat % 8 ≠ 0 ∧ 12 ≤ n.toNat ∧ n.toNat < 544
  match run cfg 100000 ⟨wrapperBase, initRegs dst src n, initMem⟩ with
  | .inl s => some s!"{tag}: fault at pc {s.pc.toHex}"
  | .inr s =>
    if s.pc != (if toFallback then fallbackAddr else returnAddr) then
      some s!"{tag}: exit at {s.pc.toHex}"
    else if toFallback then
      if (List.range 64).all fun i => s.mem (dst + BitVec.ofNat 64 i) == initMem (dst + BitVec.ofNat 64 i)
      then none else some s!"{tag}: fallback path changed memory"
    else
      let expected := memmoveResult initMem dst src n
      let window := (List.range (n.toNat + 32)).map fun i => dst - 16 + BitVec.ofNat 64 i
      let badMem := window.filter fun a => s.mem a != expected a
      let badReg := ([2, 3, 4, 8, 9, 10, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] : List Reg).filter
        fun r => s.reg r != initRegs dst src n r
      if badMem.isEmpty && badReg.isEmpty then none
      else some s!"{tag}: {badMem.length} wrong bytes, changed registers {badReg}"

def lens : List Nat := [0, 1, 2, 7, 8, 9, 11, 12, 16, 24, 79, 80, 81, 88, 89, 96, 100, 159, 160,
  255, 256, 257, 300, 543, 544, 545, 600, 700, 1030]

-- Ordinary inputs: expect `(0, [])`.
#eval
  let results := lens.flatMap fun n => ([0, 1, 3, 4, 7] : List Nat).flatMap fun sa =>
    ([0, 2, 5, 6] : List Nat).flatMap fun da => ([8, 64, 4224] : List Nat).flatMap fun delta =>
      ([false, true] : List Bool).filterMap fun rev =>
        let a : BitVec 64 := BitVec.ofNat 64 (0x100000 + sa)
        let b : BitVec 64 := BitVec.ofNat 64 (0x100000 + delta + da)
        let (src, dst) := if rev then (b, a) else (a, b)
        check dst src (BitVec.ofNat 64 n)
  (results.length, results.take 10)

-- The source range ends exactly at 2^64 (excluded by `ArgsOk`): bytes 88..95
-- are not copied by the forward word loop.
#eval check (0 - 104) (0 - 96) 96

-- The destination range ends exactly at 2^64 (excluded by `ArgsOk`): the
-- leaf's backward bound check wraps and an aligned load reads past `src + n`.
#eval check (0 - 600) (0 - 601) 600

-- One byte lower, both are fine: expect `(none, none)`.
#eval (check (0 - 105) (0 - 97) 96, check (0 - 601) (0 - 602) 600)
