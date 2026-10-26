import CopyRuntime.Machine.Semantics
import CopyRuntime.Generated.Object

/-!
# Specification vocabulary (trusted: read this file to know what is proved)

Everything here is a plain definition used by the theorem statements in
`CopyRuntime.Theorems`. No proofs live here.
-/

namespace CopyRuntime

/-! ## Linking and placement -/

/-- `sec` is loaded at `base`, and every fetch inside its address range is
served from it (no other loaded section overlaps it). -/
def Config.Loaded (cfg : Config) (base : BitVec 64) (sec : Section) : Prop :=
  ∀ pc, inSection base sec pc → cfg.sectionAt pc = some ⟨base, sec⟩

/-! ## Memory ranges and the copy result -/

/-- `a ∈ [base, base + len)`, as natural numbers. -/
def InRange (base len a : BitVec 64) : Prop :=
  base.toNat ≤ a.toNat ∧ a.toNat < base.toNat + len.toNat

instance (base len a : BitVec 64) : Decidable (InRange base len a) :=
  inferInstanceAs (Decidable (_ ∧ _))

/-- Permission predicate allowing exactly `[base, base + len)`. -/
def rangePerm (base len : BitVec 64) (a : BitVec 64) : Bool :=
  decide (InRange base len a)

/-- The memory after `memmove(dst, src, n)` from memory `m`: every byte of
`[dst, dst + n)` receives the byte that was at the same position of
`[src, src + n)` *in `m`* (a snapshot copy, so overlapping ranges are handled
as memmove requires); all other bytes are unchanged. -/
def memmoveResult (m : Mem) (dst src n : BitVec 64) : Mem :=
  fun a => if InRange dst n a then m (src + (a - dst)) else m a

/-! ## Registers -/

/-- Registers the copy routines may change: `t0`–`t2` (`x5`–`x7`), `a1`–`a7`
(`x11`–`x17`) and `t3`–`t6` (`x28`–`x31`). All others, including `ra`, `sp`,
`gp`, `tp`, `s0`–`s11` and `a0`, must be preserved. -/
def clobbered : List Reg := [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 28, 29, 30, 31]

/-- The relative offsets of `dst` and `src` modulo 8 differ. -/
def Mismatched (dst src : BitVec 64) : Prop := (dst - src).toNat % 8 ≠ 0

instance (dst src : BitVec 64) : Decidable (Mismatched dst src) :=
  inferInstanceAs (Decidable (¬ _))

/-! ## Outcomes -/

/-- Outcome (a): the routine returned to `R` having performed
`memmove(dst, src, n)`. Compared with the entry state `s0`, only the registers
in `clobbered` may differ, and `a0` still holds `dst`. -/
structure Returned (s0 : State) (R dst src n : BitVec 64) (s : State) : Prop where
  pc : s.pc = R
  a0 : s.reg 10 = dst
  mem : s.mem = memmoveResult s0.mem dst src n
  preserved : ∀ r, r ∉ clobbered → s.reg r = s0.reg r

/-- Outcome (b): control reached the misaligned-copy leaf at `L` with memory
unchanged and every register except `t0` equal to its value at entry (so `a0`,
`a1`, `a2`, `ra` still hold `dst`, `src`, `n`, the return address). -/
structure AtLeaf (s0 : State) (L : BitVec 64) (s : State) : Prop where
  pc : s.pc = L
  mem : s.mem = s0.mem
  preserved : ∀ r, r ≠ 5 → s.reg r = s0.reg r

/-- Outcome (c): control tail-called the fallback `F` with memory unchanged and
every register except `t0` and `t1` equal to its value at entry (so the
fallback receives the original arguments and return address). -/
structure AtFallback (s0 : State) (F : BitVec 64) (s : State) : Prop where
  pc : s.pc = F
  mem : s.mem = s0.mem
  preserved : ∀ r, r ≠ 5 → r ≠ 6 → s.reg r = s0.reg r

/-- The outcome of a wrapper call is determined by the arguments: the leaf for
mismatched offsets and `n ≥ 544`, the fallback for mismatched offsets and
`12 ≤ n < 544`, and a completed copy otherwise. -/
def wrapperPost (s0 : State) (L F R dst src n : BitVec 64) : State → Prop :=
  if Mismatched dst src ∧ 544 ≤ n.toNat then AtLeaf s0 L
  else if Mismatched dst src ∧ 12 ≤ n.toNat then AtFallback s0 F
  else Returned s0 R dst src n

/-! ## Entry conditions -/

/-- The calling convention at entry: `pc` is the entry point, `a0 = dst`,
`a1 = src`, `a2 = n`, `ra = R`. -/
structure CallEntry (s0 : State) (entry R dst src n : BitVec 64) : Prop where
  pc : s0.pc = entry
  a0 : s0.reg 10 = dst
  a1 : s0.reg 11 = src
  a2 : s0.reg 12 = n
  ra : s0.reg 1 = R

/-- The sandbox: the routine may read exactly `[src, src + n)` and write
exactly `[dst, dst + n)`. -/
structure Sandbox (cfg : Config) (dst src n : BitVec 64) : Prop where
  load : cfg.canLoad = rangePerm src n
  store : cfg.canStore = rangePerm dst n

/-- Conditions on the arguments shared by every entry point.

The range conditions are C's rule that the address one past the end of an
object is representable: neither range may reach the top of the address
space. They are strict because the code needs them to be. If `src + n = 2^64`,
the forward word loop's end pointer `a1 + (a2 & ~7)` wraps to 0 and up to 8
bytes are left uncopied (for example `src = 2^64 - 96`, `dst = src - 8`,
`n = 96`). If `dst + n = 2^64`, the leaf's backward bound check `addi t2, a1, 7`
can wrap, and its first aligned `ld` then reads up to 7 bytes past the end of
the source (for example `n = 600`, `dst = 2^64 - 600`, `src = dst - 1`). -/
structure ArgsOk (dst src n R : BitVec 64) : Prop where
  /-- `[src, src + n]` lies inside the address space. -/
  src_no_wrap : src.toNat + n.toNat < 2 ^ 64
  /-- `[dst, dst + n]` lies inside the address space. -/
  dst_no_wrap : dst.toNat + n.toNat < 2 ^ 64
  /-- Return addresses are instruction addresses (IALIGN = 16). -/
  ret_even : R.toNat % 2 = 0

/-- A code range that is not written by the copy and does not contain the
return address. The disjointness from `[dst, dst + n)` is not used by the
proofs; it is what makes fetching from the immutable image (rather than from
`State.mem`) faithful to the hardware. -/
structure CodeRangeOk (base : BitVec 64) (sec : Section) (dst n R : BitVec 64) : Prop where
  base_even : base.toNat % 2 = 0
  ret_outside : ¬ inSection base sec R
  not_written : ∀ a, InRange dst n a → ¬ inSection base sec a

/-- The linker accepted every relocation of `sec` loaded at `base`, given the
symbol addresses in `cfg.resolve` (lld rejects out-of-range values). -/
def RelocsInRange (cfg : Config) (base : BitVec 64) (sec : Section) : Prop :=
  ∀ r ∈ sec.relocs,
    (match r.kind with
      | .jal => jalInRange (relocValue ⟨base, sec⟩ cfg.resolve r)
      | .callPlt => callInRange (relocValue ⟨base, sec⟩ cfg.resolve r)) = true

/-- Where a wrapper loaded at `B` may exit to: the leaf `L` and the fallback
`F` lie outside the wrapper, and `F` is an instruction address. -/
structure ExitTargetsOk (B : BitVec 64) (W : Section) (L F : BitVec 64) : Prop where
  leaf_outside : ¬ inSection B W L
  fallback_outside : ¬ inSection B W F
  fallback_even : F.toNat % 2 = 0

end CopyRuntime
