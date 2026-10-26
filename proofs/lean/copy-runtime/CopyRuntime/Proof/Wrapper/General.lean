import CopyRuntime.Proof.Wrapper.Forward
import CopyRuntime.Proof.Wrapper.Backward

/-!
# The general (equal-offset) path, `.Lgeneral` at offset `0xa4`

Reached when `(dst - src) % 8 = 0` and `n ≥ 8` (aligned with `n > 88`, or
unaligned with equal offsets). `beq a0, a1` returns at once for `dst = src`;
`bltu a1, a0` selects the backward copy for `src < dst` and the forward copy
otherwise.

The forward and backward copies are separate segment lemmas (`forward_path`
at `0xac` and `backward_path` at `0x2c0`) so that they can be proved
independently.
-/

namespace CopyRuntime.Wrapper

variable {cfg : Config} {B L F : BitVec 64}

theorem memmoveResult_self (m : Mem) (d n : BitVec 64) : memmoveResult m d d n = m := by
  funext x
  unfold memmoveResult
  split
  · congr 1; bv_omega
  · rfl

/-- A `Returned` fact relative to an intermediate state transfers to the entry
state when the two agree on memory and on every preserved register. -/
theorem _root_.CopyRuntime.Returned.of_agree {s₀ s₁ t : State} {R dst src n : BitVec 64}
    (h : Returned s₁ R dst src n t) (hmem : s₁.mem = s₀.mem)
    (hregs : ∀ r, r ∉ clobbered → s₁.reg r = s₀.reg r) : Returned s₀ R dst src n t :=
  ⟨h.pc, h.a0, by rw [h.mem, hmem], fun r hr => (h.preserved r hr).trans (hregs r hr)⟩

/-- `Returned` does not depend on the `pc` of the reference state. -/
theorem _root_.CopyRuntime.Returned.change_pc {p₀ p₁ : BitVec 64} {r : Reg → BitVec 64} {m : Mem} {t : State}
    {R dst src n : BitVec 64} (h : Returned ⟨p₁, r, m⟩ R dst src n t) :
    Returned ⟨p₀, r, m⟩ R dst src n t :=
  ⟨h.pc, h.a0, h.mem, h.preserved⟩

/-- Forward copy (`src > dst`), from `mv a7, a0` at `0xac`: byte/half/word
steps to align `a1`, the 256-byte loop, the 80-byte block loop, the word loop
and the 4/2/1-byte tail (`Forward.lean`). Every load reads a source byte that
no earlier store has overwritten, because the destination lies below the
source. -/
theorem forward_path (ctx : Ctx cfg B L F) (regs : Reg → BitVec 64) (mem : Mem)
    {dst src n R : BitVec 64} (he : EntryRegs regs dst src n R) (hsb : Sandbox cfg dst src n)
    (hargs : ArgsOk dst src n R) (hRout : ¬ region B R)
    (hsame : (dst - src).toNat % 8 = 0) (hlt : dst.toNat < src.toNat) (hn8 : 8 ≤ n.toNat) :
    Exits cfg (region B) (Returned ⟨B + 0xac#64, regs, mem⟩ R dst src n)
      ⟨B + 0xac#64, regs, mem⟩ := by
  have hds : dst.toNat % 8 = src.toNat % 8 := by
    rw [BitVec.toNat_sub] at hsame
    have := dst.isLt; have := src.isLt
    omega
  exact fwd_prologue ⟨ctx, hsb, hargs, hRout, Nat.le_of_lt hlt, hds⟩ regs mem he hn8

/-- Backward copy (`src < dst`), from `add a1, a1, a2` at `0x2c0`: the mirror
image of `forward_path`, copying from the high end down (`Backward.lean`).
Every load reads a source byte below the part already copied, which no store
has reached, because the destination lies above the source. -/
theorem backward_path (ctx : Ctx cfg B L F) (regs : Reg → BitVec 64) (mem : Mem)
    {dst src n R : BitVec 64} (he : EntryRegs regs dst src n R) (hsb : Sandbox cfg dst src n)
    (hargs : ArgsOk dst src n R) (hRout : ¬ region B R)
    (hsame : (dst - src).toNat % 8 = 0) (hlt : src.toNat < dst.toNat) (hn8 : 8 ≤ n.toNat) :
    Exits cfg (region B) (Returned ⟨B + 0x2c0#64, regs, mem⟩ R dst src n)
      ⟨B + 0x2c0#64, regs, mem⟩ := by
  have hds : dst.toNat % 8 = src.toNat % 8 := by
    rw [BitVec.toNat_sub] at hsame
    have := dst.isLt; have := src.isLt
    omega
  exact bwd_prologue ⟨ctx, hsb, hargs, hRout, hlt, hds⟩ regs mem he hn8

/-- `.Lgeneral` at `0xa4`. -/
theorem general_path (ctx : Ctx cfg B L F) (regs : Reg → BitVec 64) (mem : Mem)
    {dst src n R : BitVec 64} (he : EntryRegs regs dst src n R) (hsb : Sandbox cfg dst src n)
    (hargs : ArgsOk dst src n R) (hRout : ¬ region B R)
    (hsame : (dst - src).toNat % 8 = 0) (hn8 : 8 ≤ n.toNat) :
    Exits cfg (region B) (Returned ⟨B + 0xa4#64, regs, mem⟩ R dst src n)
      ⟨B + 0xa4#64, regs, mem⟩ := by
  have hB := ctx.base_even
  have hR := hargs.ret_even
  have ha0 := he.a0; have ha1 := he.a1; have ha2 := he.a2; have hra := he.ra
  rv_step ctx.code [ha0, ha1]                              -- 0x0a4 beq a0, a1
  · -- dst = src: return without touching memory.
    rename_i heq
    subst heq
    rv_step ctx.code [hra]                                 -- 0x090 ret
    exact .exit hRout ⟨rfl, by simp [State.reg_mk, ha0], (memmoveResult_self _ _ _).symm,
      fun _ _ => rfl⟩
  rv_step ctx.code [ha0, ha1]                              -- 0x0a8 bltu a1, a0
  · -- src < dst: backward copy.
    rename_i h
    exact (backward_path ctx regs mem he hsb hargs hRout hsame h hn8).mono
      fun _ ht => ht.change_pc
  · -- src > dst: forward copy.
    rename_i hne hge
    have hlt : dst.toNat < src.toNat := by
      have : dst.toNat ≠ src.toNat := fun h => hne (BitVec.eq_of_toNat_eq h)
      omega
    exact (forward_path ctx regs mem he hsb hargs hRout hsame hlt hn8).mono
      fun _ ht => ht.change_pc

end CopyRuntime.Wrapper
