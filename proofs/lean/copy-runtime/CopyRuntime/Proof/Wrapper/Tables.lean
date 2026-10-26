import CopyRuntime.Proof.Wrapper.Exits

/-!
# The computed-jump table paths

* aligned small copies (`a0`, `a1`, `n` multiples of 8, `8 ≤ n ≤ 88`): a jump
  into the `ld` table loads all words into registers, a second jump into the
  `sd` table stores them;
* tiny copies (`.Lsmall_bytes`, `1 ≤ n ≤ 11`): the same with `lbu`/`sb`, or a
  single byte copy for `n = 1`.

Each table slot is exactly four bytes (`.option norvc`), which is what makes
the computed offsets `n/2` and `4n` land on instruction boundaries; the proof
checks this by symbolic execution of the actual bytes for every length.
Because every load precedes every store, overlapping ranges are copied as
memmove requires.
-/

namespace CopyRuntime.Wrapper

/-- Prove `Copied m m₀ dst src lo hi` for a chain of stores whose values were
loaded from the snapshot `m₀` at matching offsets, innermost store at the
highest offset. -/
macro "copied_down" : tactic => `(tactic| (
  repeat (first
    | exact Copied.empty _ _ _ _
    | refine Copied.extend_down_zero ?_ (by decide) (by decide) (by decide)
    | refine Copied.extend_down ?_ (by decide) (by decide) (by decide))))

variable {cfg : Config} {B L F : BitVec 64}

set_option maxHeartbeats 1800000 in
theorem aligned_small_path (ctx : Ctx cfg B L F) (regs : Reg → BitVec 64) (mem : Mem)
    {dst src n R : BitVec 64} (he : EntryRegs regs dst src n R) (hsb : Sandbox cfg dst src n)
    (hargs : ArgsOk dst src n R) (hRout : ¬ region B R)
    (hd : dst.toNat % 8 = 0) (hs : src.toNat % 8 = 0) (hn : n.toNat % 8 = 0)
    (hn8 : 8 ≤ n.toNat) (hn88 : n.toNat ≤ 88) :
    Exits cfg (region B) (Returned ⟨B, regs, mem⟩ R dst src n) ⟨B, regs, mem⟩ := by
  have hB := ctx.base_even
  have hR := hargs.ret_even
  have hsw : src.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hargs.src_no_wrap
  have hdw : dst.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hargs.dst_no_wrap
  have ha0 := he.a0; have ha1 := he.a1; have ha2 := he.a2; have hra := he.ra
  have hal := (or_and_seven_eq_zero_iff dst src n).2 ⟨hd, hs, hn⟩
  have hl := hsb.load; have hst := hsb.store
  have hcases : n = 8#64 ∨ n = 16#64 ∨ n = 24#64 ∨ n = 32#64 ∨ n = 40#64 ∨ n = 48#64 ∨
      n = 56#64 ∨ n = 64#64 ∨ n = 72#64 ∨ n = 80#64 ∨ n = 88#64 := by
    simp only [BitVec.toNat_eq, BitVec.toNat_ofNat]
    omega
  rcases hcases with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
  · repeat rv_step ctx.code [ha0, ha1, ha2, hra, hal, hl, hst]
    refine .exit hRout ⟨rfl, by simp [State.reg_mk, writeReg_apply, ha0], ?_, ?_⟩
    · exact Copied.memmove (by simp only [BitVec.reduceToNat]; copied_down) hdw
    · rv_preserved

set_option maxHeartbeats 2700000 in
/-- `.Lsmall_bytes`, reached for `1 ≤ n ≤ 7`, or for mismatched offsets and
`8 ≤ n ≤ 11`. For `n = 1` it copies one byte directly; otherwise it jumps into
the `lbu` table and then into the `sb` table. -/
theorem small_bytes_path (ctx : Ctx cfg B L F) (regs : Reg → BitVec 64) (mem : Mem)
    {dst src n R : BitVec 64} (he : EntryRegs regs dst src n R) (hsb : Sandbox cfg dst src n)
    (hargs : ArgsOk dst src n R) (hRout : ¬ region B R)
    (hn1 : 1 ≤ n.toNat) (hroute : n.toNat ≤ 7 ∨ (Mismatched dst src ∧ n.toNat ≤ 11)) :
    Exits cfg (region B) (Returned ⟨B, regs, mem⟩ R dst src n) ⟨B, regs, mem⟩ := by
  have hB := ctx.base_even
  have hR := hargs.ret_even
  have hsw : src.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hargs.src_no_wrap
  have hdw : dst.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hargs.dst_no_wrap
  have ha0 := he.a0; have ha1 := he.a1; have ha2 := he.a2; have hra := he.ra
  have hl := hsb.load; have hst := hsb.store
  have hun : (dst ||| src ||| n) &&& 7#64 ≠ 0#64 := by
    rcases hroute with h7 | ⟨hmis, _⟩
    · exact or_and_seven_ne_zero_of_len (by omega) dst src
    · exact or_and_seven_ne_zero_of_mismatched hmis n
  by_cases hmis : Mismatched dst src
  · have hx := (mismatched_iff dst src).1 hmis
    have hcases : n = 1#64 ∨ n = 2#64 ∨ n = 3#64 ∨ n = 4#64 ∨ n = 5#64 ∨ n = 6#64 ∨
        n = 7#64 ∨ n = 8#64 ∨ n = 9#64 ∨ n = 10#64 ∨ n = 11#64 := by
      simp only [BitVec.toNat_eq, BitVec.toNat_ofNat]
      omega
    rcases hcases with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    · repeat rv_step ctx.code [ha0, ha1, ha2, hra, hun, hx, hl, hst]
      refine .exit hRout ⟨rfl, by simp [State.reg_mk, writeReg_apply, ha0], ?_, ?_⟩
      · exact Copied.memmove (by simp only [BitVec.reduceToNat]; copied_down) hdw
      · rv_preserved
  · have hcases : n = 1#64 ∨ n = 2#64 ∨ n = 3#64 ∨ n = 4#64 ∨ n = 5#64 ∨ n = 6#64 ∨
        n = 7#64 := by
      rcases hroute with h7 | ⟨h, _⟩
      · simp only [BitVec.toNat_eq, BitVec.toNat_ofNat]
        omega
      · exact absurd h hmis
    rcases hcases with rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    · repeat rv_step ctx.code [ha0, ha1, ha2, hra, hun, hl, hst]
      refine .exit hRout ⟨rfl, by simp [State.reg_mk, writeReg_apply, ha0], ?_, ?_⟩
      · exact Copied.memmove (by simp only [BitVec.reduceToNat]; copied_down) hdw
      · rv_preserved

end CopyRuntime.Wrapper
