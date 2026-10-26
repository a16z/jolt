/-
DRAFT -- NOT BUILT, DOES NOT YET CHECK.

Unfinished proof script for `Wrapper.fwd_prologue` (stated with `sorry` in
`CopyRuntime/Proof/Wrapper/Forward.lean`). It is kept as a starting point
for finishing the forward path. Paste a proof back over the `sorry` and
compile with `lake env lean` to continue.

Status when this was set aside:
* The kernel "deep recursion" error came from `simp` rewriting
  `(x + c).toNat` with a big literal `c` (a negative immediate); see
  `Wrapper.add_ofNat_eq_sub_of_ge`, which fixed `fwd_blocks`.
* The elaborator max-recursion error in `fwd_large` came from `omega`, not
  from `fwd_pairs`; see the comment in `Wrapper.fwd_large`.
* `fwd_prologue`: case split on `src % 8`, then `rv_steps_to … 0xf0`. Not
  yet checked end to end.
-/

set_option maxHeartbeats 16000000 in
/-- The forward copy from `0xac`: `mv a7, a0`, then up to three byte/half/word
steps that make `a1` (and with it `a7`) 8-byte aligned, then the loops. -/
theorem fwd_prologue (env : FwdEnv cfg B L F dst src n R) (regs₀ : Reg → BitVec 64) (m₀ : Mem)
    (he : EntryRegs regs₀ dst src n R) (hn8 : 8 ≤ n.toNat) :
    Exits cfg (region B) (Returned ⟨B + 0xac#64, regs₀, m₀⟩ R dst src n)
      ⟨B + 0xac#64, regs₀, m₀⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hsw' := env.args.src_no_wrap
  have hsw : src.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hsw'
  have hdw : dst.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt env.args.dst_no_wrap
  have hle := env.le
  have hds := env.same
  have hl := env.sb.load; have hst := env.sb.store
  have ha0 := he.a0; have ha1 := he.a1; have ha2 := he.a2
  have hcop : Copied m₀ m₀ dst src 0 0 := Copied.empty _ _ _ _
  have hcases : src.toNat % 8 = 0 ∨ src.toNat % 8 = 1 ∨ src.toNat % 8 = 2 ∨
      src.toNat % 8 = 3 ∨ src.toNat % 8 = 4 ∨ src.toNat % 8 = 5 ∨ src.toNat % 8 = 6 ∨
      src.toNat % 8 = 7 := by omega
  rcases hcases with hc | hc | hc | hc | hc | hc | hc | hc <;>
  · rv_steps_to ctx.code 0xf0 [ha0, ha1, ha2, hl, hst]
    refine fwd_large env regs₀ m₀ he ((8 - src.toNat % 8) % 8) _ _ ?_ ?_ (by omega)
    · constructor
      · simp [writeReg_apply, ha1, hc]
      · simp [writeReg_apply, ha0, hc]
      · omega
      · copied_fwd
      · fwd_preserved (fun r _ => rfl)
    · simp only [writeReg_apply, ha2]
      first
        | omega
        | (simp only [BitVec.toNat_add, BitVec.toNat_ofNat]; omega)

