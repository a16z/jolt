import CopyRuntime.Spec
import CopyRuntime.Proof.Wrapper.Main
import CopyRuntime.Proof.Leaf

/-!
# Main theorems

All statements are about the machine code extracted from the object that
`regenerate.sh` assembles from `jolt-platform/src/memory.S`
(`CopyRuntime.Generated`), executed by the sandboxed semantics of
`CopyRuntime.Machine.Semantics`, with the vocabulary of `CopyRuntime.Spec`.

`Exits cfg region post s` means: from `s` the machine executes instructions
only inside `region`, never faults (so every fetch decodes, every load reads
only `[src, src + n)`, every store writes only `[dst, dst + n)`, and every
access is naturally aligned), and after finitely many steps leaves `region`
in a state satisfying `post`. Runs are deterministic (`exit_state_unique`).
-/

namespace CopyRuntime.Theorems

open Wrapper

/-- The entry symbols are at offset 0 and cover their whole sections, so
`B` (resp. `L`) below is the address of `__wrap_memcpy`/`__wrap_memmove`
(resp. `__jolt_copy_misaligned`). -/
theorem entry_symbols :
    Generated.wrapMemcpy.symbols = [{ name := "__wrap_memcpy", offset := 0, size := 1412 }] ∧
    Generated.wrapMemmove.symbols = [{ name := "__wrap_memmove", offset := 0, size := 1412 }] ∧
    Generated.joltCopyMisaligned.symbols =
      [{ name := "__jolt_copy_misaligned", offset := 0, size := 988 }] := by
  decide

/-- The two wrappers consist of the same machine code. -/
theorem wrappers_same_code : Generated.wrapMemcpy.bytes = Generated.wrapMemmove.bytes := by
  decide +kernel

/-- Determinism: all `Exits` facts about one start state describe the same
exit state. -/
theorem exit_state_unique {cfg : Config} {region : BitVec 64 → Prop} {P Q : State → Prop}
    {s : State} (hP : Exits cfg region P s) (hQ : Exits cfg region Q s) : ∃ t, P t ∧ Q t :=
  hP.unique hQ

/-- Shared statement for both wrappers (any section with the wrapper's code). -/
theorem wrapper_correct (W : Section) (fallback : String) (hW : IsWrapper W fallback)
    (cfg : Config) (B L F R dst src n : BitVec 64) (s₀ : State)
    (hload : cfg.Loaded B W) (hrel : RelocsInRange cfg B W)
    (hleaf : cfg.resolve "__jolt_copy_misaligned" = L) (hfb : cfg.resolve fallback = F)
    (hexit : ExitTargetsOk B W L F) (hcode : CodeRangeOk B W dst n R)
    (hsb : Sandbox cfg dst src n) (hargs : ArgsOk dst src n R)
    (hentry : CallEntry s₀ B R dst src n) :
    Exits cfg (inSection B W) (wrapperPost s₀ L F R dst src n) s₀ := by
  have ctx := Ctx.of hload hW hcode.base_even hleaf hfb hrel hexit
  have hlen : W.bytes.length = 1412 := by rw [hW.bytes, size_eq]
  have hreg := inSection_iff_codeRegion (base := B) hlen
  obtain ⟨pc, regs, mem⟩ := s₀
  have hpc : pc = B := hentry.pc
  subst hpc
  have he : EntryRegs regs dst src n R :=
    ⟨by simpa [State.reg] using hentry.a0, by simpa [State.reg] using hentry.a1,
      by simpa [State.reg] using hentry.a2, by simpa [State.reg] using hentry.ra⟩
  have hRout : ¬ region pc R := fun h => hcode.ret_outside ((hreg R).2 h)
  exact (dispatch ctx regs mem he hsb hargs hRout).congr_region fun x => (hreg x).symm

/-- **`__wrap_memcpy`.** Called with `a0 = dst`, `a1 = src`, `a2 = n`,
`ra = R`, it ends in exactly the outcome `wrapperPost` prescribes: a completed
memmove (hence memcpy) and return to `R`; a jump to the leaf `L` with
unchanged memory and arguments (mismatched offsets, `n ≥ 544`); or a tail call
to `__real_memcpy` at `F` with unchanged memory and arguments (mismatched
offsets, `12 ≤ n < 544`). -/
theorem wrap_memcpy_correct (cfg : Config) (B L F R dst src n : BitVec 64) (s₀ : State)
    (hload : cfg.Loaded B Generated.wrapMemcpy)
    (hrel : RelocsInRange cfg B Generated.wrapMemcpy)
    (hleaf : cfg.resolve "__jolt_copy_misaligned" = L)
    (hfb : cfg.resolve "__real_memcpy" = F)
    (hexit : ExitTargetsOk B Generated.wrapMemcpy L F)
    (hcode : CodeRangeOk B Generated.wrapMemcpy dst n R)
    (hsb : Sandbox cfg dst src n) (hargs : ArgsOk dst src n R)
    (hentry : CallEntry s₀ B R dst src n) :
    Exits cfg (inSection B Generated.wrapMemcpy) (wrapperPost s₀ L F R dst src n) s₀ :=
  wrapper_correct _ _ isWrapper_memcpy cfg B L F R dst src n s₀ hload hrel hleaf hfb hexit hcode
    hsb hargs hentry

/-- **`__wrap_memmove`.** Same guarantee as `wrap_memcpy_correct`, with the
fallback `__real_memmove`. -/
theorem wrap_memmove_correct (cfg : Config) (B L F R dst src n : BitVec 64) (s₀ : State)
    (hload : cfg.Loaded B Generated.wrapMemmove)
    (hrel : RelocsInRange cfg B Generated.wrapMemmove)
    (hleaf : cfg.resolve "__jolt_copy_misaligned" = L)
    (hfb : cfg.resolve "__real_memmove" = F)
    (hexit : ExitTargetsOk B Generated.wrapMemmove L F)
    (hcode : CodeRangeOk B Generated.wrapMemmove dst n R)
    (hsb : Sandbox cfg dst src n) (hargs : ArgsOk dst src n R)
    (hentry : CallEntry s₀ B R dst src n) :
    Exits cfg (inSection B Generated.wrapMemmove) (wrapperPost s₀ L F R dst src n) s₀ :=
  wrapper_correct _ _ isWrapper_memmove cfg B L F R dst src n s₀ hload hrel hleaf hfb hexit hcode
    hsb hargs hentry

/-- **`__jolt_copy_misaligned`.** Entered with mismatched offsets and
`n ≥ 544`, it performs the memmove and returns to `R`. -/
theorem jolt_copy_misaligned_correct (cfg : Config) (L R dst src n : BitVec 64) (s₀ : State)
    (hload : cfg.Loaded L Generated.joltCopyMisaligned)
    (hcode : CodeRangeOk L Generated.joltCopyMisaligned dst n R)
    (hsb : Sandbox cfg dst src n) (hargs : ArgsOk dst src n R)
    (hentry : CallEntry s₀ L R dst src n)
    (hmis : Mismatched dst src) (h544 : 544 ≤ n.toNat) :
    Exits cfg (inSection L Generated.joltCopyMisaligned) (Returned s₀ R dst src n) s₀ := by
  have ctx := Leaf.Ctx.of hload hcode.base_even
  have hreg := inSection_iff_codeRegion (base := L) Leaf.size_eq
  obtain ⟨pc, regs, mem⟩ := s₀
  have hpc : pc = L := hentry.pc
  subst hpc
  have he : EntryRegs regs dst src n R :=
    ⟨by simpa [State.reg] using hentry.a0, by simpa [State.reg] using hentry.a1,
      by simpa [State.reg] using hentry.a2, by simpa [State.reg] using hentry.ra⟩
  have hRout : ¬ Leaf.region pc R := fun h => hcode.ret_outside ((hreg R).2 h)
  exact (Leaf.leaf_path ctx regs mem he hsb hargs hRout hmis h544).congr_region
    fun x => (hreg x).symm

/-- Linking a wrapper with the leaf: with both sections loaded, a call either
completes the memmove and returns, or (for mismatched offsets and
`12 ≤ n < 544` only) tail-calls the fallback with memory and arguments
unchanged. -/
theorem wrapper_linked_correct (W : Section) (fallback : String) (hW : IsWrapper W fallback)
    (cfg : Config) (B L F R dst src n : BitVec 64) (s₀ : State)
    (hload : cfg.Loaded B W) (hrel : RelocsInRange cfg B W)
    (hleaf : cfg.resolve "__jolt_copy_misaligned" = L) (hfb : cfg.resolve fallback = F)
    (hexit : ExitTargetsOk B W L F) (hcode : CodeRangeOk B W dst n R)
    (hleafLoad : cfg.Loaded L Generated.joltCopyMisaligned)
    (hleafCode : CodeRangeOk L Generated.joltCopyMisaligned dst n R)
    (hfbLeaf : ¬ inSection L Generated.joltCopyMisaligned F)
    (hsb : Sandbox cfg dst src n) (hargs : ArgsOk dst src n R)
    (hentry : CallEntry s₀ B R dst src n) :
    Exits cfg (fun pc => inSection B W pc ∨ inSection L Generated.joltCopyMisaligned pc)
      (if Mismatched dst src ∧ 12 ≤ n.toNat ∧ n.toNat < 544 then AtFallback s₀ F
        else Returned s₀ R dst src n) s₀ := by
  have hRout : ∀ t : State, t.pc = R →
      ¬ (inSection B W t.pc ∨ inSection L Generated.joltCopyMisaligned t.pc) := by
    rintro t ht (h | h) <;> rw [ht] at h
    · exact hcode.ret_outside h
    · exact hleafCode.ret_outside h
  refine (wrapper_correct W fallback hW cfg B L F R dst src n s₀ hload hrel hleaf hfb hexit hcode
    hsb hargs hentry).bind (fun _ h => Or.inl h) fun t ht => ?_
  unfold wrapperPost at ht
  by_cases hl : Mismatched dst src ∧ 544 ≤ n.toNat
  · -- Continue into the leaf.
    rw [ite_eq_left hl] at ht
    rw [ite_eq_right (fun h => by omega)]
    have hpres : ∀ r, r ≠ 5 → t.reg r = s₀.reg r := ht.preserved
    have hentry' : CallEntry t L R dst src n :=
      ⟨ht.pc, by rw [hpres 10 (by decide), hentry.a0], by rw [hpres 11 (by decide), hentry.a1],
        by rw [hpres 12 (by decide), hentry.a2], by rw [hpres 1 (by decide), hentry.ra]⟩
    refine ((jolt_copy_misaligned_correct cfg L R dst src n t hleafLoad hleafCode hsb hargs
      hentry' hl.1 hl.2).widen (fun _ h => Or.inr h) fun u hu => hRout u hu.pc).mono
      fun u hu => Returned.of_agree hu ht.mem fun r hr => hpres r ?_
    rintro rfl
    exact hr (by decide)
  rw [ite_eq_right hl] at ht
  by_cases hf : Mismatched dst src ∧ 12 ≤ n.toNat
  · rw [ite_eq_left hf] at ht
    have h544 : n.toNat < 544 := Nat.lt_of_not_le fun h => hl ⟨hf.1, h⟩
    rw [ite_eq_left ⟨hf.1, hf.2, h544⟩]
    refine .exit ?_ ht
    rintro (h | h) <;> rw [ht.pc] at h
    · exact hexit.fallback_outside h
    · exact hfbLeaf h
  · rw [ite_eq_right hf] at ht
    rw [ite_eq_right (fun h => hf ⟨h.1, h.2.1⟩)]
    exact .exit (hRout t ht.pc) ht

/-- `__wrap_memcpy` linked with the leaf. -/
theorem wrap_memcpy_linked_correct (cfg : Config) (B L F R dst src n : BitVec 64) (s₀ : State)
    (hload : cfg.Loaded B Generated.wrapMemcpy)
    (hrel : RelocsInRange cfg B Generated.wrapMemcpy)
    (hleaf : cfg.resolve "__jolt_copy_misaligned" = L)
    (hfb : cfg.resolve "__real_memcpy" = F)
    (hexit : ExitTargetsOk B Generated.wrapMemcpy L F)
    (hcode : CodeRangeOk B Generated.wrapMemcpy dst n R)
    (hleafLoad : cfg.Loaded L Generated.joltCopyMisaligned)
    (hleafCode : CodeRangeOk L Generated.joltCopyMisaligned dst n R)
    (hfbLeaf : ¬ inSection L Generated.joltCopyMisaligned F)
    (hsb : Sandbox cfg dst src n) (hargs : ArgsOk dst src n R)
    (hentry : CallEntry s₀ B R dst src n) :
    Exits cfg
      (fun pc => inSection B Generated.wrapMemcpy pc ∨ inSection L Generated.joltCopyMisaligned pc)
      (if Mismatched dst src ∧ 12 ≤ n.toNat ∧ n.toNat < 544 then AtFallback s₀ F
        else Returned s₀ R dst src n) s₀ :=
  wrapper_linked_correct _ _ isWrapper_memcpy cfg B L F R dst src n s₀ hload hrel hleaf hfb hexit
    hcode hleafLoad hleafCode hfbLeaf hsb hargs hentry

/-- `__wrap_memmove` linked with the leaf. -/
theorem wrap_memmove_linked_correct (cfg : Config) (B L F R dst src n : BitVec 64) (s₀ : State)
    (hload : cfg.Loaded B Generated.wrapMemmove)
    (hrel : RelocsInRange cfg B Generated.wrapMemmove)
    (hleaf : cfg.resolve "__jolt_copy_misaligned" = L)
    (hfb : cfg.resolve "__real_memmove" = F)
    (hexit : ExitTargetsOk B Generated.wrapMemmove L F)
    (hcode : CodeRangeOk B Generated.wrapMemmove dst n R)
    (hleafLoad : cfg.Loaded L Generated.joltCopyMisaligned)
    (hleafCode : CodeRangeOk L Generated.joltCopyMisaligned dst n R)
    (hfbLeaf : ¬ inSection L Generated.joltCopyMisaligned F)
    (hsb : Sandbox cfg dst src n) (hargs : ArgsOk dst src n R)
    (hentry : CallEntry s₀ B R dst src n) :
    Exits cfg
      (fun pc => inSection B Generated.wrapMemmove pc ∨
        inSection L Generated.joltCopyMisaligned pc)
      (if Mismatched dst src ∧ 12 ≤ n.toNat ∧ n.toNat < 544 then AtFallback s₀ F
        else Returned s₀ R dst src n) s₀ :=
  wrapper_linked_correct _ _ isWrapper_memmove cfg B L F R dst src n s₀ hload hrel hleaf hfb hexit
    hcode hleafLoad hleafCode hfbLeaf hsb hargs hentry

end CopyRuntime.Theorems
