import CopyRuntime.Proof.Wrapper.General

/-!
# Wrapper dispatch: every argument combination reaches one of the path lemmas
-/

namespace CopyRuntime.Wrapper

variable {cfg : Config} {B L F : BitVec 64}

/-- Transfer a `Returned` fact from an intermediate state whose register file
differs from the entry state only in `t0`. -/
theorem Returned.of_t0 {p₀ p₁ : BitVec 64} {r : Reg → BitVec 64} {m : Mem} {v : BitVec 64}
    {t : State} {R dst src n : BitVec 64}
    (h : Returned ⟨p₁, writeReg r 5 v, m⟩ R dst src n t) : Returned ⟨p₀, r, m⟩ R dst src n t :=
  Returned.of_agree h rfl fun i hi => by
    have : i ≠ 5 := fun h5 => hi (by subst h5; decide)
    simp only [State.reg_mk, writeReg_apply, this, ite_false]

theorem entryRegs_t0 {regs : Reg → BitVec 64} {dst src n R v : BitVec 64}
    (he : EntryRegs regs dst src n R) : EntryRegs (writeReg regs 5 v) dst src n R :=
  ⟨by simp [writeReg_apply, he.a0], by simp [writeReg_apply, he.a1],
    by simp [writeReg_apply, he.a2], by simp [writeReg_apply, he.ra]⟩

theorem not_mismatched {dst src : BitVec 64} (h : ¬ Mismatched dst src) :
    (dst - src).toNat % 8 = 0 := by
  unfold Mismatched at h; omega

/-- From the entry point to `.Lgeneral` for equal offsets: either through the
aligned test with `n > 88`, or through `.Lunaligned` with `n ≥ 8`. -/
theorem general_entry (ctx : Ctx cfg B L F) (regs : Reg → BitVec 64) (mem : Mem)
    {dst src n R : BitVec 64} (he : EntryRegs regs dst src n R) (hsb : Sandbox cfg dst src n)
    (hargs : ArgsOk dst src n R) (hRout : ¬ region B R) (hsame : ¬ Mismatched dst src)
    (hroute : ((dst ||| src ||| n) &&& 7#64 = 0#64 ∧ 88 < n.toNat) ∨
      ((dst ||| src ||| n) &&& 7#64 ≠ 0#64 ∧ 8 ≤ n.toNat)) :
    Exits cfg (region B) (Returned ⟨B, regs, mem⟩ R dst src n) ⟨B, regs, mem⟩ := by
  have hB := ctx.base_even
  have ha0 := he.a0; have ha1 := he.a1; have ha2 := he.a2
  have hd := not_mismatched hsame
  rcases hroute with ⟨hal, h88⟩ | ⟨hun, h8⟩
  · iterate 7 rv_step ctx.code [ha0, ha1, ha2, hal]
    exact (general_path ctx _ mem (entryRegs_t0 he) hsb hargs hRout hd (by omega)).mono
      fun _ h => Returned.of_t0 h
  · have hx : (dst ^^^ src) &&& 7#64 = 0#64 :=
      Classical.not_not.1 fun h => hsame ((mismatched_iff dst src).2 h)
    iterate 10 rv_step ctx.code [ha0, ha1, ha2, hun, hx]
    exact (general_path ctx _ mem (entryRegs_t0 he) hsb hargs hRout hd h8).mono
      fun _ h => Returned.of_t0 h

/-- The wrapper, from its entry point, ends in the outcome `wrapperPost`
prescribes. -/
theorem dispatch (ctx : Ctx cfg B L F) (regs : Reg → BitVec 64) (mem : Mem)
    {dst src n R : BitVec 64} (he : EntryRegs regs dst src n R) (hsb : Sandbox cfg dst src n)
    (hargs : ArgsOk dst src n R) (hRout : ¬ region B R) :
    Exits cfg (region B) (wrapperPost ⟨B, regs, mem⟩ L F R dst src n) ⟨B, regs, mem⟩ := by
  unfold wrapperPost
  by_cases hmis : Mismatched dst src
  · by_cases h544 : 544 ≤ n.toNat
    · rw [ite_eq_left ⟨hmis, h544⟩]
      exact leaf_path ctx regs mem he hmis h544
    rw [ite_eq_right (fun h => h544 h.2)]
    by_cases h12 : 12 ≤ n.toNat
    · rw [ite_eq_left ⟨hmis, h12⟩]
      exact fallback_path ctx regs mem he hmis h12 (by omega)
    rw [ite_eq_right (fun h => h12 h.2)]
    by_cases h0 : n = 0
    · subst h0
      exact zero_path ctx regs mem he hargs.ret_even hRout
    · have : n.toNat ≠ 0 := fun h => h0 (BitVec.eq_of_toNat_eq h)
      exact small_bytes_path ctx regs mem he hsb hargs hRout (by omega)
        (Or.inr ⟨hmis, by omega⟩)
  · rw [ite_eq_right (fun h => hmis h.1), ite_eq_right (fun h => hmis h.1)]
    by_cases h0 : n = 0
    · subst h0
      exact zero_path ctx regs mem he hargs.ret_even hRout
    have hn0 : n.toNat ≠ 0 := fun h => h0 (BitVec.eq_of_toNat_eq h)
    by_cases hal : (dst ||| src ||| n) &&& 7#64 = 0#64
    · obtain ⟨hd, hs, hn⟩ := (or_and_seven_eq_zero_iff dst src n).1 hal
      by_cases h88 : n.toNat ≤ 88
      · exact aligned_small_path ctx regs mem he hsb hargs hRout hd hs hn (by omega) h88
      · exact general_entry ctx regs mem he hsb hargs hRout hmis (Or.inl ⟨hal, by omega⟩)
    · by_cases h7 : n.toNat ≤ 7
      · exact small_bytes_path ctx regs mem he hsb hargs hRout (by omega) (Or.inl h7)
      · exact general_entry ctx regs mem he hsb hargs hRout hmis (Or.inr ⟨hal, by omega⟩)

end CopyRuntime.Wrapper
