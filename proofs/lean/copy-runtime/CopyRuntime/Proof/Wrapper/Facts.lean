import CopyRuntime.Proof.Wrapper.Setup

/-!
# Bit-level facts behind the wrapper's dispatch tests
-/

namespace CopyRuntime.Wrapper

theorem toNat_mod_eight (x : BitVec 64) : x.toNat % 8 = (x &&& 7#64).toNat := by
  rw [BitVec.toNat_and]
  simp [Nat.and_two_pow_sub_one_eq_mod _ 3]

/-- `Mismatched` is the `xor`/`andi` test of `.Lunaligned`. -/
theorem mismatched_iff (dst src : BitVec 64) :
    Mismatched dst src ↔ (dst ^^^ src) &&& 7#64 ≠ 0#64 := by
  unfold Mismatched
  rw [toNat_mod_eight]
  constructor
  · intro h h2; apply h; rw [show ((dst - src) &&& 7#64) = 0#64 by bv_decide]; rfl
  · intro h h2; apply h
    have : (dst - src) &&& 7#64 = 0#64 := BitVec.eq_of_toNat_eq (by simpa using h2)
    bv_decide

/-- Mismatched offsets fail the all-aligned test. -/
theorem or_and_seven_ne_zero_of_mismatched {dst src : BitVec 64} (h : Mismatched dst src)
    (n : BitVec 64) : (dst ||| src ||| n) &&& 7#64 ≠ 0#64 := by
  have := (mismatched_iff dst src).1 h
  bv_decide

/-- Lengths with a nonzero low three bits fail the all-aligned test. -/
theorem or_and_seven_ne_zero_of_len {n : BitVec 64} (h : n.toNat % 8 ≠ 0) (dst src : BitVec 64) :
    (dst ||| src ||| n) &&& 7#64 ≠ 0#64 := by
  rw [toNat_mod_eight] at h
  have : n &&& 7#64 ≠ 0#64 := fun h' => h (by rw [h']; rfl)
  bv_decide

/-- The all-aligned test passes exactly when all three values are multiples of 8. -/
theorem or_and_seven_eq_zero_iff (dst src n : BitVec 64) :
    (dst ||| src ||| n) &&& 7#64 = 0#64 ↔
      dst.toNat % 8 = 0 ∧ src.toNat % 8 = 0 ∧ n.toNat % 8 = 0 := by
  rw [toNat_mod_eight, toNat_mod_eight, toNat_mod_eight]
  constructor
  · intro h
    refine ⟨?_, ?_, ?_⟩
    · rw [show dst &&& 7#64 = 0#64 by bv_decide]; rfl
    · rw [show src &&& 7#64 = 0#64 by bv_decide]; rfl
    · rw [show n &&& 7#64 = 0#64 by bv_decide]; rfl
  · intro ⟨h1, h2, h3⟩
    have e1 : dst &&& 7#64 = 0#64 := BitVec.eq_of_toNat_eq (by simpa using h1)
    have e2 : src &&& 7#64 = 0#64 := BitVec.eq_of_toNat_eq (by simpa using h2)
    have e3 : n &&& 7#64 = 0#64 := BitVec.eq_of_toNat_eq (by simpa using h3)
    bv_decide

/-- The PC-relative pair `auipc`/`jalr` of `R_RISCV_CALL_PLT` reaches its target. -/
@[rv_exec] theorem call_target (p F : BitVec 64) :
    p + (F - p - lo12 (F - p)) + lo12 (F - p) = F := by
  bv_omega

@[rv_exec] theorem add_sub_cancel' (x y : BitVec 64) : x + (y - x) = y := by
  bv_omega

end CopyRuntime.Wrapper
