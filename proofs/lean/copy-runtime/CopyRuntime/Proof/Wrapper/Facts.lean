import CopyRuntime.Proof.Wrapper.Setup

/-!
# Bit-level facts behind the wrapper's dispatch tests
-/

namespace CopyRuntime.Wrapper

theorem toNat_and_seven (x : BitVec 64) : (x &&& 7#64).toNat = x.toNat % 8 := by
  rw [BitVec.toNat_and]
  exact Nat.and_two_pow_sub_one_eq_mod _ 3

theorem and_seven_eq_zero_iff (x : BitVec 64) : x &&& 7#64 = 0#64 ↔ x.toNat % 8 = 0 := by
  rw [← toNat_and_seven]
  constructor
  · intro h; rw [h]; rfl
  · intro h; exact BitVec.eq_of_toNat_eq h

theorem or_and_seven (a b : BitVec 64) :
    (a ||| b) &&& 7#64 = (a &&& 7#64) ||| (b &&& 7#64) := by
  ext i hi; simp; cases a[i] <;> cases b[i] <;> simp

theorem xor_and_seven (a b : BitVec 64) :
    (a ^^^ b) &&& 7#64 = (a &&& 7#64) ^^^ (b &&& 7#64) := by
  ext i hi; simp; cases a[i] <;> cases b[i] <;> simp

theorem or_eq_zero (a b : BitVec 64) : a ||| b = 0#64 ↔ a = 0#64 ∧ b = 0#64 := by
  constructor
  · intro h
    constructor <;> (ext i hi; have := congrArg (·[i]) h; simp at this; simp [this])
  · rintro ⟨rfl, rfl⟩; rfl

theorem xor_eq_zero (a b : BitVec 64) : a ^^^ b = 0#64 ↔ a = b := by
  constructor
  · intro h; ext i hi; have := congrArg (·[i]) h; simp at this; exact this
  · rintro rfl; simp

/-- `Mismatched` is the `xor`/`andi` test of `.Lunaligned`. -/
theorem mismatched_iff (dst src : BitVec 64) :
    Mismatched dst src ↔ (dst ^^^ src) &&& 7#64 ≠ 0#64 := by
  rw [xor_and_seven, ne_eq, xor_eq_zero]
  unfold Mismatched
  have hd := toNat_and_seven dst
  have hs := toNat_and_seven src
  have : dst &&& 7#64 = src &&& 7#64 ↔ dst.toNat % 8 = src.toNat % 8 := by
    constructor
    · intro h; rw [← hd, ← hs, h]
    · intro h; apply BitVec.eq_of_toNat_eq; rw [hd, hs, h]
  rw [this, BitVec.toNat_sub]
  have := dst.isLt; have := src.isLt
  omega

/-- The all-aligned test passes exactly when all three values are multiples of 8. -/
theorem or_and_seven_eq_zero_iff (dst src n : BitVec 64) :
    (dst ||| src ||| n) &&& 7#64 = 0#64 ↔
      dst.toNat % 8 = 0 ∧ src.toNat % 8 = 0 ∧ n.toNat % 8 = 0 := by
  rw [or_and_seven, or_and_seven, or_eq_zero, or_eq_zero, and_seven_eq_zero_iff,
    and_seven_eq_zero_iff, and_seven_eq_zero_iff, and_assoc]

/-- Mismatched offsets fail the all-aligned test. -/
theorem or_and_seven_ne_zero_of_mismatched {dst src : BitVec 64} (h : Mismatched dst src)
    (n : BitVec 64) : (dst ||| src ||| n) &&& 7#64 ≠ 0#64 := by
  rw [ne_eq, or_and_seven_eq_zero_iff]
  intro ⟨hd, hs, _⟩
  apply h
  rw [BitVec.toNat_sub]
  have := dst.isLt; have := src.isLt
  omega

/-- Lengths with a nonzero low three bits fail the all-aligned test. -/
theorem or_and_seven_ne_zero_of_len {n : BitVec 64} (h : n.toNat % 8 ≠ 0) (dst src : BitVec 64) :
    (dst ||| src ||| n) &&& 7#64 ≠ 0#64 := by
  rw [ne_eq, or_and_seven_eq_zero_iff]
  exact fun ⟨_, _, hn⟩ => h hn

/-- The PC-relative pair `auipc`/`jalr` of `R_RISCV_CALL_PLT` reaches its target. -/
@[rv_exec] theorem call_target (p F : BitVec 64) :
    p + (F - p - lo12 (F - p)) + lo12 (F - p) = F := by
  bv_omega

@[rv_exec] theorem add_sub_cancel' (x y : BitVec 64) : x + (y - x) = y := by
  bv_omega

end CopyRuntime.Wrapper
