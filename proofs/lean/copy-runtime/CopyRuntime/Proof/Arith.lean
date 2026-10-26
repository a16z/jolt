/-!
# 64-bit address arithmetic

Rewrite rules that bring symbolic addresses to the normal form `x + c` (one
symbolic atom plus a literal) and facts relating `BitVec 64` and `Nat`.
-/

namespace CopyRuntime.Arith

theorem add_sub_cancel_left (x y : BitVec 64) : x + y - x = y := by
  rw [BitVec.add_comm, BitVec.add_sub_cancel]

theorem add_sub_add_left (x a b : BitVec 64) : x + a - (x + b) = a - b := by
  bv_omega

theorem add_lit_lit (x : BitVec 64) (a b : Nat) :
    x + BitVec.ofNat 64 a + BitVec.ofNat 64 b = x + BitVec.ofNat 64 (a + b) := by
  rw [BitVec.add_assoc, BitVec.ofNat_add]

theorem add_lit_sub_lit (x : BitVec 64) (a b : Nat) :
    x + BitVec.ofNat 64 a - BitVec.ofNat 64 b = x + (BitVec.ofNat 64 a - BitVec.ofNat 64 b) := by
  rw [BitVec.sub_eq_add_neg, BitVec.sub_eq_add_neg, BitVec.add_assoc]

theorem toNat_add_lit_sub (x : BitVec 64) {k : Nat} (hk : k < 2 ^ 64) :
    (x + BitVec.ofNat 64 k - x).toNat = k := by
  rw [add_sub_cancel_left, BitVec.toNat_ofNat, Nat.mod_eq_of_lt hk]

end CopyRuntime.Arith
