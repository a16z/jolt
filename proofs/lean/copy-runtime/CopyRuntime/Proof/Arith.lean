/-!
# 64-bit address arithmetic

Rewrite rules that bring symbolic addresses to the normal form `x + c` (one
symbolic atom plus a literal), and facts relating `BitVec 64` and `Nat`.

`simp` evaluates a negative immediate to a literal `c ≥ 2^63` (`-80` becomes
`0xffffffffffffffb0#64`). With a symbolic `x`, `(x + c).toNat` must not reach
`BitVec.toNat_add` (see "Performance pitfalls" in the README), so such sums are
kept as `x - c'` (`add_ofNat_eq_sub_of_ge`) or folded into the offset
(`ofNat_add_neg_lit`, `ofNat_sub_ofNat_of_le`) with a small `c'`.
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

theorem add_ofNat_congr {x : BitVec 64} {a b : Nat} (h : a = b) :
    x + BitVec.ofNat 64 a = x + BitVec.ofNat 64 b := h ▸ rfl

theorem add_ofNat_ofInt_neg (x : BitVec 64) {a k : Nat} (hk : k ≤ a) :
    x + BitVec.ofNat 64 a + BitVec.ofInt 64 (-(k : Int)) = x + BitVec.ofNat 64 (a - k) := by
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.toNat_add, BitVec.toNat_ofNat, BitVec.toNat_ofInt]
  omega

theorem add_ofNat_eq_sub_of_ge (x : BitVec 64) (c : Nat) (h : 2 ^ 63 ≤ c) (h' : c < 2 ^ 64) :
    x + BitVec.ofNat 64 c = x - BitVec.ofNat 64 (2 ^ 64 - c) := by
  rw [BitVec.sub_eq_add_neg]
  congr 1
  apply BitVec.eq_of_toNat_eq
  rw [BitVec.toNat_neg, BitVec.toNat_ofNat, BitVec.toNat_ofNat, Nat.mod_eq_of_lt h',
    Nat.mod_eq_of_lt (by omega : 2 ^ 64 - c < 2 ^ 64), Nat.mod_eq_of_lt (by omega)]
  omega

theorem toNat_sub_ofNat_of_le {x : BitVec 64} {c : Nat} (hc : c ≤ x.toNat) :
    (x - BitVec.ofNat 64 c).toNat = x.toNat - c := by
  have := x.isLt
  rw [BitVec.toNat_sub_of_le (by rw [BitVec.le_def, BitVec.toNat_ofNat]; omega),
    BitVec.toNat_ofNat, Nat.mod_eq_of_lt (by omega)]

theorem sub_ofNat_sub_ofNat (x : BitVec 64) (a b : Nat) :
    x - BitVec.ofNat 64 a - BitVec.ofNat 64 b = x - BitVec.ofNat 64 (a + b) := by
  rw [BitVec.sub_sub, BitVec.ofNat_add]

theorem ofNat_add_neg_lit {a b : Nat} (hb : 9223372036854775808 ≤ b)
    (hb' : b < 18446744073709551616) (hab : 18446744073709551616 - b ≤ a) :
    BitVec.ofNat 64 (a + b) = BitVec.ofNat 64 (a - (18446744073709551616 - b)) := by
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.toNat_ofNat]
  omega

/-- `BitVec.ofNat_sub_ofNat_of_le` without its `k < 2^64` side condition. -/
theorem ofNat_sub_ofNat_of_le {r k : Nat} (h : k ≤ r) :
    BitVec.ofNat 64 r - BitVec.ofNat 64 k = BitVec.ofNat 64 (r - k) := by
  rw [show r = (r - k) + k by omega, BitVec.ofNat_add, BitVec.add_sub_cancel,
    show r - k + k - k = r - k by omega]

theorem toNat_ofNat_of_lt {h : Nat} (hh : h < 2 ^ 64) : (BitVec.ofNat 64 h).toNat = h := by
  rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt hh]

theorem eq_ofNat_of_toNat_eq {x : BitVec 64} {h : Nat} (hx : x.toNat = h) :
    x = BitVec.ofNat 64 h := by
  apply BitVec.eq_of_toNat_eq
  rw [BitVec.toNat_ofNat, ← hx, Nat.mod_eq_of_lt x.isLt]

theorem toNat_and_seven (x : BitVec 64) : (x &&& 7#64).toNat = x.toNat % 8 := by
  rw [BitVec.toNat_and]
  exact Nat.and_two_pow_sub_one_eq_mod _ 3

theorem and_seven_eq_zero_iff (x : BitVec 64) : x &&& 7#64 = 0#64 ↔ x.toNat % 8 = 0 := by
  rw [← toNat_and_seven]
  constructor
  · intro h; rw [h]; rfl
  · intro h; exact BitVec.eq_of_toNat_eq h

private theorem and_neg8_eq (x : BitVec 64) : x &&& 0xfffffffffffffff8#64 = (x >>> 3) <<< 3 := by
  ext i hi
  have : (0xfffffffffffffff8#64 : BitVec 64) = ~~~7#64 := by decide
  rw [this]
  simp only [BitVec.getElem_and, BitVec.getElem_not, BitVec.getElem_shiftLeft,
    BitVec.getElem_ushiftRight]
  by_cases h3 : i < 3
  · have : (7#64)[i] = true := by
      rw [← BitVec.getLsbD_eq_getElem]
      rcases i with _ | _ | _ | i <;> first | omega | decide
    simp [h3, this]
  · have : (7#64)[i] = false := by
      rw [← BitVec.getLsbD_eq_getElem, BitVec.getLsbD_ofNat]
      simp only [Nat.testBit_eq_decide_div_mod_eq]
      have : 7 / 2 ^ i = 0 := Nat.div_eq_of_lt
        (Nat.lt_of_lt_of_le (by decide : 7 < 2 ^ 3) (Nat.pow_le_pow_right (by decide) (by omega)))
      simp [this]
    simp [h3, this]
    congr 1; omega

/-- `andi rd, rs, -8`. -/
theorem toNat_and_neg8 (x : BitVec 64) :
    (x &&& 0xfffffffffffffff8#64).toNat = x.toNat / 8 * 8 := by
  rw [and_neg8_eq, BitVec.toNat_shiftLeft, BitVec.toNat_ushiftRight, Nat.shiftLeft_eq,
    Nat.shiftRight_eq_div_pow]
  have := x.isLt
  rw [Nat.mod_eq_of_lt (by omega)]

end CopyRuntime.Arith
