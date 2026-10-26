import CopyRuntime.Proof.Arith
import CopyRuntime.Spec
import Std.Tactic.BVDecide

/-!
# Little-endian memory lemmas and the `Copied` invariant

`Copied m m₀ dst src lo hi` says that `m` is the snapshot copy of `m₀` on the
destination offsets `[lo, hi)`: every byte `x` whose offset `(x - dst)` lies
in `[lo, hi)` holds `m₀ (src + (x - dst))`, and every other byte is as in
`m₀`. Copy loops extend the interval one chunk at a time (`extend_down`,
`extend_up`), and `Copied m m₀ dst src 0 n` is exactly the memmove result.
-/

namespace CopyRuntime

/-! ## Bytes of a little-endian load -/

private theorem byte0_of_or (b : BitVec 8) (L : BitVec 64) :
    ((b.setWidth 64 ||| (L <<< 8)) >>> 0).setWidth 8 = b := by
  bv_decide

private theorem byte_succ_of_or (i : Nat) (hi : i < 7) (b : BitVec 8) (L : BitVec 64) :
    ((b.setWidth 64 ||| (L <<< 8)) >>> (8 * (i + 1))).setWidth 8 = (L >>> (8 * i)).setWidth 8 := by
  rcases i with _ | _ | _ | _ | _ | _ | _ | i
  all_goals first | omega | bv_decide

/-- Byte `i` of a `k`-byte little-endian load is the byte at `a + i`. -/
theorem loadLE_byte (m : Mem) : ∀ (k : Nat) (a : BitVec 64) (i : Nat), k ≤ 8 → i < k →
    (loadLE m a k >>> (8 * i)).setWidth 8 = m (a + BitVec.ofNat 64 i)
  | 0, _, _, _, hi => absurd hi (Nat.not_lt_zero _)
  | k + 1, a, 0, _, _ => by
    simp only [loadLE, Nat.mul_zero, BitVec.ofNat_eq_ofNat, BitVec.add_zero]
    exact byte0_of_or _ _
  | k + 1, a, i + 1, hk, hi => by
    simp only [loadLE]
    rw [byte_succ_of_or i (by omega), loadLE_byte m k (a + 1) i (by omega) (by omega)]
    congr 1
    bv_omega

/-- Storing (the low `k` bytes of) a `k`-byte load copies those bytes. -/
theorem storeLE_loadLE_apply (m m₀ : Mem) (a b : BitVec 64) (k : Nat) (hk : k ≤ 8)
    (x : BitVec 64) :
    storeLE m a (loadLE m₀ b k) k x = if (x - a).toNat < k then m₀ (b + (x - a)) else m x := by
  unfold storeLE
  split
  · rename_i h
    rw [loadLE_byte m₀ k b _ hk h]
    congr 2
    apply BitVec.eq_of_toNat_eq
    simp only [BitVec.toNat_ofNat]
    exact Nat.mod_eq_of_lt (by omega)
  · rfl

theorem storeLE_apply_of_ge {m : Mem} {a v x : BitVec 64} {k : Nat} (h : k ≤ (x - a).toNat) :
    storeLE m a v k x = m x := by
  unfold storeLE
  rw [ite_eq_right (by omega)]

/-- A load only depends on the bytes it reads. -/
theorem loadLE_congr {m m' : Mem} {b : BitVec 64} :
    ∀ {k : Nat}, (∀ i, i < k → m (b + BitVec.ofNat 64 i) = m' (b + BitVec.ofNat 64 i)) →
      loadLE m b k = loadLE m' b k
  | 0, _ => rfl
  | k + 1, h => by
    have h0 := h 0 (by omega)
    simp only [BitVec.add_zero] at h0
    simp only [loadLE, h0]
    congr 2
    apply loadLE_congr
    intro i hi
    have := h (i + 1) (by omega)
    have e : b + BitVec.ofNat 64 (i + 1) = b + 1 + BitVec.ofNat 64 i := by bv_omega
    rwa [e] at this

/-! ## Snapshot copies -/

/-- `m` agrees with the snapshot copy of `m₀` on destination offsets `[lo, hi)`
and with `m₀` elsewhere. -/
def Copied (m m₀ : Mem) (dst src : BitVec 64) (lo hi : Nat) : Prop :=
  ∀ x, m x = if lo ≤ (x - dst).toNat ∧ (x - dst).toNat < hi then m₀ (src + (x - dst)) else m₀ x

theorem Copied.empty (m₀ : Mem) (dst src : BitVec 64) (k : Nat) : Copied m₀ m₀ dst src k k := by
  intro x
  rw [ite_eq_right (by omega)]

private theorem sub_add_lit (x d : BitVec 64) (c : Nat) :
    x - (d + BitVec.ofNat 64 c) = x - d - BitVec.ofNat 64 c := by
  rw [BitVec.sub_sub]

private theorem toNat_sub_lit {y : BitVec 64} {c : Nat} (hc : c < 2 ^ 64) :
    (y - BitVec.ofNat 64 c).toNat =
      if c ≤ y.toNat then y.toNat - c else 2 ^ 64 - c + y.toNat := by
  rw [BitVec.toNat_sub, BitVec.toNat_ofNat, Nat.mod_eq_of_lt hc]
  have := y.isLt
  split <;> omega

private theorem ite_ite_same {α : Sort _} {P Q R : Prop} [Decidable P] [Decidable Q]
    [Decidable R] {a m : α} (h : R ↔ P ∨ Q) :
    (if P then a else if Q then a else m) = if R then a else m := by
  by_cases hP : P <;> by_cases hQ : Q <;> simp_all

/-- Prepend the chunk `[c, c + w)` (copied from the snapshot) to `[c + w, hi)`. -/
theorem Copied.extend_down {m m₀ : Mem} {dst src : BitVec 64} {c w hi : Nat}
    (h : Copied m m₀ dst src (c + w) hi) (hw : w ≤ 8) (hhi : hi < 2 ^ 64) (hcw : c + w ≤ hi) :
    Copied (storeLE m (dst + BitVec.ofNat 64 c) (loadLE m₀ (src + BitVec.ofNat 64 c) w) w)
      m₀ dst src c hi := by
  intro x
  rw [storeLE_loadLE_apply _ _ _ _ _ hw, h x, sub_add_lit, toNat_sub_lit (by omega)]
  have hsrc : src + BitVec.ofNat 64 c + (x - dst - BitVec.ofNat 64 c) = src + (x - dst) := by
    bv_omega
  rw [hsrc]
  have := (x - dst).isLt
  apply ite_ite_same
  split <;> constructor <;> intro <;> omega

/-- Append the chunk `[hi, hi + w)` (copied from the snapshot) to `[lo, hi)`. -/
theorem Copied.extend_up {m m₀ : Mem} {dst src : BitVec 64} {lo hi w : Nat}
    (h : Copied m m₀ dst src lo hi) (hw : w ≤ 8) (hhi : hi + w < 2 ^ 64) (hlo : lo ≤ hi) :
    Copied (storeLE m (dst + BitVec.ofNat 64 hi) (loadLE m₀ (src + BitVec.ofNat 64 hi) w) w)
      m₀ dst src lo (hi + w) := by
  intro x
  rw [storeLE_loadLE_apply _ _ _ _ _ hw, h x, sub_add_lit, toNat_sub_lit (by omega)]
  have hsrc : src + BitVec.ofNat 64 hi + (x - dst - BitVec.ofNat 64 hi) = src + (x - dst) := by
    bv_omega
  rw [hsrc]
  have := (x - dst).isLt
  apply ite_ite_same
  split <;> constructor <;> intro <;> omega

theorem Copied.extend_down_zero {m m₀ : Mem} {dst src : BitVec 64} {w hi : Nat}
    (h : Copied m m₀ dst src w hi) (hw : w ≤ 8) (hhi : hi < 2 ^ 64) (hcw : w ≤ hi) :
    Copied (storeLE m dst (loadLE m₀ src w) w) m₀ dst src 0 hi := by
  have := Copied.extend_down (c := 0) (by simpa using h) hw hhi (by simpa using hcw)
  simpa using this

theorem inRange_iff_toNat_sub {dst n x : BitVec 64} (hwrap : dst.toNat + n.toNat ≤ 2 ^ 64) :
    InRange dst n x ↔ (x - dst).toNat < n.toNat := by
  unfold InRange
  rw [BitVec.toNat_sub]
  have := x.isLt
  have := dst.isLt
  constructor
  · intro ⟨h1, h2⟩
    rw [show 2 ^ 64 - dst.toNat + x.toNat = (x.toNat - dst.toNat) + 2 ^ 64 by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
    omega
  · intro h
    by_cases hle : dst.toNat ≤ x.toNat
    · rw [show 2 ^ 64 - dst.toNat + x.toNat = (x.toNat - dst.toNat) + 2 ^ 64 by omega,
        Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)] at h
      omega
    · rw [Nat.mod_eq_of_lt (by omega)] at h
      omega

/-- A copy of all of `[0, n)` is the memmove result. -/
theorem Copied.memmove {m m₀ : Mem} {dst src n : BitVec 64}
    (h : Copied m m₀ dst src 0 n.toNat) (hwrap : dst.toNat + n.toNat ≤ 2 ^ 64) :
    m = memmoveResult m₀ dst src n := by
  funext x
  rw [h x, memmoveResult]
  simp only [Nat.zero_le, true_and, inRange_iff_toNat_sub hwrap]

theorem memmoveResult_zero (m : Mem) (dst src : BitVec 64) : memmoveResult m dst src 0 = m := by
  funext x
  unfold memmoveResult
  rw [ite_eq_right]
  intro ⟨_, h⟩
  have : (0 : BitVec 64).toNat = 0 := rfl
  omega

/-! ## Sandbox permissions -/

theorem toNat_sub_add_lit {a base : BitVec 64} {i : Nat} (h : (a - base).toNat + i < 2 ^ 64) :
    (a + BitVec.ofNat 64 i - base).toNat = (a - base).toNat + i := by
  have e : a + BitVec.ofNat 64 i - base = (a - base) + BitVec.ofNat 64 i := by bv_omega
  rw [e, BitVec.toNat_add, BitVec.toNat_ofNat, Nat.mod_eq_of_lt (a := i) (by omega),
    Nat.mod_eq_of_lt h]

/-- An aligned access inside `[base, base + len)` is permitted by `rangePerm`. -/
theorem accessOk_rangePerm {base len a : BitVec 64} {size : Nat}
    (hwrap : base.toNat + len.toNat ≤ 2 ^ 64) (hin : (a - base).toNat + size ≤ len.toNat)
    (halign : a.toNat % size = 0) : accessOk (rangePerm base len) a size = true := by
  unfold accessOk
  simp only [halign, beq_self_eq_true, Bool.true_and, List.all_eq_true, List.mem_range]
  intro i hi
  unfold rangePerm
  rw [decide_eq_true_eq, inRange_iff_toNat_sub hwrap, toNat_sub_add_lit (by omega)]
  omega

end CopyRuntime
