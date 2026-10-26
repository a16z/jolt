import CopyRuntime.Proof.Arith
import CopyRuntime.Spec

/-!
# Little-endian memory lemmas and the `Copied` invariant

`Copied m m₀ dst src lo hi` says that `m` is the snapshot copy of `m₀` on the
destination offsets `[lo, hi)`: every byte `x` whose offset `(x - dst)` lies
in `[lo, hi)` holds `m₀ (src + (x - dst))`, and every other byte is as in
`m₀`. Copy loops extend the interval one chunk at a time, and
`Copied m m₀ dst src 0 n` is exactly the memmove result (`Copied.memmove`).

A chunk may be loaded from the current memory rather than from `m₀` when no
earlier store has reached its source bytes. For a forward copy (`dst ≤ src`)
the copied part is a prefix `[0, k)` and loads read at offsets `≥ k`
(`Copied.read_fwd`); for a backward copy (`src ≤ dst`) it is a suffix
`[lo, hi)` and loads read below `lo` (`Copied.read_bwd`).
-/

namespace CopyRuntime

private theorem byte0_of_or (b : BitVec 8) (L : BitVec 64) :
    ((b.setWidth 64 ||| (L <<< 8)) >>> 0).setWidth 8 = b := by
  ext j hj
  simp

private theorem byte_succ_of_or (i : Nat) (hi : i < 7) (b : BitVec 8) (L : BitVec 64) :
    ((b.setWidth 64 ||| (L <<< 8)) >>> (8 * (i + 1))).setWidth 8 = (L >>> (8 * i)).setWidth 8 := by
  ext j hj
  have h1 : 8 * (i + 1) + j < 64 := by omega
  have h2 : ¬ 8 * (i + 1) + j < 8 := by omega
  have h3 : 8 * (i + 1) + j - 8 = 8 * i + j := by omega
  simp [h1, h2, h3, BitVec.getLsbD_of_ge b _ (show 8 ≤ 8 * (i + 1) + j by omega)]
  rw [BitVec.getLsbD_eq_getElem (by omega)]

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

def Copied (m m₀ : Mem) (dst src : BitVec 64) (lo hi : Nat) : Prop :=
  ∀ x, m x = if lo ≤ (x - dst).toNat ∧ (x - dst).toNat < hi then m₀ (src + (x - dst)) else m₀ x

/-- `m` agrees with the snapshot copy of `m₀` on destination offsets `[lo, hi)`
and with `base` elsewhere: a block being stored over the copy `base`.
`Copied m m₀` is `CopiedOver m m₀ m₀` by definition. -/
def CopiedOver (m base m₀ : Mem) (dst src : BitVec 64) (lo hi : Nat) : Prop :=
  ∀ x, m x = if lo ≤ (x - dst).toNat ∧ (x - dst).toNat < hi then m₀ (src + (x - dst)) else base x

theorem Copied.empty (m₀ : Mem) (dst src : BitVec 64) (k : Nat) : Copied m₀ m₀ dst src k k := by
  intro x
  rw [ite_eq_right (by omega)]

theorem Copied.congr_lo {m m₀ : Mem} {dst src : BitVec 64} {a b hi : Nat}
    (h : Copied m m₀ dst src a hi) (e : a = b) : Copied m m₀ dst src b hi := e ▸ h

theorem Copied.congr_hi {m m₀ : Mem} {dst src : BitVec 64} {lo a b : Nat}
    (h : Copied m m₀ dst src lo a) (e : a = b) : Copied m m₀ dst src lo b := e ▸ h

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
theorem CopiedOver.extend_up {m base m₀ : Mem} {dst src : BitVec 64} {lo hi w : Nat}
    (h : CopiedOver m base m₀ dst src lo hi) (hw : w ≤ 8) (hhi : hi + w < 2 ^ 64)
    (hlo : lo ≤ hi) :
    CopiedOver (storeLE m (dst + BitVec.ofNat 64 hi) (loadLE m₀ (src + BitVec.ofNat 64 hi) w) w)
      base m₀ dst src lo (hi + w) := by
  intro x
  rw [storeLE_loadLE_apply _ _ _ _ _ hw, h x, sub_add_lit, toNat_sub_lit (by omega)]
  have hsrc : src + BitVec.ofNat 64 hi + (x - dst - BitVec.ofNat 64 hi) = src + (x - dst) := by
    bv_omega
  rw [hsrc]
  have := (x - dst).isLt
  apply ite_ite_same
  split <;> constructor <;> intro <;> omega

theorem Copied.extend_up {m m₀ : Mem} {dst src : BitVec 64} {lo hi w : Nat}
    (h : Copied m m₀ dst src lo hi) (hw : w ≤ 8) (hhi : hi + w < 2 ^ 64) (hlo : lo ≤ hi) :
    Copied (storeLE m (dst + BitVec.ofNat 64 hi) (loadLE m₀ (src + BitVec.ofNat 64 hi) w) w)
      m₀ dst src lo (hi + w) :=
  CopiedOver.extend_up (base := m₀) h hw hhi hlo

theorem Copied.extend_down_zero {m m₀ : Mem} {dst src : BitVec 64} {w hi : Nat}
    (h : Copied m m₀ dst src w hi) (hw : w ≤ 8) (hhi : hi < 2 ^ 64) (hcw : w ≤ hi) :
    Copied (storeLE m dst (loadLE m₀ src w) w) m₀ dst src 0 hi := by
  have := Copied.extend_down (c := 0) (by simpa using h) hw hhi (by simpa using hcw)
  simpa using this

theorem CopiedOver.empty_of_eq {base m₀ : Mem} {dst src : BitVec 64} {a b : Nat} (e : a = b) :
    CopiedOver base base m₀ dst src a b := by
  intro x
  rw [ite_eq_right (by omega)]

/-- Glue a block `[a, b)` stored over `base` onto the copy `[b, c)` in `base`. -/
theorem CopiedOver.join {m base m₀ : Mem} {dst src : BitVec 64} {a b c : Nat}
    (h : CopiedOver m base m₀ dst src a b) (hb : Copied base m₀ dst src b c)
    (hab : a ≤ b) (hbc : b ≤ c) : Copied m m₀ dst src a c := by
  intro x
  rw [h x, hb x]
  by_cases h1 : a ≤ (x - dst).toNat ∧ (x - dst).toNat < b
  · rw [ite_eq_left h1, ite_eq_left (by omega)]
  · rw [ite_eq_right h1]
    by_cases h2 : b ≤ (x - dst).toNat ∧ (x - dst).toNat < c
    · rw [ite_eq_left h2, ite_eq_left (by omega)]
    · rw [ite_eq_right h2, ite_eq_right (by omega)]

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

private theorem toNat_add_lit_sub_of_le {src dst : BitVec 64} {j : Nat}
    (hle : dst.toNat ≤ src.toNat) (hj : src.toNat + j < 2 ^ 64) :
    (src + BitVec.ofNat 64 j - dst).toNat = src.toNat + j - dst.toNat := by
  rw [BitVec.toNat_sub, BitVec.toNat_add, BitVec.toNat_ofNat, Nat.mod_eq_of_lt (a := j) (by omega),
    Nat.mod_eq_of_lt (a := src.toNat + j) hj]
  have := dst.isLt
  rw [show 2 ^ 64 - dst.toNat + (src.toNat + j) = (src.toNat + j - dst.toNat) + 2 ^ 64 by omega,
    Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]

theorem Copied.read_fwd {m m₀ : Mem} {dst src : BitVec 64} {k j : Nat}
    (h : Copied m m₀ dst src 0 k) (hle : dst.toNat ≤ src.toNat) (hj : k ≤ j)
    (hsrc : src.toNat + j < 2 ^ 64) :
    m (src + BitVec.ofNat 64 j) = m₀ (src + BitVec.ofNat 64 j) := by
  rw [h, ite_eq_right]
  rw [toNat_add_lit_sub_of_le hle hsrc]
  omega

theorem Copied.loadLE_fwd {m m₀ : Mem} {dst src : BitVec 64} {k j w : Nat}
    (h : Copied m m₀ dst src 0 k) (hle : dst.toNat ≤ src.toNat) (hj : k ≤ j)
    (hsrc : src.toNat + j + w ≤ 2 ^ 64) :
    loadLE m (src + BitVec.ofNat 64 j) w = loadLE m₀ (src + BitVec.ofNat 64 j) w := by
  apply loadLE_congr
  intro i hi
  rw [Arith.add_lit_lit]
  exact h.read_fwd hle (by omega) (by omega)

/-- One chunk of a forward copy, with the value loaded (from offset `k`) out of
a memory `m'` that was a prefix copy of length `j ≤ k`. -/
theorem Copied.store_fwd {m m' m₀ : Mem} {dst src : BitVec 64} {j k w hi : Nat}
    (h : Copied m m₀ dst src 0 k) (h' : Copied m' m₀ dst src 0 j) (hjk : j ≤ k)
    (hle : dst.toNat ≤ src.toNat) (hw : w ≤ 8) (hsrc : src.toNat + k + w ≤ 2 ^ 64)
    (hk : k + w < 2 ^ 64) (e : k + w = hi) :
    Copied (storeLE m (dst + BitVec.ofNat 64 k) (loadLE m' (src + BitVec.ofNat 64 k) w) w)
      m₀ dst src 0 hi := by
  rw [h'.loadLE_fwd hle hjk hsrc]
  exact (h.extend_up hw hk (Nat.zero_le _)).congr_hi e

theorem Copied.store_fwd_self {m m₀ : Mem} {dst src : BitVec 64} {k w hi : Nat}
    (h : Copied m m₀ dst src 0 k) (hle : dst.toNat ≤ src.toNat) (hw : w ≤ 8)
    (hsrc : src.toNat + k + w ≤ 2 ^ 64) (hk : k + w < 2 ^ 64) (e : k + w = hi) :
    Copied (storeLE m (dst + BitVec.ofNat 64 k) (loadLE m (src + BitVec.ofNat 64 k) w) w)
      m₀ dst src 0 hi :=
  h.store_fwd h (Nat.le_refl _) hle hw hsrc hk e

theorem Copied.store_fwd_zero {m m₀ : Mem} {dst src : BitVec 64} {w hi : Nat}
    (h : Copied m m₀ dst src 0 0) (hw : w ≤ 8) (e : w = hi) :
    Copied (storeLE m dst (loadLE m src w) w) m₀ dst src 0 hi := by
  have hm : m = m₀ := by
    funext x; rw [h x, ite_eq_right (by omega)]
  subst hm e
  have := (Copied.empty m dst src 0).extend_up (w := w) hw (by omega) (Nat.le_refl _)
  simpa using this

/-- No bound on `src + j` is needed: a source byte inside the copied part
would satisfy `src + j < dst + hi < 2^64`. -/
theorem Copied.read_bwd {m m₀ : Mem} {dst src : BitVec 64} {lo hi j : Nat}
    (h : Copied m m₀ dst src lo hi) (hsd : src.toNat ≤ dst.toNat) (hj : j < lo)
    (hdst : dst.toNat + hi < 2 ^ 64) :
    m (src + BitVec.ofNat 64 j) = m₀ (src + BitVec.ofNat 64 j) := by
  rw [h, ite_eq_right]
  intro ⟨h1, h2⟩
  have := src.isLt; have := dst.isLt
  have hsj : src.toNat + j < 2 ^ 64 := by omega
  rw [BitVec.toNat_sub, BitVec.toNat_add, BitVec.toNat_ofNat, Nat.mod_eq_of_lt (a := j) (by omega),
    Nat.mod_eq_of_lt (a := src.toNat + j) hsj] at h1 h2
  by_cases hge : dst.toNat ≤ src.toNat + j
  · rw [show 2 ^ 64 - dst.toNat + (src.toNat + j) = (src.toNat + j - dst.toNat) + 2 ^ 64 by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)] at h1
    omega
  · rw [Nat.mod_eq_of_lt (by omega)] at h2
    omega

theorem Copied.loadLE_bwd {m m₀ : Mem} {dst src : BitVec 64} {lo hi j w : Nat}
    (h : Copied m m₀ dst src lo hi) (hsd : src.toNat ≤ dst.toNat) (hj : j + w ≤ lo)
    (hdst : dst.toNat + hi < 2 ^ 64) :
    loadLE m (src + BitVec.ofNat 64 j) w = loadLE m₀ (src + BitVec.ofNat 64 j) w := by
  apply loadLE_congr
  intro i hi
  rw [Arith.add_lit_lit]
  exact h.read_bwd hsd (by omega) hdst

/-- One chunk of a backward copy, loaded from the memory being stored to. -/
theorem Copied.store_bwd_self {m m₀ : Mem} {dst src : BitVec 64} {c w lo hi : Nat}
    (h : Copied m m₀ dst src (c + w) hi) (hsd : src.toNat ≤ dst.toNat) (hw : w ≤ 8)
    (hdst : dst.toNat + hi < 2 ^ 64) (hcw : c + w ≤ hi) (e : c = lo) :
    Copied (storeLE m (dst + BitVec.ofNat 64 c) (loadLE m (src + BitVec.ofNat 64 c) w) w)
      m₀ dst src lo hi := by
  rw [h.loadLE_bwd hsd (Nat.le_refl _) hdst]
  exact (h.extend_down hw (by omega) hcw).congr_lo e

theorem Copied.store_bwd_zero {m m₀ : Mem} {dst src : BitVec 64} {w lo hi : Nat}
    (h : Copied m m₀ dst src w hi) (hsd : src.toNat ≤ dst.toNat) (hw : w ≤ 8)
    (hdst : dst.toNat + hi < 2 ^ 64) (hcw : w ≤ hi) (e : 0 = lo) :
    Copied (storeLE m dst (loadLE m src w) w) m₀ dst src lo hi := by
  have := Copied.store_bwd_self (c := 0) (by simpa using h) hsd hw hdst (by simpa using hcw) e
  simpa using this

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
