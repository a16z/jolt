import CopyRuntime.Proof.Region

/-!
# Loops and forward copies

* `Exits.loop`: the loop-invariant rule, in continuation-passing form. The
  proof of one iteration is an ordinary symbolic execution from an abstract
  state satisfying the invariant; at the back edge it hands the new state to
  the continuation it was given.
* `Copied.store_fwd`: one chunk of a forward copy (`dst ≤ src`), where the
  stored value was loaded from a memory that is itself a (shorter) prefix copy.
  The loaded bytes lie at source offsets that no earlier store has reached.
* Bit tests `x &&& 2^k = 0` as `Nat` facts, so that `omega` can decide the
  alignment branches.
-/

namespace CopyRuntime

/-! ## Loop rule -/

/-- Loop-invariant rule. `I k s` describes the loop head after `k` iterations.
`hbody` proves, for a state satisfying `I k`, that execution leaves with
`post`, assuming (continuation) that it does so from every state satisfying
`I (k + 1)`. `hbound` bounds the iteration count, which makes the rule sound. -/
theorem Exits.loop {cfg : Config} {region : BitVec 64 → Prop} {post : State → Prop}
    (I : Nat → State → Prop) (K : Nat) (hbound : ∀ k s, I k s → k ≤ K)
    (hbody : ∀ k s, I k s → (∀ t, I (k + 1) t → Exits cfg region post t) →
      Exits cfg region post s) :
    ∀ k s, I k s → Exits cfg region post s := by
  suffices h : ∀ m k s, K - k ≤ m → I k s → Exits cfg region post s from
    fun k s => h (K - k) k s (Nat.le_refl _)
  intro m
  induction m with
  | zero =>
    intro k s hm hI
    exact hbody k s hI fun t ht => absurd (hbound _ _ ht) (by omega)
  | succ m ih =>
    intro k s hm hI
    exact hbody k s hI fun t ht => ih (k + 1) t (by omega) ht

/-! ## Forward copies -/

theorem toNat_add_lit_sub_of_le {src dst : BitVec 64} {j : Nat} (hle : dst.toNat ≤ src.toNat)
    (hj : src.toNat + j < 2 ^ 64) :
    (src + BitVec.ofNat 64 j - dst).toNat = src.toNat + j - dst.toNat := by
  rw [BitVec.toNat_sub, BitVec.toNat_add, BitVec.toNat_ofNat, Nat.mod_eq_of_lt (a := j) (by omega),
    Nat.mod_eq_of_lt (a := src.toNat + j) hj]
  have := dst.isLt
  rw [show 2 ^ 64 - dst.toNat + (src.toNat + j) = (src.toNat + j - dst.toNat) + 2 ^ 64 by omega,
    Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]

/-- In a forward copy, a prefix copy of length `k` has not touched the source
bytes at offsets `≥ k`. -/
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

/-- One chunk of a forward copy. The value was loaded (from offset `k`) out of a
memory `m'` that was a prefix copy of length `j ≤ k`. -/
theorem Copied.store_fwd {m m' m₀ : Mem} {dst src : BitVec 64} {j k w : Nat}
    (h : Copied m m₀ dst src 0 k) (h' : Copied m' m₀ dst src 0 j) (hjk : j ≤ k)
    (hle : dst.toNat ≤ src.toNat) (hw : w ≤ 8) (hsrc : src.toNat + k + w ≤ 2 ^ 64)
    (hk : k + w < 2 ^ 64) :
    Copied (storeLE m (dst + BitVec.ofNat 64 k) (loadLE m' (src + BitVec.ofNat 64 k) w) w)
      m₀ dst src 0 (k + w) := by
  rw [h'.loadLE_fwd hle hjk hsrc]
  exact h.extend_up hw hk (Nat.zero_le _)

/-- `Copied.store_fwd` with the value loaded from the memory being stored to. -/
theorem Copied.store_fwd_self {m m₀ : Mem} {dst src : BitVec 64} {k w : Nat}
    (h : Copied m m₀ dst src 0 k) (hle : dst.toNat ≤ src.toNat) (hw : w ≤ 8)
    (hsrc : src.toNat + k + w ≤ 2 ^ 64) (hk : k + w < 2 ^ 64) :
    Copied (storeLE m (dst + BitVec.ofNat 64 k) (loadLE m (src + BitVec.ofNat 64 k) w) w)
      m₀ dst src 0 (k + w) :=
  h.store_fwd h (Nat.le_refl _) hle hw hsrc hk

/-- The first forward chunk, at offset 0 (addresses `dst`, `src`). -/
theorem Copied.store_fwd_zero {m m₀ : Mem} {dst src : BitVec 64} {w : Nat}
    (h : Copied m m₀ dst src 0 0) (hw : w ≤ 8) :
    Copied (storeLE m dst (loadLE m src w) w) m₀ dst src 0 w := by
  have hm : m = m₀ := by
    funext x; rw [h x, ite_eq_right (by omega)]
  subst hm
  have := (Copied.empty m dst src 0).extend_up (w := w) hw (by omega) (Nat.le_refl _)
  simpa using this

/-! ## Bit tests as arithmetic -/

theorem and_pow_two_eq_zero_iff (x : BitVec 64) (k : Nat) (hk : k < 64) :
    x &&& BitVec.ofNat 64 (2 ^ k) = 0#64 ↔ x.toNat / 2 ^ k % 2 = 0 := by
  have hbit : x.getLsbD k = decide (x.toNat / 2 ^ k % 2 = 1) := by
    rw [BitVec.getLsbD, Nat.testBit_eq_decide_div_mod_eq]
  have h01 := Nat.mod_two_eq_zero_or_one (x.toNat / 2 ^ k)
  constructor
  · intro h
    have := congrArg (fun y => y.getLsbD k) h
    simp only [BitVec.getLsbD_and, BitVec.getLsbD_ofNat, Nat.testBit_two_pow_self, hk,
      decide_true, Bool.true_and, Bool.and_true, hbit] at this
    simp at this
    omega
  · intro h
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    simp only [BitVec.getLsbD_and, BitVec.getLsbD_ofNat, Nat.testBit_two_pow]
    by_cases hik : k = i
    · subst hik; simp [hbit, h]
    · simp [hik]

@[rv_exec] theorem and_one_eq_zero_iff (x : BitVec 64) : x &&& 1#64 = 0#64 ↔ x.toNat % 2 = 0 := by
  simpa using and_pow_two_eq_zero_iff x 0 (by decide)

@[rv_exec] theorem and_two_eq_zero_iff (x : BitVec 64) :
    x &&& 2#64 = 0#64 ↔ x.toNat / 2 % 2 = 0 := by
  simpa using and_pow_two_eq_zero_iff x 1 (by decide)

@[rv_exec] theorem and_four_eq_zero_iff (x : BitVec 64) :
    x &&& 4#64 = 0#64 ↔ x.toNat / 4 % 2 = 0 := by
  simpa using and_pow_two_eq_zero_iff x 2 (by decide)

end CopyRuntime
