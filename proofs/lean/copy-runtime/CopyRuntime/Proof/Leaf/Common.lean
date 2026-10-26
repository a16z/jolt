import CopyRuntime.Proof.Leaf.Basic

/-!
# Shared facts for the leaf proofs

* `Env`: everything fixed during a leaf run.
* Word lemmas: the stored combination of two loaded words is the load of the
  eight bytes in between (`storeLE_combine`), and a shifted word depends only
  on the bytes that the shift keeps (`loadLE_shr_congr`, `loadLE_shl_congr`).
  The carried word `t3` is only known up to those bytes: the leaf loads it
  before (forward) or after (backward) the bytes it covers that the copy may
  already have overwritten, and the shift discards exactly those.
* Snapshot reads for a backward copy (`copied_loadLE_bwd`), and one-byte copy
  steps in either direction.
* Address arithmetic with the leaf's immediates.
-/

namespace CopyRuntime.Leaf

/-! ## Words and bytes -/

/-- Bit `8i + t` of an 8-byte little-endian load is bit `t` of byte `i`. -/
theorem loadLE_getLsbD (M : Mem) (b : BitVec 64) {i t : Nat} (hi : i < 8) (ht : t < 8) :
    (loadLE M b 8).getLsbD (8 * i + t) = (M (b + BitVec.ofNat 64 i)).getLsbD t := by
  have h := congrArg (fun x : BitVec 8 => x.getLsbD t) (loadLE_byte M 8 b i (Nat.le_refl _) hi)
  simp only [BitVec.getLsbD_setWidth, BitVec.getLsbD_ushiftRight] at h
  rw [← h]
  simp [ht]

/-- A right shift by `8a` only depends on bytes `a .. 7`. -/
theorem loadLE_shr_congr {M M' : Mem} {b : BitVec 64} {a : Nat}
    (h : ∀ i, a ≤ i → i < 8 → M (b + BitVec.ofNat 64 i) = M' (b + BitVec.ofNat 64 i)) :
    loadLE M b 8 >>> (8 * a) = loadLE M' b 8 >>> (8 * a) := by
  apply BitVec.eq_of_getLsbD_eq
  intro j hj
  simp only [BitVec.getLsbD_ushiftRight]
  by_cases hk : 8 * a + j < 64
  · have e : 8 * a + j = 8 * ((8 * a + j) / 8) + (8 * a + j) % 8 := by omega
    rw [e, loadLE_getLsbD M b (by omega) (by omega), loadLE_getLsbD M' b (by omega) (by omega),
      h _ (by omega) (by omega)]
  · rw [BitVec.getLsbD_of_ge _ _ (by omega), BitVec.getLsbD_of_ge _ _ (by omega)]

/-- A left shift by `64 - 8a` only depends on bytes `0 .. a-1`. -/
theorem loadLE_shl_congr {M M' : Mem} {b : BitVec 64} {a : Nat} (ha : a ≤ 8)
    (h : ∀ i, i < a → M (b + BitVec.ofNat 64 i) = M' (b + BitVec.ofNat 64 i)) :
    loadLE M b 8 <<< (64 - 8 * a) = loadLE M' b 8 <<< (64 - 8 * a) := by
  apply BitVec.eq_of_getLsbD_eq
  intro j hj
  simp only [BitVec.getLsbD_shiftLeft]
  by_cases hk : j < 64 - 8 * a
  · simp [hk]
  · have e : j - (64 - 8 * a) = 8 * ((j - (64 - 8 * a)) / 8) + (j - (64 - 8 * a)) % 8 := by
      omega
    rw [e, loadLE_getLsbD M b (by omega) (by omega), loadLE_getLsbD M' b (by omega) (by omega),
      h _ (by omega)]

/-- Byte `i` of the combination of two consecutive loaded words is the byte at
offset `a + i`. -/
theorem combine_load_byte (M : Mem) (b b₁ : BitVec 64) (hb : b₁ = b + 8#64) (a i : Nat)
    (ha : 1 ≤ a ∧ a ≤ 7) (hi : i < 8) :
    (((loadLE M b 8 >>> (8 * a)) ||| (loadLE M b₁ 8 <<< (64 - 8 * a))) >>> (8 * i)).setWidth 8 =
      M (b + BitVec.ofNat 64 (a + i)) := by
  rw [combine_byte _ _ a i ha hi]
  split
  · exact loadLE_byte M 8 b (a + i) (Nat.le_refl _) (by omega)
  · rw [loadLE_byte M 8 b₁ (a + i - 8) (Nat.le_refl _) (by omega), hb, BitVec.add_assoc]
    congr 2
    apply BitVec.eq_of_toNat_eq
    simp only [BitVec.toNat_add, BitVec.toNat_ofNat]
    omega

theorem storeLE_congr_bytes {m : Mem} {A v v' : BitVec 64} {w : Nat}
    (h : ∀ i, i < w → (v >>> (8 * i)).setWidth 8 = (v' >>> (8 * i)).setWidth 8) :
    storeLE m A v w = storeLE m A v' w := by
  funext x
  unfold storeLE
  split
  · exact h _ ‹_›
  · rfl

/-- Storing the combination of two consecutive loaded words stores the load of
the eight bytes starting at offset `a` of the first. -/
theorem storeLE_combine (m M : Mem) (A b b₁ c : BitVec 64) {a : Nat} (hb : b₁ = b + 8#64)
    (hc : c = b + BitVec.ofNat 64 a) (ha : 1 ≤ a ∧ a ≤ 7) :
    storeLE m A ((loadLE M b 8 >>> (8 * a)) ||| (loadLE M b₁ 8 <<< (64 - 8 * a))) 8 =
      storeLE m A (loadLE M c 8) 8 := by
  apply storeLE_congr_bytes
  intro i hi
  rw [combine_load_byte M b b₁ hb a i ha hi, loadLE_byte M 8 c i (Nat.le_refl _) hi, hc,
    BitVec.add_assoc, ← BitVec.ofNat_add]

/-! ## Snapshot reads -/

/-- In a backward copy (`src ≤ dst`), a copy of the top part `[lo, hi)` has not
touched the source bytes below offset `lo`. -/
theorem copied_read_bwd {m m₀ : Mem} {dst src : BitVec 64} {lo hi j : Nat}
    (h : Copied m m₀ dst src lo hi) (hsd : src.toNat ≤ dst.toNat) (hj : j < lo)
    (hlo : lo ≤ hi) (hhi : dst.toNat + hi < 2 ^ 64) :
    m (src + BitVec.ofNat 64 j) = m₀ (src + BitVec.ofNat 64 j) := by
  rw [h, ite_eq_right]
  intro ⟨h1, h2⟩
  have := src.isLt; have := dst.isLt
  rw [BitVec.toNat_sub, BitVec.toNat_add, BitVec.toNat_ofNat, Nat.mod_eq_of_lt (a := j) (by omega),
    Nat.mod_eq_of_lt (a := src.toNat + j) (by omega)] at h1 h2
  by_cases hge : dst.toNat ≤ src.toNat + j
  · rw [show 2 ^ 64 - dst.toNat + (src.toNat + j) = (src.toNat + j - dst.toNat) + 2 ^ 64 by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)] at h1
    omega
  · rw [Nat.mod_eq_of_lt (by omega)] at h2
    omega

theorem copied_loadLE_bwd {m m₀ : Mem} {dst src : BitVec 64} {lo hi x w : Nat}
    (h : Copied m m₀ dst src lo hi) (hsd : src.toNat ≤ dst.toNat) (hx : x + w ≤ lo)
    (hlo : lo ≤ hi) (hhi : dst.toNat + hi < 2 ^ 64) :
    loadLE m (src + BitVec.ofNat 64 x) w = loadLE m₀ (src + BitVec.ofNat 64 x) w := by
  apply loadLE_congr
  intro i hi
  rw [Arith.add_lit_lit]
  exact copied_read_bwd h hsd (by omega) hlo hhi

theorem copied_congr_lo {m m₀ : Mem} {dst src : BitVec 64} {a b hi : Nat}
    (h : Copied m m₀ dst src a hi) (e : a = b) : Copied m m₀ dst src b hi := e ▸ h

theorem copied_congr_hi {m m₀ : Mem} {dst src : BitVec 64} {lo a b : Nat}
    (h : Copied m m₀ dst src lo a) (e : a = b) : Copied m m₀ dst src lo b := e ▸ h

/-- One byte of a forward copy (`dst ≤ src`), loaded from the memory being
stored to. -/
theorem copied_fwd_byte {m m₀ : Mem} {dst src : BitVec 64} {k hi : Nat}
    (h : Copied m m₀ dst src 0 k) (hle : dst.toNat ≤ src.toNat)
    (hsrc : src.toNat + k + 1 ≤ 2 ^ 64) (hk : k + 1 < 2 ^ 64) (e : k + 1 = hi) :
    Copied (storeLE m (dst + BitVec.ofNat 64 k) (loadLE m (src + BitVec.ofNat 64 k) 1) 1)
      m₀ dst src 0 hi :=
  copied_congr_hi (h.store_fwd_self hle (by decide) hsrc hk) e

/-- One byte of a backward copy (`src ≤ dst`), loaded from the memory being
stored to. -/
theorem copied_bwd_byte {m m₀ : Mem} {dst src : BitVec 64} {r c hi : Nat}
    (h : Copied m m₀ dst src r hi) (hc : c + 1 = r) (hsd : src.toNat ≤ dst.toNat)
    (hrhi : r ≤ hi) (hhi : dst.toNat + hi < 2 ^ 64) :
    Copied (storeLE m (dst + BitVec.ofNat 64 c) (loadLE m (src + BitVec.ofNat 64 c) 1) 1)
      m₀ dst src c hi := by
  rw [copied_loadLE_bwd h hsd (by omega) hrhi hhi]
  exact Copied.extend_down (copied_congr_lo h hc.symm) (by decide) (by omega) (by omega)

/-! ## Address arithmetic -/

theorem add_ofNat_congr {x : BitVec 64} {p q : Nat} (h : p = q) :
    x + BitVec.ofNat 64 p = x + BitVec.ofNat 64 q := h ▸ rfl

theorem imm_natCast (k : Nat) : imm (k : Int) = BitVec.ofNat 64 k := by
  apply BitVec.eq_of_toNat_eq
  simp [imm]

/-- `x + p + k` for a nonnegative immediate `k`. -/
theorem addr_fwd (x : BitVec 64) (p k q : Nat) (h : p + k = q) :
    x + BitVec.ofNat 64 p + imm (k : Int) = x + BitVec.ofNat 64 q := by
  rw [imm_natCast, Arith.add_lit_lit, h]

/-- `x + p - k` for a negative immediate `-k`, with `k ≤ p`. -/
theorem addr_bwd (x : BitVec 64) (p k q : Nat) (h : q + k = p) :
    x + BitVec.ofNat 64 p + imm (-(k : Int)) = x + BitVec.ofNat 64 q := by
  unfold imm
  rw [BitVec.ofInt_neg, BitVec.ofInt_natCast, ← h, BitVec.ofNat_add, BitVec.add_assoc,
    BitVec.add_assoc, BitVec.add_right_neg, BitVec.add_zero]

/-- Adding the literal `2^64 - k` is subtracting `k`. -/
theorem ofNat_add_wrap {r c k : Nat} (hc : c + k = 2 ^ 64) (h : k ≤ r) :
    BitVec.ofNat 64 (r + c) = BitVec.ofNat 64 (r - k) := by
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.toNat_ofNat]
  rw [show r + c = (r - k) + 2 ^ 64 by omega, Nat.add_mod_right]

theorem ofNat_add_m1 {r : Nat} (h : 1 ≤ r) :
    BitVec.ofNat 64 (r + 18446744073709551615) = BitVec.ofNat 64 (r - 1) :=
  ofNat_add_wrap (by decide) h

theorem ofNat_add_m8 {r : Nat} (h : 8 ≤ r) :
    BitVec.ofNat 64 (r + 18446744073709551608) = BitVec.ofNat 64 (r - 8) :=
  ofNat_add_wrap (by decide) h

theorem ofNat_add_m128 {r : Nat} (h : 128 ≤ r) :
    BitVec.ofNat 64 (r + 18446744073709551488) = BitVec.ofNat 64 (r - 128) :=
  ofNat_add_wrap (by decide) h

theorem and_neg8_eq' (x : BitVec 64) : x &&& 0xfffffffffffffff8#64 = (x >>> 3) <<< 3 := by
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

theorem toNat_and_neg8' (x : BitVec 64) :
    (x &&& 0xfffffffffffffff8#64).toNat = x.toNat / 8 * 8 := by
  rw [and_neg8_eq', BitVec.toNat_shiftLeft, BitVec.toNat_ushiftRight, Nat.shiftLeft_eq,
    Nat.shiftRight_eq_div_pow]
  have := x.isLt
  rw [Nat.mod_eq_of_lt (by omega)]

/-- Adding a large literal (a negative immediate) is subtracting a small one.
Rewriting with this before any `toNat` lemma matters for the kernel: simp
proves `(x + c).toNat = (x.toNat + c) % 2^64` by `rfl`, and the kernel then
rechecks it by unfolding `Nat.add x.toNat c` in unary on the literal `c`,
which hits "deep recursion". `BitVec.toNat_sub` puts the literal first
(`2^64 - c' + x.toNat`), which the kernel handles. -/
theorem leaf_add_big (x : BitVec 64) (c : Nat) (h : 2 ^ 63 ≤ c) (h' : c < 2 ^ 64) :
    x + BitVec.ofNat 64 c = x - BitVec.ofNat 64 (2 ^ 64 - c) := by
  rw [BitVec.sub_eq_add_neg]
  congr 1
  apply BitVec.eq_of_toNat_eq
  rw [BitVec.toNat_neg, BitVec.toNat_ofNat, BitVec.toNat_ofNat, Nat.mod_eq_of_lt h',
    Nat.mod_eq_of_lt (by omega : 2 ^ 64 - c < 2 ^ 64), Nat.mod_eq_of_lt (by omega)]
  omega

theorem leaf_toNat_sub {x : BitVec 64} {c : Nat} (hc : c ≤ x.toNat) :
    (x - BitVec.ofNat 64 c).toNat = x.toNat - c := by
  have := x.isLt
  rw [BitVec.toNat_sub_of_le (by rw [BitVec.le_def, BitVec.toNat_ofNat]; omega),
    BitVec.toNat_ofNat, Nat.mod_eq_of_lt (by omega)]

theorem leaf_ofNat_sub {r k : Nat} (h : k ≤ r) :
    BitVec.ofNat 64 r - BitVec.ofNat 64 k = BitVec.ofNat 64 (r - k) := by
  rw [show r = (r - k) + k by omega, BitVec.ofNat_add, BitVec.add_sub_cancel,
    show r - k + k - k = r - k by omega]

theorem toNat_and_seven' (x : BitVec 64) : (x &&& 7#64).toNat = x.toNat % 8 := by
  rw [BitVec.toNat_and]
  exact Nat.and_two_pow_sub_one_eq_mod _ 3

theorem leaf_and_seven_eq_zero_iff (x : BitVec 64) : x &&& 7#64 = 0#64 ↔ x.toNat % 8 = 0 := by
  rw [← toNat_and_seven']
  constructor
  · intro h; rw [h]; rfl
  · intro h; exact BitVec.eq_of_toNat_eq h

/-- `andi t2, a1, -8` on `src + J`, where `(src + J) % 8 = a`. -/
theorem and_neg8_add {x : BitVec 64} {J a : Nat} (hx : x.toNat + J < 2 ^ 64)
    (ha : (x.toNat + J) % 8 = a) (hJa : a ≤ J) :
    (x + BitVec.ofNat 64 J) &&& 0xfffffffffffffff8#64 = x + BitVec.ofNat 64 (J - a) := by
  apply BitVec.eq_of_toNat_eq
  rw [toNat_and_neg8']
  simp only [BitVec.toNat_add, BitVec.toNat_ofNat]
  rw [Nat.mod_eq_of_lt (a := J) (by omega), Nat.mod_eq_of_lt (a := J - a) (by omega),
    Nat.mod_eq_of_lt hx, Nat.mod_eq_of_lt (by omega)]
  omega

/-- `andi a5, a1, 7` on `src + J`. -/
theorem and_seven_add {x : BitVec 64} {J a : Nat} (hx : x.toNat + J < 2 ^ 64)
    (ha : (x.toNat + J) % 8 = a) :
    (x + BitVec.ofNat 64 J) &&& 7#64 = BitVec.ofNat 64 a := by
  apply BitVec.eq_of_toNat_eq
  rw [toNat_and_seven', BitVec.toNat_add, BitVec.toNat_ofNat, BitVec.toNat_ofNat,
    Nat.mod_eq_of_lt (a := J) (by omega), Nat.mod_eq_of_lt hx,
    Nat.mod_eq_of_lt (a := a) (by omega)]
  exact ha

/-- `slli a3, a5, 3`. -/
theorem ofNat_shl3 {a : Nat} (ha : a ≤ 7) : BitVec.ofNat 64 a <<< 3 = BitVec.ofNat 64 (8 * a) := by
  apply BitVec.eq_of_toNat_eq
  rw [BitVec.toNat_shiftLeft, BitVec.toNat_ofNat, BitVec.toNat_ofNat, Nat.shiftLeft_eq,
    Nat.mod_eq_of_lt (a := a) (by omega), Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)]
  omega

theorem bv_eq_zero_iff {x : BitVec 64} {v : Nat} (h : x.toNat = v) : x = 0#64 ↔ v = 0 := by
  constructor
  · intro hx; rw [hx] at h; rw [← h]; rfl
  · intro hv; apply BitVec.eq_of_toNat_eq; rw [h, hv]; rfl

theorem toNat_ofNat_mod64 {k : Nat} (hk : k < 64) : (BitVec.ofNat 64 k).toNat % 64 = k := by
  rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt (a := k) (by omega), Nat.mod_eq_of_lt hk]


/-! ## The run environment and the final state -/

/-- Everything fixed during a leaf run. -/
structure Env (cfg : Config) (L dst src n R : BitVec 64) : Prop where
  ctx : Ctx cfg L
  sb : Sandbox cfg dst src n
  args : ArgsOk dst src n R
  rout : ¬ region L R
  mis : dst.toNat % 8 ≠ src.toNat % 8
  big : 544 ≤ n.toNat

/-- Prove the `preserved` field of a state whose register file is a chain of
writes (to clobbered registers) on top of `regs` with `hpres` for `regs`. -/
macro "leaf_preserved" hpres:term : tactic => `(tactic| (
  intro r hr
  have hr' := hr
  simp only [clobbered, List.mem_cons, List.not_mem_nil, or_false, not_or] at hr'
  simp only [writeReg_apply, hr', ite_false]
  exact $hpres r hr))

/-- `preserved` survives a step that changes at most three clobbered registers. -/
theorem pres_frame3 {regs₀ regs regs' : Reg → BitVec 64} {a b c : Reg} (ha : a ∈ clobbered)
    (hb : b ∈ clobbered) (hc : c ∈ clobbered)
    (hfr : ∀ r, r ≠ a → r ≠ b → r ≠ c → regs' r = regs r)
    (hp : ∀ r, r ∉ clobbered → regs r = regs₀ r) : ∀ r, r ∉ clobbered → regs' r = regs₀ r :=
  fun r hr => by
    rw [hfr r (fun h => hr (h ▸ ha)) (fun h => hr (h ▸ hb)) (fun h => hr (h ▸ hc))]
    exact hp r hr

/-- `preserved` survives a run that changes at most four clobbered registers. -/
theorem pres_frame4 {regs₀ regs regs' : Reg → BitVec 64} {a b c d : Reg} (ha : a ∈ clobbered)
    (hb : b ∈ clobbered) (hc : c ∈ clobbered) (hd : d ∈ clobbered)
    (hfr : ∀ r, r ≠ a → r ≠ b → r ≠ c → r ≠ d → regs' r = regs r)
    (hp : ∀ r, r ∉ clobbered → regs r = regs₀ r) : ∀ r, r ∉ clobbered → regs' r = regs₀ r :=
  fun r hr => by
    rw [hfr r (fun h => hr (h ▸ ha)) (fun h => hr (h ▸ hb)) (fun h => hr (h ▸ hc))
      (fun h => hr (h ▸ hd))]
    exact hp r hr

theorem Env.post_ret {cfg : Config} {L dst src n R : BitVec 64} (env : Env cfg L dst src n R)
    {regs₀ regs : Reg → BitVec 64} {m₀ mem : Mem} {p : BitVec 64}
    (he : Wrapper.EntryRegs regs₀ dst src n R) (hcop : Copied mem m₀ dst src 0 n.toNat)
    (hpres : ∀ r, r ∉ clobbered → regs r = regs₀ r) :
    Returned ⟨p, regs₀, m₀⟩ R dst src n ⟨R, regs, mem⟩ :=
  ⟨rfl, by simp [State.reg_mk, hpres 10 (by decide), he.a0],
    hcop.memmove (Nat.le_of_lt env.args.dst_no_wrap),
    fun r hr => by simp [State.reg_mk, hpres r hr]⟩

end CopyRuntime.Leaf
