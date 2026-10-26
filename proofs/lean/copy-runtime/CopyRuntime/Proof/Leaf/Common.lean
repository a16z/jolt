import CopyRuntime.Proof.Region
import CopyRuntime.Proof.Returned

/-!
# Shared facts for the misaligned leaf

Storing the combination of two consecutive loaded words stores the load of the
eight bytes in between (`storeLE_combine`), and a shifted word depends only on
the bytes that the shift keeps (`loadLE_shr_congr`, `loadLE_shl_congr`).
-/

namespace CopyRuntime.Leaf

theorem size_eq : Generated.joltCopyMisaligned.bytes.length = 988 := by decide +kernel

abbrev region (L : BitVec 64) : BitVec 64 → Prop := codeRegion L 988

/-- The leaf has no relocations. -/
structure Ctx (cfg : Config) (L : BitVec 64) : Prop where
  code : CodeAt cfg L Generated.Disasm.joltCopyMisaligned []
  base_even : L.toNat % 2 = 0

theorem Ctx.of {cfg : Config} {L : BitVec 64} (hL : cfg.Loaded L Generated.joltCopyMisaligned)
    (hB : L.toNat % 2 = 0) : Ctx cfg L :=
  ⟨CodeAt.of_loaded hL hB (by rw [size_eq]; decide)
    DecoderCheck.joltCopyMisaligned_agrees_with_llvm (fun _ _ => rfl), hB⟩

/-- Everything fixed during a leaf run. -/
structure Env (cfg : Config) (L dst src n R : BitVec 64) : Prop where
  ctx : Ctx cfg L
  sb : Sandbox cfg dst src n
  args : ArgsOk dst src n R
  rout : ¬ region L R
  mis : dst.toNat % 8 ≠ src.toNat % 8
  big : 544 ≤ n.toNat

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

/-- The word-combining step of the leaf: for a source offset `a ∈ 1..7` within
an aligned word pair `w₀, w₁`, `(w₀ >>> 8a) ||| (w₁ <<< (64 - 8a))` consists of
bytes `a .. 7` of `w₀` followed by bytes `0 .. a-1` of `w₁`. (For `a = 0` the
`sll` would shift by 64 mod 64 = 0 and the identity would fail; the leaf only
reaches this code with `a ≠ 0` because the offsets are mismatched.) -/
theorem combine_byte (w₀ w₁ : BitVec 64) (a i : Nat) (ha : 1 ≤ a ∧ a ≤ 7) (hi : i < 8) :
    (((w₀ >>> (8 * a)) ||| (w₁ <<< (64 - 8 * a))) >>> (8 * i)).setWidth 8 =
      if a + i < 8 then (w₀ >>> (8 * (a + i))).setWidth 8
      else (w₁ >>> (8 * (a + i - 8))).setWidth 8 := by
  ext j hj
  split
  · rename_i h
    have e1 : 8 * a + (8 * i + j) = 8 * (a + i) + j := by omega
    have e2 : 8 * i + j < 64 - 8 * a := by omega
    simp [e1, e2]
  · rename_i h
    have e1 : 8 * i + j < 64 := by omega
    have e2 : ¬ 8 * i + j < 64 - 8 * a := by omega
    have e3 : 8 * i + j - (64 - 8 * a) = 8 * (a + i - 8) + j := by omega
    have e4 : (8 * (a + i - 8) + j) < 64 := by omega
    simp [e1, e2, e3, BitVec.getLsbD_of_ge w₀ (8 * a + (8 * i + j)) (by omega),
      BitVec.getLsbD_eq_getElem e4]

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

theorem addr_fwd (x : BitVec 64) (p k q : Nat) (h : p + k = q) :
    x + BitVec.ofNat 64 p + imm (k : Int) = x + BitVec.ofNat 64 q := by
  unfold imm
  rw [BitVec.ofInt_natCast, Arith.add_lit_lit, h]

theorem addr_bwd (x : BitVec 64) (p k q : Nat) (h : q + k = p) :
    x + BitVec.ofNat 64 p + imm (-(k : Int)) = x + BitVec.ofNat 64 q := by
  unfold imm
  rw [Arith.add_ofNat_ofInt_neg x (by omega)]
  exact Arith.add_ofNat_congr (by omega)

/-- `andi t2, a1, -8` on `src + J`, where `(src + J) % 8 = a`. -/
theorem and_neg8_add {x : BitVec 64} {J a : Nat} (hx : x.toNat + J < 2 ^ 64)
    (ha : (x.toNat + J) % 8 = a) (hJa : a ≤ J) :
    (x + BitVec.ofNat 64 J) &&& 0xfffffffffffffff8#64 = x + BitVec.ofNat 64 (J - a) := by
  apply BitVec.eq_of_toNat_eq
  rw [Arith.toNat_and_neg8]
  simp only [BitVec.toNat_add, BitVec.toNat_ofNat]
  rw [Nat.mod_eq_of_lt (a := J) (by omega), Nat.mod_eq_of_lt (a := J - a) (by omega),
    Nat.mod_eq_of_lt hx, Nat.mod_eq_of_lt (by omega)]
  omega

/-- `andi a5, a1, 7` on `src + J`. -/
theorem and_seven_add {x : BitVec 64} {J a : Nat} (hx : x.toNat + J < 2 ^ 64)
    (ha : (x.toNat + J) % 8 = a) :
    (x + BitVec.ofNat 64 J) &&& 7#64 = BitVec.ofNat 64 a := by
  apply BitVec.eq_of_toNat_eq
  rw [Arith.toNat_and_seven, BitVec.toNat_add, BitVec.toNat_ofNat, BitVec.toNat_ofNat,
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

end CopyRuntime.Leaf
