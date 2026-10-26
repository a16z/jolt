import CopyRuntime.Proof.Wrapper.Exits
import CopyRuntime.Proof.Loops

/-!
# The misaligned-copy leaf: code facts and the word-combining identity

`__jolt_copy_misaligned` is entered from either wrapper (by
`j __jolt_copy_misaligned`) for mismatched offsets and `n ≥ 544`. It peels
bytes until `a7` (the destination cursor) is 8-byte aligned, copies eight more
bytes if the first aligned source word would start below `src`, then combines
pairs of aligned source words with `srl`/`sll`/`or` into aligned destination
words (a 128-byte loop and a word loop), and finishes with a byte tail. A
mirror-image backward copy handles `src < dst`.
-/

namespace CopyRuntime.Leaf

theorem size_eq : Generated.joltCopyMisaligned.bytes.length = 988 := by decide +kernel

/-- The leaf region. -/
abbrev region (L : BitVec 64) : BitVec 64 → Prop := codeRegion L 988

/-- What the leaf path proofs need about the loaded code (it has no
relocations). -/
structure Ctx (cfg : Config) (L : BitVec 64) : Prop where
  code : CodeAt cfg L Generated.Disasm.joltCopyMisaligned []
  base_even : L.toNat % 2 = 0

theorem Ctx.of {cfg : Config} {L : BitVec 64} (hL : cfg.Loaded L Generated.joltCopyMisaligned)
    (hB : L.toNat % 2 = 0) : Ctx cfg L :=
  ⟨CodeAt.of_loaded hL hB (by rw [size_eq]; decide)
    DecoderCheck.joltCopyMisaligned_agrees_with_llvm (fun _ _ => rfl), hB⟩

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

end CopyRuntime.Leaf
