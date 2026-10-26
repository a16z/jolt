import CopyRuntime.Proof.Leaf.Common

/-!
# One combine-and-store step of the leaf

Five instructions `ld rN, ilo(t2); srl t5, ·, a3; sll t6, ·, a4;
or t5, t5, t6; sd t5, ist(a7)` store one aligned destination word made of two
aligned source words: the word `rO` loaded earlier and the word `rN` loaded
now. The lemmas take an abstract state described by predicates and hand an
abstract state to a continuation, so that chaining them never grows the
symbolic state. The immediates and the code offset are parameters; the code
hypothesis is discharged against the LLVM listing when a lemma is used.

* Forward (`fwd_comb`): `rN` is the higher word. Only bytes `a .. 7` of `rO`
  matter (`srl` by `8a` drops the rest), and only those are known to equal the
  snapshot: the copy may already have overwritten its low bytes.
* Backward (`bwd_comb`): `rN` is the lower word, and only bytes `0 .. a-1` of
  `rO` matter (`sll` by `64 - 8a` drops the rest).
-/

namespace CopyRuntime.Leaf

variable {cfg : Config} {L dst src n R : BitVec 64}

set_option maxHeartbeats 4000000 in
theorem fwd_comb (env : Env cfg L dst src n R) (hle : dst.toNat ≤ src.toNat) (m₀ : Mem)
    {post : State → Prop} (a : Nat) (ha : 1 ≤ a ∧ a ≤ 7) (o : Nat) (rO rN : Reg)
    (hr : (rO = 28 ∧ rN = 29) ∨ (rO = 29 ∧ rN = 28)) (ilo ist : BitVec 64)
    (hcode : (o, Instr.load .double rN 7 ilo, 4) ∈ Generated.Disasm.joltCopyMisaligned ∧
      (o + 4, Instr.alu .srl 30 rO 13, 4) ∈ Generated.Disasm.joltCopyMisaligned ∧
      (o + 8, Instr.alu .sll 31 rN 14, 4) ∈ Generated.Disasm.joltCopyMisaligned ∧
      (o + 12, Instr.alu .or 30 30 31, 4) ∈ Generated.Disasm.joltCopyMisaligned ∧
      (o + 16, Instr.store .double 30 17 ist, 4) ∈ Generated.Disasm.joltCopyMisaligned)
    (ho : o + 20 ≤ 988) (J : Nat) (hJa : a ≤ J) (hroom : J + 16 - a ≤ n.toNat)
    (hsal : (src.toNat + J) % 8 = a) (hdal : (dst.toNat + J) % 8 = 0)
    (regs : Reg → BitVec 64) (mem : Mem)
    (hld : regs 7 + ilo = src + BitVec.ofNat 64 (J + 8 - a))
    (hst : regs 17 + ist = dst + BitVec.ofNat 64 J)
    (h13 : (regs 13).toNat % 64 = 8 * a) (h14 : (regs 14).toNat % 64 = 64 - 8 * a)
    (hO : regs rO >>> (8 * a) = loadLE m₀ (src + BitVec.ofNat 64 (J - a)) 8 >>> (8 * a))
    (hcop : Copied mem m₀ dst src 0 J)
    (hcont : ∀ regs' mem', (∀ r, r ≠ rN → r ≠ 30 → r ≠ 31 → regs' r = regs r) →
      regs' rN = loadLE m₀ (src + BitVec.ofNat 64 (J + 8 - a)) 8 →
      Copied mem' m₀ dst src 0 (J + 8) →
      Exits cfg (region L) post ⟨L + BitVec.ofNat 64 (o + 20), regs', mem'⟩) :
    Exits cfg (region L) post ⟨L + BitVec.ofNat 64 o, regs, mem⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hsw' := env.args.src_no_wrap
  have hsw : src.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hsw'
  have hdw' := env.args.dst_no_wrap
  have hdw : dst.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hdw'
  have hl := env.sb.load; have hs := env.sb.store
  have hconv : loadLE mem (src + BitVec.ofNat 64 (J + 8 - a)) 8 =
      loadLE m₀ (src + BitVec.ofNat 64 (J + 8 - a)) 8 :=
    hcop.loadLE_fwd hle (by omega) (by omega)
  have hb : src + BitVec.ofNat 64 (J + 8 - a) = src + BitVec.ofNat 64 (J - a) + 8#64 := by
    rw [Arith.add_lit_lit]; exact add_ofNat_congr (by omega)
  have hc : src + BitVec.ofNat 64 J = src + BitVec.ofNat 64 (J - a) + BitVec.ofNat 64 a := by
    rw [Arith.add_lit_lit]; exact add_ofNat_congr (by omega)
  rcases hr with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;>
  · refine ctx.code.step o ⟨hcode.1, by simp⟩ rfl (codeRegion_lit (by omega) (by decide)) ?_
    rv_simp [hld, hl, hconv]
    refine ctx.code.step (o + 4) ⟨hcode.2.1, by simp⟩ rfl
      (codeRegion_lit (by omega) (by decide)) ?_
    rv_simp [h13, hO]
    refine ctx.code.step (o + 8) ⟨hcode.2.2.1, by simp⟩ rfl
      (codeRegion_lit (by omega) (by decide)) ?_
    rv_simp [h14]
    refine ctx.code.step (o + 12) ⟨hcode.2.2.2.1, by simp⟩ rfl
      (codeRegion_lit (by omega) (by decide)) ?_
    rv_simp
    refine ctx.code.step (o + 16) ⟨hcode.2.2.2.2, by simp⟩ rfl
      (codeRegion_lit (by omega) (by decide)) ?_
    rv_simp [hst, hs]
    rw [storeLE_combine _ _ _ _ _ _ hb hc ha]
    refine hcont _ _ ?_ ?_ (Copied.extend_up hcop (by decide) (by omega) (by omega))
    · intro r h1 h2 h3
      simp [writeReg_apply, h1, h2, h3]
    · simp [writeReg_apply]

set_option maxHeartbeats 4000000 in
theorem bwd_comb (env : Env cfg L dst src n R) (hsd : src.toNat ≤ dst.toNat) (m₀ : Mem)
    {post : State → Prop} (a : Nat) (ha : 1 ≤ a ∧ a ≤ 7) (o : Nat) (rO rN : Reg)
    (hr : (rO = 28 ∧ rN = 29) ∨ (rO = 29 ∧ rN = 28)) (ilo ist : BitVec 64)
    (hcode : (o, Instr.load .double rN 7 ilo, 4) ∈ Generated.Disasm.joltCopyMisaligned ∧
      (o + 4, Instr.alu .srl 30 rN 13, 4) ∈ Generated.Disasm.joltCopyMisaligned ∧
      (o + 8, Instr.alu .sll 31 rO 14, 4) ∈ Generated.Disasm.joltCopyMisaligned ∧
      (o + 12, Instr.alu .or 30 30 31, 4) ∈ Generated.Disasm.joltCopyMisaligned ∧
      (o + 16, Instr.store .double 30 17 ist, 4) ∈ Generated.Disasm.joltCopyMisaligned)
    (ho : o + 20 ≤ 988) (r : Nat) (hra : a + 8 ≤ r) (hrn : r ≤ n.toNat)
    (hsal : (src.toNat + r) % 8 = a) (hdal : (dst.toNat + r) % 8 = 0)
    (regs : Reg → BitVec 64) (mem : Mem)
    (hld : regs 7 + ilo = src + BitVec.ofNat 64 (r - 8 - a))
    (hst : regs 17 + ist = dst + BitVec.ofNat 64 (r - 8))
    (h13 : (regs 13).toNat % 64 = 8 * a) (h14 : (regs 14).toNat % 64 = 64 - 8 * a)
    (hO : regs rO <<< (64 - 8 * a) =
      loadLE m₀ (src + BitVec.ofNat 64 (r - a)) 8 <<< (64 - 8 * a))
    (hcop : Copied mem m₀ dst src r n.toNat)
    (hcont : ∀ regs' mem', (∀ r, r ≠ rN → r ≠ 30 → r ≠ 31 → regs' r = regs r) →
      regs' rN = loadLE m₀ (src + BitVec.ofNat 64 (r - 8 - a)) 8 →
      Copied mem' m₀ dst src (r - 8) n.toNat →
      Exits cfg (region L) post ⟨L + BitVec.ofNat 64 (o + 20), regs', mem'⟩) :
    Exits cfg (region L) post ⟨L + BitVec.ofNat 64 o, regs, mem⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hsw' := env.args.src_no_wrap
  have hsw : src.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hsw'
  have hdw' := env.args.dst_no_wrap
  have hdw : dst.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hdw'
  have hl := env.sb.load; have hs := env.sb.store
  have hconv : loadLE mem (src + BitVec.ofNat 64 (r - 8 - a)) 8 =
      loadLE m₀ (src + BitVec.ofNat 64 (r - 8 - a)) 8 :=
    copied_loadLE_bwd hcop hsd (by omega) hrn hdw'
  have hb : src + BitVec.ofNat 64 (r - a) = src + BitVec.ofNat 64 (r - 8 - a) + 8#64 := by
    rw [Arith.add_lit_lit]; exact add_ofNat_congr (by omega)
  have hc : src + BitVec.ofNat 64 (r - 8) =
      src + BitVec.ofNat 64 (r - 8 - a) + BitVec.ofNat 64 a := by
    rw [Arith.add_lit_lit]; exact add_ofNat_congr (by omega)
  have hcop' : Copied mem m₀ dst src (r - 8 + 8) n.toNat := copied_congr_lo hcop (by omega)
  rcases hr with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;>
  · refine ctx.code.step o ⟨hcode.1, by simp⟩ rfl (codeRegion_lit (by omega) (by decide)) ?_
    rv_simp [hld, hl, hconv]
    refine ctx.code.step (o + 4) ⟨hcode.2.1, by simp⟩ rfl
      (codeRegion_lit (by omega) (by decide)) ?_
    rv_simp [h13]
    refine ctx.code.step (o + 8) ⟨hcode.2.2.1, by simp⟩ rfl
      (codeRegion_lit (by omega) (by decide)) ?_
    rv_simp [h14, hO]
    refine ctx.code.step (o + 12) ⟨hcode.2.2.2.1, by simp⟩ rfl
      (codeRegion_lit (by omega) (by decide)) ?_
    rv_simp
    refine ctx.code.step (o + 16) ⟨hcode.2.2.2.2, by simp⟩ rfl
      (codeRegion_lit (by omega) (by decide)) ?_
    rv_simp [hst, hs]
    rw [storeLE_combine _ _ _ _ _ _ hb hc ha]
    refine hcont _ _ ?_ ?_ (Copied.extend_down hcop' (by decide) (by omega) (by omega))
    · intro r h1 h2 h3
      simp [writeReg_apply, h1, h2, h3]
    · simp [writeReg_apply]

end CopyRuntime.Leaf
