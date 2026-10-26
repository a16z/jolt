import CopyRuntime.Proof.Leaf.Comb

/-!
# The leaf's forward copy (`dst < src`)

Proved bottom-up in continuation style: each lemma takes an abstract state at
a code offset, described by `FAt J` (the first `J` bytes are copied, `a1`/`a7`
point `J` bytes into the ranges, `a2 = n - J`), and shows that the run ends in
`Returned`. Loops use `Exits.loop`.

Forward copying is safe for overlap because the destination lies below the
source: a store at destination offset `i` hits source offset `i - (src - dst)`,
below every offset that a later load reads in full. The one exception is the
first aligned word `t3`, whose low `a` bytes may already be overwritten; the
`srl` by `8a` discards exactly those (`fwd_comb`).
-/

namespace CopyRuntime.Leaf

/-- The cursor state after `J` bytes of a forward copy. -/
structure FAt (regs₀ : Reg → BitVec 64) (m₀ : Mem) (dst src n : BitVec 64) (J : Nat)
    (regs : Reg → BitVec 64) (mem : Mem) : Prop where
  a1 : regs 11 = src + BitVec.ofNat 64 J
  a7 : regs 17 = dst + BitVec.ofNat 64 J
  a2 : (regs 12).toNat = n.toNat - J
  le : J ≤ n.toNat
  copied : Copied mem m₀ dst src 0 J
  preserved : ∀ r, r ∉ clobbered → regs r = regs₀ r

/-- Build an `FAt` fact whose offsets are only equal (not syntactically) to `J`. -/
theorem FAt.of_eq {regs₀ regs : Reg → BitVec 64} {m₀ mem : Mem} {dst src n : BitVec 64}
    {J a b : Nat} (h1 : regs 11 = src + BitVec.ofNat 64 a)
    (h7 : regs 17 = dst + BitVec.ofNat 64 b) (h2 : (regs 12).toNat = n.toNat - J)
    (hcop : Copied mem m₀ dst src 0 J) (ea : a = J) (eb : b = J) (hle : J ≤ n.toNat)
    (hpres : ∀ r, r ∉ clobbered → regs r = regs₀ r) : FAt regs₀ m₀ dst src n J regs mem := by
  subst ea eb
  exact ⟨h1, h7, h2, hle, hcop, hpres⟩

/-- The registers of the forward combining loops at destination offset `J`
with source word offset `a`: `t2` is the aligned source word below `src + J`,
`t3` holds it (its bytes `a .. 7`, which are the ones `srl` keeps, agree with
the snapshot), and `a3`/`a4` hold the shift amounts `8a`, `64 - 8a`. -/
structure FComb (m₀ : Mem) (src : BitVec 64) (a J : Nat) (regs : Reg → BitVec 64) : Prop where
  t2 : regs 7 = src + BitVec.ofNat 64 (J - a)
  a3 : (regs 13).toNat % 64 = 8 * a
  a4 : (regs 14).toNat % 64 = 64 - 8 * a
  a5 : regs 15 = BitVec.ofNat 64 a
  t3 : regs 28 >>> (8 * a) = loadLE m₀ (src + BitVec.ofNat 64 (J - a)) 8 >>> (8 * a)

variable {cfg : Config} {L dst src n R : BitVec 64}

/-- The byte tail at `0x1dc`: copy the remaining `n - J` bytes, then return. -/
theorem fwd_tail (env : Env cfg L dst src n R) (hle : dst.toNat ≤ src.toNat)
    (regs₀ : Reg → BitVec 64) (m₀ : Mem) (he : EntryRegs regs₀ dst src n R) (J : Nat)
    (regs : Reg → BitVec 64) (mem : Mem) (hF : FAt regs₀ m₀ dst src n J regs mem) :
    Exits cfg (region L) (Returned ⟨L, regs₀, m₀⟩ R dst src n) ⟨L + 0x1dc#64, regs, mem⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hR := env.args.ret_even
  have hsw' := env.args.src_no_wrap
  have hsw : src.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hsw'
  have hdw' := env.args.dst_no_wrap
  have hdw : dst.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hdw'
  have hl := env.sb.load; have hs := env.sb.store
  refine Exits.loop
    (I := fun k s => s.pc = L + 0x1dc#64 ∧ FAt regs₀ m₀ dst src n (J + k) s.regs s.mem)
    (n.toNat - J) (fun k s h => by have := h.2.le; omega) ?_ 0 _ ⟨rfl, by simpa using hF⟩
  intro k s ⟨hpc, hF'⟩ hcont
  obtain ⟨pc, regs', mem'⟩ := s
  simp only at hpc hF'
  subst hpc
  have ha1 := hF'.a1; have ha7 := hF'.a7; have ha2 := hF'.a2; have hJn := hF'.le
  have hra : regs' 1 = R := by rw [hF'.preserved 1 (by decide), he.ra]
  rv_step ctx.code                                        -- 0x1dc beqz a2
  · rename_i hz
    rv_arith_simp at hz
    rv_step ctx.code [hra]                                -- 0x3da ret
    exact .exit env.rout
      (Returned.of_copied he env.args.dst_no_wrap (hF'.copied.congr_hi (by omega))
        hF'.preserved)
  · rename_i hz
    rv_arith_simp at hz
    rv_step ctx.code [ha1, hl]                            -- 0x1e0 lbu t1, 0(a1)
    rv_step ctx.code [ha7, hs]                            -- 0x1e4 sb t1, 0(a7)
    rv_step ctx.code [ha1]                                -- 0x1e8 addi a1, a1, 1
    rv_step ctx.code [ha7]                                -- 0x1ea addi a7, a7, 1
    rv_step ctx.code [Arith.add_ofNat_eq_sub_of_ge]       -- 0x1ec addi a2, a2, -1
    rv_step ctx.code                                      -- 0x1ee j 0x1dc
    refine hcont _ ⟨rfl, ?_⟩
    refine FAt.of_eq (a := J + k + 1) (b := J + k + 1) ?_ ?_ ?_
      (hF'.copied.store_fwd_self hle (by decide) (by omega) (by omega) rfl) (by omega) (by omega)
      (by omega) ?_
    · simp [writeReg_apply]
    · simp [writeReg_apply]
    · simp only [writeReg_apply, Fin.reduceEq, ite_false, ite_true]
      rw [Arith.toNat_sub_ofNat_of_le (by omega)]
      omega
    · rv_preserved hF'.preserved

/-- After the 128-byte loop, at `0x1b0`: `t0 = 16 - a`; the word loop at
`0x1ba` combines one word per iteration while `a2 ≥ 16 - a` (so that the next
aligned source word lies inside the source), then the byte tail. -/
theorem fwd_words (env : Env cfg L dst src n R) (hle : dst.toNat ≤ src.toNat)
    (regs₀ : Reg → BitVec 64) (m₀ : Mem) (he : EntryRegs regs₀ dst src n R)
    (a J : Nat) (ha : 1 ≤ a ∧ a ≤ 7) (hJa : a ≤ J) (hsal : (src.toNat + J) % 8 = a)
    (hdal : (dst.toNat + J) % 8 = 0) (regs : Reg → BitVec 64) (mem : Mem)
    (hF : FAt regs₀ m₀ dst src n J regs mem) (hC : FComb m₀ src a J regs) :
    Exits cfg (region L) (Returned ⟨L, regs₀, m₀⟩ R dst src n) ⟨L + 0x1b0#64, regs, mem⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hsw' := env.args.src_no_wrap
  have hdw' := env.args.dst_no_wrap
  have ha2 := hF.a2; have hJn := hF.le; have h15 := hC.a5
  rv_step ctx.code                                        -- 0x1b0 li t0, 16
  rv_step ctx.code [h15, Arith.ofNat_sub_ofNat_of_le]     -- 0x1b2 sub t0, t0, a5
  rv_step ctx.code                                        -- 0x1b6 bltu a2, t0
  · -- Fewer than `16 - a` bytes left: the byte tail.
    exact fwd_tail env hle regs₀ m₀ he J _ mem
      ⟨by simp [writeReg_apply, hF.a1], by simp [writeReg_apply, hF.a7],
        by simp [writeReg_apply, ha2], hJn, hF.copied, by rv_preserved hF.preserved⟩
  rename_i hge
  rv_arith_simp at hge
  refine Exits.loop
    (I := fun k s => s.pc = L + 0x1ba#64 ∧ FAt regs₀ m₀ dst src n (J + 8 * k) s.regs s.mem ∧
      FComb m₀ src a (J + 8 * k) s.regs ∧ s.regs 5 = BitVec.ofNat 64 (16 - a) ∧
      J + 8 * k + 16 - a ≤ n.toNat)
    (n.toNat / 8) (fun k s h => by have := h.2.2.2.2; omega) ?_ 0 _ ⟨rfl, ?_, ?_, ?_, ?_⟩
  · intro k s ⟨hpc, hF', hC', h5, hroom⟩ hcont
    obtain ⟨pc, regs', mem'⟩ := s
    simp only at hpc hF' hC' h5 hroom
    subst hpc
    refine fwd_comb env hle m₀ a ha 0x1ba 28 29 (Or.inl ⟨rfl, rfl⟩) (imm 8) (imm 0)
      (by decide +kernel) (by decide) (J + 8 * k) (by omega) (by omega) (by omega) (by omega)
      regs' mem' ?_ ?_ hC'.a3 hC'.a4 hC'.t3 hF'.copied ?_
    · rw [hC'.t2]; exact addr_fwd _ _ 8 _ (by omega)
    · rw [hF'.a7]; exact addr_fwd _ _ 0 _ (by omega)
    intro regs'' mem'' hfr h29 hcop
    rw [show (0x1ba + 20 : Nat) = 0x1ce from rfl]
    have h7 : regs'' 7 = src + BitVec.ofNat 64 (J + 8 * k - a) := by
      rw [hfr 7 (by decide) (by decide) (by decide), hC'.t2]
    have h11 : regs'' 11 = src + BitVec.ofNat 64 (J + 8 * k) := by
      rw [hfr 11 (by decide) (by decide) (by decide), hF'.a1]
    have h17 : regs'' 17 = dst + BitVec.ofNat 64 (J + 8 * k) := by
      rw [hfr 17 (by decide) (by decide) (by decide), hF'.a7]
    have h12 : (regs'' 12).toNat = n.toNat - (J + 8 * k) := by
      rw [hfr 12 (by decide) (by decide) (by decide), hF'.a2]
    have h5' : regs'' 5 = BitVec.ofNat 64 (16 - a) := by
      rw [hfr 5 (by decide) (by decide) (by decide), h5]
    rv_step ctx.code [h29, BitVec.zero_add]               -- 0x1ce mv t3, t4
    rv_step ctx.code [h7]                                 -- 0x1d0 addi t2, t2, 8
    rv_step ctx.code [h11]                                -- 0x1d2 addi a1, a1, 8
    rv_step ctx.code [h17]                                -- 0x1d4 addi a7, a7, 8
    rv_step ctx.code [Arith.add_ofNat_eq_sub_of_ge]       -- 0x1d6 addi a2, a2, -8
    rv_step ctx.code [h5']                                -- 0x1d8 bgeu a2, t0
    · -- Another word.
      rename_i hc
      rw [Arith.toNat_sub_ofNat_of_le (by omega)] at hc
      simp only [BitVec.toNat_ofNat] at hc
      refine hcont _ ⟨rfl, ?_, ⟨?_, ?_, ?_, ?_, ?_⟩, ?_, ?_⟩
      · refine FAt.of_eq (a := J + 8 * k + 8) (b := J + 8 * k + 8) ?_ ?_ ?_ hcop
          (by omega) (by omega) (by omega) ?_
        · simp [writeReg_apply]
        · simp [writeReg_apply]
        · simp only [writeReg_apply, Fin.reduceEq, ite_false, ite_true]
          rw [Arith.toNat_sub_ofNat_of_le (by omega)]
          omega
        · rv_preserved
            (pres_frame3 (by decide) (by decide) (by decide) hfr hF'.preserved)
      · simp only [writeReg_apply, Fin.reduceEq, ite_false]
        exact Arith.add_ofNat_congr (by omega)
      · simp only [writeReg_apply, Fin.reduceEq, ite_false]
        rw [hfr 13 (by decide) (by decide) (by decide)]; exact hC'.a3
      · simp only [writeReg_apply, Fin.reduceEq, ite_false]
        rw [hfr 14 (by decide) (by decide) (by decide)]; exact hC'.a4
      · simp only [writeReg_apply, Fin.reduceEq, ite_false]
        rw [hfr 15 (by decide) (by decide) (by decide)]; exact hC'.a5
      · simp only [writeReg_apply, Fin.reduceEq, ite_false, ite_true]
        rw [show J + 8 * (k + 1) - a = J + 8 * k + 8 - a by omega]
      · simp only [writeReg_apply, Fin.reduceEq, ite_false]
        exact h5'
      · omega
    · -- Fewer than `16 - a` bytes left: the byte tail.
      rename_i hc
      rw [Arith.toNat_sub_ofNat_of_le (by omega)] at hc
      simp only [BitVec.toNat_ofNat] at hc
      refine fwd_tail env hle regs₀ m₀ he (J + 8 * k + 8) _ _ ?_
      refine FAt.of_eq (a := J + 8 * k + 8) (b := J + 8 * k + 8) ?_ ?_ ?_ hcop
        rfl rfl (by omega) ?_
      · simp [writeReg_apply]
      · simp [writeReg_apply]
      · simp only [writeReg_apply, Fin.reduceEq, ite_false, ite_true]
        rw [Arith.toNat_sub_ofNat_of_le (by omega)]
        omega
      · rv_preserved (pres_frame3 (by decide) (by decide) (by decide) hfr hF'.preserved)
  · exact FAt.of_eq (a := J) (b := J) (by simp [writeReg_apply, hF.a1])
      (by simp [writeReg_apply, hF.a7]) (by simp [writeReg_apply, ha2]) hF.copied rfl rfl hJn
      (by rv_preserved hF.preserved)
  · exact ⟨by simp [writeReg_apply, hC.t2], by simp [writeReg_apply, hC.a3],
      by simp [writeReg_apply, hC.a4], by simp [writeReg_apply, hC.a5],
      by simp [writeReg_apply, hC.t3]⟩
  · simp [writeReg_apply]
  · omega

set_option synthInstance.maxSize 512 in
/-- The code of the forward 128-byte loop body: eight pairs of combine-and-store
steps, alternating `t3`/`t4`. Checked against the LLVM listing. -/
theorem fwd_bulk_code : ∀ i, i < 8 →
    ((0x5c + 40 * i, Instr.load .double 29 7 (imm ((16 * i + 8 : Nat) : Int)), 4) ∈
        Generated.Disasm.joltCopyMisaligned ∧
      (0x5c + 40 * i + 4, Instr.alu .srl 30 28 13, 4) ∈ Generated.Disasm.joltCopyMisaligned ∧
      (0x5c + 40 * i + 8, Instr.alu .sll 31 29 14, 4) ∈ Generated.Disasm.joltCopyMisaligned ∧
      (0x5c + 40 * i + 12, Instr.alu .or 30 30 31, 4) ∈ Generated.Disasm.joltCopyMisaligned ∧
      (0x5c + 40 * i + 16, Instr.store .double 30 17 (imm ((16 * i : Nat) : Int)), 4) ∈
        Generated.Disasm.joltCopyMisaligned) ∧
    ((0x5c + 40 * i + 20, Instr.load .double 28 7 (imm ((16 * i + 16 : Nat) : Int)), 4) ∈
        Generated.Disasm.joltCopyMisaligned ∧
      (0x5c + 40 * i + 20 + 4, Instr.alu .srl 30 29 13, 4) ∈
        Generated.Disasm.joltCopyMisaligned ∧
      (0x5c + 40 * i + 20 + 8, Instr.alu .sll 31 28 14, 4) ∈
        Generated.Disasm.joltCopyMisaligned ∧
      (0x5c + 40 * i + 20 + 12, Instr.alu .or 30 30 31, 4) ∈
        Generated.Disasm.joltCopyMisaligned ∧
      (0x5c + 40 * i + 20 + 16, Instr.store .double 30 17 (imm ((16 * i + 8 : Nat) : Int)), 4) ∈
        Generated.Disasm.joltCopyMisaligned) := by
  decide +kernel

/-- The straight-line body of the forward 128-byte loop, from pair `i` on:
pair `i` combines the words at source offsets `J + 16i - a`, `J + 16i + 8 - a`
and `J + 16i + 16 - a` into the destination words at `J + 16i` and
`J + 16i + 8`. Proved once by induction on the pairs with an abstract state. -/
theorem fwd_run (env : Env cfg L dst src n R) (hle : dst.toNat ≤ src.toNat) (m₀ : Mem)
    {post : State → Prop} (a : Nat) (ha : 1 ≤ a ∧ a ≤ 7) (J : Nat) (hJa : a ≤ J)
    (hroom : J + 136 - a ≤ n.toNat) (hsal : (src.toNat + J) % 8 = a)
    (hdal : (dst.toNat + J) % 8 = 0) (regsS : Reg → BitVec 64)
    (hS7 : regsS 7 = src + BitVec.ofNat 64 (J - a)) (hS17 : regsS 17 = dst + BitVec.ofNat 64 J)
    (hS13 : (regsS 13).toNat % 64 = 8 * a) (hS14 : (regsS 14).toNat % 64 = 64 - 8 * a)
    (hcont : ∀ (regs : Reg → BitVec 64) (mem : Mem),
      (∀ r, r ≠ 28 → r ≠ 29 → r ≠ 30 → r ≠ 31 → regs r = regsS r) →
      regs 28 >>> (8 * a) = loadLE m₀ (src + BitVec.ofNat 64 (J + 128 - a)) 8 >>> (8 * a) →
      Copied mem m₀ dst src 0 (J + 128) →
      Exits cfg (region L) post ⟨L + 0x19c#64, regs, mem⟩) :
    ∀ i, i ≤ 8 → ∀ (regs : Reg → BitVec 64) (mem : Mem),
      (∀ r, r ≠ 28 → r ≠ 29 → r ≠ 30 → r ≠ 31 → regs r = regsS r) →
      regs 28 >>> (8 * a) = loadLE m₀ (src + BitVec.ofNat 64 (J + 16 * i - a)) 8 >>> (8 * a) →
      Copied mem m₀ dst src 0 (J + 16 * i) →
      Exits cfg (region L) post ⟨L + BitVec.ofNat 64 (0x5c + 40 * i), regs, mem⟩ := by
  suffices h : ∀ d i, 8 - i = d → i ≤ 8 → ∀ (regs : Reg → BitVec 64) (mem : Mem),
      (∀ r, r ≠ 28 → r ≠ 29 → r ≠ 30 → r ≠ 31 → regs r = regsS r) →
      regs 28 >>> (8 * a) = loadLE m₀ (src + BitVec.ofNat 64 (J + 16 * i - a)) 8 >>> (8 * a) →
      Copied mem m₀ dst src 0 (J + 16 * i) →
      Exits cfg (region L) post ⟨L + BitVec.ofNat 64 (0x5c + 40 * i), regs, mem⟩ from
    fun i hi => h _ i rfl hi
  intro d
  induction d with
  | zero =>
    intro i hd hi regs mem hfr h28 hcop
    obtain rfl : i = 8 := by omega
    rw [show (0x5c + 40 * 8 : Nat) = 0x19c from rfl]
    exact hcont regs mem hfr (by rw [h28, show J + 16 * 8 - a = J + 128 - a by omega])
      (hcop.congr_hi (by omega))
  | succ d ih =>
    intro i hd hi regs mem hfr h28 hcop
    obtain ⟨hA, hB⟩ := fwd_bulk_code i (by omega)
    have h7 : regs 7 = src + BitVec.ofNat 64 (J - a) := by
      rw [hfr 7 (by decide) (by decide) (by decide) (by decide), hS7]
    have h17 : regs 17 = dst + BitVec.ofNat 64 J := by
      rw [hfr 17 (by decide) (by decide) (by decide) (by decide), hS17]
    have h13 : (regs 13).toNat % 64 = 8 * a := by
      rw [hfr 13 (by decide) (by decide) (by decide) (by decide), hS13]
    have h14 : (regs 14).toNat % 64 = 64 - 8 * a := by
      rw [hfr 14 (by decide) (by decide) (by decide) (by decide), hS14]
    refine fwd_comb env hle m₀ a ha (0x5c + 40 * i) 28 29 (Or.inl ⟨rfl, rfl⟩) _ _ hA
      (by omega) (J + 16 * i) (by omega) (by omega) (by omega) (by omega) regs mem ?_ ?_
      h13 h14 h28 hcop ?_
    · rw [h7]; exact addr_fwd _ _ _ _ (by omega)
    · rw [h17]; exact addr_fwd _ _ _ _ (by omega)
    intro regs' mem' hfr' h29 hcop'
    have h7' : regs' 7 = src + BitVec.ofNat 64 (J - a) := by
      rw [hfr' 7 (by decide) (by decide) (by decide), h7]
    have h17' : regs' 17 = dst + BitVec.ofNat 64 J := by
      rw [hfr' 17 (by decide) (by decide) (by decide), h17]
    have h13' : (regs' 13).toNat % 64 = 8 * a := by
      rw [hfr' 13 (by decide) (by decide) (by decide), h13]
    have h14' : (regs' 14).toNat % 64 = 64 - 8 * a := by
      rw [hfr' 14 (by decide) (by decide) (by decide), h14]
    refine fwd_comb env hle m₀ a ha (0x5c + 40 * i + 20) 29 28 (Or.inr ⟨rfl, rfl⟩) _ _ hB
      (by omega) (J + 16 * i + 8) (by omega) (by omega) (by omega) (by omega) regs' mem' ?_ ?_
      h13' h14' (by rw [h29]) hcop' ?_
    · rw [h7']; exact addr_fwd _ _ _ _ (by omega)
    · rw [h17']; exact addr_fwd _ _ _ _ (by omega)
    intro regs'' mem'' hfr'' h28'' hcop''
    rw [show 0x5c + 40 * i + 20 + 20 = 0x5c + 40 * (i + 1) by omega]
    refine ih (i + 1) (by omega) (by omega) regs'' mem'' ?_ ?_ (hcop''.congr_hi (by omega))
    · intro r h1 h2 h3 h4
      rw [hfr'' r h1 h3 h4, hfr' r h2 h3 h4, hfr r h1 h2 h3 h4]
    · rw [h28'', show J + 16 * i + 8 + 8 - a = J + 16 * (i + 1) - a by omega]

/-- The 128-byte loop at `0x5c`: sixteen combined words per iteration (`fwd_run`),
repeated while `a2 ≥ 136 - a` (the last word an iteration loads ends at source
offset `J + 136 - a`). -/
theorem fwd_bulk (env : Env cfg L dst src n R) (hle : dst.toNat ≤ src.toNat)
    (regs₀ : Reg → BitVec 64) (m₀ : Mem) (he : EntryRegs regs₀ dst src n R)
    (a J₀ : Nat) (ha : 1 ≤ a ∧ a ≤ 7) (hJa : a ≤ J₀) (hsal : (src.toNat + J₀) % 8 = a)
    (hdal : (dst.toNat + J₀) % 8 = 0) (regs : Reg → BitVec 64) (mem : Mem)
    (hF : FAt regs₀ m₀ dst src n J₀ regs mem) (hC : FComb m₀ src a J₀ regs)
    (h5 : regs 5 = BitVec.ofNat 64 (136 - a)) (hroom : J₀ + 136 - a ≤ n.toNat) :
    Exits cfg (region L) (Returned ⟨L, regs₀, m₀⟩ R dst src n) ⟨L + 0x5c#64, regs, mem⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hsw' := env.args.src_no_wrap
  have hdw' := env.args.dst_no_wrap
  refine Exits.loop
    (I := fun k s => s.pc = L + 0x5c#64 ∧ FAt regs₀ m₀ dst src n (J₀ + 128 * k) s.regs s.mem ∧
      FComb m₀ src a (J₀ + 128 * k) s.regs ∧ s.regs 5 = BitVec.ofNat 64 (136 - a) ∧
      J₀ + 128 * k + 136 - a ≤ n.toNat)
    (n.toNat / 128) (fun k s h => by have := h.2.2.2.2; omega) ?_ 0 _
    ⟨rfl, by simpa using hF, by simpa using hC, h5, by simpa using hroom⟩
  intro k s ⟨hpc, hF', hC', h5', hroom'⟩ hcont
  obtain ⟨pc, regs', mem'⟩ := s
  simp only at hpc hF' hC' h5' hroom'
  subst hpc
  have hrun := fwd_run env hle m₀ a ha (J₀ + 128 * k) (by omega) hroom' (by omega) (by omega)
    regs' hC'.t2 hF'.a7 hC'.a3 hC'.a4 (post := Returned ⟨L, regs₀, m₀⟩ R dst src n) ?_ 0
    (by omega) regs' mem' (fun _ _ _ _ _ => rfl)
  · simp only [Nat.mul_zero, Nat.add_zero] at hrun
    exact hrun hC'.t3 hF'.copied
  intro regs'' mem'' hfr h28 hcop
  have h7 : regs'' 7 = src + BitVec.ofNat 64 (J₀ + 128 * k - a) := by
    rw [hfr 7 (by decide) (by decide) (by decide) (by decide), hC'.t2]
  have h11 : regs'' 11 = src + BitVec.ofNat 64 (J₀ + 128 * k) := by
    rw [hfr 11 (by decide) (by decide) (by decide) (by decide), hF'.a1]
  have h17 : regs'' 17 = dst + BitVec.ofNat 64 (J₀ + 128 * k) := by
    rw [hfr 17 (by decide) (by decide) (by decide) (by decide), hF'.a7]
  have h12 : (regs'' 12).toNat = n.toNat - (J₀ + 128 * k) := by
    rw [hfr 12 (by decide) (by decide) (by decide) (by decide), hF'.a2]
  have h5'' : regs'' 5 = BitVec.ofNat 64 (136 - a) := by
    rw [hfr 5 (by decide) (by decide) (by decide) (by decide), h5']
  have hC'' : FComb m₀ src a (J₀ + 128 * k + 128)
      (writeReg (writeReg (writeReg (writeReg regs'' 7
        (src + BitVec.ofNat 64 (J₀ + 128 * k - a + 128))) 11
        (src + BitVec.ofNat 64 (J₀ + 128 * k + 128))) 17
        (dst + BitVec.ofNat 64 (J₀ + 128 * k + 128))) 12 (regs'' 12 - 128#64)) := by
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · simp only [writeReg_apply, Fin.reduceEq, ite_false, ite_true]
      exact Arith.add_ofNat_congr (by omega)
    · simp only [writeReg_apply, Fin.reduceEq, ite_false]
      rw [hfr 13 (by decide) (by decide) (by decide) (by decide)]; exact hC'.a3
    · simp only [writeReg_apply, Fin.reduceEq, ite_false]
      rw [hfr 14 (by decide) (by decide) (by decide) (by decide)]; exact hC'.a4
    · simp only [writeReg_apply, Fin.reduceEq, ite_false]
      rw [hfr 15 (by decide) (by decide) (by decide) (by decide)]; exact hC'.a5
    · simp only [writeReg_apply, Fin.reduceEq, ite_false]
      rw [h28, show J₀ + 128 * k + 128 - a = J₀ + 128 * k + 128 - a from rfl]
  have hF'' : FAt regs₀ m₀ dst src n (J₀ + 128 * k + 128)
      (writeReg (writeReg (writeReg (writeReg regs'' 7
        (src + BitVec.ofNat 64 (J₀ + 128 * k - a + 128))) 11
        (src + BitVec.ofNat 64 (J₀ + 128 * k + 128))) 17
        (dst + BitVec.ofNat 64 (J₀ + 128 * k + 128))) 12 (regs'' 12 - 128#64)) mem'' := by
    refine ⟨by simp [writeReg_apply], by simp [writeReg_apply], ?_, by omega, hcop, ?_⟩
    · simp only [writeReg_apply, Fin.reduceEq, ite_false, ite_true]
      rw [Arith.toNat_sub_ofNat_of_le (by omega)]
      omega
    · rv_preserved
        (pres_frame4 (by decide) (by decide) (by decide) (by decide) hfr hF'.preserved)
  rv_step ctx.code [h7]                                   -- 0x19c addi t2, t2, 128
  rv_step ctx.code [h11]                                  -- 0x1a0 addi a1, a1, 128
  rv_step ctx.code [h17]                                  -- 0x1a4 addi a7, a7, 128
  rv_step ctx.code [Arith.add_ofNat_eq_sub_of_ge]         -- 0x1a8 addi a2, a2, -128
  rv_step ctx.code [h5'']                                 -- 0x1ac bgeu a2, t0
  · -- Another 128 bytes.
    rename_i hc
    rw [Arith.toNat_sub_ofNat_of_le (by omega)] at hc
    simp only [BitVec.toNat_ofNat] at hc
    refine hcont _ ⟨rfl, ?_, ?_, ?_, ?_⟩
    · rw [show J₀ + 128 * (k + 1) = J₀ + 128 * k + 128 by omega]; exact hF''
    · rw [show J₀ + 128 * (k + 1) = J₀ + 128 * k + 128 by omega]; exact hC''
    · simp only [writeReg_apply, Fin.reduceEq, ite_false]; exact h5''
    · omega
  · -- On to the word loop.
    exact fwd_words env hle regs₀ m₀ he a (J₀ + 128 * k + 128) ha (by omega) (by omega)
      (by omega) _ _ hF'' hC''

/-- The setup at `0x3e`, with `a7 = dst + J` aligned: `t2` is the aligned source
word below `src + J`, `a5 = a` its offset (nonzero, as the offsets are
mismatched), `a3 = 8a`, `a4 = 64 - 8a`, `t3` its first word, `t0 = 136 - a`. -/
theorem fwd_setup (env : Env cfg L dst src n R) (hle : dst.toNat ≤ src.toNat)
    (regs₀ : Reg → BitVec 64) (m₀ : Mem) (he : EntryRegs regs₀ dst src n R)
    (J : Nat) (hJ : J ≤ 15) (hdal : (dst.toNat + J) % 8 = 0)
    (hJa : (src.toNat + J) % 8 ≤ J) (regs : Reg → BitVec 64) (mem : Mem)
    (hF : FAt regs₀ m₀ dst src n J regs mem) :
    Exits cfg (region L) (Returned ⟨L, regs₀, m₀⟩ R dst src n) ⟨L + 0x3e#64, regs, mem⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hsw' := env.args.src_no_wrap
  have hsw : src.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hsw'
  have hdw' := env.args.dst_no_wrap
  have hmis := env.mis; have hbig := env.big
  have hl := env.sb.load
  have ha1 := hF.a1
  obtain ⟨a, had⟩ : ∃ a, a = (src.toNat + J) % 8 := ⟨_, rfl⟩
  have ha : 1 ≤ a ∧ a ≤ 7 := by omega
  have hW := and_neg8_add (x := src) (J := J) (a := a) (by omega) (by omega) (by omega)
  have hA := and_seven_add (x := src) (J := J) (a := a) (by omega) (by omega)
  have hA3 := ofNat_shl3 (a := a) (by omega)
  rv_step ctx.code [ha1, hW]                              -- 0x3e andi t2, a1, -8
  rv_step ctx.code [ha1, hA]                              -- 0x42 andi a5, a1, 7
  rv_step ctx.code [hA3]                                  -- 0x46 slli a3, a5, 3
  rv_step ctx.code                                        -- 0x4a li a4, 64
  rv_step ctx.code [Arith.ofNat_sub_ofNat_of_le]          -- 0x4e sub a4, a4, a3
  rv_step ctx.code [hl]                                   -- 0x50 ld t3, 0(t2)
  rv_step ctx.code                                        -- 0x54 li t0, 136
  rv_step ctx.code [Arith.ofNat_sub_ofNat_of_le]          -- 0x58 sub t0, t0, a5
  refine fwd_bulk env hle regs₀ m₀ he a J ha (by omega) (by omega) hdal _ mem ?_ ?_ ?_
    (by omega)
  · exact FAt.of_eq (a := J) (b := J) (by simp [writeReg_apply, ha1])
      (by simp [writeReg_apply, hF.a7]) (by simp [writeReg_apply, hF.a2]) hF.copied rfl rfl
      hF.le (by rv_preserved hF.preserved)
  · refine ⟨by simp [writeReg_apply], ?_, ?_, by simp [writeReg_apply], ?_⟩
    · simp only [writeReg_apply, Fin.reduceEq, ite_false, ite_true]
      exact toNat_ofNat_mod64 (by omega)
    · simp only [writeReg_apply, Fin.reduceEq, ite_false, ite_true]
      exact toNat_ofNat_mod64 (by omega)
    · simp only [writeReg_apply, Fin.reduceEq, ite_false, ite_true]
      apply loadLE_shr_congr
      intro i hi1 hi2
      rw [Arith.add_lit_lit]
      exact hF.copied.read_fwd hle (by omega) (by omega)
  · simp [writeReg_apply]

/-- `.Lmis_forward_first` at `0x2a`: the first aligned source word would start
below `src`, so copy eight more bytes (counter `t0`). -/
theorem fwd_first (env : Env cfg L dst src n R) (hle : dst.toNat ≤ src.toNat)
    (regs₀ : Reg → BitVec 64) (m₀ : Mem) (he : EntryRegs regs₀ dst src n R)
    (p : Nat) (hp : p ≤ 7) (hdal : (dst.toNat + p) % 8 = 0)
    (regs : Reg → BitVec 64) (mem : Mem) (hF : FAt regs₀ m₀ dst src n p regs mem)
    (h5 : (regs 5).toNat = 8) :
    Exits cfg (region L) (Returned ⟨L, regs₀, m₀⟩ R dst src n) ⟨L + 0x2a#64, regs, mem⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hsw' := env.args.src_no_wrap
  have hsw : src.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hsw'
  have hdw' := env.args.dst_no_wrap
  have hdw : dst.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hdw'
  have hbig := env.big
  have hl := env.sb.load; have hs := env.sb.store
  refine Exits.loop
    (I := fun k s => s.pc = L + 0x2a#64 ∧ FAt regs₀ m₀ dst src n (p + k) s.regs s.mem ∧
      (s.regs 5).toNat = 8 - k ∧ k < 8)
    7 (fun k s h => by have := h.2.2.2; omega) ?_ 0 _ ⟨rfl, by simpa using hF, h5, by omega⟩
  intro k s ⟨hpc, hF', h5', hk⟩ hcont
  obtain ⟨pc, regs', mem'⟩ := s
  simp only at hpc hF' h5' hk
  subst hpc
  have ha1 := hF'.a1; have ha7 := hF'.a7; have ha2 := hF'.a2; have hJn := hF'.le
  rv_step ctx.code [ha1, hl]                              -- 0x2a lbu t1, 0(a1)
  rv_step ctx.code [ha7, hs]                              -- 0x2e sb t1, 0(a7)
  rv_step ctx.code [ha1]                                  -- 0x32 addi a1, a1, 1
  rv_step ctx.code [ha7]                                  -- 0x34 addi a7, a7, 1
  rv_step ctx.code [Arith.add_ofNat_eq_sub_of_ge]         -- 0x36 addi a2, a2, -1
  rv_step ctx.code [Arith.add_ofNat_eq_sub_of_ge]         -- 0x38 addi t0, t0, -1
  have h5'' : (regs' 5 - 1#64).toNat = 7 - k := by
    rw [Arith.toNat_sub_ofNat_of_le (by omega)]; omega
  have hF'' : FAt regs₀ m₀ dst src n (p + k + 1)
      (writeReg (writeReg (writeReg (writeReg (writeReg regs' 6
        (loadLE mem' (src + BitVec.ofNat 64 (p + k)) 1)) 11
        (src + BitVec.ofNat 64 (p + k + 1))) 17 (dst + BitVec.ofNat 64 (p + k + 1))) 12
        (regs' 12 - 1#64)) 5 (regs' 5 - 1#64))
      (storeLE mem' (dst + BitVec.ofNat 64 (p + k))
        (loadLE mem' (src + BitVec.ofNat 64 (p + k)) 1) 1) := by
    refine ⟨by simp [writeReg_apply], by simp [writeReg_apply], ?_, by omega,
      hF'.copied.store_fwd_self hle (by decide) (by omega) (by omega) rfl,
      by rv_preserved hF'.preserved⟩
    simp only [writeReg_apply, Fin.reduceEq, ite_false, ite_true]
    rw [Arith.toNat_sub_ofNat_of_le (by omega)]
    omega
  rv_step ctx.code                                        -- 0x3a bnez t0
  · -- More bytes.
    rename_i hc
    rw [bv_eq_zero_iff h5''] at hc
    refine hcont _ ⟨rfl, ?_, ?_, by omega⟩
    · rw [show p + (k + 1) = p + k + 1 by omega]; exact hF''
    · simp only [writeReg_apply, Fin.reduceEq, ite_false, ite_true]; omega
  · rename_i hc
    rw [bv_eq_zero_iff h5''] at hc
    exact fwd_setup env hle regs₀ m₀ he (p + k + 1) (by omega) (by omega) (by omega) _ _ hF''

/-- `.Lmis_forward_bound` at `0x20`, with `a7 = dst + p` aligned: if the aligned
source word below `src + p` starts below `src`, copy eight more bytes first. -/
theorem fwd_bound (env : Env cfg L dst src n R) (hle : dst.toNat ≤ src.toNat)
    (regs₀ : Reg → BitVec 64) (m₀ : Mem) (he : EntryRegs regs₀ dst src n R)
    (p : Nat) (hp : p ≤ 7) (hdal : (dst.toNat + p) % 8 = 0)
    (regs : Reg → BitVec 64) (mem : Mem) (hF : FAt regs₀ m₀ dst src n p regs mem)
    (h16 : regs 16 = src) :
    Exits cfg (region L) (Returned ⟨L, regs₀, m₀⟩ R dst src n) ⟨L + 0x20#64, regs, mem⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hsw' := env.args.src_no_wrap
  have hbig := env.big
  have ha1 := hF.a1
  rv_step ctx.code [ha1]                                  -- 0x20 andi t2, a1, -8
  rv_step ctx.code [h16, Arith.toNat_and_neg8]            -- 0x24 bgeu t2, a6
  · rename_i hc
    rv_arith_simp at hc
    refine fwd_setup env hle regs₀ m₀ he p (by omega) hdal (by omega) _ mem ?_
    exact FAt.of_eq (a := p) (b := p) (by simp [writeReg_apply, ha1])
      (by simp [writeReg_apply, hF.a7]) (by simp [writeReg_apply, hF.a2]) hF.copied rfl rfl
      hF.le (by rv_preserved hF.preserved)
  · rv_step ctx.code                                      -- 0x28 li t0, 8
    refine fwd_first env hle regs₀ m₀ he p hp hdal _ mem ?_ (by simp [writeReg_apply])
    exact FAt.of_eq (a := p) (b := p) (by simp [writeReg_apply, ha1])
      (by simp [writeReg_apply, hF.a7]) (by simp [writeReg_apply, hF.a2]) hF.copied rfl rfl
      hF.le (by rv_preserved hF.preserved)

/-- `.Lmis_forward_peel` at `0x8`: copy single bytes until `a7` is 8-byte aligned. -/
theorem fwd_peel (env : Env cfg L dst src n R) (hle : dst.toNat ≤ src.toNat)
    (regs₀ : Reg → BitVec 64) (m₀ : Mem) (he : EntryRegs regs₀ dst src n R)
    (regs : Reg → BitVec 64) (mem : Mem) (hF : FAt regs₀ m₀ dst src n 0 regs mem)
    (h16 : regs 16 = src) :
    Exits cfg (region L) (Returned ⟨L, regs₀, m₀⟩ R dst src n) ⟨L + 8#64, regs, mem⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hsw' := env.args.src_no_wrap
  have hsw : src.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hsw'
  have hdw' := env.args.dst_no_wrap
  have hdw : dst.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hdw'
  have hbig := env.big
  have hl := env.sb.load; have hs := env.sb.store
  refine Exits.loop
    (I := fun k s => s.pc = L + 8#64 ∧ FAt regs₀ m₀ dst src n k s.regs s.mem ∧
      s.regs 16 = src ∧ k ≤ (8 - dst.toNat % 8) % 8)
    7 (fun k s h => by have := h.2.2.2; omega) ?_ 0 _ ⟨rfl, hF, h16, by omega⟩
  intro k s ⟨hpc, hF', h16', hk⟩ hcont
  obtain ⟨pc, regs', mem'⟩ := s
  simp only at hpc hF' h16' hk
  subst hpc
  have ha1 := hF'.a1; have ha7 := hF'.a7; have ha2 := hF'.a2; have hJn := hF'.le
  rv_step ctx.code [ha7]                                  -- 0x08 andi t0, a7, 7
  rv_step ctx.code [Arith.and_seven_eq_zero_iff]          -- 0x0c beqz t0
  · -- Aligned.
    rename_i hc
    rv_arith_simp at hc
    refine fwd_bound env hle regs₀ m₀ he k (by omega) (by omega) _ mem' ?_
      (by simp [writeReg_apply, h16'])
    exact FAt.of_eq (a := k) (b := k) (by simp [writeReg_apply, ha1])
      (by simp [writeReg_apply, ha7]) (by simp [writeReg_apply, ha2]) hF'.copied rfl rfl
      hJn (by rv_preserved hF'.preserved)
  · rename_i hc
    rv_arith_simp at hc
    rv_step ctx.code [ha1, hl]                            -- 0x10 lbu t1, 0(a1)
    rv_step ctx.code [ha7, hs]                            -- 0x14 sb t1, 0(a7)
    rv_step ctx.code [ha1]                                -- 0x18 addi a1, a1, 1
    rv_step ctx.code [ha7]                                -- 0x1a addi a7, a7, 1
    rv_step ctx.code [Arith.add_ofNat_eq_sub_of_ge]       -- 0x1c addi a2, a2, -1
    rv_step ctx.code                                      -- 0x1e j 0x8
    refine hcont _ ⟨rfl, ?_, by simp [writeReg_apply, h16'], by omega⟩
    refine ⟨by simp [writeReg_apply], by simp [writeReg_apply], ?_, by omega,
      hF'.copied.store_fwd_self hle (by decide) (by omega) (by omega) rfl,
      by rv_preserved hF'.preserved⟩
    simp only [writeReg_apply, Fin.reduceEq, ite_false, ite_true]
    rw [Arith.toNat_sub_ofNat_of_le (by omega)]
    omega

/-- The forward copy from `0x6` (`mv a6, a1`), reached for `dst < src` with
`a7 = dst`. -/
theorem fwd_path (env : Env cfg L dst src n R) (hlt : dst.toNat < src.toNat)
    (regs₀ : Reg → BitVec 64) (m₀ : Mem) (he : EntryRegs regs₀ dst src n R)
    (regs : Reg → BitVec 64) (h11 : regs 11 = src) (h17 : regs 17 = dst) (h12 : regs 12 = n)
    (hpres : ∀ r, r ∉ clobbered → regs r = regs₀ r) :
    Exits cfg (region L) (Returned ⟨L, regs₀, m₀⟩ R dst src n) ⟨L + 6#64, regs, m₀⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  rv_step ctx.code [h11, BitVec.zero_add]                 -- 0x06 mv a6, a1
  refine fwd_peel env (Nat.le_of_lt hlt) regs₀ m₀ he _ m₀ ?_ (by simp [writeReg_apply])
  exact ⟨by simp [writeReg_apply, h11], by simp [writeReg_apply, h17],
    by simp [writeReg_apply, h12], Nat.zero_le _, Copied.empty _ _ _ _,
    by rv_preserved hpres⟩

end CopyRuntime.Leaf
