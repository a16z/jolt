import CopyRuntime.Proof.Leaf.Comb

/-!
# The leaf's backward copy (`src < dst`)

The mirror image of `Leaf/Forward.lean`. `BAt r` describes a state with `r`
bytes left: `a1 = src + r`, `a7 = dst + r`, `a2 = r`, and the top part
`[r, n)` of the destination already holds the snapshot copy. Loads read only
source offsets below `r`, which no store has reached because the destination
lies above the source (`Copied.loadLE_bwd`). Only the carried word `t3` may
already hold copied bytes, and the combining shift discards them (`bwd_comb`).

The bound check `addi t2, a1, 7; andi t2, t2, -8` does not wrap because
`a7 = dst + r` is 8-byte aligned and `dst + n < 2^64`, so
`src + r + 7 < dst + r + 7 < 2^64`.
-/

namespace CopyRuntime.Leaf

/-- The cursor state of a backward copy with `r` bytes left. -/
structure BAt (regs₀ : Reg → BitVec 64) (m₀ : Mem) (dst src n : BitVec 64) (r : Nat)
    (regs : Reg → BitVec 64) (mem : Mem) : Prop where
  a1 : regs 11 = src + BitVec.ofNat 64 r
  a7 : regs 17 = dst + BitVec.ofNat 64 r
  a2 : (regs 12).toNat = r
  le : r ≤ n.toNat
  copied : Copied mem m₀ dst src r n.toNat
  preserved : ∀ r, r ∉ clobbered → regs r = regs₀ r

theorem BAt.of_eq {regs₀ regs : Reg → BitVec 64} {m₀ mem : Mem} {dst src n : BitVec 64}
    {r a b : Nat} (h1 : regs 11 = src + BitVec.ofNat 64 a)
    (h7 : regs 17 = dst + BitVec.ofNat 64 b) (h2 : (regs 12).toNat = r)
    (hcop : Copied mem m₀ dst src r n.toNat) (ea : a = r) (eb : b = r) (hle : r ≤ n.toNat)
    (hpres : ∀ r, r ∉ clobbered → regs r = regs₀ r) : BAt regs₀ m₀ dst src n r regs mem := by
  subst ea eb
  exact ⟨h1, h7, h2, hle, hcop, hpres⟩

/-- The registers of the backward combining loops with `r` bytes left and
source word offset `a`: `t2` is the aligned source word below `src + r`, `t3`
holds it (its bytes `0 .. a-1`, which are the ones `sll` keeps, agree with the
snapshot). -/
structure BComb (m₀ : Mem) (src : BitVec 64) (a r : Nat) (regs : Reg → BitVec 64) : Prop where
  t2 : regs 7 = src + BitVec.ofNat 64 (r - a)
  a3 : (regs 13).toNat % 64 = 8 * a
  a4 : (regs 14).toNat % 64 = 64 - 8 * a
  a5 : regs 15 = BitVec.ofNat 64 a
  t3 : regs 28 <<< (64 - 8 * a) = loadLE m₀ (src + BitVec.ofNat 64 (r - a)) 8 <<< (64 - 8 * a)

variable {cfg : Config} {L dst src n R : BitVec 64}

/-- The byte tail at `0x3c8`: copy the remaining `r` bytes downwards, then
return. -/
theorem bwd_tail (env : Env cfg L dst src n R) (hsd : src.toNat ≤ dst.toNat)
    (regs₀ : Reg → BitVec 64) (m₀ : Mem) (he : EntryRegs regs₀ dst src n R) (r₀ : Nat)
    (regs : Reg → BitVec 64) (mem : Mem) (hF : BAt regs₀ m₀ dst src n r₀ regs mem) :
    Exits cfg (region L) (Returned ⟨L, regs₀, m₀⟩ R dst src n) ⟨L + 0x3c8#64, regs, mem⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hR := env.args.ret_even
  have hsw' := env.args.src_no_wrap
  have hsw : src.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hsw'
  have hdw' := env.args.dst_no_wrap
  have hdw : dst.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hdw'
  have hl := env.sb.load; have hs := env.sb.store
  refine Exits.loop
    (I := fun k s => s.pc = L + 0x3c8#64 ∧ k ≤ r₀ ∧ BAt regs₀ m₀ dst src n (r₀ - k) s.regs s.mem)
    r₀ (fun k s h => h.2.1) ?_ 0 _ ⟨rfl, Nat.zero_le _, by simpa using hF⟩
  intro k s ⟨hpc, hk, hF'⟩ hcont
  obtain ⟨pc, regs', mem'⟩ := s
  simp only at hpc hF'
  subst hpc
  have ha1 := hF'.a1; have ha7 := hF'.a7; have ha2 := hF'.a2; have hrn := hF'.le
  have hra : regs' 1 = R := by rw [hF'.preserved 1 (by decide), he.ra]
  rv_step ctx.code                                        -- 0x3c8 beqz a2
  · rename_i hz
    rv_arith_simp at hz
    rv_step ctx.code [hra]                                -- 0x3da ret
    exact .exit env.rout
      (Returned.of_copied he env.args.dst_no_wrap (hF'.copied.congr_lo (by omega))
        hF'.preserved)
  · rename_i hz
    rv_arith_simp at hz
    rv_step ctx.code [ha1, hl, Arith.ofNat_add_neg_lit]   -- 0x3ca lbu t1, -1(a1)
    rv_step ctx.code [ha7, hs, Arith.ofNat_add_neg_lit]   -- 0x3ce sb t1, -1(a7)
    rv_step ctx.code [ha1, Arith.ofNat_add_neg_lit]       -- 0x3d2 addi a1, a1, -1
    rv_step ctx.code [ha7, Arith.ofNat_add_neg_lit]       -- 0x3d4 addi a7, a7, -1
    rv_step ctx.code [Arith.add_ofNat_eq_sub_of_ge]       -- 0x3d6 addi a2, a2, -1
    rv_step ctx.code                                      -- 0x3d8 j 0x3c8
    refine hcont _ ⟨rfl, by omega, ?_⟩
    refine BAt.of_eq (a := r₀ - k - 1) (b := r₀ - k - 1) ?_ ?_ ?_
      (Copied.store_bwd_self (hF'.copied.congr_lo (by omega)) hsd (by decide) hdw' (by omega)
        (by omega)) (by omega) (by omega)
      (by omega) ?_
    · simp [writeReg_apply]
    · simp [writeReg_apply]
    · simp only [writeReg_apply, Fin.reduceEq, ite_false, ite_true]
      rw [Arith.toNat_sub_ofNat_of_le (by omega)]
      omega
    · rv_preserved hF'.preserved

/-- After the 128-byte loop, at `0x39e`: `t0 = a + 8`; the word loop at `0x3a6`
combines one word per iteration while `a2 ≥ a + 8` (so that the next aligned
source word lies inside the source), then the byte tail. -/
theorem bwd_words (env : Env cfg L dst src n R) (hsd : src.toNat ≤ dst.toNat)
    (regs₀ : Reg → BitVec 64) (m₀ : Mem) (he : EntryRegs regs₀ dst src n R)
    (a r₀ : Nat) (ha : 1 ≤ a ∧ a ≤ 7) (har : a ≤ r₀) (hsal : (src.toNat + r₀) % 8 = a)
    (hdal : (dst.toNat + r₀) % 8 = 0) (regs : Reg → BitVec 64) (mem : Mem)
    (hF : BAt regs₀ m₀ dst src n r₀ regs mem) (hC : BComb m₀ src a r₀ regs) :
    Exits cfg (region L) (Returned ⟨L, regs₀, m₀⟩ R dst src n) ⟨L + 0x39e#64, regs, mem⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hsw' := env.args.src_no_wrap
  have hdw' := env.args.dst_no_wrap
  have ha2 := hF.a2; have hrn := hF.le; have h15 := hC.a5
  have ha8 : BitVec.ofNat 64 a + 8#64 = BitVec.ofNat 64 (a + 8) := (BitVec.ofNat_add _ _).symm
  rv_step ctx.code [h15, ha8]                             -- 0x39e addi t0, a5, 8
  rv_step ctx.code                                        -- 0x3a2 bltu a2, t0
  · -- Fewer than `a + 8` bytes left: the byte tail.
    exact bwd_tail env hsd regs₀ m₀ he r₀ _ mem
      ⟨by simp [writeReg_apply, hF.a1], by simp [writeReg_apply, hF.a7],
        by simp [writeReg_apply, ha2], hrn, hF.copied, by rv_preserved hF.preserved⟩
  rename_i hge
  rv_arith_simp at hge
  refine Exits.loop
    (I := fun k s => s.pc = L + 0x3a6#64 ∧ 8 * k ≤ r₀ ∧
      BAt regs₀ m₀ dst src n (r₀ - 8 * k) s.regs s.mem ∧
      BComb m₀ src a (r₀ - 8 * k) s.regs ∧ s.regs 5 = BitVec.ofNat 64 (a + 8) ∧
      a + 8 ≤ r₀ - 8 * k)
    (r₀ / 8) (fun k s h => by have := h.2.1; omega) ?_ 0 _ ⟨rfl, Nat.zero_le _, ?_, ?_, ?_, ?_⟩
  · intro k s ⟨hpc, hk, hF', hC', h5, hroom⟩ hcont
    obtain ⟨pc, regs', mem'⟩ := s
    simp only at hpc hF' hC' h5 hroom
    subst hpc
    refine bwd_comb env hsd m₀ a ha 0x3a6 28 29 (Or.inl ⟨rfl, rfl⟩) (imm (-8)) (imm (-8))
      (by decide +kernel) (by decide) (r₀ - 8 * k) (by omega) hF'.le (by omega) (by omega)
      regs' mem' ?_ ?_ hC'.a3 hC'.a4 hC'.t3 hF'.copied ?_
    · rw [hC'.t2]; exact addr_bwd _ _ 8 _ (by omega)
    · rw [hF'.a7]; exact addr_bwd _ _ 8 _ (by omega)
    intro regs'' mem'' hfr h29 hcop
    rw [show (0x3a6 + 20 : Nat) = 0x3ba from rfl]
    have h7 : regs'' 7 = src + BitVec.ofNat 64 (r₀ - 8 * k - a) := by
      rw [hfr 7 (by decide) (by decide) (by decide), hC'.t2]
    have h11 : regs'' 11 = src + BitVec.ofNat 64 (r₀ - 8 * k) := by
      rw [hfr 11 (by decide) (by decide) (by decide), hF'.a1]
    have h17 : regs'' 17 = dst + BitVec.ofNat 64 (r₀ - 8 * k) := by
      rw [hfr 17 (by decide) (by decide) (by decide), hF'.a7]
    have h12 : (regs'' 12).toNat = r₀ - 8 * k := by
      rw [hfr 12 (by decide) (by decide) (by decide), hF'.a2]
    have h5' : regs'' 5 = BitVec.ofNat 64 (a + 8) := by
      rw [hfr 5 (by decide) (by decide) (by decide), h5]
    rv_step ctx.code [h29, BitVec.zero_add]               -- 0x3ba mv t3, t4
    rv_step ctx.code [h7, Arith.ofNat_add_neg_lit]        -- 0x3bc addi t2, t2, -8
    rv_step ctx.code [h11, Arith.ofNat_add_neg_lit]       -- 0x3be addi a1, a1, -8
    rv_step ctx.code [h17, Arith.ofNat_add_neg_lit]       -- 0x3c0 addi a7, a7, -8
    rv_step ctx.code [Arith.add_ofNat_eq_sub_of_ge]       -- 0x3c2 addi a2, a2, -8
    have hF'' : BAt regs₀ m₀ dst src n (r₀ - 8 * k - 8)
        (writeReg (writeReg (writeReg (writeReg (writeReg regs'' 28
          (loadLE m₀ (src + BitVec.ofNat 64 (r₀ - 8 * k - 8 - a)) 8)) 7
          (src + BitVec.ofNat 64 (r₀ - 8 * k - a - 8))) 11
          (src + BitVec.ofNat 64 (r₀ - 8 * k - 8))) 17
          (dst + BitVec.ofNat 64 (r₀ - 8 * k - 8))) 12 (regs'' 12 - 8#64)) mem'' := by
      refine ⟨by simp [writeReg_apply], by simp [writeReg_apply], ?_, by omega, hcop, ?_⟩
      · simp only [writeReg_apply, Fin.reduceEq, ite_false, ite_true]
        rw [Arith.toNat_sub_ofNat_of_le (by omega)]
        omega
      · rv_preserved (pres_frame3 (by decide) (by decide) (by decide) hfr hF'.preserved)
    have hC'' : BComb m₀ src a (r₀ - 8 * k - 8)
        (writeReg (writeReg (writeReg (writeReg (writeReg regs'' 28
          (loadLE m₀ (src + BitVec.ofNat 64 (r₀ - 8 * k - 8 - a)) 8)) 7
          (src + BitVec.ofNat 64 (r₀ - 8 * k - a - 8))) 11
          (src + BitVec.ofNat 64 (r₀ - 8 * k - 8))) 17
          (dst + BitVec.ofNat 64 (r₀ - 8 * k - 8))) 12 (regs'' 12 - 8#64)) := by
      refine ⟨?_, ?_, ?_, ?_, ?_⟩
      · simp only [writeReg_apply, Fin.reduceEq, ite_false, ite_true]
        exact Arith.add_ofNat_congr (by omega)
      · simp only [writeReg_apply, Fin.reduceEq, ite_false]
        rw [hfr 13 (by decide) (by decide) (by decide)]; exact hC'.a3
      · simp only [writeReg_apply, Fin.reduceEq, ite_false]
        rw [hfr 14 (by decide) (by decide) (by decide)]; exact hC'.a4
      · simp only [writeReg_apply, Fin.reduceEq, ite_false]
        rw [hfr 15 (by decide) (by decide) (by decide)]; exact hC'.a5
      · simp only [writeReg_apply, Fin.reduceEq, ite_false, ite_true]
    rv_step ctx.code [h5']                                -- 0x3c4 bgeu a2, t0
    · -- Another word.
      rename_i hc
      rw [Arith.toNat_sub_ofNat_of_le (by omega)] at hc
      simp only [BitVec.toNat_ofNat] at hc
      refine hcont _ ⟨rfl, by omega, ?_, ?_, ?_, ?_⟩
      · rw [show r₀ - 8 * (k + 1) = r₀ - 8 * k - 8 by omega]; exact hF''
      · rw [show r₀ - 8 * (k + 1) = r₀ - 8 * k - 8 by omega]; exact hC''
      · simp only [writeReg_apply, Fin.reduceEq, ite_false]; exact h5'
      · omega
    · -- Fewer than `a + 8` bytes left: the byte tail.
      exact bwd_tail env hsd regs₀ m₀ he _ _ _ hF''
  · exact BAt.of_eq (a := r₀) (b := r₀) (by simp [writeReg_apply, hF.a1])
      (by simp [writeReg_apply, hF.a7]) (by simp [writeReg_apply, ha2]) (by simpa using hF.copied)
      (by simp) (by simp) (by simp; omega) (by rv_preserved hF.preserved)
  · exact ⟨by simp [writeReg_apply, hC.t2], by simp [writeReg_apply, hC.a3],
      by simp [writeReg_apply, hC.a4], by simp [writeReg_apply, hC.a5],
      by simp [writeReg_apply, hC.t3]⟩
  · simp [writeReg_apply]
  · omega

set_option synthInstance.maxSize 512 in
/-- The code of the backward 128-byte loop body: eight pairs of combine-and-store
steps, alternating `t4`/`t3`. Checked against the LLVM listing. -/
theorem bwd_bulk_code : ∀ i, i < 8 →
    ((0x24a + 40 * i, Instr.load .double 29 7 (imm (-((16 * i + 8 : Nat) : Int))), 4) ∈
        Generated.Disasm.joltCopyMisaligned ∧
      (0x24a + 40 * i + 4, Instr.alu .srl 30 29 13, 4) ∈ Generated.Disasm.joltCopyMisaligned ∧
      (0x24a + 40 * i + 8, Instr.alu .sll 31 28 14, 4) ∈ Generated.Disasm.joltCopyMisaligned ∧
      (0x24a + 40 * i + 12, Instr.alu .or 30 30 31, 4) ∈ Generated.Disasm.joltCopyMisaligned ∧
      (0x24a + 40 * i + 16, Instr.store .double 30 17 (imm (-((16 * i + 8 : Nat) : Int))), 4) ∈
        Generated.Disasm.joltCopyMisaligned) ∧
    ((0x24a + 40 * i + 20, Instr.load .double 28 7 (imm (-((16 * i + 16 : Nat) : Int))), 4) ∈
        Generated.Disasm.joltCopyMisaligned ∧
      (0x24a + 40 * i + 20 + 4, Instr.alu .srl 30 28 13, 4) ∈
        Generated.Disasm.joltCopyMisaligned ∧
      (0x24a + 40 * i + 20 + 8, Instr.alu .sll 31 29 14, 4) ∈
        Generated.Disasm.joltCopyMisaligned ∧
      (0x24a + 40 * i + 20 + 12, Instr.alu .or 30 30 31, 4) ∈
        Generated.Disasm.joltCopyMisaligned ∧
      (0x24a + 40 * i + 20 + 16,
          Instr.store .double 30 17 (imm (-((16 * i + 16 : Nat) : Int))), 4) ∈
        Generated.Disasm.joltCopyMisaligned) := by
  decide +kernel

/-- The straight-line body of the backward 128-byte loop, from pair `i` on:
with `r` bytes left at the start of the iteration, pair `i` combines the words
at source offsets `r - 16i - a`, `r - 16i - 8 - a` and `r - 16i - 16 - a` into
the destination words at `r - 16i - 8` and `r - 16i - 16`. -/
theorem bwd_run (env : Env cfg L dst src n R) (hsd : src.toNat ≤ dst.toNat) (m₀ : Mem)
    {post : State → Prop} (a : Nat) (ha : 1 ≤ a ∧ a ≤ 7) (r : Nat) (hra : a + 128 ≤ r)
    (hrn : r ≤ n.toNat) (hsal : (src.toNat + r) % 8 = a)
    (hdal : (dst.toNat + r) % 8 = 0) (regsS : Reg → BitVec 64)
    (hS7 : regsS 7 = src + BitVec.ofNat 64 (r - a)) (hS17 : regsS 17 = dst + BitVec.ofNat 64 r)
    (hS13 : (regsS 13).toNat % 64 = 8 * a) (hS14 : (regsS 14).toNat % 64 = 64 - 8 * a)
    (hcont : ∀ (regs : Reg → BitVec 64) (mem : Mem),
      (∀ r, r ≠ 28 → r ≠ 29 → r ≠ 30 → r ≠ 31 → regs r = regsS r) →
      regs 28 <<< (64 - 8 * a) =
        loadLE m₀ (src + BitVec.ofNat 64 (r - 128 - a)) 8 <<< (64 - 8 * a) →
      Copied mem m₀ dst src (r - 128) n.toNat →
      Exits cfg (region L) post ⟨L + 0x38a#64, regs, mem⟩) :
    ∀ i, i ≤ 8 → ∀ (regs : Reg → BitVec 64) (mem : Mem),
      (∀ r, r ≠ 28 → r ≠ 29 → r ≠ 30 → r ≠ 31 → regs r = regsS r) →
      regs 28 <<< (64 - 8 * a) =
        loadLE m₀ (src + BitVec.ofNat 64 (r - 16 * i - a)) 8 <<< (64 - 8 * a) →
      Copied mem m₀ dst src (r - 16 * i) n.toNat →
      Exits cfg (region L) post ⟨L + BitVec.ofNat 64 (0x24a + 40 * i), regs, mem⟩ := by
  suffices h : ∀ d i, 8 - i = d → i ≤ 8 → ∀ (regs : Reg → BitVec 64) (mem : Mem),
      (∀ r, r ≠ 28 → r ≠ 29 → r ≠ 30 → r ≠ 31 → regs r = regsS r) →
      regs 28 <<< (64 - 8 * a) =
        loadLE m₀ (src + BitVec.ofNat 64 (r - 16 * i - a)) 8 <<< (64 - 8 * a) →
      Copied mem m₀ dst src (r - 16 * i) n.toNat →
      Exits cfg (region L) post ⟨L + BitVec.ofNat 64 (0x24a + 40 * i), regs, mem⟩ from
    fun i hi => h _ i rfl hi
  intro d
  induction d with
  | zero =>
    intro i hd hi regs mem hfr h28 hcop
    obtain rfl : i = 8 := by omega
    rw [show (0x24a + 40 * 8 : Nat) = 0x38a from rfl]
    exact hcont regs mem hfr (by rw [h28, show r - 16 * 8 - a = r - 128 - a by omega])
      (hcop.congr_lo (by omega))
  | succ d ih =>
    intro i hd hi regs mem hfr h28 hcop
    obtain ⟨hA, hB⟩ := bwd_bulk_code i (by omega)
    have h7 : regs 7 = src + BitVec.ofNat 64 (r - a) := by
      rw [hfr 7 (by decide) (by decide) (by decide) (by decide), hS7]
    have h17 : regs 17 = dst + BitVec.ofNat 64 r := by
      rw [hfr 17 (by decide) (by decide) (by decide) (by decide), hS17]
    have h13 : (regs 13).toNat % 64 = 8 * a := by
      rw [hfr 13 (by decide) (by decide) (by decide) (by decide), hS13]
    have h14 : (regs 14).toNat % 64 = 64 - 8 * a := by
      rw [hfr 14 (by decide) (by decide) (by decide) (by decide), hS14]
    refine bwd_comb env hsd m₀ a ha (0x24a + 40 * i) 28 29 (Or.inl ⟨rfl, rfl⟩) _ _ hA
      (by omega) (r - 16 * i) (by omega) (by omega) (by omega) (by omega) regs mem ?_ ?_
      h13 h14 h28 hcop ?_
    · rw [h7]; exact addr_bwd _ _ _ _ (by omega)
    · rw [h17]; exact addr_bwd _ _ _ _ (by omega)
    intro regs' mem' hfr' h29 hcop'
    have h7' : regs' 7 = src + BitVec.ofNat 64 (r - a) := by
      rw [hfr' 7 (by decide) (by decide) (by decide), h7]
    have h17' : regs' 17 = dst + BitVec.ofNat 64 r := by
      rw [hfr' 17 (by decide) (by decide) (by decide), h17]
    have h13' : (regs' 13).toNat % 64 = 8 * a := by
      rw [hfr' 13 (by decide) (by decide) (by decide), h13]
    have h14' : (regs' 14).toNat % 64 = 64 - 8 * a := by
      rw [hfr' 14 (by decide) (by decide) (by decide), h14]
    refine bwd_comb env hsd m₀ a ha (0x24a + 40 * i + 20) 29 28 (Or.inr ⟨rfl, rfl⟩) _ _ hB
      (by omega) (r - 16 * i - 8) (by omega) (by omega) (by omega) (by omega) regs' mem' ?_ ?_
      h13' h14' (by rw [h29]) hcop' ?_
    · rw [h7']; exact addr_bwd _ _ _ _ (by omega)
    · rw [h17']; exact addr_bwd _ _ _ _ (by omega)
    intro regs'' mem'' hfr'' h28'' hcop''
    rw [show 0x24a + 40 * i + 20 + 20 = 0x24a + 40 * (i + 1) by omega]
    refine ih (i + 1) (by omega) (by omega) regs'' mem'' ?_ ?_ (hcop''.congr_lo (by omega))
    · intro r h1 h2 h3 h4
      rw [hfr'' r h1 h3 h4, hfr' r h2 h3 h4, hfr r h1 h2 h3 h4]
    · rw [h28'', show r - 16 * i - 8 - 8 - a = r - 16 * (i + 1) - a by omega]

/-- The backward 128-byte loop at `0x24a`: sixteen combined words per iteration
(`bwd_run`), repeated while `a2 ≥ a + 128` (the lowest word an iteration loads
starts at source offset `r - a - 128`). -/
theorem bwd_bulk (env : Env cfg L dst src n R) (hsd : src.toNat ≤ dst.toNat)
    (regs₀ : Reg → BitVec 64) (m₀ : Mem) (he : EntryRegs regs₀ dst src n R)
    (a r₀ : Nat) (ha : 1 ≤ a ∧ a ≤ 7) (hsal : (src.toNat + r₀) % 8 = a)
    (hdal : (dst.toNat + r₀) % 8 = 0) (regs : Reg → BitVec 64) (mem : Mem)
    (hF : BAt regs₀ m₀ dst src n r₀ regs mem) (hC : BComb m₀ src a r₀ regs)
    (h5 : regs 5 = BitVec.ofNat 64 (a + 128)) (hroom : a + 128 ≤ r₀) :
    Exits cfg (region L) (Returned ⟨L, regs₀, m₀⟩ R dst src n) ⟨L + 0x24a#64, regs, mem⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hsw' := env.args.src_no_wrap
  have hdw' := env.args.dst_no_wrap
  refine Exits.loop
    (I := fun k s => s.pc = L + 0x24a#64 ∧ 128 * k ≤ r₀ ∧
      BAt regs₀ m₀ dst src n (r₀ - 128 * k) s.regs s.mem ∧
      BComb m₀ src a (r₀ - 128 * k) s.regs ∧ s.regs 5 = BitVec.ofNat 64 (a + 128) ∧
      a + 128 ≤ r₀ - 128 * k)
    (r₀ / 128) (fun k s h => by have := h.2.1; omega) ?_ 0 _
    ⟨rfl, Nat.zero_le _, by simpa using hF, by simpa using hC, h5, by simpa using hroom⟩
  intro k s ⟨hpc, hk, hF', hC', h5', hroom'⟩ hcont
  obtain ⟨pc, regs', mem'⟩ := s
  simp only at hpc hF' hC' h5' hroom'
  subst hpc
  have hrn := hF'.le
  have hrun := bwd_run env hsd m₀ a ha (r₀ - 128 * k) hroom' hrn (by omega) (by omega)
    regs' hC'.t2 hF'.a7 hC'.a3 hC'.a4 (post := Returned ⟨L, regs₀, m₀⟩ R dst src n) ?_ 0
    (by omega) regs' mem' (fun _ _ _ _ _ => rfl)
  · simp only [Nat.mul_zero, Nat.sub_zero, Nat.add_zero] at hrun
    exact hrun hC'.t3 hF'.copied
  intro regs'' mem'' hfr h28 hcop
  have h7 : regs'' 7 = src + BitVec.ofNat 64 (r₀ - 128 * k - a) := by
    rw [hfr 7 (by decide) (by decide) (by decide) (by decide), hC'.t2]
  have h11 : regs'' 11 = src + BitVec.ofNat 64 (r₀ - 128 * k) := by
    rw [hfr 11 (by decide) (by decide) (by decide) (by decide), hF'.a1]
  have h17 : regs'' 17 = dst + BitVec.ofNat 64 (r₀ - 128 * k) := by
    rw [hfr 17 (by decide) (by decide) (by decide) (by decide), hF'.a7]
  have h12 : (regs'' 12).toNat = r₀ - 128 * k := by
    rw [hfr 12 (by decide) (by decide) (by decide) (by decide), hF'.a2]
  have h5'' : regs'' 5 = BitVec.ofNat 64 (a + 128) := by
    rw [hfr 5 (by decide) (by decide) (by decide) (by decide), h5']
  rv_step ctx.code [h7, Arith.ofNat_add_neg_lit]          -- 0x38a addi t2, t2, -128
  rv_step ctx.code [h11, Arith.ofNat_add_neg_lit]         -- 0x38e addi a1, a1, -128
  rv_step ctx.code [h17, Arith.ofNat_add_neg_lit]         -- 0x392 addi a7, a7, -128
  rv_step ctx.code [Arith.add_ofNat_eq_sub_of_ge]         -- 0x396 addi a2, a2, -128
  have hF'' : BAt regs₀ m₀ dst src n (r₀ - 128 * k - 128)
      (writeReg (writeReg (writeReg (writeReg regs'' 7
        (src + BitVec.ofNat 64 (r₀ - 128 * k - a - 128))) 11
        (src + BitVec.ofNat 64 (r₀ - 128 * k - 128))) 17
        (dst + BitVec.ofNat 64 (r₀ - 128 * k - 128))) 12 (regs'' 12 - 128#64)) mem'' := by
    refine ⟨by simp [writeReg_apply], by simp [writeReg_apply], ?_, by omega, hcop, ?_⟩
    · simp only [writeReg_apply, Fin.reduceEq, ite_false, ite_true]
      rw [Arith.toNat_sub_ofNat_of_le (by omega)]
      omega
    · rv_preserved
        (pres_frame4 (by decide) (by decide) (by decide) (by decide) hfr hF'.preserved)
  have hC'' : BComb m₀ src a (r₀ - 128 * k - 128)
      (writeReg (writeReg (writeReg (writeReg regs'' 7
        (src + BitVec.ofNat 64 (r₀ - 128 * k - a - 128))) 11
        (src + BitVec.ofNat 64 (r₀ - 128 * k - 128))) 17
        (dst + BitVec.ofNat 64 (r₀ - 128 * k - 128))) 12 (regs'' 12 - 128#64)) := by
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
      exact h28
  rv_step ctx.code [h5'']                                 -- 0x39a bgeu a2, t0
  · -- Another 128 bytes.
    rename_i hc
    rw [Arith.toNat_sub_ofNat_of_le (by omega)] at hc
    simp only [BitVec.toNat_ofNat] at hc
    refine hcont _ ⟨rfl, by omega, ?_, ?_, ?_, ?_⟩
    · rw [show r₀ - 128 * (k + 1) = r₀ - 128 * k - 128 by omega]; exact hF''
    · rw [show r₀ - 128 * (k + 1) = r₀ - 128 * k - 128 by omega]; exact hC''
    · simp only [writeReg_apply, Fin.reduceEq, ite_false]; exact h5''
    · omega
  · -- On to the word loop.
    exact bwd_words env hsd regs₀ m₀ he a (r₀ - 128 * k - 128) ha (by omega) (by omega)
      (by omega) _ _ hF'' hC''

/-- The setup at `0x230`, with `a7 = dst + r` aligned and at least `8 - a`
bytes copied (so the aligned word at `src + r - a` lies inside the source). -/
theorem bwd_setup (env : Env cfg L dst src n R) (hsd : src.toNat ≤ dst.toNat)
    (regs₀ : Reg → BitVec 64) (m₀ : Mem) (he : EntryRegs regs₀ dst src n R)
    (r : Nat) (hr : n.toNat ≤ r + 15) (hdal : (dst.toNat + r) % 8 = 0)
    (hroom : 8 - (src.toNat + r) % 8 ≤ n.toNat - r) (regs : Reg → BitVec 64) (mem : Mem)
    (hF : BAt regs₀ m₀ dst src n r regs mem) :
    Exits cfg (region L) (Returned ⟨L, regs₀, m₀⟩ R dst src n) ⟨L + 0x230#64, regs, mem⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hsw' := env.args.src_no_wrap
  have hsw : src.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hsw'
  have hdw' := env.args.dst_no_wrap
  have hmis := env.mis; have hbig := env.big
  have hl := env.sb.load
  have ha1 := hF.a1; have hrn := hF.le
  obtain ⟨a, had⟩ : ∃ a, a = (src.toNat + r) % 8 := ⟨_, rfl⟩
  have ha : 1 ≤ a ∧ a ≤ 7 := by omega
  have hW := and_neg8_add (x := src) (J := r) (a := a) (by omega) (by omega) (by omega)
  have hA := and_seven_add (x := src) (J := r) (a := a) (by omega) (by omega)
  have hA3 := ofNat_shl3 (a := a) (by omega)
  have ha128 : BitVec.ofNat 64 a + 128#64 = BitVec.ofNat 64 (a + 128) :=
    (BitVec.ofNat_add _ _).symm
  rv_step ctx.code [ha1, hW]                              -- 0x230 andi t2, a1, -8
  rv_step ctx.code [ha1, hA]                              -- 0x234 andi a5, a1, 7
  rv_step ctx.code [hA3]                                  -- 0x238 slli a3, a5, 3
  rv_step ctx.code                                        -- 0x23c li a4, 64
  rv_step ctx.code [Arith.ofNat_sub_ofNat_of_le]          -- 0x240 sub a4, a4, a3
  rv_step ctx.code [hl]                                   -- 0x242 ld t3, 0(t2)
  rv_step ctx.code [ha128]                                -- 0x246 addi t0, a5, 128
  refine bwd_bulk env hsd regs₀ m₀ he a r ha (by omega) hdal _ mem ?_ ?_ ?_ (by omega)
  · exact BAt.of_eq (a := r) (b := r) (by simp [writeReg_apply, ha1])
      (by simp [writeReg_apply, hF.a7]) (by simp [writeReg_apply, hF.a2]) hF.copied rfl rfl
      hrn (by rv_preserved hF.preserved)
  · refine ⟨by simp [writeReg_apply], ?_, ?_, by simp [writeReg_apply], ?_⟩
    · simp only [writeReg_apply, Fin.reduceEq, ite_false, ite_true]
      exact toNat_ofNat_mod64 (by omega)
    · simp only [writeReg_apply, Fin.reduceEq, ite_false, ite_true]
      exact toNat_ofNat_mod64 (by omega)
    · simp only [writeReg_apply, Fin.reduceEq, ite_false, ite_true]
      apply loadLE_shl_congr (by omega)
      intro i hi
      rw [Arith.add_lit_lit]
      exact hF.copied.read_bwd hsd (by omega) hdw'
  · simp [writeReg_apply]

/-- `.Lmis_backward_first` at `0x21c`: the aligned source word at or above
`src + r` would end past `src + n`, so copy eight more bytes downwards. -/
theorem bwd_first (env : Env cfg L dst src n R) (hsd : src.toNat ≤ dst.toNat)
    (regs₀ : Reg → BitVec 64) (m₀ : Mem) (he : EntryRegs regs₀ dst src n R)
    (r₀ : Nat) (hr : n.toNat ≤ r₀ + 7) (hdal : (dst.toNat + r₀) % 8 = 0)
    (regs : Reg → BitVec 64) (mem : Mem) (hF : BAt regs₀ m₀ dst src n r₀ regs mem)
    (h5 : (regs 5).toNat = 8) :
    Exits cfg (region L) (Returned ⟨L, regs₀, m₀⟩ R dst src n) ⟨L + 0x21c#64, regs, mem⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hsw' := env.args.src_no_wrap
  have hsw : src.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hsw'
  have hdw' := env.args.dst_no_wrap
  have hdw : dst.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hdw'
  have hbig := env.big
  have hl := env.sb.load; have hs := env.sb.store
  have hrn := hF.le
  refine Exits.loop
    (I := fun k s => s.pc = L + 0x21c#64 ∧ k < 8 ∧
      BAt regs₀ m₀ dst src n (r₀ - k) s.regs s.mem ∧ (s.regs 5).toNat = 8 - k)
    7 (fun k s h => by have := h.2.1; omega) ?_ 0 _ ⟨rfl, by omega, by simpa using hF, h5⟩
  intro k s ⟨hpc, hk, hF', h5'⟩ hcont
  obtain ⟨pc, regs', mem'⟩ := s
  simp only at hpc hF' h5'
  subst hpc
  have ha1 := hF'.a1; have ha7 := hF'.a7; have ha2 := hF'.a2
  rv_step ctx.code [ha1, hl, Arith.ofNat_add_neg_lit]     -- 0x21c lbu t1, -1(a1)
  rv_step ctx.code [ha7, hs, Arith.ofNat_add_neg_lit]     -- 0x220 sb t1, -1(a7)
  rv_step ctx.code [ha1, Arith.ofNat_add_neg_lit]         -- 0x224 addi a1, a1, -1
  rv_step ctx.code [ha7, Arith.ofNat_add_neg_lit]         -- 0x226 addi a7, a7, -1
  rv_step ctx.code [Arith.add_ofNat_eq_sub_of_ge]         -- 0x228 addi a2, a2, -1
  rv_step ctx.code [Arith.add_ofNat_eq_sub_of_ge]         -- 0x22a addi t0, t0, -1
  have h5'' : (regs' 5 - 1#64).toNat = 7 - k := by
    rw [Arith.toNat_sub_ofNat_of_le (by omega)]; omega
  have hF'' : BAt regs₀ m₀ dst src n (r₀ - k - 1)
      (writeReg (writeReg (writeReg (writeReg (writeReg regs' 6
        (loadLE mem' (src + BitVec.ofNat 64 (r₀ - k - 1)) 1)) 11
        (src + BitVec.ofNat 64 (r₀ - k - 1))) 17 (dst + BitVec.ofNat 64 (r₀ - k - 1))) 12
        (regs' 12 - 1#64)) 5 (regs' 5 - 1#64))
      (storeLE mem' (dst + BitVec.ofNat 64 (r₀ - k - 1))
        (loadLE mem' (src + BitVec.ofNat 64 (r₀ - k - 1)) 1) 1) := by
    refine ⟨by simp [writeReg_apply], by simp [writeReg_apply], ?_, by omega,
      Copied.store_bwd_self (hF'.copied.congr_lo (by omega)) hsd (by decide) hdw' (by omega)
        (by omega),
      by rv_preserved hF'.preserved⟩
    simp only [writeReg_apply, Fin.reduceEq, ite_false, ite_true]
    rw [Arith.toNat_sub_ofNat_of_le (by omega)]
    omega
  rv_step ctx.code                                        -- 0x22c bnez t0
  · -- More bytes.
    rename_i hc
    rw [bv_eq_zero_iff h5''] at hc
    refine hcont _ ⟨rfl, by omega, ?_, ?_⟩
    · rw [show r₀ - (k + 1) = r₀ - k - 1 by omega]; exact hF''
    · simp only [writeReg_apply, Fin.reduceEq, ite_false, ite_true]; omega
  · rename_i hc
    rw [bv_eq_zero_iff h5''] at hc
    exact bwd_setup env hsd regs₀ m₀ he (r₀ - k - 1) (by omega) (by omega) (by omega) _ _ hF''

/-- `.Lmis_backward_bound` at `0x20e`, with `a7 = dst + r` aligned: if the
aligned source word at or above `src + r` would end past `src + n`, copy eight
more bytes first. -/
theorem bwd_bound (env : Env cfg L dst src n R) (hlt : src.toNat < dst.toNat)
    (regs₀ : Reg → BitVec 64) (m₀ : Mem) (he : EntryRegs regs₀ dst src n R)
    (r : Nat) (hr : n.toNat ≤ r + 7) (hdal : (dst.toNat + r) % 8 = 0)
    (regs : Reg → BitVec 64) (mem : Mem) (hF : BAt regs₀ m₀ dst src n r regs mem)
    (h16 : regs 16 = src + BitVec.ofNat 64 n.toNat) :
    Exits cfg (region L) (Returned ⟨L, regs₀, m₀⟩ R dst src n) ⟨L + 0x20e#64, regs, mem⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hsw' := env.args.src_no_wrap
  have hdw' := env.args.dst_no_wrap
  have hbig := env.big; have hmis := env.mis
  have ha1 := hF.a1; have hrn := hF.le
  rv_step ctx.code [ha1]                                  -- 0x20e addi t2, a1, 7
  rv_step ctx.code                                        -- 0x212 andi t2, t2, -8
  rv_step ctx.code [h16, Arith.toNat_and_neg8]            -- 0x216 bgeu a6, t2
  · rename_i hc
    rv_arith_simp at hc
    refine bwd_setup env (Nat.le_of_lt hlt) regs₀ m₀ he r (by omega) hdal (by omega) _ mem ?_
    exact BAt.of_eq (a := r) (b := r) (by simp [writeReg_apply, ha1])
      (by simp [writeReg_apply, hF.a7]) (by simp [writeReg_apply, hF.a2]) hF.copied rfl rfl
      hrn (by rv_preserved hF.preserved)
  · rv_step ctx.code                                      -- 0x21a li t0, 8
    refine bwd_first env (Nat.le_of_lt hlt) regs₀ m₀ he r hr hdal _ mem ?_
      (by simp [writeReg_apply])
    exact BAt.of_eq (a := r) (b := r) (by simp [writeReg_apply, ha1])
      (by simp [writeReg_apply, hF.a7]) (by simp [writeReg_apply, hF.a2]) hF.copied rfl rfl
      hrn (by rv_preserved hF.preserved)

/-- `.Lmis_backward_peel` at `0x1f6`: copy single bytes downwards until `a7` is
8-byte aligned. -/
theorem bwd_peel (env : Env cfg L dst src n R) (hlt : src.toNat < dst.toNat)
    (regs₀ : Reg → BitVec 64) (m₀ : Mem) (he : EntryRegs regs₀ dst src n R)
    (regs : Reg → BitVec 64) (mem : Mem) (hF : BAt regs₀ m₀ dst src n n.toNat regs mem)
    (h16 : regs 16 = src + BitVec.ofNat 64 n.toNat) :
    Exits cfg (region L) (Returned ⟨L, regs₀, m₀⟩ R dst src n) ⟨L + 0x1f6#64, regs, mem⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hsw' := env.args.src_no_wrap
  have hsw : src.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hsw'
  have hdw' := env.args.dst_no_wrap
  have hdw : dst.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hdw'
  have hbig := env.big
  have hsd := Nat.le_of_lt hlt
  have hl := env.sb.load; have hs := env.sb.store
  refine Exits.loop
    (I := fun k s => s.pc = L + 0x1f6#64 ∧ k ≤ (dst.toNat + n.toNat) % 8 ∧
      BAt regs₀ m₀ dst src n (n.toNat - k) s.regs s.mem ∧
      s.regs 16 = src + BitVec.ofNat 64 n.toNat)
    7 (fun k s h => by have := h.2.1; omega) ?_ 0 _ ⟨rfl, by omega, by simpa using hF, h16⟩
  intro k s ⟨hpc, hk, hF', h16'⟩ hcont
  obtain ⟨pc, regs', mem'⟩ := s
  simp only at hpc hF' h16'
  subst hpc
  have ha1 := hF'.a1; have ha7 := hF'.a7; have ha2 := hF'.a2; have hrn := hF'.le
  rv_step ctx.code [ha7]                                  -- 0x1f6 andi t0, a7, 7
  rv_step ctx.code [Arith.and_seven_eq_zero_iff]          -- 0x1fa beqz t0
  · -- Aligned.
    rename_i hc
    rv_arith_simp at hc
    refine bwd_bound env hlt regs₀ m₀ he (n.toNat - k) (by omega) (by omega) _ mem' ?_
      (by simp [writeReg_apply, h16'])
    exact BAt.of_eq (a := n.toNat - k) (b := n.toNat - k) (by simp [writeReg_apply, ha1])
      (by simp [writeReg_apply, ha7]) (by simp [writeReg_apply, ha2]) hF'.copied rfl rfl
      hrn (by rv_preserved hF'.preserved)
  · rename_i hc
    rv_arith_simp at hc
    rv_step ctx.code [ha1, hl, Arith.ofNat_add_neg_lit]   -- 0x1fe lbu t1, -1(a1)
    rv_step ctx.code [ha7, hs, Arith.ofNat_add_neg_lit]   -- 0x202 sb t1, -1(a7)
    rv_step ctx.code [ha1, Arith.ofNat_add_neg_lit]       -- 0x206 addi a1, a1, -1
    rv_step ctx.code [ha7, Arith.ofNat_add_neg_lit]       -- 0x208 addi a7, a7, -1
    rv_step ctx.code [Arith.add_ofNat_eq_sub_of_ge]       -- 0x20a addi a2, a2, -1
    rv_step ctx.code                                      -- 0x20c j 0x1f6
    refine hcont _ ⟨rfl, by omega, ?_, by simp [writeReg_apply, h16']⟩
    refine BAt.of_eq (a := n.toNat - k - 1) (b := n.toNat - k - 1) ?_ ?_ ?_
      (Copied.store_bwd_self (hF'.copied.congr_lo (by omega)) hsd (by decide) hdw' (by omega)
        (by omega)) (by omega) (by omega)
      (by omega) ?_
    · simp [writeReg_apply]
    · simp [writeReg_apply]
    · simp only [writeReg_apply, Fin.reduceEq, ite_false, ite_true]
      rw [Arith.toNat_sub_ofNat_of_le (by omega)]
      omega
    · rv_preserved hF'.preserved

/-- The backward copy from `0x1f0` (`add a1, a1, a2`), reached for `src < dst`
with `a7 = dst`. -/
theorem bwd_path (env : Env cfg L dst src n R) (hlt : src.toNat < dst.toNat)
    (regs₀ : Reg → BitVec 64) (m₀ : Mem) (he : EntryRegs regs₀ dst src n R)
    (regs : Reg → BitVec 64) (h11 : regs 11 = src) (h17 : regs 17 = dst) (h12 : regs 12 = n)
    (hpres : ∀ r, r ∉ clobbered → regs r = regs₀ r) :
    Exits cfg (region L) (Returned ⟨L, regs₀, m₀⟩ R dst src n) ⟨L + 0x1f0#64, regs, m₀⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hsn : src + n = src + BitVec.ofNat 64 n.toNat := by
    rw [BitVec.ofNat_toNat, BitVec.setWidth_eq]
  have hdn : dst + n = dst + BitVec.ofNat 64 n.toNat := by
    rw [BitVec.ofNat_toNat, BitVec.setWidth_eq]
  rv_step ctx.code [h11, h12, hsn]                        -- 0x1f0 add a1, a1, a2
  rv_step ctx.code [h17, h12, hdn]                        -- 0x1f2 add a7, a7, a2
  rv_step ctx.code [BitVec.zero_add]                      -- 0x1f4 mv a6, a1
  refine bwd_peel env hlt regs₀ m₀ he _ m₀ ?_ (by simp [writeReg_apply])
  exact ⟨by simp [writeReg_apply], by simp [writeReg_apply],
    by simp [writeReg_apply, h12], Nat.le_refl _, Copied.empty _ _ _ _,
    by rv_preserved hpres⟩

end CopyRuntime.Leaf
