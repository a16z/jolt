import CopyRuntime.Proof.Wrapper.Exits

/-!
# The backward general path (`src < dst`, equal offsets modulo 8, `n ≥ 8`)

The mirror image of `Forward.lean`. Each lemma takes a state at a code offset
described by `BwdAt h` (the destination offsets `[h, n)` are copied, `a1`/`a7`
point `h` bytes into the ranges) and shows that the run ends in the final
`Returned` outcome.

Backward copying is safe for overlap because every load reads a source byte
below the copied part: a store at destination offset `i` hits source offset
`i + (dst - src) > i`, and the stores so far are at offsets `≥ h`, while the
loads are at offsets `< h` (`Copied.read_bwd`).
-/

namespace CopyRuntime

/-- One store of a block that the 80-byte loop stores in ascending order below
the suffix copy `base`, with the value loaded from `base`. -/
theorem CopiedOver.store_up_to {m base m₀ : Mem} {dst src : BitVec 64} {lo k w hi lo' hi' : Nat}
    (h : CopiedOver m base m₀ dst src lo k) (h' : Copied base m₀ dst src lo' hi')
    (hlo' : k + w ≤ lo') (hsd : src.toNat ≤ dst.toNat) (hw : w ≤ 8)
    (hdst : dst.toNat + hi' < 2 ^ 64) (hk : k + w < 2 ^ 64) (hlo : lo ≤ k) (e : k + w = hi) :
    CopiedOver (storeLE m (dst + BitVec.ofNat 64 k) (loadLE base (src + BitVec.ofNat 64 k) w) w)
      base m₀ dst src lo hi := by
  rw [h'.loadLE_bwd hsd hlo' hdst, ← e]
  exact h.extend_up hw hk hlo

end CopyRuntime

namespace CopyRuntime.Wrapper

/-- Prove `Copied m m₀ dst src lo hi` for a chain of backward stores whose
values were loaded from the memory being stored to; the innermost memory must
have a `Copied` hypothesis in context. Matching is at reducible transparency
(see `copied_fwd`). -/
macro "copied_bwd" : tactic => `(tactic| (
  repeat (first
    | exact Copied.congr_lo (by with_reducible assumption) (by omega)
    | with_reducible
        (refine Copied.store_bwd_self ?_ (by omega) (by decide) (by omega) (by omega) (by omega))
    | with_reducible
        (refine Copied.store_bwd_zero ?_ (by omega) (by decide) (by omega) (by omega) (by omega)))))

/-- Prove `Copied m m₀ dst src lo hi` for a block of stores in ascending
address order on top of a memory `base` with hypothesis `hcop`, each storing a
value loaded from `base` below the copied part. -/
macro "copied_block_bwd " hcop:term : tactic => `(tactic| (
  refine CopiedOver.join ?_ $hcop (by omega) (by omega)
  repeat (first
    | with_reducible (exact CopiedOver.empty_of_eq (by omega))
    | with_reducible
        (refine CopiedOver.store_up_to ?_ $hcop (by omega) (by omega) (by decide) (by omega)
          (by omega) (by omega) (by omega)))))

/-- Everything fixed during a backward copy. -/
structure BwdEnv (cfg : Config) (B L F dst src n R : BitVec 64) : Prop where
  ctx : Ctx cfg B L F
  sb : Sandbox cfg dst src n
  args : ArgsOk dst src n R
  rout : ¬ region B R
  lt : src.toNat < dst.toNat
  same : dst.toNat % 8 = src.toNat % 8

/-- The cursor state of a backward copy with `h` bytes left: the destination
offsets `[h, n)` are copied. -/
structure BwdAt (regs₀ : Reg → BitVec 64) (m₀ : Mem) (dst src n : BitVec 64) (h : Nat)
    (regs : Reg → BitVec 64) (mem : Mem) : Prop where
  a1 : regs 11 = src + BitVec.ofNat 64 h
  a7 : regs 17 = dst + BitVec.ofNat 64 h
  le : h ≤ n.toNat
  copied : Copied mem m₀ dst src h n.toNat
  preserved : ∀ r, r ∉ clobbered → regs r = regs₀ r

/-- Build a `BwdAt` fact whose offsets are only equal (not syntactically) to `h`. -/
theorem BwdAt.of_eq {regs₀ regs : Reg → BitVec 64} {m₀ mem : Mem} {dst src n : BitVec 64}
    {h a b : Nat} (h1 : regs 11 = src + BitVec.ofNat 64 a)
    (h7 : regs 17 = dst + BitVec.ofNat 64 b) (hcop : Copied mem m₀ dst src h n.toNat)
    (ea : a = h) (eb : b = h) (hle : h ≤ n.toNat)
    (hpres : ∀ r, r ∉ clobbered → regs r = regs₀ r) : BwdAt regs₀ m₀ dst src n h regs mem := by
  subst ea eb
  exact ⟨h1, h7, hle, hcop, hpres⟩

variable {cfg : Config} {B L F dst src n R : BitVec 64}

set_option maxHeartbeats 300000 in
/-- The 4/2/1-byte tail at `0x4a6`, with `h < 8` bytes left and `a1` 8-byte
aligned. -/
theorem bwd_tail (env : BwdEnv cfg B L F dst src n R) (regs₀ : Reg → BitVec 64) (m₀ : Mem)
    (he : EntryRegs regs₀ dst src n R) (h : Nat) (regs : Reg → BitVec 64) (mem : Mem)
    (hA : BwdAt regs₀ m₀ dst src n h regs mem) (ha2 : (regs 12).toNat = h) (hh : h < 8)
    (hal : (src.toNat + h) % 8 = 0) :
    Exits cfg (region B) (Returned ⟨B + 0x2c0#64, regs₀, m₀⟩ R dst src n)
      ⟨B + 0x4a6#64, regs, mem⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hR := env.args.ret_even
  have hsw' := env.args.src_no_wrap
  have hdw' := env.args.dst_no_wrap
  have hsw : src.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hsw'
  have hdw : dst.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hdw'
  have hlt := env.lt
  have hds := env.same
  have hl := env.sb.load; have hst := env.sb.store
  have ha1 := hA.a1; have ha7 := hA.a7; have hhn := hA.le; have hcop := hA.copied
  have hra : regs 1 = R := by rw [hA.preserved 1 (by decide), he.ra]
  have hcases : h = 0 ∨ h = 1 ∨ h = 2 ∨ h = 3 ∨ h = 4 ∨ h = 5 ∨ h = 6 ∨ h = 7 := by omega
  rcases hcases with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
  · repeat rv_step ctx.code [ha1, ha7, hra, hl, hst]
    refine .exit env.rout (Returned.of_copied he env.args.dst_no_wrap ?_ ?_)
    · copied_bwd
    · rv_preserved hA.preserved

/-- The word loop: `0x488` computes the end pointer `t0 = a1 - (a2 & ~7)` and
keeps `a2 & 7`; `0x496` copies one word per iteration while `t0 < a1`. -/
theorem bwd_words (env : BwdEnv cfg B L F dst src n R) (regs₀ : Reg → BitVec 64) (m₀ : Mem)
    (he : EntryRegs regs₀ dst src n R) (h : Nat) (regs : Reg → BitVec 64) (mem : Mem)
    (hA : BwdAt regs₀ m₀ dst src n h regs mem) (ha2 : (regs 12).toNat = h)
    (hal : (src.toNat + h) % 8 = 0) :
    Exits cfg (region B) (Returned ⟨B + 0x2c0#64, regs₀, m₀⟩ R dst src n)
      ⟨B + 0x488#64, regs, mem⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hsw' := env.args.src_no_wrap
  have hdw' := env.args.dst_no_wrap
  have hsw : src.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hsw'
  have hdw : dst.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hdw'
  have hlt := env.lt
  have hds := env.same
  have hl := env.sb.load; have hst := env.sb.store
  have ha1 := hA.a1; have ha7 := hA.a7; have hhn := hA.le
  rv_step ctx.code [ha1, ha7]                                -- 0x488 andi t0, a2, -8
  generalize hT : regs 12 &&& 0xfffffffffffffff8#64 = T
  have hTn : T.toNat = h / 8 * 8 := by rw [← hT, Arith.toNat_and_neg8, ha2]
  rv_step ctx.code [ha1, ha7]                                -- 0x48c sub t0, a1, t0
  rv_step ctx.code [ha1, ha7]                                -- 0x490 c.andi a2, 7
  generalize hAA : regs 12 &&& 7#64 = A
  have hAn : A.toNat = h % 8 := by rw [← hAA, Arith.toNat_and_seven, ha2]
  have hend : src + BitVec.ofNat 64 h - T = src + BitVec.ofNat 64 (h % 8) := by
    apply BitVec.eq_of_toNat_eq
    simp only [BitVec.toNat_sub, BitVec.toNat_add, BitVec.toNat_ofNat]
    omega
  rw [hend]
  rv_step ctx.code [ha1, ha7]                                -- 0x492 beq a1, t0
  · -- No whole words left: straight to the tail.
    rename_i heq
    have h8 : h < 8 := by
      have := congrArg BitVec.toNat heq
      simp only [BitVec.toNat_add, BitVec.toNat_ofNat] at this
      omega
    refine bwd_tail env regs₀ m₀ he h _ mem ⟨?_, ?_, hhn, hA.copied, ?_⟩ ?_ h8 hal
    · simp [writeReg_apply, ha1]
    · simp [writeReg_apply, ha7]
    · rv_preserved hA.preserved
    · simp [writeReg_apply, hAn]; omega
  · rename_i hne
    have hq : 8 ≤ h := by
      apply Nat.le_of_not_lt
      intro h8
      apply hne
      rw [Nat.mod_eq_of_lt h8]
    obtain ⟨q, hqdef⟩ : ∃ q, q = h / 8 := ⟨_, rfl⟩
    refine Exits.loop
      (I := fun k s => s.pc = B + 0x496#64 ∧ BwdAt regs₀ m₀ dst src n (h - 8 * k) s.regs s.mem ∧
        s.regs 5 = src + BitVec.ofNat 64 (h % 8) ∧ s.regs 12 = A ∧ k < q)
      q (fun k s h => Nat.le_of_lt h.2.2.2.2) ?_ 0 _ ⟨rfl, ?_, ?_, ?_, ?_⟩
    · intro k s ⟨hpc, hA', h5, h12, hkq⟩ hcont
      obtain ⟨pc, regs', mem'⟩ := s
      simp only at hpc hA' h5 h12
      subst hpc
      have ha1' := hA'.a1; have ha7' := hA'.a7; have hcop' := hA'.copied
      have hk8 : 8 ≤ h - 8 * k := by omega
      rv_step ctx.code [ha1', ha7', h5, h12, hl, Arith.ofNat_add_neg_lit]  -- 0x496 ld t1, -8(a1)
      rv_step ctx.code [ha1', ha7', h5, h12, hst, Arith.ofNat_add_neg_lit]  -- 0x49a sd t1, -8(a7)
      rv_step ctx.code [ha1', ha7', h5, h12, Arith.ofNat_add_neg_lit]  -- 0x49e c.addi a1, -8
      rv_step ctx.code [ha1', ha7', h5, h12, Arith.ofNat_add_neg_lit]  -- 0x4a0 c.addi a7, -8
      rv_step ctx.code [ha1', ha7', h5, h12]                 -- 0x4a2 bltu t0, a1
      · -- Another word.
        refine hcont _ ⟨rfl, ?_, ?_, ?_, ?_⟩
        · refine BwdAt.of_eq (a := h - 8 * k - 8) (b := h - 8 * k - 8) ?_ ?_ ?_
            (by omega) (by omega) (by omega) ?_
          · simp [writeReg_apply]
          · simp [writeReg_apply]
          · copied_bwd
          · rv_preserved hA'.preserved
        · simp [writeReg_apply, h5]
        · simp [writeReg_apply, h12]
        · rename_i hlt'
          simp only [BitVec.toNat_add, BitVec.toNat_ofNat] at hlt'
          omega
      · -- Last word: on to the tail.
        rename_i hge
        refine bwd_tail env regs₀ m₀ he (h % 8) _ _ ?_ ?_ (by omega) (by omega)
        · have hk1 : h - 8 * k - 8 = h % 8 := by
            simp only [BitVec.toNat_add, BitVec.toNat_ofNat] at hge; omega
          refine BwdAt.of_eq (a := h - 8 * k - 8) (b := h - 8 * k - 8) ?_ ?_ ?_
            hk1 hk1 (by omega) ?_
          · simp [writeReg_apply]
          · simp [writeReg_apply]
          · copied_bwd
          · rv_preserved hA'.preserved
        · simp [writeReg_apply, h12, hAn]
    · refine BwdAt.of_eq (a := h) (b := h) ?_ ?_ hA.copied (by omega) (by omega) hhn ?_
      · simp [writeReg_apply, ha1]
      · simp [writeReg_apply, ha7]
      · rv_preserved hA.preserved
    · simp [writeReg_apply]
    · simp [writeReg_apply]
    · omega

set_option maxHeartbeats 600000 in
/-- The 80-byte block loop at `0x420`/`0x428`: ten loads, then ten stores. -/
theorem bwd_blocks (env : BwdEnv cfg B L F dst src n R) (regs₀ : Reg → BitVec 64) (m₀ : Mem)
    (he : EntryRegs regs₀ dst src n R) (h : Nat) (regs : Reg → BitVec 64) (mem : Mem)
    (hA : BwdAt regs₀ m₀ dst src n h regs mem) (ha2 : (regs 12).toNat = h)
    (hal : (src.toNat + h) % 8 = 0) :
    Exits cfg (region B) (Returned ⟨B + 0x2c0#64, regs₀, m₀⟩ R dst src n)
      ⟨B + 0x420#64, regs, mem⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hsw' := env.args.src_no_wrap
  have hdw' := env.args.dst_no_wrap
  have hsw : src.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hsw'
  have hdw : dst.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hdw'
  have hlt := env.lt
  have hds := env.same
  have hl := env.sb.load; have hst := env.sb.store
  have ha1 := hA.a1; have ha7 := hA.a7; have hhn := hA.le
  rv_step ctx.code [ha1, ha7]                                -- 0x420 li t0, 80
  rv_step ctx.code [ha1, ha7]                                -- 0x424 bltu a2, t0
  · exact bwd_words env regs₀ m₀ he h _ mem
      ⟨by simp [writeReg_apply, ha1], by simp [writeReg_apply, ha7], hhn, hA.copied,
        by rv_preserved hA.preserved⟩
      (by simp [writeReg_apply, ha2]) hal
  · rename_i hge
    refine Exits.loop
      (I := fun k s => s.pc = B + 0x428#64 ∧ BwdAt regs₀ m₀ dst src n (h - 80 * k) s.regs s.mem ∧
        s.regs 12 = BitVec.ofNat 64 (h - 80 * k) ∧ 80 ≤ h - 80 * k ∧ s.regs 5 = 80#64)
      (h / 80) (fun k s h => by have := h.2.2.2.1; omega) ?_ 0 _
      ⟨rfl, ?_, ?_, ?_, ?_⟩
    · intro k s ⟨hpc, hA', ha2', h80, h5⟩ hcont
      obtain ⟨pc, regs', mem'⟩ := s
      simp only at hpc hA' ha2' h80 h5
      subst hpc
      have ha1' := hA'.a1; have ha7' := hA'.a7; have hcop' := hA'.copied; have hkn := hA'.le
      have hal' : (src.toNat + (h - 80 * k)) % 8 = 0 := by omega
      rw [Nat.mul_add_one, ← Nat.sub_sub] at hcont
      iterate 23 rv_step ctx.code [ha1', ha7', h5, ha2', hl, hst, BitVec.ofNat_add_ofNat,
        Arith.ofNat_add_neg_lit]                             -- 0x428 .. 0x480
      rv_step ctx.code [ha1', ha7', h5]                      -- 0x484 bgeu a2, t0
      · rename_i hc
        rw [Arith.toNat_ofNat_of_lt (by omega)] at hc
        refine hcont _ ⟨rfl, ?_, ?_, ?_, ?_⟩
        · refine BwdAt.of_eq (a := h - 80 * k - 80) (b := h - 80 * k - 80) ?_ ?_ ?_
            rfl rfl (by omega) ?_
          · simp [writeReg_apply]
          · simp [writeReg_apply]
          · copied_block_bwd hcop'
          · rv_preserved hA'.preserved
        · simp [writeReg_apply]
        · exact hc
        · simp [writeReg_apply, h5]
      · rename_i hc
        rw [Arith.toNat_ofNat_of_lt (by omega)] at hc
        refine bwd_words env regs₀ m₀ he (h - 80 * k - 80) _ _ ?_ ?_ (by omega)
        · refine BwdAt.of_eq (a := h - 80 * k - 80) (b := h - 80 * k - 80) ?_ ?_ ?_
            rfl rfl (by omega) ?_
          · simp [writeReg_apply]
          · simp [writeReg_apply]
          · copied_block_bwd hcop'
          · rv_preserved hA'.preserved
        · simp only [writeReg_apply, Fin.reduceEq, ite_false, ite_true]
          exact Arith.toNat_ofNat_of_lt (by omega)
    · refine BwdAt.of_eq (a := h) (b := h) ?_ ?_ hA.copied (by omega) (by omega) hhn ?_
      · simp [writeReg_apply, ha1]
      · simp [writeReg_apply, ha7]
      · rv_preserved hA.preserved
    · simp only [writeReg_apply, Fin.reduceEq, ite_false]
      exact Arith.eq_ofNat_of_toNat_eq ha2
    · omega
    · simp [writeReg_apply]

/-- A run of `m` pairs `ld t1, -8(i+1)(a1); sd t1, -8(i+1)(a7)` (`i = 0 … m-1`)
at code offset `o`, with `a1 = src + h` and `a7 = dst + h` fixed: it extends a
backward suffix copy from `h - 8i` down to `h - 8m` and changes no register
but `t1`. Proved once by induction with the memory kept abstract (as
`fwd_pairs`); `hcode` is checked against the LLVM listing when it is used. -/
theorem bwd_pairs (env : BwdEnv cfg B L F dst src n R) (m₀ : Mem) {post : State → Prop}
    (o m h : Nat)
    (hcode : ∀ i < m,
      (o + 8 * i, Instr.load .double 6 11 (imm (-((8 * (i + 1) : Nat) : Int))), 4) ∈
        Generated.Disasm.wrapMemcpy ∧
      (o + 8 * i + 4, Instr.store .double 6 17 (imm (-((8 * (i + 1) : Nat) : Int))), 4) ∈
        Generated.Disasm.wrapMemcpy)
    (hend : o + 8 * m ≤ 0x578) (hfit : 8 * m ≤ h) (hhn : h ≤ n.toNat)
    (hal : (src.toNat + h) % 8 = 0)
    (regsS : Reg → BitVec 64) (hS11 : regsS 11 = src + BitVec.ofNat 64 h)
    (hS17 : regsS 17 = dst + BitVec.ofNat 64 h)
    (hcont : ∀ regs mem, (∀ r, r ≠ 6 → regs r = regsS r) →
      Copied mem m₀ dst src (h - 8 * m) n.toNat →
      Exits cfg (region B) post ⟨B + BitVec.ofNat 64 (o + 8 * m), regs, mem⟩) :
    ∀ i, i ≤ m → ∀ regs mem, (∀ r, r ≠ 6 → regs r = regsS r) →
      Copied mem m₀ dst src (h - 8 * i) n.toNat →
      Exits cfg (region B) post ⟨B + BitVec.ofNat 64 (o + 8 * i), regs, mem⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hsw' := env.args.src_no_wrap
  have hdw' := env.args.dst_no_wrap
  have hsw : src.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hsw'
  have hdw : dst.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hdw'
  have hlt := env.lt
  have hds := env.same
  have hl := env.sb.load; have hst := env.sb.store
  suffices hgen : ∀ d i, m - i = d → i ≤ m → ∀ regs mem, (∀ r, r ≠ 6 → regs r = regsS r) →
      Copied mem m₀ dst src (h - 8 * i) n.toNat →
      Exits cfg (region B) post ⟨B + BitVec.ofNat 64 (o + 8 * i), regs, mem⟩ from
    fun i hi => hgen _ i rfl hi
  intro d
  induction d with
  | zero =>
    intro i hd hi regs mem hregs hcop
    have : i = m := by omega
    subst this
    exact hcont regs mem hregs hcop
  | succ d ih =>
    intro i hd hi regs mem hregs hcop
    have him : i < m := by omega
    have hi8 : 8 * (i + 1) ≤ h := by omega
    have h11 : regs 11 = src + BitVec.ofNat 64 h := by rw [hregs 11 (by decide), hS11]
    have h17 : regs 17 = dst + BitVec.ofNat 64 h := by rw [hregs 17 (by decide), hS17]
    refine ctx.code.step (o + 8 * i) ⟨(hcode i him).1, by simp; omega⟩ rfl
      (codeRegion_lit (by omega) (by decide)) ?_
    rv_simp [h11, hl, Arith.add_ofNat_ofInt_neg]
    refine ctx.code.step (o + 8 * i + 4) ⟨(hcode i him).2, by simp; omega⟩ rfl
      (codeRegion_lit (by omega) (by decide)) ?_
    rv_simp [h17, hst, Arith.add_ofNat_ofInt_neg]
    rw [show o + 8 * i + 4 + 4 = o + 8 * (i + 1) by omega]
    refine ih (i + 1) (by omega) (by omega) _ _ ?_ ?_
    · intro r hr
      simp only [writeReg_apply, hr, ite_false]
      exact hregs r hr
    · exact Copied.store_bwd_self (hcop.congr_lo (by omega)) (Nat.le_of_lt hlt) (by decide)
        (by omega) (by omega) (by omega)

/-- The 256-byte loop at `0x308`/`0x310`: thirty-two `ld`/`sd` pairs. -/
theorem bwd_large (env : BwdEnv cfg B L F dst src n R) (regs₀ : Reg → BitVec 64) (m₀ : Mem)
    (he : EntryRegs regs₀ dst src n R) (h : Nat) (regs : Reg → BitVec 64) (mem : Mem)
    (hA : BwdAt regs₀ m₀ dst src n h regs mem) (ha2 : (regs 12).toNat = h)
    (hal : (src.toNat + h) % 8 = 0) :
    Exits cfg (region B) (Returned ⟨B + 0x2c0#64, regs₀, m₀⟩ R dst src n)
      ⟨B + 0x308#64, regs, mem⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hsw' := env.args.src_no_wrap
  have hdw' := env.args.dst_no_wrap
  have hsw : src.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hsw'
  have hdw : dst.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hdw'
  have hlt := env.lt
  have hl := env.sb.load; have hst := env.sb.store
  have ha1 := hA.a1; have ha7 := hA.a7; have hhn := hA.le
  rv_step ctx.code [ha1, ha7]                                -- 0x308 li t0, 256
  rv_step ctx.code [ha1, ha7]                                -- 0x30c bltu a2, t0
  · exact bwd_blocks env regs₀ m₀ he h _ mem
      ⟨by simp [writeReg_apply, ha1], by simp [writeReg_apply, ha7], hhn, hA.copied,
        by rv_preserved hA.preserved⟩
      (by simp [writeReg_apply, ha2]) hal
  · rename_i hge
    refine Exits.loop
      (I := fun k s => s.pc = B + 0x310#64 ∧ BwdAt regs₀ m₀ dst src n (h - 256 * k) s.regs s.mem ∧
        s.regs 12 = BitVec.ofNat 64 (h - 256 * k) ∧ 256 ≤ h - 256 * k ∧ s.regs 5 = 256#64)
      (h / 256) (fun k s h => by have := h.2.2.2.1; omega) ?_ 0 _
      ⟨rfl, ?_, ?_, ?_, ?_⟩
    · intro k s ⟨hpc, hA', ha2', h256, h5⟩ hcont
      obtain ⟨pc, regs', mem'⟩ := s
      simp only at hpc hA' ha2' h256 h5
      subst hpc
      have ha1' := hA'.a1; have ha7' := hA'.a7; have hcop' := hA'.copied; have hkn := hA'.le
      have hal' : (src.toNat + (h - 256 * k)) % 8 = 0 := by omega
      refine bwd_pairs env m₀ 0x310 32 (h - 256 * k) (by decide +kernel) (by decide) (by omega)
        hkn hal' regs' ha1' ha7' ?_ 0 (Nat.zero_le _) regs' mem' (fun _ _ => rfl)
        (hcop'.congr_lo (by omega))
      intro regs'' mem'' hr hc
      have ha1'' : regs'' 11 = src + BitVec.ofNat 64 (h - 256 * k) := by
        rw [hr 11 (by decide), ha1']
      have ha7'' : regs'' 17 = dst + BitVec.ofNat 64 (h - 256 * k) := by
        rw [hr 17 (by decide), ha7']
      have ha2'' : regs'' 12 = BitVec.ofNat 64 (h - 256 * k) := by rw [hr 12 (by decide), ha2']
      have h5'' : regs'' 5 = 256#64 := by rw [hr 5 (by decide), h5]
      have hpres : ∀ r, r ∉ clobbered → regs'' r = regs₀ r := by
        intro r hr'
        rw [hr r (fun h => by subst h; exact hr' (by decide))]
        exact hA'.preserved r hr'
      -- `omega` must not see `256 * (k + 1)` (README, "Performance pitfalls").
      rw [Nat.mul_add_one, ← Nat.sub_sub] at hcont
      show Exits _ _ _ ⟨B + 0x410#64, regs'', mem''⟩
      iterate 3 rv_step ctx.code [ha1'', ha7'', h5'', ha2'', BitVec.ofNat_add_ofNat,
        Arith.ofNat_add_neg_lit]                             -- 0x410 .. 0x418 addi -256
      rv_step ctx.code [ha1'', ha7'', h5'']                  -- 0x41c bgeu a2, t0
      · rename_i hc'
        rw [Arith.toNat_ofNat_of_lt (by omega)] at hc'
        refine hcont _ ⟨rfl, ?_, ?_, ?_, ?_⟩
        · refine BwdAt.of_eq (a := h - 256 * k - 256) (b := h - 256 * k - 256) ?_ ?_
            (hc.congr_lo (by omega)) rfl rfl (by omega) ?_
          · simp [writeReg_apply]
          · simp [writeReg_apply]
          · rv_preserved hpres
        · simp [writeReg_apply]
        · exact hc'
        · simp [writeReg_apply, h5'']
      · rename_i hc'
        rw [Arith.toNat_ofNat_of_lt (by omega)] at hc'
        refine bwd_blocks env regs₀ m₀ he (h - 256 * k - 256) _ _ ?_ ?_ (by omega)
        · refine BwdAt.of_eq (a := h - 256 * k - 256) (b := h - 256 * k - 256) ?_ ?_
            (hc.congr_lo (by omega)) rfl rfl (by omega) ?_
          · simp [writeReg_apply]
          · simp [writeReg_apply]
          · rv_preserved hpres
        · simp only [writeReg_apply, Fin.reduceEq, ite_false, ite_true]
          exact Arith.toNat_ofNat_of_lt (by omega)
    · refine BwdAt.of_eq (a := h) (b := h) ?_ ?_ hA.copied (by omega) (by omega) hhn ?_
      · simp [writeReg_apply, ha1]
      · simp [writeReg_apply, ha7]
      · rv_preserved hA.preserved
    · simp only [writeReg_apply, Fin.reduceEq, ite_false]
      exact Arith.eq_ofNat_of_toNat_eq ha2
    · omega
    · simp [writeReg_apply]

set_option maxHeartbeats 500000 in
/-- The backward copy from `0x2c0`: `a1 = src + n`, `a7 = dst + n`, then up
to three byte/half/word steps from the top that make `a1` (and with it `a7`)
8-byte aligned, then the loops. -/
theorem bwd_prologue (env : BwdEnv cfg B L F dst src n R) (regs₀ : Reg → BitVec 64) (m₀ : Mem)
    (he : EntryRegs regs₀ dst src n R) (hn8 : 8 ≤ n.toNat) :
    Exits cfg (region B) (Returned ⟨B + 0x2c0#64, regs₀, m₀⟩ R dst src n)
      ⟨B + 0x2c0#64, regs₀, m₀⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hsw' := env.args.src_no_wrap
  have hdw' := env.args.dst_no_wrap
  have hsw : src.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hsw'
  have hdw : dst.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hdw'
  have hlt := env.lt
  have hds := env.same
  have hl := env.sb.load; have hst := env.sb.store
  have ha0 := he.a0; have ha1 := he.a1
  have ha2 : regs₀ 12 = BitVec.ofNat 64 n.toNat := Arith.eq_ofNat_of_toNat_eq (by rw [he.a2])
  have hcop : Copied m₀ m₀ dst src n.toNat n.toNat := Copied.empty _ _ _ _
  have hcases : (src.toNat + n.toNat) % 8 = 0 ∨ (src.toNat + n.toNat) % 8 = 1 ∨
      (src.toNat + n.toNat) % 8 = 2 ∨ (src.toNat + n.toNat) % 8 = 3 ∨
      (src.toNat + n.toNat) % 8 = 4 ∨ (src.toNat + n.toNat) % 8 = 5 ∨
      (src.toNat + n.toNat) % 8 = 6 ∨ (src.toNat + n.toNat) % 8 = 7 := by omega
  rcases hcases with hc | hc | hc | hc | hc | hc | hc | hc <;>
  · rv_steps_to ctx.code 0x308 [ha0, ha1, ha2, hl, hst, BitVec.ofNat_add_ofNat,
      Arith.ofNat_add_neg_lit]
    refine bwd_large env regs₀ m₀ he (n.toNat - (src.toNat + n.toNat) % 8) _ _
      (BwdAt.of_eq (a := n.toNat - (src.toNat + n.toNat) % 8)
        (b := n.toNat - (src.toNat + n.toNat) % 8) ?_ ?_ ?_ rfl rfl (by omega) ?_) ?_ (by omega)
    · simp only [writeReg_apply, Fin.reduceEq, ite_true, ite_false]
      exact Arith.add_ofNat_congr (by omega)
    · simp only [writeReg_apply, Fin.reduceEq, ite_true, ite_false]
      exact Arith.add_ofNat_congr (by omega)
    · copied_bwd
    · rv_preserved
    · simp only [writeReg_apply, Fin.reduceEq, ite_true, ite_false, ha2]
      rw [Arith.toNat_ofNat_of_lt (by omega)]
      omega

end CopyRuntime.Wrapper
