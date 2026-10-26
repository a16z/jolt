import CopyRuntime.Proof.Wrapper.Exits

/-!
# The forward general path (`src > dst`, equal offsets modulo 8, `n ≥ 8`)

Proved bottom-up in continuation style: each lemma takes a state at a code
offset described by `FwdAt j` (the first `j` bytes are copied, `a1`/`a7` point
`j` bytes into the ranges) and shows that the run ends in the final `Returned`
outcome. The loops use `Exits.loop`.

Forward copying is safe for overlap because every load reads a source byte at
an offset no store has reached yet (`Copied.store_fwd`): stores at destination
offset `i` hit source offset `i - (src - dst) < i`.
-/

namespace CopyRuntime.Wrapper

/-- Everything fixed during a forward copy. -/
structure FwdEnv (cfg : Config) (B L F dst src n R : BitVec 64) : Prop where
  ctx : Ctx cfg B L F
  sb : Sandbox cfg dst src n
  args : ArgsOk dst src n R
  rout : ¬ region B R
  le : dst.toNat ≤ src.toNat
  same : dst.toNat % 8 = src.toNat % 8

/-- The cursor state after `j` bytes of a forward copy. -/
structure FwdAt (regs₀ : Reg → BitVec 64) (m₀ : Mem) (dst src n : BitVec 64) (j : Nat)
    (regs : Reg → BitVec 64) (mem : Mem) : Prop where
  a1 : regs 11 = src + BitVec.ofNat 64 j
  a7 : regs 17 = dst + BitVec.ofNat 64 j
  le : j ≤ n.toNat
  copied : Copied mem m₀ dst src 0 j
  preserved : ∀ r, r ∉ clobbered → regs r = regs₀ r

/-- Prove `Copied m m₀ dst src 0 hi` for a chain of forward stores whose
values were loaded either from the memory being stored to, or from a memory
with a `Copied` hypothesis in context; the innermost memory must have a
`Copied` hypothesis (or be the snapshot itself). All matching is at reducible
transparency: at default transparency, unifying a store chain with a lemma or
hypothesis that does not fit unfolds `Copied` and every `storeLE`. -/
macro "copied_fwd" : tactic => `(tactic| (
  repeat (first
    | exact Copied.congr_hi (by with_reducible assumption) (by omega)
    | with_reducible (exact Copied.congr_hi (Copied.empty _ _ _ _) (by omega))
    | with_reducible
        (refine Copied.store_fwd_self ?_ (by assumption) (by decide) (by omega) (by omega)
          (by omega))
    | with_reducible
        (refine Copied.store_fwd ?_ (by assumption) (by omega) (by assumption) (by decide)
          (by omega) (by omega) (by omega))
    | with_reducible (refine Copied.store_fwd_zero ?_ (by decide) (by omega)))))

variable {cfg : Config} {B L F dst src n R : BitVec 64}

set_option maxHeartbeats 400000 in
/-- The 4/2/1-byte tail at `0x286`, with `r = n - j < 8` bytes left and `a1`
8-byte aligned. -/
theorem fwd_tail (env : FwdEnv cfg B L F dst src n R) (regs₀ : Reg → BitVec 64) (m₀ : Mem)
    (he : EntryRegs regs₀ dst src n R) (j : Nat) (regs : Reg → BitVec 64) (mem : Mem)
    (hF : FwdAt regs₀ m₀ dst src n j regs mem) (ha2 : (regs 12).toNat = n.toNat - j)
    (hr : n.toNat - j < 8) (hal : (src.toNat + j) % 8 = 0) :
    Exits cfg (region B) (Returned ⟨B + 0xac#64, regs₀, m₀⟩ R dst src n)
      ⟨B + 0x286#64, regs, mem⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hR := env.args.ret_even
  have hsw' := env.args.src_no_wrap
  have hsw : src.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hsw'
  have hdw : dst.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt env.args.dst_no_wrap
  have hle := env.le
  have hds := env.same
  have hl := env.sb.load; have hst := env.sb.store
  have ha1 := hF.a1; have ha7 := hF.a7; have hjn := hF.le; have hcop := hF.copied
  have hra : regs 1 = R := by rw [hF.preserved 1 (by decide), he.ra]
  have hcases : n.toNat - j = 0 ∨ n.toNat - j = 1 ∨ n.toNat - j = 2 ∨ n.toNat - j = 3 ∨
      n.toNat - j = 4 ∨ n.toNat - j = 5 ∨ n.toNat - j = 6 ∨ n.toNat - j = 7 := by omega
  rcases hcases with h | h | h | h | h | h | h | h <;>
  · repeat rv_step ctx.code [ha1, ha7, hra, hl, hst]
    refine .exit env.rout (Returned.of_copied he env.args.dst_no_wrap ?_ ?_)
    · copied_fwd
    · rv_preserved hF.preserved

/-- Build a `FwdAt` fact whose offsets are only equal (not syntactically) to `j`. -/
theorem FwdAt.of_eq {regs₀ regs : Reg → BitVec 64} {m₀ mem : Mem} {dst src n : BitVec 64}
    {j a b : Nat} (h1 : regs 11 = src + BitVec.ofNat 64 a)
    (h7 : regs 17 = dst + BitVec.ofNat 64 b) (hcop : Copied mem m₀ dst src 0 j)
    (ea : a = j) (eb : b = j) (hle : j ≤ n.toNat)
    (hpres : ∀ r, r ∉ clobbered → regs r = regs₀ r) : FwdAt regs₀ m₀ dst src n j regs mem := by
  subst ea eb
  exact ⟨h1, h7, hle, hcop, hpres⟩

/-- The word loop: `0x26a` computes the end pointer `t0 = a1 + (a2 & ~7)` and
keeps `a2 & 7`; `0x276` copies one word per iteration while `a1 < t0`. -/
theorem fwd_words (env : FwdEnv cfg B L F dst src n R) (regs₀ : Reg → BitVec 64) (m₀ : Mem)
    (he : EntryRegs regs₀ dst src n R) (j : Nat) (regs : Reg → BitVec 64) (mem : Mem)
    (hF : FwdAt regs₀ m₀ dst src n j regs mem) (ha2 : (regs 12).toNat = n.toNat - j)
    (hal : (src.toNat + j) % 8 = 0) :
    Exits cfg (region B) (Returned ⟨B + 0xac#64, regs₀, m₀⟩ R dst src n)
      ⟨B + 0x26a#64, regs, mem⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hsw' := env.args.src_no_wrap
  have hsw : src.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hsw'
  have hdw : dst.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt env.args.dst_no_wrap
  have hle := env.le
  have hds := env.same
  have hl := env.sb.load; have hst := env.sb.store
  have ha1 := hF.a1; have ha7 := hF.a7; have hjn := hF.le
  rv_step ctx.code [ha1, ha7]                                -- 0x26a andi t0, a2, -8
  generalize hT : regs 12 &&& 0xfffffffffffffff8#64 = T
  have hTn : T.toNat = (n.toNat - j) / 8 * 8 := by rw [← hT, Arith.toNat_and_neg8, ha2]
  rv_step ctx.code [ha1, ha7]                                -- 0x26e c.add t0, a1
  rv_step ctx.code [ha1, ha7]                                -- 0x270 c.andi a2, 7
  generalize hA : regs 12 &&& 7#64 = A
  have hAn : A.toNat = (n.toNat - j) % 8 := by rw [← hA, Arith.toNat_and_seven, ha2]
  have hend : T + (src + BitVec.ofNat 64 j) = src + BitVec.ofNat 64 (j + T.toNat) := by
    apply BitVec.eq_of_toNat_eq
    simp only [BitVec.toNat_add, BitVec.toNat_ofNat]
    omega
  rw [hend]
  rv_step ctx.code [ha1, ha7]                                -- 0x272 beq a1, t0
  · -- No whole words left: straight to the tail.
    rename_i heq
    have h8 : n.toNat - j < 8 := by
      have := congrArg BitVec.toNat heq
      simp only [BitVec.toNat_add, BitVec.toNat_ofNat] at this
      omega
    refine fwd_tail env regs₀ m₀ he j _ mem ⟨?_, ?_, hjn, hF.copied, ?_⟩ ?_ h8 hal
    · simp [writeReg_apply, ha1]
    · simp [writeReg_apply, ha7]
    · rv_preserved hF.preserved
    · simp [writeReg_apply, hAn]; omega
  · rename_i hne
    have hq : 0 < T.toNat := by
      apply Nat.pos_of_ne_zero
      intro h0
      apply hne
      apply BitVec.eq_of_toNat_eq
      simp only [BitVec.toNat_add, BitVec.toNat_ofNat]
      omega
    -- Loop invariant after `k` words.
    obtain ⟨q, hqdef⟩ : ∃ q, q = T.toNat / 8 := ⟨_, rfl⟩
    refine Exits.loop
      (I := fun k s => s.pc = B + 0x276#64 ∧ FwdAt regs₀ m₀ dst src n (j + 8 * k) s.regs s.mem ∧
        s.regs 5 = src + BitVec.ofNat 64 (j + T.toNat) ∧ s.regs 12 = A ∧ k < q)
      q (fun k s h => Nat.le_of_lt h.2.2.2.2) ?_ 0 _ ⟨rfl, ?_, ?_, ?_, ?_⟩
    · intro k s ⟨hpc, hF', h5, h12, hkq⟩ hcont
      obtain ⟨pc, regs', mem'⟩ := s
      simp only at hpc hF' h5 h12
      subst hpc
      have ha1' := hF'.a1; have ha7' := hF'.a7; have hcop' := hF'.copied
      rv_step ctx.code [ha1', ha7', h5, h12, hl]             -- 0x276 ld t1, 0(a1)
      rv_step ctx.code [ha1', ha7', h5, h12, hst]            -- 0x27a sd t1, 0(a7)
      rv_step ctx.code [ha1', ha7', h5, h12]                 -- 0x27e c.addi a1, 8
      rv_step ctx.code [ha1', ha7', h5, h12]                 -- 0x280 c.addi a7, 8
      rv_step ctx.code [ha1', ha7', h5, h12]                 -- 0x282 bltu a1, t0
      · -- Another word.
        refine hcont _ ⟨rfl, ?_, ?_, ?_, ?_⟩
        · refine FwdAt.of_eq (a := j + 8 * k + 8) (b := j + 8 * k + 8) ?_ ?_ ?_
            (by omega) (by omega) (by omega) ?_
          · simp [writeReg_apply]
          · simp [writeReg_apply]
          · copied_fwd
          · rv_preserved hF'.preserved
        · simp [writeReg_apply, h5]
        · simp [writeReg_apply, h12]
        · rename_i hlt
          simp only [BitVec.toNat_add, BitVec.toNat_ofNat] at hlt
          omega
      · -- Last word: on to the tail.
        rename_i hge
        refine fwd_tail env regs₀ m₀ he (j + T.toNat) _ _ ?_ ?_ (by omega) (by omega)
        · have hk1 : j + 8 * k + 8 = j + T.toNat := by
            simp only [BitVec.toNat_add, BitVec.toNat_ofNat] at hge; omega
          refine FwdAt.of_eq (a := j + 8 * k + 8) (b := j + 8 * k + 8) ?_ ?_ ?_
            hk1 hk1 (by omega) ?_
          · simp [writeReg_apply]
          · simp [writeReg_apply]
          · copied_fwd
          · rv_preserved hF'.preserved
        · simp [writeReg_apply, h12, hAn]
          omega
    · refine FwdAt.of_eq (a := j) (b := j) ?_ ?_ hF.copied rfl rfl hjn ?_
      · simp [writeReg_apply, ha1]
      · simp [writeReg_apply, ha7]
      · rv_preserved hF.preserved
    · simp [writeReg_apply]
    · simp [writeReg_apply]
    · omega

set_option maxHeartbeats 700000 in
/-- The 80-byte block loop at `0x208`/`0x210`: ten loads, then ten stores. -/
theorem fwd_blocks (env : FwdEnv cfg B L F dst src n R) (regs₀ : Reg → BitVec 64) (m₀ : Mem)
    (he : EntryRegs regs₀ dst src n R) (j : Nat) (regs : Reg → BitVec 64) (mem : Mem)
    (hF : FwdAt regs₀ m₀ dst src n j regs mem) (ha2 : (regs 12).toNat = n.toNat - j)
    (hal : (src.toNat + j) % 8 = 0) :
    Exits cfg (region B) (Returned ⟨B + 0xac#64, regs₀, m₀⟩ R dst src n)
      ⟨B + 0x208#64, regs, mem⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hsw' := env.args.src_no_wrap
  have hsw : src.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hsw'
  have hdw : dst.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt env.args.dst_no_wrap
  have hle := env.le
  have hds := env.same
  have hl := env.sb.load; have hst := env.sb.store
  have ha1 := hF.a1; have ha7 := hF.a7; have hjn := hF.le
  rv_step ctx.code [ha1, ha7]                                -- 0x208 li t0, 80
  rv_step ctx.code [ha1, ha7]                                -- 0x20c bltu a2, t0
  · exact fwd_words env regs₀ m₀ he j _ mem
      ⟨by simp [writeReg_apply, ha1], by simp [writeReg_apply, ha7], hjn, hF.copied,
        by rv_preserved hF.preserved⟩
      (by simp [writeReg_apply, ha2]) hal
  · rename_i hge
    refine Exits.loop
      (I := fun k s => s.pc = B + 0x210#64 ∧ FwdAt regs₀ m₀ dst src n (j + 80 * k) s.regs s.mem ∧
        (s.regs 12).toNat = n.toNat - (j + 80 * k) ∧ 80 ≤ n.toNat - (j + 80 * k) ∧
        s.regs 5 = 80#64)
      (n.toNat / 80) (fun k s h => by have := h.2.2.2.1; omega) ?_ 0 _
      ⟨rfl, ?_, ?_, ?_, ?_⟩
    · intro k s ⟨hpc, hF', ha2', h80, h5⟩ hcont
      obtain ⟨pc, regs', mem'⟩ := s
      simp only at hpc hF' ha2' h80 h5
      subst hpc
      have ha1' := hF'.a1; have ha7' := hF'.a7; have hcop' := hF'.copied
      have hal' : (src.toNat + (j + 80 * k)) % 8 = 0 := by omega
      have hsub : (regs' 12 - 80#64).toNat = n.toNat - (j + 80 * k + 80) := by
        rw [Arith.toNat_sub_ofNat_of_le (by omega), ha2']; omega
      -- 0x210 .. 0x262: ten loads, ten stores, `addi a1/a7/a2`
      iterate 23 rv_step ctx.code [ha1', ha7', h5, hl, hst, Arith.add_ofNat_eq_sub_of_ge]
      rv_step ctx.code [ha1', ha7', h5]                      -- 0x266 bgeu a2, t0
      · rename_i hc
        rw [hsub] at hc
        refine hcont _ ⟨rfl, ?_, ?_, ?_, ?_⟩
        · refine FwdAt.of_eq (a := j + 80 * k + 80) (b := j + 80 * k + 80) ?_ ?_ ?_
            (by omega) (by omega) (by omega) ?_
          · simp [writeReg_apply]
          · simp [writeReg_apply]
          · copied_fwd
          · rv_preserved hF'.preserved
        · simp only [writeReg_apply]; simp only [rv_exec, hsub]; omega
        · omega
        · simp [writeReg_apply, h5]
      · rename_i hc
        refine fwd_words env regs₀ m₀ he (j + 80 * k + 80) _ _ ?_ ?_ (by omega)
        · refine FwdAt.of_eq (a := j + 80 * k + 80) (b := j + 80 * k + 80) ?_ ?_ ?_
            rfl rfl (by omega) ?_
          · simp [writeReg_apply]
          · simp [writeReg_apply]
          · copied_fwd
          · rv_preserved hF'.preserved
        · simp only [writeReg_apply]; simp only [rv_exec, hsub]
    · refine FwdAt.of_eq (a := j) (b := j) ?_ ?_ hF.copied rfl rfl hjn ?_
      · simp [writeReg_apply, ha1]
      · simp [writeReg_apply, ha7]
      · rv_preserved hF.preserved
    · simp [writeReg_apply, ha2]
    · omega
    · simp [writeReg_apply]

/-- A run of `m` pairs `ld t1, 8i(a1); sd t1, 8i(a7)` (`i = 0 … m-1`) at code offset
`o`, with `a1 = src + j` and `a7 = dst + j` fixed: it extends a forward prefix
copy from `j + 8i` to `j + 8m` bytes and changes no register but `t1`. Proved
once by induction with the memory kept abstract, so the symbolic memory term
never grows; `hcode` is checked against the LLVM listing by kernel evaluation
when the lemma is used. -/
theorem fwd_pairs (env : FwdEnv cfg B L F dst src n R) (m₀ : Mem) {post : State → Prop}
    (o m j : Nat)
    (hcode : ∀ i < m,
      (o + 8 * i, Instr.load .double 6 11 (imm ((8 * i : Nat) : Int)), 4) ∈
        Generated.Disasm.wrapMemcpy ∧
      (o + 8 * i + 4, Instr.store .double 6 17 (imm ((8 * i : Nat) : Int)), 4) ∈
        Generated.Disasm.wrapMemcpy)
    (hend : o + 8 * m ≤ 0x578) (hfit : j + 8 * m ≤ n.toNat) (hal : (src.toNat + j) % 8 = 0)
    (regsS : Reg → BitVec 64) (hS11 : regsS 11 = src + BitVec.ofNat 64 j)
    (hS17 : regsS 17 = dst + BitVec.ofNat 64 j)
    (hcont : ∀ regs mem, (∀ r, r ≠ 6 → regs r = regsS r) → Copied mem m₀ dst src 0 (j + 8 * m) →
      Exits cfg (region B) post ⟨B + BitVec.ofNat 64 (o + 8 * m), regs, mem⟩) :
    ∀ i, i ≤ m → ∀ regs mem, (∀ r, r ≠ 6 → regs r = regsS r) →
      Copied mem m₀ dst src 0 (j + 8 * i) →
      Exits cfg (region B) post ⟨B + BitVec.ofNat 64 (o + 8 * i), regs, mem⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hsw' := env.args.src_no_wrap
  have hsw : src.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hsw'
  have hdw : dst.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt env.args.dst_no_wrap
  have hle := env.le
  have hds := env.same
  have hl := env.sb.load; have hst := env.sb.store
  suffices h : ∀ d i, m - i = d → i ≤ m → ∀ regs mem, (∀ r, r ≠ 6 → regs r = regsS r) →
      Copied mem m₀ dst src 0 (j + 8 * i) →
      Exits cfg (region B) post ⟨B + BitVec.ofNat 64 (o + 8 * i), regs, mem⟩ from
    fun i hi => h _ i rfl hi
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
    have h11 : regs 11 = src + BitVec.ofNat 64 j := by rw [hregs 11 (by decide), hS11]
    have h17 : regs 17 = dst + BitVec.ofNat 64 j := by rw [hregs 17 (by decide), hS17]
    refine ctx.code.step (o + 8 * i) ⟨(hcode i him).1, by simp; omega⟩ rfl
      (codeRegion_lit (by omega) (by decide)) ?_
    rv_simp [h11, hl, BitVec.ofInt_natCast]
    refine ctx.code.step (o + 8 * i + 4) ⟨(hcode i him).2, by simp; omega⟩ rfl
      (codeRegion_lit (by omega) (by decide)) ?_
    rv_simp [h17, hst, BitVec.ofInt_natCast]
    rw [show o + 8 * i + 4 + 4 = o + 8 * (i + 1) by omega]
    refine ih (i + 1) (by omega) (by omega) _ _ ?_ ?_
    · intro r hr
      simp only [writeReg_apply, hr, ite_false]
      exact hregs r hr
    · exact Copied.store_fwd_self hcop hle (by decide) (by omega) (by omega) (by omega)

/-- The thirty-two `ld t1, 8i(a1); sd t1, 8i(a7)` pairs of the 256-byte loop. -/
theorem fwd_large_code : ∀ i < 32,
    (0xf8 + 8 * i, Instr.load .double 6 11 (imm ((8 * i : Nat) : Int)), 4) ∈
      Generated.Disasm.wrapMemcpy ∧
    (0xf8 + 8 * i + 4, Instr.store .double 6 17 (imm ((8 * i : Nat) : Int)), 4) ∈
      Generated.Disasm.wrapMemcpy := by
  decide +kernel

/-- The 256-byte loop at `0xf0`/`0xf8`: thirty-two `ld`/`sd` pairs. -/
theorem fwd_large (env : FwdEnv cfg B L F dst src n R) (regs₀ : Reg → BitVec 64) (m₀ : Mem)
    (he : EntryRegs regs₀ dst src n R) (j : Nat) (regs : Reg → BitVec 64) (mem : Mem)
    (hF : FwdAt regs₀ m₀ dst src n j regs mem) (ha2 : (regs 12).toNat = n.toNat - j)
    (hal : (src.toNat + j) % 8 = 0) :
    Exits cfg (region B) (Returned ⟨B + 0xac#64, regs₀, m₀⟩ R dst src n)
      ⟨B + 0xf0#64, regs, mem⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hsw' := env.args.src_no_wrap
  have hsw : src.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hsw'
  have hdw : dst.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt env.args.dst_no_wrap
  have hle := env.le
  have hds := env.same
  have hl := env.sb.load; have hst := env.sb.store
  have ha1 := hF.a1; have ha7 := hF.a7; have hjn := hF.le
  rv_step ctx.code [ha1, ha7]                                -- 0x0f0 li t0, 256
  rv_step ctx.code [ha1, ha7]                                -- 0x0f4 bltu a2, t0
  · exact fwd_blocks env regs₀ m₀ he j _ mem
      ⟨by simp [writeReg_apply, ha1], by simp [writeReg_apply, ha7], hjn, hF.copied,
        by rv_preserved hF.preserved⟩
      (by simp [writeReg_apply, ha2]) hal
  · rename_i hge
    refine Exits.loop
      (I := fun k s => s.pc = B + 0xf8#64 ∧ FwdAt regs₀ m₀ dst src n (j + 256 * k) s.regs s.mem ∧
        (s.regs 12).toNat = n.toNat - (j + 256 * k) ∧ 256 ≤ n.toNat - (j + 256 * k) ∧
        s.regs 5 = 256#64)
      (n.toNat / 256) (fun k s h => by have := h.2.2.2.1; omega) ?_ 0 _
      ⟨rfl, ?_, ?_, ?_, ?_⟩
    · intro k s ⟨hpc, hF', ha2', h256, h5⟩ hcont
      obtain ⟨pc, regs', mem'⟩ := s
      simp only at hpc hF' ha2' h256 h5
      subst hpc
      -- Name the offset. `omega` must never see `256 * (k + 1)`, nor two
      -- spellings of one offset (`j + 256 * k + 256` and `j + (256 * k + 256)`):
      -- it compares them by unfolding `+ 256` in unary and hits the recursion limit.
      generalize hi : j + 256 * k = i at hF' ha2' h256
      have ha1' := hF'.a1; have ha7' := hF'.a7; have hcop' := hF'.copied
      have hin := hF'.le
      have hal' : (src.toNat + i) % 8 = 0 := by
        have := hF'.a1; omega
      have hsub : (regs' 12 - 256#64).toNat = n.toNat - (i + 256) := by
        rw [Arith.toNat_sub_ofNat_of_le (by omega), ha2']; omega
      have hpairs := fwd_pairs env m₀ (post := Returned ⟨B + 0xac#64, regs₀, m₀⟩ R dst src n)
        0xf8 32 i fwd_large_code (by decide) (by omega) hal' regs' ha1' ha7'
      refine hpairs ?_ 0 (Nat.zero_le _) regs' mem' (fun _ _ => rfl) hcop'
      intro regs'' mem'' hr hc
      have ha1'' : regs'' 11 = src + BitVec.ofNat 64 i := by rw [hr 11 (by decide), ha1']
      have ha7'' : regs'' 17 = dst + BitVec.ofNat 64 i := by rw [hr 17 (by decide), ha7']
      have ha2'' : regs'' 12 = regs' 12 := hr 12 (by decide)
      have h5'' : regs'' 5 = 256#64 := by rw [hr 5 (by decide), h5]
      have hpres : ∀ r, r ∉ clobbered → regs'' r = regs₀ r := by
        intro r hr'
        rw [hr r (fun h => by subst h; exact hr' (by decide))]
        exact hF'.preserved r hr'
      have hc256 : Copied mem'' m₀ dst src 0 (i + 256) := hc.congr_hi (by omega)
      show Exits _ _ _ ⟨B + 0x1f8#64, regs'', mem''⟩
      rv_step ctx.code [ha1'', ha7'', h5'']                  -- 0x1f8 addi a1, a1, 256
      rv_step ctx.code [ha1'', ha7'', h5'']                  -- 0x1fc addi a7, a7, 256
      rv_step ctx.code [ha2'', Arith.add_ofNat_eq_sub_of_ge]  -- 0x200 addi a2, a2, -256
      rv_step ctx.code [h5'']                                -- 0x204 bgeu a2, t0
      · rename_i hc'
        rw [hsub] at hc'
        refine hcont _ ⟨rfl, ?_, ?_, ?_, by simp [writeReg_apply, h5'']⟩ <;>
          rw [show j + 256 * (k + 1) = i + 256 by
            rw [Nat.mul_add, Nat.mul_one, ← Nat.add_assoc, hi]]
        · refine FwdAt.of_eq (a := i + 256) (b := i + 256) ?_ ?_ hc256 rfl rfl (by omega) ?_
          · simp [writeReg_apply]
          · simp [writeReg_apply]
          · rv_preserved hpres
        · simp only [writeReg_apply]; simp only [rv_exec, hsub]
        · exact hc'
      · rename_i hc'
        refine fwd_blocks env regs₀ m₀ he (i + 256) _ _ ?_ ?_ (by omega)
        · refine FwdAt.of_eq (a := i + 256) (b := i + 256) ?_ ?_ hc256 rfl rfl (by omega) ?_
          · simp [writeReg_apply]
          · simp [writeReg_apply]
          · rv_preserved hpres
        · simp only [writeReg_apply]; simp only [rv_exec, hsub]
    · refine FwdAt.of_eq (a := j) (b := j) ?_ ?_ hF.copied rfl rfl hjn ?_
      · simp [writeReg_apply, ha1]
      · simp [writeReg_apply, ha7]
      · rv_preserved hF.preserved
    · simp [writeReg_apply, ha2]
    · omega
    · simp [writeReg_apply]

set_option maxHeartbeats 900000 in
/-- The forward copy from `0xac`: `mv a7, a0`, then up to three byte/half/word
steps that make `a1` (and with it `a7`) 8-byte aligned, then the loops. -/
theorem fwd_prologue (env : FwdEnv cfg B L F dst src n R) (regs₀ : Reg → BitVec 64) (m₀ : Mem)
    (he : EntryRegs regs₀ dst src n R) (hn8 : 8 ≤ n.toNat) :
    Exits cfg (region B) (Returned ⟨B + 0xac#64, regs₀, m₀⟩ R dst src n)
      ⟨B + 0xac#64, regs₀, m₀⟩ := by
  have ctx := env.ctx
  have hB := ctx.base_even
  have hsw' := env.args.src_no_wrap
  have hsw : src.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt hsw'
  have hdw : dst.toNat + n.toNat ≤ 2 ^ 64 := Nat.le_of_lt env.args.dst_no_wrap
  have hle := env.le
  have hds := env.same
  have hl := env.sb.load; have hst := env.sb.store
  have ha0 := he.a0; have ha1 := he.a1; have ha2 := he.a2
  have hcop : Copied m₀ m₀ dst src 0 0 := Copied.empty _ _ _ _
  have hcases : src.toNat % 8 = 0 ∨ src.toNat % 8 = 1 ∨ src.toNat % 8 = 2 ∨
      src.toNat % 8 = 3 ∨ src.toNat % 8 = 4 ∨ src.toNat % 8 = 5 ∨ src.toNat % 8 = 6 ∨
      src.toNat % 8 = 7 := by omega
  rcases hcases with hc | hc | hc | hc | hc | hc | hc | hc <;>
  · rv_steps_to ctx.code 0xf0 [ha0, ha1, ha2, hl, hst, BitVec.zero_add,
      Arith.add_ofNat_eq_sub_of_ge, Arith.sub_ofNat_sub_ofNat]
    refine fwd_large env regs₀ m₀ he ((8 - src.toNat % 8) % 8) _ _ ⟨?_, ?_, by omega, ?_, ?_⟩
      ?_ (by omega)
    · simp only [rv_exec, ha1, hc]
    · simp only [rv_exec, hc]
    · copied_fwd
    · rv_preserved
    · simp only [rv_exec, ha2, hc]
      first | rfl | rw [Arith.toNat_sub_ofNat_of_le (by omega)]

end CopyRuntime.Wrapper
