/-
DRAFT -- NOT BUILT, DOES NOT YET CHECK.

Unfinished proof scripts for `Wrapper.fwd_blocks`, `Wrapper.fwd_large` and
`Wrapper.fwd_prologue` (stated with `sorry` in
`CopyRuntime/Proof/Wrapper/Forward.lean`). They are kept as a starting point
for finishing the forward path. Paste a proof back over the `sorry` and
compile with `lake env lean` to continue.

Status when this was set aside:
* `fwd_blocks`: the 23 symbolic steps and the branch take about 10 s. The
  postconditions were slow, and the last full compile reported a kernel
  "deep recursion" error for the theorem. That error was not yet diagnosed;
  suspect a term with a large literal that the kernel unfolds in unary, as
  the elaborator did before `rv_split`.
* `fwd_large`: uses `fwd_pairs` for the 32 `ld`/`sd` pairs, which is proved
  and fast. It had an elaborator max-recursion error near the instantiation
  of `fwd_pairs` (probably defeq of `B + ofNat (0xf8 + 8 * 0)` with
  `B + 0xf8#64` or a similar literal), plus the same kernel issue.
* `fwd_prologue`: case split on `src % 8`, then `rv_steps_to … 0xf0`. Not
  yet checked end to end.
-/

set_option maxHeartbeats 8000000 in
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
        by fwd_preserved hF.preserved⟩
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
      iterate 23 rv_step ctx.code [ha1', ha7', h5, hl, hst]  -- 0x210 .. 0x262
      rv_step ctx.code [ha1', ha7', h5]                      -- 0x266 bgeu a2, t0
      · refine hcont _ ⟨rfl, ?_, ?_, ?_, ?_⟩
        · refine FwdAt.of_eq (a := j + 80 * k + 80) (b := j + 80 * k + 80) ?_ ?_ ?_
            (by omega) (by omega) (by omega) ?_
          · simp [writeReg_apply]
          · simp [writeReg_apply]
          · copied_fwd
          · fwd_preserved hF'.preserved
        · rename_i hc
          simp only [rv_exec]
          simp only [BitVec.toNat_add, BitVec.toNat_ofNat] at hc ⊢
          omega
        · rename_i hc
          simp only [BitVec.toNat_add, BitVec.toNat_ofNat] at hc
          omega
        · simp [writeReg_apply, h5]
      · rename_i hc
        refine fwd_words env regs₀ m₀ he (j + 80 * k + 80) _ _ ?_ ?_ (by omega)
        · refine FwdAt.of_eq (a := j + 80 * k + 80) (b := j + 80 * k + 80) ?_ ?_ ?_
            rfl rfl (by omega) ?_
          · simp [writeReg_apply]
          · simp [writeReg_apply]
          · copied_fwd
          · fwd_preserved hF'.preserved
        · simp only [rv_exec]
          simp only [BitVec.toNat_add, BitVec.toNat_ofNat]
          omega
    · refine FwdAt.of_eq (a := j) (b := j) ?_ ?_ hF.copied rfl rfl hjn ?_
      · simp [writeReg_apply, ha1]
      · simp [writeReg_apply, ha7]
      · fwd_preserved hF.preserved
    · simp [writeReg_apply, ha2]
    · omega
    · simp [writeReg_apply]


set_option maxHeartbeats 16000000 in
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
        by fwd_preserved hF.preserved⟩
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
      have ha1' := hF'.a1; have ha7' := hF'.a7; have hcop' := hF'.copied
      have hal' : (src.toNat + (j + 256 * k)) % 8 = 0 := by omega
      refine fwd_pairs env m₀ 0xf8 32 (j + 256 * k) (by decide +kernel) (by decide) (by omega)
        hal' regs' ha1' ha7' ?_ 0 (Nat.zero_le _) regs' mem' (fun _ _ => rfl) hcop'
      intro regs'' mem'' hr hc
      have ha1'' : regs'' 11 = src + BitVec.ofNat 64 (j + 256 * k) := by
        rw [hr 11 (by decide), ha1']
      have ha7'' : regs'' 17 = dst + BitVec.ofNat 64 (j + 256 * k) := by
        rw [hr 17 (by decide), ha7']
      have ha2'' : regs'' 12 = regs' 12 := hr 12 (by decide)
      have h5'' : regs'' 5 = 256#64 := by rw [hr 5 (by decide), h5]
      show Exits _ _ _ ⟨B + 0x1f8#64, regs'', mem''⟩
      rv_step ctx.code [ha1'', ha7'', h5'']                  -- 0x1f8 addi a1, a1, 256
      rv_step ctx.code [ha1'', ha7'', h5'']                  -- 0x1fc addi a7, a7, 256
      rv_step ctx.code [ha1'', ha7'', h5'']                  -- 0x200 addi a2, a2, -256
      rv_step ctx.code [ha1'', ha7'', h5'', ha2'']           -- 0x204 bgeu a2, t0
      · refine hcont _ ⟨rfl, ?_, ?_, ?_, ?_⟩
        · refine FwdAt.of_eq (a := j + 256 * k + 256) (b := j + 256 * k + 256) ?_ ?_
            (Copied.congr_hi hc (by omega)) (by omega) (by omega) (by omega) ?_
          · simp [writeReg_apply]
          · simp [writeReg_apply]
          · intro r hr'
            have hr'' := hr'
            simp only [clobbered, List.mem_cons, List.not_mem_nil, or_false, not_or] at hr''
            simp only [writeReg_apply, hr'', ite_false]
            rw [hr r (fun h => by subst h; exact hr' (by decide))]
            exact hF'.preserved r hr'
        · rename_i hc'
          simp only [rv_exec]
          simp only [BitVec.toNat_add, BitVec.toNat_ofNat] at hc' ⊢
          omega
        · rename_i hc'
          simp only [BitVec.toNat_add, BitVec.toNat_ofNat] at hc'
          omega
        · simp [writeReg_apply, h5'']
      · rename_i hc'
        refine fwd_blocks env regs₀ m₀ he (j + 256 * k + 256) _ _ ?_ ?_ (by omega)
        · refine FwdAt.of_eq (a := j + 256 * k + 256) (b := j + 256 * k + 256) ?_ ?_
            (Copied.congr_hi hc (by omega)) rfl rfl (by omega) ?_
          · simp [writeReg_apply]
          · simp [writeReg_apply]
          · intro r hr'
            have hr'' := hr'
            simp only [clobbered, List.mem_cons, List.not_mem_nil, or_false, not_or] at hr''
            simp only [writeReg_apply, hr'', ite_false]
            rw [hr r (fun h => by subst h; exact hr' (by decide))]
            exact hF'.preserved r hr'
        · simp only [rv_exec]
          simp only [BitVec.toNat_add, BitVec.toNat_ofNat]
          omega
    · refine FwdAt.of_eq (a := j) (b := j) ?_ ?_ hF.copied rfl rfl hjn ?_
      · simp [writeReg_apply, ha1]
      · simp [writeReg_apply, ha7]
      · fwd_preserved hF.preserved
    · simp [writeReg_apply, ha2]
    · omega
    · simp [writeReg_apply]


set_option maxHeartbeats 16000000 in
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
  · rv_steps_to ctx.code 0xf0 [ha0, ha1, ha2, hl, hst]
    refine fwd_large env regs₀ m₀ he ((8 - src.toNat % 8) % 8) _ _ ?_ ?_ (by omega)
    · constructor
      · simp [writeReg_apply, ha1, hc]
      · simp [writeReg_apply, ha0, hc]
      · omega
      · copied_fwd
      · fwd_preserved (fun r _ => rfl)
    · simp only [writeReg_apply, ha2]
      first
        | omega
        | (simp only [BitVec.toNat_add, BitVec.toNat_ofNat]; omega)

