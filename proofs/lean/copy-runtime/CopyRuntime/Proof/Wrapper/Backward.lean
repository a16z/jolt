import CopyRuntime.Proof.Wrapper.Tables
import CopyRuntime.Proof.Loops

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

/-! ## Memory: backward copies -/

/-- In a backward copy, a suffix copy `[lo, hi)` has not touched the source
bytes at offsets `< lo`. -/
theorem Copied.read_bwd {m m₀ : Mem} {dst src : BitVec 64} {lo hi j : Nat}
    (h : Copied m m₀ dst src lo hi) (hlt : src.toNat < dst.toNat) (hj : j < lo)
    (hdst : dst.toNat + hi < 2 ^ 64) (hsrc : src.toNat + j < 2 ^ 64) :
    m (src + BitVec.ofNat 64 j) = m₀ (src + BitVec.ofNat 64 j) := by
  rw [h, ite_eq_right]
  intro ⟨h1, h2⟩
  rw [BitVec.toNat_sub, BitVec.toNat_add, BitVec.toNat_ofNat, Nat.mod_eq_of_lt (a := j) (by omega),
    Nat.mod_eq_of_lt (a := src.toNat + j) hsrc] at h1 h2
  have := dst.isLt
  by_cases hc : dst.toNat ≤ src.toNat + j
  · rw [show 2 ^ 64 - dst.toNat + (src.toNat + j) = (src.toNat + j - dst.toNat) + 2 ^ 64 by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)] at h1
    omega
  · rw [Nat.mod_eq_of_lt (by omega)] at h2
    omega

theorem Copied.loadLE_bwd {m m₀ : Mem} {dst src : BitVec 64} {lo hi j w : Nat}
    (h : Copied m m₀ dst src lo hi) (hlt : src.toNat < dst.toNat) (hj : j + w ≤ lo)
    (hdst : dst.toNat + hi < 2 ^ 64) (hsrc : src.toNat + j + w ≤ 2 ^ 64) :
    loadLE m (src + BitVec.ofNat 64 j) w = loadLE m₀ (src + BitVec.ofNat 64 j) w := by
  apply loadLE_congr
  intro i hi
  rw [Arith.add_lit_lit]
  exact h.read_bwd hlt (by omega) hdst (by omega)

/-- One chunk of a backward copy: the value was loaded (from offset `c`) out of
a memory `m'` that was a suffix copy starting at `lo' ≥ c + w`. -/
theorem Copied.store_bwd {m m' m₀ : Mem} {dst src : BitVec 64} {c w lo' hi hi' : Nat}
    (h : Copied m m₀ dst src (c + w) hi) (h' : Copied m' m₀ dst src lo' hi')
    (hlo : c + w ≤ lo') (hlt : src.toNat < dst.toNat) (hw : w ≤ 8)
    (hdst : dst.toNat + hi' < 2 ^ 64) (hsrc : src.toNat + c + w ≤ 2 ^ 64)
    (hhi : hi < 2 ^ 64) (hcw : c + w ≤ hi) :
    Copied (storeLE m (dst + BitVec.ofNat 64 c) (loadLE m' (src + BitVec.ofNat 64 c) w) w)
      m₀ dst src c hi := by
  rw [h'.loadLE_bwd hlt hlo hdst hsrc]
  exact h.extend_down hw hhi hcw

theorem Copied.bwd_congr_lo {m m₀ : Mem} {dst src : BitVec 64} {a b hi : Nat}
    (h : Copied m m₀ dst src a hi) (e : a = b) : Copied m m₀ dst src b hi := e ▸ h

/-- `Copied.store_bwd` with the value loaded from the memory being stored to,
ending at a lower bound `lo` equal to the chunk offset `c`. -/
theorem Copied.bwd_self_to {m m₀ : Mem} {dst src : BitVec 64} {c w lo hi : Nat}
    (h : Copied m m₀ dst src (c + w) hi) (hlt : src.toNat < dst.toNat) (hw : w ≤ 8)
    (hdst : dst.toNat + hi < 2 ^ 64) (hsrc : src.toNat + c + w ≤ 2 ^ 64) (hcw : c + w ≤ hi)
    (e : c = lo) :
    Copied (storeLE m (dst + BitVec.ofNat 64 c) (loadLE m (src + BitVec.ofNat 64 c) w) w)
      m₀ dst src lo hi :=
  Copied.bwd_congr_lo (h.store_bwd h (Nat.le_refl _) hlt hw hdst hsrc (by omega) hcw) e

/-- `Copied.bwd_self_to` for the chunk at offset 0 (addresses `dst`, `src`). -/
theorem Copied.bwd_zero_self_to {m m₀ : Mem} {dst src : BitVec 64} {w lo hi : Nat}
    (h : Copied m m₀ dst src w hi) (hlt : src.toNat < dst.toNat) (hw : w ≤ 8)
    (hdst : dst.toNat + hi < 2 ^ 64) (hsrc : src.toNat + w ≤ 2 ^ 64) (hcw : w ≤ hi)
    (e : 0 = lo) :
    Copied (storeLE m dst (loadLE m src w) w) m₀ dst src lo hi := by
  have := Copied.bwd_self_to (c := 0) (by simpa using h) hlt hw hdst (by simpa using hsrc)
    (by simpa using hcw) e
  simpa using this

/-- Prove `Copied m m₀ dst src lo hi` for a chain of backward stores whose
values were loaded from the memory being stored to; the innermost memory must
have a `Copied` hypothesis in context. Matching is at reducible transparency
(see `copied_fwd`). -/
macro "copied_bwd" : tactic => `(tactic| (
  repeat (first
    | exact Copied.bwd_congr_lo (by with_reducible assumption) (by omega)
    | with_reducible (refine Copied.bwd_self_to ?_ (by assumption) (by decide) (by omega) (by omega) (by omega) (by omega))
    | with_reducible (refine Copied.bwd_zero_self_to ?_ (by assumption) (by decide) (by omega) (by omega) (by omega) (by omega)))))

/-! ### Blocks stored in ascending order

The 80-byte backward block loop loads ten words and then stores them from the
lowest address up, so the copied interval grows upward from `h - 80` inside a
gap below `[h, n)`. `CopiedOver m base m₀ dst src lo hi` describes such a
partial block: `m` is the snapshot copy on `[lo, hi)` and equals `base`
elsewhere; `CopiedOver.join` glues it onto the suffix copy `base`. -/

/-- `m` agrees with the snapshot copy of `m₀` on destination offsets
`[lo, hi)` and with `base` elsewhere. -/
def CopiedOver (m base m₀ : Mem) (dst src : BitVec 64) (lo hi : Nat) : Prop :=
  ∀ x, m x = if lo ≤ (x - dst).toNat ∧ (x - dst).toNat < hi then m₀ (src + (x - dst)) else base x

theorem CopiedOver.empty_of_eq {base m₀ : Mem} {dst src : BitVec 64} {a b : Nat} (e : a = b) :
    CopiedOver base base m₀ dst src a b := by
  intro x
  rw [ite_eq_right (by omega)]

private theorem sub_add_lit_bwd (x d : BitVec 64) (c : Nat) :
    x - (d + BitVec.ofNat 64 c) = x - d - BitVec.ofNat 64 c := by
  rw [BitVec.sub_sub]

private theorem toNat_sub_lit_bwd {y : BitVec 64} {c : Nat} (hc : c < 2 ^ 64) :
    (y - BitVec.ofNat 64 c).toNat =
      if c ≤ y.toNat then y.toNat - c else 2 ^ 64 - c + y.toNat := by
  rw [BitVec.toNat_sub, BitVec.toNat_ofNat, Nat.mod_eq_of_lt hc]
  have := y.isLt
  split <;> omega

private theorem ite_ite_same_bwd {α : Sort _} {P Q R : Prop} [Decidable P] [Decidable Q]
    [Decidable R] {a m : α} (h : R ↔ P ∨ Q) :
    (if P then a else if Q then a else m) = if R then a else m := by
  by_cases hP : P <;> by_cases hQ : Q <;> simp_all

/-- Append the chunk `[hi, hi + w)` (copied from the snapshot) to `[lo, hi)`. -/
theorem CopiedOver.extend_up {m base m₀ : Mem} {dst src : BitVec 64} {lo hi w : Nat}
    (h : CopiedOver m base m₀ dst src lo hi) (hw : w ≤ 8) (hhi : hi + w < 2 ^ 64)
    (hlo : lo ≤ hi) :
    CopiedOver (storeLE m (dst + BitVec.ofNat 64 hi) (loadLE m₀ (src + BitVec.ofNat 64 hi) w) w)
      base m₀ dst src lo (hi + w) := by
  intro x
  rw [storeLE_loadLE_apply _ _ _ _ _ hw, h x, sub_add_lit_bwd, toNat_sub_lit_bwd (by omega)]
  have hsrc : src + BitVec.ofNat 64 hi + (x - dst - BitVec.ofNat 64 hi) = src + (x - dst) := by
    bv_omega
  rw [hsrc]
  have := (x - dst).isLt
  apply ite_ite_same_bwd
  split <;> constructor <;> intro <;> omega

/-- One store of an ascending block, with the value loaded from the memory
`base` before the block, a suffix copy from `lo' ≥ k + w`. -/
theorem CopiedOver.store_up_to {m base m₀ : Mem} {dst src : BitVec 64} {lo k w hi lo' hi' : Nat}
    (h : CopiedOver m base m₀ dst src lo k) (h' : Copied base m₀ dst src lo' hi')
    (hlo' : k + w ≤ lo') (hlt : src.toNat < dst.toNat) (hw : w ≤ 8)
    (hdst : dst.toNat + hi' < 2 ^ 64) (hsrc : src.toNat + k + w ≤ 2 ^ 64)
    (hk : k + w < 2 ^ 64) (hlo : lo ≤ k) (e : k + w = hi) :
    CopiedOver (storeLE m (dst + BitVec.ofNat 64 k) (loadLE base (src + BitVec.ofNat 64 k) w) w)
      base m₀ dst src lo hi := by
  rw [h'.loadLE_bwd hlt hlo' hdst hsrc, ← e]
  exact h.extend_up hw hk hlo

/-- Glue a block `[a, b)` stored over `base` onto the suffix copy `[b, c)` of
`base`. -/
theorem CopiedOver.join {m base m₀ : Mem} {dst src : BitVec 64} {a b c : Nat}
    (h : CopiedOver m base m₀ dst src a b) (hb : Copied base m₀ dst src b c)
    (hab : a ≤ b) (hbc : b ≤ c) : Copied m m₀ dst src a c := by
  intro x
  rw [h x, hb x]
  by_cases h1 : a ≤ (x - dst).toNat ∧ (x - dst).toNat < b
  · rw [ite_eq_left h1, ite_eq_left (by omega)]
  · rw [ite_eq_right h1]
    by_cases h2 : b ≤ (x - dst).toNat ∧ (x - dst).toNat < c
    · rw [ite_eq_left h2, ite_eq_left (by omega)]
    · rw [ite_eq_right h2, ite_eq_right (by omega)]

end CopyRuntime

namespace CopyRuntime.Wrapper

/-- Prove `Copied m m₀ dst src lo hi` for a block of stores in ascending
address order on top of a memory `base` with hypothesis `hcop`, each storing a
value loaded from `base` below the copied part. -/
macro "copied_block_bwd " hcop:term : tactic => `(tactic| (
  refine CopiedOver.join ?_ $hcop (by omega) (by omega)
  repeat (first
    | with_reducible (exact CopiedOver.empty_of_eq (by omega))
    | with_reducible (refine CopiedOver.store_up_to ?_ $hcop (by omega) (by assumption) (by decide) (by omega) (by omega) (by omega) (by omega) (by omega)))))

/-! ## The backward invariant -/

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

/-- Prove the `preserved` field of a state whose register file is a chain of
writes (to clobbered registers) on top of `regs` with `hpres` for `regs`. -/
macro "bwd_preserved" hpres:term : tactic => `(tactic| (
  intro r hr
  have hr' := hr
  simp only [clobbered, List.mem_cons, List.not_mem_nil, or_false, not_or] at hr'
  simp only [writeReg_apply, hr', ite_false]
  exact $hpres r hr))

open Lean Meta Elab Tactic in
/-- `bwd_guard_not_at hc off` fails iff the goal's `pc` is `base + off` (a copy
of `rv_guard_not_at`). -/
elab "bwd_guard_not_at " hc:term:max off:num : tactic => withMainContext do
  let hcE ← elabTerm hc none
  let tgt ← instantiateMVars (← getMainTarget)
  let_expr Exits _ _ _ s := tgt | return
  let_expr State.mk pc _ _ := s.consumeMData | return
  let hcTy ← whnfR (← instantiateMVars (← inferType hcE))
  let_expr CodeAt _ base _ _ := hcTy | throwError "bwd_guard_not_at: not a CodeAt fact"
  try
    let k ← Tactic.offsetFrom pc base
    if k == off.getNat then throwError "bwd_guard_not_at: at offset {k}"
  catch e =>
    if (← e.toMessageData.toString).startsWith "bwd_guard_not_at" then throw e

/-- Step until the pc reaches `base + off` (on every branch). -/
syntax "bwd_steps_to " term:max num (" [" term,* "]")? : tactic
macro_rules
  | `(tactic| bwd_steps_to $hc $off) => `(tactic| bwd_steps_to $hc $off [])
  | `(tactic| bwd_steps_to $hc $off [$ls,*]) =>
    `(tactic| repeat (bwd_guard_not_at $hc $off; rv_step $hc [$ls,*]))

theorem Bwd.add_ofNat_congr (x : BitVec 64) {a b : Nat} (e : a = b) :
    x + BitVec.ofNat 64 a = x + BitVec.ofNat 64 b := e ▸ rfl

variable {cfg : Config} {B L F dst src n R : BitVec 64}

theorem BwdEnv.post_ret {regs₀ regs : Reg → BitVec 64} {m₀ mem : Mem} {p : BitVec 64}
    (env : BwdEnv cfg B L F dst src n R) (he : EntryRegs regs₀ dst src n R)
    (hcop : Copied mem m₀ dst src 0 n.toNat)
    (hpres : ∀ r, r ∉ clobbered → regs r = regs₀ r) :
    Returned ⟨p, regs₀, m₀⟩ R dst src n ⟨R, regs, mem⟩ :=
  ⟨rfl, by simp [State.reg_mk, hpres 10 (by decide), he.a0],
    hcop.memmove (Nat.le_of_lt env.args.dst_no_wrap),
    fun r hr => by simp [State.reg_mk, hpres r hr]⟩

/-! ## The 4/2/1-byte tail -/

set_option maxHeartbeats 4000000 in
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
    refine .exit env.rout (env.post_ret he ?_ ?_)
    · copied_bwd
    · bwd_preserved hA.preserved

theorem Bwd.and_neg8_eq (x : BitVec 64) : x &&& 0xfffffffffffffff8#64 = (x >>> 3) <<< 3 := by
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

theorem Bwd.toNat_and_neg8 (x : BitVec 64) :
    (x &&& 0xfffffffffffffff8#64).toNat = x.toNat / 8 * 8 := by
  rw [Bwd.and_neg8_eq, BitVec.toNat_shiftLeft, BitVec.toNat_ushiftRight, Nat.shiftLeft_eq,
    Nat.shiftRight_eq_div_pow]
  have := x.isLt
  rw [Nat.mod_eq_of_lt (by omega)]

/-- Address normalization for negative immediates: `x + ofNat a + imm (-c)` is
simplified by `rv_exec` to `x + ofNat (a + (2^64 - c))`; this rewrites the
offset back to `a - c` when `c ≤ a`, so that offsets stay small. -/
theorem Bwd.ofNat_add_neg_lit {a b : Nat} (hb : 9223372036854775808 ≤ b)
    (hb' : b < 18446744073709551616) (hab : 18446744073709551616 - b ≤ a) :
    BitVec.ofNat 64 (a + b) = BitVec.ofNat 64 (a - (18446744073709551616 - b)) := by
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.toNat_ofNat]
  omega

set_option maxHeartbeats 8000000 in
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
  have hTn : T.toNat = h / 8 * 8 := by rw [← hT, Bwd.toNat_and_neg8, ha2]
  rv_step ctx.code [ha1, ha7]                                -- 0x48c sub t0, a1, t0
  rv_step ctx.code [ha1, ha7]                                -- 0x490 c.andi a2, 7
  generalize hAA : regs 12 &&& 7#64 = A
  have hAn : A.toNat = h % 8 := by rw [← hAA, toNat_and_seven, ha2]
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
    · bwd_preserved hA.preserved
    · simp [writeReg_apply, hAn]; omega
  · rename_i hne
    have hq : 8 ≤ h := by
      apply Nat.le_of_not_lt
      intro h8
      apply hne
      rw [Nat.mod_eq_of_lt h8]
    -- Loop invariant after `k` words.
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
      rv_step ctx.code [ha1', ha7', h5, h12, hl, Bwd.ofNat_add_neg_lit]   -- 0x496 ld t1, -8(a1)
      rv_step ctx.code [ha1', ha7', h5, h12, hst, Bwd.ofNat_add_neg_lit]  -- 0x49a sd t1, -8(a7)
      rv_step ctx.code [ha1', ha7', h5, h12, Bwd.ofNat_add_neg_lit]       -- 0x49e c.addi a1, -8
      rv_step ctx.code [ha1', ha7', h5, h12, Bwd.ofNat_add_neg_lit]       -- 0x4a0 c.addi a7, -8
      rv_step ctx.code [ha1', ha7', h5, h12]                          -- 0x4a2 bltu t0, a1
      · -- Another word.
        refine hcont _ ⟨rfl, ?_, ?_, ?_, ?_⟩
        · refine BwdAt.of_eq (a := h - 8 * k - 8) (b := h - 8 * k - 8) ?_ ?_ ?_
            (by omega) (by omega) (by omega) ?_
          · simp [writeReg_apply]
          · simp [writeReg_apply]
          · copied_bwd
          · bwd_preserved hA'.preserved
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
          · bwd_preserved hA'.preserved
        · simp [writeReg_apply, h12, hAn]
    · refine BwdAt.of_eq (a := h) (b := h) ?_ ?_ hA.copied (by omega) (by omega) hhn ?_
      · simp [writeReg_apply, ha1]
      · simp [writeReg_apply, ha7]
      · bwd_preserved hA.preserved
    · simp [writeReg_apply]
    · simp [writeReg_apply]
    · omega

theorem Bwd.eq_ofNat_of_toNat_eq {x : BitVec 64} {h : Nat} (hx : x.toNat = h) :
    x = BitVec.ofNat 64 h := by
  apply BitVec.eq_of_toNat_eq
  rw [BitVec.toNat_ofNat, ← hx, Nat.mod_eq_of_lt x.isLt]

theorem Bwd.toNat_ofNat_of_lt {h : Nat} (hh : h < 2 ^ 64) : (BitVec.ofNat 64 h).toNat = h := by
  rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt hh]

/-! The length register `a2` is kept as `BitVec.ofNat 64 h` inside the loops,
and `addi a2, a2, -c` is normalized by `BitVec.ofNat_add_ofNat` and
`Bwd.ofNat_add_neg_lit` to `BitVec.ofNat 64 (h - c)`. A term `x + c#64` with a
symbolic `x` and a large literal `c` (such as `-80`) must never reach `simp`
with `BitVec.toNat_add`: that lemma is proved by `rfl`, and the kernel then
re-checks `(x + c).toNat = (x.toNat + c) % 2^64` by unfolding `Nat.add` on
the literal one successor at a time ("deep recursion"). -/

set_option maxHeartbeats 8000000 in
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
        by bwd_preserved hA.preserved⟩
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
        Bwd.ofNat_add_neg_lit]                                   -- 0x428 .. 0x480
      rv_step ctx.code [ha1', ha7', h5]                      -- 0x484 bgeu a2, t0
      · rename_i hc
        rw [Bwd.toNat_ofNat_of_lt (by omega)] at hc
        refine hcont _ ⟨rfl, ?_, ?_, ?_, ?_⟩
        · refine BwdAt.of_eq (a := h - 80 * k - 80) (b := h - 80 * k - 80) ?_ ?_ ?_
            rfl rfl (by omega) ?_
          · simp [writeReg_apply]
          · simp [writeReg_apply]
          · copied_block_bwd hcop'
          · bwd_preserved hA'.preserved
        · simp [writeReg_apply]
        · exact hc
        · simp [writeReg_apply, h5]
      · rename_i hc
        rw [Bwd.toNat_ofNat_of_lt (by omega)] at hc
        refine bwd_words env regs₀ m₀ he (h - 80 * k - 80) _ _ ?_ ?_ (by omega)
        · refine BwdAt.of_eq (a := h - 80 * k - 80) (b := h - 80 * k - 80) ?_ ?_ ?_
            rfl rfl (by omega) ?_
          · simp [writeReg_apply]
          · simp [writeReg_apply]
          · copied_block_bwd hcop'
          · bwd_preserved hA'.preserved
        · simp only [writeReg_apply, Fin.reduceEq, ite_false, ite_true]
          exact Bwd.toNat_ofNat_of_lt (by omega)
    · refine BwdAt.of_eq (a := h) (b := h) ?_ ?_ hA.copied (by omega) (by omega) hhn ?_
      · simp [writeReg_apply, ha1]
      · simp [writeReg_apply, ha7]
      · bwd_preserved hA.preserved
    · simp only [writeReg_apply, Fin.reduceEq, ite_false]
      exact Bwd.eq_ofNat_of_toNat_eq ha2
    · omega
    · simp [writeReg_apply]

theorem Bwd.add_lit_ofInt_neg (x : BitVec 64) {a k : Nat} (hk : k ≤ a) :
    x + BitVec.ofNat 64 a + BitVec.ofInt 64 (-(k : Int)) = x + BitVec.ofNat 64 (a - k) := by
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.toNat_add, BitVec.toNat_ofNat, BitVec.toNat_ofInt]
  omega

set_option maxHeartbeats 4000000 in
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
    rv_simp [h11, hl, Bwd.add_lit_ofInt_neg]
    refine ctx.code.step (o + 8 * i + 4) ⟨(hcode i him).2, by simp; omega⟩ rfl
      (codeRegion_lit (by omega) (by decide)) ?_
    rv_simp [h17, hst, Bwd.add_lit_ofInt_neg]
    rw [show o + 8 * i + 4 + 4 = o + 8 * (i + 1) by omega]
    refine ih (i + 1) (by omega) (by omega) _ _ ?_ ?_
    · intro r hr
      simp only [writeReg_apply, hr, ite_false]
      exact hregs r hr
    · exact Copied.bwd_self_to (Copied.bwd_congr_lo hcop (by omega)) hlt (by decide) (by omega)
        (by omega) (by omega) (by omega)


/-! In the loop bodies the offsets of the invariant `I (k + 1)` are rewritten
from `h - c * (k + 1)` to `h - c * k - c` (`Nat.mul_add_one`, `Nat.sub_sub`)
before they meet `omega`: `omega` compares atoms up to definitional equality,
and unifying two spellings of one offset with `c ≥ 256` unfolds the literal one
successor at a time ("maximum recursion depth"). -/

set_option maxHeartbeats 16000000 in
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
        by bwd_preserved hA.preserved⟩
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
        (Copied.bwd_congr_lo hcop' (by omega))
      intro regs'' mem'' hr hc
      have ha1'' : regs'' 11 = src + BitVec.ofNat 64 (h - 256 * k) := by
        rw [hr 11 (by decide), ha1']
      have ha7'' : regs'' 17 = dst + BitVec.ofNat 64 (h - 256 * k) := by
        rw [hr 17 (by decide), ha7']
      have ha2'' : regs'' 12 = BitVec.ofNat 64 (h - 256 * k) := by rw [hr 12 (by decide), ha2']
      have h5'' : regs'' 5 = 256#64 := by rw [hr 5 (by decide), h5]
      rw [Nat.mul_add_one, ← Nat.sub_sub] at hcont
      show Exits _ _ _ ⟨B + 0x410#64, regs'', mem''⟩
      iterate 3 rv_step ctx.code [ha1'', ha7'', h5'', ha2'', BitVec.ofNat_add_ofNat,
        Bwd.ofNat_add_neg_lit]                                   -- 0x410 .. 0x418 addi -256
      rv_step ctx.code [ha1'', ha7'', h5'']                  -- 0x41c bgeu a2, t0
      · rename_i hc'
        rw [Bwd.toNat_ofNat_of_lt (by omega)] at hc'
        refine hcont _ ⟨rfl, ?_, ?_, ?_, ?_⟩
        · refine BwdAt.of_eq (a := h - 256 * k - 256) (b := h - 256 * k - 256) ?_ ?_
            (Copied.bwd_congr_lo hc (by omega)) rfl rfl (by omega) ?_
          · simp [writeReg_apply]
          · simp [writeReg_apply]
          · intro r hr'
            have hr'' := hr'
            simp only [clobbered, List.mem_cons, List.not_mem_nil, or_false, not_or] at hr''
            simp only [writeReg_apply, hr'', ite_false]
            rw [hr r (fun h => by subst h; exact hr' (by decide))]
            exact hA'.preserved r hr'
        · simp [writeReg_apply]
        · exact hc'
        · simp [writeReg_apply, h5'']
      · rename_i hc'
        rw [Bwd.toNat_ofNat_of_lt (by omega)] at hc'
        refine bwd_blocks env regs₀ m₀ he (h - 256 * k - 256) _ _ ?_ ?_ (by omega)
        · refine BwdAt.of_eq (a := h - 256 * k - 256) (b := h - 256 * k - 256) ?_ ?_
            (Copied.bwd_congr_lo hc (by omega)) rfl rfl (by omega) ?_
          · simp [writeReg_apply]
          · simp [writeReg_apply]
          · intro r hr'
            have hr'' := hr'
            simp only [clobbered, List.mem_cons, List.not_mem_nil, or_false, not_or] at hr''
            simp only [writeReg_apply, hr'', ite_false]
            rw [hr r (fun h => by subst h; exact hr' (by decide))]
            exact hA'.preserved r hr'
        · simp only [writeReg_apply, Fin.reduceEq, ite_false, ite_true]
          exact Bwd.toNat_ofNat_of_lt (by omega)
    · refine BwdAt.of_eq (a := h) (b := h) ?_ ?_ hA.copied (by omega) (by omega) hhn ?_
      · simp [writeReg_apply, ha1]
      · simp [writeReg_apply, ha7]
      · bwd_preserved hA.preserved
    · simp only [writeReg_apply, Fin.reduceEq, ite_false]
      exact Bwd.eq_ofNat_of_toNat_eq ha2
    · omega
    · simp [writeReg_apply]

set_option maxHeartbeats 16000000 in
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
  have ha2 : regs₀ 12 = BitVec.ofNat 64 n.toNat := Bwd.eq_ofNat_of_toNat_eq (by rw [he.a2])
  have hcop : Copied m₀ m₀ dst src n.toNat n.toNat := Copied.empty _ _ _ _
  have hcases : (src.toNat + n.toNat) % 8 = 0 ∨ (src.toNat + n.toNat) % 8 = 1 ∨
      (src.toNat + n.toNat) % 8 = 2 ∨ (src.toNat + n.toNat) % 8 = 3 ∨
      (src.toNat + n.toNat) % 8 = 4 ∨ (src.toNat + n.toNat) % 8 = 5 ∨
      (src.toNat + n.toNat) % 8 = 6 ∨ (src.toNat + n.toNat) % 8 = 7 := by omega
  rcases hcases with hc | hc | hc | hc | hc | hc | hc | hc <;>
  · bwd_steps_to ctx.code 0x308 [ha0, ha1, ha2, hl, hst, BitVec.ofNat_add_ofNat, Bwd.ofNat_add_neg_lit]
    refine bwd_large env regs₀ m₀ he (n.toNat - (src.toNat + n.toNat) % 8) _ _
      (BwdAt.of_eq (a := n.toNat - (src.toNat + n.toNat) % 8)
        (b := n.toNat - (src.toNat + n.toNat) % 8) ?_ ?_ ?_ rfl rfl (by omega) ?_) ?_ (by omega)
    · simp only [writeReg_apply, Fin.reduceEq, ite_true, ite_false]
      exact Bwd.add_ofNat_congr _ (by omega)
    · simp only [writeReg_apply, Fin.reduceEq, ite_true, ite_false]
      exact Bwd.add_ofNat_congr _ (by omega)
    · copied_bwd
    · intro r hr
      simp only [clobbered, List.mem_cons, List.not_mem_nil, or_false, not_or] at hr
      simp only [writeReg_apply, hr, ite_false]
    · simp only [writeReg_apply, Fin.reduceEq, ite_true, ite_false, ha2]
      rw [Bwd.toNat_ofNat_of_lt (by omega)]
      omega

end CopyRuntime.Wrapper
