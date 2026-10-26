import Lean
import CopyRuntime.Proof.ExecLemmas
import CopyRuntime.Proof.Memory

/-!
# `rv_step`: symbolic execution of one listed instruction

`rv_fetch hc` expects a goal `Exits cfg region post ⟨pc, regs, mem⟩` with
`pc = base + c` (or `pc = base`) and `hc : CodeAt cfg base listing sites`. It
looks `c` up in `listing` (an LLVM listing constant, see `DecoderCheck`),
applies `CodeAt.step`, proves listing membership by kernel evaluation
(`decide +kernel`), proves the region side condition with `rv_region`, and
leaves the goal `Exits.after cfg region post (execute cfg ⟨pc, regs, mem⟩ i len)`.
It computes nothing that the kernel does not re-check: the lookup only chooses
which instruction to claim.

`rv_step hc [lemmas]` is `rv_fetch hc`, then `rv_simp [lemmas]` to evaluate the
instruction, then `rv_branch` to split on an undecided branch condition.
-/

namespace CopyRuntime.Tactic

open Lean Meta Elab Tactic

/-- Elements of a list literal (after unfolding a constant). -/
partial def listElems (e : Expr) : MetaM (Array Expr) := do
  let rec go (e : Expr) (acc : Array Expr) : MetaM (Array Expr) := do
    -- `whnfCore` removes the `have` chunks that long list literals elaborate to.
    let e ← whnfCore e.consumeMData
    match_expr e with
    | List.cons _ hd tl => go tl (acc.push hd)
    | List.nil _ => return acc
    | _ =>
      match e.getAppFn with
      | .const n lvls =>
        let some v := (← getEnv).find? n |>.bind (·.value?) |
          throwError "rv_step: cannot unfold listing {e}"
        go ((v.instantiateLevelParams (← getConstInfo n).levelParams lvls).beta e.getAppArgs) acc
      | _ => throwError "rv_step: unexpected listing shape {e}"
  go e #[]

def lookupListing (listing : Expr) (k : Nat) : MetaM (Expr × Expr) := do
  for entry in ← listElems listing do
    let_expr Prod.mk _ _ off rest := entry.consumeMData | continue
    let some off' ← getNatValue? off | continue
    if off' == k then
      let_expr Prod.mk _ _ instr len := rest.consumeMData |
        throwError "rv_step: malformed listing entry {entry}"
      return (instr, len)
  throwError "rv_step: offset {k} (0x{String.ofList (Nat.toDigits 16 k)}) is not an instruction \
    boundary of the listing"

/-- The literal offset of `pc` from `base` (syntactic match up to reducible
unfolding; never unfolds the arithmetic). -/
def offsetFrom (pc base : Expr) : MetaM Nat := do
  match_expr pc with
  | HAdd.hAdd _ _ _ _ b c =>
    if ← withReducible (isDefEq b base) then
      if let some ⟨_, v⟩ ← getBitVecValue? c then return v.toNat
  | _ => pure ()
  if ← withReducible (isDefEq pc base) then return 0
  throwError "rv_step: pc {pc} is not of the form {base} + literal"

/-- Close a region side condition `region (base + c)`. Extend with
`macro_rules | `(tactic| rv_region) => ...` for new region predicates. -/
syntax "rv_region" : tactic

def stepCore (hc : Expr) : TacticM Unit := withMainContext do
  let goal ← getMainGoal
  let tgt ← instantiateMVars (← goal.getType)
  let_expr Exits _ region post s := tgt | throwError "rv_step: goal is not `Exits`"
  let_expr State.mk pc _ _ := s.consumeMData |
    throwError "rv_step: state is not of the form ⟨pc, regs, mem⟩: {s}"
  let hcTy ← whnfR (← instantiateMVars (← inferType hc))
  let_expr CodeAt cfg base listing sites := hcTy |
    throwError "rv_step: {hc} is not a `CodeAt` fact"
  let k ← offsetFrom pc base
  let (instr, len) ← lookupListing listing k
  let thm ← mkAppOptM ``CodeAt.step
    #[some cfg, some base, some listing, some sites, some hc, some region, some post, some s,
      some (mkNatLit k), some instr, some len]
  let newGoals ← goal.apply thm
  let [hk, hpc, hreg, h] := newGoals |
    throwError "rv_step: unexpected goals after applying CodeAt.step"
  let run (g : MVarId) (tac : TSyntax `tactic) : TacticM Unit := do
    let gs ← Tactic.run g (evalTactic tac)
    unless gs.isEmpty do throwError "rv_step: side condition not closed: {gs}"
  run hk (← `(tactic| decide +kernel))
  run hpc (← `(tactic| first | with_reducible rfl | simp only [BitVec.add_zero]))
  run hreg (← `(tactic| rv_region))
  replaceMainGoal [h]

elab "rv_fetch " hc:term : tactic => withMainContext do
  let hc ← elabTerm hc none
  stepCore hc

/-- Arithmetic normalization: `toNat` of sums of a symbolic atom and literals
becomes Nat arithmetic. -/
syntax "rv_arith_simp" (Lean.Parser.Tactic.location)? : tactic
macro_rules
  | `(tactic| rv_arith_simp $[$loc]?) => `(tactic|
    simp only [Arith.add_sub_cancel_left, Arith.add_sub_add_left, Arith.add_lit_lit,
      BitVec.toNat_add, BitVec.toNat_ofNat, BitVec.toNat_sub, BitVec.ofNat_eq_ofNat,
      Nat.reduceMod, Nat.reduceAdd, Nat.reducePow, Nat.reduceSub, BitVec.reduceToNat,
      BitVec.reduceSub, BitVec.reduceAdd, ne_eq, BitVec.toNat_eq] $[$loc]?)

/-- Discharger for the conditional rewrite rules of `rv_exec`. It must not
change the local context (rewriting hypotheses inside a `simp` discharger made
the enclosing `simp` lose its cache and run exponentially long), so it only
normalizes its own goal before `omega`. The trailing `fail` keeps a failed
alternative from running in error-recovery mode (which would log an error and
admit the side goal). -/
macro "rv_disch" : tactic =>
  `(tactic| first | assumption | decide | ((try rv_arith_simp); omega) | fail "rv_disch")

/-- `rv_simp [lemmas]` is `simp (disch := rv_disch) only [rv_exec, lemmas]`
except that it tells `simp` that the discharger is well behaved (it only reads
the initial local context). Lean marks every custom discharger as not well
behaved, which disables simp's cache below binders; on long symbolic runs that
made each step exponentially slow in the length of the register-write chain. -/
syntax (name := rvSimp) "rv_simp" (" [" term,* "]")? : tactic

@[tactic rvSimp] def evalRvSimp : Tactic := fun stx => withMainContext do
  let args : Array Term :=
    if stx[1].getNumArgs == 0 then #[] else stx[1][1].getSepArgs.map (⟨·⟩)
  let lemmas ← args.mapM fun t => `(Lean.Parser.Tactic.simpLemma| $t:term)
  let simpStx ← `(tactic| simp (disch := rv_disch) only [rv_exec, $lemmas,*])
  let { ctx, simprocs, dischargeWrapper, .. } ← mkSimpContext simpStx (eraseLocal := false)
  let .custom ref disch := dischargeWrapper | throwError "rv_simp: expected a custom discharger"
  ref.set (← getThe Term.State)
  try
    let mvarId ← getMainGoal
    let target ← instantiateMVars (← mvarId.getType)
    let (r, _) ← Simp.mainCore target ctx {}
      (methods := Simp.mkMethods simprocs disch (wellBehavedDischarge := true))
    if r.expr == target then throwError "rv_simp made no progress"
    let newGoal ← applySimpResultToTarget mvarId target r
    if r.expr.isConstOf ``True then
      newGoal.assign (mkConst ``True.intro)
      replaceMainGoal []
    else
      replaceMainGoal [newGoal]
  finally
    set (← ref.get)

/-- `rv_split` turns a goal `if c then p else q` into two goals, with an
(inaccessible) hypothesis `c`, resp. `¬c`. It builds the `ite_goal` term
directly: elaborating it against the goal makes Lean's unifier unfold
`Nat.ble` in unary when `c` contains a large literal (such as the immediate
of `addi a2, a2, -80`), and `split` rewrites the whole goal. -/
elab "rv_split" : tactic => withMainContext do
  let goal ← getMainGoal
  let tgt ← instantiateMVars (← goal.getType)
  let_expr ite _ c h p q := tgt | throwError "rv_split: goal is not an `if`"
  let m₁ ← mkFreshExprSyntheticOpaqueMVar (← mkArrow c p)
  let m₂ ← mkFreshExprSyntheticOpaqueMVar (← mkArrow (mkNot c) q)
  goal.assign (mkAppN (mkConst ``ite_goal) #[c, h, p, q, m₁, m₂])
  let (_, g₁) ← m₁.mvarId!.intro1
  let (_, g₂) ← m₂.mvarId!.intro1
  replaceMainGoal [g₁, g₂]

/-- After evaluating a branch, split on its condition and drop the branches
that `omega` refutes (from the branch condition and the Nat facts in context).
Only the branch hypothesis is normalized: rewriting every hypothesis into
mod-2^64 arithmetic can make `omega` blow up on large contexts. -/
macro "rv_branch" : tactic =>
  `(tactic| (rv_split <;> try (exfalso; rename_i hbr; (try rv_arith_simp at hbr); omega)))

/-- One symbolic step: fetch, evaluate with `rv_exec` (plus extra lemmas), and
split on an undecided branch condition. -/
syntax "rv_step " term:max (" [" term,* "]")? : tactic
macro_rules
  | `(tactic| rv_step $hc) => `(tactic| rv_step $hc [])
  | `(tactic| rv_step $hc [$ls,*]) =>
    `(tactic| (rv_fetch $hc; rv_simp [$ls,*]; try rv_branch))

/-- One symbolic step through an instruction whose fetch is given by `h`
(used at relocation sites). -/
syntax "rv_step_at " term:max (" [" term,* "]")? : tactic
macro_rules
  | `(tactic| rv_step_at $h) => `(tactic| rv_step_at $h [])
  | `(tactic| rv_step_at $h [$ls,*]) =>
    `(tactic| (refine Exits.step_fetch $h (by rv_region) ?_
               rv_simp [$ls,*]
               try rv_branch))

/-- `rv_guard_not_at hc off` fails iff the goal's pc is `base + off`, where
`hc : CodeAt cfg base listing sites`. -/
elab "rv_guard_not_at " hc:term:max off:num : tactic => withMainContext do
  let hcTy ← whnfR (← instantiateMVars (← inferType (← elabTerm hc none)))
  let_expr CodeAt _ base _ _ := hcTy | throwError "rv_guard_not_at: {hc} is not a `CodeAt` fact"
  let_expr Exits _ _ _ s := ← instantiateMVars (← getMainTarget) | return
  let_expr State.mk pc _ _ := s.consumeMData | return
  if (← observing? (offsetFrom pc base)) == some off.getNat then
    throwError "rv_guard_not_at: pc is at offset {off.getNat}"

/-- Step until the pc reaches `base + off`, on every branch. -/
syntax "rv_steps_to " term:max num (" [" term,* "]")? : tactic
macro_rules
  | `(tactic| rv_steps_to $hc $off) => `(tactic| rv_steps_to $hc $off [])
  | `(tactic| rv_steps_to $hc $off [$ls,*]) =>
    `(tactic| repeat (rv_guard_not_at $hc $off; rv_step $hc [$ls,*]))

/-- Close `∀ r, r ∉ clobbered → regs r = regs₀ r` (or the same with
`State.reg`) for a register file that is a chain of writes to clobbered
registers on top of `regs₀`, or, given `rv_preserved hpres`, on top of a
register file for which `hpres` proves the same. -/
syntax "rv_preserved" (ppSpace colGt term:max)? : tactic
macro_rules
  | `(tactic| rv_preserved) => `(tactic| (
    intro r hr
    simp only [clobbered, List.mem_cons, List.not_mem_nil, or_false, not_or] at hr
    simp only [State.reg_mk, writeReg_apply, hr, ite_false]))
  | `(tactic| rv_preserved $hpres) => `(tactic| (
    intro r hr
    have hr' := hr
    simp only [clobbered, List.mem_cons, List.not_mem_nil, or_false, not_or] at hr'
    simp only [writeReg_apply, hr', ite_false]
    exact $hpres r hr))

end CopyRuntime.Tactic
