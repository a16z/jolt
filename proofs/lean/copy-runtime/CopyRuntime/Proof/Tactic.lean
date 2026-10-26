import Lean
import CopyRuntime.Proof.ExecLemmas
import CopyRuntime.Proof.Memory

/-!
# `rv_step`: symbolic execution of one listed instruction

`rv_step hc` expects a goal `Exits cfg region post ⟨pc, regs, mem⟩` with
`pc = base + c` (or `pc = base`) and `hc : CodeAt cfg base listing sites`. It
looks `c` up in `listing` (an LLVM listing constant, see `DecoderCheck`),
applies `CodeAt.step`, proves listing membership by kernel evaluation
(`decide +kernel`), proves the region side condition with `rv_region`, and
leaves the goal `Exits.after cfg region post (execute cfg ⟨pc, regs, mem⟩ i len)`.
The tactic computes nothing that the kernel does not re-check: the lookup only
chooses which instruction to claim.

`rv_step hc [lemmas]` additionally runs `simp only [rv_exec, lemmas]` on that
goal, which evaluates the instruction.
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

/-- Find the `(instruction, length)` listed at offset `k`. -/
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

/-- Apply `CodeAt.step` at the current pc; returns the remaining goal. -/
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
  -- goals: hk, hpc, hreg, h
  let [hk, hpc, hreg, h] := newGoals |
    throwError "rv_step: unexpected goals after applying CodeAt.step"
  let run (g : MVarId) (tac : TSyntax `tactic) : TacticM Unit := do
    let gs ← Tactic.run g (evalTactic tac)
    unless gs.isEmpty do throwError "rv_step: side condition not closed: {gs}"
  run hk (← `(tactic| decide +kernel))
  run hpc (← `(tactic| first | with_reducible rfl | simp only [BitVec.add_zero]))
  run hreg (← `(tactic| rv_region))
  replaceMainGoal [h]

elab "rv_fetch " hc:term : tactic => do
  let hc ← elabTerm hc none
  stepCore hc

/-- Discharger for the conditional rewrite rules of `rv_exec`. -/
macro "rv_disch" : tactic =>
  -- The trailing `fail` keeps a failed `bv_omega` from running in error-recovery
  -- mode (which would log an error and admit the side goal).
  `(tactic| first | assumption | decide | bv_omega | fail "rv_disch")

/-- After evaluating a branch, split on its condition and drop the branches
that `bv_omega` refutes from the hypotheses. -/
macro "rv_branch" : tactic =>
  `(tactic| split <;> try (exfalso; bv_omega))

/-- One symbolic step: fetch, evaluate with `rv_exec` (plus extra lemmas), and
split on an undecided branch condition. -/
syntax "rv_step " term:max (" [" term,* "]")? : tactic
macro_rules
  | `(tactic| rv_step $hc) => `(tactic| rv_step $hc [])
  | `(tactic| rv_step $hc [$ls,*]) => do
    let args ← ls.getElems.mapM fun t => `(Lean.Parser.Tactic.simpLemma| $t:term)
    `(tactic| (rv_fetch $hc; simp (disch := rv_disch) only [rv_exec, $args,*]; try rv_branch))

/-- One symbolic step through an instruction whose fetch is given by `h`
(used at relocation sites). -/
syntax "rv_step_at " term:max (" [" term,* "]")? : tactic
macro_rules
  | `(tactic| rv_step_at $h) => `(tactic| rv_step_at $h [])
  | `(tactic| rv_step_at $h [$ls,*]) => do
    let args ← ls.getElems.mapM fun t => `(Lean.Parser.Tactic.simpLemma| $t:term)
    `(tactic| (refine Exits.step_fetch $h (by rv_region) ?_
               simp (disch := rv_disch) only [rv_exec, $args,*]
               try rv_branch))

end CopyRuntime.Tactic
