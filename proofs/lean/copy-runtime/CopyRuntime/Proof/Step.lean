import CopyRuntime.Proof.Exits

/-!
# Single-step lemmas for symbolic execution
-/

namespace CopyRuntime

/-- What remains to be shown after executing one instruction. -/
def Exits.after (cfg : Config) (region : BitVec 64 → Prop) (post : State → Prop) :
    Option State → Prop
  | some s' => Exits cfg region post s'
  | none => False

@[simp] theorem Exits.after_some {cfg : Config} {region : BitVec 64 → Prop}
    {post : State → Prop} {s : State} :
    Exits.after cfg region post (some s) = Exits cfg region post s := rfl

@[simp] theorem Exits.after_none {cfg : Config} {region : BitVec 64 → Prop}
    {post : State → Prop} : Exits.after cfg region post none = False := rfl

theorem Exits.after_ite {cfg : Config} {region : BitVec 64 → Prop} {post : State → Prop}
    {c : Prop} [Decidable c] {a b : Option State} :
    Exits.after cfg region post (if c then a else b) =
      if c then Exits.after cfg region post a else Exits.after cfg region post b := by
  split <;> rfl

/-- One step through an instruction of the listing. -/
theorem Exits.step_fetch {cfg : Config} {region : BitVec 64 → Prop} {post : State → Prop}
    {s : State} {i : Instr} {len : Nat}
    (hfetch : cfg.fetch s.pc = some (i, len)) (hreg : region s.pc)
    (h : Exits.after cfg region post (execute cfg s i len)) : Exits cfg region post s := by
  unfold Exits.after at h
  split at h
  · rename_i s' hs'
    exact .step hreg (by simp [CopyRuntime.step, hfetch, hs']) h
  · exact h.elim

theorem CodeAt.step {cfg : Config} {base : BitVec 64} {listing : List (Nat × Instr × Nat)}
    {sites : List Nat} (hc : CodeAt cfg base listing sites)
    {region : BitVec 64 → Prop} {post : State → Prop} {s : State}
    (k : Nat) {i : Instr} {len : Nat}
    (hk : (k, i, len) ∈ listing ∧ k ∉ sites)
    (hpc : s.pc = base + BitVec.ofNat 64 k) (hreg : region s.pc)
    (h : Exits.after cfg region post (execute cfg s i len)) : Exits cfg region post s :=
  Exits.step_fetch (by rw [hpc]; exact hc.fetch k i len hk.1 hk.2) hreg h

end CopyRuntime
