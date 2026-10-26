import CopyRuntime.Spec
import CopyRuntime.DecoderCheck
import CopyRuntime.Proof.Arith

/-!
# Generic facts about `Exits`, and the `CodeAt` interface used by the stepping
tactic.
-/

namespace CopyRuntime

/-! ## Determinism and composition -/

/-- Runs are deterministic: two `Exits` facts about the same start state
describe the same exit state. -/
theorem Exits.unique {cfg : Config} {region : BitVec 64 → Prop} {P Q : State → Prop}
    {s : State} (hP : Exits cfg region P s) (hQ : Exits cfg region Q s) :
    ∃ t, P t ∧ Q t := by
  induction hP with
  | exit hout hp =>
    cases hQ with
    | exit _ hq => exact ⟨_, hp, hq⟩
    | step hin _ _ => exact absurd hin hout
  | step hin hstep _ ih =>
    cases hQ with
    | exit hout _ => exact absurd hin hout
    | step _ hstep' hrest =>
      rw [hstep] at hstep'
      cases hstep'
      exact ih hrest

theorem Exits.mono {cfg : Config} {region : BitVec 64 → Prop} {P Q : State → Prop}
    {s : State} (h : Exits cfg region P s) (hPQ : ∀ t, P t → Q t) :
    Exits cfg region Q s := by
  induction h with
  | exit hout hp => exact .exit hout (hPQ _ hp)
  | step hin hstep _ ih => exact .step hin hstep ih

/-- Continue a run that left a smaller region inside a larger one. -/
theorem Exits.bind {cfg : Config} {r₁ r : BitVec 64 → Prop} {P Q : State → Prop}
    {s : State} (h : Exits cfg r₁ P s) (hsub : ∀ x, r₁ x → r x)
    (hk : ∀ t, P t → Exits cfg r Q t) : Exits cfg r Q s := by
  induction h with
  | exit _ hp => exact hk _ hp
  | step hin hstep _ ih => exact .step (hsub _ hin) hstep ih

/-- A run that ends outside a larger region is also a run of that region. -/
theorem Exits.widen {cfg : Config} {r₁ r : BitVec 64 → Prop} {Q : State → Prop}
    {s : State} (h : Exits cfg r₁ Q s) (hsub : ∀ x, r₁ x → r x)
    (hout : ∀ t, Q t → ¬ r t.pc) : Exits cfg r Q s :=
  h.bind hsub fun t hq => .exit (hout t hq) hq

/-- Change the region to an equivalent one. -/
theorem Exits.congr_region {cfg : Config} {r₁ r₂ : BitVec 64 → Prop} {Q : State → Prop}
    {s : State} (h : Exits cfg r₁ Q s) (heq : ∀ x, r₁ x ↔ r₂ x) : Exits cfg r₂ Q s := by
  induction h with
  | exit hout hq => exact .exit (fun h => hout ((heq _).2 h)) hq
  | step hin hstep _ ih => exact .step ((heq _).1 hin) hstep ih

/-! ## Linear sweep facts -/

theorem decodeBytes_len {bs : List UInt8} {i : Instr} {len : Nat}
    (h : Decode.decodeBytes bs = some (i, len)) :
    (len = 2 ∨ len = 4) ∧ len ≤ bs.length := by
  unfold Decode.decodeBytes at h
  match bs, h with
  | b0 :: b1 :: rest, h =>
    simp only at h
    split at h
    · simp only [Option.map_eq_some_iff, Prod.mk.injEq] at h
      obtain ⟨_, _, _, rfl⟩ := h
      simp
    · split at h
      · simp at h
      · split at h
        · simp only [Option.map_eq_some_iff, Prod.mk.injEq] at h
          obtain ⟨_, _, _, rfl⟩ := h
          simp
        · simp at h

theorem sweep_go_spec : ∀ (fuel off : Nat) (bs : List UInt8) (tbl : List (Nat × Instr × Nat)),
    sweep.go fuel off bs = some tbl →
    ∀ k i len, (k, i, len) ∈ tbl →
      off ≤ k ∧ Decode.decodeBytes (bs.drop (k - off)) = some (i, len) ∧
      k + len ≤ off + bs.length ∧ (k - off) % 2 = 0
  | _, _, [], tbl, h => by
    simp only [sweep.go, Option.some.injEq] at h
    subst h
    simp
  | 0, _, _ :: _, _, h => by simp [sweep.go] at h
  | fuel + 1, off, b :: bs, tbl, h => by
    simp only [sweep.go] at h
    split at h
    · simp at h
    · rename_i i₀ len₀ hdec
      simp only [Option.map_eq_some_iff] at h
      obtain ⟨rest, hrest, rfl⟩ := h
      have hlen := decodeBytes_len hdec
      intro k i len hmem
      simp only [List.mem_cons, Prod.mk.injEq] at hmem
      rcases hmem with ⟨rfl, rfl, rfl⟩ | hmem
      · refine ⟨Nat.le_refl _, by simpa using hdec, ?_, by simp⟩
        simp only [List.length_cons] at hlen ⊢
        omega
      · obtain ⟨hle, hd, hb, hev⟩ := sweep_go_spec fuel (off + len₀) _ rest hrest k i len hmem
        simp only [List.drop_drop, List.length_drop, List.length_cons] at hd hb hlen ⊢
        refine ⟨by omega, ?_, by omega, by omega⟩
        have : len₀ + (k - (off + len₀)) = k - off := by omega
        rw [this] at hd
        exact hd

theorem sweep_spec {bytes : List UInt8} {tbl : List (Nat × Instr × Nat)}
    (h : sweep bytes = some tbl) {k : Nat} {i : Instr} {len : Nat} (hmem : (k, i, len) ∈ tbl) :
    Decode.decodeAt bytes k = some (i, len) ∧ k + len ≤ bytes.length ∧ k % 2 = 0 := by
  obtain ⟨_, hd, hb, hev⟩ := sweep_go_spec _ 0 bytes tbl h k i len hmem
  simp only [Nat.sub_zero, Nat.zero_add] at hd hb hev
  exact ⟨hd, hb, hev⟩

/-! ## Code facts used by the stepping tactic -/

/-- The code at `base`: every instruction of `listing` (an LLVM listing whose
agreement with the decoder is proved in `DecoderCheck`) is what `cfg.fetch`
returns at `base + offset`, except at relocation `sites`, which have their own
lemmas. -/
structure CodeAt (cfg : Config) (base : BitVec 64) (listing : List (Nat × Instr × Nat))
    (sites : List Nat) : Prop where
  base_even : base.toNat % 2 = 0
  fetch : ∀ k i len, (k, i, len) ∈ listing → k ∉ sites →
    cfg.fetch (base + BitVec.ofNat 64 k) = some (i, len)

theorem CodeAt.of_loaded {cfg : Config} {base : BitVec 64} {sec : Section}
    {listing : List (Nat × Instr × Nat)} {sites : List Nat}
    (hL : cfg.Loaded base sec) (hB : base.toNat % 2 = 0) (hsize : sec.bytes.length < 2 ^ 64)
    (hsweep : sweep sec.bytes = some listing)
    (hrel : ∀ k, k ∉ sites → sec.relocs.filter (·.covers k) = []) :
    CodeAt cfg base listing sites := by
  refine ⟨hB, fun k i len hmem hsite => ?_⟩
  obtain ⟨hdec, hb, hev⟩ := sweep_spec hsweep hmem
  have hlen := (decodeBytes_len hdec).1
  have hk : k < 2 ^ 64 := by omega
  have hoff := Arith.toNat_add_lit_sub base hk
  have hsec : cfg.sectionAt (base + BitVec.ofNat 64 k) = some ⟨base, sec⟩ :=
    hL _ (by unfold inSection; rw [hoff]; omega)
  have hpar : (base + BitVec.ofNat 64 k).toNat % 2 = 0 := by
    rw [BitVec.toNat_add, BitVec.toNat_ofNat, Nat.mod_eq_of_lt hk]
    omega
  simp only [Config.fetch, hsec, hpar, Option.bind_eq_bind, Option.bind_some, ne_eq,
    not_true_eq_false, ite_false, hoff, linkedInstr, hdec, hrel k hsite]
  rfl

end CopyRuntime
