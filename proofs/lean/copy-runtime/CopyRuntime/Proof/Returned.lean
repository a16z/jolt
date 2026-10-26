import CopyRuntime.Proof.ExecLemmas

namespace CopyRuntime

structure EntryRegs (regs : Reg → BitVec 64) (dst src n R : BitVec 64) : Prop where
  a0 : regs 10 = dst
  a1 : regs 11 = src
  a2 : regs 12 = n
  ra : regs 1 = R

/-- The final state of every copying path: back at `R`, with `[0, n)` copied
and every preserved register equal to its entry value. -/
theorem Returned.of_copied {regs₀ regs : Reg → BitVec 64} {m₀ mem : Mem}
    {p dst src n R : BitVec 64} (he : EntryRegs regs₀ dst src n R)
    (hdst : dst.toNat + n.toNat < 2 ^ 64) (hcop : Copied mem m₀ dst src 0 n.toNat)
    (hpres : ∀ r, r ∉ clobbered → regs r = regs₀ r) :
    Returned ⟨p, regs₀, m₀⟩ R dst src n ⟨R, regs, mem⟩ :=
  ⟨rfl, by simp [State.reg_mk, hpres 10 (by decide), he.a0], hcop.memmove (Nat.le_of_lt hdst),
    fun r hr => by simp [State.reg_mk, hpres r hr]⟩

/-- Transfer `Returned` from an intermediate state that agrees with the entry
state on memory and on every preserved register. -/
theorem Returned.of_agree {s₀ s₁ t : State} {R dst src n : BitVec 64}
    (h : Returned s₁ R dst src n t) (hmem : s₁.mem = s₀.mem)
    (hregs : ∀ r, r ∉ clobbered → s₁.reg r = s₀.reg r) : Returned s₀ R dst src n t :=
  ⟨h.pc, h.a0, by rw [h.mem, hmem], fun r hr => (h.preserved r hr).trans (hregs r hr)⟩

/-- `Returned` does not depend on the pc of the reference state. Use this
rather than comparing two concrete states, which unifies their pcs. -/
theorem Returned.change_pc {p₀ p₁ : BitVec 64} {r : Reg → BitVec 64} {m : Mem} {t : State}
    {R dst src n : BitVec 64} (h : Returned ⟨p₁, r, m⟩ R dst src n t) :
    Returned ⟨p₀, r, m⟩ R dst src n t :=
  ⟨h.pc, h.a0, h.mem, h.preserved⟩

end CopyRuntime
