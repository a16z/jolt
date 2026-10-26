import CopyRuntime.Proof.Tactic

/-!
# Code regions with a literal size, and the `rv_region` side-condition tactic
-/

namespace CopyRuntime

/-- The modular address range `[base, base + size)`. -/
def codeRegion (base : BitVec 64) (size : Nat) (pc : BitVec 64) : Prop :=
  (pc - base).toNat < size

theorem codeRegion_lit {base : BitVec 64} {size k : Nat} (hk : k < size)
    (hsize : size ≤ 2 ^ 64) : codeRegion base size (base + BitVec.ofNat 64 k) := by
  unfold codeRegion
  rw [Arith.toNat_add_lit_sub base (by omega)]
  exact hk

theorem codeRegion_base {base : BitVec 64} {size : Nat} (h : 0 < size) :
    codeRegion base size base := by
  unfold codeRegion
  simpa using h

theorem inSection_iff_codeRegion {base : BitVec 64} {sec : Section} {size : Nat}
    (h : sec.bytes.length = size) (pc : BitVec 64) :
    inSection base sec pc ↔ codeRegion base size pc := by
  unfold inSection codeRegion
  rw [h]

macro_rules
  | `(tactic| rv_region) =>
    `(tactic| first
      | exact codeRegion_lit (by decide) (by decide)
      | exact codeRegion_base (by decide))

end CopyRuntime
