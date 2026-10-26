import CopyRuntime.Proof.Step
import CopyRuntime.Proof.SimpAttr
import CopyRuntime.Proof.Memory

/-!
# Rewrite rules that evaluate `execute` on a symbolic state `State.mk pc regs mem`

States are kept in the form `⟨pc, regs, mem⟩` where `regs` is a chain of
`writeReg` and `mem` a chain of `storeLE` applied to the entry values.
Addresses are kept in the form `x + c` with a literal `c`.
-/

namespace CopyRuntime

attribute [rv_exec] execute BranchCond.eval AluImmOp.eval AluOp.eval Width.bytes imm
  Exits.after_some Exits.after_none Exits.after_ite
  Arith.add_lit_lit Arith.add_lit_sub_lit Arith.add_sub_cancel_left Arith.add_sub_add_left
  BitVec.add_zero
  ite_true ite_false eq_self_iff_true decide_true decide_false Bool.true_and Bool.and_true
  Bool.false_and Bool.and_false Bool.not_true Bool.not_false beq_self_eq_true
  Bool.true_or Bool.or_true Bool.false_or Bool.or_false bne_self_eq_false
  Option.some.injEq beq_iff_eq bne_iff_ne decide_eq_true_eq ne_eq not_false_eq_true
  not_true_eq_false accessOk_rangePerm Bool.false_eq_true Bool.true_eq_false

attribute [rv_exec_proc] reduceIte reduceDIte reduceCtorEq Fin.reduceEq Fin.reduceNe Fin.isValue
  BitVec.reduceOfInt BitVec.reduceOfNat BitVec.reduceAdd BitVec.reduceSub BitVec.reduceMul
  BitVec.reduceAnd BitVec.reduceOr BitVec.reduceXOr BitVec.reduceNot BitVec.reduceNeg
  BitVec.reduceHShiftLeft BitVec.reduceHShiftRight BitVec.reduceShiftLeft
  BitVec.reduceUShiftRight BitVec.reduceToNat BitVec.reduceEq BitVec.reduceNe
  BitVec.reduceBEq BitVec.reduceBNe BitVec.reduceULT BitVec.reduceULE
  Nat.reduceAdd Nat.reduceSub Nat.reduceMul Nat.reduceDiv Nat.reduceMod Nat.reducePow
  Nat.reduceLeDiff Nat.reduceEqDiff Nat.reduceBEq Nat.reduceBNe Nat.reduceLT Nat.reduceGT

@[rv_exec] theorem State.setReg_mk (p : BitVec 64) (r : Reg → BitVec 64) (m : Mem) (rd : Reg)
    (v : BitVec 64) :
    (State.mk p r m).setReg rd v = if rd = 0 then State.mk p r m else State.mk p (writeReg r rd v) m :=
  rfl

@[rv_exec] theorem State.reg_mk (p : BitVec 64) (r : Reg → BitVec 64) (m : Mem) (i : Reg) :
    (State.mk p r m).reg i = if i = 0 then 0#64 else r i := rfl

@[rv_exec] theorem writeReg_apply (r : Reg → BitVec 64) (rd i : Reg) (v : BitVec 64) :
    writeReg r rd v i = if i = rd then v else r i := rfl

/-- Collapse two adjacent writes to the same register. (Sorting writes by
register number inside `simp` was tried and is far too slow.) -/
@[rv_exec] theorem writeReg_writeReg_same (f : Reg → BitVec 64) (r : Reg) (a b : BitVec 64) :
    writeReg (writeReg f r a) r b = writeReg f r b := by
  funext i; simp only [writeReg]; split <;> rfl

theorem writeReg_writeReg_swap (f : Reg → BitVec 64) {q r : Reg} (h : q < r)
    (a b : BitVec 64) : writeReg (writeReg f r a) q b = writeReg (writeReg f q b) r a := by
  funext i; simp only [writeReg]
  split <;> split <;> first | rfl | (subst_vars; exact absurd h (Fin.lt_irrefl _))

theorem toNat_add_lit_mod_two {B : BitVec 64} (hB : B.toNat % 2 = 0) (k : Nat) :
    (B + BitVec.ofNat 64 k).toNat % 2 = k % 2 := by
  rw [BitVec.toNat_add, BitVec.toNat_ofNat]
  omega

/-- Jumps to `B + k` with `B` even and `k` even are aligned. -/
@[rv_exec] theorem jumpTo_add_lit {B : BitVec 64} (hB : B.toNat % 2 = 0) {k : Nat} (hk : k % 2 = 0)
    (s : State) : jumpTo s (B + BitVec.ofNat 64 k) = some { s with pc := B + BitVec.ofNat 64 k } := by
  unfold jumpTo
  rw [ite_eq_left (by rw [toNat_add_lit_mod_two hB]; exact hk)]

@[rv_exec] theorem jumpTo_even {t : BitVec 64} (ht : t.toNat % 2 = 0) (s : State) :
    jumpTo s t = some { s with pc := t } := by
  simp [jumpTo, ht]

/-- `jalr` clears bit 0 of its target (`~~~1`, written as the folded literal
that the simp set produces); that is a no-op on even addresses. -/
@[rv_exec] theorem and_not_one_of_even {t : BitVec 64} (ht : t.toNat % 2 = 0) :
    t &&& 0xfffffffffffffffe#64 = t := by
  have h : t &&& 1#64 = 0#64 := by
    apply BitVec.eq_of_toNat_eq
    rw [BitVec.toNat_and]
    simp [Nat.and_one_is_mod, ht]
  bv_decide

@[rv_exec] theorem add_lit_and_not_one {B : BitVec 64} (hB : B.toNat % 2 = 0) {k : Nat}
    (hk : k % 2 = 0) :
    (B + BitVec.ofNat 64 k) &&& 0xfffffffffffffffe#64 = B + BitVec.ofNat 64 k :=
  and_not_one_of_even (by rw [toNat_add_lit_mod_two hB]; exact hk)

end CopyRuntime
