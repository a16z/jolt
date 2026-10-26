import CopyRuntime.Proof.Returned
import CopyRuntime.Proof.Wrapper.Facts

/-!
# Wrapper paths that leave without copying

* `n = 0`: return immediately;
* mismatched offsets and `12 ≤ n < 544`: tail call to the fallback;
* mismatched offsets and `n ≥ 544`: jump to the misaligned leaf.
-/

namespace CopyRuntime.Wrapper

variable {cfg : Config} {B L F : BitVec 64}

theorem zero_path (ctx : Ctx cfg B L F) (regs : Reg → BitVec 64) (mem : Mem)
    {dst src R : BitVec 64} (he : EntryRegs regs dst src 0 R) (hR : R.toNat % 2 = 0)
    (hRout : ¬ region B R) :
    Exits cfg (region B) (Returned ⟨B, regs, mem⟩ R dst src 0) ⟨B, regs, mem⟩ := by
  have hB := ctx.base_even
  rv_step ctx.code [he.a2]
  rv_step ctx.code [he.ra, hR]
  exact .exit hRout ⟨rfl, by simp [State.reg_mk, he.a0], (memmoveResult_zero _ _ _).symm,
    fun _ _ => rfl⟩

theorem fallback_path (ctx : Ctx cfg B L F) (regs : Reg → BitVec 64) (mem : Mem)
    {dst src n R : BitVec 64} (he : EntryRegs regs dst src n R)
    (hmis : Mismatched dst src) (h12 : 12 ≤ n.toNat) (h544 : n.toNat < 544) :
    Exits cfg (region B) (AtFallback ⟨B, regs, mem⟩ F) ⟨B, regs, mem⟩ := by
  have hB := ctx.base_even
  have hF := ctx.fallback_even
  have ha0 := he.a0; have ha1 := he.a1; have ha2 := he.a2
  rv_step ctx.code [ha0, ha1, ha2]                     -- 0x000 c.beqz a2
  rv_step ctx.code [ha0, ha1, ha2]                     -- 0x002 or t0, a0, a1
  rv_step ctx.code [ha0, ha1, ha2]                     -- 0x006 or t0, t0, a2
  rv_step ctx.code [ha0, ha1, ha2]                     -- 0x00a andi t0, t0, 7
  rv_step ctx.code [or_and_seven_ne_zero_of_mismatched hmis n]  -- 0x00e bnez t0
  rv_step ctx.code [ha2]                               -- 0x092 c.li t0, 7
  rv_step ctx.code [ha2]                               -- 0x094 bgeu t0, a2
  rv_step ctx.code [ha0, ha1]                          -- 0x098 xor t0, a0, a1
  rv_step ctx.code                                     -- 0x09c andi t0, t0, 7
  rv_step ctx.code [(mismatched_iff dst src).1 hmis]   -- 0x0a0 bnez t0
  rv_step ctx.code                                     -- 0x56a c.li t0, 11
  rv_step ctx.code [ha2]                               -- 0x56c bgeu t0, a2
  rv_step ctx.code                                     -- 0x570 li t0, 544
  rv_step ctx.code [ha2]                               -- 0x574 bltu a2, t0
  rv_step_at ctx.fallbackAuipc                         -- 0x57c auipc t1 (relocated)
  rv_step_at ctx.fallbackJalr                          -- 0x580 jr t1 (relocated)
  refine .exit ctx.fallback_outside ⟨rfl, rfl, fun r h5 h6 => ?_⟩
  simp only [State.reg_mk, writeReg_apply, h5, h6, ite_false]

theorem leaf_path (ctx : Ctx cfg B L F) (regs : Reg → BitVec 64) (mem : Mem)
    {dst src n R : BitVec 64} (he : EntryRegs regs dst src n R)
    (hmis : Mismatched dst src) (h544 : 544 ≤ n.toNat) :
    Exits cfg (region B) (AtLeaf ⟨B, regs, mem⟩ L) ⟨B, regs, mem⟩ := by
  have hB := ctx.base_even
  have hL := ctx.leaf_even
  have ha0 := he.a0; have ha1 := he.a1; have ha2 := he.a2
  rv_step ctx.code [ha2]                               -- 0x000 c.beqz a2
  rv_step ctx.code [ha0, ha1]                          -- 0x002 or t0, a0, a1
  rv_step ctx.code [ha2]                               -- 0x006 or t0, t0, a2
  rv_step ctx.code                                     -- 0x00a andi t0, t0, 7
  rv_step ctx.code [or_and_seven_ne_zero_of_mismatched hmis n]  -- 0x00e bnez t0
  rv_step ctx.code                                     -- 0x092 c.li t0, 7
  rv_step ctx.code [ha2]                               -- 0x094 bgeu t0, a2
  rv_step ctx.code [ha0, ha1]                          -- 0x098 xor t0, a0, a1
  rv_step ctx.code                                     -- 0x09c andi t0, t0, 7
  rv_step ctx.code [(mismatched_iff dst src).1 hmis]   -- 0x0a0 bnez t0
  rv_step ctx.code                                     -- 0x56a c.li t0, 11
  rv_step ctx.code [ha2]                               -- 0x56c bgeu t0, a2
  rv_step ctx.code                                     -- 0x570 li t0, 544
  rv_step ctx.code [ha2]                               -- 0x574 bltu a2, t0
  rv_step_at ctx.leafJal                               -- 0x578 j __jolt_copy_misaligned
  refine .exit ctx.leaf_outside ⟨rfl, rfl, fun r h5 => ?_⟩
  simp only [State.reg_mk, writeReg_apply, h5, ite_false]

end CopyRuntime.Wrapper
