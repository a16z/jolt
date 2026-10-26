import CopyRuntime.Proof.Leaf.Forward
import CopyRuntime.Proof.Leaf.Backward

/-!
# The misaligned-copy leaf `__jolt_copy_misaligned`

Entered from either wrapper (by `j __jolt_copy_misaligned`) for mismatched
offsets and `n ≥ 544`. `mv a7, a0; bltu a1, a0` selects the backward copy for
`src < dst` (`Leaf/Backward.lean`) and the forward copy otherwise
(`Leaf/Forward.lean`; mismatched offsets rule out `src = dst`).

Each direction peels bytes until `a7` (the destination cursor) is 8-byte
aligned, copies eight more bytes if the first aligned source word would leave
`[src, src + n)`, then combines pairs of aligned source words with
`srl`/`sll`/`or` into aligned destination words (a 128-byte loop and a word
loop), and finishes with a byte tail. The code facts, the region, and the
word-combining identity `combine_byte` are in `Leaf/Common.lean`; the
combine-and-store step (`fwd_comb`, `bwd_comb`) is in `Leaf/Comb.lean`.
-/

namespace CopyRuntime.Leaf

/-- Correctness of the leaf, from its entry point: it performs the memmove and
returns. -/
theorem leaf_path {cfg : Config} {L : BitVec 64} (ctx : Ctx cfg L) (regs : Reg → BitVec 64)
    (mem : Mem) {dst src n R : BitVec 64} (he : EntryRegs regs dst src n R)
    (hsb : Sandbox cfg dst src n) (hargs : ArgsOk dst src n R) (hRout : ¬ region L R)
    (hmis : Mismatched dst src) (h544 : 544 ≤ n.toNat) :
    Exits cfg (region L) (Returned ⟨L, regs, mem⟩ R dst src n) ⟨L, regs, mem⟩ := by
  have hmis' : dst.toNat % 8 ≠ src.toNat % 8 := by
    unfold Mismatched at hmis
    rw [BitVec.toNat_sub] at hmis
    have := dst.isLt; have := src.isLt
    omega
  have env : Env cfg L dst src n R := ⟨ctx, hsb, hargs, hRout, hmis', h544⟩
  have hB := ctx.base_even
  have ha0 := he.a0; have ha1 := he.a1
  rv_step ctx.code [ha0, BitVec.zero_add]                 -- 0x0 mv a7, a0
  rv_step ctx.code [ha0, ha1]                             -- 0x2 bltu a1, a0
  · rename_i hlt
    exact bwd_path env hlt regs mem he _ (by simp [writeReg_apply, ha1])
      (by simp [writeReg_apply]) (by simp [writeReg_apply, he.a2])
      (by rv_preserved)
  · rename_i hge
    have hlt : dst.toNat < src.toNat := by
      have : dst.toNat ≠ src.toNat := fun h => hmis' (by rw [h])
      omega
    exact fwd_path env hlt regs mem he _ (by simp [writeReg_apply, ha1])
      (by simp [writeReg_apply]) (by simp [writeReg_apply, he.a2])
      (by rv_preserved)

end CopyRuntime.Leaf
