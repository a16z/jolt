import CopyRuntime.Proof.Region

/-!
# Common setup for both wrappers

`__wrap_memcpy` and `__wrap_memmove` have byte-identical code and differ only
in the symbol of the fallback relocation. All path proofs are done once, for
any section `W` with `IsWrapper W fallback`, against the context `Wrapper.Ctx`
which packages what the stepping tactic needs.
-/

namespace CopyRuntime.Wrapper

/-- The machine code shared by `__wrap_memcpy` and `__wrap_memmove`. -/
structure IsWrapper (W : Section) (fallback : String) : Prop where
  bytes : W.bytes = Generated.wrapMemcpy.bytes
  relocs : W.relocs =
    [{ offset := 0x578, kind := .jal, symbol := "__jolt_copy_misaligned", addend := 0 },
     { offset := 0x57c, kind := .callPlt, symbol := fallback, addend := 0 }]

theorem isWrapper_memcpy : IsWrapper Generated.wrapMemcpy "__real_memcpy" := ⟨rfl, rfl⟩

theorem isWrapper_memmove : IsWrapper Generated.wrapMemmove "__real_memmove" :=
  ⟨by decide +kernel, rfl⟩

theorem size_eq : Generated.wrapMemcpy.bytes.length = 1412 := by decide +kernel

abbrev region (B : BitVec 64) : BitVec 64 → Prop := codeRegion B 1412

/-- Everything the wrapper path proofs need about the loaded, linked code. -/
structure Ctx (cfg : Config) (B L F : BitVec 64) : Prop where
  code : CodeAt cfg B Generated.Disasm.wrapMemcpy [0x578, 0x57c, 0x580]
  leafJal : cfg.fetch (B + 0x578#64) = some (.jal 0 (L - (B + 0x578#64)), 4)
  fallbackAuipc : cfg.fetch (B + 0x57c#64) =
    some (.auipc 6 (F - (B + 0x57c#64) - lo12 (F - (B + 0x57c#64))), 4)
  fallbackJalr : cfg.fetch (B + 0x580#64) = some (.jalr 0 6 (lo12 (F - (B + 0x57c#64))), 4)
  leaf_outside : ¬ region B L
  fallback_outside : ¬ region B F
  fallback_even : F.toNat % 2 = 0
  leaf_even : L.toNat % 2 = 0
  base_even : B.toNat % 2 = 0

theorem fetch_eq_linked {cfg : Config} {B : BitVec 64} {W : Section} (hL : cfg.Loaded B W)
    (hB : B.toNat % 2 = 0) {k : Nat} (hk : k < W.bytes.length) (hk2 : k % 2 = 0)
    (hsize : W.bytes.length < 2 ^ 64) :
    cfg.fetch (B + BitVec.ofNat 64 k) = linkedInstr ⟨B, W⟩ cfg.resolve k := by
  have hoff := Arith.toNat_add_lit_sub B (k := k) (by omega)
  have hsec : cfg.sectionAt (B + BitVec.ofNat 64 k) = some ⟨B, W⟩ :=
    hL _ (by unfold inSection; rw [hoff]; exact hk)
  have hpar : (B + BitVec.ofNat 64 k).toNat % 2 = 0 := by
    rw [toNat_add_lit_mod_two hB]; exact hk2
  simp only [Config.fetch, hsec, hpar, Option.bind_eq_bind, Option.bind_some, ne_eq,
    not_true_eq_false, ite_false, hoff]

theorem linkedInstr_jal {ls : LoadedSection} {res : String → BitVec 64} {off : Nat}
    {r : Reloc} {rd : Reg} {imm₀ : BitVec 64}
    (hdec : Decode.decodeAt ls.sec.bytes off = some (.jal rd imm₀, 4))
    (hfilter : ls.sec.relocs.filter (·.covers off) = [r])
    (hkind : r.kind = .jal) (hoff : r.offset = off)
    (hrange : jalInRange (relocValue ls res r) = true) :
    linkedInstr ls res off = some (.jal rd (relocValue ls res r), 4) := by
  simp only [linkedInstr, hdec, hfilter, Option.bind_eq_bind, Option.bind_some, hkind]
  simp [hoff, hrange]

theorem linkedInstr_auipc {ls : LoadedSection} {res : String → BitVec 64} {off : Nat}
    {r : Reloc} {rd : Reg} {imm₀ : BitVec 64}
    (hdec : Decode.decodeAt ls.sec.bytes off = some (.auipc rd imm₀, 4))
    (hfilter : ls.sec.relocs.filter (·.covers off) = [r])
    (hkind : r.kind = .callPlt) (hoff : r.offset = off)
    (hrange : callInRange (relocValue ls res r) = true) :
    linkedInstr ls res off =
      some (.auipc rd (relocValue ls res r - lo12 (relocValue ls res r)), 4) := by
  simp only [linkedInstr, hdec, hfilter, Option.bind_eq_bind, Option.bind_some, hkind]
  simp [hoff, hrange]

theorem linkedInstr_jalr {ls : LoadedSection} {res : String → BitVec 64} {off : Nat}
    {r : Reloc} {rd rs1 : Reg} {imm₀ : BitVec 64}
    (hdec : Decode.decodeAt ls.sec.bytes off = some (.jalr rd rs1 imm₀, 4))
    (hfilter : ls.sec.relocs.filter (·.covers off) = [r])
    (hkind : r.kind = .callPlt) (hoff : r.offset + 4 = off)
    (hrange : callInRange (relocValue ls res r) = true) :
    linkedInstr ls res off = some (.jalr rd rs1 (lo12 (relocValue ls res r)), 4) := by
  simp only [linkedInstr, hdec, hfilter, Option.bind_eq_bind, Option.bind_some, hkind]
  simp [hoff, hrange]

theorem Ctx.of {cfg : Config} {B L F : BitVec 64} {W : Section} {fallback : String}
    (hL : cfg.Loaded B W) (hW : IsWrapper W fallback) (hB : B.toNat % 2 = 0)
    (hleaf : cfg.resolve "__jolt_copy_misaligned" = L) (hfb : cfg.resolve fallback = F)
    (hrel : RelocsInRange cfg B W) (hex : ExitTargetsOk B W L F) : Ctx cfg B L F := by
  have hlen : W.bytes.length = 1412 := by rw [hW.bytes, size_eq]
  have hreg : ∀ x, inSection B W x ↔ region B x := inSection_iff_codeRegion hlen
  have hrel1 := hrel
    { offset := 0x578, kind := .jal, symbol := "__jolt_copy_misaligned", addend := 0 }
    (by rw [hW.relocs]; simp)
  have hrel2 := hrel { offset := 0x57c, kind := .callPlt, symbol := fallback, addend := 0 }
    (by rw [hW.relocs]; simp)
  have h0 : BitVec.ofInt 64 0 = 0#64 := rfl
  simp only [relocValue, hleaf, hfb, h0, BitVec.add_zero] at hrel1 hrel2
  have hLeven : L.toNat % 2 = 0 := by
    simp only [jalInRange, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hrel1
    have e : L = B + 0x578#64 + (L - (B + 0x578#64)) := by bv_omega
    rw [e]
    have := hrel1.2
    simp only [BitVec.toNat_add, BitVec.toNat_ofNat]
    omega
  refine ⟨?_, ?_, ?_, ?_, fun h => hex.leaf_outside ((hreg _).2 h),
    fun h => hex.fallback_outside ((hreg _).2 h), hex.fallback_even, hLeven, hB⟩
  · refine CodeAt.of_loaded hL hB (by rw [hlen]; decide) ?_ ?_
    · rw [hW.bytes]; exact DecoderCheck.wrapMemcpy_agrees_with_llvm
    · intro k hk
      rw [hW.relocs]
      simp only [List.mem_cons, List.not_mem_nil, or_false, not_or] at hk
      simp [Reloc.covers]
      omega
  · rw [show (0x578#64 : BitVec 64) = BitVec.ofNat 64 0x578 from rfl,
      fetch_eq_linked hL hB (by rw [hlen]; decide) (by decide) (by rw [hlen]; decide)]
    have hdec : Decode.decodeAt W.bytes 0x578 = some (.jal 0 (imm 0), 4) := by
      rw [hW.bytes]; decide +kernel
    rw [linkedInstr_jal (ls := ⟨B, W⟩)
      (r := { offset := 0x578, kind := .jal, symbol := "__jolt_copy_misaligned", addend := 0 })
      hdec (by rw [hW.relocs]; simp [Reloc.covers]) rfl rfl
      (by simpa [relocValue, hleaf] using hrel1)]
    simp [relocValue, hleaf]
  · rw [show (0x57c#64 : BitVec 64) = BitVec.ofNat 64 0x57c from rfl,
      fetch_eq_linked hL hB (by rw [hlen]; decide) (by decide) (by rw [hlen]; decide)]
    have hdec : Decode.decodeAt W.bytes 0x57c = some (.auipc 6 (imm 0), 4) := by
      rw [hW.bytes]; decide +kernel
    rw [linkedInstr_auipc (ls := ⟨B, W⟩)
      (r := { offset := 0x57c, kind := .callPlt, symbol := fallback, addend := 0 })
      hdec (by rw [hW.relocs]; simp [Reloc.covers]) rfl rfl
      (by simpa [relocValue, hfb] using hrel2)]
    simp [relocValue, hfb]
  · rw [show (0x580#64 : BitVec 64) = BitVec.ofNat 64 0x580 from rfl,
      fetch_eq_linked hL hB (by rw [hlen]; decide) (by decide) (by rw [hlen]; decide)]
    have hdec : Decode.decodeAt W.bytes 0x580 = some (.jalr 0 6 (imm 0), 4) := by
      rw [hW.bytes]; decide +kernel
    rw [linkedInstr_jalr (ls := ⟨B, W⟩)
      (r := { offset := 0x57c, kind := .callPlt, symbol := fallback, addend := 0 })
      hdec (by rw [hW.relocs]; simp [Reloc.covers]) rfl rfl
      (by simpa [relocValue, hfb] using hrel2)]
    simp [relocValue, hfb]

end CopyRuntime.Wrapper
