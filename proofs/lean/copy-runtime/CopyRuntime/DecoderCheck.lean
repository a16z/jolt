import CopyRuntime.Machine.Decode
import CopyRuntime.Generated.Object
import CopyRuntime.Generated.Disasm

/-!
# Decoder cross-check against LLVM

`sweep` decodes a section linearly from offset 0 with the trusted Lean decoder.
The theorems below state that, for each of the three extracted sections, the
result is exactly the instruction listing that `extract.py` translated from
`llvm-objdump -M no-aliases -M numeric` (`CopyRuntime.Generated.Disasm`): same
instruction boundaries, same lengths, same operations, registers and
immediates. They are checked by kernel evaluation (`decide +kernel`, no
`native_decide`).

The proofs only ever step through instructions taken from these listings
(`CopyRuntime.CodeAt.fetch` requires listing membership), so every instruction
whose semantics a correctness theorem relies on was decoded identically by the
Lean decoder and by LLVM.
-/

namespace CopyRuntime

open Decode

/-- Linear-sweep decoding of a whole section. -/
def sweep (bytes : List UInt8) : Option (List (Nat × Instr × Nat)) :=
  go bytes.length 0 bytes
where
  go : Nat → Nat → List UInt8 → Option (List (Nat × Instr × Nat))
    | _, _, [] => some []
    | 0, _, _ :: _ => none
    | fuel + 1, off, bs@(_ :: _) =>
      match decodeBytes bs with
      | none => none
      | some (i, len) => (go fuel (off + len) (bs.drop len)).map ((off, i, len) :: ·)

namespace DecoderCheck

theorem wrapMemcpy_agrees_with_llvm :
    sweep Generated.wrapMemcpy.bytes = some Generated.Disasm.wrapMemcpy := by
  decide +kernel

theorem wrapMemmove_agrees_with_llvm :
    sweep Generated.wrapMemmove.bytes = some Generated.Disasm.wrapMemmove := by
  decide +kernel

theorem joltCopyMisaligned_agrees_with_llvm :
    sweep Generated.joltCopyMisaligned.bytes = some Generated.Disasm.joltCopyMisaligned := by
  decide +kernel

end DecoderCheck

end CopyRuntime
