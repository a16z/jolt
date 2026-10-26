/-!
# Object-file data types (trusted)

The extraction script (`extract.py`) turns the ELF relocatable object produced
from `jolt-platform/src/memory.S` into values of these types. The types only
describe data; their meaning is given by `CopyRuntime.Machine.Semantics`.
-/

namespace CopyRuntime

/-- The ELF relocation types that occur in the copy runtime object. The
extraction script rejects every other type, including `R_RISCV_RELAX`
(the source uses `.option norelax`, so the linker may not rewrite code). -/
inductive RelocKind where
  /-- `R_RISCV_JAL`: the J-type immediate of the `jal` at the site becomes
  `S + A - P`. -/
  | jal
  /-- `R_RISCV_CALL_PLT`: an `auipc` at the site and the `jalr` four bytes
  later together receive `S + A - P` (hi20 / lo12 split). -/
  | callPlt
  deriving DecidableEq, Repr

/-- One `Elf64_Rela` entry of a code section. -/
structure Reloc where
  /-- Section offset `P - base` of the patched instruction. -/
  offset : Nat
  kind : RelocKind
  /-- Name of the referenced symbol `S`. -/
  symbol : String
  /-- The explicit addend `A`. -/
  addend : Int
  deriving DecidableEq, Repr

/-- A symbol defined in a section. -/
structure SymbolDef where
  name : String
  /-- Offset within the section. -/
  offset : Nat
  size : Nat
  deriving DecidableEq, Repr

/-- One code section of the relocatable object, exactly as extracted. -/
structure Section where
  name : String
  /-- The raw section contents, in file order. -/
  bytes : List UInt8
  relocs : List Reloc
  symbols : List SymbolDef
  deriving DecidableEq, Repr

end CopyRuntime
