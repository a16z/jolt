import Lean
import CopyRuntime.Theorems

/-!
# Axiom audit

`#assert_axioms c [a, …]` fails the build unless every axiom that `c`
(transitively) depends on is one of the listed names. The standard axioms are
`propext`, `Classical.choice` and `Quot.sound`. In particular this rejects
`sorryAx`, `Lean.ofReduceBool` (`native_decide`) and the auxiliary axioms that
`bv_decide` creates for its natively checked certificates (none of which this
project may use).

Theorems that still depend on unfinished path lemmas are listed with
`sorryAx` allowed; remove it from the list when the path is proved.
-/

open Lean Elab Command

elab "#assert_axioms " id:ident " [" ax:ident,* "]" : command => do
  let n ← liftCoreM <| realizeGlobalConstNoOverloadWithInfo id
  let axs ← liftCoreM <| collectAxioms n
  let allowed := ax.getElems.map (·.getId)
  let bad := axs.filter fun a => !allowed.contains a
  unless bad.isEmpty do
    throwError "{n} depends on unexpected axioms {bad}"
  logInfo m!"{n}: {axs}"

namespace CopyRuntime

-- Decoder cross-check against LLVM (kernel evaluation only).
#assert_axioms DecoderCheck.wrapMemcpy_agrees_with_llvm [propext, Classical.choice, Quot.sound]
#assert_axioms DecoderCheck.wrapMemmove_agrees_with_llvm [propext, Classical.choice, Quot.sound]
#assert_axioms DecoderCheck.joltCopyMisaligned_agrees_with_llvm
  [propext, Classical.choice, Quot.sound]

-- Completely proved wrapper paths.
#assert_axioms Wrapper.zero_path [propext, Classical.choice, Quot.sound]
#assert_axioms Wrapper.fallback_path [propext, Classical.choice, Quot.sound]
#assert_axioms Wrapper.leaf_path [propext, Classical.choice, Quot.sound]
#assert_axioms Wrapper.aligned_small_path [propext, Classical.choice, Quot.sound]
#assert_axioms Wrapper.small_bytes_path [propext, Classical.choice, Quot.sound]
#assert_axioms Wrapper.fwd_tail [propext, Classical.choice, Quot.sound]
#assert_axioms Wrapper.fwd_words [propext, Classical.choice, Quot.sound]
#assert_axioms Wrapper.fwd_pairs [propext, Classical.choice, Quot.sound]
#assert_axioms Wrapper.fwd_blocks [propext, Classical.choice, Quot.sound]
#assert_axioms Wrapper.fwd_large [propext, Classical.choice, Quot.sound]
#assert_axioms Wrapper.fwd_prologue [propext, Classical.choice, Quot.sound]
#assert_axioms Wrapper.forward_path [propext, Classical.choice, Quot.sound]
#assert_axioms Leaf.combine_byte [propext, Classical.choice, Quot.sound]
#assert_axioms Theorems.entry_symbols [propext, Classical.choice, Quot.sound]
#assert_axioms Theorems.wrappers_same_code [propext, Classical.choice, Quot.sound]
#assert_axioms Theorems.exit_state_unique [propext, Classical.choice, Quot.sound]

-- Main theorems: still depend on `Wrapper.backward_path` and `Leaf.leaf_path`,
-- which are `sorry`.
#assert_axioms Theorems.wrap_memcpy_correct [propext, Classical.choice, Quot.sound, sorryAx]
#assert_axioms Theorems.wrap_memmove_correct [propext, Classical.choice, Quot.sound, sorryAx]
#assert_axioms Theorems.jolt_copy_misaligned_correct
  [propext, Classical.choice, Quot.sound, sorryAx]
#assert_axioms Theorems.wrap_memcpy_linked_correct
  [propext, Classical.choice, Quot.sound, sorryAx]
#assert_axioms Theorems.wrap_memmove_linked_correct
  [propext, Classical.choice, Quot.sound, sorryAx]

end CopyRuntime
