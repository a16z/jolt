(* Shared machine facts for the x86-64 Fp128 proofs. *)

needs "x86/proofs/base.ml";;
needs (Filename.concat (Sys.getenv "JOLT_FP128_PROOF_DIR")
        "fp128_common.ml");;

(* SBB r9,r9 turns the borrow flag into zero or an all-one word. AND then
   selects zero or C. This lemma records that exact two-instruction effect. *)
let JOLT_FP128_X86_64_BORROW_MASK = prove
 (`!borrow (c:int64).
        word_and (word_neg (word (bitval borrow))) c =
        if borrow then c else word 0`,
  REWRITE_TAC[WORD_AND_MASK]);;
