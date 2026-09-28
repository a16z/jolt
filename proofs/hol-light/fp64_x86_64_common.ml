(* Shared x86-64 definitions for Jolt's scalar Fp64 proofs. *)

needs "x86/proofs/base.ml";;
needs (Filename.concat (Sys.getenv "JOLT_FP64_PROOF_DIR")
        "fp64_common.ml");;

(* Reorient the SUB/CMP accumulator equation into an unsigned carry equation. *)
let JOLT_FP64_X86_COMPARE_REORIENT = prove
 (`!base carry difference left right.
      --base * carry + difference = left - right
      ==> base * carry + left = right + difference`,
  REAL_ARITH_TAC);;

(* CMP reports the same wrap bit as the preceding addition by 59. *)
let JOLT_FP64_X86_ADD_COMPARE_CARRY = prove
 (`!addcarry comparecarry corrected original difference.
      2 EXP 64 * bitval addcarry + corrected = original + 59 /\
      2 EXP 64 * bitval comparecarry + corrected = original + difference /\
      difference < 2 EXP 64
      ==> comparecarry = addcarry`,
  REPEAT GEN_TAC THEN
  ASM_CASES_TAC `addcarry:bool` THEN
  ASM_CASES_TAC `comparecarry:bool` THEN
  ASM_REWRITE_TAC[BITVAL_CLAUSES] THEN ARITH_TAC);;

(* SUB by p = 2^64 - 59 is the same wrapped operation as ADD by 59. *)
let JOLT_FP64_X86_ADD59_FROM_SUBP = prove
 (`!carry corrected original.
      --(&2 pow 64) * &(bitval carry) + &(corrected:num) =
        &(original:num) - &jolt_fp64_p
      ==> &2 pow 64 * &(bitval(~carry)) + &corrected = &original + &59`,
  REPEAT GEN_TAC THEN REWRITE_TAC[jolt_fp64_p] THEN
  ASM_CASES_TAC `carry:bool` THEN
  ASM_REWRITE_TAC[BITVAL_CLAUSES] THEN
  CONV_TAC NUM_REDUCE_CONV THEN REAL_ARITH_TAC);;
