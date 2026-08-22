(*
 * Functional correctness proof for Jolt's AArch64 Fp128 subtraction kernel.
 *
 * The object path is supplied by the build that produced fp128_sub.o. The
 * explicit instruction list makes the theorem fail if those object bytes
 * change.
 *
 * How to read this file:
 *
 * 1. define_assert_from_elf binds the proof to exact object bytes.
 * 2. ARM_MK_EXEC_RULE gives HOL Light a rule for each AArch64 instruction.
 * 3. JOLT_FP128_SUB_CORRECT proves the arithmetic body before ret.
 * 4. JOLT_FP128_SUB_SUBROUTINE_CORRECT adds ret and the calling convention.
 *
 * The input value a is stored in X0:X1, with the low word first. The input b
 * is stored in X2:X3. X4 contains C = 2^128 - p. The result is returned in
 * X0:X1. The chapter book/src/how/formal-verification/field-kernels.md
 * explains this structure and the main tactics for new HOL Light readers.
 *)

needs "arm/proofs/base.ml";;
needs (Filename.concat (Sys.getenv "JOLT_FP128_PROOF_DIR")
        "fp128_common.ml");;

let jolt_fp128_sub_object = Sys.getenv "JOLT_FP128_SUB_OBJECT";;

(*** The list includes the five arithmetic instructions and the final ret.
     Loading an object with any different instruction word fails here. ***)
let jolt_fp128_sub_mc =
  define_assert_from_elf "jolt_fp128_sub_mc" jolt_fp128_sub_object
  [
    0xeb020005;
    0xfa030026;
    0x9a8423e7;
    0xeb0700a0;
    0xda1f00c1;
    0xd65f03c0
  ];;

let JOLT_FP128_SUB_EXEC = ARM_MK_EXEC_RULE jolt_fp128_sub_mc;;

(*** ensures arm takes a precondition, a postcondition, and a frame condition.

     The precondition fixes the loaded code, program counter, and input
     registers. The postcondition fixes the next program counter and states
     the field result. The frame condition lists every part of the processor
     state that the body may change.

     The machine execution is modeled for all input words. The field equation
     is conditional on both inputs being canonical values below p. The result
     is written as (a + p - b) MOD p because natural number subtraction stops
     at zero. Adding p first makes the expression total for canonical inputs.
     ***)
let JOLT_FP128_SUB_CORRECT = time prove
 (`!a0 a1 b0 b1 pc.
        ensures arm
          (\s. aligned_bytes_loaded s (word pc) jolt_fp128_sub_mc /\
               read PC s = word pc /\
               read X0 s = a0 /\
               read X1 s = a1 /\
               read X2 s = b0 /\
               read X3 s = b1 /\
               read X4 s = word 0xffffa7f7)
          (\s. read PC s = word (pc + 0x14) /\
               (bignum_of_wordlist [a0; a1] < jolt_fp128_a7f7_p /\
                bignum_of_wordlist [b0; b1] < jolt_fp128_a7f7_p
                ==> bignum_of_wordlist [read X0 s; read X1 s] =
                    (bignum_of_wordlist [a0; a1] +
                     jolt_fp128_a7f7_p -
                     bignum_of_wordlist [b0; b1]) MOD
                    jolt_fp128_a7f7_p))
          (MAYCHANGE [PC; X0; X1; X5; X6; X7] ,,
           MAYCHANGE SOME_FLAGS ,, MAYCHANGE [events])`,
  MAP_EVERY X_GEN_TAC
   [`a0:int64`; `a1:int64`; `b0:int64`; `b1:int64`; `pc:num`] THEN
  REWRITE_TAC[SOME_FLAGS] THEN

  (* Give the two input integers short names. Each one joins a low and high
     64 bit word into one natural number. *)
  ABBREV_TAC `m = bignum_of_wordlist [a0; a1]` THEN
  ABBREV_TAC `n = bignum_of_wordlist [b0; b1]` THEN

  (* Symbolically execute instructions 1 through 5. The selected accumulator
     steps retain the borrow and correction equations needed below. *)
  ENSURES_INIT_TAC "s0" THEN
  ARM_ACCSTEPS_TAC JOLT_FP128_SUB_EXEC [1;2;4;5] (1--5) THEN
  ENSURES_FINAL_STATE_TAC THEN ASM_REWRITE_TAC[] THEN STRIP_TAC THEN

  (* d is the wrapped result of m - n. t is the corrected result. *)
  ABBREV_TAC `d = bignum_of_wordlist [sum_s1; sum_s2]` THEN
  ABBREV_TAC `t = bignum_of_wordlist [sum_s4; sum_s5]` THEN

  (* The first borrow chain computes m - n modulo 2^128. *)
  SUBGOAL_THEN `2 EXP 128 * bitval carry_s2 + m = n + d`
  ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["d"; "m"; "n"] THEN
    REWRITE_TAC[bignum_of_wordlist; MULT_CLAUSES; ADD_CLAUSES] THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DECARRY_RULE) THEN
    DISCH_THEN(fun th -> REWRITE_TAC[th]) THEN REAL_ARITH_TAC;
    ALL_TAC] THEN

  (* The selected correction is zero without a borrow, and C otherwise.
     AArch64 CF means no borrow after subtraction. The accumulator equation
     below uses carry_s2 for the mathematical borrow, so the condition appears
     as ~carry_s2 when the selected correction is zero. *)
  SUBGOAL_THEN
   `2 EXP 128 * bitval carry_s5 + d =
    val(if ~carry_s2 then (word 0:int64) else word 4294944759) + t`
  ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["t"; "d"] THEN
    REWRITE_TAC[bignum_of_wordlist; MULT_CLAUSES; ADD_CLAUSES] THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DECARRY_RULE) THEN
    DISCH_THEN(fun th -> REWRITE_TAC[th]) THEN REAL_ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN `d < 2 EXP 128 /\ t < 2 EXP 128` STRIP_ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["d"; "t"] THEN BOUNDER_TAC[];
    ALL_TAC] THEN
  DISCARD_STATE_TAC "s5" THEN
  ACCUMULATOR_POP_ASSUM_LIST(K ALL_TAC) THEN
  RULE_ASSUM_TAC(REWRITE_RULE[jolt_fp128_a7f7_p]) THEN
  REWRITE_TAC[jolt_fp128_a7f7_p] THEN

  (* Canonical inputs make a + p - b smaller than 2p. Its reduction therefore
     has only two cases. Keep the value when it is below p, or subtract p once.
     The final carry cases show that the assembly makes the same choice. *)
  SUBGOAL_THEN
   `m + 340282366920938463463374607427473266697 - n <
    2 * 340282366920938463463374607427473266697`
  ASSUME_TAC THENL
   [POP_ASSUM_LIST(MP_TAC o end_itlist CONJ) THEN ARITH_TAC;
    ALL_TAC] THEN

  SUBGOAL_THEN
   `(m + 340282366920938463463374607427473266697 - n) MOD
     340282366920938463463374607427473266697 =
    if m + 340282366920938463463374607427473266697 - n <
       340282366920938463463374607427473266697
    then m + 340282366920938463463374607427473266697 - n
    else (m + 340282366920938463463374607427473266697 - n) -
         340282366920938463463374607427473266697`
  SUBST1_TAC THENL
   [MATCH_MP_TAC MOD_CASES THEN ASM_REWRITE_TAC[];
    ALL_TAC] THEN
  ASM_CASES_TAC
   `m + 340282366920938463463374607427473266697 - n <
    340282366920938463463374607427473266697` THEN

  (ASM_CASES_TAC `carry_s2:bool` THENL
    [RULE_ASSUM_TAC
      (REWRITE_RULE[ASSUME `carry_s2:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `carry_s2:bool`);
     RULE_ASSUM_TAC
      (REWRITE_RULE[ASSUME `~carry_s2:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `~carry_s2:bool`)]) THEN
  (ASM_CASES_TAC `carry_s5:bool` THENL
    [RULE_ASSUM_TAC
      (REWRITE_RULE[ASSUME `carry_s5:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `carry_s5:bool`);
     RULE_ASSUM_TAC
      (REWRITE_RULE[ASSUME `~carry_s5:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `~carry_s5:bool`)]) THEN
  ASM_REWRITE_TAC[BITVAL_CLAUSES] THEN
  POP_ASSUM_LIST(MP_TAC o end_itlist CONJ) THEN
  CONV_TAC(DEPTH_CONV WORD_NUM_RED_CONV) THEN
  CONV_TAC NUM_REDUCE_CONV THEN
  ARITH_TAC);;

(*** Lift the body theorem to the callable function. X30 supplies the return
     address, ret transfers control there, and the final frame is the set of
     registers and flags that the AArch64 calling convention permits a callee
     to change. The function does not use stack memory. ***)
let JOLT_FP128_SUB_SUBROUTINE_CORRECT = time prove
 (`!a0 a1 b0 b1 pc returnaddress.
        ensures arm
          (\s. aligned_bytes_loaded s (word pc) jolt_fp128_sub_mc /\
               read PC s = word pc /\
               read X30 s = returnaddress /\
               read X0 s = a0 /\
               read X1 s = a1 /\
               read X2 s = b0 /\
               read X3 s = b1 /\
               read X4 s = word 0xffffa7f7)
          (\s. read PC s = returnaddress /\
               (bignum_of_wordlist [a0; a1] < jolt_fp128_a7f7_p /\
                bignum_of_wordlist [b0; b1] < jolt_fp128_a7f7_p
                ==> bignum_of_wordlist [read X0 s; read X1 s] =
                    (bignum_of_wordlist [a0; a1] +
                     jolt_fp128_a7f7_p -
                     bignum_of_wordlist [b0; b1]) MOD
                    jolt_fp128_a7f7_p))
          (MAYCHANGE_REGS_AND_FLAGS_PERMITTED_BY_ABI)`,
  ARM_ADD_RETURN_NOSTACK_TAC
    JOLT_FP128_SUB_EXEC JOLT_FP128_SUB_CORRECT);;
