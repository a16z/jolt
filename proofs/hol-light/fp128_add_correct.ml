(*
 * Functional correctness proof for Jolt's AArch64 Fp128 addition kernel.
 *
 * The object path is supplied by the build that produced fp128_add.o. The
 * explicit instruction list makes the theorem fail if those object bytes
 * change.
 *
 * How to read this file:
 *
 * 1. define_assert_from_elf binds the proof to exact object bytes.
 * 2. ARM_MK_EXEC_RULE gives HOL Light a rule for each AArch64 instruction.
 * 3. JOLT_FP128_ADD_CORRECT proves the arithmetic body before ret.
 * 4. JOLT_FP128_ADD_SUBROUTINE_CORRECT adds ret and the calling convention.
 *
 * The input value a is stored in X0:X1, with the low word first. The input b
 * is stored in X2:X3. The first instruction loads C = 2^128 - p into X4.
 * The result is returned in X0:X1. The chapter book/src/how/formal-verification/field-kernels.md
 * explains this structure and the main tactics for new HOL Light readers.
 *)

needs "arm/proofs/base.ml";;
needs (Filename.concat (Sys.getenv "JOLT_FP128_PROOF_DIR")
        "fp128_common.ml");;

let jolt_fp128_add_object = Sys.getenv "JOLT_FP128_ADD_OBJECT";;

(*** The list includes the constant load, eight arithmetic instructions, and ret.
     Loading an object with any different instruction word fails here. ***)
let jolt_fp128_add_mc =
  define_assert_from_elf "jolt_fp128_add_mc" jolt_fp128_add_object
  [
    0x529003e4;
    0x72ad4464;
    0xab020005;
    0xba030026;
    0x1a9f37e7;
    0xab0400a8;
    0xba1f00c9;
    0x7a4038e0;
    0x9a851100;
    0x9a861121;
    0xd65f03c0
  ];;

let JOLT_FP128_ADD_EXEC = ARM_MK_EXEC_RULE jolt_fp128_add_mc;;

(*** The shared arithmetic body is parameterized by C in X4. This theorem
     starts immediately after the object fixture's constant-loading
     instruction, so it applies to every valid Fp128 offset rather than only
     to the fixture's 6A23801F specialization. ***)
let JOLT_FP128_ADD_GENERIC_CORRECT = time prove
 (`!c a0 a1 b0 b1 pc.
      jolt_fp128_valid_offset c
      ==>
      ensures arm
        (\s. aligned_bytes_loaded s (word pc) jolt_fp128_add_mc /\
             read PC s = word (pc + 0x8) /\
             read X0 s = a0 /\
             read X1 s = a1 /\
             read X2 s = b0 /\
             read X3 s = b1 /\
             read X4 s = word c)
        (\s. read PC s = word (pc + 0x28) /\
             (bignum_of_wordlist [a0; a1] < jolt_fp128_p c /\
              bignum_of_wordlist [b0; b1] < jolt_fp128_p c
              ==> bignum_of_wordlist [read X0 s; read X1 s] =
                  (bignum_of_wordlist [a0; a1] +
                   bignum_of_wordlist [b0; b1]) MOD jolt_fp128_p c))
        (MAYCHANGE [PC; X0; X1; X4; X5; X6; X7; X8; X9] ,,
         MAYCHANGE SOME_FLAGS ,, MAYCHANGE [events])`,
  MAP_EVERY X_GEN_TAC
   [`c:num`; `a0:int64`; `a1:int64`; `b0:int64`; `b1:int64`; `pc:num`] THEN
  DISCH_TAC THEN REWRITE_TAC[SOME_FLAGS] THEN
  SUBGOAL_THEN `c < 2 EXP 64` ASSUME_TAC THENL
   [UNDISCH_TAC `jolt_fp128_valid_offset c` THEN
    REWRITE_TAC[jolt_fp128_valid_offset] THEN ARITH_TAC;
    ALL_TAC] THEN
  ABBREV_TAC `m = bignum_of_wordlist [a0; a1]` THEN
  ABBREV_TAC `n = bignum_of_wordlist [b0; b1]` THEN
  ENSURES_INIT_TAC "s0" THEN
  ARM_ACCSTEPS_TAC JOLT_FP128_ADD_EXEC [2;3;5;6] (2--9) THEN
  ENSURES_FINAL_STATE_TAC THEN
  ASM_SIMP_TAC[VAL_WORD; DIMINDEX_64; MOD_LT] THEN STRIP_TAC THEN
  ABBREV_TAC `l = bignum_of_wordlist [sum_s2; sum_s3]` THEN
  ABBREV_TAC `t = bignum_of_wordlist [sum_s5; sum_s6]` THEN
  SUBGOAL_THEN `2 EXP 128 * bitval carry_s3 + l = m + n` ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["l"; "m"; "n"] THEN
    REWRITE_TAC[bignum_of_wordlist; MULT_CLAUSES; ADD_CLAUSES] THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DECARRY_RULE) THEN
    DISCH_THEN(fun th -> REWRITE_TAC[th]) THEN
    ASM_SIMP_TAC[VAL_WORD; DIMINDEX_64; MOD_LT] THEN REAL_ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN `2 EXP 128 * bitval carry_s6 + t = l + c`
  ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["t"; "l"] THEN
    REWRITE_TAC[bignum_of_wordlist; MULT_CLAUSES; ADD_CLAUSES] THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DECARRY_RULE) THEN
    DISCH_THEN(fun th -> REWRITE_TAC[th]) THEN
    ASM_SIMP_TAC[VAL_WORD; DIMINDEX_64; MOD_LT] THEN REAL_ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN `l < 2 EXP 128 /\ t < 2 EXP 128` STRIP_ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["l"; "t"] THEN BOUNDER_TAC[];
    ALL_TAC] THEN
  DISCARD_STATE_TAC "s9" THEN
  ACCUMULATOR_POP_ASSUM_LIST(K ALL_TAC) THEN
  MATCH_MP_TAC EQ_TRANS THEN
  EXISTS_TAC `(if carry_s3 \/ carry_s6 then t else l):num` THEN
  CONJ_TAC THENL
   [MAP_EVERY EXPAND_TAC ["t"; "l"] THEN
    ASM_CASES_TAC `carry_s3:bool` THEN
    ASM_CASES_TAC `carry_s6:bool` THEN
    ASM_REWRITE_TAC
     [WORD_SUB_0; VAL_WORD_BITVAL; BITVAL_EQ_0; BITVAL_CLAUSES;
      COND_SWAP] THEN
    CONV_TAC WORD_REDUCE_CONV THEN CONV_TAC NUM_REDUCE_CONV THEN
    ASM_REWRITE_TAC[];
    MATCH_MP_TAC(SPECL
     [`c:num`; `m:num`; `n:num`; `l:num`; `t:num`;
      `carry_s3:bool`; `carry_s6:bool`] JOLT_FP128_ADD_GENERIC) THEN
    ASM_REWRITE_TAC[]]);;

(*** ensures arm takes a precondition, a postcondition, and a frame condition.

     The precondition fixes the loaded code, program counter, and input
     registers. The postcondition fixes the next program counter and states
     the field result. The frame condition lists every part of the processor
     state that the body may change.

     The machine execution is modeled for all input words. The field equation
     is conditional on both inputs being canonical values below p. ***)
let JOLT_FP128_ADD_CORRECT = time prove
 (`!a0 a1 b0 b1 pc.
        ensures arm
          (\s. aligned_bytes_loaded s (word pc) jolt_fp128_add_mc /\
               read PC s = word pc /\
               read X0 s = a0 /\
               read X1 s = a1 /\
               read X2 s = b0 /\
               read X3 s = b1)
          (\s. read PC s = word (pc + 0x28) /\
               (bignum_of_wordlist [a0; a1] < jolt_fp128_6a23801f_p /\
                bignum_of_wordlist [b0; b1] < jolt_fp128_6a23801f_p
                ==> bignum_of_wordlist [read X0 s; read X1 s] =
                    (bignum_of_wordlist [a0; a1] +
                     bignum_of_wordlist [b0; b1]) MOD
                    jolt_fp128_6a23801f_p))
          (MAYCHANGE [PC; X0; X1; X4; X5; X6; X7; X8; X9] ,,
           MAYCHANGE SOME_FLAGS ,, MAYCHANGE [events])`,
  MAP_EVERY X_GEN_TAC
   [`a0:int64`; `a1:int64`; `b0:int64`; `b1:int64`; `pc:num`] THEN
  REWRITE_TAC[SOME_FLAGS] THEN
  (* Execute the two-instruction constant load, then apply the body theorem
     proved for every valid offset. *)
  ENSURES_SEQUENCE_TAC `pc + 0x8`
   `\s. aligned_bytes_loaded s (word pc) jolt_fp128_add_mc /\
        read X0 s = a0 /\ read X1 s = a1 /\
        read X2 s = b0 /\ read X3 s = b1 /\
        read X4 s = word 1780711455` THEN
  CONJ_TAC THENL
   [ARM_SIM_TAC JOLT_FP128_ADD_EXEC [1;2];
    MP_TAC(SPECL
     [`1780711455`; `a0:int64`; `a1:int64`; `b0:int64`; `b1:int64`; `pc:num`]
     JOLT_FP128_ADD_GENERIC_CORRECT) THEN
    REWRITE_TAC[SOME_FLAGS; JOLT_FP128_OFFSET6A23801F_VALID;
                JOLT_FP128_6A23801F_P_GENERIC] THEN
    MATCH_MP_TAC(REWRITE_RULE[IMP_CONJ] ENSURES_PRECONDITION_THM) THEN
    SIMP_TAC[]]);;

(*** Lift the body theorem to the callable function. X30 supplies the return
     address, ret transfers control there, and the final frame is the set of
     registers and flags that the AArch64 calling convention permits a callee
     to change. The function does not use stack memory. ***)
let JOLT_FP128_ADD_SUBROUTINE_CORRECT = time prove
 (`!a0 a1 b0 b1 pc returnaddress.
        ensures arm
          (\s. aligned_bytes_loaded s (word pc) jolt_fp128_add_mc /\
               read PC s = word pc /\
               read X30 s = returnaddress /\
               read X0 s = a0 /\
               read X1 s = a1 /\
               read X2 s = b0 /\
               read X3 s = b1)
          (\s. read PC s = returnaddress /\
               (bignum_of_wordlist [a0; a1] < jolt_fp128_6a23801f_p /\
                bignum_of_wordlist [b0; b1] < jolt_fp128_6a23801f_p
                ==> bignum_of_wordlist [read X0 s; read X1 s] =
                    (bignum_of_wordlist [a0; a1] +
                     bignum_of_wordlist [b0; b1]) MOD
                    jolt_fp128_6a23801f_p))
          (MAYCHANGE_REGS_AND_FLAGS_PERMITTED_BY_ABI)`,
  ARM_ADD_RETURN_NOSTACK_TAC
    JOLT_FP128_ADD_EXEC JOLT_FP128_ADD_CORRECT);;
