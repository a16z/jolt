(*
 * Functional correctness proof for Jolt's x86-64 Fp128 subtraction kernel.
 *
 * The input value a is stored in RDI:RSI, with the low word first. The input
 * b is stored in RDX:RCX. The first instruction loads C = 2^128 - p into R8.
 * The arithmetic result is formed in RDI:RSI, then copied to the System V
 * return registers RAX:RDX. The object file imports the complete optimized
 * public witness bytes for Prime128OffsetA7F7.
 *)

needs (Filename.concat (Sys.getenv "JOLT_FP128_PROOF_DIR")
        "fp128_sub_x86_64_object.ml");;

let JOLT_FP128_SUB_X86_64_GENERIC_CORRECT = time prove
 (`!c a0 a1 b0 b1 pc.
      jolt_fp128_valid_offset c
      ==>
      ensures x86
        (\s. bytes_loaded s (word pc) (BUTLAST jolt_fp128_sub_mc) /\
             read RIP s = word (pc + 0x6) /\
             read RDI s = a0 /\
             read RSI s = a1 /\
             read RDX s = b0 /\
             read RCX s = b1 /\
             read R8 s = word c)
        (\s. read RIP s = word (pc + 0x1f) /\
             (bignum_of_wordlist [a0; a1] < jolt_fp128_p c /\
              bignum_of_wordlist [b0; b1] < jolt_fp128_p c
              ==> bignum_of_wordlist [read RAX s; read RDX s] =
                  (bignum_of_wordlist [a0; a1] + jolt_fp128_p c -
                   bignum_of_wordlist [b0; b1]) MOD jolt_fp128_p c))
        (MAYCHANGE [RIP; RAX; RDX; RDI; RSI; R9] ,,
         MAYCHANGE SOME_FLAGS ,, MAYCHANGE [events])`,
  MAP_EVERY X_GEN_TAC
   [`c:num`; `a0:int64`; `a1:int64`; `b0:int64`; `b1:int64`; `pc:num`] THEN
  DISCH_TAC THEN REWRITE_TAC[SOME_FLAGS] THEN
  SUBGOAL_THEN `c < 2 EXP 64` ASSUME_TAC THENL
   [UNDISCH_TAC `jolt_fp128_valid_offset c` THEN
    REWRITE_TAC[jolt_fp128_valid_offset] THEN ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN `c MOD 2 EXP 64 = c` ASSUME_TAC THENL
   [MATCH_MP_TAC MOD_LT THEN ASM_REWRITE_TAC[]; ALL_TAC] THEN
  ABBREV_TAC `m = bignum_of_wordlist [a0; a1]` THEN
  ABBREV_TAC `n = bignum_of_wordlist [b0; b1]` THEN
  ENSURES_INIT_TAC "s0" THEN
  X86_ACCSTEPS_TAC JOLT_FP128_SUB_X86_64_EXEC [2;3;6;7] (2--9) THEN
  ENSURES_FINAL_STATE_TAC THEN
  ASM_SIMP_TAC[VAL_WORD; DIMINDEX_64; MOD_LT] THEN STRIP_TAC THEN
  ABBREV_TAC `d = bignum_of_wordlist [sum_s2; sum_s3]` THEN
  ABBREV_TAC `t = bignum_of_wordlist [sum_s6; sum_s7]` THEN
  SUBGOAL_THEN `2 EXP 128 * bitval carry_s3 + m = n + d`
  ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["d"; "m"; "n"] THEN
    REWRITE_TAC[bignum_of_wordlist; MULT_CLAUSES; ADD_CLAUSES] THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DECARRY_RULE) THEN
    REAL_ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN
   `2 EXP 128 * bitval carry_s7 + d =
    (if carry_s3 then c else 0) + t`
  ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["t"; "d"] THEN
    REWRITE_TAC[bignum_of_wordlist; MULT_CLAUSES; ADD_CLAUSES] THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DECARRY_RULE) THEN
    REWRITE_TAC[JOLT_FP128_X86_64_BORROW_MASK] THEN
    ASM_CASES_TAC `carry_s3:bool` THEN
    ASM_REWRITE_TAC
     [BITVAL_CLAUSES; VAL_WORD_0; VAL_WORD; DIMINDEX_64; MOD_LT] THEN
    REAL_ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN `d < 2 EXP 128 /\ t < 2 EXP 128` STRIP_ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["d"; "t"] THEN BOUNDER_TAC[];
    ALL_TAC] THEN
  DISCARD_STATE_TAC "s9" THEN
  ACCUMULATOR_POP_ASSUM_LIST(K ALL_TAC) THEN
  MATCH_MP_TAC(SPECL
   [`c:num`; `m:num`; `n:num`; `d:num`; `t:num`;
    `carry_s3:bool`; `carry_s7:bool`] JOLT_FP128_SUB_GENERIC) THEN
  ASM_REWRITE_TAC[]);;

let JOLT_FP128_SUB_X86_64_CORRECT = time prove
 (`!a0 a1 b0 b1 pc.
        ensures x86
          (\s. bytes_loaded s (word pc) (BUTLAST jolt_fp128_sub_mc) /\
               read RIP s = word pc /\
               read RDI s = a0 /\
               read RSI s = a1 /\
               read RDX s = b0 /\
               read RCX s = b1)
          (\s. read RIP s = word (pc + 0x1f) /\
               (bignum_of_wordlist [a0; a1] < jolt_fp128_a7f7_p /\
                bignum_of_wordlist [b0; b1] < jolt_fp128_a7f7_p
                ==> bignum_of_wordlist [read RAX s; read RDX s] =
                    (bignum_of_wordlist [a0; a1] +
                     jolt_fp128_a7f7_p -
                     bignum_of_wordlist [b0; b1]) MOD
                    jolt_fp128_a7f7_p))
          (MAYCHANGE [RIP; RAX; RDX; RDI; RSI; R8; R9] ,,
           MAYCHANGE SOME_FLAGS ,, MAYCHANGE [events])`,
  MAP_EVERY X_GEN_TAC
   [`a0:int64`; `a1:int64`; `b0:int64`; `b1:int64`; `pc:num`] THEN
  REWRITE_TAC[SOME_FLAGS] THEN
  ABBREV_TAC `m = bignum_of_wordlist [a0; a1]` THEN
  ABBREV_TAC `n = bignum_of_wordlist [b0; b1]` THEN

  ENSURES_INIT_TAC "s0" THEN
  X86_ACCSTEPS_TAC JOLT_FP128_SUB_X86_64_EXEC [2;3;6;7] (1--9) THEN
  ENSURES_FINAL_STATE_TAC THEN ASM_REWRITE_TAC[] THEN STRIP_TAC THEN

  ABBREV_TAC `d = bignum_of_wordlist [sum_s2; sum_s3]` THEN
  ABBREV_TAC `t = bignum_of_wordlist [sum_s6; sum_s7]` THEN

  SUBGOAL_THEN `2 EXP 128 * bitval carry_s3 + m = n + d`
  ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["d"; "m"; "n"] THEN
    REWRITE_TAC[bignum_of_wordlist; MULT_CLAUSES; ADD_CLAUSES] THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DECARRY_RULE) THEN
    REAL_ARITH_TAC;
    ALL_TAC] THEN

  SUBGOAL_THEN
   `2 EXP 128 * bitval carry_s7 + d =
    val(if carry_s3 then word 4294944759 else (word 0:int64)) + t`
  ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["t"; "d"] THEN
    REWRITE_TAC[bignum_of_wordlist; MULT_CLAUSES; ADD_CLAUSES] THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DECARRY_RULE) THEN
    REWRITE_TAC[JOLT_FP128_X86_64_BORROW_MASK] THEN REAL_ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN `d < 2 EXP 128 /\ t < 2 EXP 128` STRIP_ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["d"; "t"] THEN BOUNDER_TAC[];
    ALL_TAC] THEN
  DISCARD_STATE_TAC "s9" THEN
  ACCUMULATOR_POP_ASSUM_LIST(K ALL_TAC) THEN
  RULE_ASSUM_TAC(REWRITE_RULE[jolt_fp128_a7f7_p]) THEN
  REWRITE_TAC[jolt_fp128_a7f7_p] THEN

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
  (ASM_CASES_TAC `carry_s3:bool` THENL
    [RULE_ASSUM_TAC
      (REWRITE_RULE[ASSUME `carry_s3:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `carry_s3:bool`);
     RULE_ASSUM_TAC
      (REWRITE_RULE[ASSUME `~carry_s3:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `~carry_s3:bool`)]) THEN
  (ASM_CASES_TAC `carry_s7:bool` THENL
    [RULE_ASSUM_TAC
      (REWRITE_RULE[ASSUME `carry_s7:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `carry_s7:bool`);
     RULE_ASSUM_TAC
      (REWRITE_RULE[ASSUME `~carry_s7:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `~carry_s7:bool`)]) THEN
  ASM_REWRITE_TAC[BITVAL_CLAUSES] THEN
  POP_ASSUM_LIST(MP_TAC o end_itlist CONJ) THEN
  CONV_TAC(DEPTH_CONV WORD_NUM_RED_CONV) THEN
  CONV_TAC NUM_REDUCE_CONV THEN
  ARITH_TAC);;

let JOLT_FP128_SUB_X86_64_SUBROUTINE_CORRECT = time prove
 (`!a0 a1 b0 b1 pc stackpointer returnaddress.
        ensures x86
          (\s. bytes_loaded s (word pc) jolt_fp128_sub_mc /\
               read RIP s = word pc /\
               read RSP s = stackpointer /\
               read (memory :> bytes64 stackpointer) s = returnaddress /\
               read RDI s = a0 /\
               read RSI s = a1 /\
               read RDX s = b0 /\
               read RCX s = b1)
          (\s. read RIP s = returnaddress /\
               read RSP s = word_add stackpointer (word 8) /\
               (bignum_of_wordlist [a0; a1] < jolt_fp128_a7f7_p /\
                bignum_of_wordlist [b0; b1] < jolt_fp128_a7f7_p
                ==> bignum_of_wordlist [read RAX s; read RDX s] =
                    (bignum_of_wordlist [a0; a1] +
                     jolt_fp128_a7f7_p -
                     bignum_of_wordlist [b0; b1]) MOD
                    jolt_fp128_a7f7_p))
          (MAYCHANGE [RSP] ,, MAYCHANGE_REGS_AND_FLAGS_PERMITTED_BY_ABI)`,
  X86_PROMOTE_RETURN_NOSTACK_TAC
    jolt_fp128_sub_mc JOLT_FP128_SUB_X86_64_CORRECT);;
