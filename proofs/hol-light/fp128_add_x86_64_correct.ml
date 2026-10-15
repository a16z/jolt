(*
 * Functional correctness proof for Jolt's x86-64 Fp128 addition kernel.
 *
 * The input value a is stored in RDI:RSI, with the low word first. The input
 * b is stored in RDX:RCX. The first instruction loads C = 2^128 - p into R8.
 * The arithmetic result is formed in RDI:RSI, then copied to the System V
 * return registers RAX:RDX. The object file imports the complete optimized
 * public witness bytes for Prime128OffsetA7F7.
 *)

needs (Filename.concat (Sys.getenv "JOLT_FP128_PROOF_DIR")
        "fp128_add_x86_64_object.ml");;

let JOLT_FP128_ADD_X86_64_GENERIC_CORRECT = time prove
 (`!c a0 a1 b0 b1 pc.
      jolt_fp128_valid_offset c
      ==>
      ensures x86
        (\s. bytes_loaded s (word pc) (BUTLAST jolt_fp128_add_mc) /\
             read RIP s = word (pc + 0x6) /\
             read RDI s = a0 /\
             read RSI s = a1 /\
             read RDX s = b0 /\
             read RCX s = b1 /\
             read R8 s = word c)
        (\s. read RIP s = word (pc + 0x2d) /\
             (bignum_of_wordlist [a0; a1] < jolt_fp128_p c /\
              bignum_of_wordlist [b0; b1] < jolt_fp128_p c
              ==> bignum_of_wordlist [read RAX s; read RDX s] =
                  (bignum_of_wordlist [a0; a1] +
                   bignum_of_wordlist [b0; b1]) MOD jolt_fp128_p c))
        (MAYCHANGE [RIP; RAX; RDX; RDI; RSI; R9; R10; R11] ,,
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
  X86_ACCSTEPS_TAC JOLT_FP128_ADD_X86_64_EXEC [2;3;7;8] (2--13) THEN
  ENSURES_FINAL_STATE_TAC THEN
  ASM_SIMP_TAC[VAL_WORD; DIMINDEX_64; MOD_LT] THEN STRIP_TAC THEN
  ABBREV_TAC `l = bignum_of_wordlist [sum_s2; sum_s3]` THEN
  ABBREV_TAC `t = bignum_of_wordlist [sum_s7; sum_s8]` THEN
  SUBGOAL_THEN `2 EXP 128 * bitval carry_s3 + l = m + n` ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["l"; "m"; "n"] THEN
    REWRITE_TAC[bignum_of_wordlist; MULT_CLAUSES; ADD_CLAUSES] THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DECARRY_RULE) THEN
    DISCH_THEN(fun th -> REWRITE_TAC[th]) THEN
    ASM_SIMP_TAC[VAL_WORD; DIMINDEX_64; MOD_LT] THEN REAL_ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN `2 EXP 128 * bitval carry_s8 + t = l + c`
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
  DISCARD_STATE_TAC "s13" THEN
  ACCUMULATOR_POP_ASSUM_LIST(K ALL_TAC) THEN
  MATCH_MP_TAC EQ_TRANS THEN
  EXISTS_TAC `(if carry_s3 \/ carry_s8 then t else l):num` THEN
  CONJ_TAC THENL
   [MAP_EVERY EXPAND_TAC ["t"; "l"] THEN
    ASM_CASES_TAC `carry_s3:bool` THEN
    ASM_CASES_TAC `carry_s8:bool` THEN
    ASM_REWRITE_TAC
     [WORD_SUB_0; VAL_WORD_BITVAL; BITVAL_EQ_0; BITVAL_CLAUSES;
      COND_SWAP] THEN
    CONV_TAC WORD_REDUCE_CONV THEN CONV_TAC NUM_REDUCE_CONV THEN
    ASM_REWRITE_TAC[];
    MATCH_MP_TAC(SPECL
     [`c:num`; `m:num`; `n:num`; `l:num`; `t:num`;
      `carry_s3:bool`; `carry_s8:bool`] JOLT_FP128_ADD_GENERIC) THEN
    ASM_REWRITE_TAC[]]);;

let JOLT_FP128_ADD_X86_64_CORRECT = time prove
 (`!a0 a1 b0 b1 pc.
        ensures x86
          (\s. bytes_loaded s (word pc) (BUTLAST jolt_fp128_add_mc) /\
               read RIP s = word pc /\
               read RDI s = a0 /\
               read RSI s = a1 /\
               read RDX s = b0 /\
               read RCX s = b1)
          (\s. read RIP s = word (pc + 0x2d) /\
               (bignum_of_wordlist [a0; a1] < jolt_fp128_a7f7_p /\
                bignum_of_wordlist [b0; b1] < jolt_fp128_a7f7_p
                ==> bignum_of_wordlist [read RAX s; read RDX s] =
                    (bignum_of_wordlist [a0; a1] +
                     bignum_of_wordlist [b0; b1]) MOD
                    jolt_fp128_a7f7_p))
          (MAYCHANGE [RIP; RAX; RDX; RDI; RSI; R8; R9; R10; R11] ,,
           MAYCHANGE SOME_FLAGS ,, MAYCHANGE [events])`,
  MAP_EVERY X_GEN_TAC
   [`a0:int64`; `a1:int64`; `b0:int64`; `b1:int64`; `pc:num`] THEN
  REWRITE_TAC[SOME_FLAGS] THEN
  ABBREV_TAC `m = bignum_of_wordlist [a0; a1]` THEN
  ABBREV_TAC `n = bignum_of_wordlist [b0; b1]` THEN

  ENSURES_INIT_TAC "s0" THEN
  X86_ACCSTEPS_TAC JOLT_FP128_ADD_X86_64_EXEC [2;3;7;8] (1--13) THEN
  ENSURES_FINAL_STATE_TAC THEN ASM_REWRITE_TAC[] THEN STRIP_TAC THEN

  ABBREV_TAC `l = bignum_of_wordlist [sum_s2; sum_s3]` THEN
  ABBREV_TAC `t = bignum_of_wordlist [sum_s7; sum_s8]` THEN

  SUBGOAL_THEN `2 EXP 128 * bitval carry_s3 + l = m + n` ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["l"; "m"; "n"] THEN
    REWRITE_TAC[bignum_of_wordlist; MULT_CLAUSES; ADD_CLAUSES] THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DECARRY_RULE) THEN
    DISCH_THEN(fun th -> REWRITE_TAC[th]) THEN REAL_ARITH_TAC;
    ALL_TAC] THEN

  SUBGOAL_THEN `2 EXP 128 * bitval carry_s8 + t = l + 4294944759`
  ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["t"; "l"] THEN
    REWRITE_TAC[bignum_of_wordlist; MULT_CLAUSES; ADD_CLAUSES] THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DECARRY_RULE) THEN
    DISCH_THEN(fun th -> REWRITE_TAC[th]) THEN REAL_ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN `l < 2 EXP 128 /\ t < 2 EXP 128` STRIP_ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["l"; "t"] THEN BOUNDER_TAC[];
    ALL_TAC] THEN
  DISCARD_STATE_TAC "s13" THEN
  ACCUMULATOR_POP_ASSUM_LIST(K ALL_TAC) THEN

  (ASM_CASES_TAC `carry_s3:bool` THENL
    [RULE_ASSUM_TAC(REWRITE_RULE[ASSUME `carry_s3:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `carry_s3:bool`);
     RULE_ASSUM_TAC(REWRITE_RULE[ASSUME `~carry_s3:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `~carry_s3:bool`)]) THEN
  (ASM_CASES_TAC `carry_s8:bool` THENL
    [RULE_ASSUM_TAC(REWRITE_RULE[ASSUME `carry_s8:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `carry_s8:bool`);
     RULE_ASSUM_TAC(REWRITE_RULE[ASSUME `~carry_s8:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `~carry_s8:bool`)]) THEN
  ASM_CASES_TAC `jolt_fp128_a7f7_p <= m + n` THEN
  ASM_REWRITE_TAC
   [WORD_SUB_0; VAL_WORD_BITVAL; BITVAL_EQ_0; BITVAL_CLAUSES;
    jolt_fp128_a7f7_p; MOD_ADD_CASES; GSYM NOT_LE; COND_SWAP] THEN
  CONV_TAC WORD_REDUCE_CONV THEN ASM_REWRITE_TAC[] THEN
  RULE_ASSUM_TAC(REWRITE_RULE[jolt_fp128_a7f7_p; BITVAL_CLAUSES]) THEN
  RULE_ASSUM_TAC(CONV_RULE NUM_REDUCE_CONV) THEN
  ASM_SIMP_TAC[MOD_ADD_CASES; jolt_fp128_a7f7_p; GSYM NOT_LE; COND_SWAP] THEN
  CONV_TAC NUM_REDUCE_CONV THEN
  POP_ASSUM_LIST(MP_TAC o end_itlist CONJ) THEN ARITH_TAC);;

let JOLT_FP128_ADD_X86_64_SUBROUTINE_CORRECT = time prove
 (`!a0 a1 b0 b1 pc stackpointer returnaddress.
        ensures x86
          (\s. bytes_loaded s (word pc) jolt_fp128_add_mc /\
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
                     bignum_of_wordlist [b0; b1]) MOD
                    jolt_fp128_a7f7_p))
          (MAYCHANGE [RSP] ,, MAYCHANGE_REGS_AND_FLAGS_PERMITTED_BY_ABI)`,
  X86_PROMOTE_RETURN_NOSTACK_TAC
    jolt_fp128_add_mc JOLT_FP128_ADD_X86_64_CORRECT);;
