(*
 * Functional correctness proof for Jolt's baseline x86-64 Fp128 multiply.
 *
 * Inputs are RDI:RSI and RDX:RCX, low word first. The object loads
 * C = 2^128 - p into R8. The arithmetic result is formed in RDI:RCX, then
 * copied to the System V return registers RAX:RDX. The proof follows the
 * actual carry chain through the 256-bit schoolbook product, two Solinas
 * folds, the final conditional correction, and the ABI result moves.
 *)

needs (Filename.concat (Sys.getenv "JOLT_FP128_PROOF_DIR")
        "fp128_mul_x86_64_object.ml");;

let JOLT_FP128_X86_64_ACCUMULATOR_NAT_TAC index =
  REWRITE_TAC[GSYM REAL_OF_NUM_EQ; GSYM REAL_OF_NUM_CLAUSES] THEN
  ACCUMULATOR_ASSUM_LIST(fun thl ->
    MP_TAC(hd(DESUM_RULE [List.nth (rev thl) index]))) THEN
  REWRITE_TAC[VAL_WORD_BITVAL] THEN REAL_ARITH_TAC;;

let JOLT_FP128_MUL_X86_64_GENERIC_CORRECT = time prove
 (`!c a0 a1 b0 b1 pc.
      jolt_fp128_valid_offset c
      ==>
      ensures x86
          (\s. bytes_loaded s (word pc) (BUTLAST jolt_fp128_mul_x86_64_mc) /\
               read RIP s = word (pc + 0x6) /\
               read RDI s = a0 /\
               read RSI s = a1 /\
               read RDX s = b0 /\
               read RCX s = b1 /\
               read R8 s = word c)
          (\s. read RIP s = word (pc + 0x9e) /\
               (bignum_of_wordlist [a0; a1] < jolt_fp128_p c /\
                bignum_of_wordlist [b0; b1] < jolt_fp128_p c
                ==> bignum_of_wordlist [read RAX s; read RDX s] =
                    (bignum_of_wordlist [a0; a1] *
                     bignum_of_wordlist [b0; b1]) MOD jolt_fp128_p c))
          (MAYCHANGE [RIP; RAX; RCX; RDX; RDI; RSI; R8; R9; R10; R11] ,,
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
  X86_ACCSTEPS_TAC JOLT_FP128_MUL_X86_64_EXEC
   [6;10;11;12;15;17;18;21;22;23;26;28;29;32;33;34;36;38;39;43;44;45]
   (2--49) THEN
  ENSURES_FINAL_STATE_TAC THEN
  ASM_SIMP_TAC[VAL_WORD; DIMINDEX_64; MOD_LT] THEN STRIP_TAC THEN
  SUBGOAL_THEN `val(word c:int64) = c` ASSUME_TAC THENL
   [ASM_SIMP_TAC[VAL_WORD; DIMINDEX_64; MOD_LT]; ALL_TAC] THEN
  SUBGOAL_THEN `&(val(word c:int64)):real = &c` ASSUME_TAC THENL
   [ASM_REWRITE_TAC[]; ALL_TAC] THEN

  ABBREV_TAC
   `mn4 = bignum_of_wordlist [mullo_s6; sum_s17; sum_s22; sum_s23]` THEN
  SUBGOAL_THEN `m * n < 2 EXP 256` ASSUME_TAC THENL
   [REWRITE_TAC[ARITH_RULE `2 EXP 256 = 2 EXP 128 * 2 EXP 128`] THEN
    MATCH_MP_TAC LT_MULT2 THEN
    MAP_EVERY UNDISCH_TAC
     [`m < jolt_fp128_p c`; `n < jolt_fp128_p c`] THEN
    UNDISCH_TAC `jolt_fp128_valid_offset c` THEN
    REWRITE_TAC[jolt_fp128_valid_offset; jolt_fp128_p] THEN ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN
   `2 EXP 64 * val(mulhi_s10:int64) + val(mullo_s10:int64) =
    val(a0:int64) * val(b1:int64)`
  ASSUME_TAC THENL
   [JOLT_FP128_X86_64_ACCUMULATOR_NAT_TAC 1; ALL_TAC] THEN
  SUBGOAL_THEN `val(mulhi_s10:int64) < 2 EXP 64 - 1` ASSUME_TAC THENL
   [SUBGOAL_THEN
     `val(a0:int64) * val(b1:int64) <=
      (2 EXP 64 - 1) * (2 EXP 64 - 1)` ASSUME_TAC THENL
     [MATCH_MP_TAC LE_MULT2 THEN BOUNDER_TAC[]; ALL_TAC] THEN
    SUBGOAL_THEN
     `(2 EXP 64 - 1) * (2 EXP 64 - 1) <
      2 EXP 64 * (2 EXP 64 - 1)` ASSUME_TAC THENL
     [CONV_TAC NUM_REDUCE_CONV; ALL_TAC] THEN
    MAP_EVERY UNDISCH_TAC
     [`2 EXP 64 * val(mulhi_s10:int64) + val(mullo_s10:int64) =
       val(a0:int64) * val(b1:int64)`;
      `val(a0:int64) * val(b1:int64) <=
       (2 EXP 64 - 1) * (2 EXP 64 - 1)`;
      `(2 EXP 64 - 1) * (2 EXP 64 - 1) <
       2 EXP 64 * (2 EXP 64 - 1)`] THEN
    ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN
   `2 EXP 64 * bitval carry_s12 + val(sum_s12:int64) =
    val(mulhi_s10:int64) + bitval carry_s11`
  ASSUME_TAC THENL
   [JOLT_FP128_X86_64_ACCUMULATOR_NAT_TAC 3; ALL_TAC] THEN
  SUBGOAL_THEN `~carry_s12` ASSUME_TAC THENL
   [ASM_CASES_TAC `carry_s12:bool` THEN ASM_REWRITE_TAC[] THEN
    MAP_EVERY UNDISCH_TAC
     [`2 EXP 64 * bitval carry_s12 + val(sum_s12:int64) =
       val(mulhi_s10:int64) + bitval carry_s11`;
      `val(mulhi_s10:int64) < 2 EXP 64 - 1`] THEN
    MP_TAC(SPEC `carry_s11:bool` BITVAL_BOUND) THEN
    ASM_REWRITE_TAC[BITVAL_CLAUSES] THEN ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN
   `2 EXP 256 * bitval carry_s23 + mn4 = m * n`
  ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["mn4"; "m"; "n"] THEN
    REWRITE_TAC[bignum_of_wordlist] THEN
    CONV_TAC NUM_REDUCE_CONV THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES; REAL_MUL_RZERO; REAL_ADD_RID] THEN
    CONV_TAC(RAND_CONV REAL_POLY_CONV) THEN
    ACCUMULATOR_ASSUM_LIST(fun thl ->
      MP_TAC(end_itlist CONJ
       (DESUM_RULE (fst(chop_list 10 (rev thl)))))) THEN
    DISCH_THEN(fun th -> REWRITE_TAC[th]) THEN
    ASM_REWRITE_TAC[BITVAL_CLAUSES; VAL_WORD_BITVAL] THEN
    CONV_TAC REAL_RING;
    ALL_TAC] THEN
  SUBGOAL_THEN `mn4 = m * n` ASSUME_TAC THENL
   [MATCH_MP_TAC(SPECL
      [`bitval carry_s23`; `2 EXP 256`; `mn4:num`; `m * n`]
      JOLT_FP128_NO_TOP_CARRY) THEN
    REPEAT CONJ_TAC THENL
     [CONV_TAC NUM_REDUCE_CONV;
      ACCEPT_TAC(ASSUME
       `2 EXP 256 * bitval carry_s23 + mn4 = m * n`);
      ACCEPT_TAC(ASSUME `m * n < 2 EXP 256`)];
    ALL_TAC] THEN
  ABBREV_TAC `lo0 = bignum_of_wordlist [mullo_s6; sum_s17]` THEN
  ABBREV_TAC `hi0 = bignum_of_wordlist [sum_s22; sum_s23]` THEN
  SUBGOAL_THEN `mn4 = 2 EXP 128 * hi0 + lo0` ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["mn4"; "lo0"; "hi0"] THEN
    REWRITE_TAC[bignum_of_wordlist] THEN ARITH_TAC;
    ALL_TAC] THEN

  SUBGOAL_THEN `lo0 < 2 EXP 128 /\ hi0 < 2 EXP 128`
  STRIP_ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["lo0"; "hi0"] THEN
    CONJ_TAC THEN MATCH_ACCEPT_TAC JOLT_FP128_WORDLIST2_BOUND;
    ALL_TAC] THEN

  ABBREV_TAC `t = bignum_of_wordlist [sum_s28; sum_s33; sum_s34]` THEN
  SUBGOAL_THEN `lo0 + c * hi0 < 2 EXP 192` ASSUME_TAC THENL
   [SUBGOAL_THEN `c < 2 EXP 32` ASSUME_TAC THENL
     [UNDISCH_TAC `jolt_fp128_valid_offset c` THEN
      REWRITE_TAC[jolt_fp128_valid_offset] THEN ARITH_TAC;
      ALL_TAC] THEN
    SUBGOAL_THEN `c * hi0 < 2 EXP 32 * 2 EXP 128` ASSUME_TAC THENL
     [MATCH_MP_TAC LT_MULT2 THEN ASM_REWRITE_TAC[];
      ALL_TAC] THEN
    MAP_EVERY UNDISCH_TAC
     [`lo0 < 2 EXP 128`; `c * hi0 < 2 EXP 32 * 2 EXP 128`] THEN
    CONV_TAC NUM_REDUCE_CONV THEN ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN
   `2 EXP 192 * bitval carry_s34 + t = lo0 + c * hi0`
  ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["t"; "lo0"; "hi0"] THEN
    REWRITE_TAC[bignum_of_wordlist; GSYM REAL_OF_NUM_CLAUSES;
                REAL_MUL_RZERO; REAL_ADD_RID] THEN
    CONV_TAC(RAND_CONV REAL_POLY_CONV) THEN
    ACCUMULATOR_ASSUM_LIST(fun thl ->
      let selected = fst(chop_list 6 (snd(chop_list 10 (rev thl)))) in
      MP_TAC(end_itlist CONJ (DESUM_RULE selected))) THEN
    DISCH_THEN(fun th -> REWRITE_TAC[th]) THEN
    REWRITE_TAC[ASSUME `val(word c:int64) = c`; VAL_WORD_BITVAL] THEN
    CONV_TAC REAL_RAT_REDUCE_CONV THEN CONV_TAC REAL_RING;
    ALL_TAC] THEN
  SUBGOAL_THEN `t = lo0 + c * hi0` ASSUME_TAC THENL
   [MP_TAC(SPECL
      [`bitval carry_s34`; `2 EXP 192`; `t:num`;
       `lo0 + c * hi0`] JOLT_FP128_NO_TOP_CARRY) THEN
    ANTS_TAC THENL [ALL_TAC; SIMP_TAC[]] THEN
    CONJ_TAC THENL [CONV_TAC NUM_REDUCE_CONV; ALL_TAC] THEN
    CONJ_TAC THENL
     [
      ACCEPT_TAC(ASSUME
       `2 EXP 192 * bitval carry_s34 + t = lo0 + c * hi0`);
      ACCEPT_TAC(ASSUME `lo0 + c * hi0 < 2 EXP 192`)];
    ALL_TAC] THEN

  ABBREV_TAC `lo1 = bignum_of_wordlist [sum_s28; sum_s33]` THEN
  SUBGOAL_THEN `lo1 < 2 EXP 128` ASSUME_TAC THENL
   [EXPAND_TAC "lo1" THEN MATCH_ACCEPT_TAC JOLT_FP128_WORDLIST2_BOUND;
    ALL_TAC] THEN
  SUBGOAL_THEN `t = 2 EXP 128 * val(sum_s34:int64) + lo1` ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["t"; "lo1"] THEN
    REWRITE_TAC[bignum_of_wordlist] THEN ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN `c * hi0 < c * 2 EXP 128` ASSUME_TAC THENL
   [MATCH_MP_TAC LT_LMULT THEN CONJ_TAC THENL
     [UNDISCH_TAC `jolt_fp128_valid_offset c` THEN
      REWRITE_TAC[jolt_fp128_valid_offset] THEN ARITH_TAC;
      ACCEPT_TAC(ASSUME `hi0 < 2 EXP 128`)];
    ALL_TAC] THEN
  SUBGOAL_THEN `val(sum_s34:int64) <= c` ASSUME_TAC THENL
   [MAP_EVERY UNDISCH_TAC
     [`t = lo0 + c * hi0`;
      `t = 2 EXP 128 * val(sum_s34:int64) + lo1`;
      `lo0 < 2 EXP 128`; `lo1 < 2 EXP 128`;
      `c * hi0 < c * 2 EXP 128`] THEN
    CONV_TAC NUM_REDUCE_CONV THEN ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN `val(sum_s34:int64) * c < 2 EXP 64`
  ASSUME_TAC THENL
   [SUBGOAL_THEN `val(sum_s34:int64) < 2 EXP 32 /\ c < 2 EXP 32`
    STRIP_ASSUME_TAC THENL
     [UNDISCH_TAC `val(sum_s34:int64) <= c` THEN
      UNDISCH_TAC `jolt_fp128_valid_offset c` THEN
      REWRITE_TAC[jolt_fp128_valid_offset] THEN ARITH_TAC;
      ALL_TAC] THEN
    REWRITE_TAC[ARITH_RULE `2 EXP 64 = 2 EXP 32 * 2 EXP 32`] THEN
    MATCH_MP_TAC LT_MULT2 THEN ASM_REWRITE_TAC[];
    ALL_TAC] THEN
  SUBGOAL_THEN
   `2 EXP 64 * val(mulhi_s36:int64) + val(mullo_s36:int64) =
    val(sum_s34:int64) * c`
  ASSUME_TAC THENL
   [REWRITE_TAC[GSYM REAL_OF_NUM_EQ; GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(fun thl ->
      let th = hd(DESUM_RULE [List.nth (rev thl) 16]) in
      MP_TAC(REWRITE_RULE
       [ASSUME `&(val(word c:int64)):real = &c`; VAL_WORD_BITVAL] th)) THEN
    REAL_ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN
   `val(mullo_s36:int64) = val(sum_s34:int64) * c`
  ASSUME_TAC THENL
   [MP_TAC(SPECL
      [`val(mulhi_s36:int64)`; `2 EXP 64`; `val(mullo_s36:int64)`;
       `val(sum_s34:int64) * c`] JOLT_FP128_NO_TOP_CARRY) THEN
    ASM_REWRITE_TAC[] THEN CONV_TAC NUM_REDUCE_CONV;
    ALL_TAC] THEN

  ABBREV_TAC `r = bignum_of_wordlist [sum_s38; sum_s39]` THEN
  ABBREV_TAC `u = bignum_of_wordlist [sum_s43; sum_s44]` THEN
  ABBREV_TAC `v = 2 EXP 128 * bitval carry_s39 + r` THEN
  SUBGOAL_THEN
   `2 EXP 128 * bitval carry_s39 + r =
    lo1 + val(mullo_s36:int64)`
  ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["r"; "lo1"] THEN
    REWRITE_TAC[bignum_of_wordlist; MULT_CLAUSES; ADD_CLAUSES] THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DECARRY_RULE) THEN
    DISCH_THEN(fun th -> REWRITE_TAC[th]) THEN
    ASM_REWRITE_TAC[] THEN REAL_ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN `v = lo1 + c * val(sum_s34:int64)` ASSUME_TAC THENL
   [EXPAND_TAC "v" THEN
    MP_TAC(ASSUME
     `2 EXP 128 * bitval carry_s39 + r =
      lo1 + val(mullo_s36:int64)`) THEN
    MP_TAC(ASSUME
     `val(mullo_s36:int64) = val(sum_s34:int64) * c`) THEN
    ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN `v < 2 * jolt_fp128_p c` ASSUME_TAC THENL
   [MATCH_MP_TAC(SPECL
     [`c:num`; `lo1:num`; `val(sum_s34:int64)`; `v:num`]
     JOLT_FP128_SECOND_FOLD_BOUND_GENERIC) THEN
    ASM_REWRITE_TAC[];
    ALL_TAC] THEN
  SUBGOAL_THEN `(m * n) MOD jolt_fp128_p c = v MOD jolt_fp128_p c`
  ASSUME_TAC THENL
   [ONCE_REWRITE_TAC[GSYM(ASSUME `mn4 = m * n`)] THEN
    MATCH_MP_TAC(SPECL
     [`c:num`; `mn4:num`; `t:num`; `v:num`; `hi0:num`; `lo0:num`;
      `val(sum_s34:int64)`; `lo1:num`] JOLT_FP128_TWO_FOLDS_GENERIC) THEN
    REPEAT CONJ_TAC THENL
     [ACCEPT_TAC(ASSUME `jolt_fp128_valid_offset c`);
      ACCEPT_TAC(ASSUME `mn4 = 2 EXP 128 * hi0 + lo0`);
      ACCEPT_TAC(ASSUME `t = lo0 + c * hi0`);
      ACCEPT_TAC(ASSUME
       `t = 2 EXP 128 * val(sum_s34:int64) + lo1`);
      ACCEPT_TAC(ASSUME
       `v = lo1 + c * val(sum_s34:int64)`)];
    ALL_TAC] THEN

  SUBGOAL_THEN
   `2 EXP 128 * bitval carry_s44 + u = r + c`
  ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["u"; "r"] THEN
    REWRITE_TAC[bignum_of_wordlist; MULT_CLAUSES; ADD_CLAUSES] THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DECARRY_RULE) THEN
    DISCH_THEN(fun th -> REWRITE_TAC[th]) THEN
    ASM_REWRITE_TAC[] THEN CONV_TAC REAL_RING;
    ALL_TAC] THEN
  ASM_REWRITE_TAC[] THEN
  DISCARD_STATE_TAC "s49" THEN
  ACCUMULATOR_POP_ASSUM_LIST(K ALL_TAC) THEN
  MATCH_MP_TAC EQ_TRANS THEN
  EXISTS_TAC `(if carry_s39 \/ carry_s44 then u else r):num` THEN
  CONJ_TAC THENL
   [MAP_EVERY EXPAND_TAC ["u"; "r"] THEN
    ASM_CASES_TAC `carry_s39:bool` THEN
    ASM_CASES_TAC `carry_s44:bool` THEN
    ASM_REWRITE_TAC
     [WORD_SUB_0; VAL_WORD_BITVAL; BITVAL_EQ_0; BITVAL_CLAUSES;
      COND_SWAP] THEN
    CONV_TAC WORD_REDUCE_CONV THEN CONV_TAC NUM_REDUCE_CONV THEN
    ASM_REWRITE_TAC[];
    ONCE_REWRITE_TAC[ASSUME
     `(m * n) MOD jolt_fp128_p c = v MOD jolt_fp128_p c`] THEN
    MATCH_MP_TAC(SPECL
     [`c:num`; `v:num`; `r:num`; `u:num`;
      `carry_s39:bool`; `carry_s44:bool`]
     JOLT_FP128_CANONICALIZE_GENERIC) THEN
    REPEAT CONJ_TAC THENL
     [ACCEPT_TAC(ASSUME `jolt_fp128_valid_offset c`);
      EXPAND_TAC "v" THEN ASM_REWRITE_TAC[];
      ASM_REWRITE_TAC[];
      ONCE_REWRITE_TAC[GSYM(ASSUME
      `v = lo1 + c * val(sum_s34:int64)`)] THEN
      ACCEPT_TAC(ASSUME `v < 2 * jolt_fp128_p c`);
      EXPAND_TAC "r" THEN MATCH_ACCEPT_TAC JOLT_FP128_WORDLIST2_BOUND;
      EXPAND_TAC "u" THEN MATCH_ACCEPT_TAC JOLT_FP128_WORDLIST2_BOUND]]);;

let JOLT_FP128_MUL_X86_64_CORRECT = time prove
 (`!a0 a1 b0 b1 pc.
        ensures x86
          (\s. bytes_loaded s (word pc) (BUTLAST jolt_fp128_mul_x86_64_mc) /\
               read RIP s = word pc /\
               read RDI s = a0 /\
               read RSI s = a1 /\
               read RDX s = b0 /\
               read RCX s = b1)
          (\s. read RIP s = word (pc + 0x9e) /\
               (bignum_of_wordlist [a0; a1] < jolt_fp128_a7f7_p /\
                bignum_of_wordlist [b0; b1] < jolt_fp128_a7f7_p
                ==> bignum_of_wordlist [read RAX s; read RDX s] =
                    (bignum_of_wordlist [a0; a1] *
                     bignum_of_wordlist [b0; b1]) MOD
                    jolt_fp128_a7f7_p))
          (MAYCHANGE [RIP; RAX; RCX; RDX; RDI; RSI; R8; R9; R10; R11] ,,
           MAYCHANGE SOME_FLAGS ,, MAYCHANGE [events])`,
  MAP_EVERY X_GEN_TAC
   [`a0:int64`; `a1:int64`; `b0:int64`; `b1:int64`; `pc:num`] THEN
  REWRITE_TAC[SOME_FLAGS] THEN
  ENSURES_SEQUENCE_TAC `pc + 0x6`
   `\s. bytes_loaded s (word pc) (BUTLAST jolt_fp128_mul_x86_64_mc) /\
        read RDI s = a0 /\ read RSI s = a1 /\
        read RDX s = b0 /\ read RCX s = b1 /\
        read R8 s = word 4294944759` THEN
  CONJ_TAC THENL
   [X86_SIM_TAC JOLT_FP128_MUL_X86_64_EXEC [1];
    MP_TAC(SPECL
     [`4294944759`; `a0:int64`; `a1:int64`; `b0:int64`; `b1:int64`; `pc:num`]
     JOLT_FP128_MUL_X86_64_GENERIC_CORRECT) THEN
    REWRITE_TAC[SOME_FLAGS; JOLT_FP128_OFFSETA7F7_VALID;
                JOLT_FP128_A7F7_P_GENERIC] THEN
    MATCH_MP_TAC(REWRITE_RULE[IMP_CONJ] ENSURES_PRECONDITION_THM) THEN
    SIMP_TAC[]]);;

let JOLT_FP128_MUL_X86_64_SUBROUTINE_CORRECT = time prove
 (`!a0 a1 b0 b1 pc stackpointer returnaddress.
        ensures x86
          (\s. bytes_loaded s (word pc) jolt_fp128_mul_x86_64_mc /\
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
                    (bignum_of_wordlist [a0; a1] *
                     bignum_of_wordlist [b0; b1]) MOD
                    jolt_fp128_a7f7_p))
          (MAYCHANGE [RSP] ,, MAYCHANGE_REGS_AND_FLAGS_PERMITTED_BY_ABI)`,
  X86_PROMOTE_RETURN_NOSTACK_TAC
    jolt_fp128_mul_x86_64_mc JOLT_FP128_MUL_X86_64_CORRECT);;
