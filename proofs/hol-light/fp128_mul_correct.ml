(* Functional correctness proof for the fixed A7F7 AArch64 multiply kernel. *)

needs (Filename.concat (Sys.getenv "JOLT_FP128_PROOF_DIR")
        "fp128_mul_object.ml");;

(* CSET writes a 32-bit zero or one before the value is widened to 64 bits.
   CINC adds a second carry bit. These lemmas expose those machine-level
   expressions as ordinary natural-number carry values. *)
let JOLT_FP128_VAL_CSET = prove
 (`!b. val(word_zx(word(bitval b):(32)word):(64)word) = bitval b`,
  GEN_TAC THEN BOOL_CASES_TAC `b:bool` THEN
  REWRITE_TAC[BITVAL_CLAUSES] THEN CONV_TAC WORD_REDUCE_CONV);;

let JOLT_FP128_VAL_CINC = prove
 (`!b c.
      (&(val((if c
              then word_add (word_zx(word(bitval b):(32)word)) (word 1)
              else word_zx(word(bitval b):(32)word)):(64)word)):real) =
      &(bitval b) + &(bitval c)`,
  REPEAT GEN_TAC THEN BOOL_CASES_TAC `b:bool` THEN
  BOOL_CASES_TAC `c:bool` THEN REWRITE_TAC[BITVAL_CLAUSES] THEN
  CONV_TAC WORD_REDUCE_CONV THEN ARITH_TAC);;

let JOLT_FP128_MUL_CORRECT = time prove
 (`!a0 a1 b0 b1 pc.
        ensures arm
          (\s. aligned_bytes_loaded s (word pc) jolt_fp128_mul_mc /\
               read PC s = word pc /\
               read X0 s = a0 /\
               read X1 s = a1 /\
               read X2 s = b0 /\
               read X3 s = b1)
          (\s. read PC s = word (pc + 0x90) /\
               (bignum_of_wordlist [a0; a1] < jolt_fp128_a7f7_p /\
                bignum_of_wordlist [b0; b1] < jolt_fp128_a7f7_p
                ==> bignum_of_wordlist [read X0 s; read X1 s] =
                    (bignum_of_wordlist [a0; a1] *
                     bignum_of_wordlist [b0; b1]) MOD
                    jolt_fp128_a7f7_p))
          (MAYCHANGE [PC; X0; X1; X4; X5; X6; X7; X8; X9; X10; X11; X12] ,,
           MAYCHANGE SOME_FLAGS ,, MAYCHANGE [events])`,
  MAP_EVERY X_GEN_TAC
   [`a0:int64`; `a1:int64`; `b0:int64`; `b1:int64`; `pc:num`] THEN
  REWRITE_TAC[SOME_FLAGS] THEN
  ABBREV_TAC `m = bignum_of_wordlist [a0; a1]` THEN
  ABBREV_TAC `n = bignum_of_wordlist [b0; b1]` THEN
  ENSURES_INIT_TAC "s0" THEN
  ARM_ACCSTEPS_TAC JOLT_FP128_MUL_EXEC (1--36) (1--36) THEN
  ENSURES_FINAL_STATE_TAC THEN ASM_REWRITE_TAC[] THEN STRIP_TAC THEN

  RULE_ASSUM_TAC(REWRITE_RULE[COND_SWAP; GSYM WORD_BITVAL]) THEN

  ABBREV_TAC
   `mn4 = bignum_of_wordlist [mullo_s2; sum_s16; sum_s17; sum_s18]` THEN
  SUBGOAL_THEN `m * n < 2 EXP 256` ASSUME_TAC THENL
   [REWRITE_TAC[ARITH_RULE `2 EXP 256 = 2 EXP 128 * 2 EXP 128`] THEN
    MATCH_MP_TAC LT_MULT2 THEN
    MAP_EVERY UNDISCH_TAC
     [`m < jolt_fp128_a7f7_p`; `n < jolt_fp128_a7f7_p`] THEN
    REWRITE_TAC[jolt_fp128_a7f7_p] THEN ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN
   `2 EXP 256 * bitval carry_s18 + mn4 = m * n`
  ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["mn4"; "m"; "n"] THEN
    REWRITE_TAC[bignum_of_wordlist; GSYM REAL_OF_NUM_CLAUSES;
                REAL_MUL_RZERO; REAL_ADD_RID] THEN
    CONV_TAC(RAND_CONV REAL_POLY_CONV) THEN
    ACCUMULATOR_ASSUM_LIST(fun thl ->
      MP_TAC(end_itlist CONJ
       (map (REWRITE_RULE
          [COND_SWAP; GSYM WORD_BITVAL; JOLT_FP128_VAL_CSET;
           JOLT_FP128_VAL_CINC]) (DESUM_RULE thl)))) THEN
    DISCH_THEN(fun th -> REWRITE_TAC[th]) THEN
    CONV_TAC REAL_RAT_REDUCE_CONV THEN CONV_TAC REAL_RING;
    ALL_TAC] THEN
  SUBGOAL_THEN `mn4 = m * n` ASSUME_TAC THENL
   [MP_TAC(SPECL
      [`bitval carry_s18`; `2 EXP 256`; `mn4:num`; `m * n`]
      JOLT_FP128_NO_TOP_CARRY) THEN ASM_REWRITE_TAC[] THEN
    CONV_TAC NUM_REDUCE_CONV;
    ALL_TAC] THEN

  ABBREV_TAC `lo0 = bignum_of_wordlist [mullo_s2; sum_s16]` THEN
  ABBREV_TAC `hi0 = bignum_of_wordlist [sum_s17; sum_s18]` THEN
  SUBGOAL_THEN `mn4 = 2 EXP 128 * hi0 + lo0` ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["mn4"; "lo0"; "hi0"] THEN
    REWRITE_TAC[bignum_of_wordlist] THEN ARITH_TAC;
    ALL_TAC] THEN
  ABBREV_TAC `t = bignum_of_wordlist [sum_s23; sum_s26; sum_s27]` THEN
  SUBGOAL_THEN `lo0 + 4294944759 * hi0 < 2 EXP 192` ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["lo0"; "hi0"] THEN
    REWRITE_TAC[bignum_of_wordlist] THEN BOUNDER_TAC[];
    ALL_TAC] THEN
  SUBGOAL_THEN
   `2 EXP 192 * bitval carry_s27 + t = lo0 + 4294944759 * hi0`
  ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["t"; "lo0"; "hi0"] THEN
    REWRITE_TAC[bignum_of_wordlist; GSYM REAL_OF_NUM_CLAUSES;
                REAL_MUL_RZERO; REAL_ADD_RID] THEN
    CONV_TAC(RAND_CONV REAL_POLY_CONV) THEN
    ACCUMULATOR_ASSUM_LIST(fun thl ->
      MP_TAC(end_itlist CONJ
       (map (REWRITE_RULE
          [COND_SWAP; GSYM WORD_BITVAL; JOLT_FP128_VAL_CSET;
           JOLT_FP128_VAL_CINC]) (DESUM_RULE thl)))) THEN
    DISCH_THEN(fun th -> REWRITE_TAC[th]) THEN
    CONV_TAC REAL_RAT_REDUCE_CONV THEN CONV_TAC REAL_RING;
    ALL_TAC] THEN
  SUBGOAL_THEN `t = lo0 + 4294944759 * hi0` ASSUME_TAC THENL
   [MP_TAC(SPECL
      [`bitval carry_s27`; `2 EXP 192`; `t:num`;
       `lo0 + 4294944759 * hi0`] JOLT_FP128_NO_TOP_CARRY) THEN
    ASM_REWRITE_TAC[] THEN CONV_TAC NUM_REDUCE_CONV;
    ALL_TAC] THEN

  ABBREV_TAC `lo1 = bignum_of_wordlist [sum_s23; sum_s26]` THEN
  SUBGOAL_THEN
   `lo0 < 2 EXP 128 /\ hi0 < 2 EXP 128 /\ lo1 < 2 EXP 128`
  STRIP_ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["lo0"; "hi0"; "lo1"] THEN
    REWRITE_TAC[bignum_of_wordlist] THEN BOUNDER_TAC[];
    ALL_TAC] THEN
  SUBGOAL_THEN `t = 2 EXP 128 * val(sum_s27:int64) + lo1` ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["t"; "lo1"] THEN
    REWRITE_TAC[bignum_of_wordlist] THEN ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN `val(sum_s27:int64) <= 4294944759` ASSUME_TAC THENL
   [MAP_EVERY UNDISCH_TAC
     [`t = lo0 + 4294944759 * hi0`;
      `t = 2 EXP 128 * val(sum_s27:int64) + lo1`;
      `lo0 < 2 EXP 128`; `hi0 < 2 EXP 128`; `lo1 < 2 EXP 128`] THEN
    CONV_TAC NUM_REDUCE_CONV THEN ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN `val(sum_s27:int64) * 4294944759 < 2 EXP 64`
  ASSUME_TAC THENL
   [UNDISCH_TAC `val(sum_s27:int64) <= 4294944759` THEN
    CONV_TAC NUM_REDUCE_CONV THEN ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN
   `2 EXP 64 * val(mulhi_s28:int64) + val(mullo_s28:int64) =
    val(sum_s27:int64) * 4294944759`
  ASSUME_TAC THENL
   [REWRITE_TAC[GSYM REAL_OF_NUM_EQ; GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DESUM_RULE) THEN
    DISCH_THEN(fun th -> REWRITE_TAC[th]) THEN
    CONV_TAC REAL_RAT_REDUCE_CONV THEN CONV_TAC REAL_RING;
    ALL_TAC] THEN
  SUBGOAL_THEN `val(mullo_s28:int64) = val(sum_s27:int64) * 4294944759`
  ASSUME_TAC THENL
   [MP_TAC(SPECL
      [`val(mulhi_s28:int64)`; `2 EXP 64`; `val(mullo_s28:int64)`;
       `val(sum_s27:int64) * 4294944759`] JOLT_FP128_NO_TOP_CARRY) THEN
    ASM_REWRITE_TAC[] THEN CONV_TAC NUM_REDUCE_CONV;
    ALL_TAC] THEN

  ABBREV_TAC `r = bignum_of_wordlist [sum_s29; sum_s30]` THEN
  ABBREV_TAC `u = bignum_of_wordlist [sum_s32; sum_s33]` THEN
  ABBREV_TAC `v = 2 EXP 128 * bitval carry_s30 + r` THEN
  SUBGOAL_THEN
   `2 EXP 128 * bitval carry_s30 + r =
    lo1 + val(mullo_s28:int64)`
  ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["r"; "lo1"] THEN
    REWRITE_TAC[bignum_of_wordlist; MULT_CLAUSES; ADD_CLAUSES] THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DECARRY_RULE) THEN
    DISCH_THEN(fun th -> REWRITE_TAC[th]) THEN REAL_ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN `v = lo1 + 4294944759 * val(sum_s27:int64)` ASSUME_TAC THENL
   [EXPAND_TAC "v" THEN
    MP_TAC(ASSUME
     `2 EXP 128 * bitval carry_s30 + r =
      lo1 + val(mullo_s28:int64)`) THEN
    MP_TAC(ASSUME
     `val(mullo_s28:int64) = val(sum_s27:int64) * 4294944759`) THEN
    ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN `v < 2 * jolt_fp128_a7f7_p` ASSUME_TAC THENL
   [MATCH_MP_TAC(SPECL
     [`lo1:num`; `val(sum_s27:int64)`; `v:num`]
     JOLT_FP128_SECOND_FOLD_BOUND) THEN
    REPEAT CONJ_TAC THENL
     [ACCEPT_TAC(ASSUME `lo1 < 2 EXP 128`);
      ACCEPT_TAC(ASSUME `val(sum_s27:int64) <= 4294944759`);
      ACCEPT_TAC(ASSUME
       `v = lo1 + 4294944759 * val(sum_s27:int64)`)];
    ALL_TAC] THEN
  SUBGOAL_THEN `(m * n) MOD jolt_fp128_a7f7_p =
                v MOD jolt_fp128_a7f7_p`
  ASSUME_TAC THENL
   [ONCE_REWRITE_TAC[GSYM(ASSUME `mn4 = m * n`)] THEN
    MATCH_MP_TAC(SPECL
     [`mn4:num`; `t:num`; `v:num`; `hi0:num`; `lo0:num`;
      `val(sum_s27:int64)`; `lo1:num`] JOLT_FP128_TWO_FOLDS) THEN
    REPEAT CONJ_TAC THENL
     [ACCEPT_TAC(ASSUME `mn4 = 2 EXP 128 * hi0 + lo0`);
      ACCEPT_TAC(ASSUME `t = lo0 + 4294944759 * hi0`);
      ACCEPT_TAC(ASSUME
       `t = 2 EXP 128 * val(sum_s27:int64) + lo1`);
      ACCEPT_TAC(ASSUME
       `v = lo1 + 4294944759 * val(sum_s27:int64)`)];
    ALL_TAC] THEN
  SUBGOAL_THEN `r < 2 EXP 128 /\ u < 2 EXP 128` STRIP_ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["r"; "u"] THEN BOUNDER_TAC[];
    ALL_TAC] THEN
  SUBGOAL_THEN
   `2 EXP 128 * bitval carry_s33 + u = r + 4294944759`
  ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["u"; "r"] THEN
    REWRITE_TAC[bignum_of_wordlist; MULT_CLAUSES; ADD_CLAUSES] THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DECARRY_RULE) THEN
    DISCH_THEN(fun th -> REWRITE_TAC[th]) THEN REAL_ARITH_TAC;
    ALL_TAC] THEN
  DISCARD_STATE_TAC "s36" THEN
  ACCUMULATOR_POP_ASSUM_LIST(K ALL_TAC) THEN
  MATCH_MP_TAC EQ_TRANS THEN
  EXISTS_TAC `(if carry_s30 \/ carry_s33 then u else r):num` THEN
  CONJ_TAC THENL
   [MAP_EVERY EXPAND_TAC ["u"; "r"] THEN
    ASM_CASES_TAC `carry_s30:bool` THEN
    ASM_CASES_TAC `carry_s33:bool` THEN
    ASM_REWRITE_TAC
     [WORD_SUB_0; VAL_WORD_BITVAL; BITVAL_EQ_0; BITVAL_CLAUSES;
      COND_SWAP] THEN
    CONV_TAC WORD_REDUCE_CONV THEN CONV_TAC NUM_REDUCE_CONV THEN
    ASM_REWRITE_TAC[];
    ONCE_REWRITE_TAC[ASSUME
     `(m * n) MOD jolt_fp128_a7f7_p = v MOD jolt_fp128_a7f7_p`] THEN
    MATCH_MP_TAC(SPECL
     [`v:num`; `r:num`; `u:num`; `carry_s30:bool`; `carry_s33:bool`]
     JOLT_FP128_CANONICALIZE) THEN
    REPEAT CONJ_TAC THENL
     [EXPAND_TAC "v" THEN REFL_TAC;
      ACCEPT_TAC(ASSUME
       `2 EXP 128 * bitval carry_s33 + u = r + 4294944759`);
      ACCEPT_TAC(ASSUME `v < 2 * jolt_fp128_a7f7_p`);
      ACCEPT_TAC(ASSUME `r < 2 EXP 128`);
      ACCEPT_TAC(ASSUME `u < 2 EXP 128`)]]);;

let JOLT_FP128_MUL_SUBROUTINE_CORRECT = time prove
 (`!a0 a1 b0 b1 pc returnaddress.
        ensures arm
          (\s. aligned_bytes_loaded s (word pc) jolt_fp128_mul_mc /\
               read PC s = word pc /\
               read X30 s = returnaddress /\
               read X0 s = a0 /\
               read X1 s = a1 /\
               read X2 s = b0 /\
               read X3 s = b1)
          (\s. read PC s = returnaddress /\
               (bignum_of_wordlist [a0; a1] < jolt_fp128_a7f7_p /\
                bignum_of_wordlist [b0; b1] < jolt_fp128_a7f7_p
                ==> bignum_of_wordlist [read X0 s; read X1 s] =
                    (bignum_of_wordlist [a0; a1] *
                     bignum_of_wordlist [b0; b1]) MOD
                    jolt_fp128_a7f7_p))
          MAYCHANGE_REGS_AND_FLAGS_PERMITTED_BY_ABI`,
  ARM_ADD_RETURN_NOSTACK_TAC
    JOLT_FP128_MUL_EXEC JOLT_FP128_MUL_CORRECT);;
