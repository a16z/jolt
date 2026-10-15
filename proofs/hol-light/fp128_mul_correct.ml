(* Functional correctness proofs for Jolt's AArch64 Fp128 multiply kernel. *)

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

let JOLT_FP128_MUL_GENERIC_CORRECT = time prove
 (`!c a0 a1 b0 b1 pc.
      jolt_fp128_valid_offset c
      ==>
      ensures arm
          (\s. aligned_bytes_loaded s (word pc) jolt_fp128_mul_mc /\
               read PC s = word (pc + 0x4) /\
               read X0 s = a0 /\
               read X1 s = a1 /\
               read X2 s = b0 /\
               read X3 s = b1 /\
               read X4 s = word c)
          (\s. read PC s = word (pc + 0x90) /\
               (bignum_of_wordlist [a0; a1] < jolt_fp128_p c /\
                bignum_of_wordlist [b0; b1] < jolt_fp128_p c
                ==> bignum_of_wordlist [read X0 s; read X1 s] =
                    (bignum_of_wordlist [a0; a1] *
                     bignum_of_wordlist [b0; b1]) MOD jolt_fp128_p c))
          (MAYCHANGE [PC; X0; X1; X4; X5; X6; X7; X8; X9; X10; X11; X12] ,,
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
  ARM_ACCSTEPS_TAC JOLT_FP128_MUL_EXEC (2--36) (2--36) THEN
  ENSURES_FINAL_STATE_TAC THEN
  ASM_SIMP_TAC[VAL_WORD; DIMINDEX_64; MOD_LT] THEN STRIP_TAC THEN
  SUBGOAL_THEN `val(word c:int64) = c` ASSUME_TAC THENL
   [ASM_SIMP_TAC[VAL_WORD; DIMINDEX_64; MOD_LT]; ALL_TAC] THEN

  RULE_ASSUM_TAC(REWRITE_RULE[COND_SWAP; GSYM WORD_BITVAL]) THEN

  ABBREV_TAC
   `mn4 = bignum_of_wordlist [mullo_s2; sum_s16; sum_s17; sum_s18]` THEN
  SUBGOAL_THEN `m * n < 2 EXP 256` ASSUME_TAC THENL
   [REWRITE_TAC[ARITH_RULE `2 EXP 256 = 2 EXP 128 * 2 EXP 128`] THEN
    MATCH_MP_TAC LT_MULT2 THEN
    MAP_EVERY UNDISCH_TAC
     [`m < jolt_fp128_p c`; `n < jolt_fp128_p c`] THEN
    UNDISCH_TAC `jolt_fp128_valid_offset c` THEN
    REWRITE_TAC[jolt_fp128_valid_offset; jolt_fp128_p] THEN ARITH_TAC;
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
    ASM_REWRITE_TAC[] THEN REAL_ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN `mn4 = m * n` ASSUME_TAC THENL
   [MATCH_MP_TAC(SPECL
      [`bitval carry_s18`; `2 EXP 256`; `mn4:num`; `m * n`]
      JOLT_FP128_NO_TOP_CARRY) THEN
    REPEAT CONJ_TAC THENL
     [CONV_TAC NUM_REDUCE_CONV;
      ACCEPT_TAC(ASSUME
       `2 EXP 256 * bitval carry_s18 + mn4 = m * n`);
      ACCEPT_TAC(ASSUME `m * n < 2 EXP 256`)];
    ALL_TAC] THEN

  ABBREV_TAC `lo0 = bignum_of_wordlist [mullo_s2; sum_s16]` THEN
  ABBREV_TAC `hi0 = bignum_of_wordlist [sum_s17; sum_s18]` THEN
  SUBGOAL_THEN `mn4 = 2 EXP 128 * hi0 + lo0` ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["mn4"; "lo0"; "hi0"] THEN
    REWRITE_TAC[bignum_of_wordlist] THEN ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN `lo0 < 2 EXP 128 /\ hi0 < 2 EXP 128`
  STRIP_ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["lo0"; "hi0"] THEN
    REWRITE_TAC[bignum_of_wordlist] THEN BOUNDER_TAC[];
    ALL_TAC] THEN
  ABBREV_TAC `t = bignum_of_wordlist [sum_s23; sum_s26; sum_s27]` THEN
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
   `2 EXP 192 * bitval carry_s27 + t = lo0 + c * hi0`
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
    REWRITE_TAC[ASSUME `val(word c:int64) = c`; VAL_WORD_BITVAL] THEN
    CONV_TAC REAL_RAT_REDUCE_CONV THEN CONV_TAC REAL_RING;
    ALL_TAC] THEN
  SUBGOAL_THEN `t = lo0 + c * hi0` ASSUME_TAC THENL
   [MATCH_MP_TAC(SPECL
      [`bitval carry_s27`; `2 EXP 192`; `t:num`;
       `lo0 + c * hi0`] JOLT_FP128_NO_TOP_CARRY) THEN
    REPEAT CONJ_TAC THENL
     [CONV_TAC NUM_REDUCE_CONV;
      ACCEPT_TAC(ASSUME
       `2 EXP 192 * bitval carry_s27 + t = lo0 + c * hi0`);
      ACCEPT_TAC(ASSUME `lo0 + c * hi0 < 2 EXP 192`)];
    ALL_TAC] THEN

  ABBREV_TAC `lo1 = bignum_of_wordlist [sum_s23; sum_s26]` THEN
  SUBGOAL_THEN `lo1 < 2 EXP 128` ASSUME_TAC THENL
   [EXPAND_TAC "lo1" THEN
    REWRITE_TAC[bignum_of_wordlist] THEN BOUNDER_TAC[];
    ALL_TAC] THEN
  SUBGOAL_THEN `t = 2 EXP 128 * val(sum_s27:int64) + lo1` ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["t"; "lo1"] THEN
    REWRITE_TAC[bignum_of_wordlist] THEN ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN `c * hi0 < c * 2 EXP 128` ASSUME_TAC THENL
   [MATCH_MP_TAC LT_LMULT THEN CONJ_TAC THENL
     [UNDISCH_TAC `jolt_fp128_valid_offset c` THEN
      REWRITE_TAC[jolt_fp128_valid_offset] THEN ARITH_TAC;
      ACCEPT_TAC(ASSUME `hi0 < 2 EXP 128`)];
    ALL_TAC] THEN
  SUBGOAL_THEN `val(sum_s27:int64) <= c` ASSUME_TAC THENL
   [MAP_EVERY UNDISCH_TAC
     [`t = lo0 + c * hi0`;
      `t = 2 EXP 128 * val(sum_s27:int64) + lo1`;
      `lo0 < 2 EXP 128`; `lo1 < 2 EXP 128`;
      `c * hi0 < c * 2 EXP 128`] THEN
    CONV_TAC NUM_REDUCE_CONV THEN ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN `val(sum_s27:int64) * c < 2 EXP 64`
  ASSUME_TAC THENL
   [SUBGOAL_THEN `val(sum_s27:int64) < 2 EXP 32 /\ c < 2 EXP 32`
    STRIP_ASSUME_TAC THENL
     [UNDISCH_TAC `val(sum_s27:int64) <= c` THEN
      UNDISCH_TAC `jolt_fp128_valid_offset c` THEN
      REWRITE_TAC[jolt_fp128_valid_offset] THEN ARITH_TAC;
      ALL_TAC] THEN
    REWRITE_TAC[ARITH_RULE `2 EXP 64 = 2 EXP 32 * 2 EXP 32`] THEN
    MATCH_MP_TAC LT_MULT2 THEN ASM_REWRITE_TAC[];
    ALL_TAC] THEN
  SUBGOAL_THEN
   `2 EXP 64 * val(mulhi_s28:int64) + val(mullo_s28:int64) =
    val(sum_s27:int64) * c`
  ASSUME_TAC THENL
   [REWRITE_TAC[GSYM REAL_OF_NUM_EQ; GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DESUM_RULE) THEN
    DISCH_THEN(fun th -> REWRITE_TAC[th]) THEN
    ASM_REWRITE_TAC[] THEN
    CONV_TAC REAL_RAT_REDUCE_CONV THEN CONV_TAC REAL_RING;
    ALL_TAC] THEN
  SUBGOAL_THEN `val(mullo_s28:int64) = val(sum_s27:int64) * c`
  ASSUME_TAC THENL
   [MATCH_MP_TAC(SPECL
      [`val(mulhi_s28:int64)`; `2 EXP 64`; `val(mullo_s28:int64)`;
       `val(sum_s27:int64) * c`] JOLT_FP128_NO_TOP_CARRY) THEN
    REPEAT CONJ_TAC THENL
     [CONV_TAC NUM_REDUCE_CONV;
      ACCEPT_TAC(ASSUME
       `2 EXP 64 * val(mulhi_s28:int64) + val(mullo_s28:int64) =
        val(sum_s27:int64) * c`);
      ACCEPT_TAC(ASSUME `val(sum_s27:int64) * c < 2 EXP 64`)];
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
    DISCH_THEN(fun th -> REWRITE_TAC[th]) THEN
    ASM_REWRITE_TAC[] THEN REAL_ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN `v = lo1 + c * val(sum_s27:int64)` ASSUME_TAC THENL
   [EXPAND_TAC "v" THEN
    MP_TAC(ASSUME
     `2 EXP 128 * bitval carry_s30 + r =
      lo1 + val(mullo_s28:int64)`) THEN
    MP_TAC(ASSUME
     `val(mullo_s28:int64) = val(sum_s27:int64) * c`) THEN
    ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN `v < 2 * jolt_fp128_p c` ASSUME_TAC THENL
   [MATCH_MP_TAC(SPECL
     [`c:num`; `lo1:num`; `val(sum_s27:int64)`; `v:num`]
     JOLT_FP128_SECOND_FOLD_BOUND_GENERIC) THEN
    ASM_REWRITE_TAC[];
    ALL_TAC] THEN
  SUBGOAL_THEN `(m * n) MOD jolt_fp128_p c = v MOD jolt_fp128_p c`
  ASSUME_TAC THENL
   [ONCE_REWRITE_TAC[GSYM(ASSUME `mn4 = m * n`)] THEN
    MATCH_MP_TAC(SPECL
     [`c:num`; `mn4:num`; `t:num`; `v:num`; `hi0:num`; `lo0:num`;
      `val(sum_s27:int64)`; `lo1:num`] JOLT_FP128_TWO_FOLDS_GENERIC) THEN
    REPEAT CONJ_TAC THENL
     [ACCEPT_TAC(ASSUME `jolt_fp128_valid_offset c`);
      ACCEPT_TAC(ASSUME `mn4 = 2 EXP 128 * hi0 + lo0`);
      ACCEPT_TAC(ASSUME `t = lo0 + c * hi0`);
      ACCEPT_TAC(ASSUME
       `t = 2 EXP 128 * val(sum_s27:int64) + lo1`);
      ACCEPT_TAC(ASSUME
       `v = lo1 + c * val(sum_s27:int64)`)];
    ALL_TAC] THEN
  SUBGOAL_THEN `r < 2 EXP 128 /\ u < 2 EXP 128` STRIP_ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["r"; "u"] THEN BOUNDER_TAC[];
    ALL_TAC] THEN
  SUBGOAL_THEN
   `2 EXP 128 * bitval carry_s33 + u = r + c`
  ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["u"; "r"] THEN
    REWRITE_TAC[bignum_of_wordlist; MULT_CLAUSES; ADD_CLAUSES] THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DECARRY_RULE) THEN
    DISCH_THEN(fun th -> REWRITE_TAC[th]) THEN
    ASM_REWRITE_TAC[] THEN REAL_ARITH_TAC;
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
     `(m * n) MOD jolt_fp128_p c = v MOD jolt_fp128_p c`] THEN
    MATCH_MP_TAC(SPECL
     [`c:num`; `v:num`; `r:num`; `u:num`;
      `carry_s30:bool`; `carry_s33:bool`]
     JOLT_FP128_CANONICALIZE_GENERIC) THEN
    REPEAT CONJ_TAC THENL
     [ACCEPT_TAC(ASSUME `jolt_fp128_valid_offset c`);
      EXPAND_TAC "v" THEN REFL_TAC;
      ACCEPT_TAC(ASSUME
       `2 EXP 128 * bitval carry_s33 + u = r + c`);
      ACCEPT_TAC(ASSUME `v < 2 * jolt_fp128_p c`);
      ACCEPT_TAC(ASSUME `r < 2 EXP 128`);
      ACCEPT_TAC(ASSUME `u < 2 EXP 128`)]]);;

(*** The object fixture loads A7F7 before entering the generic body. This
     corollary preserves the whole-function theorem used by the ABI proof. ***)
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
  ENSURES_SEQUENCE_TAC `pc + 0x4`
   `\s. aligned_bytes_loaded s (word pc) jolt_fp128_mul_mc /\
        read X0 s = a0 /\ read X1 s = a1 /\
        read X2 s = b0 /\ read X3 s = b1 /\
        read X4 s = word 4294944759` THEN
  CONJ_TAC THENL
   [ARM_SIM_TAC JOLT_FP128_MUL_EXEC [1];
    MP_TAC(SPECL
     [`4294944759`; `a0:int64`; `a1:int64`; `b0:int64`; `b1:int64`; `pc:num`]
     JOLT_FP128_MUL_GENERIC_CORRECT) THEN
    REWRITE_TAC[SOME_FLAGS; JOLT_FP128_OFFSETA7F7_VALID;
                JOLT_FP128_A7F7_P_GENERIC] THEN
    MATCH_MP_TAC(REWRITE_RULE[IMP_CONJ] ENSURES_PRECONDITION_THM) THEN
    SIMP_TAC[]]);;

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
