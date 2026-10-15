(* Exact AArch64 correctness proof for scalar Fp64 multiplication. *)

needs (Filename.concat (Sys.getenv "JOLT_FP64_PROOF_DIR")
        "fp64_mul_object.ml");;

let JOLT_FP64_MUL_CORRECT = time prove
 (`!a b pc.
        ensures arm
          (\s. aligned_bytes_loaded s (word pc) jolt_fp64_mul_mc /\
               read PC s = word pc /\ read X0 s = a /\ read X1 s = b)
          (\s. read PC s = word (pc + 0x44) /\
               (val a < jolt_fp64_p /\ val b < jolt_fp64_p
                ==> val(read X0 s) = (val a * val b) MOD jolt_fp64_p))
          (MAYCHANGE [PC; X0; X8; X9; X10; X11; X12] ,,
           MAYCHANGE SOME_FLAGS ,, MAYCHANGE [events])`,
  MAP_EVERY X_GEN_TAC [`a:int64`; `b:int64`; `pc:num`] THEN
  REWRITE_TAC[SOME_FLAGS] THEN
  ENSURES_INIT_TAC "s0" THEN
  ARM_ACCSTEPS_TAC JOLT_FP64_MUL_EXEC (1--16) (1--17) THEN
  ENSURES_FINAL_STATE_TAC THEN ASM_REWRITE_TAC[] THEN STRIP_TAC THEN
  RULE_ASSUM_TAC(REWRITE_RULE[COND_SWAP; GSYM WORD_BITVAL]) THEN
  SUBGOAL_THEN
   `2 EXP 64 * val(mulhi_s1:int64) + val(mullo_s1:int64) =
    val(a:int64) * val(b:int64)`
  (LABEL_TAC "product") THENL
   [REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DESUM_RULE) THEN
    DISCH_THEN(fun th -> REWRITE_TAC[th]) THEN
    CONV_TAC REAL_RAT_REDUCE_CONV THEN CONV_TAC REAL_RING;
    ALL_TAC] THEN
  SUBGOAL_THEN `val(mulhi_s4:int64) < 59` ASSUME_TAC THENL
   [MP_TAC(SPEC `mulhi_s1:int64` VAL_BOUND_64) THEN
    MP_TAC(SPEC `val(mullo_s4:int64)` REAL_POS) THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DESUM_RULE) THEN
    CONV_TAC REAL_RAT_REDUCE_CONV THEN REAL_ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN
   `2 EXP 64 * val(sum_s8:int64) + val(sum_s7:int64) =
    val(mullo_s1:int64) + 59 * val(mulhi_s1:int64)`
  (LABEL_TAC "fold1") THENL
   [ASM_CASES_TAC `carry_s5:bool` THEN ASM_CASES_TAC `carry_s7:bool` THEN
    ASM_CASES_TAC `carry_s8:bool` THEN
    UNDISCH_TAC `val(mulhi_s4:int64) < 59` THEN
    MAP_EVERY (fun t -> MP_TAC(SPEC t VAL_BOUND_64))
     [`mulhi_s1:int64`; `mulhi_s4:int64`; `sum_s5:int64`; `sum_s8:int64`] THEN
    MAP_EVERY MP_TAC
     [SPEC `val(mulhi_s1:int64)` REAL_POS;
      SPEC `val(mulhi_s4:int64)` REAL_POS;
      SPEC `val(sum_s5:int64)` REAL_POS;
      SPEC `val(sum_s8:int64)` REAL_POS] THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DESUM_RULE) THEN
    ASM_REWRITE_TAC[BITVAL_CLAUSES] THEN
    CONV_TAC REAL_RAT_REDUCE_CONV THEN REAL_ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN `val(mulhi_s9:int64) < 59` ASSUME_TAC THENL
   [MP_TAC(SPEC `sum_s8:int64` VAL_BOUND_64) THEN
    MP_TAC(SPEC `val(mullo_s9:int64)` REAL_POS) THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DESUM_RULE) THEN
    CONV_TAC REAL_RAT_REDUCE_CONV THEN REAL_ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN
   `2 EXP 64 * val(sum_s13:int64) + val(sum_s12:int64) =
    val(sum_s7:int64) + 59 * val(sum_s8:int64)`
  (LABEL_TAC "fold2") THENL
   [ASM_CASES_TAC `carry_s10:bool` THEN ASM_CASES_TAC `carry_s12:bool` THEN
    ASM_CASES_TAC `carry_s13:bool` THEN
    UNDISCH_TAC `val(mulhi_s9:int64) < 59` THEN
    MAP_EVERY (fun t -> MP_TAC(SPEC t VAL_BOUND_64))
     [`sum_s8:int64`; `mulhi_s9:int64`; `sum_s10:int64`; `sum_s13:int64`] THEN
    MAP_EVERY MP_TAC
     [SPEC `val(sum_s8:int64)` REAL_POS;
      SPEC `val(mulhi_s9:int64)` REAL_POS;
      SPEC `val(sum_s10:int64)` REAL_POS;
      SPEC `val(sum_s13:int64)` REAL_POS] THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DESUM_RULE) THEN
    ASM_REWRITE_TAC[BITVAL_CLAUSES] THEN
    CONV_TAC REAL_RAT_REDUCE_CONV THEN REAL_ARITH_TAC;
    ALL_TAC] THEN

  SUBGOAL_THEN `val(sum_s8:int64) <= 59` (LABEL_TAC "fold1_bound") THENL
   [USE_THEN "fold1" MP_TAC THEN
    MAP_EVERY (fun t -> MP_TAC(SPEC t VAL_BOUND_64))
     [`sum_s7:int64`; `mullo_s1:int64`; `mulhi_s1:int64`] THEN
    CONV_TAC NUM_REDUCE_CONV THEN ARITH_TAC;
    ALL_TAC] THEN
  ABBREV_TAC `t = 2 EXP 64 * val(sum_s13:int64) + val(sum_s12:int64)` THEN
  SUBGOAL_THEN `t < 2 * jolt_fp64_p` (LABEL_TAC "reduced_bound") THENL
   [USE_THEN "fold2" MP_TAC THEN USE_THEN "fold1_bound" MP_TAC THEN
    MP_TAC(SPEC `sum_s7:int64` VAL_BOUND_64) THEN
    REWRITE_TAC[jolt_fp64_p] THEN CONV_TAC NUM_REDUCE_CONV THEN ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN
   `(val(a:int64) * val(b:int64)) MOD jolt_fp64_p = t MOD jolt_fp64_p`
  (LABEL_TAC "congruence") THENL
   [EXPAND_TAC "t" THEN
    MATCH_MP_TAC(SPECL
     [`val(a:int64) * val(b:int64)`;
      `val(mulhi_s1:int64)`; `val(mullo_s1:int64)`;
      `val(sum_s8:int64)`; `val(sum_s7:int64)`;
      `val(sum_s13:int64)`; `val(sum_s12:int64)`]
     JOLT_FP64_FOLD_TWICE) THEN
    REPEAT CONJ_TAC THENL
     [USE_THEN "product" MP_TAC THEN ARITH_TAC;
      USE_THEN "fold1" MP_TAC THEN ARITH_TAC;
      USE_THEN "fold2" MP_TAC THEN EXPAND_TAC "t" THEN ARITH_TAC];
    ALL_TAC] THEN
  USE_THEN "congruence" (fun th -> ONCE_REWRITE_TAC[th]) THEN
  SUBGOAL_THEN
   `2 EXP 64 * bitval carry_s14 + val(sum_s14:int64) =
    val(sum_s12:int64) + 59`
  (LABEL_TAC "tail14") THENL
   [ASM_CASES_TAC `carry_s14:bool` THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DESUM_RULE) THEN
    ASM_REWRITE_TAC[BITVAL_CLAUSES] THEN
    CONV_TAC REAL_RAT_REDUCE_CONV THEN REAL_ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN
   `2 EXP 64 * bitval carry_s15 + val(sum_s15:int64) =
    val(sum_s12:int64) + 59`
  (LABEL_TAC "tail15") THENL
   [ASM_CASES_TAC `carry_s15:bool` THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DESUM_RULE) THEN
    ASM_REWRITE_TAC[BITVAL_CLAUSES] THEN
    CONV_TAC REAL_RAT_REDUCE_CONV THEN REAL_ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN
   `2 EXP 64 * bitval carry_s16 + val(sum_s13:int64) =
    val(sum_s16:int64) + bitval(~carry_s15)`
  (LABEL_TAC "tail16") THENL
   [ASM_CASES_TAC `carry_s15:bool` THEN ASM_CASES_TAC `carry_s16:bool` THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DESUM_RULE) THEN
    ASM_REWRITE_TAC[BITVAL_CLAUSES] THEN
    CONV_TAC REAL_RAT_REDUCE_CONV THEN REAL_ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN
   `2 EXP 64 * bitval carry_s15 + val(sum_s14:int64) =
    val(sum_s12:int64) + 59`
  (LABEL_TAC "tail_correction") THENL
   [ASM_CASES_TAC `carry_s14:bool` THEN ASM_CASES_TAC `carry_s15:bool` THEN
    MAP_EVERY UNDISCH_TAC
     [`2 EXP 64 * bitval carry_s14 + val(sum_s14:int64) =
       val(sum_s12:int64) + 59`;
      `2 EXP 64 * bitval carry_s15 + val(sum_s15:int64) =
       val(sum_s12:int64) + 59`] THEN
    MAP_EVERY (fun t -> MP_TAC(SPEC t VAL_BOUND_64))
     [`sum_s14:int64`; `sum_s15:int64`] THEN
    ASM_REWRITE_TAC[BITVAL_CLAUSES] THEN CONV_TAC NUM_REDUCE_CONV THEN
    ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN
  `val(sum_s12:int64) < 2 EXP 64 /\ val(sum_s13:int64) < 2 EXP 64 /\
    val(sum_s14:int64) < 2 EXP 64 /\ val(sum_s15:int64) < 2 EXP 64 /\
    val(sum_s16:int64) < 2 EXP 64`
  STRIP_ASSUME_TAC THENL [BOUNDER_TAC[]; ALL_TAC] THEN
  EXPAND_TAC "t" THEN
  GEN_REWRITE_TAC LAND_CONV [COND_RAND] THEN
  MATCH_MP_TAC(SPECL
   [`val(sum_s13:int64)`; `val(sum_s12:int64)`;
    `val(sum_s14:int64)`; `val(sum_s16:int64)`;
    `carry_s15:bool`; `carry_s16:bool`]
  JOLT_FP64_FINAL_REDUCTION) THEN
  REPEAT CONJ_TAC THENL
   [UNDISCH_TAC
     `2 EXP 64 * val(sum_s13:int64) + val(sum_s12:int64) = t` THEN
    USE_THEN "reduced_bound" MP_TAC THEN ARITH_TAC;
    USE_THEN "tail_correction" ACCEPT_TAC;
    USE_THEN "tail16" ACCEPT_TAC;
    ASM_REWRITE_TAC[];
    ASM_REWRITE_TAC[];
    ASM_REWRITE_TAC[];
    ASM_REWRITE_TAC[]]);;

let JOLT_FP64_MUL_SUBROUTINE_CORRECT = time prove
 (`!a b pc returnaddress.
        ensures arm
          (\s. aligned_bytes_loaded s (word pc) jolt_fp64_mul_mc /\
               read PC s = word pc /\ read X30 s = returnaddress /\
               read X0 s = a /\ read X1 s = b)
          (\s. read PC s = returnaddress /\
               (val a < jolt_fp64_p /\ val b < jolt_fp64_p
                ==> val(read X0 s) = (val a * val b) MOD jolt_fp64_p))
          MAYCHANGE_REGS_AND_FLAGS_PERMITTED_BY_ABI`,
  ARM_ADD_RETURN_NOSTACK_TAC JOLT_FP64_MUL_EXEC JOLT_FP64_MUL_CORRECT);;
