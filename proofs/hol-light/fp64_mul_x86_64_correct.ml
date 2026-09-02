(* Exact baseline x86-64 correctness proof for scalar Fp64 multiplication. *)

needs (Filename.concat (Sys.getenv "JOLT_FP64_PROOF_DIR")
        "fp64_mul_x86_64_object.ml");;

let JOLT_FP64_MUL_X86_64_CORRECT = time prove
 (`!a b pc.
        ensures x86
          (\s. bytes_loaded s (word pc) (BUTLAST jolt_fp64_mul_x86_64_mc) /\
               read RIP s = word pc /\ read RDI s = a /\ read RSI s = b)
          (\s. read RIP s = word (pc + 0x45) /\
               (val a < jolt_fp64_p /\ val b < jolt_fp64_p
                ==> val(read RAX s) = (val a * val b) MOD jolt_fp64_p))
          (MAYCHANGE [RIP; RAX; RCX; RDX; RDI; RSI; R8; R9] ,,
           MAYCHANGE SOME_FLAGS ,, MAYCHANGE [events])`,
  MAP_EVERY X_GEN_TAC [`a:int64`; `b:int64`; `pc:num`] THEN
  REWRITE_TAC[SOME_FLAGS] THEN ENSURES_INIT_TAC "s0" THEN
  X86_ACCSTEPS_TAC JOLT_FP64_MUL_X86_64_EXEC
   [2;7;10;11;12;14;15;16;17;19;20] (1--21) THEN
  ENSURES_FINAL_STATE_TAC THEN ASM_REWRITE_TAC[] THEN STRIP_TAC THEN

  SUBGOAL_THEN
   `2 EXP 64 * val(mulhi_s2:int64) + val(mullo_s2:int64) =
    val(a:int64) * val(b:int64)`
  (LABEL_TAC "product") THENL
   [REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DESUM_RULE) THEN
    DISCH_THEN(fun th -> REWRITE_TAC[th]) THEN CONV_TAC REAL_RING;
    ALL_TAC] THEN
  SUBGOAL_THEN `val(mulhi_s7:int64) < 59` ASSUME_TAC THENL
   [MP_TAC(SPEC `mulhi_s2:int64` VAL_BOUND_64) THEN
    MP_TAC(SPEC `val(mullo_s7:int64)` REAL_POS) THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DESUM_RULE) THEN
    CONV_TAC REAL_RAT_REDUCE_CONV THEN REAL_ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN
   `2 EXP 64 * val(sum_s12:int64) + val(sum_s11:int64) =
    val(mullo_s2:int64) + 59 * val(mulhi_s2:int64)`
  (LABEL_TAC "fold1") THENL
   [ASM_CASES_TAC `carry_s10:bool` THEN
    ASM_CASES_TAC `carry_s11:bool` THEN
    ASM_CASES_TAC `carry_s12:bool` THEN
    UNDISCH_TAC `val(mulhi_s7:int64) < 59` THEN
    MAP_EVERY (fun t -> MP_TAC(SPEC t VAL_BOUND_64))
     [`mulhi_s2:int64`; `mulhi_s7:int64`; `sum_s10:int64`;
      `sum_s11:int64`; `sum_s12:int64`] THEN
    MAP_EVERY MP_TAC
     [SPEC `val(mulhi_s2:int64)` REAL_POS;
      SPEC `val(mulhi_s7:int64)` REAL_POS;
      SPEC `val(sum_s10:int64)` REAL_POS;
      SPEC `val(sum_s11:int64)` REAL_POS;
      SPEC `val(sum_s12:int64)` REAL_POS] THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DESUM_RULE) THEN
    ASM_REWRITE_TAC[BITVAL_CLAUSES] THEN
    CONV_TAC REAL_RAT_REDUCE_CONV THEN REAL_ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN `val(mulhi_s14:int64) < 59` ASSUME_TAC THENL
   [MP_TAC(SPEC `sum_s12:int64` VAL_BOUND_64) THEN
    MP_TAC(SPEC `val(mullo_s14:int64)` REAL_POS) THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DESUM_RULE) THEN
    CONV_TAC REAL_RAT_REDUCE_CONV THEN REAL_ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN
   `2 EXP 64 * val(sum_s17:int64) + val(sum_s16:int64) =
    val(sum_s11:int64) + 59 * val(sum_s12:int64)`
  (LABEL_TAC "fold2") THENL
   [ASM_CASES_TAC `carry_s15:bool` THEN
    ASM_CASES_TAC `carry_s16:bool` THEN
    ASM_CASES_TAC `carry_s17:bool` THEN
    UNDISCH_TAC `val(mulhi_s14:int64) < 59` THEN
    MAP_EVERY (fun t -> MP_TAC(SPEC t VAL_BOUND_64))
     [`sum_s12:int64`; `mulhi_s14:int64`; `sum_s15:int64`;
      `sum_s16:int64`; `sum_s17:int64`] THEN
    MAP_EVERY MP_TAC
     [SPEC `val(sum_s12:int64)` REAL_POS;
      SPEC `val(mulhi_s14:int64)` REAL_POS;
      SPEC `val(sum_s15:int64)` REAL_POS;
      SPEC `val(sum_s16:int64)` REAL_POS;
      SPEC `val(sum_s17:int64)` REAL_POS] THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DESUM_RULE) THEN
    ASM_REWRITE_TAC[BITVAL_CLAUSES] THEN
    CONV_TAC REAL_RAT_REDUCE_CONV THEN REAL_ARITH_TAC;
    ALL_TAC] THEN

  SUBGOAL_THEN `val(sum_s12:int64) <= 59` (LABEL_TAC "fold1_bound") THENL
   [USE_THEN "fold1" MP_TAC THEN
    MAP_EVERY (fun t -> MP_TAC(SPEC t VAL_BOUND_64))
     [`sum_s11:int64`; `mullo_s2:int64`; `mulhi_s2:int64`] THEN
    CONV_TAC NUM_REDUCE_CONV THEN ARITH_TAC;
    ALL_TAC] THEN
  ABBREV_TAC `t = 2 EXP 64 * val(sum_s17:int64) + val(sum_s16:int64)` THEN
  SUBGOAL_THEN `t < 2 * jolt_fp64_p` (LABEL_TAC "reduced_bound") THENL
   [USE_THEN "fold2" MP_TAC THEN USE_THEN "fold1_bound" MP_TAC THEN
    MP_TAC(SPEC `sum_s11:int64` VAL_BOUND_64) THEN
    REWRITE_TAC[jolt_fp64_p] THEN CONV_TAC NUM_REDUCE_CONV THEN ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN
   `(val(a:int64) * val(b:int64)) MOD jolt_fp64_p = t MOD jolt_fp64_p`
  (LABEL_TAC "congruence") THENL
   [EXPAND_TAC "t" THEN
    MATCH_MP_TAC(SPECL
     [`val(a:int64) * val(b:int64)`;
      `val(mulhi_s2:int64)`; `val(mullo_s2:int64)`;
      `val(sum_s12:int64)`; `val(sum_s11:int64)`;
      `val(sum_s17:int64)`; `val(sum_s16:int64)`]
     JOLT_FP64_FOLD_TWICE) THEN
    REPEAT CONJ_TAC THENL
      [USE_THEN "product" MP_TAC THEN ARITH_TAC;
      USE_THEN "fold1" MP_TAC THEN ARITH_TAC;
      USE_THEN "fold2" MP_TAC THEN EXPAND_TAC "t" THEN ARITH_TAC];
    ALL_TAC] THEN
  USE_THEN "congruence" (fun th -> ONCE_REWRITE_TAC[th]) THEN

  SUBGOAL_THEN
   `2 EXP 64 * bitval(~carry_s19) + val(sum_s19:int64) =
    val(sum_s16:int64) + 59`
  (LABEL_TAC "tail19") THENL
   [REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    MATCH_MP_TAC JOLT_FP64_X86_ADD59_FROM_SUBP THEN
    REWRITE_TAC[jolt_fp64_p] THEN
    ACCUMULATOR_ASSUM_LIST(fun theorems -> ACCEPT_TAC(el 1 theorems));
    ALL_TAC] THEN
  SUBGOAL_THEN
   `2 EXP 64 * bitval carry_s20 + val(sum_s17:int64) =
    bitval carry_s19 + val(sum_s20:int64)`
  (LABEL_TAC "tail20") THENL
   [REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    MATCH_MP_TAC JOLT_FP64_X86_COMPARE_REORIENT THEN
    ACCUMULATOR_ASSUM_LIST
     (fun theorems ->
       ACCEPT_TAC(REWRITE_RULE[REAL_SUB_RZERO] (hd theorems)));
    ALL_TAC] THEN
  SUBGOAL_THEN
   `val(sum_s16:int64) < 2 EXP 64 /\ val(sum_s17:int64) < 2 EXP 64 /\
    val(sum_s19:int64) < 2 EXP 64 /\ val(sum_s20:int64) < 2 EXP 64`
  STRIP_ASSUME_TAC THENL [BOUNDER_TAC[]; ALL_TAC] THEN
  EXPAND_TAC "t" THEN
  GEN_REWRITE_TAC LAND_CONV [COND_RAND] THEN
  GEN_REWRITE_TAC LAND_CONV [COND_SWAP] THEN
  MATCH_MP_TAC(SPECL
   [`val(sum_s17:int64)`; `val(sum_s16:int64)`;
    `val(sum_s19:int64)`; `val(sum_s20:int64)`;
    `~carry_s19`; `carry_s20:bool`] JOLT_FP64_FINAL_REDUCTION) THEN
  REPEAT CONJ_TAC THENL
   [USE_THEN "reduced_bound" MP_TAC THEN EXPAND_TAC "t" THEN ARITH_TAC;
    USE_THEN "tail19" ACCEPT_TAC;
    USE_THEN "tail20" MP_TAC THEN ARITH_TAC;
    ASM_REWRITE_TAC[];
    ASM_REWRITE_TAC[];
    ASM_REWRITE_TAC[];
    ASM_REWRITE_TAC[]]);;

let JOLT_FP64_MUL_X86_64_SUBROUTINE_CORRECT = time prove
 (`!a b pc stackpointer returnaddress.
        ensures x86
          (\s. bytes_loaded s (word pc) jolt_fp64_mul_x86_64_mc /\
               read RIP s = word pc /\ read RSP s = stackpointer /\
               read (memory :> bytes64 stackpointer) s = returnaddress /\
               read RDI s = a /\ read RSI s = b)
          (\s. read RIP s = returnaddress /\
               read RSP s = word_add stackpointer (word 8) /\
               (val a < jolt_fp64_p /\ val b < jolt_fp64_p
                ==> val(read RAX s) = (val a * val b) MOD jolt_fp64_p))
          (MAYCHANGE [RSP] ,, MAYCHANGE_REGS_AND_FLAGS_PERMITTED_BY_ABI)`,
  X86_PROMOTE_RETURN_NOSTACK_TAC
    jolt_fp64_mul_x86_64_mc JOLT_FP64_MUL_X86_64_CORRECT);;
