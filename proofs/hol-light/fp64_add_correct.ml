(* Exact AArch64 correctness proof for scalar Fp64 addition. *)

needs (Filename.concat (Sys.getenv "JOLT_FP64_PROOF_DIR")
        "fp64_add_object.ml");;

let JOLT_FP64_ADD_CORRECT = time prove
 (`!a b pc.
        ensures arm
          (\s. aligned_bytes_loaded s (word pc) jolt_fp64_add_mc /\
               read PC s = word pc /\ read X0 s = a /\ read X1 s = b)
          (\s. read PC s = word (pc + 0x1c) /\
               (val a < jolt_fp64_p /\ val b < jolt_fp64_p
                ==> val(read X0 s) = (val a + val b) MOD jolt_fp64_p))
          (MAYCHANGE [PC; X0; X8; X9] ,,
           MAYCHANGE SOME_FLAGS ,, MAYCHANGE [events])`,
  MAP_EVERY X_GEN_TAC [`a:int64`; `b:int64`; `pc:num`] THEN
  REWRITE_TAC[SOME_FLAGS] THEN
  ENSURES_INIT_TAC "s0" THEN
  ARM_ACCSTEPS_TAC JOLT_FP64_ADD_EXEC [1] (1--1) THEN
  SUBGOAL_THEN
   `2 EXP 64 * bitval carry_s1 + val(sum_s1:int64) =
    val(a:int64) + val(b:int64)`
  (LABEL_TAC "sum1") THENL
   [REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DECARRY_RULE) THEN
    DISCH_THEN(fun th -> REWRITE_TAC[th]) THEN REAL_ARITH_TAC;
    ALL_TAC] THEN
  ACCUMULATOR_POP_ASSUM_LIST(K ALL_TAC) THEN
  ARM_STEPS_TAC JOLT_FP64_ADD_EXEC (2--3) THEN
  ARM_ACCSTEPS_TAC JOLT_FP64_ADD_EXEC [4] (4--4) THEN
  SUBGOAL_THEN
   `2 EXP 64 * bitval carry_s4 + val(sum_s4:int64) =
    val(sum_s1:int64) + 59 * bitval carry_s1`
  (LABEL_TAC "sum4") THENL
   [REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DECARRY_RULE) THEN
    DISCH_THEN(fun th -> REWRITE_TAC[th]) THEN
    COND_CASES_TAC THEN ASM_REWRITE_TAC[BITVAL_CLAUSES] THEN
    CONV_TAC(DEPTH_CONV WORD_NUM_RED_CONV) THEN
    CONV_TAC NUM_REDUCE_CONV THEN REAL_ARITH_TAC;
    ALL_TAC] THEN
  ACCUMULATOR_POP_ASSUM_LIST(K ALL_TAC) THEN
  ARM_ACCSTEPS_TAC JOLT_FP64_ADD_EXEC [5] (5--5) THEN
  SUBGOAL_THEN
   `2 EXP 64 * bitval carry_s5 + val(sum_s5:int64) =
    val(sum_s4:int64) + 59`
  (LABEL_TAC "sum5") THENL
   [REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DECARRY_RULE) THEN
    DISCH_THEN(fun th -> REWRITE_TAC[th]) THEN
    CONV_TAC(DEPTH_CONV WORD_NUM_RED_CONV) THEN
    CONV_TAC NUM_REDUCE_CONV THEN REAL_ARITH_TAC;
    ALL_TAC] THEN
  ACCUMULATOR_POP_ASSUM_LIST(K ALL_TAC) THEN
  ARM_STEPS_TAC JOLT_FP64_ADD_EXEC (6--7) THEN
  ENSURES_FINAL_STATE_TAC THEN ASM_REWRITE_TAC[] THEN STRIP_TAC THEN
  SUBGOAL_THEN
   `val(sum_s1:int64) < 2 EXP 64 /\
    val(sum_s4:int64) < 2 EXP 64 /\
    val(sum_s5:int64) < 2 EXP 64`
  STRIP_ASSUME_TAC THENL [BOUNDER_TAC[]; ALL_TAC] THEN
  DISCARD_STATE_TAC "s7" THEN
  RULE_ASSUM_TAC(REWRITE_RULE[jolt_fp64_p]) THEN
  REWRITE_TAC[jolt_fp64_p] THEN
  (ASM_CASES_TAC `carry_s1:bool` THENL
    [RULE_ASSUM_TAC(REWRITE_RULE[ASSUME `carry_s1:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `carry_s1:bool`);
     RULE_ASSUM_TAC(REWRITE_RULE[ASSUME `~carry_s1:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `~carry_s1:bool`)]) THEN
  (ASM_CASES_TAC `carry_s4:bool` THENL
    [RULE_ASSUM_TAC(REWRITE_RULE[ASSUME `carry_s4:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `carry_s4:bool`);
     RULE_ASSUM_TAC(REWRITE_RULE[ASSUME `~carry_s4:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `~carry_s4:bool`)]) THEN
  (ASM_CASES_TAC `carry_s5:bool` THENL
    [RULE_ASSUM_TAC(REWRITE_RULE[ASSUME `carry_s5:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `carry_s5:bool`);
     RULE_ASSUM_TAC(REWRITE_RULE[ASSUME `~carry_s5:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `~carry_s5:bool`)]) THEN
  ASM_REWRITE_TAC[BITVAL_CLAUSES] THEN
  CONV_TAC(DEPTH_CONV WORD_NUM_RED_CONV) THEN
  CONV_TAC NUM_REDUCE_CONV THEN
  ASM_SIMP_TAC[MOD_ADD_CASES; GSYM NOT_LE] THEN
  POP_ASSUM_LIST(MP_TAC o end_itlist CONJ) THEN ARITH_TAC);;

let JOLT_FP64_ADD_SUBROUTINE_CORRECT = time prove
 (`!a b pc returnaddress.
        ensures arm
          (\s. aligned_bytes_loaded s (word pc) jolt_fp64_add_mc /\
               read PC s = word pc /\ read X30 s = returnaddress /\
               read X0 s = a /\ read X1 s = b)
          (\s. read PC s = returnaddress /\
               (val a < jolt_fp64_p /\ val b < jolt_fp64_p
                ==> val(read X0 s) = (val a + val b) MOD jolt_fp64_p))
          MAYCHANGE_REGS_AND_FLAGS_PERMITTED_BY_ABI`,
  ARM_ADD_RETURN_NOSTACK_TAC JOLT_FP64_ADD_EXEC JOLT_FP64_ADD_CORRECT);;
