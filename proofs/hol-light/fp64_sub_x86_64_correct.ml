(* Exact x86-64 correctness proof for scalar Fp64 subtraction. *)

needs (Filename.concat (Sys.getenv "JOLT_FP64_PROOF_DIR")
        "fp64_sub_x86_64_object.ml");;

let JOLT_FP64_SUB_X86_64_CORRECT = time prove
 (`!a b pc.
        ensures x86
          (\s. bytes_loaded s (word pc) (BUTLAST jolt_fp64_sub_x86_64_mc) /\
               read RIP s = word pc /\ read RDI s = a /\ read RSI s = b)
          (\s. read RIP s = word (pc + 0xb) /\
               (val a < jolt_fp64_p /\ val b < jolt_fp64_p
                ==> val(read RAX s) =
                    (val a + jolt_fp64_p - val b) MOD jolt_fp64_p))
          (MAYCHANGE [RIP; RAX; RDI] ,,
           MAYCHANGE SOME_FLAGS ,, MAYCHANGE [events])`,
  MAP_EVERY X_GEN_TAC [`a:int64`; `b:int64`; `pc:num`] THEN
  REWRITE_TAC[SOME_FLAGS] THEN ENSURES_INIT_TAC "s0" THEN
  X86_ACCSTEPS_TAC JOLT_FP64_SUB_X86_64_EXEC [1] (1--1) THEN
  SUBGOAL_THEN
   `2 EXP 64 * bitval carry_s1 + val(a:int64) =
    val(b:int64) + val(sum_s1:int64)`
  (LABEL_TAC "difference") THENL
   [REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    MATCH_MP_TAC JOLT_FP64_X86_COMPARE_REORIENT THEN
    ACCUMULATOR_ASSUM_LIST(fun theorems -> ACCEPT_TAC(hd theorems));
    ALL_TAC] THEN
  ACCUMULATOR_POP_ASSUM_LIST(K ALL_TAC) THEN
  X86_ACCSTEPS_TAC JOLT_FP64_SUB_X86_64_EXEC [2] (2--3) THEN
  SUBGOAL_THEN
   `2 EXP 64 * bitval carry_s2 + val(sum_s2:int64) =
    val(sum_s1:int64) + jolt_fp64_p`
  (LABEL_TAC "correction") THENL
   [REWRITE_TAC[jolt_fp64_p; GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ) THEN
    DISCH_THEN(fun th -> REWRITE_TAC[th]) THEN REAL_ARITH_TAC;
    ALL_TAC] THEN
  ACCUMULATOR_POP_ASSUM_LIST(K ALL_TAC) THEN
  ENSURES_FINAL_STATE_TAC THEN ASM_REWRITE_TAC[] THEN STRIP_TAC THEN
  SUBGOAL_THEN
   `val(sum_s1:int64) < 2 EXP 64 /\ val(sum_s2:int64) < 2 EXP 64`
  STRIP_ASSUME_TAC THENL [BOUNDER_TAC[]; ALL_TAC] THEN
  DISCARD_STATE_TAC "s3" THEN
  GEN_REWRITE_TAC LAND_CONV [COND_RAND] THEN
  MATCH_MP_TAC(SPECL
   [`val(a:int64)`; `val(b:int64)`; `val(sum_s1:int64)`;
    `carry_s1:bool`; `val(sum_s2:int64)`; `carry_s2:bool`]
   JOLT_FP64_SUB_REDUCTION) THEN
  REPEAT CONJ_TAC THEN ASM_REWRITE_TAC[]);;

let JOLT_FP64_SUB_X86_64_SUBROUTINE_CORRECT = time prove
 (`!a b pc stackpointer returnaddress.
        ensures x86
          (\s. bytes_loaded s (word pc) jolt_fp64_sub_x86_64_mc /\
               read RIP s = word pc /\ read RSP s = stackpointer /\
               read (memory :> bytes64 stackpointer) s = returnaddress /\
               read RDI s = a /\ read RSI s = b)
          (\s. read RIP s = returnaddress /\
               read RSP s = word_add stackpointer (word 8) /\
               (val a < jolt_fp64_p /\ val b < jolt_fp64_p
                ==> val(read RAX s) =
                    (val a + jolt_fp64_p - val b) MOD jolt_fp64_p))
          (MAYCHANGE [RSP] ,, MAYCHANGE_REGS_AND_FLAGS_PERMITTED_BY_ABI)`,
  X86_PROMOTE_RETURN_NOSTACK_TAC
    jolt_fp64_sub_x86_64_mc JOLT_FP64_SUB_X86_64_CORRECT);;
