(* Arithmetic definitions shared by the AArch64 and x86-64 Fp128 proofs. *)

let jolt_fp128_a7f7_p = new_definition
 `jolt_fp128_a7f7_p = 0xffffffffffffffffffffffff00005809`;;

(* Architecture-neutral arithmetic facts used after the processor-specific
   instruction trace has been reduced to equations over natural numbers. *)
let JOLT_FP128_NO_TOP_CARRY = prove
 (`!c b r n. 0 < b /\ b * c + r = n /\ n < b ==> r = n`,
  INDUCT_TAC THENL
   [REPEAT GEN_TAC THEN REWRITE_TAC[MULT_CLAUSES; ADD_CLAUSES] THEN
    MESON_TAC[];
    REPEAT GEN_TAC THEN REWRITE_TAC[MULT_SUC] THEN ARITH_TAC]);;

let JOLT_FP128_FOLD_128 = prove
 (`!lo hi.
      (2 EXP 128 * hi + lo) MOD jolt_fp128_a7f7_p =
      (4294944759 * hi + lo) MOD jolt_fp128_a7f7_p`,
  REPEAT GEN_TAC THEN ONCE_REWRITE_TAC[GSYM MOD_ADD_MOD] THEN
  ONCE_REWRITE_TAC[GSYM MOD_MULT_LMOD] THEN
  REWRITE_TAC[jolt_fp128_a7f7_p] THEN CONV_TAC NUM_REDUCE_CONV);;

let JOLT_FP128_SECOND_FOLD_BOUND = prove
 (`!lo q v.
      lo < 2 EXP 128 /\ q <= 4294944759 /\
      v = lo + 4294944759 * q
      ==> v < 2 * jolt_fp128_a7f7_p`,
  REPEAT GEN_TAC THEN REWRITE_TAC[jolt_fp128_a7f7_p] THEN
  CONV_TAC NUM_REDUCE_CONV THEN ARITH_TAC);;

let JOLT_FP128_TWO_FOLDS = prove
 (`!mn t v hi lo q lo'.
      mn = 2 EXP 128 * hi + lo /\
      t = lo + 4294944759 * hi /\
      t = 2 EXP 128 * q + lo' /\
      v = lo' + 4294944759 * q
      ==> mn MOD jolt_fp128_a7f7_p = v MOD jolt_fp128_a7f7_p`,
  REPEAT GEN_TAC THEN STRIP_TAC THEN
  MATCH_MP_TAC EQ_TRANS THEN
  EXISTS_TAC `(4294944759 * hi + lo) MOD jolt_fp128_a7f7_p` THEN
  CONJ_TAC THENL
   [ASM_REWRITE_TAC[JOLT_FP128_FOLD_128];
    SUBGOAL_THEN
     `4294944759 * hi + lo = 2 EXP 128 * q + lo'`
    SUBST1_TAC THENL
     [ASM_MESON_TAC[ADD_SYM];
      ASM_REWRITE_TAC[ADD_SYM; JOLT_FP128_FOLD_128]]]);;

let JOLT_FP128_CANONICALIZE = prove
 (`!v r u c1 c2.
      v = 2 EXP 128 * bitval c1 + r /\
      2 EXP 128 * bitval c2 + u = r + 4294944759 /\
      v < 2 * jolt_fp128_a7f7_p /\
      r < 2 EXP 128 /\ u < 2 EXP 128
      ==> (if c1 \/ c2 then u else r) = v MOD jolt_fp128_a7f7_p`,
  REPEAT GEN_TAC THEN STRIP_TAC THEN
  SUBGOAL_THEN
   `v MOD jolt_fp128_a7f7_p =
    if v < jolt_fp128_a7f7_p then v else v - jolt_fp128_a7f7_p`
  SUBST1_TAC THENL
   [MATCH_MP_TAC MOD_CASES THEN ASM_REWRITE_TAC[];
    ALL_TAC] THEN
  (ASM_CASES_TAC `c1:bool` THENL
    [RULE_ASSUM_TAC(REWRITE_RULE[ASSUME `c1:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `c1:bool`);
     RULE_ASSUM_TAC(REWRITE_RULE[ASSUME `~c1:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `~c1:bool`)]) THEN
  (ASM_CASES_TAC `c2:bool` THENL
    [RULE_ASSUM_TAC(REWRITE_RULE[ASSUME `c2:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `c2:bool`);
     RULE_ASSUM_TAC(REWRITE_RULE[ASSUME `~c2:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `~c2:bool`)]) THEN
  RULE_ASSUM_TAC(REWRITE_RULE[jolt_fp128_a7f7_p]) THEN
  ASM_REWRITE_TAC[BITVAL_CLAUSES; jolt_fp128_a7f7_p] THEN
  CONV_TAC NUM_REDUCE_CONV THEN
  POP_ASSUM_LIST(MP_TAC o end_itlist CONJ) THEN ARITH_TAC);;
