(* Arithmetic definitions shared by the AArch64 and x86-64 Fp128 proofs. *)

let jolt_fp128_p = new_definition
 `jolt_fp128_p c = 2 EXP 128 - c`;;

let jolt_fp128_valid_offset = new_definition
 `jolt_fp128_valid_offset c <=>
    0 < c /\ c < 2 EXP 32 /\ c * (c + 1) < jolt_fp128_p c`;;

let jolt_fp128_a7f7_p = new_definition
 `jolt_fp128_a7f7_p = 0xffffffffffffffffffffffff00005809`;;

let JOLT_FP128_A7F7_P_GENERIC = prove
 (`jolt_fp128_p 4294944759 = jolt_fp128_a7f7_p`,
  REWRITE_TAC[jolt_fp128_p; jolt_fp128_a7f7_p] THEN
  CONV_TAC NUM_REDUCE_CONV);;

let JOLT_FP128_OFFSET275_VALID = prove
 (`jolt_fp128_valid_offset 275`,
  REWRITE_TAC[jolt_fp128_valid_offset; jolt_fp128_p] THEN
  CONV_TAC NUM_REDUCE_CONV);;

let JOLT_FP128_OFFSETA7F7_VALID = prove
 (`jolt_fp128_valid_offset 4294944759`,
  REWRITE_TAC[jolt_fp128_valid_offset; jolt_fp128_p] THEN
  CONV_TAC NUM_REDUCE_CONV);;

let JOLT_FP128_RADIX_DECOMPOSITION = prove
 (`!c. jolt_fp128_valid_offset c
       ==> 2 EXP 128 = jolt_fp128_p c + c /\ c < jolt_fp128_p c`,
  GEN_TAC THEN REWRITE_TAC[jolt_fp128_valid_offset; jolt_fp128_p] THEN
  ARITH_TAC);;

let JOLT_FP128_RADIX_MOD = prove
 (`!c. jolt_fp128_valid_offset c
       ==> (2 EXP 128) MOD jolt_fp128_p c = c`,
  REPEAT STRIP_TAC THEN MATCH_MP_TAC MOD_UNIQ THEN
  EXISTS_TAC `1` THEN REWRITE_TAC[MULT_CLAUSES] THEN
  MP_TAC(SPEC `c:num` JOLT_FP128_RADIX_DECOMPOSITION) THEN
  ASM_REWRITE_TAC[] THEN ARITH_TAC);;

let JOLT_FP128_FOLD_128_GENERIC = prove
 (`!c lo hi.
      jolt_fp128_valid_offset c
      ==> (2 EXP 128 * hi + lo) MOD jolt_fp128_p c =
          (c * hi + lo) MOD jolt_fp128_p c`,
  REPEAT STRIP_TAC THEN MATCH_MP_TAC MOD_EQ THEN EXISTS_TAC `hi:num` THEN
  MP_TAC(SPEC `c:num` JOLT_FP128_RADIX_DECOMPOSITION) THEN
  ASM_REWRITE_TAC[] THEN STRIP_TAC THEN
  ASM_REWRITE_TAC[LEFT_ADD_DISTRIB] THEN CONV_TAC NUM_RING);;

let JOLT_FP128_SECOND_FOLD_BOUND_GENERIC = prove
 (`!c lo q v.
      jolt_fp128_valid_offset c /\ lo < 2 EXP 128 /\ q <= c /\
      v = lo + c * q
      ==> v < 2 * jolt_fp128_p c`,
  REPEAT GEN_TAC THEN STRIP_TAC THEN
  SUBGOAL_THEN `c * q <= c * c` ASSUME_TAC THENL
   [ASM_REWRITE_TAC[LE_MULT_LCANCEL] THEN
    UNDISCH_TAC `jolt_fp128_valid_offset c` THEN
    REWRITE_TAC[jolt_fp128_valid_offset] THEN ARITH_TAC;
    ALL_TAC] THEN
  RULE_ASSUM_TAC(REWRITE_RULE
   [jolt_fp128_valid_offset; jolt_fp128_p; RIGHT_ADD_DISTRIB;
    MULT_CLAUSES]) THEN
  REWRITE_TAC[jolt_fp128_p] THEN
  POP_ASSUM_LIST(MP_TAC o end_itlist CONJ) THEN ARITH_TAC);;

let JOLT_FP128_TWO_FOLDS_GENERIC = prove
 (`!c mn t v hi lo q lo'.
      jolt_fp128_valid_offset c /\
      mn = 2 EXP 128 * hi + lo /\
      t = lo + c * hi /\
      t = 2 EXP 128 * q + lo' /\
      v = lo' + c * q
      ==> mn MOD jolt_fp128_p c = v MOD jolt_fp128_p c`,
  REPEAT GEN_TAC THEN STRIP_TAC THEN
  SUBGOAL_THEN
   `mn MOD jolt_fp128_p c = t MOD jolt_fp128_p c`
  ASSUME_TAC THENL
   [MP_TAC(SPECL [`c:num`; `lo:num`; `hi:num`]
      JOLT_FP128_FOLD_128_GENERIC) THEN
    ASM_REWRITE_TAC[] THEN ASM_MESON_TAC[ADD_SYM];
    ALL_TAC] THEN
  SUBGOAL_THEN
   `t MOD jolt_fp128_p c = v MOD jolt_fp128_p c`
  ASSUME_TAC THENL
   [MP_TAC(SPECL [`c:num`; `lo':num`; `q:num`]
      JOLT_FP128_FOLD_128_GENERIC) THEN
    ASM_REWRITE_TAC[] THEN ASM_MESON_TAC[ADD_SYM];
    ASM_MESON_TAC[]]);;

let JOLT_FP128_CANONICALIZE_GENERIC = prove
 (`!c v r u c1 c2.
      jolt_fp128_valid_offset c /\
      v = 2 EXP 128 * bitval c1 + r /\
      2 EXP 128 * bitval c2 + u = r + c /\
      v < 2 * jolt_fp128_p c /\
      r < 2 EXP 128 /\ u < 2 EXP 128
      ==> (if c1 \/ c2 then u else r) = v MOD jolt_fp128_p c`,
  REPEAT GEN_TAC THEN STRIP_TAC THEN
  SUBGOAL_THEN
   `v MOD jolt_fp128_p c =
    if v < jolt_fp128_p c then v else v - jolt_fp128_p c`
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
  RULE_ASSUM_TAC(REWRITE_RULE[jolt_fp128_valid_offset; jolt_fp128_p]) THEN
  ASM_REWRITE_TAC[BITVAL_CLAUSES; jolt_fp128_p] THEN
  POP_ASSUM_LIST(MP_TAC o end_itlist CONJ) THEN ARITH_TAC);;

let JOLT_FP128_ADD_GENERIC = prove
 (`!c m n l t c1 c2.
      jolt_fp128_valid_offset c /\
      m < jolt_fp128_p c /\ n < jolt_fp128_p c /\
      2 EXP 128 * bitval c1 + l = m + n /\
      2 EXP 128 * bitval c2 + t = l + c /\
      l < 2 EXP 128 /\ t < 2 EXP 128
      ==> (if c1 \/ c2 then t else l) =
          (m + n) MOD jolt_fp128_p c`,
  REPEAT GEN_TAC THEN STRIP_TAC THEN
  MATCH_MP_TAC(SPECL
   [`c:num`; `m + n:num`; `l:num`; `t:num`; `c1:bool`; `c2:bool`]
   JOLT_FP128_CANONICALIZE_GENERIC) THEN
  ASM_REWRITE_TAC[] THEN
  RULE_ASSUM_TAC(REWRITE_RULE[jolt_fp128_valid_offset; jolt_fp128_p]) THEN
  REWRITE_TAC[jolt_fp128_p] THEN
  POP_ASSUM_LIST(MP_TAC o end_itlist CONJ) THEN ARITH_TAC);;

let JOLT_FP128_SUB_GENERIC = prove
 (`!c m n d t borrow carry.
      jolt_fp128_valid_offset c /\
      m < jolt_fp128_p c /\ n < jolt_fp128_p c /\
      2 EXP 128 * bitval borrow + m = n + d /\
      2 EXP 128 * bitval carry + d =
        (if borrow then c else 0) + t /\
      d < 2 EXP 128 /\ t < 2 EXP 128
      ==> t = (m + jolt_fp128_p c - n) MOD jolt_fp128_p c`,
  REPEAT GEN_TAC THEN STRIP_TAC THEN
  SUBGOAL_THEN
   `m + jolt_fp128_p c - n < 2 * jolt_fp128_p c`
  ASSUME_TAC THENL
   [RULE_ASSUM_TAC(REWRITE_RULE[jolt_fp128_valid_offset; jolt_fp128_p]) THEN
    REWRITE_TAC[jolt_fp128_p] THEN
    POP_ASSUM_LIST(MP_TAC o end_itlist CONJ) THEN ARITH_TAC;
    ALL_TAC] THEN
  MP_TAC(SPECL
   [`m + jolt_fp128_p c - n`; `jolt_fp128_p c`] MOD_CASES) THEN
  ASM_REWRITE_TAC[] THEN DISCH_THEN SUBST1_TAC THEN
  ASM_CASES_TAC `m + jolt_fp128_p c - n < jolt_fp128_p c` THEN
  (ASM_CASES_TAC `borrow:bool` THENL
    [RULE_ASSUM_TAC(REWRITE_RULE[ASSUME `borrow:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `borrow:bool`);
     RULE_ASSUM_TAC(REWRITE_RULE[ASSUME `~borrow:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `~borrow:bool`)]) THEN
  (ASM_CASES_TAC `carry:bool` THENL
    [RULE_ASSUM_TAC(REWRITE_RULE[ASSUME `carry:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `carry:bool`);
     RULE_ASSUM_TAC(REWRITE_RULE[ASSUME `~carry:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `~carry:bool`)]) THEN
  RULE_ASSUM_TAC(REWRITE_RULE[jolt_fp128_valid_offset; jolt_fp128_p]) THEN
  ASM_REWRITE_TAC[BITVAL_CLAUSES; jolt_fp128_p] THEN
  POP_ASSUM_LIST(MP_TAC o end_itlist CONJ) THEN ARITH_TAC);;

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
