(* Architecture-neutral arithmetic facts for p = 2^64 - 59. *)

let jolt_fp64_p = new_definition
 `jolt_fp64_p = 18446744073709551557`;;

let JOLT_FP64_FOLD = prove
 (`!lo hi.
      (2 EXP 64 * hi + lo) MOD jolt_fp64_p =
      (59 * hi + lo) MOD jolt_fp64_p`,
  REPEAT GEN_TAC THEN ONCE_REWRITE_TAC[GSYM MOD_ADD_MOD] THEN
  ONCE_REWRITE_TAC[GSYM MOD_MULT_LMOD] THEN
  REWRITE_TAC[jolt_fp64_p] THEN CONV_TAC NUM_REDUCE_CONV);;

let JOLT_FP64_NO_TOP_CARRY = prove
 (`!c b r n. 0 < b /\ b * c + r = n /\ n < b ==> r = n`,
  INDUCT_TAC THENL
   [REPEAT GEN_TAC THEN REWRITE_TAC[MULT_CLAUSES; ADD_CLAUSES] THEN
    MESON_TAC[];
    REPEAT GEN_TAC THEN REWRITE_TAC[MULT_SUC] THEN ARITH_TAC]);;

(* The two conditional additions used by the scalar addition kernels. *)
let JOLT_FP64_ADD_REDUCTION = prove
 (`!a b first firstcarry correction correctioncarry final finalcarry.
      a < jolt_fp64_p /\ b < jolt_fp64_p /\
      2 EXP 64 * bitval firstcarry + first = a + b /\
      2 EXP 64 * bitval correctioncarry + correction = first + 59 /\
      2 EXP 64 * bitval finalcarry + final =
        (if firstcarry then correction else first) + 59 /\
      first < 2 EXP 64 /\ correction < 2 EXP 64 /\ final < 2 EXP 64
      ==> (if finalcarry then final
           else if ~firstcarry then first else correction) =
          (a + b) MOD jolt_fp64_p`,
  REPEAT GEN_TAC THEN REWRITE_TAC[jolt_fp64_p] THEN STRIP_TAC THEN
  (ASM_CASES_TAC `firstcarry:bool` THENL
    [RULE_ASSUM_TAC
      (REWRITE_RULE[ASSUME `firstcarry:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `firstcarry:bool`);
     RULE_ASSUM_TAC
      (REWRITE_RULE[ASSUME `~firstcarry:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `~firstcarry:bool`)]) THEN
  (ASM_CASES_TAC `correctioncarry:bool` THENL
    [RULE_ASSUM_TAC
      (REWRITE_RULE[ASSUME `correctioncarry:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `correctioncarry:bool`);
     RULE_ASSUM_TAC
      (REWRITE_RULE[ASSUME `~correctioncarry:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `~correctioncarry:bool`)]) THEN
  (ASM_CASES_TAC `finalcarry:bool` THENL
    [RULE_ASSUM_TAC
      (REWRITE_RULE[ASSUME `finalcarry:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `finalcarry:bool`);
     RULE_ASSUM_TAC
      (REWRITE_RULE[ASSUME `~finalcarry:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `~finalcarry:bool`)]) THEN
  ASM_REWRITE_TAC[BITVAL_CLAUSES] THEN CONV_TAC NUM_REDUCE_CONV THEN
  ASM_SIMP_TAC[MOD_ADD_CASES; GSYM NOT_LE] THEN
  POP_ASSUM_LIST(MP_TAC o end_itlist CONJ) THEN ARITH_TAC);;

(* Borrow correction used by the scalar subtraction kernels. *)
let JOLT_FP64_SUB_REDUCTION = prove
 (`!a b difference borrow corrected correctioncarry.
      a < jolt_fp64_p /\ b < jolt_fp64_p /\
      2 EXP 64 * bitval borrow + a = b + difference /\
      2 EXP 64 * bitval correctioncarry + corrected =
        difference + jolt_fp64_p /\
      difference < 2 EXP 64 /\ corrected < 2 EXP 64
      ==> (if ~borrow then difference else corrected) =
          (a + jolt_fp64_p - b) MOD jolt_fp64_p`,
  REPEAT GEN_TAC THEN REWRITE_TAC[jolt_fp64_p] THEN STRIP_TAC THEN
  SUBGOAL_THEN `a + 18446744073709551557 - b <
                2 * 18446744073709551557`
  ASSUME_TAC THENL [ASM_ARITH_TAC; ALL_TAC] THEN
  MP_TAC(SPECL
   [`a + 18446744073709551557 - b`; `18446744073709551557`]
   MOD_CASES) THEN
  ANTS_TAC THENL [ASM_REWRITE_TAC[]; DISCH_THEN SUBST1_TAC] THEN
  (ASM_CASES_TAC `borrow:bool` THENL
    [RULE_ASSUM_TAC(REWRITE_RULE[ASSUME `borrow:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `borrow:bool`);
     RULE_ASSUM_TAC(REWRITE_RULE[ASSUME `~borrow:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `~borrow:bool`)]) THEN
  (ASM_CASES_TAC `correctioncarry:bool` THENL
    [RULE_ASSUM_TAC
      (REWRITE_RULE[ASSUME `correctioncarry:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `correctioncarry:bool`);
     RULE_ASSUM_TAC
      (REWRITE_RULE[ASSUME `~correctioncarry:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `~correctioncarry:bool`)]) THEN
  ASM_REWRITE_TAC[BITVAL_CLAUSES] THEN CONV_TAC NUM_REDUCE_CONV THEN
  POP_ASSUM_LIST(MP_TAC o end_itlist CONJ) THEN ARITH_TAC);;

(* Two applications of 2^64 = 59 (mod p) preserve the residue. *)
let JOLT_FP64_FOLD_TWICE = prove
 (`!n hi0 lo0 hi1 lo1 hi2 lo2.
      2 EXP 64 * hi0 + lo0 = n /\
      2 EXP 64 * hi1 + lo1 = lo0 + 59 * hi0 /\
      2 EXP 64 * hi2 + lo2 = lo1 + 59 * hi1
      ==> n MOD jolt_fp64_p =
          (2 EXP 64 * hi2 + lo2) MOD jolt_fp64_p`,
  REPEAT STRIP_TAC THEN
  UNDISCH_TAC `2 EXP 64 * hi0 + lo0 = n` THEN
  DISCH_THEN(SUBST1_TAC o SYM) THEN
  REWRITE_TAC[JOLT_FP64_FOLD] THEN
  SUBGOAL_THEN `59 * hi0 + lo0 = 2 EXP 64 * hi1 + lo1`
  SUBST1_TAC THENL [ASM_ARITH_TAC; ALL_TAC] THEN
  REWRITE_TAC[JOLT_FP64_FOLD] THEN
  SUBGOAL_THEN `59 * hi1 + lo1 = 2 EXP 64 * hi2 + lo2`
  SUBST1_TAC THENL [ASM_ARITH_TAC; ALL_TAC] THEN
  REWRITE_TAC[JOLT_FP64_FOLD]);;

(* Select the unique canonical representative after the second fold. *)
let JOLT_FP64_FINAL_REDUCTION = prove
 (`!hi lo corrected scratch addcarry selectcarry.
      2 EXP 64 * hi + lo < 2 * jolt_fp64_p /\
      2 EXP 64 * bitval addcarry + corrected = lo + 59 /\
      2 EXP 64 * bitval selectcarry + hi =
        scratch + bitval(~addcarry) /\
      lo < 2 EXP 64 /\ hi < 2 EXP 64 /\
      corrected < 2 EXP 64 /\ scratch < 2 EXP 64
      ==> (if selectcarry then lo else corrected) =
          (2 EXP 64 * hi + lo) MOD jolt_fp64_p`,
  REPEAT GEN_TAC THEN REWRITE_TAC[jolt_fp64_p] THEN STRIP_TAC THEN
  SUBGOAL_THEN `hi <= 1` ASSUME_TAC THENL [ASM_ARITH_TAC; ALL_TAC] THEN
  SUBGOAL_THEN `?hib. hi = bitval hib` CHOOSE_TAC THENL
   [ASM_REWRITE_TAC[GSYM NUM_AS_BITVAL]; ALL_TAC] THEN
  MP_TAC(SPECL
   [`2 EXP 64 * hi + lo`; `18446744073709551557`] MOD_CASES) THEN
  ANTS_TAC THENL [ASM_REWRITE_TAC[]; DISCH_THEN SUBST1_TAC] THEN
  UNDISCH_TAC `hi = bitval hib` THEN DISCH_THEN SUBST_ALL_TAC THEN
  (ASM_CASES_TAC `hib:bool` THENL
    [RULE_ASSUM_TAC(REWRITE_RULE[ASSUME `hib:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `hib:bool`);
     RULE_ASSUM_TAC(REWRITE_RULE[ASSUME `~hib:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `~hib:bool`)]) THEN
  (ASM_CASES_TAC `addcarry:bool` THENL
    [RULE_ASSUM_TAC(REWRITE_RULE[ASSUME `addcarry:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `addcarry:bool`);
     RULE_ASSUM_TAC(REWRITE_RULE[ASSUME `~addcarry:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `~addcarry:bool`)]) THEN
  (ASM_CASES_TAC `selectcarry:bool` THENL
    [RULE_ASSUM_TAC
      (REWRITE_RULE[ASSUME `selectcarry:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `selectcarry:bool`);
     RULE_ASSUM_TAC
      (REWRITE_RULE[ASSUME `~selectcarry:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `~selectcarry:bool`)]) THEN
  ASM_REWRITE_TAC[BITVAL_CLAUSES] THEN CONV_TAC NUM_REDUCE_CONV THEN
  COND_CASES_TAC THEN ASM_REWRITE_TAC[] THEN
  POP_ASSUM_LIST(MP_TAC o end_itlist CONJ) THEN ARITH_TAC);;
