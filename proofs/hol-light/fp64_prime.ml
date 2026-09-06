(* Primality and quadratic-extension facts for the production Fp64 modulus. *)

needs "Library/pocklington.ml";;
needs (Filename.concat (Sys.getenv "JOLT_FP64_PROOF_DIR")
        "fp64_common.ml");;

let JOLT_FP64_PRIME = prove
 (`prime jolt_fp64_p`,
  REWRITE_TAC[jolt_fp64_p] THEN (CONV_TAC o PRIME_RULE)
   ["2"; "3"; "5"; "7"; "11"; "13"; "17"; "23"; "31";
    "71"; "73"; "137"; "547"; "1427"; "2131"; "15331";
    "5594472617641";
    "18446744073709551557"]);;
