(* Primality certificate for the production Prime128OffsetA7F7 modulus. *)

needs "Library/pocklington.ml";;
needs (Filename.concat (Sys.getenv "JOLT_FP128_PROOF_DIR")
        "fp128_common.ml");;

(* The Rust type checks the Solinas shape at compile time, but trial division
   over a 128 bit number is not practical during constant evaluation. *)
let JOLT_FP128_A7F7_PRIME = prove
 (`prime jolt_fp128_a7f7_p`,
  REWRITE_TAC[jolt_fp128_a7f7_p] THEN (CONV_TAC o PRIME_RULE)
   ["2"; "3"; "5"; "7"; "11"; "17"; "19"; "23"; "41"; "61";
    "307"; "433"; "367"; "491"; "983"; "1229"; "3037"; "36373";
    "90437"; "18223"; "459647"; "1964143"; "23569717";
    "942788681"; "3296066903"; "33908337700847";
    "54317376720913331118684727";
    "340282366920938463463374607427473266697"]);;
