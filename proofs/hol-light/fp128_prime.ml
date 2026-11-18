(* Primality certificate for the production Prime128Offset6A23801F modulus. *)

needs "Library/pocklington.ml";;
needs (Filename.concat (Sys.getenv "JOLT_FP128_PROOF_DIR")
        "fp128_common.ml");;

(* The Rust type checks the Solinas shape at compile time, but trial division
   over a 128 bit number is not practical during constant evaluation. *)
let JOLT_FP128_6A23801F_PRIME = prove
 (`prime jolt_fp128_6a23801f_p`,
  REWRITE_TAC[jolt_fp128_6a23801f_p] THEN (CONV_TAC o PRIME_RULE)
   ["2"; "3"; "5"; "7"; "11"; "13"; "17"; "29"; "53"; "59"; "71";
    "73"; "107"; "127"; "137"; "199"; "241"; "797"; "1499"; "1753";
    "7013"; "15053"; "5306461"; "12373567"; "48787559"; "3317554013";
    "6635108027"; "151955935914427270231";
    "340282366920938463463374607429987500001"]);;
