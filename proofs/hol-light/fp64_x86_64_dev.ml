(* Interactive entry point for reloadable scalar Fp64 x86-64 proofs. *)

loadt "x86/proofs/base.ml";;

let jolt_fp64_proof_dir = Sys.getenv "JOLT_FP64_PROOF_DIR";;
let jolt_fp64_dev_object = Sys.getenv "JOLT_FP64_DEV_OBJECT_SOURCE";;
let jolt_fp64_dev_theorem = Sys.getenv "JOLT_FP64_DEV_CORRECT_SOURCE";;

loadt (Filename.concat jolt_fp64_proof_dir jolt_fp64_dev_object);;
loadt (Filename.concat jolt_fp64_proof_dir jolt_fp64_dev_theorem);;

print_endline
  ("Reload after an edit with: loadt \"" ^
   Filename.concat jolt_fp64_proof_dir jolt_fp64_dev_theorem ^
   "\";;");;
