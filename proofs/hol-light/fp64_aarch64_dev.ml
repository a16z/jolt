(* Persistent AArch64 development session for the scalar Fp64 proofs. *)

loadt "arm/proofs/base.ml";;

let jolt_fp64_proof_dir = Sys.getenv "JOLT_FP64_PROOF_DIR";;
let jolt_fp64_dev_object = Sys.getenv "JOLT_FP64_DEV_OBJECT_SOURCE";;
let jolt_fp64_dev_theorem = Sys.getenv "JOLT_FP64_DEV_CORRECT_SOURCE";;

loadt (Filename.concat jolt_fp64_proof_dir "fp64_common.ml");;

loadt (Filename.concat jolt_fp64_proof_dir jolt_fp64_dev_object);;

loadt (Filename.concat jolt_fp64_proof_dir jolt_fp64_dev_theorem);;

print_endline
  ("Reload after an edit with: loadt \"" ^
   Filename.concat jolt_fp64_proof_dir jolt_fp64_dev_theorem ^
   "\";;");;
