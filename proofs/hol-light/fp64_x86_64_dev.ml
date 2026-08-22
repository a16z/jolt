(* Interactive entry point for reloadable scalar Fp64 x86-64 proofs. *)

loadt "x86/proofs/base.ml";;

let jolt_fp64_dev_operation = Sys.getenv "JOLT_FP64_DEV_OPERATION";;
let jolt_fp64_proof_dir = Sys.getenv "JOLT_FP64_PROOF_DIR";;

let jolt_fp64_dev_theorem =
  match jolt_fp64_dev_operation with
  | "add" -> "fp64_add_x86_64_correct.ml"
  | "sub" -> "fp64_sub_x86_64_correct.ml"
  | "mul" -> "fp64_mul_x86_64_correct.ml"
  | "mul_bmi2" -> "fp64_mul_x86_64_bmi2_correct.ml"
  | operation -> failwith ("unsupported Fp64 proof operation: " ^ operation);;

loadt (Filename.concat jolt_fp64_proof_dir jolt_fp64_dev_theorem);;

print_endline
  ("Reload after an edit with: loadt \"" ^
   Filename.concat jolt_fp64_proof_dir jolt_fp64_dev_theorem ^
   "\";;");;
