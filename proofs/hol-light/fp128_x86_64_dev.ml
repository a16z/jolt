(* Interactive entry point. Load the stable model and object once, then run
   the selected reloadable theorem file. *)

let jolt_fp128_dev_operation = Sys.getenv "JOLT_FP128_DEV_OPERATION";;
let jolt_fp128_proof_dir = Sys.getenv "JOLT_FP128_PROOF_DIR";;

let jolt_fp128_dev_theorem =
  match jolt_fp128_dev_operation with
  | "add" -> "fp128_add_x86_64_correct.ml"
  | "sub" -> "fp128_sub_x86_64_correct.ml"
  | "mul" -> "fp128_mul_x86_64_correct.ml"
  | "mul_bmi2_adx" -> "fp128_mul_x86_64_bmi2_adx_correct.ml"
  | operation -> failwith ("unsupported Fp128 proof operation: " ^ operation);;

loadt (Filename.concat jolt_fp128_proof_dir jolt_fp128_dev_theorem);;

print_endline
  ("Reload after an edit with: loadt \"" ^
   Filename.concat jolt_fp128_proof_dir jolt_fp128_dev_theorem ^
   "\";;");;
