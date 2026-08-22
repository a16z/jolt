(* Interactive AArch64 entry point. The stable model and object stay loaded
   while the selected correctness theorem is reloaded after each edit. *)

let jolt_fp128_dev_operation = Sys.getenv "JOLT_FP128_DEV_OPERATION";;
let jolt_fp128_proof_dir = Sys.getenv "JOLT_FP128_PROOF_DIR";;

let jolt_fp128_dev_theorem =
  match jolt_fp128_dev_operation with
  | "mul" -> "fp128_mul_correct.ml"
  | operation -> failwith ("unsupported Fp128 proof operation: " ^ operation);;

loadt (Filename.concat jolt_fp128_proof_dir "fp128_mul_object.ml");;
loadt (Filename.concat jolt_fp128_proof_dir jolt_fp128_dev_theorem);;

print_endline
  ("Reload after an edit with: loadt \"" ^
   Filename.concat jolt_fp128_proof_dir jolt_fp128_dev_theorem ^
   "\";;");;
