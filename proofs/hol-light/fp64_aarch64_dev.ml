(* Persistent AArch64 development session for the scalar Fp64 proofs. *)

loadt "arm/proofs/base.ml";;

let jolt_fp64_proof_dir = Sys.getenv "JOLT_FP64_PROOF_DIR";;
let jolt_fp64_dev_operation = Sys.getenv "JOLT_FP64_DEV_OPERATION";;

loadt (Filename.concat jolt_fp64_proof_dir "fp64_common.ml");;

let jolt_fp64_dev_theorem =
  match jolt_fp64_dev_operation with
  | "add" -> "fp64_add_correct.ml"
  | "sub" -> "fp64_sub_correct.ml"
  | "mul" -> "fp64_mul_correct.ml"
  | operation -> failwith ("unsupported scalar Fp64 operation: " ^ operation);;

let jolt_fp64_dev_object =
  match jolt_fp64_dev_operation with
  | "add" -> "fp64_add_object.ml"
  | "sub" -> "fp64_sub_object.ml"
  | "mul" -> "fp64_mul_object.ml"
  | operation -> failwith ("unsupported scalar Fp64 operation: " ^ operation);;

loadt (Filename.concat jolt_fp64_proof_dir jolt_fp64_dev_object);;

loadt (Filename.concat jolt_fp64_proof_dir jolt_fp64_dev_theorem);;

print_endline
  ("Reload after an edit with: loadt \"" ^
   Filename.concat jolt_fp64_proof_dir jolt_fp64_dev_theorem ^
   "\";;");;
