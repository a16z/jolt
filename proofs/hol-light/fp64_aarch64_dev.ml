(* Persistent AArch64 development session for the scalar Fp64 proofs. *)

loadt "arm/proofs/base.ml";;

let jolt_fp64_proof_dir = Sys.getenv "JOLT_FP64_PROOF_DIR";;
let jolt_fp64_dev_operation = Sys.getenv "JOLT_FP64_DEV_OPERATION";;
let jolt_fp64_target_os = Sys.getenv "JOLT_FP64_TARGET_OS";;
let jolt_fp64_linux = jolt_fp64_target_os = "linux";;

loadt (Filename.concat jolt_fp64_proof_dir "fp64_common.ml");;

let jolt_fp64_dev_theorem =
  match jolt_fp64_linux,jolt_fp64_dev_operation with
  | true,"add" -> "fp64_add_aarch64_linux_correct.ml"
  | true,"sub" -> "fp64_sub_aarch64_linux_correct.ml"
  | true,"mul" -> "fp64_mul_aarch64_linux_correct.ml"
  | false,"add" -> "fp64_add_correct.ml"
  | false,"sub" -> "fp64_sub_correct.ml"
  | false,"mul" -> "fp64_mul_correct.ml"
  | _,operation ->
      failwith ("unsupported scalar Fp64 operation: " ^ operation);;

let jolt_fp64_dev_object =
  match jolt_fp64_linux,jolt_fp64_dev_operation with
  | true,"add" -> "fp64_add_aarch64_linux_object.ml"
  | true,"sub" -> "fp64_sub_aarch64_linux_object.ml"
  | true,"mul" -> "fp64_mul_aarch64_linux_object.ml"
  | false,"add" -> "fp64_add_object.ml"
  | false,"sub" -> "fp64_sub_object.ml"
  | false,"mul" -> "fp64_mul_object.ml"
  | _,operation ->
      failwith ("unsupported scalar Fp64 operation: " ^ operation);;

loadt (Filename.concat jolt_fp64_proof_dir jolt_fp64_dev_object);;

loadt (Filename.concat jolt_fp64_proof_dir jolt_fp64_dev_theorem);;

print_endline
  ("Reload after an edit with: loadt \"" ^
   Filename.concat jolt_fp64_proof_dir jolt_fp64_dev_theorem ^
   "\";;");;
