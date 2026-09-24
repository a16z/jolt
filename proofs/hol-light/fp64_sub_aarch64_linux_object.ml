(* Exact Linux AArch64 compiler object and execution rule for Fp64 subtraction. *)

needs "arm/proofs/base.ml";;
needs (Filename.concat (Sys.getenv "JOLT_FP64_PROOF_DIR")
        "fp64_common.ml");;

let jolt_fp64_sub_aarch64_linux_object =
  Sys.getenv "JOLT_FP64_SUB_OBJECT";;

let jolt_fp64_sub_aarch64_linux_mc =
  define_assert_from_elf "jolt_fp64_sub_aarch64_linux_mc"
    jolt_fp64_sub_aarch64_linux_object
  [
    0x92800748; 0xeb010009; 0x9a9f3108; 0x8b080120; 0xd65f03c0
  ];;

let JOLT_FP64_SUB_AARCH64_LINUX_EXEC =
  ARM_MK_EXEC_RULE jolt_fp64_sub_aarch64_linux_mc;;
