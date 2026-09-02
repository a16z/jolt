(* Exact Linux AArch64 compiler object and execution rule for Fp64 addition. *)

needs "arm/proofs/base.ml";;
needs (Filename.concat (Sys.getenv "JOLT_FP64_PROOF_DIR")
        "fp64_common.ml");;

let jolt_fp64_add_aarch64_linux_object =
  Sys.getenv "JOLT_FP64_ADD_OBJECT";;

let jolt_fp64_add_aarch64_linux_mc =
  define_assert_from_elf "jolt_fp64_add_aarch64_linux_mc"
    jolt_fp64_add_aarch64_linux_object
  [
    0x52800768; 0xab000029; 0x9a9f2108; 0x8b090108;
    0x9100ed09; 0xeb08013f; 0x9a883120; 0xd65f03c0
  ];;

let JOLT_FP64_ADD_AARCH64_LINUX_EXEC =
  ARM_MK_EXEC_RULE jolt_fp64_add_aarch64_linux_mc;;
