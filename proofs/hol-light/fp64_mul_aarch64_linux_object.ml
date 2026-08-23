(* Exact Linux AArch64 compiler object and execution rule for Fp64 multiplication. *)

needs "arm/proofs/base.ml";;
needs (Filename.concat (Sys.getenv "JOLT_FP64_PROOF_DIR")
        "fp64_common.ml");;

let jolt_fp64_mul_aarch64_linux_object =
  Sys.getenv "JOLT_FP64_MUL_OBJECT";;

let jolt_fp64_mul_aarch64_linux_mc =
  define_assert_from_elf "jolt_fp64_mul_aarch64_linux_mc"
    jolt_fp64_mul_aarch64_linux_object
  [
    0x9bc07c29; 0x52800768; 0x9b007c2a; 0x9bc87d2b;
    0x9b087d2c; 0xcb09016b; 0xab0a018a; 0x9a090169;
    0x9bc87d2b; 0x9b087d28; 0xcb09016b; 0xab0a0108;
    0x9a090169; 0xb100ed1f; 0x9100ed0a; 0xfa1f013f;
    0x9a8a3100; 0xd65f03c0
  ];;

let JOLT_FP64_MUL_AARCH64_LINUX_EXEC =
  ARM_MK_EXEC_RULE jolt_fp64_mul_aarch64_linux_mc;;
