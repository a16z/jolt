(* Exact compiler-emitted AArch64 object and execution rule for Fp64 multiplication. *)

needs "arm/proofs/base.ml";;
needs (Filename.concat (Sys.getenv "JOLT_FP64_PROOF_DIR")
        "fp64_common.ml");;

let jolt_fp64_mul_object = Sys.getenv "JOLT_FP64_MUL_OBJECT";;

let jolt_fp64_mul_mc =
  define_assert_from_elf "jolt_fp64_mul_mc" jolt_fp64_mul_object
  [
    0x9b007c28; 0x9bc07c29; 0x5280076a; 0x9bca7d2b;
    0xcb09016b; 0x9b0a7d2c; 0xab080188; 0x9a090169;
    0x9bca7d2b; 0xcb09016b; 0x9b0a7d2a; 0xab080148;
    0x9a090169; 0x9100ed0a; 0xb100ed1f; 0xfa1f013f;
    0x9a8a3100; 0xd65f03c0
  ];;

let JOLT_FP64_MUL_EXEC = ARM_MK_EXEC_RULE jolt_fp64_mul_mc;;
