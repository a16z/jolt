(* Exact object words and execution rule for AArch64 Fp128 multiplication. *)

needs "arm/proofs/base.ml";;
needs (Filename.concat (Sys.getenv "JOLT_FP128_PROOF_DIR")
        "fp128_common.ml");;

let jolt_fp128_mul_object = Sys.getenv "JOLT_FP128_MUL_OBJECT";;

let jolt_fp128_mul_mc =
  define_assert_from_elf "jolt_fp128_mul_mc" jolt_fp128_mul_object
  [
    0x128b0104;
    0x9b027c05;
    0x9bc27c06;
    0x9b037c07;
    0x9bc37c08;
    0x9b027c29;
    0x9bc27c2a;
    0x9b037c2b;
    0x9bc37c2c;
    0xab0700c6;
    0x1a9f37e7;
    0xab0a0108;
    0x1a9f37ea;
    0xab0b0108;
    0x9a8a354a;
    0xab0900c6;
    0xba070108;
    0x9a0a018c;
    0x9b047d07;
    0x9bc47d09;
    0x9b047d8a;
    0x9bc47d8b;
    0xab0700a5;
    0xba0900c6;
    0x1a9f37e8;
    0xab0a00c6;
    0x9a08016c;
    0x9b047d87;
    0xab0700a5;
    0xba1f00c6;
    0x1a9f37e7;
    0xab0400a9;
    0xba1f00ca;
    0x7a4038e0;
    0x9a851120;
    0x9a861141;
    0xd65f03c0
  ];;

let JOLT_FP128_MUL_EXEC = ARM_MK_EXEC_RULE jolt_fp128_mul_mc;;
