(* Exact compiler-emitted AArch64 object and execution rule for Fp64 subtraction. *)

needs "arm/proofs/base.ml";;
needs (Filename.concat (Sys.getenv "JOLT_FP64_PROOF_DIR")
        "fp64_common.ml");;

let jolt_fp64_sub_object = Sys.getenv "JOLT_FP64_SUB_OBJECT";;

let jolt_fp64_sub_mc =
  define_assert_from_elf "jolt_fp64_sub_mc" jolt_fp64_sub_object
  [0xeb010008; 0x92800749; 0x9a9f3129; 0x8b090100; 0xd65f03c0];;

let JOLT_FP64_SUB_EXEC = ARM_MK_EXEC_RULE jolt_fp64_sub_mc;;
