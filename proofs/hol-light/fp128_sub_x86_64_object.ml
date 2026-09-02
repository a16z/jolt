(* Exact object bytes and execution rule for x86-64 Fp128 subtraction. *)

needs (Filename.concat (Sys.getenv "JOLT_FP128_PROOF_DIR")
        "fp128_x86_64_common.ml");;

let jolt_fp128_sub_object = Sys.getenv "JOLT_FP128_SUB_OBJECT";;

let jolt_fp128_sub_mc =
  define_assert_from_elf "jolt_fp128_sub_mc" jolt_fp128_sub_object
  [
    0x41; 0xb8; 0xf7; 0xa7; 0xff; 0xff;
    0x48; 0x29; 0xd7;
    0x48; 0x19; 0xce;
    0x4d; 0x19; 0xc9;
    0x4d; 0x21; 0xc1;
    0x4c; 0x29; 0xcf;
    0x48; 0x83; 0xde; 0x00;
    0x48; 0x89; 0xf2;
    0x48; 0x89; 0xf8;
    0xc3
  ];;

let JOLT_FP128_SUB_X86_64_EXEC =
  X86_MK_CORE_EXEC_RULE jolt_fp128_sub_mc;;
