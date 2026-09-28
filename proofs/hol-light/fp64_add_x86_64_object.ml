(* Exact compiler-emitted x86-64 object and execution rule for Fp64 addition. *)

needs (Filename.concat (Sys.getenv "JOLT_FP64_PROOF_DIR")
        "fp64_x86_64_common.ml");;

let jolt_fp64_add_x86_64_object = Sys.getenv "JOLT_FP64_ADD_OBJECT";;

let jolt_fp64_add_x86_64_mc =
  define_assert_from_elf "jolt_fp64_add_x86_64_mc"
    jolt_fp64_add_x86_64_object
  [
    0x48; 0x01; 0xf7;
    0x48; 0x8d; 0x47; 0x3b;
    0x48; 0x0f; 0x43; 0xc7;
    0x48; 0x8d; 0x48; 0x3b;
    0x48; 0x39; 0xc1;
    0x48; 0x0f; 0x42; 0xc1;
    0xc3
  ];;

let JOLT_FP64_ADD_X86_64_EXEC =
  X86_MK_CORE_EXEC_RULE jolt_fp64_add_x86_64_mc;;
