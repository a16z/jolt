#if defined(__APPLE__)
#define C_SYMBOL(name) _##name
#else
#define C_SYMBOL(name) name
#endif

.text
.p2align 2
.globl C_SYMBOL(jolt_fp64_sub_asm)
#if !defined(__APPLE__)
.type C_SYMBOL(jolt_fp64_sub_asm), %function
#endif
C_SYMBOL(jolt_fp64_sub_asm):
#if defined(__APPLE__)
#include "fp64_sub_body.inc"
#else
#include "fp64_sub_linux_body.inc"
#endif
ret
#if !defined(__APPLE__)
.size C_SYMBOL(jolt_fp64_sub_asm), .-C_SYMBOL(jolt_fp64_sub_asm)
.section .note.GNU-stack,"",%progbits
#endif
