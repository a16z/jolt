#if defined(__APPLE__)
#define C_SYMBOL(name) _##name
#else
#define C_SYMBOL(name) name
#endif

.text
.p2align 2
.globl C_SYMBOL(jolt_fp64_mul_asm)
#if !defined(__APPLE__)
.type C_SYMBOL(jolt_fp64_mul_asm), %function
#endif
C_SYMBOL(jolt_fp64_mul_asm):
#include "fp64_mul_body.inc"
ret
#if !defined(__APPLE__)
.size C_SYMBOL(jolt_fp64_mul_asm), .-C_SYMBOL(jolt_fp64_mul_asm)
.section .note.GNU-stack,"",%progbits
#endif
