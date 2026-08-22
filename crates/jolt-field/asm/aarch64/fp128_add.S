#if defined(__APPLE__)
#define C_SYMBOL(name) _##name
#else
#define C_SYMBOL(name) name
#endif

.text
.p2align 2
.globl C_SYMBOL(jolt_fp128_add_asm)
#if !defined(__APPLE__)
.type C_SYMBOL(jolt_fp128_add_asm), %function
#endif
C_SYMBOL(jolt_fp128_add_asm):
#include "fp128_load_a7f7.inc"
#include "fp128_add_body.inc"
ret
#if !defined(__APPLE__)
.size C_SYMBOL(jolt_fp128_add_asm), .-C_SYMBOL(jolt_fp128_add_asm)
.section .note.GNU-stack,"",%progbits
#endif
