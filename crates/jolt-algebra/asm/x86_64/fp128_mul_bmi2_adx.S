#if defined(__APPLE__)
#define C_SYMBOL(name) _##name
#else
#define C_SYMBOL(name) name
#endif

.intel_syntax noprefix
.text
.p2align 4
.globl C_SYMBOL(jolt_fp128_mul_bmi2_adx_asm)
#if !defined(__APPLE__)
.type C_SYMBOL(jolt_fp128_mul_bmi2_adx_asm), @function
#endif
C_SYMBOL(jolt_fp128_mul_bmi2_adx_asm):
#include "fp128_mul_bmi2_adx_body.inc"
ret
#if !defined(__APPLE__)
.size C_SYMBOL(jolt_fp128_mul_bmi2_adx_asm), .-C_SYMBOL(jolt_fp128_mul_bmi2_adx_asm)
.section .note.GNU-stack,"",@progbits
#endif
