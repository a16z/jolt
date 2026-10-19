#if defined(__APPLE__)
#define C_SYMBOL(name) _##name
#else
#define C_SYMBOL(name) name
#endif

.intel_syntax noprefix
.text
.p2align 4
.globl C_SYMBOL(jolt_fp128_sub_asm)
#if !defined(__APPLE__)
.type C_SYMBOL(jolt_fp128_sub_asm), @function
#endif
C_SYMBOL(jolt_fp128_sub_asm):
#include "fp128_load_a7f7.inc"
#include "fp128_sub_body.inc"
mov rdx, rsi
mov rax, rdi
ret
#if !defined(__APPLE__)
.size C_SYMBOL(jolt_fp128_sub_asm), .-C_SYMBOL(jolt_fp128_sub_asm)
.section .note.GNU-stack,"",@progbits
#endif
