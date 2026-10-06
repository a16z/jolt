#if defined(__APPLE__) && defined(__MACH__)
#define JOLT_FP64_DARWIN 1
#define C_SYMBOL(name) _##name
#elif defined(__linux__) && defined(__ELF__)
#define JOLT_FP64_LINUX 1
#define C_SYMBOL(name) name
#else
#error "Fp64 proof assembly has no registered target for this platform"
#endif

.intel_syntax noprefix
.text
.p2align 4
.globl C_SYMBOL(jolt_fp64_mul_asm)
#if defined(JOLT_FP64_LINUX)
.type C_SYMBOL(jolt_fp64_mul_asm), @function
#endif
C_SYMBOL(jolt_fp64_mul_asm):
#include "fp64_mul_body.inc"
ret
#if defined(JOLT_FP64_LINUX)
.size C_SYMBOL(jolt_fp64_mul_asm), .-C_SYMBOL(jolt_fp64_mul_asm)
.section .note.GNU-stack,"",@progbits
#endif
