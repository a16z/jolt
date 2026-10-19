#if defined(__APPLE__) && defined(__MACH__)
#define JOLT_FP64_DARWIN 1
#define C_SYMBOL(name) _##name
#elif defined(__linux__) && defined(__ELF__)
#define JOLT_FP64_LINUX 1
#define C_SYMBOL(name) name
#else
#error "Fp64 proof assembly has no registered target for this platform"
#endif

.text
.p2align 2
.globl C_SYMBOL(jolt_fp64_sub_asm)
#if defined(JOLT_FP64_LINUX)
.type C_SYMBOL(jolt_fp64_sub_asm), %function
#endif
C_SYMBOL(jolt_fp64_sub_asm):
#if defined(JOLT_FP64_DARWIN)
#include "fp64_sub_body.inc"
#elif defined(JOLT_FP64_LINUX)
#include "fp64_sub_linux_body.inc"
#endif
ret
#if defined(JOLT_FP64_LINUX)
.size C_SYMBOL(jolt_fp64_sub_asm), .-C_SYMBOL(jolt_fp64_sub_asm)
.section .note.GNU-stack,"",%progbits
#endif
