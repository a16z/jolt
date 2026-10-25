// The CLI wraps external calls and leaves the existing implementation as
// __real_memcpy/__real_memmove. All loads precede stores for overlap safety.
.macro small_copy name, fallback
.section .text.\name,"ax",@progbits
.globl \name
.hidden \name
.type \name,@function
\name:
.option push
.option norelax
    beqz a2, 9f
    or t0, a0, a1
    or t0, t0, a2
    andi t0, t0, 7
    bnez t0, 8f
    li t0, 80
    bltu t0, a2, 8f
    add a1, a1, a2
    add a7, a0, a2
    lla t0, 2f
    srli a2, a2, 1
    sub t0, t0, a2
    jr t0
// Computed offsets require exactly four bytes per load/store.
.option push
.option norvc
    ld t1, -80(a1)
    ld t2, -72(a1)
    ld t3, -64(a1)
    ld t4, -56(a1)
    ld t5, -48(a1)
    ld t6, -40(a1)
    ld a3, -32(a1)
    ld a4, -24(a1)
    ld a5, -16(a1)
    ld a6, -8(a1)
.option pop
2:
    lla t0, 3f
    sub t0, t0, a2
    jr t0
.option push
.option norvc
    sd t1, -80(a7)
    sd t2, -72(a7)
    sd t3, -64(a7)
    sd t4, -56(a7)
    sd t5, -48(a7)
    sd t6, -40(a7)
    sd a3, -32(a7)
    sd a4, -24(a7)
    sd a5, -16(a7)
    sd a6, -8(a7)
.option pop
3:
9:
    ret
8:
    tail \fallback
.option pop
.size \name, .-\name
.endm

small_copy __wrap_memcpy, __real_memcpy
small_copy __wrap_memmove, __real_memmove
