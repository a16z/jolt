// The CLI wraps external calls; unlike the copy routines, there is no
// fallback to __real_memcmp or __real_bcmp.
// Like Jolt's subword loads, partial windows read containing doublewords.
// Masking excludes every byte outside the requested ranges; a second word
// is loaded only when the requested window crosses a doubleword boundary.
.macro compare name, equality_only
.pushsection .text.\name,"ax",@progbits
.globl \name
.hidden \name
.type \name,@function
\name:
    beqz a2, .Lequal_\name
    li a3, 8
    or t0, a0, a1
    andi t0, t0, 7
    bnez t0, .Lunaligned_\name
    bltu a2, a3, .Ltail_\name
.Lwords_\name:
    ld t0, 0(a0)
    ld t1, 0(a1)
    bne t0, t1, .Ldifferent_\name
    addi a2, a2, -8
    addi a0, a0, 8
    addi a1, a1, 8
    bgeu a2, a3, .Lwords_\name
    beqz a2, .Lequal_\name
.Ltail_\name:
    ld t0, 0(a0)
    ld t1, 0(a1)
.Lmask_\name:
    sub t2, a3, a2
    slli t2, t2, 3
    sll t0, t0, t2
    sll t1, t1, t2
    bne t0, t1, .Ldifferent_\name
.Lequal_\name:
    li a0, 0
    ret
.Lunaligned_\name:
    xor t0, a0, a1
    andi t0, t0, 7
    bnez t0, .Lwindows_\name
    bltu a2, a3, .Lwindows_\name
// Equal offsets: compare the common head, then enter the aligned loop.
    andi t2, a0, 7
    sub t3, a3, t2
    andi a0, a0, -8
    andi a1, a1, -8
    ld t0, 0(a0)
    ld t1, 0(a1)
    slli t2, t2, 3
    srl t0, t0, t2
    srl t1, t1, t2
    bne t0, t1, .Ldifferent_\name
    sub a2, a2, t3
    addi a0, a0, 8
    addi a1, a1, 8
    bgeu a2, a3, .Lwords_\name
    j .Ltail_\name
.Lwindows_\name:
// a4/a5 hold bit shifts; t4/t5 hold bytes left in each first word.
// Load a second word only when the remaining range crosses into it.
    andi a4, a0, 7
    andi a5, a1, 7
    sub t4, a3, a4
    sub t5, a3, a5
    slli a4, a4, 3
    slli a5, a5, 3
    neg a6, a4
    neg a7, a5
    andi a0, a0, -8
    andi a1, a1, -8
.Lwindow_loop_\name:
    ld t0, 0(a0)
    beqz a4, .Lright_\name
    srl t0, t0, a4
    bgeu t4, a2, .Lright_\name
    ld t2, 8(a0)
    sll t2, t2, a6
    or t0, t0, t2
.Lright_\name:
    ld t1, 0(a1)
    beqz a5, .Lwindow_compare_\name
    srl t1, t1, a5
    bgeu t5, a2, .Lwindow_compare_\name
    ld t2, 8(a1)
    sll t2, t2, a7
    or t1, t1, t2
.Lwindow_compare_\name:
    bltu a2, a3, .Lmask_\name
    bne t0, t1, .Ldifferent_\name
    addi a2, a2, -8
    addi a0, a0, 8
    addi a1, a1, 8
    bnez a2, .Lwindow_loop_\name
    j .Lequal_\name
.Ldifferent_\name:
.if \equality_only
    li a0, 1
    ret
.else
    andi a0, t0, 255
    andi t2, t1, 255
    sub a0, a0, t2
    bnez a0, .Ldifferent_return_\name
// Little-endian word ordering is not byte ordering. Discard equal low
// halves until the first unequal byte is in the low lane of both words.
    xor t2, t0, t1
    slli t3, t2, 32
    bnez t3, .Lhalf_\name
    srli t0, t0, 32
    srli t1, t1, 32
    srli t2, t2, 32
.Lhalf_\name:
    slli t3, t2, 48
    bnez t3, .Lbyte_\name
    srli t0, t0, 16
    srli t1, t1, 16
    srli t2, t2, 16
.Lbyte_\name:
    andi t3, t2, 255
    bnez t3, .Llow_\name
    srli t0, t0, 8
    srli t1, t1, 8
.Llow_\name:
    andi t0, t0, 255
    andi t1, t1, 255
    sub a0, t0, t1
.Ldifferent_return_\name:
    ret
.endif
.size \name, .-\name
.popsection
.endm

compare __wrap_memcmp, 0
compare __wrap_bcmp, 1
