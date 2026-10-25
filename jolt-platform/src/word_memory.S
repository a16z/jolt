/* RV64 little-endian memory routines. Only caller-saved registers are used.
 * memcpy reads/writes exactly the requested ranges and returns the original
 * destination. Overlapping ranges retain memcpy's ordinary undefined contract.
 * memcmp returns the difference of the first unequal unsigned bytes.
 */
.text
.option push
.option norvc
.p2align 2
.globl memcpy
.type memcpy,@function
memcpy:
    beqz a2, .Lmc_done
    mv t0, a0
    xor t1, a0, a1
    andi t1, t1, 7
    bnez t1, .Lmc_shift_prefix
.Lmc_align:
    andi t1, t0, 7
    beqz t1, .Lmc_aligned
    lbu t2, 0(a1)
    sb t2, 0(t0)
    addi a1, a1, 1
    addi t0, t0, 1
    addi a2, a2, -1
    bnez a2, .Lmc_align
    ret
.Lmc_aligned:
    li t1, 64
    bltu a2, t1, .Lmc_words
.Lmc_64:
    ld t2, 0(a1)
    sd t2, 0(t0)
    ld t2, 8(a1)
    sd t2, 8(t0)
    ld t2, 16(a1)
    sd t2, 16(t0)
    ld t2, 24(a1)
    sd t2, 24(t0)
    ld t2, 32(a1)
    sd t2, 32(t0)
    ld t2, 40(a1)
    sd t2, 40(t0)
    ld t2, 48(a1)
    sd t2, 48(t0)
    ld t2, 56(a1)
    sd t2, 56(t0)
    addi a1, a1, 64
    addi t0, t0, 64
    addi a2, a2, -64
    bgeu a2, t1, .Lmc_64
.Lmc_words:
    li t1, 8
    bltu a2, t1, .Lmc_byte_check
.Lmc_word:
    ld t2, 0(a1)
    sd t2, 0(t0)
    addi a1, a1, 8
    addi t0, t0, 8
    addi a2, a2, -8
    bgeu a2, t1, .Lmc_word
    j .Lmc_byte_check

/* Align the source first, then the destination. Thereafter floor(src/8)*8
 * is at or above the first aligned source address inside the allocation.
 * If fewer bytes exist, the byte loop returns before any wide load.
 */
.Lmc_shift_prefix:
    andi t1, a1, 7
    beqz t1, .Lmc_shift_align
.Lmc_source_align:
    lbu t2, 0(a1)
    sb t2, 0(t0)
    addi a1, a1, 1
    addi t0, t0, 1
    addi a2, a2, -1
    beqz a2, .Lmc_done
    andi t1, a1, 7
    bnez t1, .Lmc_source_align
.Lmc_shift_align:
    andi t1, t0, 7
    beqz t1, .Lmc_shift_init
    lbu t2, 0(a1)
    sb t2, 0(t0)
    addi a1, a1, 1
    addi t0, t0, 1
    addi a2, a2, -1
    bnez a2, .Lmc_shift_align
    ret
.Lmc_shift_init:
    li t1, 16
    bltu a2, t1, .Lmc_byte
    andi t5, a1, 7
    slli t3, t5, 3
    li t4, 64
    sub t4, t4, t3
    andi a3, a1, -8
    ld t2, 0(a3)
    li t1, 40
    bltu a2, t1, .Lmc_shift_words
/* At least 40 source bytes remain. The highest aligned read ends no later
 * than src+40. Four exact destination words are produced, and the final
 * source word becomes the next iteration's lower word.
 */
.Lmc_shift_32:
    ld t6, 8(a3)
    srl a4, t2, t3
    sll a6, t6, t4
    or a4, a4, a6
    sd a4, 0(t0)
    ld t2, 16(a3)
    srl a4, t6, t3
    sll a6, t2, t4
    or a4, a4, a6
    sd a4, 8(t0)
    ld t6, 24(a3)
    srl a4, t2, t3
    sll a6, t6, t4
    or a4, a4, a6
    sd a4, 16(t0)
    ld t2, 32(a3)
    srl a4, t6, t3
    sll a6, t2, t4
    or a4, a4, a6
    sd a4, 24(t0)
    addi a3, a3, 32
    addi t0, t0, 32
    addi a2, a2, -32
    bgeu a2, t1, .Lmc_shift_32
.Lmc_shift_words:
    li t1, 16
    bltu a2, t1, .Lmc_shift_tail
.Lmc_shift_word:
    ld t6, 8(a3)
    srl a4, t2, t3
    sll a6, t6, t4
    or a4, a4, a6
    sd a4, 0(t0)
    mv t2, t6
    addi a3, a3, 8
    addi t0, t0, 8
    addi a2, a2, -8
    bgeu a2, t1, .Lmc_shift_word
.Lmc_shift_tail:
    add a1, a3, t5
.Lmc_byte_check:
    beqz a2, .Lmc_done
.Lmc_byte:
    lbu t2, 0(a1)
    sb t2, 0(t0)
    addi a1, a1, 1
    addi t0, t0, 1
    addi a2, a2, -1
    bnez a2, .Lmc_byte
.Lmc_done:
    ret
.size memcpy, .-memcpy

.p2align 2
.globl memcmp
.type memcmp,@function
memcmp:
    beqz a2, .Lcmp_equal
    xor t0, a0, a1
    andi t0, t0, 7
    bnez t0, .Lcmp_byte
.Lcmp_align:
    andi t0, a0, 7
    beqz t0, .Lcmp_words
    lbu t1, 0(a0)
    lbu t2, 0(a1)
    bne t1, t2, .Lcmp_different
    addi a0, a0, 1
    addi a1, a1, 1
    addi a2, a2, -1
    bnez a2, .Lcmp_align
    j .Lcmp_equal
.Lcmp_words:
    li t0, 8
    bltu a2, t0, .Lcmp_byte_check
.Lcmp_word:
    ld t1, 0(a0)
    ld t2, 0(a1)
    bne t1, t2, .Lcmp_byte
    addi a0, a0, 8
    addi a1, a1, 8
    addi a2, a2, -8
    bgeu a2, t0, .Lcmp_word
.Lcmp_byte_check:
    beqz a2, .Lcmp_equal
.Lcmp_byte:
    lbu t1, 0(a0)
    lbu t2, 0(a1)
    bne t1, t2, .Lcmp_different
    addi a0, a0, 1
    addi a1, a1, 1
    addi a2, a2, -1
    bnez a2, .Lcmp_byte
.Lcmp_equal:
    li a0, 0
    ret
.Lcmp_different:
    sub a0, t1, t2
    ret
.size memcmp, .-memcmp
.option pop
.section .note.GNU-stack,"",@progbits
