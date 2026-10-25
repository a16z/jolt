// The CLI wraps external calls and leaves the existing implementation as
// __real_memcpy/__real_memmove. Small copies load completely before storing.
.macro copy name, fallback
.section .text.\name,"ax",@progbits
.globl \name
.hidden \name
.type \name,@function
\name:
.option push
.option norelax
    beqz a2, .Lreturn_\name
    or t0, a0, a1
    or t0, t0, a2
    andi t0, t0, 7
    bnez t0, .Lunaligned_\name
    li t0, 88
    bltu t0, a2, .Lgeneral_\name
    lla t0, .Lstoredispatch_\name
    srli a2, a2, 1
    sub t0, t0, a2
    jr t0
// Computed offsets require exactly four bytes per load/store.
.option push
.option norvc
    ld t1, 80(a1)
    ld t2, 72(a1)
    ld t3, 64(a1)
    ld t4, 56(a1)
    ld t5, 48(a1)
    ld t6, 40(a1)
    ld a3, 32(a1)
    ld a4, 24(a1)
    ld a5, 16(a1)
    ld a6, 8(a1)
    ld a7, 0(a1)
.option pop
.Lstoredispatch_\name:
    lla t0, .Lreturn_\name
    sub t0, t0, a2
    jr t0
.option push
.option norvc
    sd t1, 80(a0)
    sd t2, 72(a0)
    sd t3, 64(a0)
    sd t4, 56(a0)
    sd t5, 48(a0)
    sd t6, 40(a0)
    sd a3, 32(a0)
    sd a4, 24(a0)
    sd a5, 16(a0)
    sd a6, 8(a0)
    sd a7, 0(a0)
.option pop
.Lreturn_\name:
    ret
.Lunaligned_\name:
    li t0, 7
    bgeu t0, a2, .Lsmall_bytes_\name
    xor t0, a0, a1
    andi t0, t0, 7
    bnez t0, .Lmismatched_\name
// Equal offsets can reach word alignment without reading outside either range.
// Direction protects overlapping ranges, including the byte edges.
// The general path has more than 7 bytes, so alignment cannot exhaust it.
.Lgeneral_\name:
    beq a0, a1, .Lreturn_\name
    bltu a1, a0, .Lbackward_\name
    mv a7, a0
    andi t0, a1, 1
    beqz t0, .Lforward_align_1_\name
    lbu t1, 0(a1)
    sb t1, 0(a7)
    addi a1, a1, 1
    addi a7, a7, 1
    addi a2, a2, -1
.Lforward_align_1_\name:
    andi t0, a1, 2
    beqz t0, .Lforward_align_2_\name
    lhu t1, 0(a1)
    sh t1, 0(a7)
    addi a1, a1, 2
    addi a7, a7, 2
    addi a2, a2, -2
.Lforward_align_2_\name:
    andi t0, a1, 4
    beqz t0, .Lforward_align_4_\name
    lwu t1, 0(a1)
    sw t1, 0(a7)
    addi a1, a1, 4
    addi a7, a7, 4
    addi a2, a2, -4
.Lforward_align_4_\name:
    li t0, 256
    bltu a2, t0, .Lforward_blocks_\name
.Lforward_large_\name:
    ld t1, 0(a1)
    sd t1, 0(a7)
    ld t1, 8(a1)
    sd t1, 8(a7)
    ld t1, 16(a1)
    sd t1, 16(a7)
    ld t1, 24(a1)
    sd t1, 24(a7)
    ld t1, 32(a1)
    sd t1, 32(a7)
    ld t1, 40(a1)
    sd t1, 40(a7)
    ld t1, 48(a1)
    sd t1, 48(a7)
    ld t1, 56(a1)
    sd t1, 56(a7)
    ld t1, 64(a1)
    sd t1, 64(a7)
    ld t1, 72(a1)
    sd t1, 72(a7)
    ld t1, 80(a1)
    sd t1, 80(a7)
    ld t1, 88(a1)
    sd t1, 88(a7)
    ld t1, 96(a1)
    sd t1, 96(a7)
    ld t1, 104(a1)
    sd t1, 104(a7)
    ld t1, 112(a1)
    sd t1, 112(a7)
    ld t1, 120(a1)
    sd t1, 120(a7)
    ld t1, 128(a1)
    sd t1, 128(a7)
    ld t1, 136(a1)
    sd t1, 136(a7)
    ld t1, 144(a1)
    sd t1, 144(a7)
    ld t1, 152(a1)
    sd t1, 152(a7)
    ld t1, 160(a1)
    sd t1, 160(a7)
    ld t1, 168(a1)
    sd t1, 168(a7)
    ld t1, 176(a1)
    sd t1, 176(a7)
    ld t1, 184(a1)
    sd t1, 184(a7)
    ld t1, 192(a1)
    sd t1, 192(a7)
    ld t1, 200(a1)
    sd t1, 200(a7)
    ld t1, 208(a1)
    sd t1, 208(a7)
    ld t1, 216(a1)
    sd t1, 216(a7)
    ld t1, 224(a1)
    sd t1, 224(a7)
    ld t1, 232(a1)
    sd t1, 232(a7)
    ld t1, 240(a1)
    sd t1, 240(a7)
    ld t1, 248(a1)
    sd t1, 248(a7)
    addi a1, a1, 256
    addi a7, a7, 256
    addi a2, a2, -256
    bgeu a2, t0, .Lforward_large_\name
.Lforward_blocks_\name:
    li t0, 80
    bltu a2, t0, .Lforward_words_\name
.Lforward_block_\name:
    ld t1, 0(a1)
    ld t2, 8(a1)
    ld t3, 16(a1)
    ld t4, 24(a1)
    ld t5, 32(a1)
    ld t6, 40(a1)
    ld a3, 48(a1)
    ld a4, 56(a1)
    ld a5, 64(a1)
    ld a6, 72(a1)
    sd t1, 0(a7)
    sd t2, 8(a7)
    sd t3, 16(a7)
    sd t4, 24(a7)
    sd t5, 32(a7)
    sd t6, 40(a7)
    sd a3, 48(a7)
    sd a4, 56(a7)
    sd a5, 64(a7)
    sd a6, 72(a7)
    addi a1, a1, 80
    addi a7, a7, 80
    addi a2, a2, -80
    bgeu a2, t0, .Lforward_block_\name
.Lforward_words_\name:
    andi t0, a2, -8
    add t0, a1, t0
    andi a2, a2, 7
    beq a1, t0, .Lforward_bytes_\name
.Lforward_word_\name:
    ld t1, 0(a1)
    sd t1, 0(a7)
    addi a1, a1, 8
    addi a7, a7, 8
    bltu a1, t0, .Lforward_word_\name
.Lforward_bytes_\name:
    andi t0, a2, 4
    beqz t0, .Lforward_tail_4_\name
    lwu t1, 0(a1)
    sw t1, 0(a7)
    addi a1, a1, 4
    addi a7, a7, 4
.Lforward_tail_4_\name:
    andi t0, a2, 2
    beqz t0, .Lforward_tail_2_\name
    lhu t1, 0(a1)
    sh t1, 0(a7)
    addi a1, a1, 2
    addi a7, a7, 2
.Lforward_tail_2_\name:
    andi t0, a2, 1
    beqz t0, .Lreturn_\name
    lbu t1, 0(a1)
    sb t1, 0(a7)
    ret
.Lbackward_\name:
    add a1, a1, a2
    add a7, a0, a2
    andi t0, a1, 1
    beqz t0, .Lbackward_align_1_\name
    lbu t1, -1(a1)
    sb t1, -1(a7)
    addi a1, a1, -1
    addi a7, a7, -1
    addi a2, a2, -1
.Lbackward_align_1_\name:
    andi t0, a1, 2
    beqz t0, .Lbackward_align_2_\name
    lhu t1, -2(a1)
    sh t1, -2(a7)
    addi a1, a1, -2
    addi a7, a7, -2
    addi a2, a2, -2
.Lbackward_align_2_\name:
    andi t0, a1, 4
    beqz t0, .Lbackward_align_4_\name
    lwu t1, -4(a1)
    sw t1, -4(a7)
    addi a1, a1, -4
    addi a7, a7, -4
    addi a2, a2, -4
.Lbackward_align_4_\name:
    li t0, 256
    bltu a2, t0, .Lbackward_blocks_\name
.Lbackward_large_\name:
    ld t1, -8(a1)
    sd t1, -8(a7)
    ld t1, -16(a1)
    sd t1, -16(a7)
    ld t1, -24(a1)
    sd t1, -24(a7)
    ld t1, -32(a1)
    sd t1, -32(a7)
    ld t1, -40(a1)
    sd t1, -40(a7)
    ld t1, -48(a1)
    sd t1, -48(a7)
    ld t1, -56(a1)
    sd t1, -56(a7)
    ld t1, -64(a1)
    sd t1, -64(a7)
    ld t1, -72(a1)
    sd t1, -72(a7)
    ld t1, -80(a1)
    sd t1, -80(a7)
    ld t1, -88(a1)
    sd t1, -88(a7)
    ld t1, -96(a1)
    sd t1, -96(a7)
    ld t1, -104(a1)
    sd t1, -104(a7)
    ld t1, -112(a1)
    sd t1, -112(a7)
    ld t1, -120(a1)
    sd t1, -120(a7)
    ld t1, -128(a1)
    sd t1, -128(a7)
    ld t1, -136(a1)
    sd t1, -136(a7)
    ld t1, -144(a1)
    sd t1, -144(a7)
    ld t1, -152(a1)
    sd t1, -152(a7)
    ld t1, -160(a1)
    sd t1, -160(a7)
    ld t1, -168(a1)
    sd t1, -168(a7)
    ld t1, -176(a1)
    sd t1, -176(a7)
    ld t1, -184(a1)
    sd t1, -184(a7)
    ld t1, -192(a1)
    sd t1, -192(a7)
    ld t1, -200(a1)
    sd t1, -200(a7)
    ld t1, -208(a1)
    sd t1, -208(a7)
    ld t1, -216(a1)
    sd t1, -216(a7)
    ld t1, -224(a1)
    sd t1, -224(a7)
    ld t1, -232(a1)
    sd t1, -232(a7)
    ld t1, -240(a1)
    sd t1, -240(a7)
    ld t1, -248(a1)
    sd t1, -248(a7)
    ld t1, -256(a1)
    sd t1, -256(a7)
    addi a1, a1, -256
    addi a7, a7, -256
    addi a2, a2, -256
    bgeu a2, t0, .Lbackward_large_\name
.Lbackward_blocks_\name:
    li t0, 80
    bltu a2, t0, .Lbackward_words_\name
.Lbackward_block_\name:
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
    addi a1, a1, -80
    addi a7, a7, -80
    addi a2, a2, -80
    bgeu a2, t0, .Lbackward_block_\name
.Lbackward_words_\name:
    andi t0, a2, -8
    sub t0, a1, t0
    andi a2, a2, 7
    beq a1, t0, .Lbackward_bytes_\name
.Lbackward_word_\name:
    ld t1, -8(a1)
    sd t1, -8(a7)
    addi a1, a1, -8
    addi a7, a7, -8
    bltu t0, a1, .Lbackward_word_\name
.Lbackward_bytes_\name:
    andi t0, a2, 4
    beqz t0, .Lbackward_tail_4_\name
    lwu t1, -4(a1)
    sw t1, -4(a7)
    addi a1, a1, -4
    addi a7, a7, -4
.Lbackward_tail_4_\name:
    andi t0, a2, 2
    beqz t0, .Lbackward_tail_2_\name
    lhu t1, -2(a1)
    sh t1, -2(a7)
    addi a1, a1, -2
    addi a7, a7, -2
.Lbackward_tail_2_\name:
    andi t0, a2, 1
    beqz t0, .Lreturn_\name
    lbu t1, -1(a1)
    sb t1, -1(a7)
    ret
.Lsmall_bytes_\name:
    addi t0, a2, -1
    beqz t0, .Lone_byte_\name
    lla t0, .Lbyte_stores_\name
    slli a2, a2, 2
    sub t0, t0, a2
    jr t0
.option push
.option norvc
    lbu t1, 10(a1)
    lbu t2, 9(a1)
    lbu t3, 8(a1)
    lbu t4, 7(a1)
    lbu t5, 6(a1)
    lbu t6, 5(a1)
    lbu a3, 4(a1)
    lbu a4, 3(a1)
    lbu a5, 2(a1)
    lbu a6, 1(a1)
    lbu a7, 0(a1)
.option pop
.Lbyte_stores_\name:
    lla t0, .Lbyte_return_\name
    sub t0, t0, a2
    jr t0
.option push
.option norvc
    sb t1, 10(a0)
    sb t2, 9(a0)
    sb t3, 8(a0)
    sb t4, 7(a0)
    sb t5, 6(a0)
    sb t6, 5(a0)
    sb a3, 4(a0)
    sb a4, 3(a0)
    sb a5, 2(a0)
    sb a6, 1(a0)
    sb a7, 0(a0)
.option pop
.Lbyte_return_\name:
    ret
.Lone_byte_\name:
    lbu t1, 0(a1)
    sb t1, 0(a0)
    ret
.Lmismatched_\name:
    li t0, 11
    bgeu t0, a2, .Lsmall_bytes_\name
.Lfallback_\name:
    tail \fallback
.option pop
.size \name, .-\name
.endm

copy __wrap_memcpy, __real_memcpy
copy __wrap_memmove, __real_memmove
