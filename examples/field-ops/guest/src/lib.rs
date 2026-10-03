//! eq-polynomial MLE evaluation in the proof field through the raw
//! field-inline instructions: eq(r, x) = prod_i (r_i·x_i + (1 − r_i)(1 − x_i)),
//! folded in the field register file and checked against a host-provided
//! expected value with FIELD_ASSERT_EQ.

#![cfg_attr(feature = "guest", no_std)]

/// Evaluates eq(r, x) over the `(r_i, x_i)` coordinate pairs in the field-inline
/// register file and FIELD_ASSERT_EQs it against the expected value, imported
/// from its canonical little-endian u64 limbs with `field_from_limbs!`. Returns
/// 42, read out of the field-inline file as `acc − expected + 42` with
/// `field_to_limbs!`.
#[jolt::provable(heap_size = 32768, max_trace_length = 65536)]
fn eval_eq_mle(pairs: [[u64; 2]; 4], expected_limbs: [u64; 4]) -> u64 {
    // field register map: field[0] = 1, field[1] = eq accumulator, field[2]/field[3] = (r_i, x_i),
    // field[4]-field[6] = per-pair scratch, field[8] = expected value,
    // field[10]/field[11] = result values; field[12] = memory round-trip value.
    jolt::field_load_imm!(0, 1);
    jolt::field_load_imm!(1, 1);
    for [r, x] in pairs {
        jolt::field_load_imm!(2, 0);
        jolt::field_load_accumulate_from_register!(2, r);
        jolt::field_load_imm!(3, 0);
        jolt::field_load_accumulate_from_register!(3, x);
        jolt::field_mul!(4, 2, 3); // r·x
        jolt::field_sub!(5, 0, 2); // 1 − r
        jolt::field_sub!(6, 0, 3); // 1 − x
        jolt::field_mul!(5, 5, 6); // (1 − r)(1 − x)
        jolt::field_add!(4, 4, 5); // the eq factor for this coordinate
        jolt::field_mul!(1, 1, 4); // fold it into the accumulator
    }

    // The host computes the expected value in the proof field, so its limbs
    // are canonical there; an fp128 value leaves the upper two limbs zero.
    jolt::field_from_limbs!(8, expected_limbs);
    jolt::field_assert_eq!(1, 8);

    // Memory ingress round-trips an independently pinned 128-bit integer
    // through the checked readout.
    #[cfg(target_arch = "riscv64")]
    {
        let limbs = [0x1234_5678_9abc_def0u64, 9];
        jolt::field_load_imm!(12, 0);
        // SAFETY: the two loads read this live two-word array through a0;
        // a1 is clobbered, and only field register 12 is changed.
        unsafe {
            core::arch::asm!(
                ".word {load_high}",
                ".word {load_low}",
                load_high = const jolt::field_inline::memory_accumulate_word(12, 1),
                load_low = const jolt::field_inline::memory_accumulate_word(12, 0),
                in("a0") limbs.as_ptr(),
                out("a1") _,
                options(nostack, readonly),
            );
        }
        // The readout restores register 12, so a second one reads it again.
        for _ in 0..2 {
            assert_eq!(jolt::field_to_limbs!(12, 2), limbs);
        }
    }
    // Exercise inversion separately: 3 · 3⁻¹ = 1 (field register 0 holds 1).
    jolt::field_load_imm!(13, 3);
    jolt::field_inv!(12, 13);
    jolt::field_mul!(12, 12, 13);
    jolt::field_assert_eq!(12, 0);

    // acc − expected is zero by the assert above, so the result fits one limb.
    jolt::field_sub!(10, 1, 8);
    jolt::field_load_imm!(11, 42);
    jolt::field_add!(10, 10, 11);
    let [result] = jolt::field_to_limbs!(10, 1);
    result
}
