//! eq-polynomial MLE evaluation over BN254 Fr through the raw field-inline
//! instructions: eq(r, x) = prod_i (r_i·x_i + (1 − r_i)(1 − x_i)), folded in
//! the field register file and checked against a host-provided expected value
//! with FIELD_ASSERT_EQ.

#![cfg_attr(feature = "guest", no_std)]

/// Evaluates eq(r, x) over the `(r_i, x_i)` coordinate pairs in the field-inline
/// register file and FIELD_ASSERT_EQs it against the expected value, supplied
/// as canonical little-endian u64 limbs and recomposed in-field by accumulating
/// the limbs from most significant to least significant. Returns
/// 42, bridged out of the field-inline file as `acc − expected + 42` — provably small,
/// so the StoreToX range restriction holds exactly when the assert did.
#[jolt::provable(heap_size = 32768, max_trace_length = 65536)]
fn eval_eq_mle(pairs: [[u64; 2]; 4], expected_limbs: [u64; 4]) -> u64 {
    // field register map: field[0] = 1, field[1] = eq accumulator, field[2]/field[3] = (r_i, x_i),
    // field[4]-field[6] = per-pair scratch, field[8] = recomposed expected value,
    // field[10]/field[11] = result-bridge scratch.
    jolt::field_load_imm!(0, 1);
    jolt::field_load_imm!(1, 1);
    for [r, x] in pairs {
        jolt::field_load_imm!(2, 0);
        jolt::field_load_accumulate_from_x!(2, r);
        jolt::field_load_imm!(3, 0);
        jolt::field_load_accumulate_from_x!(3, x);
        jolt::field_mul!(4, 2, 3); // r·x
        jolt::field_sub!(5, 0, 2); // 1 − r
        jolt::field_sub!(6, 0, 3); // 1 − x
        jolt::field_mul!(5, 5, 6); // (1 − r)(1 − x)
        jolt::field_add!(4, 4, 5); // the eq factor for this coordinate
        jolt::field_mul!(1, 1, 4); // fold it into the accumulator
    }

    // Horner-recompose the expected value: ((l3·2^64 + l2)·2^64 + l1)·2^64 + l0.
    // Each limb crosses the bridge as a u64; only the in-field partial sums
    // exceed 64 bits.
    let [l0, l1, l2, l3] = expected_limbs;
    jolt::field_load_imm!(8, 0);
    for limb in [l3, l2, l1, l0] {
        jolt::field_load_accumulate_from_x!(8, limb);
    }

    jolt::field_assert_eq!(1, 8);

    // Memory ingress and advice egress round-trip an independently pinned
    // 128-bit integer, below both supported proof fields' moduli.
    #[cfg(target_arch = "riscv64")]
    {
        let limbs = [0x1234_5678_9abc_def0u64, 9];
        let low: u64;
        jolt::field_load_imm!(12, 0);
        // SAFETY: the two loads read this live two-word array through a0;
        // a1 is clobbered, and only field registers 12 and 13 are changed.
        unsafe {
            core::arch::asm!(
                ".word {load_high}",
                ".word {load_low}",
                ".word {advice}",
                load_high = const jolt::field_inline_r_word(0x61, 5, 11, 10, 12),
                load_low = const jolt::field_inline_r_word(0x60, 5, 11, 10, 12),
                advice = const jolt::field_inline_r_word(1, 6, 11, 12, 13),
                in("a0") limbs.as_ptr(),
                out("a1") low,
                options(nostack, readonly),
            );
        }
        let high = jolt::field_store_to_x!(13);
        assert_eq!([low, high], limbs);
    }
    // Exercise inversion separately: 3 · 3⁻¹ = 1 (field register 0 holds 1).
    jolt::field_load_imm!(13, 3);
    jolt::field_inv!(12, 13);
    jolt::field_mul!(12, 12, 13);
    jolt::field_assert_eq!(12, 0);

    // acc − expected is zero by the assert above, so the bridged result is
    // exactly 42 and the StoreToX < 2^64 range restriction holds.
    jolt::field_sub!(10, 1, 8);
    jolt::field_load_imm!(11, 42);
    jolt::field_add!(10, 10, 11);
    jolt::field_store_to_x!(10)
}
