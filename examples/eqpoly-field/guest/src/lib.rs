//! eq-polynomial MLE evaluation over BN254 Fr through the raw field-inline
//! instructions: eq(r, x) = prod_i (r_i·x_i + (1 − r_i)(1 − x_i)), folded in
//! the FR register file and checked against a host-provided expected value
//! with FIELD_ASSERT_EQ.

#![cfg_attr(feature = "guest", no_std)]

/// Evaluates eq(r, x) over the `(r_i, x_i)` coordinate pairs in the FR
/// register file and FIELD_ASSERT_EQs it against the expected value, supplied
/// as canonical little-endian u64 limbs and recomposed in-field (Horner in
/// radix 2^64, the radix built by repeated squaring of a LoadImm 2). Returns
/// 42, bridged out of the FR file as `acc − expected + 42` — provably small,
/// so the StoreToX range restriction holds exactly when the assert did.
#[jolt::provable(heap_size = 32768, max_trace_length = 65536)]
fn eval_eq_mle(pairs: [[u64; 2]; 4], expected_limbs: [u64; 4]) -> u64 {
    // FR register map: fr0 = 1, fr1 = eq accumulator, fr2/fr3 = (r_i, x_i),
    // fr4-fr6 = per-pair scratch, fr7 = 2^64, fr8 = recomposed expected
    // value, fr9 = limb bridge, fr10/fr11 = result-bridge scratch.
    jolt::field_load_imm!(0, 1);
    jolt::field_load_imm!(1, 1);
    for [r, x] in pairs {
        jolt::field_load_from_x!(2, r);
        jolt::field_load_from_x!(3, x);
        jolt::field_mul!(4, 2, 3); // r·x
        jolt::field_sub!(5, 0, 2); // 1 − r
        jolt::field_sub!(6, 0, 3); // 1 − x
        jolt::field_mul!(5, 5, 6); // (1 − r)(1 − x)
        jolt::field_add!(4, 4, 5); // the eq factor for this coordinate
        jolt::field_mul!(1, 1, 4); // fold it into the accumulator
    }

    // 2^64 by repeated squaring of 2: LoadImm immediates are 12-bit, so the
    // limb radix cannot be loaded directly.
    jolt::field_load_imm!(7, 2);
    jolt::field_mul!(7, 7, 7); // 2^2
    jolt::field_mul!(7, 7, 7); // 2^4
    jolt::field_mul!(7, 7, 7); // 2^8
    jolt::field_mul!(7, 7, 7); // 2^16
    jolt::field_mul!(7, 7, 7); // 2^32
    jolt::field_mul!(7, 7, 7); // 2^64

    // Horner-recompose the expected value: ((l3·2^64 + l2)·2^64 + l1)·2^64 + l0.
    // Each limb crosses the bridge as a u64; only the FR-internal partial sums
    // exceed 64 bits.
    let [l0, l1, l2, l3] = expected_limbs;
    jolt::field_load_from_x!(8, l3);
    for limb in [l2, l1, l0] {
        jolt::field_mul!(8, 8, 7);
        jolt::field_load_from_x!(9, limb);
        jolt::field_add!(8, 8, 9);
    }

    jolt::field_assert_eq!(1, 8);

    // acc − expected is zero by the assert above, so the bridged result is
    // exactly 42 and the StoreToX < 2^64 range restriction holds.
    jolt::field_sub!(10, 1, 8);
    jolt::field_load_imm!(11, 42);
    jolt::field_add!(10, 10, 11);
    jolt::field_store_to_x!(10)
}
