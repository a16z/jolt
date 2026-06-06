#![cfg_attr(feature = "guest", no_std)]

#[jolt::provable(heap_size = 32768, max_trace_length = 65536)]
fn field_inline_eq_poly(input: u32) -> u32 {
    jolt::field_load_imm!(1, 1);
    jolt::field_load_imm!(2, 0);

    jolt::field_sub!(3, 1, 1);
    jolt::field_sub!(4, 1, 1);
    jolt::field_mul!(5, 1, 1);
    jolt::field_mul!(6, 3, 4);
    jolt::field_add!(7, 5, 6);

    jolt::field_sub!(8, 1, 2);
    jolt::field_sub!(9, 1, 2);
    jolt::field_mul!(10, 2, 2);
    jolt::field_mul!(11, 8, 9);
    jolt::field_add!(12, 10, 11);

    jolt::field_mul!(13, 7, 12);
    jolt::field_assert_eq!(13, 1);

    input ^ 1
}
