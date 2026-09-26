// One kernel per field operation, generic over the field type. Operands are
// read straight from device memory, as consumer kernels do.

template <typename F>
kernel void jolt_test_field_add(device const F* a [[buffer(0)]],
                                device const F* b [[buffer(1)]],
                                device F* out [[buffer(2)]],
                                uint i [[thread_position_in_grid]]) {
    out[i] = a[i] + b[i];
}

template <typename F>
kernel void jolt_test_field_sub(device const F* a [[buffer(0)]],
                                device const F* b [[buffer(1)]],
                                device F* out [[buffer(2)]],
                                uint i [[thread_position_in_grid]]) {
    out[i] = a[i] - b[i];
}

template <typename F>
kernel void jolt_test_field_mul(device const F* a [[buffer(0)]],
                                device const F* b [[buffer(1)]],
                                device F* out [[buffer(2)]],
                                uint i [[thread_position_in_grid]]) {
    out[i] = a[i] * b[i];
}

template <typename F>
kernel void jolt_test_field_neg(device const F* a [[buffer(0)]],
                                device F* out [[buffer(1)]],
                                uint i [[thread_position_in_grid]]) {
    out[i] = -a[i];
}

template <typename F>
kernel void jolt_test_field_square(device const F* a [[buffer(0)]],
                                   device F* out [[buffer(1)]],
                                   uint i [[thread_position_in_grid]]) {
    out[i] = square(a[i]);
}

template <typename F>
kernel void jolt_test_field_mul_u64(device const F* a [[buffer(0)]],
                                    device const ulong* s [[buffer(1)]],
                                    device F* out [[buffer(2)]],
                                    uint i [[thread_position_in_grid]]) {
    out[i] = mul_u64(a[i], s[i]);
}

template <typename F>
kernel void jolt_test_field_mul_i64(device const F* a [[buffer(0)]],
                                    device const long* s [[buffer(1)]],
                                    device F* out [[buffer(2)]],
                                    uint i [[thread_position_in_grid]]) {
    out[i] = mul_i64(a[i], s[i]);
}

template <typename F>
kernel void jolt_test_field_from_u64(device const ulong* v [[buffer(0)]],
                                     device F* out [[buffer(1)]],
                                     uint i [[thread_position_in_grid]]) {
    out[i] = F::from_u64(v[i]);
}

template <typename F>
kernel void jolt_test_field_from_i64(device const long* v [[buffer(0)]],
                                     device F* out [[buffer(1)]],
                                     uint i [[thread_position_in_grid]]) {
    out[i] = F::from_i64(v[i]);
}

// Writes the all-ones pattern, which is not canonical for any 2^128 - C.
template <typename F>
kernel void jolt_test_field_write_non_canonical(device F* out [[buffer(0)]],
                                                uint i [[thread_position_in_grid]]) {
    out[i].limb = uint4(0xffffffffu);
}
