// Extension-only operations, generic over the extension type E.

template <typename E>
kernel void jolt_test_ext_mul_base(device const E* a [[buffer(0)]],
                                   device const typename E::Base* x [[buffer(1)]],
                                   device E* out [[buffer(2)]],
                                   uint i [[thread_position_in_grid]]) {
    out[i] = mul_base(a[i], x[i]);
}
