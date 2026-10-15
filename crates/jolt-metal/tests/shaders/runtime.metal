#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void jolt_test_vec_add(device const T* a [[buffer(0)]],
                              device const T* b [[buffer(1)]],
                              device T* out [[buffer(2)]],
                              uint i [[thread_position_in_grid]]) {
    out[i] = a[i] + b[i];
}

kernel void jolt_test_fill_bytes(device uchar* out [[buffer(0)]],
                                 constant uchar& value [[buffer(1)]],
                                 uint i [[thread_position_in_grid]]) {
    out[i] = value;
}
