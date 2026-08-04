struct SolinasProbeParams {
    uint elements;
    uint iterations;
};

kernel void solinas_noop() {}

kernel void solinas_copy(
    device const SolinasFp128* lhs [[buffer(0)]],
    device const SolinasFp128* rhs [[buffer(1)]],
    device SolinasFp128* output [[buffer(2)]],
    constant SolinasProbeParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    (void)rhs;
    if (gid < params.elements) {
        output[gid] = lhs[gid];
    }
}

kernel void solinas_add_probe(
    device const SolinasFp128* lhs [[buffer(0)]],
    device const SolinasFp128* rhs [[buffer(1)]],
    device SolinasFp128* output [[buffer(2)]],
    constant SolinasProbeParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < params.elements) {
        output[gid] = solinas_add(lhs[gid], rhs[gid]);
    }
}

kernel void solinas_sub_probe(
    device const SolinasFp128* lhs [[buffer(0)]],
    device const SolinasFp128* rhs [[buffer(1)]],
    device SolinasFp128* output [[buffer(2)]],
    constant SolinasProbeParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < params.elements) {
        output[gid] = solinas_sub(lhs[gid], rhs[gid]);
    }
}

kernel void solinas_mul_wide_probe(
    device const SolinasFp128* lhs [[buffer(0)]],
    device const SolinasFp128* rhs [[buffer(1)]],
    device SolinasFp128* output [[buffer(2)]],
    constant SolinasProbeParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < params.elements) {
        output[gid] = solinas_mul_wide(lhs[gid], rhs[gid]);
    }
}

#define DEFINE_CHAIN_PROBE(NAME, ILP, MUL) \
kernel void NAME( \
    device const SolinasFp128* lhs [[buffer(0)]], \
    device const SolinasFp128* rhs [[buffer(1)]], \
    device SolinasFp128* output [[buffer(2)]], \
    constant SolinasProbeParams& params [[buffer(3)]], \
    uint gid [[thread_position_in_grid]]) \
{ \
    uint base = gid * ILP; \
    if (base >= params.elements) { \
        return; \
    } \
    SolinasFp128 accumulators[ILP]; \
    SolinasFp128 factors[ILP]; \
    for (uint lane = 0; lane < ILP; lane++) { \
        uint index = base + lane; \
        accumulators[lane] = index < params.elements ? lhs[index] : solinas_zero(); \
        factors[lane] = index < params.elements ? rhs[index] : solinas_zero(); \
    } \
    for (uint iteration = 0; iteration < params.iterations; iteration++) { \
        for (uint lane = 0; lane < ILP; lane++) { \
            accumulators[lane] = MUL(accumulators[lane], factors[lane]); \
        } \
    } \
    for (uint lane = 0; lane < ILP; lane++) { \
        uint index = base + lane; \
        if (index < params.elements) { \
            output[index] = accumulators[lane]; \
        } \
    } \
}

DEFINE_CHAIN_PROBE(solinas_chain_wide_1, 1, solinas_mul_wide)
DEFINE_CHAIN_PROBE(solinas_chain_wide_2, 2, solinas_mul_wide)
DEFINE_CHAIN_PROBE(solinas_chain_wide_4, 4, solinas_mul_wide)
DEFINE_CHAIN_PROBE(solinas_chain_wide_8, 8, solinas_mul_wide)

kernel void solinas_u32_mad_ilp8(
    device const uint4* lhs [[buffer(0)]],
    device const uint4* rhs [[buffer(1)]],
    device uint4* output [[buffer(2)]],
    constant SolinasProbeParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.elements) {
        return;
    }
    uint4 x = lhs[gid];
    uint4 y = rhs[gid];
    uint4 mx = rhs[gid] | 1u;
    uint4 my = lhs[gid] | 1u;
    for (uint iteration = 0; iteration < params.iterations; iteration++) {
        x = x * mx + uint4(0x9e3779b9u, 0x7f4a7c15u, 0xf39cc060u, 0x106aa070u);
        y = y * my + uint4(0x94d049bbu, 0x369dea0fu, 0xd2b74407u, 0xb7e15163u);
    }
    output[gid] = x ^ y;
}
