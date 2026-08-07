struct SolinasHalfWidthOperand {
    ulong2 words;
};

struct SolinasHalfWidthProbeParams {
    uint elements;
    uint iterations;
};

kernel void solinas_half_width_mul_u64_probe(
    device const SolinasFp128* coefficients [[buffer(0)]],
    device const SolinasHalfWidthOperand* operands [[buffer(1)]],
    device SolinasFp128* output [[buffer(2)]],
    constant SolinasHalfWidthProbeParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < params.elements) {
        output[gid] = solinas_half_width_mul_u64(
            coefficients[gid], operands[gid].words.x);
    }
}

kernel void solinas_half_width_mul_signed_u64_probe(
    device const SolinasFp128* coefficients [[buffer(0)]],
    device const SolinasHalfWidthOperand* operands [[buffer(1)]],
    device SolinasFp128* output [[buffer(2)]],
    constant SolinasHalfWidthProbeParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < params.elements) {
        ulong2 operand = operands[gid].words;
        output[gid] = solinas_half_width_mul_signed_u64(
            coefficients[gid], operand.x, operand.y != 0ul);
    }
}

kernel void solinas_half_width_mul_u64_delta_probe(
    device const SolinasFp128* coefficients [[buffer(0)]],
    device const SolinasHalfWidthOperand* operands [[buffer(1)]],
    device SolinasFp128* output [[buffer(2)]],
    constant SolinasHalfWidthProbeParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < params.elements) {
        ulong2 endpoints = operands[gid].words;
        output[gid] = solinas_half_width_mul_u64_delta(
            coefficients[gid], endpoints.x, endpoints.y);
    }
}

#define HALF_WIDTH_APPLY_1(M) M(0)
#define HALF_WIDTH_APPLY_2(M) HALF_WIDTH_APPLY_1(M) M(1)
#define HALF_WIDTH_APPLY_4(M) HALF_WIDTH_APPLY_2(M) M(2) M(3)
#define HALF_WIDTH_APPLY_8(M) HALF_WIDTH_APPLY_4(M) M(4) M(5) M(6) M(7)

#define HALF_WIDTH_DECLARE_U64(LANE) \
    uint index_##LANE = base + LANE; \
    SolinasFp128 accumulator_##LANE = coefficients[index_##LANE]; \
    ulong factor_##LANE = operands[index_##LANE].words.x;

#define HALF_WIDTH_STEP_U64(LANE) \
    accumulator_##LANE = solinas_half_width_mul_u64( \
        accumulator_##LANE, factor_##LANE);

#define HALF_WIDTH_DECLARE_SIGNED_U64(LANE) \
    uint index_##LANE = base + LANE; \
    SolinasFp128 accumulator_##LANE = coefficients[index_##LANE]; \
    ulong magnitude_##LANE = operands[index_##LANE].words.x; \
    bool negative_##LANE = operands[index_##LANE].words.y != 0ul;

#define HALF_WIDTH_STEP_SIGNED_U64(LANE) \
    accumulator_##LANE = solinas_half_width_mul_signed_u64( \
        accumulator_##LANE, magnitude_##LANE, negative_##LANE);

#define HALF_WIDTH_DECLARE_U64_DELTA(LANE) \
    uint index_##LANE = base + LANE; \
    SolinasFp128 accumulator_##LANE = coefficients[index_##LANE]; \
    ulong2 endpoints_##LANE = operands[index_##LANE].words; \
    bool negative_##LANE = endpoints_##LANE.x < endpoints_##LANE.y; \
    ulong forward_##LANE = endpoints_##LANE.x - endpoints_##LANE.y; \
    ulong reverse_##LANE = endpoints_##LANE.y - endpoints_##LANE.x; \
    ulong magnitude_##LANE = negative_##LANE \
        ? reverse_##LANE \
        : forward_##LANE;

#define HALF_WIDTH_STEP_U64_DELTA(LANE) \
    accumulator_##LANE = solinas_half_width_mul_signed_u64( \
        accumulator_##LANE, magnitude_##LANE, negative_##LANE);

#define HALF_WIDTH_STORE(LANE) output[index_##LANE] = accumulator_##LANE;

// The host contract requires elements to be divisible by ILP. Explicit lane
// variables keep array indexing out of the hot loop and make spills auditable.
#define DEFINE_HALF_WIDTH_CHAIN(NAME, ILP, APPLY, DECLARE, STEP) \
kernel void NAME( \
    device const SolinasFp128* coefficients [[buffer(0)]], \
    device const SolinasHalfWidthOperand* operands [[buffer(1)]], \
    device SolinasFp128* output [[buffer(2)]], \
    constant SolinasHalfWidthProbeParams& params [[buffer(3)]], \
    uint gid [[thread_position_in_grid]]) \
{ \
    uint base = gid * ILP; \
    if (base >= params.elements) { \
        return; \
    } \
    APPLY(DECLARE) \
    for (uint iteration = 0u; iteration < params.iterations; iteration++) { \
        APPLY(STEP) \
    } \
    APPLY(HALF_WIDTH_STORE) \
}

DEFINE_HALF_WIDTH_CHAIN(
    solinas_half_width_chain_u64_ilp1,
    1u,
    HALF_WIDTH_APPLY_1,
    HALF_WIDTH_DECLARE_U64,
    HALF_WIDTH_STEP_U64)
DEFINE_HALF_WIDTH_CHAIN(
    solinas_half_width_chain_u64_ilp2,
    2u,
    HALF_WIDTH_APPLY_2,
    HALF_WIDTH_DECLARE_U64,
    HALF_WIDTH_STEP_U64)
DEFINE_HALF_WIDTH_CHAIN(
    solinas_half_width_chain_u64_ilp4,
    4u,
    HALF_WIDTH_APPLY_4,
    HALF_WIDTH_DECLARE_U64,
    HALF_WIDTH_STEP_U64)
DEFINE_HALF_WIDTH_CHAIN(
    solinas_half_width_chain_u64_ilp8,
    8u,
    HALF_WIDTH_APPLY_8,
    HALF_WIDTH_DECLARE_U64,
    HALF_WIDTH_STEP_U64)

DEFINE_HALF_WIDTH_CHAIN(
    solinas_half_width_chain_signed_u64_ilp1,
    1u,
    HALF_WIDTH_APPLY_1,
    HALF_WIDTH_DECLARE_SIGNED_U64,
    HALF_WIDTH_STEP_SIGNED_U64)
DEFINE_HALF_WIDTH_CHAIN(
    solinas_half_width_chain_signed_u64_ilp2,
    2u,
    HALF_WIDTH_APPLY_2,
    HALF_WIDTH_DECLARE_SIGNED_U64,
    HALF_WIDTH_STEP_SIGNED_U64)
DEFINE_HALF_WIDTH_CHAIN(
    solinas_half_width_chain_signed_u64_ilp4,
    4u,
    HALF_WIDTH_APPLY_4,
    HALF_WIDTH_DECLARE_SIGNED_U64,
    HALF_WIDTH_STEP_SIGNED_U64)
DEFINE_HALF_WIDTH_CHAIN(
    solinas_half_width_chain_signed_u64_ilp8,
    8u,
    HALF_WIDTH_APPLY_8,
    HALF_WIDTH_DECLARE_SIGNED_U64,
    HALF_WIDTH_STEP_SIGNED_U64)

DEFINE_HALF_WIDTH_CHAIN(
    solinas_half_width_chain_u64_delta_ilp1,
    1u,
    HALF_WIDTH_APPLY_1,
    HALF_WIDTH_DECLARE_U64_DELTA,
    HALF_WIDTH_STEP_U64_DELTA)
DEFINE_HALF_WIDTH_CHAIN(
    solinas_half_width_chain_u64_delta_ilp2,
    2u,
    HALF_WIDTH_APPLY_2,
    HALF_WIDTH_DECLARE_U64_DELTA,
    HALF_WIDTH_STEP_U64_DELTA)
DEFINE_HALF_WIDTH_CHAIN(
    solinas_half_width_chain_u64_delta_ilp4,
    4u,
    HALF_WIDTH_APPLY_4,
    HALF_WIDTH_DECLARE_U64_DELTA,
    HALF_WIDTH_STEP_U64_DELTA)
DEFINE_HALF_WIDTH_CHAIN(
    solinas_half_width_chain_u64_delta_ilp8,
    8u,
    HALF_WIDTH_APPLY_8,
    HALF_WIDTH_DECLARE_U64_DELTA,
    HALF_WIDTH_STEP_U64_DELTA)

#undef DEFINE_HALF_WIDTH_CHAIN
#undef HALF_WIDTH_STORE
#undef HALF_WIDTH_STEP_U64_DELTA
#undef HALF_WIDTH_DECLARE_U64_DELTA
#undef HALF_WIDTH_STEP_SIGNED_U64
#undef HALF_WIDTH_DECLARE_SIGNED_U64
#undef HALF_WIDTH_STEP_U64
#undef HALF_WIDTH_DECLARE_U64
#undef HALF_WIDTH_APPLY_8
#undef HALF_WIDTH_APPLY_4
#undef HALF_WIDTH_APPLY_2
#undef HALF_WIDTH_APPLY_1
