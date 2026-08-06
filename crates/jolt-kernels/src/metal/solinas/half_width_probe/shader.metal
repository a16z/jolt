struct SolinasHalfWidthWide192 {
    uint limb[6];
};

struct SolinasHalfWidthOperand {
    ulong2 words;
};

struct SolinasHalfWidthProbeParams {
    uint elements;
    uint iterations;
};

inline SolinasHalfWidthWide192 solinas_half_width_product_u64(
    SolinasFp128 lhs,
    ulong rhs)
{
    uint rhs_lo = (uint)rhs;
    uint rhs_hi = (uint)(rhs >> 32);
    SolinasHalfWidthWide192 product;
    product.limb[0] = 0u;
    product.limb[1] = 0u;
    product.limb[2] = 0u;
    product.limb[3] = 0u;
    product.limb[4] = 0u;
    product.limb[5] = 0u;

    ulong carry = 0ul;
    ulong word = (ulong)lhs.limb[0] * (ulong)rhs_lo;
    product.limb[0] = (uint)word;
    carry = word >> 32;
    word = (ulong)lhs.limb[0] * (ulong)rhs_hi + carry;
    product.limb[1] = (uint)word;
    product.limb[2] = (uint)(word >> 32);

    word = (ulong)lhs.limb[1] * (ulong)rhs_lo
        + (ulong)product.limb[1];
    product.limb[1] = (uint)word;
    carry = word >> 32;
    word = (ulong)lhs.limb[1] * (ulong)rhs_hi
        + (ulong)product.limb[2]
        + carry;
    product.limb[2] = (uint)word;
    product.limb[3] = (uint)(word >> 32);

    word = (ulong)lhs.limb[2] * (ulong)rhs_lo
        + (ulong)product.limb[2];
    product.limb[2] = (uint)word;
    carry = word >> 32;
    word = (ulong)lhs.limb[2] * (ulong)rhs_hi
        + (ulong)product.limb[3]
        + carry;
    product.limb[3] = (uint)word;
    product.limb[4] = (uint)(word >> 32);

    word = (ulong)lhs.limb[3] * (ulong)rhs_lo
        + (ulong)product.limb[3];
    product.limb[3] = (uint)word;
    carry = word >> 32;
    word = (ulong)lhs.limb[3] * (ulong)rhs_hi
        + (ulong)product.limb[4]
        + carry;
    product.limb[4] = (uint)word;
    product.limb[5] = (uint)(word >> 32);
    return product;
}

inline SolinasFp128 solinas_half_width_reduce_u192(
    SolinasHalfWidthWide192 product)
{
    SolinasFp128 folded;
    ulong word = (ulong)product.limb[4] * (ulong)SOLINAS_OFFSET
        + (ulong)product.limb[0];
    folded.limb[0] = (uint)word;
    ulong carry = word >> 32;
    word = (ulong)product.limb[5] * (ulong)SOLINAS_OFFSET
        + (ulong)product.limb[1]
        + carry;
    folded.limb[1] = (uint)word;
    carry = word >> 32;
    word = (ulong)product.limb[2] + carry;
    folded.limb[2] = (uint)word;
    carry = word >> 32;
    word = (ulong)product.limb[3] + carry;
    folded.limb[3] = (uint)word;
    ulong first_fold_carry = word >> 32;

    // For a canonical 128-by-64 product, the first carry is at most one and
    // its residue is below 2^96. Adding one offset therefore cannot overflow.
    word = (ulong)folded.limb[0]
        + first_fold_carry * (ulong)SOLINAS_OFFSET;
    folded.limb[0] = (uint)word;
    carry = word >> 32;
    word = (ulong)folded.limb[1] + carry;
    folded.limb[1] = (uint)word;
    carry = word >> 32;
    word = (ulong)folded.limb[2] + carry;
    folded.limb[2] = (uint)word;
    carry = word >> 32;
    word = (ulong)folded.limb[3] + carry;
    folded.limb[3] = (uint)word;

    SolinasCorrection corrected = solinas_add_offset(folded);
    return solinas_select(corrected.carry != 0u, corrected.value, folded);
}

inline SolinasFp128 solinas_half_width_mul_u64(
    SolinasFp128 coefficient,
    ulong scalar)
{
    return solinas_half_width_reduce_u192(
        solinas_half_width_product_u64(coefficient, scalar));
}

inline SolinasFp128 solinas_half_width_mul_signed_u64(
    SolinasFp128 coefficient,
    ulong magnitude,
    bool negative)
{
    SolinasFp128 positive = solinas_half_width_mul_u64(coefficient, magnitude);
    SolinasFp128 negated = solinas_sub(solinas_zero(), positive);
    return solinas_select(!negative, positive, negated);
}

inline SolinasFp128 solinas_half_width_mul_u64_delta(
    SolinasFp128 coefficient,
    ulong minuend,
    ulong subtrahend)
{
    bool negative = minuend < subtrahend;
    ulong forward = minuend - subtrahend;
    ulong reverse = subtrahend - minuend;
    ulong magnitude = negative ? reverse : forward;
    return solinas_half_width_mul_signed_u64(
        coefficient, magnitude, negative);
}

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
