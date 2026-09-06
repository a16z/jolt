struct alignas(16) Radix26State {
    int32_t digit[5][4];

    U128 value(unsigned component) const {
        U128 result = 0;
        for (unsigned index = 0; index < 5; ++index) {
            const int64_t coefficient = digit[index][component];
            uint64_t magnitude = uint64_t(coefficient < 0 ? -coefficient : coefficient);
            U128 power = U128(1) << (26 * index);
            U128 term = 0;
            while (magnitude) {
                if (magnitude & 1) term = add(term, power);
                power = add(power, power);
                magnitude >>= 1;
            }
            result = coefficient < 0 ? sub(result, term) : add(result, term);
        }
        return result;
    }
};
static_assert(sizeof(Radix26State) == 80);

static void check_radix26_normalizer(id<MTLDevice> device, id<MTLCommandQueue> queue,
                                    id<MTLComputePipelineState> pipeline) {
    constexpr int64_t base = int64_t(1) << 26, top = int64_t(1) << 24;
    constexpr int64_t small = (int64_t(1) << 32) - OFFSET;
    const int64_t lower[] = {-17 * small - 16 * (base - 1), -17 * 64 - 16 * (base - 1),
        -16 * (base - 1), -16 * (base - 1), -16 * (top - 1)};
    const int64_t upper[] = {17 * (base - 1) + 17 * small, 17 * (base - 1) + 17 * 64,
        17 * (base - 1), 17 * (base - 1), 17 * (top - 1)};
    std::vector<Radix26State> inputs(128);
    for (unsigned index = 0; index < inputs.size(); ++index)
        for (unsigned component = 0; component < 4; ++component) {
            const unsigned state = index * 4 + component;
            for (unsigned digit = 0; digit < 5; ++digit) {
                const int64_t value = state < 32 ? ((state >> digit) & 1 ? upper[digit] : lower[digit])
                    : lower[digit] + int64_t(mix(state * 5 + digit) % uint64_t(upper[digit] - lower[digit] + 1));
                require(value >= INT32_MIN && value <= INT32_MAX, "radix26 input range");
                inputs[index].digit[digit][component] = int32_t(value);
            }
        }
    id<MTLBuffer> input = [device newBufferWithBytes:inputs.data() length:inputs.size() * sizeof(Radix26State)
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [device newBufferWithLength:input.length options:MTLResourceStorageModeShared];
    require(input && output && pipeline.maxTotalThreadsPerThreadgroup >= 128, "radix26 probe resources");
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(inputs.size(), 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
    const auto started = Clock::now();
    [command commit];
    finish_command(command, started);
    const auto *actual = static_cast<const Radix26State *>(output.contents);
    for (unsigned index = 0; index < inputs.size(); ++index)
        for (unsigned component = 0; component < 4; ++component) {
            require(actual[index].value(component) == inputs[index].value(component),
                "radix26 independent weighted modular sum");
            require(actual[index].digit[0][component] >= -17 * small
                && actual[index].digit[0][component] <= base - 1 + 17 * small
                && actual[index].digit[1][component] >= -17 * 64
                && actual[index].digit[1][component] <= base - 1 + 17 * 64,
                "radix26 folded low-digit bounds");
            for (unsigned digit = 2; digit < 5; ++digit)
                require(actual[index].digit[digit][component] >= 0
                    && actual[index].digit[digit][component] < (digit == 4 ? top : base),
                    "radix26 normalized high-digit bounds");
        }
    std::puts("RADIX26_NORMALIZER states=512 corners=32 modular_sum=pass post_bounds=pass");
    std::fflush(stdout);
}
