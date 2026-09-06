#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>

using Clock = std::chrono::steady_clock;
using U128 = unsigned __int128;
static constexpr uint32_t OFFSET = 0xffffa7f7u;
static constexpr U128 MODULUS = U128(0) - U128(OFFSET);

struct PackedParams {
    uint64_t rows, columns, lane_stride, capacity, k, d, rank, positions, digits;
    uint64_t blocks, full_blocks, boundary_columns, tasks, task_offset, dispatch_tasks;
    uint64_t lane_row_offset, output_coefficients, columns_per_group, partials;
    uint64_t positions_per_partial, log_d, zero_mask;
};
static_assert(sizeof(PackedParams) == 176);

static void require(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "DIAGNOSTIC_FAILURE %s\n", message);
        std::exit(1);
    }
}

static uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

static U128 add(U128 a, U128 b) {
    U128 sum = a + b;
    if (sum < a) sum += OFFSET;
    return sum >= MODULUS ? sum - MODULUS : sum;
}

static U128 sub(U128 a, U128 b) {
    return a >= b ? a - b : MODULUS - (b - a);
}

struct Case {
    PackedParams params;
    id<MTLBuffer> matrix, lanes, output, zero_rows;
    std::vector<uint64_t> task_hot;

    Case(id<MTLDevice> device, uint64_t positions, uint64_t blocks,
         uint64_t columns, uint64_t maximum_tasks, uint64_t zero_mask) {
        const uint64_t rows = positions / 2 * blocks;
        params = {rows, columns, columns, 32, 256, 128, 3, positions, 1,
            blocks, blocks, 0, blocks * columns, 0, maximum_tasks, 0,
            32 * blocks * 3 * 128, 1, 16, positions / 16, 7, zero_mask};
        const uint64_t matrix_count = 3 * positions * 128;
        matrix = [device newBufferWithLength:matrix_count * 16 options:MTLResourceStorageModeShared];
        lanes = [device newBufferWithLength:rows * columns options:MTLResourceStorageModeShared];
        output = [device newBufferWithLength:params.output_coefficients * 16 * 16
            options:MTLResourceStorageModeShared];
        zero_rows = [device newBufferWithLength:rows / 8 options:MTLResourceStorageModeShared];
        require(matrix && lanes && output && zero_rows, "buffer allocation");
        auto *values = static_cast<U128 *>(matrix.contents);
        auto *hot = static_cast<uint8_t *>(lanes.contents);
        auto *zero = static_cast<uint64_t *>(zero_rows.contents);
        for (uint64_t index = 0; index < matrix_count; ++index) {
            U128 value = U128(mix(index)) << 64 | mix(index + 13);
            values[index] = value >= MODULUS ? value - MODULUS : value;
        }
        const uint64_t touched_blocks = (maximum_tasks + columns - 1) / columns;
        const uint64_t touched_rows = touched_blocks * positions / 2;
        require(touched_rows <= rows, "selector prefix bounds");
        std::memset(zero_rows.contents, 0, zero_rows.length);
        std::memset(output.contents, 0, output.length);
        task_hot.resize(touched_blocks * columns);
        for (uint64_t row = 0; row < touched_rows; ++row) {
            const bool active_zero = row % 11 == 0;
            if (active_zero) zero[row / 64] |= uint64_t(1) << (row % 64);
            for (uint64_t column = 0; column < columns; ++column) {
                const uint64_t index = row * columns + column;
                const uint64_t random = mix(index);
                const uint8_t value = random % 100 < 56 ? uint8_t(mix(random) % 255 + 1) : 0;
                hot[index] = value;
                if (value || (active_zero && ((zero_mask >> column) & 1)))
                    ++task_hot[(row / (positions / 2)) * columns + column];
            }
        }
    }

    U128 oracle(uint64_t task, uint64_t part, uint64_t element, uint64_t coefficient) const {
        const uint64_t block = task / params.columns, column = task % params.columns;
        const auto *values = static_cast<const U128 *>(matrix.contents);
        const auto *hot = static_cast<const uint8_t *>(lanes.contents);
        const auto *zero = static_cast<const uint64_t *>(zero_rows.contents);
        U128 result = 0;
        const uint64_t part_rows = params.positions_per_partial / 2;
        for (uint64_t local = 0; local < part_rows; ++local) {
            const uint64_t block_row = part * part_rows + local;
            const uint64_t row = block * params.positions / 2 + block_row;
            const uint8_t selected = hot[row * params.columns + column];
            const bool selected_zero = ((params.zero_mask >> column) & 1)
                && ((zero[row / 64] >> (row % 64)) & 1);
            if (!selected && !selected_zero) continue;
            const uint64_t position = 2 * block_row + (selected >> 7);
            const uint64_t shift = selected & 127;
            const U128 value = values[(element * params.positions + position) * 128
                + ((coefficient + 128 - shift) % 128)];
            result = coefficient >= shift ? add(result, value) : sub(result, value);
        }
        return result;
    }

    size_t output_index(uint64_t task, uint64_t part, uint64_t element,
                        uint64_t coefficient) const {
        const uint64_t block = task / params.columns, column = task % params.columns;
        return part * params.output_coefficients
            + ((column * params.blocks + block) * 3 + element) * 128 + coefficient;
    }

    uint64_t verify(bool full) const {
        const auto *actual = static_cast<const U128 *>(output.contents);
        uint64_t hash = 14695981039346656037ull;
        auto check = [&](uint64_t task, uint64_t part, uint64_t element, uint64_t coefficient) {
            const U128 value = actual[output_index(task, part, element, coefficient)];
            require(value == oracle(task, part, element, coefficient), "independent u128 oracle");
            for (unsigned byte = 0; byte < 16; ++byte)
                hash = (hash ^ uint8_t(value >> (byte * 8))) * 1099511628211ull;
        };
        if (full) {
            for (uint64_t task = 0; task < params.dispatch_tasks; ++task)
                for (uint64_t part = 0; part < 16; ++part)
                    for (uint64_t element = 0; element < 3; ++element)
                        for (uint64_t coefficient = 0; coefficient < 128; ++coefficient)
                            check(task, part, element, coefficient);
            for (uint64_t column = params.columns; column < 32; ++column)
                for (uint64_t block = 0; block < params.blocks; ++block)
                    for (uint64_t part = 0; part < 16; ++part)
                        for (uint64_t element = 0; element < 3; ++element)
                            for (uint64_t coefficient = 0; coefficient < 128; ++coefficient)
                                require(actual[part * params.output_coefficients
                                    + ((column * params.blocks + block) * 3 + element) * 128
                                    + coefficient] == 0, "padding untouched");
        } else {
            for (uint64_t sample = 0; sample < 65; ++sample) {
                const uint64_t task = sample < 2 ? sample * (params.dispatch_tasks - 1)
                    : mix(sample) % params.dispatch_tasks;
                check(task, sample < 2 ? sample * 15 : sample % 16,
                    sample < 2 ? sample * 2 : sample % 3,
                    sample < 2 ? sample * 127 : mix(sample + 7) % 128);
            }
        }
        return hash;
    }

    void run(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
             unsigned order, bool warmup, bool full, size_t shared_bytes = 0) {
        const auto started = Clock::now();
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:matrix offset:0 atIndex:0];
        [encoder setBuffer:lanes offset:0 atIndex:1];
        [encoder setBuffer:output offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        [encoder setBuffer:zero_rows offset:0 atIndex:4];
        if (shared_bytes) [encoder setThreadgroupMemoryLength:shared_bytes atIndex:0];
        const uint64_t streams = (params.dispatch_tasks + 63) / 64;
        [encoder dispatchThreadgroups:MTLSizeMake(streams * 48, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        [encoder endEncoding];
        const double epoch = [[NSDate date] timeIntervalSince1970];
        [command commit];
        while (command.status != MTLCommandBufferStatusCompleted
            && command.status != MTLCommandBufferStatusError) {
            if (std::chrono::duration<double>(Clock::now() - started).count() > 5) {
                std::fprintf(stderr, "DIAGNOSTIC_WATCHDOG five-second command limit\n");
                std::_Exit(124);
            }
            usleep(1000);
        }
        if (command.error) std::fprintf(stderr, "%s\n", command.error.localizedDescription.UTF8String);
        require(command.status == MTLCommandBufferStatusCompleted && !command.error, "command completion");
        const double wall = std::chrono::duration<double>(Clock::now() - started).count();
        const double gpu = command.GPUEndTime - command.GPUStartTime;
        require(std::isfinite(gpu) && gpu > 0, "GPU timestamp");
        uint64_t hot = 0;
        for (uint64_t task = 0; task < params.dispatch_tasks; ++task) hot += task_hot[task];
        const uint64_t hash = verify(full);
        std::printf("SATURATION positions=%llu order=%u warmup=%u streams=%llu groups=%llu tasks=%llu hot=%llu gpu_ms=%.6f wall_ms=%.6f giga_updates_s=%.6f oracle=%s checksum=%016llx epoch=%.6f\n",
            (unsigned long long)params.positions, order, unsigned(warmup),
            (unsigned long long)streams, (unsigned long long)(streams * 48),
            (unsigned long long)params.dispatch_tasks, (unsigned long long)hot,
            gpu * 1e3, wall * 1e3, double(hot) * 384 / gpu / 1e9,
            full ? "full" : "65_samples", (unsigned long long)hash, epoch);
        std::fflush(stdout);
    }
};

#ifndef COMMIT_DIAGNOSTIC_LIBRARY
int main(int argc, const char **argv) {
    require(argc == 3, "usage: saturation accepted-onehot.metal archive.bin");
    @autoreleasepool {
        NSError *error = nil;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "Metal device");
        NSString *source = [NSString stringWithContentsOfFile:@(argv[1])
            encoding:NSUTF8StringEncoding error:&error];
        require(source && !error, "accepted source read");
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (error) std::fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
        require(library && !error, "accepted source compilation");
        id<MTLFunction> function = [library newFunctionWithName:@"akita_packed_onehot_commit_fp128_d128_rank3"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        require(pipeline && !error, "accepted pipeline");
        id<MTLBinaryArchive> archive = [device newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&error];
        MTLComputePipelineDescriptor *descriptor = [MTLComputePipelineDescriptor new];
        descriptor.computeFunction = function;
        require([archive addComputePipelineFunctionsWithDescriptor:descriptor error:&error], "archive creation");
        require([archive serializeToURL:[NSURL fileURLWithPath:@(argv[2])] error:&error], "archive serialization");
        std::printf("SATURATION_PIPELINE device=%s simd=%lu max_threads=%lu shared_bytes=%lu seed=splitmix64-index synthetic=true\n",
            device.name.UTF8String, (unsigned long)pipeline.threadExecutionWidth,
            (unsigned long)pipeline.maxTotalThreadsPerThreadgroup,
            (unsigned long)pipeline.staticThreadgroupMemoryLength);
        require(pipeline.maxTotalThreadsPerThreadgroup == 1024
            && pipeline.threadExecutionWidth == 32 && pipeline.staticThreadgroupMemoryLength == 32768,
            "production resource fingerprint");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        { Case parity(device, 256, 2, 5, 10, 10); parity.run(queue, pipeline, 0, false, true); }
        Case target(device, 1 << 19, 1024, 29, 1024, 0);
        target.params.dispatch_tasks = 512;
        target.run(queue, pipeline, 0, true, false);
        unsigned order = 0;
        for (unsigned streams : {1u, 8u, 2u, 16u, 4u, 8u}) {
            target.params.dispatch_tasks = streams * 64;
            target.run(queue, pipeline, ++order, false, false);
        }
        std::puts("SATURATION_COMPLETE target_observations=6 parity=pass");
    }
}
#endif
