#define COMMIT_DIAGNOSTIC_LIBRARY
#include "saturation.mm"

struct ReuseParams { uint64_t columns, rows_per_block, blocks, zero_mask; };
struct ReuseEntry { uint32_t hot, representative; };
static_assert(sizeof(ReuseParams) == 32 && sizeof(ReuseEntry) == 8);

struct ReuseCase {
    ReuseParams params;
    id<MTLBuffer> lanes, zeros, metadata;
    std::vector<ReuseEntry> reference;

    ReuseCase(id<MTLDevice> device, id<MTLBuffer> source, id<MTLBuffer> zero_rows, ReuseParams shape)
        : params(shape), lanes(source), zeros(zero_rows) {
        require(params.columns > 0 && params.columns <= 32 && params.blocks > 0
            && params.rows_per_block > 0 && params.rows_per_block <= 262144,
            "reuse shape bounds");
        require(lanes.length >= params.columns * params.rows_per_block * params.blocks
            && zeros.length >= ((params.rows_per_block * params.blocks + 63) / 64) * 8,
            "reuse source lengths");
        metadata = [device newBufferWithLength:params.columns * params.blocks * sizeof(ReuseEntry)
            options:MTLResourceStorageModeShared];
        require(metadata != nil, "reuse metadata allocation");
    }

    uint16_t symbol(uint64_t row, uint64_t column) const {
        const auto *source = static_cast<const uint8_t *>(lanes.contents);
        const auto *bits = static_cast<const uint64_t *>(zeros.contents);
        const uint8_t value = source[row * params.columns + column];
        if (value) return value;
        return ((params.zero_mask >> column) & 1) && ((bits[row / 64] >> (row % 64)) & 1) ? 256 : 0;
    }

    void validate(bool target) {
        const auto start = Clock::now();
        const uint64_t tasks = params.columns * params.blocks;
        const uint32_t first = params.blocks > 1 ? 1 : 0;
        const uint32_t references = uint32_t(std::min(uint64_t(8), params.blocks > 1 ? params.blocks - 1 : 1));
        std::vector<ReuseEntry> expected(tasks);
        std::vector<uint16_t> matches(tasks, uint16_t((1u << references) - 1u));
        for (uint64_t row = 0; row < params.rows_per_block * params.blocks; ++row) {
            const uint64_t block = row / params.rows_per_block;
            for (uint64_t column = 0; column < params.columns; ++column) {
                const uint64_t task = block * params.columns + column;
                const uint16_t value = symbol(row, column);
                expected[task].hot += value != 0;
                for (uint32_t reference_index = 0; reference_index < references && matches[task]; ++reference_index) {
                    const uint16_t bit = uint16_t(1u << reference_index);
                    if ((matches[task] & bit) && block != first + reference_index
                        && value != symbol((first + reference_index) * params.rows_per_block
                            + row % params.rows_per_block, column))
                        matches[task] &= uint16_t(~bit);
                }
            }
        }
        const auto *actual = static_cast<const ReuseEntry *>(metadata.contents);
        uint64_t total_hot = 0, executed_hot = 0, unique = 0, zero_tasks = 0;
        for (uint64_t task = 0; task < tasks; ++task) {
            auto &entry = expected[task];
            entry.representative = uint32_t(task / params.columns);
            if (entry.hot && matches[task]) entry.representative = first + __builtin_ctz(unsigned(matches[task]));
            require(actual[task].hot == entry.hot && actual[task].representative == entry.representative,
                "every metadata pair equals independent serial oracle");
            require(actual[task].representative < params.blocks, "representative range");
            total_hot += entry.hot;
            if (!entry.hot) ++zero_tasks;
            else if (entry.representative == task / params.columns) {
                ++unique;
                executed_hot += entry.hot;
            }
        }
        for (uint64_t task = 0; task < tasks; ++task) {
            const auto &entry = actual[task];
            const auto &representative = actual[entry.representative * params.columns + task % params.columns];
            require(representative.representative == entry.representative && representative.hot == entry.hot,
                "representative idempotence and conserved hot count");
        }
        if (target) require(total_hot == 3263846381ull && executed_hot == 2863290349ull
            && unique == 19239 && zero_tasks == 1534, "frozen D14 independent census totals");
        reference = expected;
        std::printf("REUSE_ORACLE tasks=%llu hot=%llu executed_hot=%llu unique=%llu zeros=%llu full=pass cpu_ms=%.6f\n",
            (unsigned long long)tasks, (unsigned long long)total_hot, (unsigned long long)executed_hot,
            (unsigned long long)unique, (unsigned long long)zero_tasks,
            std::chrono::duration<double>(Clock::now() - start).count() * 1e3);
        std::fflush(stdout);
    }

    void run(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline, unsigned order, bool warmup) {
        const auto start = Clock::now();
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:lanes offset:0 atIndex:0];
        [encoder setBuffer:zeros offset:0 atIndex:1];
        [encoder setBuffer:metadata offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(params.columns * params.blocks, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];
        [command commit];
        finish_command(command, start);
        const double gpu = command.GPUEndTime - command.GPUStartTime;
        const double wall = std::chrono::duration<double>(Clock::now() - start).count();
        require(std::isfinite(gpu) && gpu > 0, "reuse GPU timestamp");
        if (!reference.empty()) require(std::memcmp(reference.data(), metadata.contents, metadata.length) == 0,
            "repeat complete metadata equality");
        std::printf("REUSE_METADATA tasks=%llu order=%u warmup=%u gpu_ms=%.6f wall_ms=%.6f bytes=%lu\n",
            (unsigned long long)(params.columns * params.blocks), order, unsigned(warmup), gpu * 1e3,
            wall * 1e3, (unsigned long)metadata.length);
        std::fflush(stdout);
    }
};

int main(int argc, const char **argv) {
    require(argc == 4, "usage: reuse-metadata shader.metal capture-directory archive.bin");
    @autoreleasepool {
        NSError *error = nil;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "Metal device");
        NSString *source = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
        require(source && !error, "reuse shader source");
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (error) std::fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
        require(library && !error, "reuse library");
        id<MTLFunction> function = [library newFunctionWithName:@"diagnostic_reuse_metadata"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        require(pipeline && !error && pipeline.maxTotalThreadsPerThreadgroup >= 128
            && pipeline.threadExecutionWidth == 32, "reuse pipeline");
        id<MTLBinaryArchive> archive = [device newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&error];
        MTLComputePipelineDescriptor *descriptor = [MTLComputePipelineDescriptor new];
        descriptor.computeFunction = function;
        require([archive addComputePipelineFunctionsWithDescriptor:descriptor error:&error], "reuse archive");
        require([archive serializeToURL:[NSURL fileURLWithPath:@(argv[3])] error:&error], "archive serialization");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        for (uint64_t blocks : {10ull, 3ull, 1ull}) {
            const uint64_t block_rows = blocks == 10 ? 257 : 5;
            std::vector<uint8_t> fixture(blocks * block_rows * 3);
            std::vector<uint64_t> zero_bits((blocks * block_rows + 63) / 64);
            const uint8_t classes[] = {9,1,2,1,2,3,4,5,6,1};
            for (uint64_t row = 0; row < blocks * block_rows; ++row) {
                const uint64_t block = row / block_rows;
                fixture[row * 3] = classes[block];
                if ((block == 1 || block == 3) && row % block_rows % 3 == 0)
                    zero_bits[row / 64] |= 1ull << (row % 64);
            }
            ReuseCase small(device, [device newBufferWithBytes:fixture.data() length:fixture.size()
                options:MTLResourceStorageModeShared], [device newBufferWithBytes:zero_bits.data()
                length:zero_bits.size() * 8 options:MTLResourceStorageModeShared], {3, block_rows, blocks, 2});
            small.run(queue, pipeline, 0, false);
            small.validate(false);
            if (blocks == 10) {
                const auto *actual = static_cast<const ReuseEntry *>(small.metadata.contents);
                require(actual[27].representative == 1 && actual[27].hot == 257
                    && actual[0].representative == 0 && actual[10].representative == 1
                    && actual[1].hot == 0 && actual[1].representative == 0,
                    "hand-counted multiple classes, no-match and selected-zero fixture");
            }
        }
        const auto setup_start = Clock::now();
        NSString *directory = @(argv[2]);
        NSData *metadata_bytes = [NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:@"metadata.json"]];
        NSDictionary *shape = [NSJSONSerialization JSONObjectWithData:metadata_bytes options:0 error:&error];
        require(shape && !error && [shape[@"rows"] unsignedLongLongValue] == (1ull << 28)
            && [shape[@"positions"] unsignedLongLongValue] == (1ull << 19)
            && [shape[@"columns"] unsignedLongLongValue] == 29
            && [shape[@"full_blocks"] unsignedLongLongValue] == 769
            && [shape[@"zero_mask"] unsignedLongLongValue] == 402653184, "frozen reuse shape");
        NSData *lanes = [NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:@"lanes.u8"]
            options:NSDataReadingMappedAlways error:&error];
        require(lanes && !error && lanes.length == (1ull << 28) * 29, "captured selectors");
        NSData *zeros = [NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:@"zeros.u64le"]
            options:NSDataReadingMappedAlways error:&error];
        require(zeros && !error && zeros.length == (1ull << 28) / 8, "captured selected-zero bits");
        id<MTLBuffer> source_buffer = [device newBufferWithBytesNoCopy:const_cast<void *>(lanes.bytes)
            length:lanes.length options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> zero_buffer = [device newBufferWithBytesNoCopy:const_cast<void *>(zeros.bytes)
            length:zeros.length options:MTLResourceStorageModeShared deallocator:nil];
        require(source_buffer && zero_buffer, "zero-copy capture buffers, no fallback copy");
        ReuseCase target(device, source_buffer, zero_buffer, {29, 262144, 769, 402653184});
        std::printf("REUSE_SETUP wall_ms=%.6f zero_copy=true\n",
            std::chrono::duration<double>(Clock::now() - setup_start).count() * 1e3);
        target.run(queue, pipeline, 0, true);
        const auto *warm = static_cast<const ReuseEntry *>(target.metadata.contents);
        target.reference.assign(warm, warm + target.params.columns * target.params.blocks);
        target.run(queue, pipeline, 1, false);
        target.run(queue, pipeline, 2, false);
        target.validate(true);
        NSString *output = [NSString stringWithFormat:@"%s.metadata.u32le", argv[3]];
        require(![[NSFileManager defaultManager] fileExistsAtPath:output], "fresh reuse metadata artifact");
        NSData *result = [NSData dataWithBytesNoCopy:target.metadata.contents length:target.metadata.length freeWhenDone:NO];
        require([result writeToFile:output options:0 error:&error] && !error, "metadata artifact write");
        std::puts("REUSE_METADATA_COMPLETE target_observations=2 parity=pass");
    }
}
