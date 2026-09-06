#define COMMIT_DIAGNOSTIC_LIBRARY
#include "saturation.mm"

struct CountParams {
    uint64_t columns, rows_per_fragment, tasks;
};

struct CountCase {
    CountParams params;
    id<MTLBuffer> lanes, counts;
    std::vector<uint16_t> reference;

    CountCase(id<MTLDevice> device, id<MTLBuffer> source, uint64_t columns,
              uint64_t rows_per_fragment, uint64_t tasks)
        : params{columns, rows_per_fragment, tasks}, lanes(source) {
        require(tasks % columns == 0 && rows_per_fragment <= 16384, "count shape bounds");
        counts = [device newBufferWithLength:tasks * 128 * 2 options:MTLResourceStorageModeShared];
        require(counts != nil, "negative count storage");
    }

    uint16_t oracle(uint64_t task, uint64_t coefficient) const {
        const auto *source = static_cast<const uint8_t *>(lanes.contents);
        const uint64_t first = (task / params.columns) * params.rows_per_fragment;
        uint64_t count = 0;
        for (uint64_t row = 0; row < params.rows_per_fragment; ++row)
            count += (source[(first + row) * params.columns + task % params.columns] & 127) > coefficient;
        require(count <= 16384, "independent negative-count bound");
        return uint16_t(count);
    }

    void validate(bool full) {
        const auto *actual = static_cast<const uint16_t *>(counts.contents);
        const auto *source = static_cast<const uint8_t *>(lanes.contents);
        std::vector<uint64_t> weights(params.tasks);
        const uint64_t rows = params.tasks / params.columns * params.rows_per_fragment;
        for (uint64_t row = 0; row < rows; ++row)
            for (uint64_t column = 0; column < params.columns; ++column)
                weights[(row / params.rows_per_fragment) * params.columns + column]
                    += source[row * params.columns + column] & 127;
        for (uint64_t task = 0; task < params.tasks; ++task) {
            uint64_t weight = 0, previous = params.rows_per_fragment;
            for (uint64_t coefficient = 0; coefficient < 128; ++coefficient) {
                const uint64_t count = actual[task * 128 + coefficient];
                require(count <= previous, "count range and suffix monotonicity");
                if (full) require(count == oracle(task, coefficient), "full direct-comparison count oracle");
                previous = count;
                weight += count;
            }
            require(previous == 0 && weight == weights[task], "last-zero and full fragment moment identity");
        }
        if (!full) {
            for (uint64_t sample = 0; sample < 129; ++sample) {
                const uint64_t task = sample < 2 ? sample * (params.tasks - 1) : mix(sample) % params.tasks;
                const uint64_t coefficient = sample < 2 ? sample * 127 : mix(sample + 13) % 128;
                require(actual[task * 128 + coefficient] == oracle(task, coefficient), "129 direct count samples");
            }
        }
        reference.assign(actual, actual + params.tasks * 128);
        std::printf("NEGATIVE_COUNTS_ORACLE tasks=%llu full=%u fragment_moments=pass range=pass samples=%u\n",
            (unsigned long long)params.tasks, unsigned(full), full ? unsigned(params.tasks * 128) : 129);
    }

    void run(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline, unsigned order, bool warmup) {
        const auto start = Clock::now();
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:lanes offset:0 atIndex:0];
        [encoder setBuffer:counts offset:0 atIndex:1];
        [encoder setBytes:&params length:sizeof(params) atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(params.tasks, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];
        [command commit];
        finish_command(command, start);
        const double gpu = command.GPUEndTime - command.GPUStartTime;
        const double wall = std::chrono::duration<double>(Clock::now() - start).count();
        require(std::isfinite(gpu) && gpu > 0, "negative-count GPU timestamp");
        if (!reference.empty()) require(std::memcmp(reference.data(), counts.contents, counts.length) == 0,
            "repeat full count equality");
        std::printf("NEGATIVE_COUNTS tasks=%llu order=%u warmup=%u gpu_ms=%.6f wall_ms=%.6f bytes=%lu\n",
            (unsigned long long)params.tasks, order, unsigned(warmup), gpu * 1e3,
            wall * 1e3,
            (unsigned long)counts.length);
        std::fflush(stdout);
    }
};

int main(int argc, const char **argv) {
    require(argc == 4, "usage: counts shader.metal capture-directory archive.bin");
    @autoreleasepool {
        NSError *error = nil;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "Metal device");
        NSString *source = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
        require(source && !error, "negative-count shader source");
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (error) std::fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
        require(library && !error, "negative-count library");
        id<MTLFunction> function = [library newFunctionWithName:@"diagnostic_negative_counts"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        require(pipeline && !error && pipeline.maxTotalThreadsPerThreadgroup >= 128
            && pipeline.threadExecutionWidth == 32, "negative-count pipeline");
        id<MTLBinaryArchive> archive = [device newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&error];
        MTLComputePipelineDescriptor *descriptor = [MTLComputePipelineDescriptor new];
        descriptor.computeFunction = function;
        require([archive addComputePipelineFunctionsWithDescriptor:descriptor error:&error], "negative-count archive");
        require([archive serializeToURL:[NSURL fileURLWithPath:@(argv[3])] error:&error], "archive serialization");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        const uint8_t fixture[] = {0,1,128,255,127,32,129,0,255,128,0,2,63,64,64,63};
        CountCase small(device, [device newBufferWithBytes:fixture length:sizeof(fixture)
            options:MTLResourceStorageModeShared], 2, 4, 4);
        small.run(queue, pipeline, 0, false);
        small.validate(true);
        NSString *directory = @(argv[2]);
        NSData *metadata_bytes = [NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:@"metadata.json"]];
        NSDictionary *metadata = [NSJSONSerialization JSONObjectWithData:metadata_bytes options:0 error:&error];
        require(metadata && !error && [metadata[@"rows"] unsignedLongLongValue] == (1ull << 28)
            && [metadata[@"positions"] unsignedLongLongValue] == (1ull << 19)
            && [metadata[@"columns"] unsignedLongLongValue] == 29
            && [metadata[@"full_blocks"] unsignedLongLongValue] == 769, "frozen count shape");
        NSData *lanes = [NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:@"lanes.u8"]
            options:NSDataReadingMappedAlways error:&error];
        require(lanes && !error && lanes.length == (1ull << 28) * 29, "captured selectors");
        id<MTLBuffer> source_buffer = [device newBufferWithBytesNoCopy:const_cast<void *>(lanes.bytes)
            length:lanes.length options:MTLResourceStorageModeShared deallocator:nil];
        require(source_buffer != nil, "captured selector buffer");
        CountCase target(device, source_buffer, 29, 16384, 769 * 16 * 29);
        target.run(queue, pipeline, 0, true);
        const auto *warm_counts = static_cast<const uint16_t *>(target.counts.contents);
        target.reference.assign(warm_counts, warm_counts + target.params.tasks * 128);
        target.run(queue, pipeline, 1, false);
        target.run(queue, pipeline, 2, false);
        target.validate(false);
        NSString *output = [NSString stringWithFormat:@"%s.counts.u16le", argv[3]];
        require(![[NSFileManager defaultManager] fileExistsAtPath:output], "fresh counts artifact");
        NSData *result = [NSData dataWithBytesNoCopy:target.counts.contents length:target.counts.length freeWhenDone:NO];
        require([result writeToFile:output options:0 error:&error] && !error, "counts artifact write");
        std::puts("NEGATIVE_COUNTS_COMPLETE target_observations=2 parity=pass");
    }
}
