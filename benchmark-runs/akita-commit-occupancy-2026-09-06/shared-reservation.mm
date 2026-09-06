#define COMMIT_DIAGNOSTIC_LIBRARY
#include "saturation.mm"

static NSString *section(NSString *source, NSString *begin, NSString *end) {
    NSRange start = [source rangeOfString:begin];
    require(start.location != NSNotFound, "source section start");
    NSRange finish = [source rangeOfString:end options:0
        range:NSMakeRange(start.location, source.length - start.location)];
    require(finish.location != NSNotFound, "source section end");
    return [source substringWithRange:NSMakeRange(start.location, finish.location - start.location)];
}

static NSString *replace(NSString *source, NSString *old, NSString *value) {
    require([source containsString:old], "diagnostic substitution anchor");
    return [source stringByReplacingOccurrencesOfString:old withString:value];
}

int main(int argc, const char **argv) {
    require(argc == 4, "usage: shared-reservation production.metal helpers.metal archive-prefix");
    @autoreleasepool {
        NSError *error = nil;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "Metal device");
        NSString *source = [NSString stringWithContentsOfFile:@(argv[1])
            encoding:NSUTF8StringEncoding error:&error];
        require(source && !error, "accepted source");
        NSString *helpers = [NSString stringWithContentsOfFile:@(argv[2])
            encoding:NSUTF8StringEncoding error:&error];
        require(helpers && !error, "diagnostic helpers");
        NSString *task = section(source,
            @"inline void akita_fp128_d128_rank3_accumulate_task_tile(",
            @"inline void akita_store_fp128_d128_rank3(");
        NSString *kernel = section(source,
            @"kernel void akita_packed_onehot_commit_fp128_d128_rank3(",
            @"// Packed decompose-fold for the D128 rank-3 row.");
        kernel = replace(kernel,
            @"    threadgroup uint shared_matrix[PACKED_FP128_D512_PANEL_TILE_ELEMENTS * 4];",
            @"");
        kernel = replace(kernel, @"    uint thread_index [[thread_index_in_threadgroup]],",
            @"    threadgroup uint *shared_matrix [[threadgroup(0)]],\n    uint thread_index [[thread_index_in_threadgroup]],");
        NSString *body = [task stringByAppendingString:kernel];
        body = replace(body, @"akita_fp128_d128_rank3_accumulate_task_tile", @"diagnostic_accumulate_task_tile");
        body = replace(body, @"akita_packed_onehot_commit_fp128_d128_rank3", @"diagnostic_shared_commit");
        body = replace(body, @"akita_fp128_d512_accumulate_mixed", @"diagnostic_accumulate_mixed");
        body = replace(body, @"PACKED_FP128_D128_RANK3_ROWS_PER_TILE", @"(diagnostic_tile_positions / 2u)");
        body = replace(body, @"PACKED_FP128_D128_RANK3_TILE_POSITIONS", @"diagnostic_tile_positions");
        body = replace(body, @"PACKED_FP128_D512_PANEL_TILE_ELEMENTS", @"(diagnostic_tile_positions * 128u)");
        NSString *combined = [[source stringByAppendingString:helpers] stringByAppendingString:body];
        NSString *generated_path = [NSString stringWithFormat:@"%s.generated.metal", argv[3]];
        require([combined writeToFile:generated_path atomically:NO encoding:NSUTF8StringEncoding error:&error],
            "generated shader artifact");
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        id<MTLLibrary> library = [device newLibraryWithSource:combined options:options error:&error];
        if (error) std::fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
        require(library && !error, "diagnostic library compilation");
        id<MTLComputePipelineState> pipelines[3];
        for (unsigned variant = 0; variant < 3; ++variant) {
            id<MTLFunction> function;
            const uint32_t tile = variant == 2 ? 8 : 16;
            if (variant == 0) {
                function = [library newFunctionWithName:@"akita_packed_onehot_commit_fp128_d128_rank3"];
            } else {
                MTLFunctionConstantValues *values = [MTLFunctionConstantValues new];
                [values setConstantValue:&tile type:MTLDataTypeUInt atIndex:20];
                function = [library newFunctionWithName:@"diagnostic_shared_commit"
                    constantValues:values error:&error];
                require(function && !error, "tile specialization");
            }
            pipelines[variant] = [device newComputePipelineStateWithFunction:function error:&error];
            require(pipelines[variant] && !error, "diagnostic pipeline");
            id<MTLBinaryArchive> archive = [device newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&error];
            MTLComputePipelineDescriptor *descriptor = [MTLComputePipelineDescriptor new];
            descriptor.computeFunction = function;
            require([archive addComputePipelineFunctionsWithDescriptor:descriptor error:&error], "archive creation");
            NSString *path = [NSString stringWithFormat:@"%s.variant%u.bin", argv[3], variant];
            require([archive serializeToURL:[NSURL fileURLWithPath:path] error:&error], "archive serialization");
            std::printf("RESERVATION_PIPELINE variant=%u tile_positions=%u max_threads=%lu simd=%lu static_bytes=%lu\n",
                variant, tile, (unsigned long)pipelines[variant].maxTotalThreadsPerThreadgroup,
                (unsigned long)pipelines[variant].threadExecutionWidth,
                (unsigned long)pipelines[variant].staticThreadgroupMemoryLength);
            require(pipelines[variant].maxTotalThreadsPerThreadgroup == 1024
                && pipelines[variant].threadExecutionWidth == 32
                && pipelines[variant].staticThreadgroupMemoryLength == (variant == 0 ? 32768 : 0),
                "diagnostic resource fingerprint");
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        for (unsigned variant = 0; variant < 3; ++variant) {
            Case parity(device, 256, 2, 5, 10, 10);
            std::printf("RESERVATION_CASE phase=parity variant=%u reserved_bytes=%u\n",
                variant, variant == 0 ? 0 : variant == 1 ? 32768 : 16384);
            parity.run(queue, pipelines[variant], variant, false, true,
                variant == 0 ? 0 : variant == 1 ? 32768 : 16384);
        }
        Case target(device, 1 << 19, 1024, 29, 512, 0);
        std::puts("RESERVATION_CASE phase=warmup variant=0 reserved_bytes=0");
        target.run(queue, pipelines[0], 0, true, false);
        std::puts("RESERVATION_CASE phase=warmup variant=2 reserved_bytes=16384");
        target.run(queue, pipelines[2], 0, true, false, 16384);
        const unsigned variants[] = {0, 1, 2, 2, 2, 0};
        const unsigned reservation[] = {0, 32768, 16384, 32768, 24576, 0};
        for (unsigned order = 0; order < 6; ++order) {
            std::printf("RESERVATION_CASE phase=measure variant=%u reserved_bytes=%u\n",
                variants[order], reservation[order]);
            target.run(queue, pipelines[variants[order]], order, false, false, reservation[order]);
        }
        std::puts("RESERVATION_COMPLETE target_observations=6 parity=pass");
    }
}
