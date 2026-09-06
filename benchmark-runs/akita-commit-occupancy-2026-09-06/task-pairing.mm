#define COMMIT_DIAGNOSTIC_LIBRARY
#include "saturation.mm"

static void check_active_output(const Case &test, std::vector<U128> &reference) {
    const size_t count = test.params.dispatch_tasks * 16 * 3 * 128;
    const bool first = reference.empty();
    if (first) reference.resize(count);
    const auto *actual = static_cast<const U128 *>(test.output.contents);
    size_t cursor = 0;
    for (uint64_t part = 0; part < 16; ++part) {
        for (uint64_t task = test.params.task_offset;
             task < test.params.task_offset + test.params.dispatch_tasks; ++task) {
            const size_t index = test.output_index(task, part, 0, 0);
            const size_t bytes = 3 * 128 * sizeof(U128);
            if (first) std::memcpy(reference.data() + cursor, actual + index, bytes);
            else require(std::memcmp(reference.data() + cursor, actual + index, bytes) == 0,
                "full active-output equality");
            cursor += 3 * 128;
        }
    }
    require(cursor == count, "active-output coverage");
}

int main(int argc, const char **argv) {
    require(argc == 5 || argc == 6, "usage: pairing production.metal capture-directory map.u32le archive-prefix [interleaving.metal]");
    const bool interleaved = argc == 6;
    @autoreleasepool {
        NSError *error = nil;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "Metal device");
        NSString *source = [NSString stringWithContentsOfFile:@(argv[1])
            encoding:NSUTF8StringEncoding error:&error];
        require(source && !error, "accepted source");
        NSRange start = [source rangeOfString:@"kernel void akita_packed_onehot_commit_fp128_d128_rank3("];
        NSRange end = [source rangeOfString:@"// Packed decompose-fold for the D128 rank-3 row."];
        require(start.location != NSNotFound && end.location > start.location, "production shader boundary");
        NSString *body = [source substringWithRange:NSMakeRange(start.location, end.location - start.location)];
        NSString *variant_name = interleaved ? @"diagnostic_interleaved_commit" : @"diagnostic_paired_commit";
        body = [body stringByReplacingOccurrencesOfString:@"akita_packed_onehot_commit_fp128_d128_rank3"
            withString:variant_name];
        NSString *combined;
        if (interleaved) {
            NSString *helper = [NSString stringWithContentsOfFile:@(argv[5])
                encoding:NSUTF8StringEncoding error:&error];
            require(helper && !error, "interleaved helper source");
            NSString *anchor = @"        if (active_0) {\n            akita_fp128_d128_rank3_accumulate_task_tile(\n                accumulator_0, shared_matrix, lanes, active_zero_rows, params,\n                (ulong)block_0 * rows_per_block + tile_rows, column_0, simd_lane);\n        }\n        if (active_1) {\n            akita_fp128_d128_rank3_accumulate_task_tile(\n                accumulator_1, shared_matrix, lanes, active_zero_rows, params,\n                (ulong)block_1 * rows_per_block + tile_rows, column_1, simd_lane);\n        }";
            require([body containsString:anchor], "original task-loop boundary");
            body = [body stringByReplacingOccurrencesOfString:anchor withString:
                @"        diagnostic_interleaved_task_tile(\n            accumulator_0, accumulator_1, shared_matrix, lanes, active_zero_rows, params,\n            (ulong)block_0 * rows_per_block + tile_rows,\n            (ulong)block_1 * rows_per_block + tile_rows,\n            column_0, column_1, simd_lane, active_0, active_1);"];
            combined = [[source stringByAppendingString:helper] stringByAppendingString:body];
        } else {
        NSString *anchor = @"    uint global_1 = global_0 + 1u;";
        require([body containsString:anchor], "task identity anchor");
        body = [body stringByReplacingOccurrencesOfString:anchor withString:
            @"    uint global_1 = global_0 + 1u;\n    if (active_0) global_0 = task_order[global_0];\n    if (active_1) global_1 = task_order[global_1];"];
        anchor = @"    uint thread_index [[thread_index_in_threadgroup]],";
        require([body containsString:anchor], "mapping argument anchor");
        body = [body stringByReplacingOccurrencesOfString:anchor withString:
            @"    device const uint *task_order [[buffer(5)]],\n    uint thread_index [[thread_index_in_threadgroup]],"];
        combined = [source stringByAppendingString:body];
        }
        require([combined writeToFile:[NSString stringWithFormat:@"%s.generated.metal", argv[4]]
            atomically:NO encoding:NSUTF8StringEncoding error:&error], "shader snapshot");
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        id<MTLLibrary> library = [device newLibraryWithSource:combined options:options error:&error];
        if (error) std::fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
        require(library && !error, "pairing library compilation");
        const NSArray<NSString *> *names = @[@"akita_packed_onehot_commit_fp128_d128_rank3", variant_name];
        id<MTLComputePipelineState> pipelines[2];
        for (unsigned variant = 0; variant < 2; ++variant) {
            id<MTLFunction> function = [library newFunctionWithName:names[variant]];
            pipelines[variant] = [device newComputePipelineStateWithFunction:function error:&error];
            if (error) std::fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
            require(pipelines[variant] && !error, "pairing pipeline");
            require(pipelines[variant].maxTotalThreadsPerThreadgroup == 1024
                && pipelines[variant].threadExecutionWidth == 32
                && pipelines[variant].staticThreadgroupMemoryLength == 32768, "pairing resource fingerprint");
            id<MTLBinaryArchive> archive = [device newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&error];
            MTLComputePipelineDescriptor *descriptor = [MTLComputePipelineDescriptor new];
            descriptor.computeFunction = function;
            require([archive addComputePipelineFunctionsWithDescriptor:descriptor error:&error], "pairing archive");
            NSString *path = [NSString stringWithFormat:@"%s.variant%u.bin", argv[4], variant];
            require([archive serializeToURL:[NSURL fileURLWithPath:path] error:&error], "archive serialization");
            std::printf("PAIRING_PIPELINE variant=%u shared_bytes=32768 max_threads=1024 simd=32\n", variant);
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        const uint32_t small_map[] = {9,0,8,1,7,2,6,3,5,4};
        for (unsigned variant = 0; variant < 2; ++variant) {
            Case small(device, 256, 2, 5, 10, 10);
            small.make_private(device, queue);
            if (variant && !interleaved) small.task_mapping = [device newBufferWithBytes:small_map length:sizeof(small_map)
                options:MTLResourceStorageModeShared];
            std::printf("PAIRING_CASE phase=parity variant=%u\n", variant);
            small.run(queue, pipelines[variant], variant, false, true);
        }
        NSString *directory = @(argv[2]);
        NSData *metadata_bytes = [NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:@"metadata.json"]];
        NSDictionary *metadata = [NSJSONSerialization JSONObjectWithData:metadata_bytes options:0 error:&error];
        require(metadata && !error, "capture metadata");
        require([metadata[@"rows"] unsignedLongLongValue] == (1ull << 28)
            && [metadata[@"positions"] unsignedLongLongValue] == (1ull << 19)
            && [metadata[@"columns"] unsignedLongLongValue] == 29, "frozen captured shape");
        NSData *lanes = [NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:@"lanes.u8"]
            options:NSDataReadingMappedAlways error:&error];
        require(lanes && !error && lanes.length == (1ull << 28) * 29, "captured selectors");
        NSData *zeros = [NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:@"active_zero_rows.u64le"]
            options:NSDataReadingMappedAlways error:&error];
        require(zeros && !error && zeros.length == (1ull << 25), "captured selected-zero rows");
        NSData *mapping = [NSData dataWithContentsOfFile:@(argv[3])];
        NSData *hot = [NSData dataWithContentsOfFile:[NSString stringWithFormat:@"%s.hot.u64le", argv[3]]];
        const uint64_t tasks = [metadata[@"full_blocks"] unsignedLongLongValue] * 29;
        require(mapping.length == tasks * 4 && hot.length == tasks * 8, "mapping and exact work lengths");
        Case target(device, 1 << 19, 1024, 29, 512, 0);
        target.lanes = [device newBufferWithBytesNoCopy:const_cast<void *>(lanes.bytes) length:lanes.length
            options:MTLResourceStorageModeShared deallocator:nil];
        target.zero_rows = [device newBufferWithBytes:zeros.bytes length:zeros.length options:MTLResourceStorageModeShared];
        require(target.lanes && target.zero_rows, "captured source buffers");
        target.params.zero_mask = [metadata[@"zero_mask"] unsignedLongLongValue];
        target.params.tasks = tasks;
        target.params.full_blocks = [metadata[@"full_blocks"] unsignedLongLongValue];
        target.params.task_offset = (((tasks + 63) / 64 - 8) / 2) * 64;
        target.params.dispatch_tasks = 512;
        require(target.params.task_offset == 10880, "preregistered interior command");
        target.task_hot.resize(tasks);
        std::memcpy(target.task_hot.data(), hot.bytes, hot.length);
        target.make_private(device, queue);
        id<MTLBuffer> mapping_buffer = [device newBufferWithBytes:mapping.bytes length:mapping.length
            options:MTLResourceStorageModeShared];
        require(mapping_buffer != nil, "mapping buffer");
        std::vector<U128> reference;
        for (unsigned variant = 0; variant < 2; ++variant) {
            target.task_mapping = variant && !interleaved ? mapping_buffer : nil;
            std::printf("PAIRING_CASE phase=warmup variant=%u task_offset=10880\n", variant);
            target.run(queue, pipelines[variant], variant, true, false);
            check_active_output(target, reference);
        }
        unsigned order = 0;
        for (unsigned variant : {0u, 1u, 1u, 0u}) {
            target.task_mapping = variant && !interleaved ? mapping_buffer : nil;
            std::printf("PAIRING_CASE phase=measure variant=%u task_offset=10880\n", variant);
            target.run(queue, pipelines[variant], ++order, false, false);
            check_active_output(target, reference);
        }
        std::printf("%s_COMPLETE target_observations=4 parity=pass active_coefficients=%llu\n",
            interleaved ? "INTERLEAVING" : "PAIRING", (unsigned long long)reference.size());
    }
}
