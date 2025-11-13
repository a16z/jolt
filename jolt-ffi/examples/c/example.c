#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../../target/debug/jolt-ffi.h"

// Helper function to read a file into a buffer
uint8_t* read_file(const char* filename, size_t* size) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t* buffer = malloc(*size);
    if (!buffer) {
        fclose(f);
        return NULL;
    }

    if (fread(buffer, 1, *size, f) != *size) {
        fprintf(stderr, "Failed to read file: %s\n", filename);
        free(buffer);
        fclose(f);
        return NULL;
    }

    fclose(f);
    return buffer;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <elf_file> <proof_output_file> [preprocessing_file]\n", argv[0]);
        return 1;
    }

    const char* elf_file = argv[1];
    const char* proof_output = argv[2];
    const char* preprocessing_file = (argc > 3) ? argv[3] : NULL;

    printf("Jolt FFI Example\n");
    printf("================\n\n");

    // Step 1: Load or create preprocessing
    printf("Step 1: Loading preprocessing...\n");
    struct JoltProverPreprocessingHandle* preprocessing = NULL;

    if (preprocessing_file) {
        preprocessing = jolt_prover_preprocessing_load(preprocessing_file);
        if (!preprocessing) {
            const char* error = jolt_last_error();
            fprintf(stderr, "Failed to load preprocessing: %s\n", error ? error : "unknown error");
            return 1;
        }
        printf("  ✓ Loaded preprocessing from %s\n\n", preprocessing_file);
    } else {
        fprintf(stderr, "Error: This example requires preprocessing file.\n");
        fprintf(stderr, "Please provide a preprocessing file as the third argument.\n");
        fprintf(stderr, "You can generate preprocessing using the jolt CLI.\n");
        return 1;
    }

    // Step 2: Read ELF file
    printf("Step 2: Reading ELF file...\n");
    size_t elf_size;
    uint8_t* elf_contents = read_file(elf_file, &elf_size);
    if (!elf_contents) {
        jolt_prover_preprocessing_free(preprocessing);
        return 1;
    }
    printf("  ✓ Read %zu bytes from %s\n\n", elf_size, elf_file);

    // Step 3: Create prover from ELF
    printf("Step 3: Creating prover from ELF...\n");

    // Example inputs (empty for this demo - modify as needed)
    uint8_t inputs[] = {};
    size_t inputs_len = 0;

    struct JoltCpuProverHandle* prover = jolt_cpu_prover_gen_from_elf(
        preprocessing,
        elf_contents, elf_size,
        inputs, inputs_len,
        NULL, 0,  // untrusted_advice
        NULL, 0   // trusted_advice
    );

    free(elf_contents);

    if (!prover) {
        const char* error = jolt_last_error();
        fprintf(stderr, "Failed to create prover: %s\n", error ? error : "unknown error");
        jolt_prover_preprocessing_free(preprocessing);
        return 1;
    }
    printf("  ✓ Prover created successfully\n\n");

    // Step 4: Generate proof
    printf("Step 4: Generating proof (this may take a while)...\n");
    int result = jolt_cpu_prover_prove(prover, proof_output);

    // Note: jolt_cpu_prover_prove() consumes the prover, so don't free it

    if (result != 0) {
        const char* error = jolt_last_error();
        fprintf(stderr, "Failed to generate proof: %s\n", error ? error : "unknown error");
        jolt_prover_preprocessing_free(preprocessing);
        return 1;
    }
    printf("  ✓ Proof generated and saved to %s\n\n", proof_output);

    // Step 5: Cleanup
    jolt_prover_preprocessing_free(preprocessing);

    printf("Done! Proof saved to: %s\n", proof_output);
    return 0;
}
