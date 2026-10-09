#![cfg(feature = "prover-fixtures")]

use jolt_host::Program;

// Repeated and partial unmaps must not corrupt ZeroOS's kernel heap.
#[test]
fn large_alloc_munmap_trace() {
    let mut program = Program::new("large-alloc-guest");
    program.set_std(true);
    program.set_func("large_alloc_roundtrip");
    program.set_heap_size(16_777_216);
    program.set_stack_size(1_048_576);
    let (_, _, _, io_device) = program.trace(&[], &[], &[]);
    assert!(
        !io_device.panic,
        "large-alloc guest panicked: a non-exact munmap corrupted the ZeroOS kernel heap"
    );
}

// Under guest-size-class-alloc, musl's mmap-backed memory (C malloc, pthread
// stacks) and Rust allocations draw from one arena.
#[test]
fn size_class_heap_serves_musl_mappings() {
    let mut program = Program::new("large-alloc-guest");
    program.set_std(true);
    program.set_func("shared_heap_roundtrip");
    program.add_guest_feature("size-class-alloc");
    program.set_heap_size(16_777_216);
    program.set_stack_size(1_048_576);
    let (_, io_device) = program.execute_with_output(&[], &[], &[]);
    assert!(
        !io_device.panic,
        "shared-heap guest panicked: musl could not map memory beside the Rust heap"
    );
}
