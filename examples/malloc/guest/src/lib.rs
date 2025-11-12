#![cfg_attr(feature = "guest", no_std)]

#[jolt::provable(memory_size = 65536)]
fn alloc(val: i32) -> u32 {
    // The C functions are linked by the build script
    extern "C" {
        fn alloc_and_set(val: i32) -> *mut i32;
        fn free_me(ptr: *mut i32);
    }

    let ptr = unsafe { alloc_and_set(val) };

    // The value should have been set correctly
    assert_eq!(unsafe { *ptr }, val);

    unsafe { free_me(ptr) };

    0
}
