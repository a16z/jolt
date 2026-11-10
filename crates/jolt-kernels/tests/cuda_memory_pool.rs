#![cfg(feature = "cuda")]
#![expect(
    clippy::expect_used,
    reason = "CUDA contract test failures must fail the test"
)]

use cudarc::driver::result::{device, mem_pool};
use cudarc::driver::sys::{
    CUmemAllocationHandleType, CUmemAllocationType, CUmemLocation, CUmemLocationType,
    CUmemPoolProps, CUmemPool_attribute as MemoryPoolAttribute,
};
use cudarc::driver::CudaContext;
use jolt_kernels::cuda::CudaKernelContext;
use std::ptr;

#[test]
fn context_preserves_caller_memory_pool_policy() {
    let Ok(driver) = CudaContext::new(0) else {
        return;
    };
    if !driver.has_async_alloc() {
        return;
    }
    // SAFETY: driver keeps the device and its default pool alive. Attribute
    // pointers below are u64s, the CUDA type for RELEASE_THRESHOLD. The custom
    // pool properties have valid zero defaults for the remaining C fields; that
    // pool is deselected and destroyed only after its kernel context is dropped.
    unsafe {
        let default = device::get_default_mem_pool(driver.cu_device()).expect("default pool");
        let attr = MemoryPoolAttribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD;
        let mut original = 0u64;
        mem_pool::get_attribute(default, attr, ptr::from_mut(&mut original).cast())
            .expect("original threshold");
        let mut requested = 64u64 << 20;
        mem_pool::set_attribute(default, attr, ptr::from_mut(&mut requested).cast())
            .expect("caller threshold");
        let context = CudaKernelContext::new(0).expect("CUDA kernels");
        let mut observed = 0u64;
        mem_pool::get_attribute(default, attr, ptr::from_mut(&mut observed).cast())
            .expect("retained threshold");
        assert_eq!(observed, requested);
        drop(context);

        let mut zero = 0u64;
        mem_pool::set_attribute(default, attr, ptr::from_mut(&mut zero).cast())
            .expect("reset default threshold");
        let props = CUmemPoolProps {
            allocType: CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
            handleTypes: CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE,
            location: CUmemLocation {
                type_: CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                id: driver.cu_device(),
            },
            ..std::mem::zeroed()
        };
        let custom = mem_pool::create(ptr::from_ref(&props)).expect("custom pool");
        device::set_mem_pool(driver.cu_device(), custom).expect("select custom pool");
        let context = CudaKernelContext::new(0).expect("CUDA kernels with custom pool");
        assert_eq!(
            device::get_mem_pool(driver.cu_device()).expect("selected pool"),
            custom
        );
        for pool in [default, custom] {
            mem_pool::get_attribute(pool, attr, ptr::from_mut(&mut observed).cast())
                .expect("unchanged threshold");
            assert_eq!(observed, 0);
        }
        drop(context);
        device::set_mem_pool(driver.cu_device(), default).expect("restore default pool");
        mem_pool::destroy(custom).expect("destroy unused custom pool");
        mem_pool::set_attribute(default, attr, ptr::from_mut(&mut original).cast())
            .expect("restore original threshold");
    }
}
