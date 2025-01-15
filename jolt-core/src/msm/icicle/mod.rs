#[cfg(not(feature = "icicle"))]
use ark_bn254::G1Projective;
use ark_ec::{CurveGroup, ScalarMul};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;

#[cfg(feature = "icicle")]
pub(crate) mod adapter;
#[cfg(feature = "icicle")]
pub use adapter::*;

static ICICLE_INIT: Once = Once::new();
static ICICLE_READY: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "icicle")]
pub trait CurveGroupConfig: CurveGroup + Icicle {}
#[cfg(not(feature = "icicle"))]
pub trait CurveGroupConfig: CurveGroup {}

#[cfg(feature = "icicle")]
pub trait ScalarMulConfig: ScalarMul + Icicle {}
#[cfg(not(feature = "icicle"))]
pub trait ScalarMulConfig: ScalarMul {}
#[cfg(not(feature = "icicle"))]
pub trait Icicle {}
#[cfg(not(feature = "icicle"))]
impl Icicle for G1Projective {}

/// Initializes the icicle backend and sets the CUDA device as active and returns true if successful.
///
/// Safe to call multiple times on the main thread; will only initialize the backend once.
///
/// Todo(sagar) this takes almost 1 second - likely due to license check
/// Todo(sagar) Remove set_device from here.
#[tracing::instrument()]
pub fn icicle_init() -> bool {
    let mut initialized = false;

    ICICLE_INIT.call_once(|| {
        #[cfg(feature = "icicle")]
        if icicle_runtime::load_backend_from_env_or_default().is_ok() {
            if let Ok(devices) = icicle_runtime::get_registered_devices() {
                println!("Initializing icicle: available devices {:?}", devices);

                // Attempt to set the CUDA device as active
                let device = icicle_runtime::Device::new("CUDA", 0);
                if icicle_runtime::set_device(&device).is_ok() {
                    println!("icicle using device: {:?}", device);
                    initialized = true;
                } else {
                    println!("Failed to set CUDA device; falling back to CPU.");
                }
            }
        }

        #[cfg(not(feature = "icicle"))]
        {
            initialized = false;
        }

        #[cfg(feature = "icicle")]
        if !initialized {
            println!("Failed to initialize icicle backend; using JOLT CPU implementations.");
        }

        ICICLE_READY.store(initialized, Ordering::Relaxed);
    });

    ICICLE_READY.load(Ordering::Relaxed)
}

/// Returns the total memory available on the system in bits.
///
/// If icicle is enabled, it will return the total memory available on the GPU in bits.
#[allow(dead_code)]
pub fn total_memory_bits() -> usize {
    const DEFAULT_MEM_GB: usize = 30;
    const BITS_PER_BYTE: usize = 8;
    const BYTES_PER_KB: usize = 1024;
    const BYTES_PER_GB: usize = 1024 * 1024 * 1024;

    #[cfg(feature = "icicle")]
    if let Ok((total_bytes, _)) = icicle_runtime::get_available_memory() {
        // If icicle is enabled and memory is available, return the total memory in bits.
        return total_bytes.checked_mul(BITS_PER_BYTE).unwrap_or(usize::MAX);
    }

    // Fallback to system memory if icicle is unavailable or not enabled.
    #[cfg(not(target_arch = "wasm32"))]
    if let Ok(mem_info) = sys_info::mem_info() {
        return (mem_info.total as usize * BYTES_PER_KB)
            .checked_mul(BITS_PER_BYTE)
            .unwrap_or(usize::MAX);
    }

    // Fallback to "default" memory if system memory retrieval fails.
    DEFAULT_MEM_GB
        .checked_mul(
            BYTES_PER_GB
                .checked_mul(BITS_PER_BYTE)
                .unwrap_or(usize::MAX),
        )
        .unwrap_or(usize::MAX)
}
