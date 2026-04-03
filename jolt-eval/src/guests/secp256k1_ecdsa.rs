use common::jolt_device::MemoryConfig;

use super::GuestConfig;

/// Secp256k1 ECDSA signature verification guest.
pub struct Secp256k1EcdsaVerify {
    pub z: [u64; 4],
    pub r: [u64; 4],
    pub s: [u64; 4],
    pub q: [u64; 8],
}

impl Default for Secp256k1EcdsaVerify {
    fn default() -> Self {
        // Test vector from examples/secp256k1-ecdsa-verify: "hello world"
        Self {
            z: [
                0x9088f7ace2efcde9,
                0xc484efe37a5380ee,
                0xa52e52d7da7dabfa,
                0xb94d27b9934d3e08,
            ],
            r: [
                0xb8fc413b4b967ed8,
                0x248d4b0b2829ab00,
                0x587f69296af3cd88,
                0x3a5d6a386e6cf7c0,
            ],
            s: [
                0x66a82f274e3dcafc,
                0x299a02486be40321,
                0x6212d714118f617e,
                0x9d452f63cf91018d,
            ],
            q: [
                0x0012563f32ed0216,
                0xee00716af6a73670,
                0x91fc70e34e00e6c8,
                0xeeb6be8b9e68868b,
                0x4780de3d5fda972d,
                0xcb1b42d72491e47f,
                0xdc7f31262e4ba2b7,
                0xdc7b004d3bb2800d,
            ],
        }
    }
}

impl GuestConfig for Secp256k1EcdsaVerify {
    fn package(&self) -> &str {
        "secp256k1-ecdsa-verify-guest"
    }
    fn memory_config(&self) -> MemoryConfig {
        use common::constants::{DEFAULT_MAX_TRUSTED_ADVICE_SIZE, DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE};
        MemoryConfig {
            max_input_size: 4096,
            max_output_size: 4096,
            max_untrusted_advice_size: DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE,
            max_trusted_advice_size: DEFAULT_MAX_TRUSTED_ADVICE_SIZE,
            stack_size: 4096,
            heap_size: 100000,
            program_size: None,
        }
    }
    fn input(&self) -> Vec<u8> {
        postcard::to_stdvec(&(self.z, self.r, self.s, self.q)).unwrap()
    }
    fn bench_name(&self) -> String {
        "prover_time_secp256k1_ecdsa_verify".to_string()
    }
}
