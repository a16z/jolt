//! Common constants for Jolt inline implementations
//!
//! This module defines the opcode, funct3, and funct7 values used by all inline
//! implementations to ensure consistency across the codebase.

/// Common opcode for all inline instructions
pub const INLINE_OPCODE: u32 = 0x0B;

/// SHA-256 inline instruction constants
pub mod sha256 {
    /// SHA256 default compression instruction
    pub mod default {
        pub const FUNCT3: u32 = 0x00;
        pub const FUNCT7: u32 = 0x00;
        pub const NAME: &str = "SHA256_INLINE";
    }

    /// SHA256 initialization instruction
    pub mod init {
        pub const FUNCT3: u32 = 0x01;
        pub const FUNCT7: u32 = 0x00;
        pub const NAME: &str = "SHA256_INIT_INLINE";
    }
}

/// Keccak-256 inline instruction constants
pub mod keccak256 {
    pub const FUNCT3: u32 = 0x00;
    pub const FUNCT7: u32 = 0x01;
    pub const NAME: &str = "KECCAK256_INLINE";
}

pub mod blake2 {
    pub const FUNCT3: u32 = 0x00;
    pub const FUNCT7: u32 = 0x02;
    pub const NAME: &str = "BLAKE2_INLINE";
}

pub mod blake3 {
    pub const FUNCT3: u32 = 0x00;
    pub const FUNCT7: u32 = 0x03;
    pub const NAME: &str = "BLAKE3_INLINE";
}

/// BigInt operations inline instruction constants
pub mod bigint {
    /// 256-bit multiplication
    pub mod mul256 {
        pub const FUNCT3: u32 = 0x00;
        pub const FUNCT7: u32 = 0x04; // Changed from 0x02 to avoid collision with blake2
        pub const NAME: &str = "BIGINT256_MUL";
    }
}
