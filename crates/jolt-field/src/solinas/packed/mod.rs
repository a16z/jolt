//! Packed SIMD backend selection for the Solinas fields.
//!
//! `simd.rs` holds the per-ISA primitive vocabularies, `engine.rs` and
//! `fp128.rs` the shared packed algebra stamped per (width × ISA), and
//! `ext.rs` the packed extension towers. This module picks the widest
//! backend the compilation target supports, falling back to the one-lane
//! [`NoPacking`].

mod engine;
mod ext;
mod fp128;
mod simd;

pub use ext::{PackedFpExt2, PackedFpExt4, PackedFpExt8};

use crate::solinas::{Fp128, Fp32, Fp64};
use crate::WithPacking;

/// Selects the packed backend for a Solinas prime at compile time:
/// NEON on aarch64, AVX-512 then AVX2 on x86-64, scalar [`NoPacking`]
/// otherwise.
macro_rules! select_packing {
    ($alias:ident<$p:ident: $ty:ty>, $scalar:ident, $engine:ident) => {
        /// Selected packed backend for this prime width.
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        pub type $alias<const $p: $ty> = engine::$engine<$p, simd::Neon>;

        /// Selected packed backend for this prime width.
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512dq"
        ))]
        pub type $alias<const $p: $ty> = engine::$engine<$p, simd::Avx512>;

        /// Selected packed backend for this prime width.
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            not(all(target_feature = "avx512f", target_feature = "avx512dq"))
        ))]
        pub type $alias<const $p: $ty> = engine::$engine<$p, simd::Avx2>;

        /// Selected packed backend for this prime width.
        #[cfg(not(any(
            all(target_arch = "aarch64", target_feature = "neon"),
            all(target_arch = "x86_64", target_feature = "avx2")
        )))]
        pub type $alias<const $p: $ty> = crate::NoPacking<$scalar<$p>>;

        impl<const $p: $ty> WithPacking for $scalar<$p> {
            type Packing = $alias<$p>;
        }
    };
}

select_packing!(Fp32Packing<P: u32>, Fp32, PackedFp32);
select_packing!(Fp64Packing<P: u64>, Fp64, PackedFp64);
select_packing!(Fp128Packing<P: u128>, Fp128, PackedFp128);
