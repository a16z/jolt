use jolt_field::packed::{Fp32Packing, HasPacking};
use jolt_field::{
    Fp32, Prime128Offset275, Prime31Offset19, Prime32Offset99, Prime40Offset195, Prime48Offset59,
    Prime56Offset27, Prime64Offset59,
};

pub(crate) type Mersenne31 = Fp32<{ (1u32 << 31) - 1 }>;
pub(crate) type PackedMersenne31 = Fp32Packing<{ (1u32 << 31) - 1 }>;
pub(crate) type P31O19 = <Prime31Offset19 as HasPacking>::Packing;
pub(crate) type P32O99 = <Prime32Offset99 as HasPacking>::Packing;
pub(crate) type P40O195 = <Prime40Offset195 as HasPacking>::Packing;
pub(crate) type P48O59 = <Prime48Offset59 as HasPacking>::Packing;
pub(crate) type P56O27 = <Prime56Offset27 as HasPacking>::Packing;
pub(crate) type P64O59 = <Prime64Offset59 as HasPacking>::Packing;
pub(crate) type P128O275 = <Prime128Offset275 as HasPacking>::Packing;
pub(crate) type F128 = Prime128Offset275;

pub(crate) const PRIME31_OFFSET19: &str = "prime31_offset19";
pub(crate) const MERSENNE31: &str = "mersenne31";
pub(crate) const PRIME32_OFFSET99: &str = "prime32_offset99";
pub(crate) const PRIME40_OFFSET195: &str = "prime40_offset195";
pub(crate) const PRIME48_OFFSET59: &str = "prime48_offset59";
pub(crate) const PRIME56_OFFSET27: &str = "prime56_offset27";
pub(crate) const PRIME64_OFFSET59: &str = "prime64_offset59";
pub(crate) const PRIME128_OFFSET275: &str = "prime128_offset275";
