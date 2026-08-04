use jolt_field::AkitaField;
use jolt_kernels::metal::solinas::Fp128;

#[inline(always)]
pub(crate) fn add(lhs: Fp128, rhs: Fp128) -> Fp128 {
    Fp128::from_jolt_field(
        &(lhs.into_jolt_field::<AkitaField>() + rhs.into_jolt_field::<AkitaField>()),
    )
}

#[inline(always)]
pub(crate) fn sub(lhs: Fp128, rhs: Fp128) -> Fp128 {
    Fp128::from_jolt_field(
        &(lhs.into_jolt_field::<AkitaField>() - rhs.into_jolt_field::<AkitaField>()),
    )
}

#[inline(always)]
pub(crate) fn bind(lo: Fp128, hi: Fp128, challenge: Fp128) -> Fp128 {
    add(lo, mul(challenge, sub(hi, lo)))
}

#[inline(always)]
pub(crate) fn mul(lhs: Fp128, rhs: Fp128) -> Fp128 {
    Fp128::from_jolt_field(
        &(lhs.into_jolt_field::<AkitaField>() * rhs.into_jolt_field::<AkitaField>()),
    )
}
