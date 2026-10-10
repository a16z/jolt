use std::arch::aarch64::vmull_p64;

#[inline]
pub(super) fn clmul(a: u64, b: u64) -> u128 {
    // SAFETY: this module requires cfg(all(target_arch = "aarch64", target_feature = "aes")),
    // which enables vmull_p64.
    unsafe { vmull_p64(a, b) }
}
