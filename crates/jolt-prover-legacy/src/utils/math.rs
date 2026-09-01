use ark_ff::biginteger::S64;

pub use jolt_utils::Math;

#[inline(always)]
pub fn s64_from_diff_u64s(a: u64, b: u64) -> S64 {
    if a < b {
        S64::new([b - a], false)
    } else {
        S64::new([a - b], true)
    }
}
