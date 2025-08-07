/// Helper function to convert Vec<u64> to iterator of i128
pub fn u64_vec_to_i128_iter(vec: &[u64]) -> impl Iterator<Item = i128> + '_ {
    vec.iter().map(|v| *v as i128)
}
