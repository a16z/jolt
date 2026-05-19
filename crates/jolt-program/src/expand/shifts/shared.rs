/// Returns the mask-shaped immediate expected by the virtual right-shift rows.
///
/// The virtual shift instructions recover the actual shift amount from the
/// immediate's trailing-zero count. Keeping the upper `len - shift` bits set
/// gives the lookup table the same mask value used for range/correctness
/// checks while still encoding the immediate shift compactly.
pub(in crate::expand) fn right_shift_bitmask(shift: u32, len: u32) -> u64 {
    let ones = (1u128 << (len - shift)) - 1;
    (ones << shift) as u64
}
