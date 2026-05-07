pub(in crate::expand) fn right_shift_bitmask(shift: u32, len: u32) -> u64 {
    let ones = (1u128 << (len - shift)) - 1;
    (ones << shift) as u64
}
