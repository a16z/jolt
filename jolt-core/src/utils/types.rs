/// Right-hand instruction input value used by zkVM instruction logic.
///
/// Captures the semantic signedness of the right operand at XLEN width.
/// - `Unsigned(u64)`: operand is interpreted as an XLEN-bit unsigned word
/// - `Signed(i64)`: operand is interpreted as an XLEN-bit two's-complement signed word
///
/// Helper methods provide width-aware projections to `u64`/`i64` and a
/// canonical unsigned representation for lookup key construction.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum U64OrI64 {
    Unsigned(u64),
    Signed(i64),
}

impl U64OrI64 {
    /// Return the value as an unsigned 64-bit word (XLEN=64 view).
    #[inline]
    pub fn as_u64(&self) -> u64 {
        match *self {
            U64OrI64::Unsigned(u) => u,
            U64OrI64::Signed(s) => s as u64,
        }
    }

    /// Return the value as an unsigned 32-bit word (XLEN=32 view).
    #[inline]
    pub fn as_u32(&self) -> u32 {
        match *self {
            U64OrI64::Unsigned(u) => u as u32,
            U64OrI64::Signed(s) => s as u32,
        }
    }

    /// Return the value as an unsigned 8-bit word (XLEN=8 view).
    #[inline]
    pub fn as_u8(&self) -> u8 {
        match *self {
            U64OrI64::Unsigned(u) => u as u8,
            U64OrI64::Signed(s) => s as u8,
        }
    }

    /// Return the value as a signed 64-bit word (XLEN=64 view).
    #[inline]
    pub fn as_i64(&self) -> i64 {
        match *self {
            U64OrI64::Unsigned(u) => u as i64,
            U64OrI64::Signed(s) => s,
        }
    }

    /// Return the value as a signed 32-bit word (XLEN=32 view).
    /// This is a truncating conversion.
    #[inline]
    pub fn as_i32(&self) -> i32 {
        match *self {
            U64OrI64::Unsigned(u) => u as i32,
            U64OrI64::Signed(s) => s as i32,
        }
    }

    /// Return the value as a signed 8-bit word (XLEN=8 view).
    /// This is a truncating conversion.
    #[inline]
    pub fn as_i8(&self) -> i8 {
        match *self {
            U64OrI64::Unsigned(u) => u as i8,
            U64OrI64::Signed(s) => s as i8,
        }
    }

    /// Return the value widened to i128.
    #[inline]
    pub fn as_i128(&self) -> i128 {
        match *self {
            U64OrI64::Unsigned(u) => u as i128,
            U64OrI64::Signed(s) => s as i128,
        }
    }

    /// Return a canonical unsigned representation suitable for lookup keys.
    ///
    /// This is the XLEN-masked view of the value, promoted to `u128`.
    #[inline]
    pub fn to_u128_lookup<const XLEN: usize>(&self) -> u128 {
        match XLEN {
            64 => self.as_u64() as u128,
            32 => self.as_u32() as u128,
            8 => self.as_u8() as u128,
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum U128OrI128 {
    Unsigned(u128),
    Signed(i128),
}

#[cfg(test)]
/// Validate that specialized projections as_{u,i}{8,32,64} are equivalent to
/// widening to i128 and narrowing back for both Unsigned and Signed variants
/// under XLEN views 8, 32, and 64.
mod tests {
    use super::U64OrI64 as RIV;
    use rand::Rng;

    fn check_equivalence(v: RIV) {
        let i128_wide = v.as_i128();

        // XLEN=8
        assert_eq!(i128_wide as u8, v.as_u8());
        assert_eq!(i128_wide as i8, v.as_i8());

        // XLEN=32
        assert_eq!(i128_wide as u32, v.as_u32());
        assert_eq!(i128_wide as i32, v.as_i32());

        // XLEN=64
        assert_eq!(i128_wide as u64, v.as_u64());
        assert_eq!(i128_wide as i64, v.as_i64());
    }

    #[test]
    fn projections_match_i128_path_unsigned() {
        let cases: &[u64] = &[
            0,
            1,
            0x7F,
            0x80,
            0xFF,
            0x7FFF_FFFF,
            0x8000_0000,
            0xFFFF_FFFF,
            0x7FFF_FFFF_FFFF_FFFF,
            0x8000_0000_0000_0000,
            0xFFFF_FFFF_FFFF_FFFF,
        ];
        for &u in cases {
            check_equivalence(RIV::Unsigned(u));
        }

        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            check_equivalence(RIV::Unsigned(rng.gen()));
        }
    }

    #[test]
    fn projections_match_i128_path_signed() {
        let cases: &[i64] = &[
            0,
            1,
            -1,
            i64::MIN,
            i64::MAX,
            -128,
            127,
            -0x8000_0000,
            0x7FFF_FFFF,
        ];
        for &s in cases {
            check_equivalence(RIV::Signed(s));
        }

        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            check_equivalence(RIV::Signed(rng.gen()));
        }
    }
}
