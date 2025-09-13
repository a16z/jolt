/// Right-hand instruction input value used by zkVM instruction logic.
///
/// Captures the semantic signedness of the right operand at XLEN width.
/// - `Unsigned(u64)`: operand is interpreted as an XLEN-bit unsigned word
/// - `Signed(i64)`: operand is interpreted as an XLEN-bit two's-complement signed word
///
/// Helper methods provide width-aware projections to `u64`/`i64` and a
/// canonical unsigned representation for lookup key construction.
use allocative::Allocative;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Allocative)]
pub enum U64OrI64 {
    Unsigned(u64),
    Signed(i64),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Allocative)]
pub enum U128OrI128 {
    Unsigned(u128),
    Signed(i128),
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

impl U128OrI128 {
    #[inline]
    pub fn as_u128(&self) -> u128 {
        match *self {
            U128OrI128::Unsigned(u) => u,
            U128OrI128::Signed(s) => s as u128,
        }
    }

    #[inline]
    pub fn as_i128(&self) -> i128 {
        match *self {
            U128OrI128::Unsigned(u) => u as i128,
            U128OrI128::Signed(s) => s,
        }
    }
}

impl core::cmp::PartialOrd for U64OrI64 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl core::cmp::Ord for U64OrI64 {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        match (self, other) {
            (U64OrI64::Unsigned(a), U64OrI64::Unsigned(b)) => a.cmp(b),
            (U64OrI64::Signed(a), U64OrI64::Signed(b)) => a.cmp(b),
            (U64OrI64::Unsigned(a), U64OrI64::Signed(b)) => {
                if *b < 0 {
                    core::cmp::Ordering::Greater
                } else {
                    a.cmp(&(*b as u64))
                }
            }
            (U64OrI64::Signed(a), U64OrI64::Unsigned(b)) => {
                if *a < 0 {
                    core::cmp::Ordering::Less
                } else {
                    (*a as u64).cmp(b)
                }
            }
        }
    }
}

impl core::cmp::PartialOrd for U128OrI128 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl core::cmp::Ord for U128OrI128 {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        match (self, other) {
            (U128OrI128::Unsigned(a), U128OrI128::Unsigned(b)) => a.cmp(b),
            (U128OrI128::Signed(a), U128OrI128::Signed(b)) => a.cmp(b),
            (U128OrI128::Unsigned(a), U128OrI128::Signed(b)) => {
                if *b < 0 {
                    core::cmp::Ordering::Greater
                } else {
                    a.cmp(&(*b as u128))
                }
            }
            (U128OrI128::Signed(a), U128OrI128::Unsigned(b)) => {
                if *a < 0 {
                    core::cmp::Ordering::Less
                } else {
                    (*a as u128).cmp(b)
                }
            }
        }
    }
}

impl Default for U64OrI64 {
    fn default() -> Self {
        U64OrI64::Unsigned(0)
    }
}

impl Default for U128OrI128 {
    fn default() -> Self {
        U128OrI128::Unsigned(0)
    }
}

impl Valid for U64OrI64 {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl Valid for U128OrI128 {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalSerialize for U64OrI64 {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            U64OrI64::Unsigned(u) => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                u.serialize_with_mode(writer, compress)
            }
            U64OrI64::Signed(s) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                s.serialize_with_mode(writer, compress)
            }
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        0u8.serialized_size(compress)
            + match self {
                U64OrI64::Unsigned(u) => u.serialized_size(compress),
                U64OrI64::Signed(s) => s.serialized_size(compress),
            }
    }
}

impl CanonicalDeserialize for U64OrI64 {
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: Compress,
        _validate: Validate,
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(&mut reader, compress, Validate::No)?;
        match tag {
            0 => {
                let u = u64::deserialize_with_mode(reader, compress, Validate::No)?;
                Ok(U64OrI64::Unsigned(u))
            }
            1 => {
                let s = i64::deserialize_with_mode(reader, compress, Validate::No)?;
                Ok(U64OrI64::Signed(s))
            }
            _ => Err(SerializationError::InvalidData),
        }
    }
}

impl CanonicalSerialize for U128OrI128 {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            U128OrI128::Unsigned(u) => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                u.serialize_with_mode(writer, compress)
            }
            U128OrI128::Signed(s) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                s.serialize_with_mode(writer, compress)
            }
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        0u8.serialized_size(compress)
            + match self {
                U128OrI128::Unsigned(u) => u.serialized_size(compress),
                U128OrI128::Signed(s) => s.serialized_size(compress),
            }
    }
}

impl CanonicalDeserialize for U128OrI128 {
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: Compress,
        _validate: Validate,
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(&mut reader, compress, Validate::No)?;
        match tag {
            0 => {
                let u = u128::deserialize_with_mode(reader, compress, Validate::No)?;
                Ok(U128OrI128::Unsigned(u))
            }
            1 => {
                let s = i128::deserialize_with_mode(reader, compress, Validate::No)?;
                Ok(U128OrI128::Signed(s))
            }
            _ => Err(SerializationError::InvalidData),
        }
    }
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
