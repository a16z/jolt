/// Embed primitive integer values into a scalar object.
pub trait FromPrimitiveInt: Sized {
    #[inline]
    fn from_bool(v: bool) -> Self {
        if v {
            Self::from_u64(1)
        } else {
            Self::from_u64(0)
        }
    }

    #[inline]
    fn from_u8(v: u8) -> Self {
        Self::from_u64(v as u64)
    }

    #[inline]
    fn from_i8(v: i8) -> Self {
        Self::from_i64(v as i64)
    }

    #[inline]
    fn from_u16(v: u16) -> Self {
        Self::from_u64(v as u64)
    }

    #[inline]
    fn from_i16(v: i16) -> Self {
        Self::from_i64(v as i64)
    }

    #[inline]
    fn from_u32(v: u32) -> Self {
        Self::from_u64(v as u64)
    }

    #[inline]
    fn from_i32(v: i32) -> Self {
        Self::from_i64(v as i64)
    }

    fn from_u64(v: u64) -> Self;
    fn from_i64(v: i64) -> Self;
    fn from_u128(v: u128) -> Self;
    fn from_i128(v: i128) -> Self;
}
