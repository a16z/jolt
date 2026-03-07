//! Type-erased buffer view for heterogeneous polynomial inputs.
//!
//! During sumcheck, polynomials have different underlying scalar types
//! (`u8`, `i64`, `Fr`, etc.) but must all feed into the same kernel
//! evaluation. [`AnyBuffer`] wraps a typed slice and provides field-element
//! pair access without virtual dispatch — variant matching branch-predicts
//! perfectly since all elements in a buffer share the same type.

use jolt_field::Field;

/// Type-erased view into a buffer that provides field-element pairs.
///
/// Wraps a slice of any scalar type that converts to `F` and provides
/// indexed pair access for [`pairwise_reduce_mixed`](crate::CpuBackend::pairwise_reduce_mixed).
///
/// # Performance
///
/// The `match` in [`pair`](Self::pair) is branch-predicted perfectly because
/// all elements in a buffer share the same type. This is faster than
/// `dyn` trait dispatch (no indirect call) and avoids the type-erasure
/// overhead of function pointers.
///
/// # Construction
///
/// Use the [`field`](Self::field) constructor for field-element buffers,
/// or the `From<&[T]>` impls for compact scalar types.
pub enum AnyBuffer<'a, F: Field> {
    Field(&'a [F]),
    Bool(&'a [bool]),
    U8(&'a [u8]),
    U16(&'a [u16]),
    U32(&'a [u32]),
    U64(&'a [u64]),
    U128(&'a [u128]),
    I64(&'a [i64]),
    I128(&'a [i128]),
}

impl<'a, F: Field> AnyBuffer<'a, F> {
    /// Wraps a field-element slice.
    #[inline]
    pub fn field(s: &'a [F]) -> Self {
        Self::Field(s)
    }

    /// Number of elements in the underlying slice.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Field(s) => s.len(),
            Self::Bool(s) => s.len(),
            Self::U8(s) => s.len(),
            Self::U16(s) => s.len(),
            Self::U32(s) => s.len(),
            Self::U64(s) => s.len(),
            Self::U128(s) => s.len(),
            Self::I64(s) => s.len(),
            Self::I128(s) => s.len(),
        }
    }

    /// Number of interleaved pairs: `len() / 2`.
    #[inline]
    pub fn half_len(&self) -> usize {
        self.len() / 2
    }

    /// Returns `true` if the buffer has no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reads the `i`-th interleaved pair as field elements.
    ///
    /// Returns `(buf[2*i], buf[2*i+1])` promoted to `F`.
    #[inline]
    pub fn pair(&self, i: usize) -> (F, F) {
        match self {
            Self::Field(s) => (s[2 * i], s[2 * i + 1]),
            Self::Bool(s) => (F::from_bool(s[2 * i]), F::from_bool(s[2 * i + 1])),
            Self::U8(s) => (F::from_u8(s[2 * i]), F::from_u8(s[2 * i + 1])),
            Self::U16(s) => (F::from_u16(s[2 * i]), F::from_u16(s[2 * i + 1])),
            Self::U32(s) => (F::from_u32(s[2 * i]), F::from_u32(s[2 * i + 1])),
            Self::U64(s) => (F::from_u64(s[2 * i]), F::from_u64(s[2 * i + 1])),
            Self::U128(s) => (F::from_u128(s[2 * i]), F::from_u128(s[2 * i + 1])),
            Self::I64(s) => (F::from_i64(s[2 * i]), F::from_i64(s[2 * i + 1])),
            Self::I128(s) => (F::from_i128(s[2 * i]), F::from_i128(s[2 * i + 1])),
        }
    }
}

macro_rules! impl_from_slice {
    ($variant:ident, $ty:ty) => {
        impl<'a, F: Field> From<&'a [$ty]> for AnyBuffer<'a, F> {
            #[inline]
            fn from(s: &'a [$ty]) -> Self {
                Self::$variant(s)
            }
        }
        impl<'a, F: Field> From<&'a Vec<$ty>> for AnyBuffer<'a, F> {
            #[inline]
            fn from(s: &'a Vec<$ty>) -> Self {
                Self::$variant(s.as_slice())
            }
        }
    };
}

impl_from_slice!(Bool, bool);
impl_from_slice!(U8, u8);
impl_from_slice!(U16, u16);
impl_from_slice!(U32, u32);
impl_from_slice!(U64, u64);
impl_from_slice!(U128, u128);
impl_from_slice!(I64, i64);
impl_from_slice!(I128, i128);
