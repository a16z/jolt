use crate::field::JoltField;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SmallScalarConversionError {
    OutOfRange { value: String, target_type: &'static str },
}

impl std::fmt::Display for SmallScalarConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SmallScalarConversionError::OutOfRange { value, target_type } => {
                write!(f, "Value {} is out of range for type {}", value, target_type)
            }
        }
    }
}

impl std::error::Error for SmallScalarConversionError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmallScalar {
    Bool(bool),
    U8(u8),
    U64(u64),
    I64(i64),
    U128(u128),
    I128(i128),
}

impl SmallScalar {
    pub fn as_i128(self) -> i128 {
        match self {
            SmallScalar::Bool(v) => v as i128,
            SmallScalar::U8(v) => v as i128,
            SmallScalar::U64(v) => v as i128,
            SmallScalar::I64(v) => v as i128,
            SmallScalar::U128(v) => v as i128, // lossy if > i128::MAX; callers choose variant appropriately
            SmallScalar::I128(v) => v,
        }
    }

    pub fn as_u64_clamped(self) -> u64 {
        match self {
            SmallScalar::Bool(v) => v as u64,
            SmallScalar::U8(v) => v as u64,
            SmallScalar::U64(v) => v,
            SmallScalar::I64(v) => v.max(0) as u64,
            SmallScalar::U128(v) => v as u64,
            SmallScalar::I128(v) => {
                if v >= 0 {
                    v as u64
                } else {
                    0
                }
            }
        }
    }

    pub fn to_i8(self) -> i8 {
        match self {
            SmallScalar::Bool(v) => v as i8,
            SmallScalar::U8(v) => v as i8,
            SmallScalar::U64(v) => (v as i128).clamp(i8::MIN as i128, i8::MAX as i128) as i8,
            SmallScalar::I64(v) => v.clamp(i8::MIN as i64, i8::MAX as i64) as i8,
            SmallScalar::U128(v) => (v as i128).clamp(i8::MIN as i128, i8::MAX as i128) as i8,
            SmallScalar::I128(v) => v.clamp(i8::MIN as i128, i8::MAX as i128) as i8,
        }
    }

    pub fn as_bool(self) -> bool {
        match self {
            SmallScalar::Bool(v) => v,
            SmallScalar::U8(v) => v != 0,
            SmallScalar::U64(v) => v != 0,
            SmallScalar::I64(v) => v != 0,
            SmallScalar::U128(v) => v != 0,
            SmallScalar::I128(v) => v != 0,
        }
    }

    /// Optimized multiplication with a field element, avoiding conversions when possible
    pub fn mul_field<F: JoltField>(self, field: F) -> F {
        match self {
            // Most efficient: Bool multiplication
            SmallScalar::Bool(v) => {
                if v {
                    field
                } else {
                    F::zero()
                }
            }
            // Use specialized field multiplication methods for different sizes
            SmallScalar::U8(v) => field.mul_u64(v as u64),
            SmallScalar::U64(v) => field.mul_u64(v),
            SmallScalar::I64(v) => field.mul_i128(v as i128),
            SmallScalar::U128(v) => field.mul_u128(v),
            SmallScalar::I128(v) => field.mul_i128(v),
        }
    }
}

// ===============================================
// Conversion trait implementations
// ===============================================

impl From<SmallScalar> for bool {
    fn from(scalar: SmallScalar) -> bool {
        match scalar {
            SmallScalar::Bool(v) => v,
            SmallScalar::U8(v) => v != 0,
            SmallScalar::U64(v) => v != 0,
            SmallScalar::I64(v) => v != 0,
            SmallScalar::U128(v) => v != 0,
            SmallScalar::I128(v) => v != 0,
        }
    }
}

impl From<SmallScalar> for i8 {
    /// Clamped conversion to i8 - values outside i8 range are clamped to i8::MIN or i8::MAX
    fn from(scalar: SmallScalar) -> i8 {
        match scalar {
            SmallScalar::Bool(v) => v as i8,
            SmallScalar::U8(v) => v as i8,
            SmallScalar::U64(v) => (v as i128).clamp(i8::MIN as i128, i8::MAX as i128) as i8,
            SmallScalar::I64(v) => v.clamp(i8::MIN as i64, i8::MAX as i64) as i8,
            SmallScalar::U128(v) => (v as i128).clamp(i8::MIN as i128, i8::MAX as i128) as i8,
            SmallScalar::I128(v) => v.clamp(i8::MIN as i128, i8::MAX as i128) as i8,
        }
    }
}


impl From<SmallScalar> for u64 {
    /// Clamped conversion to u64 - negative values become 0, larger values are truncated
    fn from(scalar: SmallScalar) -> u64 {
        match scalar {
            SmallScalar::Bool(v) => v as u64,
            SmallScalar::U8(v) => v as u64,
            SmallScalar::U64(v) => v,
            SmallScalar::I64(v) => v.max(0) as u64,
            SmallScalar::U128(v) => v as u64, // truncates if > u64::MAX
            SmallScalar::I128(v) => {
                if v >= 0 {
                    v as u64 // truncates if > u64::MAX
                } else {
                    0
                }
            }
        }
    }
}


impl TryFrom<SmallScalar> for i128 {
    type Error = SmallScalarConversionError;

    /// Exact conversion to i128 - fails if U128 value > i128::MAX
    fn try_from(scalar: SmallScalar) -> Result<i128, Self::Error> {
        let result = match scalar {
            SmallScalar::Bool(v) => v as i128,
            SmallScalar::U8(v) => v as i128,
            SmallScalar::U64(v) => v as i128,
            SmallScalar::I64(v) => v as i128,
            SmallScalar::U128(v) => {
                if v <= i128::MAX as u128 {
                    v as i128
                } else {
                    return Err(SmallScalarConversionError::OutOfRange {
                        value: v.to_string(),
                        target_type: "i128",
                    });
                }
            }
            SmallScalar::I128(v) => v,
        };
        Ok(result)
    }
}