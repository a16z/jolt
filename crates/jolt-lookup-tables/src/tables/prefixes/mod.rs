//! Prefix polynomial evaluations for the sparse-dense decomposition.
//!
//! Each prefix captures the "contribution" of high-order bound variables
//! to a lookup table's MLE during sumcheck. At the start of each phase,
//! prefixes are evaluated at binary points to materialize a dense polynomial,
//! which is then bound during the phase's sumcheck rounds. The fully bound
//! value becomes the prefix's checkpoint for the next phase. Checkpoints are
//! initialized via [`SparseDensePrefix::default_checkpoint`].

pub mod align_addr;
pub mod and;
pub mod andn;
pub mod div_by_zero;
pub mod eq;
pub mod left_is_zero;
pub mod left_msb_right_operand;
pub mod left_msb_right_operand_is_zero;
pub mod left_operand_msb;
pub mod left_shift;
pub mod left_shift_helper;
pub mod left_shift_w;
pub mod left_shift_w_helper;
pub mod lower_half_word;
pub mod lower_word;
pub mod lsb;
pub mod lt;
pub mod offset_scale;
pub mod or;
pub mod overflow_bits_zero;
pub mod pow2;
pub mod pow2_offset;
pub mod pow2_w;
pub mod rev8w;
pub mod right_is_zero;
pub mod right_operand;
pub mod right_operand_msb;
pub mod right_operand_w;
pub mod right_shift;
pub mod right_shift_w;
pub mod shift_data;
pub mod sign_extension;
pub mod sign_extension_right_operand;
pub mod sign_extension_upper_half;
pub mod sign_extension_w;
pub mod srlw_sext;
pub mod two_lsb;
pub mod upper_word;
pub mod window_sign;
pub mod window_sign_pow2;
pub mod word_msb;
pub mod xor;
pub mod xor_rot;
pub mod xor_rotw;

use jolt_field::JoltField;
use std::fmt::Display;
use std::ops::Index;

use crate::lookup_bits::LookupBits;
use align_addr::AlignAddrPrefix;
use pow2_offset::Pow2OffsetPrefix;

/// A prefix polynomial evaluated at binary points during materialization.
///
/// Implementations provide:
/// - `default_checkpoint()`: the initial checkpoint value before any phases
/// - `evaluate()`: the prefix value at a binary point, given accumulated
///   checkpoints from previous phases
pub trait SparseDensePrefix<F: JoltField>: 'static + Sync {
    /// Default checkpoint value for this prefix before any phases have run.
    fn default_checkpoint() -> F;

    /// Evaluate this prefix at binary point `b`, given accumulated checkpoints
    /// from previous phases and the number of remaining suffix variables.
    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F;
}

/// Wrapper for prefix polynomial evaluations, used for type safety.
#[derive(Clone, Copy)]
pub struct PrefixEval<F>(pub(crate) F);

/// Full RV64 instruction lookup address width.
pub const LOG_K: usize = 2 * crate::XLEN;

impl<F: Display> Display for PrefixEval<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<F> From<F> for PrefixEval<F> {
    fn from(value: F) -> Self {
        Self(value)
    }
}

impl<F: Copy> PrefixEval<F> {
    /// Returns the underlying field evaluation.
    pub fn value(self) -> F {
        self.0
    }
}

impl<F> Index<Prefixes> for &[PrefixEval<F>] {
    type Output = F;

    #[expect(
        clippy::unwrap_used,
        clippy::get_unwrap,
        reason = "checkpoint slices are sized to Prefixes::COUNT, so every variant index is in range"
    )]
    fn index(&self, prefix: Prefixes) -> &Self::Output {
        let index = prefix as usize;
        &self.get(index).unwrap().0
    }
}

/// All prefix types used by Jolt's lookup tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, strum::EnumCount, strum::VariantArray)]
#[repr(u8)]
pub enum Prefixes {
    LowerWord,
    LowerHalfWord,
    UpperWord,
    Eq,
    And,
    Andn,
    Or,
    Xor,
    LessThan,
    LeftOperandIsZero,
    RightOperandIsZero,
    LeftOperandMsb,
    RightOperandMsb,
    DivByZero,
    Lsb,
    Pow2,
    Pow2W,
    Rev8W,
    RightShift,
    SignExtension,
    LeftShift,
    LeftShiftHelper,
    TwoLsb,
    SignExtensionUpperHalf,
    RightOperand,
    LeftMsbRightOperand,
    LeftMsbRightOperandIsZero,
    RightOperandW,
    SignExtensionRightOperand,
    RightShiftW,
    LeftShiftWHelper,
    LeftShiftW,
    OverflowBitsZero,
    XorRot16,
    XorRot24,
    XorRot32,
    XorRot63,
    XorRotW7,
    XorRotW8,
    XorRotW12,
    XorRotW16,
    Pow2OffsetW,
    WindowSign,
    WindowSignPow2,
    XorRotW22,
    XorRotW19,
    XorRotW6,
    /// The low word's most-significant source bit, `x_{XLEN/2-1}`.
    WordMsb,
    /// SRAW sign-fill terms whose variables have entered the prefix.
    SignExtensionW,
    /// The prefix-owned portion of SRLW's `x_{XLEN/2-1} * y_0` predicate.
    SrlwSext,
    Pow2OffsetB,
    Pow2OffsetH,
    AlignAddr,
    ShiftDataB,
    ShiftDataH,
    ShiftDataW,
    OffsetScaleB,
    OffsetScaleH,
    OffsetScaleW,
}

/// Total number of prefix variants.
pub const NUM_PREFIXES: usize = <Prefixes as strum::EnumCount>::COUNT;

/// All prefix variants in discriminant order.
pub const ALL_PREFIXES: &[Prefixes] = <Prefixes as strum::VariantArray>::VARIANTS;

/// Dispatches a `SparseDensePrefix` method call to the concrete type for each `Prefixes` variant.
macro_rules! dispatch_prefix {
    ($self:expr, $method:ident) => {
        dispatch_prefix!($self, $method,)
    };
    ($self:expr, $method:ident, $($args:expr),* $(,)?) => {
        match $self {
            Prefixes::LowerWord => lower_word::LowerWordPrefix::$method($($args),*),
            Prefixes::LowerHalfWord => lower_half_word::LowerHalfWordPrefix::$method($($args),*),
            Prefixes::UpperWord => upper_word::UpperWordPrefix::$method($($args),*),
            Prefixes::Eq => eq::EqPrefix::$method($($args),*),
            Prefixes::And => and::AndPrefix::$method($($args),*),
            Prefixes::Andn => andn::AndnPrefix::$method($($args),*),
            Prefixes::Or => or::OrPrefix::$method($($args),*),
            Prefixes::Xor => xor::XorPrefix::$method($($args),*),
            Prefixes::LessThan => lt::LessThanPrefix::$method($($args),*),
            Prefixes::LeftOperandIsZero => left_is_zero::LeftOperandIsZeroPrefix::$method($($args),*),
            Prefixes::RightOperandIsZero => right_is_zero::RightOperandIsZeroPrefix::$method($($args),*),
            Prefixes::LeftOperandMsb => left_operand_msb::LeftOperandMsbPrefix::$method($($args),*),
            Prefixes::RightOperandMsb => right_operand_msb::RightOperandMsbPrefix::$method($($args),*),
            Prefixes::DivByZero => div_by_zero::DivByZeroPrefix::$method($($args),*),
            Prefixes::Lsb => lsb::LsbPrefix::$method($($args),*),
            Prefixes::Pow2 => pow2::Pow2Prefix::$method($($args),*),
            Prefixes::Pow2W => pow2_w::Pow2WPrefix::$method($($args),*),
            Prefixes::Rev8W => rev8w::Rev8WPrefix::$method($($args),*),
            Prefixes::RightShift => right_shift::RightShiftPrefix::$method($($args),*),
            Prefixes::SignExtension => sign_extension::SignExtensionPrefix::$method($($args),*),
            Prefixes::LeftShift => left_shift::LeftShiftPrefix::$method($($args),*),
            Prefixes::LeftShiftHelper => left_shift_helper::LeftShiftHelperPrefix::$method($($args),*),
            Prefixes::TwoLsb => two_lsb::TwoLsbPrefix::$method($($args),*),
            Prefixes::SignExtensionUpperHalf => sign_extension_upper_half::SignExtensionUpperHalfPrefix::$method($($args),*),
            Prefixes::RightOperand => right_operand::RightOperandPrefix::$method($($args),*),
            Prefixes::LeftMsbRightOperand => left_msb_right_operand::LeftMsbRightOperandPrefix::$method($($args),*),
            Prefixes::LeftMsbRightOperandIsZero => left_msb_right_operand_is_zero::LeftMsbRightOperandIsZeroPrefix::$method($($args),*),
            Prefixes::RightOperandW => right_operand_w::RightOperandWPrefix::$method($($args),*),
            Prefixes::SignExtensionRightOperand => sign_extension_right_operand::SignExtensionRightOperandPrefix::$method($($args),*),
            Prefixes::RightShiftW => right_shift_w::RightShiftWPrefix::$method($($args),*),
            Prefixes::LeftShiftWHelper => left_shift_w_helper::LeftShiftWHelperPrefix::$method($($args),*),
            Prefixes::LeftShiftW => left_shift_w::LeftShiftWPrefix::$method($($args),*),
            Prefixes::OverflowBitsZero => overflow_bits_zero::OverflowBitsZeroPrefix::$method($($args),*),
            Prefixes::XorRot16 => xor_rot::XorRotPrefix::<16>::$method($($args),*),
            Prefixes::XorRot24 => xor_rot::XorRotPrefix::<24>::$method($($args),*),
            Prefixes::XorRot32 => xor_rot::XorRotPrefix::<32>::$method($($args),*),
            Prefixes::XorRot63 => xor_rot::XorRotPrefix::<63>::$method($($args),*),
            Prefixes::XorRotW7 => xor_rotw::XorRotWPrefix::<7>::$method($($args),*),
            Prefixes::XorRotW8 => xor_rotw::XorRotWPrefix::<8>::$method($($args),*),
            Prefixes::XorRotW12 => xor_rotw::XorRotWPrefix::<12>::$method($($args),*),
            Prefixes::XorRotW16 => xor_rotw::XorRotWPrefix::<16>::$method($($args),*),
            Prefixes::Pow2OffsetW => Pow2OffsetPrefix::<2>::$method($($args),*),
            Prefixes::WindowSign => window_sign::WindowSignPrefix::$method($($args),*),
            Prefixes::WindowSignPow2 => window_sign_pow2::WindowSignPow2Prefix::$method($($args),*),
            Prefixes::XorRotW22 => xor_rotw::XorRotWPrefix::<22>::$method($($args),*),
            Prefixes::XorRotW19 => xor_rotw::XorRotWPrefix::<19>::$method($($args),*),
            Prefixes::XorRotW6 => xor_rotw::XorRotWPrefix::<6>::$method($($args),*),
            Prefixes::WordMsb => word_msb::WordMsbPrefix::$method($($args),*),
            Prefixes::SignExtensionW => sign_extension_w::SignExtensionWPrefix::$method($($args),*),
            Prefixes::SrlwSext => srlw_sext::SrlwSextPrefix::$method($($args),*),
            Prefixes::Pow2OffsetB => Pow2OffsetPrefix::<0>::$method($($args),*),
            Prefixes::Pow2OffsetH => Pow2OffsetPrefix::<1>::$method($($args),*),
            Prefixes::AlignAddr => AlignAddrPrefix::$method($($args),*),
            Prefixes::ShiftDataB => shift_data::ShiftDataPrefix::<1>::$method($($args),*),
            Prefixes::ShiftDataH => shift_data::ShiftDataPrefix::<2>::$method($($args),*),
            Prefixes::ShiftDataW => shift_data::ShiftDataPrefix::<4>::$method($($args),*),
            Prefixes::OffsetScaleB => offset_scale::OffsetScalePrefix::<1>::$method($($args),*),
            Prefixes::OffsetScaleH => offset_scale::OffsetScalePrefix::<2>::$method($($args),*),
            Prefixes::OffsetScaleW => offset_scale::OffsetScalePrefix::<4>::$method($($args),*),
        }
    };
}

impl Prefixes {
    /// Return the default checkpoint value for this prefix variant.
    pub fn default_checkpoint<F: JoltField>(&self) -> PrefixEval<F> {
        PrefixEval(dispatch_prefix!(self, default_checkpoint))
    }

    /// Evaluate this prefix at binary point `b`.
    pub fn evaluate<F: JoltField>(
        &self,
        checkpoints: &[PrefixEval<F>],
        b: LookupBits,
        suffix_len: usize,
    ) -> PrefixEval<F> {
        PrefixEval(dispatch_prefix!(self, evaluate, checkpoints, b, suffix_len))
    }
}
