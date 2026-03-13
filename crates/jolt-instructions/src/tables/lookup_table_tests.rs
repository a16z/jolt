#![allow(unused_results)]

//! Comprehensive lookup table correctness tests.
//!
//! Three test tiers per table:
//! 1. `mle_full_hypercube` — exhaustive 2^16 check at XLEN=8
//! 2. `mle_random` — 1000 random points at XLEN=64
//! 3. `prefix_suffix` — sparse-dense decomposition vs MLE across all sumcheck rounds

use jolt_field::Fr;

use super::test_utils::{mle_full_hypercube_test, mle_random_test, prefix_suffix_test};

use super::and::AndTable;
use super::andn::AndnTable;
use super::equal::EqualTable;
use super::halfword_alignment::HalfwordAlignmentTable;
use super::lower_half_word::LowerHalfWordTable;
use super::movsign::MovsignTable;
use super::mulu_no_overflow::MulUNoOverflowTable;
use super::not_equal::NotEqualTable;
use super::or::OrTable;
use super::pow2::Pow2Table;
use super::pow2_w::Pow2WTable;
use super::range_check::RangeCheckTable;
use super::range_check_aligned::RangeCheckAlignedTable;
use super::shift_right_bitmask::ShiftRightBitmaskTable;
use super::sign_extend_half_word::SignExtendHalfWordTable;
use super::signed_greater_than_equal::SignedGreaterThanEqualTable;
use super::signed_less_than::SignedLessThanTable;
use super::unsigned_greater_than_equal::UnsignedGreaterThanEqualTable;
use super::unsigned_less_than::UnsignedLessThanTable;
use super::unsigned_less_than_equal::UnsignedLessThanEqualTable;
use super::upper_word::UpperWordTable;
use super::valid_div0::ValidDiv0Table;
use super::valid_signed_remainder::ValidSignedRemainderTable;
use super::valid_unsigned_remainder::ValidUnsignedRemainderTable;
use super::virtual_change_divisor::VirtualChangeDivisorTable;
use super::virtual_change_divisor_w::VirtualChangeDivisorWTable;
use super::virtual_rev8w::VirtualRev8WTable;
use super::virtual_rotr::VirtualRotrTable;
use super::virtual_rotrw::VirtualRotrWTable;
use super::virtual_sra::VirtualSRATable;
use super::virtual_srl::VirtualSRLTable;
use super::virtual_xor_rot::VirtualXORROTTable;
use super::virtual_xor_rotw::VirtualXORROTWTable;
use super::word_alignment::WordAlignmentTable;
use super::xor::XorTable;

macro_rules! table_tests {
    ($mod:ident, $table8:ty, $table64:ty) => {
        mod $mod {
            use super::*;

            #[test]
            fn mle_full_hypercube() {
                mle_full_hypercube_test::<Fr, $table8>();
            }

            #[test]
            fn mle_random() {
                mle_random_test::<64, Fr, $table64>();
            }

            #[test]
            fn prefix_suffix() {
                prefix_suffix_test::<64, Fr, $table64>();
            }
        }
    };
}

// Arithmetic / range-check
table_tests!(range_check, RangeCheckTable<8>, RangeCheckTable<64>);
table_tests!(
    range_check_aligned,
    RangeCheckAlignedTable<8>,
    RangeCheckAlignedTable<64>
);

// Bitwise
table_tests!(and, AndTable<8>, AndTable<64>);
table_tests!(andn, AndnTable<8>, AndnTable<64>);
table_tests!(or, OrTable<8>, OrTable<64>);
table_tests!(xor, XorTable<8>, XorTable<64>);

// Comparison
table_tests!(equal, EqualTable<8>, EqualTable<64>);
table_tests!(not_equal, NotEqualTable<8>, NotEqualTable<64>);
table_tests!(
    signed_less_than,
    SignedLessThanTable<8>,
    SignedLessThanTable<64>
);
table_tests!(
    unsigned_less_than,
    UnsignedLessThanTable<8>,
    UnsignedLessThanTable<64>
);
table_tests!(
    signed_greater_than_equal,
    SignedGreaterThanEqualTable<8>,
    SignedGreaterThanEqualTable<64>
);
table_tests!(
    unsigned_greater_than_equal,
    UnsignedGreaterThanEqualTable<8>,
    UnsignedGreaterThanEqualTable<64>
);
table_tests!(
    unsigned_less_than_equal,
    UnsignedLessThanEqualTable<8>,
    UnsignedLessThanEqualTable<64>
);

// Word extraction
table_tests!(upper_word, UpperWordTable<8>, UpperWordTable<64>);
table_tests!(
    lower_half_word,
    LowerHalfWordTable<8>,
    LowerHalfWordTable<64>
);
table_tests!(
    sign_extend_half_word,
    SignExtendHalfWordTable<8>,
    SignExtendHalfWordTable<64>
);

// Sign/conditional
table_tests!(movsign, MovsignTable<8>, MovsignTable<64>);

// Power of 2
table_tests!(pow2, Pow2Table<8>, Pow2Table<64>);
table_tests!(pow2_w, Pow2WTable<8>, Pow2WTable<64>);

// Shift
table_tests!(
    shift_right_bitmask,
    ShiftRightBitmaskTable<8>,
    ShiftRightBitmaskTable<64>
);
table_tests!(virtual_srl, VirtualSRLTable<8>, VirtualSRLTable<64>);
table_tests!(virtual_sra, VirtualSRATable<8>, VirtualSRATable<64>);
table_tests!(virtual_rotr, VirtualRotrTable<8>, VirtualRotrTable<64>);
table_tests!(virtual_rotrw, VirtualRotrWTable<8>, VirtualRotrWTable<64>);

// Division validation
table_tests!(valid_div0, ValidDiv0Table<8>, ValidDiv0Table<64>);
table_tests!(
    valid_unsigned_remainder,
    ValidUnsignedRemainderTable<8>,
    ValidUnsignedRemainderTable<64>
);
table_tests!(
    valid_signed_remainder,
    ValidSignedRemainderTable<8>,
    ValidSignedRemainderTable<64>
);
table_tests!(
    virtual_change_divisor,
    VirtualChangeDivisorTable<8>,
    VirtualChangeDivisorTable<64>
);
table_tests!(
    virtual_change_divisor_w,
    VirtualChangeDivisorWTable<8>,
    VirtualChangeDivisorWTable<64>
);

// Alignment
table_tests!(
    halfword_alignment,
    HalfwordAlignmentTable<8>,
    HalfwordAlignmentTable<64>
);
table_tests!(
    word_alignment,
    WordAlignmentTable<8>,
    WordAlignmentTable<64>
);

// Multiply overflow
table_tests!(
    mulu_no_overflow,
    MulUNoOverflowTable<8>,
    MulUNoOverflowTable<64>
);

// Byte manipulation
table_tests!(virtual_rev8w, VirtualRev8WTable<8>, VirtualRev8WTable<64>);

// XOR-rotate (SHA) — 64-bit only (no XLEN=8 hypercube test)
macro_rules! xor_rot_tests {
    ($mod:ident, $table64:ty) => {
        mod $mod {
            use super::*;

            #[test]
            fn mle_random() {
                mle_random_test::<64, Fr, $table64>();
            }

            #[test]
            fn prefix_suffix() {
                prefix_suffix_test::<64, Fr, $table64>();
            }
        }
    };
}

xor_rot_tests!(virtual_xor_rot_32, VirtualXORROTTable<64, 32>);
xor_rot_tests!(virtual_xor_rot_24, VirtualXORROTTable<64, 24>);
xor_rot_tests!(virtual_xor_rot_16, VirtualXORROTTable<64, 16>);
xor_rot_tests!(virtual_xor_rot_63, VirtualXORROTTable<64, 63>);
xor_rot_tests!(virtual_xor_rotw_16, VirtualXORROTWTable<64, 16>);
xor_rot_tests!(virtual_xor_rotw_12, VirtualXORROTWTable<64, 12>);
xor_rot_tests!(virtual_xor_rotw_8, VirtualXORROTWTable<64, 8>);
xor_rot_tests!(virtual_xor_rotw_7, VirtualXORROTWTable<64, 7>);
