use crate::subprotocols::sparse_dense_shout::LookupBits;
use eq::EqSuffix;
use lt::LessThanSuffix;
use or::OrSuffix;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

use and::AndSuffix;
use lower_word::LowerWordSuffix;
use one::OneSuffix;
use upper_word::UpperWordSuffix;
use xor::XorSuffix;

pub mod and;
pub mod eq;
pub mod lower_word;
pub mod lt;
pub mod one;
pub mod or;
pub mod upper_word;
pub mod xor;

pub trait SparseDenseSuffix: 'static + Sync {
    fn suffix_mle(&self, b: LookupBits) -> u32;
}

#[repr(u8)]
#[derive(EnumCountMacro, EnumIter)]
pub enum Suffixes<const WORD_SIZE: usize> {
    One(OneSuffix),
    And(AndSuffix),
    Xor(XorSuffix),
    Or(OrSuffix),
    UpperWord(UpperWordSuffix<WORD_SIZE>),
    LowerWord(LowerWordSuffix<WORD_SIZE>),
    LessThan(LessThanSuffix),
    Eq(EqSuffix),
}

impl<const WORD_SIZE: usize> SparseDenseSuffix for Suffixes<WORD_SIZE> {
    fn suffix_mle(&self, b: LookupBits) -> u32 {
        match self {
            Suffixes::One(suffix) => SparseDenseSuffix::suffix_mle(suffix, b),
            Suffixes::And(suffix) => SparseDenseSuffix::suffix_mle(suffix, b),
            Suffixes::Or(suffix) => SparseDenseSuffix::suffix_mle(suffix, b),
            Suffixes::Xor(suffix) => SparseDenseSuffix::suffix_mle(suffix, b),
            Suffixes::UpperWord(suffix) => SparseDenseSuffix::suffix_mle(suffix, b),
            Suffixes::LowerWord(suffix) => SparseDenseSuffix::suffix_mle(suffix, b),
            Suffixes::LessThan(suffix) => SparseDenseSuffix::suffix_mle(suffix, b),
            Suffixes::Eq(suffix) => SparseDenseSuffix::suffix_mle(suffix, b),
        }
    }
}
