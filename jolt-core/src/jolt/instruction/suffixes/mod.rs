use std::{fmt::Display, ops::Index};

use crate::{field::JoltField, subprotocols::sparse_dense_shout::LookupBits};
use eq::EqSuffix;
use lt::LessThanSuffix;
use num_derive::FromPrimitive;
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
    fn suffix_mle(b: LookupBits) -> u32;
}

#[repr(u8)]
#[derive(EnumCountMacro, EnumIter, FromPrimitive)]
pub enum Suffixes {
    One,
    And,
    Xor,
    Or,
    UpperWord,
    LowerWord,
    LessThan,
    Eq,
}

#[derive(Clone, Copy)]
pub struct SuffixEval<F>(F);
pub type SuffixCheckpoint<F: JoltField> = SuffixEval<Option<F>>;

impl<F: Display> Display for SuffixEval<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<F> From<F> for SuffixEval<F> {
    fn from(value: F) -> Self {
        Self(value)
    }
}

impl<F> SuffixCheckpoint<F> {
    pub fn unwrap(self) -> SuffixEval<F> {
        self.0.unwrap().into()
    }
}

impl<F> Index<Suffixes> for &[SuffixEval<F>] {
    type Output = F;

    fn index(&self, prefix: Suffixes) -> &Self::Output {
        let index = prefix as usize;
        &self.get(index).unwrap().0
    }
}

impl Suffixes {
    pub fn suffix_mle<const WORD_SIZE: usize>(&self, b: LookupBits) -> u32 {
        match self {
            Suffixes::One => OneSuffix::suffix_mle(b),
            Suffixes::And => AndSuffix::suffix_mle(b),
            Suffixes::Or => OrSuffix::suffix_mle(b),
            Suffixes::Xor => XorSuffix::suffix_mle(b),
            Suffixes::UpperWord => UpperWordSuffix::<WORD_SIZE>::suffix_mle(b),
            Suffixes::LowerWord => LowerWordSuffix::<WORD_SIZE>::suffix_mle(b),
            Suffixes::LessThan => LessThanSuffix::suffix_mle(b),
            Suffixes::Eq => EqSuffix::suffix_mle(b),
        }
    }
}
