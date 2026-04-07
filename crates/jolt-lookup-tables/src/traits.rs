use jolt_field::Field;
use std::fmt::Debug;

use crate::challenge_ops::{ChallengeOps, FieldOps};

pub trait LookupTable: Clone + Debug + Send + Sync {
    fn materialize_entry(&self, index: u128) -> u64;

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>;
}
