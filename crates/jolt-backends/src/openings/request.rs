use jolt_field::Field;
use jolt_witness::{OracleRef, WitnessNamespace};

use crate::BackendValueSlot;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct OpeningSlot(pub u32);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpeningQueryRequest<F: Field, N: WitnessNamespace> {
    pub slot: OpeningSlot,
    pub oracle: OracleRef<N>,
    pub point: Vec<F>,
    pub eval: BackendValueSlot,
    pub scalar: F,
    pub use_opening_hint: bool,
}

impl<F: Field, N: WitnessNamespace> OpeningQueryRequest<F, N> {
    pub fn new(
        slot: OpeningSlot,
        oracle: OracleRef<N>,
        point: Vec<F>,
        eval: BackendValueSlot,
        scalar: F,
        use_opening_hint: bool,
    ) -> Self {
        Self {
            slot,
            oracle,
            point,
            eval,
            scalar,
            use_opening_hint,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpeningRequest<F: Field, N: WitnessNamespace> {
    pub label: &'static str,
    pub queries: Vec<OpeningQueryRequest<F, N>>,
    pub joint_claim: BackendValueSlot,
    #[cfg(feature = "zk")]
    pub hiding_eval: bool,
}

impl<F: Field, N: WitnessNamespace> OpeningRequest<F, N> {
    pub const fn new(
        label: &'static str,
        queries: Vec<OpeningQueryRequest<F, N>>,
        joint_claim: BackendValueSlot,
    ) -> Self {
        Self {
            label,
            queries,
            joint_claim,
            #[cfg(feature = "zk")]
            hiding_eval: false,
        }
    }
}
