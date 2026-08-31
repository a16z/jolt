//! Shared Product and Instruction claim-reduction round service.

mod runtime;

pub(super) const SOURCE: &str = include_str!("shader.metal");

pub(crate) use runtime::{
    PendingProductInstructionInitialMessage, ProductInstructionOpenings,
    ProductInstructionRoundService, ProductInstructionRoundStats,
};
