mod request;
mod result;

pub use request::{
    OpeningQueryRequest, OpeningRequest, OpeningRlcComponent, OpeningRlcMaterializationRequest,
    OpeningSlot,
};
pub use result::{
    OpeningEvaluationOutput, OpeningProofOutput, OpeningResult, OpeningRlcMaterializationResult,
};
