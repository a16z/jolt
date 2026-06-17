pub mod inputs;
pub mod outputs;
mod verify;

pub use inputs::{deps, Deps};
pub use outputs::{
    Stage8BatchStatement, Stage8ClearBatchStatement, Stage8ClearOutput, Stage8OpeningId,
    Stage8OpeningStatement, Stage8Output, Stage8ZkBatchStatement, Stage8ZkOutput,
};
pub use verify::{batch_statement, verify};
