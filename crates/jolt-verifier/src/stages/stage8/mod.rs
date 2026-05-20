pub mod inputs;
pub mod outputs;
mod verify;

pub use inputs::{deps, Deps};
pub use outputs::{Stage8ClearOutput, Stage8Output, Stage8ZkOutput};
pub use verify::verify;
