pub mod inputs;
pub mod outputs;
mod verify;

pub use inputs::{deps, Deps};
pub use outputs::Stage8Output;
pub use verify::verify;
