pub mod inputs;
pub mod outputs;
pub mod verify;

#[cfg(feature = "zk")]
pub use inputs::SpartanZkInputs;
pub use inputs::{deps, SpartanDeps, SpartanInputs};
pub use outputs::SpartanOutput;
#[cfg(feature = "zk")]
pub use outputs::SpartanZkOutput;
pub use verify::verify;
#[cfg(feature = "zk")]
pub use verify::verify_zk;
