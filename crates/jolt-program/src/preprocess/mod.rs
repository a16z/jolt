pub mod bytecode;
pub mod error;
pub mod program;
pub mod ram;

pub use bytecode::{BytecodePCMapper, BytecodePreprocessing};
pub use error::PreprocessingError;
pub use program::JoltProgramPreprocessing;
pub use ram::RAMPreprocessing;
