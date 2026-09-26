//! Safe runtime over Metal: open a [`Device`], build a [`ShaderLibrary`] with
//! every pipeline created up front, move data through [`DeviceBuffer`]s, and
//! run [`Batch`]es of dispatches.

mod batch;
mod buffer;
mod device;
mod library;
mod sys;

pub use batch::{Batch, Binding, Grid};
pub use buffer::DeviceBuffer;
pub use device::{Device, DeviceLimits};
pub use library::{host_name, LibrarySpec, MslType, Pipeline, ShaderLibrary};
