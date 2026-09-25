//! Non-macOS backend: no device can be constructed, so every other type is
//! uninhabited and its methods are statically unreachable.

use crate::error::MetalError;
use crate::runtime::batch::Binding;
use crate::runtime::device::DeviceInfo;
use crate::runtime::library::PipelineInfo;

pub(crate) enum Device {}

impl Device {
    pub(crate) fn system_default() -> Result<(Self, DeviceInfo), MetalError> {
        Err(MetalError::Unavailable {
            reason: format!("Metal requires macOS; this is {}", std::env::consts::OS),
        })
    }

    pub(crate) fn allocated_bytes(&self) -> Result<usize, MetalError> {
        match *self {}
    }

    pub(crate) fn new_zeroed_buffer(&self, _bytes: usize) -> Result<Buffer, MetalError> {
        match *self {}
    }

    pub(crate) fn new_buffer_with_bytes(&self, _data: &[u8]) -> Result<Buffer, MetalError> {
        match *self {}
    }

    pub(crate) fn compile(&self, _source: &str) -> Result<Library, MetalError> {
        match *self {}
    }

    pub(crate) fn command_batch(&self) -> Result<CommandBatch, MetalError> {
        match *self {}
    }
}

pub(crate) enum Library {}

impl Library {
    pub(crate) fn pipeline(
        &self,
        _device: &Device,
        _kernel: &str,
    ) -> Result<(Pipeline, PipelineInfo), MetalError> {
        match *self {}
    }
}

pub(crate) enum Pipeline {}

pub(crate) enum Buffer {}

impl Buffer {
    pub(crate) fn host_bytes(&mut self, _len: usize) -> Option<&[u8]> {
        match *self {}
    }
}

pub(crate) enum CommandBatch {}

impl CommandBatch {
    pub(crate) fn dispatch(
        &mut self,
        _pipeline: &Pipeline,
        _bindings: &[Binding<'_>],
        _threads: usize,
        _threads_per_threadgroup: usize,
    ) -> Result<(), MetalError> {
        match *self {}
    }

    pub(crate) fn commit_and_wait(self) -> Result<(), MetalError> {
        match self {}
    }
}
