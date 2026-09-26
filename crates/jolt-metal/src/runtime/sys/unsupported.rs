//! Non-macOS backend: no device can be constructed, so every other type is
//! uninhabited and its methods are statically unreachable.

use std::env::consts::OS;
use std::time::Duration;

use crate::error::MetalError;
use crate::runtime::batch::Binding;
use crate::runtime::device::DeviceInfo;
use crate::runtime::library::PipelineInfo;

pub(crate) enum RawDevice {}

impl RawDevice {
    pub(crate) fn system_default() -> Result<(Self, DeviceInfo), MetalError> {
        Err(MetalError::Unavailable {
            reason: format!("Metal requires macOS; this is {OS}"),
        })
    }

    pub(crate) fn allocated_bytes(&self) -> Result<usize, MetalError> {
        match *self {}
    }

    pub(crate) fn new_zeroed_buffer(&self, _bytes: usize) -> Result<RawBuffer, MetalError> {
        match *self {}
    }

    pub(crate) fn new_buffer_with_bytes(&self, _data: &[u8]) -> Result<RawBuffer, MetalError> {
        match *self {}
    }

    pub(crate) fn compile(&self, _source: &str) -> Result<RawLibrary, MetalError> {
        match *self {}
    }

    pub(crate) fn command_batch(&self) -> Result<RawCommandBatch, MetalError> {
        match *self {}
    }
}

pub(crate) enum RawLibrary {}

impl RawLibrary {
    pub(crate) fn pipeline(
        &self,
        _device: &RawDevice,
        _kernel: &str,
    ) -> Result<(RawPipeline, PipelineInfo), MetalError> {
        match *self {}
    }
}

pub(crate) enum RawPipeline {}

pub(crate) enum RawBuffer {}

impl RawBuffer {
    pub(crate) fn host_bytes(&mut self, _len: usize) -> Option<&[u8]> {
        match *self {}
    }
}

pub(crate) enum RawCommandBatch {}

impl RawCommandBatch {
    pub(crate) fn dispatch(
        &mut self,
        _pipeline: &RawPipeline,
        _bindings: &[Binding<'_>],
        _threads: usize,
        _threads_per_threadgroup: usize,
    ) -> Result<(), MetalError> {
        match *self {}
    }

    pub(crate) fn commit_and_wait(self) -> Result<Duration, MetalError> {
        match self {}
    }
}
