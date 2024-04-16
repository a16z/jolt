#![cfg_attr(not(feature = "host"), no_std)]

extern crate jolt_sdk_macros;

#[cfg(feature = "host")]
use std::fs::File;
#[cfg(feature = "host")]
use std::path::PathBuf;

use core::{
    alloc::{GlobalAlloc, Layout},
    cell::UnsafeCell,
};

#[cfg(feature = "host")]
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
#[cfg(feature = "host")]
use eyre::Result;

pub use jolt_sdk_macros::provable;
pub use postcard;

#[cfg(feature = "host")]
pub use ark_bn254::{Fr as F, G1Projective as G};
#[cfg(feature = "host")]
pub use ark_ec::CurveGroup;
#[cfg(feature = "host")]
pub use ark_ff::PrimeField;
#[cfg(feature = "host")]
pub use common::{
    constants::MEMORY_OPS_PER_INSTRUCTION,
    rv_trace::{MemoryOp, RV32IM},
};
#[cfg(feature = "host")]
pub use jolt_core::host;
#[cfg(feature = "host")]
pub use jolt_core::jolt::instruction;
#[cfg(feature = "host")]
pub use jolt_core::jolt::vm::{
    bytecode::BytecodeRow,
    rv32i_vm::{RV32IJoltProof, RV32IJoltVM, RV32I},
    Jolt, JoltCommitments, JoltPreprocessing, JoltProof,
};
#[cfg(feature = "host")]
pub use tracer;

#[cfg(feature = "host")]
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct Proof {
    pub proof: RV32IJoltProof<F, G>,
    pub commitments: JoltCommitments<G>,
}

#[cfg(feature = "host")]
impl Proof {
    /// Gets the byte size of the full proof
    pub fn size(&self) -> Result<usize> {
        let mut buffer = Vec::new();
        self.serialize_compressed(&mut buffer)?;
        Ok(buffer.len())
    }

    /// Saves the proof to a file
    pub fn save_to_file<P: Into<PathBuf>>(&self, path: P) -> Result<()> {
        let file = File::create(path.into())?;
        self.serialize_compressed(file)?;
        Ok(())
    }

    /// Reads a proof from a file
    pub fn from_file<P: Into<PathBuf>>(path: P) -> Result<Self> {
        let file = File::open(path.into())?;
        Ok(Proof::deserialize_compressed(file)?)
    }
}

pub struct BumpAllocator {
    offset: UnsafeCell<usize>,
}

unsafe impl Sync for BumpAllocator {}

extern "C" {
    static _HEAP_PTR: u8;
}

fn heap_start() -> usize {
    unsafe { _HEAP_PTR as *const u8 as usize }
}

impl BumpAllocator {
    pub const fn new() -> Self {
        Self {
            offset: UnsafeCell::new(0),
        }
    }

    pub fn free_memory(&self) -> usize {
        heap_start() + (self.offset.get() as usize)
    }
}

unsafe impl GlobalAlloc for BumpAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let alloc_start = align_up(self.free_memory(), layout.align());
        let alloc_end = alloc_start + layout.size();
        *self.offset.get() = alloc_end - self.free_memory();

        alloc_start as *mut u8
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
}

fn align_up(addr: usize, align: usize) -> usize {
    (addr + align - 1) & !(align - 1)
}
