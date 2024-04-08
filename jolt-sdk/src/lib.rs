#![cfg_attr(not(feature = "std"), no_std)]

extern crate jolt_sdk_macros;

use core::{alloc::{GlobalAlloc, Layout}, cell::UnsafeCell, ptr, slice::SliceIndex};
#[cfg(feature = "std")]
use std::fs::File;
#[cfg(feature = "std")]
use std::path::PathBuf;

#[cfg(feature = "std")]
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
#[cfg(feature = "std")]
use eyre::Result;

pub use jolt_sdk_macros::provable;
pub use postcard;

#[cfg(feature = "std")]
pub use ark_ec::CurveGroup;
#[cfg(feature = "std")]
pub use ark_ff::PrimeField;
#[cfg(feature = "std")]
pub use ark_bn254::{Fr as F, G1Projective as G};
#[cfg(feature = "std")]
pub use common::{constants::MEMORY_OPS_PER_INSTRUCTION, rv_trace::{MemoryOp, RV32IM}};
#[cfg(feature = "std")]
pub use liblasso::host;
#[cfg(feature = "std")]
pub use liblasso::jolt::instruction;
#[cfg(feature = "std")]
pub use liblasso::jolt::vm::{
    bytecode::BytecodeRow,
    rv32i_vm::{RV32IJoltProof, RV32IJoltVM, RV32I},
    Jolt, JoltCommitments, JoltPreprocessing, JoltProof,
};
#[cfg(feature = "std")]
pub use tracer;

#[cfg(feature = "std")]
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct Proof {
    pub proof: RV32IJoltProof<F, G>,
    pub commitments: JoltCommitments<G>,
}

#[cfg(feature = "std")]
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
    free_memory: UnsafeCell<usize>,
}
 
unsafe impl Sync for BumpAllocator {}

impl BumpAllocator {
    pub const fn new() -> Self {
        Self {
            free_memory: UnsafeCell::new(0),
        }
    }

    pub fn init(&self, heap_start: usize) {
        unsafe {
            *self.free_memory.get() = heap_start;
        }
    }

    pub fn free_memory(&self) -> usize {
        self.free_memory.get() as usize
    }
}

unsafe impl GlobalAlloc for BumpAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let alloc_start = align_up(*self.free_memory.get(), layout.align());
        let alloc_end = alloc_start + layout.size();
        *self.free_memory.get() = alloc_end;

        alloc_start as *mut u8
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        
    }
}

fn align_up(addr: usize, align: usize) -> usize {
    let remainder = addr % align;
    if remainder == 0 {
        addr // addr already aligned
    } else {
        addr - remainder + align
    }
}
