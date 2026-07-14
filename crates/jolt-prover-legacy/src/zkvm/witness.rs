use allocative::Allocative;
use common::constants::XLEN;
use common::jolt_device::MemoryLayout;
use rayon::prelude::*;
use tracer::instruction::Cycle;

#[cfg(feature = "prover")]
use crate::curve::JoltCurve;
#[cfg(feature = "prover")]
use crate::poly::commitment::commitment_scheme::StreamingCommitmentScheme;
use crate::zkvm::config::OneHotParams;
use crate::zkvm::instruction::InstructionFlags;
#[cfg(feature = "prover")]
use crate::zkvm::prover::JoltProverPreprocessing;
use crate::{
    field::JoltField,
    poly::{multilinear_polynomial::MultilinearPolynomial, one_hot_polynomial::OneHotPolynomial},
    zkvm::ram::remap_address,
};

use super::instruction::{CircuitFlags, LookupQuery};

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum CommittedPolynomial {
    /*  Twist/Shout witnesses */
    /// Inc polynomial for the registers instance of Twist
    RdInc,
    /// Inc polynomial for the RAM instance of Twist
    RamInc,
    /// One-hot ra polynomial for the instruction lookups instance of Shout.
    /// There are d=8 of these polynomials, `InstructionRa(0) .. InstructionRa(7)`
    InstructionRa(usize),
    /// One-hot ra polynomial for the bytecode instance of Shout
    BytecodeRa(usize),
    /// Dense committed bytecode chunk polynomial for committed program mode.
    BytecodeChunk(usize),
    /// One-hot ra/wa polynomial for the RAM instance of Twist
    /// Note that for RAM, ra and wa are the same polynomial because
    /// there is at most one load or store per cycle.
    RamRa(usize),
    /// Trusted advice polynomial - committed before proving, verifier has commitment.
    /// Length cannot exceed max_trace_length.
    TrustedAdvice,
    /// Untrusted advice polynomial - committed during proving, commitment in proof.
    /// Length cannot exceed max_trace_length.
    UntrustedAdvice,
    /// Program image (initial RAM image) polynomial for committed program mode.
    ProgramImageInit,
    /// One-hot chunk column `j` of the fused unsigned increment stream
    /// (lattice/packed mode only; a slot of the packed witness `W`).
    UnsignedIncChunk(usize),
    /// Boolean msb column of the fused unsigned increment stream
    /// (lattice/packed mode only; a slot of the packed witness `W`).
    UnsignedIncMsb,
    /// One-hot register selector `(chunk, lane)` of the precommitted
    /// bytecode decomposition (lattice/packed mode; a `W_prog` slot). Lane
    /// order is rs1, rs2, rd.
    BytecodeRegisterSelector(usize, usize),
    /// Boolean circuit-flag column `(chunk, flag)` of the precommitted
    /// bytecode decomposition (lattice/packed mode; a `W_prog` slot).
    BytecodeCircuitFlag(usize, usize),
    /// Boolean instruction-flag column `(chunk, flag)` of the precommitted
    /// bytecode decomposition (lattice/packed mode; a `W_prog` slot).
    BytecodeInstructionFlag(usize, usize),
    /// One-hot lookup-table selector of a precommitted bytecode chunk
    /// (lattice/packed mode; a `W_prog` slot).
    BytecodeLookupSelector(usize),
    /// Boolean RAF flag column of a precommitted bytecode chunk
    /// (lattice/packed mode; a `W_prog` slot).
    BytecodeRafFlag(usize),
    /// Byte one-hot decomposition of a chunk's unexpanded PCs
    /// (lattice/packed mode; a `W_prog` slot).
    BytecodeUnexpandedPcBytes(usize),
    /// Byte one-hot decomposition of a chunk's canonical immediate field
    /// bytes (lattice/packed mode; a `W_prog` slot).
    BytecodeImmBytes(usize),
    /// Byte one-hot decomposition of the program image words
    /// (lattice/packed mode; a `W_prog` slot).
    ProgramImageBytes,
}

/// Returns a list of symbols representing all committed polynomials.
pub fn all_committed_polynomials(one_hot_params: &OneHotParams) -> Vec<CommittedPolynomial> {
    let mut polynomials = vec![CommittedPolynomial::RdInc, CommittedPolynomial::RamInc];
    for i in 0..one_hot_params.instruction_d {
        polynomials.push(CommittedPolynomial::InstructionRa(i));
    }
    for i in 0..one_hot_params.ram_d {
        polynomials.push(CommittedPolynomial::RamRa(i));
    }
    for i in 0..one_hot_params.bytecode_d {
        polynomials.push(CommittedPolynomial::BytecodeRa(i));
    }
    polynomials
}

impl CommittedPolynomial {
    /// Generate witness data and compute tier 1 commitment for a single row
    #[cfg(feature = "prover")]
    pub fn stream_witness_and_commit_rows<F, C, PCS>(
        &self,
        preprocessing: &JoltProverPreprocessing<F, C, PCS>,
        row_cycles: &[tracer::instruction::Cycle],
        one_hot_params: &OneHotParams,
    ) -> <PCS as StreamingCommitmentScheme>::ChunkState
    where
        F: JoltField,
        C: JoltCurve<F = F>,
        PCS: StreamingCommitmentScheme<Field = F>,
    {
        match self {
            CommittedPolynomial::RdInc => {
                let row: Vec<i128> = row_cycles
                    .iter()
                    .map(|cycle| {
                        let (_, pre_value, post_value) = cycle.rd_write().unwrap_or_default();
                        post_value as i128 - pre_value as i128
                    })
                    .collect();
                PCS::process_chunk(&preprocessing.generators, &row)
            }
            CommittedPolynomial::RamInc => {
                let row: Vec<i128> = row_cycles
                    .iter()
                    .map(|cycle| match cycle.ram_access() {
                        tracer::instruction::RAMAccess::Write(write) => {
                            write.post_value as i128 - write.pre_value as i128
                        }
                        _ => 0,
                    })
                    .collect();
                PCS::process_chunk(&preprocessing.generators, &row)
            }
            CommittedPolynomial::InstructionRa(idx) => {
                let row: Vec<Option<usize>> = row_cycles
                    .iter()
                    .map(|cycle| {
                        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                        Some(one_hot_params.lookup_index_chunk(lookup_index, *idx) as usize)
                    })
                    .collect();
                PCS::process_chunk_onehot(&preprocessing.generators, one_hot_params.k_chunk, &row)
            }
            CommittedPolynomial::BytecodeRa(idx) => {
                let row: Vec<Option<usize>> = row_cycles
                    .iter()
                    .map(|cycle| {
                        let pc = preprocessing.materialized_program().get_pc(cycle);
                        Some(one_hot_params.bytecode_pc_chunk(pc, *idx) as usize)
                    })
                    .collect();
                PCS::process_chunk_onehot(&preprocessing.generators, one_hot_params.k_chunk, &row)
            }
            CommittedPolynomial::RamRa(idx) => {
                let row: Vec<Option<usize>> = row_cycles
                    .iter()
                    .map(|cycle| {
                        remap_address(
                            cycle.ram_access().address() as u64,
                            &preprocessing.shared.memory_layout,
                        )
                        .map(|address| one_hot_params.ram_address_chunk(address, *idx) as usize)
                    })
                    .collect();
                PCS::process_chunk_onehot(&preprocessing.generators, one_hot_params.k_chunk, &row)
            }
            CommittedPolynomial::TrustedAdvice
            | CommittedPolynomial::UntrustedAdvice
            | CommittedPolynomial::ProgramImageInit
            | CommittedPolynomial::BytecodeChunk(_) => {
                panic!("Precommitted polynomials should not use streaming witness generation")
            }
            CommittedPolynomial::BytecodeRegisterSelector(..)
            | CommittedPolynomial::BytecodeCircuitFlag(..)
            | CommittedPolynomial::BytecodeInstructionFlag(..)
            | CommittedPolynomial::BytecodeLookupSelector(_)
            | CommittedPolynomial::BytecodeRafFlag(_)
            | CommittedPolynomial::BytecodeUnexpandedPcBytes(_)
            | CommittedPolynomial::BytecodeImmBytes(_)
            | CommittedPolynomial::ProgramImageBytes
            | CommittedPolynomial::UnsignedIncChunk(_)
            | CommittedPolynomial::UnsignedIncMsb => {
                panic!("Lattice columns commit through the packed witness, not per-polynomial streaming")
            }
        }
    }

    #[tracing::instrument(skip_all, name = "CommittedPolynomial::generate_witness")]
    pub fn generate_witness<F>(
        &self,
        bytecode_preprocessing: &crate::zkvm::bytecode::BytecodePreprocessing,
        memory_layout: &MemoryLayout,
        trace: &[Cycle],
        one_hot_params: Option<&OneHotParams>,
    ) -> MultilinearPolynomial<F>
    where
        F: JoltField,
    {
        match self {
            CommittedPolynomial::BytecodeRa(i) => {
                let one_hot_params = one_hot_params.unwrap();
                let addresses: Vec<_> = trace
                    .par_iter()
                    .map(|cycle| {
                        let pc =
                            crate::zkvm::bytecode::get_pc_for_cycle(bytecode_preprocessing, cycle);
                        Some(one_hot_params.bytecode_pc_chunk(pc, *i))
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    one_hot_params.k_chunk,
                ))
            }
            CommittedPolynomial::RamRa(i) => {
                let one_hot_params = one_hot_params.unwrap();
                let addresses: Vec<_> = trace
                    .par_iter()
                    .map(|cycle| {
                        remap_address(cycle.ram_access().address() as u64, memory_layout)
                            .map(|address| one_hot_params.ram_address_chunk(address, *i))
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    one_hot_params.k_chunk,
                ))
            }
            CommittedPolynomial::RdInc => {
                let coeffs: Vec<i128> = trace
                    .par_iter()
                    .map(|cycle| {
                        let (_, pre_value, post_value) = cycle.rd_write().unwrap_or_default();
                        post_value as i128 - pre_value as i128
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomial::RamInc => {
                let coeffs: Vec<i128> = trace
                    .par_iter()
                    .map(|cycle| {
                        let ram_op = cycle.ram_access();
                        match ram_op {
                            tracer::instruction::RAMAccess::Write(write) => {
                                write.post_value as i128 - write.pre_value as i128
                            }
                            _ => 0,
                        }
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomial::InstructionRa(i) => {
                let one_hot_params = one_hot_params.unwrap();
                let addresses: Vec<_> = trace
                    .par_iter()
                    .map(|cycle| {
                        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                        Some(one_hot_params.lookup_index_chunk(lookup_index, *i))
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    one_hot_params.k_chunk,
                ))
            }
            #[cfg(feature = "prover")]
            CommittedPolynomial::UnsignedIncChunk(i) => {
                let one_hot_params = one_hot_params.unwrap();
                let width = one_hot_params.log_k_chunk;
                let addresses: Vec<_> = trace
                    .par_iter()
                    .map(|cycle| {
                        Some(
                            crate::zkvm::packed_witness::FusedIncCycle::from_cycle(cycle)
                                .chunk_symbol_bits(width, *i) as u8,
                        )
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    one_hot_params.k_chunk,
                ))
            }
            #[cfg(feature = "prover")]
            CommittedPolynomial::UnsignedIncMsb => {
                let coeffs: Vec<u8> = trace
                    .par_iter()
                    .map(|cycle| {
                        u8::from(
                            crate::zkvm::packed_witness::FusedIncCycle::from_cycle(cycle).msb(),
                        )
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomial::TrustedAdvice
            | CommittedPolynomial::UntrustedAdvice
            | CommittedPolynomial::ProgramImageInit
            | CommittedPolynomial::BytecodeChunk(_) => {
                panic!("Precommitted polynomials should not use generate_witness")
            }
            #[cfg(feature = "prover")]
            CommittedPolynomial::BytecodeRegisterSelector(..)
            | CommittedPolynomial::BytecodeCircuitFlag(..)
            | CommittedPolynomial::BytecodeInstructionFlag(..)
            | CommittedPolynomial::BytecodeLookupSelector(_)
            | CommittedPolynomial::BytecodeRafFlag(_)
            | CommittedPolynomial::BytecodeUnexpandedPcBytes(_)
            | CommittedPolynomial::BytecodeImmBytes(_)
            | CommittedPolynomial::ProgramImageBytes => {
                panic!("W_prog sub-columns commit through the packed precommitted witness, not generate_witness")
            }
            #[cfg(not(feature = "prover"))]
            CommittedPolynomial::BytecodeRegisterSelector(..)
            | CommittedPolynomial::BytecodeCircuitFlag(..)
            | CommittedPolynomial::BytecodeInstructionFlag(..)
            | CommittedPolynomial::BytecodeLookupSelector(_)
            | CommittedPolynomial::BytecodeRafFlag(_)
            | CommittedPolynomial::BytecodeUnexpandedPcBytes(_)
            | CommittedPolynomial::BytecodeImmBytes(_)
            | CommittedPolynomial::ProgramImageBytes
            | CommittedPolynomial::UnsignedIncChunk(_)
            | CommittedPolynomial::UnsignedIncMsb => {
                panic!("Lattice columns require the prover feature")
            }
        }
    }

    pub fn get_onehot_k(&self, one_hot_params: &OneHotParams) -> Option<usize> {
        match self {
            CommittedPolynomial::InstructionRa(_)
            | CommittedPolynomial::BytecodeRa(_)
            | CommittedPolynomial::RamRa(_)
            | CommittedPolynomial::UnsignedIncChunk(_) => Some(one_hot_params.k_chunk),
            _ => None,
        }
    }
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum VirtualPolynomial {
    PC,
    UnexpandedPC,
    NextPC,
    NextUnexpandedPC,
    NextIsNoop,
    NextIsVirtual,
    NextIsFirstInSequence,
    LeftLookupOperand,
    RightLookupOperand,
    LeftInstructionInput,
    RightInstructionInput,
    Product,
    ShouldJump,
    ShouldBranch,
    Rd,
    Imm,
    Rs1Value,
    Rs2Value,
    RdWriteValue,
    Rs1Ra,
    Rs2Ra,
    RdWa,
    LookupOutput,
    InstructionRaf,
    InstructionRafFlag,
    InstructionRa(usize),
    RegistersVal,
    RamAddress,
    RamRa,
    RamReadValue,
    RamWriteValue,
    RamVal,
    RamValInit,
    RamValFinal,
    RamHammingWeight,
    UnivariateSkip,
    OpFlags(CircuitFlags),
    InstructionFlags(InstructionFlags),
    LookupTableFlag(usize),
    BytecodeValStage(usize),
    BytecodeReadRafAddrClaim,
    BooleanityAddrClaim,
    BytecodeClaimReductionIntermediate,
    ProgramImageInitContributionRw,
    /// The gamma-batched RamInc/RdInc stream of the lattice mode; its
    /// destination selector is the existing `OpFlags(Store)` flag.
    FusedInc,
}
