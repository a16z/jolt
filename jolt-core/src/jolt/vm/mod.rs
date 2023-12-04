use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use merlin::Transcript;
use std::any::TypeId;
use strum::{EnumCount, IntoEnumIterator};

use crate::{
    lasso::{
        memory_checking::{MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier},
        surge::{Surge, SurgeProof},
    },
    utils::math::Math,
};

use crate::jolt::{
    instruction::{sltu::SLTUInstruction, JoltInstruction, Opcode},
    subtable::LassoSubtable,
};
use crate::poly::structured_poly::BatchablePolynomials;
use crate::utils::{errors::ProofVerifyError, random::RandomTape};

use self::bytecode::{
    BytecodeCommitment, BytecodeInitFinalOpenings, BytecodePolynomials, BytecodeReadWriteOpenings,
    ELFRow,
};
use self::instruction_lookups::{InstructionLookups, InstructionLookupsProof};
use self::read_write_memory::{
    MemoryCommitment, MemoryInitFinalOpenings, MemoryOp, MemoryReadWriteOpenings, ReadWriteMemory,
};

struct JoltProof<F: PrimeField, G: CurveGroup<ScalarField = F>> {
    instruction_lookups: InstructionLookupsProof<F, G>,
}

pub trait Jolt<F: PrimeField, G: CurveGroup<ScalarField = F>, const C: usize, const M: usize> {
    type InstructionSet: JoltInstruction + Opcode + IntoEnumIterator + EnumCount;
    type Subtables: LassoSubtable<F> + IntoEnumIterator + EnumCount + From<TypeId> + Into<usize>;

    fn prove() {
        // preprocess?
        // emulate
        // prove_bytecode
        // prove_memory
        // prove_lookups
        // prove_r1cs
        unimplemented!("todo");
    }

    fn prove_instruction_lookups(
        ops: Vec<Self::InstructionSet>,
        transcript: &mut Transcript,
        random_tape: &mut RandomTape<G>,
    ) -> InstructionLookupsProof<F, G> {
        let instruction_lookups =
            InstructionLookups::<F, G, Self::InstructionSet, Self::Subtables, C, M>::new(ops);
        instruction_lookups.prove_lookups(transcript, random_tape)
    }

    fn verify_instruction_lookups(
        proof: InstructionLookupsProof<F, G>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        InstructionLookups::<F, G, Self::InstructionSet, Self::Subtables, C, M>::verify(
            proof, transcript,
        )
    }

    fn prove_bytecode(
        mut bytecode: Vec<ELFRow>,
        mut trace: Vec<ELFRow>,
        transcript: &mut Transcript,
        random_tape: &mut RandomTape<G>,
    ) -> (
        MemoryCheckingProof<
            G,
            BytecodePolynomials<F, G>,
            BytecodeReadWriteOpenings<F, G>,
            BytecodeInitFinalOpenings<F, G>,
        >,
        BytecodeCommitment<G>,
    ) {
        let polys: BytecodePolynomials<F, G> = BytecodePolynomials::new(bytecode, trace);
        let batched_polys = polys.batch();
        let commitments = BytecodePolynomials::commit(&batched_polys);

        (
            polys.prove_memory_checking(
                &polys,
                &batched_polys,
                &commitments,
                transcript,
                random_tape,
            ),
            commitments,
        )
    }

    fn verify_bytecode(
        proof: MemoryCheckingProof<
            G,
            BytecodePolynomials<F, G>,
            BytecodeReadWriteOpenings<F, G>,
            BytecodeInitFinalOpenings<F, G>,
        >,
        commitment: BytecodeCommitment<G>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        BytecodePolynomials::verify_memory_checking(proof, &commitment, transcript)
    }

    fn prove_memory(
        memory_trace: Vec<MemoryOp>,
        memory_size: usize,
        transcript: &mut Transcript,
        random_tape: &mut RandomTape<G>,
    ) -> (
        MemoryCheckingProof<
            G,
            ReadWriteMemory<F, G>,
            MemoryReadWriteOpenings<F, G>,
            MemoryInitFinalOpenings<F, G>,
        >,
        MemoryCommitment<G>,
        SurgeProof<F, G>,
    ) {
        const MAX_TRACE_SIZE: usize = 1 << 22;
        // TODO: Support longer traces
        assert!(memory_trace.len() <= MAX_TRACE_SIZE);

        todo!("Load program bytecode into memory");

        let (memory, read_timestamps) = ReadWriteMemory::new(memory_trace, memory_size, transcript);
        let batched_polys = memory.batch();
        let commitments: MemoryCommitment<G> = ReadWriteMemory::commit(&batched_polys);

        let memory_checking_proof = memory.prove_memory_checking(
            &memory,
            &batched_polys,
            &commitments,
            transcript,
            random_tape,
        );

        let timestamp_validity_lookups: Vec<SLTUInstruction> = read_timestamps
            .iter()
            .enumerate()
            .map(|(i, &ts)| SLTUInstruction(ts, i as u64 + 1))
            .collect();

        let timestamp_validity_proof =
            <Surge<F, G, SLTUInstruction, 2, MAX_TRACE_SIZE>>::new(timestamp_validity_lookups)
                .prove(transcript);

        (memory_checking_proof, commitments, timestamp_validity_proof)
    }

    fn verify_memory(
        memory_checking_proof: MemoryCheckingProof<
            G,
            ReadWriteMemory<F, G>,
            MemoryReadWriteOpenings<F, G>,
            MemoryInitFinalOpenings<F, G>,
        >,
        commitment: MemoryCommitment<G>,
        transcript: &mut Transcript,
        timestamp_validity_proof: SurgeProof<F, G>,
    ) -> Result<(), ProofVerifyError> {
        const MAX_TRACE_SIZE: usize = 1 << 22;
        ReadWriteMemory::verify_memory_checking(memory_checking_proof, &commitment, transcript)?;
        <Surge<F, G, SLTUInstruction, 2, MAX_TRACE_SIZE>>::verify(
            timestamp_validity_proof,
            transcript,
        )
    }

    fn prove_r1cs() {
        unimplemented!("todo")
    }
}

pub mod bytecode;
pub mod instruction_lookups;
pub mod read_write_memory;
pub mod rv32i_vm;
