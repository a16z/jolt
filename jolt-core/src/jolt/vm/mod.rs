use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use merlin::Transcript;
use std::any::TypeId;
use strum::{EnumCount, IntoEnumIterator};

use crate::jolt::{
    instruction::{sltu::SLTUInstruction, JoltInstruction, Opcode},
    subtable::LassoSubtable,
};
use crate::poly::structured_poly::BatchablePolynomials;
use crate::utils::{errors::ProofVerifyError, random::RandomTape};
use crate::{
    lasso::{
        memory_checking::{MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier},
        surge::{Surge, SurgeProof},
    },
    utils::math::Math,
};
use common::ELFInstruction;

use self::bytecode::{
    BytecodeCommitment, BytecodeInitFinalOpenings, BytecodePolynomials, BytecodeProof,
    BytecodeReadWriteOpenings, ELFRow,
};
use self::instruction_lookups::{InstructionLookups, InstructionLookupsProof};
use self::read_write_memory::{
    MemoryCommitment, MemoryInitFinalOpenings, MemoryOp, MemoryReadWriteOpenings, ReadWriteMemory,
    ReadWriteMemoryProof,
};

struct JoltProof<F: PrimeField, G: CurveGroup<ScalarField = F>> {
    instruction_lookups: InstructionLookupsProof<F, G>,
    read_write_memory: ReadWriteMemoryProof<F, G>,
    bytecode: BytecodeProof<F, G>,
    // TODO: r1cs
}

pub trait Jolt<F: PrimeField, G: CurveGroup<ScalarField = F>, const C: usize, const M: usize> {
    type InstructionSet: JoltInstruction + Opcode + IntoEnumIterator + EnumCount;
    type Subtables: LassoSubtable<F> + IntoEnumIterator + EnumCount + From<TypeId> + Into<usize>;

    fn prove(
        bytecode: Vec<ELFInstruction>,
        mut bytecode_trace: Vec<ELFRow>,
        memory_trace: Vec<MemoryOp>,
        instructions: Vec<Self::InstructionSet>,
    ) -> JoltProof<F, G> {
        let mut transcript = Transcript::new(b"Jolt transcript");
        let mut random_tape = RandomTape::new(b"Jolt prover randomness");
        let bytecode_proof =
            Self::prove_bytecode(&bytecode, bytecode_trace, &mut transcript, &mut random_tape);
        let memory_proof =
            Self::prove_memory(bytecode, memory_trace, &mut transcript, &mut random_tape);
        let instruction_lookups =
            Self::prove_instruction_lookups(instructions, &mut transcript, &mut random_tape);
        todo!("rics");
        JoltProof {
            instruction_lookups,
            read_write_memory: memory_proof,
            bytecode: bytecode_proof,
        }
    }

    fn verify(proof: JoltProof<F, G>) -> Result<(), ProofVerifyError> {
        let mut transcript = Transcript::new(b"Jolt transcript");
        Self::verify_bytecode(proof.bytecode, &mut transcript)?;
        Self::verify_memory(proof.read_write_memory, &mut transcript)?;
        Self::verify_instruction_lookups(proof.instruction_lookups, &mut transcript)?;
        todo!("r1cs");
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
        bytecode: &Vec<ELFInstruction>,
        mut trace: Vec<ELFRow>,
        transcript: &mut Transcript,
        random_tape: &mut RandomTape<G>,
    ) -> BytecodeProof<F, G> {
        let mut bytecode_rows = bytecode.iter().map(ELFRow::from).collect();
        let polys: BytecodePolynomials<F, G> = BytecodePolynomials::new(bytecode_rows, trace);
        let batched_polys = polys.batch();
        let commitment = BytecodePolynomials::commit(&batched_polys);

        let memory_checking_proof = polys.prove_memory_checking(
            &polys,
            &batched_polys,
            &commitment,
            transcript,
            random_tape,
        );
        BytecodeProof {
            memory_checking_proof,
            commitment,
        }
    }

    fn verify_bytecode(
        proof: BytecodeProof<F, G>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        BytecodePolynomials::verify_memory_checking(
            proof.memory_checking_proof,
            &proof.commitment,
            transcript,
        )
    }

    fn prove_memory(
        bytecode: Vec<ELFInstruction>,
        memory_trace: Vec<MemoryOp>,
        transcript: &mut Transcript,
        random_tape: &mut RandomTape<G>,
    ) -> ReadWriteMemoryProof<F, G> {
        const MAX_TRACE_SIZE: usize = 1 << 22;
        // TODO: Support longer traces
        assert!(memory_trace.len() <= MAX_TRACE_SIZE);

        todo!("Load program bytecode into memory");

        let (memory, read_timestamps) = ReadWriteMemory::new(bytecode, memory_trace, transcript);
        let batched_polys = memory.batch();
        let commitment: MemoryCommitment<G> = ReadWriteMemory::commit(&batched_polys);

        let memory_checking_proof = memory.prove_memory_checking(
            &memory,
            &batched_polys,
            &commitment,
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

        ReadWriteMemoryProof {
            memory_checking_proof,
            commitment,
            timestamp_validity_proof,
        }
    }

    fn verify_memory(
        proof: ReadWriteMemoryProof<F, G>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        const MAX_TRACE_SIZE: usize = 1 << 22;
        ReadWriteMemory::verify_memory_checking(
            proof.memory_checking_proof,
            &proof.commitment,
            transcript,
        )?;
        <Surge<F, G, SLTUInstruction, 2, MAX_TRACE_SIZE>>::verify(
            proof.timestamp_validity_proof,
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
