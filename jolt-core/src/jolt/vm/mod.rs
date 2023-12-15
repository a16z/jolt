use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use merlin::Transcript;
use std::any::TypeId;
use strum::{EnumCount, IntoEnumIterator};
use ark_std::log2;

use crate::r1cs::snark::{JoltCircuit, run_jolt_spartan_with_circuit};
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
        let mut bytecode_rows = bytecode.iter().map(ELFRow::from).collect();
        let bytecode_proof = Self::prove_bytecode(
            bytecode_rows,
            bytecode_trace,
            &mut transcript,
            &mut random_tape,
        );
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

    #[tracing::instrument(skip_all, name = "Jolt::prove_instruction_lookups")]
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

    #[tracing::instrument(skip_all, name = "Jolt::prove_bytecode")]
    fn prove_bytecode(
        mut bytecode_rows: Vec<ELFRow>,
        mut trace: Vec<ELFRow>,
        transcript: &mut Transcript,
        random_tape: &mut RandomTape<G>,
    ) -> BytecodeProof<F, G> {
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

    #[tracing::instrument(skip_all, name = "Jolt::prove_memory")]
    fn prove_memory(
        bytecode: Vec<ELFInstruction>,
        memory_trace: Vec<MemoryOp>,
        transcript: &mut Transcript,
        random_tape: &mut RandomTape<G>,
    ) -> ReadWriteMemoryProof<F, G> {
        const MAX_TRACE_SIZE: usize = 1 << 22;
        // TODO: Support longer traces
        assert!(memory_trace.len() <= MAX_TRACE_SIZE);

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

    fn prove_r1cs(
        instructions: Vec<Self::InstructionSet>,
        mut bytecode_rows: Vec<ELFRow>,
        mut trace: Vec<ELFRow>,
        bytecode: Vec<ELFInstruction>,
        memory_trace: Vec<MemoryOp>,
        circuit_flags: Vec<F>,
        transcript: &mut Transcript,
        random_tape: &mut RandomTape<G>,
    ) {
        let log_M = log2(M) as usize;
        let [prog_a_rw, prog_v_rw, prog_t_reads] = 
            BytecodePolynomials::<F, G>::r1cs_polys_from_bytecode(bytecode_rows, trace);

        let [memreg_a_rw, memreg_v_reads, memreg_v_writes, memreg_t_reads] = ReadWriteMemory::<F, G>::get_r1cs_polys(bytecode, memory_trace, transcript);

        let (chunks_x, chunks_y): (Vec<F>, Vec<F>) = 
            instructions
            .iter()
            .flat_map(|op| {
                let chunks_xy = op.operand_chunks(C, log_M);
                let chunks_x = chunks_xy[0].clone();
                let chunks_y = chunks_xy[1].clone();
                chunks_x.into_iter().zip(chunks_y.into_iter())
            })
            .map(|(x, y)| (F::from(x as u64), F::from(y as u64)))
            .unzip();

        let chunks_query = instructions.iter()
            .flat_map(|op| op.to_indices(C, log_M))
            .map(|x| x as u64)
            .map(F::from)
            .collect::<Vec<F>>();

        let lookup_outputs = instructions.iter().map(|op| op.lookup_entry::<F>(C, M)).collect::<Vec<F>>();

        let inputs = vec![
            prog_a_rw,
            prog_v_rw,
            prog_t_reads,
            memreg_a_rw,
            memreg_v_reads, 
            memreg_v_writes,
            memreg_t_reads, 
            chunks_x, 
            chunks_y,
            chunks_query,
            lookup_outputs,
            circuit_flags,
        ];

        // TODO: move this conversion to the r1cs module 
        use spartan2::provider::bn256_grumpkin::bn256;
        use bn256::Scalar as Spartan2Fr;
        type G1 = bn256::Point;
        use common::field_conversion::ark_to_spartan; 
        type EE = spartan2::provider::ipa_pc::EvaluationEngine<G1>;
        type S = spartan2::spartan::snark::RelaxedR1CSSNARK<G1, EE>;

        let inputs_ff = inputs
            .into_iter()
            .map(|input| input
                .into_iter()
                .map(|x| ark_to_spartan(x))
                .collect::<Vec<Spartan2Fr>>()
            ).collect::<Vec<Vec<Spartan2Fr>>>();

        let jolt_circuit = JoltCircuit::<Spartan2Fr>::new_from_inputs(32, C, inputs_ff);
        let result_verify = run_jolt_spartan_with_circuit::<G1, S>(jolt_circuit);
        // assert!(result_verify.is_ok());
    }
}

pub mod bytecode;
pub mod instruction_lookups;
pub mod read_write_memory;
pub mod rv32i_vm;
