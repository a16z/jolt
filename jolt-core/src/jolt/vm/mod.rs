use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use merlin::Transcript;
use std::any::TypeId;
use strum::{EnumCount, IntoEnumIterator};

use crate::lasso::{
    memory_checking::MemoryCheckingProver,
    surge::Surge,
};

use crate::jolt::{
    instruction::{sltu::SLTUInstruction, JoltInstruction, Opcode},
    subtable::LassoSubtable,
};
use crate::poly::structured_poly::BatchablePolynomials;
use crate::utils::{errors::ProofVerifyError, random::RandomTape};

use self::instruction_lookups::{InstructionLookups, InstructionLookupsProof};
use self::read_write_memory::{MemoryCommitment, MemoryOp, ReadWriteMemory};

pub trait Jolt<F: PrimeField, G: CurveGroup<ScalarField = F>, const C: usize, const M: usize> {
    type InstructionSet: JoltInstruction + Opcode + IntoEnumIterator + EnumCount;
    type Subtables: LassoSubtable<F> + IntoEnumIterator + EnumCount + From<TypeId> + Into<usize>;

    fn prove() {
        // preprocess?
        // emulate
        // prove_program_code
        // prove_memory
        // prove_lookups
        // prove_r1cs
        unimplemented!("todo");
    }

    fn prove_instruction_lookups(
        ops: Vec<Self::InstructionSet>,
        r: Vec<F>,
        transcript: &mut Transcript,
    ) -> InstructionLookupsProof<F, G> {
        let instruction_lookups =
            InstructionLookups::<F, G, Self::InstructionSet, Self::Subtables, C, M>::new(ops);
        instruction_lookups.prove_lookups(r, transcript)
    }

    fn verify_instruction_lookups(
        proof: InstructionLookupsProof<F, G>,
        r: Vec<F>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        InstructionLookups::<F, G, Self::InstructionSet, Self::Subtables, C, M>::verify(
            proof, &r, transcript,
        )
    }

    fn prove_program_code(
        program_code: &[u64],
        access_sequence: &[usize],
        code_size: usize,
        contiguous_reads_per_access: usize,
        r_mem_check: &(F, F),
        transcript: &mut Transcript,
    ) {
        // let (gamma, tau) = r_mem_check;
        // let hash_func = |a: &F, v: &F, t: &F| -> F { *t * gamma.square() + *v * *gamma + *a - tau };

        // let m: usize = (access_sequence.len() * contiguous_reads_per_access).next_power_of_two();
        // // TODO(moodlezoup): resize access_sequence?

        // let mut read_addrs: Vec<usize> = Vec::with_capacity(m);
        // let mut final_cts: Vec<usize> = vec![0; code_size];
        // let mut read_cts: Vec<usize> = Vec::with_capacity(m);
        // let mut read_values: Vec<u64> = Vec::with_capacity(m);

        // for (j, code_address) in access_sequence.iter().enumerate() {
        //   debug_assert!(code_address + contiguous_reads_per_access <= code_size);
        //   debug_assert!(code_address % contiguous_reads_per_access == 0);

        //   for offset in 0..contiguous_reads_per_access {
        //     let addr = code_address + offset;
        //     let counter = final_cts[addr];
        //     read_addrs.push(addr);
        //     read_values.push(program_code[addr]);
        //     read_cts.push(counter);
        //     final_cts[addr] = counter + 1;
        //   }
        // }

        // let E_poly: DensePolynomial<F> = DensePolynomial::from_u64(&read_values); // v_ops
        // let dim: DensePolynomial<F> = DensePolynomial::from_usize(access_sequence); // a_ops
        // let read_cts: DensePolynomial<F> = DensePolynomial::from_usize(&read_cts); // t_read
        // let final_cts: DensePolynomial<F> = DensePolynomial::from_usize(&final_cts); // t_final
        // let init_values: DensePolynomial<F> = DensePolynomial::from_u64(program_code); // v_mem

        // let polys = PCPolys::new(dim, E_poly, init_values, read_cts, final_cts, 0);
        // let (gens, commitments) = polys.commit::<G>();

        todo!("decide how to represent nested proofs, gens, commitments");
        // MemoryCheckingProof::<G, PCFingerprintProof<G>>::prove(
        //   &polys,
        //   r_fingerprints,
        //   &gens,
        //   &mut transcript,
        //   &mut random_tape,
        // )
    }

    fn prove_memory(memory_trace: Vec<MemoryOp>, memory_size: usize, transcript: &mut Transcript) {
        const MAX_TRACE_SIZE: usize = 1 << 22;
        // TODO: Support longer traces
        assert!(memory_trace.len() <= MAX_TRACE_SIZE);

        todo!("Load program bytecode into memory");

        let (memory, read_timestamps) = ReadWriteMemory::new(memory_trace, memory_size, transcript);
        let batched_polys = memory.batch();
        let commitments: MemoryCommitment<G> = ReadWriteMemory::commit(&batched_polys);

        let mut random_tape = RandomTape::new(b"proof");
        memory.prove_memory_checking(
            &memory,
            &batched_polys,
            &commitments,
            transcript,
            &mut random_tape,
        );

        let timestamp_validity_lookups: Vec<SLTUInstruction> = read_timestamps
            .iter()
            .enumerate()
            .map(|(i, &ts)| SLTUInstruction(ts, i as u64 + 1))
            .collect();

        let timestamp_validity_proof =
            <Surge<F, G, SLTUInstruction, 2, MAX_TRACE_SIZE>>::new(timestamp_validity_lookups)
                .prove(transcript);
    }

    fn prove_r1cs(
        read_write_memory: ReadWriteMemory<F, G>, // for memory checking
        ops: Vec<Self::InstructionSet>, // for instruction lookups 
    ) {
        // Program vectors 

        // Memory vectors 
        let memreg_a_rw = read_write_memory.a_read_write;
        let memreg_v_reads = read_write_memory.v_read;
        let memreg_v_writes= read_write_memory.v_write;
        let memreg_t_reads= read_write_memory.t_read;

        // Lookup vectors 
        // TODO: get chunks_x, chunks_y from the lookup ops  
        // let chunks_x = ops.iter().map(|op| op.to_indices::<F>(C, M)).collect::<Vec<F>>();
        let chunks_query = ops.iter().map(|op| op.to_indices::<F>(C, M)).collect::<Vec<F>>();
        let lookup_outputs = ops.iter().map(|op| op.lookup_entry::<F>(C, M)).collect::<Vec<F>>();

        // op_flags 


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
            op_flags,
        ];
    }
}

pub mod instruction_lookups;
pub mod pc;
pub mod read_write_memory;
pub mod rv32i_vm;
