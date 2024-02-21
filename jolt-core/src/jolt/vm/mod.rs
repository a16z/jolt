use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_std::log2;
use merlin::Transcript;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::any::TypeId;
use strum::{EnumCount, IntoEnumIterator};

use crate::{jolt::{
    instruction::{JoltInstruction, Opcode},
    subtable::LassoSubtable,
    vm::timestamp_range_check::TimestampValidityProof,
}, poly::{hyrax::HyraxGenerators, pedersen::PedersenInit}};
use crate::lasso::memory_checking::{MemoryCheckingProver, MemoryCheckingVerifier};
use crate::poly::structured_poly::BatchablePolynomials;
use crate::r1cs::snark::prove_r1cs;
use crate::r1cs::snark::JoltCircuit;
use crate::utils::errors::ProofVerifyError;
use common::{constants::MEMORY_OPS_PER_INSTRUCTION, ELFInstruction};

use self::instruction_lookups::{
    InstructionCommitment, InstructionLookups, InstructionLookupsProof,
};
use self::read_write_memory::{MemoryCommitment, MemoryOp, ReadWriteMemory, ReadWriteMemoryProof};
use self::{
    bytecode::{BytecodeCommitment, BytecodePolynomials, BytecodeProof, ELFRow},
    instruction_lookups::InstructionPolynomials,
};

struct JoltProof<F, G, Subtables>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
    Subtables: LassoSubtable<F> + IntoEnumIterator,
{
    bytecode: BytecodeProof<F, G>,
    read_write_memory: ReadWriteMemoryProof<F, G>,
    instruction_lookups: InstructionLookupsProof<F, G, Subtables>,
    // TODO: r1cs
}

pub struct JoltPolynomials<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    pub bytecode: BytecodePolynomials<F, G>,
    pub read_write_memory: ReadWriteMemory<F, G>,
    pub instruction_lookups: InstructionPolynomials<F, G>,
}

pub struct JoltCommitments<G: CurveGroup> {
    pub bytecode: BytecodeCommitment<G>,
    pub read_write_memory: MemoryCommitment<G>,
    pub instruction_lookups: InstructionCommitment<G>,
}

pub trait Jolt<'a, F: PrimeField, G: CurveGroup<ScalarField = F>, const C: usize, const M: usize> {
    type InstructionSet: JoltInstruction + Opcode + IntoEnumIterator + EnumCount;
    type Subtables: LassoSubtable<F> + IntoEnumIterator + EnumCount + From<TypeId> + Into<usize>;

    fn prove(
        bytecode: Vec<ELFInstruction>,
        mut bytecode_trace: Vec<ELFRow>,
        memory_trace: Vec<[MemoryOp; MEMORY_OPS_PER_INSTRUCTION]>,
        instructions: Vec<Self::InstructionSet>,
    ) -> (JoltProof<F, G, Self::Subtables>, JoltCommitments<G>) {
        let mut transcript = Transcript::new(b"Jolt transcript");
        let mut bytecode_rows = bytecode.iter().map(ELFRow::from).collect();
        let (bytecode_proof, bytecode_polynomials, bytecode_commitment) =
            Self::prove_bytecode(bytecode_rows, bytecode_trace, &mut transcript);
        let (memory_proof, memory_polynomials, memory_commitment) =
            Self::prove_memory(bytecode, memory_trace, &mut transcript);
        let (
            instruction_lookups_proof,
            instruction_lookups_polynomials,
            instruction_lookups_commitment,
        ) = Self::prove_instruction_lookups(instructions, &mut transcript);
        todo!("r1cs");
        let jolt_proof = JoltProof {
            bytecode: bytecode_proof,
            read_write_memory: memory_proof,
            instruction_lookups: instruction_lookups_proof,
        };
        let jolt_polynomials = JoltPolynomials {
            bytecode: bytecode_polynomials,
            read_write_memory: memory_polynomials,
            instruction_lookups: instruction_lookups_polynomials,
        };
        let jolt_commitments = JoltCommitments {
            bytecode: bytecode_commitment,
            read_write_memory: memory_commitment,
            instruction_lookups: instruction_lookups_commitment,
        };
        (jolt_proof, jolt_commitments)
    }

    fn verify(
        proof: JoltProof<F, G, Self::Subtables>,
        commitments: JoltCommitments<G>,
    ) -> Result<(), ProofVerifyError> {
        let mut transcript = Transcript::new(b"Jolt transcript");
        Self::verify_bytecode(proof.bytecode, commitments.bytecode, &mut transcript)?;
        Self::verify_memory(
            proof.read_write_memory,
            commitments.read_write_memory,
            &mut transcript,
        )?;
        Self::verify_instruction_lookups(
            proof.instruction_lookups,
            commitments.instruction_lookups,
            &mut transcript,
        )?;
        todo!("r1cs");
    }

    #[tracing::instrument(skip_all, name = "Jolt::prove_instruction_lookups")]
    fn prove_instruction_lookups(
        ops: Vec<Self::InstructionSet>,
        transcript: &mut Transcript,
    ) -> (
        InstructionLookupsProof<F, G, Self::Subtables>,
        InstructionPolynomials<F, G>,
        InstructionCommitment<G>,
    ) {
        let instruction_lookups =
            InstructionLookups::<F, G, Self::InstructionSet, Self::Subtables, C, M>::new(ops);
        instruction_lookups.prove_lookups(transcript)
    }

    fn verify_instruction_lookups(
        proof: InstructionLookupsProof<F, G, Self::Subtables>,
        commitment: InstructionCommitment<G>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        InstructionLookups::<F, G, Self::InstructionSet, Self::Subtables, C, M>::verify(
            proof, commitment, transcript,
        )
    }

    #[tracing::instrument(skip_all, name = "Jolt::prove_bytecode")]
    fn prove_bytecode(
        bytecode_rows: Vec<ELFRow>,
        trace: Vec<ELFRow>,
        transcript: &mut Transcript,
    ) -> (
        BytecodeProof<F, G>,
        BytecodePolynomials<F, G>,
        BytecodeCommitment<G>,
    ) {
        let polys: BytecodePolynomials<F, G> = BytecodePolynomials::new(bytecode_rows, trace);
        let batched_polys = polys.batch();
        let initializer: PedersenInit<G> = HyraxGenerators::new_initializer(polys.max_generator_size(), b"LassoV1");
        let commitment = BytecodePolynomials::commit(&batched_polys, &initializer);

        let proof = polys.prove_memory_checking(&polys, &batched_polys, transcript);
        (proof, polys, commitment)
    }

    fn verify_bytecode(
        proof: BytecodeProof<F, G>,
        commitment: BytecodeCommitment<G>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        BytecodePolynomials::verify_memory_checking(proof, &commitment, transcript)
    }

    #[tracing::instrument(skip_all, name = "Jolt::prove_memory")]
    fn prove_memory(
        bytecode: Vec<ELFInstruction>,
        memory_trace: Vec<[MemoryOp; MEMORY_OPS_PER_INSTRUCTION]>,
        transcript: &mut Transcript,
    ) -> (
        ReadWriteMemoryProof<F, G>,
        ReadWriteMemory<F, G>,
        MemoryCommitment<G>,
    ) {
        let (memory, read_timestamps) = ReadWriteMemory::new(bytecode, memory_trace, transcript);
        let batched_polys = memory.batch();
        let initializer: PedersenInit<G> = HyraxGenerators::new_initializer(memory.max_generator_size(), b"LassoV1");
        let commitment: MemoryCommitment<G> = ReadWriteMemory::commit(&batched_polys, &initializer);

        let memory_checking_proof =
            memory.prove_memory_checking(&memory, &batched_polys, transcript);

        let timestamp_validity_proof = TimestampValidityProof::prove(
            read_timestamps,
            &memory,
            &batched_polys,
            &commitment,
            transcript,
        );

        (
            ReadWriteMemoryProof {
                memory_checking_proof,
                timestamp_validity_proof,
            },
            memory,
            commitment,
        )
    }

    fn verify_memory(
        mut proof: ReadWriteMemoryProof<F, G>,
        commitment: MemoryCommitment<G>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        ReadWriteMemory::verify_memory_checking(
            proof.memory_checking_proof,
            &commitment,
            transcript,
        )?;
        TimestampValidityProof::verify(&mut proof.timestamp_validity_proof, &commitment, transcript)
    }

    #[tracing::instrument(skip_all, name = "Jolt::prove_r1cs")]
    fn prove_r1cs(
        instructions: Vec<Self::InstructionSet>,
        bytecode_rows: Vec<ELFRow>,
        trace: Vec<ELFRow>,
        bytecode: Vec<ELFInstruction>,
        memory_trace: Vec<MemoryOp>,
        circuit_flags: Vec<F>,
        transcript: &mut Transcript,
    ) {
        let N_FLAGS = 17;
        let TRACE_LEN = trace.len();

        let log_M = log2(M) as usize;

        let [mut prog_a_rw, mut prog_v_rw, _] =
            BytecodePolynomials::<F, G>::r1cs_polys_from_bytecode(bytecode_rows, trace);

        // Add circuit_flags_packed to prog_v_rw. Pack them in little-endian order.
        prog_v_rw.extend(circuit_flags.chunks(N_FLAGS).map(|x| {
            x.iter().enumerate().fold(F::zero(), |packed, (i, flag)| {
                packed + *flag * F::from(2u64.pow((N_FLAGS - 1 - i) as u32))
            })
        }));

        /* Transformation for single-step version */
        let prog_v_components = prog_v_rw.chunks(TRACE_LEN).collect::<Vec<_>>();
        let mut new_prog_v_rw = Vec::with_capacity(prog_v_rw.len());

        for i in 0..TRACE_LEN {
            for component in &prog_v_components {
                if let Some(value) = component.get(i) {
                    new_prog_v_rw.push(value.clone());
                }
            }
        }

        prog_v_rw = new_prog_v_rw;
        /* End of transformation for single-step version */

        let [mut memreg_a_rw, mut memreg_v_reads, mut memreg_v_writes, _] =
            ReadWriteMemory::<F, G>::get_r1cs_polys(bytecode, memory_trace, transcript);

        let span = tracing::span!(tracing::Level::INFO, "compute chunks operands");
        let _guard = span.enter();
        let (mut chunks_x, mut chunks_y): (Vec<F>, Vec<F>) = instructions
            .iter()
            .flat_map(|op| {
                let chunks_xy = op.operand_chunks(C, log_M);
                let chunks_x = chunks_xy[0].clone();
                let chunks_y = chunks_xy[1].clone();
                chunks_x.into_iter().zip(chunks_y.into_iter())
            })
            .map(|(x, y)| (F::from(x as u64), F::from(y as u64)))
            .unzip();

        let mut chunks_query = instructions
            .iter()
            .flat_map(|op| op.to_indices(C, log_M))
            .map(|x| x as u64)
            .map(F::from)
            .collect::<Vec<F>>();
        drop(_guard);
        drop(span);

        let mut lookup_outputs = Self::compute_lookup_outputs(&instructions);

        // assert lengths
        assert_eq!(prog_a_rw.len(), TRACE_LEN);
        assert_eq!(prog_v_rw.len(), TRACE_LEN * 6);
        assert_eq!(memreg_a_rw.len(), TRACE_LEN * 7);
        assert_eq!(memreg_v_reads.len(), TRACE_LEN * 7);
        assert_eq!(memreg_v_writes.len(), TRACE_LEN * 7);
        assert_eq!(chunks_x.len(), TRACE_LEN * C);
        assert_eq!(chunks_y.len(), TRACE_LEN * C);
        assert_eq!(chunks_query.len(), TRACE_LEN * C);
        assert_eq!(lookup_outputs.len(), TRACE_LEN);
        assert_eq!(circuit_flags.len(), TRACE_LEN * N_FLAGS);

        // let padded_trace_len = next power of 2 from trace_eln 
        let PADDED_TRACE_LEN = TRACE_LEN.next_power_of_two();

        // pad each of the above vectors to be of length PADDED_TRACE_LEN * their multiple 
        prog_a_rw.resize(PADDED_TRACE_LEN, Default::default());
        prog_v_rw.resize(PADDED_TRACE_LEN * 6, Default::default());
        memreg_a_rw.resize(PADDED_TRACE_LEN * 7, Default::default());
        memreg_v_reads.resize(PADDED_TRACE_LEN * 7, Default::default());
        memreg_v_writes.resize(PADDED_TRACE_LEN * 7, Default::default());
        chunks_x.resize(PADDED_TRACE_LEN * C, Default::default());
        chunks_y.resize(PADDED_TRACE_LEN * C, Default::default());
        chunks_query.resize(PADDED_TRACE_LEN * C, Default::default());
        lookup_outputs.resize(PADDED_TRACE_LEN, Default::default());

        let mut circuit_flags_padded = circuit_flags.clone();
        circuit_flags_padded.extend(vec![F::from(0_u64); PADDED_TRACE_LEN * N_FLAGS - circuit_flags.len()]);
        // circuit_flags.resize(PADDED_TRACE_LEN * N_FLAGS, Default::default());

        let inputs = vec![
            prog_a_rw,
            prog_v_rw,
            memreg_a_rw,
            memreg_v_reads,
            memreg_v_writes,
            chunks_x,
            chunks_y,
            chunks_query,
            lookup_outputs,
            circuit_flags_padded,
        ];

        let res = prove_r1cs(32, C, PADDED_TRACE_LEN, inputs);
        assert!(res.is_ok());
    }

    #[tracing::instrument(skip_all, name = "Jolt::compute_lookup_outputs")]
    fn compute_lookup_outputs(instructions: &Vec<Self::InstructionSet>) -> Vec<F> {
        instructions
            .par_iter()
            .map(|op| op.lookup_entry::<F>(C, M))
            .collect()
    }
}

pub mod bytecode;
pub mod instruction_lookups;
pub mod read_write_memory;
pub mod rv32i_vm;
pub mod timestamp_range_check;
