use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_std::log2;
use circom_scotia::r1cs;
use halo2curves::bn256;
use itertools::max;
use merlin::Transcript;
use rayon::prelude::*;
use std::any::TypeId;
use strum::{EnumCount, IntoEnumIterator};

use crate::lasso::memory_checking::{MemoryCheckingProver, MemoryCheckingVerifier};
use crate::poly::pedersen::PedersenGenerators;
use crate::poly::structured_poly::BatchablePolynomials;
use crate::r1cs::snark::R1CSProof;
use crate::utils::errors::ProofVerifyError;
use crate::{
    jolt::{
        instruction::{JoltInstruction, Opcode},
        subtable::LassoSubtable,
        vm::timestamp_range_check::TimestampValidityProof,
    },
    lasso::memory_checking::NoPreprocessing,
};
use common::{
    constants::MEMORY_OPS_PER_INSTRUCTION, field_conversion::IntoSpartan, ELFInstruction,
};

use self::instruction_lookups::{
    InstructionCommitment, InstructionLookupsPreprocessing, InstructionLookupsProof,
};
use self::read_write_memory::{MemoryCommitment, MemoryOp, ReadWriteMemory, ReadWriteMemoryProof};
use self::{
    bytecode::{BytecodeCommitment, BytecodePolynomials, BytecodeProof, ELFRow},
    instruction_lookups::InstructionPolynomials,
};

#[derive(Clone)]
pub struct JoltPreprocessing<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    pub generators: PedersenGenerators<G>,
    pub spartan_generators: Vec<bn256::G1Affine>,
    pub instruction_lookups: InstructionLookupsPreprocessing<F>,
}

pub struct JoltProof<const C: usize, const M: usize, F, G, InstructionSet, Subtables>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
    InstructionSet: JoltInstruction + Opcode + IntoEnumIterator + EnumCount,
    Subtables: LassoSubtable<F> + IntoEnumIterator,
{
    bytecode: BytecodeProof<F, G>,
    read_write_memory: ReadWriteMemoryProof<F, G>,
    instruction_lookups: InstructionLookupsProof<C, M, F, G, InstructionSet, Subtables>,
    r1cs: R1CSProof,
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

pub trait Jolt<'a, F: PrimeField, G: CurveGroup<ScalarField = F>, const C: usize, const M: usize>
where
    G::Affine: IntoSpartan,
{
    type InstructionSet: JoltInstruction + Opcode + IntoEnumIterator + EnumCount;
    type Subtables: LassoSubtable<F> + IntoEnumIterator + EnumCount + From<TypeId> + Into<usize>;

    #[tracing::instrument(skip_all, name = "Jolt::preprocess")]
    fn preprocess(
        max_bytecode_size: usize,
        max_memory_address: usize,
        max_trace_length: usize,
    ) -> JoltPreprocessing<F, G> {
        // TODO(moodlezoup): move more stuff into preprocessing

        let num_bytecode_generators =
            BytecodePolynomials::<F, G>::num_generators(max_bytecode_size, max_trace_length);
        let num_read_write_memory_generators =
            ReadWriteMemory::<F, G>::num_generators(max_memory_address, max_trace_length);
        let timestamp_range_check_generators =
            TimestampValidityProof::<F, G>::num_generators(max_trace_length);
        let preprocessing = InstructionLookupsPreprocessing::preprocess::<
            C,
            M,
            Self::InstructionSet,
            Self::Subtables,
        >();
        let num_instruction_lookup_generators = InstructionLookupsProof::<
            C,
            M,
            F,
            G,
            Self::InstructionSet,
            Self::Subtables,
        >::num_generators(
            &preprocessing, max_trace_length
        );

        let max_num_generators = max([
            num_bytecode_generators,
            num_read_write_memory_generators,
            timestamp_range_check_generators,
            num_instruction_lookup_generators,
        ])
        .unwrap();
        let generators = PedersenGenerators::new(max_num_generators, b"Jolt v1 Hyrax generators");

        JoltPreprocessing {
            generators: generators.clone(),
            spartan_generators: generators.to_spartan_bn256(),
            instruction_lookups: preprocessing,
        }
    }

    #[tracing::instrument(skip_all, name = "Jolt::prove")]
    fn prove(
        bytecode: Vec<ELFInstruction>,
        bytecode_trace: Vec<ELFRow>,
        memory_trace: Vec<[MemoryOp; MEMORY_OPS_PER_INSTRUCTION]>,
        instructions: Vec<Self::InstructionSet>,
        circuit_flags: Vec<F>,
        preprocessing: JoltPreprocessing<F, G>,
    ) -> (
        JoltProof<C, M, F, G, Self::InstructionSet, Self::Subtables>,
        JoltCommitments<G>,
    ) {
        let mut transcript = Transcript::new(b"Jolt transcript");
        let bytecode_rows: Vec<ELFRow> = bytecode.iter().map(ELFRow::from).collect();
        let (bytecode_proof, bytecode_polynomials, bytecode_commitment) = Self::prove_bytecode(
            bytecode_rows.clone(),
            bytecode_trace.clone(),
            &preprocessing.generators,
            &mut transcript,
        );

        // - prove_r1cs() memory_trace R1CS is not 2-padded
        // - prove_memory() memory_trace    is 2-padded
        let mut padded_memory_trace = memory_trace.clone();
        padded_memory_trace.resize(
            memory_trace.len().next_power_of_two(),
            std::array::from_fn(|_| MemoryOp::no_op()),
        );


        let (memory_proof, memory_polynomials, memory_commitment) = Self::prove_memory(
            bytecode.clone(),
            padded_memory_trace,
            &preprocessing.generators,
            &mut transcript,
        );

        let (
            instruction_lookups_proof,
            instruction_lookups_polynomials,
            instruction_lookups_commitment,
        ) = Self::prove_instruction_lookups(
            &preprocessing.instruction_lookups,
            instructions.clone(),
            &preprocessing.generators,
            &mut transcript,
        );

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

        let r1cs_proof = Self::prove_r1cs(
            preprocessing,
            instructions,
            bytecode_rows,
            bytecode_trace,
            bytecode,
            memory_trace.into_iter().flatten().collect(),
            circuit_flags,
            &jolt_polynomials, 
            &jolt_commitments, 
            &mut transcript,
        );

        let jolt_proof = JoltProof {
            bytecode: bytecode_proof,
            read_write_memory: memory_proof,
            instruction_lookups: instruction_lookups_proof,
            r1cs: r1cs_proof,
        };

        (jolt_proof, jolt_commitments)
    }

    fn verify(
        preprocessing: JoltPreprocessing<F, G>,
        proof: JoltProof<C, M, F, G, Self::InstructionSet, Self::Subtables>,
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
            &preprocessing.instruction_lookups,
            proof.instruction_lookups,
            commitments.instruction_lookups,
            &mut transcript,
        )?;
        proof
            .r1cs
            .verify()
            .map_err(|e| ProofVerifyError::SpartanError(e.to_string()))?;
        Ok(())
    }

    #[tracing::instrument(skip_all, name = "Jolt::prove_instruction_lookups")]
    fn prove_instruction_lookups(
        preprocessing: &InstructionLookupsPreprocessing<F>,
        ops: Vec<Self::InstructionSet>,
        generators: &PedersenGenerators<G>,
        transcript: &mut Transcript,
    ) -> (
        InstructionLookupsProof<C, M, F, G, Self::InstructionSet, Self::Subtables>,
        InstructionPolynomials<F, G>,
        InstructionCommitment<G>,
    ) {
        InstructionLookupsProof::prove_lookups(preprocessing, ops, generators, transcript)
    }

    fn verify_instruction_lookups(
        preprocessing: &InstructionLookupsPreprocessing<F>,
        proof: InstructionLookupsProof<C, M, F, G, Self::InstructionSet, Self::Subtables>,
        commitment: InstructionCommitment<G>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        InstructionLookupsProof::verify(preprocessing, proof, commitment, transcript)
    }

    #[tracing::instrument(skip_all, name = "Jolt::prove_bytecode")]
    fn prove_bytecode(
        bytecode_rows: Vec<ELFRow>,
        trace: Vec<ELFRow>,
        generators: &PedersenGenerators<G>,
        transcript: &mut Transcript,
    ) -> (
        BytecodeProof<F, G>,
        BytecodePolynomials<F, G>,
        BytecodeCommitment<G>,
    ) {
        let polys: BytecodePolynomials<F, G> = BytecodePolynomials::new(bytecode_rows, trace);
        let batched_polys = polys.batch();
        let commitment = BytecodePolynomials::commit(&polys, &batched_polys, &generators);

        let proof = BytecodeProof::prove_memory_checking(
            &NoPreprocessing,
            &polys,
            &batched_polys,
            transcript,
        );
        (proof, polys, commitment)
    }

    fn verify_bytecode(
        proof: BytecodeProof<F, G>,
        commitment: BytecodeCommitment<G>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        BytecodeProof::verify_memory_checking(&NoPreprocessing, proof, &commitment, transcript)
    }

    #[tracing::instrument(skip_all, name = "Jolt::prove_memory")]
    fn prove_memory(
        bytecode: Vec<ELFInstruction>,
        memory_trace: Vec<[MemoryOp; MEMORY_OPS_PER_INSTRUCTION]>,
        generators: &PedersenGenerators<G>,
        transcript: &mut Transcript,
    ) -> (
        ReadWriteMemoryProof<F, G>,
        ReadWriteMemory<F, G>,
        MemoryCommitment<G>,
    ) {
        let (memory, read_timestamps) = ReadWriteMemory::new(bytecode, memory_trace, transcript);
        let batched_polys = memory.batch();
        let commitment: MemoryCommitment<G> =
            ReadWriteMemory::commit(&memory, &batched_polys, &generators);

        let memory_checking_proof = ReadWriteMemoryProof::prove_memory_checking(
            &NoPreprocessing,
            &memory,
            &batched_polys,
            transcript,
        );

        let timestamp_validity_proof = TimestampValidityProof::prove(
            read_timestamps,
            &memory,
            &batched_polys,
            &generators,
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
        ReadWriteMemoryProof::verify_memory_checking(
            &NoPreprocessing,
            proof.memory_checking_proof,
            &commitment,
            transcript,
        )?;
        TimestampValidityProof::verify(&mut proof.timestamp_validity_proof, &commitment, transcript)
    }

    #[tracing::instrument(skip_all, name = "Jolt::prove_r1cs")]
    fn prove_r1cs(
        preprocessing: JoltPreprocessing<F, G>,
        instructions: Vec<Self::InstructionSet>,
        bytecode_rows: Vec<ELFRow>,
        trace: Vec<ELFRow>,
        bytecode: Vec<ELFInstruction>,
        memory_trace: Vec<MemoryOp>,
        circuit_flags: Vec<F>,
        jolt_polynomials: &JoltPolynomials<F, G>,
        jolt_commitments: &JoltCommitments<G>,
        transcript: &mut Transcript,
    ) -> R1CSProof {
        let N_FLAGS = 17;
        let TRACE_LEN = trace.len();

        let log_M = log2(M) as usize;

        let [mut prog_a_rw, mut prog_v_rw, _] =
            BytecodePolynomials::<F, G>::r1cs_polys_from_bytecode(bytecode_rows, trace);

        // Add circuit_flags_packed to prog_v_rw. Pack them in little-endian order.
        let span = tracing::span!(tracing::Level::INFO, "pack_flags");
        let _enter = span.enter();
        let precomputed_powers: Vec<F> = (0..N_FLAGS)
            .map(|i| F::from_u64(2u64.pow(i as u32)).unwrap())
            .collect();
        let packed_flags: Vec<F> = circuit_flags
            .par_chunks(N_FLAGS)
            .map(|x| {
                x.iter().enumerate().fold(F::zero(), |packed, (i, flag)| {
                    packed + *flag * precomputed_powers[N_FLAGS - 1 - i]
                })
            })
            .collect();
        prog_v_rw.extend(packed_flags);
        drop(_enter);
        drop(span);

        /* Transformation for single-step version */
        let span = tracing::span!(tracing::Level::INFO, "transform_to_single_step");
        let _enter = span.enter();
        let prog_v_components: Vec<&[F]> = prog_v_rw.chunks(TRACE_LEN).collect::<Vec<_>>();
        let mut new_prog_v_rw = Vec::with_capacity(prog_v_rw.len());
        for i in 0..TRACE_LEN {
            for component in &prog_v_components {
                if let Some(value) = component.get(i) {
                    new_prog_v_rw.push(value.clone());
                }
            }
        }
        drop(_enter);
        drop(span);

        prog_v_rw = new_prog_v_rw;
        /* End of transformation for single-step version */

        let [mut memreg_a_rw, mut memreg_v_reads, mut memreg_v_writes, _] =
            ReadWriteMemory::<F, G>::get_r1cs_polys(bytecode, memory_trace, transcript);

        let span = tracing::span!(tracing::Level::INFO, "compute chunks operands");
        let _guard = span.enter();
        let (mut chunks_x, mut chunks_y): (Vec<F>, Vec<F>) = instructions
            .iter()
            .flat_map(|op| {
                let [chunks_x, chunks_y] = op.operand_chunks(C, log_M);
                chunks_x.into_iter().zip(chunks_y.into_iter())
            })
            .map(|(x, y)| {
                (
                    F::from_u64(x as u64).unwrap(),
                    F::from_u64(y as u64).unwrap(),
                )
            })
            .unzip();

        let mut chunks_query = instructions
            .par_iter()
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
        circuit_flags_padded.extend(vec![
            F::zero();
            PADDED_TRACE_LEN * N_FLAGS - circuit_flags.len()
        ]);
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

        let mut jolt_commitments_spartan =  vec![
            // skip 1 (it is timestamp), and move rd (3) to the end
            jolt_commitments.bytecode.read_write_commitments[0].to_spartan_bn256(), // a
            jolt_commitments.bytecode.read_write_commitments[2].to_spartan_bn256(), // opcode, 
            jolt_commitments.bytecode.read_write_commitments[4].to_spartan_bn256(), // rs1
            jolt_commitments.bytecode.read_write_commitments[5].to_spartan_bn256(), // rs2
            jolt_commitments.bytecode.read_write_commitments[3].to_spartan_bn256(), // rd
            jolt_commitments.bytecode.read_write_commitments[6].to_spartan_bn256(), // imm
        ];

        let memory_comms = jolt_commitments.read_write_memory.a_v_read_write_commitments.iter().map(|x| x.to_spartan_bn256()).collect::<Vec<Vec<bn256::G1Affine>>>(); 

        jolt_commitments_spartan.extend(memory_comms);

        R1CSProof::prove(
            32, C, PADDED_TRACE_LEN, 
            inputs, 
            preprocessing.spartan_generators, 
            jolt_polynomials,
            &jolt_commitments_spartan, 
        ).expect("R1CS proof failed")
    }

    #[tracing::instrument(skip_all, name = "Jolt::compute_lookup_outputs")]
    fn compute_lookup_outputs(instructions: &Vec<Self::InstructionSet>) -> Vec<F> {
        instructions
            .par_iter()
            .map(|op| F::from_u64(op.lookup_entry()).unwrap())
            .collect()
    }
}

pub mod bytecode;
pub mod instruction_lookups;
pub mod read_write_memory;
pub mod rv32i_vm;
pub mod timestamp_range_check;
