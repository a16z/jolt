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
        let TRACE_LEN = trace.len();
        let log_M = log2(M) as usize;

        let [prog_a_rw, prog_v_rw_wo_flags_packed, prog_t_reads] = 
            BytecodePolynomials::<F, G>::r1cs_polys_from_bytecode(bytecode_rows, trace);

        // get circuit_flags_packed 
        // for every 15 values in circuit_flags, convert to a single value
        let circuit_flags_packed = circuit_flags
            .chunks(15)
            .map(|x| {
                let mut packed = F::ZERO;
                for (i, flag) in x.iter().enumerate() {
                    packed += *flag * F::from(2u64.pow(i as u32));
                }
                packed
            })
            .collect::<Vec<F>>();

        // append this to prog_v_rw
        let mut prog_v_rw = prog_v_rw_wo_flags_packed;
        prog_v_rw.extend(circuit_flags_packed);

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

        // assert
        assert_eq!(prog_a_rw.len(), TRACE_LEN);
        assert_eq!(prog_v_rw.len(), TRACE_LEN * 6); 
        assert_eq!(memreg_a_rw.len(), TRACE_LEN * 7);
        assert_eq!(memreg_v_reads.len(), TRACE_LEN * 7);
        assert_eq!(memreg_v_writes.len(), TRACE_LEN * 7);
        assert_eq!(chunks_x.len(), TRACE_LEN * C);
        assert_eq!(chunks_y.len(), TRACE_LEN * C);
        assert_eq!(chunks_query.len(), TRACE_LEN * C);
        assert_eq!(lookup_outputs.len(), TRACE_LEN);
        assert_eq!(circuit_flags.len(), TRACE_LEN * 15);

        let inputs = vec![
            prog_a_rw.clone(),
            prog_v_rw.clone(),
            // memreg_a_rw,
            // memreg_v_reads, 
            // memreg_v_writes,
            // memreg_t_reads, 
            // chunks_x, 
            // chunks_y,
            // chunks_query,
            // lookup_outputs,
            circuit_flags.to_vec(),
        ];

        // print the last 15 circuit flags 
        // println!("circuit flags: {:?}", circuit_flags[TRACE_LEN * 15 - 15*117..TRACE_LEN * 15 - 15*116].to_vec());
        println!("circuit flags: {:?}", circuit_flags[15 * 6.. 15 * 7].to_vec());

        // print the last trace_len values in prog_v_rw
        // println!("prog_v_rw: {:?}", prog_v_rw[TRACE_LEN * 6 - 117..TRACE_LEN*6-116].to_vec());
        println!("prog_v_rw: {:?}", prog_v_rw[TRACE_LEN * 6 - TRACE_LEN + 6]);

        // let NUM_STEPS = TRACE_LEN;
        // let inputs = vec![
        //     prog_a_rw.clone()[..NUM_STEPS].to_vec(),
        //     prog_v_rw[..NUM_STEPS * 6].to_vec(),
        //     // memreg_a_rw,
        //     // memreg_v_reads, 
        //     // memreg_v_writes,
        //     // memreg_t_reads, 
        //     // chunks_x, 
        //     // chunks_y,
        //     // chunks_query,
        //     // lookup_outputs,
        //     circuit_flags[..NUM_STEPS * 15].to_vec(),
        // ];



        // // print prog_a_rw
        // println!("prog_a_rw: {:?}", prog_a_rw); 

        // // print flags 
        // println!("flags len : {:?}", circuit_flags.len());

        // TODO: move this conversion to the r1cs module 
        use spartan2::provider::bn256_grumpkin::bn256;
        use spartan2::traits::Group;
        use bn256::Scalar as Spartan2Fr;
        type G1 = bn256::Point;
        use ff::Field;
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

        // let end_index = std::cmp::min(15*7, inputs_ff[2].len());
        // println!("inputs_ff[2][0..{}]: {:?}", end_index, &inputs_ff[2][0..end_index]);
        // // print the last trace_len elements of inputs_ff[1]
        // println!("inputs_ff[1]: {:?}", inputs_ff[1][TRACE_LEN*6-TRACE_LEN+5]);
        // convert inputs_ff[2] to binary values 
        println!("inputs_ff circuit_flags: {:?}", &inputs_ff[2]);

        // println!("inputs_ff.len : {:?}", inputs_ff.len());

        let jolt_circuit = JoltCircuit::<Spartan2Fr>::new_from_inputs(32, C, inputs_ff);
        let result_verify = run_jolt_spartan_with_circuit::<G1, S>(jolt_circuit);
        assert!(result_verify.is_ok());
        println!("The circuit passed.");
        assert!(false);
    }
}

pub mod bytecode;
pub mod instruction_lookups;
pub mod read_write_memory;
pub mod rv32i_vm;
