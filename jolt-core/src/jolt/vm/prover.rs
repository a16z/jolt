#![allow(clippy::type_complexity)]
#![allow(dead_code)]

use crate::field::JoltField;
use crate::jolt::vm::bytecode::BytecodeShoutProof;
use crate::jolt::vm::instruction_lookups::LookupsProof;
use crate::jolt::vm::ram::RAMTwistProof;
use crate::jolt::vm::registers::RegistersTwistProof;
use crate::jolt::vm::rv32im_vm::Serializable;
use crate::jolt::vm::{JoltCommon, JoltProof, JoltProverPreprocessing, JoltSharedPreprocessing};
use crate::msm::icicle;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::opening_proof::ProverOpeningAccumulator;
use crate::r1cs::constraints::R1CSConstraints;
use crate::r1cs::spartan::UniformSpartanProof;
use crate::utils::transcript::Transcript;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::jolt_device::MemoryLayout;
use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};
use tracer::instruction::{RV32IMCycle, RV32IMInstruction};
use tracer::JoltDevice;

impl<F, PCS, ProofTranscript> Serializable for JoltProverPreprocessing<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
}

impl<F, PCS, ProofTranscript> JoltProverPreprocessing<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub fn save_to_target_dir(&self, target_dir: &str) -> std::io::Result<()> {
        let filename = Path::new(target_dir).join("jolt_prover_preprocessing.dat");
        let mut file = File::create(filename.as_path())?;
        let mut data = Vec::new();
        self.serialize_compressed(&mut data).unwrap();
        file.write_all(&data)?;
        Ok(())
    }

    pub fn read_from_target_dir(target_dir: &str) -> std::io::Result<Self> {
        let filename = Path::new(target_dir).join("jolt_prover_preprocessing.dat");
        let mut file = File::open(filename.as_path())?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        Ok(Self::deserialize_compressed(&*data).unwrap())
    }
}

pub struct ProverDebugInfo<F, ProofTranscript, PCS>
where
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
{
    pub transcript: ProofTranscript,
    pub opening_accumulator: ProverOpeningAccumulator<F, ProofTranscript>,
    pub prover_setup: PCS::ProverSetup,
}

pub trait JoltProver<const WORD_SIZE: usize, F, PCS, ProofTranscript>:
    JoltCommon<WORD_SIZE, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    type Constraints: R1CSConstraints<F>;

    #[tracing::instrument(skip_all, name = "Jolt::preprocess")]
    fn prover_preprocess(
        bytecode: Vec<RV32IMInstruction>,
        memory_layout: MemoryLayout,
        memory_init: Vec<(u64, u8)>,
        max_bytecode_size: usize,
        max_memory_size: usize,
        max_trace_length: usize,
    ) -> JoltProverPreprocessing<F, PCS, ProofTranscript> {
        let small_value_lookup_tables = F::compute_lookup_tables();
        F::initialize_lookup_tables(small_value_lookup_tables.clone());

        let shared = <Self as JoltCommon<WORD_SIZE, F, PCS, ProofTranscript>>::shared_preprocess(
            bytecode,
            memory_layout,
            memory_init,
        );

        let max_poly_len: usize = [
            (max_bytecode_size + 1).next_power_of_two(), // Account for no-op prepended to bytecode
            max_trace_length.next_power_of_two(),
            max_memory_size.next_power_of_two(),
        ]
        .into_iter()
        .max()
        .unwrap();
        let generators = PCS::setup_prover(max_poly_len);

        JoltProverPreprocessing {
            generators,
            shared,
            field: small_value_lookup_tables,
        }
    }

    #[tracing::instrument(skip_all, name = "Jolt::prove")]
    fn prove(
        program_io: JoltDevice,
        mut trace: Vec<RV32IMCycle>,
        mut preprocessing: JoltProverPreprocessing<F, PCS, ProofTranscript>,
    ) -> (
        JoltProof<WORD_SIZE, F, PCS, ProofTranscript>,
        // JoltCommitments<PCS, ProofTranscript>,
        JoltDevice,
        Option<ProverDebugInfo<F, ProofTranscript, PCS>>,
    ) {
        icicle::icicle_init();
        let trace_length = trace.len();
        println!("Trace length: {trace_length}");

        F::initialize_lookup_tables(std::mem::take(&mut preprocessing.field));

        // TODO(moodlezoup): Truncate generators

        // TODO(JP): Drop padding on number of steps
        let padded_trace_length = trace_length.next_power_of_two();
        let padding = padded_trace_length - trace_length;
        let last_address = trace.last().unwrap().instruction().normalize().address;
        if padding != 0 {
            // Pad with NoOps (with sequential addresses) followed by a final JALR
            trace.extend((0..padding - 1).map(|i| RV32IMCycle::NoOp(last_address + 4 * i)));
            // Final JALR sets NextUnexpandedPC = 0
            trace.push(RV32IMCycle::last_jalr(last_address + 4 * (padding - 1)));
        } else {
            // Replace last JAL with JALR to set NextUnexpandedPC = 0
            assert!(matches!(trace.last().unwrap(), RV32IMCycle::JAL(_)));
            *trace.last_mut().unwrap() = RV32IMCycle::last_jalr(last_address);
        }

        let mut transcript = ProofTranscript::new(b"Jolt transcript");
        let mut opening_accumulator: ProverOpeningAccumulator<F, ProofTranscript> =
            ProverOpeningAccumulator::new();

        Self::fiat_shamir_preamble(
            &mut transcript,
            &program_io,
            &program_io.memory_layout,
            trace_length,
        );

        // jolt_commitments
        //     .read_write_values()
        //     .iter()
        //     .for_each(|value| value.append_to_transcript(&mut transcript));
        // jolt_commitments
        //     .init_final_values()
        //     .iter()
        //     .for_each(|value| value.append_to_transcript(&mut transcript));

        let constraint_builder = Self::Constraints::construct_constraints(padded_trace_length);
        let spartan_key = UniformSpartanProof::<F, ProofTranscript>::setup(
            &constraint_builder,
            padded_trace_length,
        );
        transcript.append_scalar(&spartan_key.vk_digest);

        let r1cs_proof = UniformSpartanProof::prove::<PCS>(
            &preprocessing,
            &constraint_builder,
            &spartan_key,
            &trace,
            &mut opening_accumulator,
            &mut transcript,
        )
        .ok()
        .unwrap();

        let instruction_proof =
            LookupsProof::prove(&trace, &mut opening_accumulator, &mut transcript);

        let registers_proof =
            RegistersTwistProof::prove(&trace, &mut opening_accumulator, &mut transcript);

        let ram_proof = RAMTwistProof::prove(
            &preprocessing.shared.ram,
            &trace,
            &program_io,
            1 << 16, // TODO(moodlezoup)
            &mut opening_accumulator,
            &mut transcript,
        );

        let bytecode_proof =
            BytecodeShoutProof::prove(&preprocessing.shared.bytecode, &trace, &mut transcript);

        // Batch-prove all openings
        // let opening_proof =
        //     opening_accumulator.reduce_and_prove::<PCS>(&preprocessing.generators, &mut transcript);

        let jolt_proof = JoltProof {
            trace_length,
            bytecode: bytecode_proof,
            instruction_lookups: instruction_proof,
            ram: ram_proof,
            registers: registers_proof,
            r1cs: r1cs_proof,
            // opening_proof,
        };

        #[cfg(test)]
        let debug_info = Some(ProverDebugInfo {
            transcript,
            opening_accumulator,
            prover_setup: preprocessing.generators.clone(),
        });
        #[cfg(not(test))]
        let debug_info = None;
        (jolt_proof, program_io, debug_info)
    }
}
