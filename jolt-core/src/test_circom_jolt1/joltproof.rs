use core::fmt;

use ark_bn254::{Bn254, Fr as Scalar};

use super::{
    jolt_device::{convert_from_jolt_device_to_circom, JoltDeviceCircom},
    joltproof_bytecode_proof::{convert_from_bytecode_proof_to_circom, BytecodeProofCircom},
    joltproof_inst_proof::{
        convert_from_inst_lookups_proof_to_circom, InstructionLookupsProofCircom,
    },
    joltproof_red_opening::{convert_reduced_opening_proof_to_circom, ReducedOpeningProofCircom},
    joltproof_rw_mem_proof::{
        convert_from_read_write_mem_proof_to_circom, ReadWriteMemoryProofCircom,
    },
    joltproof_uniform_spartan::{compute_uniform_spartan_to_circom, UniformSpartanProofCircom},
    pi_proof::{convert_piproof_to_circom, PIProofCircom},
    preprocess,
};

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct JoltproofCircom {
    pub trace_length: Scalar,
    pub program_io: JoltDeviceCircom,
    pub bytecode: BytecodeProofCircom,
    pub read_write_memory: ReadWriteMemoryProofCircom,
    pub instruction_lookups: InstructionLookupsProofCircom,
    pub r1cs: UniformSpartanProofCircom,
    pub opening_proof: ReducedOpeningProofCircom,
    pub pi_proof: PIProofCircom,
}

impl fmt::Debug for JoltproofCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
            "trace_length": "{}",
            "program_io": {:?},
            "bytecode": {:?},
            "read_write_memory": {:?},
            "instruction_lookups": {:?},
            "r1cs": {:?},
            "opening_proof": {:?},
            "pi_proof": {:?}
            }}"#,
            self.trace_length,
            self.program_io,
            self.bytecode,
            self.read_write_memory,
            self.instruction_lookups,
            self.r1cs,
            self.opening_proof,
            self.pi_proof
        )
    }
}

use crate::{
    jolt::vm::{
        rv32i_vm::{RV32ISubtables, C, M, RV32I},
        JoltPreprocessing, JoltProof,
    },
    poly::commitment::hyperkzg::HyperKZG,
    r1cs::inputs::JoltR1CSInputs,
    utils::poseidon_transcript::PoseidonTranscript,
};

pub fn convert_jolt_proof_to_circom(
    proof: JoltProof<
        { C },
        { M },
        JoltR1CSInputs,
        Scalar,
        HyperKZG<Bn254, PoseidonTranscript<Scalar, Scalar>>,
        RV32I,
        RV32ISubtables<Scalar>,
        PoseidonTranscript<Scalar, Scalar>,
    >,
    jolt_preprocessing: JoltPreprocessing<
        C,
        Scalar,
        HyperKZG<Bn254, PoseidonTranscript<Scalar, Scalar>>,
        PoseidonTranscript<Scalar, Scalar>,
    >,
) -> JoltproofCircom {
    let bytecode = convert_from_bytecode_proof_to_circom(proof.bytecode);
    JoltproofCircom {
        trace_length: Scalar::from(proof.trace_length as u128),
        program_io: convert_from_jolt_device_to_circom(proof.program_io),
        bytecode,
        read_write_memory: convert_from_read_write_mem_proof_to_circom(proof.read_write_memory),
        instruction_lookups: convert_from_inst_lookups_proof_to_circom(proof.instruction_lookups),
        r1cs: compute_uniform_spartan_to_circom(proof.r1cs),
        opening_proof: convert_reduced_opening_proof_to_circom(proof.opening_proof),
        pi_proof: convert_piproof_to_circom(jolt_preprocessing),
    }
}
