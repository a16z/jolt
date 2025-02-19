pub mod helper_commitms;
pub mod jolt_device;
pub mod joltproof;
pub mod joltproof_bytecode_proof;
pub mod joltproof_inst_proof;
pub mod joltproof_red_opening;
pub mod joltproof_rw_mem_proof;
pub mod joltproof_uniform_spartan;
pub mod pi_proof;
pub mod preprocess;
pub mod struct_fq;
pub mod sum_check_gkr;
pub mod transcript;
use std::{
    fs::{File, OpenOptions},
    io::Write,
    sync::{LazyLock, Mutex},
};

use crate::{
    field::JoltField,
    host,
    jolt::vm::{
        rv32i_vm::{RV32IJoltVM, C, M},
        Jolt,
    },
    poly::commitment::{
        commitment_scheme::CommitmentScheme,
        hyperkzg::{HyperKZG, HyperKZGCommitment},
    },
    utils::{poseidon_transcript::PoseidonTranscript, transcript::Transcript},
};

use crate::jolt::vm::{JoltPreprocessing, JoltProof, JoltStuff};

static FIB_FILE_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));
use crate::jolt::vm::rv32i_vm::{RV32ISubtables, RV32I};
use crate::r1cs::inputs::{ConstraintInput, JoltR1CSInputs};
use ark_bn254::{Bn254, Fq as Fp, Fr as Scalar};
use ark_ff::{AdditiveGroup, PrimeField};
use helper_commitms::{convert_from_jolt_stuff_to_circom, JoltStuffCircom};
use joltproof::{convert_jolt_proof_to_circom, JoltproofCircom};
use num_bigint::BigUint;
use preprocess::{convert_joltpreprocessing_to_circom, JoltPreprocessingCircom};
use transcript::convert_transcript_to_circom;

#[test]
fn fib_e2e_hyperkzg() {
    println!("Running Fib");
    fib_e2everify::<
        Scalar,
        HyperKZG<Bn254, PoseidonTranscript<Scalar, Scalar>>,
        PoseidonTranscript<Scalar, Scalar>,
    >();
    let (preprocessing, proof_from_rust, commitments) = fib_e2e::<
        Scalar,
        HyperKZG<Bn254, PoseidonTranscript<Scalar, Scalar>>,
        PoseidonTranscript<Scalar, Scalar>,
    >();

    // let circom_linking_hyperkzg_proof = hyper_kzg_proof_to_hyper_kzg_circomfor_linking(&proof_from_rust.opening_proof.joint_opening_proof);
    // let circom_linking_vk = convert_hyperkzg_verifier_key_to_hyperkzg_verifier_key_circom_for_linking(preprocessing.generators.1);
    let (circom_preprocessing, circom_proof, circom_stuff) =
        convert_full_proof_to_circom(preprocessing, proof_from_rust, &commitments);
    // let circom_linking_stuff = convert_from_jolt_stuff_to_circom_for_linking(&commitments);

    let transcipt_init =
        <PoseidonTranscript<Scalar, Scalar> as Transcript>::new(b"Jolt transcript");

    let input_json = format!(
        r#"{{
        "transcript_init": {:?},
        "preprocessing": {:?},
        "proof": {:?},
        "commitments": {:?}
    }}"#,
        convert_transcript_to_circom(transcipt_init),
        circom_preprocessing,
        circom_proof,
        circom_stuff
    );

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(input_json.as_bytes())
        .expect("Failed to write to input.json");
    println!("Input JSON file created successfully.");

    // let input_json_stuff = format!(
    //     r#" }},
    //      "commitments": {:?} }},

    //     "vk": {:?},
    //     "pi": {:?}
    // }}
    //     "#, circom_linking_stuff, circom_linking_vk, circom_linking_hyperkzg_proof

    // );
    // let input_file_path = "input_link.json";
    // let mut input_file = OpenOptions::new()
    //     .append(true)
    //     .create(true)
    //     .open(input_file_path)
    //     .expect("Failed to open input.json");
    // input_file
    //     .write_all(input_json_stuff.as_bytes())
    //     .expect("Failed to write to input.json");
    // println!("Input JSON file appended successfully.");
}

fn fib_e2e<F, PCS, ProofTranscript>() -> (
    JoltPreprocessing<C, F, PCS, ProofTranscript>,
    JoltProof<4, 65536, JoltR1CSInputs, F, PCS, RV32I, RV32ISubtables<F>, ProofTranscript>,
    JoltStuff<<PCS as CommitmentScheme<ProofTranscript>>::Commitment>,
)
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    let artifact_guard = FIB_FILE_LOCK.lock().unwrap();
    let mut program = host::Program::new("fibonacci-guest");
    program.set_input(&9u32);
    let (bytecode, memory_init) = program.decode();
    let (io_device, trace) = program.trace();
    drop(artifact_guard);

    let preprocessing: JoltPreprocessing<C, F, PCS, ProofTranscript> = RV32IJoltVM::preprocess(
        bytecode.clone(),
        io_device.memory_layout.clone(),
        memory_init,
        1 << 20,
        1 << 20,
        1 << 20,
    );

    let (proof, commitments, debug_info) =
        <RV32IJoltVM as Jolt<F, PCS, C, M, ProofTranscript>>::prove(
            io_device,
            trace,
            preprocessing.clone(),
        );

    // println!("bytecode stuff is {:?}", commitments.bytecode.a_read_write);
    // let verification_result =
    //     RV32IJoltVM::verify(preprocessing, proof, commitments, debug_info);
    // assert!(
    //     verification_result.is_ok(),
    //     "Verification failed with error: {:?}",
    //     verification_result.err()
    // );
    return (preprocessing, proof, commitments);
}

fn fib_e2everify<F, PCS, ProofTranscript>()
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    let artifact_guard = FIB_FILE_LOCK.lock().unwrap();
    let mut program = host::Program::new("fibonacci-guest");
    program.set_input(&9u32);
    let (bytecode, memory_init) = program.decode();
    let (io_device, trace) = program.trace();
    drop(artifact_guard);

    let preprocessing: JoltPreprocessing<C, F, PCS, ProofTranscript> = RV32IJoltVM::preprocess(
        bytecode.clone(),
        io_device.memory_layout.clone(),
        memory_init,
        1 << 20,
        1 << 20,
        1 << 20,
    );

    let (proof, commitments, debug_info) =
        <RV32IJoltVM as Jolt<F, PCS, C, M, ProofTranscript>>::prove(
            io_device,
            trace,
            preprocessing.clone(),
        );
    let verification_result = RV32IJoltVM::verify(preprocessing, proof, commitments, debug_info);
    assert!(
        verification_result.is_ok(),
        "Verification failed with error: {:?}",
        verification_result.err()
    );
}

pub fn convert_full_proof_to_circom(
    jolt_preprocessing: JoltPreprocessing<
        C,
        Scalar,
        HyperKZG<Bn254, PoseidonTranscript<Scalar, Scalar>>,
        PoseidonTranscript<Scalar, Scalar>,
    >,
    jolt_proof: JoltProof<
        { C },
        { M },
        JoltR1CSInputs,
        Scalar,
        HyperKZG<Bn254, PoseidonTranscript<Scalar, Scalar>>,
        RV32I,
        RV32ISubtables<Scalar>,
        PoseidonTranscript<Scalar, Scalar>,
    >,
    jolt_stuff: &JoltStuff<HyperKZGCommitment<Bn254>>,
) -> (JoltPreprocessingCircom, JoltproofCircom, JoltStuffCircom) {
    (
        convert_joltpreprocessing_to_circom(&jolt_preprocessing),
        convert_jolt_proof_to_circom(jolt_proof, jolt_preprocessing),
        convert_from_jolt_stuff_to_circom(jolt_stuff),
    )
}

pub fn convert_fp_to_3_limbs_of_scalar(r: &Fp) -> [Scalar; 3] {
    let mut limbs = [Scalar::ZERO; 3];

    let mask = BigUint::from((1u128 << 125) - 1);

    limbs[0] = Scalar::from(BigUint::from(r.into_bigint()) & mask.clone());

    limbs[1] = Scalar::from((BigUint::from(r.into_bigint()) >> 125) & mask.clone());

    limbs[2] = Scalar::from((BigUint::from(r.into_bigint()) >> 250) & mask.clone());

    limbs
}
