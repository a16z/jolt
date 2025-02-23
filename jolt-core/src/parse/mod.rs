mod commit;
mod jolt;
mod spartan_hkzg;
mod spartan_hyrax;
pub(crate) trait Parse {
    fn format(&self) -> serde_json::Value {
        unimplemented!("")
    }
    fn format_non_native(&self) -> serde_json::Value {
        unimplemented!("")
    }
    fn format_setup(&self, _size: usize) -> serde_json::Value {
        unimplemented!("added for setup")
    }
}
pub(crate) const NUM_MEMORIES: usize = 54;
pub(crate) const NUM_INSTRUCTIONS: usize = 26;
pub(crate) const MEMORY_OPS_PER_INSTRUCTION: usize = 4;
pub(crate) static CHUNKS_X_SIZE: usize = 4;
pub(crate) static CHUNKS_Y_SIZE: usize = 4;
pub(crate) const NUM_CIRCUIT_FLAGS: usize = 11;
pub(crate) static RELEVANT_Y_CHUNKS_LEN: usize = 4;

// TODO: This is inner_sumcheck_rounds - 1. Fix this dynamically?
pub(crate) static POSTPONED_POINT_LEN: usize = 10;

#[cfg(test)]
mod test {
    use std::{
        fs::File,
        io::Write,
        sync::{LazyLock, Mutex},
    };

    use num_bigint::BigUint;
    use serde_json::json;

    use crate::{
        field::JoltField,
        host,
        jolt::vm::{
            rv32i_vm::{RV32IJoltVM, RV32ISubtables, C, M, RV32I},
            Jolt, JoltPreprocessing, JoltProof, JoltStuff, ProverDebugInfo,
        },
        parse::{
            spartan_hkzg::{spartan_hkzg, LinkingStuff1},
            Parse,
        },
        poly::commitment::{commitment_scheme::CommitmentScheme, hyperkzg::HyperKZG},
        r1cs::inputs::JoltR1CSInputs,
        utils::{poseidon_transcript::PoseidonTranscript, transcript::Transcript},
    };
    type Fr = ark_bn254::Fr;
    type ProofTranscrip = PoseidonTranscript<Fr, Fr>;
    type PCS = HyperKZG<ark_bn254::Bn254, ProofTranscrip>;

    // If multiple tests try to read the same trace artifacts simultaneously, they will fail
    static FIB_FILE_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    #[test]
    fn end_to_end_testing() {
        println!("Running Fib");

        let (jolt_preprocessing, jolt_proof, jolt_commitments, _debug_info) =
            fib_e2e::<Fr, PCS, ProofTranscrip>();
        // let verification_result =
        //     RV32IJoltVM::verify(jolt_preprocessing, jolt_proof, jolt_commitments, debug_info);
        // assert!(
        //     verification_result.is_ok(),
        //     "Verification failed with error: {:?}",
        //     verification_result.err()
        // );

        let jolt1_input = json!(
        {
            "preprocessing": {
                "v_init_final_hash": jolt_preprocessing.bytecode.v_init_final_hash.to_string(),
                "bytecode_words_hash": jolt_preprocessing.read_write_memory.hash.to_string()
            },
            "proof": jolt_proof.format(),
            "commitments": jolt_commitments.format_non_native(),
            "pi_proof": jolt_preprocessing.format()
        });

        // Convert the JSON to a pretty-printed string
        let pretty_json =
            serde_json::to_string_pretty(&jolt1_input).expect("Failed to serialize JSON");

        let input_file_path = "jolt1_input.json";
        let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
        input_file
            .write_all(pretty_json.as_bytes())
            .expect("Failed to write to input.json");

        //TODO(Ashish):- Add code to generate witness

        //Note:- How to test
        //1.Comment the below part --> generate jolt1_input --> CP to CircomJolt
        //2.Generate witness.json from circom --> CP witness.json here
        //3.Uncomment the below part --> generate jolt2_input --> CP to CircomJolt
        //4.Test Jolt2
        //5. After 1st iteration repeate 3 to 4.zzzz

        // Read the witness.json file
        let witness_file_path = "src/spartan/witness.json";
        let witness_file = File::open(witness_file_path).expect("Failed to open witness.json");
        let witness: Vec<String> = serde_json::from_reader(witness_file).unwrap();

        let mut z = Vec::new();
        for value in witness {
            let val: BigUint = value.parse().unwrap();
            let mut bytes = val.to_bytes_le();
            bytes.resize(32, 0u8);
            let val = Fr::from_bytes(&bytes);
            z.push(val);
        }
        let linking_stuff = LinkingStuff1::new(jolt_commitments, z);
        let jolt_vk = jolt_preprocessing.generators.1.format();
        let jolt2_input = json!(
        {
            "linkingstuff": linking_stuff.format(),
            "vk": jolt_vk,
            "pi": jolt_proof.opening_proof.joint_opening_proof.format()
        });

        let input_file_path = "jolt2_input.json";
        let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
        let pretty_json =
            serde_json::to_string_pretty(&jolt2_input).expect("Failed to serialize JSON");
        input_file
            .write_all(pretty_json.as_bytes())
            .expect("Failed to write to input.json");

        let linking_stuff = linking_stuff.format_non_native();
        let jolt_pi = json!({
            "v_init_final_hash": jolt_preprocessing.bytecode.v_init_final_hash.format_non_native(),
                "bytecode_words_hash": jolt_preprocessing.read_write_memory.hash.format_non_native()
        });

        spartan_hkzg(linking_stuff, jolt_pi, jolt2_input, jolt_vk);
    }

    fn fib_e2e<F, PCS, ProofTranscript>() -> (
        JoltPreprocessing<C, F, PCS, ProofTranscript>,
        JoltProof<C, M, JoltR1CSInputs, F, PCS, RV32I, RV32ISubtables<F>, ProofTranscript>,
        JoltStuff<<PCS as CommitmentScheme<ProofTranscript>>::Commitment>,
        std::option::Option<ProverDebugInfo<F, ProofTranscript>>,
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

        let preprocessing = RV32IJoltVM::preprocess(
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

        (preprocessing, proof, commitments, debug_info)
    }
}
