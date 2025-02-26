#![allow(dead_code)]
use crate::field::JoltField;
use num_bigint::BigUint;
use std::{
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

mod commit;
mod jolt;
mod spartan1;
mod spartan2;
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

pub(crate) fn write_json(input: &serde_json::Value, out_dir: &str, package_name: &str) {
    let file_name = format!("{}_input.json", package_name);
    let file_path = Path::new(out_dir).join(&file_name);

    // Convert the JSON to a pretty-printed string
    let pretty_json = serde_json::to_string_pretty(&input).expect("Failed to serialize JSON");

    let mut input_file = File::create(file_path).expect("Failed to create input.json");
    input_file
        .write_all(pretty_json.as_bytes())
        .expect("Failed to write input file");
}

pub(crate) fn read_witness<F: JoltField>(witness_file_path: &str) -> Vec<F> {
    let witness_file = File::open(witness_file_path).expect("Failed to open witness.json");
    let witness: Vec<String> = serde_json::from_reader(witness_file).unwrap();
    let mut z = Vec::new();

    for value in witness {
        let val: BigUint = value.parse().unwrap();
        let mut bytes = val.to_bytes_le();
        bytes.resize(32, 0u8);
        let val = F::from_bytes(&bytes);
        z.push(val);
    }
    z
}

fn get_paths(package_names: &[&str]) -> Vec<PathBuf> {
    let mut package_paths = Vec::new();

    let mut build_args = vec!["build"];
    for package_name in package_names {
        build_args.push("--package");
        build_args.push(package_name);
    }

    let status = Command::new("cargo")
        .args(&build_args)
        .status()
        .expect("Failed to build packages");

    if !status.success() {
        panic!("Failed to build one or more packages.");
    }

    // Get Cargo metadata once to find where dependencies are stored
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1"])
        .output()
        .expect("Failed to get cargo metadata");

    let metadata: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("Failed to parse cargo metadata");

    let packages = metadata
        .get("packages")
        .and_then(|p| p.as_array())
        .expect("Invalid metadata format");

    for package_name in package_names {
        let mut repo_b_path = None;

        for package in packages {
            if package.get("name").and_then(|n| n.as_str()) == Some(*package_name) {
                repo_b_path = package
                    .get("manifest_path")
                    .and_then(|m| m.as_str())
                    .map(|m| PathBuf::from(m).parent().unwrap().to_path_buf());
                break;
            }
        }

        let repo_b_path =
            repo_b_path.unwrap_or_else(|| panic!("Could not find package: {}", package_name));

        let circom_file_name = format!("{}.circom", package_name);
        let circom_file = repo_b_path.join(circom_file_name);

        // Ensure the file exists
        assert!(circom_file.exists(), "File not found at {:?}", circom_file);

        package_paths.push(circom_file);
    }
    package_paths
}

#[cfg(test)]
mod test {
    use crate::{
        field::JoltField,
        host,
        jolt::vm::{
            rv32i_vm::{RV32IJoltVM, RV32ISubtables, C, M, RV32I},
            Jolt, JoltPreprocessing, JoltProof, JoltStuff, ProverDebugInfo,
        },
        parse::{
            spartan1::{spartan_hkzg, LinkingStuff1},
            write_json, Parse,
        },
        poly::commitment::{commitment_scheme::CommitmentScheme, hyperkzg::HyperKZG},
        r1cs::inputs::JoltR1CSInputs,
        utils::{poseidon_transcript::PoseidonTranscript, transcript::Transcript},
    };
    use common::{
        constants::{MEMORY_OPS_PER_INSTRUCTION, RAM_START_ADDRESS, REGISTER_COUNT},
        rv_trace::NUM_CIRCUIT_FLAGS,
    };
    use serde_json::json;
    use std::{
        env,
        fs::read_to_string,
        path::Path,
        sync::{LazyLock, Mutex},
    };
    use strum::EnumCount;

    use super::{generate_r1cs, get_paths, read_witness};
    type Fr = ark_bn254::Fr;
    type ProofTranscript = PoseidonTranscript<Fr, Fr>;
    type PCS = HyperKZG<ark_bn254::Bn254, ProofTranscript>;

    // If multiple tests try to read the same trace artifacts simultaneously, they will fail
    static FIB_FILE_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    const NUM_SUBTABLES: usize = RV32ISubtables::<Fr>::COUNT;
    const NUM_INSTRUCTIONS: usize = RV32I::COUNT;
    #[test]
    fn end_to_end_testing() {
        let binding = env::current_dir().unwrap().join("src/parse/requirements");
        let output_dir = binding.to_str().unwrap();

        let jolt_package = "jolt1";
        let combine_r1cs_package = "combined_r1cs";
        let spartan_hyrax_package = "spartan_hyrax";
        let packages = &[jolt_package, combine_r1cs_package, spartan_hyrax_package];

        let file_paths = get_paths(packages);

        let circom_template = "verify";
        let prime = "bn128";
        let (jolt_preprocessing, jolt_proof, jolt_commitments, _debug_info) =
            fib_e2e::<Fr, PCS, ProofTranscript>();
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

        write_json(&jolt1_input, output_dir, packages[0]);

        let jolt1_params: Vec<usize> = get_jolt_args(&jolt_proof, &jolt_preprocessing);
        generate_r1cs(
            &file_paths[0],
            &output_dir,
            circom_template,
            jolt1_params,
            prime,
        );

        // Read the witness.json file
        let witness_file_path = format!("{}/{}_witness.json", output_dir, jolt_package);
        let z = read_witness::<Fr>(&witness_file_path.to_string());

        let hyperkzg_proof = jolt_proof.opening_proof.joint_opening_proof.format();

        let jolt_pi = json!({
            "v_init_final_hash": jolt_preprocessing.bytecode.v_init_final_hash.format_non_native(),
            "bytecode_words_hash": jolt_preprocessing.read_write_memory.hash.format_non_native()
        });

        let bytecode_stuff_size = 6 * 9;
        let read_write_memory_stuff_size = 6 * 13;
        let instruction_lookups_stuff_size = 6 * (C + 3 * 54 + NUM_INSTRUCTIONS + 1); //NUM_MEMORIES = 54
        let timestamp_range_check_stuff_size = 6 * (4 * MEMORY_OPS_PER_INSTRUCTION);
        let aux_variable_stuff_size = 6 * (8 + 4); //RELEVANT_Y_CHUNKS_LEN = 4
        let r1cs_stuff_size = 6 * (4 + 4 + NUM_CIRCUIT_FLAGS) + aux_variable_stuff_size; //CHUNKS_X_SIZE + CHUNKS_Y_SIZE = 4 + 4
        let jolt_stuff_size = bytecode_stuff_size
            + read_write_memory_stuff_size
            + instruction_lookups_stuff_size
            + timestamp_range_check_stuff_size
            + r1cs_stuff_size;

        // Length of public IO of V_{Jolt, 1} including the 1 at index 0.
        // 1 + counter_jolt_1 (1) + linking stuff size (jolt stuff size + 15) + jolt pi size (2).
        let pub_io_len = 1 + 1 + jolt_stuff_size + 15 + 2;

        let linking_stuff = LinkingStuff1::new(jolt_commitments, jolt_stuff_size, &z);

        let linking_stuff_1 = linking_stuff.format_non_native();
        let linking_stuff_2 = linking_stuff.format();

        let vk_jolt_2 = jolt_preprocessing.generators.1.format();
        let vk_jolt_2_nn = jolt_preprocessing.generators.1.format_non_native();
        let jolt_openining_point_len = jolt_proof.opening_proof.joint_opening_proof.com.len() + 1;

        println!("Running Spartan Hyperkzg");
        spartan_hkzg(
            jolt_pi,
            linking_stuff_1,
            linking_stuff_2,
            vk_jolt_2,
            vk_jolt_2_nn,
            hyperkzg_proof,
            pub_io_len,
            jolt_stuff_size,
            jolt_openining_point_len,
            &file_paths,
            packages,
            &z,
            output_dir,
        );
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

    fn get_jolt_args(
        proof: &JoltProof<
            C,
            M,
            JoltR1CSInputs,
            Fr,
            PCS,
            RV32I,
            RV32ISubtables<Fr>,
            ProofTranscript,
        >,
        preprocessing: &JoltPreprocessing<C, Fr, PCS, ProofTranscript>,
    ) -> Vec<usize> {
        let mut verify_args = vec![
            preprocessing.bytecode.v_init_final[0].len(),
            preprocessing.read_write_memory.bytecode_words.len(),
            proof.program_io.inputs.len(),
            proof.program_io.outputs.len(),
            proof.bytecode.multiset_hashes.read_hashes.len(),
            proof.bytecode.multiset_hashes.init_hashes.len(),
            proof.bytecode.read_write_grand_product.gkr_layers.len(),
            proof.bytecode.init_final_grand_product.gkr_layers.len(),
            proof.bytecode.read_write_grand_product.gkr_layers
                [proof.bytecode.read_write_grand_product.gkr_layers.len() - 1]
                .proof
                .uni_polys
                .len(),
            proof
                .read_write_memory
                .memory_checking_proof
                .read_write_grand_product
                .gkr_layers[proof
                .read_write_memory
                .memory_checking_proof
                .read_write_grand_product
                .gkr_layers
                .len()
                - 1]
            .proof
            .uni_polys
            .len(),
            proof
                .read_write_memory
                .memory_checking_proof
                .init_final_grand_product
                .gkr_layers[proof
                .read_write_memory
                .memory_checking_proof
                .init_final_grand_product
                .gkr_layers
                .len()
                - 1]
            .proof
            .uni_polys
            .len(),
            proof
                .read_write_memory
                .memory_checking_proof
                .multiset_hashes
                .read_hashes
                .len(),
            proof
                .read_write_memory
                .memory_checking_proof
                .multiset_hashes
                .init_hashes
                .len(),
            proof
                .read_write_memory
                .memory_checking_proof
                .read_write_grand_product
                .gkr_layers
                .len(),
            proof
                .read_write_memory
                .memory_checking_proof
                .init_final_grand_product
                .gkr_layers
                .len(),
            proof
                .read_write_memory
                .timestamp_validity_proof
                .batched_grand_product
                .gkr_layers[proof
                .read_write_memory
                .timestamp_validity_proof
                .batched_grand_product
                .gkr_layers
                .len()
                - 1]
            .proof
            .uni_polys
            .len(),
            proof
                .read_write_memory
                .timestamp_validity_proof
                .batched_grand_product
                .gkr_layers
                .len(),
            proof
                .read_write_memory
                .timestamp_validity_proof
                .multiset_hashes
                .read_hashes
                .len(),
            proof
                .read_write_memory
                .timestamp_validity_proof
                .multiset_hashes
                .init_hashes
                .len(),
            proof
                .read_write_memory
                .timestamp_validity_proof
                .exogenous_openings
                .len(),
            proof.read_write_memory.output_proof.num_rounds,
            proof
                .instruction_lookups
                .memory_checking
                .read_write_grand_product
                .gkr_layers[proof
                .instruction_lookups
                .memory_checking
                .read_write_grand_product
                .gkr_layers
                .len()
                - 1]
            .proof
            .uni_polys
            .len(),
            proof
                .instruction_lookups
                .memory_checking
                .init_final_grand_product
                .gkr_layers[proof
                .instruction_lookups
                .memory_checking
                .init_final_grand_product
                .gkr_layers
                .len()
                - 1]
            .proof
            .uni_polys
            .len(),
            proof
                .instruction_lookups
                .primary_sumcheck
                .sumcheck_proof
                .uni_polys[0]
                .coeffs
                .len()
                - 1,
            proof.instruction_lookups.primary_sumcheck.num_rounds,
            preprocessing.instruction_lookups.num_memories,
            NUM_INSTRUCTIONS,
            NUM_SUBTABLES,
            proof
                .instruction_lookups
                .memory_checking
                .read_write_grand_product
                .gkr_layers
                .len(),
            proof
                .instruction_lookups
                .memory_checking
                .init_final_grand_product
                .gkr_layers
                .len(),
            proof.r1cs.outer_sumcheck_proof.uni_polys.len(),
            proof.r1cs.inner_sumcheck_proof.uni_polys.len(),
            proof.opening_proof.sumcheck_proof.uni_polys.len(),
            proof.r1cs.claimed_witness_evals.len(),
            proof.opening_proof.sumcheck_claims.len(),
            32, // Word Size,
            C,
            4, // chunks_x_size,
            4, // chunks_y_size,
            NUM_CIRCUIT_FLAGS,
            4, //relevant_y_chunks_len,
            (1 << 16),
            REGISTER_COUNT as usize,
            preprocessing
                .read_write_memory
                .min_bytecode_address
                .try_into()
                .unwrap(),
            RAM_START_ADDRESS as usize,
            preprocessing.memory_layout.input_start as usize,
            preprocessing.memory_layout.output_start as usize,
            preprocessing.memory_layout.panic as usize,
            preprocessing.memory_layout.termination as usize,
            (proof.program_io.panic as u8) as usize,
        ];
        let binding = env::current_dir().unwrap().join("src/parse/requirements");
        let file_name = format!("{}", "args.txt");
        let file_path = Path::new(&binding).join(&file_name);

        let content = read_to_string(file_path).unwrap();

        let mut values = content
            .split(',')
            .map(|s| s.trim().parse::<usize>().unwrap());

        let num_steps = values.next().unwrap();
        let num_cons_total = values.next().unwrap();
        let num_vars = values.next().unwrap();
        let num_cols = values.next().unwrap();
        verify_args.push(num_steps);
        verify_args.push(num_cons_total);
        verify_args.push(num_vars);
        verify_args.push(num_cols);
        verify_args.push(preprocessing.memory_layout.max_output_size as usize);
        verify_args.push(preprocessing.memory_layout.max_input_size as usize);

        verify_args
    }
}

pub(crate) fn generate_r1cs(
    circom_file_path: &PathBuf,
    output_dir: &str,
    circom_template: &str,
    params: Vec<usize>,
    prime: &str,
) {
    let circom_file_path = circom_file_path.to_str().unwrap();
    let circom_file_name = Path::new(circom_file_path)
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap();

    let backup_file = format!("{}.bak", circom_file_path);
    println!("circom file name is {:?}", circom_file_name);
    // Backup original file
    fs::copy(circom_file_path, &backup_file).expect("Failed to create backup");

    // Add component main to Circom file
    let args_string = params
        .iter()
        .map(|p| p.to_string())
        .collect::<Vec<_>>()
        .join(",");

    let component_line = if args_string.is_empty() {
        format!("\ncomponent main = {}();", circom_template)
    } else {
        format!("\ncomponent main = {}({});", circom_template, args_string)
    };

    let mut circom_file = fs::OpenOptions::new()
        .append(true)
        .open(circom_file_path)
        .expect("Failed to open Circom file");

    circom_file
        .write_all(component_line.as_bytes())
        .expect("Failed to write to Circom file");

    // Compile Circom file
    let output = Command::new("circom")
        .args([
            circom_file_path,
            "--wasm",
            "--json",
            "--prime",
            prime,
            "--O2",
            "--output",
            output_dir,
        ])
        .stderr(Stdio::piped())
        .output()
        .expect("Failed to execute Circom compilation");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        println!("Circom compilation failed with error:\n{}", stderr);
        fs::rename(&backup_file, circom_file_path).expect("Failed to restore Circom file");
        panic!("Circom compilation failed");
    } else {
        fs::rename(&backup_file, circom_file_path).expect("Failed to restore Circom file");
    }

    println!("Circom compilation successful.");

    let js_dir = format!("{}/{}_js", output_dir, circom_file_name);

    // Generate witness
    let wasm_file = format!("{}/{}.wasm", js_dir, circom_file_name);

    let witness_output = format!("{}/{}_witness.wtns", output_dir, circom_file_name);
    let input_path = format!("{}/{}_input.json", output_dir, circom_file_name,);

    let witness_status = Command::new("node")
        .args([
            &format!("{}/generate_witness.js", js_dir),
            &wasm_file,
            &input_path,
            &witness_output,
        ])
        .status()
        .expect("Failed to execute witness generation");

    if !witness_status.success() {
        panic!("Witness generation failed");
    }

    let witness_file_path = format!("{}/{}_witness.json", output_dir, circom_file_name);

    let export_witnes = Command::new("snarkjs")
        .args([
            "wtns",
            "export",
            "json",
            &witness_output,
            &witness_file_path,
        ])
        .status()
        .expect("Failed to execute export witness");
    if !export_witnes.success() {
        panic!("Failed to convert wtns into json");
    }
}
