use std::{fs::File, io::Read, process::Command};
// use matrix::check_inner_product_sparse;
use serde_json::Value;

#[test]
fn test_parsing(){
    let input_file_path = "input.json";
    let script_path = "./../../../scripts/compile_and_generate_witness_bn_scalar.sh";
    let circom_template = "verify";
    let circom_file_path = "./jolt1.circom";
    let circom_file = "jolt1";
    let params: [usize; 56] = [ 1 << 9, 293, 1, 1, 1, 1, 9, 9, 9,  // num_evals,bytecode_words_size, input_size, output_size, num_read_write_hashes_bytecode,num_init_final_hashes_bytecode,read_write_grand_product_layers_bytecode,init_final_grand_product_layers_bytecode,max_rounds_bytecode, 
        11, 13, 4,1,9,13,  // max_rounds_read_write,num_read_write_hashes_read_write_memory_checking, num_init_final_hashes_read_write_memory_checking,read_write_grand_product_layers_read_write_memory_checking,init_final_grand_product_layers_read_write_memory_checking, 
        13, 9, 8, 1,4, 
        13,  // max_rounds_timestamp,ts_validity_grand_product_layers_timestamp,num_read_write_hashes_timestamp,num_init_hashes_timestamp,MEMORY_OPS_PER_INSTRUCTION,max_rounds_outputsumcheck,
        16, 22, 8, 9, 54, 26, 26, 10, 16, // max_rounds_instruction_lookups,primary_sumcheck_degree_instruction_lookups, primary_sumcheck_num_rounds_instruction_lookups, NUM_MEMORIES, NUM_INSTRUCTIONS,   NUM_SUBTABLES,read_write_grand_product_layers_instruction_lookups, init_final_grand_product_layers_instruction_lookups,
        16, 17, //  outer_num_rounds_uniform_spartan_proof, inner_num_rounds_uniform_spartan_proof,
        16, 76, 10, //rounds_reduced_opening_proof, num_spartan_witness_evals, num_sumcheck_claims,
    
    //  WORD_SIZE, C, chunks_y_size, chunks_x_size, NUM_CIRCUIT_FLAGS, relevant_y_chunks_len,M,
        32, 4, 4, 4, 11, 4, 1 << 16, 

    // REGISTER_COUNT, min_bytecode_address,RAM_START_ADDRESS,,memory_layout_input_start,memory_layout_output_start,memory_layout_panic,memory_layout_termination,program_io_panic,
        64, 2147483648, 2147483648, 2147467520, 2147471616,2147475712, 2147475716, 0,
    //  num_steps, num_cons_total, num_vars, num_rows
        512, 65536, 76, 67, 4096, 4096]; // spartan;

    let mut args = [circom_file_path,
    circom_template,
    input_file_path,
    circom_file].to_vec();

    let mut param_strings: Vec<String> = Vec::new();
    for i in 0..params.len() {
        param_strings.push(params[i].to_string());
    }
    for param_str in &param_strings {
        args.push(param_str);
    }

    let output = Command::new(script_path)
    .args(&args)
    .output()
    .expect("Failed to execute shell script");

    if !output.status.success() {
        panic!(
            "Shell script execution failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    println!(
        "Shell script output: {}",
        String::from_utf8_lossy(&output.stdout)
    );

    // Step 4: Verify witness generation
    println!("Witness generated successfully.");

    // Read the witness.json file
    let witness_file_path = "witness.json"; // Make sure this is the correct path to your witness.json
    let mut witness_file = File::open(witness_file_path).expect("Failed to open witness.json");

    // Read the file contents
    let mut witness_contents = String::new();
    witness_file
        .read_to_string(&mut witness_contents)
        .expect("Failed to read witness.json");

    // Parse the JSON contents
    let witness_json: Value =
        serde_json::from_str(&witness_contents).expect("Failed to parse witness.json");

//  check_inner_product_sparse(vec![circom_file.to_string()]);

}

