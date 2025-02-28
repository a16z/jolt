
pragma circom 2.2.1;
// include "./../utils.circom"

bus MultisetHashes(num_read_write_hashes, num_init_hashes, num_final_hashes) {
    signal read_hashes[num_read_write_hashes];
    signal write_hashes[num_read_write_hashes];
    signal init_hashes[num_init_hashes];
    signal final_hashes[num_final_hashes];
}



bus JoltPreprocessing() { 
    signal v_init_final_hash;
    signal bytecode_words_hash;
}

bus PIProof(num_evals, bytecode_words_size) {

    BytecodePreprocessing( num_evals) bytecode;

    ReadWriteMemoryPreprocessing(bytecode_words_size) read_write_memory;

}

bus JoltDevice(input_size, output_size) {

    signal inputs[input_size];

    signal outputs[output_size];

    signal panic; 
}

bus JoltProof(input_size, 
              output_size, 
              num_read_write_hashes_bytecode,
              num_init_final_hashes_bytecode,
              read_write_grand_product_layers_bytecode,
              init_final_grand_product_layers_bytecode,
              max_rounds_bytecode, 
            
              max_rounds_read_write,max_rounds_init_final, num_read_write_hashes_read_write_memory_checking, 
              num_init_final_hashes_read_write_memory_checking,
              read_write_grand_product_layers_read_write_memory_checking,
              init_final_grand_product_layers_read_write_memory_checking,
              max_rounds_timestamp,
              ts_validity_grand_product_layers_timestamp,
              num_read_write_hashes_timestamp,
              num_init_hashes_timestamp,
              MEMORY_OPS_PER_INSTRUCTION,
              max_rounds_outputsumcheck,

              max_rounds_instruction_lookups,
              max_round_init_final_lookups,
              primary_sumcheck_degree_instruction_lookups, 
              primary_sumcheck_num_rounds_instruction_lookups, NUM_MEMORIES, NUM_INSTRUCTIONS,   NUM_SUBTABLES,
              read_write_grand_product_layers_instruction_lookups, init_final_grand_product_layers_instruction_lookups,

              outer_num_rounds_uniform_spartan_proof, inner_num_rounds_uniform_spartan_proof,
              rounds_reduced_opening_proof,
              num_spartan_witness_evals, 
              num_sumcheck_claims           
              ) {

    signal trace_length;

    JoltDevice(input_size, output_size) program_io;

    BytecodeProof(num_read_write_hashes_bytecode, 
                    num_init_final_hashes_bytecode,
                    read_write_grand_product_layers_bytecode,
                    init_final_grand_product_layers_bytecode,max_rounds_bytecode) bytecode;

    ReadWriteMemoryProof(max_rounds_read_write, max_rounds_init_final, num_read_write_hashes_read_write_memory_checking, 
                    num_init_final_hashes_read_write_memory_checking,
                    read_write_grand_product_layers_read_write_memory_checking,
                    init_final_grand_product_layers_read_write_memory_checking,
                    max_rounds_timestamp,
                    ts_validity_grand_product_layers_timestamp,
                    num_read_write_hashes_timestamp,
                    num_init_hashes_timestamp,
                    MEMORY_OPS_PER_INSTRUCTION,
                    max_rounds_outputsumcheck) read_write_memory;

    InstructionLookupsProof(max_rounds_instruction_lookups, max_round_init_final_lookups,  primary_sumcheck_degree_instruction_lookups, primary_sumcheck_num_rounds_instruction_lookups, NUM_MEMORIES, NUM_INSTRUCTIONS, NUM_SUBTABLES,
                    read_write_grand_product_layers_instruction_lookups, init_final_grand_product_layers_instruction_lookups) instruction_lookups;

    UniformSpartanProof(outer_num_rounds_uniform_spartan_proof, inner_num_rounds_uniform_spartan_proof, num_spartan_witness_evals) r1cs;

    ReducedOpeningProof(rounds_reduced_opening_proof , num_sumcheck_claims) opening_proof;

    // PIProof( num_evals, bytecode_words_size) pi_proof;
}


bus JoltStuff(C, 
              NUM_MEMORIES, 
              NUM_INSTRUCTIONS, 
              MEMORY_OPS_PER_INSTRUCTION,
              chunks_x_size, 
              chunks_y_size, 
              NUM_CIRCUIT_FLAGS, 
              relevant_y_chunks_len) {

    BytecodeStuff() bytecode;

    ReadWriteMemoryStuff() read_write_memory;

    InstructionLookupStuff(C, 
                            NUM_MEMORIES, 
                            NUM_INSTRUCTIONS
                            ) instruction_lookups;

    TimestampRangeCheckStuff(MEMORY_OPS_PER_INSTRUCTION) timestamp_range_check;

    R1CSStuff(chunks_x_size, 
              chunks_y_size, 
              NUM_CIRCUIT_FLAGS, 
              relevant_y_chunks_len) r1cs;
}



bus BytecodeStuff() {
    HyperKZGCommitment() a_read_write;
    
    HyperKZGCommitment() v_read_write[6];

    HyperKZGCommitment() t_read;
    
    HyperKZGCommitment() t_final;

}

bus ReadWriteMemoryStuff() {
    HyperKZGCommitment() a_ram;

    HyperKZGCommitment() v_read_rd;

    HyperKZGCommitment() v_read_rs1;
    
    HyperKZGCommitment() v_read_rs2;

    HyperKZGCommitment() v_read_ram;

    HyperKZGCommitment() v_write_rd;
    
    HyperKZGCommitment() v_write_ram;
    
    HyperKZGCommitment() v_final;
    
    HyperKZGCommitment() t_read_rd;

    HyperKZGCommitment() t_read_rs1;
    
    HyperKZGCommitment() t_read_rs2;
    
    HyperKZGCommitment() t_read_ram;
   
    HyperKZGCommitment() t_final;

}


bus TimestampRangeCheckStuff(MEMORY_OPS_PER_INSTRUCTION) {

    HyperKZGCommitment() read_cts_read_timestamp[MEMORY_OPS_PER_INSTRUCTION];

    HyperKZGCommitment() read_cts_global_minus_read[MEMORY_OPS_PER_INSTRUCTION];

    HyperKZGCommitment() final_cts_read_timestamp[MEMORY_OPS_PER_INSTRUCTION];

    HyperKZGCommitment() final_cts_global_minus_read[MEMORY_OPS_PER_INSTRUCTION];


}



bus R1CSStuff(chunks_x_size, 
              chunks_y_size, 
              NUM_CIRCUIT_FLAGS, 
              relevant_y_chunks_len) {

    HyperKZGCommitment() chunks_x[chunks_x_size];

    HyperKZGCommitment() chunks_y[chunks_y_size];

    HyperKZGCommitment() circuit_flags[NUM_CIRCUIT_FLAGS];

    AuxVariableStuff(relevant_y_chunks_len) aux;
}

bus AuxVariableStuff(relevant_y_chunks_len) {

    HyperKZGCommitment() left_lookup_operand;

    HyperKZGCommitment() right_lookup_operand;

    HyperKZGCommitment() product;

    HyperKZGCommitment() relevant_y_chunks[relevant_y_chunks_len];

    HyperKZGCommitment() write_lookup_output_to_rd;

    HyperKZGCommitment() write_pc_to_rd;

    HyperKZGCommitment() next_pc_jump;

    HyperKZGCommitment() should_branch;

    HyperKZGCommitment() next_pc;
}


bus InstructionLookupStuff(C, NUM_MEMORIES, NUM_INSTRUCTIONS) {
    
    /// `C`-sized vector of polynomials/commitments/openings corresponding to the
    /// indices at which subtables are queried.
    HyperKZGCommitment() dim[C];

    /// `num_memories`-sized vector of polynomials/commitments/openings corresponding to
    /// the read access counts for each memory.
    HyperKZGCommitment() read_cts[NUM_MEMORIES];

    /// `num_memories`-sized vector of polynomials/commitments/openings corresponding to
    /// the final access counts for each memory.
    HyperKZGCommitment() final_cts[NUM_MEMORIES];

    /// `num_memories`-sized vector of polynomials/commitments/openings corresponding to
    /// the values read from each memory.
    HyperKZGCommitment() E_polys[NUM_MEMORIES];

    /// `NUM_INSTRUCTIONS`-sized vector of polynomials/commitments/openings corresponding
    /// to the indicator bitvectors designating which lookup to perform at each step of
    /// the execution trace.
    HyperKZGCommitment() instruction_flags[NUM_INSTRUCTIONS];

    /// The polynomial/commitment/opening corresponding to the lookup output for each
    /// step of the execution trace.
    HyperKZGCommitment() lookup_outputs;

}
