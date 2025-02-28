pragma circom 2.2.1;

bus LinkingStuff1NN(
                    C, 
                    NUM_MEMORIES, 
                    NUM_INSTRUCTIONS, 
                    MEMORY_OPS_PER_INSTRUCTION,
                    chunks_x_size, 
                    chunks_y_size, 
                    NUM_CIRCUIT_FLAGS, 
                    relevant_y_chunks_len
                    ) {
    JoltStuffNN(
                    C, 
                    NUM_MEMORIES, 
                    NUM_INSTRUCTIONS, 
                    MEMORY_OPS_PER_INSTRUCTION,
                    chunks_x_size, 
                    chunks_y_size, 
                    NUM_CIRCUIT_FLAGS,
                    relevant_y_chunks_len
                    ) commitments;


    OpeningCombinersNN() opening_combiners;

    HyperKzgVerifierAdviceNN() hyperkzg_verifier_advice;
 
}

bus JoltStuffNN(C, 
              NUM_MEMORIES, 
              NUM_INSTRUCTIONS, 
              MEMORY_OPS_PER_INSTRUCTION,
              chunks_x_size, 
              chunks_y_size, 
              NUM_CIRCUIT_FLAGS, 
              relevant_y_chunks_len) {

    BytecodeStuffNN() bytecode;

    ReadWriteMemoryStuffNN() read_write_memory;

    InstructionLookupStuffNN(C, 
                            NUM_MEMORIES, 
                            NUM_INSTRUCTIONS
                            ) instruction_lookups;

    TimestampRangeCheckStuffNN(MEMORY_OPS_PER_INSTRUCTION) timestamp_range_check;

    R1CSStuffNN(chunks_x_size, 
              chunks_y_size, 
              NUM_CIRCUIT_FLAGS, 
              relevant_y_chunks_len) r1cs;
}

bus BytecodeStuffNN() {
    HyperKZGCommitmentNN() a_read_write;
    
    HyperKZGCommitmentNN() v_read_write[6];

    HyperKZGCommitmentNN() t_read;
    
    HyperKZGCommitmentNN() t_final;
}

bus HyperKZGCommitmentNN() {
    G1AffineNN() commitment;
}

bus ReadWriteMemoryStuffNN() {
    HyperKZGCommitmentNN() a_ram;

    HyperKZGCommitmentNN() v_read_rd;

    HyperKZGCommitmentNN() v_read_rs1;
    
    HyperKZGCommitmentNN() v_read_rs2;

    HyperKZGCommitmentNN() v_read_ram;

    HyperKZGCommitmentNN() v_write_rd;
    
    HyperKZGCommitmentNN() v_write_ram;
    
    HyperKZGCommitmentNN() v_final;
    
    HyperKZGCommitmentNN() t_read_rd;

    HyperKZGCommitmentNN() t_read_rs1;
    
    HyperKZGCommitmentNN() t_read_rs2;
    
    HyperKZGCommitmentNN() t_read_ram;
   
    HyperKZGCommitmentNN() t_final;

}

bus InstructionLookupStuffNN(C, 
                            NUM_MEMORIES, 
                            NUM_INSTRUCTIONS
                         //    instruction_flag_bitvectors_rows, 
                         //    instruction_flag_bitvectors_cols
                            ) {
    
    /// `C`-sized vector of polynomials/commitments/openings corresponding to the
    /// indices at which subtables are queried.
    HyperKZGCommitmentNN() dim[C];

    /// `num_memories`-sized vector of polynomials/commitments/openings corresponding to
    /// the read access counts for each memory.
    HyperKZGCommitmentNN() read_cts[NUM_MEMORIES];

    /// `num_memories`-sized vector of polynomials/commitments/openings corresponding to
    /// the final access counts for each memory.
    HyperKZGCommitmentNN() final_cts[NUM_MEMORIES];

    /// `num_memories`-sized vector of polynomials/commitments/openings corresponding to
    /// the values read from each memory.
    HyperKZGCommitmentNN() E_polys[NUM_MEMORIES];

    /// `NUM_INSTRUCTIONS`-sized vector of polynomials/commitments/openings corresponding
    /// to the indicator bitvectors designating which lookup to perform at each step of
    /// the execution trace.
    HyperKZGCommitmentNN() instruction_flags[NUM_INSTRUCTIONS];

    /// The polynomial/commitment/opening corresponding to the lookup output for each
    /// step of the execution trace.
    HyperKZGCommitmentNN() lookup_outputs;


}

bus TimestampRangeCheckStuffNN(MEMORY_OPS_PER_INSTRUCTION) {

    HyperKZGCommitmentNN() read_cts_read_timestamp[MEMORY_OPS_PER_INSTRUCTION];

    HyperKZGCommitmentNN() read_cts_global_minus_read[MEMORY_OPS_PER_INSTRUCTION];

    HyperKZGCommitmentNN() final_cts_read_timestamp[MEMORY_OPS_PER_INSTRUCTION];

    HyperKZGCommitmentNN() final_cts_global_minus_read[MEMORY_OPS_PER_INSTRUCTION];


}

bus R1CSStuffNN(chunks_x_size, 
              chunks_y_size, 
              NUM_CIRCUIT_FLAGS, 
              relevant_y_chunks_len) {

    HyperKZGCommitmentNN() chunks_x[chunks_x_size];

    HyperKZGCommitmentNN() chunks_y[chunks_y_size];

    HyperKZGCommitmentNN() circuit_flags[NUM_CIRCUIT_FLAGS];

    AuxVariableStuffNN(relevant_y_chunks_len) aux;
}

bus AuxVariableStuffNN(relevant_y_chunks_len) {

    HyperKZGCommitmentNN() left_lookup_operand;

    HyperKZGCommitmentNN() right_lookup_operand;

    HyperKZGCommitmentNN() product;

    HyperKZGCommitmentNN() relevant_y_chunks[relevant_y_chunks_len];

    HyperKZGCommitmentNN() write_lookup_output_to_rd;

    HyperKZGCommitmentNN() write_pc_to_rd;

    HyperKZGCommitmentNN() next_pc_jump;

    HyperKZGCommitmentNN() should_branch;

    HyperKZGCommitmentNN() next_pc;
}

bus OpeningCombinersNN() {

    BytecodeCombinersNN()     bytecode_combiners;
  
    InstructionLookupCombinersNN() instruction_lookup_combiners;

    ReadWriteOutputTimestampCombinersNN() read_write_output_timestamp_combiners;

    SpartanCombinersNN() spartan_combiners;

    Fq() coefficient;


}

bus BytecodeCombinersNN() {
    Fq() rho[2] ;
}

bus InstructionLookupCombinersNN() {
    Fq() rho[3] ;
}

bus ReadWriteOutputTimestampCombinersNN() {
    Fq() rho[4] ;
}

bus SpartanCombinersNN() {
    Fq() rho ;
}

bus HyperKzgVerifierAdviceNN() {
   Fq()  r;
   Fq()  d_0;
   Fq()  v;
   Fq()  q_power;
}

bus JoltPreprocessingNN() {
    Fq() v_init_final_hash;
    Fq() bytecode_words_hash;
}

// bus Fq() {
//     // signal element;
//     signal limbs[3];
// }

// template test() {
//     input LinkingStuffNN(1, 2, 3, 4, 5,6,7,8,9) ls;
// }

// component main = test();
