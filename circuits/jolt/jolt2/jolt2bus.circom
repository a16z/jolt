pragma circom 2.2.1;

bus LinkingStuff2(C, 
                    NUM_MEMORIES, 
                    NUM_INSTRUCTIONS, 
                    MEMORY_OPS_PER_INSTRUCTION,
                    chunks_x_size, 
                    chunks_y_size, 
                    NUM_CIRCUIT_FLAGS, 
                    relevant_y_chunks_len,rounds_reduced_opening_proof) {

    OpeningCombiners() opening_combiners;


    HyperKzgVerifierAdvice() hyperkzg_verifier_advice;


    JoltStuff(C, 
                NUM_MEMORIES, 
                NUM_INSTRUCTIONS, 
                MEMORY_OPS_PER_INSTRUCTION,
                chunks_x_size, 
                chunks_y_size, 
                NUM_CIRCUIT_FLAGS, 
                relevant_y_chunks_len) commitments;

}

bus HyperKzgVerifierAdvice() {
     Fq()  r;
     Fq()  d_0;
     Fq()  v;
     Fq()  q_power;
}

bus OpeningCombiners() {

    BytecodeCombiners() bytecode_combiners;
  
    InstructionLookupCombiners() instruction_lookup_combiners;

    ReadWriteOutputTimestampCombiners() read_write_output_timestamp_combiners;

    SpartanCombiners() spartan_combiners;

    Fq()  coefficient;

}


bus BytecodeCombiners() {
    Fq() rho[2] ;
    
}

bus InstructionLookupCombiners() {
    Fq() rho[3] ;
  
}

bus ReadWriteOutputTimestampCombiners() {
    Fq() rho[4] ; 
}

bus SpartanCombiners() {
    Fq() rho ;
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


bus InstructionLookupStuff(C, 
                            NUM_MEMORIES, 
                            NUM_INSTRUCTIONS
                            ) {
    
    HyperKZGCommitment() dim[C];
    
    HyperKZGCommitment() read_cts[NUM_MEMORIES];
   
    HyperKZGCommitment() final_cts[NUM_MEMORIES];

    HyperKZGCommitment() E_polys[NUM_MEMORIES];

    HyperKZGCommitment() instruction_flags[NUM_INSTRUCTIONS];

    HyperKZGCommitment() lookup_outputs;


}

