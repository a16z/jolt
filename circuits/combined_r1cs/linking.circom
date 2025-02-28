pragma circom 2.2.1;
include "./../jolt/jolt2/jolt2bus.circom";
include "./../pcs/hyperkzg_utils.circom";

template Linking(
                    C, 
                    NUM_MEMORIES, 
                    NUM_INSTRUCTIONS, 
                    MEMORY_OPS_PER_INSTRUCTION,
                    chunks_x_size, 
                    chunks_y_size, 
                    NUM_CIRCUIT_FLAGS, 
                    relevant_y_chunks_len,
                    rounds_reduced_opening_proof
) {
    input LinkingStuff1NN(
                    C, 
                    NUM_MEMORIES, 
                    NUM_INSTRUCTIONS, 
                    MEMORY_OPS_PER_INSTRUCTION,
                    chunks_x_size, 
                    chunks_y_size, 
                    NUM_CIRCUIT_FLAGS, 
                    relevant_y_chunks_len
    ) linking_stuff_1;

    input LinkingStuff2(
                    C, 
                    NUM_MEMORIES, 
                    NUM_INSTRUCTIONS, 
                    MEMORY_OPS_PER_INSTRUCTION,
                    chunks_x_size, 
                    chunks_y_size, 
                    NUM_CIRCUIT_FLAGS, 
                    relevant_y_chunks_len,
                    rounds_reduced_opening_proof
    ) linking_stuff_2;

    CompareJoltStuff(
                    C, 
                    NUM_MEMORIES, 
                    NUM_INSTRUCTIONS, 
                    MEMORY_OPS_PER_INSTRUCTION,
                    chunks_x_size, 
                    chunks_y_size, 
                    NUM_CIRCUIT_FLAGS, 
                    relevant_y_chunks_len
    )(linking_stuff_1.commitments, linking_stuff_2.commitments);

    CompareOpeningCombiners()(linking_stuff_1.opening_combiners, linking_stuff_2.opening_combiners);
    CompareHyperKZGVerifierAdvice()(linking_stuff_1.hyperkzg_verifier_advice, linking_stuff_2.hyperkzg_verifier_advice);
}

template CompareOpeningCombiners {
    input OpeningCombinersNN() combiners_1;
    input OpeningCombiners() combiners_2;

    for (var i = 0; i < 2; i++) {
        combiners_1.bytecode_combiners.rho[i] === combiners_2.bytecode_combiners.rho[i];
    }

    for (var i = 0; i < 3; i++) {
        combiners_1.instruction_lookup_combiners.rho[i] === combiners_2.instruction_lookup_combiners.rho[i];
    }

    for (var i = 0; i < 4; i++) {
        combiners_1.read_write_output_timestamp_combiners.rho[i] === combiners_2.read_write_output_timestamp_combiners.rho[i];
    }

    combiners_1.spartan_combiners.rho === combiners_2.spartan_combiners.rho; 

    combiners_1.coefficient === combiners_2.coefficient;
} 

template CompareHyperKZGVerifierAdvice {
    input HyperKzgVerifierAdviceNN() advice_1;
    input HyperKzgVerifierAdvice() advice_2;

    advice_1.r === advice_2.r;
    advice_1.d_0 === advice_2.d_0;
    advice_1.v === advice_2.v;
    advice_1.q_power === advice_2.q_power;
} 

template CompareJoltStuff(
                    C, 
                    NUM_MEMORIES, 
                    NUM_INSTRUCTIONS, 
                    MEMORY_OPS_PER_INSTRUCTION,
                    chunks_x_size, 
                    chunks_y_size, 
                    NUM_CIRCUIT_FLAGS, 
                    relevant_y_chunks_len
) {
    input JoltStuffNN(
                    C, 
                    NUM_MEMORIES, 
                    NUM_INSTRUCTIONS, 
                    MEMORY_OPS_PER_INSTRUCTION,
                    chunks_x_size, 
                    chunks_y_size, 
                    NUM_CIRCUIT_FLAGS, 
                    relevant_y_chunks_len
    ) jolt_stuff_1;

    input JoltStuff(
                    C, 
                    NUM_MEMORIES, 
                    NUM_INSTRUCTIONS, 
                    MEMORY_OPS_PER_INSTRUCTION,
                    chunks_x_size, 
                    chunks_y_size, 
                    NUM_CIRCUIT_FLAGS, 
                    relevant_y_chunks_len
    ) jolt_stuff_2;

    compareBytecodeStuff()(jolt_stuff_1.bytecode, jolt_stuff_2.bytecode);
    compareReadWriteMemoryStuff()(jolt_stuff_1.read_write_memory, jolt_stuff_2.read_write_memory);
    compareInstructionLookupStuff(
        C, 
        NUM_MEMORIES, 
        NUM_INSTRUCTIONS
    )(jolt_stuff_1.instruction_lookups, jolt_stuff_2.instruction_lookups);
    compareTimestampRangeCheckStuff(MEMORY_OPS_PER_INSTRUCTION)(jolt_stuff_1.timestamp_range_check, jolt_stuff_2.timestamp_range_check);

    compareR1CSStuff(
            chunks_x_size, 
            chunks_y_size, 
            NUM_CIRCUIT_FLAGS, 
            relevant_y_chunks_len
    )(jolt_stuff_1.r1cs, jolt_stuff_2.r1cs);
}

template compareBytecodeStuff() {
    input BytecodeStuffNN() bc1;
    input BytecodeStuff() bc2;

    compareHyperKZGCommitment()(bc1.a_read_write, bc2.a_read_write);
    
    for(var i = 0; i < 6; i++) {
        compareHyperKZGCommitment()(bc1.v_read_write[i], bc2.v_read_write[i]);
    }

    compareHyperKZGCommitment()(bc1.t_read, bc2.t_read);
    compareHyperKZGCommitment()(bc1.t_final, bc2.t_final);
}

template compareHyperKZGCommitment() {
    input HyperKZGCommitmentNN() h1;
    input HyperKZGCommitment() h2;

    compareG1Affine()(h1.commitment, h2.commitment);
}

template compareG1Affine() {
    input G1AffineNN() ga1;
    input G1Affine() ga2;

    compareFq()(ga1.x, ga2.x);
    compareFq()(ga1.y, ga2.y);
}

template compareReadWriteMemoryStuff() {
    input ReadWriteMemoryStuffNN() rw1;
    input ReadWriteMemoryStuff() rw2;

    compareHyperKZGCommitment()(rw1.a_ram, rw2.a_ram);
    compareHyperKZGCommitment()(rw1.v_read_rd, rw2.v_read_rd);
    compareHyperKZGCommitment()(rw1.v_read_rs1, rw2.v_read_rs1);
    compareHyperKZGCommitment()(rw1.v_read_rs1, rw2.v_read_rs1);
    compareHyperKZGCommitment()(rw1.v_read_ram, rw2.v_read_ram);
    compareHyperKZGCommitment()(rw1.v_write_rd, rw2.v_write_rd);
    compareHyperKZGCommitment()(rw1.v_write_ram, rw2.v_write_ram);
    compareHyperKZGCommitment()(rw1.v_final, rw2.v_final);
    compareHyperKZGCommitment()(rw1.t_read_rd, rw2.t_read_rd);
    compareHyperKZGCommitment()(rw1.t_read_rs1, rw2.t_read_rs1);
    compareHyperKZGCommitment()(rw1.t_read_rs2, rw2.t_read_rs2);
    compareHyperKZGCommitment()(rw1.t_read_ram, rw2.t_read_ram);
    compareHyperKZGCommitment()(rw1.t_final, rw2.t_final);
}

template compareInstructionLookupStuff(
    C, 
    NUM_MEMORIES, 
    NUM_INSTRUCTIONS
) {
    input InstructionLookupStuffNN(
        C,
        NUM_MEMORIES,
        NUM_INSTRUCTIONS
    ) il1;
    input InstructionLookupStuff(
        C,
        NUM_MEMORIES,
        NUM_INSTRUCTIONS
    ) il2;

    for(var i = 0; i < C; i++) {
        compareHyperKZGCommitment()(il1.dim[i], il2.dim[i]);
    }
    for(var i = 0; i < NUM_MEMORIES; i++) {
        compareHyperKZGCommitment()(il1.read_cts[i], il2.read_cts[i]);
    }
    for(var i = 0; i < NUM_MEMORIES; i++) {
        compareHyperKZGCommitment()(il1.final_cts[i], il2.final_cts[i]);
    }
    for(var i = 0; i < NUM_MEMORIES; i++) {
        compareHyperKZGCommitment()(il1.E_polys[i], il2.E_polys[i]);
    }
    for(var i = 0; i < NUM_INSTRUCTIONS; i++) {
        compareHyperKZGCommitment()(il1.instruction_flags[i], il2.instruction_flags[i]);
    }
    compareHyperKZGCommitment()(il1.lookup_outputs, il2.lookup_outputs);
}

template compareTimestampRangeCheckStuff(MEMORY_OPS_PER_INSTRUCTION) {
    input TimestampRangeCheckStuffNN(MEMORY_OPS_PER_INSTRUCTION) tr1;
    input TimestampRangeCheckStuff(MEMORY_OPS_PER_INSTRUCTION) tr2;

    for(var i = 0; i < MEMORY_OPS_PER_INSTRUCTION; i++) {
        compareHyperKZGCommitment()(tr1.read_cts_read_timestamp[i], tr2.read_cts_read_timestamp[i]);
    }
    for(var i = 0; i < MEMORY_OPS_PER_INSTRUCTION; i++) {
        compareHyperKZGCommitment()(tr1.read_cts_global_minus_read[i], tr2.read_cts_global_minus_read[i]);
    }
    for(var i = 0; i < MEMORY_OPS_PER_INSTRUCTION; i++) {
        compareHyperKZGCommitment()(tr1.final_cts_read_timestamp[i], tr2.final_cts_read_timestamp[i]);
    }
    for(var i = 0; i < MEMORY_OPS_PER_INSTRUCTION; i++) {
        compareHyperKZGCommitment()(tr1.final_cts_global_minus_read[i], tr2.final_cts_global_minus_read[i]);
    }
}

template compareR1CSStuff(
    chunks_x_size, 
    chunks_y_size, 
    NUM_CIRCUIT_FLAGS, 
    relevant_y_chunks_len
) {
    input R1CSStuffNN(chunks_x_size, 
              chunks_y_size, 
              NUM_CIRCUIT_FLAGS, 
              relevant_y_chunks_len) r1;
    input R1CSStuff(chunks_x_size, 
              chunks_y_size, 
              NUM_CIRCUIT_FLAGS, 
              relevant_y_chunks_len) r2;

    for(var i = 0; i < chunks_x_size; i++) {
        compareHyperKZGCommitment()(r1.chunks_x[i], r2.chunks_x[i]);
    }
    for(var i = 0; i < chunks_y_size; i++) {
        compareHyperKZGCommitment()(r1.chunks_y[i], r2.chunks_y[i]);
    }
    for(var i = 0; i < NUM_CIRCUIT_FLAGS; i++) {
        compareHyperKZGCommitment()(r1.circuit_flags[i], r2.circuit_flags[i]);
    }

    compareAuxVariableStuff(relevant_y_chunks_len)(r1.aux, r2.aux);
}

template compareAuxVariableStuff(relevant_y_chunks_len) {
    input AuxVariableStuffNN(relevant_y_chunks_len) aux1;
    input AuxVariableStuff(relevant_y_chunks_len) aux2;

    compareHyperKZGCommitment()(aux1.left_lookup_operand, aux2.left_lookup_operand);
    compareHyperKZGCommitment()(aux1.right_lookup_operand, aux2.right_lookup_operand);
    compareHyperKZGCommitment()(aux1.product, aux2.product);

    for(var i = 0; i < relevant_y_chunks_len; i++) {
        compareHyperKZGCommitment()(aux1.relevant_y_chunks[i], aux2.relevant_y_chunks[i]);
    }

    compareHyperKZGCommitment()(aux1.write_lookup_output_to_rd, aux2.write_lookup_output_to_rd);
    compareHyperKZGCommitment()(aux1.write_pc_to_rd, aux2.write_pc_to_rd);
    compareHyperKZGCommitment()(aux1.next_pc_jump, aux2.next_pc_jump);
    compareHyperKZGCommitment()(aux1.should_branch, aux2.should_branch);
    compareHyperKZGCommitment()(aux1.next_pc, aux2.next_pc);

}

template compareFq() {
    input Fq() op1;
    signal input op2;

    signal comb <== op1.limbs[0] + op1.limbs[1] * (1 << 125) + op1.limbs[2] * (1 << 250);
    comb === op2;
}

template LinkingStuff1NNRangeCheck(
    C, 
    NUM_MEMORIES, 
    NUM_INSTRUCTIONS, 
    MEMORY_OPS_PER_INSTRUCTION,
    chunks_x_size, 
    chunks_y_size, 
    NUM_CIRCUIT_FLAGS, 
    relevant_y_chunks_len
) {
    input LinkingStuff1NN(
        C, 
        NUM_MEMORIES, 
        NUM_INSTRUCTIONS, 
        MEMORY_OPS_PER_INSTRUCTION,
        chunks_x_size, 
        chunks_y_size, 
        NUM_CIRCUIT_FLAGS, 
        relevant_y_chunks_len
    ) ls;

    JoltStuffNNRangeCheck(
        C, 
        NUM_MEMORIES, 
        NUM_INSTRUCTIONS, 
        MEMORY_OPS_PER_INSTRUCTION,
        chunks_x_size, 
        chunks_y_size, 
        NUM_CIRCUIT_FLAGS,
        relevant_y_chunks_len        
    )(ls.commitments);

    OpeningCombinersNNRangeCheck()(ls.opening_combiners);

    HyperKzgVerifierAdviceNNRangeCheck()(ls.hyperkzg_verifier_advice);
}

template BytecodeStuffNNRangeCheck() {
    input BytecodeStuffNN() bs;

    HyperKZGCommitmentNNRangeCheck()(bs.a_read_write);
    
    for(var i = 0; i < 6; i++) {
        HyperKZGCommitmentNNRangeCheck()(bs.v_read_write[i]);
    }

    HyperKZGCommitmentNNRangeCheck()(bs.t_read);
    
    HyperKZGCommitmentNNRangeCheck()(bs.t_final);
}

template HyperKZGCommitmentNNRangeCheck() {
    input HyperKZGCommitmentNN() hc;

    G1AffineNNRangeCheck()(hc.commitment);
}

template G1AffineNNRangeCheck() {
    input G1AffineNN() point;

    FqRangeCheck(1)(point.x);
    FqRangeCheck(1)(point.y);
}

template G2AffineNNRangeCheck() {
    input G2AffineNN() point;

    Fp2NNRangeCheck()(point.x);
    Fp2NNRangeCheck()(point.y);
}

template Fp2NNRangeCheck() {
    input Fp2NN() elem;

    FqRangeCheck(1)(elem.x);
    FqRangeCheck(1)(elem.y);
}

template ReadWriteMemoryStuffNNRangeCheck() {
    input ReadWriteMemoryStuffNN() rwm;

    HyperKZGCommitmentNNRangeCheck()(rwm.a_ram);
    HyperKZGCommitmentNNRangeCheck()(rwm.v_read_rd);
    HyperKZGCommitmentNNRangeCheck()(rwm.v_read_rs1);
    HyperKZGCommitmentNNRangeCheck()(rwm.v_read_rs2);
    HyperKZGCommitmentNNRangeCheck()(rwm.v_read_ram);
    HyperKZGCommitmentNNRangeCheck()(rwm.v_write_rd);
    HyperKZGCommitmentNNRangeCheck()(rwm.v_write_ram);
    HyperKZGCommitmentNNRangeCheck()(rwm.v_final);
    HyperKZGCommitmentNNRangeCheck()(rwm.t_read_rd);
    HyperKZGCommitmentNNRangeCheck()(rwm.t_read_rs1);
    HyperKZGCommitmentNNRangeCheck()(rwm.t_read_rs2);
    HyperKZGCommitmentNNRangeCheck()(rwm.t_read_ram);
    HyperKZGCommitmentNNRangeCheck()(rwm.t_final);
}

template InstructionLookupStuffNNRangeCheck(C, 
                        NUM_MEMORIES, 
                        NUM_INSTRUCTIONS
) {
    input InstructionLookupStuffNN(
        C, NUM_MEMORIES, NUM_INSTRUCTIONS
    ) instruction_lookups;

    for(var i = 0; i < C; i++) {
        HyperKZGCommitmentNNRangeCheck()(instruction_lookups.dim[i]);
    }
    for(var i = 0; i < NUM_MEMORIES; i++) {
        HyperKZGCommitmentNNRangeCheck()(instruction_lookups.read_cts[i]);
    }
    for(var i = 0; i < NUM_MEMORIES; i++) {
        HyperKZGCommitmentNNRangeCheck()(instruction_lookups.final_cts[i]);
    }
    for(var i = 0; i < NUM_MEMORIES; i++) {
        HyperKZGCommitmentNNRangeCheck()(instruction_lookups.E_polys[i]);
    }
    for(var i = 0; i < NUM_INSTRUCTIONS; i++) {
        HyperKZGCommitmentNNRangeCheck()(instruction_lookups.instruction_flags[i]);
    }

    HyperKZGCommitmentNNRangeCheck()(instruction_lookups.lookup_outputs);
}

template TimestampRangeCheckStuffNNRangeCheck(MEMORY_OPS_PER_INSTRUCTION) {
    input TimestampRangeCheckStuffNN(MEMORY_OPS_PER_INSTRUCTION) timestamp_range_check;

    for(var i = 0; i < MEMORY_OPS_PER_INSTRUCTION; i++) {
        HyperKZGCommitmentNNRangeCheck()(timestamp_range_check.read_cts_read_timestamp[i]);
    }
    for(var i = 0; i < MEMORY_OPS_PER_INSTRUCTION; i++) {
        HyperKZGCommitmentNNRangeCheck()(timestamp_range_check.read_cts_global_minus_read[i]);
    }
    for(var i = 0; i < MEMORY_OPS_PER_INSTRUCTION; i++) {
        HyperKZGCommitmentNNRangeCheck()(timestamp_range_check.final_cts_read_timestamp[i]);
    }
    for(var i = 0; i < MEMORY_OPS_PER_INSTRUCTION; i++) {
        HyperKZGCommitmentNNRangeCheck()(timestamp_range_check.final_cts_global_minus_read[i]);
    }
}

template R1CSStuffNNRangeCheck(chunks_x_size, 
            chunks_y_size, 
            NUM_CIRCUIT_FLAGS, 
            relevant_y_chunks_len) {
    input R1CSStuffNN(chunks_x_size, 
              chunks_y_size, 
              NUM_CIRCUIT_FLAGS, 
              relevant_y_chunks_len) r1cs;

    for(var i = 0; i < chunks_x_size; i++) {
        HyperKZGCommitmentNNRangeCheck()(r1cs.chunks_x[i]);
    }                
    for(var i = 0; i < chunks_y_size; i++) {
        HyperKZGCommitmentNNRangeCheck()(r1cs.chunks_y[i]);
    }                
    for(var i = 0; i < NUM_CIRCUIT_FLAGS; i++) {
        HyperKZGCommitmentNNRangeCheck()(r1cs.circuit_flags[i]);
    }                
    AuxVariableStuffNNRangeCheck(relevant_y_chunks_len)(r1cs.aux);
}

template AuxVariableStuffNNRangeCheck(relevant_y_chunks_len) {
    input AuxVariableStuffNN(relevant_y_chunks_len) aux;

    HyperKZGCommitmentNNRangeCheck()(aux.left_lookup_operand);
    HyperKZGCommitmentNNRangeCheck()(aux.right_lookup_operand);
    HyperKZGCommitmentNNRangeCheck()(aux.product);

    for(var i = 0; i < relevant_y_chunks_len; i++) {
        HyperKZGCommitmentNNRangeCheck()(aux.relevant_y_chunks[i]);
    }

    HyperKZGCommitmentNNRangeCheck()(aux.write_lookup_output_to_rd);
    HyperKZGCommitmentNNRangeCheck()(aux.write_pc_to_rd);
    HyperKZGCommitmentNNRangeCheck()(aux.next_pc_jump);
    HyperKZGCommitmentNNRangeCheck()(aux.should_branch);
    HyperKZGCommitmentNNRangeCheck()(aux.next_pc);
}

template OpeningCombinersNNRangeCheck() {
    input OpeningCombinersNN() oc;

    BytecodeCombinersNNRangeCheck()(oc.bytecode_combiners);
    InstructionLookupCombinersNNRangeCheck()(oc.instruction_lookup_combiners);
    ReadWriteOutputTimestampCombinersNNRangeCheck()(oc.read_write_output_timestamp_combiners);
    SpartanCombinersNNRangeCheck()(oc.spartan_combiners);
    FqRangeCheck(0)(oc.coefficient);
}

template BytecodeCombinersNNRangeCheck() {
    input BytecodeCombinersNN() bc;

    for(var i = 0; i < 2; i++) {
        FqRangeCheck(0)(bc.rho[i]);
    }
}

template InstructionLookupCombinersNNRangeCheck() {
    input InstructionLookupCombinersNN() ilc;

    for(var i = 0; i < 3; i++) {
        FqRangeCheck(0)(ilc.rho[i]);
    }
}

template ReadWriteOutputTimestampCombinersNNRangeCheck() {
    input ReadWriteOutputTimestampCombinersNN() rwc;

    for(var i = 0; i < 4; i++) {
        FqRangeCheck(0)(rwc.rho[i]);
    }
}

template SpartanCombinersNNRangeCheck() {
    input SpartanCombinersNN() sc;

    FqRangeCheck(0)(sc.rho);
}

template HyperKzgVerifierAdviceNNRangeCheck() {
    input HyperKzgVerifierAdviceNN() advice;

    FqRangeCheck(0)(advice.r);
    FqRangeCheck(0)(advice.d_0);
    FqRangeCheck(0)(advice.v);
    FqRangeCheck(0)(advice.q_power);
}


template JoltStuffNNRangeCheck(
    C, 
    NUM_MEMORIES, 
    NUM_INSTRUCTIONS, 
    MEMORY_OPS_PER_INSTRUCTION,
    chunks_x_size, 
    chunks_y_size, 
    NUM_CIRCUIT_FLAGS,
    relevant_y_chunks_len    
) {
    input JoltStuffNN(
        C, 
        NUM_MEMORIES, 
        NUM_INSTRUCTIONS, 
        MEMORY_OPS_PER_INSTRUCTION,
        chunks_x_size, 
        chunks_y_size, 
        NUM_CIRCUIT_FLAGS,
        relevant_y_chunks_len  
    ) js;

    BytecodeStuffNNRangeCheck()(js.bytecode);

    ReadWriteMemoryStuffNNRangeCheck()(js.read_write_memory);

    InstructionLookupStuffNNRangeCheck(C, 
                            NUM_MEMORIES, 
                            NUM_INSTRUCTIONS
                            )(js.instruction_lookups);

    TimestampRangeCheckStuffNNRangeCheck(MEMORY_OPS_PER_INSTRUCTION)(js.timestamp_range_check);

    R1CSStuffNNRangeCheck(chunks_x_size, 
              chunks_y_size, 
              NUM_CIRCUIT_FLAGS, 
              relevant_y_chunks_len) (js.r1cs);
}
