pragma circom 2.2.1;
include "./../../fields/non_native/utils.circom";
include "./../../fields/non_native/non_native_over_bn_base.circom";
include "./../../groups/bn254_g1.circom";
include "./../../groups/utils.circom";
include "./../../pcs/hyperkzgjolt2.circom";
include "jolt2bus.circom";


template PairingCheck(C, 
                    NUM_MEMORIES, 
                    NUM_INSTRUCTIONS, 
                    MEMORY_OPS_PER_INSTRUCTION,
                    chunks_x_size, 
                    chunks_y_size, 
                    NUM_CIRCUIT_FLAGS, 
                    relevant_y_chunks_len,rounds_reduced_opening_proof){


    input LinkingStuff2(C, 
                    NUM_MEMORIES, 
                    NUM_INSTRUCTIONS, 
                    MEMORY_OPS_PER_INSTRUCTION,
                    chunks_x_size, 
                    chunks_y_size, 
                    NUM_CIRCUIT_FLAGS, 
                    relevant_y_chunks_len,rounds_reduced_opening_proof) linkingstuff;

    input HyperKZGVerifierKey() vk;

    input HyperKZGProof(rounds_reduced_opening_proof) pi; 


    component combine_commit = CombiningCommitmentsForHyperKzgVerifier(C, 
                                NUM_MEMORIES, 
                                NUM_INSTRUCTIONS, 
                                MEMORY_OPS_PER_INSTRUCTION,
                                chunks_x_size, 
                                chunks_y_size, 
                                NUM_CIRCUIT_FLAGS, 
                                relevant_y_chunks_len,rounds_reduced_opening_proof);

    combine_commit.linkingstuff <== linkingstuff;
    HyperKZGCommitment() joint_commitment <== combine_commit.joint_commitment;

    HyperKzgVerifierJolt2(rounds_reduced_opening_proof)(vk, pi, joint_commitment, linkingstuff.hyperkzg_verifier_advice.r, linkingstuff.hyperkzg_verifier_advice.d_0,linkingstuff.hyperkzg_verifier_advice.v, linkingstuff.hyperkzg_verifier_advice.q_power );

}


template CombiningCommitmentsForHyperKzgVerifier(C, 
                    NUM_MEMORIES, 
                    NUM_INSTRUCTIONS, 
                    MEMORY_OPS_PER_INSTRUCTION,
                    chunks_x_size, 
                    chunks_y_size, 
                    NUM_CIRCUIT_FLAGS, 
                    relevant_y_chunks_len,rounds_reduced_opening_proof){
        
    input LinkingStuff2(C, 
                    NUM_MEMORIES, 
                    NUM_INSTRUCTIONS, 
                    MEMORY_OPS_PER_INSTRUCTION,
                    chunks_x_size, 
                    chunks_y_size, 
                    NUM_CIRCUIT_FLAGS, 
                    relevant_y_chunks_len,rounds_reduced_opening_proof) linkingstuff;



    G1Projective() final_commitment[10] ;

    // Computing commitment for Bytecode
    var bytecode_read_write_commits_len =8;
    HyperKZGCommitment() bytecode_read_write_commits[bytecode_read_write_commits_len];

    bytecode_read_write_commits[0] <== linkingstuff.commitments.bytecode.a_read_write;
    for (var i = 0; i < 6; i++) {
        bytecode_read_write_commits[i + 1] <== linkingstuff.commitments.bytecode.v_read_write[i];
    }
    bytecode_read_write_commits[7] <== linkingstuff.commitments.bytecode.t_read;

    G1Projective() bytecode_read_write_commits_projective[bytecode_read_write_commits_len];
    bytecode_read_write_commits_projective <== ConvertToProjective(bytecode_read_write_commits_len)(bytecode_read_write_commits);
    final_commitment[0] <== CombineCommitments(bytecode_read_write_commits_len)(bytecode_read_write_commits_projective, linkingstuff.opening_combiners.bytecode_combiners.rho[0]);


    var bytecode_init_final_commitments_len =1;
    HyperKZGCommitment() bytecode_init_final_commitments[bytecode_init_final_commitments_len];
    bytecode_init_final_commitments[0] <== linkingstuff.commitments.bytecode.t_final;  

    G1Projective() bytecode_init_final_commitments_projective[bytecode_init_final_commitments_len];
    bytecode_init_final_commitments_projective <== ConvertToProjective(bytecode_init_final_commitments_len)(bytecode_init_final_commitments);
    final_commitment[1] <== bytecode_init_final_commitments_projective[0];

    // Computing commitment for Instruction lookups
    HyperKZGCommitment() primary_sumcheck_commitments[NUM_MEMORIES + NUM_INSTRUCTIONS + 1];
    for (var i = 0; i < NUM_MEMORIES; i++){
        primary_sumcheck_commitments[i] <== linkingstuff.commitments.instruction_lookups.E_polys[i];
       
    }
    for (var i = 0; i < NUM_INSTRUCTIONS; i++){
        primary_sumcheck_commitments[i + NUM_MEMORIES] <== linkingstuff.commitments.instruction_lookups.instruction_flags[i];

    }
    primary_sumcheck_commitments[NUM_MEMORIES + NUM_INSTRUCTIONS] <== linkingstuff.commitments.instruction_lookups.lookup_outputs;


    G1Projective() primary_sumcheck_commitments_projective[NUM_MEMORIES + NUM_INSTRUCTIONS + 1];
    primary_sumcheck_commitments_projective <== ConvertToProjective(NUM_MEMORIES + NUM_INSTRUCTIONS + 1)(primary_sumcheck_commitments);
    final_commitment[2] <== CombineCommitments(NUM_MEMORIES + NUM_INSTRUCTIONS + 1)(primary_sumcheck_commitments_projective, linkingstuff.opening_combiners.instruction_lookup_combiners.rho[0]);


    var inst_num_read_write_commits_len = C + 2 * NUM_MEMORIES + NUM_INSTRUCTIONS + 1;
    
    HyperKZGCommitment() inst_read_write_commits[inst_num_read_write_commits_len];

      for (var i = 0; i < C; i++){
        inst_read_write_commits[i] <== linkingstuff.commitments.instruction_lookups.dim[i];
    }
    for (var i = 0; i < NUM_MEMORIES; i++) {
        inst_read_write_commits[i + C] <==linkingstuff.commitments.instruction_lookups.read_cts[i ];
        inst_read_write_commits[i + C +  NUM_MEMORIES] <==linkingstuff.commitments.instruction_lookups.E_polys[i];

    }

    for (var i = 0; i < NUM_INSTRUCTIONS; i++) {
        inst_read_write_commits[i + C +  2 * NUM_MEMORIES] <== linkingstuff.commitments.instruction_lookups.instruction_flags[i ];
    }

    inst_read_write_commits[inst_num_read_write_commits_len - 1] <== linkingstuff.commitments.instruction_lookups.lookup_outputs;

    G1Projective() inst_read_write_commits_projective[inst_num_read_write_commits_len];
    inst_read_write_commits_projective <== ConvertToProjective(inst_num_read_write_commits_len)(inst_read_write_commits);
    final_commitment[3] <== CombineCommitments(inst_num_read_write_commits_len)(inst_read_write_commits_projective, linkingstuff.opening_combiners.instruction_lookup_combiners.rho[1]);

    G1Projective() inst_init_final_commit_projective[NUM_MEMORIES];
    inst_init_final_commit_projective <== ConvertToProjective(NUM_MEMORIES)(linkingstuff.commitments.instruction_lookups.final_cts);
    final_commitment[4] <== CombineCommitments(NUM_MEMORIES)(inst_init_final_commit_projective, linkingstuff.opening_combiners.instruction_lookup_combiners.rho[2]);

    // Computing commitment for Read write memory
    var memory_read_write_commitments_len = 14;
    HyperKZGCommitment() memory_read_write_commitments[memory_read_write_commitments_len];

    memory_read_write_commitments[0] <== linkingstuff.commitments.read_write_memory.a_ram;
    memory_read_write_commitments[1] <== linkingstuff.commitments.read_write_memory.v_read_rd;
    memory_read_write_commitments[2] <== linkingstuff.commitments.read_write_memory.v_read_rs1;
    memory_read_write_commitments[3] <== linkingstuff.commitments.read_write_memory.v_read_rs2;
    memory_read_write_commitments[4] <== linkingstuff.commitments.read_write_memory.v_read_ram;
    memory_read_write_commitments[5] <== linkingstuff.commitments.read_write_memory.v_write_rd;
    memory_read_write_commitments[6] <== linkingstuff.commitments.read_write_memory.v_write_ram;
    memory_read_write_commitments[7] <== linkingstuff.commitments.read_write_memory.t_read_rd;
    memory_read_write_commitments[8] <== linkingstuff.commitments.read_write_memory.t_read_rs1;
    memory_read_write_commitments[9] <== linkingstuff.commitments.read_write_memory.t_read_rs2;
    memory_read_write_commitments[10] <== linkingstuff.commitments.read_write_memory.t_read_ram;


    memory_read_write_commitments[11] <== linkingstuff.commitments.bytecode.v_read_write[2];
    memory_read_write_commitments[12] <== linkingstuff.commitments.bytecode.v_read_write[3];
    memory_read_write_commitments[13] <== linkingstuff.commitments.bytecode.v_read_write[4];

    G1Projective() memory_read_write_commitments_projective[memory_read_write_commitments_len];
    memory_read_write_commitments_projective <== ConvertToProjective(memory_read_write_commitments_len)(memory_read_write_commitments);
    final_commitment[5] <== CombineCommitments(memory_read_write_commitments_len)(memory_read_write_commitments_projective, linkingstuff.opening_combiners.read_write_output_timestamp_combiners.rho[0]);


    var init_final_commitments_len = 2;
    HyperKZGCommitment() memory_init_final_commitments[init_final_commitments_len];
    memory_init_final_commitments[0] <== linkingstuff.commitments.read_write_memory.v_final;  
    memory_init_final_commitments[1] <== linkingstuff.commitments.read_write_memory.t_final;

    G1Projective() memory_init_final_commitments_projective[init_final_commitments_len];
    memory_init_final_commitments_projective <== ConvertToProjective(init_final_commitments_len)(memory_init_final_commitments);
    final_commitment[6] <== CombineCommitments(init_final_commitments_len)(memory_init_final_commitments_projective, linkingstuff.opening_combiners.read_write_output_timestamp_combiners.rho[1]);


    // Output sum check
    HyperKZGCommitment() outputsumcheckcommitments[1];
    outputsumcheckcommitments[0] <== linkingstuff.commitments.read_write_memory.v_final;


    G1Projective() outputsumcheckcommitments_projective[1];
    outputsumcheckcommitments_projective <== ConvertToProjective(1)(outputsumcheckcommitments);
    final_commitment[7] <== outputsumcheckcommitments_projective[0];


    // Timestamp
    HyperKZGCommitment() ts_commitments[4* MEMORY_OPS_PER_INSTRUCTION + 4];
        for (var i = 0; i < MEMORY_OPS_PER_INSTRUCTION; i++) {
        ts_commitments[i] <== linkingstuff.commitments.timestamp_range_check.read_cts_read_timestamp[i];
    }
    for (var i = 0; i < MEMORY_OPS_PER_INSTRUCTION; i++) {
        ts_commitments[i + MEMORY_OPS_PER_INSTRUCTION] <== linkingstuff.commitments.timestamp_range_check.read_cts_global_minus_read[i];
    }
    for (var i = 0; i < MEMORY_OPS_PER_INSTRUCTION; i++) {
        ts_commitments[i + 2 * MEMORY_OPS_PER_INSTRUCTION] <== linkingstuff.commitments.timestamp_range_check.final_cts_read_timestamp[i];
    }
    for (var i = 0; i < MEMORY_OPS_PER_INSTRUCTION; i++) {
        ts_commitments[i + 3 * MEMORY_OPS_PER_INSTRUCTION] <== linkingstuff.commitments.timestamp_range_check.final_cts_global_minus_read[i];
    }
    ts_commitments[ 4 * MEMORY_OPS_PER_INSTRUCTION] <== linkingstuff.commitments.read_write_memory.t_read_rd;
    ts_commitments[ 4 * MEMORY_OPS_PER_INSTRUCTION +1] <== linkingstuff.commitments.read_write_memory.t_read_rs1;
    ts_commitments[ 4 * MEMORY_OPS_PER_INSTRUCTION +2] <== linkingstuff.commitments.read_write_memory.t_read_rs2;
    ts_commitments[ 4 * MEMORY_OPS_PER_INSTRUCTION +3] <== linkingstuff.commitments.read_write_memory.t_read_ram;


    G1Projective() ts_commitments_projective[4* MEMORY_OPS_PER_INSTRUCTION + 4];
    ts_commitments_projective <== ConvertToProjective(4 * MEMORY_OPS_PER_INSTRUCTION + 4)(ts_commitments);
    final_commitment[8] <== CombineCommitments(4 * MEMORY_OPS_PER_INSTRUCTION + 4)(ts_commitments_projective, linkingstuff.opening_combiners.read_write_output_timestamp_combiners.rho[3]);

    // Spartan
    component  flatten_commitments = FlattenCommits(
              C, 
              NUM_MEMORIES, 
              NUM_INSTRUCTIONS, 
              MEMORY_OPS_PER_INSTRUCTION,
              chunks_x_size, 
              chunks_y_size, 
              NUM_CIRCUIT_FLAGS, 
              relevant_y_chunks_len);
    flatten_commitments.jolt <== linkingstuff.commitments;

    var spartan_commitments_len = 76;
    G1Projective() spartan_commitments_projective[spartan_commitments_len];
    spartan_commitments_projective <== ConvertToProjective(spartan_commitments_len)(flatten_commitments.commitments_flattened);
    final_commitment[9] <== CombineCommitments(spartan_commitments_len)(spartan_commitments_projective, linkingstuff.opening_combiners.spartan_combiners.rho);

    var num_openings = 10;
    Fq() gamma <== linkingstuff.opening_combiners.coefficient;
   

    G1Projective() joint_commitments <== CombineCommitments(num_openings)(final_commitment, gamma);
    G1Affine() affine_joint_commitment <== G1ToAffine()(joint_commitments);
    
    output HyperKZGCommitment() joint_commitment;
    joint_commitment.commitment <== affine_joint_commitment;
 
}     



template ConvertToProjective(num_commitments){
    input HyperKZGCommitment() commitments[num_commitments];
    output G1Projective() projective_commitments[num_commitments];
    
    for (var i = 0; i < num_commitments; i++) {
        projective_commitments[i] <== toProjectiveNew()(commitments[i].commitment);
    }
}


template FlattenCommits(
                C, 
              NUM_MEMORIES, 
              NUM_INSTRUCTIONS, 
              MEMORY_OPS_PER_INSTRUCTION,
              chunks_x_size, 
              chunks_y_size, 
              NUM_CIRCUIT_FLAGS, 
              relevant_y_chunks_len){
    input JoltStuff(
        C, 
              NUM_MEMORIES, 
              NUM_INSTRUCTIONS, 
              MEMORY_OPS_PER_INSTRUCTION,
              chunks_x_size, 
              chunks_y_size, 
              NUM_CIRCUIT_FLAGS, 
              relevant_y_chunks_len
    ) jolt;

    output HyperKZGCommitment() commitments_flattened[76];

    // order same as in get_ref of "impl ConstraintInput for JoltR1CSInputs"
    commitments_flattened[0] <== jolt.bytecode.a_read_write;
    commitments_flattened[1] <== jolt.bytecode.v_read_write[0];
    commitments_flattened[2] <== jolt.bytecode.v_read_write[1];
    commitments_flattened[3] <== jolt.bytecode.v_read_write[3];
    commitments_flattened[4] <== jolt.bytecode.v_read_write[4];
    commitments_flattened[5] <== jolt.bytecode.v_read_write[2];
    commitments_flattened[6] <== jolt.bytecode.v_read_write[5];

    commitments_flattened[7] <== jolt.read_write_memory.a_ram;
    commitments_flattened[8] <== jolt.read_write_memory.v_read_rs1;
    commitments_flattened[9] <== jolt.read_write_memory.v_read_rs2;
    commitments_flattened[10] <== jolt.read_write_memory.v_read_rd;
    commitments_flattened[11] <== jolt.read_write_memory.v_read_ram;
    commitments_flattened[12] <== jolt.read_write_memory.v_write_rd;
    commitments_flattened[13] <== jolt.read_write_memory.v_write_ram;

    for (var i = 0; i < C; i++) {
        commitments_flattened[14 + i] <== jolt.instruction_lookups.dim[i];
    }
    commitments_flattened[14 + C ] <== jolt.instruction_lookups.lookup_outputs;

    for (var i = 0; i < chunks_x_size; i++) {
        commitments_flattened[14 + C + 1 + i] <== jolt.r1cs.chunks_x[i];
    }
    for (var i = 0; i < chunks_y_size; i++) {
        commitments_flattened[14 + C + 1 + chunks_x_size + i] <== jolt.r1cs.chunks_y[i];
    }
    for (var i = 0; i < NUM_CIRCUIT_FLAGS; i++) {
        commitments_flattened[14 + C + 1 + chunks_x_size + chunks_y_size + i] <== jolt.r1cs.circuit_flags[i];
    }

    for (var i = 0; i < NUM_INSTRUCTIONS; i++) {
        commitments_flattened[14 + C + 1 + chunks_x_size + chunks_y_size + NUM_CIRCUIT_FLAGS + i] <== jolt.instruction_lookups.instruction_flags[i];
    }

    commitments_flattened[14 + C + 1 + chunks_x_size + chunks_y_size + NUM_CIRCUIT_FLAGS + NUM_INSTRUCTIONS + 0] <== jolt.r1cs.aux
    .left_lookup_operand;
    commitments_flattened[14 + C + 1 + chunks_x_size + chunks_y_size + NUM_CIRCUIT_FLAGS + NUM_INSTRUCTIONS + 1] <== jolt.r1cs.aux
    .right_lookup_operand;
    commitments_flattened[14 + C + 1 + chunks_x_size + chunks_y_size + NUM_CIRCUIT_FLAGS + NUM_INSTRUCTIONS + 2] <== jolt.r1cs.aux
    .product;
    
    for (var i = 0; i < relevant_y_chunks_len; i++) {
        commitments_flattened[14 + C + 1 + chunks_x_size + chunks_y_size + NUM_CIRCUIT_FLAGS + NUM_INSTRUCTIONS + 3 + i] <== jolt.r1cs.aux.relevant_y_chunks[i];
    }
    commitments_flattened[14 + C + 1 + chunks_x_size + chunks_y_size + NUM_CIRCUIT_FLAGS + NUM_INSTRUCTIONS + 3 + relevant_y_chunks_len + 0] <== jolt.r1cs.aux
    .write_lookup_output_to_rd;
    commitments_flattened[14 + C + 1 + chunks_x_size + chunks_y_size + NUM_CIRCUIT_FLAGS + NUM_INSTRUCTIONS + 3 + relevant_y_chunks_len + 1] <== jolt.r1cs.aux
    .write_pc_to_rd;
    commitments_flattened[14 + C + 1 + chunks_x_size + chunks_y_size + NUM_CIRCUIT_FLAGS + NUM_INSTRUCTIONS +3 + relevant_y_chunks_len + 2] <== jolt.r1cs.aux
    .next_pc_jump;
    commitments_flattened[14 + C + 1 + chunks_x_size + chunks_y_size + NUM_CIRCUIT_FLAGS + NUM_INSTRUCTIONS + 3 + relevant_y_chunks_len + 3] <== jolt.r1cs.aux
    .should_branch;
    commitments_flattened[14 + C + 1 + chunks_x_size + chunks_y_size + NUM_CIRCUIT_FLAGS + NUM_INSTRUCTIONS + 3 + relevant_y_chunks_len + 4] <== jolt.r1cs.aux
    .next_pc;
}


// component main =  PairingCheck( 4, 54, 26,
//                     4, 4, 4, 11, 4, 16
// );