pragma circom 2.2.1;
include "./sum_check/sumcheck.circom";
include "./../../pcs/hyrax.circom";
include "./../../combined_r1cs/linking_buses.circom";
include "./../../combined_r1cs/linking.circom";

template VerifySpartan(outer_num_rounds, inner_num_rounds, num_vars, postponed_point_len) { 
    
    var C = 4; 
    var NUM_MEMORIES = 54; 
    var NUM_INSTRUCTIONS = 26; 
    var MEMORY_OPS_PER_INSTRUCTION = 4;
    var chunks_x_size = 4; 
    var chunks_y_size = 4; 
    var NUM_CIRCUIT_FLAGS = 11; 
    var relevant_y_chunks_len = 4;

    // Public output of Combined R1CS.
    input signal counter_combined_r1cs;

    // Public output of Combined R1CS. It is the evaluation that was postponed. This verifier is supposed to do this eval natively.

    input PostponedEval(postponed_point_len) to_eval;
    
    // Public input of Combined R1CS.
    input Spartan1PI(C, 
                    NUM_MEMORIES, 
                    NUM_INSTRUCTIONS, 
                    MEMORY_OPS_PER_INSTRUCTION,
                    chunks_x_size, 
                    chunks_y_size, 
                    NUM_CIRCUIT_FLAGS, 
                    relevant_y_chunks_len) pub_io;   

    // The above five are the public IO of Combined R1CS.

    input Fq() digest;

    input PedersenGenerators(num_vars) setup;

    // The above five are the public input of V_{Spartan, 2}.

    // Public output of V_{Spartan, 2}.
    output signal counter_spartan_2;
    counter_combined_r1cs === 2;
    counter_spartan_2 <== counter_combined_r1cs + 1;
    
    output PostponedEval(inner_num_rounds - 1) postponed_eval;

    input NonUniformSpartanProof(outer_num_rounds, inner_num_rounds, num_vars) proof;

    
    input HyraxCommitment(num_vars) w_commitment;


    Transcript() transcript <== TranscriptNew()(8115760143138964446454059462739);
    Transcript() temp_transcript <== AppendScalar()(digest, transcript);

    Transcript() int_transcript[7];   

    Spartan1PIRangeCheck(
        C, 
        NUM_MEMORIES, 
        NUM_INSTRUCTIONS, 
        MEMORY_OPS_PER_INSTRUCTION,
        chunks_x_size, 
        chunks_y_size, 
        NUM_CIRCUIT_FLAGS, 
        relevant_y_chunks_len
    ) (pub_io);   

    PostponedEvalRangeCheck(postponed_point_len)(to_eval);

    NonUniformSpartanProofRangeCheck(
        outer_num_rounds,
        inner_num_rounds,
        num_vars
    )(proof);

    // Verifying the postponed eval.
    VerifyPostponedEval(postponed_point_len, 
                        C, 
                        NUM_MEMORIES, 
                        NUM_INSTRUCTIONS, 
                        MEMORY_OPS_PER_INSTRUCTION,
                        chunks_x_size, 
                        chunks_y_size, 
                        NUM_CIRCUIT_FLAGS, 
                        relevant_y_chunks_len)(to_eval, pub_io.linking_stuff, pub_io.jolt_pi);
    
    Fq() tau[outer_num_rounds];
  
    var w_num_commitments = 2**(num_vars - (num_vars>>1));

    int_transcript[0] <== AppendCommitment(w_num_commitments)(w_commitment.row_commitments, temp_transcript);


    (int_transcript[1], tau) <== ChallengeVector(outer_num_rounds)(int_transcript[0]);

    // Outer sum-check
    Fq() claim_outer_final, r_x[outer_num_rounds];
    Fq() zero;
    zero.limbs <== [0, 0, 0];

    (int_transcript[2], claim_outer_final, r_x) <== NonNativeSumCheck(outer_num_rounds, 3)
                                                    (zero, proof.outer_sumcheck_proof, int_transcript[1]);

    // verify claim_outer_final
    Fq() taus_bound_rx <== EvaluateEq(outer_num_rounds)(tau, r_x);
    
    Fq() claim_Az_Bz <== NonNativeMul()(proof.outer_sumcheck_claims[0], proof.outer_sumcheck_claims[1]);

    Fq() claim_Az_Bz_minus_Cz <== NonNativeSub()(claim_Az_Bz, proof.outer_sumcheck_claims[2]);

    Fq() expected_claim_outer_final <== NonNativeMul()(claim_Az_Bz_minus_Cz, taus_bound_rx);
    NonNativeEqualityReducedLHS()(expected_claim_outer_final, claim_outer_final);

    int_transcript[3] <== AppendScalars(3)(proof.outer_sumcheck_claims, int_transcript[2]);
    
    // inner sum-check
    Fq() r_inner_sumcheck_RLC;
    (int_transcript[4], r_inner_sumcheck_RLC) <== ChallengeScalar()(int_transcript[3]);

    Fq() claim_inner_final, r_y[inner_num_rounds];
    Fq() claim_inner_joint <== EvalUniPoly(2)(proof.outer_sumcheck_claims, r_inner_sumcheck_RLC);
    (int_transcript[5], claim_inner_final, r_y) <== NonNativeSumCheck(inner_num_rounds, 2)
                                                    (claim_inner_joint, proof.inner_sumcheck_proof, int_transcript[4]);

    Fq() expected_eval_z;
    expected_eval_z <== EvaluateZMle(inner_num_rounds)(r_y, proof.pub_io_eval, proof.inner_sumcheck_claims[3]);

    Fq() inner_sum_check_truncated_claim[3] <== TruncateVec(0, 3, 4)(proof.inner_sumcheck_claims);
    Fq() claim_combined_Az_Bz_Cz <== EvalUniPoly(2)(inner_sum_check_truncated_claim, r_inner_sumcheck_RLC);
    Fq() expected_claim_inner_final <== NonNativeMul()(claim_combined_Az_Bz_Cz, expected_eval_z);
    NonNativeEqualityReducedLHS()(expected_claim_inner_final, claim_inner_final);

    int_transcript[6] <== AppendScalars(4)(proof.inner_sumcheck_claims, int_transcript[5]);

    Fq() r_rest[inner_num_rounds - 1] <== TruncateVec(1, inner_num_rounds, inner_num_rounds)(r_y);

    // Postponing the eval of public io.
    postponed_eval.point <== r_rest;
    postponed_eval.eval <== proof.pub_io_eval;
         
    // VerifyEval(num_vars)(w_commitment , proof.joint_opening_proof, setup, proof.inner_sumcheck_claims[3], r_rest);
}

template EvaluateZMle(r_len) {    
    input Fq() r[r_len];
    input Fq() pub_io_eval;
    input Fq() w_eval;
    output Fq() eval_z;

    Fq() one;
    one.limbs <== [1, 0, 0];

    Fq() r_const <== r[0];
    Fq() r_rest[r_len - 1] <== TruncateVec(1, r_len, r_len)(r);

    Fq() t1 <== NonNativeSub()(one, r_const); 
    Fq() t2 <== NonNativeMul()(t1, pub_io_eval); 
    Fq() t3 <== NonNativeMul()(r_const, w_eval);
    eval_z <== NonNativeAdd()(t2, t3);
}

// Template to compute the postponed evaluation of teh public IO of V_{jolt, 1}.
template VerifyPostponedEval(point_len, 

                            C, 
                            NUM_MEMORIES, 
                            NUM_INSTRUCTIONS, 
                            MEMORY_OPS_PER_INSTRUCTION,
                            chunks_x_size, 
                            chunks_y_size, 
                            NUM_CIRCUIT_FLAGS, 
                            relevant_y_chunks_len) {

    input PostponedEval(point_len) postponed_eval;

    input LinkingStuff1NN(C, 
                        NUM_MEMORIES, 
                        NUM_INSTRUCTIONS, 
                        MEMORY_OPS_PER_INSTRUCTION,
                        chunks_x_size, 
                        chunks_y_size, 
                        NUM_CIRCUIT_FLAGS, 
                        relevant_y_chunks_len) pub_out;

    input JoltPreprocessingNN() pub_in;

    var bytecode_stuff_size = 6 * 9;

    var read_write_memory_stuff_size = 6 * 13;

    var instruction_lookups_stuff_size = 6 * (C + 3 * NUM_MEMORIES + NUM_INSTRUCTIONS + 1);

    var timestamp_range_check_stuff_size = 6 * (4 * MEMORY_OPS_PER_INSTRUCTION);

    var aux_variable_stuff_size = 6 * (8 + relevant_y_chunks_len);

    var r1cs_stuff_size = 6 * (chunks_x_size + chunks_y_size + NUM_CIRCUIT_FLAGS) + aux_variable_stuff_size;

    var jolt_stuff_size = bytecode_stuff_size + read_write_memory_stuff_size + instruction_lookups_stuff_size 
                            + timestamp_range_check_stuff_size + r1cs_stuff_size;

    var bytecode_combiners_size = 2;

    var instruction_lookups_combiners_size = 3;

    var read_write_output_timestamp_combiners_size = 4;

    var spartan_combiners_size = 1;

    var opening_combiners_size = bytecode_combiners_size + instruction_lookups_combiners_size
                                    + read_write_output_timestamp_combiners_size + spartan_combiners_size + 1;

    var hyperkzg_verifier_advice_size = 4;

    var linking_stuff_1_size = jolt_stuff_size + opening_combiners_size + hyperkzg_verifier_advice_size;

    var pub_io_len = 1 + 1 + linking_stuff_1_size + 2;

    signal pub_io[pub_io_len] <== SerialisePubIO(C, 
                                                NUM_MEMORIES, 
                                                NUM_INSTRUCTIONS, 
                                                MEMORY_OPS_PER_INSTRUCTION,
                                                chunks_x_size, 
                                                chunks_y_size, 
                                                NUM_CIRCUIT_FLAGS, 
                                                relevant_y_chunks_len, 

                                                pub_io_len)(pub_out, pub_in);

    var next_pow_2_pub_io_len = NextPowerOf2(pub_io_len);
    var ceil_log_pub_io_len = log2(next_pow_2_pub_io_len);

    signal pub_io_padded[1 << ceil_log_pub_io_len];

    for (var i = 0; i < pub_io_len; i++) {
        pub_io_padded[i] <== pub_io[i];
    }

    for (var i = pub_io_len; i < next_pow_2_pub_io_len; i++) {
        pub_io_padded[i] <== 0;
    }

    signal native_point[point_len];
    for (var i = 0; i < point_len; i++) {
        native_point[i] <== CombineLimbs()(postponed_eval.point[i]);
    }

    signal _rest_point[point_len - ceil_log_pub_io_len];
    signal required_point[ceil_log_pub_io_len];

    (_rest_point, required_point) <== NativeSplitAt(point_len - ceil_log_pub_io_len, point_len)(native_point);

    signal mul[point_len - ceil_log_pub_io_len + 1];
    signal evals[1 << ceil_log_pub_io_len] <== NativeEvals(ceil_log_pub_io_len)(required_point);
    mul[0] <== NativeComputeDotProduct(1 << ceil_log_pub_io_len)(evals, pub_io_padded);
    for (var i = 0; i < point_len - ceil_log_pub_io_len; i++) {
        mul[i + 1] <== mul[i] * (1 - native_point[i]);
    }

    signal computed_eval <== mul[point_len - ceil_log_pub_io_len];

    signal claimed_eval <== CombineLimbs()(postponed_eval.eval);

    claimed_eval === computed_eval;
}

template SerialisePubIO(C, NUM_MEMORIES, NUM_INSTRUCTIONS, MEMORY_OPS_PER_INSTRUCTION,chunks_x_size, chunks_y_size, NUM_CIRCUIT_FLAGS, relevant_y_chunks_len,pub_io_len) {

    input LinkingStuff1NN(C, 
                        NUM_MEMORIES, 
                        NUM_INSTRUCTIONS, 
                        MEMORY_OPS_PER_INSTRUCTION,
                        chunks_x_size, 
                        chunks_y_size, 
                        NUM_CIRCUIT_FLAGS, 
                        relevant_y_chunks_len) pub_out;

    input JoltPreprocessingNN() pub_in;

    output signal pub_io[pub_io_len];

    var bytecode_stuff_size = 6 * 9;

    var read_write_memory_stuff_size = 6 * 13;

    var instruction_lookups_stuff_size = 6 * (C + 3 * NUM_MEMORIES + NUM_INSTRUCTIONS + 1);

    var timestamp_range_check_stuff_size = 6 * (4 * MEMORY_OPS_PER_INSTRUCTION);

    var aux_variable_stuff_size = 6 * (8 + relevant_y_chunks_len);

    var r1cs_stuff_size = 6 * (chunks_x_size + chunks_y_size + NUM_CIRCUIT_FLAGS) + aux_variable_stuff_size;

    var jolt_stuff_size = bytecode_stuff_size + read_write_memory_stuff_size + instruction_lookups_stuff_size 
                            + timestamp_range_check_stuff_size + r1cs_stuff_size;

    pub_io[0] <== 1;
    pub_io[1] <== 1;

    // Serialising pub_out.
    // Serialising JoltStuffNN().
    component serialise = SerialiseNNCommitments(jolt_stuff_size / 6);

    // Serialising BytecodeStuffNN().
    serialise.commits[0] <== pub_out.commitments.bytecode.a_read_write;
    for (var i = 0; i < 6; i++) {
        serialise.commits[i + 1] <== pub_out.commitments.bytecode.v_read_write[i];
    }
    serialise.commits[7] <== pub_out.commitments.bytecode.t_read;
    serialise.commits[8] <== pub_out.commitments.bytecode.t_final;

    // Serialising ReadWriteMemoryStuffNN().
    serialise.commits[9] <== pub_out.commitments.read_write_memory.a_ram;
    serialise.commits[10] <== pub_out.commitments.read_write_memory.v_read_rd;
    serialise.commits[11] <== pub_out.commitments.read_write_memory.v_read_rs1;
    serialise.commits[12] <== pub_out.commitments.read_write_memory.v_read_rs2;
    serialise.commits[13] <== pub_out.commitments.read_write_memory.v_read_ram;
    serialise.commits[14] <== pub_out.commitments.read_write_memory.v_write_rd;
    serialise.commits[15] <== pub_out.commitments.read_write_memory.v_write_ram;
    serialise.commits[16] <== pub_out.commitments.read_write_memory.v_final;
    serialise.commits[17] <== pub_out.commitments.read_write_memory.t_read_rd;
    serialise.commits[18] <== pub_out.commitments.read_write_memory.t_read_rs1;
    serialise.commits[19] <== pub_out.commitments.read_write_memory.t_read_rs2;
    serialise.commits[20] <== pub_out.commitments.read_write_memory.t_read_ram;
    serialise.commits[21] <== pub_out.commitments.read_write_memory.t_final;

    // Serialising InstructionsLookupStuffNN().
    for (var i = 0; i < C; i++) {
        serialise.commits[22 + i] <== pub_out.commitments.instruction_lookups.dim[i];
    }
    for (var i = 0; i < NUM_MEMORIES; i++) {
        serialise.commits[22 + C + i] <== pub_out.commitments.instruction_lookups.read_cts[i];
        serialise.commits[22 + C + NUM_MEMORIES + i] <== pub_out.commitments.instruction_lookups.final_cts[i];
        serialise.commits[22 + C + 2 * NUM_MEMORIES + i] <== pub_out.commitments.instruction_lookups.E_polys[i];
    }
    for (var i = 0; i < NUM_INSTRUCTIONS; i++) {
        serialise.commits[22 + C + 3 * NUM_MEMORIES + i] <== pub_out.commitments.instruction_lookups.instruction_flags[i];
    }
    serialise.commits[22 + C + 3 * NUM_MEMORIES + NUM_INSTRUCTIONS] <== pub_out.commitments.instruction_lookups.lookup_outputs;

    // Serialising TimeStampRangeCheckStuffNN().
    var count = 23 + C + 3 * NUM_MEMORIES + NUM_INSTRUCTIONS;
    for (var i = 0; i < MEMORY_OPS_PER_INSTRUCTION; i++) {
        serialise.commits[count + i] <== pub_out.commitments.timestamp_range_check.read_cts_read_timestamp[i];
        serialise.commits[count + MEMORY_OPS_PER_INSTRUCTION + i] <== pub_out.commitments.timestamp_range_check.read_cts_global_minus_read[i];
        serialise.commits[count + 2 * MEMORY_OPS_PER_INSTRUCTION + i] <== pub_out.commitments.timestamp_range_check.final_cts_read_timestamp[i];
        serialise.commits[count + 3 * MEMORY_OPS_PER_INSTRUCTION + i] <== pub_out.commitments.timestamp_range_check.final_cts_global_minus_read[i];
    }

    // Serialising R1CSStuff().
    count += 4 * MEMORY_OPS_PER_INSTRUCTION;
    for (var i = 0; i < chunks_x_size; i++) {
        serialise.commits[count + i] <== pub_out.commitments.r1cs.chunks_x[i];
    }
    count += chunks_x_size;
    for (var i = 0; i < chunks_y_size; i++) {
        serialise.commits[count + i] <== pub_out.commitments.r1cs.chunks_y[i];
    }
    count += chunks_y_size;
    for (var i = 0; i < NUM_CIRCUIT_FLAGS; i++) {
        serialise.commits[count + i] <== pub_out.commitments.r1cs.circuit_flags[i];
    }
    count += NUM_CIRCUIT_FLAGS;

    // Serialsing AuxVariableStuff().
    serialise.commits[count + 0] <== pub_out.commitments.r1cs.aux.left_lookup_operand;
    serialise.commits[count + 1] <== pub_out.commitments.r1cs.aux.right_lookup_operand;
    serialise.commits[count + 2] <== pub_out.commitments.r1cs.aux.product;
    count += 3;
    for (var i = 0; i < relevant_y_chunks_len; i++) {
        serialise.commits[count + i] <== pub_out.commitments.r1cs.aux.relevant_y_chunks[i];
    }
    count += relevant_y_chunks_len;
    serialise.commits[count + 0] <== pub_out.commitments.r1cs.aux.write_lookup_output_to_rd;
    serialise.commits[count + 1] <== pub_out.commitments.r1cs.aux.write_pc_to_rd;
    serialise.commits[count + 2] <== pub_out.commitments.r1cs.aux.next_pc_jump;
    serialise.commits[count + 3] <== pub_out.commitments.r1cs.aux.should_branch;
    serialise.commits[count + 4] <== pub_out.commitments.r1cs.aux.next_pc;

    assert(count + 5 == jolt_stuff_size / 6);

    for (var i = 2; i <= jolt_stuff_size + 1; i++) {
        pub_io[i] <== serialise.serialised_comms[i - 2];
    }

    // Serialising opening_combiners(). 6 * jolt_stuff_size + 1.

    pub_io[2 + jolt_stuff_size] <== CombineLimbs()(pub_out.opening_combiners.bytecode_combiners.rho[0]);
    pub_io[2 + jolt_stuff_size + 1] <== CombineLimbs()(pub_out.opening_combiners.bytecode_combiners.rho[1]);

    pub_io[2 + jolt_stuff_size + 2] <== CombineLimbs()(pub_out.opening_combiners.instruction_lookup_combiners.rho[0]);
    pub_io[2 + jolt_stuff_size + 3] <== CombineLimbs()(pub_out.opening_combiners.instruction_lookup_combiners.rho[1]);
    pub_io[2 + jolt_stuff_size + 4] <== CombineLimbs()(pub_out.opening_combiners.instruction_lookup_combiners.rho[2]);

    pub_io[2 + jolt_stuff_size + 5] <== CombineLimbs()(pub_out.opening_combiners.read_write_output_timestamp_combiners.rho[0]);
    pub_io[2 + jolt_stuff_size + 6] <== CombineLimbs()(pub_out.opening_combiners.read_write_output_timestamp_combiners.rho[1]);
    pub_io[2 + jolt_stuff_size + 7] <== CombineLimbs()(pub_out.opening_combiners.read_write_output_timestamp_combiners.rho[2]);
    pub_io[2 + jolt_stuff_size + 8] <== CombineLimbs()(pub_out.opening_combiners.read_write_output_timestamp_combiners.rho[3]);

    pub_io[2 + jolt_stuff_size + 9] <== CombineLimbs()(pub_out.opening_combiners.spartan_combiners.rho);

    pub_io[2 + jolt_stuff_size + 10] <== CombineLimbs()(pub_out.opening_combiners.coefficient);


    // Serialising HyperKzgVerifierAdvice().
    pub_io[2 + jolt_stuff_size + 11] <== CombineLimbs()(pub_out.hyperkzg_verifier_advice.r);
    pub_io[2 + jolt_stuff_size + 12] <== CombineLimbs()(pub_out.hyperkzg_verifier_advice.d_0);
    pub_io[2 + jolt_stuff_size + 13] <== CombineLimbs()(pub_out.hyperkzg_verifier_advice.v);
    pub_io[2 + jolt_stuff_size + 14] <== CombineLimbs()(pub_out.hyperkzg_verifier_advice.q_power);

    // Serialsing JoltPI.
    pub_io[2 + jolt_stuff_size + 15] <== CombineLimbs()(pub_in.v_init_final_hash);
    pub_io[2 + jolt_stuff_size + 16] <== CombineLimbs()(pub_in.bytecode_words_hash);

    assert(pub_io_len == 2 + jolt_stuff_size + 17);
}

template CombineLimbs() {
    input Fq() in;
    output signal elem;

    elem <== in.limbs[0] + (1 << 125) * in.limbs[1] + (1 << 250) * in.limbs[2];
}

template SerialiseNNCommitments(num_commits) {
    input HyperKZGCommitmentNN() commits[num_commits];
    output signal serialised_comms[6 * num_commits];

    for (var i = 0; i < num_commits; i++) {
        for (var j = 0; j < 3; j++) {
            serialised_comms[6 * i + j] <== commits[i].commitment.x.limbs[j];
            serialised_comms[6 * i + j + 3] <== commits[i].commitment.y.limbs[j];
        }
    }
}

template AppendCommitment(w_num_commitments) {
    input G1Projective() commitments[w_num_commitments];

    input Transcript() transcript;
    output Transcript() up_transcript;

    component append_points = AppendPoints(w_num_commitments);

    signal x_square[w_num_commitments];
    signal x_cube[w_num_commitments];
    signal y_square[w_num_commitments];
    signal z_square[w_num_commitments];
    signal z_cube[w_num_commitments];

    for (var i = 0; i < w_num_commitments; i++) {
        x_square[i] <== commitments[i].x * commitments[i].x;
        x_cube[i] <== x_square[i] * commitments[i].x;

        y_square[i] <== commitments[i].y * commitments[i].y;

        z_square[i] <== commitments[i].z * commitments[i].z;
        z_cube[i] <== z_square[i] * commitments[i].z;
        
        //Curve Check
        commitments[i].z *  y_square[i] === x_cube[i]  - 17 *  z_cube[i];

        //To Check z = 0/1
        (1 - commitments[i].z) * commitments[i].z === 0;
        
        //To make sure that if commitment is identity we append with zeroes. 
        append_points.points[i].x <== commitments[i].x;
        append_points.points[i].y <== commitments[i].z *  commitments[i].y;
    }

    append_points.transcript <== transcript;
    up_transcript <== append_points.up_transcript;
}

template Spartan1PIRangeCheck(
    C, 
    NUM_MEMORIES, 
    NUM_INSTRUCTIONS, 
    MEMORY_OPS_PER_INSTRUCTION,
    chunks_x_size, 
    chunks_y_size, 
    NUM_CIRCUIT_FLAGS, 
    relevant_y_chunks_len
) {
    input Spartan1PI(
        C, 
        NUM_MEMORIES, 
        NUM_INSTRUCTIONS, 
        MEMORY_OPS_PER_INSTRUCTION,
        chunks_x_size, 
        chunks_y_size, 
        NUM_CIRCUIT_FLAGS, 
        relevant_y_chunks_len
    ) pi;

    JoltPreprocessingNNRangeCheck()(pi.jolt_pi);
    LinkingStuff1NNRangeCheck (C, 
        NUM_MEMORIES, 
        NUM_INSTRUCTIONS, 
        MEMORY_OPS_PER_INSTRUCTION,
        chunks_x_size, 
        chunks_y_size, 
        NUM_CIRCUIT_FLAGS, 
        relevant_y_chunks_len) (pi.linking_stuff);
    HyperKZGVerifierKeyNNRangeCheck()(pi.vk_spartan_1);
    HyperKZGVerifierKeyNNRangeCheck()(pi.vk_jolt_2);

}  

template JoltPreprocessingNNRangeCheck() {
    input JoltPreprocessingNN() jp;

    FqRangeCheck(0)(jp.v_init_final_hash);
    FqRangeCheck(0)(jp.bytecode_words_hash);
}

template HyperKZGVerifierKeyNNRangeCheck() {
    input HyperKZGVerifierKeyNN() vk;

    KZGVerifierKeyNNRangeCheck()(vk.kzg_vk);
}

template KZGVerifierKeyNNRangeCheck() {
    input KZGVerifierKeyNN() vk;

    G1AffineNNRangeCheck()(vk.g1);
    G2AffineNNRangeCheck()(vk.g2);
    G2AffineNNRangeCheck()(vk.beta_g2);
}

template PostponedEvalRangeCheck(point_len) {
    input PostponedEval(point_len) p_eval;

    for(var i = 0; i < point_len; i++) {
        FqRangeCheck(0)(p_eval.point[i]);
    }
    FqRangeCheck(0)(p_eval.eval);
}

template NonUniformSpartanProofRangeCheck(
    outer_num_rounds,
    inner_num_rounds,
    num_vars
) {
    input NonUniformSpartanProof(
        outer_num_rounds,
        inner_num_rounds,
        num_vars
    ) proof;
    
    for(var i = 0; i < 3; i++) {
        FqRangeCheck(1)(proof.outer_sumcheck_claims[i]);
    }

    for(var i = 0; i < 4; i++) {
        FqRangeCheck(1)(proof.inner_sumcheck_claims[i]);
    }

    FqRangeCheck(1)(proof.pub_io_eval);
}

bus NonUniformSpartanProof (outer_num_rounds, 
                            inner_num_rounds, 
                            num_vars) {

    SumcheckInstanceProof(3, outer_num_rounds) outer_sumcheck_proof; 

    Fq() outer_sumcheck_claims[3];

    SumcheckInstanceProof(2, inner_num_rounds) inner_sumcheck_proof;

    Fq() inner_sumcheck_claims[4];              

    Fq() pub_io_eval;

    EvalProof(num_vars) joint_opening_proof;
}

bus Spartan1PI(C, 
                NUM_MEMORIES, 
                NUM_INSTRUCTIONS, 
                MEMORY_OPS_PER_INSTRUCTION,
                chunks_x_size, 
                chunks_y_size, 
                NUM_CIRCUIT_FLAGS, 
                relevant_y_chunks_len) {

    signal counter_jolt_1;                

    JoltPreprocessingNN() jolt_pi;

    LinkingStuff1NN(C, 
                    NUM_MEMORIES, 
                    NUM_INSTRUCTIONS, 
                    MEMORY_OPS_PER_INSTRUCTION,
                    chunks_x_size, 
                    chunks_y_size, 
                    NUM_CIRCUIT_FLAGS, 
                    relevant_y_chunks_len) linking_stuff;

    Fq() digest;
    
    HyperKZGVerifierKeyNN() vk_spartan_1;

    HyperKZGVerifierKeyNN() vk_jolt_2;
}

template NativeSplitAt(mid, size){
    assert(mid <= size);

    input signal vec[size];
    output signal low[mid];
    output signal hi[size - mid];

    for (var i = 0; i < mid; i++){
        low[i] <== vec[i];
    }
    for (var i = mid; i < size; i++){
        hi[i - mid] <== vec[i];
    }
}
 
// component main = VerifySpartan( 23, 24, 23, 22);
