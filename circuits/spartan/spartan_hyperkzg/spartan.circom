pragma circom 2.2.1;
include "./sum_check/sumcheck.circom";
include "./../../pcs/hyperkzg.circom";
include "./utils.circom";
include "./../../combined_r1cs/linking_buses.circom";
include "./../../combined_r1cs/linking.circom";

template VerifySpartan(outer_num_rounds, inner_num_rounds, num_vars,rounds_reduced_opening_proof) { 

    var  C = 4; 
    var NUM_MEMORIES = 54; 
    var NUM_INSTRUCTIONS = 26; 
    var MEMORY_OPS_PER_INSTRUCTION = 4;
    var chunks_x_size = 4; 
    var chunks_y_size = 4; 
    var NUM_CIRCUIT_FLAGS = 11; 
    var relevant_y_chunks_len = 4;

    // Public input of V_{jolt, 1}.
    input JoltPreprocessingNN() jolt_pi;

    // Public output of V_{jolt, 1}.
    input LinkingStuff1NN(C, 
                    NUM_MEMORIES, 
                    NUM_INSTRUCTIONS, 
                    MEMORY_OPS_PER_INSTRUCTION,
                    chunks_x_size, 
                    chunks_y_size, 
                    NUM_CIRCUIT_FLAGS, 
                    relevant_y_chunks_len
                    ) linking_stuff;

    input HyperKZGVerifierKey() vk;
    input Fq() digest;
    
    // Public output of V_{Spartan, 1}.
    output PostponedEval(inner_num_rounds - 1) postponed_eval;

    input NonUniformSpartanProof(outer_num_rounds, 
                                inner_num_rounds, 
                                num_vars) proof;

    input HyperKZGCommitment() w_commitment;


    JoltPreprocessingNNRangeCheck()(jolt_pi);

    LinkingStuff1NNRangeCheck(
        C, 
        NUM_MEMORIES, 
        NUM_INSTRUCTIONS, 
        MEMORY_OPS_PER_INSTRUCTION,
        chunks_x_size, 
        chunks_y_size, 
        NUM_CIRCUIT_FLAGS, 
        relevant_y_chunks_len
    )(linking_stuff);

    NonUniformSpartanProofRangeCheck(
        outer_num_rounds,
        inner_num_rounds,
        num_vars
    )(proof);

    Transcript() transcript <== TranscriptNew()(8115760143138964446454059462739);
    Transcript() temp_transcript <== AppendScalar()(digest, transcript);
    
    Transcript() int_transcript[7];   

    Fq() tau[outer_num_rounds];

    int_transcript[0] <== AppendPoint()(w_commitment.commitment, temp_transcript);

    (int_transcript[1], tau) <== ChallengeVector(outer_num_rounds)(int_transcript[0]);

    // Outer sum-check
    Fq() claim_outer_final, r_x[outer_num_rounds];
    Fq() zero;
    zero.limbs <== [0, 0, 0];

    (int_transcript[2], claim_outer_final, r_x) <== NonNativeSumCheck(outer_num_rounds, 3)
                                                    (zero, proof.outer_sumcheck_proof, 
                                                    int_transcript[1]);


    // verify claim_outer_final
    Fq() taus_bound_rx <== EvaluateEq(outer_num_rounds)(tau, r_x);
    Fq() claim_Az_Bz <== NonNativeMul()(proof.outer_sumcheck_claims[0],  
                                        proof.outer_sumcheck_claims[1]);
    Fq() claim_Az_Bz_minus_Cz <== NonNativeSub()(claim_Az_Bz,  
                                        proof.outer_sumcheck_claims[2]);
    Fq() expected_claim_outer_final <== NonNativeMul()(claim_Az_Bz_minus_Cz, taus_bound_rx);
    NonNativeEqualityReducedLHS()(expected_claim_outer_final, claim_outer_final);
    log("expected_claim_outer_final ", expected_claim_outer_final.limbs[0], expected_claim_outer_final.limbs[1], expected_claim_outer_final.limbs[2]);
    log("claim_outer_final = ", claim_outer_final.limbs[0], claim_outer_final.limbs[1], claim_outer_final.limbs[2]);
    log();

    int_transcript[3] <== AppendScalars(3)(proof.outer_sumcheck_claims, int_transcript[2]);
    
    // inner sum-check
    Fq() r_inner_sumcheck_RLC;
    (int_transcript[4], r_inner_sumcheck_RLC) <== ChallengeScalar()(int_transcript[3]);

    Fq() claim_inner_final, r_y[inner_num_rounds];
    Fq() claim_inner_joint <== EvalUniPoly(2)(proof.outer_sumcheck_claims, r_inner_sumcheck_RLC);
    (int_transcript[5], claim_inner_final, r_y) <== NonNativeSumCheck(inner_num_rounds, 2)
                                                                (claim_inner_joint, 
                                                                proof.inner_sumcheck_proof, 
                                                                int_transcript[4]);

    Fq() expected_eval_z;
    expected_eval_z <== EvaluateZMle(inner_num_rounds)
                                        (r_y, proof.pub_io_eval, proof.inner_sumcheck_claims[3]);

    
    Fq() inner_sum_check_truncated_claim[3] <== TruncateVec(0, 3, 4)(proof.inner_sumcheck_claims);
    Fq() claim_combined_Az_Bz_Cz <== EvalUniPoly(2)(inner_sum_check_truncated_claim, r_inner_sumcheck_RLC);
    Fq() expected_claim_inner_final <== NonNativeMul()(claim_combined_Az_Bz_Cz, expected_eval_z);
    NonNativeEqualityReducedLHS()(expected_claim_inner_final, claim_inner_final);

    int_transcript[6] <== AppendScalars(4)(proof.inner_sumcheck_claims, int_transcript[5]);

    Fq() r_rest[inner_num_rounds - 1] <== TruncateVec(1, inner_num_rounds, inner_num_rounds)(r_y);

    // Postponing the eval of public io.
    postponed_eval.point <== r_rest;
    postponed_eval.eval <== proof.pub_io_eval;

    HyperKzgVerifier(num_vars)(
        vk,
        w_commitment,
        r_rest,
        proof.joint_opening_proof,
        proof.inner_sumcheck_claims[3],
        int_transcript[6]
    );
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

template NonUniformSpartanPIRangeCheck(
    C, 
    NUM_MEMORIES, 
    NUM_INSTRUCTIONS, 
    MEMORY_OPS_PER_INSTRUCTION,
    chunks_x_size, 
    chunks_y_size, 
    NUM_CIRCUIT_FLAGS, 
    relevant_y_chunks_len
) {
    input NonUniformSpartanPI(
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

    LinkingStuff1NNRangeCheck(
        C, 
        NUM_MEMORIES, 
        NUM_INSTRUCTIONS, 
        MEMORY_OPS_PER_INSTRUCTION,
        chunks_x_size, 
        chunks_y_size, 
        NUM_CIRCUIT_FLAGS, 
        relevant_y_chunks_len
    )(pi.linking_stuff);

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
        FqRangeCheck(0)(proof.outer_sumcheck_claims[i]);
    }

    for(var i = 0; i < 4; i++) {
        FqRangeCheck(0)(proof.inner_sumcheck_claims[i]);
    }

    FqRangeCheck(0)(proof.pub_io_eval);
}

template JoltPreprocessingNNRangeCheck() {
    input JoltPreprocessingNN() jp;

    FqRangeCheck(0)(jp.v_init_final_hash);
    FqRangeCheck(0)(jp.bytecode_words_hash);
}

bus NonUniformSpartanStuff() {
    HyperKZGCommitment() w;

    HyperKZGCommitment() val[3];
    HyperKZGCommitment() row[3];
    HyperKZGCommitment() col[3];

    HyperKZGCommitment() read_ts_row[3];
    HyperKZGCommitment() read_ts_col[3];

    HyperKZGCommitment() final_ts_row[3];
    HyperKZGCommitment() final_ts_col[3];

    HyperKZGCommitment() e_row[3];
    HyperKZGCommitment() e_col[3];
}

bus NonUniformSpartanPI(C, 
                    NUM_MEMORIES, 
                    NUM_INSTRUCTIONS, 
                    MEMORY_OPS_PER_INSTRUCTION,
                    chunks_x_size, 
                    chunks_y_size, 
                    NUM_CIRCUIT_FLAGS, 
                    relevant_y_chunks_len) {
    JoltPreprocessingNN() jolt_pi;
    LinkingStuff1NN(C, 
                    NUM_MEMORIES, 
                    NUM_INSTRUCTIONS, 
                    MEMORY_OPS_PER_INSTRUCTION,
                    chunks_x_size, 
                    chunks_y_size, 
                    NUM_CIRCUIT_FLAGS, 
                    relevant_y_chunks_len) linking_stuff;
    
    HyperKZGVerifierKey() vk;
}

bus NonUniformSpartanProof (outer_num_rounds, inner_num_rounds, num_vars) {

    SumcheckInstanceProof(3, outer_num_rounds) outer_sumcheck_proof; 

    Fq() outer_sumcheck_claims[3];

    SumcheckInstanceProof(2, inner_num_rounds) inner_sumcheck_proof;

    Fq() inner_sumcheck_claims[4];              

    Fq() pub_io_eval;

    HyperKZGProof(num_vars) joint_opening_proof;
}

// component main = VerifySpartan(10, 22, 23, 22, 4, 54, 26, 4, 4, 4, 11, 4, 16);