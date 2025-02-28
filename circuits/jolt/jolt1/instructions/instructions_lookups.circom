pragma circom 2.2.1;
include "./../grand_product/grand_product.circom";
 include "./../grand_product/sparse_grand_product.circom";
include "./instruction_bus.circom";
include "./instruction_combine_lookups.circom";
include "./instruction_subtables.circom";
include "./../utils.circom";
include "./instructions_helper.circom";
include "./../opening_proof/opening_proof_bus.circom";
include "./../mem_checking/common.circom";

template VerifyInstructionLookups(max_rounds, max_rounds_init_final,
                primary_sumcheck_degree, primary_sumcheck_num_rounds,
                C, NUM_MEMORIES, NUM_INSTRUCTIONS, NUM_SUBTABLES,
                read_write_grand_product_layers, init_final_grand_product_layers, WORD_SIZE, M
                ) {

    input InstructionLookupsProof(max_rounds, max_rounds_init_final,
            primary_sumcheck_degree, primary_sumcheck_num_rounds, NUM_MEMORIES, NUM_INSTRUCTIONS, NUM_SUBTABLES,
            read_write_grand_product_layers, init_final_grand_product_layers) proof;

    input Transcript() transcript;
    output Transcript() up_transcript;

    Transcript() int_transcript[3]; 

    signal r_eq[primary_sumcheck_num_rounds];
    signal r_primary_sumcheck[primary_sumcheck_num_rounds];
    signal claim_last;
    (int_transcript[0], r_eq) <== ChallengeVector(primary_sumcheck_num_rounds)(transcript);


    
    (int_transcript[1], claim_last, r_primary_sumcheck) <== SumCheck(primary_sumcheck_num_rounds, primary_sumcheck_degree)(0, proof.primary_sumcheck.sumcheck_proof, int_transcript[0]);

    // Verify that eq(r, r_z) * [f_1(r_z) * g(E_1(r_z)) + ... + f_F(r_z) * E_F(r_z))] = claim_last
    signal eq_eval <== EvaluateEq(primary_sumcheck_num_rounds)(r_eq, r_primary_sumcheck);
        

   signal combined_lookups <== CombineLookups(C, M, WORD_SIZE,
                        NUM_INSTRUCTIONS,       
                        NUM_MEMORIES)(proof.primary_sumcheck.openings.E_poly_openings, proof.primary_sumcheck.openings.flag_openings);


    signal combined_lookups_minus_lookup_outputs_opening <== (combined_lookups - proof.primary_sumcheck.openings.lookup_outputs_opening);

    signal  left_value <== (eq_eval * combined_lookups_minus_lookup_outputs_opening);
 
    left_value === claim_last;

    signal primary_sumcheck_openings[NUM_MEMORIES + NUM_INSTRUCTIONS + 1];
       
    for (var i = 0; i < NUM_MEMORIES; i++){
        primary_sumcheck_openings[i] <== proof.primary_sumcheck.openings.E_poly_openings[i];
    }
    for (var i = 0; i < NUM_INSTRUCTIONS; i++){
        primary_sumcheck_openings[i + NUM_MEMORIES] <== proof.primary_sumcheck.openings.flag_openings[i];
    }

    primary_sumcheck_openings[NUM_MEMORIES + NUM_INSTRUCTIONS] <== proof.primary_sumcheck.openings.lookup_outputs_opening;
        
    output VerifierOpening(primary_sumcheck_num_rounds) instruction_openings;
    
     (int_transcript[2], instruction_openings) <==  OpeningAccumulator(NUM_MEMORIES + NUM_INSTRUCTIONS + 1, primary_sumcheck_num_rounds)
     ( r_primary_sumcheck, primary_sumcheck_openings, int_transcript[1]);
     
      output VerifierOpening(read_write_grand_product_layers - 1) inst_read_write_openings;
      output VerifierOpening(init_final_grand_product_layers) inst_init_final_openings;

    component verify_memory_checking = VerifyMemoryChecking(max_rounds, max_rounds_init_final, C, 
                        NUM_MEMORIES, 
                        NUM_INSTRUCTIONS, 
                        NUM_SUBTABLES,
                        read_write_grand_product_layers,
                        init_final_grand_product_layers,
                        WORD_SIZE);

    verify_memory_checking.proof <==   proof.memory_checking_proof;       
    verify_memory_checking.transcript <==  int_transcript[2];

    up_transcript <== verify_memory_checking.up_transcript;

    inst_read_write_openings <== verify_memory_checking.inst_read_write_verifier_opening;
    inst_init_final_openings <== verify_memory_checking.inst_init_final_verifier_opening;
    
}    

template VerifyMemoryChecking(max_rounds, max_rounds_init_final, C, 
                                NUM_MEMORIES, 
                                NUM_INSTRUCTIONS, 
                                NUM_SUBTABLES,
                                read_write_grand_product_layers,
                                init_final_grand_product_layers,
                                WORD_SIZE){
    
    input InstMemoryCheckingProof(max_rounds, max_rounds_init_final,  NUM_MEMORIES, 
                    NUM_SUBTABLES, NUM_INSTRUCTIONS,
                    read_write_grand_product_layers,
                    init_final_grand_product_layers) proof;

    input Transcript() transcript;
    output Transcript() up_transcript;

    Transcript() int_transcript[7];  
    signal gamma, tau;
    (int_transcript[0], gamma) <== ChallengeScalar()(transcript);
   
    (int_transcript[1], tau) <== ChallengeScalar()(int_transcript[0]);
   
    CheckMultisetEqualityInst(NUM_MEMORIES, NUM_SUBTABLES)(proof.multiset_hashes);

    int_transcript[2] <== AppendToTranscript(NUM_MEMORIES, NUM_SUBTABLES, NUM_MEMORIES)(int_transcript[1], proof.multiset_hashes);

    signal read_write_hashes[2 * NUM_MEMORIES];
    signal init_final_hashes[NUM_MEMORIES + NUM_SUBTABLES];

    (read_write_hashes, init_final_hashes) <== InterleaveHashesForGKR(NUM_MEMORIES, NUM_SUBTABLES, NUM_MEMORIES)(proof.multiset_hashes);

    var log_num_read_write_hashes = log2(NextPowerOf2(2 * NUM_MEMORIES));

    var r_read_write_len = read_write_grand_product_layers + log_num_read_write_hashes - 1;

    signal read_write_claim, r_read_write[r_read_write_len];
    
    (int_transcript[3], read_write_claim, r_read_write) <== SparseVerifyGrandProduct(max_rounds, read_write_grand_product_layers, 2 * NUM_MEMORIES)(proof.read_write_grand_product, read_write_hashes, int_transcript[2]);

    signal r_read_write_batch_index[log_num_read_write_hashes];

    signal r_read_write_opening[read_write_grand_product_layers - 1];

    (r_read_write_batch_index, r_read_write_opening) <== SplitAt(log_num_read_write_hashes, r_read_write_len)(r_read_write);

 


    var log_num_init_final_hashes = log2(NextPowerOf2(NUM_MEMORIES + NUM_SUBTABLES));

    var r_init_final_len = init_final_grand_product_layers + log_num_init_final_hashes;

    signal init_final_claim, r_init_final[r_init_final_len];




     (int_transcript[4], init_final_claim, r_init_final)  <== VerifyGrandProduct(max_rounds_init_final, init_final_grand_product_layers, NUM_MEMORIES + NUM_SUBTABLES)(proof.init_final_grand_product, init_final_hashes, int_transcript[3]);

    signal r_init_final_batch_index[log_num_init_final_hashes];
    
    signal r_init_final_opening[init_final_grand_product_layers];
    
    (r_init_final_batch_index, r_init_final_opening) <== SplitAt(log_num_init_final_hashes, r_init_final_len)(r_init_final);

   
    var num_read_write_commits = C +  2 * NUM_MEMORIES + NUM_INSTRUCTIONS + 1;
    
    signal read_write_claims[num_read_write_commits]; 
    for (var i = 0; i < C; i++){
        read_write_claims[i] <== proof.openings[i];
    }
    for (var i = 0; i < NUM_MEMORIES; i++) {
        read_write_claims[i + C] <== proof.openings[i + C];
        read_write_claims[i + C +  NUM_MEMORIES] <== proof.openings[i + C + 2 * NUM_MEMORIES];

    }

    for (var i = 0; i < NUM_INSTRUCTIONS; i++) {
        read_write_claims[i + C +  2 * NUM_MEMORIES] <== proof.openings[i + C +  3 * NUM_MEMORIES];
    }

    read_write_claims[num_read_write_commits - 1] <== proof.openings[ C +   3 * NUM_MEMORIES +  NUM_INSTRUCTIONS];

    output VerifierOpening(read_write_grand_product_layers - 1) inst_read_write_verifier_opening;
    (int_transcript[5], inst_read_write_verifier_opening) <== OpeningAccumulator(num_read_write_commits, read_write_grand_product_layers - 1)( r_read_write_opening, read_write_claims, int_transcript[4]);
    
    signal init_final_claims[NUM_MEMORIES];
    for (var i = 0; i < NUM_MEMORIES; i++){
        init_final_claims[i] <== proof.openings[C + NUM_MEMORIES + i];
    }

    output VerifierOpening(init_final_grand_product_layers) inst_init_final_verifier_opening;
    (int_transcript[6], inst_init_final_verifier_opening) <== OpeningAccumulator(NUM_MEMORIES, init_final_grand_product_layers) 
                                                        ( r_init_final_opening,
                                                        init_final_claims, int_transcript[5]);
    

    up_transcript <== int_transcript[6];

    signal a_init_final;
    signal v_init_final[26];
    (a_init_final, v_init_final)  <== ComputeVerifierOpenings(init_final_grand_product_layers, WORD_SIZE)(r_init_final_opening);

    signal int_openings[C + 3 * NUM_MEMORIES + NUM_INSTRUCTIONS + 28] ;

    for (var i = 0; i < (C + 3 * NUM_MEMORIES+NUM_INSTRUCTIONS + 1  ); i++){
        int_openings[i] <== proof.openings[i];
    }
    
    int_openings[C + 3 * NUM_MEMORIES+NUM_INSTRUCTIONS + 1] <== a_init_final;
    for (var i = C + 3 * NUM_MEMORIES+NUM_INSTRUCTIONS + 2; i < C + 3 * NUM_MEMORIES+NUM_INSTRUCTIONS + 28; i++){
        int_openings[i] <== v_init_final[i - (C + 3 * NUM_MEMORIES+NUM_INSTRUCTIONS +2)];
    }


    CheckFingerprintsInstructionsInstructionLookups(log_num_read_write_hashes,
                    log_num_init_final_hashes,
                    C, 
                    NUM_MEMORIES, 
                    NUM_INSTRUCTIONS,
                    NUM_SUBTABLES  
        )(
        read_write_claim,
        init_final_claim,
        r_read_write_batch_index,
        r_init_final_batch_index,
        int_openings,
        gamma,
        tau
    );


}


template ComputeVerifierOpenings(r_init_final_len, WORD_SIZE){
        input signal r_init_final[r_init_final_len];
        output signal a_init_final;
        output signal v_init_final[26];
        a_init_final <== EvaluateIdentityPoly(r_init_final_len)(r_init_final);  

        v_init_final[0]  <== evaluate_mle_and(r_init_final_len)(r_init_final);
        v_init_final[1] <==  evaluate_mle_eq_abs(r_init_final_len)(r_init_final);
        v_init_final[2] <==  evaluate_mle_eq(r_init_final_len)(r_init_final);
        v_init_final[3] <==  evaluate_mle_left_msb(r_init_final_len)(r_init_final);
        v_init_final[4] <==  evaluate_mle_right_msb(r_init_final_len)(r_init_final);
        v_init_final[5] <==  evaluate_mle_identiy(r_init_final_len)(r_init_final);
        v_init_final[6] <==  evaluate_mle_lt_abs(r_init_final_len)(r_init_final);
        v_init_final[7] <==  evaluate_mle_ltu(r_init_final_len)(r_init_final);
        v_init_final[8] <==  evaluate_mle_or(r_init_final_len)(r_init_final);
        v_init_final[9] <==  evaluate_mle_sign_extend(r_init_final_len, 16)(r_init_final);    // jolt fizes WIDTH to 16.

        // eq_vec is used by the following 9 template calls. Computing it outside as it is non-native
        // operations intensive.
        var b = r_init_final_len / 2;
        var min = 0;
        var log_WORD_SIZE = log2(WORD_SIZE);
        if (log_WORD_SIZE < b) {
            min = log_WORD_SIZE;
        } else {
            min =  b;
        }
        signal y[min] <== TruncateVec(r_init_final_len - min, r_init_final_len, r_init_final_len)(r_init_final);
        signal eq_vec[1 << min] <== Evals(min)(y);
        v_init_final[10] <==  evaluate_mle_sll(r_init_final_len, 1 << min, 0, WORD_SIZE)(r_init_final, eq_vec);
        v_init_final[11] <==  evaluate_mle_sll(r_init_final_len, 1 << min, 1, WORD_SIZE)(r_init_final, eq_vec);
        v_init_final[12] <==  evaluate_mle_sll(r_init_final_len, 1 << min, 2, WORD_SIZE)(r_init_final, eq_vec);
        v_init_final[13] <==  evaluate_mle_sll(r_init_final_len, 1 << min, 3, WORD_SIZE)(r_init_final, eq_vec);
        v_init_final[14] <==  evaluate_mle_sra_sign(r_init_final_len,1 << min, WORD_SIZE)(r_init_final, eq_vec);
        v_init_final[15] <==  evaluate_mle_srl(r_init_final_len, 1 << min, 0, WORD_SIZE)(r_init_final, eq_vec);
        v_init_final[16] <==  evaluate_mle_srl(r_init_final_len, 1 << min, 1, WORD_SIZE)(r_init_final, eq_vec);
        v_init_final[17] <==  evaluate_mle_srl(r_init_final_len, 1 << min, 2, WORD_SIZE)(r_init_final, eq_vec);
        v_init_final[18] <==  evaluate_mle_srl(r_init_final_len, 1 << min, 3, WORD_SIZE)(r_init_final, eq_vec);
        v_init_final[19] <==  evaluate_mle_truncate_overflow(r_init_final_len, WORD_SIZE)(r_init_final);
        v_init_final[20] <==  evaluate_mle_xor(r_init_final_len)(r_init_final);
        v_init_final[21] <==  evaluate_mle_left_is_zero(r_init_final_len)(r_init_final);
        v_init_final[22] <==  evaluate_mle_right_is_zero(r_init_final_len)(r_init_final);
        v_init_final[23] <==  evaluate_mle_div_by_zero(r_init_final_len)(r_init_final);
        v_init_final[24] <==  evaluate_mle_low_bit(r_init_final_len,0)(r_init_final);
        v_init_final[25] <==  evaluate_mle_low_bit(r_init_final_len,1)(r_init_final);

 }



template CheckFingerprintsInstructionsInstructionLookups(
                    log_num_read_write_hashes,
                    log_num_init_final_hashes,
                    C, 
                    NUM_MEMORIES, 
                    NUM_INSTRUCTIONS,
                    NUM_SUBTABLES  
                    ){

    input signal read_write_claim;
    input signal init_final_claim;

    input signal r_read_write_batch_index[log_num_read_write_hashes];
    input signal r_init_final_batch_index[log_num_init_final_hashes];

    input signal openings[C + 3 * NUM_MEMORIES + NUM_INSTRUCTIONS + 28]; 

    input signal  gamma;
    input signal  tau;

     signal read_values[NUM_MEMORIES][4];




    read_values  <== ComputeReadtuple(NUM_INSTRUCTIONS, NUM_MEMORIES, C)(openings);

    signal write_values[NUM_MEMORIES][4] <== ComputeWritetuple(NUM_INSTRUCTIONS, NUM_MEMORIES, C)(openings);

    signal init_values[NUM_SUBTABLES][4] <== ComputeInittuple(NUM_INSTRUCTIONS, NUM_MEMORIES, C, NUM_SUBTABLES)(openings);

    signal final_values[NUM_MEMORIES][4] <== ComputeFinaltuple(NUM_INSTRUCTIONS, NUM_MEMORIES, C)(openings);

    signal read_write_tuples[2 * NUM_MEMORIES][4];

    signal init_final_tuples[NUM_SUBTABLES +  NUM_MEMORIES][4];

   (read_write_tuples, init_final_tuples)  <== InterleaveInstructionLookups(NUM_MEMORIES, NUM_SUBTABLES)
                                                (read_values, write_values, init_values, final_values);

    var read_write_flags_len = NextPowerOf2(2 * NUM_MEMORIES);

    signal  read_write_flags[read_write_flags_len];

    for (var i = 0; i < 2 * NUM_MEMORIES; i++) {
        read_write_flags[i] <== read_write_tuples[i][3];
    }

    for (var i = 2 * NUM_MEMORIES; i < read_write_flags_len; i++) {
        read_write_flags[i] <== 1;
    }

    var log_num_read_write_hashes_pow_2 = 1 << log_num_read_write_hashes; 
    signal  evals[log_num_read_write_hashes_pow_2] <== Evals(log_num_read_write_hashes)(r_read_write_batch_index);

    signal combined_flags <== ComputeDotProduct(read_write_flags_len)(evals, read_write_flags );

    signal combined_read_write_fingerprint[2 * NUM_MEMORIES + 1];
    signal temp_combined_read_write_fingerprint[2 * NUM_MEMORIES ];

    signal fingerprint_vec[2 * NUM_MEMORIES ];
    combined_read_write_fingerprint[0] <== 0;

    for (var i = 0; i < 2 * NUM_MEMORIES; i++) {
        fingerprint_vec[i] <== FingerprintInstructionLookups()(read_write_tuples[i][0], read_write_tuples[i][1], read_write_tuples[i][2], gamma, tau);
        temp_combined_read_write_fingerprint[i] <== (evals[i] *  fingerprint_vec[i] );
        combined_read_write_fingerprint[i+1] <== (combined_read_write_fingerprint[i] + temp_combined_read_write_fingerprint[i]);
    }

    signal combined_flags_and_fingerprint[3];
   
    combined_flags_and_fingerprint[0] <== (combined_flags * combined_read_write_fingerprint[2 * NUM_MEMORIES] );
    combined_flags_and_fingerprint[1] <== (1 + combined_flags_and_fingerprint[0]);

    combined_flags_and_fingerprint[2] <== (combined_flags_and_fingerprint[1] - combined_flags);
    read_write_claim === combined_flags_and_fingerprint[2];

    var log_num_init_final_hashes_pow_2 = 1 << log_num_init_final_hashes; 
    signal  r_init_final_batch_index_evals[log_num_init_final_hashes_pow_2] <== Evals(log_num_init_final_hashes)(r_init_final_batch_index);

    signal combined_init_final_fingerprint[NUM_SUBTABLES +  NUM_MEMORIES + 1];
     signal temp_combined_init_final_fingerprint[NUM_SUBTABLES +  NUM_MEMORIES ];

    signal init_final_tuples_fingerprint_vec[NUM_SUBTABLES +  NUM_MEMORIES ];
    combined_init_final_fingerprint[0] <== 0;

    for (var i = 0; i < NUM_SUBTABLES +  NUM_MEMORIES; i++) {
        init_final_tuples_fingerprint_vec[i] <== FingerprintInstructionLookups()(init_final_tuples[i][0], init_final_tuples[i][1], init_final_tuples[i][2], gamma, tau);
        temp_combined_init_final_fingerprint[i] <== (r_init_final_batch_index_evals[i] * init_final_tuples_fingerprint_vec[i] );
        combined_init_final_fingerprint[i+1] <== (combined_init_final_fingerprint[i] + temp_combined_init_final_fingerprint[i]);
    }

    combined_init_final_fingerprint[NUM_SUBTABLES +  NUM_MEMORIES] === init_final_claim;

}

template FingerprintInstructionLookups() {
    // Declare the input signals
    input signal a , v, t; 
    input signal gamma, tau;   
    output signal result; 

    signal gamma_square;
    gamma_square <==  (gamma  * gamma);  
    
    signal gamma_square_t;
    gamma_square_t <==  (gamma_square * t);  

    signal gamma_v;
    gamma_v <== (gamma * v); 
    signal temp_res;
    temp_res  <==  (gamma_square_t + gamma_v); 
    signal temp_res_a;
    temp_res_a <==  (temp_res + a); 
    result <== (temp_res_a  - tau); 
}

template ComputeReadtuple(NUM_INSTRUCTIONS, NUM_MEMORIES, C){


    input signal openings[ C + 3 * NUM_MEMORIES + NUM_INSTRUCTIONS + 28];
    output signal  read_tuple[NUM_MEMORIES][4];

    var temp_instruction_flags[NUM_INSTRUCTIONS];
    for (var i = 0; i < NUM_INSTRUCTIONS; i++) {
        temp_instruction_flags[i] = openings[C+ 3* NUM_MEMORIES + i];
    }


    signal memory_flags[NUM_MEMORIES] <==  MemoryFlags(NUM_INSTRUCTIONS, NUM_MEMORIES)(temp_instruction_flags);


    for (var i = 0; i < NUM_MEMORIES; i++) {
        var dim_index = MemorytoDimensionindex(i);
        
        read_tuple[i][0] <==  openings[dim_index];
        read_tuple[i][1] <==   openings[C + 2 * NUM_MEMORIES +i];
        read_tuple[i][2] <==   openings[C + i];
        read_tuple[i][3] <==   memory_flags[i];
    }
 }


template ComputeWritetuple(NUM_INSTRUCTIONS, NUM_MEMORIES, C){

     
    input signal openings[C + 3 * NUM_MEMORIES+NUM_INSTRUCTIONS + 28];
    output signal write_tuple[NUM_MEMORIES][4];
    signal temp_write_tuple[NUM_MEMORIES][4];

    temp_write_tuple <== ComputeReadtuple(NUM_INSTRUCTIONS, NUM_MEMORIES, C)(openings);

    for (var i = 0; i < NUM_MEMORIES; i++) { 
        write_tuple[i][0] <==  temp_write_tuple[i][0];
        write_tuple[i][1] <==  temp_write_tuple[i][1];
        write_tuple[i][2] <==  (temp_write_tuple[i][2] + 1);
        write_tuple[i][3] <==  temp_write_tuple[i][3];
    }

 }


template ComputeInittuple(NUM_INSTRUCTIONS, NUM_MEMORIES, C ,NUM_SUBTABLES){

    input signal openings[C + 3 * NUM_MEMORIES + NUM_INSTRUCTIONS + 28];
    output signal init_tuple[NUM_SUBTABLES][4];

    for (var i = 0; i < NUM_SUBTABLES; i++) { 
        init_tuple[i][0] <==  openings[C+ 3* NUM_MEMORIES+NUM_INSTRUCTIONS + 1];
        init_tuple[i][1] <==  openings[C+ 3* NUM_MEMORIES+NUM_INSTRUCTIONS+ 2 + i];
        init_tuple[i][2] <==  0;
        init_tuple[i][3] <==  0;
    }
}

template ComputeFinaltuple(NUM_INSTRUCTIONS, NUM_MEMORIES, C){
    input  signal openings[C + 3 * NUM_MEMORIES + NUM_INSTRUCTIONS + 28];
    output signal  final_tuple[NUM_MEMORIES][4];

    for (var i = 0; i < NUM_MEMORIES; i++) {   
        final_tuple[i][0] <==  openings[C+ 3 * NUM_MEMORIES + NUM_INSTRUCTIONS + 1];
        var index = MemoryToSubtableIndex(i);
        final_tuple[i][1] <==  openings[ C+ 3 * NUM_MEMORIES + NUM_INSTRUCTIONS + 2 + index];
        final_tuple[i][2] <==  openings[C + NUM_MEMORIES + i];
        final_tuple[i][3] <==  0;
    }
 }


template MemoryFlags(NUM_INSTRUCTIONS, NUM_MEMORIES) {
    signal input instruction_flags[NUM_INSTRUCTIONS];
    signal output memory_flags_valid[NUM_MEMORIES];

    var  memory_flags[NUM_MEMORIES];

    // Initialize all memory flags to 0
    for (var i = 0; i < NUM_MEMORIES; i++) {
        memory_flags[i] = 0;
    }
    var instruction_to_memory_indices_len[26] = [4, 4, 4, 4, 4, 4, 9, 7, 4, 9, 7, 4, 5, 4, 5, 4, 4, 2, 4, 4, 8, 18, 11, 8, 1, 2] ; 
    for (var instruction_index = 0; instruction_index < NUM_INSTRUCTIONS; instruction_index++) {
         for (var i = 0; i < instruction_to_memory_indices_len[instruction_index]; i++) {
             var temp_memory_index = InstructionToMemoryIndices(instruction_index, i );
              memory_flags[temp_memory_index] += instruction_flags[instruction_index];
         }
     }

    for (var i = 0; i < NUM_MEMORIES; i++) {
        memory_flags_valid[i] <-- memory_flags[i];
    }
}


template InterleaveInstructionLookups(NUM_MEMORIES, NUM_SUBTABLES) {
   input  signal read_values[NUM_MEMORIES][4];
   input  signal write_values[NUM_MEMORIES][4];
   input  signal init_values[NUM_SUBTABLES][4];
   input  signal final_values[NUM_MEMORIES][4];
   output signal read_write_values[2 * NUM_MEMORIES][4];
   output signal init_final_values[NUM_SUBTABLES +  NUM_MEMORIES][4];

    for (var i = 0; i < NUM_MEMORIES; i++) {
        read_write_values[2 * i][0] <== read_values[i][0];
        read_write_values[2 * i][1] <== read_values[i][1];
        read_write_values[2 * i][2] <== read_values[i][2];
        read_write_values[2 * i][3] <== read_values[i][3];
        
        
        read_write_values[2 * i + 1][0] <== write_values[i][0];
        read_write_values[2 * i + 1][1] <== write_values[i][1];
        read_write_values[2 * i + 1][2] <== write_values[i][2];
        read_write_values[2 * i + 1][3] <== write_values[i][3];
    }

    var index = 0;
    var index2 = 0;
    var  subtable_to_memory_indices_len[26] = [4, 1, 4, 1, 1, 4, 1, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 4, 4, 4, 1, 1] ;
    for (var i = 0; i < NUM_SUBTABLES; i++) {
        init_final_values[index][0] <== init_values[i][0];
        init_final_values[index][1] <== init_values[i][1];
        init_final_values[index][2] <== init_values[i][2];
        init_final_values[index][3] <== init_values[i][3];
        index += 1;
    
        for (var j = 0; j < subtable_to_memory_indices_len[i]; j++) {
            init_final_values[index][0] <== final_values[index2][0];
            init_final_values[index][1] <== final_values[index2][1];
            init_final_values[index][2] <== final_values[index2][2];
            init_final_values[index][3] <== final_values[index2][3];

            index += 1;
            index2 += 1;
        }
    }
}

template InterleaveHashesForGKR(
    num_read_write_hashes,
    num_init_hashes,
    num_final_hashes
)
{
        input MultisetHashes(num_read_write_hashes,
                                num_init_hashes, num_final_hashes) multiset_hashes;
           
        signal output  read_write_hashes[2 * num_read_write_hashes];
        signal output  init_final_hashes[num_init_hashes +  num_final_hashes];

    for (var i = 0; i < num_read_write_hashes; i++) {
        read_write_hashes[2*i] <== multiset_hashes.read_hashes[i];
        read_write_hashes[2*i + 1] <== multiset_hashes.write_hashes[i];
    }
    
    var subtable_to_memory_indices_len[26] = [4, 1, 4, 1, 1, 4, 1, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 4, 4, 4, 1, 1] ;
    
    var count = 0;
    var count2 = 0;
    for (var i = 0; i < 26; i++){
        init_final_hashes[count] <== multiset_hashes.init_hashes[i];
        count += 1;
        for (var j = 0; j < subtable_to_memory_indices_len[i]; j++) {
            init_final_hashes[count] <== multiset_hashes.final_hashes[count2];
            count += 1;
            count2 += 1;
        }
    }

}

