pragma circom 2.2.1;
include "./../../../fields/non_native/utils.circom";
include "./../utils.circom";
include "./../../../transcript/transcript.circom";
include "./../sum_check/sumcheck.circom";

template VerifyGrandProduct(max_rounds, proof_layers_size, claimed_outputs_size){
    input BatchedGrandProductProof(max_rounds, proof_layers_size) proof;
    input signal claimed_outputs[claimed_outputs_size];
    input Transcript() transcript;

    output Transcript() up_transcript;
    output signal final_claim;

    Transcript() int_transcript[2];
    int_transcript[0] <== AppendScalars(claimed_outputs_size)(claimed_outputs, transcript);

    var np2_claimed_output = NextPowerOf2(claimed_outputs_size);
    var num_challenges = log2(np2_claimed_output);
    signal r[num_challenges];
    (int_transcript[1], r) <== ChallengeVector(num_challenges)(int_transcript[0]);
    
    signal pad_claimed_out[np2_claimed_output] <== Pad(claimed_outputs_size)(claimed_outputs);

    signal eq_r[np2_claimed_output] <== Evals(num_challenges)(r);
    signal claim <== ComputeDotProduct(np2_claimed_output)(pad_claimed_out, eq_r);

    output signal r_grand_product[num_challenges + proof_layers_size];
    (up_transcript, final_claim, r_grand_product) <== Verifylayers(max_rounds, num_challenges, proof_layers_size)(proof.gkr_layers, claim, r, int_transcript[1]);
}

template Verifylayers(max_rounds, r_start_size, proof_layers_size){
    input BatchedGrandProductLayerProof(max_rounds) proof_layers[proof_layers_size];
    input signal claim;
    input signal r_start[r_start_size];
    input Transcript() transcript;
    output Transcript() up_transcript; 
    output signal final_claim;
    output signal r_grand_product[proof_layers_size + r_start_size];

    
    var fixed_at_start = r_start_size;
    signal int_r_grand_product[proof_layers_size + 1][proof_layers_size + fixed_at_start]; 
    
    for (var i = 0; i < r_start_size; i++){
        int_r_grand_product[0][i] <== r_start[i];
    }

    Transcript() int_transcript[4 * proof_layers_size + 1]; 
    int_transcript[0] <== transcript;

    signal r_sumcheck[proof_layers_size][proof_layers_size + fixed_at_start];
   

    signal sumcheck_claim[proof_layers_size];
    signal eq_eval[proof_layers_size];
    signal int_claim[proof_layers_size + 1]; 
    int_claim[0] <== claim;

    component sum_check[proof_layers_size];
    component evaluate_eq[proof_layers_size];

    for (var layer_index = 0; layer_index < proof_layers_size; layer_index++){
        sum_check[layer_index] =  SumCheck(layer_index + fixed_at_start, 3);
        sum_check[layer_index].initialClaim <== int_claim[layer_index];
        
        for (var i = 0; i < layer_index + fixed_at_start; i++){
            sum_check[layer_index].sumcheck_proof.uni_polys[i] <== proof_layers[layer_index].proof.uni_polys[i];
        }
   
        sum_check[layer_index].transcript <== int_transcript[4 * layer_index];
        int_transcript[4 * layer_index + 1] <== sum_check[layer_index].up_transcript;
        sumcheck_claim[layer_index] <== sum_check[layer_index].finalClaim;

        for (var i = 0; i < layer_index + fixed_at_start; i ++){
            r_sumcheck[layer_index][i] <== sum_check[layer_index].randomPoints[i];
        }

        int_transcript[4 * layer_index + 2] <== AppendScalar()(proof_layers[layer_index].left_claim, int_transcript[4 * layer_index + 1]);
        int_transcript[4 * layer_index + 3] <== AppendScalar()(proof_layers[layer_index].right_claim, int_transcript[4 * layer_index + 2]);
        
        evaluate_eq[layer_index] = EvaluateEq(layer_index + fixed_at_start);
        
        for (var i = 0; i < layer_index + fixed_at_start; i++){
            evaluate_eq[layer_index].rx[i] <== r_sumcheck[layer_index][layer_index + fixed_at_start - 1 - i];  //TODO(Ashish):- If this fn doesn't work verify indexes of this signal. 
            evaluate_eq[layer_index].r[i] <== int_r_grand_product[layer_index][i];  
            int_r_grand_product[layer_index + 1][i] <== r_sumcheck[layer_index][layer_index + fixed_at_start - 1 - i]; 
        }
        
        eq_eval[layer_index] <==  evaluate_eq[layer_index].result;
                   
        (int_r_grand_product[layer_index + 1][layer_index + fixed_at_start], int_claim[layer_index + 1], int_transcript[4 * layer_index + 4] ) <== VerifySumcheckClaim()(proof_layers[layer_index].left_claim, proof_layers[layer_index].right_claim, sumcheck_claim[layer_index], eq_eval[layer_index], int_transcript[4 * layer_index + 3]);
    }
    up_transcript <== int_transcript[4 * proof_layers_size ];
    final_claim <== int_claim[proof_layers_size];
    r_grand_product <== int_r_grand_product[proof_layers_size]; 
}

template VerifySumcheckClaim() {
    input signal left_claim;
    input signal right_claim;
    input signal sumcheck_claim;    
    input signal eq_eval;                
    input Transcript() transcript; 
    output signal r_grand_product; 
    output signal grand_product_claim;  
    output Transcript() up_transcript; 
    signal r_layer;

    signal left_right_claim <== left_claim * right_claim;
    
    signal expected_sumcheck_claim <== left_right_claim * eq_eval;
    expected_sumcheck_claim === sumcheck_claim;
    
    (up_transcript, r_layer) <== ChallengeScalar()(transcript);

    signal right_minus_left_claim <== right_claim - left_claim;
    signal temp <== r_layer * right_minus_left_claim;
    grand_product_claim <== left_claim + temp;
    
    r_grand_product <== r_layer;
}


bus BatchedGrandProductLayerProof(rounds) {
    SumcheckInstanceProof(3, rounds) proof;
    signal left_claim;
    signal right_claim;
}

bus BatchedGrandProductProof(max_rounds, no_of_layers){
    BatchedGrandProductLayerProof(max_rounds) gkr_layers[no_of_layers];
}

// component main = VerifyGrandProduct(8, 5, 3);
