pragma circom 2.2.1;
include "./../../../fields/non_native/utils.circom";
include "./../utils.circom";
include "./../../../transcript/transcript.circom";
include "./../sum_check/sumcheck.circom";
include "grand_product.circom";

//TODO:- Currently we don't handle the case when claimed_outputs_size = 1;
template SparseVerifyGrandProduct(max_rounds, proof_layers_size, claimed_outputs_size){
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

    output signal r_grand_product[num_challenges + proof_layers_size - 1];
    (up_transcript, final_claim, r_grand_product) <== SparseVerifylayers(max_rounds, num_challenges, proof_layers_size)(proof.gkr_layers, claim, r, int_transcript[1]);
}


template SparseVerifylayers(max_rounds, r_start_size, proof_layers_size){
    input BatchedGrandProductLayerProof(max_rounds) proof_layers[proof_layers_size];
    input signal claim;
    input signal r_start[r_start_size];
    input Transcript() transcript;
    output Transcript() up_transcript; 
    output signal final_claim;
    output signal r_grand_product[proof_layers_size + r_start_size - 1];

    
    var fixed_at_start = r_start_size;
    
    signal int_r_grand_product[proof_layers_size][proof_layers_size + fixed_at_start]; 
    signal final_r_grand_product[proof_layers_size + fixed_at_start - 1];
    for (var i = 0; i < r_start_size; i++){
        int_r_grand_product[0][i] <== r_start[i];
    }

    Transcript() int_transcript[4 * proof_layers_size]; 
    int_transcript[0] <== transcript;

    signal r_sumcheck[proof_layers_size][proof_layers_size + fixed_at_start];
   

    signal sumcheck_claim[proof_layers_size];
    signal eq_eval[proof_layers_size];
    signal int_claim[proof_layers_size + 1]; 
    int_claim[0] <== claim;

    component non_native_sum_check[proof_layers_size];
    component evaluate_eq[proof_layers_size];

    for (var layer_index = 0; layer_index < proof_layers_size; layer_index++){
        non_native_sum_check[layer_index] =  SumCheck(layer_index + fixed_at_start, 3);
        non_native_sum_check[layer_index].initialClaim <== int_claim[layer_index];
        
        for (var i = 0; i < layer_index + fixed_at_start; i++){
            non_native_sum_check[layer_index].sumcheck_proof.uni_polys[i] <== proof_layers[layer_index].proof.uni_polys[i];
        }
   
        non_native_sum_check[layer_index].transcript <== int_transcript[4 * layer_index];
        int_transcript[4 * layer_index + 1] <== non_native_sum_check[layer_index].up_transcript;
        sumcheck_claim[layer_index] <== non_native_sum_check[layer_index].finalClaim;

        for (var i = 0; i < layer_index + fixed_at_start; i ++){
            r_sumcheck[layer_index][i] <== non_native_sum_check[layer_index].randomPoints[i];
        }

        int_transcript[4 * layer_index + 2] <== AppendScalar()(proof_layers[layer_index].left_claim, int_transcript[4 * layer_index + 1]);
        int_transcript[4 * layer_index + 3] <== AppendScalar()(proof_layers[layer_index].right_claim, int_transcript[4 * layer_index + 2]);
        
        evaluate_eq[layer_index] = EvaluateEq(layer_index + fixed_at_start);
        
        for (var i = 0; i < layer_index + fixed_at_start; i++){
            evaluate_eq[layer_index].rx[i] <== r_sumcheck[layer_index][layer_index + fixed_at_start - 1 - i];  //TODO(Ashish):- If this fn doesn't work verify indexes of this signal. 
            evaluate_eq[layer_index].r[i] <== int_r_grand_product[layer_index][i];  

            if (layer_index != proof_layers_size - 1){
                int_r_grand_product[layer_index + 1][i] <== r_sumcheck[layer_index][layer_index + fixed_at_start - 1 - i]; 
            }
            else {
                final_r_grand_product[i] <== r_sumcheck[layer_index][layer_index + fixed_at_start - 1 - i]; 
            }
        }
        
        eq_eval[layer_index] <==  evaluate_eq[layer_index].result;

        if (layer_index != proof_layers_size - 1){
            (int_r_grand_product[layer_index + 1][layer_index + fixed_at_start], int_claim[layer_index + 1], int_transcript[4 * layer_index + 4] ) <== SparseVerifySumcheckClaim()(proof_layers[layer_index].left_claim, proof_layers[layer_index].right_claim, sumcheck_claim[layer_index], eq_eval[layer_index], int_transcript[4 * layer_index + 3]);
        }
        else {
            int_claim[layer_index + 1] <== SparseVerifySumcheckFinalClaim()(proof_layers[layer_index].left_claim, proof_layers[layer_index].right_claim, sumcheck_claim[layer_index], eq_eval[layer_index]);
        }
    }
    up_transcript <== int_transcript[4 * proof_layers_size - 1];
    final_claim <== int_claim[proof_layers_size];
    r_grand_product <== final_r_grand_product;
}

template SparseVerifySumcheckClaim() {
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

template SparseVerifySumcheckFinalClaim() {
    input signal left_claim;
    input signal right_claim;
    input signal sumcheck_claim;    
    input signal eq_eval;                

    output signal grand_product_claim;  
    signal r_layer;

    signal left_right_claim <== left_claim * right_claim;
    
    signal one_minus_left_claim <== 1 - left_claim;
    signal left_right_claim_plus_one_minus_left_claim <== one_minus_left_claim + left_right_claim;

    signal expected_sumcheck_claim <== left_right_claim_plus_one_minus_left_claim * eq_eval;
    expected_sumcheck_claim === sumcheck_claim;

    grand_product_claim <== left_right_claim_plus_one_minus_left_claim;

}

// component main = SparseVerifyGrandProduct(8, 5, 3);
