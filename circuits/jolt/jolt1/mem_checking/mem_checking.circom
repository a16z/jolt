pragma circom 2.2.1;
include "./../../../fields/non_native/utils.circom";
include "./../utils.circom";
include "./../../../transcript/transcript.circom";
include "./../jolt1_buses.circom";
include "mem_checking_buses.circom";
include "./../grand_product/grand_product.circom";
include "common.circom";

template VerifyMemoryCheckingBytecode( 
                                num_evals,
                                num_read_write_hashes, 
                                num_init_final_hashes,
                                read_write_grand_product_layers,
                                init_final_grand_product_layers,
                                C,                         
                                max_rounds) {

    input BytecodePreprocessing( 
                                num_evals) preprocessing;


    input BytecodeProof(num_read_write_hashes, 
                        num_init_final_hashes,
                        read_write_grand_product_layers,
                        init_final_grand_product_layers,max_rounds) proof;

    input Transcript() transcript;
    output Transcript() up_transcript;


    Transcript() int_transcript[7];  
    signal gamma, tau ;
    (int_transcript[0], gamma) <== ChallengeScalar()(transcript);


     (int_transcript[1], tau) <== ChallengeScalar()(int_transcript[0]);
   

    CheckMultisetEquality( 
                            num_evals,
                            num_read_write_hashes,
                            num_init_final_hashes)(preprocessing, proof.multiset_hashes);

    int_transcript[2] <== AppendToTranscript(num_read_write_hashes,
                                            num_init_final_hashes, num_init_final_hashes)(int_transcript[1], proof.multiset_hashes);
   
    var  read_write_hashes_len = 2 * num_read_write_hashes;
    var  init_final_hashes_len = 2 * num_init_final_hashes;
    signal read_write_hashes[read_write_hashes_len];
    signal init_final_hashes[init_final_hashes_len];
     
    (read_write_hashes, init_final_hashes) <== InterleaveHashes(num_read_write_hashes, num_init_final_hashes, num_init_final_hashes)(proof.multiset_hashes);
   
    var log_num_read_write_hashes = log2(NextPowerOf2(read_write_hashes_len));
    var r_read_write_len = read_write_grand_product_layers + log_num_read_write_hashes;
    signal read_write_claim, r_read_write[r_read_write_len];
     (int_transcript[3], read_write_claim, r_read_write) <== VerifyGrandProduct(max_rounds, read_write_grand_product_layers, read_write_hashes_len)(proof.read_write_grand_product, read_write_hashes, int_transcript[2]);

    signal r_read_write_batch_index[log_num_read_write_hashes];
    
    signal r_read_write_opening[read_write_grand_product_layers];

    (r_read_write_batch_index, r_read_write_opening) <== SplitAt(log_num_read_write_hashes, 
                                                                    r_read_write_len) (r_read_write);

    var log_num_init_final_hashes = log2(NextPowerOf2( init_final_hashes_len));
    var r_init_final_len = init_final_grand_product_layers + log_num_init_final_hashes;

    signal init_final_claim, r_init_final[r_init_final_len];
    (int_transcript[4], init_final_claim, r_init_final)  <== VerifyGrandProduct(max_rounds,init_final_grand_product_layers, init_final_hashes_len)(proof.init_final_grand_product, init_final_hashes, int_transcript[3]);


    signal r_init_final_batch_index[log_num_init_final_hashes];
    signal r_init_final_opening[init_final_grand_product_layers];
    (r_init_final_batch_index, r_init_final_opening) <== SplitAt(log_num_init_final_hashes, r_init_final_len)
                                                                (r_init_final);

    signal read_write_claims[8]; 
    for (var i = 0; i < 8; i++) {
        read_write_claims[i] <== proof.openings[i];
    }
    
     output VerifierOpening(read_write_grand_product_layers) byte_code_read_write_verifier_opening;

     (int_transcript[5], byte_code_read_write_verifier_opening) <== OpeningAccumulator(8, read_write_grand_product_layers)( r_read_write_opening, read_write_claims, int_transcript[4]);
         
    signal init_final_claims[1];
    init_final_claims[0] <== proof.openings[8];
    output VerifierOpening(init_final_grand_product_layers) init_final_verifier_opening;
    (int_transcript[6], init_final_verifier_opening) <== OpeningAccumulator(1, init_final_grand_product_layers)
                                                    ( r_init_final_opening,
                                                    init_final_claims, int_transcript[5]);


    up_transcript <== int_transcript[6];
    signal verifier_openings[16]  <== ComputeVerifierOpeningsBytecode(
        init_final_grand_product_layers,
        
        num_evals)(r_init_final_opening, proof.openings, preprocessing);

    CheckFingerprintsBytecode(
            num_evals,
            log_num_read_write_hashes,
            log_num_init_final_hashes
    )(preprocessing,read_write_claim, init_final_claim, r_read_write_batch_index,r_init_final_batch_index, verifier_openings, gamma, tau);

}


template CheckMultisetEquality( 
                                num_evals,
                                num_read_write_hashes,
                                num_init_final_hashes) {

    input BytecodePreprocessing( num_evals) preprocessing;
    input MultisetHashes(num_read_write_hashes, num_init_final_hashes, num_init_final_hashes) multiset_hashes;

    var NUM_MEMORIES = num_read_write_hashes;


    signal left_value[NUM_MEMORIES];
    signal right_value[NUM_MEMORIES];

    for (var i = 0; i < NUM_MEMORIES; i++) {
        left_value[i] <== multiset_hashes.read_hashes[i] *  multiset_hashes.final_hashes[i];
        right_value[i]  <==  multiset_hashes.init_hashes[i] * multiset_hashes.write_hashes[i];   
        left_value[i] === right_value[i]; 
    }
}


template  ComputeVerifierOpeningsBytecode(
    init_final_grand_product_layers,
    
    num_evals){

    input   signal  r_init_final[init_final_grand_product_layers] ;
    input   signal  openings[9];
    input BytecodePreprocessing( 
        num_evals) preprocessing;

    output  signal int_openings[16] ;


    for (var i = 0; i < 9; i++) {
        int_openings[i] <== openings[i];
    }

   
                   
    int_openings[9]  <== EvaluateIdentityPoly(init_final_grand_product_layers)(r_init_final);
   

    signal  chi[1 << init_final_grand_product_layers]   <==   Evals(init_final_grand_product_layers)(r_init_final);

    for (var i = 0; i < 6; i++) { 
        int_openings[10 + i] <==  ComputeDotProduct(1 << init_final_grand_product_layers)(preprocessing.v_init_final[i], chi);
    }

}


template FingerprintBytecode() {
    // Declare the input signals
    input  signal inputs[8]; 
    input  signal gamma, tau ;   
    output  signal out; 
    signal gamma_term[9];
    gamma_term[0] <== 1;
    signal result[9];
    result[0] <== 0;
    signal temp_result[8];
    for (var i = 0; i < 8; i++) {
        temp_result[i] <== inputs[i] * gamma_term[i];
        result[i + 1] <==  temp_result[i] + result[i];
        gamma_term[i + 1] <== gamma * gamma_term[i];
    }
    out <== result[8] -  tau; 
}


template CheckFingerprintsBytecode(
    num_evals,
    log_num_read_write_hashes,
    log_num_init_final_hashes
    ){


    input BytecodePreprocessing( 
            num_evals) preprocessing;

    input signal  read_write_claim;
    input signal  init_final_claim;
    input signal  r_read_write_batch_index[log_num_read_write_hashes];
    input signal  r_init_final_batch_index[log_num_init_final_hashes];


    input   signal   openings[16];

    input signal  gamma;
    input signal  tau;


    signal temp_read_hashes[8];
    temp_read_hashes[0] <== openings[0]; 
    temp_read_hashes[1] <== openings[1]; 
    temp_read_hashes[2] <== openings[2]; 
    temp_read_hashes[3] <== openings[3]; 
    temp_read_hashes[4] <== openings[4];
    temp_read_hashes[5] <== openings[5];
    temp_read_hashes[6] <== openings[6];
    temp_read_hashes[7] <== openings[7];  
    signal read_hashes[1];
    read_hashes[0]  <== FingerprintBytecode()(
                    temp_read_hashes,
                    gamma,tau);
    
    signal one <==  1;
     signal temp_write_hashes[8];
    temp_write_hashes[0] <== openings[0];
    temp_write_hashes[1] <== openings[1];
    temp_write_hashes[2] <== openings[2]; 
    temp_write_hashes[3] <== openings[3]; 
    temp_write_hashes[4] <== openings[4];
    temp_write_hashes[5] <== openings[5];
    temp_write_hashes[6] <== openings[6];
    temp_write_hashes[7] <== (openings[7] +  one) ;  

    signal write_hashes[1];

    write_hashes[0] <== FingerprintBytecode()(
                    temp_write_hashes,
                    gamma,tau);

    
    signal temp_init_hashes[8];

    signal zero <== 0;
    temp_init_hashes[0] <== openings[9]; 
    temp_init_hashes[1] <== openings[10];  
    temp_init_hashes[2] <== openings[11]; 
    temp_init_hashes[3] <== openings[12]; 
    temp_init_hashes[4] <== openings[13];
    temp_init_hashes[5] <== openings[14];
    temp_init_hashes[6] <== openings[15];
    temp_init_hashes[7] <== zero ;  

     signal init_hashes[1];
    init_hashes[0] <== FingerprintBytecode()(
            temp_init_hashes,
            gamma,tau);


    signal temp_final_hashes[8];
    temp_final_hashes[0] <== openings[9]; 
    temp_final_hashes[1] <== openings[10];  
    temp_final_hashes[2] <== openings[11]; 
    temp_final_hashes[3] <== openings[12]; 
    temp_final_hashes[4] <== openings[13];
    temp_final_hashes[5] <== openings[14];
    temp_final_hashes[6] <== openings[15];
    temp_final_hashes[7] <== openings[8];  

     signal final_hashes[1];
    final_hashes[0] <== FingerprintBytecode()(
            temp_final_hashes,
            gamma,tau);

    
    //ineterleave
    signal read_write_hashes[2];
    signal init_final_hashes[2];
    
    read_write_hashes[0] <== read_hashes[0];
    read_write_hashes[1] <== write_hashes[0];

    init_final_hashes[0] <== init_hashes[0];
    init_final_hashes[1] <== final_hashes[0];

    var pow_2 = 1 << log_num_read_write_hashes; 
    signal  evals[pow_2] <== Evals(log_num_read_write_hashes)(r_read_write_batch_index);

    signal combined_read_write_hash;

    combined_read_write_hash <== ComputeDotProduct(pow_2)(evals, read_write_hashes );

    combined_read_write_hash === read_write_claim;

    var log_num_init_final_hashes_pow2 = 1 << log_num_init_final_hashes; 
    signal  r_init_evals[log_num_init_final_hashes_pow2] <== Evals(log_num_init_final_hashes)(r_init_final_batch_index);

    signal combined_init_final_hash;
    combined_init_final_hash <== ComputeDotProduct(log_num_init_final_hashes_pow2)(r_init_evals, init_final_hashes);
    combined_init_final_hash === init_final_claim;
}



