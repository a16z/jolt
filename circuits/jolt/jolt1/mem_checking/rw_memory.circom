pragma circom 2.2.1;
include "./../../../fields/non_native/utils.circom";
include "./../utils.circom";
include "./../../../transcript/transcript.circom";
include "./../jolt1_buses.circom";
include "mem_checking_buses.circom";
include "./../grand_product/grand_product.circom";
include "common.circom";

template VerifyMemoryCheckingReadWrite(bytecode_words_size,
        input_size,
        output_size,
        num_read_write_hashes,
        num_init_final_hashes,
        read_write_grand_product_layers,
        init_final_grand_product_layers,
        C, 
        NUM_MEMORIES, 
        NUM_INSTRUCTIONS, 
        MEMORY_OPS_PER_INSTRUCTION,
        chunks_x_size, 
        chunks_y_size, 
        NUM_CIRCUIT_FLAGS, 
        relevant_y_chunks_len,
        REGISTER_COUNT,
        memory_layout_input_start,
        min_bytecode_address,
        max_rounds,
        max_rounds_init_final
        )
{
        input ReadWriteMemoryPreprocessing(bytecode_words_size)
                                     preprocessing;

        input  ReadWriteMemoryCheckingProof(max_rounds, max_rounds_init_final, num_read_write_hashes, 
                  num_init_final_hashes,
                read_write_grand_product_layers,
                init_final_grand_product_layers) proof;


        input   JoltDevice(input_size, output_size) program_io;           
  
 
        input Transcript() transcript;

        output Transcript() up_transcript;


        Transcript() int_transcript[7];  
        signal gamma;
        (int_transcript[0], gamma) <== ChallengeScalar()(transcript);
    
 
        signal tau;    
        (int_transcript[1], tau) <== ChallengeScalar()(int_transcript[0]);




        CheckMultisetEqualityReadWriteMemory(
                num_read_write_hashes,
                 num_init_final_hashes) (proof.multiset_hashes);


         var  read_write_hashes_len = 2 * num_read_write_hashes;
         var  init_final_hashes_len = 2 * num_init_final_hashes;

        signal read_write_hashes[read_write_hashes_len];
        signal init_final_hashes[init_final_hashes_len];

        int_transcript[2] <== AppendToTranscript(num_read_write_hashes,
                                                     num_init_final_hashes, num_init_final_hashes) (int_transcript[1], proof.multiset_hashes);



        (read_write_hashes, init_final_hashes) <== InterleaveHashes(num_read_write_hashes,
                                                                    num_init_final_hashes, num_init_final_hashes) (proof.multiset_hashes);

                                                                  
        
        
     var log_num_read_write_hashes = log2(NextPowerOf2(read_write_hashes_len));

     var r_read_write_len = read_write_grand_product_layers + log_num_read_write_hashes ;

     signal read_write_claim, r_read_write[r_read_write_len];

     (int_transcript[3], read_write_claim, r_read_write) <== VerifyGrandProduct(max_rounds,read_write_grand_product_layers, read_write_hashes_len)(proof.read_write_grand_product, read_write_hashes, int_transcript[2]);


        
                                                
    signal r_read_write_batch_index[log_num_read_write_hashes];
    

    signal r_read_write_opening[read_write_grand_product_layers];

    
    (r_read_write_batch_index, r_read_write_opening) <== SplitAt(log_num_read_write_hashes, 
                                                                    r_read_write_len) (r_read_write);

    var log_num_init_final_hashes = log2(NextPowerOf2(init_final_hashes_len));
    var r_init_final_len = init_final_grand_product_layers + log_num_init_final_hashes;

    signal init_final_claim, r_init_final[r_init_final_len];
    (int_transcript[4], init_final_claim, r_init_final)  <== VerifyGrandProduct(max_rounds_init_final,init_final_grand_product_layers, init_final_hashes_len)(proof.init_final_grand_product, init_final_hashes, int_transcript[3]);



    signal r_init_final_batch_index[log_num_init_final_hashes];
    

    signal r_init_final_opening[init_final_grand_product_layers];
    (r_init_final_batch_index, r_init_final_opening) <== SplitAt(log_num_init_final_hashes, r_init_final_len)
                                                                (r_init_final);


        signal read_write_claims[14]; 
        for (var i = 0; i < 7; i++) {
            read_write_claims[i] <== proof.openings[i];
        }

        for (var i = 8; i < 12; i++) {
                read_write_claims[i-1] <== proof.openings[i];
        }

        read_write_claims[11] <==  proof.exogenous_openings[0];   
        read_write_claims[12] <==  proof.exogenous_openings[1];   
        read_write_claims[13] <==  proof.exogenous_openings[2];   


  
     output  VerifierOpening(read_write_grand_product_layers) read_write_verifier_opening;
   
    (int_transcript[5], read_write_verifier_opening) <== OpeningAccumulator(14, read_write_grand_product_layers) 
                                 (
                                                r_read_write_opening,
                                                read_write_claims,
                                                int_transcript[4]);
              



       signal init_final_claims[2];
       init_final_claims[0] <== proof.openings[7];
       init_final_claims[1] <== proof.openings[12];
         
       output  VerifierOpening(init_final_grand_product_layers) init_final_verifier_opening;

       (int_transcript[6], init_final_verifier_opening) <== OpeningAccumulator(2, init_final_grand_product_layers) 
       (
           r_init_final_opening,
           init_final_claims,
           int_transcript[5]);
   
        up_transcript <== int_transcript[6];

            
        signal verifier_openings[16] <==  ComputeVerifierOpeningsReadWriteMemory(read_write_grand_product_layers,init_final_grand_product_layers,bytecode_words_size,
                input_size,
                output_size,
                REGISTER_COUNT,
                memory_layout_input_start,
                min_bytecode_address)(
                r_read_write_opening,
                r_init_final_opening,
                proof.openings,
                preprocessing, program_io);


        CheckFingerprintsReadWriteMemory(
                        bytecode_words_size,
                        log_num_read_write_hashes,
                        log_num_init_final_hashes,
                        input_size,
                        output_size)(
                        preprocessing,
                        read_write_claim,
                        init_final_claim,
                        r_read_write_batch_index,
                        r_init_final_batch_index,
                        verifier_openings,
                        proof.exogenous_openings,
                        gamma,
                        tau
                        );
           
}


template CheckFingerprintsReadWriteMemory(
        bytecode_words_size,
        log_num_read_write_hashes,
        log_num_init_final_hashes,
        input_size,
        output_size
        ){
        input  ReadWriteMemoryPreprocessing(bytecode_words_size)
                                     preprocessing;
        input signal  read_write_claim;
        input signal  init_final_claim;
        input signal  r_read_write_batch_index[log_num_read_write_hashes];
        input signal  r_init_final_batch_index[log_num_init_final_hashes];


        input signal   openings[16];

        input signal   exogenous_openings[3];

        input signal  gamma;
        input signal  tau;
        
        // Computing  read_hashes
        signal read_hashes[4] <== ComputeReadHashes()(exogenous_openings, openings, gamma, tau);

        // Computing  write_hashes
        signal write_hashes[4] <== ComputeWriteHashes()(exogenous_openings, openings, gamma, tau);

        // Computing  init_hashes and final_hashes
        signal init_hashes[1] <== ComputeInitHashes()(exogenous_openings, openings, gamma, tau);

        signal final_hashes[1] <== ComputeFinalHashes()(exogenous_openings, openings, gamma, tau);

        
        //ineterleave
        signal read_write_hashes[2 * 4];
        signal init_final_hashes[2 * 1];
        for (var i = 0; i < 4; i++) {
                read_write_hashes[2*i] <== read_hashes[i];
                read_write_hashes[2*i + 1] <== write_hashes[i];
        }
        
        for (var i = 0; i < 1; i++) {
                init_final_hashes[2*i] <== init_hashes[i];
                init_final_hashes[2*i + 1] <== final_hashes[i];
        }


        var pow_2 = 1 << log_num_read_write_hashes; 
        signal  evals[pow_2] <== Evals(log_num_read_write_hashes)(r_read_write_batch_index);

        signal combined_read_write_hash;

        signal ZERO <== 0;
      
        combined_read_write_hash <== ComputeDotProduct(pow_2)(evals ,  read_write_hashes );
          

      combined_read_write_hash ===  read_write_claim;

       var log_num_init_final_hashes_pow2 = 1 << log_num_init_final_hashes; 
       signal  r_init_evals[log_num_init_final_hashes_pow2] <== Evals(log_num_init_final_hashes)(r_init_final_batch_index);
         
      
       signal combined_init_final_hash <== ComputeDotProduct(log_num_init_final_hashes_pow2)(r_init_evals, init_final_hashes );
       combined_init_final_hash === init_final_claim;
}


template FingerprintReadWriteMemory() {

    input  signal a, v, t; 
    input  signal gamma, tau;
    output  signal result; 

  
    signal gamma_square;
    gamma_square <==  (gamma  *  gamma);  
    
    signal gamma_square_t;
    gamma_square_t <==  (gamma_square *  t);  

    signal gamma_v;
    gamma_v <==  (gamma *  v); 

    signal temp_res;
    temp_res  <==  (gamma_square_t + gamma_v); 

    signal temp_res_a;
    temp_res_a  <==  (temp_res + a); 

    result <== (temp_res_a - tau); 

}



template  ComputeVerifierOpeningsReadWriteMemory(read_write_grand_product_layers,
        init_final_grand_product_layers,
        bytecode_words_size,
        input_size,
        output_size,
        REGISTER_COUNT,
        memory_layout_input_start,
        min_bytecode_address){

        input  signal r_read_write[read_write_grand_product_layers] ;
        input  signal r_init_final[init_final_grand_product_layers] ;
        input  signal openings[13];
        input  ReadWriteMemoryPreprocessing(bytecode_words_size)
                                         preprocessing;

        input   JoltDevice(input_size, output_size) program_io;                           

        output  signal int_openings[16] ;
        for (var i = 0; i < 13; i++) {
                int_openings[i] <== openings[i];
        }
          
        int_openings[15]  <== EvaluateIdentityPoly(read_write_grand_product_layers)(r_read_write);
        int_openings[13]  <== EvaluateIdentityPoly(init_final_grand_product_layers)(r_init_final);
 
        var memory_size  = 1 << init_final_grand_product_layers ;

        var v_init[memory_size];
        var v_init_index =  MemoryAddressToWitnessIndex(REGISTER_COUNT, min_bytecode_address,  memory_layout_input_start);
        var starting_v_init_index = v_init_index;
        for (var i = 0; i < bytecode_words_size; i++) {
            v_init[v_init_index + i ] = preprocessing.bytecode_words[i];
        }

        var v_init_index_1 = MemoryAddressToWitnessIndex(REGISTER_COUNT, memory_layout_input_start, memory_layout_input_start );    
        var starting_input_idx = v_init_index_1;
        for (var i = 0; i < input_size; i += 4) {
                var word = 0;
                var input_idx = 1;
                for (var j = 0; j < 4; j++) {              
                        if ( i + j < input_size) {
                             word = word + (program_io.inputs[i + j] * (input_idx));
                             input_idx *= 256;
                        } 
                }
                v_init[v_init_index_1] = word;
                v_init_index_1 += 1;
        }

        signal temp_v_init[memory_size];
        for (var i = 0; i < memory_size; i++) {
                temp_v_init[i] <==  v_init[i];
        }
        signal one <== 1;
        // starting index for bytecode  = 4096
        // (1 - r_0) (1 - r_1) (1 - r_2) (1 - r_3) (1 - r_4) (1 - r_5) (1 - r_6) (1 - r_7) (1 - r_8) (1 - r_9) (1 - r_10) (1 - r_11) r_12
        var log_bytecode_words_size = log2(NextPowerOf2(bytecode_words_size));

        signal r_init_final_bytecode[log_bytecode_words_size];

        for (var i = 0; i < log_bytecode_words_size; i++) {
                r_init_final_bytecode[i] <==  r_init_final[init_final_grand_product_layers - log_bytecode_words_size + i];   

        }
        var r_eq_bytecode_len = 1 << log_bytecode_words_size;
        signal  r_eq_bytecode[r_eq_bytecode_len ]  <==  Evals(log_bytecode_words_size)(r_init_final_bytecode);
        signal temp_v_init_bytecode[r_eq_bytecode_len];  
        for (var i = 0; i < r_eq_bytecode_len; i++) {
                temp_v_init_bytecode[i] <==  temp_v_init[i + starting_v_init_index]; 
        }

        signal temp_opening[init_final_grand_product_layers-log_bytecode_words_size + 1  ];
        temp_opening[0] <== ComputeDotProduct(r_eq_bytecode_len)(temp_v_init_bytecode, r_eq_bytecode);
        temp_opening[1] <== r_init_final[0] *  temp_opening[0];
        for (var i = 1; i < init_final_grand_product_layers-log_bytecode_words_size ; i++) {
                temp_opening[i + 1] <== (one  -   r_init_final[i]) *   temp_opening[i];    
        }
      
        // Computation for the values of input_index
        // This will only work when input_size <= 32
        // Input index should start from 64
        // (1 - r_0) (1 - r_1) (1 - r_2) (1 - r_3) (1 - r_4) (1 - r_5) r_6 (1 - r_7) (1 - r_8) (1 - r_9) (1 - r_10) (1 - r_11)(1-r_12)
        var non_zero_values_input_size =   v_init_index_1 - starting_input_idx;
        if  (non_zero_values_input_size == 1){
                non_zero_values_input_size = 2;
        }
        
        var  log_input_size = log2(NextPowerOf2(non_zero_values_input_size)); 
        signal r_init_final_input_size[log_input_size];
        for (var i = 0; i < log_input_size; i++) {
               r_init_final_input_size[i] <==  r_init_final[init_final_grand_product_layers - log_input_size + i ];    
        }
        var r_eq_input_size_len = 1 << log_input_size;
        signal  r_eq_input_size[r_eq_input_size_len]   <==  Evals(log_input_size)(r_init_final_input_size);

        signal temp_v_init_input_size[r_eq_input_size_len];
        for (var i = 0; i < r_eq_input_size_len; i++) {
             temp_v_init_input_size[i] <==  temp_v_init[i + starting_input_idx]; 
         
        }
      
        signal  eval_opening_input_size <== ComputeDotProduct(r_eq_input_size_len)(temp_v_init_input_size, r_eq_input_size);
        var  eval_input_size_len =  6;
        signal eval_input_size[eval_input_size_len + 1 + init_final_grand_product_layers - log_input_size -7];
        eval_input_size[0] <== (eval_opening_input_size * r_init_final[6]);

        for (var i = 0; i < eval_input_size_len; i++) {
            eval_input_size[i + 1] <== ( eval_input_size[i] *  (one -  r_init_final[i]));
        }
      

        for (var i = 0; i < init_final_grand_product_layers -  log_input_size - 7 ; i++) {    
          eval_input_size[i + eval_input_size_len + 1] <== ( eval_input_size[i + eval_input_size_len]  * (one - r_init_final[i+7] ) );
         }

        signal input_size_final_eval <== eval_input_size[eval_input_size_len + init_final_grand_product_layers - log_input_size -7];

        int_openings[14] <== ( input_size_final_eval +  temp_opening[init_final_grand_product_layers-log_bytecode_words_size ]);
     
}












template CheckMultisetEqualityReadWriteMemory(num_read_write_hashes,
         num_init_final_hashes) {

     input MultisetHashes(num_read_write_hashes,  num_init_final_hashes, num_init_final_hashes) multiset_hashes;

     //TODO(Bhargav) Asserts? Can be avoided by setting num_init_final_hashes = num_read_write_hashes.

     var NUM_MEMORIES = num_read_write_hashes;

     signal read_hash[NUM_MEMORIES+1];  // Temporary signal to accumulate the read_hash
     read_hash[0]  <== 1;
   

     signal write_hash[NUM_MEMORIES+1];  // Temporary signal to accumulate the product
     write_hash[0]  <== 1;
   

	//TODO: This is wrong. Need to check LHS = RHS in each iteration.
     for (var i = 0; i < NUM_MEMORIES; i++) {
       read_hash[i+1] <==   (read_hash[i]  *  multiset_hashes.read_hashes[i]); // Multiply each element
       write_hash[i+1] <==   (write_hash[i] *  multiset_hashes.write_hashes[i]); // Multiply each element
     }

     signal init_hash_write_hash <== (multiset_hashes.init_hashes[0] *  write_hash[NUM_MEMORIES]);

     signal final_hash_read_hash <==(multiset_hashes.final_hashes[0] *  read_hash[NUM_MEMORIES]);



     init_hash_write_hash === final_hash_read_hash;

}




template ComputeReadHashes() {
    input  signal exogenous_openings[3];
    input  signal openings[16];
    input  signal gamma;
    input  signal tau;
    output signal  read_hashes[4];

    signal temp_read_hashes[4][3];
    
    temp_read_hashes[0][0] <== exogenous_openings[1]; 
    temp_read_hashes[0][1] <== openings[2]; 
    temp_read_hashes[0][2] <== openings[9]; 
    temp_read_hashes[1][0] <== exogenous_openings[2]; 
    temp_read_hashes[1][1] <== openings[3]; 
    temp_read_hashes[1][2] <== openings[10]; 
    temp_read_hashes[2][0] <== exogenous_openings[0]; 
    temp_read_hashes[2][1] <== openings[1]; 
    temp_read_hashes[2][2] <== openings[8]; 
    temp_read_hashes[3][0] <== openings[0]; 
    temp_read_hashes[3][1] <== openings[4]; 
    temp_read_hashes[3][2] <== openings[11]; 

    for (var i = 0; i < 4; i++) {
        read_hashes[i] <== FingerprintReadWriteMemory()(
            temp_read_hashes[i][0],
            temp_read_hashes[i][1],
            temp_read_hashes[i][2],
            gamma, tau
        );
    }
}






template ComputeWriteHashes() {
    input  signal exogenous_openings[3];
    input  signal openings[16];
    input  signal gamma;
    input  signal tau;
    output signal  write_hashes[4];

     signal temp_write_hashes[4][3];
        temp_write_hashes[0][0] <== exogenous_openings[1]; 
        temp_write_hashes[0][1] <== openings[2]; 
        temp_write_hashes[0][2] <== openings[15];
        temp_write_hashes[1][0] <== exogenous_openings[2];
        temp_write_hashes[1][1] <== openings[3]; 
        temp_write_hashes[1][2] <== openings[15]; 
        temp_write_hashes[2][0] <== exogenous_openings[0]; 
        temp_write_hashes[2][1] <== openings[5]; 
        temp_write_hashes[2][2] <== openings[15];
        temp_write_hashes[3][0] <== openings[0]; 
        temp_write_hashes[3][1] <== openings[6]; 
        temp_write_hashes[3][2] <== openings[15]; 
        for (var i = 0; i < 4; i++) {
                write_hashes[i] <== FingerprintReadWriteMemory()(
                        temp_write_hashes[i][0],
                        temp_write_hashes[i][1],
                        temp_write_hashes[i][2],
                        gamma,tau);
        }

       
}




template ComputeInitHashes() {
    input  signal exogenous_openings[3];
    input  signal openings[16];
    input  signal gamma;
    input  signal tau;

    output signal  init_hashes[1];

        signal temp_init_hashes[3];
      
        temp_init_hashes[0] <== openings[13]; 
        temp_init_hashes[1] <== openings[14]; 
        temp_init_hashes[2] <== 0;

        init_hashes[0] <== FingerprintReadWriteMemory()(
                temp_init_hashes[0],
                temp_init_hashes[1],
                temp_init_hashes[2],
                gamma,tau);

       
}



template ComputeFinalHashes() {
    input  signal exogenous_openings[3];
    input  signal openings[16];
    input  signal gamma;
    input  signal tau;
    output signal  final_hashes[1];

        signal temp_final_hashes[3];
        temp_final_hashes[0] <== openings[13]; 
        temp_final_hashes[1] <== openings[7]; 
        temp_final_hashes[2] <== openings[12]; 

        final_hashes[0] <== FingerprintReadWriteMemory()(
                temp_final_hashes[0],
                temp_final_hashes[1],
                temp_final_hashes[2],
                gamma,tau);
}



