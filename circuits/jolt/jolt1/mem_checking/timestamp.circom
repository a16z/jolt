pragma circom 2.2.1;
include "./../../../fields/non_native/utils.circom";
include "./../utils.circom";
include "./../../../transcript/transcript.circom";
include "./../jolt1_buses.circom";
include "mem_checking_buses.circom";
include "./../grand_product/grand_product.circom";
include "common.circom";
template TimestampValidityVerifier(max_rounds,ts_validity_grand_product_layers,
    num_read_write_hashes,
    num_init_hashes,
    C, 
    num_memories, 
    NUM_INSTRUCTIONS, 
    MEMORY_OPS_PER_INSTRUCTION,
    chunks_x_size, 
    chunks_y_size, 
    NUM_CIRCUIT_FLAGS, 
    relevant_y_chunks_len
) {

    input TimestampValidityProof(max_rounds,ts_validity_grand_product_layers,num_read_write_hashes,
         num_init_hashes,  MEMORY_OPS_PER_INSTRUCTION)   proof;


    input Transcript() transcript;
    output Transcript() up_transcript;

     Transcript() int_transcript[5];  

    signal gamma, tau;
    (int_transcript[0], gamma) <== ChallengeScalar()(transcript);
  
    (int_transcript[1], tau) <== ChallengeScalar()(int_transcript[0]);

   
    
    CheckMultisetEqualityTimestamp(num_read_write_hashes,
        num_init_hashes)( proof.multiset_hashes);



    int_transcript[2] <== AppendToTranscript(num_read_write_hashes,
            num_init_hashes, num_read_write_hashes)(int_transcript[1], proof.multiset_hashes);


    var  read_write_hashes_len = 2 * num_read_write_hashes;
    var  init_final_hashes_len = num_init_hashes + num_read_write_hashes;

    signal read_write_hashes[read_write_hashes_len];
    signal init_final_hashes[init_final_hashes_len];
        
    (read_write_hashes, init_final_hashes) <== InterleaveHashesTimestamp(num_read_write_hashes, num_init_hashes) (proof.multiset_hashes.read_hashes,
        proof.multiset_hashes.write_hashes,proof.multiset_hashes.init_hashes,proof.multiset_hashes.final_hashes);

    var  batch_size = read_write_hashes_len +  init_final_hashes_len;


     signal concatenated_hashes [  batch_size ];

    for (var i = 0; i < read_write_hashes_len; i++) {
        concatenated_hashes[i] <== read_write_hashes[i];
    }

    for (var i = 0; i < init_final_hashes_len; i++) {
        concatenated_hashes[i + read_write_hashes_len ] <== init_final_hashes[i];
    }
   
    var log_num_batch_size = log2(NextPowerOf2(batch_size));

     var r_grand_product_len = ts_validity_grand_product_layers + log_num_batch_size;

    signal grand_product_claim, r_grand_product[r_grand_product_len];

     (int_transcript[3], grand_product_claim, r_grand_product) <== VerifyGrandProduct(max_rounds,ts_validity_grand_product_layers, batch_size)(proof.batched_grand_product, concatenated_hashes, int_transcript[2]);


    signal r_batch_index[log_num_batch_size];

    signal r_opening[ts_validity_grand_product_layers];

    (r_batch_index, r_opening) <== SplitAt(log_num_batch_size, 
        r_grand_product_len) (r_grand_product);


 


     signal ts_claims[4* MEMORY_OPS_PER_INSTRUCTION + 4];
   
    for (var i = 0; i < MEMORY_OPS_PER_INSTRUCTION; i++) {
        ts_claims[i] <== proof.openings.read_cts_read_timestamp[i];
    }
    for (var i = 0; i < MEMORY_OPS_PER_INSTRUCTION; i++) {
        ts_claims[i + MEMORY_OPS_PER_INSTRUCTION] <== proof.openings.read_cts_global_minus_read[i];
    }
    for (var i = 0; i < MEMORY_OPS_PER_INSTRUCTION; i++) {
        ts_claims[i + 2 * MEMORY_OPS_PER_INSTRUCTION] <== proof.openings.final_cts_read_timestamp[i];
    }
    for (var i = 0; i < MEMORY_OPS_PER_INSTRUCTION; i++) {
        ts_claims[i + 3 * MEMORY_OPS_PER_INSTRUCTION] <== proof.openings.final_cts_global_minus_read[i];
    }

    for (var i = 0; i < 4; i++) {
        ts_claims[i + 4 * MEMORY_OPS_PER_INSTRUCTION] <== proof.exogenous_openings[i];
    }


    output VerifierOpening(ts_validity_grand_product_layers) ts_verifier_opening;

    (int_transcript[4], ts_verifier_opening) <== OpeningAccumulator(4* MEMORY_OPS_PER_INSTRUCTION + 4, ts_validity_grand_product_layers)( r_opening, ts_claims, int_transcript[3]);


    up_transcript <== int_transcript[4];
 
    signal int_openings[4*MEMORY_OPS_PER_INSTRUCTION +1] ;
    for (var i = 0; i < MEMORY_OPS_PER_INSTRUCTION; i++) {
        int_openings[i] <== proof.openings.read_cts_read_timestamp[i];
    }
    for (var i = 0; i < MEMORY_OPS_PER_INSTRUCTION; i++) {
        int_openings[i + MEMORY_OPS_PER_INSTRUCTION] <== proof.openings.read_cts_global_minus_read[i];
    }
    for (var i = 0; i < MEMORY_OPS_PER_INSTRUCTION; i++) {
        int_openings[i + 2 * MEMORY_OPS_PER_INSTRUCTION] <== proof.openings.final_cts_read_timestamp[i];
    }
    for (var i = 0; i < MEMORY_OPS_PER_INSTRUCTION; i++) {
        int_openings[i + 3 * MEMORY_OPS_PER_INSTRUCTION] <== proof.openings.final_cts_global_minus_read[i];
    }


   int_openings[4 * MEMORY_OPS_PER_INSTRUCTION]  <== EvaluateIdentityPoly(ts_validity_grand_product_layers)(r_opening);
    signal read_hashes[2 * MEMORY_OPS_PER_INSTRUCTION] <==  Readhashes(MEMORY_OPS_PER_INSTRUCTION)(
     proof.exogenous_openings, int_openings, gamma, tau);



    signal write_hashes[2 * MEMORY_OPS_PER_INSTRUCTION] <==  Writehashes(MEMORY_OPS_PER_INSTRUCTION)(
             proof.exogenous_openings, int_openings, gamma, tau);
       

    signal init_hashes[1] <==  Inithashes(MEMORY_OPS_PER_INSTRUCTION)(
                int_openings, gamma, tau);


    signal final_hashes[2 * MEMORY_OPS_PER_INSTRUCTION] <==  Finalhashes(MEMORY_OPS_PER_INSTRUCTION)(
                proof.exogenous_openings, int_openings, gamma, tau);



    signal zero <== 0;
    signal read_write_hash[4 * MEMORY_OPS_PER_INSTRUCTION];
    signal init_final_hash[2 * MEMORY_OPS_PER_INSTRUCTION + 1];
        
    (read_write_hash, init_final_hash) <== InterleaveHashesTimestamp(2 * MEMORY_OPS_PER_INSTRUCTION, 1) (read_hashes,
        write_hashes,init_hashes,final_hashes);

    
        var pow_2 = 1 << log_num_batch_size; 
        signal  evals[pow_2] <== Evals(log_num_batch_size)(r_batch_index);


       var hashes_len = 6*MEMORY_OPS_PER_INSTRUCTION +1;
       signal combined_hash ; 
  
        signal read_write_hashes_init_final_hashes[hashes_len];
  
        for (var i = 0; i < 4 * MEMORY_OPS_PER_INSTRUCTION; i++) {
         read_write_hashes_init_final_hashes[i] <== read_write_hash[i];
        }

        for (var i = 0; i < (2 * MEMORY_OPS_PER_INSTRUCTION + 1); i++) {
            read_write_hashes_init_final_hashes[4 * MEMORY_OPS_PER_INSTRUCTION + i] <== init_final_hash[i];
        }

      signal evals_temp[hashes_len] <== TruncateVec(0, hashes_len, pow_2)(evals);
     combined_hash <== ComputeDotProduct(hashes_len)(evals_temp, read_write_hashes_init_final_hashes );

     combined_hash ===  grand_product_claim;
 
}




template FingerprintTimestamp() {
    input  signal a , t; 
    input  signal gamma, tau ;   
    output  signal out; 

  
    signal gamma_a;
    gamma_a <== (gamma *  a); 

    signal temp_res;
    temp_res  <==  (gamma_a + t); 

    out <== (temp_res -  tau); 


}



template CheckMultisetEqualityTimestamp(
    num_read_write_hashes,
    num_init_hashes) {


input MultisetHashes(num_read_write_hashes, num_init_hashes, num_read_write_hashes) multiset_hashes;


         var num_memories = num_read_write_hashes;
         
         signal ZERO <== 0;


         signal lhs[num_memories];
         signal rhs[num_memories];
         signal res[num_memories];

         
         for (var i = 0; i < num_memories; i++) {
            if ( num_memories == num_read_write_hashes) {

                lhs[i] <==  (multiset_hashes.read_hashes[i] * multiset_hashes.final_hashes[i]);
                rhs[i] <==  (multiset_hashes.init_hashes[0] *  multiset_hashes.write_hashes[i]);
               
               lhs[i] === rhs[i];
            }
         
         }
}



template InterleaveHashesTimestamp(num_read_write_hashes,
    num_init_hashes) {

   input  signal read_hashes[num_read_write_hashes];
   input  signal write_hashes[num_read_write_hashes];
   input  signal init_hashes[num_init_hashes];
   input  signal final_hashes[num_read_write_hashes];

   output signal read_write_hashes[2 * num_read_write_hashes];
   output signal init_final_hashes[num_init_hashes + num_read_write_hashes];
   
   for (var i = 0; i < num_read_write_hashes; i++) {
   read_write_hashes[2*i] <== read_hashes[i];
   read_write_hashes[2*i + 1] <== write_hashes[i];
   }
   
    for (var i = 0; i < num_read_write_hashes; i++) {
       init_final_hashes[i ] <== final_hashes[i];
   }

   for (var i = 0; i < num_init_hashes; i++) {
       init_final_hashes[i + num_read_write_hashes ] <== init_hashes[i];
   }

   

}



template Readhashes(MEMORY_OPS_PER_INSTRUCTION) {


   input  signal exogenous_openings[MEMORY_OPS_PER_INSTRUCTION];
   input  signal openings[4*MEMORY_OPS_PER_INSTRUCTION +1];
   input  signal gamma, tau ; 

    output signal read_hashes[2 * MEMORY_OPS_PER_INSTRUCTION];

  signal read_tuple[MEMORY_OPS_PER_INSTRUCTION][4] ;
    for (var i = 0; i < MEMORY_OPS_PER_INSTRUCTION; i++) {

        read_tuple[i][0] <== exogenous_openings[i];
        read_tuple[i][1] <== openings[i];

        read_tuple[i][2] <== (openings[4 * MEMORY_OPS_PER_INSTRUCTION] - exogenous_openings[i]);
        read_tuple[i][3] <== openings[i + MEMORY_OPS_PER_INSTRUCTION];
    }

    for (var i = 0; i < MEMORY_OPS_PER_INSTRUCTION; i++) {

        read_hashes[2 * i]  <== FingerprintTimestamp()(read_tuple[i][0], read_tuple[i][1], gamma,tau );
        read_hashes[2 * i +1]  <== FingerprintTimestamp()(read_tuple[i][2], read_tuple[i][3], gamma,tau );

    }


}






template Writehashes(MEMORY_OPS_PER_INSTRUCTION) {


   input  signal exogenous_openings[MEMORY_OPS_PER_INSTRUCTION];
   input  signal openings[4*MEMORY_OPS_PER_INSTRUCTION +1];
   input  signal gamma, tau ; 

    output signal write_hashes[2 * MEMORY_OPS_PER_INSTRUCTION];
 
     signal write_tuple[MEMORY_OPS_PER_INSTRUCTION][4] ;

    for (var i = 0; i < MEMORY_OPS_PER_INSTRUCTION; i++) {

        write_tuple[i][0] <== exogenous_openings[i];
        write_tuple[i][1] <== (openings[i] +   1);

        write_tuple[i][2] <== (openings[4 * MEMORY_OPS_PER_INSTRUCTION] - exogenous_openings[i]);
        write_tuple[i][3] <==  (openings[i + MEMORY_OPS_PER_INSTRUCTION] +  1);
    }


    for (var i = 0; i < MEMORY_OPS_PER_INSTRUCTION; i++) {

        write_hashes[2 * i]  <== FingerprintTimestamp()(write_tuple[i][0], write_tuple[i][1], gamma,tau );
        write_hashes[2 * i +1]  <== FingerprintTimestamp()(write_tuple[i][2], write_tuple[i][3], gamma,tau );

    }

}

template Inithashes(MEMORY_OPS_PER_INSTRUCTION) {

   input  signal openings[4*MEMORY_OPS_PER_INSTRUCTION +1];
   input  signal gamma, tau ; 

    output signal init_hashes[1] ;
 
    signal init_tuple[2] ;
    init_tuple[0] <==  openings[4*MEMORY_OPS_PER_INSTRUCTION];
    init_tuple[1] <== 0;


    init_hashes[0]  <== FingerprintTimestamp()(init_tuple[0], init_tuple[1], gamma,tau );


}



template Finalhashes(MEMORY_OPS_PER_INSTRUCTION) {


     input  signal exogenous_openings[MEMORY_OPS_PER_INSTRUCTION];
     input  signal openings[4*MEMORY_OPS_PER_INSTRUCTION +1];
     input  signal gamma, tau ; 
     output signal final_hashes[2 * MEMORY_OPS_PER_INSTRUCTION];
  
     signal final_tuple[MEMORY_OPS_PER_INSTRUCTION][4] ;
     for (var i = 0; i < MEMORY_OPS_PER_INSTRUCTION; i++) {

        final_tuple[i][0] <== openings[4 * MEMORY_OPS_PER_INSTRUCTION];
        final_tuple[i][1] <== openings[i + 2 * MEMORY_OPS_PER_INSTRUCTION];

        final_tuple[i][2] <== openings[4 * MEMORY_OPS_PER_INSTRUCTION];
        final_tuple[i][3] <==  openings[i + 3 * MEMORY_OPS_PER_INSTRUCTION];
     }

    for (var i = 0; i < MEMORY_OPS_PER_INSTRUCTION; i++) {
        final_hashes[2 * i]  <== FingerprintTimestamp()(final_tuple[i][0], final_tuple[i][1], gamma,tau );
        final_hashes[2 * i +1]  <== FingerprintTimestamp()(final_tuple[i][2], final_tuple[i][3], gamma,tau );
    }
}


