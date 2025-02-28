
include "./../../../fields/non_native/utils.circom";
include "./../utils.circom";
include "./../../../transcript/transcript.circom";
include "./../jolt1_buses.circom";
include "mem_checking_buses.circom";
include "./../grand_product/grand_product.circom";
include "common.circom";

template VerifyMemoryCheckingOutputSumCheck(
    input_size,
    output_size,
    rounds,
    REGISTER_COUNT,
    RAM_START_ADDRESS,
    memory_layout_input_start,
    memory_layout_output_start,
    memory_layout_panic,
    memory_layout_termination,
    program_io_panic
){

    input OutputSumcheckProof(3, rounds) proof;
    
    input   JoltDevice(input_size, output_size) program_io;  

    // input ReadWriteMemoryStuff() commitments;

    input Transcript() transcript;

    output Transcript() up_transcript;

    Transcript() int_transcript[3]; 

    signal  r_eq[rounds];
    (int_transcript[0], r_eq) <== ChallengeVector(rounds)(transcript);

   signal zero <== 0;
  
   signal sumcheck_claim;
   signal r_sumcheck[rounds];
  (int_transcript[1], sumcheck_claim,  r_sumcheck) <== SumCheck(rounds, 3)(zero, proof.sumcheck_proof, int_transcript[0]);

   signal  eq_eval <== EvaluateEq(rounds)(r_eq, r_sumcheck);

    var input_start_index = MemoryAddressToWitnessIndex(REGISTER_COUNT, memory_layout_input_start , memory_layout_input_start); 
                   
    var io_memory_size = MemoryAddressToWitnessIndex(REGISTER_COUNT,RAM_START_ADDRESS, memory_layout_input_start); 

    var log_io_memory_size = log2(io_memory_size); 

   signal one <== 1;


    var start =  rounds -  log_io_memory_size; 
    var temp_r_sumcheck_len = rounds - start; 
    signal temp_r_sumcheck[temp_r_sumcheck_len];

    for (var i = start ; i < rounds; i++) {
        temp_r_sumcheck[i - start] <== r_sumcheck[i];

    } 


    var log_input_size = log2(NextPowerOf2(input_start_index)); 
    signal r_sumcheck_io_witness_range[log_input_size];
    for (var i = 0 ; i < log_input_size; i++) {
        r_sumcheck_io_witness_range[i] <== temp_r_sumcheck[temp_r_sumcheck_len - log_input_size +i];


    }

    signal r_eq_io_witness_range[input_start_index] <==  Evals(log_input_size)(r_sumcheck_io_witness_range);
    signal  temp_r_eq_io_witness_range[input_start_index + 1] ;
    temp_r_eq_io_witness_range[0] <== zero;
    for (var i = 0 ; i < input_start_index; i++) {
        temp_r_eq_io_witness_range[i+1] <== r_eq_io_witness_range[i] + temp_r_eq_io_witness_range[i];
    }

    var fc_io_witness_range_len = temp_r_sumcheck_len - log_input_size;
    signal  fc_io_witness_range[fc_io_witness_range_len + 1] ;
    fc_io_witness_range[0] <== temp_r_eq_io_witness_range[input_start_index];

    for (var i = 0 ; i < fc_io_witness_range_len; i++) {
      fc_io_witness_range[i + 1] <==  fc_io_witness_range[i]  *  (one  -  temp_r_sumcheck[i]);
    }

    signal  io_witness_range_eval[2] ;
    io_witness_range_eval[0]  <==   (one  - fc_io_witness_range[fc_io_witness_range_len]);

    signal r_prod[start+1];
    r_prod[0] <== one;
    for (var i = 0; i < start; i++) {
        r_prod[i+1] <== r_prod[i] * ( one -  r_sumcheck[i]);
    }

    io_witness_range_eval[1]  <== (r_prod[start] * io_witness_range_eval[0]);
    var v_io[io_memory_size];
    var input_index =  MemoryAddressToWitnessIndex(REGISTER_COUNT, memory_layout_input_start,  memory_layout_input_start); 
    var starting_input_idx = input_index; 
    for (var i = 0; i < input_size; i += 4) {
        var word = 0;
        var input_idx = 1;
            for (var j = 0; j < 4; j++) {
                if ( i + j < input_size) {
                             word = word + (program_io.inputs[i + j] * ( input_idx));     
                              input_idx *= 256;        
                } 
            }
            v_io[input_index]  = word;
            input_index += 1;
    }
 

    var output_index =  MemoryAddressToWitnessIndex(REGISTER_COUNT, memory_layout_output_start,  memory_layout_input_start); 
    var starting_output_idx = output_index; 

    for (var i = 0; i < output_size; i += 4) {
            var output_word = 0;
            var input_idx = 1;
            for (var j = 0; j < 4; j++) {
                if ( i + j < output_size) {
                output_word  = output_word + (program_io.outputs[i + j] * ( input_idx)); 
                input_idx *= 256;
                } 
            }
            v_io[output_index ]  = output_word;
            output_index +=1;
    }
 

    var panic_index =  MemoryAddressToWitnessIndex(REGISTER_COUNT, memory_layout_panic,  memory_layout_input_start); 
    v_io[panic_index] = program_io.panic;


    var termination_index =  MemoryAddressToWitnessIndex(REGISTER_COUNT, memory_layout_termination,  memory_layout_input_start);
    v_io[termination_index]   =  1 -  program_io_panic ;


    // Computation for the values of input_index
    // This will only work when input_size <= 32
    // Input index should start from 64
    var non_zero_v_io_input_size =   input_index - starting_input_idx; 
        if  ( non_zero_v_io_input_size == 1){
                non_zero_v_io_input_size = 2;
        }

    var log_non_zero_v_io_input_size = log2(NextPowerOf2(non_zero_v_io_input_size)); 

  

    signal r_sumcheck_io_input_size[log_non_zero_v_io_input_size];  
   
    for (var i = 0 ; i < log_non_zero_v_io_input_size; i++) {
        r_sumcheck_io_input_size[i] <== temp_r_sumcheck[temp_r_sumcheck_len - log_non_zero_v_io_input_size + i];
    }

    signal fc_input_size[1 << log_non_zero_v_io_input_size] <== Evals(log_non_zero_v_io_input_size)(r_sumcheck_io_input_size); 
    signal temp_vio_input_size[1 << log_non_zero_v_io_input_size]; 
    for (var i = 0 ; i < (1 << log_non_zero_v_io_input_size); i++) {
       temp_vio_input_size[i] <== v_io[i + starting_input_idx];
    }
 
    signal eval_vio_input <== ComputeDotProduct(1 << log_non_zero_v_io_input_size)(temp_vio_input_size , fc_input_size);


    var  eval_vio_input_size_len =  5 ;
    signal eval_vio_input_size[eval_vio_input_size_len + 1 + temp_r_sumcheck_len - log_non_zero_v_io_input_size -6];
    eval_vio_input_size[0] <== (eval_vio_input *  temp_r_sumcheck[5]);
    for (var i = 0; i < eval_vio_input_size_len; i++) {
        eval_vio_input_size[i + 1] <== ( eval_vio_input_size[i]  * (one - temp_r_sumcheck[i]) );
    }
    for (var i = 0; i < temp_r_sumcheck_len - log_non_zero_v_io_input_size -6 ; i++) {      
      eval_vio_input_size[i + eval_vio_input_size_len+ 1] <== ( eval_vio_input_size[i + eval_vio_input_size_len] *  (one - temp_r_sumcheck[i+6]) );
    }
    signal input_size_final_eval <== eval_vio_input_size[eval_vio_input_size_len + temp_r_sumcheck_len - log_non_zero_v_io_input_size -6];



    // output index
    // This will only work when output_size <= 32
    // output index should start from 1088
    var non_zero_v_io_output_size =   output_index - starting_output_idx;  
    var log_non_zero_v_io_output_size = 1;
    if  (non_zero_v_io_output_size > 1) {
       log_non_zero_v_io_output_size = log2(NextPowerOf2(non_zero_v_io_output_size)); 
    }
    signal r_sumcheck_io_output_size[log_non_zero_v_io_output_size]; 
    for (var i = 0 ; i < log_non_zero_v_io_output_size; i++) {
        r_sumcheck_io_output_size[i] <-- temp_r_sumcheck[ temp_r_sumcheck_len- log_non_zero_v_io_output_size +i];
      
    }

    signal fc_output_size[1 << log_non_zero_v_io_output_size] <== Evals(log_non_zero_v_io_output_size)(r_sumcheck_io_output_size); 
    signal temp_vio_output_size[1 << log_non_zero_v_io_output_size]; 
  
    for (var i = 0 ; i < (1 << log_non_zero_v_io_output_size); i++) {
       temp_vio_output_size[i]  <== v_io[i + starting_output_idx];
    }

    signal eval_vio_output_size <== ComputeDotProduct(1 << log_non_zero_v_io_output_size)(temp_vio_output_size , fc_output_size);

    signal eval_vio_output_size_vec[ temp_r_sumcheck_len -  log_non_zero_v_io_output_size -2];
    signal rc1_rc5 <== (temp_r_sumcheck[1] *  temp_r_sumcheck[5])  ;

    eval_vio_output_size_vec[0] <==    (eval_vio_output_size * rc1_rc5);
    for (var i = 0 ; i <(3); i++) {
        eval_vio_output_size_vec[i + 1] <== (eval_vio_output_size_vec[i] *  (one  - temp_r_sumcheck[i + 2]));
    }

    for (var i = 0 ; i < (temp_r_sumcheck_len -  log_non_zero_v_io_output_size - 6); i++) {
        eval_vio_output_size_vec[i + 4] <== (eval_vio_output_size_vec[i+ 3] * (one  - temp_r_sumcheck[i + 6]));
    }


   signal output_size_final_eval <== (eval_vio_output_size_vec[temp_r_sumcheck_len -  log_non_zero_v_io_output_size - 3] * (one  -  temp_r_sumcheck[0]));
 

    // panic and termination evals
    // Panic index = 2112  and termination index = 2113
   signal r_sumcheck_io_panic_termination[1];  
    for (var i = 0 ; i < 1; i++) {
        r_sumcheck_io_panic_termination[i] <-- temp_r_sumcheck[temp_r_sumcheck_len - 1];
    }

   signal fc_panic_termination[2] <== Evals(1)(r_sumcheck_io_panic_termination); 

   signal temp_vio_panic_termination[2]; 
    for (var i = 0 ; i < 2 ; i++) {
       temp_vio_panic_termination[i] <== v_io[i + panic_index];
 
    }

    signal eval_panic_termination <== ComputeDotProduct(2)(temp_vio_panic_termination , fc_panic_termination);

   signal eval_vio_panic_termination[ temp_r_sumcheck_len - 2];
   signal rc0_rc5 <== (temp_r_sumcheck[0] * temp_r_sumcheck[5]);
    eval_vio_panic_termination[0] <== (eval_panic_termination * rc0_rc5 );

    for (var i = 0 ; i < 4; i++) {
        eval_vio_panic_termination[i + 1] <== (eval_vio_panic_termination[i] * (one  - temp_r_sumcheck[i + 1]));
    }

    for (var i = 0 ; i < temp_r_sumcheck_len - 7; i++) {
        eval_vio_panic_termination[i + 5] <== (eval_vio_panic_termination[i + 4] *  (one  - temp_r_sumcheck[i + 6]));
    }

    signal  v_io_eval[3] ;
    v_io_eval[0]  <== (input_size_final_eval + output_size_final_eval);
    v_io_eval[1]  <==  (v_io_eval[0] + eval_vio_panic_termination[temp_r_sumcheck_len - 3]);
   
    v_io_eval[2]  <== (r_prod[start] * v_io_eval[1]);
    signal  eq_eval_io_witness_range_eval[3];
    eq_eval_io_witness_range_eval[0] <== (eq_eval * io_witness_range_eval[1]);

    eq_eval_io_witness_range_eval[1] <== (proof.opening - v_io_eval[2]);

    eq_eval_io_witness_range_eval[2] <==   eq_eval_io_witness_range_eval[0] *  eq_eval_io_witness_range_eval[1];

    eq_eval_io_witness_range_eval[2] === sumcheck_claim;


   signal final_claims[1];
    final_claims[0] <== proof.opening;
    
    output  VerifierOpening(rounds) init_final_verifier_opening;

    (int_transcript[2], init_final_verifier_opening) <== OpeningAccumulator(1, rounds) 
    ( r_sumcheck,
         final_claims,
         int_transcript[1]);
         
     up_transcript <== int_transcript[2];
}

