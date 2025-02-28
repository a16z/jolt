pragma circom 2.2.1;
include "jolt1_buses.circom";
include  "mem_checking/mem_checking_buses.circom";
include  "spartan/spartan_buses.circom";
include  "opening_proof/opening_proof_bus.circom";
include  "instructions/instruction_bus.circom";
include "./../../transcript/transcript.circom";
include "mem_checking/mem_checking.circom";
include "mem_checking/rw_memory.circom";
include "mem_checking/outputsumcheck.circom";
include "mem_checking/timestamp.circom";
include "instructions/instructions_lookups.circom";
include "spartan/spartan.circom";
include "opening_proof/opening_proof.circom";

template verify(     
            num_evals,
            bytecode_words_size, 
            input_size, 
            output_size, 
            num_read_write_hashes_bytecode,
            num_init_final_hashes_bytecode,
            read_write_grand_product_layers_bytecode,
            init_final_grand_product_layers_bytecode,
            max_rounds_bytecode, 
            
            max_rounds_read_write, max_rounds_init_final, num_read_write_hashes_read_write_memory_checking, 
            num_init_final_hashes_read_write_memory_checking,
            read_write_grand_product_layers_read_write_memory_checking,
            init_final_grand_product_layers_read_write_memory_checking,
            max_rounds_timestamp,
            ts_validity_grand_product_layers_timestamp,
            num_read_write_hashes_timestamp,
            num_init_hashes_timestamp,
            MEMORY_OPS_PER_INSTRUCTION,
            max_rounds_outputsumcheck,

            max_rounds_instruction_lookups,
            max_round_init_final_lookups,
            primary_sumcheck_degree_instruction_lookups, 
            primary_sumcheck_num_rounds_instruction_lookups, NUM_MEMORIES, NUM_INSTRUCTIONS,   NUM_SUBTABLES,
            read_write_grand_product_layers_instruction_lookups, init_final_grand_product_layers_instruction_lookups,

            outer_num_rounds_uniform_spartan_proof, inner_num_rounds_uniform_spartan_proof,
            rounds_reduced_opening_proof,
            num_spartan_witness_evals, 
            num_sumcheck_claims,


            WORD_SIZE,
            C, 
            chunks_x_size, 
            chunks_y_size, 
            NUM_CIRCUIT_FLAGS, 
            relevant_y_chunks_len,
            M,
      
            REGISTER_COUNT,
            min_bytecode_address,
            RAM_START_ADDRESS,
            memory_layout_input_start,
            memory_layout_output_start,
            memory_layout_panic,
            memory_layout_termination,
            program_io_panic,
            num_steps, num_cons_total, num_vars, num_rows,

            max_output_size,
            max_input_size

            ) {

    // Scalar representation of label name: b"Jolt transcript"
    Transcript() transcript_init <== TranscriptNew()(604586419824232873836833680384618314);

    input JoltPreprocessing() preprocessing;

    input JoltProof(input_size, 
              output_size, 
              num_read_write_hashes_bytecode,
              num_init_final_hashes_bytecode,
              read_write_grand_product_layers_bytecode,
              init_final_grand_product_layers_bytecode,
              max_rounds_bytecode, 
            
              max_rounds_read_write, max_rounds_init_final, num_read_write_hashes_read_write_memory_checking, 
              num_init_final_hashes_read_write_memory_checking,
              read_write_grand_product_layers_read_write_memory_checking,
              init_final_grand_product_layers_read_write_memory_checking,
              max_rounds_timestamp,
              ts_validity_grand_product_layers_timestamp,
              num_read_write_hashes_timestamp,
              num_init_hashes_timestamp,
              MEMORY_OPS_PER_INSTRUCTION,
              max_rounds_outputsumcheck,

             max_rounds_instruction_lookups,
             max_round_init_final_lookups,
             primary_sumcheck_degree_instruction_lookups, 
             primary_sumcheck_num_rounds_instruction_lookups, NUM_MEMORIES, NUM_INSTRUCTIONS,   NUM_SUBTABLES,
             read_write_grand_product_layers_instruction_lookups, init_final_grand_product_layers_instruction_lookups,
             outer_num_rounds_uniform_spartan_proof, inner_num_rounds_uniform_spartan_proof,
             rounds_reduced_opening_proof,
             num_spartan_witness_evals, 
             num_sumcheck_claims
            ) proof;

    input JoltStuff(C, 
                    NUM_MEMORIES, 
                    NUM_INSTRUCTIONS, 
                    MEMORY_OPS_PER_INSTRUCTION,
                    chunks_x_size, 
                    chunks_y_size, 
                    NUM_CIRCUIT_FLAGS, 
                    relevant_y_chunks_len) commitments;

    input PIProof( num_evals, bytecode_words_size) pi_proof;

    Transcript() transcript[10];
    transcript[0].state <== transcript_init.state;
    transcript[0].nRounds <== transcript_init.nRounds;



    component preamble = FiatShamirPreamble(input_size, output_size, C , M , max_output_size,
                        max_input_size, NUM_INSTRUCTIONS, NUM_SUBTABLES);
    
    preamble.transcript <== transcript[0];
    preamble.program_io <== proof.program_io;
    preamble.trace_length <== proof.trace_length;
    transcript[1] <== preamble.up_transcript;
    
    component append_read_write_values = AppendReadWriteValues(C, 
                                                                NUM_MEMORIES, 
                                                                NUM_INSTRUCTIONS,      
                                                                MEMORY_OPS_PER_INSTRUCTION,
                                                                chunks_x_size, 
                                                                chunks_y_size, 
                                                                NUM_CIRCUIT_FLAGS, 
                                                                relevant_y_chunks_len);
    append_read_write_values.commitments <== commitments;
    append_read_write_values.transcript <==   transcript[1];
    transcript[2] <== append_read_write_values.up_transcript;

    component append_init_final_values = AppendInitFinalValues(C, 
                                                                NUM_MEMORIES, 
                                                                NUM_INSTRUCTIONS, 
                                                                MEMORY_OPS_PER_INSTRUCTION,
                                                                chunks_x_size, 
                                                                chunks_y_size, 
                                                                NUM_CIRCUIT_FLAGS, 
                                                                relevant_y_chunks_len);
    append_init_final_values.commitments <== commitments;
    append_init_final_values.transcript <==  transcript[2];
    transcript[3] <== append_init_final_values.up_transcript;


    component verify_pi = VerifyPI( num_evals, bytecode_words_size);
    verify_pi.preprocessing <== preprocessing;
    verify_pi.proof <== pi_proof;


    component verify_bytecode = VerifyMemoryCheckingBytecode( 
                                                        num_evals,
                                                        num_read_write_hashes_bytecode, 
                                                        num_init_final_hashes_bytecode,
                                                        read_write_grand_product_layers_bytecode,
                                                        init_final_grand_product_layers_bytecode,C,                                        
                                                        max_rounds_bytecode);
    verify_bytecode.preprocessing <== pi_proof.bytecode;
    verify_bytecode.proof <== proof.bytecode;
    verify_bytecode.transcript <==  transcript[3];
    transcript[4] <== verify_bytecode.up_transcript;
    VerifierOpening(read_write_grand_product_layers_bytecode) bytecode_read_write_openings <== verify_bytecode.byte_code_read_write_verifier_opening;
    VerifierOpening(init_final_grand_product_layers_bytecode) bytecode_init_final_openings <== verify_bytecode.init_final_verifier_opening;


    component verify_instruction_lookups = VerifyInstructionLookups(
        max_rounds_instruction_lookups,
        max_round_init_final_lookups,
        primary_sumcheck_degree_instruction_lookups,
        primary_sumcheck_num_rounds_instruction_lookups,
        C, NUM_MEMORIES, NUM_INSTRUCTIONS, NUM_SUBTABLES,
        read_write_grand_product_layers_instruction_lookups, init_final_grand_product_layers_instruction_lookups,
        WORD_SIZE, M
    );
        

    verify_instruction_lookups.proof <== proof.instruction_lookups;
    verify_instruction_lookups.transcript <== transcript[4];
    transcript[5] <== verify_instruction_lookups.up_transcript;
    var instruction_read_write_opening_len = read_write_grand_product_layers_instruction_lookups - 1;
    VerifierOpening(primary_sumcheck_num_rounds_instruction_lookups) instruction_primary_sum_check_opening <== verify_instruction_lookups.instruction_openings;
    VerifierOpening(instruction_read_write_opening_len) instruction_read_write_opening <== verify_instruction_lookups.inst_read_write_openings;
    VerifierOpening(init_final_grand_product_layers_instruction_lookups) instruction_init_final_opening <== verify_instruction_lookups.inst_init_final_openings;

     component rw_memory = VerifyMemoryCheckingReadWrite(
        bytecode_words_size,
        input_size,
        output_size,
        num_read_write_hashes_read_write_memory_checking,
        num_init_final_hashes_read_write_memory_checking,
        read_write_grand_product_layers_read_write_memory_checking,
        init_final_grand_product_layers_read_write_memory_checking,
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
        max_rounds_read_write,
        max_rounds_init_final
    );

    // rw_memory.preprocessing <== proof.pi_proof.read_write_memory;
    rw_memory.preprocessing <== pi_proof.read_write_memory;
    rw_memory.proof <== proof.read_write_memory.memory_checking_proof;
    rw_memory.program_io <== proof.program_io;
    rw_memory.transcript <== transcript[5];
    transcript[6] <== rw_memory.up_transcript;
    VerifierOpening(read_write_grand_product_layers_read_write_memory_checking) memory_checking_read_write_opening <== rw_memory.read_write_verifier_opening;
    VerifierOpening(init_final_grand_product_layers_read_write_memory_checking) memory_checking_init_final_opening <== rw_memory.init_final_verifier_opening;
    


    component output_sum_check = VerifyMemoryCheckingOutputSumCheck(
            input_size,
            output_size,
            max_rounds_outputsumcheck,
            REGISTER_COUNT,
            RAM_START_ADDRESS,
            memory_layout_input_start,
            memory_layout_output_start,
            memory_layout_panic,
            memory_layout_termination,
            program_io_panic
    );

    output_sum_check.proof <== proof.read_write_memory.output_proof;
    output_sum_check.program_io <== proof.program_io;
    output_sum_check.transcript <== transcript[6];

    transcript[7] <== output_sum_check.up_transcript;
    VerifierOpening(max_rounds_outputsumcheck) output_sum_check_opening <== output_sum_check.init_final_verifier_opening;

    

    component timestamp_validity_proof = TimestampValidityVerifier(
                         max_rounds_timestamp,
                         ts_validity_grand_product_layers_timestamp,
                         num_read_write_hashes_timestamp,
                         num_init_hashes_timestamp,
                         C, 
                         NUM_MEMORIES, 
                         NUM_INSTRUCTIONS, 
                         MEMORY_OPS_PER_INSTRUCTION,
                         chunks_x_size, 
                         chunks_y_size, 
                         NUM_CIRCUIT_FLAGS, 
                         relevant_y_chunks_len
    );


    timestamp_validity_proof.proof <== proof.read_write_memory.timestamp_validity_proof;
    timestamp_validity_proof.transcript <== transcript[7];
    transcript[8] <== timestamp_validity_proof.up_transcript;
    VerifierOpening(ts_validity_grand_product_layers_timestamp) timestamp_validity_opening <== timestamp_validity_proof.ts_verifier_opening;
       

    var num_var_next_power2 =  NextPowerOf2(num_vars);
    var num_var_next_power2_log2 = log2(num_var_next_power2);
    var n_prefix = num_var_next_power2_log2 + 1;
    var r1cs_opening_len = inner_num_rounds_uniform_spartan_proof - n_prefix;

    R1CSProof(outer_num_rounds_uniform_spartan_proof, inner_num_rounds_uniform_spartan_proof, num_spartan_witness_evals)  spartan_proof; 
    spartan_proof.proof <== proof.r1cs;


   component uniform_spartan_proof = VerifyR1CS(num_spartan_witness_evals, num_steps, num_cons_total, num_vars, num_rows);
    uniform_spartan_proof.r1cs_proof <== spartan_proof;
    uniform_spartan_proof.transcript <== transcript[8];
    transcript[9] <== uniform_spartan_proof.up_transcript;
    VerifierOpening(r1cs_opening_len) r1cs_opening <== uniform_spartan_proof.r1cs_opening;


    component reduce_and_verify = ReduceAndVerify(
        read_write_grand_product_layers_bytecode, init_final_grand_product_layers_bytecode,
                        primary_sumcheck_num_rounds_instruction_lookups, instruction_read_write_opening_len, init_final_grand_product_layers_instruction_lookups,
                        read_write_grand_product_layers_read_write_memory_checking, init_final_grand_product_layers_read_write_memory_checking,
                        max_rounds_outputsumcheck, ts_validity_grand_product_layers_timestamp,
                        r1cs_opening_len,
                        rounds_reduced_opening_proof);

    reduce_and_verify.transcript <== transcript[9];
    reduce_and_verify.reduced_opening_proof <== proof.opening_proof;
    reduce_and_verify.byte_code_read_write_openings <== bytecode_read_write_openings;
    reduce_and_verify.byte_code_init_final_openings <== bytecode_init_final_openings;
    reduce_and_verify.inst_primary_sum_check_openings <== instruction_primary_sum_check_opening;
    reduce_and_verify.inst_read_write_openings <== instruction_read_write_opening;
    reduce_and_verify.inst_init_final_openings <== instruction_init_final_opening;

    reduce_and_verify.memory_checking_read_write_openings <== memory_checking_read_write_opening;
    reduce_and_verify.memory_checking_init_final_openings <== memory_checking_init_final_opening;

    reduce_and_verify.output_sum_check_openings <== output_sum_check_opening;
    reduce_and_verify.timestamp_validity_openings <== timestamp_validity_opening;
    reduce_and_verify.r1cs_openings <== r1cs_opening;

    signal opening_combiners_coeff <== reduce_and_verify.gamma;
    HyperKzgVerifierAdvice()  hyperkzg_verifier_advice <== reduce_and_verify.hyperkzg_verifier_advice;

    output signal counter_jolt_1 <== 1;
    
    output   LinkingStuff1(C, 
                    NUM_MEMORIES, 
                    NUM_INSTRUCTIONS, 
                    MEMORY_OPS_PER_INSTRUCTION,
                    chunks_x_size, 
                    chunks_y_size, 
                    NUM_CIRCUIT_FLAGS, 
                    relevant_y_chunks_len, rounds_reduced_opening_proof) linkingstuff ;

    linkingstuff.commitments <== commitments;
    linkingstuff.opening_combiners.bytecode_combiners.rho[0] <==  bytecode_read_write_openings.rho ;
    linkingstuff.opening_combiners.bytecode_combiners.rho[1] <==  bytecode_init_final_openings.rho ;

    linkingstuff.opening_combiners.instruction_lookup_combiners.rho[0] <==  instruction_primary_sum_check_opening.rho ;
    linkingstuff.opening_combiners.instruction_lookup_combiners.rho[1] <==  instruction_read_write_opening.rho ;
    linkingstuff.opening_combiners.instruction_lookup_combiners.rho[2] <==  instruction_init_final_opening.rho ;


    linkingstuff.opening_combiners.read_write_output_timestamp_combiners.rho[0] <==  memory_checking_read_write_opening.rho ;
    linkingstuff.opening_combiners.read_write_output_timestamp_combiners.rho[1] <==  memory_checking_init_final_opening.rho ;
    linkingstuff.opening_combiners.read_write_output_timestamp_combiners.rho[2] <==  output_sum_check_opening.rho ;
    linkingstuff.opening_combiners.read_write_output_timestamp_combiners.rho[3] <==  timestamp_validity_opening.rho ;

    linkingstuff.opening_combiners.spartan_combiners.rho <==  r1cs_opening.rho ;

    linkingstuff.opening_combiners.coefficient <==  opening_combiners_coeff;

    linkingstuff.hyperkzg_verifier_advice <==  hyperkzg_verifier_advice;
}





bus LinkingStuff1(C, 
                    NUM_MEMORIES, 
                    NUM_INSTRUCTIONS, 
                    MEMORY_OPS_PER_INSTRUCTION,
                    chunks_x_size, 
                    chunks_y_size, 
                    NUM_CIRCUIT_FLAGS, 
                    relevant_y_chunks_len,rounds_reduced_opening_proof) {
   JoltStuff(C, 
                    NUM_MEMORIES, 
                    NUM_INSTRUCTIONS, 
                    MEMORY_OPS_PER_INSTRUCTION,
                    chunks_x_size, 
                    chunks_y_size, 
                    NUM_CIRCUIT_FLAGS, 
                    relevant_y_chunks_len) commitments;


    OpeningCombiners() opening_combiners;

    HyperKzgVerifierAdvice() hyperkzg_verifier_advice;
 
}


bus HyperKzgVerifierAdvice() {
   signal  r;
   signal  d_0;
   signal  v;
   signal  q_power;
}




bus OpeningCombiners() {

    BytecodeCombiners()     bytecode_combiners;
  
    InstructionLookupCombiners() instruction_lookup_combiners;

    ReadWriteOutputTimestampCombiners() read_write_output_timestamp_combiners;

    SpartanCombiners() spartan_combiners;

    signal coefficient;


}



bus BytecodeCombiners() {
    signal rho[2] ;
}


bus InstructionLookupCombiners() {
    signal rho[3] ;
}


bus ReadWriteOutputTimestampCombiners() {
    signal rho[4] ;
}


bus SpartanCombiners() {
    signal rho ;
}




template AppendReadWriteValues(C, 
                                NUM_MEMORIES, 
                                NUM_INSTRUCTIONS, 
                                MEMORY_OPS_PER_INSTRUCTION,
                                chunks_x_size, 
                                chunks_y_size, 
                                NUM_CIRCUIT_FLAGS, 
                                relevant_y_chunks_len) {

    input JoltStuff(C, 
                    NUM_MEMORIES, 
                    NUM_INSTRUCTIONS, 
                    MEMORY_OPS_PER_INSTRUCTION,
                    chunks_x_size, 
                    chunks_y_size, 
                    NUM_CIRCUIT_FLAGS, 
                    relevant_y_chunks_len) commitments;

    input Transcript() transcript;
    output Transcript() up_transcript;

    component append_points = AppendPoints(20 + C + 2 * NUM_MEMORIES + NUM_INSTRUCTIONS 
                                            + 4 * MEMORY_OPS_PER_INSTRUCTION
                                            + chunks_x_size + chunks_y_size 
                                            + NUM_CIRCUIT_FLAGS + 8 + relevant_y_chunks_len);


    append_points.points[0] <== commitments.bytecode.a_read_write.commitment;
    for (var i = 0; i < 6; i++) {
        append_points.points[i+1] <== commitments.bytecode.v_read_write[i].commitment;
    }
    append_points.points[7] <== commitments.bytecode.t_read.commitment;


    append_points.points[8] <== commitments.read_write_memory.a_ram.commitment;
    append_points.points[9] <== commitments.read_write_memory.v_read_rd.commitment;
    append_points.points[10] <== commitments.read_write_memory.v_read_rs1.commitment;
    append_points.points[11] <== commitments.read_write_memory.v_read_rs2.commitment;
    append_points.points[12] <== commitments.read_write_memory.v_read_ram.commitment;
    append_points.points[13] <== commitments.read_write_memory.v_write_rd.commitment;
    append_points.points[14] <== commitments.read_write_memory.v_write_ram.commitment;
    append_points.points[15] <== commitments.read_write_memory.t_read_rd.commitment;
    append_points.points[16] <== commitments.read_write_memory.t_read_rs1.commitment;
    append_points.points[17] <== commitments.read_write_memory.t_read_rs2.commitment;
    append_points.points[18] <== commitments.read_write_memory.t_read_ram.commitment;

    for (var i = 0; i < C; i++) {
        append_points.points[19 + i] <== commitments.instruction_lookups.dim[i].commitment;
    }
    for (var i = 0; i < NUM_MEMORIES; i++) {
        append_points.points[19 + C + i] <== commitments.instruction_lookups.read_cts[i].commitment;
        append_points.points[19 + C + NUM_MEMORIES + i] <== commitments.instruction_lookups.E_polys[i].commitment;
    }
    for (var i = 0; i < NUM_INSTRUCTIONS; i++) {
        append_points.points[19 + C + 2 * NUM_MEMORIES + i] <== commitments.instruction_lookups.instruction_flags[i].commitment;
    }

    append_points.points[19 + C + 2 * NUM_MEMORIES + NUM_INSTRUCTIONS ] <== commitments.instruction_lookups.lookup_outputs.commitment;

    var count = 20 + C + 2 * NUM_MEMORIES + NUM_INSTRUCTIONS;
     for (var i = 0; i < MEMORY_OPS_PER_INSTRUCTION; i++) {
         append_points.points[count + i] <== commitments.timestamp_range_check.read_cts_read_timestamp[i].commitment;
         append_points.points[count + MEMORY_OPS_PER_INSTRUCTION + i] <== commitments.timestamp_range_check.read_cts_global_minus_read[i].commitment;
         append_points.points[count + 2 * MEMORY_OPS_PER_INSTRUCTION + i] <== commitments.timestamp_range_check.final_cts_read_timestamp[i].commitment;
         append_points.points[count + 3 * MEMORY_OPS_PER_INSTRUCTION + i] <== commitments.timestamp_range_check.final_cts_global_minus_read[i].commitment;
     }


    count += 4 * MEMORY_OPS_PER_INSTRUCTION;
    for (var i = 0; i < chunks_x_size; i++) {
        append_points.points[count + i] <== commitments.r1cs.chunks_x[i].commitment;
    }
    count += chunks_x_size;
    for (var i = 0; i < chunks_y_size; i++) {
        append_points.points[count + i] <== commitments.r1cs.chunks_y[i].commitment;
    }
    count += chunks_y_size;
    for (var i = 0; i < NUM_CIRCUIT_FLAGS; i++) {
        append_points.points[count + i] <== commitments.r1cs.circuit_flags[i].commitment;
    }
    count += NUM_CIRCUIT_FLAGS;

    // // To append from aux
    append_points.points[count + 0] <== commitments.r1cs.aux.left_lookup_operand.commitment;
    append_points.points[count + 1] <== commitments.r1cs.aux.right_lookup_operand.commitment;
    append_points.points[count + 2] <== commitments.r1cs.aux.product.commitment;
    count += 3;
    for (var i = 0; i < relevant_y_chunks_len; i++) {
        append_points.points[count + i] <== commitments.r1cs.aux.relevant_y_chunks[i].commitment;
    }
    count += relevant_y_chunks_len;
    append_points.points[count + 0] <== commitments.r1cs.aux.write_lookup_output_to_rd.commitment;
    append_points.points[count + 1] <== commitments.r1cs.aux.write_pc_to_rd.commitment;
    append_points.points[count + 2] <== commitments.r1cs.aux.next_pc_jump.commitment;
    append_points.points[count + 3] <== commitments.r1cs.aux.should_branch.commitment;
    append_points.points[count + 4] <== commitments.r1cs.aux.next_pc.commitment;

    append_points.transcript <== transcript;
    up_transcript <== append_points.up_transcript;

}


template AppendInitFinalValues(C, 
                                NUM_MEMORIES, 
                                NUM_INSTRUCTIONS, 
                                MEMORY_OPS_PER_INSTRUCTION,
                                chunks_x_size, 
                                chunks_y_size, 
                                NUM_CIRCUIT_FLAGS, 
                                relevant_y_chunks_len) {

    input JoltStuff(C, 
                    NUM_MEMORIES, 
                    NUM_INSTRUCTIONS, 
                    MEMORY_OPS_PER_INSTRUCTION,
                    chunks_x_size, 
                    chunks_y_size, 
                    NUM_CIRCUIT_FLAGS, 
                    relevant_y_chunks_len) commitments;

    input Transcript() transcript;
    output Transcript() up_transcript;

    component append_points = AppendPoints(3 + NUM_MEMORIES);

    append_points.points[0] <== commitments.bytecode.t_final.commitment;

    append_points.points[1] <== commitments.read_write_memory.v_final.commitment;
    append_points.points[2] <== commitments.read_write_memory.t_final.commitment;

    for (var i = 0; i < NUM_MEMORIES; i++) {
        append_points.points[3 + i] <== commitments.instruction_lookups.final_cts[i].commitment;
    }

    append_points.transcript <== transcript;
    up_transcript <== append_points.up_transcript;
}



template FiatShamirPreamble(input_size, output_size, C , M, maximum_output_size, 
                            maximum_input_size, NUM_INSTRUCTIONS, NUM_SUBTABLES) {
    
    input Transcript() transcript;
    input JoltDevice(input_size, output_size) program_io;
   
    input signal trace_length;
    
    output Transcript() up_transcript;

    Transcript() int_transcript[9];

    int_transcript[0]  <== AppendScalar()(trace_length, transcript);
 
    component append_2 = AppendScalar();
    append_2.scalar <-- C;
    append_2.transcript <== int_transcript[0];
    int_transcript[1] <== append_2.up_transcript;

    component append_3 = AppendScalar();
    append_3.scalar <-- M;
    append_3.transcript <== int_transcript[1];
    int_transcript[2] <== append_3.up_transcript;

    component append_4 = AppendScalar();
    append_4.scalar <-- NUM_INSTRUCTIONS;
    append_4.transcript <== int_transcript[2];
    int_transcript[3] <== append_4.up_transcript;

    component append_5 = AppendScalar(); 
    append_5.scalar <-- NUM_SUBTABLES;
    append_5.transcript <== int_transcript[3];
    int_transcript[4] <== append_5.up_transcript;

    component append_6 = AppendScalar();
    append_6.scalar <== maximum_input_size;
    append_6.transcript <== int_transcript[4];
    int_transcript[5] <== append_6.up_transcript;


    component append_7 = AppendScalar();
    append_7.scalar <-- maximum_output_size;
    append_7.transcript <== int_transcript[5];
    int_transcript[6] <== append_7.up_transcript;
    int_transcript[7] <== AppendBytes(input_size)(program_io.inputs, int_transcript[6]);

    int_transcript[8] <== AppendBytes(output_size)(program_io.outputs, int_transcript[7]);
   
    up_transcript <== AppendScalar()(program_io.panic, int_transcript[8]);
}

template VerifyPI ( num_evals, bytecode_words_size) {
    input JoltPreprocessing() preprocessing;
    input PIProof( num_evals, bytecode_words_size) proof;

    signal v_init_final_flattened[6 * num_evals];

    for (var i = 0; i < 6; i++) {
        for (var j = 0; j < num_evals; j++) {
            v_init_final_flattened[num_evals * i + j] <== proof.bytecode.v_init_final[i][j];
        }
    }

    VerifyHashing(6 * num_evals)(preprocessing.v_init_final_hash, v_init_final_flattened);
    VerifyHashing(bytecode_words_size)(preprocessing.bytecode_words_hash, proof.read_write_memory.bytecode_words);
}


// component main =  verify( 1<<9, 293, 1, 1, 1, 1, 9, 9, 9,  // num_evals,bytecode_words_size, input_size, output_size, num_read_write_hashes_bytecode,num_init_final_hashes_bytecode,read_write_grand_product_layers_bytecode,init_final_grand_product_layers_bytecode,max_rounds_bytecode, 
//         11, 13, 4,1,9,13,  // max_rounds_read_write,num_read_write_hashes_read_write_memory_checking, num_init_final_hashes_read_write_memory_checking,read_write_grand_product_layers_read_write_memory_checking,init_final_grand_product_layers_read_write_memory_checking, 
//         13,9,8,1,4, 
//         13,  // max_rounds_timestamp,ts_validity_grand_product_layers_timestamp,num_read_write_hashes_timestamp,num_init_hashes_timestamp,MEMORY_OPS_PER_INSTRUCTION,max_rounds_outputsumcheck,
//         16, 22, 8, 9,  54, 26, 26, 10, 16, // max_rounds_instruction_lookups,primary_sumcheck_degree_instruction_lookups, primary_sumcheck_num_rounds_instruction_lookups, NUM_MEMORIES, NUM_INSTRUCTIONS,   NUM_SUBTABLES,read_write_grand_product_layers_instruction_lookups, init_final_grand_product_layers_instruction_lookups,
//         16,17, //  outer_num_rounds_uniform_spartan_proof, inner_num_rounds_uniform_spartan_proof,
//         16,76,10, //rounds_reduced_opening_proof, num_spartan_witness_evals, num_sumcheck_claims,
    
//        //  WORD_SIZE, C, chunks_y_size, chunks_x_size, NUM_CIRCUIT_FLAGS, relevant_y_chunks_len,M,
//         32, 4, 4, 4, 11, 4, 1 << 16, 

//        // REGISTER_COUNT, min_bytecode_address,RAM_START_ADDRESS,,memory_layout_input_start,memory_layout_output_start,memory_layout_panic,memory_layout_termination,program_io_panic,
//         64,    2147483648, 2147483648, 2147467520, 2147471616,2147475712, 2147475716, 0,
//        //  num_steps, num_cons_total, num_vars, num_rows
//         512, 65536, 76, 67, 4096, 4096

// );