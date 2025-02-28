pragma circom 2.2.1;
include "./../../../fields/non_native/utils.circom";
include "./../utils.circom";
include "./../../../transcript/transcript.circom";
include "./../jolt1_buses.circom";
include "mem_checking_buses.circom";
include "./../grand_product/grand_product.circom";



bus OutputSumcheckProof(degree, rounds) {
        SumcheckInstanceProof(degree,rounds) sumcheck_proof;
        signal opening;
}


bus BytecodePreprocessing( num_evals) {

    signal v_init_final[6][num_evals];
}

bus ReadWriteMemoryPreprocessing(bytecode_words_size) {

    signal bytecode_words[bytecode_words_size];


}


bus MemoryLayout() {
    signal max_input_size;
    signal max_output_size;
    signal input_start;
    signal input_end;
    signal output_start;
    signal output_end;
    signal panic;
    signal termination;
}



bus BytecodeProof(num_read_write_hashes, 
                    num_init_final_hashes,
                    read_write_grand_product_layers,
                    init_final_grand_product_layers,max_rounds) {
    MultisetHashes(num_read_write_hashes, num_init_final_hashes, num_init_final_hashes) multiset_hashes;
    BatchedGrandProductProof(max_rounds,read_write_grand_product_layers) read_write_grand_product;
    BatchedGrandProductProof(max_rounds,init_final_grand_product_layers) init_final_grand_product;
    
    signal openings[9];
}


bus ReadWriteMemoryProof(max_rounds_read_write, max_rounds_init_final, num_read_write_hashes_read_write_memory_checking, 
                    num_init_final_hashes_read_write_memory_checking,
                    read_write_grand_product_layers_read_write_memory_checking,
                    init_final_grand_product_layers_read_write_memory_checking,
                    max_rounds_timestamp,
                    ts_validity_grand_product_layers_timestamp,
                    num_read_write_hashes_timestamp,
                    num_init_hashes_timestamp,
                    MEMORY_OPS_PER_INSTRUCTION,
                    max_rounds_outputsumcheck
                    ) {

    ReadWriteMemoryCheckingProof(max_rounds_read_write, max_rounds_init_final, num_read_write_hashes_read_write_memory_checking, 
                   num_init_final_hashes_read_write_memory_checking,
                    read_write_grand_product_layers_read_write_memory_checking,
                    init_final_grand_product_layers_read_write_memory_checking) memory_checking_proof;


    TimestampValidityProof(max_rounds_timestamp, ts_validity_grand_product_layers_timestamp, num_read_write_hashes_timestamp, num_init_hashes_timestamp ,MEMORY_OPS_PER_INSTRUCTION) timestamp_validity_proof;
    OutputSumcheckProof(3,max_rounds_outputsumcheck) output_proof;
}



bus ReadWriteMemoryCheckingProof(max_rounds, 
                    max_rounds_init_final,
                    num_read_write_hashes,
                    num_init_final_hashes, 
                    read_write_grand_product_layers,
                    init_final_grand_product_layers) {
    
    MultisetHashes(num_read_write_hashes, num_init_final_hashes, num_init_final_hashes) multiset_hashes;
    BatchedGrandProductProof(max_rounds,read_write_grand_product_layers) read_write_grand_product;
    BatchedGrandProductProof(max_rounds_init_final, init_final_grand_product_layers) init_final_grand_product;
    
    signal openings[13];
    signal exogenous_openings[3];
}



bus TimestampValidityProof(max_rounds, ts_validity_grand_product_layers,num_read_write_hashes, num_init_hashes,MEMORY_OPS_PER_INSTRUCTION) {
    MultisetHashes(num_read_write_hashes, num_init_hashes, num_read_write_hashes ) multiset_hashes;
    TimestampRangeCheckOpenings(MEMORY_OPS_PER_INSTRUCTION) openings;
    signal exogenous_openings[MEMORY_OPS_PER_INSTRUCTION];
    BatchedGrandProductProof(max_rounds,ts_validity_grand_product_layers) batched_grand_product;
}



bus TimestampRangeCheckOpenings(MEMORY_OPS_PER_INSTRUCTION) {
    signal read_cts_read_timestamp[MEMORY_OPS_PER_INSTRUCTION];
    signal read_cts_global_minus_read[MEMORY_OPS_PER_INSTRUCTION];
    signal final_cts_read_timestamp[MEMORY_OPS_PER_INSTRUCTION];
    signal final_cts_global_minus_read[MEMORY_OPS_PER_INSTRUCTION];
    // signal identity;
}


