pragma circom 2.2.1;
include "./../grand_product/grand_product.circom";

bus InstructionLookupsProof(max_rounds, max_rounds_init_final,
                    primary_sumcheck_degree, primary_sumcheck_num_rounds, NUM_MEMORIES, NUM_INSTRUCTIONS, NUM_SUBTABLES,  
                    read_write_grand_product_layers, init_final_grand_product_layers) {

    PrimarySumcheck(primary_sumcheck_degree, primary_sumcheck_num_rounds, NUM_MEMORIES, NUM_INSTRUCTIONS) primary_sumcheck;
    
    InstMemoryCheckingProof(max_rounds, max_rounds_init_final,  NUM_MEMORIES, NUM_SUBTABLES, NUM_INSTRUCTIONS,
                    read_write_grand_product_layers, init_final_grand_product_layers) memory_checking_proof;
}

bus InstMemoryCheckingProof(max_rounds, max_rounds_init_final,  NUM_MEMORIES, NUM_INSTRUCTIONS,
                            NUM_SUBTABLES,
                            read_write_grand_product_layers,
                            init_final_grand_product_layers) {
    
    MultisetHashes(NUM_MEMORIES, NUM_SUBTABLES, NUM_MEMORIES) multiset_hashes;
    BatchedGrandProductProof(max_rounds, read_write_grand_product_layers) read_write_grand_product;
    BatchedGrandProductProof(max_rounds_init_final, init_final_grand_product_layers) init_final_grand_product;
    signal openings[4 + 3 * NUM_MEMORIES + NUM_INSTRUCTIONS + 1];

}