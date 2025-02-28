pragma circom 2.2.1;

include "./../spartan/spartan_hyperkzg/spartan.circom";
include "./linking.circom";
include "./../pcs/hyperkzg.circom";
include "./../jolt/jolt2/jolt2.circom";
// include "./../jolt/jolt1/jolt1_buses.circom";

template Combine(outer_num_rounds, inner_num_rounds, num_vars, rounds_reduced_opening_proof) {

    var  C = 4; 
    var NUM_MEMORIES = 54; 
    var NUM_INSTRUCTIONS = 26; 
    var MEMORY_OPS_PER_INSTRUCTION = 4;
    var chunks_x_size = 4; 
    var chunks_y_size = 4; 
    var NUM_CIRCUIT_FLAGS = 11; 
    var relevant_y_chunks_len = 4;
                
    // Public input of V_{jolt, 1}.
    input JoltPreprocessingNN() jolt_pi;

    // Public output of V_{jolt, 1}.
    input signal counter_jolt_1;

    input LinkingStuff1NN(C, 
                    NUM_MEMORIES, 
                    NUM_INSTRUCTIONS, 
                    MEMORY_OPS_PER_INSTRUCTION,
                    chunks_x_size, 
                    chunks_y_size, 
                    NUM_CIRCUIT_FLAGS, 
                    relevant_y_chunks_len) linking_stuff_1;

    input HyperKZGVerifierKey() vk_spartan_1;
    input Fq() digest;

    // The above four together are the public input of V_{Spartan, 1}. Thus also are part of public input R1CS, say Combined R1CS.

    input NonUniformSpartanProof(outer_num_rounds, 
                                inner_num_rounds, 
                                num_vars) spartan_proof;
                        
    input HyperKZGCommitment() w_commitment;

    // input Transcript() transcript;

    input LinkingStuff2(
        C, 
        NUM_MEMORIES, 
        NUM_INSTRUCTIONS, 
        MEMORY_OPS_PER_INSTRUCTION,
        chunks_x_size, 
        chunks_y_size, 
        NUM_CIRCUIT_FLAGS, 
        relevant_y_chunks_len,
        rounds_reduced_opening_proof
    ) linking_stuff_2;

    // This is the public input of V_{jolt, 2}. Thus also a part of the public input of Combined R1CS.
    input HyperKZGVerifierKey() vk_jolt_2;

    input HyperKZGProof(rounds_reduced_opening_proof) hyperkzg_proof;

    counter_jolt_1 === 1;

    output signal counter_combined_r1cs;
    
    counter_combined_r1cs <== counter_jolt_1 + 1;

    // Public output of V_{Spartan, 1}. Thus also of Combined R1CS.
    output PostponedEval(inner_num_rounds - 1) postponed_eval;

    postponed_eval <== VerifySpartan(outer_num_rounds, inner_num_rounds, num_vars,rounds_reduced_opening_proof)
                                    (jolt_pi, linking_stuff_1, vk_spartan_1, digest, spartan_proof, w_commitment);
    
    PairingCheck(
        C, 
        NUM_MEMORIES, 
        NUM_INSTRUCTIONS, 
        MEMORY_OPS_PER_INSTRUCTION,
        chunks_x_size, 
        chunks_y_size, 
        NUM_CIRCUIT_FLAGS, 
        relevant_y_chunks_len,
        rounds_reduced_opening_proof
    )(linking_stuff_2, vk_jolt_2, hyperkzg_proof);

    Linking(
        C, 
        NUM_MEMORIES, 
        NUM_INSTRUCTIONS, 
        MEMORY_OPS_PER_INSTRUCTION,
        chunks_x_size, 
        chunks_y_size, 
        NUM_CIRCUIT_FLAGS, 
        relevant_y_chunks_len,
        rounds_reduced_opening_proof
    )(linking_stuff_1, linking_stuff_2);
}

// component main = Combine(22, 23, 22, 16);