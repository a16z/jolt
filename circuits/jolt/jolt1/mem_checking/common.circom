pragma circom 2.2.1;
include "./../../../fields/non_native/utils.circom";
include "./../utils.circom";
include "./../../../transcript/transcript.circom";
include "./../jolt1_buses.circom";
include "mem_checking_buses.circom";
include "./../grand_product/grand_product.circom";


template AppendToTranscript(num_read_write_hashes, num_init_hashes, num_final_hashes) {
    input Transcript() transcript;
    input MultisetHashes(num_read_write_hashes,  num_init_hashes, num_final_hashes) multiset_hashes;
    output Transcript() up_transcript;

    var num_of_hashes = num_read_write_hashes;

    Transcript() int_transcript_1[num_of_hashes + 1];

    int_transcript_1[0] <== transcript;
    for (var i = 0; i < num_of_hashes; i++) {
        int_transcript_1[i + 1] <== AppendScalar() (multiset_hashes.read_hashes[i], int_transcript_1[i]);
    }

    Transcript() int_transcript_2[num_of_hashes + 1];
    int_transcript_2[0] <== int_transcript_1[num_of_hashes];
    for (var i = 0; i < num_of_hashes; i++) {
        int_transcript_2[i + 1] <== AppendScalar() (multiset_hashes.write_hashes[i], int_transcript_2[i]);
    }


    Transcript() int_transcript_3[num_init_hashes + 1];
  
    int_transcript_3[0] <== int_transcript_2[num_of_hashes];
    for (var i = 0; i < num_init_hashes; i++) {
        int_transcript_3[i + 1] <== AppendScalar() (multiset_hashes.init_hashes[i], int_transcript_3[i]);
    }


    Transcript() int_transcript_4[num_final_hashes + 1];
    int_transcript_4[0] <== int_transcript_3[num_init_hashes];
    for (var i = 0; i < num_final_hashes; i++) {
        int_transcript_4[i + 1] <== AppendScalar() (multiset_hashes.final_hashes[i], int_transcript_4[i]);
    }
    
    up_transcript <== int_transcript_4[num_final_hashes];
}

template InterleaveHashes(num_read_write_hashes,
                                num_init_hashes, num_final_hashes) {

    input MultisetHashes(num_read_write_hashes,
                                num_init_hashes, num_final_hashes) multiset_hashes;

   signal output  read_write_hashes[2 * num_read_write_hashes];
   signal output  init_final_hashes[num_init_hashes +  num_final_hashes];

    for (var i = 0; i < num_read_write_hashes; i++) {
        read_write_hashes[2*i] <== multiset_hashes.read_hashes[i];
        read_write_hashes[2*i + 1] <== multiset_hashes.write_hashes[i];
    }


    for (var i = 0; i < (num_init_hashes ); i++) {
        init_final_hashes[2*i] <== multiset_hashes.init_hashes[i];
        init_final_hashes[2*i + 1] <== multiset_hashes.final_hashes[i];
    }
}


function MemoryAddressToWitnessIndex(REGISTER_COUNT ,  address ,  memory_layout_input_start) {
        
        var  witness_index = 0;
     
        var offset = address - memory_layout_input_start;
    
        var index = offset / 4;
    
        witness_index = REGISTER_COUNT + index;

        return witness_index;
       
}

