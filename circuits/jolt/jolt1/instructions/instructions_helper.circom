
pragma circom 2.2.1;
include "./instruction_combine_lookups.circom";
include "./../utils.circom";




template CheckMultisetEqualityInst(NUM_MEMORIES, NUM_SUBTABLES) {
    input MultisetHashes(NUM_MEMORIES, NUM_SUBTABLES, NUM_MEMORIES) multiset_hahses;
    
    signal left_value[NUM_MEMORIES];
    signal right_value[NUM_MEMORIES];

    for (var i = 0; i < NUM_MEMORIES; i++) {
        left_value[i] <== multiset_hahses.read_hashes[i] * multiset_hahses.final_hashes[i];
        right_value[i] <== multiset_hahses.init_hashes[MemoryToSubtableIndex(i)] * multiset_hahses.write_hashes[i];
        left_value[i] === right_value[i];
    }

}

template CombineLookups(C, M, WORD_SIZE, NUM_INSTRUCTIONS, NUM_MEMORIES){
    input signal  e_poly_openings[NUM_MEMORIES];
    input signal flag_openings[NUM_INSTRUCTIONS];
    output signal result;

    signal  sum[26];

    var instruction_index = 0;
    var instruction_to_memory_indices_len[26] = [4, 4, 4, 4, 4, 4, 9, 7, 4, 9, 7, 4, 5, 4, 5, 4, 4, 2, 4, 4, 8, 18, 11, 8, 1, 2] ; 

    signal filtered_operands_add[instruction_to_memory_indices_len[instruction_index]];
    for (var i = 0; i < instruction_to_memory_indices_len[instruction_index]; i++) {
        filtered_operands_add[i] <== e_poly_openings[InstructionToMemoryIndices(instruction_index ,i)];
    }
    signal temp_value_add <== CombineLookupsAdd(C, M)(filtered_operands_add);
    signal temp_add <==  flag_openings[instruction_index] *  temp_value_add ;
    sum[0] <== temp_add;
  

    // Instruction SUB
    instruction_index += 1;
    signal filtered_operands_sub[instruction_to_memory_indices_len[instruction_index]];
    for (var i = 0; i < instruction_to_memory_indices_len[instruction_index]; i++) {
        filtered_operands_sub[i] <== e_poly_openings[InstructionToMemoryIndices(instruction_index ,i)];
    }
    
    signal temp_value_sub <== CombineLookupsSub(C, M)(filtered_operands_sub);
    signal temp_sub <==   flag_openings[instruction_index] * temp_value_sub;
    sum[1] <== (sum[0] + temp_sub);
  

    // Instruction AND
    instruction_index += 1;
 
    signal filtered_operands_and[instruction_to_memory_indices_len[instruction_index]];
    
    for (var i = 0; i < instruction_to_memory_indices_len[instruction_index]; i++) {
        filtered_operands_and[i] <== e_poly_openings[InstructionToMemoryIndices(instruction_index ,i)];
    }

    signal temp_value_and <== CombineLookupsAnd(C, M)(filtered_operands_and);
    signal temp_and <==   flag_openings[instruction_index] *  temp_value_and;
    sum[2] <== (sum[1] + temp_and);
  
    // Instruction OR
    instruction_index += 1;
  
    signal filtered_operands_or[instruction_to_memory_indices_len[instruction_index]];
    for (var i = 0; i < instruction_to_memory_indices_len[instruction_index]; i++) {
        filtered_operands_or[i] <== e_poly_openings[InstructionToMemoryIndices(instruction_index ,i)];
    }
    signal temp_value_or <== CombineLookupsOr(C, M)(filtered_operands_or);  
    signal temp_or <==   flag_openings[instruction_index] *  temp_value_or;
    sum[3] <== (sum[2] + temp_or);
  

    // Instruction XOR
    instruction_index += 1;
  
    signal filtered_operands_xor[instruction_to_memory_indices_len[instruction_index]];
    for (var i = 0; i < instruction_to_memory_indices_len[instruction_index]; i++) {
        filtered_operands_xor[i] <== e_poly_openings[InstructionToMemoryIndices(instruction_index ,i)];
    }
    signal temp_value_xor <== CombineLookupsXor(C, M)(filtered_operands_xor);
    signal temp_xor <==  flag_openings[instruction_index] * temp_value_xor;
    sum[4] <== (sum[3] + temp_xor);

    // Instruction BEQ
    instruction_index += 1;
  
    signal filtered_operands_beq[instruction_to_memory_indices_len[instruction_index]];
    for (var i = 0; i < instruction_to_memory_indices_len[instruction_index]; i++) {
        filtered_operands_beq[i] <== e_poly_openings[InstructionToMemoryIndices(instruction_index ,i)];
    }
    signal temp_value_beq <== CombineLookupsBeq(C, M)(filtered_operands_beq);
    signal temp_beq <==   flag_openings[instruction_index] * temp_value_beq;
    sum[5] <== (sum[4] + temp_beq);
  

    // Instruction BGE

    instruction_index += 1;
  
    signal filtered_operands_bge[instruction_to_memory_indices_len[instruction_index]];
    for (var i = 0; i < instruction_to_memory_indices_len[instruction_index]; i++) {
        filtered_operands_bge[i] <== e_poly_openings[InstructionToMemoryIndices(instruction_index ,i)];
    }

    signal temp_value_bge <== CombineLookupsBge(C, M)(filtered_operands_bge);
    signal temp_bge <==flag_openings[instruction_index] * temp_value_bge;
    sum[6] <== (sum[5] + temp_bge);
  
    // Instruction BGEU
    instruction_index += 1;
    signal filtered_operands_bgeu[instruction_to_memory_indices_len[instruction_index]];
    for (var i = 0; i < instruction_to_memory_indices_len[instruction_index]; i++) {
        filtered_operands_bgeu[i] <== e_poly_openings[InstructionToMemoryIndices(instruction_index ,i)];
    }       

    signal temp_value_bgeu <== CombineLookupsBgeu(C, M)(filtered_operands_bgeu);
    signal temp_bgeu <==  flag_openings[instruction_index] * temp_value_bgeu;
    sum[7] <== (sum[6] + temp_bgeu);
  
    // Instruction BNE
    instruction_index += 1;
    signal filtered_operands_bne[instruction_to_memory_indices_len[instruction_index]];
    for (var i = 0; i < instruction_to_memory_indices_len[instruction_index]; i++) {
        filtered_operands_bne[i] <==e_poly_openings[InstructionToMemoryIndices(instruction_index ,i)];
    }
    signal temp_value_bne <== CombineLookupsBne(C, M)(filtered_operands_bne);
    signal temp_bne <==  flag_openings[instruction_index] * temp_value_bne;
    sum[8] <== (sum[7] + temp_bne);
  
    // Instruction SLT
    instruction_index += 1;
    signal filtered_operands_slt[instruction_to_memory_indices_len[instruction_index]];
    for (var i = 0; i < instruction_to_memory_indices_len[instruction_index]; i++) {
        filtered_operands_slt[i] <== e_poly_openings[InstructionToMemoryIndices(instruction_index ,i)];
    }

    signal temp_value_slt <== CombineLookupsSlt(C, M)(filtered_operands_slt);
    signal temp_slt <==  flag_openings[instruction_index] * temp_value_slt;
    sum[9] <== (sum[8] + temp_slt);
  

    // Instruction SLTU
    instruction_index += 1;
    signal filtered_operands_sltu[instruction_to_memory_indices_len[instruction_index]];
    for (var i = 0; i < instruction_to_memory_indices_len[instruction_index]; i++) {
        filtered_operands_sltu[i] <== e_poly_openings[InstructionToMemoryIndices(instruction_index ,i)];
    }

    signal temp_value_sltu <== CombineLookupsSltu(C, M)(filtered_operands_sltu);
    signal temp_sltu <==  flag_openings[instruction_index] *  temp_value_sltu;
    sum[10] <== (sum[9] + temp_sltu);
  

    // Instruction SLL
    instruction_index += 1;
    signal filtered_operands_sll[instruction_to_memory_indices_len[instruction_index]];
    for (var i = 0; i < instruction_to_memory_indices_len[instruction_index]; i++) {
        filtered_operands_sll[i] <== e_poly_openings[InstructionToMemoryIndices(instruction_index, i)];
    }

    signal temp_value_sll <== CombineLookupsSll(C, M)(filtered_operands_sll);
    signal temp_sll <== flag_openings[instruction_index] * temp_value_sll;
    sum[11] <== (sum[10] + temp_sll);
  
    // Instruction SRA
    instruction_index += 1;
    signal filtered_operands_sra[instruction_to_memory_indices_len[instruction_index]];
    for (var i = 0; i < instruction_to_memory_indices_len[instruction_index]; i++) {
        filtered_operands_sra[i] <==e_poly_openings[InstructionToMemoryIndices(instruction_index, i)];
    }

    signal temp_value_sra <== CombineLookupsSra(C, M)(filtered_operands_sra);
    signal temp_sra <==  flag_openings[instruction_index] * temp_value_sra;
    sum[12] <== (sum[11] + temp_sra);
  

    // Instruction SRL
    instruction_index += 1;
  
    signal filtered_operands_srl[instruction_to_memory_indices_len[instruction_index]];
    for (var i = 0; i < instruction_to_memory_indices_len[instruction_index]; i++) {
        filtered_operands_srl[i] <== e_poly_openings[InstructionToMemoryIndices(instruction_index ,i)];
    }

    signal temp_value_srl <== CombineLookupsSrl(C, M)(filtered_operands_srl);
    signal temp_srl <==  flag_openings[instruction_index] * temp_value_srl;
    sum[13] <== (sum[12] + temp_srl);
  

    // Instruction MOVSIGN
    instruction_index += 1;
  
    signal filtered_operands_movsign[instruction_to_memory_indices_len[instruction_index]];
    for (var i = 0; i < instruction_to_memory_indices_len[instruction_index]; i++) {
        filtered_operands_movsign[i] <== e_poly_openings[InstructionToMemoryIndices(instruction_index, i)];
    }

    signal temp_value_movsign <== CombineLookupsVirtualMovsign(instruction_to_memory_indices_len[instruction_index], M, WORD_SIZE)(filtered_operands_movsign);
    signal temp_movsign <==  flag_openings[instruction_index] * temp_value_movsign;
    sum[14] <== (sum[13] + temp_movsign);
  

    // Instruction MUL
    instruction_index += 1;
   
    signal filtered_operands_mul[instruction_to_memory_indices_len[instruction_index]];
    for (var i = 0; i < instruction_to_memory_indices_len[instruction_index]; i++) {
        filtered_operands_mul[i] <== e_poly_openings[InstructionToMemoryIndices(instruction_index ,i)];
    }

    signal temp_value_mul <==  CombineLookupsMul(instruction_to_memory_indices_len[instruction_index], M)(filtered_operands_mul);
    signal temp_mul <==flag_openings[instruction_index] * temp_value_mul;
    sum[15] <== (sum[14] + temp_mul);
  

    // Instruction MULU
    instruction_index += 1;
   
    signal filtered_operands_mulu[instruction_to_memory_indices_len[instruction_index]];
    for (var i = 0; i < instruction_to_memory_indices_len[instruction_index]; i++) {
        filtered_operands_mulu[i] <==e_poly_openings[InstructionToMemoryIndices(instruction_index ,i)];
    }

    signal temp_value_mulu <==  CombineLookupsMulu(instruction_to_memory_indices_len[instruction_index], M)(filtered_operands_mulu);
    signal temp_mulu <==  flag_openings[instruction_index] * temp_value_mulu;
    sum[16] <== (sum[15] + temp_mulu);
      

    // Instruction MULHU
    instruction_index += 1;
  
    signal filtered_operands_mulhu[instruction_to_memory_indices_len[instruction_index]];
    for (var i = 0; i < instruction_to_memory_indices_len[instruction_index]; i++) {
        filtered_operands_mulhu[i] <== e_poly_openings[InstructionToMemoryIndices(instruction_index ,i)];
    }

    signal temp_value_mulhu <==  CombineLookupsMulhu(instruction_to_memory_indices_len[instruction_index], M)(filtered_operands_mulhu);
    signal temp_mulhu <==  flag_openings[instruction_index] * temp_value_mulhu;
    sum[17] <== (sum[16] + temp_mulhu);
  

    // Instruction VIRTUAL_ADVICE
    instruction_index += 1;
   
    signal filtered_operands_virtual_advice[instruction_to_memory_indices_len[instruction_index]];
    for (var i = 0; i < instruction_to_memory_indices_len[instruction_index]; i++) {
        filtered_operands_virtual_advice[i] <== e_poly_openings[InstructionToMemoryIndices(instruction_index ,i)];
    }

    signal temp_value_virtual_advice <==  CombineLookupsVirtualAdvice(instruction_to_memory_indices_len[instruction_index], M)(filtered_operands_virtual_advice);
    signal temp_virtual_advice <==  flag_openings[instruction_index] * temp_value_virtual_advice;
    sum[18] <== (sum[17] + temp_virtual_advice);
  


    // Instruction VIRTUAL_MOVE
    instruction_index += 1;
  
    signal filtered_operands_virtual_move[instruction_to_memory_indices_len[instruction_index]];
    for (var i = 0; i < instruction_to_memory_indices_len[instruction_index]; i++) {
        filtered_operands_virtual_move[i] <== e_poly_openings[InstructionToMemoryIndices(instruction_index ,i)];
    }

    signal temp_value_virtual_move <==  CombineLookupsVirtualMove(instruction_to_memory_indices_len[instruction_index], M)(filtered_operands_virtual_move);
    signal temp_virtual_move <==  flag_openings[instruction_index] * temp_value_virtual_move;
    sum[19] <== (sum[18] + temp_virtual_move);
  

    // Instruction VIRTUAL_ASSERT_LTE
    instruction_index += 1;

    signal filtered_operands_virtual_assert_lte[instruction_to_memory_indices_len[instruction_index]];
    for (var i = 0; i < instruction_to_memory_indices_len[instruction_index]; i++) {
        filtered_operands_virtual_assert_lte[i] <== e_poly_openings[InstructionToMemoryIndices(instruction_index ,i)];
    }

    signal temp_value_virtual_assert_lte <==  CombineLookupsVirtualAssertLte(C, M)(filtered_operands_virtual_assert_lte);
    signal temp_virtual_assert_lte <== flag_openings[instruction_index] * temp_value_virtual_assert_lte;
    sum[20] <== (sum[19] + temp_virtual_assert_lte);
  

    // Instruction VIRTUAL_ASSERT_VALID_SIGNED_REMAINDER
    instruction_index += 1;
  
    signal filtered_operands_virtual_assert_valid_signed_remainder[instruction_to_memory_indices_len[instruction_index]];
    for (var i = 0; i < instruction_to_memory_indices_len[instruction_index]; i++) {
        filtered_operands_virtual_assert_valid_signed_remainder[i] <== e_poly_openings[InstructionToMemoryIndices(instruction_index ,i)];
    }

    signal temp_value_virtual_assert_valid_signed_remainder <== CombineLookupsVirtualAssertValidSignedRemainder(C, M)(filtered_operands_virtual_assert_valid_signed_remainder);
    signal temp_virtual_assert_valid_signed_remainder <==  flag_openings[instruction_index] *  temp_value_virtual_assert_valid_signed_remainder;
    sum[21] <== (sum[20] + temp_virtual_assert_valid_signed_remainder);
  


    // Instruction VIRTUAL_ASSERT_VALID_UNSIGNED_REMAINDER
    instruction_index += 1;
    signal filtered_operands_virtual_assert_valid_unsigned_remainder[instruction_to_memory_indices_len[instruction_index]];
    for (var i = 0; i < instruction_to_memory_indices_len[instruction_index]; i++) {
        filtered_operands_virtual_assert_valid_unsigned_remainder[i] <== e_poly_openings[InstructionToMemoryIndices(instruction_index ,i)];
    }

    signal temp_value_virtual_assert_valid_unsigned_remainder <== CombineLookupsVirtualAssertValidUnsignedRemainder(C ,M)(filtered_operands_virtual_assert_valid_unsigned_remainder);
    signal temp_virtual_assert_valid_unsigned_remainder <== flag_openings[instruction_index] * temp_value_virtual_assert_valid_unsigned_remainder ;
    sum[22] <== (sum[21] + temp_virtual_assert_valid_unsigned_remainder);
  

    // Instruction VIRTUAL_ASSERT_VALID_DIV0
    instruction_index += 1;
  
    signal filtered_operands_virtual_assert_valid_div0[instruction_to_memory_indices_len[instruction_index]];
    for (var i = 0; i < instruction_to_memory_indices_len[instruction_index]; i++) {
        filtered_operands_virtual_assert_valid_div0[i] <== e_poly_openings[InstructionToMemoryIndices(instruction_index ,i)];
    }

    signal temp_value_virtual_assert_valid_div0 <== CombineLookupsVirtualAssertValidDiv0(C, M)(filtered_operands_virtual_assert_valid_div0);
    signal temp_virtual_assert_valid_div0 <== flag_openings[instruction_index] *  temp_value_virtual_assert_valid_div0;
    sum[23] <== (sum[22] + temp_virtual_assert_valid_div0);
  


    // Instruction VIRTUAL_ASSERT_HALFWORD_ALIGNMENT
    instruction_index += 1;
   
    signal filtered_operands_virtual_assert_halfword_alignment[instruction_to_memory_indices_len[instruction_index]];
    for (var i = 0; i < instruction_to_memory_indices_len[instruction_index]; i++) {
        filtered_operands_virtual_assert_halfword_alignment[i] <==e_poly_openings[InstructionToMemoryIndices(instruction_index ,i)];
    }

    signal temp_value_virtual_assert_halfword_alignment <== CombineLookupsAlignedMemoryAccess(instruction_to_memory_indices_len[instruction_index], 2)(filtered_operands_virtual_assert_halfword_alignment);
    signal temp_virtual_assert_halfword_alignment <==  flag_openings[instruction_index] * temp_value_virtual_assert_halfword_alignment ;
    sum[24] <== (sum[23] + temp_virtual_assert_halfword_alignment);
  


    // Instruction VIRTUAL_ASSERT_WORD_ALIGNMENT
    instruction_index += 1;
    signal filtered_operands_virtual_assert_word_alignment[instruction_to_memory_indices_len[instruction_index]];
    for (var i = 0; i < instruction_to_memory_indices_len[instruction_index]; i++) {
        filtered_operands_virtual_assert_word_alignment[i] <== e_poly_openings[InstructionToMemoryIndices(instruction_index ,i)];
    }

    signal temp_value_virtual_assert_word_alignment <==  CombineLookupsAlignedMemoryAccess(instruction_to_memory_indices_len[instruction_index], 4)(filtered_operands_virtual_assert_word_alignment);
    signal temp_virtual_assert_word_alignment <== flag_openings[instruction_index] * temp_value_virtual_assert_word_alignment;
    sum[25] <== (sum[24] + temp_virtual_assert_word_alignment);
    result <== sum[25];

}






function InstructionToMemoryIndices(outer_idx, inner_idx) { 

    var arrays[26][18] =  [[34, 35, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [34, 35, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [20, 21, 22, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [36, 37, 38, 39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [9, 10, 17, 18, 19, 6, 7, 15, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0], [16, 17, 18, 19, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [9, 10, 17, 18, 19, 6, 7, 15, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0], [16, 17, 18, 19, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [28, 27, 26, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [33, 32, 31, 30, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [33, 32, 31, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [24, 11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [34, 35, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [34, 35, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [11, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [34, 35, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [16, 17, 18, 19, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [9, 10, 6, 7, 8, 17, 18, 19, 4, 15, 40, 41, 42, 43, 44, 45, 46, 47], [16, 17, 18, 19, 5, 6, 7, 44, 45, 46, 47, 0, 0, 0, 0, 0, 0, 0], [40, 41, 42, 43, 48, 49, 50, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [52, 53, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]];
    var result = arrays[outer_idx][inner_idx];
   
    return result;

}

function SubtableToMemoryIndices(outer_idx, inner_idx) { 

    var arrays[26][4] = [[0, 1, 2, 3], [4, 0, 0, 0], [5, 6, 7, 8], [9, 0, 0, 0], [10, 0, 0, 0], [11, 12, 13, 14], [15, 0, 0, 0], [16, 17, 18, 19], [20, 21, 22, 23], [24, 0, 0, 0], [25, 0, 0, 0], [26, 0, 0, 0], [27, 0, 0, 0], [28, 0, 0, 0], [29, 0, 0, 0], [30, 0, 0, 0], [31, 0, 0, 0], [32, 0, 0, 0], [33, 0, 0, 0], [34, 35, 0, 0], [36, 37, 38, 39], [40, 41, 42, 43], [44, 45, 46, 47], [48, 49, 50, 51], [52, 0, 0, 0], [53, 0, 0, 0]];
  
   var result = arrays[outer_idx][inner_idx];

    return result;

}



function MemorytoDimensionindex(inner_idx) { 
    var arrays[54] = [0, 1, 2, 3, 0, 0, 1, 2, 3, 0, 0, 0, 1, 2, 3, 0, 0, 1, 2, 3, 0, 1, 2, 3, 2, 3, 2, 1, 0, 0, 3, 2, 1, 0, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 3, 3];
    var result = arrays[inner_idx];
    return result;

}

function MemoryToSubtableIndex(inner_idx) { 
    var arrays[54] =  [0, 0, 0, 0, 1, 2, 2, 2, 2, 3, 4, 5, 5, 5, 5, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 24, 25];
    var result = arrays[inner_idx];
    return result;

}
