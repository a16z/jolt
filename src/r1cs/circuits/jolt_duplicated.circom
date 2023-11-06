pragma circom 2.1.6;

/*  Constants: written as functions because circom.
*/
function N_CHUNKS() {return 6;}
function L_CHUNK() {return 11;}
function L_MS_CHUNK() {return 9;}
function N_FLAGS() {return 15;}

function WIT_PER_STEP() {return 6 * 3 + 1 + 2 * 3 + N_FLAGS() + 8 * 3 + 1 + 2 + 4 * N_CHUNKS();} // 54 for N=1, NC=6, NF=15

function ALL_ONES() {return 0x1111111111111111;} 

// state: [field; 2] = [step_num, pc]
function STEP_NUM_IDX() {return 0;}
function PC_IDX() {return 1;}

// Utility functions
template subarray(S, L, N) {
    signal input in[N];
    signal output out[L];

    for (var i=S; i<S+L; i++) {
        out[i-S] <== in[i];
    }
}

template submatrix(S, R, C, N) {
    signal input in[N];
    signal output out[R][C];

    for (var r=0; r<R; r++) {
        for (var c=0; c<C; c++) {
            out[r][c] <== in[S + r * R + c];
        }
    }
}

template if_else() {
    signal input in[3]; 
    signal output out; 

    signal zero_for_a <== in[0];
    signal a <== in[1];
    signal b <== in[2];

    signal _please_help_me_god <== (1-zero_for_a) * a;
    out <== _please_help_me_god + (zero_for_a) * b;
}

template combine_chunks(N, L) {
    signal input in[N];
    signal output out;

    signal combine[N];
    for (var i=0; i<N; i++) {
        if (i==0) {
            combine[i] <== in[0];
        } else {
            combine[i] <== combine[i-1] + 2**(i*L) * in[i];
        }
    }
    
    out <== in[N-1];
}

// One CPU step of jolt 
template JoltStep() {
    /*  Input and Output states 
    */
    signal input input_state[2];
    signal output output_state[2];

    signal input witnesses[WIT_PER_STEP()];

    /* The rest of this function has three broad components: 
    1. Parse the witness to get the individual elements 
    2. Write out the constraints
    */

    /*********************** 1. PARSE WITNESSES *******************************/
    /* These Per-step witnesses
    */

    signal prog_read_avt[6][3];
    signal op_flags_packed;
    signal op_flag[N_FLAGS()]; // just advice, not read from code

    /* Register reading witnesses 
    */
    signal rs1_read_avt[3];
    signal rs2_read_avt[3];


    signal mem_read_write_avt[8][3];
    signal mem_read_val; // advice for loads; asserted for stores

    signal lookup_output;
    signal instr_lookup_query;

    /* Among x, y and z: either (x, y) are both used OR only z is used.
    chunks_lookup are derived using chunks_{(x,y)/z}
    */
    signal chunks_x[N_CHUNKS()];
    signal chunks_y[N_CHUNKS()];
    signal chunks_z[N_CHUNKS()];
    signal chunks_lookup[N_CHUNKS()];

    /* Parse them
    */

    var running_idx = 0;
    prog_read_avt <== submatrix(running_idx, 6, 3, WIT_PER_STEP())(witnesses);
    running_idx += 6 * 3;

    op_flags_packed <== witnesses[running_idx];
    running_idx += 1;

    rs1_read_avt <== subarray(running_idx, 3, WIT_PER_STEP())(witnesses);
    running_idx += 3;
    rs2_read_avt <== subarray(running_idx, 3, WIT_PER_STEP())(witnesses);
    running_idx += 3;

    mem_read_write_avt <== submatrix(running_idx, 8, 3, WIT_PER_STEP())(witnesses);
    running_idx += 8 * 3;
    mem_read_val <== witnesses[running_idx];
    running_idx += 1;

    lookup_output <== witnesses[running_idx];
    running_idx += 1;
    instr_lookup_query <== witnesses[running_idx];
    running_idx += 1;

    op_flag <== subarray(running_idx, N_FLAGS(), WIT_PER_STEP())(witnesses);
    running_idx += N_FLAGS();

    chunks_x <== subarray(running_idx, N_CHUNKS(), WIT_PER_STEP())(witnesses);
    running_idx += N_CHUNKS();

    chunks_y <== subarray(running_idx, N_CHUNKS(), WIT_PER_STEP())(witnesses);
    running_idx += N_CHUNKS();

    chunks_z <== subarray(running_idx, N_CHUNKS(), WIT_PER_STEP())(witnesses);
    running_idx += N_CHUNKS();

    chunks_lookup <== subarray(running_idx, N_CHUNKS(), WIT_PER_STEP())(witnesses);

    /*********************** 2. CONSTRAINTS *******************************/

    /* Parse Input State: 
    */
    signal step_num <== input_state[STEP_NUM_IDX()];
    signal PC <== input_state[PC_IDX()];
    
    /* Reading program code: 
        1. They all have the same address
        2. Parse
    */
    for (var i=0; i<6; i++) {
        prog_read_avt[i][0] === PC;
    }

    signal op_code <== prog_read_avt[0][1];
    signal rs1 <== prog_read_avt[1][1];
    signal rs2 <== prog_read_avt[2][1];
    signal rd <== prog_read_avt[3][1];
    signal immediate <== prog_read_avt[4][1];
    signal op_flags <== prog_read_avt[5][1];

    /* Constraints on op_flags: 
        1. Verify they combine to form op_flags_packed
        2. Parse
    */
    signal flags_combined <== combine_chunks(N_FLAGS(), 1)(op_flag);
    op_flags_packed === flags_combined;
    
    signal is_load_instr <== op_flag[2];
    signal is_store_instr <== op_flag[3];
    signal is_jump_instr <== op_flag[4];
    signal is_branch_instr <== op_flag[5];
    signal if_update_rd <== op_flag[6];
    signal is_add_instr <== op_flag[7];
    signal is_sub_instr <== op_flag[8];
    signal is_mul_instr <== op_flag[9];
    signal is_advice_instr <== op_flag[10]; // these use a dummy lookup where output is y
    signal is_assert_false_instr <== op_flag[11];
    signal is_assert_true_instr <== op_flag[12];
    signal sign_imm <== op_flag[13];
    signal is_lui <== op_flag[14];

    /* Constraints on reading registers 
    */
    rs1_read_avt[0] === rs1; 
    rs2_read_avt[0] === rs2; 

    signal rs1_val <== rs1_read_avt[1];
    signal rs2_val <== rs2_read_avt[1];

    /* Assigning operands x and y
    */
    signal x <== if_else()([op_flag[0], rs1_val, PC]);

    signal __y <== if_else()([op_flag[1], rs2_val, immediate]);
    signal _y <== if_else()([is_advice_instr, lookup_output, __y]);
    signal y <== if_else()([is_load_instr, mem_read_val, _y]);

    /* Loads and stores
    */
    signal write_v <== if_else()([is_store_instr, lookup_output, mem_read_val]);

    /* Create the lookup query 
        - First, obtain z.
        - Then verify that the chunks of x, y, z are correct. 
    */

    signal _is_concat <== (1-is_add_instr) * (1-is_sub_instr);
    signal is_concat <== _is_concat * (1-is_mul_instr);

    signal z_concat <== x * (2**64) + y;
    signal z_add <== x + y;
    signal z_jump <== z_add + 4;
    signal z_sub <== x + (ALL_ONES() - y + 1);
    signal z_mul <== x * y;

    signal z__5 <== is_concat * z_concat;
    signal z__4 <== z__5 + is_add_instr * z_add;
    signal z__3 <== z__4 + is_jump_instr * z_jump;
    signal z__2 <== z__3 + is_sub_instr * z_sub;
    signal z__1 <== z__2 + is_mul_instr * z_mul;
    signal z <== z__1;

    // verify chunks_x
    signal combined_x_chunks <== combine_chunks(N_CHUNKS(), L_CHUNK())(chunks_x);
    assert((combined_x_chunks - x) * is_concat == 0);

    // verify chunks_y 
    signal combined_y_chunks <== combine_chunks(N_CHUNKS(), L_CHUNK())(chunks_y);
    assert((combined_y_chunks-y) * is_concat == 0);

    // verify chunks of z -- doesn't matter if concat, though
    signal combined_z_chunks <== combine_chunks(N_CHUNKS(), L_CHUNK())(chunks_z);
    assert((combined_z_chunks-z) * (1-is_concat) == 0);

    // verify that chunks_i == chunks_x || chunks_y -- only for concat
    for (var i=0; i<N_CHUNKS(); i++) {
      assert(chunks_lookup[i] - (chunks_x[i] + chunks_y[i] * 2**L_CHUNK()) * is_concat == 0);
    } 
    // the most significant chunk has a shorter length!
    var idx_ms_chunk = N_CHUNKS()-1;
    assert(chunks_lookup[idx_ms_chunk] - (chunks_x[idx_ms_chunk] + chunks_y[idx_ms_chunk] * 2**(L_MS_CHUNK())) * is_concat == 0);

    // Get final lookup index (query)
    signal lookup_index <== op_code * (2**128) + z;

    is_assert_false_instr * (1-lookup_output) === 0;
    is_assert_true_instr * lookup_output === 0;

    // lui doesn't need a lookup 
   is_lui * (lookup_output - immediate) === 0;

   /*
        Store into destination register rd 
   */
    signal to_store_in_rd <== if_update_rd * lookup_output;
    signal to_store_in_rd_j <== if_else()([is_jump_instr, lookup_output-4, to_store_in_rd]);

    /*
        Store into output state
    */
    output_state[STEP_NUM_IDX()] <== input_state[STEP_NUM_IDX()]+1;

    // set next PC 
    signal next_pc_j <== if_else()([is_jump_instr, lookup_output-4, input_state[PC_IDX()] + 4]);
    signal next_pc_j_b <== if_else()([is_branch_instr * lookup_output, input_state[PC_IDX()] + sign_imm * immediate, next_pc_j]);
    output_state[PC_IDX()] <== next_pc_j_b;
    
}

template JoltLoop(N) {
    var TOTAL_INPUT_LENGTH = N * WIT_PER_STEP();
    signal input in[TOTAL_INPUT_LENGTH];


    signal output out[2]; 

    /* Variables used in the for loop
    */
    component jolt_steps[N];

    for (var i=0; i<N; i++) {
        jolt_steps[i] = JoltStep();

        if (i==0) {
            jolt_steps[i].input_state <== [0, 0];
        } else {
            jolt_steps[i].input_state <== jolt_steps[i-1].output_state;
        }

        jolt_steps[i].witnesses <== subarray(i * WIT_PER_STEP(), WIT_PER_STEP(), TOTAL_INPUT_LENGTH)(in); 

    }

    out <== jolt_steps[N-1].output_state;
}

template JoltMain(N) {
    signal input in[N * WIT_PER_STEP()];

    component jolt_loop; 
    jolt_loop = JoltLoop(N); 
    jolt_loop.in <== in;

 }

component main {public [in]} 
    = JoltMain(1);
