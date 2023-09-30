pragma circom 2.1.6;

/*  Constants: written as functions because circom.
*/
function N_CHUNKS() {return 6;}
function L_CHUNK() {return 11;}
function L_MS_CHUNK() {return 9;}
function N_FLAGS() {return 15;}

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
    signal input input_state[2];
    signal output output_state[2];

    /* Per-step witnesses
    */
    signal input op_code;
    signal input rs1;
    signal input rs2;
    signal input rd;
    signal input immediate;
    signal input op_flags_packed;
    signal input code_read_t;
    signal input op_flag[N_FLAGS()]; // just advice, not read from code

    signal input rs1_val;
    signal input rs1_read_ts;
    signal input rs2_val;
    signal input rs2_read_ts;

    signal input mem_read_val; // advice for loads; asserted for stores
    signal input mem_read_ts; // for both loads and stores

    signal input lookup_output;
    signal input instr_lookup_query;

    /* Among x, y and z: either (x, y) are both used OR only z is used.
    chunks_lookup are derived using chunks_{(x,y)/z}
    */
    signal input chunks_x[N_CHUNKS()];
    signal input chunks_y[N_CHUNKS()];
    signal input chunks_z[N_CHUNKS()];
    signal input chunks_lookup[N_CHUNKS()];

    /* op_flags: 
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


    /* Assigning operands x and y
    */
    signal x <== if_else()([op_flag[0], rs1_val, input_state[1]]);

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
    signal input op_code[N];
    signal input rs1[N];
    signal input rs2[N];
    signal input rd[N];
    signal input immediate[N];
    signal input op_flags_packed[N];
    signal input code_read_t[N];
    signal input rs1_val[N];
    signal input rs1_read_ts[N];
    signal input rs2_val[N];
    signal input rs2_read_ts[N];
    signal input mem_read_val[N];
    signal input mem_read_ts[N];
    signal input lookup_output[N];
    signal input instr_lookup_query[N];
    signal input op_flag[N][N_FLAGS()];
    signal input chunks_x[N][N_CHUNKS()];
    signal input chunks_y[N][N_CHUNKS()];
    signal input chunks_z[N][N_CHUNKS()];
    signal input chunks_lookup[N][N_CHUNKS()];

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

        jolt_steps[i].op_code <== op_code[i];
        jolt_steps[i].rs1 <== rs1[i];
        jolt_steps[i].rs2 <== rs2[i];
        jolt_steps[i].rd <== rd[i];
        jolt_steps[i].immediate <== immediate[i];
        jolt_steps[i].op_flags_packed <== op_flags_packed[i];
        jolt_steps[i].code_read_t <== code_read_t[i];
        jolt_steps[i].rs1_val <== rs1_val[i];
        jolt_steps[i].rs1_read_ts <== rs1_read_ts[i];
        jolt_steps[i].rs2_val <== rs2_val[i];
        jolt_steps[i].rs2_read_ts <== rs2_read_ts[i];
        jolt_steps[i].mem_read_val <== mem_read_val[i];
        jolt_steps[i].mem_read_ts <== mem_read_ts[i];
        jolt_steps[i].lookup_output <== lookup_output[i];
        jolt_steps[i].instr_lookup_query <== instr_lookup_query[i];
        jolt_steps[i].op_flag <== op_flag[i];
        jolt_steps[i].chunks_x <== chunks_x[i];
        jolt_steps[i].chunks_y <== chunks_y[i];
        jolt_steps[i].chunks_z <== chunks_z[i];
        jolt_steps[i].chunks_lookup <== chunks_lookup[i];
    }

    out <== jolt_steps[N-1].output_state;
}

template JoltMain(N) {
    var INPUT_LEN = 15 * N + N * N_FLAGS() + 4 * N * N_CHUNKS(); // 54 for N=1, NC=6, NF=15
    signal input in[INPUT_LEN];

    component jolt_loop; 
    jolt_loop = JoltLoop(N); 
    jolt_loop.op_code <== subarray(0, N, INPUT_LEN)(in);
    jolt_loop.rs1 <== subarray(N, N, INPUT_LEN)(in);
    jolt_loop.rs2 <== subarray(2 * N, N, INPUT_LEN)(in);
    jolt_loop.rd <== subarray(3 * N, N, INPUT_LEN)(in);
    jolt_loop.immediate <== subarray(4 * N, N, INPUT_LEN)(in);
    jolt_loop.op_flags_packed <== subarray(5 * N, N, INPUT_LEN)(in);
    jolt_loop.code_read_t <== subarray(6 * N, N, INPUT_LEN)(in);
    jolt_loop.rs1_val <== subarray(7 * N, N, INPUT_LEN)(in);
    jolt_loop.rs1_read_ts <== subarray(8 * N, N, INPUT_LEN)(in);
    jolt_loop.rs2_val <== subarray(9 * N, N, INPUT_LEN)(in);
    jolt_loop.rs2_read_ts <== subarray(10 * N, N, INPUT_LEN)(in);
    jolt_loop.mem_read_val <== subarray(11 * N, N, INPUT_LEN)(in);
    jolt_loop.mem_read_ts <== subarray(12 * N, N, INPUT_LEN)(in);
    jolt_loop.lookup_output <== subarray(13 * N, N, INPUT_LEN)(in);
    jolt_loop.instr_lookup_query <== subarray(14 * N, N, INPUT_LEN)(in);

    jolt_loop.op_flag <== submatrix(15 * N, N , N_FLAGS(), INPUT_LEN)(in);

    var CHUNKS_START_IDX = 15 * N + N * N_FLAGS();

    jolt_loop.chunks_x <== submatrix(CHUNKS_START_IDX, N , N_CHUNKS(), INPUT_LEN)(in);
    jolt_loop.chunks_y <== submatrix(CHUNKS_START_IDX + N * N_CHUNKS(), N , N_CHUNKS(), INPUT_LEN)(in);
    jolt_loop.chunks_z <== submatrix(CHUNKS_START_IDX + 2 * N * N_CHUNKS(), N , N_CHUNKS(), INPUT_LEN)(in);
    jolt_loop.chunks_lookup <== submatrix(CHUNKS_START_IDX + 3 * N * N_CHUNKS(), N , N_CHUNKS(), INPUT_LEN)(in);

}

component main {public [in]} 
    = JoltMain(1);
