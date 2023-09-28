pragma circom 2.1.6;

include "node_modules/circomlib/circuits/poseidon.circom";

// function is_load_instr() {return 2;}
// function is_store_instr() {return 3;}
// function is_jump_instr() {return 4;}
// function is_branch_instr() {return 5;}
// function if_update_rd() {return 6;}
// function is_add_instr() {return 7;}
// function is_sub_instr() {return 8;}
// function is_mul_instr() {return 9;}
// function is_advice_instr() {return 10;}
// function is_assert_false_instr() {return 11;}
// function is_assert_true_instr() {return 12;}
// function sign_imm() {return 13;}
// function is_lui() {return 14;}

// function N_CHUNKS() {return 6;}
// function L_CHUNK() {return 11;}

// function N_FLAGS() {return 15;}

// template subarray(S, L, N) {
//     signal input in[N];
//     signal output out[L];

//     for (var i=S; i<S+L; i++) {
//         out[i-S] <== in[i];
//     }
// }

// template submatrix(S, R, C, N) {
//     signal input in[N];
//     signal output out[R][C];

//     for (var r=0; r<R; r++) {
//         for (var c=0; c<C; c++) {
//             out[r][c] <== in[S + r * R + c];
//         }
//     }
// }



// template if_else() {
//     signal input in[3]; 
//     signal output out; 

//     signal zero_for_a <== in[0];
//     signal a <== in[1];
//     signal b <== in[2];

//     signal _please_help_me_god <== (1-zero_for_a) * a;
//     out <== _please_help_me_god + (zero_for_a) * b;
// }

// template combine_chunks(N, L) {
//     signal input in[N];
//     signal output out;

//     signal combine[N];
//     for (var i=0; i<N; i++) {
//         if (i==0) {
//             combine[i] <== in[0];
//         } else {
//             combine[i] <== combine[i-1] + 2**(i*L) * in[i];
//         }
//     }
    
//     out <== in[N-1];
// }

// template jolt_step() {
//     signal input input_state[2];
//     signal output output_state[2];

//     signal input op_code;
//     signal input rs1;
//     signal input rs2;
//     signal input rd;
//     signal input immediate;
//     signal input op_flags_packed;
//     signal input code_read_t;

//     signal input rs1_val;
//     signal input rs1_read_ts;
//     signal input rs2_val;
//     signal input rs2_read_ts;

//     // Memory read from RAM 
//     signal input mem_read_val;
//     signal input mem_read_ts;

//     signal input lookup_output;
//     signal input instr_lookup_query;

//     // Not related to memory-checks
//     signal input op_flag[N_FLAGS()];

//     /* Among x, y and z: either (x, y) are both used OR only z is used.
//     chunks_lookup are derived using chunks_{(x,y)/z}
//     */
//     signal input chunks_x[N_CHUNKS()];
//     signal input chunks_y[N_CHUNKS()];
//     signal input chunks_z[N_CHUNKS()];
//     signal input chunks_lookup[N_CHUNKS()];

//     /* Actually starting the core step 
//     */
//     signal is_load_instr <== op_flag[2];
//     signal is_store_instr <== op_flag[3];
//     signal is_jump_instr <== op_flag[4];
//     signal is_branch_instr <== op_flag[5];
//     signal if_update_rd <== op_flag[6];
//     signal is_add_instr <== op_flag[7];
//     signal is_sub_instr <== op_flag[8];
//     signal is_mul_instr <== op_flag[9];
//     signal is_advice_instr <== op_flag[10]; // these use a dummy lookup where output is operand2
//     signal is_assert_false_instr <== op_flag[11];
//     signal is_assert_true_instr <== op_flag[12];
//     signal sign_imm <== op_flag[13];
//     signal is_lui <== op_flag[14];

//     // The setting of operand1

//     // component choose_op1 = if_else();
//     // choose_op1.inp <== if_else()([op_flag[0], val_rs1, input_state[1]]);
//     // signal operand1 <== choose_op1.out; 
//     signal operand1 <== if_else()([op_flag[0], rs1_val, input_state[1]]);

//     // The setting of operand2

//     // component __choose_op2 = if_else();
//     // __choose_op2.inp <== [op_flag[1], rs2_val, immediate];
//     // signal __operand2 <== __choose_op2.out; 
//     signal __operand2 <== if_else()([op_flag[1], rs2_val, immediate]);

//     // component _choose_op2 = if_else();
//     // _choose_op2.inp <== [is_advice_instr, lookup_output, __operand2];
//     // signal _operand2 <== _choose_op2.out; 
//     signal _operand2 <== if_else()([is_advice_instr, lookup_output, __operand2]);

//     // component choose_op2 = if_else();
//     // choose_op2.inp <== [is_load_instr, mem_read_val, _operand2];
//     // signal operand2 <== choose_op2.out; 
//     signal operand2 <== if_else()([is_load_instr, mem_read_val, _operand2]);

//     component choose_write_v = if_else();
//     choose_write_v.in <== [is_store_instr, lookup_output, mem_read_val];
//     signal write_v <== choose_write_v.out;


//     /*
//         Create the lookup query 
//     */

//     signal _is_concat <== (1-is_add_instr) * (1-is_sub_instr);
//     signal is_concat <== _is_concat * (1-is_mul_instr);

//     signal z_concat <== operand1 * (2**64) + operand2;
//     signal z_add <== operand1 + operand2;
//     signal z_mul <== operand1 * operand2;

//     signal __z <== is_concat * z_concat;
//     signal _z <== __z + is_add_instr * z_add;
//     signal z <== _z + is_mul_instr * z_mul;

//     // verify chunks_x
//     component combine_chunks_x = combine_chunks(N_CHUNKS(), L_CHUNK());
//     combine_chunks_x.in <== chunks_x;
//     signal combined_x <== combine_chunks_x.out;

//     assert((combined_x-operand1) * is_concat == 0);

//     // verify chunks_y 
//     component combine_chunks_y = combine_chunks(N_CHUNKS(), L_CHUNK());
//     combine_chunks_y.in <== chunks_y;
//     signal combined_y <== combine_chunks_y.out;

//     assert((combined_y-operand2) * is_concat == 0);

//     // verify chunks of z -- doesn't matter if concat, though
//     component combine_chunks_z = combine_chunks(N_CHUNKS(), L_CHUNK());
//     combine_chunks_z.in <== chunks_z;
//     signal combined_z <== combine_chunks_z.out;

//     assert((combined_z-z) * (1-is_concat) == 0);

//     // verify that chunks_i == chunks_x || chunks_y -- only for concat
//     for (var i=0; i<N_CHUNKS(); i++) {
//       assert(chunks_lookup[i] - (chunks_x[i] + chunks_y[i] * 2**L_CHUNK()) * is_concat == 0);
//     } 

//     var idx_msc = N_CHUNKS()-1;
//     assert(chunks_lookup[idx_msc] - (chunks_x[idx_msc] + chunks_y[idx_msc] * 2**L_CHUNK()) * is_concat == 0);

//     // resume after line 407 
// }

// template JoltLoop(N) {
//     signal input op_code[N];
//     signal input rs1[N];
//     signal input rs2[N];
//     signal input rd[N];
//     signal input immediate[N];
//     signal input op_flags_packed[N];
//     signal input code_read_t[N];
//     signal input rs1_val[N];
//     signal input rs1_read_ts[N];
//     signal input rs2_val[N];
//     signal input rs2_read_ts[N];
//     signal input mem_read_val[N];
//     signal input mem_read_ts[N];
//     signal input lookup_output[N];
//     signal input instr_lookup_query[N];
//     signal input op_flag[N][N_FLAGS()];
//     signal input chunks_x[N][N_CHUNKS()];
//     signal input chunks_y[N][N_CHUNKS()];
//     signal input chunks_z[N][N_CHUNKS()];
//     signal input chunks_lookup[N][N_CHUNKS()];

//     signal output out[2]; 

//     /* Variables used in the for loop
//     */
//     component jolt_steps[N];

//     for (var i=0; i<N; i++) {
//         jolt_steps[i] = jolt_step();

//         if (i==0) {
//             jolt_steps[i].input_state <== [0, 0];
//         } else {
//             jolt_steps[i].input_state <== jolt_steps[i-1].output_state;
//         }

//         jolt_steps[i].op_code <== op_code[i];
//         jolt_steps[i].rs1 <== rs1[i];
//         jolt_steps[i].rs2 <== rs2[i];
//         jolt_steps[i].rd <== rd[i];
//         jolt_steps[i].immediate <== immediate[i];
//         jolt_steps[i].op_flags_packed <== op_flags_packed[i];
//         jolt_steps[i].code_read_t <== code_read_t[i];
//         jolt_steps[i].rs1_val <== rs1_val[i];
//         jolt_steps[i].rs1_read_ts <== rs1_read_ts[i];
//         jolt_steps[i].rs2_val <== rs2_val[i];
//         jolt_steps[i].rs2_read_ts <== rs2_read_ts[i];
//         jolt_steps[i].mem_read_val <== mem_read_val[i];
//         jolt_steps[i].mem_read_ts <== mem_read_ts[i];
//         jolt_steps[i].lookup_output <== lookup_output[i];
//         jolt_steps[i].instr_lookup_query <== instr_lookup_query[i];
//         jolt_steps[i].op_flag <== op_flag[i];
//         jolt_steps[i].chunks_x <== chunks_x[i];
//         jolt_steps[i].chunks_y <== chunks_y[i];
//         jolt_steps[i].chunks_z <== chunks_z[i];
//         jolt_steps[i].chunks_lookup <== chunks_lookup[i];
//     }

//     out <== jolt_steps[N-1].output_state;
// }

// template jolt_main(N) {
//     var INPUT_LEN = 15 * N + N * N_FLAGS() + 4 * N * N_CHUNKS();
//     signal input in[INPUT_LEN];

//     component jolt_loop = JoltLoop(N); 
//     jolt_loop.op_code <== subarray(0, N, INPUT_LEN)(in);
//     jolt_loop.rs1 <== subarray(N, N, INPUT_LEN)(in);
//     jolt_loop.rs2 <== subarray(2 * N, N, INPUT_LEN)(in);
//     jolt_loop.rd <== subarray(3 * N, N, INPUT_LEN)(in);
//     jolt_loop.immediate <== subarray(4 * N, N, INPUT_LEN)(in);
//     jolt_loop.op_flags_packed <== subarray(5 * N, N, INPUT_LEN)(in);
//     jolt_loop.code_read_t <== subarray(6 * N, N, INPUT_LEN)(in);
//     jolt_loop.rs1_val <== subarray(7 * N, N, INPUT_LEN)(in);
//     jolt_loop.rs1_read_ts <== subarray(8 * N, N, INPUT_LEN)(in);
//     jolt_loop.rs2_val <== subarray(9 * N, N, INPUT_LEN)(in);
//     jolt_loop.rs2_read_ts <== subarray(10 * N, N, INPUT_LEN)(in);
//     jolt_loop.mem_read_val <== subarray(11 * N, N, INPUT_LEN)(in);
//     jolt_loop.mem_read_ts <== subarray(12 * N, N, INPUT_LEN)(in);
//     jolt_loop.lookup_output <== subarray(13 * N, N, INPUT_LEN)(in);
//     jolt_loop.instr_lookup_query <== subarray(14 * N, N, INPUT_LEN)(in);

//     jolt_loop.op_flag <== submatrix(15 * N, N , N_FLAGS(), INPUT_LEN)(in);

//     var CHUNKS_START_IDX = 15 * N + N * N_FLAGS();

//     jolt_loop.chunks_x <== subarray(CHUNKS_START_IDX, N , N_CHUNKS(), INPUT_LEN);
//     jolt_loop.chunks_y <== subarray(CHUNKS_START_IDX + N * N_CHUNKS(), N , N_CHUNKS(), INPUT_LEN);
//     jolt_loop.chunks_z <== subarray(CHUNKS_START_IDX + 2 * N * N_CHUNKS(), N , N_CHUNKS(), INPUT_LEN);
//     jolt_loop.chunks_lookup <== subarray(CHUNKS_START_IDX + 3 * N * N_CHUNKS(), N , N_CHUNKS(), INPUT_LEN);

// }

template test_main() {
    signal input step_in[2];
    signal input in1;
    signal input in2;
    signal output step_out[2];

    in1 === in2;

    signal in[2] <== [in1, in2];

    step_out[0] <== in[0] + step_in[1];
    step_out[1] <== in[0] * step_in[0];
}

// template test_main() {
//     signal input in[2];
//     signal output out;
//     // signal output out[2];

//     // out[0] <== arg_in[0] + arg_in[1];
//     // out[1] <== arg_in[0] * arg_in[1];

//     component pos = Poseidon(2);
//     pos.inputs <== in;
//     out <== pos.out;
// }

component main {public [step_in, in1, in2]} 
    = test_main();
