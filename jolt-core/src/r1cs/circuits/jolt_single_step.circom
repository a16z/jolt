pragma circom 2.1.6;

/* Compiler Variables */
function NUM_STEPS() {return 182;} // NOTE: Fibonacci ignores first 3
function W() {return 32;}
function C() {return 4;}
function PROG_START_ADDR() {return 2147483664;}
function N_FLAGS() {return 18;}
function LOG_M() { return 16; }
/* End of Compiler Variables */

function L_CHUNK() { return LOG_M()/2; } 
// "memreg ops per step" 
function MOPS() {if (W() == 32) {return 7;} else {return 11;}}

/*  Constants: written as functions because circom.
*/

function ALL_ONES() {if (W() == 32) {return 0xffffffff;} else {return 0xffffffffffffffff;} } 

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

// big-endian 
template combine_chunks(N, L) {
    signal input in[N];
    signal output out;

    signal combine[N];
    for (var i=0; i<N; i++) {
        if (i==0) {
            combine[i] <==  (1 << ((N-1-i)*L)) * in[i];
        } else {
            combine[i] <== combine[i-1] + (1 << ((N-1-i)*L)) * in[i];
        }
    }
    
    out <== combine[N-1];
}

template combine_chunks_le(N, L) {
    signal input in[N];
    signal output out;

    signal combine[N];
    for (var i=0; i<N; i++) {
        if (i==0) {
            combine[i] <==  (1 << (i*L)) * in[i];
        } else {
            combine[i] <== combine[i-1] + (1 << (i*L)) * in[i];
        }
    }
    
    out <== combine[N-1];
}

template prodZeroTest(N) {
    signal input in[N];
    signal prod[N];
    signal output out; 

    for (var i=0; i<N; i++) {
        if (i==0) {
            prod[i] <== in[i];
        } else {
            prod[i] <== prod[i-1] * in[i];
        }
    }
    
    0 === prod[N-1];
    out <== 0; 
}

// One CPU step of jolt 
template JoltStep() {
    signal input input_state[2];
    signal output output_state[2];

    /* Per-step inputs. See JoltMain() for information.
    */
    signal input prog_a_rw;

    signal input prog_v_rw[6]; 
    signal opcode <== prog_v_rw[0];
    signal rs1 <== prog_v_rw[1];
    signal rs2 <== prog_v_rw[2];
    signal rd <== prog_v_rw[3];
    signal immediate_before_processing <== prog_v_rw[4];
    signal op_flags_packed <== prog_v_rw[5];

    signal input memreg_a_rw[MOPS()];
    signal input memreg_v_reads[MOPS()];
    signal input memreg_v_writes[MOPS()];

    signal input chunks_x[C()];
    signal input chunks_y[C()];
    signal input chunks_query[C()];

    signal input lookup_output;

    signal input op_flags[N_FLAGS()];

    /* Enforce that prog_a_rw === input_state.pc */
    prog_a_rw === input_state[PC_IDX()];
        
    /* Constraints for op_flags: 
        1. Verify they combine to form op_flags_packed
        2. Parse them.
    */
    signal flags_combined <== combine_chunks(N_FLAGS(), 1)(op_flags);
    op_flags_packed === flags_combined;

    signal is_load_instr <== op_flags[2];
    signal is_store_instr <== op_flags[3];
    signal is_jump_instr <== op_flags[4];
    signal is_branch_instr <== op_flags[5];
    signal if_update_rd_with_lookup_output <== op_flags[6];
    signal is_add_instr <== op_flags[7];
    signal is_sub_instr <== op_flags[8];
    signal is_mul_instr <== op_flags[9];
    signal is_advice_instr <== op_flags[10]; // these use a dummy lookup where output is y
    signal is_assert_false_instr <== op_flags[11];
    signal is_assert_true_instr <== op_flags[12];
    signal sign_imm_flag <== op_flags[13];
    signal is_concat <== op_flags[14];
    signal is_lui_auipc <== op_flags[15];
    signal is_jal <== op_flags[16];
    signal is_shift <== op_flags[17];

    // Pre-processing the imm 
    signal immediate <== if_else()([is_lui_auipc, immediate_before_processing, immediate_before_processing * (2**12)]);
    // signal immediate <== if_else()([is_jal, _immediate, immediate_before_processing * 2]);

    /*******  Register Reading Constraints: 
    Of the 7 (or 11) memory reads, the first 3 are reads from rs1, rs2, rd. 

    1. Ensure that the address of the reads and writes are indeed rs1, rs2, rd.
    2. For memory "reads", the same value is written back, so check that memreg_v_reads[i] === memread_v_writes[i].
    // */
    rs1 === memreg_a_rw[0];
    memreg_v_reads[0] === memreg_v_writes[0];
    signal rs1_val <== memreg_v_reads[0];

    rs2 === memreg_a_rw[1];
    memreg_v_reads[1] === memreg_v_writes[1];
    signal rs2_val <== memreg_v_reads[1];

    rd === memreg_a_rw[2]; // the correctness of the write value will be handled later

    /******* Assigning operands x and y */
    signal x <== if_else()([op_flags[0], rs1_val, prog_a_rw]); // TODO: change this for virtual instructions

    signal _y <== if_else()([op_flags[1], rs2_val, immediate]);
    signal y <== if_else()([1-is_advice_instr, lookup_output, _y]);

    /******* LOAD-STORE CONSTRAINTS */

    /* Take the 4 (or 8) bytes of memory read/written and combine into one 64-bit value.
    */
    signal mem_v_bytes[MOPS()-3] <== subarray(3, MOPS()-3, MOPS())(memreg_v_writes);
    signal load_or_store_value <== combine_chunks_le(MOPS()-3, 8)(mem_v_bytes); 

    /* Verify all 4 (or 8) addresses involved. The starting should be rs1_val + immediate
    */

    signal is_load_store_instr <== is_load_instr + is_store_instr;

    signal immediate_absolute <== if_else()([sign_imm_flag, immediate, ALL_ONES() - immediate + 1]);
    signal sign_of_immediate <== 1-2*sign_imm_flag;
    signal immediate_signed <== sign_of_immediate * immediate_absolute;

    signal _load_store_addr <== rs1_val + immediate_signed;
    signal load_store_addr <== is_load_store_instr * _load_store_addr;

    memreg_a_rw[3] === (is_load_instr + is_store_instr) * load_store_addr; 

    for (var i=1; i<MOPS()-3; i++) {
        // the first three are rs1, rs2, rd so memory starts are index 3
        (memreg_a_rw[3+i] - (memreg_a_rw[3] + i)) *  memreg_a_rw[3+i] === 0; 
    }

    /* As "loads" are memory reads, we ensure that memreg_v_reads[2..10] === memreg_v_writes[2..10]
    */
    for (var i=0; i<MOPS()-3; i++) {
        (memreg_v_reads[3+i] - memreg_v_writes[3+i]) * is_load_instr === 0;
    }
    // NOTE: we will ensure that the loaded value is stored into rd near the end

    /*  For stores, ensure that the value stored is what is rs2.
        As "stores" are memory writes, we do not check memreg_v_reads against memreg_v_writes, as in loads
    */
    is_store_instr * (load_or_store_value - rs2_val) === 0;

    // /******** Constraints for Lookup Query Chunking  */

    /* Create the lookup query 
        - First, obtain z.
        - Then verify that the chunks of x, y, z are correct. 
    */

    // Store the right query format into z
    signal z_concat <== x * (2**W()) + y;
    signal z_add <== x + y;
    signal z_jump <== z_add; // This is superfluous. TODO: change 
    signal z_sub <== x + (ALL_ONES() - y + 1);
    signal z_mul <== x * y;

    signal z__5 <== is_concat * z_concat;
    signal z__4 <== z__5 + is_add_instr * z_add;
    signal z__3 <== z__4 + is_jump_instr * z_jump;
    signal z__2 <== z__3 + is_sub_instr * z_sub;
    signal z__1 <== z__2 + is_mul_instr * z_mul;
    signal z <== z__1;

    // verify chunks_x
    signal combined_x_chunks <== combine_chunks(C(), L_CHUNK())(chunks_x);
    (combined_x_chunks - x) * is_concat === 0;

    // verify chunks_y 
    signal combined_y_chunks <== combine_chunks(C(), L_CHUNK())(chunks_y);
    (combined_y_chunks-y) * is_concat === 0;

    /* Constraints to check correctness of chunks_query 
        If NOT a concat query: chunks_query === chunks_z 
        If its a concat query: then chunks_query === zip(chunks_x, chunks_y)
    */
    signal combined_z_chunks <== combine_chunks(C(), LOG_M())(chunks_query);
    (combined_z_chunks-z) * (1-(is_concat)) === 0;

    // the concat checks: 
    // the most significant chunk has a shorter length!
    signal chunk_y_used[C()]; 
    for (var i=0; i<C(); i++) {
        chunk_y_used[i] <== if_else()([is_shift, chunks_y[i], chunks_y[C()-1]]);
        (chunks_query[i] - (chunk_y_used[i] + chunks_x[i] * 2**L_CHUNK())) * is_concat === 0;
    } 

    // TODO: handle case when C() doesn't divide W() 
    // var idx_ms_chunk = C()-1;
    // (chunks_query[idx_ms_chunk] - (chunks_x[idx_ms_chunk] + chunks_y[idx_ms_chunk] * 2**(L_MS_CHUNK()))) * is_concat === 0;

    // For assert instructions 
    is_assert_false_instr * (1-lookup_output) === 0;
    is_assert_true_instr * lookup_output === 0;

    /* Constraints for storing value in register rd.
    */
    // lui doesn't need a lookup and simply requires the lookup_output to be set to immediate 
    // so it can be stored in the destination register. 

    signal rd_val <== memreg_v_writes[2]; 
    is_load_instr * (rd_val - load_or_store_value) === 0;
    signal _rd_val_test1 <== prodZeroTest(3)([rd, if_update_rd_with_lookup_output, (rd_val - lookup_output)]);
    signal _rd_val_test2 <== prodZeroTest(3)([rd, is_jump_instr, (rd_val - (prog_a_rw+4))]);
    // TODO: LUI - add another flag for lui (again)
    // is_lui * (rd_val - immediate) === 0;

    /* Store into output state
    */
    output_state[STEP_NUM_IDX()] <== input_state[STEP_NUM_IDX()]+1;

    // set next PC 
    signal next_pc_j <== if_else()([
        is_jump_instr,  
        input_state[PC_IDX()] + 4, 
        lookup_output
    ]);
    signal next_pc_j_b <== if_else()([
        is_branch_instr * lookup_output, 
        next_pc_j, 
        input_state[PC_IDX()] + immediate_signed
    ]);
    output_state[PC_IDX()] <== next_pc_j_b;
}

/* Input elements: 

N = number of CPU steps 

Program reads: 
As code is read-only, the same a, v vectors are used for RS and WS. 
The t_writes vector is just the counter so need not be fed in separately. 
Each of the vectors below are 6 * N in length.
1. prog_a_rw 
2. prog_v_rw 
3. prog_t_reads

Memory: Here, we need a separate vector for v_reads and v_writes. 
Each vector is (7 or 11) * N in length. 
4. mem_a_rw
5. mem_v_reads
6. mem_v_writes
7. mem_t_reads

The chunks of the lookups: (if a group below is not involved, just make it all 0)
Each of these vectors is N * C in length. 
8. chunks_x
9. chunks_y
10. chunk_query = x+y, or x*y, or [x_i || y_i]_{i=1}^C

The lookup outputs: 
11. lookup_outputs: N in length

The N_FLAGS op_flags involved in each step
12. op_flags: this should be a vector N_FLAGS * N bits 

*/

component main {public [
        input_state,
        prog_a_rw,
        prog_v_rw,
        memreg_a_rw, 
        memreg_v_reads, 
        memreg_v_writes, 
        chunks_x, 
        chunks_y, 
        chunks_query, 
        lookup_output, 
        op_flags
        ]} 
    = JoltStep();
