pragma circom 2.1.6;

/* Compiler Variables */
function NUM_STEPS() {return 1;}
function W() {return 64;}
function C() {return 6;}
function N_FLAGS() {return 15;}

/*  Constants: written as functions because circom.
*/
function L_CHUNK() {
    if (W() % C() == 0) {
        return W()/C();
    } else {
        return W() / C() + 1;
    }
}
function L_MS_CHUNK() {return W() % L_CHUNK();}
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

    /* Per-step inputs. See JoltMain() for information.
    */
    signal input read_pc;
    signal input opcode;
    signal input rs1;
    signal input rs2;
    signal input rd;
    signal input immediate;
    signal input op_flags_packed;

    signal input memreg_a_rw[11];
    signal input memreg_v_reads[11];
    signal input memreg_v_writes[11];
    signal input memreg_t_reads[11];

    signal input chunks_x[C()];
    signal input chunks_y[C()];
    signal input chunks_query[C()];

    signal input lookup_output;

    signal input op_flags[N_FLAGS()];

    /* Enforce that read_pc === input_state.pc */
    read_pc === input_state[PC_IDX()];
        
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
    signal sign_imm <== op_flags[13];
    signal is_lui <== op_flags[14];

    /*******  Register Reading Constraints: 

    Of the 11 memory reads, the first 2 are reads from rs1, rs2, 
    and the last is the write to rd.

    1. Ensure that the address of the reads and writes are indeed rs1, rs2, rd.
    2. For memory "reads", the same value is written back, so check that memreg_v_reads[i] === memread_v_writes[i].

    // TODO: encode virtual address for memory and registers
    */
    rs1 === memreg_a_rw[0];
    memreg_v_reads[0] === memreg_v_writes[0]; 
    signal rs1_val <== memreg_v_reads[0];

    rs2 === memreg_a_rw[1];
    memreg_v_reads[1] === memreg_v_writes[1];
    signal rs2_val <== memreg_v_reads[1];

    rd === memreg_a_rw[10]; // the correctness of the write value will be handled later

    /******* Assigning operands x and y */
    signal x <== if_else()([op_flags[0], rs1_val, input_state[1]]);

    signal _y <== if_else()([op_flags[1], rs2_val, immediate]);
    signal y <== if_else()([is_advice_instr, lookup_output, _y]);

    /******* LOAD-STORE CONSTRAINTS */

    /* Take the 8 bytes of memory read/written and combine into one 64-bit value.
    */
    signal mem_v_bytes[8] <== subarray(2, 8, 11)(memreg_v_writes);
    signal load_or_store_value <== combine_chunks(8, 8)(mem_v_bytes); 

    /* Verify all 8 addresses involved. The starting should be rs1_val + immediate
    */
    memreg_a_rw[2] === rs1_val + sign_imm * immediate; 
    for (var i=1; i<8; i++) {
        memreg_a_rw[2+i] === memreg_a_rw[2]+i; // the first two are rs1, rs2 so memory starts are index 2
    }

    /* As "loads" are memory reads, we ensure that memreg_v_reads[2..10] === memreg_v_writes[2..10]
    */
    for (var i=0; i<8; i++) {
        (memreg_v_reads[2+i] - memreg_v_writes[2+i]) * is_load_instr === 0;
    }
    // NOTE: we will ensure that the loaded value is stored into rd near the end

    /*  For stores, ensure that the value stored is what is rs2.
        As "stores" are memory writes, we do not check memreg_v_reads against memreg_v_writes, as in loads
    */
    load_or_store_value === rs2_val;

    /******** Constraints for Lookup Query Chunking  */

    /* Create the lookup query 
        - First, obtain z.
        - Then verify that the chunks of x, y, z are correct. 
    */

    // Figure out if it's a concat-style query
    // which it is if it's neither ADD, SUB or MUL
    signal _is_concat <== (1-is_add_instr) * (1-is_sub_instr);
    signal is_concat <== _is_concat * (1-is_mul_instr);

    // Store the right query format into z
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
    signal combined_x_chunks <== combine_chunks(C(), L_CHUNK())(chunks_x);
    assert((combined_x_chunks - x) * is_concat == 0);

    // verify chunks_y 
    signal combined_y_chunks <== combine_chunks(C(), L_CHUNK())(chunks_y);
    assert((combined_y_chunks-y) * is_concat == 0);

    /* Constraints to check correctness of chunks_query 
    If NOT a concat query: chunks_query === chunks_z 
    If its a concat query: then chunks_query === zip(chunks_x, chunks_y)
    */
    signal combined_z_chunks <== combine_chunks(C(), L_CHUNK())(chunks_query);
    assert((combined_z_chunks-z) * (1-is_concat) == 0);

    // the concat checks: 
    // the most significant chunk has a shorter length!
    for (var i=0; i<C(); i++) {
      assert(chunks_query[i] - (chunks_x[i] + chunks_y[i] * 2**L_CHUNK()) * is_concat == 0);
    } 
    // handles the "most significant" chunk here
    var idx_ms_chunk = C()-1;
    assert(chunks_query[idx_ms_chunk] - (chunks_x[idx_ms_chunk] + chunks_y[idx_ms_chunk] * 2**(L_MS_CHUNK())) * is_concat == 0);

    // For assert instructions 
    is_assert_false_instr * (1-lookup_output) === 0;
    is_assert_true_instr * lookup_output === 0;

    /* Constraints for storing value in register rd.
    */
    // lui doesn't need a lookup and simply requires the lookup_output to be set to immediate 
    // so it can be stored in the destination register. 

    signal rd_val <== memreg_v_writes[10]; 
    is_load_instr * (rd_val - load_or_store_value) === 0;
    is_lui * (rd_val - immediate) === 0;
    if_update_rd_with_lookup_output * (rd_val - lookup_output) === 0;
    is_jump_instr * (rd_val - (lookup_output-4)) === 0;

    /* Store into output state
    */
    output_state[STEP_NUM_IDX()] <== input_state[STEP_NUM_IDX()]+1;

    // set next PC 
    signal next_pc_j <== if_else()([is_jump_instr, lookup_output-4, input_state[PC_IDX()] + 4]);
    signal next_pc_j_b <== if_else()([is_branch_instr * lookup_output, input_state[PC_IDX()] + sign_imm * immediate, next_pc_j]);
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
Each vector is 8 * N in length. 
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

template JoltMain(N) {
    // The 3 program vectors are ordered by element. 
    // The address/value/timestamp of: 
    // [opcode for step 1, opcode for step 2, ... || rs1,... || rs2, ... || op_flags_packed, ...]
    signal input prog_a_rw[N * 6]; 
    signal input prog_v_rw[N * 6];
    signal input prog_t_reads[N * 6];

    // The combined registers and memory a/v/t vectors. 
    // These are ordered chronologically in terms of reads. 
    /* Each step has 11 mem ops: 
            1-2. Reading the two source registers. 
            3-10. The 8 bytes of memory read/written.
            11. Writing to the destination register. 
    */
    signal input memreg_a_rw[N * 11];
    signal input memreg_v_reads[N * 11];
    signal input memreg_v_writes[N * 11];
    signal input memreg_t_reads[N * 11];

    // These are the chunks of the two operands and the 'query'.
    // Here, query could be the z = x+y or z=x*y or,
    // in the case of concatenation, the chunks are [x_i || y_i]
    signal input chunks_x[N * C()];
    signal input chunks_y[N * C()];
    signal input chunks_query[N * C()]; // dim_i

    // The 'a' vector from Lasso containing the table entries looked up.
    signal input lookup_outputs[N];

    // The individual op_flags that guide the circuit. 
    // Unpacked from op_flags_packed, which is read from code.
    signal input op_flags[N * N_FLAGS()];

    // The final [step number, program counter]
    signal output out[2]; 

    /* Parse the program a/v/t vectors by element.
    */
    signal opcode_a_rw[N]         <== subarray(0, N, N*6)((prog_a_rw));
    signal rs1_a_rw[N]            <== subarray(N, N, N*6)((prog_a_rw));
    signal rs2_a_rw[N]            <== subarray(2*N, N, N*6)((prog_a_rw));
    signal rd_a_rw[N]             <== subarray(3*N, N, N*6)((prog_a_rw));
    signal immediate_a_rw[N]      <== subarray(4*N, N, N*6)((prog_a_rw));
    signal opflags_packed_a_rw[N] <== subarray(5*N, N, N*6)((prog_a_rw));

    signal opcode_v_rw[N]         <== subarray(0, N, N*6)((prog_v_rw));
    signal rs1_v_rw[N]            <== subarray(N, N, N*6)((prog_v_rw));
    signal rs2_v_rw[N]            <== subarray(2*N, N, N*6)((prog_v_rw));
    signal rd_v_rw[N]             <== subarray(3*N, N, N*6)((prog_v_rw));
    signal immediate_v_rw[N]      <== subarray(4*N, N, N*6)((prog_v_rw));
    signal opflags_packed_v_rw[N] <== subarray(5*N, N, N*6)((prog_v_rw));

    signal opcode_t_rw[N]         <== subarray(0, N, N*6)((prog_t_reads));
    signal rs1_t_rw[N]            <== subarray(N, N, N*6)((prog_t_reads));
    signal rs2_t_rw[N]            <== subarray(2*N, N, N*6)((prog_t_reads));
    signal rd_t_rw[N]             <== subarray(3*N, N, N*6)((prog_t_reads));
    signal immediate_t_rw[N]      <== subarray(4*N, N, N*6)((prog_t_reads));
    signal opflags_packed_t_rw[N] <== subarray(5*N, N, N*6)((prog_t_reads));


    /* Ensure that all program reads of a step are from the same address
    */
    for (var i=0; i<N; i++) {
        opcode_a_rw[i] === rs1_a_rw[i];
        opcode_a_rw[i] === rs2_a_rw[i];
        opcode_a_rw[i] === rd_a_rw[i];
        opcode_a_rw[i] === immediate_a_rw[i];
        opcode_a_rw[i] === opflags_packed_a_rw[i];
    }

    component jolt_steps[N];

    for (var i=0; i<N; i++) {
        jolt_steps[i] = JoltStep();

        if (i==0) {
            jolt_steps[i].input_state <== [0, 0];
        } else {
            jolt_steps[i].input_state <== jolt_steps[i-1].output_state;
        }

        jolt_steps[i].read_pc <== opcode_a_rw[i];

        jolt_steps[i].opcode <== opcode_v_rw[i];
        jolt_steps[i].rs1 <== rs1_v_rw[i];
        jolt_steps[i].rs2 <== rs2_v_rw[i];
        jolt_steps[i].rd <== rd_v_rw[i];
        jolt_steps[i].immediate <== immediate_v_rw[i];
        jolt_steps[i].op_flags_packed <== opflags_packed_v_rw[i];

        jolt_steps[i].memreg_a_rw <== subarray(i*11, 11, N*11)(memreg_a_rw);
        jolt_steps[i].memreg_v_reads <== subarray(i*11, 11, N*11)(memreg_v_reads);
        jolt_steps[i].memreg_v_writes <== subarray(i*11, 11, N*11)(memreg_v_writes);
        jolt_steps[i].memreg_t_reads <== subarray(i*11, 11, N*11)(memreg_t_reads);

        jolt_steps[i].chunks_x <== subarray(i*C(), C(), N*C())(chunks_x);
        jolt_steps[i].chunks_y <== subarray(i*C(), C(), N*C())(chunks_y);
        jolt_steps[i].chunks_query <== subarray(i*C(), C(), N*C())(chunks_y);

        jolt_steps[i].lookup_output <== lookup_outputs[i];

        jolt_steps[i].op_flags <== subarray(i*N_FLAGS(), N_FLAGS(), N*N_FLAGS())(op_flags);
    }

    out <== jolt_steps[N-1].output_state;
}

component main {public [
        prog_a_rw, prog_v_rw, prog_t_reads, 
        memreg_a_rw, memreg_v_reads, memreg_v_writes, memreg_t_reads,
        chunks_x, chunks_y, chunks_query, lookup_outputs, 
        op_flags]} 
    = JoltMain(NUM_STEPS());
