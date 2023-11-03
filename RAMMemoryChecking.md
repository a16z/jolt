# Memory Checking in Jolt

We'll briefly go through the read-write memory checking procedure. Then, how it's handled in Jolt. 

Things to iron out: 
1. What is produced by the emulator
2. What are lookups used for
3. What is the input to the grand product argument

### Memory-Checking Tuples 

First, we'll go through the algorithm for maintaining a read-write memory. Similar to Lasso, the format of a read or write tuple is (a, v, t). The operations are handled as follows: 

When reading address `a` : 
1. The prover claims that the value read is `v` and this value was written to it at time `t_read`. Both `v` and `t_read` are given as advice. Thus update `RS <-- RS \union (a, v, t_read)` .
2. Then the same value is figuratively "written back" to the register. This means updating the WS as `WS <-- WS \union (a, v, t)` where `t` is the current time.

Note that when address `a` is next read, the value the prover should provide as advice is the same `v, t` just written now. This is how RS and WS will eventually balance out (after accounting for the initial writes at time 0).

Writing value `v` to address `a`: 
1. Before every write, a figurative read is performed. As before, the prover provides `v_read, t_read` as advice and thus `RS <-- RS \union (a, v_read, t_read)`.
2. The difference now is that the new value is written. Update the WS as `WS <-- WS \union (a, v, t)`

As with a read-only memory, the WS should be updated to incorporate the initial writes at time `0` for each address: `WS <-- WS \union {(a, v_a, 0)} for all a`. 

The final RS and WS imply a sequence of `a`, `v`, `t` values that can be fed to the grand product argument.

#### Range checks

The only major check that we need to ensure on the prover's advice is that the value `t_read` provided as advice (for both reads and writes) is "honest". At least, it should be less than the current time counter. (If not, there exists attacks where the Prover can provide "future" values that are not yet written as advice and still have the final RS and WS balance out.)

Thus, our system must ensure that `ts_read < t_current` for all read-write memory. (This isn't required for read-only memories). 

### 1. The Emulator

The emulator is augumented to maintain the correct RS and WS at all times. Basically, this means maintaining an access counter (aka timestamp) for each address. A global step counter is also maintained. Everytime a memory operation is performed, this counter, along with the read set RS and write set WS are updated. We can treat the Program Code, Registers and RAM as all living in one memory space, so there is only one RS and WS at the end. 

For simplicity (in fact, this is easy to ensure), let's assume that the number of reads and writes per step is **constant**. This is because we know an upper bound on the exact number of PC, Regs and RAM operations per CPU step (6, 3 and 1, resp.). If some step doesn't actually perform an operation (eg, some instructions don't write to a destination register and only loads/stores ops access RAM), we can add a dummy operation to the emulator transcript.

Thus, we have exactly 10 memory operations happening per CPU step: 
* 6 are reads from PC, which is read-only
* 1 is a write to a destination register
* 2 are reads from source registers (rs1, rs2)
* 1 is a read/write to RAM (depending on load/store)

As the latter four are to read-write memory (Regs and RAM), we must perform a range check on the provided `ts_read` advice. 

Thus, we can treat the output of the emulator as follows:
* `RS = RS_1 \union RS_2 .... RS_N`: where `RS_i` is the set of 10 read tuples for step `i` 
* Analagously define `WS` and `WS_i`  where the latter is the set of 10 write tuples for step `i` 

### 2. What are lookups needed for? 

Recall that each CPU step has 4 `ts_read` values that are to read-write memories. These need to be range-checked to be less than the step number `t` itself. 

Let `RS'_i` be the subset of `RS` consisting of the four reads at step `i` that require a range-check. The timestamps of these four reads must be range-checked to be less than `i`. 
* These range checks can be done with Lasso parameter `c=1` and thus require only one advice element that's at most `i` itself. 
* Thus, each CPU step requires four additional values to be committed to by the prover. 

Note that these range checks and elements committed are not involved in R1CS. 

### 3. What is send to the grand product scheme?

I believe the grand product argument is agnostic to whether the memory was read-only or read-write. So it can be treated the same way as in Lasso, taking as input the RS and WS tuples produced by the transcript. 
