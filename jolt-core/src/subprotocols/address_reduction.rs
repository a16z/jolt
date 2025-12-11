//! RA Address Reduction Sumcheck
//!
//! This module implements a sumcheck that aligns the address portion of opening points
//! across all ra_i one-hot committed polynomials (InstructionRa, BytecodeRa, RamRa).
//!
//! ## Background
//!
//! After Stage 6, each ra_i one-hot polynomial has TWO claims at different address points
//! but the SAME cycle point (r_cycle_stage6):
//!
//! 1. **Booleanity claim**: `ra_i(r_address_bool, r_cycle_stage6)`
//!    - From `BooleanitySumcheck` in Stage 6
//!    - r_address_bool comes from booleanity sumcheck challenges
//!
//! 2. **Virtualization claim**: `ra_i(r_address_virt, r_cycle_stage6)`
//!    - For BytecodeRa: from `BytecodeReadRaf` in Stage 6
//!    - For InstructionRa: from `InstructionRaVirtualization` in Stage 6
//!    - For RamRa: from `RamRaVirtualization` in Stage 6
//!    - r_address_virt comes from the respective sumcheck challenges
//!
//! The third claim (HammingWeight) is handled separately: it runs in Stage 7 **batched**
//! with this AddressReduction sumcheck. Since they share sumcheck challenges, the
//! HammingWeight opening and AddressReduction output will have the SAME address point.
//!
//! ## Stage 7 Structure
//!
//! Stage 7 runs a batched sumcheck containing:
//! 1. All HammingWeight sumchecks (BytecodeHammingWeight, InstructionHammingWeight, RamHammingWeight)
//! 2. This AddressReduction sumcheck
//!
//! All instances run for `log_k_chunk` rounds (address variables only).
//! Since they're batched, they share the same challenges ρ_1, ..., ρ_{log_k_chunk}.
//!
//! After Stage 7:
//! - HammingWeight produces claims at (ρ, r_cycle_stage6)
//! - AddressReduction reduces Booleanity+Virtualization claims to (ρ, r_cycle_stage6)
//! - Each ra_i now has a SINGLE canonical opening point: (ρ, r_cycle_stage6)
//!
//! ## Sumcheck Relation
//!
//! Let N = total number of ra polynomials = instruction_d + bytecode_d + ram_d.
//!
//! For each ra_i (where i ∈ 0..N), we have:
//!   - claim_bool_i = ra_i(r_addr_bool_i, r_cycle)
//!   - claim_virt_i = ra_i(r_addr_virt_i, r_cycle)
//!
//! Define the "pushforward" polynomial:
//!   G_i(k) := Σ_j eq(r_cycle, j) · ra_i(k, j)
//!
//! Since ra_i is one-hot (ra_i(k,j) = 1 iff cycle j's address chunk i equals k):
//!   G_i(k) = Σ_{j: addr_chunk_i(j) = k} eq(r_cycle, j)
//!
//! The sumcheck proves:
//!
//!   Σ_k [Σ_i (γ^{2i} · eq(r_addr_bool_i, k) + γ^{2i+1} · eq(r_addr_virt_i, k))] · G_i(k)
//!     = Σ_i (γ^{2i} · claim_bool_i + γ^{2i+1} · claim_virt_i)
//!
//! This is rewritten as a sum over address k:
//!
//!   Σ_k W(k) = input_claim
//!
//! where W(k) = Σ_i (γ^{2i} · eq(r_addr_bool_i, k) + γ^{2i+1} · eq(r_addr_virt_i, k)) · G_i(k)
//!
//! ## Degree Analysis
//!
//! Each round polynomial has degree 2:
//!   - eq(r_addr_*, k) contributes degree 1 in the binding variable
//!   - G_i(k) contributes degree 1 in the binding variable (it's a dense MLE)
//!   - Product is degree 2
//!
//! ## Prover Initialization (Dominant Cost)
//!
//! Computing G_i(k) for all i and k is the expensive part. We use the split-eq optimization
//! from Section 6.3 of the Twist/Shout paper.
//!
//! ### Split-Eq Setup
//!
//! Given r_cycle = (r_out, r_in, r_last) where |r_out| ≈ |r_in| ≈ log(T)/2:
//!   - Precompute E_out = eq(r_out, ·) of size sqrt(T)
//!   - Precompute E_in = eq(r_in, ·) of size sqrt(T)
//!   - factor_0 = (1 - r_last), factor_1 = r_last
//!
//! The eq evaluation factorizes as:
//!   eq(r_cycle, j) = E_out[x_out] · E_in[x_in] · factor_{last_bit}
//!
//! where j = x_out · sqrt(T) · 2 + x_in · 2 + last_bit.
//!
//! ### Computing ra_i(k, j) On-The-Fly from Trace
//!
//! For each cycle j in the trace, we compute all ra_i chunk indices:
//!
//! ```text
//! fn compute_ra_indices(
//!     cycle: &Cycle,
//!     preprocessing: &JoltProverPreprocessing,
//!     one_hot_params: &OneHotParams,
//! ) -> RaIndices {
//!     // 1. InstructionRa: from lookup index
//!     let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
//!     let instruction_ra: [u16; instruction_d] = (0..instruction_d)
//!         .map(|i| one_hot_params.lookup_index_chunk(lookup_index, i))
//!         .collect();
//!
//!     // 2. BytecodeRa: from PC
//!     let pc = preprocessing.bytecode.get_pc(cycle);
//!     let bytecode_ra: [u16; bytecode_d] = (0..bytecode_d)
//!         .map(|i| one_hot_params.bytecode_pc_chunk(pc, i))
//!         .collect();
//!
//!     // 3. RamRa: from remapped address (may be None for non-memory cycles)
//!     let address = remap_address(cycle.ram_access().address() as u64, &memory_layout);
//!     let ram_ra: [Option<u16>; ram_d] = (0..ram_d)
//!         .map(|i| address.map(|a| one_hot_params.ram_address_chunk(a, i)))
//!         .collect();
//!
//!     RaIndices { instruction_ra, bytecode_ra, ram_ra }
//! }
//! ```
//!
//! ### Streaming Algorithm (Single Pass)
//!
//! ```text
//! fn compute_all_G(
//!     trace: &[Cycle],
//!     r_cycle: &[F::Challenge],
//!     preprocessing: &JoltProverPreprocessing,
//!     one_hot_params: &OneHotParams,
//! ) -> AllGPolynomials {
//!     let K = one_hot_params.k_chunk;  // size of each G_i array
//!     let N = instruction_d + bytecode_d + ram_d;  // total number of ra polynomials
//!     let sqrt_T = 1 << (log_T / 2);
//!     
//!     // Split r_cycle into (r_out, r_in, r_last)
//!     let (r_out, r_in, r_last) = split_cycle_challenges(r_cycle);
//!     
//!     // Precompute eq tables
//!     let E_out: Vec<F> = EqPolynomial::evals(&r_out);  // size sqrt_T
//!     let E_in: Vec<F> = EqPolynomial::evals(&r_in);    // size sqrt_T
//!     let factor_0 = F::one() - r_last;
//!     let factor_1 = r_last;
//!     
//!     // Parallel over x_out (outer chunks of trace)
//!     let local_results: Vec<Vec<Vec<F>>> = (0..sqrt_T)
//!         .into_par_iter()
//!         .map(|x_out| {
//!             // Each thread has local G_i arrays of size K
//!             let mut local_G: Vec<Vec<F>> = vec![vec![F::zero(); K]; N];
//!             
//!             // Sequential over x_in (inner chunks)
//!             for x_in in 0..sqrt_T {
//!                 let eq_in_term = E_in[x_in];
//!                 
//!                 // Process two consecutive cycles (last_bit = 0 and 1)
//!                 let j_base = x_out * sqrt_T * 2 + x_in * 2;
//!                 
//!                 for last_bit in 0..2 {
//!                     let j = j_base + last_bit;
//!                     if j >= trace.len() { continue; }  // Handle padding
//!                     
//!                     let factor = if last_bit == 0 { factor_0 } else { factor_1 };
//!                     let eq_contrib = eq_in_term * factor;
//!                     
//!                     // Compute all ra indices for cycle j
//!                     let cycle = &trace[j];
//!                     let ra_indices = compute_ra_indices(cycle, preprocessing, one_hot_params);
//!                     
//!                     // InstructionRa contributions (always present)
//!                     for i in 0..instruction_d {
//!                         let k = ra_indices.instruction_ra[i] as usize;
//!                         local_G[i][k] += eq_contrib;
//!                     }
//!                     
//!                     // BytecodeRa contributions (always present)
//!                     for i in 0..bytecode_d {
//!                         let k = ra_indices.bytecode_ra[i] as usize;
//!                         local_G[instruction_d + i][k] += eq_contrib;
//!                     }
//!                     
//!                     // RamRa contributions (may be None for non-memory cycles)
//!                     for i in 0..ram_d {
//!                         if let Some(k) = ra_indices.ram_ra[i] {
//!                             local_G[instruction_d + bytecode_d + i][k as usize] += eq_contrib;
//!                         }
//!                         // If None, this cycle doesn't contribute to RamRa
//!                     }
//!                 }
//!             }
//!             
//!             // Scale local_G by E_out[x_out]
//!             let e_out_term = E_out[x_out];
//!             for g in local_G.iter_mut() {
//!                 for val in g.iter_mut() {
//!                     *val *= e_out_term;
//!                 }
//!             }
//!             
//!             local_G
//!         })
//!         .collect();
//!     
//!     // Reduce: sum all local_G arrays into final G arrays
//!     let mut G: Vec<Vec<F>> = vec![vec![F::zero(); K]; N];
//!     for local_G in local_results {
//!         for (i, local_g) in local_G.into_iter().enumerate() {
//!             for (k, val) in local_g.into_iter().enumerate() {
//!                 G[i][k] += val;
//!             }
//!         }
//!     }
//!     
//!     // Convert to MultilinearPolynomials
//!     G.into_iter()
//!         .map(|coeffs| MultilinearPolynomial::from(coeffs))
//!         .collect()
//! }
//! ```
//!
//! ### Complexity Analysis
//!
//! - **Time**: O(T · N) where T = trace length, N = total ra polynomials
//!   - Single pass over trace (T iterations)
//!   - Each cycle computes N chunk indices and N scalar additions
//!   - Parallel over sqrt(T) outer chunks
//!
//! - **Memory**: O(sqrt(T) · N · K) for local_G arrays across threads
//!   - K = one_hot_params.k_chunk (typically 256)
//!   - Much smaller than materializing full ra_i polynomials (which would be K × T each)
//!
//! - **I/O**: Single streaming read of trace (cache-friendly)
//!
//! ### Key Optimizations
//!
//! 1. **No witness materialization**: We never build the full ra_i polynomials.
//!    Instead, we compute ra_i(k, j) on-the-fly from cycle data.
//!
//! 2. **Shared eq table**: All N polynomials share the same E_out, E_in tables.
//!    (vs. generic opening reduction which computes per-polynomial)
//!
//! 3. **Batched trace access**: We access each cycle exactly once and extract
//!    all N chunk indices together.
//!
//! 4. **Parallel outer loop**: The sqrt(T) outer chunks are independent,
//!    allowing good parallelism without synchronization during the main loop.
//!
//! ## Implementation Outline
//!
//! ### 1. SumcheckId
//!
//! Uses `SumcheckId::RaAddressReduction` (already defined in opening_proof.rs).
//!
//! ### 2. RaAddressReductionParams
//!
//! Fields:
//!   - gamma_powers: Vec<F>              // γ^0, γ^1, ..., γ^{2N-1}
//!   - r_cycle: Vec<F::Challenge>        // shared r_cycle from Stage 6
//!   - r_addr_bool: Vec<Vec<F::Challenge>>  // r_address per ra_i from Booleanity
//!   - r_addr_virt: Vec<Vec<F::Challenge>>  // r_address per ra_i from Virtualization
//!   - claims_bool: Vec<F>               // claims from Booleanity per ra_i
//!   - claims_virt: Vec<F>               // claims from Virtualization per ra_i
//!   - log_k_chunk: usize                // number of sumcheck rounds
//!   - polynomial_types: Vec<CommittedPolynomial>  // InstructionRa(0..d), BytecodeRa(0..d), RamRa(0..d)
//!
//! Methods:
//!   - new(): Fetches claims and r_address values from Stage 6 openings, samples gamma
//!   - input_claim(): Returns Σ_i (γ^{2i} · claim_bool_i + γ^{2i+1} · claim_virt_i)
//!   - degree(): Returns 2
//!   - num_rounds(): Returns log_k_chunk
//!
//! ### 3. RaAddressReductionProver
//!
//! State:
//!   - G: Vec<MultilinearPolynomial<F>>  // G_i for each ra polynomial
//!   - eq_bool: Vec<MultilinearPolynomial<F>>  // eq(r_addr_bool_i, ·) for each i
//!   - eq_virt: Vec<MultilinearPolynomial<F>>  // eq(r_addr_virt_i, ·) for each i
//!   - params: RaAddressReductionParams<F>
//!
//! Initialization:
//!   1. Compute all G_i using split-eq optimization (single trace streaming pass)
//!   2. Compute eq tables for each r_addr_bool_i and r_addr_virt_i
//!
//! compute_message(round, previous_claim):
//!   - Evaluate degree-2 univariate at points {0, 2}
//!   - For each j in 0..K/2:
//!       For each i:
//!         eq_b_evals = eq_bool[i].sumcheck_evals(j)
//!         eq_v_evals = eq_virt[i].sumcheck_evals(j)
//!         G_evals = G[i].sumcheck_evals(j)
//!         contribution = γ^{2i} * eq_b_evals * G_evals + γ^{2i+1} * eq_v_evals * G_evals
//!         accumulate contribution
//!   - Return UniPoly::from_evals_and_hint(previous_claim, &evals)
//!
//! ingest_challenge(r_j, round):
//!   - Bind all G_i, eq_bool_i, eq_virt_i polynomials at r_j
//!
//! cache_openings():
//!   - For each i: append sparse opening for ra_i at (ρ, r_cycle) with claim G_i(ρ)
//!   - SumcheckId: RaAddressReduction
//!
//! ### 4. RaAddressReductionVerifier
//!
//! expected_output_claim(challenges ρ):
//!   - For each i:
//!       eq_bool_eval = eq(r_addr_bool_i, ρ)
//!       eq_virt_eval = eq(r_addr_virt_i, ρ)
//!       G_i_claim = accumulator.get_committed_polynomial_opening(ra_i, RaAddressReduction)
//!       contribution = γ^{2i} * eq_bool_eval * G_i_claim + γ^{2i+1} * eq_virt_eval * G_i_claim
//!   - Return sum of contributions
//!
//! cache_openings():
//!   - For each i: append sparse opening for ra_i at (ρ, r_cycle)
//!
//! ### 5. Integration in Prover/Verifier (Stage 7)
//!
//! **Prover (prove_stage7):**
//!
//! ```text
//! // 1. Collect params for HammingWeight sumchecks
//! let bytecode_hw_params = bytecode::ra_hamming_weight_params(..., r_cycle_stage6);
//! let instruction_hw_params = instruction_lookups::ra_hamming_weight_params(..., r_cycle_stage6);
//! let ram_hw_params = ram::ra_hamming_weight_params(..., r_cycle_stage6);
//!
//! // 2. Collect params for AddressReduction
//! let address_reduction_params = RaAddressReductionParams::new(
//!     &opening_accumulator,
//!     &one_hot_params,
//!     r_cycle_stage6,
//!     &mut transcript,
//! );
//!
//! // 3. Initialize all provers (AddressReduction does the expensive G computation here)
//! let bytecode_hw = HammingWeightSumcheckProver::gen(bytecode_hw_params, ...);
//! let instruction_hw = HammingWeightSumcheckProver::gen(instruction_hw_params, ...);
//! let ram_hw = HammingWeightSumcheckProver::gen(ram_hw_params, ...);
//! let address_reduction = RaAddressReductionProver::initialize(
//!     address_reduction_params,
//!     &trace,
//!     &one_hot_params,
//! );
//!
//! // 4. Run batched sumcheck (all share log_k_chunk rounds)
//! let instances = vec![
//!     &mut bytecode_hw,
//!     &mut instruction_hw,
//!     &mut ram_hw,
//!     &mut address_reduction,
//! ];
//! let (sumcheck_proof, r_stage7) = BatchedSumcheck::prove(instances, ...);
//!
//! // After this:
//! // - HammingWeight claims are at (r_stage7, r_cycle_stage6)
//! // - AddressReduction reduced claims are at (r_stage7, r_cycle_stage6)
//! // - Each ra_i has ONE opening at this point, ready for Dory
//! ```
//!
//! **Verifier (verify_stage7):**
//!
//! Similar structure with verifier instances.
//!
//! ### 6. Removing OpeningReduction
//!
//! With IncReduction (Stage 6) and AddressReduction (Stage 7):
//! - RamInc, RdInc: reduced to single point in Stage 6
//! - All ra_i: reduced to single point in Stage 7
//!
//! These are the ONLY committed polynomials. The generic OpeningReduction sumcheck
//! (`SumcheckId::OpeningReduction`) is no longer needed.
//!
//! Stage 8 (Dory) then directly opens:
//! - RamInc(r_cycle_stage6)
//! - RdInc(r_cycle_stage6)
//! - InstructionRa(i)(ρ_addr, r_cycle_stage6) for each i
//! - BytecodeRa(i)(ρ_addr, r_cycle_stage6) for each i
//! - RamRa(i)(ρ_addr, r_cycle_stage6) for each i
//!
//! All share r_cycle_stage6 — maximum alignment achieved!
//!
//! ### 7. Benefits
//!
//! - **Prover efficiency**: G computation is a single streaming pass over trace
//!   (vs. generic opening reduction which did this per-polynomial per-opening)
//! - **Proof size**: Each ra_i has 1 opening instead of 3
//!   (saves ~2 field elements per ra_i polynomial)
//! - **Sumcheck cost**: Only log_k_chunk ≤ 8 rounds (vs. log_k_chunk + log_T for generic)
//! - **Memory**: G arrays are size K (not K*T), shared eq table

//! ## Detailed Wiring Changes
//!
//! ### Current Stage Layout (before optimization)
//!
//! ```text
//! Stage 4:
//!   - RegistersReadWriteChecking (emits RdInc claim)
//!   - RamBooleanity (emits RamRa claims) ← MOVE TO STAGE 6
//!   - RamValEvaluation (emits RamInc claim)
//!   - RamValFinal (emits RamInc claim, same point as ValEval)
//!
//! Stage 5:
//!   - RegistersValEvaluation (emits RdInc claim)
//!   - RamHammingBooleanity (emits virtual RamHammingWeight claim)
//!   - RamRaVirtualization (emits RamRa claims) ← MOVE TO STAGE 6
//!   - InstructionReadRaf (emits InstructionRa virtual claim)
//!
//! Stage 6:
//!   - BytecodeReadRaf (emits BytecodeRa claims)
//!   - BytecodeHammingWeight (emits BytecodeRa claims) ← MOVE TO STAGE 7
//!   - BytecodeBooleanity (emits BytecodeRa claims)
//!   - RamHammingWeight (emits RamRa claims) ← MOVE TO STAGE 7
//!   - InstructionRaVirtualization (emits InstructionRa claims)
//!   - InstructionBooleanity (emits InstructionRa claims)
//!   - InstructionHammingWeight (emits InstructionRa claims) ← MOVE TO STAGE 7
//!
//! Stage 7:
//!   - OpeningReduction (generic, expensive) ← REMOVE
//!
//! Stage 8:
//!   - Dory opening proof
//! ```
//!
//! ### New Stage Layout (after optimization)
//!
//! ```text
//! Stage 4:
//!   - RegistersReadWriteChecking (emits RdInc claim)
//!   - RamValEvaluation (emits RamInc claim)
//!   - RamValFinal (emits RamInc claim)
//!
//! Stage 5:
//!   - RegistersValEvaluation (emits RdInc claim)
//!   - RamHammingBooleanity (emits virtual RamHammingWeight claim)
//!   - InstructionReadRaf (emits InstructionRa virtual claim)
//!
//! Stage 6:
//!   - BytecodeReadRaf (emits BytecodeRa claims at r_addr_readraf, r_cycle_stage6)
//!   - BytecodeBooleanity (emits BytecodeRa claims at r_addr_bool, r_cycle_stage6)
//!   - RamBooleanity (emits RamRa claims at r_addr_bool, r_cycle_stage6)  ← MOVED HERE
//!   - RamRaVirtualization (emits RamRa claims at r_addr_virt, r_cycle_stage6)  ← MOVED HERE
//!   - InstructionRaVirtualization (emits InstructionRa claims at r_addr_virt, r_cycle_stage6)
//!   - InstructionBooleanity (emits InstructionRa claims at r_addr_bool, r_cycle_stage6)
//!   - IncReduction (reduces RamInc + RdInc to single point ρ_inc)  ← NEW
//!
//! Stage 7:
//!   - BytecodeHammingWeight (uses r_cycle_stage6, produces r_addr = ρ)  ← MOVED HERE
//!   - InstructionHammingWeight (uses r_cycle_stage6, produces r_addr = ρ)  ← MOVED HERE
//!   - RamHammingWeight (uses r_cycle_stage6, produces r_addr = ρ)  ← MOVED HERE
//!   - RaAddressReduction (aligns Stage 6 claims to r_addr = ρ)  ← NEW
//!   // All batched together, sharing challenges ρ_1..ρ_{log_k_chunk}
//!
//! Stage 8:
//!   - Dory opening proof (directly on aligned claims)
//! ```
//!
//! ### Claim Flow Summary
//!
//! After Stage 6:
//!   - RamInc: 1 claim at r_cycle_stage6 (from IncReduction, batched with RA sumchecks)
//!   - RdInc: 1 claim at r_cycle_stage6 (from IncReduction, batched with RA sumchecks)
//!   - BytecodeRa(i): 2 claims at (r_addr_bool, r_cycle_stage6), (r_addr_readraf, r_cycle_stage6)
//!   - InstructionRa(i): 2 claims at (r_addr_bool, r_cycle_stage6), (r_addr_virt, r_cycle_stage6)
//!   - RamRa(i): 2 claims at (r_addr_bool, r_cycle_stage6), (r_addr_virt, r_cycle_stage6)
//!
//! After Stage 7:
//!   - RamInc: 1 claim at r_cycle_stage6 (unchanged, already fully reduced)
//!   - RdInc: 1 claim at r_cycle_stage6 (unchanged, already fully reduced)
//!   - BytecodeRa(i): 1 claim at (ρ_addr, r_cycle_stage6) ← HammingWeight + AddressReduction
//!   - InstructionRa(i): 1 claim at (ρ_addr, r_cycle_stage6) ← HammingWeight + AddressReduction
//!   - RamRa(i): 1 claim at (ρ_addr, r_cycle_stage6) ← HammingWeight + AddressReduction
//!
//! All committed polynomials share r_cycle_stage6 as their cycle component!
//!
//! ### Key Invariants
//!
//! 1. **Shared r_cycle in Stage 6**: All RA-related sumchecks (Booleanity, Virtualization,
//!    ReadRaf) must be in the same batched sumcheck so they share r_cycle from the
//!    sumcheck challenges.
//!
//! 2. **Shared r_address in Stage 7**: HammingWeight and AddressReduction must be in the
//!    same batched sumcheck so they share r_address from the sumcheck challenges.
//!
//! 3. **HammingWeight uses Stage 6 r_cycle**: The HammingWeight params must pass in
//!    r_cycle_stage6 (from Stage 6 sumcheck challenges), not sample new challenges.
//!
//! 4. **AddressReduction only needs 2 claims per ra_i**: Booleanity + Virtualization.
//!    HammingWeight claim is automatically aligned since it shares challenges.
//!
//! ### How r_cycle Flows from Stage 6 to Stage 7
//!
//! Stage 6 runs a batched sumcheck with instances that all have log(T) cycle rounds.
//! The shared sumcheck challenges become r_cycle_stage6.
//!
//! At the end of Stage 6, `cache_openings` is called for each instance. The RA-related
//! instances append their claims with r_cycle = r_cycle_stage6.
//!
//! Stage 7 then needs to access r_cycle_stage6. There are two options:
//!
//! **Option A**: Store r_cycle_stage6 in prover state
//!   - After Stage 6 BatchedSumcheck::prove returns (proof, r_stage6)
//!   - Extract the cycle portion: r_cycle_stage6 = r_stage6[log_k_chunk..]
//!   - Pass r_cycle_stage6 to Stage 7 HammingWeight and AddressReduction params
//!
//! **Option B**: Retrieve from opening accumulator
//!   - Stage 6 cache_openings appended claims with r_cycle_stage6
//!   - Stage 7 can retrieve any RA claim's opening point and extract r_cycle
//!   - e.g., get_committed_polynomial_opening(BytecodeRa(0), BytecodeBooleanity).0
//!
//! Option A is cleaner and more explicit. The prover would:
//!
//! ```text
//! // Stage 6
//! let (stage6_proof, r_stage6) = BatchedSumcheck::prove(stage6_instances, ...);
//! let r_cycle_stage6 = r_stage6[log_k_chunk..].to_vec();  // Extract cycle portion
//!
//! // Stage 7
//! let hw_params = HammingWeightParams {
//!     r_cycle: r_cycle_stage6.clone(),
//!     ...
//! };
//! let addr_reduction_params = RaAddressReductionParams::new(
//!     r_cycle_stage6,
//!     &opening_accumulator,
//!     ...
//! );
//! ```
//!
//! ### Opening Accumulator Changes
//!
//! The `append_sparse` calls from Stage 6 sumchecks will accumulate openings at
//! different r_address values but the SAME r_cycle_stage6. The AddressReduction sumcheck
//! reads these accumulated claims and produces a single reduced opening per ra_i.
//!
//! After Stage 7, the opening accumulator should contain:
//!   - RamInc at r_cycle_stage6 (from IncReduction)
//!   - RdInc at r_cycle_stage6 (from IncReduction)
//!   - Each ra_i at (ρ_addr, r_cycle_stage6) (from AddressReduction)
//!
//! All polynomials share r_cycle_stage6. They go directly to Dory without any
//! generic opening reduction sumcheck.
//!
//! ### Dory Integration
//!
//! The `OpeningReductionState` (currently produced by Stage 7 generic opening reduction)
//! needs to be replaced with a simpler structure that just lists:
//!   - The final opening points
//!   - The corresponding claims
//!   - The polynomial commitments
//!
//! The RLC polynomial construction remains the same, but operates on fewer, aligned openings.

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::multilinear_polynomial::BindingOrder;
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::utils::math::Math;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::zkvm::config::OneHotParams;
use crate::zkvm::instruction::LookupQuery;
use crate::zkvm::prover::JoltProverPreprocessing;
use crate::zkvm::ram::remap_address;
use common::constants::XLEN;
use fixedbitset::FixedBitSet;
use rayon::prelude::*;
use tracer::instruction::Cycle;

/// Stores the chunk indices for all ra polynomials for a single cycle.
struct RaIndices {
    /// InstructionRa chunk indices (always present)
    instruction_ra: Vec<u16>,
    /// BytecodeRa chunk indices (always present)
    bytecode_ra: Vec<u16>,
    /// RamRa chunk indices (may be None for non-memory cycles)
    ram_ra: Vec<Option<u16>>,
}

impl RaIndices {
    /// Compute all ra chunk indices for a single cycle from trace data.
    #[inline]
    fn from_cycle<F: JoltField, PCS: CommitmentScheme<Field = F>>(
        cycle: &Cycle,
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        one_hot_params: &OneHotParams,
    ) -> Self {
        // 1. InstructionRa: from lookup index
        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
        let instruction_ra: Vec<u16> = (0..one_hot_params.instruction_d)
            .map(|i| one_hot_params.lookup_index_chunk(lookup_index, i))
            .collect();

        // 2. BytecodeRa: from PC
        let pc = preprocessing.bytecode.get_pc(cycle);
        let bytecode_ra: Vec<u16> = (0..one_hot_params.bytecode_d)
            .map(|i| one_hot_params.bytecode_pc_chunk(pc, i))
            .collect();

        // 3. RamRa: from remapped address (may be None for non-memory cycles)
        let address = remap_address(
            cycle.ram_access().address() as u64,
            &preprocessing.memory_layout,
        );
        let ram_ra: Vec<Option<u16>> = (0..one_hot_params.ram_d)
            .map(|i| address.map(|a| one_hot_params.ram_address_chunk(a, i)))
            .collect();

        Self {
            instruction_ra,
            bytecode_ra,
            ram_ra,
        }
    }
}

/// Computes all G_i polynomials in a single streaming pass over the trace.
///
/// G_i(k) = Σ_j eq(r_cycle, j) · ra_i(k, j)
///
/// For one-hot ra polynomials:
/// G_i(k) = Σ_{j: addr_chunk_i(j) = k} eq(r_cycle, j)
#[tracing::instrument(skip_all, name = "RaAddressReduction::compute_all_G")]
pub fn compute_all_G<F: JoltField, PCS: CommitmentScheme<Field = F>>(
    trace: &[Cycle],
    r_cycle: &[F::Challenge],
    preprocessing: &JoltProverPreprocessing<F, PCS>,
    one_hot_params: &OneHotParams,
) -> Vec<Vec<F>> {
    let K = one_hot_params.k_chunk;
    let instruction_d = one_hot_params.instruction_d;
    let bytecode_d = one_hot_params.bytecode_d;
    let ram_d = one_hot_params.ram_d;
    let N = instruction_d + bytecode_d + ram_d; // Total number of ra polynomials
    let T = trace.len();

    // Build split-eq polynomial over r_cycle
    // This gives us E_out, E_in tables and the current_w (last challenge)
    let split_eq = GruenSplitEqPolynomial::<F>::new(r_cycle, BindingOrder::LowToHigh);

    let E_in = split_eq.E_in_current();
    let E_out = split_eq.E_out_current();
    let w_current = split_eq.get_current_w();
    let factor_0 = F::one() - w_current;
    let factor_1: F = w_current.into();

    let in_len = E_in.len();
    let x_in_bits = in_len.log_2();

    // Precompute merged inner weights: [E_in[x_in] * (1-w), E_in[x_in] * w] for all x_in
    // This avoids recomputing the product for each (x_out, x_in) pair
    let merged_in_unreduced: Vec<F::Unreduced<9>> = {
        let mut merged: Vec<F::Unreduced<9>> = unsafe_allocate_zero_vec(2 * in_len);
        merged
            .par_chunks_exact_mut(2)
            .zip(E_in.par_iter())
            .for_each(|(chunk, &low)| {
                chunk[0] = low.mul_unreduced::<9>(factor_0);
                chunk[1] = low.mul_unreduced::<9>(factor_1);
            });
        merged
    };

    // Parallel fold over E_out indices
    // Each thread maintains local G arrays for all N polynomials
    let G: Vec<Vec<F>> = E_out
        .par_iter()
        .enumerate()
        .fold(
            || vec![unsafe_allocate_zero_vec::<F>(K); N],
            |mut partial_G, (x_out, &e_out)| {
                // Local unreduced accumulators for this x_out chunk
                let mut local_unreduced: Vec<Vec<F::Unreduced<9>>> =
                    vec![unsafe_allocate_zero_vec(K); N];

                // Track which indices were touched for efficient reduction
                let mut touched_flags: Vec<FixedBitSet> = vec![FixedBitSet::with_capacity(K); N];

                let x_out_base = x_out << (x_in_bits + 1);

                // Sequential over x_in
                for x_in in 0..in_len {
                    let j0 = x_out_base + (x_in << 1);
                    let j1 = j0 + 1;
                    let off = 2 * x_in;
                    let add0_unr = merged_in_unreduced[off];
                    let add1_unr = merged_in_unreduced[off + 1];

                    // Process cycle j0 (last_bit = 0)
                    if j0 < T {
                        let ra_indices =
                            RaIndices::from_cycle(&trace[j0], preprocessing, one_hot_params);

                        // InstructionRa contributions
                        for i in 0..instruction_d {
                            let k = ra_indices.instruction_ra[i] as usize;
                            if !touched_flags[i].contains(k) {
                                touched_flags[i].insert(k);
                            }
                            local_unreduced[i][k] += add0_unr;
                        }

                        // BytecodeRa contributions
                        for i in 0..bytecode_d {
                            let poly_idx = instruction_d + i;
                            let k = ra_indices.bytecode_ra[i] as usize;
                            if !touched_flags[poly_idx].contains(k) {
                                touched_flags[poly_idx].insert(k);
                            }
                            local_unreduced[poly_idx][k] += add0_unr;
                        }

                        // RamRa contributions (may be None)
                        for i in 0..ram_d {
                            let poly_idx = instruction_d + bytecode_d + i;
                            if let Some(k) = ra_indices.ram_ra[i] {
                                let k = k as usize;
                                if !touched_flags[poly_idx].contains(k) {
                                    touched_flags[poly_idx].insert(k);
                                }
                                local_unreduced[poly_idx][k] += add0_unr;
                            }
                        }
                    }

                    // Process cycle j1 (last_bit = 1)
                    if j1 < T {
                        let ra_indices =
                            RaIndices::from_cycle(&trace[j1], preprocessing, one_hot_params);

                        // InstructionRa contributions
                        for i in 0..instruction_d {
                            let k = ra_indices.instruction_ra[i] as usize;
                            if !touched_flags[i].contains(k) {
                                touched_flags[i].insert(k);
                            }
                            local_unreduced[i][k] += add1_unr;
                        }

                        // BytecodeRa contributions
                        for i in 0..bytecode_d {
                            let poly_idx = instruction_d + i;
                            let k = ra_indices.bytecode_ra[i] as usize;
                            if !touched_flags[poly_idx].contains(k) {
                                touched_flags[poly_idx].insert(k);
                            }
                            local_unreduced[poly_idx][k] += add1_unr;
                        }

                        // RamRa contributions (may be None)
                        for i in 0..ram_d {
                            let poly_idx = instruction_d + bytecode_d + i;
                            if let Some(k) = ra_indices.ram_ra[i] {
                                let k = k as usize;
                                if !touched_flags[poly_idx].contains(k) {
                                    touched_flags[poly_idx].insert(k);
                                }
                                local_unreduced[poly_idx][k] += add1_unr;
                            }
                        }
                    }
                }

                // Reduce and scale by E_out[x_out]
                for poly_idx in 0..N {
                    for k in touched_flags[poly_idx].ones() {
                        let reduced = F::from_montgomery_reduce::<9>(local_unreduced[poly_idx][k]);
                        partial_G[poly_idx][k] += e_out * reduced;
                    }
                }

                partial_G
            },
        )
        .reduce(
            || vec![unsafe_allocate_zero_vec::<F>(K); N],
            |mut a, b| {
                for poly_idx in 0..N {
                    for (x, y) in a[poly_idx].iter_mut().zip(&b[poly_idx]) {
                        *x += *y;
                    }
                }
                a
            },
        );

    G
}

// ============================================================================
// IMPORTS FOR SUMCHECK
// ============================================================================

use allocative::Allocative;

use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier};
use crate::transcripts::Transcript;
use crate::zkvm::witness::CommittedPolynomial;

const DEGREE_BOUND: usize = 2;

// ============================================================================
// PARAMS
// ============================================================================

/// Parameters for the RA address reduction sumcheck.
///
/// This sumcheck aligns the address portion of opening points across all ra_i
/// one-hot polynomials. After Stage 6, each ra_i has two claims at different
/// r_address values but the same r_cycle. This sumcheck reduces them to a
/// single opening point per ra_i.
#[derive(Allocative, Clone)]
pub struct RaAddressReductionParams<F: JoltField> {
    /// γ^0, γ^1, ..., γ^{2N-1} for batching (N = total ra polynomials)
    pub gamma_powers: Vec<F>,
    /// Shared r_cycle from Stage 6 (all ra claims share this)
    pub r_cycle: Vec<F::Challenge>,
    /// r_address values from Booleanity sumcheck for each ra_i
    pub r_addr_bool: Vec<Vec<F::Challenge>>,
    /// r_address values from Virtualization/ReadRaf sumcheck for each ra_i
    pub r_addr_virt: Vec<Vec<F::Challenge>>,
    /// Claims from Booleanity sumcheck for each ra_i
    pub claims_bool: Vec<F>,
    /// Claims from Virtualization/ReadRaf sumcheck for each ra_i
    pub claims_virt: Vec<F>,
    /// log_2(k_chunk) - number of sumcheck rounds
    pub log_k_chunk: usize,
    /// Polynomial labels: InstructionRa(0..d), BytecodeRa(0..d), RamRa(0..d)
    pub polynomial_types: Vec<CommittedPolynomial>,
}

impl<F: JoltField> RaAddressReductionParams<F> {
    /// Create params by fetching claims from Stage 6 and sampling batching challenge.
    pub fn new(
        r_cycle: Vec<F::Challenge>,
        one_hot_params: &OneHotParams,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let instruction_d = one_hot_params.instruction_d;
        let bytecode_d = one_hot_params.bytecode_d;
        let ram_d = one_hot_params.ram_d;
        let N = instruction_d + bytecode_d + ram_d;

        // Build polynomial types list
        let mut polynomial_types = Vec::with_capacity(N);
        for i in 0..instruction_d {
            polynomial_types.push(CommittedPolynomial::InstructionRa(i));
        }
        for i in 0..bytecode_d {
            polynomial_types.push(CommittedPolynomial::BytecodeRa(i));
        }
        for i in 0..ram_d {
            polynomial_types.push(CommittedPolynomial::RamRa(i));
        }

        // Sample batching challenge γ and compute powers
        let gamma: F = transcript.challenge_scalar();
        let mut gamma_powers = Vec::with_capacity(2 * N);
        let mut power = F::one();
        for _ in 0..(2 * N) {
            gamma_powers.push(power);
            power *= gamma;
        }

        // Fetch claims and r_address values from Stage 6 openings
        let mut r_addr_bool = Vec::with_capacity(N);
        let mut r_addr_virt = Vec::with_capacity(N);
        let mut claims_bool = Vec::with_capacity(N);
        let mut claims_virt = Vec::with_capacity(N);

        for poly_type in &polynomial_types {
            // Booleanity claim
            let (bool_sumcheck_id, virt_sumcheck_id) = match poly_type {
                CommittedPolynomial::InstructionRa(_) => (
                    SumcheckId::InstructionBooleanity,
                    SumcheckId::InstructionRaVirtualization,
                ),
                CommittedPolynomial::BytecodeRa(_) => {
                    (SumcheckId::BytecodeBooleanity, SumcheckId::BytecodeReadRaf)
                }
                CommittedPolynomial::RamRa(_) => {
                    (SumcheckId::RamBooleanity, SumcheckId::RamRaVirtualization)
                }
                _ => unreachable!(),
            };

            let (bool_point, bool_claim) =
                accumulator.get_committed_polynomial_opening(*poly_type, bool_sumcheck_id);
            let (virt_point, virt_claim) =
                accumulator.get_committed_polynomial_opening(*poly_type, virt_sumcheck_id);

            // Extract just the address portion (first log_k_chunk challenges)
            let log_k_chunk = one_hot_params.log_k_chunk;
            r_addr_bool.push(bool_point.r[..log_k_chunk].to_vec());
            r_addr_virt.push(virt_point.r[..log_k_chunk].to_vec());
            claims_bool.push(bool_claim);
            claims_virt.push(virt_claim);
        }

        Self {
            gamma_powers,
            r_cycle,
            r_addr_bool,
            r_addr_virt,
            claims_bool,
            claims_virt,
            log_k_chunk: one_hot_params.log_k_chunk,
            polynomial_types,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for RaAddressReductionParams<F> {
    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        // Σ_i (γ^{2i} · claim_bool_i + γ^{2i+1} · claim_virt_i)
        let mut claim = F::zero();
        for i in 0..self.polynomial_types.len() {
            claim += self.gamma_powers[2 * i] * self.claims_bool[i];
            claim += self.gamma_powers[2 * i + 1] * self.claims_virt[i];
        }
        claim
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.log_k_chunk
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        // Address challenges come from sumcheck (little-endian), convert to big-endian
        // Then concatenate with r_cycle to form full opening point
        let r_addr: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness();
        let full_point = [r_addr.r.as_slice(), self.r_cycle.as_slice()].concat();
        OpeningPoint::<BIG_ENDIAN, F>::new(full_point)
    }
}

// ============================================================================
// PROVER
// ============================================================================

/// Prover for RA address reduction sumcheck.
///
/// This is a standard sumcheck (no prefix-suffix optimization needed since
/// log_k_chunk is small, typically ≤ 8).
#[derive(Allocative)]
pub struct RaAddressReductionProver<F: JoltField> {
    /// G_i polynomials (pushforward of ra_i over r_cycle)
    /// G_i(k) = Σ_j eq(r_cycle, j) · ra_i(k, j)
    G: Vec<MultilinearPolynomial<F>>,
    /// eq(r_addr_bool_i, ·) for each ra polynomial
    eq_bool: Vec<MultilinearPolynomial<F>>,
    /// eq(r_addr_virt_i, ·) for each ra polynomial
    eq_virt: Vec<MultilinearPolynomial<F>>,
    params: RaAddressReductionParams<F>,
}

impl<F: JoltField> RaAddressReductionProver<F> {
    /// Initialize the prover by computing all G_i polynomials.
    #[tracing::instrument(skip_all, name = "RaAddressReductionProver::initialize")]
    pub fn initialize<PCS: CommitmentScheme<Field = F>>(
        params: RaAddressReductionParams<F>,
        trace: &[Cycle],
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        one_hot_params: &OneHotParams,
    ) -> Self {
        // Compute all G_i polynomials via streaming
        let G_vecs = compute_all_G(trace, &params.r_cycle, preprocessing, one_hot_params);
        let G: Vec<MultilinearPolynomial<F>> = G_vecs
            .into_iter()
            .map(|v| MultilinearPolynomial::from(v))
            .collect();

        // Compute eq tables for each r_address
        let N = params.polynomial_types.len();
        let mut eq_bool = Vec::with_capacity(N);
        let mut eq_virt = Vec::with_capacity(N);

        for i in 0..N {
            eq_bool.push(MultilinearPolynomial::from(EqPolynomial::evals(
                &params.r_addr_bool[i],
            )));
            eq_virt.push(MultilinearPolynomial::from(EqPolynomial::evals(
                &params.r_addr_virt[i],
            )));
        }

        Self {
            G,
            eq_bool,
            eq_virt,
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for RaAddressReductionProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "RaAddressReductionProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let N = self.params.polynomial_types.len();
        let half_n = self.G[0].len() / 2;

        let mut evals = [F::zero(); DEGREE_BOUND];

        for j in 0..half_n {
            for i in 0..N {
                let g_evals =
                    self.G[i].sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                let eq_b_evals = self.eq_bool[i]
                    .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                let eq_v_evals = self.eq_virt[i]
                    .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);

                // γ^{2i} · eq_bool · G + γ^{2i+1} · eq_virt · G
                let gamma_bool = self.params.gamma_powers[2 * i];
                let gamma_virt = self.params.gamma_powers[2 * i + 1];

                for k in 0..DEGREE_BOUND {
                    evals[k] += gamma_bool * eq_b_evals[k] * g_evals[k]
                        + gamma_virt * eq_v_evals[k] * g_evals[k];
                }
            }
        }

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    #[tracing::instrument(skip_all, name = "RaAddressReductionProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        // Bind all polynomials in parallel
        let (g_slice, eq_slices) = (&mut self.G, (&mut self.eq_bool, &mut self.eq_virt));

        rayon::scope(|s| {
            s.spawn(|_| {
                g_slice.par_iter_mut().for_each(|g| {
                    g.bind_parallel(r_j, BindingOrder::LowToHigh);
                });
            });
            s.spawn(|_| {
                eq_slices.0.par_iter_mut().for_each(|eq| {
                    eq.bind_parallel(r_j, BindingOrder::LowToHigh);
                });
            });
            s.spawn(|_| {
                eq_slices.1.par_iter_mut().for_each(|eq| {
                    eq.bind_parallel(r_j, BindingOrder::LowToHigh);
                });
            });
        });
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let N = self.params.polynomial_types.len();

        // Extract r_address portion (just the sumcheck challenges, converted to big-endian)
        let r_address: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness();
        let r_address = r_address.r;

        for i in 0..N {
            // Final claim is G_i(ρ) where ρ is the sumcheck challenges
            let claim = self.G[i].final_sumcheck_claim();

            accumulator.append_sparse(
                transcript,
                vec![self.params.polynomial_types[i]],
                SumcheckId::RaAddressReduction,
                r_address.clone(),
                self.params.r_cycle.clone(),
                vec![claim],
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

// ============================================================================
// VERIFIER
// ============================================================================

pub struct RaAddressReductionVerifier<F: JoltField> {
    params: RaAddressReductionParams<F>,
}

impl<F: JoltField> RaAddressReductionVerifier<F> {
    pub fn new(
        r_cycle: Vec<F::Challenge>,
        one_hot_params: &OneHotParams,
        accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params =
            RaAddressReductionParams::new(r_cycle, one_hot_params, accumulator, transcript);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for RaAddressReductionVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let N = self.params.polynomial_types.len();

        // Compute ρ (final address point)
        let rho: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness();
        let rho = rho.r;

        let mut output_claim = F::zero();

        for i in 0..N {
            // eq evaluations at final point ρ
            let eq_bool_eval = EqPolynomial::mle(&rho, &self.params.r_addr_bool[i]);
            let eq_virt_eval = EqPolynomial::mle(&rho, &self.params.r_addr_virt[i]);

            // Fetch G_i(ρ) from accumulator (prover provided this)
            let (_, g_i_claim) = accumulator.get_committed_polynomial_opening(
                self.params.polynomial_types[i],
                SumcheckId::RaAddressReduction,
            );

            // γ^{2i} · eq_bool(ρ) · G_i(ρ) + γ^{2i+1} · eq_virt(ρ) · G_i(ρ)
            let gamma_bool = self.params.gamma_powers[2 * i];
            let gamma_virt = self.params.gamma_powers[2 * i + 1];

            output_claim += gamma_bool * eq_bool_eval * g_i_claim;
            output_claim += gamma_virt * eq_virt_eval * g_i_claim;
        }

        output_claim
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let N = self.params.polynomial_types.len();

        // Compute full opening point (r_address || r_cycle)
        let r_address: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness();
        let r_address = r_address.r;
        let full_point = [r_address.as_slice(), self.params.r_cycle.as_slice()].concat();

        for i in 0..N {
            accumulator.append_sparse(
                transcript,
                vec![self.params.polynomial_types[i]],
                SumcheckId::RaAddressReduction,
                full_point.clone(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    // TODO: Add tests comparing compute_all_G output against naive computation
    // TODO: Add tests for sumcheck correctness
}
