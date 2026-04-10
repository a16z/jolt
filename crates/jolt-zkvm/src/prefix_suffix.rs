//! Prefix-suffix decomposition prover for instruction lookup sumchecks.
//!
//! Implements the multi-phase prefix-suffix decomposition used by
//! InstructionReadRaf. The 128-bit address space is decomposed into
//! `num_phases` sub-phases of `chunk_bits` each.
//!
//! Address rounds: for each sub-phase, builds prefix (P) and suffix (Q)
//! polynomials, computes round poly as Σ P×Q (read-checking + RAF).
//! Between phases: update expanding tables, prefix checkpoints, condensed u_evals.
//!
//! At the address→cycle transition, materializes ra_polys and combined_val
//! into output buffers for the Dense cycle-phase kernel.

use std::collections::HashMap;

use jolt_compiler::kernel_spec::Iteration;
use jolt_compiler::PolynomialId;
use jolt_field::Field;
use jolt_instructions::tables::prefixes::{
    PrefixCheckpoint, PrefixEval, Prefixes, ALL_PREFIXES, NUM_PREFIXES,
};
use jolt_instructions::tables::suffixes::Suffixes;
use jolt_instructions::{LookupBits, LookupTableKind, LookupTables};
use jolt_poly::EqPolynomial;

/// XLEN = 64 for RV64.
const XLEN: usize = 64;
/// LOG_K = 128 for instruction lookups (2 operands × 64 bits).
const LOG_K: usize = 128;

/// Per-cycle data needed for the prefix-suffix decomposition.
///
/// Extracted from the execution trace and passed to PrefixSuffixState
/// at initialization.
pub struct LookupTraceData {
    /// Per-cycle lookup key (128-bit packed), T entries.
    pub lookup_keys: Vec<u128>,
    /// Per-cycle lookup table kind, T entries. None for cycles with no lookup.
    pub table_kinds: Vec<Option<LookupTableKind>>,
    /// Per-cycle interleaved-operands flag, T entries.
    pub is_interleaved: Vec<bool>,
}

/// State for one PrefixSuffix instance, persisting across address rounds.
///
/// Created at instance activation (first active round). Consumed at the
/// address→cycle transition.
pub struct PrefixSuffixState<F: Field> {
    // -- Parameters --
    chunk_bits: usize,
    num_phases: usize,
    ra_virtual_log_k_chunk: usize,
    gamma: F,
    gamma_sqr: F,

    // -- Trace data --
    lookup_keys: Vec<LookupBits>,
    table_kinds: Vec<Option<LookupTableKind>>,
    /// Indices of cycles grouped by table kind. `[table_idx] -> [cycle indices]`
    lookup_indices_by_table: Vec<Vec<usize>>,
    is_interleaved: Vec<bool>,
    num_cycles: usize,

    // -- Address round state --
    /// eq(r_reduction, j) for each cycle j, condensed each sub-phase.
    u_evals: Vec<F>,

    /// Expanding tables: one per sub-phase. Each is `2^chunk_bits` elements,
    /// tracking the running product of prefix-chunk eq evaluations per cycle.
    /// `v[phase][k]` starts as F::one() at reset and expands via update(r_j).
    v: Vec<ExpandingTable<F>>,

    /// Running list of sumcheck challenges (address vars, in binding order).
    r: Vec<F>,

    /// Prefix checkpoints (one per Prefix variant), updated every 2 rounds.
    prefix_checkpoints: Vec<PrefixCheckpoint<F>>,

    /// Per-table suffix accumulators for the current phase.
    /// `suffix_polys[table_idx][suffix_idx]` has `2^chunk_bits` elements.
    suffix_polys: Vec<Vec<Vec<F>>>,

    // -- RAF decomposition state --
    /// Q polynomials for left operand: [shift_half, left_op], each 2^chunk_bits.
    q_left: [Vec<F>; 2],
    /// Q polynomials for right operand: [shift_half, right_op], each 2^chunk_bits.
    q_right: [Vec<F>; 2],
    /// Q polynomials for identity: [shift_full, identity], each 2^chunk_bits.
    q_identity: [Vec<F>; 2],

    /// P polynomial evaluations for left/right/identity. Each has ORDER=2 arrays.
    /// These are computed from the prefix_registry checkpoints per phase.
    /// `p_left[i]` = Option<Vec<F>> of length 2^chunk_bits.
    p_left: [Option<Vec<F>>; 2],
    p_right: [Option<Vec<F>>; 2],
    p_identity: [Option<Vec<F>>; 2],

    /// Prefix registry checkpoints for P construction.
    /// `registry_checkpoints[Prefix variant]` = Option<F>
    registry_checkpoints: [Option<F>; 3],

    /// Current sub-phase index (0-based).
    current_phase: usize,
    /// Total rounds completed.
    total_round: usize,

    /// Output PolynomialIds for materialized RA polys.
    output_ra_poly_ids: Vec<PolynomialId>,
    /// Output PolynomialId for materialized combined_val.
    output_combined_val_id: PolynomialId,
}

/// Simple expanding table: eq(r_1, ..., r_j) evaluations built incrementally.
/// HighToLow binding order (consistent with address-phase convention).
struct ExpandingTable<F: Field> {
    values: Vec<F>,
    len: usize,
}

impl<F: Field> ExpandingTable<F> {
    fn new(capacity: usize) -> Self {
        let mut values = vec![F::zero(); capacity];
        values[0] = F::one();
        Self { values, len: 1 }
    }

    fn reset(&mut self) {
        self.values[0] = F::one();
        self.len = 1;
    }

    /// Expand by incorporating challenge r_j (HighToLow binding order).
    /// New values: v[2i] = v[i] - r_j*v[i], v[2i+1] = r_j*v[i].
    fn update(&mut self, r_j: F) {
        for i in (0..self.len).rev() {
            let v_i = self.values[i];
            let eval_1 = r_j * v_i;
            self.values[2 * i] = v_i - eval_1;
            self.values[2 * i + 1] = eval_1;
        }
        self.len *= 2;
    }

    #[inline]
    fn get(&self, index: usize) -> F {
        debug_assert!(index < self.len);
        self.values[index]
    }
}

impl<F: Field> PrefixSuffixState<F> {
    /// Create a new PrefixSuffix state for an InstructionReadRaf instance.
    ///
    /// Called at instance activation (first active round).
    pub fn new(
        iteration: &Iteration,
        challenges: &[F],
        trace_data: &LookupTraceData,
    ) -> Self {
        let (
            chunk_bits,
            num_phases,
            ra_virtual_log_k_chunk,
            gamma_idx,
            r_reduction,
            output_ra_polys,
            output_combined_val,
        ) = match iteration {
            Iteration::PrefixSuffix {
                chunk_bits,
                num_phases,
                ra_virtual_log_k_chunk,
                gamma,
                r_reduction,
                output_ra_polys,
                output_combined_val,
                ..
            } => (
                *chunk_bits,
                *num_phases,
                *ra_virtual_log_k_chunk,
                *gamma,
                r_reduction,
                output_ra_polys.clone(),
                *output_combined_val,
            ),
            _ => panic!("PrefixSuffixState::new called with non-PrefixSuffix iteration"),
        };

        let gamma = challenges[gamma_idx];
        let gamma_sqr = gamma * gamma;
        let num_cycles = trace_data.lookup_keys.len();
        let m = 1usize << chunk_bits;

        // Build lookup_keys as LookupBits
        let lookup_keys: Vec<LookupBits> = trace_data
            .lookup_keys
            .iter()
            .map(|&k| LookupBits::new(k, LOG_K))
            .collect();

        // Build u_evals = eq(r_reduction, j) for all cycles
        let r_point: Vec<F> = r_reduction.iter().map(|&ci| challenges[ci]).collect();
        let u_evals = EqPolynomial::<F>::evals(&r_point, None);

        // Group cycles by table kind
        let num_tables = LookupTableKind::COUNT;
        let mut lookup_indices_by_table = vec![Vec::new(); num_tables];
        for (j, kind) in trace_data.table_kinds.iter().enumerate() {
            if let Some(k) = kind {
                lookup_indices_by_table[k.index()].push(j);
            }
        }

        // Initialize expanding tables
        let v: Vec<ExpandingTable<F>> = (0..num_phases).map(|_| ExpandingTable::new(m)).collect();

        // Initialize suffix_polys (empty, filled by init_phase)
        let suffix_polys: Vec<Vec<Vec<F>>> = (0..num_tables)
            .map(|t_idx| {
                let table = LookupTables::<XLEN>::from(
                    ALL_TABLE_KINDS[t_idx],
                );
                let suffixes = table.suffixes();
                suffixes.iter().map(|_| vec![F::zero(); m]).collect()
            })
            .collect();

        let empty_q = || [vec![F::zero(); m], vec![F::zero(); m]];

        let mut state = Self {
            chunk_bits,
            num_phases,
            ra_virtual_log_k_chunk,
            gamma,
            gamma_sqr,
            lookup_keys,
            table_kinds: trace_data.table_kinds.clone(),
            lookup_indices_by_table,
            is_interleaved: trace_data.is_interleaved.clone(),
            num_cycles,
            u_evals,
            v,
            r: Vec::with_capacity(LOG_K),
            prefix_checkpoints: vec![PrefixCheckpoint::from(None); NUM_PREFIXES],
            suffix_polys,
            q_left: empty_q(),
            q_right: empty_q(),
            q_identity: empty_q(),
            p_left: [None, None],
            p_right: [None, None],
            p_identity: [None, None],
            registry_checkpoints: [None; 3],
            current_phase: 0,
            total_round: 0,
            output_ra_poly_ids: output_ra_polys,
            output_combined_val_id: output_combined_val,
        };

        state.init_phase(0);
        state
    }

    /// Phase initialization: condense u_evals, build Q, suffix, P polys.
    fn init_phase(&mut self, phase: usize) {
        let log_m = self.chunk_bits;
        let m = 1usize << log_m;
        let m_mask = m - 1;

        // Condensation: multiply u_evals by expanding table from previous phase
        if phase != 0 {
            for (j, k) in self.lookup_keys.iter().enumerate() {
                let (prefix, _) = k.split((self.num_phases - phase) * log_m);
                let k_bound: usize = prefix & m_mask;
                self.u_evals[j] *= self.v[phase - 1].get(k_bound);
            }
        }

        // Build Q polynomials for RAF (left/right/identity) decompositions
        self.init_q_raf(phase);

        // Build per-table suffix accumulators
        self.init_suffix_polys(phase);

        // Build P polynomials from prefix registry
        self.init_p_polys(phase);

        // Reset expanding table for this phase
        self.v[phase].reset();
    }

    /// Fused Q initialization for RAF decompositions.
    ///
    /// Mirrors `PrefixSuffixDecomposition::init_Q_raf` from jolt-core.
    fn init_q_raf(&mut self, phase: usize) {
        let log_m = self.chunk_bits;
        let m = 1usize << log_m;
        let suffix_len = (self.num_phases - phase - 1) * log_m;

        let shift_half: u128 = if suffix_len >= 2 {
            1u128 << (suffix_len / 2)
        } else {
            1
        };
        let shift_full: u128 = if suffix_len > 0 {
            1u128 << suffix_len
        } else {
            1
        };
        let shift_half_f = F::from_u128(shift_half);
        let shift_full_f = F::from_u128(shift_full);

        // Accumulate into 5 buckets: [shift_half, left, right, shift_full, identity]
        let mut acc_sh = vec![F::zero(); m];
        let mut acc_l = vec![F::zero(); m];
        let mut acc_r = vec![F::zero(); m];
        let mut acc_sf = vec![F::zero(); m];
        let mut acc_id = vec![F::zero(); m];

        for (j, k) in self.lookup_keys.iter().enumerate() {
            let (prefix_bits, suffix_bits) = k.split(suffix_len);
            let r_index: usize = prefix_bits & (m - 1);
            let u = self.u_evals[j];

            if self.is_interleaved[j] {
                // Operand path
                acc_sh[r_index] += u;

                let (lo_bits, ro_bits) = suffix_bits.uninterleave();
                let lo: u64 = lo_bits.into();
                if lo != 0 {
                    acc_l[r_index] += u.mul_u64(lo);
                }
                let ro: u64 = ro_bits.into();
                if ro != 0 {
                    acc_r[r_index] += u.mul_u64(ro);
                }
            } else {
                // Identity path
                acc_sf[r_index] += u;

                let id: u128 = suffix_bits.into();
                if id != 0 {
                    acc_id[r_index] += u.mul_u128(id);
                }
            }
        }

        // Apply shift scaling
        if shift_half != 1 {
            for v in &mut acc_sh {
                *v *= shift_half_f;
            }
        }
        if shift_full != 1 {
            for v in &mut acc_sf {
                *v *= shift_full_f;
            }
        }

        // Assign to Q polynomials
        self.q_left = [acc_sh.clone(), acc_l];
        self.q_right = [acc_sh, acc_r];
        self.q_identity = [acc_sf, acc_id];
    }

    /// Per-table suffix accumulation for the read-checking component.
    ///
    /// Mirrors `InstructionReadRafSumcheckProver::init_suffix_polys`.
    fn init_suffix_polys(&mut self, phase: usize) {
        let log_m = self.chunk_bits;
        let m = 1usize << log_m;
        let m_mask = m - 1;

        for (t_idx, indices) in self.lookup_indices_by_table.iter().enumerate() {
            let table = LookupTables::<XLEN>::from(ALL_TABLE_KINDS[t_idx]);
            let suffixes = table.suffixes();
            let num_suffixes = suffixes.len();

            // Zero out
            for s_idx in 0..num_suffixes {
                self.suffix_polys[t_idx][s_idx] = vec![F::zero(); m];
            }

            if indices.is_empty() {
                continue;
            }

            for &j in indices {
                let k = self.lookup_keys[j];
                let (prefix_bits, suffix_bits) =
                    k.split((self.num_phases - 1 - phase) * log_m);
                let idx: usize = prefix_bits & m_mask;
                let u = self.u_evals[j];

                for (s_idx, suffix) in suffixes.iter().enumerate() {
                    match suffix {
                        Suffixes::One => {
                            self.suffix_polys[t_idx][s_idx][idx] += u;
                        }
                        _ if suffix.is_01_valued() => {
                            let t = suffix.suffix_mle::<XLEN>(suffix_bits);
                            debug_assert!(t == 0 || t == 1);
                            if t == 1 {
                                self.suffix_polys[t_idx][s_idx][idx] += u;
                            }
                        }
                        _ => {
                            let t = suffix.suffix_mle::<XLEN>(suffix_bits);
                            if t != 0 {
                                self.suffix_polys[t_idx][s_idx][idx] += u.mul_u64(t);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Build P polynomials from prefix registry checkpoints.
    ///
    /// For each of the 3 RAF families (left operand, right operand, identity),
    /// constructs prefix MLEs over `2^chunk_bits` variables.
    ///
    /// Left/Right operand: P[0] = ShiftHalfPoly (constant = checkpoint * 2^{chunk/2} + i_operand),
    ///                     P[1] = OperandPoly (checkpoint * 2^{chunk/2} + operand(i))
    /// Identity:           P[0] = ShiftFullPoly (constant = checkpoint * 2^{chunk} + i),
    ///                     P[1] = IdentityPoly (checkpoint * 2^{chunk} + i)
    ///
    /// For phase 0, P[0] is None (shift prefix = 1, handled as constant).
    /// For P[1], the polynomial evaluates the operand/identity on the chunk bits.
    fn init_p_polys(&mut self, _phase: usize) {
        let log_m = self.chunk_bits;
        let m = 1usize << log_m;

        // Registry checkpoints: [RightOperand, LeftOperand, Identity]
        let right_cp = self.registry_checkpoints[0];
        let left_cp = self.registry_checkpoints[1];
        let identity_cp = self.registry_checkpoints[2];

        // Build identity P polynomial:
        // P_identity[1](x) = identity_checkpoint * 2^chunk_bits + x  (as integer)
        let id_base = identity_cp.unwrap_or(F::zero()) * F::from_u64(m as u64);
        let p_identity_1: Vec<F> = (0..m).map(|i| id_base + F::from_u64(i as u64)).collect();

        // Build operand P polynomials:
        // P_left[1](x) = left_checkpoint * 2^{chunk/2} + left_operand(x)
        // P_right[1](x) = right_checkpoint * 2^{chunk/2} + right_operand(x)
        let half_m = 1usize << (log_m / 2);
        let left_base = left_cp.unwrap_or(F::zero()) * F::from_u64(half_m as u64);
        let right_base = right_cp.unwrap_or(F::zero()) * F::from_u64(half_m as u64);

        let mut p_left_1 = vec![F::zero(); m];
        let mut p_right_1 = vec![F::zero(); m];
        for i in 0..m {
            let bits = LookupBits::new(i as u128, log_m);
            let (lo, ro) = bits.uninterleave();
            let lo_val: u64 = lo.into();
            let ro_val: u64 = ro.into();
            p_left_1[i] = left_base + F::from_u64(lo_val);
            p_right_1[i] = right_base + F::from_u64(ro_val);
        }

        // P[0] = operand/identity prefix poly, P[1] = None (shift prefix
        // is constant 1 in phase 0; in later phases, shift is folded into Q
        // via init_q_raf). This matches jolt-core's ordering:
        // OperandPolynomial::prefixes() → [Some(OperandPrefix), None]
        // IdentityPolynomial::prefixes() → [Some(IdentityPrefix), None]
        self.p_left = [Some(p_left_1), None];
        self.p_right = [Some(p_right_1), None];
        self.p_identity = [Some(p_identity_1), None];
    }

    /// Compute one round's evaluation vector during the address phase.
    ///
    /// Returns `num_evals` evaluation points (degree-2 → 3 points at {0, 1, 2}).
    /// The caller computes the UniPoly via `from_evals_and_hint(previous_claim, &[eval_0, eval_2])`.
    pub fn compute_address_round(&self) -> [F; 2] {
        let read_checking = self.prover_msg_read_checking(self.total_round);
        let raf = self.prover_msg_raf();

        [read_checking[0] + raf[0], read_checking[1] + raf[1]]
    }

    /// RAF component: combines left/right/identity P×Q products with γ-weights.
    ///
    /// Returns [eval_0, eval_2].
    fn prover_msg_raf(&self) -> [F; 2] {
        let len = self.q_identity[0].len();
        let half = len / 2;

        let mut left_0 = F::zero();
        let mut left_2 = F::zero();
        let mut right_0 = F::zero();
        let mut right_2 = F::zero();

        for b in 0..half {
            let (l0, l2) = Self::ps_sumcheck_evals(&self.p_left, &self.q_left, b, len);
            let (i0, i2) = Self::ps_sumcheck_evals(&self.p_identity, &self.q_identity, b, len);
            let (r0, r2) = Self::ps_sumcheck_evals(&self.p_right, &self.q_right, b, len);

            left_0 += l0;
            left_2 += l2;
            // Combine identity + right (both get γ² weight)
            right_0 += i0 + r0;
            right_2 += i2 + r2;
        }

        [
            self.gamma * left_0 + self.gamma_sqr * right_0,
            self.gamma * left_2 + self.gamma_sqr * right_2,
        ]
    }

    /// Evaluate Σ_i P[i](X) * Q[i](X) at X=0 and X=2 for a given bucket index.
    ///
    /// Uses HighToLow binding: pairs at (index, index + len/2).
    /// If P[i] is None, treat as constant 1.
    fn ps_sumcheck_evals(
        p: &[Option<Vec<F>>; 2],
        q: &[Vec<F>; 2],
        index: usize,
        len: usize,
    ) -> (F, F) {
        let half = len / 2;
        let mut eval_0 = F::zero();
        let mut eval_2_left = F::zero();
        let mut eval_2_right = F::zero();

        for i in 0..2 {
            let (p0, p2) = if let Some(ref p_poly) = p[i] {
                // P evaluated at X=0 and X=2 for HighToLow binding:
                // p(0) = p[index], p(1) = p[index + half]
                // p(2) = 2*p[index+half] - p[index]
                let p_left = p_poly[index];
                let p_right = p_poly[index + half];
                (p_left, p_right + p_right - p_left)
            } else {
                (F::one(), F::one())
            };

            let q_left = q[i][index];
            let q_right = q[i][index + half];

            eval_0 += p0 * q_left;
            eval_2_left += p2 * q_left;
            eval_2_right += p2 * q_right;
        }

        (eval_0, eval_2_right + eval_2_right - eval_2_left)
    }

    /// Read-checking component: per-table prefix×suffix evaluation.
    ///
    /// For each lookup table, evaluates prefix MLEs at X=0 and X=2, combines
    /// with suffix accumulators via table.combine().
    fn prover_msg_read_checking(&self, j: usize) -> [F; 2] {
        // Use current suffix poly length (polys shrink after each bind)
        let current_len = self.suffix_polys.first()
            .and_then(|table_polys| table_polys.first())
            .map_or(0, |sp| sp.len());
        let half = current_len / 2;
        let log_len = current_len.trailing_zeros() as usize;

        let r_x: Option<F> = if j % 2 == 1 {
            self.r.last().copied()
        } else {
            None
        };

        let mut eval_0 = F::zero();
        let mut eval_2_left = F::zero();
        let mut eval_2_right = F::zero();

        for b_val in 0..half {
            let b = LookupBits::new(b_val as u128, log_len - 1);

            // Compute prefix evaluations for ALL prefix types at c=0 and c=2
            let prefixes_c0: Vec<PrefixEval<F>> = ALL_PREFIXES
                .iter()
                .map(|prefix| {
                    prefix.prefix_mle::<XLEN, F, F>(
                        &self.prefix_checkpoints,
                        r_x,
                        0,
                        b,
                        j,
                    )
                })
                .collect();
            let prefixes_c2: Vec<PrefixEval<F>> = ALL_PREFIXES
                .iter()
                .map(|prefix| {
                    prefix.prefix_mle::<XLEN, F, F>(
                        &self.prefix_checkpoints,
                        r_x,
                        2,
                        b,
                        j,
                    )
                })
                .collect();

            // For each table, combine prefix evals with suffix accumulators
            for (t_idx, suffix_polys) in self.suffix_polys.iter().enumerate() {
                let table = LookupTables::<XLEN>::from(ALL_TABLE_KINDS[t_idx]);

                let suffixes_left: Vec<F> =
                    suffix_polys.iter().map(|sp| sp[b_val]).collect();
                let suffixes_right: Vec<F> =
                    suffix_polys.iter().map(|sp| sp[b_val + half]).collect();

                eval_0 += table.combine(&prefixes_c0, &suffixes_left);
                eval_2_left += table.combine(&prefixes_c2, &suffixes_left);
                eval_2_right += table.combine(&prefixes_c2, &suffixes_right);
            }
        }

        [eval_0, eval_2_right + eval_2_right - eval_2_left]
    }

    /// Ingest a challenge and advance state (called after each address round).
    ///
    /// Binds suffix polys, Q polys, P polys, and expanding table.
    /// Updates prefix checkpoints every 2 rounds.
    /// Initializes next phase when crossing phase boundaries.
    pub fn ingest_challenge(&mut self, r_j: F, round: usize) {
        let log_m = self.chunk_bits;
        let phase = round / log_m;

        self.r.push(r_j);

        // Bind suffix polys (HighToLow)
        for table_polys in &mut self.suffix_polys {
            for poly in table_polys {
                bind_in_place(poly, r_j);
            }
        }

        // Bind Q polys (HighToLow)
        for q in [&mut self.q_left, &mut self.q_right, &mut self.q_identity] {
            for poly in q.iter_mut() {
                bind_in_place(poly, r_j);
            }
        }

        // Bind P polys (HighToLow)
        for p in [&mut self.p_left, &mut self.p_right, &mut self.p_identity] {
            for poly in p.iter_mut().flatten() {
                bind_in_place(poly, r_j);
            }
        }

        // Update expanding table
        self.v[phase].update(r_j);

        // Update prefix checkpoints every 2 rounds
        if self.r.len().is_multiple_of(2) {
            let suffix_len = LOG_K - (round / log_m + 1) * log_m;
            Prefixes::update_checkpoints::<XLEN, F, F>(
                &mut self.prefix_checkpoints,
                self.r[self.r.len() - 2],
                self.r[self.r.len() - 1],
                round,
                suffix_len,
            );
        }

        // Check if this is the last round in the phase
        if (round + 1).is_multiple_of(log_m) {
            // Update registry checkpoints from P final claims
            self.registry_checkpoints[0] = Self::p_final_claim(&self.p_right);  // RightOperand
            self.registry_checkpoints[1] = Self::p_final_claim(&self.p_left);   // LeftOperand
            self.registry_checkpoints[2] = Self::p_final_claim(&self.p_identity); // Identity

            if phase != self.num_phases - 1 {
                self.init_phase(phase + 1);
            }
        }

        self.total_round = round + 1;
        self.current_phase = phase;
    }

    /// Extract the final sumcheck claim from a P polynomial pair.
    fn p_final_claim(p: &[Option<Vec<F>>; 2]) -> Option<F> {
        // P[0] is the operand/identity prefix polynomial (the meaningful one).
        // P[1] is None (shift is folded into Q).
        // After binding chunk_bits vars, P[0] reduces to a single scalar.
        if let Some(ref poly) = p[0] {
            if poly.len() == 1 {
                Some(poly[0])
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn total_round(&self) -> usize {
        self.total_round
    }

    /// Whether all address rounds are complete (ready for cycle transition).
    pub fn address_rounds_complete(&self) -> bool {
        self.total_round >= LOG_K
    }

    /// Materialize RA polys and combined_val at the address→cycle transition.
    ///
    /// Returns a map of PolynomialId → Vec<F> for the cycle phase.
    pub fn materialize_outputs(&self) -> HashMap<PolynomialId, Vec<F>> {
        let mut outputs = HashMap::new();
        let log_m = self.chunk_bits;
        let m = 1usize << log_m;
        let m_mask = m - 1;

        // Materialize ra_polys
        let n_vra = LOG_K / self.ra_virtual_log_k_chunk;
        let chunk_size = self.num_phases / n_vra;

        for chunk_i in 0..n_vra {
            let phase_offset = chunk_i * chunk_size;
            let ra: Vec<F> = self
                .lookup_keys
                .iter()
                .map(|k| {
                    let v: u128 = (*k).into();
                    let mut shift = (self.num_phases - 1 - phase_offset) * log_m;
                    let first_idx = ((v >> shift) as usize) & m_mask;
                    let mut acc = self.v[phase_offset].get(first_idx);
                    for phase in (phase_offset + 1)..(phase_offset + chunk_size) {
                        shift -= log_m;
                        let idx = ((v >> shift) as usize) & m_mask;
                        acc *= self.v[phase].get(idx);
                    }
                    acc
                })
                .collect();

            let _ = outputs.insert(self.output_ra_poly_ids[chunk_i], ra);
        }

        // Materialize combined_val_polynomial = Val_j(k) + γ·RafVal_j(k)
        let left_prefix = self.registry_checkpoints[1].unwrap_or(F::zero());
        let right_prefix = self.registry_checkpoints[0].unwrap_or(F::zero());
        let identity_prefix = self.registry_checkpoints[2].unwrap_or(F::zero());
        let raf_interleaved = self.gamma * left_prefix + self.gamma_sqr * right_prefix;
        let raf_identity = self.gamma_sqr * identity_prefix;

        // Compute per-table values at the fully-bound address point.
        // At this point, suffix variable set is empty, so suffix_mle evaluates on empty bits.
        let empty_suffix = LookupBits::new(0, 0);
        let prefix_evals: Vec<PrefixEval<F>> = self
            .prefix_checkpoints
            .iter()
            .map(|cp| cp.unwrap())
            .collect();

        let table_values_at_r_addr: Vec<F> = ALL_TABLE_KINDS
            .iter()
            .map(|&kind| {
                let table = LookupTables::<XLEN>::from(kind);
                let suffix_evals: Vec<F> = table
                    .suffixes()
                    .iter()
                    .map(|suffix| F::from_u64(suffix.suffix_mle::<XLEN>(empty_suffix)))
                    .collect();
                table.combine(&prefix_evals, &suffix_evals)
            })
            .collect();

        let combined_val: Vec<F> = (0..self.num_cycles)
            .map(|j| {
                let mut val = F::zero();
                // Add lookup table value
                if let Some(kind) = self.table_kinds[j] {
                    val += table_values_at_r_addr[kind.index()];
                }
                // Add RAF operand contribution
                if self.is_interleaved[j] {
                    val += raf_interleaved;
                } else {
                    val += raf_identity;
                }
                val
            })
            .collect();

        let _ = outputs.insert(self.output_combined_val_id, combined_val);
        outputs
    }
}

/// HighToLow bind in place: halves the polynomial length.
/// `poly[i] = poly[i] * (1 - r) + poly[i + half] * r`
fn bind_in_place<F: Field>(poly: &mut Vec<F>, r: F) {
    let half = poly.len() / 2;
    for i in 0..half {
        let left = poly[i];
        let right = poly[i + half];
        poly[i] = left + r * (right - left);
    }
    poly.truncate(half);
}

/// All 41 LookupTableKind variants in order, for indexing.
const ALL_TABLE_KINDS: [LookupTableKind; LookupTableKind::COUNT] = [
    LookupTableKind::RangeCheck,
    LookupTableKind::RangeCheckAligned,
    LookupTableKind::And,
    LookupTableKind::Andn,
    LookupTableKind::Or,
    LookupTableKind::Xor,
    LookupTableKind::Equal,
    LookupTableKind::NotEqual,
    LookupTableKind::SignedLessThan,
    LookupTableKind::UnsignedLessThan,
    LookupTableKind::SignedGreaterThanEqual,
    LookupTableKind::UnsignedGreaterThanEqual,
    LookupTableKind::UnsignedLessThanEqual,
    LookupTableKind::UpperWord,
    LookupTableKind::LowerHalfWord,
    LookupTableKind::SignExtendHalfWord,
    LookupTableKind::Movsign,
    LookupTableKind::Pow2,
    LookupTableKind::Pow2W,
    LookupTableKind::ShiftRightBitmask,
    LookupTableKind::VirtualSRL,
    LookupTableKind::VirtualSRA,
    LookupTableKind::VirtualROTR,
    LookupTableKind::VirtualROTRW,
    LookupTableKind::ValidDiv0,
    LookupTableKind::ValidUnsignedRemainder,
    LookupTableKind::ValidSignedRemainder,
    LookupTableKind::VirtualChangeDivisor,
    LookupTableKind::VirtualChangeDivisorW,
    LookupTableKind::HalfwordAlignment,
    LookupTableKind::WordAlignment,
    LookupTableKind::MulUNoOverflow,
    LookupTableKind::VirtualRev8W,
    LookupTableKind::VirtualXORROT32,
    LookupTableKind::VirtualXORROT24,
    LookupTableKind::VirtualXORROT16,
    LookupTableKind::VirtualXORROT63,
    LookupTableKind::VirtualXORROTW16,
    LookupTableKind::VirtualXORROTW12,
    LookupTableKind::VirtualXORROTW8,
    LookupTableKind::VirtualXORROTW7,
];
