//! CPU prefix-suffix decomposition for instruction lookup sumchecks.

use std::collections::HashMap;

use jolt_compiler::kernel_spec::Iteration;
use jolt_compiler::PolynomialId;
use jolt_compute::LookupTraceData;
use jolt_field::Field;
use jolt_instructions::tables::prefixes::{
    PrefixCheckpoint, PrefixEval, Prefixes, ALL_PREFIXES, NUM_PREFIXES,
};
use jolt_instructions::tables::suffixes::Suffixes;
use jolt_instructions::{LookupBits, LookupTableKind, LookupTables};
use jolt_poly::EqPolynomial;

const XLEN: usize = 64;
const LOG_K: usize = 128;

pub struct CpuPrefixSuffixState<F: Field> {
    chunk_bits: usize,
    num_phases: usize,
    ra_virtual_log_k_chunk: usize,
    gamma: F,
    gamma_sqr: F,

    lookup_keys: Vec<LookupBits>,
    table_kinds: Vec<Option<LookupTableKind>>,
    lookup_indices_by_table: Vec<Vec<usize>>,
    is_interleaved: Vec<bool>,
    num_cycles: usize,

    u_evals: Vec<F>,
    v: Vec<ExpandingTable<F>>,
    r: Vec<F>,
    prefix_checkpoints: Vec<PrefixCheckpoint<F>>,
    suffix_polys: Vec<Vec<Vec<F>>>,

    q_left: [Vec<F>; 2],
    q_right: [Vec<F>; 2],
    q_identity: [Vec<F>; 2],

    p_left: [Option<Vec<F>>; 2],
    p_right: [Option<Vec<F>>; 2],
    p_identity: [Option<Vec<F>>; 2],

    registry_checkpoints: [Option<F>; 3],

    current_phase: usize,
    total_round: usize,

    output_ra_poly_ids: Vec<PolynomialId>,
    output_combined_val_id: PolynomialId,
}

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

impl<F: Field> CpuPrefixSuffixState<F> {
    pub fn new(iteration: &Iteration, challenges: &[F], trace_data: &LookupTraceData) -> Self {
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
            _ => panic!("CpuPrefixSuffixState::new called with non-PrefixSuffix iteration"),
        };

        let gamma = challenges[gamma_idx];
        let gamma_sqr = gamma * gamma;
        let num_cycles = trace_data.lookup_keys.len();
        let m = 1usize << chunk_bits;

        let lookup_keys: Vec<LookupBits> = trace_data
            .lookup_keys
            .iter()
            .map(|&k| LookupBits::new(k, LOG_K))
            .collect();

        let r_point: Vec<F> = r_reduction.iter().map(|&ci| challenges[ci]).collect();
        let u_evals = EqPolynomial::<F>::evals(&r_point, None);

        let num_tables = LookupTableKind::COUNT;
        let mut lookup_indices_by_table = vec![Vec::new(); num_tables];
        for (j, kind) in trace_data.table_kinds.iter().enumerate() {
            if let Some(k) = kind {
                lookup_indices_by_table[k.index()].push(j);
            }
        }

        let v: Vec<ExpandingTable<F>> = (0..num_phases).map(|_| ExpandingTable::new(m)).collect();

        let suffix_polys: Vec<Vec<Vec<F>>> = (0..num_tables)
            .map(|t_idx| {
                let table = LookupTables::<XLEN>::from(ALL_TABLE_KINDS[t_idx]);
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

    fn init_phase(&mut self, phase: usize) {
        let log_m = self.chunk_bits;
        let m = 1usize << log_m;
        let m_mask = m - 1;

        if phase != 0 {
            for (j, k) in self.lookup_keys.iter().enumerate() {
                let (prefix, _) = k.split((self.num_phases - phase) * log_m);
                let k_bound: usize = prefix & m_mask;
                self.u_evals[j] *= self.v[phase - 1].get(k_bound);
            }
        }

        self.init_q_raf(phase);
        self.init_suffix_polys(phase);
        self.init_p_polys(phase);
        self.v[phase].reset();
    }

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
                acc_sf[r_index] += u;
                let id: u128 = suffix_bits.into();
                if id != 0 {
                    acc_id[r_index] += u.mul_u128(id);
                }
            }
        }

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

        self.q_left = [acc_sh.clone(), acc_l];
        self.q_right = [acc_sh, acc_r];
        self.q_identity = [acc_sf, acc_id];
    }

    fn init_suffix_polys(&mut self, phase: usize) {
        let log_m = self.chunk_bits;
        let m = 1usize << log_m;
        let m_mask = m - 1;

        for (t_idx, indices) in self.lookup_indices_by_table.iter().enumerate() {
            let table = LookupTables::<XLEN>::from(ALL_TABLE_KINDS[t_idx]);
            let suffixes = table.suffixes();
            let num_suffixes = suffixes.len();

            for s_idx in 0..num_suffixes {
                self.suffix_polys[t_idx][s_idx] = vec![F::zero(); m];
            }

            if indices.is_empty() {
                continue;
            }

            for &j in indices {
                let k = self.lookup_keys[j];
                let (prefix_bits, suffix_bits) = k.split((self.num_phases - 1 - phase) * log_m);
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

    fn init_p_polys(&mut self, _phase: usize) {
        let log_m = self.chunk_bits;
        let m = 1usize << log_m;

        let right_cp = self.registry_checkpoints[0];
        let left_cp = self.registry_checkpoints[1];
        let identity_cp = self.registry_checkpoints[2];

        let id_base = identity_cp.unwrap_or(F::zero()) * F::from_u64(m as u64);
        let p_identity_1: Vec<F> = (0..m).map(|i| id_base + F::from_u64(i as u64)).collect();

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

        self.p_left = [Some(p_left_1), None];
        self.p_right = [Some(p_right_1), None];
        self.p_identity = [Some(p_identity_1), None];
    }

    pub fn compute_address_round(&self) -> [F; 2] {
        let read_checking = self.prover_msg_read_checking(self.total_round);
        let raf = self.prover_msg_raf();
        [read_checking[0] + raf[0], read_checking[1] + raf[1]]
    }

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
            right_0 += i0 + r0;
            right_2 += i2 + r2;
        }

        [
            self.gamma * left_0 + self.gamma_sqr * right_0,
            self.gamma * left_2 + self.gamma_sqr * right_2,
        ]
    }

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

    fn prover_msg_read_checking(&self, j: usize) -> [F; 2] {
        let current_len = self
            .suffix_polys
            .first()
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

            let prefixes_c0: Vec<PrefixEval<F>> = ALL_PREFIXES
                .iter()
                .map(|prefix| {
                    prefix.prefix_mle::<XLEN, F, F>(&self.prefix_checkpoints, r_x, 0, b, j)
                })
                .collect();
            let prefixes_c2: Vec<PrefixEval<F>> = ALL_PREFIXES
                .iter()
                .map(|prefix| {
                    prefix.prefix_mle::<XLEN, F, F>(&self.prefix_checkpoints, r_x, 2, b, j)
                })
                .collect();

            for (t_idx, suffix_polys) in self.suffix_polys.iter().enumerate() {
                let table = LookupTables::<XLEN>::from(ALL_TABLE_KINDS[t_idx]);

                let suffixes_left: Vec<F> = suffix_polys.iter().map(|sp| sp[b_val]).collect();
                let suffixes_right: Vec<F> =
                    suffix_polys.iter().map(|sp| sp[b_val + half]).collect();

                eval_0 += table.combine(&prefixes_c0, &suffixes_left);
                eval_2_left += table.combine(&prefixes_c2, &suffixes_left);
                eval_2_right += table.combine(&prefixes_c2, &suffixes_right);
            }
        }

        [eval_0, eval_2_right + eval_2_right - eval_2_left]
    }

    pub fn ingest_challenge(&mut self, r_j: F) {
        let round = self.total_round;
        let log_m = self.chunk_bits;
        let phase = round / log_m;

        self.r.push(r_j);

        for table_polys in &mut self.suffix_polys {
            for poly in table_polys {
                bind_in_place(poly, r_j);
            }
        }

        for q in [&mut self.q_left, &mut self.q_right, &mut self.q_identity] {
            for poly in q.iter_mut() {
                bind_in_place(poly, r_j);
            }
        }

        for p in [&mut self.p_left, &mut self.p_right, &mut self.p_identity] {
            for poly in p.iter_mut().flatten() {
                bind_in_place(poly, r_j);
            }
        }

        self.v[phase].update(r_j);

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

        if (round + 1).is_multiple_of(log_m) {
            self.registry_checkpoints[0] = Self::p_final_claim(&self.p_right);
            self.registry_checkpoints[1] = Self::p_final_claim(&self.p_left);
            self.registry_checkpoints[2] = Self::p_final_claim(&self.p_identity);

            if phase != self.num_phases - 1 {
                self.init_phase(phase + 1);
            }
        }

        self.total_round = round + 1;
        self.current_phase = phase;
    }

    fn p_final_claim(p: &[Option<Vec<F>>; 2]) -> Option<F> {
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

    pub fn materialize_outputs(&self) -> HashMap<PolynomialId, Vec<F>> {
        let mut outputs = HashMap::new();
        let log_m = self.chunk_bits;
        let m = 1usize << log_m;
        let m_mask = m - 1;

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

        let left_prefix = self.registry_checkpoints[1].unwrap_or(F::zero());
        let right_prefix = self.registry_checkpoints[0].unwrap_or(F::zero());
        let identity_prefix = self.registry_checkpoints[2].unwrap_or(F::zero());
        let raf_interleaved = self.gamma * left_prefix + self.gamma_sqr * right_prefix;
        let raf_identity = self.gamma_sqr * identity_prefix;

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
                if let Some(kind) = self.table_kinds[j] {
                    val += table_values_at_r_addr[kind.index()];
                }
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

fn bind_in_place<F: Field>(poly: &mut Vec<F>, r: F) {
    let half = poly.len() / 2;
    for i in 0..half {
        let left = poly[i];
        let right = poly[i + half];
        poly[i] = left + r * (right - left);
    }
    poly.truncate(half);
}

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
