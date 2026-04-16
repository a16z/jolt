//! Reference prefix-suffix evaluator for testing.
//!
//! Bridges jolt-instructions (prefix/suffix/table definitions) into
//! evaluation methods used by equivalence tests to validate the
//! data-driven runtime.

use jolt_field::Field;
use jolt_instructions::tables::prefixes::{
    PrefixCheckpoint, PrefixEval, Prefixes, ALL_PREFIXES, NUM_PREFIXES,
};
use jolt_instructions::tables::suffixes::Suffixes;
use jolt_instructions::{LookupBits, LookupTableKind, LookupTables};
use jolt_zkvm::runtime::prefix_suffix::PhaseBuffers;

const XLEN: usize = 64;

struct FlatCombineEntry {
    table_idx: usize,
    prefix_idx: Option<usize>,
    suffix_local_idx: usize,
    coefficient: i128,
}

pub struct JoltPrefixSuffixEvaluator {
    suffix_at_empty: Vec<Vec<u64>>,
    combine_matrix: Vec<FlatCombineEntry>,
}

impl Default for JoltPrefixSuffixEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl JoltPrefixSuffixEvaluator {
    pub fn new() -> Self {
        let num_tables = LookupTableKind::COUNT;
        let mut suffix_at_empty = Vec::with_capacity(num_tables);

        let all_kinds = all_table_kinds();
        let mut combine_matrix = Vec::new();
        for (t_idx, &kind) in all_kinds.iter().enumerate() {
            let table = LookupTables::<XLEN>::from(kind);
            let suffixes = table.suffixes();
            let empty = LookupBits::new(0, 0);
            suffix_at_empty.push(
                suffixes
                    .iter()
                    .map(|s| s.suffix_mle::<XLEN>(empty))
                    .collect(),
            );
            for entry in table.combine_entries() {
                combine_matrix.push(FlatCombineEntry {
                    table_idx: t_idx,
                    prefix_idx: entry.prefix.map(|p| p as usize),
                    suffix_local_idx: entry.suffix_idx,
                    coefficient: entry.coefficient,
                });
            }
        }

        Self {
            suffix_at_empty,
            combine_matrix,
        }
    }

    pub fn num_tables(&self) -> usize {
        LookupTableKind::COUNT
    }

    pub fn num_prefixes(&self) -> usize {
        NUM_PREFIXES
    }

    #[allow(clippy::too_many_arguments)]
    pub fn init_phase_buffers<F: Field>(
        &self,
        phase: usize,
        lookup_keys: &[u128],
        u_evals: &[F],
        _table_kind_indices: &[Option<usize>],
        is_interleaved: &[bool],
        lookup_indices_by_table: &[Vec<usize>],
        registry_checkpoints: &[Option<F>; 3],
        chunk_bits: usize,
        num_phases: usize,
    ) -> PhaseBuffers<F> {
        let log_m = chunk_bits;
        let m = 1usize << log_m;
        let m_mask = m - 1;
        let all_kinds = all_table_kinds();
        let num_tables = all_kinds.len();

        let mut suffix_polys: Vec<Vec<Vec<F>>> = Vec::with_capacity(num_tables);
        for t_idx in 0..num_tables {
            let table = LookupTables::<XLEN>::from(all_kinds[t_idx]);
            let suffixes = table.suffixes();
            let num_suffixes = suffixes.len();
            let mut table_polys = Vec::with_capacity(num_suffixes);
            for _ in 0..num_suffixes {
                table_polys.push(vec![F::zero(); m]);
            }

            let indices = &lookup_indices_by_table[t_idx];
            for &j in indices {
                let k = LookupBits::new(lookup_keys[j], 128);
                let (prefix_bits, suffix_bits) = k.split((num_phases - 1 - phase) * log_m);
                let idx: usize = prefix_bits & m_mask;
                let u = u_evals[j];

                for (s_idx, suffix) in suffixes.iter().enumerate() {
                    match suffix {
                        Suffixes::One => {
                            table_polys[s_idx][idx] += u;
                        }
                        _ if suffix.is_01_valued() => {
                            let t = suffix.suffix_mle::<XLEN>(suffix_bits);
                            debug_assert!(t == 0 || t == 1);
                            if t == 1 {
                                table_polys[s_idx][idx] += u;
                            }
                        }
                        _ => {
                            let t = suffix.suffix_mle::<XLEN>(suffix_bits);
                            if t != 0 {
                                table_polys[s_idx][idx] += u.mul_u64(t);
                            }
                        }
                    }
                }
            }
            suffix_polys.push(table_polys);
        }

        let suffix_len = (num_phases - phase - 1) * log_m;
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

        for (j, &key) in lookup_keys.iter().enumerate() {
            let k = LookupBits::new(key, 128);
            let (prefix_bits, suffix_bits) = k.split(suffix_len);
            let r_index: usize = prefix_bits & (m - 1);
            let u = u_evals[j];

            if is_interleaved[j] {
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

        let q_left = [acc_sh.clone(), acc_l];
        let q_right = [acc_sh, acc_r];
        let q_identity = [acc_sf, acc_id];

        let right_cp = registry_checkpoints[0];
        let left_cp = registry_checkpoints[1];
        let identity_cp = registry_checkpoints[2];

        let id_base = identity_cp.unwrap_or(F::zero()) * F::from_u64(m as u64);
        let p_identity = (0..m).map(|i| id_base + F::from_u64(i as u64)).collect();

        let half_m = 1usize << (log_m / 2);
        let left_base = left_cp.unwrap_or(F::zero()) * F::from_u64(half_m as u64);
        let right_base = right_cp.unwrap_or(F::zero()) * F::from_u64(half_m as u64);

        let mut p_left = vec![F::zero(); m];
        let mut p_right = vec![F::zero(); m];
        for i in 0..m {
            let bits = LookupBits::new(i as u128, log_m);
            let (lo, ro) = bits.uninterleave();
            let lo_val: u64 = lo.into();
            let ro_val: u64 = ro.into();
            p_left[i] = left_base + F::from_u64(lo_val);
            p_right[i] = right_base + F::from_u64(ro_val);
        }

        PhaseBuffers {
            suffix_polys,
            q_left,
            q_right,
            q_identity,
            p_left: [Some(p_left), None],
            p_right: [Some(p_right), None],
            p_identity: [Some(p_identity), None],
        }
    }

    pub fn compute_read_checking<F: Field>(
        &self,
        round: usize,
        suffix_polys: &[Vec<Vec<F>>],
        checkpoints: &[Option<F>],
        r_x: Option<F>,
    ) -> [F; 2] {
        let current_len = suffix_polys
            .first()
            .and_then(|table_polys| table_polys.first())
            .map_or(0, |sp| sp.len());
        let half = current_len / 2;
        let log_len = current_len.trailing_zeros() as usize;

        let prefix_cps: Vec<PrefixCheckpoint<F>> = checkpoints
            .iter()
            .map(|&v| PrefixCheckpoint::from(v))
            .collect();

        let coeffs: Vec<F> = self
            .combine_matrix
            .iter()
            .map(|e| F::from_i128(e.coefficient))
            .collect();

        let mut eval_0 = F::zero();
        let mut eval_2_left = F::zero();
        let mut eval_2_right = F::zero();

        for b_val in 0..half {
            let b = LookupBits::new(b_val as u128, log_len - 1);

            let prefixes_c0: Vec<PrefixEval<F>> = ALL_PREFIXES
                .iter()
                .map(|prefix| prefix.prefix_mle::<XLEN, F, F>(&prefix_cps, r_x, 0, b, round))
                .collect();
            let prefixes_c2: Vec<PrefixEval<F>> = ALL_PREFIXES
                .iter()
                .map(|prefix| prefix.prefix_mle::<XLEN, F, F>(&prefix_cps, r_x, 2, b, round))
                .collect();

            for (i, entry) in self.combine_matrix.iter().enumerate() {
                let p_c0 = match entry.prefix_idx {
                    Some(p) => prefixes_c0[p].0,
                    None => F::one(),
                };
                let p_c2 = match entry.prefix_idx {
                    Some(p) => prefixes_c2[p].0,
                    None => F::one(),
                };
                let s_left = suffix_polys[entry.table_idx][entry.suffix_local_idx][b_val];
                let s_right = suffix_polys[entry.table_idx][entry.suffix_local_idx][b_val + half];

                eval_0 += coeffs[i] * p_c0 * s_left;
                eval_2_left += coeffs[i] * p_c2 * s_left;
                eval_2_right += coeffs[i] * p_c2 * s_right;
            }
        }

        [eval_0, eval_2_right + eval_2_right - eval_2_left]
    }

    pub fn update_checkpoints<F: Field>(
        &self,
        checkpoints: &mut [Option<F>],
        r_x: F,
        r_y: F,
        round: usize,
        suffix_len: usize,
    ) {
        let mut prefix_cps: Vec<PrefixCheckpoint<F>> = checkpoints
            .iter()
            .map(|&v| PrefixCheckpoint::from(v))
            .collect();
        Prefixes::update_checkpoints::<XLEN, F, F>(&mut prefix_cps, r_x, r_y, round, suffix_len);
        for (i, cp) in prefix_cps.iter().enumerate() {
            checkpoints[i] = cp.0;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compute_combined_val<F: Field>(
        &self,
        checkpoints: &[Option<F>],
        gamma: F,
        gamma_sqr: F,
        table_kind_indices: &[Option<usize>],
        is_interleaved: &[bool],
        registry_checkpoints: &[Option<F>; 3],
        num_cycles: usize,
    ) -> Vec<F> {
        let left_prefix = registry_checkpoints[1].unwrap_or(F::zero());
        let right_prefix = registry_checkpoints[0].unwrap_or(F::zero());
        let identity_prefix = registry_checkpoints[2].unwrap_or(F::zero());
        let raf_interleaved = gamma * left_prefix + gamma_sqr * right_prefix;
        let raf_identity = gamma_sqr * identity_prefix;

        let prefix_vals: Vec<F> = checkpoints.iter().map(|v| v.unwrap()).collect();

        let num_tables = self.suffix_at_empty.len();
        let mut table_values_at_r_addr = vec![F::zero(); num_tables];
        for entry in &self.combine_matrix {
            let p_val = match entry.prefix_idx {
                Some(p) => prefix_vals[p],
                None => F::one(),
            };
            let s_val = F::from_u64(self.suffix_at_empty[entry.table_idx][entry.suffix_local_idx]);
            let coeff = F::from_i128(entry.coefficient);
            table_values_at_r_addr[entry.table_idx] += coeff * p_val * s_val;
        }

        (0..num_cycles)
            .map(|j| {
                let mut val = F::zero();
                if let Some(t_idx) = table_kind_indices[j] {
                    val += table_values_at_r_addr[t_idx];
                }
                if is_interleaved[j] {
                    val += raf_interleaved;
                } else {
                    val += raf_identity;
                }
                val
            })
            .collect()
    }
}

fn all_table_kinds() -> [LookupTableKind; LookupTableKind::COUNT] {
    [
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
    ]
}
