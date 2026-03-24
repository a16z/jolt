//! Sparse read-write matrix evaluator for RAM and register sumchecks.
//!
//! Implements [`SumcheckCompute`] for the sparse entry-list representation
//! used in RAM and register read-write checking. Entries represent non-zero
//! positions in a K×T matrix. Binding merges adjacent entry pairs; message
//! contribution evaluates the formula at each pair.
//!
//! This evaluator handles one phase of the multi-phase sumcheck (either
//! cycle-major or address-major). Phase transitions are managed by
//! [`PhasedEvaluator`](jolt_sumcheck::PhasedEvaluator).

use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::SumcheckCompute;

/// A sparse matrix entry for read-write checking.
///
/// Represents one non-zero position in the (address, cycle) matrix.
/// The `ra` coefficient is the address selector (1 at the correct address),
/// and `val` is the memory/register value at this position.
///
/// `prev_val` and `next_val` carry boundary values for checkpoint threading
/// in address-major binding (where missing entries need default values).
#[derive(Clone, Debug)]
pub struct RwEntry<F> {
    /// Position in the binding dimension (row for cycle-major, col for address-major).
    pub bind_pos: usize,
    /// Position in the free dimension (col for cycle-major, row for address-major).
    pub free_pos: usize,
    /// Address selector coefficient.
    pub ra: F,
    /// Value coefficient (memory/register value at this entry).
    pub val: F,
    /// Boundary value before this entry (used in address-major checkpoint threading).
    pub prev_val: F,
    /// Boundary value after this entry.
    pub next_val: F,
}

/// Formula that computes a sumcheck contribution from an entry pair.
///
/// Given two entries (at positions 2i and 2i+1 in the binding dimension),
/// computes `[f(0), f(2)]` — two evaluation points of the degree-d round
/// polynomial contribution.
pub trait RwFormula<F: Field>: Send + Sync {
    /// Degree of the formula's round polynomial.
    fn degree(&self) -> usize;

    /// Compute round polynomial contributions from an entry pair.
    ///
    /// `even`/`odd`: entries at adjacent binding positions (2i, 2i+1).
    /// Either may be None (missing entry — use default values).
    /// `inc_evals`: `[inc(even_pos), inc(odd_pos)]` from the dense inc polynomial.
    ///
    /// Returns `[eval_at_0, eval_at_2]` (for degree 2) or more points for higher degree.
    fn eval_pair(
        &self,
        even: Option<&RwEntry<F>>,
        odd: Option<&RwEntry<F>>,
        inc_evals: [F; 2],
        default_val: F,
    ) -> Vec<F>;
}

/// Standard RAM read-write formula: `ra · (val + γ · (inc + val))`.
pub struct RamRwFormula<F> {
    pub gamma: F,
}

impl<F: Field> RwFormula<F> for RamRwFormula<F> {
    fn degree(&self) -> usize {
        2
    }

    fn eval_pair(
        &self,
        even: Option<&RwEntry<F>>,
        odd: Option<&RwEntry<F>>,
        inc_evals: [F; 2],
        default_val: F,
    ) -> Vec<F> {
        // Extract coefficients with defaults for missing entries
        let (ra_even, val_even) = match even {
            Some(e) => (e.ra, e.val),
            None => (F::zero(), default_val),
        };
        let (ra_odd, val_odd) = match odd {
            Some(e) => (e.ra, e.val),
            None => (F::zero(), default_val),
        };

        // Linear interpolation coefficients: f(t) = a + t*(b - a)
        let ra_0 = ra_even;
        let ra_slope = ra_odd - ra_even;
        let val_0 = val_even;
        let val_slope = val_odd - val_even;
        let inc_0 = inc_evals[0];
        let inc_slope = inc_evals[1] - inc_evals[0];

        // f(t) = ra(t) · (val(t) + γ · (inc(t) + val(t)))
        //      = ra(t) · ((1+γ)·val(t) + γ·inc(t))
        let one_plus_gamma = F::one() + self.gamma;

        // Evaluate at t=0 and t=2
        let eval_0 = ra_0 * (one_plus_gamma * val_0 + self.gamma * inc_0);
        let ra_2 = ra_0 + ra_slope + ra_slope;
        let val_2 = val_0 + val_slope + val_slope;
        let inc_2 = inc_0 + inc_slope + inc_slope;
        let eval_2 = ra_2 * (one_plus_gamma * val_2 + self.gamma * inc_2);

        vec![eval_0, eval_2]
    }
}

/// Sparse read-write evaluator for one phase of a multi-phase sumcheck.
///
/// Holds a sorted list of sparse entries and evaluates the formula
/// contribution for each entry pair during sumcheck rounds.
pub struct SparseRwEvaluator<F: Field, Fm: RwFormula<F>> {
    /// Sparse entries sorted by `bind_pos`.
    entries: Vec<RwEntry<F>>,
    /// Dense eq polynomial evaluations (weighting).
    eq_evals: Vec<F>,
    /// Dense increment polynomial (size 2^remaining_vars).
    inc: Vec<F>,
    /// Formula for computing per-entry-pair contributions.
    formula: Fm,
    /// Default value for missing entries (e.g., val_init for addresses never accessed).
    default_val: F,
    /// Running claim for `set_claim` optimization.
    claim: F,
}

impl<F: Field, Fm: RwFormula<F>> SparseRwEvaluator<F, Fm> {
    pub fn new(
        entries: Vec<RwEntry<F>>,
        eq_evals: Vec<F>,
        inc: Vec<F>,
        formula: Fm,
        default_val: F,
    ) -> Self {
        Self {
            entries,
            eq_evals,
            inc,
            formula,
            default_val,
            claim: F::zero(),
        }
    }

    /// Number of entries.
    pub fn num_entries(&self) -> usize {
        self.entries.len()
    }
}

impl<F: Field, Fm: RwFormula<F>> SumcheckCompute<F> for SparseRwEvaluator<F, Fm> {
    fn set_claim(&mut self, claim: F) {
        self.claim = claim;
    }

    fn round_polynomial(&self) -> UnivariatePoly<F> {
        let half_eq = self.eq_evals.len() / 2;
        let half_inc = self.inc.len() / 2;
        let degree = self.formula.degree();

        // Initialize evaluation accumulators
        let mut evals = vec![F::zero(); degree + 1];

        // Walk entries sorted by bind_pos. Group by free_pos.
        let mut i = 0;
        while i < self.entries.len() {
            let entry = &self.entries[i];
            let bind = entry.bind_pos;

            // The eq weight for this entry's bind position: eq[bind/2] paired
            let eq_idx = bind / 2;
            if eq_idx >= half_eq {
                i += 1;
                continue;
            }

            // eq interpolation: eq(0) = eq[idx], eq(1) = eq[idx + half]
            let eq_0 = self.eq_evals[eq_idx];
            let eq_1 = self.eq_evals[eq_idx + half_eq];

            // inc interpolation at the bind position
            let inc_idx = bind / 2;
            let inc_0 = if inc_idx < half_inc { self.inc[inc_idx] } else { F::zero() };
            let inc_1 = if inc_idx + half_inc < self.inc.len() {
                self.inc[inc_idx + half_inc]
            } else {
                F::zero()
            };

            // Find the entry pair: even (bind=2k) and odd (bind=2k+1)
            let (actual_even, actual_odd, advance) = if entry.bind_pos.is_multiple_of(2) {
                // Even entry
                let has_odd = i + 1 < self.entries.len()
                    && self.entries[i + 1].bind_pos == entry.bind_pos + 1
                    && self.entries[i + 1].free_pos == entry.free_pos;
                if has_odd {
                    (Some(entry), Some(&self.entries[i + 1]), 2)
                } else {
                    (Some(entry), None, 1)
                }
            } else {
                // Odd entry (no even)
                (None, Some(entry), 1)
            };

            let pair_evals = self.formula.eval_pair(
                actual_even,
                actual_odd,
                [inc_0, inc_1],
                self.default_val,
            );

            // Weight by eq and accumulate
            // eval_at_t = eq(t) * formula(t)
            // eq(t) = eq_0 + t*(eq_1 - eq_0)
            let eq_slope = eq_1 - eq_0;
            evals[0] += eq_0 * pair_evals[0];
            if degree >= 2 && pair_evals.len() > 1 {
                let eq_2 = eq_0 + eq_slope + eq_slope;
                evals[2] += eq_2 * pair_evals[1];
            }

            i += advance;
        }

        // P(1) = claim - P(0)
        UnivariatePoly::from_evals_and_hint(self.claim, &[evals[0], evals[2]])
    }

    fn bind(&mut self, challenge: F) {
        // Bind eq polynomial (HighToLow: pair (j, j+half))
        let half = self.eq_evals.len() / 2;
        for j in 0..half {
            let lo = self.eq_evals[j];
            let hi = self.eq_evals[j + half];
            self.eq_evals[j] = lo + challenge * (hi - lo);
        }
        self.eq_evals.truncate(half);

        // Bind inc polynomial (same convention)
        let half_inc = self.inc.len() / 2;
        for j in 0..half_inc {
            let lo = self.inc[j];
            let hi = self.inc[j + half_inc];
            self.inc[j] = lo + challenge * (hi - lo);
        }
        self.inc.truncate(half_inc);

        // Merge entry pairs: entries at bind_pos 2k and 2k+1 merge to k
        let mut merged = Vec::with_capacity(self.entries.len());
        let mut i = 0;
        while i < self.entries.len() {
            let entry = &self.entries[i];
            let even_pos = (entry.bind_pos / 2) * 2;
            let odd_pos = even_pos + 1;

            if entry.bind_pos.is_multiple_of(2) {
                // Check if next entry is the odd partner
                let has_odd = i + 1 < self.entries.len()
                    && self.entries[i + 1].bind_pos == odd_pos
                    && self.entries[i + 1].free_pos == entry.free_pos;

                if has_odd {
                    let odd = &self.entries[i + 1];
                    merged.push(RwEntry {
                        bind_pos: entry.bind_pos / 2,
                        free_pos: entry.free_pos,
                        ra: entry.ra + challenge * (odd.ra - entry.ra),
                        val: entry.val + challenge * (odd.val - entry.val),
                        prev_val: entry.prev_val,
                        next_val: odd.next_val,
                    });
                    i += 2;
                } else {
                    // Even only (odd is implicit zero ra, default val)
                    merged.push(RwEntry {
                        bind_pos: entry.bind_pos / 2,
                        free_pos: entry.free_pos,
                        ra: entry.ra * (F::one() - challenge),
                        val: entry.val + challenge * (self.default_val - entry.val),
                        prev_val: entry.prev_val,
                        next_val: entry.next_val,
                    });
                    i += 1;
                }
            } else {
                // Odd only (even is implicit zero ra, default val)
                merged.push(RwEntry {
                    bind_pos: entry.bind_pos / 2,
                    free_pos: entry.free_pos,
                    ra: challenge * entry.ra,
                    val: self.default_val + challenge * (entry.val - self.default_val),
                    prev_val: entry.prev_val,
                    next_val: entry.next_val,
                });
                i += 1;
            }
        }
        self.entries = merged;
    }
}

impl<F: Field, Fm: RwFormula<F>> SparseRwEvaluator<F, Fm> {
    /// Returns the remaining sparse entries (for phase transitions).
    pub fn into_entries(self) -> Vec<RwEntry<F>> {
        self.entries
    }

    /// Returns remaining inc polynomial (for phase transitions).
    pub fn inc(&self) -> &[F] {
        &self.inc
    }
}

/// Builds cycle-major sparse entries for RAM read-write checking from a trace.
///
/// Each cycle with a RAM access produces one entry. Entries are sorted by
/// `bind_pos` (= cycle index) for cycle-major binding.
///
/// `padded_len` is the next power of 2 ≥ trace length.
pub fn ram_entries_from_trace<F: Field>(
    trace: &[tracer::instruction::Cycle],
    padded_len: usize,
) -> Vec<RwEntry<F>> {
    use tracer::instruction::RAMAccess;

    let mut entries = Vec::new();
    for (j, cycle) in trace.iter().enumerate() {
        let access = cycle.ram_access();
        match &access {
            RAMAccess::Read(read) => {
                entries.push(RwEntry {
                    bind_pos: j,
                    free_pos: read.address as usize,
                    ra: F::one(),
                    val: F::from_u64(read.value),
                    prev_val: F::from_u64(read.value),
                    next_val: F::from_u64(read.value),
                });
            }
            RAMAccess::Write(write) => {
                entries.push(RwEntry {
                    bind_pos: j,
                    free_pos: write.address as usize,
                    ra: F::one(),
                    val: F::from_u64(write.pre_value),
                    prev_val: F::from_u64(write.pre_value),
                    next_val: F::from_u64(write.post_value),
                });
            }
            RAMAccess::NoOp => {}
        }
    }
    // Pad positions are within [trace.len()..padded_len) — no entries (implicit zeros)
    let _ = padded_len;
    entries
}

/// Converts cycle-major entries to address-major entries.
///
/// Swaps `bind_pos` and `free_pos`, then sorts by `bind_pos` (now address).
pub fn to_address_major<F: Field + Ord>(mut entries: Vec<RwEntry<F>>) -> Vec<RwEntry<F>> {
    for entry in &mut entries {
        std::mem::swap(&mut entry.bind_pos, &mut entry.free_pos);
    }
    entries.sort_by_key(|e| (e.bind_pos, e.free_pos));
    entries
}

/// Materializes sparse entries into dense polynomial tables.
///
/// Returns `(ra_table, val_table)` each of size `domain_size`.
/// Missing positions get `ra=0, val=default_val`.
pub fn materialize_dense<F: Field>(
    entries: &[RwEntry<F>],
    domain_size: usize,
    default_val: F,
) -> (Vec<F>, Vec<F>) {
    let mut ra = vec![F::zero(); domain_size];
    let mut val = vec![default_val; domain_size];
    for entry in entries {
        if entry.bind_pos < domain_size {
            ra[entry.bind_pos] = entry.ra;
            val[entry.bind_pos] = entry.val;
        }
    }
    (ra, val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr};
    use jolt_poly::EqPolynomial;
    use jolt_sumcheck::{SumcheckClaim, SumcheckProver, SumcheckVerifier};
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use num_traits::{One, Zero};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    /// Dense reference: Σ_x eq(r, x) · ra(x) · ((1+γ)·val(x) + γ·inc(x))
    fn dense_ram_rw_sum(eq: &[Fr], ra: &[Fr], val: &[Fr], inc: &[Fr], gamma: Fr) -> Fr {
        let one_plus_gamma = Fr::one() + gamma;
        eq.iter()
            .zip(ra.iter())
            .zip(val.iter())
            .zip(inc.iter())
            .map(|(((&e, &r), &v), &i)| e * r * (one_plus_gamma * v + gamma * i))
            .sum()
    }

    #[test]
    fn sparse_rw_single_entry() {
        let num_vars = 3;
        let size = 1 << num_vars;
        let gamma = Fr::from_u64(7);

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq = EqPolynomial::new(r).evaluations();

        // One entry at position 3 (bind_pos=3)
        let entry = RwEntry {
            bind_pos: 3,
            free_pos: 0,
            ra: Fr::from_u64(1),
            val: Fr::from_u64(42),
            prev_val: Fr::zero(),
            next_val: Fr::zero(),
        };

        // Dense reference: ra[3]=1, val[3]=42, all others zero
        let mut ra_dense = vec![Fr::zero(); size];
        let mut val_dense = vec![Fr::zero(); size];
        ra_dense[3] = Fr::from_u64(1);
        val_dense[3] = Fr::from_u64(42);
        let inc = vec![Fr::zero(); size]; // no increments

        let claimed_sum = dense_ram_rw_sum(&eq, &ra_dense, &val_dense, &inc, gamma);

        let mut evaluator = SparseRwEvaluator::new(
            vec![entry],
            eq.clone(),
            inc,
            RamRwFormula { gamma },
            Fr::zero(),
        );

        let claim = SumcheckClaim {
            num_vars,
            degree: 3, // ra * val = degree 2, + eq = 3
            claimed_sum,
        };

        let mut pt = Blake2bTranscript::new(b"sparse_single");
        let proof = SumcheckProver::prove(&claim, &mut evaluator, &mut pt);

        let mut vt = Blake2bTranscript::new(b"sparse_single");
        let result = SumcheckVerifier::verify(&claim, &proof, &mut vt);
        assert!(result.is_ok(), "sumcheck verification failed: {result:?}");
    }

    #[test]
    fn sparse_rw_multiple_entries() {
        let num_vars = 4;
        let size = 1 << num_vars;
        let gamma = Fr::from_u64(3);

        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq = EqPolynomial::new(r).evaluations();

        // Entries at positions 1, 5, 9, 13 (spread across the domain)
        let entries = vec![
            RwEntry { bind_pos: 1, free_pos: 0, ra: Fr::from_u64(1), val: Fr::from_u64(10), prev_val: Fr::zero(), next_val: Fr::zero() },
            RwEntry { bind_pos: 5, free_pos: 0, ra: Fr::from_u64(1), val: Fr::from_u64(20), prev_val: Fr::zero(), next_val: Fr::zero() },
            RwEntry { bind_pos: 9, free_pos: 0, ra: Fr::from_u64(1), val: Fr::from_u64(30), prev_val: Fr::zero(), next_val: Fr::zero() },
            RwEntry { bind_pos: 13, free_pos: 0, ra: Fr::from_u64(1), val: Fr::from_u64(40), prev_val: Fr::zero(), next_val: Fr::zero() },
        ];

        // Dense reference
        let mut ra_dense = vec![Fr::zero(); size];
        let mut val_dense = vec![Fr::zero(); size];
        for e in &entries {
            ra_dense[e.bind_pos] = e.ra;
            val_dense[e.bind_pos] = e.val;
        }
        let inc = vec![Fr::zero(); size];

        let claimed_sum = dense_ram_rw_sum(&eq, &ra_dense, &val_dense, &inc, gamma);

        let mut evaluator = SparseRwEvaluator::new(
            entries,
            eq.clone(),
            inc,
            RamRwFormula { gamma },
            Fr::zero(),
        );

        let claim = SumcheckClaim {
            num_vars,
            degree: 3,
            claimed_sum,
        };

        let mut pt = Blake2bTranscript::new(b"sparse_multi");
        let proof = SumcheckProver::prove(&claim, &mut evaluator, &mut pt);

        let mut vt = Blake2bTranscript::new(b"sparse_multi");
        let result = SumcheckVerifier::verify(&claim, &proof, &mut vt);
        assert!(result.is_ok(), "sumcheck verification failed: {result:?}");
    }
}
