//! Optimized RAM Hamming-weight booleanity (stage 6b) kernel, porting the
//! legacy `HammingBooleanitySumcheckProver`.
//!
//! The summand is `eq(r_cycle, j) · (H(j)² − H(j))` with `H` the RAM
//! Hamming-weight indicator. Ported techniques:
//!
//! - **Split-eq / Gruen round messages.** The eq factor lives in a
//!   [`GruenSplitEqPolynomial`] (never a dense bound table); per round only
//!   the inner quadratic's constant (`h₀² − h₀`) and leading (`(h₁ − h₀)²`)
//!   coefficients are accumulated and the cubic is reconstructed from
//!   `s(0)+s(1) = previous_claim` — two point-evaluations per pair instead
//!   of the naive tier's four full-summand evaluations plus an eq-table
//!   bind.
//! - **In-place parallel binding** of the single dense `H` table
//!   (`Polynomial::bind_with_order`, rayon inside).
//! - **Block-pattern startup.** The first [`STARTUP_ROUNDS`] rounds run on
//!   `H` packed as one bit pattern per block of low cycle bits: before any
//!   bind, a single pass accumulates each pattern's weight under the
//!   split-eq tail factor, and each startup message is a sum over the
//!   (complement-merged) patterns — no division, exact coefficients. After
//!   the last startup challenge the bound table is read off a subset-sum
//!   lookup indexed by pattern, and the dense Gruen rounds resume on it.
//!   Dense rounds still invert `current_scalar · c_j`
//!   (`gruen_poly_deg_3`), so a zero cycle coordinate past the startup
//!   depth panics as before; the startup rounds themselves accept any
//!   coordinate.
//!
//! Byte parity with the reference kernel holds because field arithmetic is
//! exact: the Gruen-reconstructed evaluations equal the true round
//! polynomial's, and both sides interpolate the same four points.

use jolt_claims::protocols::jolt::{JoltDerivedId, RamHammingBooleanityPublic};
use jolt_claims::NoChallenges;
use jolt_field::JoltField;
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::relations::{
    SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage6b::ram_hamming_booleanity::{
    RamHammingBooleanity, RamHammingBooleanityOutputClaims,
};
use jolt_witness::witnesses::RamHammingWeight;
use jolt_witness::{JoltWitnessPlane, WitnessBundle};

use super::support::{
    collect_rows, map_indices, map_reduce_chunks, pin_derived_term_if_derived, scan_chunk_size,
    RoundProgress,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// Low cycle rounds proved from block patterns before `H` is materialized
/// (a block of `2^STARTUP_ROUNDS` cycles packs into one `u16`).
const STARTUP_ROUNDS: usize = 4;
const _: () = assert!(STARTUP_ROUNDS <= 4);

/// Slot front for the stage-6b RAM Hamming-weight booleanity member.
pub struct OptimizedRamHammingBooleanity;

impl<F: JoltField> PrepareKernel<F, RamHammingBooleanity<F>> for OptimizedRamHammingBooleanity {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamHammingBooleanity<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamHammingBooleanity<F>>>, KernelError<F>>
    {
        let relation = inputs.relation;
        let trace_dimensions = relation.trace_dimensions();
        let stage1_cycle_binding = relation.stage1_cycle_binding();
        if stage1_cycle_binding.len() != trace_dimensions.log_t() {
            return Err(KernelError::InvariantViolation {
                reason: "stage-1 cycle binding has the wrong variable count",
            });
        }
        let rows: Vec<HammingRow> = collect_rows(witness, 1 << trace_dimensions.log_t())?;
        // The verifier's `derive_output_term` pairs the raw sumcheck point
        // against the stage-1 binding positionally, so the eq table's
        // big-endian point is the binding reversed — same orientation as the
        // reference's derived table.
        let eq_point: Vec<F> = stage1_cycle_binding.iter().rev().copied().collect();
        let eq = GruenSplitEqPolynomial::new(&eq_point, BindingOrder::LowToHigh);
        let depth = STARTUP_ROUNDS.min(trace_dimensions.log_t());
        let hamming = if depth == 0 {
            HammingState::Dense(Polynomial::new(
                rows.iter().map(|row| F::from_bool(row.weight.0)).collect(),
            ))
        } else {
            HammingState::Startup(HammingStartup::new(&rows, depth, &eq))
        };

        Ok(Box::new(OptimizedRamHammingBooleanityKernel {
            progress: RoundProgress::new(relation.rounds()),
            eq,
            hamming,
        }))
    }
}

#[derive(Clone, Copy, Debug, WitnessBundle)]
struct HammingRow {
    weight: RamHammingWeight,
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct OptimizedRamHammingBooleanityKernel<F: JoltField> {
    progress: RoundProgress,
    eq: GruenSplitEqPolynomial<F>,
    hamming: HammingState<F>,
}

impl<F: JoltField> OptimizedRamHammingBooleanityKernel<F> {
    fn bind(&mut self, challenge: F) {
        self.eq.bind(challenge);
        match &mut self.hamming {
            HammingState::Startup(startup) => {
                startup.challenges.push(challenge);
                if startup.challenges.len() == startup.depth {
                    self.hamming = HammingState::Dense(startup.materialize());
                }
            }
            HammingState::Dense(hamming) => {
                hamming.bind_with_order(challenge, BindingOrder::LowToHigh);
            }
        }
        self.progress.advance();
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum HammingState<F: JoltField> {
    Startup(HammingStartup<F>),
    Dense(Polynomial<F>),
}

/// `H` during the first `depth` rounds. Block `z` covers cycles
/// `2^depth·z + u`; bit `u` of `patterns[z]` is `H` there. `histogram[p]`
/// sums the tail weight `eq(c_{depth..}, z)` over the blocks whose pattern
/// is `p` or its complement `!p` — both give the same defect, so bins are
/// keyed by the representative with the top bit clear, and bin 0 (constant
/// blocks, zero defect) stays empty.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct HammingStartup<F: JoltField> {
    depth: usize,
    patterns: Vec<u16>,
    histogram: Vec<F>,
    challenges: Vec<F>,
}

impl<F: JoltField> HammingStartup<F> {
    fn new(rows: &[HammingRow], depth: usize, eq: &GruenSplitEqPolynomial<F>) -> Self {
        let width = 1usize << depth;
        let patterns = map_indices(rows.len() / width, |block| {
            rows[block * width..][..width]
                .iter()
                .enumerate()
                .fold(0u16, |pattern, (offset, row)| {
                    pattern | u16::from(row.weight.0) << offset
                })
        });

        let (e_out, e_in) = eq.e_out_in_for_window(depth);
        let in_bits = e_in.len().trailing_zeros();
        let bins = 1usize << (width - 1);
        let histogram = map_reduce_chunks(
            e_out.len(),
            scan_chunk_size(e_out.len()),
            |range| {
                let mut histogram = vec![F::zero(); bins];
                let mut inner = vec![F::zero(); bins];
                if bins <= 128 {
                    for x_out in range {
                        let mut touched = 0u128;
                        let blocks = &patterns[x_out << in_bits..][..e_in.len()];
                        for (&pattern, &weight) in blocks.iter().zip(e_in) {
                            let bin = canonical_bin(pattern, width);
                            if bin == 0 {
                                continue;
                            }
                            if touched >> bin & 1 == 0 {
                                touched |= 1 << bin;
                                inner[bin] = weight;
                            } else {
                                inner[bin] += weight;
                            }
                        }
                        while touched != 0 {
                            let bin = touched.trailing_zeros() as usize;
                            touched &= touched - 1;
                            histogram[bin] += e_out[x_out] * inner[bin];
                        }
                    }
                } else {
                    let mut seen = vec![false; bins];
                    let mut touched = Vec::with_capacity(bins.min(e_in.len()));
                    for x_out in range {
                        let blocks = &patterns[x_out << in_bits..][..e_in.len()];
                        for (&pattern, &weight) in blocks.iter().zip(e_in) {
                            let bin = canonical_bin(pattern, width);
                            if bin == 0 {
                                continue;
                            }
                            if !seen[bin] {
                                seen[bin] = true;
                                touched.push(bin);
                                inner[bin] = weight;
                            } else {
                                inner[bin] += weight;
                            }
                        }
                        for &bin in &touched {
                            histogram[bin] += e_out[x_out] * inner[bin];
                            seen[bin] = false;
                        }
                        touched.clear();
                    }
                }
                histogram
            },
            |mut left, right| {
                for (left, right) in left.iter_mut().zip(right) {
                    *left += right;
                }
                left
            },
            || vec![F::zero(); bins],
        );

        Self {
            depth,
            patterns,
            histogram,
            challenges: Vec::with_capacity(depth),
        }
    }

    /// Round `j = challenges.len()`: `s(t) = l(t)·Q(t)` with `l` the split-eq
    /// linear factor and `Q(t) = Σ_p M[p] Σ_v eq(c_{j+1..depth}, v)·(P² − P)`
    /// at `P = P_p(s_{..j}, t, v)`, the pattern's multilinear extension —
    /// quadratic in `t`, so the coefficients come out exactly.
    fn round_poly(
        &self,
        eq: &GruenSplitEqPolynomial<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let bound = self.challenges.len();
        let reversed: Vec<F> = self.challenges.iter().rev().copied().collect();
        let prefix = EqPolynomial::<F>::evals(&reversed, None);
        let suffix = eq.e_active_for_window(self.depth - bound);
        let [q_0, q_1, q_2] = if self.depth == 4 {
            self.marginalized_coefficients(bound, &prefix, &suffix)
        } else {
            let [mut q_0, mut q_1, mut q_2] = [F::zero(); 3];
            for (pattern, &weight) in self.histogram.iter().enumerate() {
                if weight.is_zero() {
                    continue;
                }
                let [mut p_0, mut p_1, mut p_2] = [F::zero(); 3];
                for (high, &e_high) in suffix.iter().enumerate() {
                    let [mut at_0, mut at_1] = [F::zero(); 2];
                    for (low, &e_low) in prefix.iter().enumerate() {
                        let offset = low | high << (bound + 1);
                        if pattern >> offset & 1 == 1 {
                            at_0 += e_low;
                        }
                        if pattern >> (offset | 1 << bound) & 1 == 1 {
                            at_1 += e_low;
                        }
                    }
                    let delta = at_1 - at_0;
                    p_0 += e_high * (at_0 * at_0 - at_0);
                    p_1 += e_high * delta * (at_0 + at_0 - F::one());
                    p_2 += e_high * delta * delta;
                }
                q_0 += weight * p_0;
                q_1 += weight * p_1;
                q_2 += weight * p_2;
            }
            [q_0, q_1, q_2]
        };
        let (l_0, l_1) = eq.current_linear_evals();
        let slope = l_1 - l_0;
        let actual = l_0 * q_0 + l_1 * (q_0 + q_1 + q_2);
        if actual != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual,
            });
        }
        Ok(UnivariatePoly::new(vec![
            l_0 * q_0,
            l_0 * q_1 + slope * q_0,
            l_0 * q_2 + slope * q_1,
            slope * q_2,
        ]))
    }

    /// Marginalize the 16-bit pattern histogram to the endpoint bits used by
    /// this round before multiplying quadratic defects. The first four rounds
    /// need only 2, 8, 128, and 32,768 complement classes respectively.
    fn marginalized_coefficients(&self, bound: usize, prefix: &[F], suffix: &[F]) -> [F; 3] {
        let endpoint_bits = 1usize << (bound + 1);
        let mask = (1usize << endpoint_bits) - 1;
        let mut reduced = Vec::new();
        let histogram = if bound + 1 == self.depth {
            &self.histogram
        } else {
            reduced.resize(1 << (endpoint_bits - 1), F::zero());
            let mut local = vec![F::zero(); reduced.len()];
            for (high, &e_high) in suffix.iter().enumerate() {
                local.fill(F::zero());
                for (pattern, &weight) in self.histogram.iter().enumerate() {
                    if weight.is_zero() {
                        continue;
                    }
                    let subpattern = (pattern >> (high * endpoint_bits)) & mask;
                    let bin = canonical_bin(subpattern as u16, endpoint_bits);
                    if bin != 0 {
                        local[bin] += weight;
                    }
                }
                for (total, &weight) in reduced.iter_mut().zip(&local) {
                    *total += e_high * weight;
                }
            }
            &reduced
        };

        let mut subset_sums = vec![F::zero(); 1 << prefix.len()];
        for pattern in 1..subset_sums.len() {
            subset_sums[pattern] =
                subset_sums[pattern & (pattern - 1)] + prefix[pattern.trailing_zeros() as usize];
        }
        let endpoint_mask = (1 << prefix.len()) - 1;
        let [mut q_0, mut q_1, mut q_2] = [F::zero(); 3];
        for (pattern, &weight) in histogram.iter().enumerate().skip(1) {
            if weight.is_zero() {
                continue;
            }
            let at_0 = subset_sums[pattern & endpoint_mask];
            let at_1 = subset_sums[(pattern >> prefix.len()) & endpoint_mask];
            let delta = at_1 - at_0;
            q_0 += weight * (at_0 * at_0 - at_0);
            q_1 += weight * delta * (at_0 + at_0 - F::one());
            q_2 += weight * delta * delta;
        }
        [q_0, q_1, q_2]
    }

    /// `H` bound at `s_{..depth}`: a pattern's multilinear extension is the
    /// sum of `eq(s, u)` over its set bits, tabulated for every pattern by
    /// peeling the lowest bit.
    fn materialize(&self) -> Polynomial<F> {
        let reversed: Vec<F> = self.challenges.iter().rev().copied().collect();
        let weights = EqPolynomial::<F>::evals(&reversed, None);
        let mut values = vec![F::zero(); 1 << (1 << self.depth)];
        for pattern in 1..values.len() {
            values[pattern] =
                values[pattern & (pattern - 1)] + weights[pattern.trailing_zeros() as usize];
        }
        Polynomial::new(map_indices(self.patterns.len(), |block| {
            values[usize::from(self.patterns[block])]
        }))
    }
}

/// Histogram bin of a `width`-bit block pattern: the pattern or its
/// complement, whichever has the top bit clear.
fn canonical_bin(pattern: u16, width: usize) -> usize {
    let pattern = usize::from(pattern);
    if pattern >> (width - 1) & 1 == 1 {
        pattern ^ ((1 << width) - 1)
    } else {
        pattern
    }
}

impl<F: JoltField> ProveRounds<F> for OptimizedRamHammingBooleanityKernel<F> {
    fn num_rounds(&self) -> usize {
        self.progress.total()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        let hamming = match &self.hamming {
            HammingState::Startup(startup) => {
                return startup.round_poly(&self.eq, round, previous_claim);
            }
            HammingState::Dense(hamming) => hamming,
        };
        let [constant, leading] = self.eq.par_fold_out_in(
            || [F::zero(); 2],
            |accumulator, row, _x_in, e_in| {
                let (h_0, h_1) = hamming.sumcheck_eval_pair(row, BindingOrder::LowToHigh);
                let delta = h_1 - h_0;
                accumulator[0] += e_in * (h_0 * h_0 - h_0);
                accumulator[1] += e_in * (delta * delta);
            },
            |_x_out, e_out, inner| [e_out * inner[0], e_out * inner[1]],
            |left, right| [left[0] + right[0], left[1] + right[1]],
        );
        Ok(self.eq.gruen_poly_deg_3(constant, leading, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: JoltField> SumcheckKernel<F> for OptimizedRamHammingBooleanityKernel<F> {
    type Relation = RamHammingBooleanity<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<RamHammingBooleanityOutputClaims<F>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        let HammingState::Dense(hamming) = &self.hamming else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "RAM Hamming startup outlived its rounds",
            });
        };
        Ok(RamHammingBooleanityOutputClaims {
            ram_hamming_weight: hamming.evals()[0],
        })
    }

    /// The split-eq scalar (fully bound `EqCycle`) against the verifier's
    /// `derive_output_term` — the same drift detector the naive tier runs on
    /// its hand-materialized derived table.
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &NoChallenges<F>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        pin_derived_term_if_derived(
            relation,
            JoltDerivedId::from(RamHammingBooleanityPublic::EqCycle),
            input_points,
            output_points,
            challenges,
            self.eq.current_scalar(),
        )
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_claims::protocols::jolt::geometry::ram::ram_hamming_weight;
    use jolt_field::{Fr, One, Ring, Zero};
    use jolt_program::execution::{OwnedTrace, TraceRow};
    use jolt_witness::{JoltWitnessOracle, TraceBackend};

    use super::*;
    use crate::optimized::booleanity::testing::{
        load_row, no_op_row, store_row, test_challenge, with_booleanity_backend, with_trace_backend,
    };
    use crate::ReferenceBackend;
    use jolt_verifier::stages::stage6b::ram_hamming_booleanity::RamHammingBooleanityInputClaims;

    fn generic_binding(log_t: usize) -> Vec<Fr> {
        (0..log_t as u64)
            .map(|index| Fr::from_u64(600 + 41 * index))
            .collect()
    }

    /// Lockstep parity drive against the reference kernel: identical round
    /// polynomials every round, identical output claims, and the split-eq
    /// scalar passing the verifier's derived-term cross-check.
    fn parity(backend: &TraceBackend<OwnedTrace>, log_t: usize, stage1_cycle_binding: Vec<Fr>) {
        let relation = RamHammingBooleanity::new(TraceDimensions::new(log_t), stage1_cycle_binding);
        let claims = RamHammingBooleanityInputClaims::default();
        let points = RamHammingBooleanityInputClaims::default();
        let challenges = NoChallenges::default();
        let inputs = || ProverInputs {
            relation: &relation,
            claims: &claims,
            points: &points,
            challenges: &challenges,
        };
        let mut reference = ReferenceBackend
            .prepare(&mut ProofSession::default(), backend, inputs())
            .unwrap();
        let mut optimized = OptimizedRamHammingBooleanity
            .prepare(&mut ProofSession::default(), backend, inputs())
            .unwrap();

        // The Hamming indicator is boolean, so the input claim is zero.
        let mut claim = Fr::from_u64(0);
        let mut bind = None;
        let mut drawn = Vec::new();
        for round in 0..reference.num_rounds() {
            let expected = reference.prove_round(bind, round, claim).unwrap();
            let actual = optimized.prove_round(bind, round, claim).unwrap();
            assert_eq!(expected, actual, "round {round} polynomial mismatch");
            let challenge = test_challenge(round);
            claim = expected.evaluate(challenge);
            drawn.push(challenge);
            bind = Some(challenge);
        }
        if let Some(last) = drawn.last() {
            reference.finish_rounds(*last).unwrap();
            optimized.finish_rounds(*last).unwrap();
        }

        assert_eq!(
            reference.output_claims(&claims).unwrap(),
            optimized.output_claims(&claims).unwrap()
        );
        let output_points = relation.derive_opening_points(&drawn, &points).unwrap();
        reference
            .validate_derived_tables(&relation, &points, &output_points, &challenges)
            .unwrap();
        optimized
            .validate_derived_tables(&relation, &points, &output_points, &challenges)
            .unwrap();
    }

    /// Rows whose Hamming indicator is `bits`, alternating the RAM shapes
    /// that realize each value: nonzero-address loads and stores for ones,
    /// no-ops and address-0 loads (a RAM access, but no Hamming weight) for
    /// zeros.
    fn hamming_rows(bits: &[bool]) -> Vec<TraceRow> {
        bits.iter()
            .enumerate()
            .map(|(cycle, &bit)| {
                let address = 0x8000_1000 + 8 * cycle as u64;
                match (bit, cycle.is_multiple_of(2)) {
                    (true, true) => load_row(address),
                    (true, false) => store_row(address),
                    (false, true) => no_op_row(),
                    (false, false) => load_row(0),
                }
            })
            .collect()
    }

    /// Parity over a trace whose first cycles carry `bits`; the backend pads
    /// the rest with no-op rows.
    fn hamming_parity(log_t: usize, bits: &[bool], stage1_cycle_binding: Vec<Fr>) {
        with_trace_backend(log_t, 4, hamming_rows(bits), |backend, _| {
            let mut expected: Vec<Fr> = bits.iter().map(|&bit| Fr::from_bool(bit)).collect();
            expected.resize(1 << log_t, Fr::from_u64(0));
            assert_eq!(
                JoltWitnessOracle::<Fr>::oracle_table(
                    backend,
                    ram_hamming_weight().polynomial_id()
                )
                .unwrap(),
                expected,
                "fixture rows realize the requested Hamming indicator"
            );
            parity(backend, log_t, stage1_cycle_binding);
        });
    }

    fn pattern_bits(pattern: usize, width: usize) -> impl Iterator<Item = bool> {
        (0..width).map(move |offset| pattern >> offset & 1 == 1)
    }

    /// Compare marginalization against the direct pattern MLE formula, with
    /// both low and high bits set across each round's endpoint windows.
    #[test]
    fn four_round_marginals_match_direct_pattern_formula() {
        let patterns = [
            0u16, 1, 0x0100, 0x00ff, 0x1234, 0x5555, 0x6a95, 0x7fff, 0x8000, 0xffff,
        ];
        for pattern in patterns {
            let representative = canonical_bin(pattern, 16);
            let weight = Fr::from_u64(31 + u64::from(pattern));
            let mut histogram = vec![Fr::zero(); 1 << 15];
            histogram[representative] = weight;
            let mut startup = HammingStartup {
                depth: 4,
                patterns: Vec::new(),
                histogram,
                challenges: Vec::new(),
            };
            for bound in 0..4 {
                startup.challenges = (0..bound).map(|j| Fr::from_u64(9 + j as u64)).collect();
                let reversed: Vec<_> = startup.challenges.iter().rev().copied().collect();
                let prefix = EqPolynomial::<Fr>::evals(&reversed, None);
                let suffix_point: Vec<_> = (0..3 - bound)
                    .map(|j| Fr::from_u64(101 + j as u64))
                    .collect();
                let suffix = EqPolynomial::<Fr>::evals(&suffix_point, None);
                let actual = startup.marginalized_coefficients(bound, &prefix, &suffix);

                let mut expected = [Fr::zero(); 3];
                for (high, &e_high) in suffix.iter().enumerate() {
                    let mut left = Fr::zero();
                    let mut right = Fr::zero();
                    for (low, &e_low) in prefix.iter().enumerate() {
                        let bit = low | high << (bound + 1);
                        if pattern >> bit & 1 == 1 {
                            left += e_low;
                        }
                        if pattern >> (bit | 1 << bound) & 1 == 1 {
                            right += e_low;
                        }
                    }
                    let delta = right - left;
                    expected[0] += weight * e_high * (left * left - left);
                    expected[1] += weight * e_high * delta * (left + left - Fr::one());
                    expected[2] += weight * e_high * delta * delta;
                }
                assert_eq!(actual, expected, "pattern={pattern:#06x}, bound={bound}");
            }
        }
    }

    #[test]
    fn matches_reference() {
        with_booleanity_backend(2, 4, |backend, _| parity(backend, 2, generic_binding(2)));
    }

    #[test]
    fn matches_reference_single_round() {
        with_booleanity_backend(1, 4, |backend, _| parity(backend, 1, generic_binding(1)));
    }

    #[test]
    fn matches_reference_with_padding_rows() {
        with_booleanity_backend(3, 4, |backend, _| parity(backend, 3, generic_binding(3)));
    }

    /// Short traces use startup messages through the final bind. Exercise
    /// every pattern through three rounds and representative 16-bit patterns
    /// at four; enumerating 65,536 full trace backends would obscure the
    /// actual kernel regression this test guards.
    #[test]
    fn every_pattern_within_startup_depth() {
        for log_t in 1..=STARTUP_ROUNDS.min(3) {
            let width = 1 << log_t;
            for pattern in 0..1 << width {
                let bits: Vec<bool> = pattern_bits(pattern, width).collect();
                hamming_parity(log_t, &bits, generic_binding(log_t));
            }
        }
        for pattern in [0, 1, 0x5555, 0xaaaa, 0x8001, 0xffff] {
            let bits: Vec<bool> = pattern_bits(pattern, 16).collect();
            hamming_parity(4, &bits, generic_binding(4));
        }
    }

    /// A trace longer than startup contains varied low and high halves of
    /// 16-bit patterns, followed by dense rounds checked against reference.
    #[test]
    fn every_pattern_above_startup_depth() {
        let width = 1 << STARTUP_ROUNDS;
        let log_t = STARTUP_ROUNDS + 7;
        let bits: Vec<bool> = (0..1 << (log_t - STARTUP_ROUNDS))
            .flat_map(|block| pattern_bits((block * 0x9e37) & 0xffff, width))
            .collect();
        hamming_parity(log_t, &bits, generic_binding(log_t));
    }

    /// Startup rounds never divide by the cycle coordinate, so `0` and `1`
    /// coordinates there are fine (the dense Gruen rounds would invert a
    /// zero one); later coordinates stay generic.
    #[test]
    fn boolean_startup_coordinates() {
        let bits: Vec<bool> = (0..40).map(|cycle| cycle % 3 != 1).collect();
        for log_t in [STARTUP_ROUNDS, STARTUP_ROUNDS + 3] {
            for corner in 0..1usize << STARTUP_ROUNDS {
                let mut binding = generic_binding(log_t);
                for (coordinate, bit) in
                    binding.iter_mut().zip(pattern_bits(corner, STARTUP_ROUNDS))
                {
                    *coordinate = Fr::from_bool(bit);
                }
                hamming_parity(log_t, &bits[..bits.len().min(1 << log_t)], binding);
            }
        }
    }

    #[test]
    fn startup_rejects_inconsistent_claim() {
        let log_t = STARTUP_ROUNDS + 1;
        let rows = hamming_rows(&[true, false, false, true]);
        with_trace_backend(log_t, 4, rows, |backend, _| {
            let relation =
                RamHammingBooleanity::new(TraceDimensions::new(log_t), generic_binding(log_t));
            let claims = RamHammingBooleanityInputClaims::default();
            let points = RamHammingBooleanityInputClaims::default();
            let mut optimized = OptimizedRamHammingBooleanity
                .prepare(
                    &mut ProofSession::default(),
                    backend,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &NoChallenges::default(),
                    },
                )
                .unwrap();
            assert!(matches!(
                optimized.prove_round(None, 0, Fr::from_u64(1)),
                Err(SumcheckError::RoundCheckFailed { round: 0, expected, actual })
                    if expected == Fr::from_u64(1) && actual == Fr::from_u64(0)
            ));
        });
    }
}
