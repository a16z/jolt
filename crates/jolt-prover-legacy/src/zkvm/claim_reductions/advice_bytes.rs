//! The 7x advice-bytes reconstruction sumcheck (lattice/packed mode only).
//!
//! One sumcheck over the untrusted-advice byte one-hot column's
//! `(symbol ‖ limb ‖ word)` cell domain, γ-batching three legs (see
//! `UntrustedAdviceReconstruction` in jolt-claims):
//!
//! - booleanity (`γ⁰`): `eq(cell, r_ref) · (B² − B)` sums to zero,
//! - hamming (`γ¹`): `eq((limb,word), r_ref_lw) · B` sums to one,
//! - word reconstruction (`γ²`): `id(symbol) · 256^limb · eq(word, r_word) ·
//!   B` sums to the completed untrusted-advice word claim.
//!
//! The reference point `r_ref` is drawn fresh over the cell domain before
//! the instance gamma; the word kernel binds the completed advice claim's
//! own point. The column opening produced here is the leaf claim the packed
//! stage-8 `UntrustedAdviceOneHot` statement consumes.

use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use jolt_claims::protocols::jolt::lattice::geometry::{word_byte_num_vars, BYTE_BITS, WORD_BYTES};
use rayon::prelude::*;

use crate::field::JoltField;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId, BIG_ENDIAN,
    LITTLE_ENDIAN,
};
use crate::poly::unipoly::UniPoly;
#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::{InputClaimConstraint, OutputClaimConstraint};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceParams;
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::zkvm::claim_reductions::advice::AdviceKind;

const DEGREE_BOUND: usize = 3;

#[derive(Allocative, Clone)]
pub struct UntrustedAdviceReconstructionSumcheckParams<F: JoltField> {
    pub word_vars: usize,
    /// The fresh reference point over the cell domain (msb-first), drawn
    /// before `gamma`.
    pub r_reference: Vec<F::Challenge>,
    pub gamma: F,
    /// The completed untrusted-advice word claim's point (the word kernel).
    pub r_word: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> UntrustedAdviceReconstructionSumcheckParams<F> {
    pub fn new(
        word_vars: usize,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let r_reference = transcript.challenge_vector_optimized::<F>(word_byte_num_vars(word_vars));
        let gamma: F = transcript.challenge_scalar();
        let (r_word, _) = accumulator
            .get_advice_opening(AdviceKind::Untrusted, SumcheckId::AdviceClaimReduction)
            .expect("completed untrusted advice claim must exist before the 7x phase");
        Self {
            word_vars,
            r_reference,
            gamma,
            r_word,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for UntrustedAdviceReconstructionSumcheckParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        word_byte_num_vars(self.word_vars)
    }

    /// Booleanity sums to zero, hamming to one, reconstruction to the
    /// completed word claim: `γ + γ²·A(r_word)`.
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, word_claim) = accumulator
            .get_advice_opening(AdviceKind::Untrusted, SumcheckId::AdviceClaimReduction)
            .expect("completed untrusted advice claim must exist before the 7x phase");
        self.gamma + self.gamma * self.gamma * word_claim
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        unimplemented!(
            "zk x lattice is rejected fail-closed; UntrustedAdviceReconstruction carries no BlindFold plumbing"
        )
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(&self, _: &dyn OpeningAccumulator<F>) -> Vec<F> {
        unimplemented!(
            "zk x lattice is rejected fail-closed; UntrustedAdviceReconstruction carries no BlindFold plumbing"
        )
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        unimplemented!(
            "zk x lattice is rejected fail-closed; UntrustedAdviceReconstruction carries no BlindFold plumbing"
        )
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, _sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        unimplemented!(
            "zk x lattice is rejected fail-closed; UntrustedAdviceReconstruction carries no BlindFold plumbing"
        )
    }
}

/// The prover never materializes the three cell-domain tables the relation is
/// phrased over — at 1 MB advice those would be ~4 GiB *per kernel*. Instead
/// it exploits their structure (the same one-hot machinery the trace columns
/// use, adapted to this instance's rows-first binding order):
///
/// - the column is a strict K=256 one-hot, held as per-row sparse lane
///   weights that start singleton and only densify as row rounds merge rows;
/// - `eq(cell, r_ref)` factors as `eq_byte ⊗ eq_row`, and the hamming kernel
///   is literally the same `eq_row` table;
/// - the decode kernel factors as `value(byte) · (256^limb · eq_word)`.
///
/// The round polynomials are the same multilinear-extension sums as the dense
/// form — same degree, rounds, and transcript — computed per row pair over
/// the union of the two sparse lane supports (absent lanes contribute zero to
/// both `kb·(B²−B)` and `kl·B`). After the `3 + word_vars` row rounds the
/// state collapses to three 256-entry lane tables and binds densely.
#[derive(Allocative)]
pub struct UntrustedAdviceReconstructionSumcheckProver<F: JoltField> {
    phase: Phase<F>,
    pub params: UntrustedAdviceReconstructionSumcheckParams<F>,
}

#[derive(Allocative)]
enum Phase<F: JoltField> {
    /// The first `3 + word_vars` rounds bind the `(limb ‖ word)` row
    /// variables (the cell domain's low bits under LowToHigh binding).
    Rows {
        /// Per merged row, the nonzero `(lane, weight)` pairs sorted by lane.
        rows: Vec<Vec<(u8, F)>>,
        /// `eq(row, r_ref_lw)` — the booleanity kernel's row factor and,
        /// γ-scaled, the whole hamming kernel.
        eq_row: MultilinearPolynomial<F>,
        /// `256^limb · eq(word, r_word)` — the decode kernel's row factor.
        pw_row: MultilinearPolynomial<F>,
        /// `eq(byte, r_ref_byte)` over the 256 lanes; fixed until the lane
        /// rounds.
        eq_byte: Vec<F>,
        /// `γ²·value(byte)` over the 256 lanes — the decode kernel's lane
        /// factor, γ²-scaled once here.
        value_byte: Vec<F>,
    },
    /// The last `BYTE_BITS` rounds bind the byte-lane variables densely over
    /// 256-entry tables (the shape the dense implementation had, collapsed).
    Lanes {
        bytes: MultilinearPolynomial<F>,
        k_bool: MultilinearPolynomial<F>,
        k_lin: MultilinearPolynomial<F>,
    },
}

/// Walks the union of two lane-sorted sparse rows, yielding each lane with
/// its two weights (zero where absent). Both the round-message accumulation
/// and the bind-merge consume the exact per-lane sequence the dense
/// implementation saw, so sharing the walk keeps the two consensus-critical
/// loops from drifting apart.
fn for_each_lane_union<F: JoltField>(
    lo: &[(u8, F)],
    hi: &[(u8, F)],
    mut visit: impl FnMut(u8, F, F),
) {
    let (mut i, mut j) = (0, 0);
    while i < lo.len() || j < hi.len() {
        // Lanes are u8, so u16::MAX is a safe exhausted-side sentinel.
        let l0 = lo.get(i).map_or(u16::MAX, |&(lane, _)| lane as u16);
        let l1 = hi.get(j).map_or(u16::MAX, |&(lane, _)| lane as u16);
        match l0.cmp(&l1) {
            std::cmp::Ordering::Less => {
                visit(lo[i].0, lo[i].1, F::zero());
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                visit(hi[j].0, F::zero(), hi[j].1);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                visit(lo[i].0, lo[i].1, hi[j].1);
                i += 1;
                j += 1;
            }
        }
    }
}

impl<F: JoltField> UntrustedAdviceReconstructionSumcheckProver<F> {
    #[tracing::instrument(
        skip_all,
        name = "UntrustedAdviceReconstructionSumcheckProver::initialize"
    )]
    pub fn initialize(
        params: UntrustedAdviceReconstructionSumcheckParams<F>,
        words: &[u64],
    ) -> Self {
        let word_vars = params.word_vars;
        let limb_bits = WORD_BYTES.log_2();
        let cell_vars = word_byte_num_vars(word_vars);
        debug_assert!(words.len() <= 1 << word_vars);
        debug_assert_eq!(cell_vars, BYTE_BITS + limb_bits + word_vars);

        // Rows past `words.len()` encode byte 0 hot, matching the committed
        // column's padding (the shifted-zero encoding the hamming leg needs).
        let rows: Vec<Vec<(u8, F)>> = (0..WORD_BYTES << word_vars)
            .map(|row| {
                let word_index = row & ((1 << word_vars) - 1);
                let limb = row >> word_vars;
                let byte = words
                    .get(word_index)
                    .map_or(0, |word| (word >> (8 * limb)) as u8);
                vec![(byte, F::one())]
            })
            .collect();

        let eq_byte = EqPolynomial::<F>::evals(&params.r_reference[..BYTE_BITS]);
        let eq_row = EqPolynomial::<F>::evals(&params.r_reference[BYTE_BITS..]);
        let eq_word = EqPolynomial::<F>::evals(&params.r_word.r);
        let places: Vec<F> = (0..WORD_BYTES)
            .map(|limb| F::from_u64(1u64 << (8 * limb)))
            .collect();
        let pw_row = (0..WORD_BYTES << word_vars)
            .into_par_iter()
            .map(|row| {
                let word_index = row & ((1 << word_vars) - 1);
                let limb = row >> word_vars;
                places[limb] * eq_word[word_index]
            })
            .collect::<Vec<F>>();
        let gamma_squared = params.gamma * params.gamma;
        let value_byte: Vec<F> = (0..1u64 << BYTE_BITS)
            .map(|lane| gamma_squared * F::from_u64(lane))
            .collect();

        Self {
            phase: Phase::Rows {
                rows,
                eq_row: MultilinearPolynomial::from(eq_row),
                pw_row: MultilinearPolynomial::from(pw_row),
                eq_byte,
                value_byte,
            },
            params,
        }
    }

    /// Collapses the fully-row-bound state to the three dense 256-entry lane
    /// tables the last `BYTE_BITS` rounds bind.
    fn transition_to_lanes(&mut self) {
        let Phase::Rows {
            rows,
            eq_row,
            pw_row,
            eq_byte,
            value_byte,
        } = &self.phase
        else {
            unreachable!("the transition fires exactly once, at the end of the row rounds");
        };
        debug_assert_eq!(rows.len(), 1);
        let e_row = eq_row.final_sumcheck_claim();
        let p_row = pw_row.final_sumcheck_claim();
        let mut bytes = vec![F::zero(); 1 << BYTE_BITS];
        for &(lane, weight) in &rows[0] {
            bytes[lane as usize] = weight;
        }
        let gamma_e = self.params.gamma * e_row;
        let k_bool: Vec<F> = eq_byte.iter().map(|eq| *eq * e_row).collect();
        let k_lin: Vec<F> = value_byte
            .iter()
            .map(|value| gamma_e + *value * p_row)
            .collect();
        self.phase = Phase::Lanes {
            bytes: MultilinearPolynomial::from(bytes),
            k_bool: MultilinearPolynomial::from(k_bool),
            k_lin: MultilinearPolynomial::from(k_lin),
        };
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for UntrustedAdviceReconstructionSumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(
        skip_all,
        name = "UntrustedAdviceReconstructionSumcheckProver::compute_message"
    )]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let term = |b: F, kb: F, kl: F| kb * (b.square() - b) + kl * b;
        let evals = match &self.phase {
            Phase::Rows {
                rows,
                eq_row,
                pw_row,
                eq_byte,
                value_byte,
            } => {
                let gamma = self.params.gamma;
                (0..rows.len() / 2)
                    .into_par_iter()
                    .map(|p| {
                        let e = eq_row.sumcheck_evals_array::<3>(p, BindingOrder::LowToHigh);
                        let pw = pw_row.sumcheck_evals_array::<3>(p, BindingOrder::LowToHigh);
                        // The hamming leg's per-pair factor, hoisted out of
                        // the lane walk.
                        let gamma_e = [gamma * e[0], gamma * e[1], gamma * e[2]];
                        let mut acc = [F::zero(); 3];
                        for_each_lane_union(&rows[2 * p], &rows[2 * p + 1], |lane, b0, b1| {
                            let eq_b = eq_byte[lane as usize];
                            let value = value_byte[lane as usize];
                            let b_delta = b1 - b0;
                            let (b2, b3) = (b1 + b_delta, b1 + b_delta + b_delta);
                            // `b0` is exactly zero for lanes hot only in the
                            // odd half; the point-0 term is then an exact
                            // zero and is skipped.
                            if !b0.is_zero() {
                                acc[0] += term(b0, eq_b * e[0], gamma_e[0] + value * pw[0]);
                            }
                            acc[1] += term(b2, eq_b * e[1], gamma_e[1] + value * pw[1]);
                            acc[2] += term(b3, eq_b * e[2], gamma_e[2] + value * pw[2]);
                        });
                        acc
                    })
                    .reduce(
                        || [F::zero(); 3],
                        |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
                    )
            }
            Phase::Lanes {
                bytes,
                k_bool,
                k_lin,
            } => (0..bytes.len() / 2)
                .into_par_iter()
                .map(|g| {
                    let b = bytes.sumcheck_evals_array::<3>(g, BindingOrder::LowToHigh);
                    let kb = k_bool.sumcheck_evals_array::<3>(g, BindingOrder::LowToHigh);
                    let kl = k_lin.sumcheck_evals_array::<3>(g, BindingOrder::LowToHigh);
                    [
                        term(b[0], kb[0], kl[0]),
                        term(b[1], kb[1], kl[1]),
                        term(b[2], kb[2], kl[2]),
                    ]
                })
                .reduce(
                    || [F::zero(); 3],
                    |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
                ),
        };
        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    #[tracing::instrument(
        skip_all,
        name = "UntrustedAdviceReconstructionSumcheckProver::ingest_challenge"
    )]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        match &mut self.phase {
            Phase::Rows {
                rows,
                eq_row,
                pw_row,
                ..
            } => {
                let merged_rows: Vec<Vec<(u8, F)>> = (0..rows.len() / 2)
                    .into_par_iter()
                    .map(|p| {
                        let (lo, hi) = (&rows[2 * p], &rows[2 * p + 1]);
                        let mut merged = Vec::with_capacity(lo.len() + hi.len());
                        for_each_lane_union(lo, hi, |lane, w0, w1| {
                            merged.push((lane, w0 + (w1 - w0) * r_j));
                        });
                        merged
                    })
                    .collect();
                *rows = merged_rows;
                eq_row.bind_parallel(r_j, BindingOrder::LowToHigh);
                pw_row.bind_parallel(r_j, BindingOrder::LowToHigh);
                if rows.len() == 1 {
                    self.transition_to_lanes();
                }
            }
            Phase::Lanes {
                bytes,
                k_bool,
                k_lin,
            } => {
                bytes.bind_parallel(r_j, BindingOrder::LowToHigh);
                k_bool.bind_parallel(r_j, BindingOrder::LowToHigh);
                k_lin.bind_parallel(r_j, BindingOrder::LowToHigh);
            }
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let Phase::Lanes { bytes, .. } = &self.phase else {
            unreachable!("cache_openings runs after every round is bound");
        };
        accumulator.append_untrusted_advice(
            SumcheckId::UntrustedAdviceReconstruction,
            self.params.normalize_opening_point(sumcheck_challenges),
            bytes.final_sumcheck_claim(),
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// The trusted-advice byte reconstruction: the standalone degree-2
/// reconstruction leg (`id(symbol) · 256^limb · eq(word, r_word) · B`) over
/// the precommitted trusted byte column — no booleanity or hamming legs (the
/// trusted committer attests the encoding), no drawn challenges.
#[derive(Allocative, Clone)]
pub struct TrustedAdviceReconstructionSumcheckParams<F: JoltField> {
    pub word_vars: usize,
    /// The completed trusted-advice word claim's point (the word kernel).
    pub r_word: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> TrustedAdviceReconstructionSumcheckParams<F> {
    pub fn new(word_vars: usize, accumulator: &dyn OpeningAccumulator<F>) -> Self {
        let (r_word, _) = accumulator
            .get_advice_opening(AdviceKind::Trusted, SumcheckId::AdviceClaimReduction)
            .expect("completed trusted advice claim must exist before the 7x phase");
        Self { word_vars, r_word }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for TrustedAdviceReconstructionSumcheckParams<F> {
    fn degree(&self) -> usize {
        2
    }

    /// Only the `(byte ‖ place)` variables bind; the word point stays fixed
    /// by the incoming claim (the byte column is pre-bound at `r_word`).
    fn num_rounds(&self) -> usize {
        word_byte_num_vars(0)
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        accumulator
            .get_advice_opening(AdviceKind::Trusted, SumcheckId::AdviceClaimReduction)
            .expect("completed trusted advice claim must exist before the 7x phase")
            .1
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        unimplemented!(
            "zk x lattice is rejected fail-closed; TrustedAdviceReconstruction carries no BlindFold plumbing"
        )
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(&self, _: &dyn OpeningAccumulator<F>) -> Vec<F> {
        unimplemented!(
            "zk x lattice is rejected fail-closed; TrustedAdviceReconstruction carries no BlindFold plumbing"
        )
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        unimplemented!(
            "zk x lattice is rejected fail-closed; TrustedAdviceReconstruction carries no BlindFold plumbing"
        )
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, _sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        unimplemented!(
            "zk x lattice is rejected fail-closed; TrustedAdviceReconstruction carries no BlindFold plumbing"
        )
    }
}

#[derive(Allocative)]
pub struct TrustedAdviceReconstructionSumcheckProver<F: JoltField> {
    /// The byte one-hot column pre-bound at `r_word` over the word
    /// variables: a scatter of `eq(r_word, ·)` into the `(byte ‖ place)`
    /// cells.
    bytes: MultilinearPolynomial<F>,
    /// The decode kernel `byte · 256^place`.
    kernel: MultilinearPolynomial<F>,
    pub params: TrustedAdviceReconstructionSumcheckParams<F>,
}

impl<F: JoltField> TrustedAdviceReconstructionSumcheckProver<F> {
    #[tracing::instrument(
        skip_all,
        name = "TrustedAdviceReconstructionSumcheckProver::initialize"
    )]
    pub fn initialize(params: TrustedAdviceReconstructionSumcheckParams<F>, words: &[u64]) -> Self {
        let word_vars = params.word_vars;
        let limb_bits = WORD_BYTES.log_2();
        let cell_vars = word_byte_num_vars(0);
        debug_assert!(words.len() <= 1 << word_vars);
        debug_assert_eq!(params.r_word.r.len(), word_vars);

        // Zero-padded words scatter byte 0 (the witness encodes padding as
        // symbol-0 hot), so the column matches the committed one exactly.
        let eq_word = EqPolynomial::<F>::evals(&params.r_word.r);
        let mut bytes = vec![F::zero(); 1 << cell_vars];
        for limb in 0..WORD_BYTES {
            for word_index in 0..(1usize << word_vars) {
                let byte = words
                    .get(word_index)
                    .map_or(0, |word| (word >> (8 * limb)) as u8)
                    as usize;
                bytes[(byte << limb_bits) | limb] += eq_word[word_index];
            }
        }

        let kernel = (0..1usize << cell_vars)
            .into_par_iter()
            .map(|cell| {
                let limb = cell & (WORD_BYTES - 1);
                let symbol = cell >> limb_bits;
                F::from_u64(symbol as u64) * F::from_u64(1u64 << (8 * limb))
            })
            .collect::<Vec<F>>();

        Self {
            bytes: MultilinearPolynomial::from(bytes),
            kernel: MultilinearPolynomial::from(kernel),
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for TrustedAdviceReconstructionSumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(
        skip_all,
        name = "TrustedAdviceReconstructionSumcheckProver::compute_message"
    )]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let half = self.bytes.len() / 2;
        let [eval_0, eval_2] = (0..half)
            .into_par_iter()
            .map(|g| {
                let b0 = self.bytes.get_bound_coeff(2 * g);
                let b1 = self.bytes.get_bound_coeff(2 * g + 1);
                let k0 = self.kernel.get_bound_coeff(2 * g);
                let k1 = self.kernel.get_bound_coeff(2 * g + 1);
                [k0 * b0, (k1 + (k1 - k0)) * (b1 + (b1 - b0))]
            })
            .reduce(|| [F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]]);
        UniPoly::from_evals(&[eval_0, previous_claim - eval_0, eval_2])
    }

    #[tracing::instrument(
        skip_all,
        name = "TrustedAdviceReconstructionSumcheckProver::ingest_challenge"
    )]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.bytes.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.kernel.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        // The byte-column opening lands at the column's own packed-slot
        // point: the bound (byte ‖ place) prefix suffixed with the fixed
        // word point.
        let mut point: Vec<F::Challenge> = sumcheck_challenges.to_vec();
        point.reverse();
        point.extend_from_slice(&self.params.r_word.r);
        accumulator.append_trusted_advice(
            SumcheckId::TrustedAdviceReconstruction,
            OpeningPoint::new(point),
            self.bytes.final_sumcheck_claim(),
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::poly::multilinear_polynomial::PolynomialEvaluation;
    use crate::transcripts::Blake2bTranscript;
    use ark_bn254::Fr;

    type Challenge = <Fr as JoltField>::Challenge;

    const WORD_VARS: usize = 2;

    /// Drives the full round loop over a synthetic advice column and checks
    /// every message folds the previous claim, the final claim equals the
    /// legacy verifier half's closed form, and the cached byte opening
    /// decodes directly against the column — pinning the msb-first cell
    /// conventions non-circularly.
    #[test]
    fn round_loop_reduces_to_the_byte_column_opening() {
        let words: Vec<u64> = vec![0x0102030405060708, 0, u64::MAX, 0xdeadbeef];
        let r_word: Vec<Challenge> = (0..WORD_VARS)
            .map(|i| Challenge::from((31 + 7 * i as u64) as u128))
            .collect();
        let word_claim = MultilinearPolynomial::<Fr>::from(words.clone()).evaluate(&r_word);

        let mut accumulator = ProverOpeningAccumulator::<Fr>::new(WORD_VARS);
        accumulator.append_untrusted_advice(
            SumcheckId::AdviceClaimReduction,
            OpeningPoint::new(r_word.clone()),
            word_claim,
        );

        let mut transcript = Blake2bTranscript::new(b"advice-bytes-test");
        let params = UntrustedAdviceReconstructionSumcheckParams::<Fr>::new(
            WORD_VARS,
            &accumulator,
            &mut transcript,
        );
        let mut prover =
            UntrustedAdviceReconstructionSumcheckProver::initialize(params.clone(), &words);

        let mut claim = params.input_claim(&accumulator);
        let mut challenges = Vec::new();
        for round in 0..params.num_rounds() {
            let message = SumcheckInstanceProver::<Fr, Blake2bTranscript>::compute_message(
                &mut prover,
                round,
                claim,
            );
            assert_eq!(
                message.eval_at_zero() + message.eval_at_one(),
                claim,
                "round {round} message must fold the previous claim"
            );
            let r_j = Challenge::from((211 + 13 * round) as u128);
            claim = message.evaluate(&r_j);
            challenges.push(r_j);
            SumcheckInstanceProver::<Fr, Blake2bTranscript>::ingest_challenge(
                &mut prover,
                r_j,
                round,
            );
        }
        SumcheckInstanceProver::<Fr, Blake2bTranscript>::cache_openings(
            &prover,
            &mut accumulator,
            &challenges,
        );

        // The verifier's closed form over the cached opening: the same five
        // publics the jolt-verifier ConcreteSumcheck derives (eq splits,
        // msb-first identity and place kernels).
        let opening_point = params.normalize_opening_point(&challenges);
        let (_, bytes_claim) = accumulator
            .get_advice_opening(
                AdviceKind::Untrusted,
                SumcheckId::UntrustedAdviceReconstruction,
            )
            .unwrap();
        let limb_bits = WORD_BYTES.log_2();
        let point: Vec<Fr> = opening_point.r.iter().map(|&c| c.into()).collect();
        let reference: Vec<Fr> = params.r_reference.iter().map(|&c| c.into()).collect();
        let (r_symbol, r_limb_word) = point.split_at(BYTE_BITS);
        let (r_limb, r_word_bound) = r_limb_word.split_at(limb_bits);
        let r_word_field: Vec<Fr> = r_word.iter().map(|&c| c.into()).collect();
        let eq_cell_mle: Fr = EqPolynomial::mle(&point, &reference);
        let eq_limb_word: Fr = EqPolynomial::mle(r_limb_word, &reference[BYTE_BITS..]);
        let eq_word: Fr = EqPolynomial::mle(r_word_bound, &r_word_field);
        let identity_at_symbol = r_symbol
            .iter()
            .fold(Fr::from_u64(0), |acc, coordinate| acc + acc + *coordinate);
        let place_at_limb =
            r_limb
                .iter()
                .enumerate()
                .fold(Fr::from_u64(1), |acc, (position, coordinate)| {
                    let weight = 1usize << (limb_bits - 1 - position);
                    let place = Fr::from_u64(1u64 << (8 * weight));
                    acc * ((place - Fr::from_u64(1)) * *coordinate + Fr::from_u64(1))
                });
        let gamma = params.gamma;
        let expected = eq_cell_mle * (bytes_claim.square() - bytes_claim)
            + gamma * eq_limb_word * bytes_claim
            + gamma * gamma * identity_at_symbol * place_at_limb * eq_word * bytes_claim;
        assert_eq!(claim, expected, "final claim must equal the closed form");

        // The cached opening must equal the byte column evaluated directly at
        // the produced msb-first cell point.
        let cell_vars = word_byte_num_vars(WORD_VARS);
        let eq_cell = EqPolynomial::<Fr>::evals(&opening_point.r);
        let limb_bits = WORD_BYTES.log_2();
        let mut direct = Fr::from_u64(0);
        for limb in 0..WORD_BYTES {
            for (word_index, word) in words.iter().enumerate() {
                let byte = ((word >> (8 * limb)) & 0xff) as usize;
                direct += eq_cell[(((byte << limb_bits) | limb) << WORD_VARS) | word_index];
            }
        }
        assert_eq!(cell_vars, opening_point.len());
        assert_eq!(bytes_claim, direct, "byte opening must decode the column");
    }

    /// The dense cell-domain tables the prover materialized before the
    /// factored rewrite — kept as a reference oracle: the factored prover
    /// must produce identical round polynomials and final opening claim.
    struct DenseReference {
        bytes: MultilinearPolynomial<Fr>,
        k_bool: MultilinearPolynomial<Fr>,
        k_lin: MultilinearPolynomial<Fr>,
    }

    impl DenseReference {
        fn initialize(
            params: &UntrustedAdviceReconstructionSumcheckParams<Fr>,
            words: &[u64],
        ) -> Self {
            let word_vars = params.word_vars;
            let limb_bits = WORD_BYTES.log_2();
            let cell_vars = word_byte_num_vars(word_vars);
            let mut bytes = vec![0u8; 1 << cell_vars];
            for limb in 0..WORD_BYTES {
                for word_index in 0..(1usize << word_vars) {
                    let byte = words
                        .get(word_index)
                        .map_or(0, |word| (word >> (8 * limb)) as u8)
                        as usize;
                    bytes[(((byte << limb_bits) | limb) << word_vars) | word_index] = 1;
                }
            }
            let k_bool = EqPolynomial::<Fr>::evals(&params.r_reference);
            let eq_lw = EqPolynomial::<Fr>::evals(&params.r_reference[BYTE_BITS..]);
            let eq_word = EqPolynomial::<Fr>::evals(&params.r_word.r);
            let gamma = params.gamma;
            let gamma_squared = gamma * gamma;
            let k_lin = (0..1usize << cell_vars)
                .map(|cell| {
                    let word_index = cell & ((1 << word_vars) - 1);
                    let limb = (cell >> word_vars) & (WORD_BYTES - 1);
                    let symbol = cell >> (word_vars + limb_bits);
                    let place = Fr::from_u64(1u64 << (8 * limb));
                    gamma * eq_lw[cell & ((1 << (limb_bits + word_vars)) - 1)]
                        + gamma_squared * Fr::from_u64(symbol as u64) * place * eq_word[word_index]
                })
                .collect::<Vec<Fr>>();
            Self {
                bytes: MultilinearPolynomial::from(bytes),
                k_bool: MultilinearPolynomial::from(k_bool),
                k_lin: MultilinearPolynomial::from(k_lin),
            }
        }

        fn compute_message(&self, previous_claim: Fr) -> UniPoly<Fr> {
            let term = |b: Fr, kb: Fr, kl: Fr| kb * (b.square() - b) + kl * b;
            let mut evals = [Fr::from_u64(0); 3];
            for g in 0..self.bytes.len() / 2 {
                let b0 = self.bytes.get_bound_coeff(2 * g);
                let b1 = self.bytes.get_bound_coeff(2 * g + 1);
                let kb0 = self.k_bool.get_bound_coeff(2 * g);
                let kb1 = self.k_bool.get_bound_coeff(2 * g + 1);
                let kl0 = self.k_lin.get_bound_coeff(2 * g);
                let kl1 = self.k_lin.get_bound_coeff(2 * g + 1);
                let (b_delta, kb_delta, kl_delta) = (b1 - b0, kb1 - kb0, kl1 - kl0);
                let (b2, b3) = (b1 + b_delta, b1 + b_delta + b_delta);
                evals[0] += term(b0, kb0, kl0);
                evals[1] += term(b2, kb1 + kb_delta, kl1 + kl_delta);
                evals[2] += term(b3, kb1 + kb_delta + kb_delta, kl1 + kl_delta + kl_delta);
            }
            UniPoly::from_evals(&[evals[0], previous_claim - evals[0], evals[1], evals[2]])
        }

        fn bind(&mut self, r_j: Challenge) {
            self.bytes.bind_parallel(r_j, BindingOrder::LowToHigh);
            self.k_bool.bind_parallel(r_j, BindingOrder::LowToHigh);
            self.k_lin.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
    }

    /// Runs the factored prover and the dense reference through the full
    /// round loop on the same challenges, comparing every round polynomial
    /// coefficient-for-coefficient and the final opening claim — across word
    /// patterns exercising duplicate lanes, all-zero rows, implicit padding
    /// rows (`words.len() < 2^word_vars`), and both row/lane phase shapes.
    #[test]
    fn factored_prover_matches_the_dense_reference() {
        let cases: Vec<(usize, Vec<u64>)> = vec![
            (2, vec![0x0102030405060708, 0, u64::MAX, 0xdeadbeef]),
            // Duplicate lanes: every row of a word hits the same byte value.
            (2, vec![0x4242424242424242; 4]),
            // Pure padding: the all-zero column (lane 0 hot everywhere).
            (2, vec![]),
            // Implicit padding rows past words.len().
            (3, vec![0xa5, 0x00ff00ff00ff00ff, 0x8000000000000001]),
        ];
        for (case_index, (word_vars, words)) in cases.into_iter().enumerate() {
            let r_word: Vec<Challenge> = (0..word_vars)
                .map(|i| Challenge::from((17 + 5 * (i + case_index) as u64) as u128))
                .collect();
            let mut padded = words.clone();
            padded.resize(1 << word_vars, 0);
            let word_claim = MultilinearPolynomial::<Fr>::from(padded).evaluate(&r_word);
            let mut accumulator = ProverOpeningAccumulator::<Fr>::new(word_vars);
            accumulator.append_untrusted_advice(
                SumcheckId::AdviceClaimReduction,
                OpeningPoint::new(r_word),
                word_claim,
            );
            let mut transcript = Blake2bTranscript::new(b"advice-bytes-dense-diff");
            let params = UntrustedAdviceReconstructionSumcheckParams::<Fr>::new(
                word_vars,
                &accumulator,
                &mut transcript,
            );
            let mut prover =
                UntrustedAdviceReconstructionSumcheckProver::initialize(params.clone(), &words);
            let mut reference = DenseReference::initialize(&params, &words);

            let mut claim = params.input_claim(&accumulator);
            let mut challenges = Vec::new();
            for round in 0..params.num_rounds() {
                let message = SumcheckInstanceProver::<Fr, Blake2bTranscript>::compute_message(
                    &mut prover,
                    round,
                    claim,
                );
                let expected = reference.compute_message(claim);
                assert_eq!(
                    message.coeffs, expected.coeffs,
                    "case {case_index} round {round}: factored round polynomial diverges"
                );
                let r_j = Challenge::from((97 + 29 * (round + case_index)) as u128);
                claim = message.evaluate(&r_j);
                challenges.push(r_j);
                SumcheckInstanceProver::<Fr, Blake2bTranscript>::ingest_challenge(
                    &mut prover,
                    r_j,
                    round,
                );
                reference.bind(r_j);
            }
            SumcheckInstanceProver::<Fr, Blake2bTranscript>::cache_openings(
                &prover,
                &mut accumulator,
                &challenges,
            );
            let (_, bytes_claim) = accumulator
                .get_advice_opening(
                    AdviceKind::Untrusted,
                    SumcheckId::UntrustedAdviceReconstruction,
                )
                .expect("factored prover caches the byte opening");
            assert_eq!(
                bytes_claim,
                reference.bytes.final_sumcheck_claim(),
                "case {case_index}: final byte opening diverges from the dense reference"
            );
        }
    }
}
