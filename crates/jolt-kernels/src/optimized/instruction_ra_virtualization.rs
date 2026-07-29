//! Optimized instruction RA virtualization (stage 6b) kernel.
//!
//! The summand is `eq(r_cycle, j) · Σ_v γ^v · Π_{i<N} ra_{N·v+i}(chunk_i, j)`
//! over the cycle domain. The reference kernel materializes every committed
//! one-hot `(K × T)` grid through `oracle_table` and address-folds it (`K·T`
//! multiply-adds per committed chunk), then interprets the relation
//! expression through the naive prover. This kernel exploits the one-hot
//! structure instead: each committed `InstructionRa` chunk is a point mass at
//! that chunk of the cycle's 128-bit lookup index, so the address-folded
//! value is a single eq-table lookup,
//!
//! `ra_i(r_chunk_i, j) = eq(r_chunk_i, chunk_i(k_j)) = eq_table_i[chunk_i(k_j)]`
//!
//! — `T` lookups per committed chunk instead of a `K × T` grid walk, and no
//! grid materialization at all. The per-cycle lookup indices are reclaimed
//! from the [`ProofSession`] when the stage-5 optimized kernel already
//! collected them (see
//! [`SharedInstructionRows`](super::instruction_read_raf::SharedInstructionRows)),
//! or collected fresh otherwise.
//!
//! Round messages use the Gruen split-eq factorization: `eq(r_cycle, ·)` is
//! never materialized or bound as a `T`-sized table; each round emits
//! `s(t) = ℓ(t) · Σ_y E_out·E_in · Σ_v γ^v Π_i ra(t, y)` at the naive
//! prover's `t = 0..=degree` sample points through the same `from_evals`
//! constructor, so round polynomials and output claims are byte-identical
//! (field arithmetic is exact under any regrouping).
//!
//! The bound eq factor is pinned to the verifier's scalar path by
//! [`validate_derived_tables`](crate::SumcheckKernel::validate_derived_tables):
//! the fully-bound Gruen scalar must equal `derive_output_term(EqCycle)`,
//! exactly as the naive tier's bound `EqCycle` table is checked.

use std::sync::Arc;

use jolt_claims::protocols::jolt::geometry::dimensions::committed_address_chunks;
use jolt_claims::protocols::jolt::relations::instruction::InstructionRaVirtualizationOutputClaims;
use jolt_claims::protocols::jolt::{InstructionRaVirtualizationPublic, JoltDerivedId};
use jolt_field::Field;
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage6b::instruction_ra_virtualization::InstructionRaVirtualization;
use jolt_witness::{collect_bundles, JoltWitnessPlane};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::instruction_read_raf::SharedInstructionRows;
use crate::reference::instruction_read_raf::InstructionReadRafWitness;
use crate::reference::views::eq_table;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// Optimized [`PrepareKernel`] implementor for the
/// `instruction_ra_virtualization` slot.
pub struct OptimizedInstructionRaVirtualization;

impl<F: Field> PrepareKernel<F, InstructionRaVirtualization<F>>
    for OptimizedInstructionRaVirtualization
{
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, InstructionRaVirtualization<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionRaVirtualization<F>>>, KernelError<F>>
    {
        let relation = inputs.relation;
        let cycles = 1usize << relation.dimensions().log_t();
        // Reclaim the stage-5 rows when the optimized read-RAF kernel parked
        // them (mixed-backend registries may not have); the length guard
        // makes a stale carry impossible to consume.
        let rows = match session.take::<SharedInstructionRows>() {
            Some(SharedInstructionRows(rows)) if rows.len() == cycles => rows,
            _ => Arc::new(collect_bundles(witness, cycles)?),
        };
        Ok(Box::new(OptimizedInstructionRaVirtualizationKernel::new(
            relation.dimensions().log_t(),
            relation.dimensions().num_virtual_ra_polys(),
            relation.dimensions().num_committed_per_virtual(),
            relation.instruction_address(),
            relation.instruction_read_raf_cycle(),
            relation.committed_chunk_bits(),
            &rows,
            inputs.challenges.gamma,
        )?))
    }
}

/// Collect `f(0), …, f(len − 1)`.
fn map_indices<T: Send>(len: usize, f: impl Fn(usize) -> T + Send + Sync) -> Vec<T> {
    #[cfg(feature = "parallel")]
    {
        (0..len).into_par_iter().map(f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..len).map(f).collect()
    }
}

pub struct OptimizedInstructionRaVirtualizationKernel<F: Field> {
    log_t: usize,
    num_committed_per_virtual: usize,
    gamma_powers: Vec<F>,
    /// Address-folded committed RA selectors, one per committed chunk:
    /// `folded[i][j] = eq(r_chunk_i, chunk_i(k_j))`.
    folded_ra: Vec<Polynomial<F>>,
    gruen: GruenSplitEqPolynomial<F>,
    bind_scratch: Vec<F>,
    rounds_bound: usize,
}

impl<F: Field> OptimizedInstructionRaVirtualizationKernel<F> {
    #[expect(clippy::too_many_arguments, reason = "mirrors the relation accessors")]
    pub fn new(
        log_t: usize,
        num_virtual: usize,
        num_committed_per_virtual: usize,
        instruction_address: &[F],
        instruction_read_raf_cycle: &[F],
        committed_chunk_bits: usize,
        rows: &[InstructionReadRafWitness],
        gamma: F,
    ) -> Result<Self, KernelError<F>> {
        let num_committed = num_virtual * num_committed_per_virtual;
        let chunks = committed_address_chunks(instruction_address, committed_chunk_bits);
        if chunks.len() != num_committed
            || instruction_address.len() != num_committed * committed_chunk_bits
        {
            return Err(KernelError::InvariantViolation {
                reason: "instruction address chunk count disagrees with the committed RA count",
            });
        }
        if committed_chunk_bits == 0 || committed_chunk_bits > 32 {
            return Err(KernelError::Unsupported {
                reason: "committed RA chunk width outside the supported one-hot range",
            });
        }
        if rows.len() != 1 << log_t {
            return Err(KernelError::TableSizeMismatch {
                table: "stage-6b instruction rows".to_owned(),
                expected: 1 << log_t,
                got: rows.len(),
            });
        }
        if instruction_read_raf_cycle.len() != log_t {
            return Err(KernelError::TableSizeMismatch {
                table: "instruction read-RAF cycle point".to_owned(),
                expected: log_t,
                got: instruction_read_raf_cycle.len(),
            });
        }

        // One eq table per committed chunk point (each `2^w` entries), then
        // the point-mass fold: one table lookup per cycle per chunk.
        let chunk_tables: Vec<Vec<F>> = map_indices(chunks.len(), |i| eq_table(&chunks[i]));
        let mask = (1u128 << committed_chunk_bits) - 1;
        let folded_ra: Vec<Polynomial<F>> = chunk_tables
            .iter()
            .enumerate()
            .map(|(i, table)| {
                let shift = (num_committed - 1 - i) * committed_chunk_bits;
                Polynomial::new(map_indices(rows.len(), |j| {
                    table[((rows[j].lookup_index.0 >> shift) & mask) as usize]
                }))
            })
            .collect();

        let mut gamma_powers = Vec::with_capacity(num_virtual);
        let mut power = F::one();
        for _ in 0..num_virtual {
            gamma_powers.push(power);
            power *= gamma;
        }

        Ok(Self {
            log_t,
            num_committed_per_virtual,
            gamma_powers,
            folded_ra,
            gruen: GruenSplitEqPolynomial::new(instruction_read_raf_cycle, BindingOrder::LowToHigh),
            bind_scratch: Vec::new(),
            rounds_bound: 0,
        })
    }

    /// `s(t) = ℓ(t) · q(t)` at the naive prover's sample points, with
    /// `q(t) = Σ_y E(y) · Σ_v γ^v Π_{i<N} ra_{N·v+i}(t, y)`.
    fn message(
        &self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        // The relation degree: one eq factor plus the N-wide products.
        let degree = self.num_committed_per_virtual + 1;
        let points = degree + 1;
        let num_committed = self.folded_ra.len();

        let q_evals = self.gruen.par_fold_out_in(
            || {
                (
                    vec![F::zero(); points],
                    vec![F::zero(); num_committed],
                    vec![F::zero(); num_committed],
                )
            },
            |(acc, evals, steps), row, _x_in, e_in| {
                for (position, ra) in self.folded_ra.iter().enumerate() {
                    let table = ra.evals();
                    let lo = table[2 * row];
                    let hi = table[2 * row + 1];
                    evals[position] = lo;
                    steps[position] = hi - lo;
                }
                for value in acc.iter_mut() {
                    let mut sum = F::zero();
                    for (v, gamma_power) in self.gamma_powers.iter().enumerate() {
                        let base = v * self.num_committed_per_virtual;
                        let mut product = evals[base];
                        for eval in &evals[base + 1..base + self.num_committed_per_virtual] {
                            product *= *eval;
                        }
                        sum += *gamma_power * product;
                    }
                    *value += e_in * sum;
                    for (eval, step) in evals.iter_mut().zip(steps.iter()) {
                        *eval += *step;
                    }
                }
            },
            |_x_out, e_out, (mut acc, _, _)| {
                for value in &mut acc {
                    *value *= e_out;
                }
                acc
            },
            |mut a, b| {
                for (a, b) in a.iter_mut().zip(&b) {
                    *a += *b;
                }
                a
            },
        );

        let (l_at_0, l_at_1) = self.gruen.current_linear_evals();
        let l_step = l_at_1 - l_at_0;
        let mut l_eval = l_at_0;
        let mut evals = Vec::with_capacity(points);
        for q in &q_evals {
            evals.push(l_eval * *q);
            l_eval += l_step;
        }

        let round_sum = evals[0] + evals[1];
        if round_sum != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual: round_sum,
            });
        }
        Ok(UnivariatePoly::from_evals(&evals))
    }

    fn bind(&mut self, challenge: F) {
        self.gruen.bind(challenge);
        for ra in &mut self.folded_ra {
            ra.bind_low_to_high_reusing_scratch(challenge, &mut self.bind_scratch);
        }
        self.rounds_bound += 1;
    }
}

impl<F: Field> ProveRounds<F> for OptimizedInstructionRaVirtualizationKernel<F> {
    fn num_rounds(&self) -> usize {
        self.log_t
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
        self.message(round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: Field> SumcheckKernel<F> for OptimizedInstructionRaVirtualizationKernel<F> {
    type Relation = InstructionRaVirtualization<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<InstructionRaVirtualizationOutputClaims<F>, SumcheckKernelError<F>> {
        if self.rounds_bound != self.log_t {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.log_t - self.rounds_bound,
            });
        }
        Ok(InstructionRaVirtualizationOutputClaims {
            committed_instruction_ra: self.folded_ra.iter().map(|ra| ra.evals()[0]).collect(),
        })
    }

    /// The Gruen scalar after full binding is the bound `EqCycle` value; pin
    /// it to the verifier's `derive_output_term`, exactly as the naive tier's
    /// materialized eq table is pinned.
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        if self.rounds_bound != self.log_t {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.log_t - self.rounds_bound,
            });
        }
        let id = JoltDerivedId::from(InstructionRaVirtualizationPublic::EqCycle);
        let expected = relation.derive_output_term(&id, input_points, output_points, challenges)?;
        let got = self.gruen.current_scalar();
        if got != expected {
            return Err(SumcheckKernelError::DerivedTableDrift { id, expected, got });
        }
        Ok(())
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use std::collections::BTreeMap;
    use std::num::NonZeroUsize;

    use jolt_claims::protocols::jolt::geometry::dimensions::committed_address_chunks;
    use jolt_claims::protocols::jolt::geometry::instruction::{
        committed_instruction_ra, InstructionRaVirtualizationDimensions,
    };
    use jolt_claims::protocols::jolt::relations::instruction::{
        InstructionRaVirtualizationChallenges, InstructionRaVirtualizationInputClaims,
    };
    use jolt_claims::protocols::jolt::{InstructionRaVirtualizationPublic, JoltDerivedId};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::{BindingOrder, Polynomial};
    use jolt_sumcheck::ProveRounds;
    use jolt_verifier::stages::relations::ConcreteSumcheck;
    use jolt_verifier::stages::stage6b::instruction_ra_virtualization::InstructionRaVirtualization;
    use jolt_witness::witnesses::{InstructionRafFlag, LookupIndex, TableIndex};
    use jolt_witness::{FixedBackend, PolynomialEncoding, Shape};

    use crate::reference::instruction_read_raf::InstructionReadRafWitness;
    use crate::reference::views::{address_fold, eq_table};
    use crate::{NaiveSumcheckProver, ProverInputs, SumcheckKernel};

    use super::OptimizedInstructionRaVirtualizationKernel;

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn challenge(round: usize) -> Fr {
        fr(0xD1B5_4A32_D192_ED03 ^ (round as u64).wrapping_mul(0x94D0_49BB_1331_11EB) ^ 3)
    }

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn fixture_rows(log_t: usize, seed: u64) -> Vec<InstructionReadRafWitness> {
        let mut state = seed;
        (0..1usize << log_t)
            .map(|j| {
                let lookup_index = match j {
                    0 => 0u128,
                    1 => u128::MAX,
                    _ => ((splitmix(&mut state) as u128) << 64) | splitmix(&mut state) as u128,
                };
                InstructionReadRafWitness {
                    lookup_index: LookupIndex(lookup_index),
                    table_index: TableIndex(None),
                    raf_flag: InstructionRafFlag(false),
                }
            })
            .collect()
    }

    /// Builds the committed one-hot `(K × T)` grid for chunk `i` exactly as
    /// the trace backend serves it: address-major, hot at that chunk of the
    /// cycle's lookup index, every cycle hot.
    fn one_hot_grid(
        rows: &[InstructionReadRafWitness],
        chunk_index: usize,
        num_committed: usize,
        chunk_bits: usize,
    ) -> Vec<Fr> {
        let k = 1usize << chunk_bits;
        let t = rows.len();
        let shift = (num_committed - 1 - chunk_index) * chunk_bits;
        let mask = (1u128 << chunk_bits) - 1;
        let mut grid = vec![fr(0); k * t];
        for (j, row) in rows.iter().enumerate() {
            let hot = ((row.lookup_index.0 >> shift) & mask) as usize;
            grid[(hot * t) | j] = fr(1);
        }
        grid
    }

    /// Reference (naive prover over address-folded oracle grids, exactly as
    /// the reference `prepare` assembles it) vs the optimized kernel, same
    /// challenges: byte-equal round polynomials and output claims, and the
    /// optimized eq-scalar passes the derived-table cross-check.
    fn assert_parity(
        log_t: usize,
        num_virtual: usize,
        per_virtual: usize,
        chunk_bits: usize,
        seed: u64,
    ) {
        let num_committed = num_virtual * per_virtual;
        let dimensions = InstructionRaVirtualizationDimensions::new(
            log_t,
            NonZeroUsize::new(num_virtual).unwrap(),
            NonZeroUsize::new(per_virtual).unwrap(),
        )
        .unwrap();
        let rows = fixture_rows(log_t, seed);
        let instruction_address: Vec<Fr> = (0..num_committed * chunk_bits)
            .map(|i| fr(300 + 13 * i as u64))
            .collect();
        let r_cycle: Vec<Fr> = (0..log_t).map(|i| fr(7000 + 29 * i as u64)).collect();
        let gamma = fr(0xFEED_5EED);
        let relation = InstructionRaVirtualization::<Fr>::new(
            dimensions,
            instruction_address.clone(),
            r_cycle.clone(),
            chunk_bits,
        );

        // The reference tier, assembled exactly as its `prepare` does: one-hot
        // grids behind a fixed oracle, address-folded per committed chunk.
        let mut backend = FixedBackend::new();
        for index in 0..num_committed {
            let grid = one_hot_grid(&rows, index, num_committed, chunk_bits);
            backend
                .insert(
                    committed_instruction_ra(index).polynomial_id(),
                    Shape::new(chunk_bits + log_t, PolynomialEncoding::Dense),
                    grid,
                )
                .unwrap();
        }
        let chunks = committed_address_chunks(&instruction_address, chunk_bits);
        let mut opening_tables = BTreeMap::new();
        for (index, chunk) in chunks.iter().enumerate() {
            let folded =
                address_fold::<Fr>(&backend, committed_instruction_ra(index), log_t, chunk)
                    .unwrap();
            let _ = opening_tables.insert(committed_instruction_ra(index), Polynomial::new(folded));
        }
        let derived_tables = BTreeMap::from([(
            JoltDerivedId::from(InstructionRaVirtualizationPublic::EqCycle),
            Polynomial::new(eq_table(&r_cycle)),
        )]);

        let input_claims = InstructionRaVirtualizationInputClaims {
            instruction_ra: vec![fr(0); num_virtual],
        };
        let input_points = InstructionRaVirtualizationInputClaims {
            instruction_ra: vec![Vec::new(); num_virtual],
        };
        let challenges = InstructionRaVirtualizationChallenges { gamma };
        let inputs = ProverInputs {
            relation: &relation,
            claims: &input_claims,
            points: &input_points,
            challenges: &challenges,
        };
        let mut reference = NaiveSumcheckProver::new(
            &inputs,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )
        .unwrap();

        let mut optimized = OptimizedInstructionRaVirtualizationKernel::new(
            log_t,
            num_virtual,
            per_virtual,
            &instruction_address,
            &r_cycle,
            chunk_bits,
            &rows,
            gamma,
        )
        .unwrap();

        // True input claim: the full hypercube sum of the output summand.
        let eq_cycle = eq_table(&r_cycle);
        let mut claim = fr(0);
        for j in 0..rows.len() {
            let mut sum = fr(0);
            let mut gamma_power = fr(1);
            for v in 0..num_virtual {
                let mut product = fr(1);
                for i in 0..per_virtual {
                    let index = v * per_virtual + i;
                    let shift = (num_committed - 1 - index) * chunk_bits;
                    let hot =
                        ((rows[j].lookup_index.0 >> shift) & ((1u128 << chunk_bits) - 1)) as usize;
                    product *= eq_table(&chunks[index])[hot];
                }
                sum += gamma_power * product;
                gamma_power *= gamma;
            }
            claim += eq_cycle[j] * sum;
        }

        let rounds = reference.num_rounds();
        assert_eq!(rounds, optimized.num_rounds());
        assert_eq!(rounds, log_t);
        for round in 0..rounds {
            let bind = round.checked_sub(1).map(challenge);
            let reference_poly = reference.prove_round(bind, round, claim).unwrap();
            let optimized_poly = optimized.prove_round(bind, round, claim).unwrap();
            assert_eq!(
                reference_poly.coefficients(),
                optimized_poly.coefficients(),
                "round {round} polynomial mismatch (log_t={log_t}, V={num_virtual}, N={per_virtual})"
            );
            claim = reference_poly.evaluate(challenge(round));
        }
        reference.finish_rounds(challenge(rounds - 1)).unwrap();
        optimized.finish_rounds(challenge(rounds - 1)).unwrap();

        let reference_outputs = reference.output_claims(&input_claims).unwrap();
        let optimized_outputs = optimized.output_claims(&input_claims).unwrap();
        assert_eq!(
            reference_outputs.committed_instruction_ra,
            optimized_outputs.committed_instruction_ra
        );

        // The optimized eq scalar passes the same derived-table cross-check
        // the naive tier's materialized table does.
        let sumcheck_point: Vec<Fr> = (0..rounds).map(challenge).collect();
        let output_points = relation
            .derive_opening_points(&sumcheck_point, &input_points)
            .unwrap();
        optimized
            .validate_derived_tables(&relation, &input_points, &output_points, &challenges)
            .unwrap();
        reference
            .validate_derived_tables(&relation, &input_points, &output_points, &challenges)
            .unwrap();
    }

    /// Production shape: 8 virtuals × 4 committed each, 4-bit chunks (the
    /// 128-bit instruction address).
    #[test]
    fn parity_production_geometry() {
        assert_parity(4, 8, 4, 4, 42);
    }

    /// Odd geometry: 3 virtuals × 2 committed, 2-bit chunks, odd log_t.
    #[test]
    fn parity_small_odd_geometry() {
        assert_parity(3, 3, 2, 2, 1337);
    }
}
