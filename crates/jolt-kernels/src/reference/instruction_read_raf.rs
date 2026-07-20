//! The instruction read+RAF checking (stage 5) kernel — the one relation the
//! naive tier cannot serve: its domain is `(address ∈ 2^128) × (cycle ∈ 2^T)`,
//! so no leaf table can be materialized. Instead the kernel exploits the
//! summand's structure
//!
//! `Σ_{j,k} eq(r_reduction, j) · ra(k, j) · (Val_j(k) + γ·RafVal_j(k))`
//!
//! where `ra(k, j)` is a point mass at the cycle's 128-bit lookup index
//! `k_j`: every hypercube sum collapses to a per-cycle sum, evaluated with
//! the prefix–suffix decompositions of the lookup tables (eprint 2025/611,
//! Appendix A).
//!
//! **Address rounds** (first 128, MSB-first) run in phases of
//! [`CHUNK_LEN`]` = 8` variables. Per phase, each of the 46 table prefixes is
//! materialized as a dense 256-entry chunk polynomial from its checkpoints
//! (`jolt-lookup-tables`' binary-point `evaluate` API is built for exactly
//! this), each present table's suffixes are accumulated into 256-entry `Q`
//! polynomials by one trace scan (`Q[x] += u_j · suffix_mle(low bits of
//! k_j)`), and the three RAF operand/identity decompositions get their
//! chunk-prefix and shift/value `Q` polynomials. The round polynomial is the
//! true quadratic `Σ_b Σ_t combine_t(P(c,b), Q_t(c,b)) + raf(c,b)`, sampled
//! at `c ∈ {0,1,2}`; binding is plain `HighToLow` dense binding, and each
//! fully bound prefix becomes its checkpoint for the next phase. The per-cycle
//! eq weights `u_j` are condensed at each phase start by the previous phase's
//! bound-challenge eq table.
//!
//! **Cycle rounds** (last `log_T`, `LowToHigh`) are a plain multilinear
//! product over cycle-indexed tables: `eq(r_reduction, ·)`, the combined
//! `Val + γ·RafVal` values at the bound address, and the
//! `D = num_virtual_ra_polys` virtual `ra` chunk selectors (8 in the default
//! config) — the true degree-`D+2` polynomial sampled at `D+3` points.
//! (The legacy prover's Gruen split-eq and phase-count selection are
//! optimizations of the same true polynomials; byte parity needs only
//! exactness, so this kernel always uses 8-variable phases.)

use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafDimensions;
use jolt_claims::protocols::jolt::relations::instruction::{
    InstructionReadRafChallenges, InstructionReadRafOutputClaims,
};
use jolt_field::Field;
use jolt_lookup_tables::tables::prefixes::{PrefixEval, ALL_PREFIXES};
use jolt_lookup_tables::tables::suffixes::SuffixEval;
use jolt_lookup_tables::{LookupBits, LookupTableKind, XLEN as RISCV_XLEN};
use jolt_poly::{BindingOrder, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::stage5::InstructionReadRaf;
use jolt_witness::protocols::jolt_vm::Stage5InstructionReadRafRow;

use super::views::eq_table;
use crate::instruction_read_raf::InstructionReadRafProver;
use crate::{KernelError, ProofSession, ProveSumcheck, ReferenceBackend};

/// Address variables bound per phase. Fixed at 8 (the legacy prover picks 8
/// or 16 by trace size, but the emitted polynomials are identical — see the
/// module docs).
const CHUNK_LEN: usize = 8;
const CHUNK_SIZE: usize = 1 << CHUNK_LEN;

impl<F: Field> InstructionReadRafProver<F> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        dimensions: InstructionReadRafDimensions,
        r_reduction: &[F],
        rows: Vec<Stage5InstructionReadRafRow>,
        challenges: &InstructionReadRafChallenges<F>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = InstructionReadRaf<F>>>, KernelError<F>> {
        Ok(Box::new(InstructionReadRafKernel::new(
            dimensions,
            r_reduction,
            rows,
            challenges.gamma,
        )?))
    }
}

/// One RAF prefix–suffix decomposition (left operand, right operand, or
/// address identity): `poly(k) = P(chunk) · Q_shift + Q_value` over the
/// current phase's chunk domain, with the fully bound `P` becoming the next
/// phase's checkpoint.
struct RafDecomposition<F: Field> {
    prefix: Polynomial<F>,
    q_shift: Polynomial<F>,
    q_value: Polynomial<F>,
    checkpoint: F,
}

impl<F: Field> RafDecomposition<F> {
    fn empty() -> Self {
        Self {
            prefix: Polynomial::new(vec![F::zero()]),
            q_shift: Polynomial::new(vec![F::zero()]),
            q_value: Polynomial::new(vec![F::zero()]),
            checkpoint: F::zero(),
        }
    }

    fn message_eval(&self, b: usize, half: usize, c: usize) -> F {
        extension_eval(self.prefix.evals(), b, half, c)
            * extension_eval(self.q_shift.evals(), b, half, c)
            + extension_eval(self.q_value.evals(), b, half, c)
    }

    fn bind(&mut self, challenge: F) {
        self.prefix
            .bind_with_order(challenge, BindingOrder::HighToLow);
        self.q_shift
            .bind_with_order(challenge, BindingOrder::HighToLow);
        self.q_value
            .bind_with_order(challenge, BindingOrder::HighToLow);
    }
}

/// The linear extension of a dense table's current top variable: `evals[b]`
/// at 0, `evals[b + half]` at 1, `2·hi − lo` at 2.
fn extension_eval<F: Field>(evals: &[F], b: usize, half: usize, c: usize) -> F {
    let lo = evals[b];
    let hi = evals[b + half];
    match c {
        0 => lo,
        1 => hi,
        _ => hi + hi - lo,
    }
}

/// Cycle-indexed tables for the last `log_T` rounds: `eq(r_reduction, ·)`,
/// the combined `Val + γ·RafVal` at the bound address, and the virtual `ra`
/// chunk selectors.
struct CycleTables<F: Field> {
    eq_reduction: Polynomial<F>,
    combined_val: Polynomial<F>,
    ra: Vec<Polynomial<F>>,
}

pub struct InstructionReadRafKernel<F: Field> {
    relation: InstructionReadRaf<F>,
    dimensions: InstructionReadRafDimensions,
    gamma: F,
    r_reduction: Vec<F>,
    rows: Vec<Stage5InstructionReadRafRow>,
    /// Per-table cycle buckets, indexed by `LookupTableKind::index()`.
    buckets: Vec<Vec<usize>>,
    /// Condensed per-cycle eq weights: after phase `p` starts,
    /// `u[j] = eq(r_reduction, j) · Π_{q<p} eq(phase-q challenges, chunk_q(k_j))`.
    u_evals: Vec<F>,
    /// The 46 table-prefix checkpoints (fully bound values of completed
    /// phases' prefix chunk polynomials).
    prefix_checkpoints: Vec<PrefixEval<F>>,
    /// The 46 materialized prefix chunk polynomials for the current phase,
    /// in `ALL_PREFIXES` order.
    prefix_tables: Vec<Polynomial<F>>,
    /// Per present table (enum index, suffix `Q` polynomials in
    /// `table.suffixes()` order) for the current phase.
    suffix_tables: Vec<(LookupTableKind<RISCV_XLEN>, Vec<Polynomial<F>>)>,
    raf_left: RafDecomposition<F>,
    raf_right: RafDecomposition<F>,
    raf_identity: RafDecomposition<F>,
    /// Completed phases' bound-challenge eq tables (`v[p][x] =
    /// eq(phase-p challenges, x)`, MSB-first).
    v_tables: Vec<Vec<F>>,
    phase_challenges: Vec<F>,
    cycle_challenges: Vec<F>,
    cycle_tables: Option<CycleTables<F>>,
    rounds_bound: usize,
}

impl<F: Field> InstructionReadRafKernel<F> {
    pub fn new(
        dimensions: InstructionReadRafDimensions,
        r_reduction: &[F],
        rows: Vec<Stage5InstructionReadRafRow>,
        gamma: F,
    ) -> Result<Self, KernelError<F>> {
        let address_bits = dimensions.instruction_address_bits();
        let log_t = dimensions.log_t();
        if address_bits != 2 * RISCV_XLEN {
            return Err(KernelError::Unsupported {
                reason: "instruction read-RAF supports only the 2·XLEN interleaved-operand \
                         address width",
            });
        }
        let ra_count = dimensions.num_virtual_ra_polys();
        if !address_bits.is_multiple_of(ra_count)
            || !(address_bits / ra_count).is_multiple_of(CHUNK_LEN)
        {
            return Err(KernelError::Unsupported {
                reason: "virtual RA chunk width must be a multiple of the phase width",
            });
        }
        if rows.len() != 1 << log_t {
            return Err(KernelError::TableSizeMismatch {
                table: "stage-5 instruction rows".to_owned(),
                expected: 1 << log_t,
                got: rows.len(),
            });
        }
        if r_reduction.len() != log_t {
            return Err(KernelError::TableSizeMismatch {
                table: "instruction claim-reduction point".to_owned(),
                expected: log_t,
                got: r_reduction.len(),
            });
        }

        let mut buckets = vec![Vec::new(); LookupTableKind::<RISCV_XLEN>::COUNT];
        for (j, row) in rows.iter().enumerate() {
            if let Some(table_index) = row.table_index {
                buckets
                    .get_mut(table_index)
                    .ok_or(KernelError::InvariantViolation {
                        reason: "stage-5 row selects an unknown lookup table",
                    })?
                    .push(j);
            }
        }

        let mut kernel = Self {
            relation: InstructionReadRaf::new(dimensions),
            dimensions,
            gamma,
            r_reduction: r_reduction.to_vec(),
            rows,
            buckets,
            u_evals: eq_table(r_reduction),
            prefix_checkpoints: ALL_PREFIXES
                .iter()
                .map(|prefix| prefix.default_checkpoint::<F>())
                .collect(),
            prefix_tables: Vec::new(),
            suffix_tables: Vec::new(),
            raf_left: RafDecomposition::empty(),
            raf_right: RafDecomposition::empty(),
            raf_identity: RafDecomposition::empty(),
            v_tables: Vec::new(),
            phase_challenges: Vec::new(),
            cycle_challenges: Vec::new(),
            cycle_tables: None,
            rounds_bound: 0,
        };
        kernel.init_phase(0);
        Ok(kernel)
    }

    fn address_bits(&self) -> usize {
        self.dimensions.instruction_address_bits()
    }

    fn phases(&self) -> usize {
        self.address_bits() / CHUNK_LEN
    }

    /// Bits below (and excluding) phase `p`'s chunk.
    fn suffix_len(&self, phase: usize) -> usize {
        self.address_bits() - (phase + 1) * CHUNK_LEN
    }

    /// Phase `p`'s chunk of a lookup index (the `CHUNK_LEN` bits directly
    /// above that phase's suffix).
    fn chunk(&self, lookup_index: u128, phase: usize) -> usize {
        ((lookup_index >> self.suffix_len(phase)) as usize) & (CHUNK_SIZE - 1)
    }

    fn init_phase(&mut self, phase: usize) {
        // Condensation: fold the previous phase's bound-challenge eq weights
        // into the per-cycle mass.
        if phase != 0 {
            let shift = self.suffix_len(phase - 1);
            let Self {
                u_evals,
                v_tables,
                rows,
                ..
            } = self;
            let v_prev = &v_tables[phase - 1];
            for (u, row) in u_evals.iter_mut().zip(rows.iter()) {
                *u *= v_prev[((row.lookup_index >> shift) as usize) & (CHUNK_SIZE - 1)];
            }
        }

        let suffix_len = self.suffix_len(phase);
        let suffix_mask = if suffix_len == 128 {
            u128::MAX
        } else {
            (1u128 << suffix_len) - 1
        };

        // RAF suffix accumulators: one fused scan. The shift suffixes are
        // constant per phase (`2^{suffix_len/2}`, `2^{suffix_len}`), so raw
        // eq mass is accumulated and scaled afterwards.
        let mut q_shift_half_raw = [F::zero(); CHUNK_SIZE];
        let mut q_left = [F::zero(); CHUNK_SIZE];
        let mut q_right = [F::zero(); CHUNK_SIZE];
        let mut q_shift_full_raw = [F::zero(); CHUNK_SIZE];
        let mut q_identity = [F::zero(); CHUNK_SIZE];
        for (row, &u) in self.rows.iter().zip(&self.u_evals) {
            let chunk = self.chunk(row.lookup_index, phase);
            let suffix_bits = row.lookup_index & suffix_mask;
            if row.interleaved_operands {
                q_shift_half_raw[chunk] += u;
                let (left, right) = LookupBits::new(suffix_bits, suffix_len).uninterleave();
                let left = u64::from(left);
                if left != 0 {
                    q_left[chunk] += u * F::from_u64(left);
                }
                let right = u64::from(right);
                if right != 0 {
                    q_right[chunk] += u * F::from_u64(right);
                }
            } else {
                q_shift_full_raw[chunk] += u;
                if suffix_bits != 0 {
                    q_identity[chunk] += u * F::from_u128(suffix_bits);
                }
            }
        }
        let q_shift_half = q_shift_half_raw.map(|value| value.mul_pow_2(suffix_len / 2));
        let q_shift_full = q_shift_full_raw.map(|value| value.mul_pow_2(suffix_len));

        // RAF prefix chunk polynomials, from the registry checkpoints: the
        // identity prefix extends its bound value by the chunk's integer
        // value; the operand prefixes by their uninterleaved half.
        let identity_prefix: Vec<F> = (0..CHUNK_SIZE)
            .map(|x| self.raf_identity.checkpoint.mul_pow_2(CHUNK_LEN) + F::from_u64(x as u64))
            .collect();
        let (left_prefix, right_prefix): (Vec<F>, Vec<F>) = (0..CHUNK_SIZE)
            .map(|x| {
                let (left, right) = LookupBits::new(x as u128, CHUNK_LEN).uninterleave();
                (
                    self.raf_left.checkpoint.mul_pow_2(CHUNK_LEN / 2)
                        + F::from_u64(u64::from(left)),
                    self.raf_right.checkpoint.mul_pow_2(CHUNK_LEN / 2)
                        + F::from_u64(u64::from(right)),
                )
            })
            .unzip();
        self.raf_left.prefix = Polynomial::new(left_prefix);
        self.raf_left.q_shift = Polynomial::new(q_shift_half.to_vec());
        self.raf_left.q_value = Polynomial::new(q_left.to_vec());
        self.raf_right.prefix = Polynomial::new(right_prefix);
        self.raf_right.q_shift = Polynomial::new(q_shift_half.to_vec());
        self.raf_right.q_value = Polynomial::new(q_right.to_vec());
        self.raf_identity.prefix = Polynomial::new(identity_prefix);
        self.raf_identity.q_shift = Polynomial::new(q_shift_full.to_vec());
        self.raf_identity.q_value = Polynomial::new(q_identity.to_vec());

        // Read-checking suffix accumulators, per present table.
        self.suffix_tables = LookupTableKind::<RISCV_XLEN>::iter()
            .filter(|table| !self.buckets[table.index()].is_empty())
            .map(|table| {
                let suffixes = table.suffixes();
                let mut accumulators = vec![vec![F::zero(); CHUNK_SIZE]; suffixes.len()];
                for &j in &self.buckets[table.index()] {
                    let row = &self.rows[j];
                    let u = self.u_evals[j];
                    let chunk = self.chunk(row.lookup_index, phase);
                    let suffix_bits = LookupBits::new(row.lookup_index & suffix_mask, suffix_len);
                    for (accumulator, suffix) in accumulators.iter_mut().zip(suffixes) {
                        let value = suffix.suffix_mle(suffix_bits);
                        if value != 0 {
                            accumulator[chunk] += u * F::from_u64(value);
                        }
                    }
                }
                (
                    table,
                    accumulators.into_iter().map(Polynomial::new).collect(),
                )
            })
            .collect();

        // Table-prefix chunk polynomials from the checkpoints.
        self.prefix_tables = ALL_PREFIXES
            .iter()
            .map(|prefix| {
                Polynomial::new(
                    (0..CHUNK_SIZE)
                        .map(|x| {
                            prefix
                                .evaluate::<F>(
                                    &self.prefix_checkpoints,
                                    LookupBits::new(x as u128, CHUNK_LEN),
                                    suffix_len,
                                )
                                .value()
                        })
                        .collect(),
                )
            })
            .collect();

        self.phase_challenges.clear();
    }

    /// The true quadratic for an address round, sampled at `c ∈ {0,1,2}`.
    fn address_message(&self) -> [F; 3] {
        let gamma_sqr = self.gamma * self.gamma;
        let half = self.prefix_tables[0].evals().len() / 2;
        let mut evals = [F::zero(); 3];
        for (c, eval) in evals.iter_mut().enumerate() {
            let mut read = F::zero();
            let mut left = F::zero();
            let mut right = F::zero();
            let mut identity = F::zero();
            for b in 0..half {
                let prefix_evals: Vec<PrefixEval<F>> = self
                    .prefix_tables
                    .iter()
                    .map(|table| PrefixEval::from(extension_eval(table.evals(), b, half, c)))
                    .collect();
                for (table, suffixes) in &self.suffix_tables {
                    let suffix_evals: Vec<SuffixEval<F>> = suffixes
                        .iter()
                        .map(|q| SuffixEval::from(extension_eval(q.evals(), b, half, c)))
                        .collect();
                    read += table.combine(&prefix_evals, &suffix_evals);
                }
                left += self.raf_left.message_eval(b, half, c);
                right += self.raf_right.message_eval(b, half, c);
                identity += self.raf_identity.message_eval(b, half, c);
            }
            *eval = read + self.gamma * left + gamma_sqr * (right + identity);
        }
        evals
    }

    /// The true degree-`(ra_count + 2)` polynomial for a cycle round, sampled
    /// at `degree + 1` integer points.
    fn cycle_message(&self) -> Result<Vec<F>, SumcheckError<F>> {
        let tables = self
            .cycle_tables
            .as_ref()
            .ok_or(SumcheckError::MissingEvaluationSource { kind: "opening" })?;
        let degree = self.dimensions.num_virtual_ra_polys() + 2;
        let half = tables.eq_reduction.evals().len() / 2;
        let evals =
            (0..=degree)
                .map(|c| {
                    let point = F::from_u64(c as u64);
                    (0..half)
                        .map(|y| {
                            let mut product = tables.eq_reduction.sumcheck_round_eval_with_order(
                                y,
                                point,
                                BindingOrder::LowToHigh,
                            ) * tables
                                .combined_val
                                .sumcheck_round_eval_with_order(y, point, BindingOrder::LowToHigh);
                            for ra in &tables.ra {
                                product *= ra.sumcheck_round_eval_with_order(
                                    y,
                                    point,
                                    BindingOrder::LowToHigh,
                                );
                            }
                            product
                        })
                        .sum()
                })
                .collect();
        Ok(evals)
    }

    /// Handoff at the address/cycle boundary: the fully bound prefix
    /// checkpoints collapse each table's `Val` MLE to a constant, the RAF
    /// checkpoints to the γ-weighted operand/identity constants, and the
    /// per-phase eq tables materialize the virtual `ra` selectors.
    fn init_cycle_rounds(&mut self) {
        let gamma_sqr = self.gamma * self.gamma;
        let empty_bits = LookupBits::new(0, 0);
        let table_values: Vec<F> = LookupTableKind::<RISCV_XLEN>::iter()
            .map(|table| {
                let suffix_evals: Vec<SuffixEval<F>> = table
                    .suffixes()
                    .iter()
                    .map(|suffix| SuffixEval::from(F::from_u64(suffix.suffix_mle(empty_bits))))
                    .collect();
                table.combine(&self.prefix_checkpoints, &suffix_evals)
            })
            .collect();
        let raf_interleaved =
            self.gamma * self.raf_left.checkpoint + gamma_sqr * self.raf_right.checkpoint;
        let raf_identity = gamma_sqr * self.raf_identity.checkpoint;

        let combined_val: Vec<F> = self
            .rows
            .iter()
            .map(|row| {
                let table_value = row
                    .table_index
                    .map_or_else(F::zero, |index| table_values[index]);
                let raf_value = if row.interleaved_operands {
                    raf_interleaved
                } else {
                    raf_identity
                };
                table_value + raf_value
            })
            .collect();

        let ra_count = self.dimensions.num_virtual_ra_polys();
        let phases_per_ra = self.phases() / ra_count;
        let ra = (0..ra_count)
            .map(|i| {
                Polynomial::new(
                    self.rows
                        .iter()
                        .map(|row| {
                            (0..phases_per_ra)
                                .map(|q| {
                                    let phase = i * phases_per_ra + q;
                                    self.v_tables[phase][self.chunk(row.lookup_index, phase)]
                                })
                                .product()
                        })
                        .collect(),
                )
            })
            .collect();

        self.cycle_tables = Some(CycleTables {
            eq_reduction: Polynomial::new(eq_table(&self.r_reduction)),
            combined_val: Polynomial::new(combined_val),
            ra,
        });

        // The address-phase state is dead past this point.
        self.u_evals = Vec::new();
        self.prefix_tables = Vec::new();
        self.suffix_tables = Vec::new();
        self.v_tables = Vec::new();
    }
}

impl<F: Field> ProveRounds<F> for InstructionReadRafKernel<F> {
    fn num_rounds(&self) -> usize {
        self.dimensions.sumcheck_rounds()
    }

    fn compute_message(
        &mut self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let evals = if self.rounds_bound < self.address_bits() {
            self.address_message().to_vec()
        } else {
            self.cycle_message()?
        };
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

    fn ingest_challenge(&mut self, challenge: F, _round: usize) -> Result<(), SumcheckError<F>> {
        if self.rounds_bound < self.address_bits() {
            for table in &mut self.prefix_tables {
                table.bind_with_order(challenge, BindingOrder::HighToLow);
            }
            for (_, suffixes) in &mut self.suffix_tables {
                for q in suffixes {
                    q.bind_with_order(challenge, BindingOrder::HighToLow);
                }
            }
            self.raf_left.bind(challenge);
            self.raf_right.bind(challenge);
            self.raf_identity.bind(challenge);
            self.phase_challenges.push(challenge);

            if self.phase_challenges.len() == CHUNK_LEN {
                let phase = self.rounds_bound / CHUNK_LEN;
                self.v_tables.push(eq_table(&self.phase_challenges));
                for (checkpoint, table) in
                    self.prefix_checkpoints.iter_mut().zip(&self.prefix_tables)
                {
                    *checkpoint = PrefixEval::from(table.evals()[0]);
                }
                self.raf_left.checkpoint = self.raf_left.prefix.evals()[0];
                self.raf_right.checkpoint = self.raf_right.prefix.evals()[0];
                self.raf_identity.checkpoint = self.raf_identity.prefix.evals()[0];

                if phase + 1 < self.phases() {
                    self.init_phase(phase + 1);
                } else {
                    self.init_cycle_rounds();
                }
            }
        } else {
            let tables = self
                .cycle_tables
                .as_mut()
                .ok_or(SumcheckError::MissingEvaluationSource { kind: "opening" })?;
            tables
                .eq_reduction
                .bind_with_order(challenge, BindingOrder::LowToHigh);
            tables
                .combined_val
                .bind_with_order(challenge, BindingOrder::LowToHigh);
            for ra in &mut tables.ra {
                ra.bind_with_order(challenge, BindingOrder::LowToHigh);
            }
            self.cycle_challenges.push(challenge);
        }
        self.rounds_bound += 1;
        Ok(())
    }
}

impl<F: Field> ProveSumcheck<F> for InstructionReadRafKernel<F> {
    type Relation = InstructionReadRaf<F>;

    fn relation(&self) -> &InstructionReadRaf<F> {
        &self.relation
    }

    fn output_claims(&mut self) -> Result<InstructionReadRafOutputClaims<F>, KernelError<F>> {
        if self.rounds_bound != self.num_rounds() {
            return Err(KernelError::NotFullyBound {
                remaining: self.num_rounds() - self.rounds_bound,
            });
        }
        let tables = self
            .cycle_tables
            .as_ref()
            .ok_or(KernelError::InvariantViolation {
                reason: "cycle tables absent after full binding",
            })?;

        // Flag claims at the normalized (big-endian) cycle point: the flags
        // never participate in the round loop (they only appear in the output
        // claim), so they are direct eq-weighted sums over the trace.
        let r_cycle: Vec<F> = self.cycle_challenges.iter().rev().copied().collect();
        let eq_cycle = eq_table(&r_cycle);
        let mut lookup_table_flags = vec![F::zero(); LookupTableKind::<RISCV_XLEN>::COUNT];
        let mut instruction_raf_flag = F::zero();
        for (row, &eq) in self.rows.iter().zip(&eq_cycle) {
            if let Some(index) = row.table_index {
                lookup_table_flags[index] += eq;
            }
            if !row.interleaved_operands {
                instruction_raf_flag += eq;
            }
        }

        Ok(InstructionReadRafOutputClaims {
            lookup_table_flags,
            instruction_ra: tables.ra.iter().map(|ra| ra.evals()[0]).collect(),
            instruction_raf_flag,
        })
    }
}
