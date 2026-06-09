use jolt_field::{AdditiveAccumulator, Field, RingAccumulator, WithAccumulator};
use jolt_lookup_tables::{
    tables::{
        prefixes::{PrefixCheckpoint, NUM_PREFIXES},
        PrefixEval, Prefixes, Suffixes,
    },
    LookupBits, LookupTableKind, ALL_PREFIXES,
};
use jolt_poly::{
    thread::unsafe_allocate_zero_vec, BindingOrder, EqPolynomial, GruenSplitEqPolynomial,
    Polynomial,
};
use rayon::prelude::*;

use crate::{
    cpu::field, BackendError, SumcheckInstructionReadRafOutput,
    SumcheckInstructionReadRafStateRequest,
};

const RV64_XLEN: usize = 64;
const MAX_SUFFIXES_PER_TABLE: usize = 4;
const MAX_INSTRUCTION_READ_RAF_FACTORS: usize = 16;

#[cfg(feature = "prover-harness")]
fn record_instruction_timing(label: &'static str, start: std::time::Instant) {
    crate::timing::record_backend_timing(label, start.elapsed().as_secs_f64() * 1000.0);
}

#[cfg(not(feature = "prover-harness"))]
#[expect(
    dead_code,
    reason = "fallback timing hook is unused without prover-harness"
)]
const fn record_instruction_timing(_label: &'static str, _start: ()) {}

pub struct InstructionReadRafState<F: Field> {
    round: usize,
    log_t: usize,
    address_bits: usize,
    log_m: usize,
    phases: usize,
    ra_virtual_chunk_bits: usize,
    gamma: F,
    gamma2: F,
    lookup_indices: Vec<LookupBits>,
    table_indices: Vec<Option<usize>>,
    active_tables: Vec<LookupTableKind<RV64_XLEN>>,
    active_table_buckets: Vec<Vec<usize>>,
    active_prefix_indices: Vec<usize>,
    interleaved_operands: Vec<bool>,
    prefix_checkpoints: Vec<PrefixCheckpoint<F>>,
    suffix_polys: Vec<Vec<Polynomial<F>>>,
    v: Vec<ExpandingTable<F>>,
    u_evals: Vec<F>,
    raf: RafPrefixSuffixState<F>,
    eq_cycle: GruenSplitEqPolynomial<F>,
    combined_val: Option<Polynomial<F>>,
    ra_polys: Option<Vec<Polynomial<F>>>,
    cycle_bind_scratch: InstructionCycleBindScratch<F>,
    handoff_claim: Option<F>,
    address_challenges: Vec<F>,
    cycle_challenges: Vec<F>,
}

#[derive(Default)]
struct InstructionCycleBindScratch<F: Field> {
    combined_val: Vec<F>,
    ra_polys: Vec<Vec<F>>,
}

impl<F> InstructionReadRafState<F>
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    pub fn new(
        backend: &'static str,
        task: &'static str,
        request: &SumcheckInstructionReadRafStateRequest<F>,
    ) -> Result<Self, BackendError> {
        validate_request(backend, task, request)?;

        let log_m = request.address_bits / request.phases;
        let table_count = LookupTableKind::<RV64_XLEN>::COUNT;
        let mut table_buckets = vec![Vec::new(); table_count];
        let mut lookup_indices = Vec::with_capacity(request.rows.len());
        let mut table_indices = Vec::with_capacity(request.rows.len());
        let mut interleaved_operands = Vec::with_capacity(request.rows.len());

        for (cycle, row) in request.rows.iter().enumerate() {
            lookup_indices.push(LookupBits::new(row.lookup_index, request.address_bits));
            if let Some(table_index) = row.table_index {
                if table_index >= table_count {
                    return invalid(
                        backend,
                        task,
                        format!(
                            "instruction read-RAF table index {table_index} is outside {table_count} tables"
                        ),
                    );
                }
                table_buckets[table_index].push(cycle);
            }
            table_indices.push(row.table_index);
            interleaved_operands.push(row.interleaved_operands);
        }
        let prefix_checkpoints = vec![None.into(); NUM_PREFIXES];
        let eq_evals = EqPolynomial::new(request.fixed_cycle_point.clone()).evaluations();
        let (active_tables, active_table_buckets): (
            Vec<LookupTableKind<RV64_XLEN>>,
            Vec<Vec<usize>>,
        ) = LookupTableKind::<RV64_XLEN>::iter()
            .zip(table_buckets)
            .filter(|(_, rows)| !rows.is_empty())
            .unzip();
        let suffix_polys = active_tables
            .iter()
            .map(|table| {
                table
                    .suffixes()
                    .iter()
                    .map(|_| Polynomial::zeros(log_m))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let active_prefix_indices = active_prefix_indices(&active_tables);

        let mut state = Self {
            round: 0,
            log_t: request.log_t,
            address_bits: request.address_bits,
            log_m,
            phases: request.phases,
            ra_virtual_chunk_bits: request.ra_virtual_chunk_bits,
            gamma: request.gamma,
            gamma2: request.gamma * request.gamma,
            lookup_indices,
            table_indices,
            active_tables,
            active_table_buckets,
            active_prefix_indices,
            interleaved_operands,
            prefix_checkpoints,
            suffix_polys,
            v: (0..request.phases)
                .map(|_| ExpandingTable::new(1 << log_m))
                .collect(),
            u_evals: eq_evals,
            raf: RafPrefixSuffixState::new(),
            eq_cycle: GruenSplitEqPolynomial::new(
                &request.fixed_cycle_point,
                BindingOrder::LowToHigh,
            ),
            combined_val: None,
            ra_polys: None,
            cycle_bind_scratch: InstructionCycleBindScratch::default(),
            handoff_claim: None,
            address_challenges: Vec::with_capacity(request.address_bits),
            cycle_challenges: Vec::with_capacity(request.log_t),
        };
        state.init_phase(0);
        Ok(state)
    }

    pub fn evaluate_round(
        &self,
        backend: &'static str,
        task: &'static str,
        previous_claim: F,
    ) -> Result<jolt_poly::UnivariatePoly<F>, BackendError> {
        if self.round >= self.address_bits + self.log_t {
            return invalid(
                backend,
                task,
                format!(
                    "instruction read-RAF round {} is outside {} rounds",
                    self.round,
                    self.address_bits + self.log_t
                ),
            );
        }
        let polynomial = if self.round < self.address_bits {
            self.address_round_message(previous_claim)
        } else {
            self.cycle_round_message(backend, task, previous_claim)?
        };
        Ok(polynomial)
    }

    pub fn bind(
        &mut self,
        backend: &'static str,
        task: &'static str,
        challenge: F,
    ) -> Result<(), BackendError> {
        if self.round >= self.address_bits + self.log_t {
            return invalid(
                backend,
                task,
                format!(
                    "instruction read-RAF bind round {} is outside {} rounds",
                    self.round,
                    self.address_bits + self.log_t
                ),
            );
        }

        if self.round < self.address_bits {
            let phase = self.round / self.log_m;
            let round = self.round;
            #[cfg(feature = "prover-harness")]
            let address_bind_start = std::time::Instant::now();
            rayon::scope(|scope| {
                let suffix_polys = &mut self.suffix_polys;
                scope.spawn(move |_| {
                    #[cfg(feature = "prover-harness")]
                    let start = std::time::Instant::now();
                    suffix_polys.par_iter_mut().for_each(|table| {
                        table.par_iter_mut().for_each(|poly| {
                            poly.bind_with_order(challenge, BindingOrder::HighToLow);
                        });
                    });
                    #[cfg(feature = "prover-harness")]
                    record_instruction_timing(
                        "stage5.backend.bind.instruction_read_raf.address.suffix",
                        start,
                    );
                });
                let raf = &mut self.raf;
                scope.spawn(move |_| {
                    #[cfg(feature = "prover-harness")]
                    let start = std::time::Instant::now();
                    raf.bind_parallel(challenge);
                    #[cfg(feature = "prover-harness")]
                    record_instruction_timing(
                        "stage5.backend.bind.instruction_read_raf.address.raf",
                        start,
                    );
                });
                let v = &mut self.v[phase];
                scope.spawn(move |_| {
                    #[cfg(feature = "prover-harness")]
                    let start = std::time::Instant::now();
                    v.update(challenge);
                    #[cfg(feature = "prover-harness")]
                    record_instruction_timing(
                        "stage5.backend.bind.instruction_read_raf.address.v",
                        start,
                    );
                });
            });
            #[cfg(feature = "prover-harness")]
            record_instruction_timing(
                "stage5.backend.bind.instruction_read_raf.address.total",
                address_bind_start,
            );
            self.address_challenges.push(challenge);

            self.round += 1;
            if self.address_challenges.len().is_multiple_of(2) {
                let suffix_len = self.address_bits - (round / self.log_m + 1) * self.log_m;
                let len = self.address_challenges.len();
                Prefixes::update_checkpoints(
                    &mut self.prefix_checkpoints,
                    &self.active_prefix_indices,
                    self.address_challenges[len - 2],
                    self.address_challenges[len - 1],
                    round,
                    suffix_len,
                );
            }
            if self.round.is_multiple_of(self.log_m) {
                #[cfg(feature = "prover-harness")]
                let start = std::time::Instant::now();
                self.raf.update_checkpoints();
                #[cfg(feature = "prover-harness")]
                record_instruction_timing(
                    "stage5.backend.bind.instruction_read_raf.address.update_checkpoints",
                    start,
                );
                if phase + 1 < self.phases {
                    #[cfg(feature = "prover-harness")]
                    let start = std::time::Instant::now();
                    self.init_phase(phase + 1);
                    #[cfg(feature = "prover-harness")]
                    record_instruction_timing(
                        "stage5.backend.bind.instruction_read_raf.address.init_phase",
                        start,
                    );
                }
            }
            if self.round == self.address_bits {
                #[cfg(feature = "prover-harness")]
                let start = std::time::Instant::now();
                self.init_cycle_rounds();
                #[cfg(feature = "prover-harness")]
                record_instruction_timing(
                    "stage5.backend.bind.instruction_read_raf.init_cycle_rounds",
                    start,
                );
            }
            return Ok(());
        }

        let eq_cycle = &mut self.eq_cycle;
        let combined_val = &mut self.combined_val;
        let ra_polys = &mut self.ra_polys;
        let combined_val_scratch = &mut self.cycle_bind_scratch.combined_val;
        let ra_polys_scratch = &mut self.cycle_bind_scratch.ra_polys;
        #[cfg(feature = "prover-harness")]
        let cycle_bind_start = std::time::Instant::now();
        rayon::scope(|scope| {
            scope.spawn(|_| {
                #[cfg(feature = "prover-harness")]
                let start = std::time::Instant::now();
                eq_cycle.bind(challenge);
                #[cfg(feature = "prover-harness")]
                record_instruction_timing(
                    "stage5.backend.bind.instruction_read_raf.cycle.eq",
                    start,
                );
            });
            scope.spawn(|_| {
                #[cfg(feature = "prover-harness")]
                let start = std::time::Instant::now();
                if let Some(combined_val) = combined_val {
                    combined_val.bind_low_to_high_reusing_scratch(challenge, combined_val_scratch);
                }
                #[cfg(feature = "prover-harness")]
                record_instruction_timing(
                    "stage5.backend.bind.instruction_read_raf.cycle.combined_val",
                    start,
                );
            });
            scope.spawn(|_| {
                #[cfg(feature = "prover-harness")]
                let start = std::time::Instant::now();
                if let Some(ra_polys) = ra_polys {
                    if ra_polys_scratch.len() < ra_polys.len() {
                        ra_polys_scratch.resize_with(ra_polys.len(), Vec::new);
                    }
                    ra_polys
                        .iter_mut()
                        .zip(ra_polys_scratch.iter_mut())
                        .for_each(|(poly, scratch)| {
                            poly.bind_low_to_high_reusing_scratch(challenge, scratch);
                        });
                }
                #[cfg(feature = "prover-harness")]
                record_instruction_timing(
                    "stage5.backend.bind.instruction_read_raf.cycle.ra_polys",
                    start,
                );
            });
        });
        #[cfg(feature = "prover-harness")]
        record_instruction_timing(
            "stage5.backend.bind.instruction_read_raf.cycle.total",
            cycle_bind_start,
        );
        self.cycle_challenges.push(challenge);
        self.round += 1;
        Ok(())
    }

    pub fn output_claims(&self) -> Result<SumcheckInstructionReadRafOutput<F>, BackendError> {
        let Some(ra_polys) = &self.ra_polys else {
            return invalid(
                "cpu",
                "instruction read-RAF sumcheck output claims",
                "instruction virtual RA polynomials were not materialized",
            );
        };
        if self.round != self.address_bits + self.log_t {
            return invalid(
                "cpu",
                "instruction read-RAF sumcheck output claims",
                format!(
                    "instruction read-RAF output requested after {} of {} rounds",
                    self.round,
                    self.address_bits + self.log_t
                ),
            );
        }
        let r_cycle = self
            .cycle_challenges
            .iter()
            .rev()
            .copied()
            .collect::<Vec<_>>();
        let (lookup_table_flags, instruction_raf_flag) = self.compute_flag_claims(&r_cycle);
        let instruction_ra = ra_polys.iter().map(final_eval).collect::<Vec<_>>();
        let Some(combined_val) = &self.combined_val else {
            return invalid(
                "cpu",
                "instruction read-RAF sumcheck output claims",
                "instruction combined value polynomial was not materialized",
            );
        };
        let final_claim = self.eq_cycle.current_scalar()
            * final_eval(combined_val)
            * instruction_ra.iter().copied().product::<F>();

        Ok(SumcheckInstructionReadRafOutput {
            lookup_table_flags,
            instruction_ra,
            instruction_raf_flag,
            handoff_claim: self.handoff_claim.unwrap_or_else(F::zero),
            final_claim,
        })
    }

    fn init_phase(&mut self, phase: usize) {
        if phase != 0 {
            #[cfg(feature = "prover-harness")]
            let start = std::time::Instant::now();
            let previous = &self.v[phase - 1];
            let suffix_len = (self.phases - phase) * self.log_m;
            let mask = (1usize << self.log_m) - 1;
            self.lookup_indices
                .par_iter()
                .zip(self.u_evals.par_iter_mut())
                .for_each(|(lookup_index, u_eval)| {
                    let (prefix_bits, _) = lookup_index.split(suffix_len);
                    let bound_index = usize::from(prefix_bits) & mask;
                    *u_eval *= previous[bound_index];
                });
            #[cfg(feature = "prover-harness")]
            record_instruction_timing(
                "stage5.backend.instruction_read_raf.init_phase.u_evals",
                start,
            );
        }

        #[cfg(feature = "prover-harness")]
        let start = std::time::Instant::now();
        self.raf.init_q(
            self.log_m,
            phase,
            self.phases,
            &self.u_evals,
            &self.lookup_indices,
            &self.interleaved_operands,
        );
        #[cfg(feature = "prover-harness")]
        record_instruction_timing(
            "stage5.backend.instruction_read_raf.init_phase.raf_init_q",
            start,
        );
        #[cfg(feature = "prover-harness")]
        let start = std::time::Instant::now();
        self.init_suffix_polys(phase);
        #[cfg(feature = "prover-harness")]
        record_instruction_timing(
            "stage5.backend.instruction_read_raf.init_phase.suffix_polys",
            start,
        );
        #[cfg(feature = "prover-harness")]
        let start = std::time::Instant::now();
        self.raf.init_prefix_polys(self.log_m);
        #[cfg(feature = "prover-harness")]
        record_instruction_timing(
            "stage5.backend.instruction_read_raf.init_phase.prefix_polys",
            start,
        );
        #[cfg(feature = "prover-harness")]
        let start = std::time::Instant::now();
        self.v[phase].reset(F::one());
        #[cfg(feature = "prover-harness")]
        record_instruction_timing(
            "stage5.backend.instruction_read_raf.init_phase.v_reset",
            start,
        );
    }

    fn init_suffix_polys(&mut self, phase: usize) {
        let m = 1usize << self.log_m;
        let suffix_len = (self.phases - phase - 1) * self.log_m;
        let mask = m - 1;
        self.suffix_polys = self
            .active_tables
            .par_iter()
            .zip(self.active_table_buckets.par_iter())
            .map(|(table, rows)| {
                let suffixes = table.suffixes();
                let suffix_count = suffixes.len();
                debug_assert!(suffix_count <= MAX_SUFFIXES_PER_TABLE);

                let mut one_suffix = None;
                let mut bit_suffixes = [0usize; MAX_SUFFIXES_PER_TABLE];
                let mut bit_suffix_count = 0usize;
                let mut other_suffixes = [0usize; MAX_SUFFIXES_PER_TABLE];
                let mut other_suffix_count = 0usize;
                for (index, suffix) in suffixes.iter().enumerate() {
                    if matches!(suffix, Suffixes::One) {
                        one_suffix = Some(index);
                    } else if suffix.is_01_valued() {
                        bit_suffixes[bit_suffix_count] = index;
                        bit_suffix_count += 1;
                    } else {
                        other_suffixes[other_suffix_count] = index;
                        other_suffix_count += 1;
                    }
                }

                let chunk_size = rows.len().div_ceil(rayon::current_num_threads()).max(1);
                let flat = rows
                    .par_chunks(chunk_size)
                    .fold(
                        || vec![<F as WithAccumulator>::Accumulator::default(); suffix_count * m],
                        |mut acc, chunk| {
                            for &row in chunk {
                                let lookup_index = self.lookup_indices[row];
                                let (prefix_bits, suffix_bits) = lookup_index.split(suffix_len);
                                let index = usize::from(prefix_bits) & mask;
                                let u = self.u_evals[row];

                                if let Some(suffix_index) = one_suffix {
                                    acc[suffix_index * m + index].add(u);
                                }
                                for &suffix_index in &bit_suffixes[..bit_suffix_count] {
                                    if suffixes[suffix_index].suffix_mle(suffix_bits) == 1 {
                                        acc[suffix_index * m + index].add(u);
                                    }
                                }
                                for &suffix_index in &other_suffixes[..other_suffix_count] {
                                    let value = suffixes[suffix_index].suffix_mle(suffix_bits);
                                    if value != 0 {
                                        acc[suffix_index * m + index].fmadd_u64(u, value);
                                    }
                                }
                            }
                            acc
                        },
                    )
                    .reduce(
                        || vec![<F as WithAccumulator>::Accumulator::default(); suffix_count * m],
                        |mut acc, values| {
                            merge_accumulator_vec::<F>(&mut acc, values);
                            acc
                        },
                    );

                (0..suffix_count)
                    .map(|suffix_index| {
                        let start = suffix_index * m;
                        let values = flat[start..start + m]
                            .par_iter()
                            .copied()
                            .map(AdditiveAccumulator::reduce)
                            .collect::<Vec<_>>();
                        Polynomial::new(values)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
    }

    fn address_round_message(&self, previous_claim: F) -> jolt_poly::UnivariatePoly<F> {
        let read_checking = self.read_checking_message();
        let raf = self.raf.message(self.gamma, self.gamma2);
        let evals = [read_checking[0] + raf[0], read_checking[1] + raf[1]];
        jolt_poly::UnivariatePoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn read_checking_message(&self) -> [F; 2] {
        if self.suffix_polys.is_empty() {
            return [F::zero(); 2];
        }
        let len = self.suffix_polys[0][0].len();
        let b_len = len.trailing_zeros() as usize - 1;
        let r_x = if self.round % 2 == 1 {
            self.address_challenges.last().copied()
        } else {
            None
        };

        let [eval_0, eval_2_left, eval_2_right] = (0..len / 2)
            .into_par_iter()
            .map(|row| {
                let b = LookupBits::new(row as u128, b_len);
                let mut prefixes_at_0 = [PrefixEval::from(F::zero()); NUM_PREFIXES];
                let mut prefixes_at_2 = [PrefixEval::from(F::zero()); NUM_PREFIXES];
                for &index in &self.active_prefix_indices {
                    let prefix = ALL_PREFIXES[index];
                    prefixes_at_0[index] =
                        prefix.prefix_mle(&self.prefix_checkpoints, r_x, 0, b, self.round);
                    prefixes_at_2[index] =
                        prefix.prefix_mle(&self.prefix_checkpoints, r_x, 2, b, self.round);
                }

                let mut evals = [<F as WithAccumulator>::Accumulator::default(); 3];
                for (table, suffix_polys) in self.active_tables.iter().zip(self.suffix_polys.iter())
                {
                    let suffix_count = suffix_polys.len();
                    let mut suffixes_left = [F::zero(); MAX_SUFFIXES_PER_TABLE];
                    let mut suffixes_right = [F::zero(); MAX_SUFFIXES_PER_TABLE];
                    for (index, suffix) in suffix_polys.iter().enumerate() {
                        suffixes_left[index] = suffix.evaluations()[row];
                        suffixes_right[index] = suffix.evaluations()[row + len / 2];
                    }
                    let suffixes_left = &suffixes_left[..suffix_count];
                    let suffixes_right = &suffixes_right[..suffix_count];
                    evals[0].add(table.combine(&prefixes_at_0, suffixes_left));
                    evals[1].add(table.combine(&prefixes_at_2, suffixes_left));
                    evals[2].add(table.combine(&prefixes_at_2, suffixes_right));
                }
                evals
            })
            .reduce(
                || [<F as WithAccumulator>::Accumulator::default(); 3],
                sum_accumulator_arrays::<F, 3>,
            )
            .map(AdditiveAccumulator::reduce);
        [eval_0, eval_2_right + eval_2_right - eval_2_left]
    }

    fn init_cycle_rounds(&mut self) {
        let n_ra = self.address_bits / self.ra_virtual_chunk_bits;
        let phases_per_ra = self.phases / n_ra;
        let mask = (1usize << self.log_m) - 1;
        let ra_polys = self
            .v
            .chunks(phases_per_ra)
            .enumerate()
            .map(|(chunk_index, tables)| {
                let phase_offset = chunk_index * phases_per_ra;
                let evals = self
                    .lookup_indices
                    .par_iter()
                    .map(|lookup_index| {
                        let value = u128::from(*lookup_index);
                        if tables.is_empty() {
                            return F::one();
                        }

                        let mut shift = (self.phases - 1 - phase_offset) * self.log_m;
                        let mut iter = tables.iter();
                        let Some(first) = iter.next() else {
                            return F::one();
                        };
                        let first_index = ((value >> shift) as usize) & mask;
                        let mut acc = first[first_index];

                        for table in iter {
                            shift -= self.log_m;
                            let index = ((value >> shift) as usize) & mask;
                            acc *= table[index];
                        }
                        acc
                    })
                    .collect::<Vec<_>>();
                Polynomial::new(evals)
            })
            .collect::<Vec<_>>();

        debug_assert_eq!(self.address_challenges.len(), self.address_bits);
        let mut prefix_evals = ALL_PREFIXES
            .iter()
            .map(Prefixes::default_checkpoint::<F>)
            .collect::<Vec<_>>();
        for &index in &self.active_prefix_indices {
            prefix_evals[index] = self.prefix_checkpoints[index].unwrap();
        }
        let empty_suffix_bits = LookupBits::new(0, 0);
        let mut table_values = [F::zero(); LookupTableKind::<RV64_XLEN>::COUNT];
        for table in &self.active_tables {
            let suffix_evals = table
                .suffixes()
                .iter()
                .map(|suffix| suffix.evaluate::<F>(empty_suffix_bits))
                .collect::<Vec<_>>();
            table_values[table.index()] = table.combine(&prefix_evals, &suffix_evals);
        }
        let left_prefix = self.raf.left_checkpoint;
        let right_prefix = self.raf.right_checkpoint;
        let identity_prefix = self.raf.identity_checkpoint;
        let raf_interleaved = self.gamma * left_prefix + self.gamma2 * right_prefix;
        let raf_identity = self.gamma2 * identity_prefix;
        let combined_val = self
            .table_indices
            .par_iter()
            .zip(self.interleaved_operands.par_iter())
            .map(|(table_index, interleaved)| {
                let mut value = table_index.map_or_else(F::zero, |index| table_values[index]);
                value += if *interleaved {
                    raf_interleaved
                } else {
                    raf_identity
                };
                value
            })
            .collect::<Vec<_>>();
        self.combined_val = Some(Polynomial::new(combined_val));
        let scratch_capacity = self.lookup_indices.len() / 2;
        self.cycle_bind_scratch.combined_val = Vec::with_capacity(scratch_capacity);
        self.cycle_bind_scratch.ra_polys = (0..ra_polys.len())
            .map(|_| Vec::with_capacity(scratch_capacity))
            .collect();
        self.ra_polys = Some(ra_polys);
        self.u_evals.clear();
        self.suffix_polys.clear();
    }

    fn cycle_round_message(
        &self,
        backend: &'static str,
        task: &'static str,
        previous_claim: F,
    ) -> Result<jolt_poly::UnivariatePoly<F>, BackendError> {
        let Some(ra_polys) = &self.ra_polys else {
            return invalid(
                backend,
                task,
                "instruction RA polynomials are not materialized for cycle round",
            );
        };
        let Some(combined_val) = &self.combined_val else {
            return invalid(
                backend,
                task,
                "instruction combined value polynomial is not materialized for cycle round",
            );
        };
        let quotient_degree = ra_polys.len() + 1;
        if quotient_degree > MAX_INSTRUCTION_READ_RAF_FACTORS {
            return Ok(self.cycle_round_message_heap(ra_polys, combined_val, previous_claim));
        }
        let evals = self
            .eq_cycle
            .e_out_current()
            .par_iter()
            .enumerate()
            .map(|(out_index, &e_out)| {
                let mut accumulators = [<F as WithAccumulator>::Accumulator::default();
                    MAX_INSTRUCTION_READ_RAF_FACTORS];
                let mut pairs = [(F::zero(), F::zero()); MAX_INSTRUCTION_READ_RAF_FACTORS];
                for (in_index, &e_in) in self.eq_cycle.e_in_current().iter().enumerate() {
                    let row = self.eq_cycle.group_index(out_index, in_index);
                    let (val_0, val_1) =
                        combined_val.sumcheck_eval_pair(row, BindingOrder::LowToHigh);
                    pairs[0] = (e_in * val_0, e_in * val_1);
                    for (index, poly) in ra_polys.iter().enumerate() {
                        pairs[index + 1] = poly.sumcheck_eval_pair(row, BindingOrder::LowToHigh);
                    }
                    accumulate_linear_product_evals(
                        &pairs[..quotient_degree],
                        &mut accumulators[..quotient_degree],
                    );
                }
                let mut evals = [F::zero(); MAX_INSTRUCTION_READ_RAF_FACTORS];
                for (eval, accumulator) in evals.iter_mut().zip(accumulators).take(quotient_degree)
                {
                    *eval = accumulator.reduce() * e_out;
                }
                evals
            })
            .reduce(
                || [F::zero(); MAX_INSTRUCTION_READ_RAF_FACTORS],
                sum_arrays::<F, MAX_INSTRUCTION_READ_RAF_FACTORS>,
            );
        Ok(self
            .eq_cycle
            .gruen_poly_from_evals(&evals[..quotient_degree], previous_claim))
    }

    fn cycle_round_message_heap(
        &self,
        ra_polys: &[Polynomial<F>],
        combined_val: &Polynomial<F>,
        previous_claim: F,
    ) -> jolt_poly::UnivariatePoly<F> {
        let quotient_degree = ra_polys.len() + 1;
        let evals = self
            .eq_cycle
            .e_out_current()
            .par_iter()
            .enumerate()
            .map(|(out_index, &e_out)| {
                let mut accumulators =
                    vec![<F as WithAccumulator>::Accumulator::default(); quotient_degree];
                let mut pairs = vec![(F::zero(), F::zero()); quotient_degree];
                for (in_index, &e_in) in self.eq_cycle.e_in_current().iter().enumerate() {
                    let row = self.eq_cycle.group_index(out_index, in_index);
                    let (val_0, val_1) =
                        combined_val.sumcheck_eval_pair(row, BindingOrder::LowToHigh);
                    pairs[0] = (e_in * val_0, e_in * val_1);
                    for (index, poly) in ra_polys.iter().enumerate() {
                        pairs[index + 1] = poly.sumcheck_eval_pair(row, BindingOrder::LowToHigh);
                    }
                    accumulate_linear_product_evals(&pairs, &mut accumulators);
                }
                accumulators
                    .into_iter()
                    .map(|accumulator| accumulator.reduce() * e_out)
                    .collect::<Vec<_>>()
            })
            .reduce(
                || vec![F::zero(); quotient_degree],
                |mut acc, values| {
                    acc.iter_mut()
                        .zip(values)
                        .for_each(|(acc, value)| *acc += value);
                    acc
                },
            );
        self.eq_cycle.gruen_poly_from_evals(&evals, previous_claim)
    }

    fn compute_flag_claims(&self, r_cycle: &[F]) -> (Vec<F>, F) {
        let eq = EqPolynomial::new(r_cycle.to_vec()).evaluations();
        let table_count = LookupTableKind::<RV64_XLEN>::COUNT;
        eq.par_iter()
            .zip(self.table_indices.par_iter())
            .zip(self.interleaved_operands.par_iter())
            .fold(
                || (vec![F::zero(); table_count], F::zero()),
                |(mut flags, mut raf), ((&eq, table_index), interleaved)| {
                    if let Some(table_index) = table_index {
                        flags[*table_index] += eq;
                    }
                    if !*interleaved {
                        raf += eq;
                    }
                    (flags, raf)
                },
            )
            .reduce(
                || (vec![F::zero(); table_count], F::zero()),
                |(mut left_flags, left_raf), (right_flags, right_raf)| {
                    left_flags
                        .iter_mut()
                        .zip(right_flags)
                        .for_each(|(left, right)| *left += right);
                    (left_flags, left_raf + right_raf)
                },
            )
    }
}

struct RafPrefixSuffixState<F: Field> {
    left_checkpoint: F,
    right_checkpoint: F,
    identity_checkpoint: F,
    left: OperandPrefixSuffix<F>,
    right: OperandPrefixSuffix<F>,
    identity: IdentityPrefixSuffix<F>,
}

impl<F: Field> RafPrefixSuffixState<F> {
    fn new() -> Self {
        Self {
            left_checkpoint: F::zero(),
            right_checkpoint: F::zero(),
            identity_checkpoint: F::zero(),
            left: OperandPrefixSuffix::default(),
            right: OperandPrefixSuffix::default(),
            identity: IdentityPrefixSuffix::default(),
        }
    }

    fn init_q(
        &mut self,
        log_m: usize,
        phase: usize,
        phases: usize,
        u_evals: &[F],
        lookup_indices: &[LookupBits],
        interleaved_operands: &[bool],
    ) where
        <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
    {
        let m = 1usize << log_m;
        let suffix_len = (phases - phase - 1) * log_m;
        let mask = m - 1;
        let shift_half = F::one().mul_pow_2(suffix_len / 2);
        let shift_full = F::one().mul_pow_2(suffix_len);
        let chunk_size = lookup_indices
            .len()
            .div_ceil(rayon::current_num_threads())
            .max(1);
        let total_len = 5 * m;

        let rows = lookup_indices
            .par_chunks(chunk_size)
            .zip(u_evals.par_chunks(chunk_size))
            .zip(interleaved_operands.par_chunks(chunk_size))
            .fold(
                || vec![<F as WithAccumulator>::Accumulator::default(); total_len],
                |mut acc, ((lookup_chunk, u_chunk), interleaved_chunk)| {
                    let shift_half_offset = 0;
                    let left_offset = m;
                    let right_offset = 2 * m;
                    let shift_full_offset = 3 * m;
                    let identity_offset = 4 * m;
                    for ((lookup_index, &u), &interleaved) in lookup_chunk
                        .iter()
                        .zip(u_chunk.iter())
                        .zip(interleaved_chunk.iter())
                    {
                        let (prefix_bits, suffix_bits) = lookup_index.split(suffix_len);
                        let index = usize::from(prefix_bits) & mask;
                        if interleaved {
                            acc[shift_half_offset + index].add(u);
                            let (left_bits, right_bits) = suffix_bits.uninterleave();
                            let left_value = u128::from(left_bits) as u64;
                            if left_value != 0 {
                                acc[left_offset + index].fmadd_u64(u, left_value);
                            }
                            let right_value = u128::from(right_bits) as u64;
                            if right_value != 0 {
                                acc[right_offset + index].fmadd_u64(u, right_value);
                            }
                        } else {
                            acc[shift_full_offset + index].add(u);
                            let identity_value = u128::from(suffix_bits);
                            if identity_value != 0 {
                                if identity_value <= u64::MAX as u128 {
                                    acc[identity_offset + index]
                                        .fmadd_u64(u, identity_value as u64);
                                } else {
                                    acc[identity_offset + index]
                                        .fmadd(u, F::from_u128(identity_value));
                                }
                            }
                        }
                    }
                    acc
                },
            )
            .reduce(
                || vec![<F as WithAccumulator>::Accumulator::default(); total_len],
                |mut acc, values| {
                    merge_accumulator_vec::<F>(&mut acc, values);
                    acc
                },
            );

        let q_shift_half = rows[..m]
            .par_iter()
            .copied()
            .map(|value| value.reduce() * shift_half)
            .collect::<Vec<_>>();
        let left_rows = rows[m..2 * m]
            .par_iter()
            .copied()
            .map(AdditiveAccumulator::reduce)
            .collect::<Vec<_>>();
        let right_rows = rows[2 * m..3 * m]
            .par_iter()
            .copied()
            .map(AdditiveAccumulator::reduce)
            .collect::<Vec<_>>();
        let q_shift_full = rows[3 * m..4 * m]
            .par_iter()
            .copied()
            .map(|value| value.reduce() * shift_full)
            .collect::<Vec<_>>();
        let identity_rows = rows[4 * m..5 * m]
            .par_iter()
            .copied()
            .map(AdditiveAccumulator::reduce)
            .collect::<Vec<_>>();

        self.left.q_shift = Polynomial::new(q_shift_half.clone());
        self.left.q_operand = Polynomial::new(left_rows);
        self.right.q_shift = Polynomial::new(q_shift_half);
        self.right.q_operand = Polynomial::new(right_rows);
        self.identity.q_shift = Polynomial::new(q_shift_full);
        self.identity.q_identity = Polynomial::new(identity_rows);
    }

    fn init_prefix_polys(&mut self, log_m: usize) {
        self.left
            .init_prefix_poly(log_m, self.left_checkpoint, OperandSide::Left);
        self.right
            .init_prefix_poly(log_m, self.right_checkpoint, OperandSide::Right);
        self.identity
            .init_prefix_poly(log_m, self.identity_checkpoint);
    }

    fn message(&self, gamma: F, gamma2: F) -> [F; 2]
    where
        <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
    {
        let len = self.identity.q_identity.len();
        let [left_0, left_2, gamma2_0, gamma2_2] = (0..len / 2)
            .into_par_iter()
            .map(|row| {
                let left = self.left.sumcheck_evals(row);
                let right = self.right.sumcheck_evals(row);
                let identity = self.identity.sumcheck_evals(row);
                let mut evals = [<F as WithAccumulator>::Accumulator::default(); 4];
                evals[0].add(left[0]);
                evals[1].add(left[1]);
                evals[2].add(right[0]);
                evals[2].add(identity[0]);
                evals[3].add(right[1]);
                evals[3].add(identity[1]);
                evals
            })
            .reduce(
                || [<F as WithAccumulator>::Accumulator::default(); 4],
                sum_accumulator_arrays::<F, 4>,
            )
            .map(AdditiveAccumulator::reduce);
        [
            gamma * left_0 + gamma2 * gamma2_0,
            gamma * left_2 + gamma2 * gamma2_2,
        ]
    }

    fn bind_parallel(&mut self, challenge: F) {
        rayon::scope(|scope| {
            let left = &mut self.left;
            scope.spawn(move |_| left.bind_parallel(challenge));
            let right = &mut self.right;
            scope.spawn(move |_| right.bind_parallel(challenge));
            let identity = &mut self.identity;
            scope.spawn(move |_| identity.bind_parallel(challenge));
        });
    }

    fn update_checkpoints(&mut self) {
        self.left_checkpoint = final_eval(&self.left.prefix);
        self.right_checkpoint = final_eval(&self.right.prefix);
        self.identity_checkpoint = final_eval(&self.identity.prefix);
    }
}

struct OperandPrefixSuffix<F: Field> {
    prefix: Polynomial<F>,
    q_shift: Polynomial<F>,
    q_operand: Polynomial<F>,
}

impl<F: Field> Default for OperandPrefixSuffix<F> {
    fn default() -> Self {
        Self {
            prefix: Polynomial::new(vec![F::zero()]),
            q_shift: Polynomial::new(vec![F::zero()]),
            q_operand: Polynomial::new(vec![F::zero()]),
        }
    }
}

impl<F: Field> OperandPrefixSuffix<F> {
    fn init_prefix_poly(&mut self, log_m: usize, checkpoint: F, side: OperandSide) {
        let shift = F::one().mul_pow_2(log_m / 2);
        let evals = (0..1usize << log_m)
            .map(|index| {
                let bits = LookupBits::new(index as u128, log_m);
                let (left, right) = bits.uninterleave();
                let operand = match side {
                    OperandSide::Left => u128::from(left),
                    OperandSide::Right => u128::from(right),
                };
                checkpoint * shift + F::from_u128(operand)
            })
            .collect::<Vec<_>>();
        self.prefix = Polynomial::new(evals);
    }

    fn sumcheck_evals(&self, row: usize) -> [F; 2] {
        let prefix = sumcheck_eval_at_0_and_2(&self.prefix, row, BindingOrder::HighToLow);
        let shift = sumcheck_eval_at_0_and_2(&self.q_shift, row, BindingOrder::HighToLow);
        let operand = sumcheck_eval_at_0_and_2(&self.q_operand, row, BindingOrder::HighToLow);
        [
            prefix.0 * shift.0 + operand.0,
            prefix.1 * shift.1 + operand.1,
        ]
    }

    fn bind_parallel(&mut self, challenge: F) {
        rayon::scope(|scope| {
            let prefix = &mut self.prefix;
            scope.spawn(move |_| prefix.bind_with_order(challenge, BindingOrder::HighToLow));
            let q_shift = &mut self.q_shift;
            scope.spawn(move |_| q_shift.bind_with_order(challenge, BindingOrder::HighToLow));
            let q_operand = &mut self.q_operand;
            scope.spawn(move |_| q_operand.bind_with_order(challenge, BindingOrder::HighToLow));
        });
    }
}

struct IdentityPrefixSuffix<F: Field> {
    prefix: Polynomial<F>,
    q_shift: Polynomial<F>,
    q_identity: Polynomial<F>,
}

impl<F: Field> Default for IdentityPrefixSuffix<F> {
    fn default() -> Self {
        Self {
            prefix: Polynomial::new(vec![F::zero()]),
            q_shift: Polynomial::new(vec![F::zero()]),
            q_identity: Polynomial::new(vec![F::zero()]),
        }
    }
}

impl<F: Field> IdentityPrefixSuffix<F> {
    fn init_prefix_poly(&mut self, log_m: usize, checkpoint: F) {
        let shift = F::one().mul_pow_2(log_m);
        let evals = (0..1usize << log_m)
            .map(|index| checkpoint * shift + F::from_u64(index as u64))
            .collect::<Vec<_>>();
        self.prefix = Polynomial::new(evals);
    }

    fn sumcheck_evals(&self, row: usize) -> [F; 2] {
        let prefix = sumcheck_eval_at_0_and_2(&self.prefix, row, BindingOrder::HighToLow);
        let shift = sumcheck_eval_at_0_and_2(&self.q_shift, row, BindingOrder::HighToLow);
        let identity = sumcheck_eval_at_0_and_2(&self.q_identity, row, BindingOrder::HighToLow);
        [
            prefix.0 * shift.0 + identity.0,
            prefix.1 * shift.1 + identity.1,
        ]
    }

    fn bind_parallel(&mut self, challenge: F) {
        rayon::scope(|scope| {
            let prefix = &mut self.prefix;
            scope.spawn(move |_| prefix.bind_with_order(challenge, BindingOrder::HighToLow));
            let q_shift = &mut self.q_shift;
            scope.spawn(move |_| q_shift.bind_with_order(challenge, BindingOrder::HighToLow));
            let q_identity = &mut self.q_identity;
            scope.spawn(move |_| q_identity.bind_with_order(challenge, BindingOrder::HighToLow));
        });
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OperandSide {
    Left,
    Right,
}

#[derive(Clone, Debug)]
struct ExpandingTable<F: Field> {
    values: Vec<F>,
    scratch: Vec<F>,
    len: usize,
}

impl<F: Field> ExpandingTable<F> {
    fn new(capacity: usize) -> Self {
        Self {
            values: unsafe_allocate_zero_vec(capacity),
            scratch: unsafe_allocate_zero_vec(capacity),
            len: 0,
        }
    }

    fn reset(&mut self, value: F) {
        self.values[0] = value;
        self.len = 1;
    }

    fn update(&mut self, challenge: F) {
        self.values[..self.len]
            .par_iter()
            .zip(self.scratch.par_chunks_mut(2))
            .for_each(|(&value, dest)| {
                let eval_1 = value * challenge;
                dest[0] = value - eval_1;
                dest[1] = eval_1;
            });
        std::mem::swap(&mut self.values, &mut self.scratch);
        self.len *= 2;
    }
}

impl<F: Field> std::ops::Index<usize> for ExpandingTable<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < self.len);
        &self.values[index]
    }
}

fn active_prefix_indices(tables: &[LookupTableKind<RV64_XLEN>]) -> Vec<usize> {
    let mut active = [false; NUM_PREFIXES];
    tables
        .iter()
        .flat_map(LookupTableKind::required_prefixes)
        .for_each(|&prefix| mark_prefix_and_dependencies(prefix, &mut active));
    active
        .iter()
        .enumerate()
        .filter_map(|(index, is_active)| is_active.then_some(index))
        .collect()
}

fn mark_prefix_and_dependencies(prefix: Prefixes, active: &mut [bool; NUM_PREFIXES]) {
    let index = prefix as usize;
    if active[index] {
        return;
    }
    active[index] = true;

    match prefix {
        Prefixes::LessThan => mark_prefix_and_dependencies(Prefixes::Eq, active),
        Prefixes::PositiveRemainderLessThanDivisor => {
            mark_prefix_and_dependencies(Prefixes::PositiveRemainderEqualsDivisor, active);
        }
        Prefixes::NegativeDivisorGreaterThanRemainder => {
            mark_prefix_and_dependencies(Prefixes::NegativeDivisorEqualsRemainder, active);
        }
        Prefixes::SignExtension => {
            mark_prefix_and_dependencies(Prefixes::LeftOperandMsb, active);
        }
        Prefixes::LeftShift => {
            mark_prefix_and_dependencies(Prefixes::LeftShiftHelper, active);
        }
        Prefixes::LeftShiftW => {
            mark_prefix_and_dependencies(Prefixes::LeftShiftWHelper, active);
        }
        _ => {}
    }
}

fn sumcheck_eval_at_0_and_2<F: Field>(
    polynomial: &Polynomial<F>,
    row: usize,
    order: BindingOrder,
) -> (F, F) {
    let (lo, hi) = polynomial.sumcheck_eval_pair(row, order);
    (lo, hi + hi - lo)
}

fn accumulate_linear_product_evals<F>(
    pairs: &[(F, F)],
    evals: &mut [<F as WithAccumulator>::Accumulator],
) where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    debug_assert_eq!(pairs.len(), evals.len());
    if pairs.len() == 9 {
        // SAFETY: guarded by the exact length check above.
        let pairs = unsafe { &*pairs.as_ptr().cast::<[(F, F); 9]>() };
        field::eval_linear_product_d9_accumulate(pairs, evals);
        return;
    }
    for (index, eval) in evals.iter_mut().enumerate().take(pairs.len() - 1) {
        let point = F::from_u64((index + 1) as u64);
        let mut product = F::one();
        for &(lo, hi) in pairs {
            product *= lo + point * (hi - lo);
        }
        eval.add(product);
    }

    let mut leading = F::one();
    for &(lo, hi) in pairs {
        leading *= hi - lo;
    }
    if let Some(eval_at_infinity) = evals.last_mut() {
        eval_at_infinity.add(leading);
    }
}

fn final_eval<F: Field>(polynomial: &Polynomial<F>) -> F {
    polynomial
        .evaluations()
        .first()
        .copied()
        .unwrap_or_else(F::zero)
}

fn merge_accumulator_vec<F>(
    left: &mut [<F as WithAccumulator>::Accumulator],
    right: Vec<<F as WithAccumulator>::Accumulator>,
) where
    F: Field,
    <F as WithAccumulator>::Accumulator: AdditiveAccumulator<Element = F>,
{
    left.iter_mut()
        .zip(right)
        .for_each(|(left, right)| left.merge(right));
}

fn sum_arrays<F: Field, const N: usize>(left: [F; N], right: [F; N]) -> [F; N] {
    std::array::from_fn(|index| left[index] + right[index])
}

fn sum_accumulator_arrays<F, const N: usize>(
    mut left: [<F as WithAccumulator>::Accumulator; N],
    right: [<F as WithAccumulator>::Accumulator; N],
) -> [<F as WithAccumulator>::Accumulator; N]
where
    F: Field,
    <F as WithAccumulator>::Accumulator: AdditiveAccumulator<Element = F>,
{
    left.iter_mut()
        .zip(right)
        .for_each(|(left, right)| left.merge(right));
    left
}

fn validate_request<F: Field>(
    backend: &'static str,
    task: &'static str,
    request: &SumcheckInstructionReadRafStateRequest<F>,
) -> Result<(), BackendError> {
    let expected_rows = 1usize << request.log_t;
    if request.rows.len() != expected_rows {
        return invalid(
            backend,
            task,
            format!(
                "instruction read-RAF has {} rows, expected {expected_rows}",
                request.rows.len()
            ),
        );
    }
    if request.fixed_cycle_point.len() != request.log_t {
        return invalid(
            backend,
            task,
            format!(
                "instruction read-RAF fixed cycle point has {} challenges, expected {}",
                request.fixed_cycle_point.len(),
                request.log_t
            ),
        );
    }
    if request.address_bits != 2 * RV64_XLEN {
        return invalid(
            backend,
            task,
            format!(
                "instruction read-RAF address width is {}, expected {}",
                request.address_bits,
                2 * RV64_XLEN
            ),
        );
    }
    if request.phases == 0 || !request.address_bits.is_multiple_of(request.phases) {
        return invalid(
            backend,
            task,
            format!(
                "instruction read-RAF address width {} is not divisible by {} phases",
                request.address_bits, request.phases
            ),
        );
    }
    let log_m = request.address_bits / request.phases;
    if log_m == 0 || log_m > 16 {
        return invalid(
            backend,
            task,
            format!("instruction read-RAF phase width {log_m} is unsupported"),
        );
    }
    if request.ra_virtual_chunk_bits == 0
        || !request
            .address_bits
            .is_multiple_of(request.ra_virtual_chunk_bits)
    {
        return invalid(
            backend,
            task,
            format!(
                "instruction read-RAF address width {} is not divisible by virtual RA chunk {}",
                request.address_bits, request.ra_virtual_chunk_bits
            ),
        );
    }
    let virtual_ra_count = request.address_bits / request.ra_virtual_chunk_bits;
    if !request.phases.is_multiple_of(virtual_ra_count) {
        return invalid(
            backend,
            task,
            format!(
                "instruction read-RAF phase count {} is not divisible by {virtual_ra_count} virtual RA polynomials",
                request.phases
            ),
        );
    }
    Ok(())
}

fn invalid<T>(
    backend: &'static str,
    task: &'static str,
    reason: impl Into<String>,
) -> Result<T, BackendError> {
    Err(BackendError::InvalidRequest {
        backend,
        task,
        reason: reason.into(),
    })
}
