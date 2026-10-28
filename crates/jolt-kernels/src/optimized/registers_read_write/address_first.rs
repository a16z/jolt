//! Address-first register checking with O(T) storage for the fixed register domain.
//!
//! Raw increment checkpoints let each address round scan compact access
//! chunks independently, weighted by the bound address prefix. The same
//! checkpoints parallelize the handoff to five cycle tables; the dead address
//! equality buffer becomes the rs1 table. Gruen factoring handles the tail.

use core::mem::MaybeUninit;

use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::protocols::jolt::{JoltDerivedId, JoltPolynomialId, RegistersReadWritePublic};
use jolt_field::JoltField;
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, TensorEqTable, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage4::registers_read_write_checking::{
    RegistersReadWriteChecking, RegistersReadWriteOutputClaims,
};
use jolt_witness::__private::TraceRow;
use jolt_witness::witnesses::WitnessEnv;
use jolt_witness::{JoltWitnessPlane, WitnessBundle, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::rows::{raw_rd_inc, RegisterCycleRow, SharedRdIndices};
use crate::optimized::prefix_product::PrefixColumn;
use crate::optimized::support::{
    bind_prefix_pairs_buffer_len, fallible_collect_bytes, map_indices, pin_derived_term,
    pool_workers, BundleStore, GruenRoundMessage, RoundProgress,
};
use crate::{KernelError, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError};

const CHUNK: usize = 1 << 12;

#[derive(Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct Access {
    rs1: Option<u8>,
    rs2: Option<u8>,
    rd: Option<u8>,
    inc: i128,
}

impl WitnessBundle for Access {
    fn from_row(
        row: &TraceRow,
        next: Option<&TraceRow>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let row = RegisterCycleRow::from_row(row, next, env)?;
        Ok(Self {
            rs1: row.rs1.map(|(k, _)| k),
            rs2: row.rs2.map(|(k, _)| k),
            rd: row.rd.map(|(k, ..)| k),
            inc: raw_rd_inc(&row),
        })
    }
    fn annotated_ids() -> Vec<JoltPolynomialId> {
        Vec::new()
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum Phase<F: JoltField> {
    Address {
        rows: Vec<Access>,
        checkpoints: Vec<[i128; 1 << REGISTER_ADDRESS_BITS]>,
        weights: Vec<F>,
        eq: Vec<F>,
        final_values: Box<[i128; 1 << REGISTER_ADDRESS_BITS]>,
    },
    Cycle {
        rs1: PrefixColumn<F>,
        rs2: PrefixColumn<F>,
        wa: PrefixColumn<F>,
        val: PrefixColumn<F>,
        inc: PrefixColumn<F>,
    },
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(super) struct AddressFirstKernel<F: JoltField> {
    phase: Phase<F>,
    gruen: GruenSplitEqPolynomial<F>,
    progress: RoundProgress,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    gamma: F,
}

impl<F: JoltField> AddressFirstKernel<F> {
    /// Peak bytes over the `prefix_cycles` cycles before the trace's constant
    /// suffix that `prepare` keeps, or `None` past `usize::MAX`: the most of
    /// the access rows' fallible collect (`fallible_collect_bytes`), the
    /// handoff to the cycle rounds and the cycle rounds. The handoff holds the
    /// rows, the increment checkpoints, the final register values and the `K`
    /// bound address weights while it fills the five `prefix_cycles`-entry
    /// cycle tables (`rs1` in the address eq table's allocation) through one
    /// checkpoint value per chunk task; the cycle rounds hold the five tables
    /// and the bound half the first bind collects
    /// (`bind_prefix_pairs_buffer_len`). Preparation after the collect and
    /// the address rounds hold the rows, the checkpoints, the eq table and at
    /// most `K` checkpoint values per chunk task, less than the handoff; the
    /// eq table's tensor factors, live while it fills, are not counted.
    pub(super) fn peak_bytes(prefix_cycles: usize) -> Option<usize> {
        let field = |count: usize| count.checked_mul(size_of::<F>());
        let checkpoint = size_of::<[i128; 1 << REGISTER_ADDRESS_BITS]>();
        let chunks = prefix_cycles.div_ceil(CHUNK);
        let tables = field(prefix_cycles.checked_mul(5)?)?;
        let handoff = [
            prefix_cycles.checked_mul(size_of::<Access>())?,
            chunks.checked_mul(checkpoint)?,
            checkpoint,
            field(1 << REGISTER_ADDRESS_BITS)?,
            tables,
            field(pool_workers().min(chunks))?,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)?;
        let cycle_rounds =
            tables.checked_add(field(bind_prefix_pairs_buffer_len(prefix_cycles))?)?;
        Some(
            fallible_collect_bytes::<Access>(prefix_cycles)?
                .max(handoff)
                .max(cycle_rounds),
        )
    }

    pub(super) fn prepare(
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: &ProverInputs<'_, F, RegistersReadWriteChecking<F>>,
    ) -> Result<Self, KernelError<F>> {
        let dimensions = inputs.relation.register_dimensions();
        let cycles = 1usize << dimensions.log_t();
        let store = BundleStore::<Access>::resolve(witness, cycles)?;
        let access = store.access();
        let prefix = if access.row(cycles - 1)? == Access::default() {
            access.constant_suffix_start(cycles)?
        } else {
            cycles
        };
        #[cfg(feature = "parallel")]
        let rows: Vec<Access> = (0..prefix)
            .into_par_iter()
            .map(|row| access.row(row))
            .collect::<Result<_, _>>()?;
        #[cfg(not(feature = "parallel"))]
        let rows: Vec<Access> = (0..prefix)
            .map(|row| access.row(row))
            .collect::<Result<_, _>>()?;
        let mut checkpoints = Vec::with_capacity(rows.len().div_ceil(CHUNK));
        let mut values = [0i128; 1 << REGISTER_ADDRESS_BITS];
        let rd_indices = rows
            .iter()
            .enumerate()
            .map(|(cycle, row)| {
                if cycle % CHUNK == 0 {
                    checkpoints.push(values);
                }
                if let Some(rd) = row.rd {
                    values[usize::from(rd)] += row.inc;
                }
                row.rd
            })
            .collect();
        session.park(SharedRdIndices::new(rd_indices, cycles));
        let gruen =
            GruenSplitEqPolynomial::new(&inputs.points.rd_write_value, BindingOrder::LowToHigh);
        Ok(Self {
            phase: Phase::Address {
                rows,
                checkpoints,
                weights: vec![F::one()],
                eq: {
                    let eq = TensorEqTable::new(&inputs.points.rd_write_value);
                    map_indices(prefix, |row| eq.evaluate_index(row))
                },
                final_values: Box::new(values),
            },
            gruen,
            progress: RoundProgress::new(dimensions.read_write_rounds()),
            gamma: inputs.challenges.gamma,
        })
    }

    fn address_message(
        &self,
        rows: &[Access],
        checkpoints: &[[i128; 1 << REGISTER_ADDRESS_BITS]],
        weights: &[F],
        eq: &[F],
        claim: F,
    ) -> UnivariatePoly<F> {
        let bound = weights.len().trailing_zeros() as usize;
        let gamma_sq = self.gamma * self.gamma;
        let read_weights: Vec<_> = weights
            .iter()
            .map(|&w| [self.gamma * w, gamma_sq * w])
            .collect();
        let fold = |(chunk, rows): (usize, &[Access])| {
            let mut values = checkpoint_values(&checkpoints[chunk], weights);
            let mut evals = [F::zero(); 2];
            let eq = &eq[chunk * CHUNK..];
            for (j, row) in rows.iter().enumerate() {
                let inc = F::from_i128(row.inc);
                let mut cycle = [F::zero(); 2];
                for (operand, (index, extra)) in
                    [(row.rs1, F::zero()), (row.rs2, F::zero()), (row.rd, inc)]
                        .into_iter()
                        .enumerate()
                {
                    if let Some(index) = index {
                        let index = usize::from(index);
                        let column = index >> bound;
                        let low = index & (weights.len() - 1);
                        let weight = if operand < 2 {
                            read_weights[low][operand]
                        } else {
                            weights[low]
                        };
                        let even = values[column & !1];
                        let odd = values[column | 1];
                        let at_two = odd + odd - even + extra;
                        if column & 1 == 0 {
                            cycle[0] += weight * (even + extra);
                            cycle[1] -= weight * at_two;
                        } else {
                            cycle[1] += (weight + weight) * at_two;
                        }
                    }
                }
                for i in 0..2 {
                    evals[i] += eq[j] * cycle[i];
                }
                if let Some(rd) = row.rd {
                    let rd = usize::from(rd);
                    values[rd >> bound] += weights[rd & (weights.len() - 1)] * inc;
                }
            }
            evals
        };
        let add = |a: [F; 2], b: [F; 2]| [a[0] + b[0], a[1] + b[1]];
        #[cfg(feature = "parallel")]
        let evals = rows
            .par_chunks(CHUNK)
            .enumerate()
            .map(fold)
            .reduce(|| [F::zero(); 2], add);
        #[cfg(not(feature = "parallel"))]
        let evals = rows
            .chunks(CHUNK)
            .enumerate()
            .map(fold)
            .fold([F::zero(); 2], add);
        UnivariatePoly::from_evals_and_hint(claim, &evals)
    }

    fn bind(&mut self, r: F) {
        match &mut self.phase {
            Phase::Address {
                rows,
                checkpoints,
                weights,
                eq,
                final_values,
            } => {
                let n = weights.len();
                for i in 0..n {
                    let high = weights[i] * r;
                    weights.push(high);
                    weights[i] -= high;
                }
                if weights.len() == 1usize << REGISTER_ADDRESS_BITS {
                    // The address equality table is dead; reuse its allocation
                    // for rs1 so the handoff never holds six dense cycle tables.
                    let mut rs1 = std::mem::take(eq);
                    rs1.clear();
                    let mut rs2 = Vec::with_capacity(rows.len());
                    let mut wa = Vec::with_capacity(rows.len());
                    let mut val = Vec::with_capacity(rows.len());
                    let mut inc = Vec::with_capacity(rows.len());
                    let fill =
                        |(chunk, [rs1, rs2, wa, val, inc]): (usize, [&mut [MaybeUninit<F>]; 5])| {
                            let mut value = checkpoint_values(&checkpoints[chunk], weights)[0];
                            for (i, row) in rows[chunk * CHUNK..].iter().take(rs1.len()).enumerate()
                            {
                                let hot = |index: Option<u8>| {
                                    index.map_or(F::zero(), |k| weights[usize::from(k)])
                                };
                                let write = hot(row.rd);
                                let delta = F::from_i128(row.inc);
                                let _ = rs1[i].write(hot(row.rs1));
                                let _ = rs2[i].write(hot(row.rs2));
                                let _ = wa[i].write(write);
                                let _ = val[i].write(value);
                                let _ = inc[i].write(delta);
                                value += write * delta;
                            }
                        };
                    #[cfg(feature = "parallel")]
                    (
                        rs1.spare_capacity_mut()[..rows.len()].par_chunks_mut(CHUNK),
                        rs2.spare_capacity_mut()[..rows.len()].par_chunks_mut(CHUNK),
                        wa.spare_capacity_mut()[..rows.len()].par_chunks_mut(CHUNK),
                        val.spare_capacity_mut()[..rows.len()].par_chunks_mut(CHUNK),
                        inc.spare_capacity_mut()[..rows.len()].par_chunks_mut(CHUNK),
                    )
                        .into_par_iter()
                        .map(|(a, b, c, d, e)| [a, b, c, d, e])
                        .enumerate()
                        .for_each(fill);
                    #[cfg(not(feature = "parallel"))]
                    rs1.spare_capacity_mut()[..rows.len()]
                        .chunks_mut(CHUNK)
                        .zip(rs2.spare_capacity_mut()[..rows.len()].chunks_mut(CHUNK))
                        .zip(wa.spare_capacity_mut()[..rows.len()].chunks_mut(CHUNK))
                        .zip(val.spare_capacity_mut()[..rows.len()].chunks_mut(CHUNK))
                        .zip(inc.spare_capacity_mut()[..rows.len()].chunks_mut(CHUNK))
                        .map(|((((a, b), c), d), e)| [a, b, c, d, e])
                        .enumerate()
                        .for_each(fill);
                    // SAFETY: every disjoint chunk initialized its full slice
                    // of all five tables before the iterators completed.
                    unsafe {
                        rs1.set_len(rows.len());
                        rs2.set_len(rows.len());
                        wa.set_len(rows.len());
                        val.set_len(rows.len());
                        inc.set_len(rows.len());
                    }
                    self.phase = Phase::Cycle {
                        rs1: PrefixColumn {
                            values: rs1,
                            tail: [F::zero(); 2],
                        },
                        rs2: PrefixColumn {
                            values: rs2,
                            tail: [F::zero(); 2],
                        },
                        wa: PrefixColumn {
                            values: wa,
                            tail: [F::zero(); 2],
                        },
                        val: PrefixColumn {
                            values: val,
                            tail: [checkpoint_values(final_values, weights)[0]; 2],
                        },
                        inc: PrefixColumn {
                            values: inc,
                            tail: [F::zero(); 2],
                        },
                    };
                }
            }
            Phase::Cycle {
                rs1,
                rs2,
                wa,
                val,
                inc,
            } => {
                for table in [rs1, rs2, wa, val, inc] {
                    let _ = table.bind(r);
                }
                self.gruen.bind(r);
            }
        }
        self.progress.advance();
    }
}

impl<F: JoltField> ProveRounds<F> for AddressFirstKernel<F> {
    fn num_rounds(&self) -> usize {
        self.progress.total()
    }
    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = bind {
            self.bind(r);
        }
        match &self.phase {
            Phase::Address {
                rows,
                checkpoints,
                weights,
                eq,
                ..
            } => Ok(self.address_message(rows, checkpoints, weights, eq, claim)),
            Phase::Cycle {
                rs1,
                rs2,
                wa,
                val,
                inc,
            } => {
                let summand = |rs1: F, rs2: F, wa: F, val: F, inc: F| {
                    wa * (val + inc) + self.gamma * (rs1 + self.gamma * rs2) * val
                };
                let q = self.gruen.par_fold_out_in_prefix(
                    rs1.values.len().div_ceil(2),
                    || [F::zero(); 2],
                    |acc, row, _x_in, e_in| {
                        let pair = |table: &PrefixColumn<F>| {
                            let (lo, hi) = table.pair(row);
                            [lo, hi - lo]
                        };
                        let (rs1, rs2, wa, val, inc) =
                            (pair(rs1), pair(rs2), pair(wa), pair(val), pair(inc));
                        for i in 0..2 {
                            acc[i] += e_in * summand(rs1[i], rs2[i], wa[i], val[i], inc[i]);
                        }
                    },
                    |_x_out, e_out, acc| acc.map(|value| e_out * value),
                    |a, b| [a[0] + b[0], a[1] + b[1]],
                );
                self.gruen.checked_cubic(q[0], q[1], claim, round, || {
                    self.gruen.par_fold_out_in_prefix(
                        rs1.values.len().div_ceil(2),
                        F::zero,
                        |sum, pair, _, weight| {
                            let row = 2 * pair + 1;
                            *sum += weight
                                * summand(
                                    rs1.value(row),
                                    rs2.value(row),
                                    wa.value(row),
                                    val.value(row),
                                    inc.value(row),
                                );
                        },
                        |_, weight, sum| weight * sum,
                        |a, b| a + b,
                    )
                })
            }
        }
    }
    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

/// Each checkpoint is the raw increment prefix at a chunk boundary. Binding
/// that K-sized state lets chunks scan independently without another trace pass.
fn checkpoint_values<F: JoltField>(
    checkpoint: &[i128; 1 << REGISTER_ADDRESS_BITS],
    weights: &[F],
) -> Vec<F> {
    checkpoint
        .chunks(weights.len())
        .map(|values| {
            values
                .iter()
                .zip(weights)
                .map(|(&value, &weight)| F::from_i128(value) * weight)
                .sum()
        })
        .collect()
}

impl<F: JoltField> SumcheckKernel<F> for AddressFirstKernel<F> {
    type Relation = RegistersReadWriteChecking<F>;
    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<RegistersReadWriteOutputClaims<F>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        let Phase::Cycle {
            rs1,
            rs2,
            wa,
            val,
            inc,
        } = &self.phase
        else {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.num_rounds(),
            });
        };
        Ok(RegistersReadWriteOutputClaims {
            registers_val: val.value(0),
            rs1_ra: rs1.value(0),
            rs2_ra: rs2.value(0),
            rd_wa: wa.value(0),
            rd_inc: inc.value(0),
        })
    }
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        inputs: &SumcheckInputPoints<F, Self::Relation>,
        outputs: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        pin_derived_term(
            relation,
            JoltDerivedId::from(RegistersReadWritePublic::EqCycle),
            inputs,
            outputs,
            challenges,
            self.gruen.current_scalar(),
        )
    }
}
