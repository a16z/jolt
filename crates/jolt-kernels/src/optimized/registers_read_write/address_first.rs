//! Address-first register checking with O(T + K) storage.
//!
//! Each address round scans compact access rows, maintaining the current
//! register values weighted by the bound address prefix. Once addresses are
//! bound, only five cycle tables remain; Gruen factoring handles their tail.

use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::protocols::jolt::{JoltDerivedId, JoltPolynomialId, RegistersReadWritePublic};
use jolt_field::JoltField;
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
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

use super::rows::{raw_rd_inc, RegisterCycleRow, SharedRdIndices};
use crate::optimized::support::{collect_rows, pin_derived_term, RoundProgress};
use crate::{KernelError, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError};

#[derive(Clone, Copy)]
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
        weights: Vec<F>,
        eq: Polynomial<F>,
    },
    Cycle {
        rs1: Polynomial<F>,
        rs2: Polynomial<F>,
        wa: Polynomial<F>,
        val: Polynomial<F>,
        inc: Polynomial<F>,
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
    pub(super) fn prepare(
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: &ProverInputs<'_, F, RegistersReadWriteChecking<F>>,
    ) -> Result<Self, KernelError<F>> {
        let dimensions = inputs.relation.register_dimensions();
        let rows = collect_rows::<Access>(witness, 1usize << dimensions.log_t())?;
        session.park(SharedRdIndices(rows.iter().map(|row| row.rd).collect()));
        let gruen =
            GruenSplitEqPolynomial::new(&inputs.points.rd_write_value, BindingOrder::LowToHigh);
        Ok(Self {
            phase: Phase::Address {
                rows,
                weights: vec![F::one()],
                eq: gruen.merge(),
            },
            gruen,
            progress: RoundProgress::new(dimensions.read_write_rounds()),
            gamma: inputs.challenges.gamma,
        })
    }

    fn address_message(
        &self,
        rows: &[Access],
        weights: &[F],
        eq: &Polynomial<F>,
        claim: F,
    ) -> UnivariatePoly<F> {
        let bound = weights.len().trailing_zeros() as usize;
        let mut values = vec![F::zero(); (1usize << REGISTER_ADDRESS_BITS) >> bound];
        let mut evals = [F::zero(); 2];
        let gamma_sq = self.gamma * self.gamma;
        for (j, row) in rows.iter().enumerate() {
            let inc = F::from_i128(row.inc);
            let mut cycle = [F::zero(); 2];
            for (index, coefficient, extra) in [
                (row.rs1, self.gamma, F::zero()),
                (row.rs2, gamma_sq, F::zero()),
                (row.rd, F::one(), inc),
            ] {
                if let Some(index) = index {
                    let index = usize::from(index);
                    let column = index >> bound;
                    let weight = coefficient * weights[index & (weights.len() - 1)];
                    let even = values[column & !1];
                    let odd = values[column | 1];
                    let (at_zero, at_two) = if column & 1 == 0 {
                        (weight, -weight)
                    } else {
                        (F::zero(), weight + weight)
                    };
                    cycle[0] += at_zero * (even + extra);
                    cycle[1] += at_two * (odd + odd - even + extra);
                }
            }
            for i in 0..2 {
                evals[i] += eq.evals()[j] * cycle[i];
            }
            if let Some(rd) = row.rd {
                let rd = usize::from(rd);
                values[rd >> bound] += weights[rd & (weights.len() - 1)] * inc;
            }
        }
        UnivariatePoly::from_evals_and_hint(claim, &evals)
    }

    fn bind(&mut self, r: F) {
        match &mut self.phase {
            Phase::Address { rows, weights, .. } => {
                let n = weights.len();
                for i in 0..n {
                    let high = weights[i] * r;
                    weights.push(high);
                    weights[i] -= high;
                }
                if weights.len() == 1usize << REGISTER_ADDRESS_BITS {
                    let mut rs1 = Vec::with_capacity(rows.len());
                    let mut rs2 = Vec::with_capacity(rows.len());
                    let mut wa = Vec::with_capacity(rows.len());
                    let mut val = Vec::with_capacity(rows.len());
                    let mut inc = Vec::with_capacity(rows.len());
                    let mut value = F::zero();
                    for row in rows.iter() {
                        let hot = |index: Option<u8>| {
                            index.map_or(F::zero(), |k| weights[usize::from(k)])
                        };
                        let write = hot(row.rd);
                        let delta = F::from_i128(row.inc);
                        rs1.push(hot(row.rs1));
                        rs2.push(hot(row.rs2));
                        wa.push(write);
                        val.push(value);
                        inc.push(delta);
                        value += write * delta;
                    }
                    self.phase = Phase::Cycle {
                        rs1: Polynomial::new(rs1),
                        rs2: Polynomial::new(rs2),
                        wa: Polynomial::new(wa),
                        val: Polynomial::new(val),
                        inc: Polynomial::new(inc),
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
                    let _ = table.bind_low_to_high_in_place(r);
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
        _round: usize,
        claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = bind {
            self.bind(r);
        }
        match &self.phase {
            Phase::Address { rows, weights, eq } => {
                Ok(self.address_message(rows, weights, eq, claim))
            }
            Phase::Cycle {
                rs1,
                rs2,
                wa,
                val,
                inc,
            } => {
                let q = self.gruen.par_fold_out_in(
                    || [F::zero(); 2],
                    |acc, row, _x_in, e_in| {
                        let pair = |table: &Polynomial<F>| {
                            let lo = table.evals()[2 * row];
                            [lo, table.evals()[2 * row + 1] - lo]
                        };
                        let (rs1, rs2, wa, val, inc) =
                            (pair(rs1), pair(rs2), pair(wa), pair(val), pair(inc));
                        for i in 0..2 {
                            acc[i] += e_in
                                * (wa[i] * (val[i] + inc[i])
                                    + self.gamma * (rs1[i] + self.gamma * rs2[i]) * val[i]);
                        }
                    },
                    |_x_out, e_out, acc| acc.map(|value| e_out * value),
                    |a, b| [a[0] + b[0], a[1] + b[1]],
                );
                Ok(self.gruen.gruen_poly_deg_3(q[0], q[1], claim))
            }
        }
    }
    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
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
            registers_val: val.evals()[0],
            rs1_ra: rs1.evals()[0],
            rs2_ra: rs2.evals()[0],
            rd_wa: wa.evals()[0],
            rd_inc: inc.evals()[0],
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
