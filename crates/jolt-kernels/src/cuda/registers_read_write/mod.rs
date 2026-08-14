use jolt_claims::protocols::jolt::geometry::registers::{rd_inc_read_write, rs2_ra_read_write};
use jolt_claims::protocols::jolt::relations::registers::{
    RegistersReadWriteInputClaims, RegistersReadWriteOutputClaims,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::{Field, Fr, FromPrimitiveInt};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage4::registers_read_write_checking::RegistersReadWriteChecking;
use jolt_witness::{collect_bundles, JoltWitnessPlane};

use super::{require_context, CudaBackend};
use crate::cuda::common::address_major_matrix::DeviceAddressMajorMatrix;
use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::device::{fr_into, require_fr, require_fr_slice, DeviceFrVec};
use crate::cuda::common::read_write_matrix::DeviceReadWriteMatrix;
use crate::reference::views::dense_view;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

pub(crate) mod rs2_claim;
pub(crate) mod witness;

const COEFF_WIDTH: usize = 2;

pub struct RegistersReadWriteKernel<F: Field> {
    context: &'static CudaKernelContext,
    relation: RegistersReadWriteChecking<F>,
    log_t: usize,
    log_k: usize,
    cycle: Option<DeviceReadWriteMatrix>,
    address: Option<DeviceAddressMajorMatrix>,
    inc: DeviceFrVec,
    eq: GruenSplitEqPolynomial<F>,
    merged_eq: Option<DeviceFrVec>,
    val_init: Vec<Fr>,
    rs2_hot: Vec<Option<usize>>,
    gamma: F,
    challenges: Vec<F>,
    finals: Option<[F; 3]>,
    rounds_bound: usize,
}

impl<F: Field> RegistersReadWriteKernel<F> {
    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        let failed = || SumcheckError::MissingEvaluationSource {
            kind: "cuda registers read-write bind",
        };
        self.challenges.push(challenge);
        if self.rounds_bound < self.log_t {
            let cycle = self.cycle.as_mut().ok_or_else(failed)?;
            cycle.bind(self.context, challenge).map_err(|_| failed())?;
            self.inc = self
                .context
                .bind(
                    &self.inc,
                    require_fr(challenge).map_err(|_| failed())?,
                    BindingOrder::LowToHigh,
                )
                .map_err(|_| failed())?;
            self.eq.bind(challenge);
            self.rounds_bound += 1;
            if self.rounds_bound == self.log_t {
                self.transition().map_err(|_| failed())?;
            }
            return Ok(());
        }
        let address = self.address.as_mut().ok_or_else(failed)?;
        address
            .bind(self.context, challenge)
            .map_err(|_| failed())?;
        self.rounds_bound += 1;
        if self.rounds_bound == self.log_t + self.log_k {
            self.materialize().map_err(|_| failed())?;
        }
        Ok(())
    }

    fn transition(&mut self) -> Result<(), crate::cuda::common::error::CudaError> {
        let merged = self.eq.merge();
        let merged = require_fr_slice(merged.evals())?;
        self.merged_eq = Some(self.context.upload(merged)?);
        let cycle =
            self.cycle
                .take()
                .ok_or(crate::cuda::common::error::CudaError::InvariantViolation {
                    reason: "registers read-write phase 1 ended without a cycle-major matrix",
                })?;
        self.address = Some(cycle.to_address_major(self.context, &self.val_init)?);
        Ok(())
    }

    fn materialize(&mut self) -> Result<(), crate::cuda::common::error::CudaError> {
        let address = self.address.as_ref().ok_or(
            crate::cuda::common::error::CudaError::InvariantViolation {
                reason: "registers read-write phase 2 ended without an address-major matrix",
            },
        )?;
        let [ra, wa, val] = address.materialize(self.context, 1, 1)?;
        let lift = |device: &DeviceFrVec| -> Result<F, crate::cuda::common::error::CudaError> {
            let value = device.first()?;
            fr_into(value).ok_or(crate::cuda::common::error::CudaError::NotImplemented {
                kernel: "CUDA kernels support only the BN254 scalar field",
            })
        };
        self.finals = Some([lift(&ra)?, lift(&wa)?, lift(&val)?]);
        Ok(())
    }
}

impl<F: Field> ProveRounds<F> for RegistersReadWriteKernel<F> {
    fn num_rounds(&self) -> usize {
        self.relation.symbolic().rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge)?;
        }
        let failed = || SumcheckError::MissingEvaluationSource {
            kind: "cuda registers read-write round",
        };
        if round < self.log_t {
            let cycle = self.cycle.as_ref().ok_or_else(failed)?;
            let coeffs: [F; 2] = cycle
                .quadratic_coeffs(self.context, &self.inc, &self.eq)
                .map_err(|_| failed())?;
            return Ok(self
                .eq
                .gruen_poly_deg_3(coeffs[0], coeffs[1], previous_claim));
        }
        let address = self.address.as_ref().ok_or_else(failed)?;
        let merged_eq = self.merged_eq.as_ref().ok_or_else(failed)?;
        let evals: [F; 2] = address
            .round_evals(self.context, &self.inc, merged_eq)
            .map_err(|_| failed())?;
        let mut coefficients =
            UnivariatePoly::from_evals_and_hint(previous_claim, &evals).into_coefficients();
        coefficients.resize(self.relation.degree() + 1, F::zero());
        Ok(UnivariatePoly::new(coefficients))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind)
    }
}

impl<F: Field> SumcheckKernel<F> for RegistersReadWriteKernel<F> {
    type Relation = RegistersReadWriteChecking<F>;

    fn output_claims(
        &mut self,
        _inputs: &RegistersReadWriteInputClaims<F>,
    ) -> Result<RegistersReadWriteOutputClaims<F>, SumcheckKernelError<F>> {
        let remaining = self.relation.symbolic().rounds() - self.rounds_bound;
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let [combined_ra, rd_wa, registers_val] =
            self.finals.ok_or(SumcheckKernelError::InvariantViolation {
                reason: "CUDA registers read-write never materialized its bound tables",
            })?;
        let point = self
            .relation
            .register_dimensions()
            .read_write_opening_point(&self.challenges)
            .map_err(|_| SumcheckKernelError::InvariantViolation {
                reason: "CUDA registers read-write could not normalize its opening point",
            })?;
        let rs2_ra = rs2_claim::rs2_ra_claim(
            self.context,
            &self.rs2_hot,
            &point.r_address,
            &point.r_cycle,
        )
        .map_err(|_| SumcheckKernelError::InvariantViolation {
            reason: "CUDA registers read-write rs2 claim failed",
        })?;
        let gamma_inverse =
            self.gamma
                .inverse()
                .ok_or(SumcheckKernelError::InvariantViolation {
                    reason: "CUDA registers read-write needs an invertible gamma",
                })?;
        let rs1_ra = (combined_ra - self.gamma * self.gamma * rs2_ra) * gamma_inverse;
        let rd_inc = self.inc.first().ok().and_then(fr_into).ok_or(
            SumcheckKernelError::InvariantViolation {
                reason: "CUDA registers read-write increment readback failed",
            },
        )?;
        Ok(RegistersReadWriteOutputClaims {
            registers_val,
            rs1_ra,
            rs2_ra,
            rd_wa,
            rd_inc,
        })
    }
}

impl<F: Field> PrepareKernel<F, RegistersReadWriteChecking<F>> for CudaBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RegistersReadWriteChecking<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RegistersReadWriteChecking<F>>>, KernelError<F>>
    {
        let context = require_context()?;
        let relation = inputs.relation;
        let dimensions = relation.register_dimensions();
        let log_t = dimensions.log_t();
        let log_k = dimensions.log_k();
        if dimensions.phase1_num_rounds() != log_t || dimensions.phase2_num_rounds() != log_k {
            return Err(KernelError::Unsupported {
                reason: "CUDA registers read-write checking supports only the default \
                         read-write config (phase 1 = all cycle rounds, phase 2 = all address \
                         rounds)",
            });
        }
        let r_cycle: &[F] = &inputs.points.rd_write_value;
        if r_cycle.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "registers read-write input point has the wrong variable count",
            });
        }

        let gamma = inputs.challenges.gamma;
        let rows = collect_bundles::<witness::RegistersReadWriteWitness>(witness, 1usize << log_t)?;
        let entries = witness::matrix_entries(
            &rows,
            require_fr(gamma).map_err(|_| KernelError::Unsupported {
                reason: "CUDA kernels support only the BN254 scalar field",
            })?,
        );
        let cycle = DeviceReadWriteMatrix::new(context, &entries, COEFF_WIDTH)?;
        let inc = context.upload(require_fr_slice(&dense_view(
            witness,
            rd_inc_read_write(),
        )?)?)?;

        Ok(Box::new(RegistersReadWriteKernel {
            context,
            relation: relation.clone(),
            log_t,
            log_k,
            cycle: Some(cycle),
            address: None,
            inc,
            eq: GruenSplitEqPolynomial::new(r_cycle, BindingOrder::LowToHigh),
            merged_eq: None,
            val_init: vec![Fr::from_u64(0); 1usize << log_k],
            rs2_hot: witness.hot_indices(rs2_ra_read_write().polynomial_id())?,
            gamma,
            challenges: Vec::with_capacity(log_t + log_k),
            finals: None,
            rounds_bound: 0,
        }))
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations and fixture errors fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::ReadWriteDimensions;
    use jolt_claims::protocols::jolt::relations::registers::{
        RegistersReadWriteChallenges, RegistersReadWriteInputClaims,
    };
    use jolt_claims::protocols::jolt::{
        JoltCommittedPolynomial, JoltPolynomialId, JoltVirtualPolynomial,
    };
    use jolt_claims::OutputClaims;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_program::execution::{RegisterRead, RegisterState, RegisterWrite};
    use jolt_verifier::stages::stage4::registers_read_write_checking::RegistersReadWriteChecking;
    use jolt_witness::__private::TraceRow;
    use jolt_witness::witnesses::{Extract, RdInc, ToField, WitnessEnv};
    use jolt_witness::{FixedBackend, PolynomialEncoding, Shape};

    use super::CudaBackend;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{arb_point, drive, fr, reference_input_claim, RowPlane};
    use crate::reference::ReferenceBackend;
    use crate::{PrepareKernel, ProofSession, ProverInputs};
    use proptest::prelude::*;

    const LOG_T: usize = 6;
    const LOG_K: usize = 7;

    type Activity = (Option<u8>, Option<u8>, Option<u8>);

    const ACTIVITY: [Activity; 12] = [
        (None, None, None),
        (Some(3), None, None),
        (None, Some(5), None),
        (None, None, Some(7)),
        (Some(9), Some(9), None),
        (Some(11), None, Some(11)),
        (None, Some(13), Some(13)),
        (Some(2), Some(4), Some(6)),
        (Some(6), Some(6), Some(6)),
        (Some(1), Some(2), Some(1)),
        (Some(1), Some(2), Some(2)),
        (Some(120), Some(121), Some(122)),
    ];

    fn rows(seed: u64) -> (Vec<TraceRow>, Vec<Vec<Fr>>, Vec<Fr>) {
        let cycles = 1usize << LOG_T;
        let registers = 1usize << LOG_K;
        let mut state = vec![0u64; registers];
        let mut rows = Vec::with_capacity(cycles);
        let mut val = vec![Fr::from_u64(0); registers * cycles];
        let mut rs1_ra = vec![Fr::from_u64(0); registers * cycles];
        let mut rs2_ra = vec![Fr::from_u64(0); registers * cycles];
        let mut rd_wa = vec![Fr::from_u64(0); registers * cycles];

        for cycle in 0..cycles {
            for (register, value) in state.iter().copied().enumerate() {
                val[register * cycles + cycle] = Fr::from_u64(value);
            }

            let (rs1, rs2, rd) = ACTIVITY[(cycle + seed as usize) % ACTIVITY.len()];
            let mut registers_state = RegisterState::default();
            if let Some(register) = rs1 {
                registers_state.rs1 = Some(RegisterRead {
                    register,
                    value: state[register as usize],
                });
                rs1_ra[register as usize * cycles + cycle] = Fr::from_u64(1);
            }
            if let Some(register) = rs2 {
                registers_state.rs2 = Some(RegisterRead {
                    register,
                    value: state[register as usize],
                });
                rs2_ra[register as usize * cycles + cycle] = Fr::from_u64(1);
            }
            if let Some(register) = rd {
                let pre_value = state[register as usize];
                let post_value = pre_value
                    .wrapping_add(seed.wrapping_mul(cycle as u64 + 1))
                    .wrapping_add(u64::from(register));
                registers_state.rd = Some(RegisterWrite {
                    register,
                    pre_value,
                    post_value,
                });
                rd_wa[register as usize * cycles + cycle] = Fr::from_u64(1);
                state[register as usize] = post_value;
            }
            rows.push(TraceRow {
                registers: registers_state,
                ..TraceRow::default()
            });
        }

        let preprocessing = RowPlane::new(FixedBackend::new(), "inc probe", LOG_T, Vec::new());
        let env = WitnessEnv::new(jolt_witness::ProgramSource::program_preprocessing(
            &preprocessing,
        ));
        let inc: Vec<Fr> = rows
            .iter()
            .enumerate()
            .map(|(index, row)| {
                RdInc::extract(row, rows.get(index + 1), &env)
                    .expect("rd increment")
                    .to_field()
            })
            .collect();

        (rows, vec![val, rs1_ra, rs2_ra, rd_wa], inc)
    }

    fn witness(seed: u64) -> RowPlane {
        let (rows, grids, inc) = rows(seed);
        let mut backend = FixedBackend::new();
        let ids = [
            JoltVirtualPolynomial::RegistersVal,
            JoltVirtualPolynomial::Rs1Ra,
            JoltVirtualPolynomial::Rs2Ra,
            JoltVirtualPolynomial::RdWa,
        ];
        for (id, grid) in ids.into_iter().zip(grids) {
            backend
                .insert(
                    JoltPolynomialId::Virtual(id),
                    Shape::new(LOG_K + LOG_T, PolynomialEncoding::Dense),
                    grid,
                )
                .expect("insert register grid");
        }
        backend
            .insert(
                JoltPolynomialId::Committed(JoltCommittedPolynomial::RdInc),
                Shape::new(LOG_T, PolynomialEncoding::Dense),
                inc,
            )
            .expect("insert rd_inc");
        RowPlane::new(backend, "cuda registers_read_write fixture", LOG_T, rows)
    }

    #[test]
    fn fixture_grids_agree_with_trace_rows() {
        let cycles = 1usize << LOG_T;
        for seed in 0..ACTIVITY.len() as u64 {
            let (rows, grids, inc) = rows(seed);
            let [val, rs1_ra, rs2_ra, rd_wa] = [&grids[0], &grids[1], &grids[2], &grids[3]];
            let mut reads = 0usize;
            let mut writes = 0usize;

            for (cycle, row) in rows.iter().enumerate() {
                let hot = |grid: &Vec<Fr>, register: usize| grid[register * cycles + cycle];
                let value = |register: usize| val[register * cycles + cycle];

                for read in [row.registers.rs1, row.registers.rs2].into_iter().flatten() {
                    assert_eq!(
                        value(usize::from(read.register)),
                        Fr::from_u64(read.value),
                        "seed {seed} cycle {cycle}: val grid disagrees with the read value",
                    );
                    reads += 1;
                }
                if let Some(write) = row.registers.rd {
                    assert_eq!(
                        value(usize::from(write.register)),
                        Fr::from_u64(write.pre_value),
                        "seed {seed} cycle {cycle}: val grid disagrees with the write pre-value",
                    );
                    assert_eq!(
                        inc[cycle],
                        Fr::from_u64(write.post_value) - Fr::from_u64(write.pre_value),
                        "seed {seed} cycle {cycle}: inc disagrees with the write",
                    );
                    writes += 1;
                }

                for register in 0..1usize << LOG_K {
                    let expect_hot = |operand: Option<u8>| {
                        Fr::from_u64(u64::from(operand == Some(register as u8)))
                    };
                    assert_eq!(
                        hot(rs1_ra, register),
                        expect_hot(row.registers.rs1.map(|read| read.register)),
                        "seed {seed} cycle {cycle} register {register}: rs1_ra grid is wrong",
                    );
                    assert_eq!(
                        hot(rs2_ra, register),
                        expect_hot(row.registers.rs2.map(|read| read.register)),
                        "seed {seed} cycle {cycle} register {register}: rs2_ra grid is wrong",
                    );
                    assert_eq!(
                        hot(rd_wa, register),
                        expect_hot(row.registers.rd.map(|write| write.register)),
                        "seed {seed} cycle {cycle} register {register}: rd_wa grid is wrong",
                    );
                }
            }

            assert!(
                reads > 0 && writes > 0,
                "seed {seed} exercised no reads or no writes",
            );
        }
    }

    proptest! {
        #[test]
        fn registers_read_write_matches_reference(
            seed in any::<u64>(),
            rd_write_value in arb_point(LOG_T),
            rs1_value in arb_point(LOG_T),
            rs2_value in arb_point(LOG_T),
            gamma in any::<u64>().prop_map(fr),
            challenges in arb_point(LOG_K + LOG_T),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };
            let witness = witness(seed);
            let relation = RegistersReadWriteChecking::<Fr>::new(ReadWriteDimensions::new(
                LOG_T,
                LOG_K,
                LOG_T,
                LOG_K,
            ));
            let claims = RegistersReadWriteInputClaims {
                rd_write_value: Fr::from_u64(0),
                rs1_value: Fr::from_u64(0),
                rs2_value: Fr::from_u64(0),
            };
            let points = RegistersReadWriteInputClaims {
                rd_write_value,
                rs1_value,
                rs2_value,
            };
            let challenge_set = RegistersReadWriteChallenges { gamma };
            let make_inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenge_set,
            };

            let input_claim = reference_input_claim(&witness, make_inputs);
            let mut expected_kernel = ReferenceBackend
                .prepare(&mut ProofSession::default(), &witness, make_inputs())
                .expect("reference prepare");
            let mut got_kernel = CudaBackend
                .prepare(&mut ProofSession::default(), &witness, make_inputs())
                .expect("cuda prepare");

            let expected = drive(&mut *expected_kernel, input_claim, &challenges);
            let got = drive(&mut *got_kernel, input_claim, &challenges);
            prop_assert_eq!(got, expected);

            let expected_claims = expected_kernel.output_claims(&claims).expect("reference claims");
            let got_claims = got_kernel.output_claims(&claims).expect("cuda claims");
            prop_assert_eq!(got_claims.opening_values(), expected_claims.opening_values());
        }
    }
}
