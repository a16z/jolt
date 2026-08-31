use jolt_claims::protocols::jolt::relations::registers::{
    RegistersReadWriteInputClaims, RegistersReadWriteOutputClaims,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::{Field, Fr, FromPrimitiveInt};
use jolt_poly::{BindingOrder, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage4::registers_read_write_checking::RegistersReadWriteChecking;
use jolt_witness::JoltWitnessPlane;

use crate::cuda::witness::session_window_residency;

use super::{require_context, CudaBackend};
use crate::cuda::common::address_major_matrix::DeviceAddressMajorMatrix;
use crate::cuda::common::context::{context_for, CudaKernelContext};
use crate::cuda::common::device::{fr_into, require_fr, DeviceFrVec};
use crate::cuda::common::devices::{fan_out, witness_windows, CycleWindow, DeviceTask};
use crate::cuda::common::error::{backend, CudaError};
use crate::cuda::common::read_write_matrix::{CycleShard, ShardedReadWriteMatrix};
use crate::cuda::common::split_eq::DeviceSplitEq;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

pub(crate) mod device_rows;
pub(crate) mod rs2_claim;
pub(crate) mod witness;

type PreparedShard<F> = (
    CycleShard<F>,
    (usize, cudarc::driver::CudaSlice<u32>, CycleWindow),
);

pub struct RegistersReadWriteKernel<F: Field> {
    context: &'static CudaKernelContext,
    relation: RegistersReadWriteChecking<F>,
    log_t: usize,
    log_k: usize,
    cycle: Option<ShardedReadWriteMatrix<F>>,
    address: Option<DeviceAddressMajorMatrix>,
    inc: Option<DeviceFrVec>,
    eq: DeviceSplitEq<F>,
    merged_eq: Option<DeviceFrVec>,
    val_init: Vec<Fr>,
    rs2_windows: Vec<(usize, cudarc::driver::CudaSlice<u32>, CycleWindow)>,
    gamma: F,
    challenges: Vec<F>,
    finals: Option<[F; 3]>,
    rounds_bound: usize,
}

#[cfg(feature = "allocative")]
impl<F: Field> allocative::Allocative for RegistersReadWriteKernel<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("inc"),
            self.inc.as_ref().map_or(0, DeviceFrVec::device_bytes),
        );
        visitor.visit_simple(
            allocative::Key::new("cycle_matrix"),
            self.cycle
                .as_ref()
                .map_or(0, ShardedReadWriteMatrix::device_bytes),
        );
        visitor.visit_simple(
            allocative::Key::new("address_matrix"),
            self.address
                .as_ref()
                .map_or(0, DeviceAddressMajorMatrix::device_bytes),
        );
        visitor.visit_simple(allocative::Key::new("eq"), self.eq.device_bytes());
        visitor.visit_simple(
            allocative::Key::new("merged_eq"),
            self.merged_eq.as_ref().map_or(0, DeviceFrVec::device_bytes),
        );
        visitor.visit_simple(
            allocative::Key::new("val_init"),
            self.val_init.len() * size_of::<Fr>(),
        );
        visitor.visit_simple(
            allocative::Key::new("challenges"),
            self.challenges.len() * size_of::<F>(),
        );
        visitor.exit();
    }
}

impl<F: Field> RegistersReadWriteKernel<F> {
    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        let failed = || SumcheckError::MissingEvaluationSource {
            kind: "cuda registers read-write bind",
        };
        self.challenges.push(challenge);
        if self.rounds_bound < self.log_t {
            let bound = self.rounds_bound;
            let cycle = self.cycle.as_mut().ok_or_else(failed)?;
            cycle
                .bind(challenge, bound)
                .map_err(backend("cuda registers read-write cycle bind"))?;
            self.eq.bind(challenge);
            self.rounds_bound += 1;
            if self.rounds_bound == self.log_t {
                self.transition().map_err(backend(
                    "cuda registers read-write cycle-to-address transition",
                ))?;
            }
            return Ok(());
        }
        let address = self.address.as_mut().ok_or_else(failed)?;
        address
            .bind(self.context, challenge)
            .map_err(backend("cuda registers read-write address bind"))?;
        self.rounds_bound += 1;
        if self.rounds_bound == self.log_t + self.log_k {
            self.materialize()
                .map_err(backend("cuda registers read-write materialize"))?;
        }
        Ok(())
    }

    fn transition(&mut self) -> Result<(), crate::cuda::common::error::CudaError> {
        self.merged_eq = Some(self.eq.merge(self.context)?);
        let (cycle, inc) = self
            .cycle
            .as_mut()
            .and_then(ShardedReadWriteMatrix::take_parts)
            .ok_or(crate::cuda::common::error::CudaError::InvariantViolation {
                reason: "registers read-write phase 1 ended without a cycle-major matrix",
            })?;
        self.cycle = None;
        self.address = Some(cycle.to_address_major(self.context, &self.val_init)?);
        self.inc = Some(inc);
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
                .quadratic_coeffs(&self.eq)
                .map_err(backend("cuda registers read-write round"))?;
            return Ok(self
                .eq
                .gruen_poly_deg_3(coeffs[0], coeffs[1], previous_claim));
        }
        let address = self.address.as_ref().ok_or_else(failed)?;
        let merged_eq = self.merged_eq.as_ref().ok_or_else(failed)?;
        let inc = self.inc.as_ref().ok_or_else(failed)?;
        let evals: [F; 2] = address
            .round_evals(self.context, inc, merged_eq)
            .map_err(backend("cuda registers read-write round"))?;
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
        let shards = self.rs2_windows.len();
        let mut rs2_ra = F::zero();
        for (ordinal, indices, window) in &self.rs2_windows {
            let device = context_for(*ordinal).ok_or(SumcheckKernelError::InvariantViolation {
                reason: "a registers read-write rs2 window names an absent device",
            })?;
            rs2_ra += rs2_claim::rs2_ra_claim_window(
                device,
                indices,
                window.len,
                &point.r_address,
                &point.r_cycle,
                *ordinal,
                shards,
            )
            .map_err(|_| SumcheckKernelError::InvariantViolation {
                reason: "CUDA registers read-write rs2 claim failed",
            })?;
        }
        let gamma_inverse =
            self.gamma
                .inverse()
                .ok_or(SumcheckKernelError::InvariantViolation {
                    reason: "CUDA registers read-write needs an invertible gamma",
                })?;
        let rs1_ra = (combined_ra - self.gamma * self.gamma * rs2_ra) * gamma_inverse;
        let rd_inc = self
            .inc
            .as_ref()
            .and_then(|inc| inc.first().ok())
            .and_then(fr_into)
            .ok_or(SumcheckKernelError::InvariantViolation {
                reason: "CUDA registers read-write increment readback failed",
            })?;
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
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RegistersReadWriteChecking<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RegistersReadWriteChecking<F>>>, KernelError<F>>
    {
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
        let device_gamma = require_fr(gamma).map_err(|_| KernelError::Unsupported {
            reason: "CUDA kernels support only the BN254 scalar field",
        })?;
        let cycles = 1usize << log_t;
        let windows = witness_windows(cycles);
        let shards = windows.len();
        let mut resident = Vec::with_capacity(shards);
        for (ordinal, window) in windows.iter().enumerate() {
            let device = context_for(ordinal).ok_or(KernelError::InvariantViolation {
                reason: "a registers read-write window names an absent device",
            })?;
            let (trace, atoms) =
                session_window_residency(device, session, witness, cycles, window)?;
            resident.push((ordinal, device, trace, atoms));
        }

        let tasks: Vec<DeviceTask<'_, PreparedShard<F>, CudaError>> = resident
            .into_iter()
            .zip(windows.iter())
            .map(|((ordinal, device, trace, atoms), window)| {
                let task: DeviceTask<'_, PreparedShard<F>, CudaError> = Box::new(move || {
                    let rows = device_rows::DeviceRegisterRows::from_device(
                        device, &trace, &atoms, window.len,
                    )?;
                    let matrix = rows.matrix(device, device_gamma)?;
                    let inc = rows.inc(device)?;
                    let eq = DeviceSplitEq::new_window(
                        device,
                        r_cycle,
                        BindingOrder::LowToHigh,
                        ordinal,
                        shards,
                    )?;
                    Ok((
                        CycleShard {
                            ordinal,
                            matrix,
                            inc,
                            eq,
                        },
                        (
                            ordinal,
                            rows.into_rs2_address(),
                            CycleWindow {
                                start: window.start,
                                len: window.len,
                            },
                        ),
                    ))
                });
                task
            })
            .collect();

        let mut cycle_shards = Vec::with_capacity(shards);
        let mut rs2_windows = Vec::with_capacity(shards);
        for (shard, rs2) in fan_out(tasks)? {
            cycle_shards.push(shard);
            rs2_windows.push(rs2);
        }

        Ok(Box::new(RegistersReadWriteKernel {
            context: require_context()?,
            relation: relation.clone(),
            log_t,
            log_k,
            cycle: Some(ShardedReadWriteMatrix::new(cycle_shards, log_t)?),
            address: None,
            inc: None,
            eq: DeviceSplitEq::new(require_context()?, r_cycle, BindingOrder::LowToHigh)?,
            merged_eq: None,
            val_init: vec![Fr::from_u64(0); 1usize << log_k],
            rs2_windows,
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
    use jolt_verifier::stages::stage4::registers_read_write_checking::RegistersReadWriteChecking;
    use jolt_witness::__private::TraceRow;
    use jolt_witness::{FixedBackend, PolynomialEncoding, Shape};

    use super::CudaBackend;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{
        arb_point, drive, fr, reference_input_claim, register_rows, RowPlane, REGISTER_ACTIVITY,
    };
    use crate::reference::ReferenceBackend;
    use crate::{PrepareKernel, ProofSession, ProverInputs};
    use proptest::prelude::*;

    const LOG_T: usize = 6;
    const LOG_K: usize = 7;

    #[test]
    fn device_register_rows_match_the_host_encoder() {
        let Some(context) = shared_context() else {
            return;
        };
        let plane = witness(7);
        let cycles = 1usize << LOG_T;
        let rows: Vec<super::witness::RegistersReadWriteWitness> =
            jolt_witness::collect_bundles(&plane, cycles).expect("reference register rows");
        let expected = super::device_rows::DeviceRegisterRows::upload(context, &rows)
            .expect("host-encoded register rows");

        let mut session = ProofSession::default();
        let trace =
            crate::cuda::witness::session_device_trace::<Fr>(context, &mut session, &plane, cycles)
                .expect("device residency");
        let atoms =
            crate::cuda::witness::session_atom_columns::<Fr>(context, &mut session, &plane, cycles)
                .expect("atom columns");
        let got =
            super::device_rows::DeviceRegisterRows::from_device(context, &trace, &atoms, cycles)
                .expect("device-gathered register rows");

        for (name, got, expected) in [
            ("rs1 address", got.rs1_address(), expected.rs1_address()),
            ("rs2 address", got.rs2_address(), expected.rs2_address()),
            ("rd address", got.rd_address(), expected.rd_address()),
        ] {
            let expected = context.download_u32(expected).expect("download");
            assert!(
                expected.iter().any(|&slot| slot != expected[0]),
                "every {name} is identical, so a kernel ignoring the row would pass",
            );
            assert_eq!(
                context.download_u32(got).expect("download"),
                expected,
                "the {name} column diverges",
            );
        }
        for (name, got, expected) in [
            ("rs1 value", got.rs1_value(), expected.rs1_value()),
            ("rs2 value", got.rs2_value(), expected.rs2_value()),
            ("rd pre value", got.rd_pre_value(), expected.rd_pre_value()),
            (
                "rd post value",
                got.rd_post_value(),
                expected.rd_post_value(),
            ),
        ] {
            assert_eq!(
                context.download_u64(got).expect("download"),
                context.download_u64(expected).expect("download"),
                "the {name} column diverges",
            );
        }
    }

    fn rows(seed: u64) -> (Vec<TraceRow>, Vec<Vec<Fr>>, Vec<Fr>) {
        let fixture = register_rows(LOG_T, LOG_K, seed);
        (
            fixture.rows,
            vec![fixture.val, fixture.rs1_ra, fixture.rs2_ra, fixture.rd_wa],
            fixture.inc,
        )
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
        for seed in 0..REGISTER_ACTIVITY.len() as u64 {
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
