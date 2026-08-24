use jolt_claims::protocols::jolt::relations::ram::{
    RamReadWriteInputClaims, RamReadWriteOutputClaims,
};
use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial};
use jolt_claims::SymbolicSumcheck;
use jolt_field::{Field, Fr};
use jolt_poly::{BindingOrder, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage2::ram_read_write_checking::RamReadWriteChecking;
use jolt_witness::JoltWitnessPlane;

use crate::cuda::witness::session_window_residency;

use super::{require_context, CudaBackend};
use crate::cuda::common::address_major_matrix::DeviceAddressMajorMatrix;
use crate::cuda::common::context::{context_for, CudaKernelContext};
use crate::cuda::common::device::{fr_into, require_fr, require_fr_slice, DeviceFrVec};
use crate::cuda::common::devices::witness_windows;
use crate::cuda::common::error::CudaError;
use crate::cuda::common::read_write_matrix::{CycleShard, ShardedReadWriteMatrix};
use crate::cuda::common::split_eq::DeviceSplitEq;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

pub(crate) mod device_rows;
pub(crate) mod witness;

pub struct RamReadWriteKernel<F: Field> {
    context: &'static CudaKernelContext,
    relation: RamReadWriteChecking<F>,
    log_t: usize,
    log_k: usize,
    cycle: Option<ShardedReadWriteMatrix<F>>,
    address: Option<DeviceAddressMajorMatrix>,
    inc: Option<DeviceFrVec>,
    eq: DeviceSplitEq<F>,
    merged_eq: Option<DeviceFrVec>,
    val_init: Vec<Fr>,
    finals: Option<[F; 2]>,
    rounds_bound: usize,
}

#[cfg(feature = "allocative")]
impl<F: Field> allocative::Allocative for RamReadWriteKernel<F> {
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
        visitor.exit();
    }
}

impl<F: Field> RamReadWriteKernel<F> {
    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        let failed = || SumcheckError::MissingEvaluationSource {
            kind: "cuda RAM read-write bind",
        };
        if self.rounds_bound < self.log_t {
            let bound = self.rounds_bound;
            let cycle = self.cycle.as_mut().ok_or_else(failed)?;
            cycle.bind(challenge, bound).map_err(|_| failed())?;
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

    fn transition(&mut self) -> Result<(), CudaError> {
        self.merged_eq = Some(self.eq.merge(self.context)?);
        let (cycle, inc) = self
            .cycle
            .as_mut()
            .and_then(ShardedReadWriteMatrix::take_parts)
            .ok_or(CudaError::InvariantViolation {
                reason: "RAM read-write phase 1 ended without a cycle-major matrix",
            })?;
        self.cycle = None;
        self.address = Some(cycle.to_address_major(self.context, &self.val_init)?);
        self.inc = Some(inc);
        Ok(())
    }

    fn materialize(&mut self) -> Result<(), CudaError> {
        let address = self.address.as_ref().ok_or(CudaError::InvariantViolation {
            reason: "RAM read-write phase 2 ended without an address-major matrix",
        })?;
        let [ra, _, val] = address.materialize(self.context, 1, 1)?;
        let lift = |device: &DeviceFrVec| -> Result<F, CudaError> {
            fr_into(device.first()?).ok_or(CudaError::NotImplemented {
                kernel: "CUDA kernels support only the BN254 scalar field",
            })
        };
        self.finals = Some([lift(&ra)?, lift(&val)?]);
        Ok(())
    }
}

impl<F: Field> ProveRounds<F> for RamReadWriteKernel<F> {
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
            kind: "cuda RAM read-write round",
        };
        if round < self.log_t {
            let cycle = self.cycle.as_ref().ok_or_else(failed)?;
            let coeffs: [F; 2] = cycle.quadratic_coeffs(&self.eq).map_err(|_| failed())?;
            return Ok(self
                .eq
                .gruen_poly_deg_3(coeffs[0], coeffs[1], previous_claim));
        }
        let address = self.address.as_ref().ok_or_else(failed)?;
        let merged_eq = self.merged_eq.as_ref().ok_or_else(failed)?;
        let inc = self.inc.as_ref().ok_or_else(failed)?;
        let evals: [F; 2] = address
            .round_evals(self.context, inc, merged_eq)
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

impl<F: Field> SumcheckKernel<F> for RamReadWriteKernel<F> {
    type Relation = RamReadWriteChecking<F>;

    fn output_claims(
        &mut self,
        _inputs: &RamReadWriteInputClaims<F>,
    ) -> Result<RamReadWriteOutputClaims<F>, SumcheckKernelError<F>> {
        let remaining = self.relation.symbolic().rounds() - self.rounds_bound;
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let [ra, val] = self.finals.ok_or(SumcheckKernelError::InvariantViolation {
            reason: "CUDA RAM read-write never materialized its bound tables",
        })?;
        let inc = self
            .inc
            .as_ref()
            .and_then(|inc| inc.first().ok())
            .and_then(fr_into)
            .ok_or(SumcheckKernelError::InvariantViolation {
                reason: "CUDA RAM read-write increment readback failed",
            })?;
        Ok(RamReadWriteOutputClaims { val, ra, inc })
    }
}

impl<F: Field> PrepareKernel<F, RamReadWriteChecking<F>> for CudaBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamReadWriteChecking<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamReadWriteChecking<F>>>, KernelError<F>>
    {
        let context = require_context()?;
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let log_t = dimensions.log_t();
        let log_k = relation.ram_log_k();
        if dimensions.phase1_num_rounds() != log_t || dimensions.phase2_num_rounds() != log_k {
            return Err(KernelError::Unsupported {
                reason: "CUDA RAM read-write checking supports only the default read-write \
                         config (phase 1 = all cycle rounds, phase 2 = all address rounds)",
            });
        }
        let tau_low = relation.product_tau_low();
        if tau_low.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "RAM read-write cycle-eq point has the wrong variable count",
            });
        }

        let gamma = require_fr(inputs.challenges.gamma).map_err(|_| KernelError::Unsupported {
            reason: "CUDA kernels support only the BN254 scalar field",
        })?;
        let cycles = 1usize << log_t;
        let windows = witness_windows(cycles);
        let shards = windows.len();
        let mut cycle_shards = Vec::with_capacity(shards);
        for (ordinal, window) in windows.iter().enumerate() {
            let device = context_for(ordinal).ok_or(KernelError::InvariantViolation {
                reason: "a RAM read-write window names an absent device",
            })?;
            let (trace, _) = session_window_residency(device, session, witness, cycles, window)?;
            let rows =
                device_rows::DeviceRamRows::from_device(&trace, 1usize << log_k, window.len)?;
            cycle_shards.push(CycleShard {
                ordinal,
                matrix: rows.matrix(device, gamma)?,
                inc: rows.inc(device)?,
                eq: DeviceSplitEq::new_window(
                    device,
                    tau_low,
                    BindingOrder::LowToHigh,
                    ordinal,
                    shards,
                )?,
            });
        }
        let val_init = require_fr_slice(
            &witness.oracle_table(JoltPolynomialId::Virtual(JoltVirtualPolynomial::RamValInit))?,
        )?
        .to_vec();
        if val_init.len() != 1usize << log_k {
            return Err(KernelError::InvariantViolation {
                reason: "RAM initial-state table does not span the address domain",
            });
        }

        Ok(Box::new(RamReadWriteKernel {
            context,
            relation: relation.clone(),
            log_t,
            log_k,
            cycle: Some(ShardedReadWriteMatrix::new(cycle_shards, log_t)?),
            address: None,
            inc: None,
            eq: DeviceSplitEq::new(context, tau_low, BindingOrder::LowToHigh)?,
            merged_eq: None,
            val_init,
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
    use jolt_claims::protocols::jolt::{
        JoltCommittedPolynomial, JoltPolynomialId, JoltVirtualPolynomial,
    };
    use jolt_claims::OutputClaims;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_program::execution::{RamAccess, RamRead, RamWrite, TraceRow};
    use jolt_verifier::stages::stage2::ram_read_write_checking::{
        RamReadWriteChallenges, RamReadWriteChecking, RamReadWriteInputClaims,
    };
    use jolt_witness::{FixedBackend, PolynomialEncoding, Shape};

    use super::CudaBackend;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::{arb_point, drive, fr, reference_input_claim, RowPlane};
    use crate::reference::ReferenceBackend;
    use crate::{PrepareKernel, ProofSession, ProverInputs};
    use proptest::prelude::*;

    const LOG_T: usize = 6;
    const RAM_LOG_K: usize = 4;

    struct Fixture {
        rows: Vec<TraceRow>,
        val: Vec<Fr>,
        ra: Vec<Fr>,
        inc: Vec<Fr>,
        val_init: Vec<Fr>,
    }

    fn fixture(seed: u64) -> Fixture {
        let cycles = 1usize << LOG_T;
        let words = 1usize << RAM_LOG_K;
        let mut state: Vec<u64> = (0..words as u64)
            .map(|word| seed.wrapping_mul(word + 1) % 5_000)
            .collect();
        let val_init: Vec<Fr> = state.iter().map(|value| Fr::from_u64(*value)).collect();

        let mut rows = Vec::with_capacity(cycles);
        let mut val = vec![Fr::from_u64(0); words * cycles];
        let mut ra = vec![Fr::from_u64(0); words * cycles];
        let mut inc = vec![Fr::from_u64(0); cycles];

        for cycle in 0..cycles {
            for (word, value) in state.iter().copied().enumerate() {
                val[word * cycles + cycle] = Fr::from_u64(value);
            }

            let word = 1 + (cycle + seed as usize) % (words - 1);
            let address = 8 * word as u64;
            let access = match (cycle + seed as usize) % 4 {
                0 => RamAccess::NoOp,
                1 => RamAccess::Read(RamRead {
                    address,
                    value: state[word],
                }),
                2 => RamAccess::Write(RamWrite {
                    address,
                    pre_value: state[word],
                    post_value: state[word].wrapping_add(1 + cycle as u64),
                }),
                _ => RamAccess::Write(RamWrite {
                    address,
                    pre_value: state[word],
                    post_value: state[word],
                }),
            };

            match access {
                RamAccess::Read(_) => {
                    ra[word * cycles + cycle] = Fr::from_u64(1);
                }
                RamAccess::Write(write) => {
                    ra[word * cycles + cycle] = Fr::from_u64(1);
                    inc[cycle] = Fr::from_u64(write.post_value) - Fr::from_u64(write.pre_value);
                    state[word] = write.post_value;
                }
                RamAccess::NoOp => {}
            }

            rows.push(TraceRow {
                ram_access: access,
                ..TraceRow::default()
            });
        }

        Fixture {
            rows,
            val,
            ra,
            inc,
            val_init,
        }
    }

    #[test]
    fn device_ram_rows_match_the_host_encoder() {
        let Some(context) = shared_context() else {
            return;
        };
        let plane = witness(7);
        let cycles = 1usize << LOG_T;
        let rows: Vec<super::witness::RamReadWriteWitness> =
            jolt_witness::collect_bundles(&plane, cycles).expect("reference RAM rows");
        let expected = super::device_rows::DeviceRamRows::upload(context, &rows)
            .expect("host-encoded RAM rows");

        let mut session = ProofSession::default();
        let trace =
            crate::cuda::witness::session_device_trace::<Fr>(context, &mut session, &plane, cycles)
                .expect("device residency");
        let got =
            super::device_rows::DeviceRamRows::from_device(&trace, 1usize << RAM_LOG_K, cycles)
                .expect("device-gathered RAM rows");

        let expected_address = context.download_u32(expected.address()).expect("download");
        assert!(
            expected_address
                .iter()
                .any(|&word| word != expected_address[0]),
            "every remapped RAM address is identical, so a kernel ignoring the row would pass",
        );
        assert_eq!(
            context.download_u32(got.address()).expect("download"),
            expected_address,
            "the remapped RAM address column diverges",
        );
        for (name, got, expected) in [
            ("read value", got.read_value(), expected.read_value()),
            ("write value", got.write_value(), expected.write_value()),
        ] {
            assert_eq!(
                context.download_u64(got).expect("download"),
                context.download_u64(expected).expect("download"),
                "the RAM {name} column diverges",
            );
        }
    }

    fn witness(seed: u64) -> RowPlane {
        let f = fixture(seed);
        let mut backend = FixedBackend::new();
        for (id, grid) in [
            (JoltVirtualPolynomial::RamVal, f.val),
            (JoltVirtualPolynomial::RamRa, f.ra),
        ] {
            backend
                .insert(
                    JoltPolynomialId::Virtual(id),
                    Shape::new(RAM_LOG_K + LOG_T, PolynomialEncoding::Dense),
                    grid,
                )
                .expect("insert ram grid");
        }
        backend
            .insert(
                JoltPolynomialId::Virtual(JoltVirtualPolynomial::RamValInit),
                Shape::new(RAM_LOG_K, PolynomialEncoding::Dense),
                f.val_init,
            )
            .expect("insert ram_val_init");
        backend
            .insert(
                JoltPolynomialId::Committed(JoltCommittedPolynomial::RamInc),
                Shape::new(LOG_T, PolynomialEncoding::Dense),
                f.inc,
            )
            .expect("insert ram_inc");
        RowPlane::new(backend, "cuda ram_read_write fixture", LOG_T, f.rows)
    }

    #[test]
    fn fixture_grids_agree_with_trace_rows() {
        let cycles = 1usize << LOG_T;
        let words = 1usize << RAM_LOG_K;
        for seed in 0..6u64 {
            let f = fixture(seed);
            let mut accesses = 0usize;
            let mut writes = 0usize;

            for (cycle, row) in f.rows.iter().enumerate() {
                let touched = match row.ram_access {
                    RamAccess::Read(read) => {
                        let word = (read.address / 8) as usize;
                        assert_eq!(
                            f.val[word * cycles + cycle],
                            Fr::from_u64(read.value),
                            "seed {seed} cycle {cycle}: val grid disagrees with the read value",
                        );
                        assert_eq!(f.inc[cycle], Fr::from_u64(0), "reads must not increment");
                        accesses += 1;
                        Some(word)
                    }
                    RamAccess::Write(write) => {
                        let word = (write.address / 8) as usize;
                        assert_eq!(
                            f.val[word * cycles + cycle],
                            Fr::from_u64(write.pre_value),
                            "seed {seed} cycle {cycle}: val grid disagrees with the write pre-value",
                        );
                        assert_eq!(
                            f.inc[cycle],
                            Fr::from_u64(write.post_value) - Fr::from_u64(write.pre_value),
                            "seed {seed} cycle {cycle}: inc disagrees with the write",
                        );
                        accesses += 1;
                        writes += 1;
                        Some(word)
                    }
                    RamAccess::NoOp => {
                        assert_eq!(f.inc[cycle], Fr::from_u64(0), "no-ops must not increment");
                        None
                    }
                };

                for word in 0..words {
                    assert_eq!(
                        f.ra[word * cycles + cycle],
                        Fr::from_u64(u64::from(touched == Some(word))),
                        "seed {seed} cycle {cycle} word {word}: ra grid is wrong",
                    );
                }
            }

            for word in 0..words {
                assert_eq!(
                    f.val[word * cycles],
                    f.val_init[word],
                    "seed {seed} word {word}: val grid does not start at val_init",
                );
            }
            assert!(
                accesses > 0 && writes > 0,
                "seed {seed} exercised no accesses or no writes",
            );
        }
    }

    proptest! {
        #[test]
        fn ram_read_write_matches_reference(
            seed in any::<u64>(),
            ram_read_value in arb_point(LOG_T),
            ram_write_value in arb_point(LOG_T),
            tau_low in arb_point(LOG_T),
            gamma in any::<u64>().prop_map(fr),
            challenges in arb_point(RAM_LOG_K + LOG_T),
        ) {
            let Some(_) = shared_context() else { return Ok(()); };
            let witness = witness(seed);
            let relation = RamReadWriteChecking::<Fr>::new(
                ReadWriteDimensions::new(LOG_T, RAM_LOG_K, LOG_T, RAM_LOG_K),
                RAM_LOG_K,
                tau_low,
            );
            let claims = RamReadWriteInputClaims {
                ram_read_value: Fr::from_u64(0),
                ram_write_value: Fr::from_u64(0),
            };
            let points = RamReadWriteInputClaims {
                ram_read_value,
                ram_write_value,
            };
            let challenge_set = RamReadWriteChallenges { gamma };
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
