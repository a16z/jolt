//! Metal RAM read/write-checking cycle rounds with an early host handoff.

use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial};
use jolt_field::{Fr, Ring};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputClaims,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage2::ram_read_write_checking::RamReadWriteChecking;
use jolt_witness::JoltWitnessPlane;

use super::{num_threadgroups, Partials, RoundTable};
use crate::metal::buffers::{OwnedDeviceBuffer, PageAlignedVec};
use crate::metal::field::{fr_as_u32s, fr_to_u32_limbs};
use crate::metal::runtime::{KernelId, MetalContext};
use crate::metal::{metal_gate, testing, MetalError};
use crate::optimized::ram_read_write::RamReadWriteKernel;
use crate::optimized::ram_trace::{RamAccessColumns, RamAccessValues, NO_ACCESS};
use crate::optimized::rw_matrix::{CycleMajorEntry, CycleMajorMatrix};
use crate::optimized::OptimizedBackend;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

const KIND: &str = "ram_read_write";
const HANDOFF_ROWS: usize = 4096;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RawRamRwEntry {
    val: Fr,
    ra: Fr,
    prev_val: u64,
    next_val: u64,
    col: u32,
    pad: u32,
}

const _: () = assert!(std::mem::size_of::<RawRamRwEntry>() == 88);

/// The cycle-major CSR: one entry per RAM access, `offsets[t]..offsets[t+1]`
/// spanning cycle `t`'s row.
fn ram_rw_entries(
    columns: &RamAccessColumns,
    values: &RamAccessValues,
) -> (Vec<RawRamRwEntry>, Vec<u32>) {
    let mut entries = Vec::new();
    let mut offsets = Vec::with_capacity(columns.addresses.len() + 1);
    offsets.push(0);
    for (cycle, &address) in columns.addresses.iter().enumerate() {
        if address != NO_ACCESS {
            let pre_value = values.pre_values[cycle];
            entries.push(RawRamRwEntry {
                val: Fr::from_u64(pre_value),
                ra: Fr::from_u64(1),
                prev_val: pre_value,
                next_val: values.post_values[cycle],
                col: address,
                pad: 0,
            });
        }
        offsets.push(entries.len() as u32);
    }
    (entries, offsets)
}

struct PingPong<T: Copy> {
    cur: OwnedDeviceBuffer<T>,
    nxt: OwnedDeviceBuffer<T>,
}

impl<T: Copy> PingPong<T> {
    fn swap(&mut self) {
        std::mem::swap(&mut self.cur, &mut self.nxt);
    }
}

struct DeviceRamRwState {
    context: &'static MetalContext,
    entries: PingPong<RawRamRwEntry>,
    entry_count: usize,
    offsets: PingPong<u32>,
    counts: OwnedDeviceBuffer<u32>,
    inc: RoundTable,
    partials: Partials,
    rows: usize,
    counts_valid: bool,
}

impl DeviceRamRwState {
    fn new(
        context: &'static MetalContext,
        entries: Vec<RawRamRwEntry>,
        offsets: Vec<u32>,
        inc: Vec<Fr>,
    ) -> Result<Self, MetalError> {
        let entry_count = entries.len();
        let entries = context.own_vec(entries)?;
        let offsets = context.own_vec(offsets)?;
        let rows = inc.len();
        Ok(Self {
            context,
            entries: PingPong {
                cur: entries,
                nxt: context.own_page_aligned(PageAlignedVec::from_elem(
                    RawRamRwEntry::default(),
                    entry_count,
                ))?,
            },
            entry_count,
            offsets: PingPong {
                cur: offsets,
                nxt: context.own_page_aligned(PageAlignedVec::from_elem(0_u32, rows + 1))?,
            },
            counts: context
                .own_page_aligned(PageAlignedVec::from_elem(0_u32, (rows / 2).max(1)))?,
            inc: RoundTable::new(context, inc)?,
            partials: Partials::new(context, 2, (rows / 2).max(1))?,
            rows,
            counts_valid: false,
        })
    }

    fn scan_bind_offsets(&mut self) -> Result<usize, MetalError> {
        if !self.counts_valid {
            return Err(MetalError::Execution(
                "RAM read/write bind without a preceding message".to_string(),
            ));
        }
        let pairs = self.rows / 2;
        let mut total = 0_u32;
        let counts = &self.counts.as_slice()[..pairs];
        let offsets = &mut self.offsets.nxt.as_mut_slice()[..=pairs];
        offsets[0] = 0;
        for (index, &count) in counts.iter().enumerate() {
            total = total.checked_add(count).ok_or_else(|| {
                MetalError::Execution("RAM read/write entry count overflow".to_string())
            })?;
            offsets[index + 1] = total;
        }
        let new_count = total as usize;
        if new_count > self.entries.nxt.len() {
            return Err(MetalError::Execution(
                "RAM read/write bind exceeded sparse entry capacity".to_string(),
            ));
        }
        Ok(new_count)
    }

    fn message(
        &mut self,
        gruen: &GruenSplitEqPolynomial<Fr>,
        gamma: Fr,
    ) -> Result<[Fr; 2], MetalError> {
        let pairs = self.rows / 2;
        let num_tgs = num_threadgroups(pairs);
        let e_in = gruen.e_in_current();
        let e_out = gruen.e_out_current();
        let e_in_buffer = self.context.wrap_slice(fr_as_u32s(e_in))?;
        let e_out_buffer = self.context.wrap_slice(fr_as_u32s(e_out))?;
        let entry_buffer = self.entries.cur.device_buffer();
        let offset_buffer = self.offsets.cur.device_buffer();
        let inc_buffer = self.inc.cur().device_buffer();
        let partial_buffer = self.partials.buffer().device_buffer();
        let count_buffer = self.counts.device_buffer();
        let mut params = vec![
            pairs as u32,
            num_tgs as u32,
            e_in.len().trailing_zeros(),
            e_in.len() as u32,
        ];
        params.extend_from_slice(&fr_to_u32_limbs(gamma));
        let mut pass = self.context.begin_pass()?;
        pass.dispatch(
            KernelId::RamRwMessage,
            &params,
            &[
                &entry_buffer,
                &offset_buffer,
                &inc_buffer,
                &e_out_buffer,
                &e_in_buffer,
                &partial_buffer,
                &count_buffer,
            ],
            pairs,
        );
        pass.run()?;
        testing::note_device_round();
        self.counts_valid = true;
        let sums = self.partials.sums(num_tgs);
        Ok([sums[0], sums[1]])
    }

    fn bind(&mut self, challenge: Fr) -> Result<(), MetalError> {
        let pairs = self.rows / 2;
        let new_count = self.scan_bind_offsets()?;
        {
            let entry_buffer = self.entries.cur.device_buffer();
            let offset_buffer = self.offsets.cur.device_buffer();
            let out_offset_buffer = self.offsets.nxt.device_buffer();
            let out_entry_buffer = self.entries.nxt.device_buffer();
            let inc_buffer = self.inc.cur().device_buffer();
            let out_inc_buffer = self.inc.nxt().device_buffer();
            let mut params = vec![pairs as u32];
            params.extend_from_slice(&fr_to_u32_limbs(challenge));
            let mut pass = self.context.begin_pass()?;
            pass.dispatch(
                KernelId::RamRwBind,
                &params,
                &[
                    &entry_buffer,
                    &offset_buffer,
                    &out_offset_buffer,
                    &out_entry_buffer,
                    &inc_buffer,
                    &out_inc_buffer,
                ],
                pairs,
            );
            pass.run()?;
        }
        testing::note_device_round();
        self.entries.swap();
        self.offsets.swap();
        self.inc.swap();
        self.entry_count = new_count;
        self.rows = pairs;
        self.counts_valid = false;
        Ok(())
    }

    fn into_cycle_state(self) -> Result<(CycleMajorMatrix<Fr>, Polynomial<Fr>), ()> {
        let offsets = &self.offsets.cur.as_slice()[..=self.rows];
        if offsets[self.rows] as usize != self.entry_count {
            return Err(());
        }
        let raw_entries = &self.entries.cur.as_slice()[..self.entry_count];
        let mut entries = Vec::with_capacity(self.entry_count);
        for row in 0..self.rows {
            let start = offsets[row] as usize;
            let end = offsets[row + 1] as usize;
            let Some(row_entries) = raw_entries.get(start..end) else {
                return Err(());
            };
            entries.extend(row_entries.iter().map(|raw| CycleMajorEntry {
                row,
                col: raw.col as usize,
                prev_val: raw.prev_val,
                next_val: raw.next_val,
                val: raw.val,
                ra: raw.ra,
            }));
        }
        let inc = Polynomial::new(self.inc.cur_slice(self.rows).to_vec());
        Ok((CycleMajorMatrix { entries }, inc))
    }
}

pub struct MetalRamReadWriteChecking {
    fallback: OptimizedBackend,
    handoff_rows: usize,
}

impl MetalRamReadWriteChecking {
    pub fn new() -> Self {
        Self {
            fallback: OptimizedBackend,
            handoff_rows: HANDOFF_ROWS,
        }
    }

    #[cfg(test)]
    fn with_handoff_rows(handoff_rows: usize) -> Self {
        Self {
            fallback: OptimizedBackend,
            handoff_rows,
        }
    }
}

impl Default for MetalRamReadWriteChecking {
    fn default() -> Self {
        Self::new()
    }
}

impl PrepareKernel<Fr, RamReadWriteChecking<Fr>> for MetalRamReadWriteChecking {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<Fr>,
        inputs: ProverInputs<'_, Fr, RamReadWriteChecking<Fr>>,
    ) -> Result<Box<dyn SumcheckKernel<Fr, Relation = RamReadWriteChecking<Fr>>>, KernelError<Fr>>
    {
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let log_t = dimensions.log_t();
        let log_k = relation.ram_log_k();
        let tau_low = relation.product_tau_low();
        if dimensions.phase1_num_rounds() != log_t {
            return Err(KernelError::Unsupported {
                reason: "Metal RAM read-write checking supports only the default read-write config",
            });
        }
        if log_t == 0 || dimensions.log_k() != log_k || tau_low.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "RAM read-write checking geometry is inconsistent",
            });
        }
        let cycles = 1usize << log_t;
        if !metal_gate(KIND, cycles) || log_t >= 32 {
            return self.fallback.prepare(session, witness, inputs);
        }

        let columns = RamAccessColumns::shared(session, witness, log_t)?;
        columns.validate_addresses(1usize << log_k)?;
        let values =
            std::sync::Arc::clone(session.state::<std::sync::Arc<RamAccessValues>>().ok_or(
                KernelError::InvariantViolation {
                    reason: "RAM access value columns were already consumed",
                },
            )?);
        let (entries, offsets) = ram_rw_entries(&columns, &values);
        if entries.is_empty() {
            return self.fallback.prepare(session, witness, inputs);
        }
        let inc = values.inc_column::<Fr>();
        let val_final = witness.oracle_table(JoltPolynomialId::Virtual(
            JoltVirtualPolynomial::RamValFinal,
        ))?;
        if inc.len() != cycles || val_final.len() != 1usize << log_k {
            return Err(KernelError::InvariantViolation {
                reason: "RAM read-write witness tables disagree with the relation geometry",
            });
        }
        let val_init = Polynomial::new(columns.reconstruct_val_init(&values.pre_values, val_final));
        let context = match MetalContext::global() {
            Ok(context) => context,
            Err(error) => {
                tracing::warn!(slot = KIND, %error, "no device context; using optimized fallback");
                return self.fallback.prepare(session, witness, inputs);
            }
        };
        let device = match DeviceRamRwState::new(context, entries, offsets, inc) {
            Ok(device) => device,
            Err(error) => {
                tracing::warn!(slot = KIND, %error, "device preparation failed; using optimized fallback");
                return self.fallback.prepare(session, witness, inputs);
            }
        };
        // Device state built: this kernel is the value columns' stage-4
        // consumer; drop the session's strong ref (mirrors the optimized
        // tier's `shared_with_values` take).
        drop(values);
        let _ = session.take::<std::sync::Arc<RamAccessValues>>();
        Ok(Box::new(MetalRamRwKernel {
            device: Some(device),
            gruen: Some(GruenSplitEqPolynomial::new(
                tau_low,
                BindingOrder::LowToHigh,
            )),
            host: None,
            val_init: Some(val_init),
            gamma: inputs.challenges.gamma,
            log_t,
            log_k,
            handoff_rows: self.handoff_rows.max(1),
            rounds_bound: 0,
        }))
    }
}

fn missing_device_state() -> SumcheckError<Fr> {
    SumcheckError::MissingEvaluationSource {
        kind: "Metal RAM read/write device state",
    }
}

struct MetalRamRwKernel {
    device: Option<DeviceRamRwState>,
    gruen: Option<GruenSplitEqPolynomial<Fr>>,
    host: Option<RamReadWriteKernel<Fr>>,
    val_init: Option<Polynomial<Fr>>,
    gamma: Fr,
    log_t: usize,
    log_k: usize,
    handoff_rows: usize,
    rounds_bound: usize,
}

impl MetalRamRwKernel {
    fn transition(&mut self) -> Result<(), SumcheckError<Fr>> {
        let state = self.device.take().ok_or_else(missing_device_state)?;
        let (matrix, inc) = state
            .into_cycle_state()
            .map_err(|()| missing_device_state())?;
        let gruen = self.gruen.take().ok_or_else(missing_device_state)?;
        let val_init = self.val_init.take().ok_or_else(missing_device_state)?;
        self.host = Some(RamReadWriteKernel::from_cycle_state(
            matrix,
            gruen,
            inc,
            val_init,
            self.gamma,
            self.log_t,
            self.log_k,
            self.rounds_bound,
        )?);
        Ok(())
    }

    fn maybe_transition(&mut self) -> Result<(), SumcheckError<Fr>> {
        if self
            .device
            .as_ref()
            .is_some_and(|state| state.rows <= self.handoff_rows)
        {
            self.transition()?;
        }
        Ok(())
    }

    fn host_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        self.host
            .as_mut()
            .ok_or_else(missing_device_state)?
            .prove_round(bind, round, previous_claim)
    }
}

impl ProveRounds<Fr> for MetalRamRwKernel {
    fn num_rounds(&self) -> usize {
        self.log_t + self.log_k
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        if self.host.is_some() {
            return self.host_round(bind, round, previous_claim);
        }
        if let Some(challenge) = bind {
            let bind_result = self
                .device
                .as_mut()
                .ok_or_else(missing_device_state)?
                .bind(challenge);
            if let Err(error) = bind_result {
                tracing::warn!(slot = KIND, %error, "device bind failed; finishing on CPU");
                self.transition()?;
                return self.host_round(Some(challenge), round, previous_claim);
            }
            self.gruen
                .as_mut()
                .ok_or_else(missing_device_state)?
                .bind(challenge);
            self.rounds_bound += 1;
        }
        self.maybe_transition()?;
        if self.host.is_some() {
            return self.host_round(None, round, previous_claim);
        }
        let result = self
            .device
            .as_mut()
            .ok_or_else(missing_device_state)?
            .message(
                self.gruen.as_ref().ok_or_else(missing_device_state)?,
                self.gamma,
            );
        match result {
            Ok([q_0, q_inf]) => Ok(self
                .gruen
                .as_ref()
                .ok_or_else(missing_device_state)?
                .gruen_poly_deg_3(q_0, q_inf, previous_claim)),
            Err(error) => {
                tracing::warn!(slot = KIND, %error, "device message failed; finishing on CPU");
                self.transition()?;
                self.host_round(None, round, previous_claim)
            }
        }
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        if let Some(host) = &mut self.host {
            return host.finish_rounds(bind);
        }
        let bind_result = self
            .device
            .as_mut()
            .ok_or_else(missing_device_state)?
            .bind(bind);
        if let Err(error) = bind_result {
            tracing::warn!(slot = KIND, %error, "final device bind failed; finishing on CPU");
            self.transition()?;
            return self
                .host
                .as_mut()
                .ok_or_else(missing_device_state)?
                .finish_rounds(bind);
        }
        self.gruen
            .as_mut()
            .ok_or_else(missing_device_state)?
            .bind(bind);
        self.rounds_bound += 1;
        self.transition()
    }
}

impl SumcheckKernel<Fr> for MetalRamRwKernel {
    type Relation = RamReadWriteChecking<Fr>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<Fr, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<Fr, Self::Relation>, SumcheckKernelError<Fr>> {
        let remaining = self.num_rounds().saturating_sub(self.rounds_bound);
        self.host
            .as_mut()
            .ok_or(SumcheckKernelError::NotFullyBound { remaining })?
            .output_claims(inputs)
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<Fr, Self::Relation>,
        output_points: &SumcheckOutputPoints<Fr, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<Fr, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<Fr>> {
        self.host
            .as_ref()
            .ok_or(SumcheckKernelError::NotFullyBound {
                remaining: self.num_rounds().saturating_sub(self.rounds_bound),
            })?
            .validate_derived_tables(relation, input_points, output_points, challenges)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::ReadWriteDimensions;
    use jolt_claims::protocols::jolt::geometry::ram::{ram_inc, ram_ra, ram_val};
    use jolt_poly::EqPolynomial;
    use jolt_verifier::stages::stage2::ram_read_write_checking::{
        RamReadWriteChallenges, RamReadWriteInputClaims,
    };

    use super::*;
    use crate::metal::testing::{device_probe_count, gpu_lock};
    use crate::optimized::testing::{
        assert_parity, random_scalars, with_ram_fixture, FixtureShape, RamOp,
    };

    fn dense_input_claim(
        witness: &dyn JoltWitnessPlane<Fr>,
        tau_low: &[Fr],
        gamma: Fr,
        ram_k: usize,
    ) -> Fr {
        let cycles = 1usize << tau_low.len();
        let eq = EqPolynomial::new(tau_low.to_vec()).evaluations();
        let ra: Vec<Fr> = witness.oracle_table(ram_ra().polynomial_id()).unwrap();
        let val: Vec<Fr> = witness.oracle_table(ram_val().polynomial_id()).unwrap();
        let inc: Vec<Fr> = witness.oracle_table(ram_inc().polynomial_id()).unwrap();
        let mut claim = Fr::from_u64(0);
        for k in 0..ram_k {
            for j in 0..cycles {
                let index = (k << tau_low.len()) | j;
                claim += eq[j] * ra[index] * (val[index] + gamma * (val[index] + inc[j]));
            }
        }
        claim
    }

    fn run_parity(shape: FixtureShape, ops: Vec<RamOp>, handoff_rows: usize) {
        with_ram_fixture(shape, ops, |witness| {
            let tau_low = random_scalars(shape.log_t, 17);
            let gamma = random_scalars(1, 23)[0];
            let relation = RamReadWriteChecking::<Fr>::new(
                ReadWriteDimensions::new(shape.log_t, shape.log_k(), shape.log_t, shape.log_k()),
                shape.log_k(),
                tau_low.clone(),
            );
            let claims = RamReadWriteInputClaims {
                ram_read_value: Fr::from_u64(0),
                ram_write_value: Fr::from_u64(0),
            };
            let points = RamReadWriteInputClaims::<Vec<Fr>>::default();
            let challenges = RamReadWriteChallenges { gamma };
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };
            let optimized = PrepareKernel::<Fr, _>::prepare(
                &OptimizedBackend,
                &mut ProofSession::default(),
                witness,
                inputs(),
            )
            .unwrap();
            let before = device_probe_count();
            let metal = PrepareKernel::<Fr, _>::prepare(
                &MetalRamReadWriteChecking::with_handoff_rows(handoff_rows),
                &mut ProofSession::default(),
                witness,
                inputs(),
            )
            .unwrap();
            assert_parity(
                optimized,
                metal,
                dense_input_claim(witness, &tau_low, gamma, shape.ram_k),
                &inputs(),
                71,
            );
            assert!(device_probe_count() > before, "device path did not engage");
        });
    }

    fn force_device() {
        std::env::remove_var("JOLT_METAL_DISABLE");
        std::env::set_var("JOLT_METAL_MIN_TERMS_RAM_READ_WRITE", "0");
    }

    #[test]
    fn ram_rw_matches_optimized_mid_cycle_handoff() {
        let _lock = gpu_lock();
        force_device();
        run_parity(
            FixtureShape {
                log_t: 5,
                ram_k: 16,
            },
            vec![
                RamOp::Write { word: 3, post: 5 },
                RamOp::Read { word: 3 },
                RamOp::Write { word: 3, post: 9 },
                RamOp::Read { word: 7 },
                RamOp::None,
                RamOp::Write { word: 4, post: 2 },
                RamOp::Read { word: 3 },
                RamOp::Write { word: 7, post: 6 },
            ],
            4,
        );
    }

    #[test]
    fn ram_rw_matches_optimized_device_cycle_boundary() {
        let _lock = gpu_lock();
        force_device();
        run_parity(
            FixtureShape { log_t: 4, ram_k: 8 },
            vec![
                RamOp::None,
                RamOp::Write { word: 5, post: 11 },
                RamOp::None,
                RamOp::Read { word: 5 },
                RamOp::None,
                RamOp::Write { word: 5, post: 3 },
            ],
            1,
        );
    }
}
