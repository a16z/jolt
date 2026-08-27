//! Metal Hamming-weight claim reduction (stage 7): device twin of
//! [`OptimizedHammingWeightClaimReduction`], byte-identical round
//! polynomials by construction.
//!
//! Construction (shared-eq pushforwards `G_i` + fused weights `W_i`) is the
//! optimized kernel's ([`build_hamming_weight_tables`] — the `O(T)` bundle
//! walk stays on the CPU; scatter accumulation is a poor device fit at this
//! pool size). The `N` table pairs are CONCATENATED into two flat device
//! tables (one prepare-time memcpy — Metal's 31-buffer argument table cannot
//! hold `2N` separate bindings, and the flat layout keeps every round a
//! single `jk_table_pairs_round` dispatch over `N · groups` threads). Tables
//! are contiguous at the current per-table stride, so the flat pairwise fold
//! IS the per-table fold with compact repacking — the CPU tail reuses the
//! plain flat `RoundTable` helpers unchanged.

use jolt_claims::protocols::jolt::JoltOpeningId;
use jolt_claims::OutputClaims;
use jolt_field::{Fr, Ring};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::SumcheckOutputClaims;
use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::HammingWeightClaimReduction;
use jolt_witness::JoltWitnessPlane;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{num_threadgroups, DeviceRound, Partials, RoundTable};
use crate::metal::field::fr_to_u32_limbs;
use crate::metal::runtime::{KernelId, MetalContext};
use crate::metal::{metal_gate, testing, MetalError};
use crate::optimized::hamming_weight_claim_reduction::{
    build_hamming_weight_tables, HammingWeightTables, OptimizedHammingWeightClaimReduction,
};
#[cfg(feature = "parallel")]
use crate::optimized::support::merge_evals;
use crate::optimized::support::round_poly_from_skipped_evals;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

const KIND: &str = "hamming_weight_claim_reduction";

/// Slot front: device kernel above the [`metal_gate`] threshold, the
/// optimized fallback below it or on any device failure.
pub struct MetalHammingWeightClaimReduction {
    pub fallback: OptimizedHammingWeightClaimReduction,
}

impl PrepareKernel<Fr, HammingWeightClaimReduction<Fr>> for MetalHammingWeightClaimReduction {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<Fr>,
        inputs: ProverInputs<'_, Fr, HammingWeightClaimReduction<Fr>>,
    ) -> Result<
        Box<dyn SumcheckKernel<Fr, Relation = HammingWeightClaimReduction<Fr>>>,
        KernelError<Fr>,
    > {
        let dimensions = inputs.relation.dimensions();
        let work_items = dimensions.layout.total() << dimensions.log_k_chunk;
        if metal_gate(KIND, work_items) {
            match MetalContext::global() {
                Ok(context) => {
                    // Structural errors propagate — the fallback would fail
                    // identically; only device failures fall back.
                    let tables = build_hamming_weight_tables(session, witness, &inputs)?;
                    match MetalHammingWeightKernel::new(context, tables) {
                        Ok(kernel) => return Ok(Box::new(kernel)),
                        Err(error) => tracing::warn!(
                            slot = KIND,
                            %error,
                            "device prepare failed; using the optimized fallback"
                        ),
                    }
                }
                Err(error) => tracing::warn!(
                    slot = KIND,
                    %error,
                    "no device context; using the optimized fallback"
                ),
            }
        }
        self.fallback.prepare(session, witness, inputs)
    }
}

struct MetalHammingWeightKernel {
    rounds: usize,
    rounds_bound: usize,
    num_tables: usize,
    /// Current per-table length (the concatenation stride).
    len: usize,
    /// Pushforwards `G_i`, concatenated in canonical layout order.
    g: RoundTable,
    /// Combined weights `W_i`, same layout.
    w: RoundTable,
    output_openings: Vec<JoltOpeningId>,
    partials: Partials,
    device: DeviceRound,
}

impl MetalHammingWeightKernel {
    fn new(
        context: &'static MetalContext,
        tables: HammingWeightTables<Fr>,
    ) -> Result<Self, MetalError> {
        let num_tables = tables.g_tables.len();
        let len = tables.g_tables.first().map_or(0, Vec::len);
        if num_tables == 0 || len < 2 {
            return Err(MetalError::UnsupportedShape(
                "hamming weight reduction needs at least one table pair of length >= 2",
            ));
        }
        let concat = |tables: Vec<Vec<Fr>>| -> Vec<Fr> {
            let mut flat = Vec::with_capacity(num_tables * len);
            for table in tables {
                flat.extend_from_slice(&table);
            }
            flat
        };
        Ok(Self {
            rounds: tables.rounds,
            rounds_bound: 0,
            num_tables,
            len,
            g: RoundTable::new(context, concat(tables.g_tables))?,
            w: RoundTable::new(context, concat(tables.weight_tables))?,
            output_openings: tables.output_openings,
            partials: Partials::new(context, 2, num_tables * len / 2)?,
            device: DeviceRound::new(context, KIND),
        })
    }

    fn bind_bookkeeping(&mut self) {
        self.len /= 2;
        self.rounds_bound += 1;
    }

    /// The fused device round over all `N` table pairs: one dispatch, one
    /// command buffer, one wait.
    fn device_round(
        &self,
        context: &MetalContext,
        bind: Option<Fr>,
        groups_per_table: usize,
    ) -> Result<Vec<Fr>, MetalError> {
        let threads = self.num_tables * groups_per_table;
        let num_tgs = num_threadgroups(threads);
        // TablePairsParams: [log_h, num_tables, len, do_bind, num_tgs, r].
        let mut params = vec![
            groups_per_table.ilog2(),
            self.num_tables as u32,
            self.len as u32,
            u32::from(bind.is_some()),
            num_tgs as u32,
        ];
        params.extend_from_slice(&fr_to_u32_limbs(bind.unwrap_or_else(|| Fr::from_u64(0))));
        let buffers = [
            self.g.cur().device_buffer(),
            self.w.cur().device_buffer(),
            self.g.nxt().device_buffer(),
            self.w.nxt().device_buffer(),
            self.partials.buffer().device_buffer(),
        ];
        let mut pass = context.begin_pass()?;
        pass.dispatch(
            KernelId::TablePairsRound,
            &params,
            &[
                &buffers[0],
                &buffers[1],
                &buffers[2],
                &buffers[3],
                &buffers[4],
            ],
            threads,
        );
        pass.run()?;
        testing::note_device_round();
        Ok(self.partials.sums(num_tgs))
    }

    /// The CPU twin: tables are contiguous and even-length, so the flat
    /// pairwise walk covers exactly the per-table pairs.
    fn cpu_round(&mut self, bind: Option<Fr>) -> Vec<Fr> {
        if let Some(challenge) = bind {
            let total = self.num_tables * self.len;
            self.g.bind_cpu(total, challenge);
            self.w.bind_cpu(total, challenge);
            self.bind_bookkeeping();
        }
        let total = self.num_tables * self.len;
        let g = self.g.cur_slice(total);
        let w = self.w.cur_slice(total);
        let group = |j: usize| -> [Fr; 2] {
            let at_two = |lo: Fr, hi: Fr| hi + hi - lo;
            [
                g[2 * j] * w[2 * j],
                at_two(g[2 * j], g[2 * j + 1]) * at_two(w[2 * j], w[2 * j + 1]),
            ]
        };
        let half = total / 2;
        #[cfg(feature = "parallel")]
        {
            (0..half)
                .into_par_iter()
                .fold(
                    || vec![Fr::from_u64(0); 2],
                    |mut acc, j| {
                        let evals = group(j);
                        acc[0] += evals[0];
                        acc[1] += evals[1];
                        acc
                    },
                )
                .reduce(|| vec![Fr::from_u64(0); 2], merge_evals)
        }
        #[cfg(not(feature = "parallel"))]
        (0..half).fold(vec![Fr::from_u64(0); 2], |mut acc, j| {
            let evals = group(j);
            acc[0] += evals[0];
            acc[1] += evals[1];
            acc
        })
    }
}

impl ProveRounds<Fr> for MetalHammingWeightKernel {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        _round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        // Post-bind per-table pair count.
        let groups_per_table = if bind.is_some() {
            self.len / 4
        } else {
            self.len / 2
        };
        let device = if groups_per_table == 0 {
            None
        } else {
            self.device.gated(self.num_tables * self.len)
        };
        let evals = match device {
            Some(context) => match self.device_round(context, bind, groups_per_table) {
                Ok(sums) => {
                    if bind.is_some() {
                        self.g.swap();
                        self.w.swap();
                        self.bind_bookkeeping();
                    }
                    sums
                }
                Err(error) => {
                    self.device.failed(&error);
                    self.cpu_round(bind)
                }
            },
            None => self.cpu_round(bind),
        };
        Ok(round_poly_from_skipped_evals(&evals, previous_claim))
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        let total = self.num_tables * self.len;
        self.g.bind_cpu(total, bind);
        self.w.bind_cpu(total, bind);
        self.bind_bookkeeping();
        Ok(())
    }
}

impl SumcheckKernel<Fr> for MetalHammingWeightKernel {
    type Relation = HammingWeightClaimReduction<Fr>;

    fn output_claims(
        &mut self,
        _inputs: &jolt_verifier::stages::relations::SumcheckInputClaims<Fr, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<Fr, Self::Relation>, SumcheckKernelError<Fr>> {
        if self.rounds_bound != self.rounds {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.rounds - self.rounds_bound,
            });
        }
        // Fully bound: each table is one element, so the flat layout is the
        // per-table final values in canonical order.
        let finals = self.g.cur_slice(self.num_tables);
        let output_openings = &self.output_openings;
        SumcheckOutputClaims::<Fr, Self::Relation>::from_opening_values(|id: &JoltOpeningId| {
            output_openings
                .iter()
                .position(|opening| opening == id)
                .map(|index| finals[index])
        })
        .map_err(SumcheckKernelError::from)
    }
}

/// Lockstep parity against the optimized kernel with the device path forced
/// and probed; the sample backend serves all three RA families, so the
/// multi-table concatenated layout is genuinely multi-family here.
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::claim_reductions::hamming_weight::HammingWeightClaimReductionDimensions;
    use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
    use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltPolynomialId};
    use jolt_field::Ring;
    use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::{
        HammingWeightClaimReductionChallenges, HammingWeightClaimReductionInputClaims,
    };
    use jolt_witness::testing::with_sample_backend;
    use jolt_witness::JoltWitnessOracle;

    use super::*;
    use crate::metal::testing::{device_probe_count, gpu_lock};
    use crate::optimized::parity::{
        probe_input_claim, probe_one_hot_family, run_lockstep, synthetic_point,
    };
    use crate::ReferenceBackend;

    #[test]
    fn hamming_weight_reduction_matches_optimized() {
        let _lock = gpu_lock();
        // nextest runs one process per test, so env mutation is safe.
        std::env::remove_var("JOLT_METAL_DISABLE");
        std::env::set_var("JOLT_METAL_MIN_TERMS", "0");
        with_sample_backend(|backend| {
            let log_t = JoltWitnessOracle::<Fr>::shape(
                backend,
                JoltPolynomialId::Committed(JoltCommittedPolynomial::RdInc),
            )
            .unwrap()
            .rows()
            .ilog2() as usize;
            let (instruction_d, log_k_chunk) =
                probe_one_hot_family(backend, JoltCommittedPolynomial::InstructionRa, log_t);
            let (bytecode_d, _) =
                probe_one_hot_family(backend, JoltCommittedPolynomial::BytecodeRa, log_t);
            let (ram_d, _) = probe_one_hot_family(backend, JoltCommittedPolynomial::RamRa, log_t);
            let layout = JoltRaPolynomialLayout::new(instruction_d, bytecode_d, ram_d).unwrap();
            let dimensions = HammingWeightClaimReductionDimensions::new(layout, log_k_chunk);

            let relation = HammingWeightClaimReduction::new(
                dimensions,
                synthetic_point(log_t, 3),
                synthetic_point(log_k_chunk, 5),
                (0..layout.total())
                    .map(|index| synthetic_point(log_k_chunk, 7 + index as u64))
                    .collect(),
            );
            let challenges = HammingWeightClaimReductionChallenges {
                gamma: Fr::from_u64(23),
            };
            let claims = HammingWeightClaimReductionInputClaims::<Fr>::default();
            let input_points = HammingWeightClaimReductionInputClaims::<Vec<Fr>>::default();

            let mut session = ProofSession::default();
            // The reference kernel exists only to recover the honest input
            // claim through its round check.
            let mut reference =
                <ReferenceBackend as PrepareKernel<Fr, HammingWeightClaimReduction<Fr>>>::prepare(
                    &ReferenceBackend,
                    &mut session,
                    backend,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &input_points,
                        challenges: &challenges,
                    },
                )
                .unwrap();
            let mut optimized = OptimizedHammingWeightClaimReduction
                .prepare(
                    &mut session,
                    backend,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &input_points,
                        challenges: &challenges,
                    },
                )
                .unwrap();
            let metal_slot = MetalHammingWeightClaimReduction {
                fallback: OptimizedHammingWeightClaimReduction,
            };
            let mut metal = metal_slot
                .prepare(
                    &mut session,
                    backend,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &input_points,
                        challenges: &challenges,
                    },
                )
                .unwrap();

            let rounds_before = device_probe_count();
            let claim = probe_input_claim(reference.as_mut());
            let sumcheck_challenges = synthetic_point(log_k_chunk, 301);
            run_lockstep(
                optimized.as_mut(),
                metal.as_mut(),
                claim,
                &sumcheck_challenges,
            );
            assert_eq!(
                optimized.output_claims(&claims).unwrap(),
                metal.output_claims(&claims).unwrap()
            );
            assert!(
                device_probe_count() > rounds_before,
                "the metal kernel never dispatched on the device"
            );
        });
    }
}
