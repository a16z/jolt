pub(super) mod address_driver;
pub(super) mod address_phase;
mod combine;
mod cycle_handoff;
mod cycle_rounds;
mod prefix_suffix;
mod prefixes;
mod suffixes;

use std::sync::Arc;

use jolt_claims::protocols::jolt::geometry::instruction::CANONICAL_INSTRUCTION_ADDRESS;
use jolt_claims::protocols::jolt::relations::instruction::{
    InstructionReadRafInputClaims, InstructionReadRafOutputClaims,
};
use jolt_field::{Field, Fr};
use jolt_lookup_tables::lookup_bits::LookupBits;
use jolt_lookup_tables::tables::prefixes::{PrefixEval, ALL_PREFIXES};
use jolt_lookup_tables::tables::suffixes::SuffixEval;
use jolt_lookup_tables::tables::LookupTableKind;
use jolt_lookup_tables::XLEN as RISCV_XLEN;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::stage5::instruction_read_raf::InstructionReadRaf;
use jolt_witness::{collect_bundles, JoltWitnessPlane};

use self::address_driver::DeviceAddressPhase;
use self::address_phase::{flag_claims, DeviceRows, NO_TABLE};
use self::cycle_handoff::{build_cycle_tables, HandoffInputs};
use self::cycle_rounds::DeviceCycleRounds;
use super::{require_context, CudaBackend};
use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::device::{fr_into, require_fr, require_fr_slice};
use crate::reference::instruction_read_raf::InstructionReadRafWitness;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

const CHUNK_LEN: usize = 8;
const ADDRESS_BITS: usize = 128;
const RAF_CHECKPOINTS: usize = 4;
const HINT_POINTS: usize = 2;

fn raf_initial_checkpoints<F: Field>() -> [F; RAF_CHECKPOINTS] {
    let mut checkpoints = [F::zero(); RAF_CHECKPOINTS];
    if CANONICAL_INSTRUCTION_ADDRESS {
        checkpoints[3] = F::one();
    }
    checkpoints
}

pub struct DeviceInstructionReadRaf<F: Field> {
    device: Option<DeviceAddressPhase>,
    cycle: Option<DeviceCycleRounds>,
    rows: Arc<DeviceRows>,
    r_reduction: Vec<Fr>,
    cycle_challenges: Vec<Fr>,
    prefix_checkpoints: Vec<PrefixEval<F>>,
    raf_checkpoints: [F; RAF_CHECKPOINTS],
    ra_count: usize,
    rounds: usize,
    gamma: F,
    context: &'static CudaKernelContext,
    rounds_bound: usize,
}

impl<F: Field> DeviceInstructionReadRaf<F> {
    fn field(value: jolt_field::Fr) -> Result<F, SumcheckError<F>> {
        fr_into(value).ok_or(SumcheckError::MissingEvaluationSource {
            kind: "cuda instruction read-RAF field",
        })
    }

    fn enter_cycle_rounds(&mut self) -> Result<(), SumcheckError<F>> {
        let device = self
            .device
            .take()
            .ok_or(SumcheckError::MissingEvaluationSource {
                kind: "cuda address phase",
            })?;
        let failed = || SumcheckError::MissingEvaluationSource {
            kind: "cuda address-phase handoff",
        };

        let prefix_checkpoints = device.checkpoints(self.context).map_err(|_| failed())?;
        let prefix_checkpoints: Vec<F> = prefix_checkpoints
            .into_iter()
            .map(Self::field)
            .collect::<Result<_, _>>()?;

        let raf = device.raf_checkpoints(self.context).map_err(|_| failed())?;
        if raf.len() != RAF_CHECKPOINTS {
            return Err(failed());
        }
        let mut raf_checkpoints = [F::zero(); RAF_CHECKPOINTS];
        for (slot, value) in raf_checkpoints.iter_mut().zip(raf) {
            *slot = Self::field(value)?;
        }

        if prefix_checkpoints.len() != self.prefix_checkpoints.len()
            || device.v_tables().len() != ADDRESS_BITS / CHUNK_LEN
        {
            return Err(failed());
        }

        self.prefix_checkpoints = prefix_checkpoints
            .into_iter()
            .map(PrefixEval::from)
            .collect();
        self.raf_checkpoints = raf_checkpoints;

        let gamma_sqr = self.gamma * self.gamma;
        let empty = LookupBits::new(0, 0);
        let table_values: Vec<Fr> = LookupTableKind::<RISCV_XLEN>::iter()
            .map(|table| {
                let suffixes: Vec<SuffixEval<F>> = table
                    .suffixes()
                    .iter()
                    .map(|suffix| SuffixEval::from(F::from_u64(suffix.suffix_mle(empty))))
                    .collect();
                require_fr(table.combine(&self.prefix_checkpoints, &suffixes))
            })
            .collect::<Result<_, _>>()
            .map_err(|_| failed())?;
        let raf_interleaved =
            self.gamma * self.raf_checkpoints[0] + gamma_sqr * self.raf_checkpoints[1];
        let mut raf_identity = gamma_sqr * self.raf_checkpoints[2];
        if CANONICAL_INSTRUCTION_ADDRESS {
            raf_identity += gamma_sqr * self.gamma * self.raf_checkpoints[3];
        }

        let tables = build_cycle_tables(
            self.context,
            &HandoffInputs {
                rows: &self.rows,
                v_tables: device.v_tables(),
                table_values: &table_values,
                raf_interleaved: require_fr(raf_interleaved).map_err(|_| failed())?,
                raf_identity: require_fr(raf_identity).map_err(|_| failed())?,
                ra_count: self.ra_count,
                address_bits: ADDRESS_BITS,
            },
        )
        .map_err(|_| failed())?;

        self.cycle = Some(
            DeviceCycleRounds::from_device(
                &self.r_reduction,
                tables.combined_val,
                tables.ra,
                self.rounds - ADDRESS_BITS,
            )
            .map_err(|_| failed())?,
        );
        Ok(())
    }
}

impl<F: Field> PrepareKernel<F, InstructionReadRaf<F>> for CudaBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, InstructionReadRaf<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionReadRaf<F>>>, KernelError<F>> {
        let context = require_context()?;
        let dimensions = inputs.relation.dimensions();
        if dimensions.instruction_address_bits() != ADDRESS_BITS
            || !ADDRESS_BITS.is_multiple_of(CHUNK_LEN)
        {
            return Err(KernelError::Unsupported {
                reason: "the CUDA instruction read-RAF address phase supports only the \
                         2·XLEN interleaved-operand address width in 8-variable phases",
            });
        }
        let rows: Vec<InstructionReadRafWitness> =
            collect_bundles(witness, 1 << dimensions.log_t())?;

        let mut bits = Vec::with_capacity(rows.len() * 2);
        let mut table_index = Vec::with_capacity(rows.len());
        let mut raf_flag = Vec::with_capacity(rows.len());
        for row in &rows {
            bits.push(row.lookup_index.0 as u64);
            bits.push((row.lookup_index.0 >> 64) as u64);
            table_index.push(row.table_index.0.map_or(NO_TABLE, |index| index as u32));
            raf_flag.push(u8::from(row.raf_flag.0));
        }
        drop(rows);

        let unsupported = || KernelError::Unsupported {
            reason: "the CUDA instruction read-RAF kernel supports only the BN254 scalar field",
        };
        let device_rows = Arc::new(
            DeviceRows::from_encoded(context, &bits, &table_index, &raf_flag)
                .map_err(|_| unsupported())?,
        );
        drop(bits);

        let device = DeviceAddressPhase::with_rows(
            context,
            Arc::clone(&device_rows),
            &table_index,
            &inputs.points.lookup_output,
            ADDRESS_BITS,
        )
        .map_err(|_| unsupported())?;

        let r_reduction = require_fr_slice(&inputs.points.lookup_output)
            .map_err(|_| unsupported())?
            .to_vec();
        Ok(Box::new(DeviceInstructionReadRaf {
            device: Some(device),
            cycle: None,
            rows: device_rows,
            r_reduction,
            cycle_challenges: Vec::with_capacity(dimensions.log_t()),
            prefix_checkpoints: ALL_PREFIXES
                .iter()
                .map(|prefix| prefix.default_checkpoint::<F>())
                .collect(),
            raf_checkpoints: raf_initial_checkpoints(),
            ra_count: dimensions.num_virtual_ra_polys(),
            rounds: dimensions.sumcheck_rounds(),
            gamma: inputs.challenges.gamma,
            context,
            rounds_bound: 0,
        }))
    }
}

impl<F: Field> ProveRounds<F> for DeviceInstructionReadRaf<F> {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge)?;
        }
        if self.rounds_bound < ADDRESS_BITS {
            let device = self
                .device
                .as_ref()
                .ok_or(SumcheckError::MissingEvaluationSource {
                    kind: "cuda address phase",
                })?;
            let evals = device
                .round_message_hinted(
                    self.context,
                    require_fr(self.gamma).map_err(|_| SumcheckError::MissingEvaluationSource {
                        kind: "cuda address gamma",
                    })?,
                    require_fr(previous_claim).map_err(|_| {
                        SumcheckError::MissingEvaluationSource {
                            kind: "cuda address claim hint",
                        }
                    })?,
                )
                .map_err(|_| SumcheckError::MissingEvaluationSource {
                    kind: "cuda address round message",
                })?;
            let mut host = [F::zero(); HINT_POINTS];
            for (slot, value) in host.iter_mut().zip(evals) {
                *slot = Self::field(value)?;
            }
            return Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &host));
        }

        let cycle = self
            .cycle
            .as_ref()
            .ok_or(SumcheckError::MissingEvaluationSource {
                kind: "cuda cycle rounds",
            })?;
        cycle
            .round_message(self.context, previous_claim)
            .map_err(|_| SumcheckError::MissingEvaluationSource {
                kind: "cuda cycle round message",
            })
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind)
    }
}

impl<F: Field> DeviceInstructionReadRaf<F> {
    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        if self.rounds_bound < ADDRESS_BITS {
            let scalar =
                require_fr(challenge).map_err(|_| SumcheckError::MissingEvaluationSource {
                    kind: "cuda address-phase challenge",
                })?;
            self.device
                .as_mut()
                .ok_or(SumcheckError::MissingEvaluationSource {
                    kind: "cuda address phase",
                })?
                .bind(self.context, scalar)
                .map_err(|_| SumcheckError::MissingEvaluationSource {
                    kind: "cuda address-phase bind",
                })?;
            self.rounds_bound += 1;
            if self.rounds_bound == ADDRESS_BITS {
                self.enter_cycle_rounds()?;
            }
            Ok(())
        } else {
            let scalar =
                require_fr(challenge).map_err(|_| SumcheckError::MissingEvaluationSource {
                    kind: "cuda cycle-round challenge",
                })?;
            self.cycle
                .as_mut()
                .ok_or(SumcheckError::MissingEvaluationSource {
                    kind: "cuda cycle rounds",
                })?
                .bind(self.context, scalar)
                .map_err(|_| SumcheckError::MissingEvaluationSource {
                    kind: "cuda cycle-round bind",
                })?;
            self.rounds_bound += 1;
            self.cycle_challenges
                .push(require_fr(challenge).map_err(|_| {
                    SumcheckError::MissingEvaluationSource {
                        kind: "cuda cycle-round challenge",
                    }
                })?);
            Ok(())
        }
    }
}

impl<F: Field> SumcheckKernel<F> for DeviceInstructionReadRaf<F> {
    type Relation = InstructionReadRaf<F>;

    fn output_claims(
        &mut self,
        _inputs: &InstructionReadRafInputClaims<F>,
    ) -> Result<InstructionReadRafOutputClaims<F>, SumcheckKernelError<F>> {
        let remaining = self.rounds - self.rounds_bound;
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let cycle = self
            .cycle
            .as_ref()
            .ok_or(SumcheckKernelError::InvariantViolation {
                reason: "cycle rounds absent after full binding",
            })?;
        let instruction_ra: Vec<F> =
            cycle
                .ra_finals(self.context)
                .map_err(|_| SumcheckKernelError::InvariantViolation {
                    reason: "CUDA instruction RA claim readback failed",
                })?;
        let r_cycle: Vec<Fr> = self.cycle_challenges.iter().rev().copied().collect();
        let eq_cycle = self.context.eq_evals(&r_cycle).map_err(|_| {
            SumcheckKernelError::InvariantViolation {
                reason: "CUDA cycle eq table construction failed",
            }
        })?;
        let (flags, raf_flag) = flag_claims(
            self.context,
            &self.rows,
            &eq_cycle,
            LookupTableKind::<RISCV_XLEN>::COUNT,
        )
        .map_err(|_| SumcheckKernelError::InvariantViolation {
            reason: "CUDA flag claim readback failed",
        })?;
        let lookup_table_flags: Vec<F> = flags
            .into_iter()
            .map(|value| {
                fr_into(value).ok_or(SumcheckKernelError::InvariantViolation {
                    reason: "CUDA kernels support only the BN254 scalar field",
                })
            })
            .collect::<Result<_, _>>()?;
        let instruction_raf_flag =
            fr_into(raf_flag).ok_or(SumcheckKernelError::InvariantViolation {
                reason: "CUDA kernels support only the BN254 scalar field",
            })?;
        Ok(InstructionReadRafOutputClaims {
            lookup_table_flags,
            instruction_ra,
            instruction_raf_flag,
        })
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    reason = "test module: device and fixture errors fail loudly"
)]
mod legacy_oracle {
    use std::sync::Arc;

    use jolt_field::Fr;

    use super::ADDRESS_BITS;

    use ark_bn254::Fr as LegacyFr;
    use jolt_prover_legacy::field::JoltField as LegacyJoltField;
    use jolt_prover_legacy::poly::opening_proof::ProverOpeningAccumulator;
    use jolt_prover_legacy::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
    use jolt_prover_legacy::subprotocols::sumcheck_prover::SumcheckInstanceProver;
    use jolt_prover_legacy::transcripts::{Blake2bTranscript, Transcript};
    use jolt_prover_legacy::zkvm::config::OneHotParams;
    use jolt_prover_legacy::zkvm::instruction::{
        Flags as LegacyFlags, InstructionLookup as LegacyLookup, InterleavedBitsMarker,
        JoltTraceCycle, LookupQuery as LegacyLookupQuery,
    };
    use jolt_prover_legacy::zkvm::instruction_lookups::read_raf_checking::{
        InstructionReadRafSumcheckParams, InstructionReadRafSumcheckProver,
    };
    use jolt_prover_legacy::zkvm::lookup_table::LookupTables;
    use jolt_prover_legacy::zkvm::witness::VirtualPolynomial;
    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};
    use strum::IntoEnumIterator;
    use tracer::instruction::Cycle;

    use super::address_driver::DeviceAddressPhase;
    use crate::cuda::common::context::shared_context;

    const LOG_T: usize = 8;

    fn random_cycle(rng: &mut StdRng) -> Cycle {
        let variants: Vec<Cycle> = Cycle::iter().collect();
        for _ in 0..10_000 {
            let index = rng.next_u64() as usize % variants.len();
            let candidate = variants[index].random(rng);
            if JoltTraceCycle::try_new(&candidate).is_ok() {
                return candidate;
            }
        }
        panic!("no convertible cycle variant found");
    }

    fn trace(log_t: usize, seed: u64) -> Vec<Cycle> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..1usize << log_t)
            .map(|_| random_cycle(&mut rng))
            .collect()
    }

    fn consumed_openings(trace: &[Cycle], eq: &[LegacyFr]) -> (LegacyFr, LegacyFr, LegacyFr) {
        let mut rv = <LegacyFr as LegacyJoltField>::from_u64(0);
        let mut left = <LegacyFr as LegacyJoltField>::from_u64(0);
        let mut right = <LegacyFr as LegacyJoltField>::from_u64(0);
        for (index, cycle) in trace.iter().enumerate() {
            let jolt = JoltTraceCycle::try_new(cycle).expect("final Jolt row");
            let lookup_index = LegacyLookupQuery::<64>::to_lookup_index(&jolt);
            if let Some(table) = LegacyLookup::<64>::lookup_table(&jolt) {
                rv += LegacyJoltField::mul_u64(&eq[index], table.materialize_entry(lookup_index));
            }
            let (lo, ro) = LegacyLookupQuery::<64>::to_lookup_operands(cycle);
            left += LegacyJoltField::mul_u64(&eq[index], lo);
            right += LegacyJoltField::mul_u128(&eq[index], ro);
        }
        (rv, left, right)
    }

    #[test]
    fn instruction_read_raf_matches_legacy() {
        let Some(_) = shared_context() else {
            return;
        };
        let trace = Arc::new(trace(LOG_T, 12345));

        let transcript = &mut Blake2bTranscript::new(&[]);
        let mut accumulator = ProverOpeningAccumulator::new(LOG_T);
        let r_cycle: Vec<<LegacyFr as LegacyJoltField>::Challenge> =
            transcript.challenge_vector_optimized::<LegacyFr>(LOG_T);
        let eq = jolt_prover_legacy::poly::eq_poly::EqPolynomial::<LegacyFr>::evals(&r_cycle);
        let (rv, left, right) = consumed_openings(&trace, &eq);

        for (polynomial, claim) in [
            (VirtualPolynomial::LookupOutput, rv),
            (VirtualPolynomial::LeftLookupOperand, left),
            (VirtualPolynomial::RightLookupOperand, right),
        ] {
            accumulator.append_virtual(
                polynomial,
                SumcheckId::InstructionClaimReduction,
                OpeningPoint::<BIG_ENDIAN, LegacyFr>::new(r_cycle.clone()),
                claim,
            );
        }
        accumulator.append_virtual(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanProductVirtualization,
            OpeningPoint::<BIG_ENDIAN, LegacyFr>::new(r_cycle.clone()),
            rv,
        );

        let one_hot = OneHotParams::new(LOG_T, 100, 100);
        let params =
            InstructionReadRafSumcheckParams::new(LOG_T, &one_hot, &accumulator, transcript);
        let legacy_gamma = params.gamma;
        let mut legacy = InstructionReadRafSumcheckProver::initialize(params, Arc::clone(&trace));

        let lookup_index: Vec<u128> = trace
            .iter()
            .map(|cycle| {
                let jolt = JoltTraceCycle::try_new(cycle).expect("final Jolt row");
                LegacyLookupQuery::<64>::to_lookup_index(&jolt)
            })
            .collect();
        let table_index: Vec<Option<usize>> = trace
            .iter()
            .map(|cycle| {
                let jolt = JoltTraceCycle::try_new(cycle).expect("final Jolt row");
                LegacyLookup::<64>::lookup_table(&jolt)
                    .map(|table| LookupTables::<64>::enum_index(&table))
            })
            .collect();
        let raf_flag: Vec<bool> = trace
            .iter()
            .map(|cycle| {
                let jolt = JoltTraceCycle::try_new(cycle).expect("final Jolt row");
                !LegacyFlags::circuit_flags(&jolt).is_interleaved_operands()
            })
            .collect();
        let r_reduction: Vec<Fr> = r_cycle
            .iter()
            .map(|challenge| {
                let value: LegacyFr = (*challenge).into();
                Fr::from(value)
            })
            .collect();
        let context = shared_context().expect("cuda context");
        let mut device = DeviceAddressPhase::new(
            context,
            &lookup_index,
            &table_index,
            &raf_flag,
            &r_reduction,
            ADDRESS_BITS,
        )
        .expect("device address phase");
        let gamma = Fr::from(legacy_gamma);

        let mut claim = <LegacyFr as LegacyJoltField>::from_u64(0);
        let mut rounds_checked = 0usize;
        for round in 0..ADDRESS_BITS {
            rounds_checked += 1;
            let want = SumcheckInstanceProver::<LegacyFr, Blake2bTranscript>::compute_message(
                &mut legacy,
                round,
                claim,
            );
            let got = device
                .round_message_hinted(context, gamma, Fr::from(claim))
                .expect("device message");
            let want_at = |x: u64| want.evaluate(&<LegacyFr as LegacyJoltField>::from_u64(x));
            assert_eq!(
                Fr::from(want_at(0)),
                got[0],
                "round {round} message at X = 0 diverged"
            );
            assert_eq!(
                Fr::from(want_at(2)),
                got[1],
                "round {round} message at X = 2 diverged"
            );
            let challenge = r_cycle[round % r_cycle.len()];
            claim = want.evaluate(&<LegacyFr as From<_>>::from(challenge));
            SumcheckInstanceProver::<LegacyFr, Blake2bTranscript>::ingest_challenge(
                &mut legacy,
                challenge,
                round,
            );
            device
                .bind(context, Fr::from(<LegacyFr as From<_>>::from(challenge)))
                .expect("device bind");
        }
        assert_eq!(
            rounds_checked, ADDRESS_BITS,
            "the oracle must compare every address round"
        );
    }
}
