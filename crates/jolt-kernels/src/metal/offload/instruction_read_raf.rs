//! Metal offload seams for the stage-5 instruction read-RAF kernel: the
//! resident Stage-1 construction, the external address-phase handoff, and
//! the device-resident cycle-round state machine.

use std::sync::Arc;

use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafDimensions;
use jolt_field::AkitaField;
use jolt_lookup_tables::tables::prefixes::ALL_PREFIXES;
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
use jolt_poly::{Polynomial, UnivariatePoly};
use jolt_sumcheck::SumcheckError;
use jolt_verifier::stages::stage5::InstructionReadRaf;
use jolt_witness::JoltWitnessPlane;

use crate::metal::solinas::{
    AddressPhaseSequence, AddressPhaseSequenceConfig, AddressPhaseSums, AddressRafScanRow,
    BooleanityRow, BooleanityRows, Fp128, InstructionReadRafStage1Lease, MetalError,
    Product5Sequence, Product5SequenceConfig, SolinasMetal, PRODUCT5_FACTORS,
};
use crate::optimized::instruction_read_raf::{
    collect_instruction_cycle_rows, CycleTables, InstructionCycleRow,
    InstructionReadRafClaimColumns, OptimizedInstructionReadRafKernel, RafDecomposition, RafSums,
    SharedInstructionRows, CHUNK_LEN, CHUNK_SIZE,
};
use crate::{KernelError, ProofSession, ProverInputs};

impl InstructionCycleRow {
    pub(crate) fn metal_booleanity_rows(rows: &[Self]) -> &[BooleanityRow] {
        // SAFETY: both repr(C) row types are five aligned u64 words in the
        // same order. Booleanity masks the stage-5-only flag bits.
        unsafe { std::slice::from_raw_parts(rows.as_ptr().cast(), rows.len()) }
    }
}

const _: () = assert!(
    std::mem::size_of::<InstructionCycleRow>() == std::mem::size_of::<BooleanityRow>()
        && std::mem::align_of::<InstructionCycleRow>() == std::mem::align_of::<BooleanityRow>()
);

impl InstructionReadRafClaimColumns {
    pub(crate) const fn is_stage1(&self) -> bool {
        matches!(self, Self::Stage1(_))
    }
}

pub(crate) fn prepare_metal_instruction_read_raf(
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<AkitaField>,
    inputs: ProverInputs<'_, AkitaField, InstructionReadRaf<AkitaField>>,
    external_address_phases: bool,
) -> Result<OptimizedInstructionReadRafKernel<AkitaField>, KernelError<AkitaField>> {
    let dimensions = inputs.relation.dimensions();
    let rows: Arc<Vec<InstructionCycleRow>> = Arc::new(collect_instruction_cycle_rows(
        witness,
        1 << dimensions.log_t(),
    )?);
    session.park(SharedInstructionRows(Arc::clone(&rows)));
    OptimizedInstructionReadRafKernel::new_inner(
        dimensions,
        &inputs.points.lookup_output,
        rows,
        inputs.challenges.gamma,
        external_address_phases,
    )
}

impl OptimizedInstructionReadRafKernel<AkitaField> {
    pub(crate) fn new_metal_resident(
        dimensions: InstructionReadRafDimensions,
        r_reduction: &[AkitaField],
        claims: InstructionReadRafStage1Lease,
        gamma: AkitaField,
    ) -> Result<Self, KernelError<AkitaField>> {
        let address_bits = dimensions.instruction_address_bits();
        let log_t = dimensions.log_t();
        let ra_count = dimensions.num_virtual_ra_polys();
        if address_bits != 2 * RISCV_XLEN {
            return Err(KernelError::Unsupported {
                reason: "instruction read-RAF supports only the 2·XLEN interleaved-operand address width",
            });
        }
        if !address_bits.is_multiple_of(ra_count)
            || !(address_bits / ra_count).is_multiple_of(CHUNK_LEN)
        {
            return Err(KernelError::Unsupported {
                reason: "virtual RA chunk width must be a multiple of the phase width",
            });
        }
        if ra_count + 1 != PRODUCT5_FACTORS {
            return Err(KernelError::Unsupported {
                reason: "resident instruction read-RAF requires the four-RA Product5 geometry",
            });
        }
        if log_t >= 32 {
            return Err(KernelError::Unsupported {
                reason: "resident instruction read-RAF cycle indices are u32",
            });
        }
        if r_reduction.len() != log_t {
            return Err(KernelError::TableSizeMismatch {
                table: "instruction claim-reduction point".to_owned(),
                expected: log_t,
                got: r_reduction.len(),
            });
        }
        let rows = 1usize << log_t;
        if claims.receipt().rows() != rows || claims.claim_slice().len() != rows {
            return Err(KernelError::TableSizeMismatch {
                table: "resident instruction read-RAF claims".to_owned(),
                expected: rows,
                got: claims.claim_slice().len(),
            });
        }

        Ok(Self {
            dimensions,
            gamma,
            r_reduction: r_reduction.to_vec(),
            rows: Arc::new(Vec::new()),
            buckets: Vec::new(),
            u_evals: Vec::new(),
            prefix_checkpoints: ALL_PREFIXES
                .iter()
                .map(|prefix| prefix.default_checkpoint::<AkitaField>())
                .collect(),
            prefix_tables: Vec::new(),
            suffix_tables: Vec::new(),
            raf_left: RafDecomposition::empty(),
            raf_right: RafDecomposition::empty(),
            raf_identity: RafDecomposition::empty(),
            raf_upper_all_ones: RafDecomposition::empty_product(),
            v_tables: Vec::new(),
            phase_challenges: Vec::new(),
            cycle_challenges: Vec::new(),
            cycle: None,
            claim_columns: InstructionReadRafClaimColumns::Stage1(claims),
            rounds_bound: 0,
            external_address_phases: true,
        })
    }

    pub(crate) fn metal_prepare_booleanity_rows(
        &self,
        context: &SolinasMetal,
    ) -> Result<BooleanityRows, MetalError> {
        if self.claim_columns.is_stage1() {
            return Err(MetalError::InvalidInstructionReadRafGrouped(
                "resident Stage-1 rows must be borrowed from their owner".to_owned(),
            ));
        }
        context.prepare_booleanity_rows(InstructionCycleRow::metal_booleanity_rows(&self.rows))
    }

    pub(crate) fn metal_prepare_address_sequence(
        &mut self,
        context: &SolinasMetal,
        config: AddressPhaseSequenceConfig,
    ) -> Result<AddressPhaseSequence, SumcheckError<AkitaField>> {
        if self.claim_columns.is_stage1() {
            return Err(metal_state_error(
                "resident Stage-1 state requires prebuilt grouped planes",
            ));
        }
        if !self.external_address_phases || self.rounds_bound != 0 {
            return Err(metal_state_error(
                "resident address handoff requires an unbound external address state",
            ));
        }
        let sequence = context
            .prepare_address_phase_sequence_from_buckets(
                self.rows.len(),
                &self.buckets,
                config,
                |index| {
                    let row = &self.rows[index];
                    (
                        AddressRafScanRow::new_with_table(
                            row.lookup_index(),
                            row.table_index(),
                            row.raf_flag(),
                        ),
                        Fp128::from_jolt_field(&self.u_evals[index]),
                    )
                },
            )
            .map_err(metal_sumcheck_error)?;
        self.u_evals = Vec::new();
        self.buckets = Vec::new();
        Ok(sequence)
    }

    pub(crate) fn metal_address_phase_request(
        &self,
    ) -> Result<(u32, Option<[Fp128; CHUNK_SIZE]>), SumcheckError<AkitaField>> {
        if !self.external_address_phases
            || self.rounds_bound >= self.address_bits()
            || !self.rounds_bound.is_multiple_of(CHUNK_LEN)
        {
            return Err(metal_state_error(
                "resident address phase requested outside a phase boundary",
            ));
        }
        let phase = self.rounds_bound / CHUNK_LEN;
        let previous = if phase == 0 {
            None
        } else {
            let table = self.v_tables.get(phase - 1).ok_or_else(|| {
                metal_state_error("resident address condensation table is absent")
            })?;
            if table.len() != CHUNK_SIZE {
                return Err(metal_state_error(
                    "resident address condensation table has the wrong length",
                ));
            }
            Some(std::array::from_fn(|index| {
                Fp128::from_jolt_field(&table[index])
            }))
        };
        Ok((self.suffix_len(phase) as u32, previous))
    }

    pub(crate) fn metal_install_address_phase(
        &mut self,
        sums: AddressPhaseSums,
    ) -> Result<(), SumcheckError<AkitaField>> {
        if !self.external_address_phases
            || self.rounds_bound >= self.address_bits()
            || !self.rounds_bound.is_multiple_of(CHUNK_LEN)
        {
            return Err(metal_state_error(
                "resident address output arrived outside a phase boundary",
            ));
        }
        let convert = |values: &[Fp128]| {
            values
                .iter()
                .copied()
                .map(Fp128::into_jolt_field::<AkitaField>)
                .collect::<Vec<_>>()
        };
        let raf = RafSums {
            shift_half: convert(sums.raf().shift_half()),
            left: convert(sums.raf().left()),
            right: convert(sums.raf().right()),
            shift_full: convert(sums.raf().shift_full()),
            identity: convert(sums.raf().identity()),
            upper_all_ones: convert(sums.raf().upper_all_ones()),
        };
        let mut suffix_tables = Vec::with_capacity(LookupTableKind::<RISCV_XLEN>::COUNT);
        for table in LookupTableKind::<RISCV_XLEN>::iter() {
            let flat = sums
                .suffix()
                .table(table.index())
                .ok_or_else(|| metal_state_error("resident address suffix table is absent"))?;
            let polynomials = flat
                .chunks_exact(CHUNK_SIZE)
                .map(|coefficients| Polynomial::new(convert(coefficients)))
                .collect();
            suffix_tables.push((table, polynomials));
        }
        self.install_address_phase(self.rounds_bound / CHUNK_LEN, raf, suffix_tables);
        Ok(())
    }

    pub(crate) fn metal_address_active(&self) -> bool {
        self.external_address_phases && self.rounds_bound < self.address_bits()
    }

    pub(crate) fn metal_address_phase_pending(&self) -> bool {
        self.metal_address_active() && self.rounds_bound.is_multiple_of(CHUNK_LEN)
    }

    pub(crate) fn metal_bind_address(
        &mut self,
        challenge: AkitaField,
    ) -> Result<(), SumcheckError<AkitaField>> {
        if !self.metal_address_active() {
            return Err(metal_state_error(
                "resident address bind requested after the address rounds",
            ));
        }
        self.bind(challenge)
    }

    pub(crate) fn metal_address_message(
        &self,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        if !self.metal_address_active() {
            return Err(metal_state_error(
                "resident address message requested after the address rounds",
            ));
        }
        Ok(self.address_message(previous_claim))
    }

    pub(crate) fn metal_resident_cycle_message(
        &self,
        sequence: &mut AddressPhaseSequence,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        let cycle = self
            .cycle
            .as_ref()
            .ok_or(SumcheckError::MissingEvaluationSource { kind: "opening" })?;
        let CycleTables::Pending(pending) = &cycle.tables else {
            return Err(metal_state_error(
                "resident first cycle message requires pending tables",
            ));
        };
        let q_evals = sequence
            .cycle_message(
                &self.v_tables,
                &pending.table_values,
                pending.raf_interleaved,
                pending.raf_identity,
                cycle.gruen.e_in_current(),
                cycle.gruen.e_out_current(),
            )
            .map_err(metal_sumcheck_error)?;
        Ok(cycle.gruen.gruen_poly_from_evals(&q_evals, previous_claim))
    }

    pub(crate) fn metal_resident_cycle_available(&self) -> bool {
        self.dimensions.num_virtual_ra_polys() + 1 == PRODUCT5_FACTORS
    }

    pub(crate) fn metal_offload_resident_bind(
        &mut self,
        challenge: AkitaField,
        sequence: AddressPhaseSequence,
        config: Product5SequenceConfig,
    ) -> Result<(Product5Sequence, [AkitaField; PRODUCT5_FACTORS]), SumcheckError<AkitaField>> {
        let pending = {
            let cycle = self
                .cycle
                .as_mut()
                .ok_or(SumcheckError::MissingEvaluationSource { kind: "opening" })?;
            cycle.gruen.bind(challenge);
            let tables = core::mem::replace(&mut cycle.tables, CycleTables::Offloaded);
            let CycleTables::Pending(pending) = tables else {
                return Err(metal_state_error(
                    "resident cycle handoff requires pending tables",
                ));
            };
            pending
        };
        self.cycle_challenges.push(challenge);
        self.rounds_bound += 1;
        let cycle = self
            .cycle
            .as_ref()
            .ok_or(SumcheckError::MissingEvaluationSource { kind: "opening" })?;
        let result = sequence
            .fused_cycle_transition(
                &self.v_tables,
                &pending.table_values,
                pending.raf_interleaved,
                pending.raf_identity,
                challenge,
                cycle.gruen.e_in_current(),
                cycle.gruen.e_out_current(),
                config,
            )
            .map_err(metal_sumcheck_error)?;
        self.rows = Arc::new(Vec::new());
        self.v_tables = Vec::new();
        Ok(result)
    }

    pub(crate) fn metal_handoff_available(&self, cutoff: usize) -> bool {
        self.dimensions.num_virtual_ra_polys() + 1 == PRODUCT5_FACTORS
            && !self.claim_columns.is_stage1()
            && self.claim_columns.len() / 2 > cutoff
            && self
                .cycle
                .as_ref()
                .is_some_and(|cycle| matches!(cycle.tables, CycleTables::Pending(_)))
    }

    pub(crate) fn metal_offload_pending_bind(
        &mut self,
        challenge: AkitaField,
        context: &SolinasMetal,
        config: Product5SequenceConfig,
    ) -> Result<Product5Sequence, SumcheckError<AkitaField>> {
        if self.claim_columns.is_stage1() {
            return Err(metal_state_error(
                "resident Stage-1 state cannot use the CPU pending-table handoff",
            ));
        }
        let (pending, e_in, e_out) = {
            let cycle = self
                .cycle
                .as_mut()
                .ok_or(SumcheckError::MissingEvaluationSource { kind: "opening" })?;
            cycle.gruen.bind(challenge);
            let tables = core::mem::replace(&mut cycle.tables, CycleTables::Offloaded);
            let CycleTables::Pending(pending) = tables else {
                return Err(SumcheckError::ComputeBackend {
                    backend: "metal",
                    message: "dense-cycle handoff requires pending CPU tables".to_owned(),
                });
            };
            (
                pending,
                cycle.gruen.e_in_current().to_vec(),
                cycle.gruen.e_out_current().to_vec(),
            )
        };

        let claim_columns = self
            .claim_columns
            .as_slice()
            .ok_or_else(|| metal_state_error("cycle claim columns are absent"))?;
        let elements = claim_columns.len() / 2;
        let sequence = context
            .prepare_product5_sequence_from_fn(elements, &e_in, &e_out, config, |index| {
                let factor = index / elements;
                let position = index % elements;
                let source = 2 * position;
                let (lo, hi) = if factor == 0 {
                    (
                        Self::pending_combined_base(&pending, claim_columns, source),
                        Self::pending_combined_base(&pending, claim_columns, source + 1),
                    )
                } else {
                    (
                        self.pending_ra_base(factor - 1, source),
                        self.pending_ra_base(factor - 1, source + 1),
                    )
                };
                lo + challenge * (hi - lo)
            })
            .map_err(metal_sumcheck_error)?;

        self.rows = Arc::new(Vec::new());
        self.v_tables = Vec::new();
        self.cycle_challenges.push(challenge);
        self.rounds_bound += 1;
        Ok(sequence)
    }

    pub(crate) fn metal_bind_offloaded(
        &mut self,
        challenge: AkitaField,
    ) -> Result<(), SumcheckError<AkitaField>> {
        let cycle = self
            .cycle
            .as_mut()
            .ok_or(SumcheckError::MissingEvaluationSource { kind: "opening" })?;
        if !matches!(cycle.tables, CycleTables::Offloaded) {
            return Err(SumcheckError::ComputeBackend {
                backend: "metal",
                message: "device bind requires offloaded cycle tables".to_owned(),
            });
        }
        cycle.gruen.bind(challenge);
        self.cycle_challenges.push(challenge);
        self.rounds_bound += 1;
        Ok(())
    }

    pub(crate) fn metal_cycle_weights(
        &self,
    ) -> Result<(&[AkitaField], &[AkitaField]), SumcheckError<AkitaField>> {
        let cycle = self
            .cycle
            .as_ref()
            .ok_or(SumcheckError::MissingEvaluationSource { kind: "opening" })?;
        Ok((cycle.gruen.e_in_current(), cycle.gruen.e_out_current()))
    }

    pub(crate) fn metal_cycle_message(
        &self,
        q_evals: &[AkitaField; PRODUCT5_FACTORS],
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        let cycle = self
            .cycle
            .as_ref()
            .ok_or(SumcheckError::MissingEvaluationSource { kind: "opening" })?;
        if !matches!(cycle.tables, CycleTables::Offloaded) {
            return Err(SumcheckError::ComputeBackend {
                backend: "metal",
                message: "device message requires offloaded cycle tables".to_owned(),
            });
        }
        Ok(cycle.gruen.gruen_poly_from_evals(q_evals, previous_claim))
    }

    pub(crate) fn metal_restore_dense(
        &mut self,
        tables: [Vec<AkitaField>; PRODUCT5_FACTORS],
    ) -> Result<(), SumcheckError<AkitaField>> {
        let cycle = self
            .cycle
            .as_mut()
            .ok_or(SumcheckError::MissingEvaluationSource { kind: "opening" })?;
        if !matches!(cycle.tables, CycleTables::Offloaded) {
            return Err(SumcheckError::ComputeBackend {
                backend: "metal",
                message: "device readback requires offloaded cycle tables".to_owned(),
            });
        }
        let [combined_val, ra_0, ra_1, ra_2, ra_3] = tables;
        cycle.tables = CycleTables::Dense {
            combined_val: Polynomial::new(combined_val),
            ra: vec![
                Polynomial::new(ra_0),
                Polynomial::new(ra_1),
                Polynomial::new(ra_2),
                Polynomial::new(ra_3),
            ],
        };
        Ok(())
    }
}

fn metal_sumcheck_error(error: crate::metal::solinas::MetalError) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: error.to_string(),
    }
}

fn metal_state_error(message: impl Into<String>) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: message.into(),
    }
}
