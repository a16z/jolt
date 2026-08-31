use metal::foreign_types::ForeignType;

use super::super::{
    InstructionReadRafStage1Lease, InstructionReadRafStage1Receipt, MetalError, SolinasMetal,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RegistersValInstructionSourceReceipt {
    cycles: usize,
    explicit_rows: usize,
    device_registry_id: u64,
    generation: u64,
    completion_serial: u64,
    source_storage_ids: [usize; 2],
    source_storage_bytes: [u64; 2],
    instruction_rows_storage_id: usize,
    instruction_rows_bytes: u64,
}

impl RegistersValInstructionSourceReceipt {
    copy_field_getters! { pub(crate), {
        cycles: usize,
        explicit_rows: usize,
        device_registry_id: u64,
        generation: u64,
        completion_serial: u64,
        source_storage_ids: [usize; 2],
        source_storage_bytes: [u64; 2],
        instruction_rows_storage_id: usize,
        instruction_rows_bytes: u64,
    } }
}

pub(crate) struct RegistersValInstructionSourceLease {
    receipt: RegistersValInstructionSourceReceipt,
    source: InstructionReadRafStage1Lease,
}

impl RegistersValInstructionSourceLease {
    copy_field_getters! { pub(crate), { receipt: RegistersValInstructionSourceReceipt }}

    pub(crate) fn into_parts(
        self,
        context: &SolinasMetal,
        expected_cycles: usize,
    ) -> Result<
        (
            RegistersValInstructionSourceReceipt,
            InstructionReadRafStage1Lease,
        ),
        MetalError,
    > {
        validate_instruction_source_lease(&self, context, expected_cycles)?;
        Ok((self.receipt, self.source))
    }
}

pub(crate) struct RegistersValInstructionSourceRequest {
    cycles: usize,
    explicit_rows: usize,
    device_registry_id: u64,
    source_storage_ids: [usize; 2],
    source_storage_bytes: [u64; 2],
    instruction_source: InstructionReadRafStage1Receipt,
}

impl RegistersValInstructionSourceRequest {
    pub(crate) fn publish(
        self,
        context: &SolinasMetal,
        source: InstructionReadRafStage1Lease,
    ) -> Result<RegistersValInstructionSourceLease, MetalError> {
        let source_receipt = source.receipt();
        if source_receipt != self.instruction_source
            || source_receipt.device_registry_id() != self.device_registry_id
            || source_receipt.rows() != self.cycles
            || source.row_buffer().device().registry_id() != self.device_registry_id
            || source.row_buffer().as_ptr() as usize != source_receipt.row_allocation_identity()
            || source.row_buffer().length() != source_receipt.row_bytes()
        {
            return Err(invalid(
                "RegistersVal instruction source does not match its Stage-1 request",
            ));
        }
        let lease = RegistersValInstructionSourceLease {
            receipt: RegistersValInstructionSourceReceipt {
                cycles: self.cycles,
                explicit_rows: self.explicit_rows,
                device_registry_id: self.device_registry_id,
                generation: source_receipt.source_generation(),
                completion_serial: source_receipt.completion_serial(),
                source_storage_ids: self.source_storage_ids,
                source_storage_bytes: self.source_storage_bytes,
                instruction_rows_storage_id: source_receipt.row_allocation_identity(),
                instruction_rows_bytes: source_receipt.row_bytes(),
            },
            source,
        };
        validate_instruction_source_lease(&lease, context, self.cycles)?;
        Ok(lease)
    }
}

impl SolinasMetal {
    #[expect(
        clippy::too_many_arguments,
        reason = "the request validates two source allocations and their provenance at one boundary"
    )]
    pub(crate) fn prepare_registers_val_instruction_source_request(
        &self,
        cycles: usize,
        explicit_rows: usize,
        source_compact_storage_id: usize,
        source_compact_bytes: u64,
        source_residual_storage_id: usize,
        source_residual_bytes: u64,
        instruction_source: InstructionReadRafStage1Receipt,
    ) -> Result<RegistersValInstructionSourceRequest, MetalError> {
        if cycles < 4
            || !cycles.is_power_of_two()
            || explicit_rows > cycles
            || source_compact_storage_id == 0
            || source_residual_storage_id == 0
            || source_compact_storage_id == source_residual_storage_id
            || source_compact_bytes == 0
            || source_residual_bytes == 0
            || instruction_source.rows() != cycles
            || instruction_source.device_registry_id() != self.device_registry_id()
            || instruction_source.row_allocation_identity() == 0
            || instruction_source.claim_allocation_identity() == 0
        {
            return Err(invalid(
                "RegistersVal request does not match its instruction source",
            ));
        }
        Ok(RegistersValInstructionSourceRequest {
            cycles,
            device_registry_id: self.device_registry_id(),
            explicit_rows,
            source_storage_ids: [source_compact_storage_id, source_residual_storage_id],
            source_storage_bytes: [source_compact_bytes, source_residual_bytes],
            instruction_source,
        })
    }
}

fn validate_instruction_source_lease(
    lease: &RegistersValInstructionSourceLease,
    context: &SolinasMetal,
    expected_cycles: usize,
) -> Result<(), MetalError> {
    let receipt = lease.receipt;
    let source = lease.source.receipt();
    if receipt.cycles != expected_cycles
        || receipt.device_registry_id != context.device_registry_id()
        || receipt.generation != source.source_generation()
        || receipt.completion_serial != source.completion_serial()
        || receipt.instruction_rows_storage_id != source.row_allocation_identity()
        || receipt.instruction_rows_bytes != source.row_bytes()
        || lease.source.row_buffer().device().registry_id() != receipt.device_registry_id
        || lease.source.row_buffer().as_ptr() as usize != receipt.instruction_rows_storage_id
        || lease.source.row_buffer().length() != receipt.instruction_rows_bytes
    {
        return Err(invalid(
            "resident RegistersVal instruction source does not match its sealed receipt",
        ));
    }
    Ok(())
}

fn invalid(reason: &'static str) -> MetalError {
    MetalError::InvalidRegistersValState(reason)
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for RegistersValInstructionSourceRequest {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.exit();
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for RegistersValInstructionSourceLease {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("borrowed_instruction_rows"),
            self.receipt.instruction_rows_bytes as usize,
        );
        visitor.exit();
    }
}

#[cfg(test)]
#[expect(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_field::{AkitaField, FromPrimitiveInt};

    use crate::metal::solinas::{
        BooleanityRow, RegistersValFirstMessageConfig, RegistersValTransitionConfig,
        INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS,
    };

    #[test]
    fn stage1_owner_covers_padding_and_signed_increments() {
        let context = match SolinasMetal::for_akita() {
            Ok(context) => context,
            Err(MetalError::DeviceUnavailable) => return,
            Err(error) => panic!("Metal setup failed: {error}"),
        };
        let cycles = 1 << 13;
        let mut instruction = context
            .prepare_instruction_read_raf_stage1_storage(cycles)
            .unwrap();
        instruction
            .with_chunk_writers(|writers| {
                for (chunk, writer) in writers.iter_mut().enumerate() {
                    for local in 0..writer.len() {
                        let cycle = chunk * INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS + local;
                        let (fused_increment, rd_write) = match cycle {
                            0 => (-5, Some((3, 9, 4))),
                            4096 => (22, Some((127, 5, 27))),
                            _ => (0, None),
                        };
                        let row = BooleanityRow::new(0, None, None, fused_increment)?;
                        writer.push_with_register_write(row, 0, false, 0, rd_write)?;
                    }
                }
                Ok(())
            })
            .unwrap();
        let instruction = instruction.seal().unwrap();
        let request = context
            .prepare_registers_val_instruction_source_request(
                cycles,
                4097,
                11,
                48 * cycles as u64,
                12,
                112 * cycles as u64,
                instruction.receipt(),
            )
            .unwrap();
        let source = instruction
            .lease(cycles, context.device_registry_id())
            .unwrap();
        let lease = request.publish(&context, source).unwrap();
        let receipt = lease.receipt();
        assert_eq!(receipt.explicit_rows(), 4097);
        assert_eq!(receipt.cycles(), cycles);
        assert_eq!(
            receipt.instruction_rows_storage_id(),
            instruction.receipt().row_allocation_identity()
        );
        assert_eq!(
            receipt.instruction_rows_bytes(),
            instruction.receipt().row_bytes()
        );

        let mut rd = vec![u8::MAX; cycles];
        let mut inc = vec![AkitaField::zero(); cycles];
        rd[0] = 3;
        rd[4096] = 127;
        inc[0] = AkitaField::from_i128(-5);
        inc[4096] = AkitaField::from_i128(22);
        let r_address = (0..7)
            .map(|index| AkitaField::from_u64((index + 2) as u64))
            .collect::<Vec<_>>();
        let r_cycle = (0..cycles.ilog2() as usize)
            .map(|index| AkitaField::from_u64((index + 11) as u64))
            .collect::<Vec<_>>();
        let baseline = context
            .prepare_registers_val_first_message(
                &inc,
                &rd,
                &r_address,
                &r_cycle,
                RegistersValFirstMessageConfig::default(),
            )
            .unwrap();
        let direct = context
            .prepare_registers_val_first_message_instruction_rows(
                lease,
                &r_address,
                &r_cycle,
                RegistersValFirstMessageConfig::default(),
            )
            .unwrap();
        baseline.execute().unwrap();
        direct.execute().unwrap();
        assert_eq!(
            direct.read_message().unwrap(),
            baseline.read_message().unwrap()
        );

        let bound_lt_lo = (0..32)
            .map(|index| AkitaField::from_u64((index + 29) as u64))
            .collect::<Vec<_>>();
        let challenge = AkitaField::from_u64(313);
        let baseline = baseline
            .into_first_transition(&bound_lt_lo, RegistersValTransitionConfig::default())
            .unwrap();
        let direct = direct
            .into_first_transition(&bound_lt_lo, RegistersValTransitionConfig::default())
            .unwrap();
        baseline.execute(challenge).unwrap();
        direct.execute(challenge).unwrap();
        assert_eq!(
            direct.read_message().unwrap(),
            baseline.read_message().unwrap()
        );
    }
}
