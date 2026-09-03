//! The public column: the outsourced evaluations the native verifier
//! computes from the program (`ValIo`, `InitEval`, the five bytecode-table
//! stage folds) and the challenge wires those evaluations are taken at.
//! Public variables occupy `z[1..=num_public]` so the wrapping Spartan can
//! treat them as its public-input prefix.

use common::jolt_device::JoltDevice;
use jolt_claims::protocols::jolt::geometry::bytecode::{
    read_raf_stage_values, BytecodeReadRafStageValueInputs, BYTECODE_STAGE_GAMMA_COUNTS,
};
use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_field::{Fr, One, Zero};
use jolt_poly::{sparse_segments_mle_msb, EqPolynomial};
use jolt_program::preprocess::{PublicInitialRam, PublicIoMemory};
use jolt_r1cs::Variable;

use super::ctx::{lc_var, Ctx, Lc};
use super::{Native, Preprocessing, PublicLayout, RelationError, NUM_STAGE_VALUES};
use crate::profile::WrapperProfile;

pub const NUM_BYTECODE_GAMMAS: usize = 6;

/// The challenge wires the outsourced inputs are evaluated at, copied into
/// the public column so the outside evaluator sees exactly the wires the
/// relation used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicOutputs {
    /// RAM address point (`log_k_ram` coordinates): `InitEval`'s and `ValIo`'s
    /// point (the read-write and output-check members bind the same wires).
    pub ram_address: Vec<Variable>,
    /// Bytecode address point (`log_k_bytecode` coordinates): the stage folds' point.
    pub bytecode_address: Vec<Variable>,
    /// `[gamma, stage1_gamma, …, stage5_gamma]` of the bytecode read-RAF.
    pub bytecode_gammas: [Variable; NUM_BYTECODE_GAMMAS],
    /// Register address point (7 coordinates): the read-write and value-evaluation
    /// members bind the same wires.
    pub register_address: [Variable; REGISTER_ADDRESS_BITS],
}

/// The outsourced public inputs, as the native verifier computes them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutsourcedInputs {
    pub val_io: Fr,
    pub init_eval: Fr,
    pub stage_values: [Fr; NUM_STAGE_VALUES],
}

pub(crate) struct PublicSlots {
    layout: PublicLayout,
}

pub(crate) fn allocate(ctx: &mut Ctx, profile: &WrapperProfile) -> PublicSlots {
    ctx.section("public");
    let mut next = || ctx.alloc(None);
    let val_io = next();
    let init_eval = next();
    let stage_values = std::array::from_fn(|_| next());
    let ram_address = (0..profile.log_k_ram).map(|_| next()).collect();
    let bytecode_address = (0..profile.log_k_bytecode).map(|_| next()).collect();
    let bytecode_gammas = std::array::from_fn(|_| next());
    let register_address = std::array::from_fn(|_| next());
    let num_public = 2
        + NUM_STAGE_VALUES
        + profile.log_k_ram
        + profile.log_k_bytecode
        + NUM_BYTECODE_GAMMAS
        + REGISTER_ADDRESS_BITS;
    PublicSlots {
        layout: PublicLayout {
            num_public,
            val_io,
            init_eval,
            stage_values,
            outputs: PublicOutputs {
                ram_address,
                bytecode_address,
                bytecode_gammas,
                register_address,
            },
        },
    }
}

impl PublicSlots {
    pub(crate) fn val_io(&self) -> Lc {
        lc_var(self.layout.val_io)
    }

    pub(crate) fn init_eval(&self) -> Lc {
        lc_var(self.layout.init_eval)
    }

    pub(crate) fn stage_value(&self, stage: usize) -> Lc {
        lc_var(self.layout.stage_values[stage])
    }

    pub(crate) fn outputs(&self) -> &PublicOutputs {
        &self.layout.outputs
    }

    pub(crate) fn val_io_slot(&self) -> Variable {
        self.layout.val_io
    }

    pub(crate) fn init_eval_slot(&self) -> Variable {
        self.layout.init_eval
    }

    pub(crate) fn stage_value_slot(&self, stage: usize) -> Variable {
        self.layout.stage_values[stage]
    }

    /// Copies a challenge wire into its public slot (one equality row).
    pub(crate) fn bind_outputs(
        ctx: &mut Ctx,
        slots: &[Variable],
        wires: &[Lc],
    ) -> Result<(), RelationError> {
        assert_eq!(slots.len(), wires.len(), "public output arity");
        for (&slot, wire) in slots.iter().zip(wires) {
            if let Some(value) = ctx.value(wire) {
                ctx.assign(slot, value)?;
            }
            ctx.assert_eq(&lc_var(slot), wire);
        }
        Ok(())
    }

    /// Assigns an outsourced input (assign mode only; build mode leaves the
    /// slot free).
    pub(crate) fn set_input(
        ctx: &mut Ctx,
        slot: Variable,
        value: Option<Fr>,
    ) -> Result<(), RelationError> {
        if let Some(value) = value {
            ctx.assign(slot, value)?;
        }
        Ok(())
    }

    pub(crate) fn finish(self) -> PublicLayout {
        self.layout
    }
}

/// `ValIo(r)`: the public-IO MLE (inputs, outputs, panic, termination) at the
/// output-check address, masked to the IO region by the high coordinates.
pub(crate) fn val_io(native: &Native<'_>, address: &[Fr]) -> Result<Fr, RelationError> {
    let memory = PublicIoMemory::new(native.public_io)
        .map_err(|error| RelationError::Geometry(error.to_string()))?;
    let split = address
        .len()
        .checked_sub(memory.io_num_vars())
        .ok_or_else(|| RelationError::Geometry("output address shorter than IO domain".into()))?;
    let (r_hi, r_lo) = address.split_at(split);
    let hi_scale = r_hi
        .iter()
        .fold(Fr::one(), |acc, coordinate| acc * (Fr::one() - *coordinate));
    Ok(hi_scale
        * sparse_segments_mle_msb(
            memory
                .segments
                .iter()
                .map(|segment| (segment.start_index, segment.words.as_slice())),
            r_lo,
        ))
}

/// `InitEval(r)`: the public initial RAM (program image + inputs) MLE at the
/// RAM read-write address.
pub(crate) fn init_eval(native: &Native<'_>, address: &[Fr]) -> Result<Fr, RelationError> {
    let full = native
        .preprocessing
        .program
        .as_full()
        .ok_or(RelationError::Unsupported("committed program"))?;
    let initial = PublicInitialRam::new(&full.ram, native.public_io)
        .map_err(|error| RelationError::Geometry(error.to_string()))?;
    Ok(sparse_segments_mle_msb(
        initial
            .segments
            .iter()
            .map(|segment| (segment.start_index, segment.words.as_slice())),
        address,
    ))
}

pub struct StageValueInputs<'a> {
    pub bytecode_address: &'a [Fr],
    pub register_address: &'a [Fr],
    /// `[stage1_gamma, …, stage5_gamma]`.
    pub stage_gammas: [Fr; NUM_STAGE_VALUES],
}

/// The address-only bytecode-table folds: each row's five staged values,
/// folded against `eq(r_address, row)`.
pub(crate) fn stage_values(
    native: &Native<'_>,
    inputs: StageValueInputs<'_>,
) -> Result<[Fr; NUM_STAGE_VALUES], RelationError> {
    let full = native
        .preprocessing
        .program
        .as_full()
        .ok_or(RelationError::Unsupported("committed program"))?;
    let bytecode = full.bytecode.bytecode.as_slice();
    if bytecode.len() != 1 << inputs.bytecode_address.len() {
        return Err(RelationError::Geometry(format!(
            "bytecode has {} rows, address domain is 2^{}",
            bytecode.len(),
            inputs.bytecode_address.len()
        )));
    }
    let powers: [Vec<Fr>; NUM_STAGE_VALUES] = std::array::from_fn(|stage| {
        let mut powers = vec![Fr::one(); BYTECODE_STAGE_GAMMA_COUNTS[stage]];
        for index in 1..powers.len() {
            powers[index] = powers[index - 1] * inputs.stage_gammas[stage];
        }
        powers
    });
    let rows = read_raf_stage_values(BytecodeReadRafStageValueInputs {
        bytecode,
        register_read_write_point: inputs.register_address,
        register_val_evaluation_point: inputs.register_address,
        stage1_gammas: &powers[0],
        stage2_gammas: &powers[1],
        stage3_gammas: &powers[2],
        stage4_gammas: &powers[3],
        stage5_gammas: &powers[4],
    });
    let eq_evals = EqPolynomial::<Fr>::evals(inputs.bytecode_address, None);
    let mut folded = [Fr::zero(); NUM_STAGE_VALUES];
    for (row, eq) in rows.into_iter().zip(eq_evals) {
        for (acc, value) in folded.iter_mut().zip(row) {
            *acc += value * eq;
        }
    }
    Ok(folded)
}

/// The outsourced inputs from the verifier's data alone, for callers that
/// hold the challenge values (the public outputs) but not a witness.
pub fn outsourced_inputs(
    native_preprocessing: &Preprocessing,
    public_io: &JoltDevice,
    ram_address: &[Fr],
    stage_inputs: StageValueInputs<'_>,
) -> Result<OutsourcedInputs, RelationError> {
    let native = Native {
        preprocessing: native_preprocessing,
        public_io,
    };
    Ok(OutsourcedInputs {
        val_io: val_io(&native, ram_address)?,
        init_eval: init_eval(&native, ram_address)?,
        stage_values: stage_values(&native, stage_inputs)?,
    })
}
