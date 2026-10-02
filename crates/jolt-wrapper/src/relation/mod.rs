//! The Jolt verifier's stage algebra as an R1CS over `Fr`.
//!
//! `build_relation` walks the verifier's Fiat-Shamir schedule once per
//! [`WrapperProfile`] and emits, for stages 1–7, the stage-8 RLC and the Dory
//! `Fr` scalar algebra: every absorbed field element is a witness variable,
//! every squeeze is a plain challenge wire (its hash binding is the transcript
//! table's job), every sumcheck round is a Horner chain, and every verifier
//! formula is lowered from the `jolt-claims` symbolic expressions or the
//! per-`JoltDerivedId` gadget mirroring the native `derive_*_term`.
//! `generate_witness` runs the same walk in assign mode against a recording of
//! the native verifier, so the schedule is asserted event-for-event.

mod ctx;
mod dory;
mod gadgets;
mod lower;
mod public_io;
mod replay;
mod stage1;
mod stage2;
mod stage3;
mod stage4;
mod stage5;
mod stage6a;
mod stage6b;
mod stage7;
mod stage8;
mod sumcheck;
mod tables;
mod wiring;

use common::jolt_device::JoltDevice;
use jolt_crypto::{Bn254G1, Pedersen};
use jolt_dory::DoryScheme;
use jolt_field::Fr;
use jolt_r1cs::{ConstraintMatrices, Variable};
use jolt_verifier::{JoltProof, JoltVerifierPreprocessing, VerifierError};
use thiserror::Error;

pub use dory::{DoryLinks, DoryScalar};
pub use public_io::{outsourced_inputs, EvaluationPoints, OutsourcedInputs, StageValueInputs};
pub use replay::SqueezeKind;

use crate::hash_table::Recorded;
use crate::profile::WrapperProfile;
use ctx::Ctx;
use replay::Replay;
use wiring::Wires;

pub type Pcs = DoryScheme;
pub type Vc = Pedersen<Bn254G1>;
pub type Proof = JoltProof<Pcs, Vc>;
pub type Preprocessing = JoltVerifierPreprocessing<Pcs, Vc>;

pub const NUM_STAGE_VALUES: usize = 5;

#[derive(Debug, Error)]
pub enum RelationError {
    #[error("unsupported proof shape: {0}")]
    Unsupported(&'static str),
    #[error("verifier replay: {0}")]
    Verifier(#[from] VerifierError),
    #[error(
        "transcript schedule mismatch at event {index}: circuit {expected}, replay {recorded}"
    )]
    Schedule {
        index: usize,
        expected: String,
        recorded: String,
    },
    #[error("missing {kind} source {id}")]
    MissingSource { kind: &'static str, id: String },
    #[error("{kind} {id}: gadget {gadget:?}, native {native:?}")]
    NativeMismatch {
        kind: &'static str,
        id: String,
        gadget: Fr,
        native: Fr,
    },
    #[error("geometry: {0}")]
    Geometry(String),
    #[error("witness value missing for variable {0}")]
    Witness(usize),
    #[error("public input {0} was never assigned")]
    PublicInput(&'static str),
}

/// One transcript event of the wrapped schedule, in Fiat-Shamir order. The
/// transcript table (T1) consumes this list to bind the challenge wires to
/// the hash chain: constant bytes are labels/counts, `Fr` entries are
/// 32-byte big-endian encodings of the named variable, opaque entries are
/// prover bytes the relation never sees (Dory group elements).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScheduleEntry {
    Bytes(Vec<u8>),
    Fr(Variable),
    Opaque { len: usize },
    Squeeze { kind: SqueezeKind, var: Variable },
}

/// Which witness variables carry protocol meaning for the other tables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinkTable {
    pub schedule: Vec<ScheduleEntry>,
    pub dory: DoryLinks,
}

/// The seven public columns and the private wires at which they are evaluated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicLayout {
    pub num_public: usize,
    pub val_io: Variable,
    pub init_eval: Variable,
    pub stage_values: [Variable; NUM_STAGE_VALUES],
    pub evaluation_points: EvaluationPoints,
}

/// A labeled run of constraint rows, for attributing `check_witness` failures.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowSpan {
    pub label: String,
    pub start: usize,
    pub end: usize,
}

pub struct Relation {
    pub matrices: ConstraintMatrices<Fr>,
    pub public: PublicLayout,
    pub link: LinkTable,
    pub rows: Vec<RowSpan>,
}

impl Relation {
    pub fn row_label(&self, row: usize) -> Option<&str> {
        self.rows
            .iter()
            .find(|span| span.start <= row && row < span.end)
            .map(|span| span.label.as_str())
    }
}

pub struct Witness {
    /// The full assignment `z` (index 0 is the constant one).
    pub values: Vec<Fr>,
    /// Transcript state after the natively absorbed preamble and commitments.
    pub state_in: [u8; 32],
    pub outsourced: OutsourcedInputs,
}

/// The native data the assign-mode walk evaluates the outsourced public
/// inputs against.
pub(crate) struct Native<'a> {
    pub preprocessing: &'a Preprocessing,
    pub public_io: &'a JoltDevice,
}

pub fn build_relation(profile: &WrapperProfile) -> Result<Relation, RelationError> {
    let mut ctx = Ctx::new(None);
    let walked = walk(&mut ctx, profile, None)?;
    let (matrices, _, schedule, rows) = ctx.finish();
    Ok(Relation {
        matrices,
        public: walked.public,
        link: LinkTable {
            schedule,
            dory: walked.dory,
        },
        rows,
    })
}

pub fn generate_witness(
    profile: &WrapperProfile,
    preprocessing: &Preprocessing,
    public_io: &JoltDevice,
    proof: &Proof,
) -> Result<Witness, RelationError> {
    generate_witness_with_records(profile, preprocessing, public_io, proof)
        .map(|(witness, _)| witness)
}

pub(crate) fn generate_witness_with_records(
    profile: &WrapperProfile,
    preprocessing: &Preprocessing,
    public_io: &JoltDevice,
    proof: &Proof,
) -> Result<(Witness, Vec<Recorded>), RelationError> {
    let Replay {
        events,
        state_in,
        records,
    } = replay::replay(preprocessing, public_io, proof)?;
    let mut ctx = Ctx::new(Some(events));
    let native = Native {
        preprocessing,
        public_io,
    };
    let Walked { public, .. } = walk(&mut ctx, profile, Some(&native))?;
    let (_, values, _, _) = ctx.finish();
    let values = values
        .into_iter()
        .enumerate()
        .map(|(index, value)| value.ok_or(RelationError::Witness(index)))
        .collect::<Result<Vec<_>, _>>()?;
    let read = |variable: Variable| values[variable.index()];
    let outsourced = OutsourcedInputs {
        val_io: read(public.val_io),
        init_eval: read(public.init_eval),
        stage_values: public.stage_values.map(read),
    };
    Ok((
        Witness {
            values,
            state_in,
            outsourced,
        },
        records,
    ))
}

/// The whole schedule, stage by stage. Build mode (`native == None`) leaves
/// every prover-supplied and challenge wire unassigned; assign mode reads them
/// off the replay and evaluates the outsourced inputs natively.
struct Walked {
    public: PublicLayout,
    dory: DoryLinks,
}

fn walk(
    ctx: &mut Ctx,
    profile: &WrapperProfile,
    native: Option<&Native<'_>>,
) -> Result<Walked, RelationError> {
    let public = public_io::allocate(ctx, profile);
    let mut wires = Wires::default();
    let stage1 = stage1::walk(ctx, profile, &mut wires)?;
    let stage2 = stage2::walk(ctx, profile, native, &public, &mut wires, &stage1)?;
    let stage3 = stage3::walk(ctx, profile, &mut wires, &stage2)?;
    let stage4 = stage4::walk(ctx, profile, native, &public, &mut wires, &stage2, &stage3)?;
    let stage5 = stage5::walk(ctx, profile, &mut wires, &stage2, &stage4)?;
    let stage6a = stage6a::walk(ctx, profile, &public, &mut wires, &stage5)?;
    let stage6b = stage6b::walk(
        ctx, profile, native, &public, &mut wires, &stage1, &stage2, &stage3, &stage4, &stage5,
        &stage6a,
    )?;
    let stage7 = stage7::walk(ctx, profile, &mut wires, &stage5, &stage6a, &stage6b)?;
    let dory = stage8::walk(ctx, profile, &wires, &stage5, &stage7)?;
    ctx.expect_replay_consumed()?;
    Ok(Walked {
        public: public.finish(),
        dory,
    })
}
