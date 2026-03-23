//! Protocol data types for the Jolt proving pipeline.
//!
//! These structs carry evaluation data between stages. Both prover and
//! verifier populate them — the prover from polynomial evaluation, the
//! verifier from proof data. They contain only scalar values (field
//! elements), no proof artifacts or transcript state.

use jolt_field::Field;
use jolt_openings::{CommittedEval, VirtualEval};
use jolt_spartan::UniformSpartanProof;

/// S1 output: Spartan challenge vectors + virtual polynomial evaluations.
#[derive(Clone)]
pub struct SpartanOutput<F: Field> {
    pub proof: UniformSpartanProof<F>,
    pub r_x: Vec<F>,
    pub r_y: Vec<F>,
    pub evals: SpartanVirtualEvals<F>,
}

/// Virtual polynomial evaluations at `r_cycle` from the Spartan witness.
#[derive(Clone, Copy, Debug)]
pub struct SpartanVirtualEvals<F> {
    pub ram_read_value: VirtualEval<F>,
    pub ram_write_value: VirtualEval<F>,
    pub ram_address: VirtualEval<F>,
    pub lookup_output: VirtualEval<F>,
    pub left_operand: VirtualEval<F>,
    pub right_operand: VirtualEval<F>,
    pub left_instruction_input: VirtualEval<F>,
    pub right_instruction_input: VirtualEval<F>,
    pub rd_write_value: VirtualEval<F>,
    pub rs1_value: VirtualEval<F>,
    pub rs2_value: VirtualEval<F>,
}

/// S2 evaluations: PV factors + claim reduction scalars.
#[derive(Clone, Debug)]
pub struct S2Evals<F: Field> {
    pub eval_point: Vec<F>,
    pub ram_val: VirtualEval<F>,
    pub ram_inc: CommittedEval<F>,
    pub next_is_noop: VirtualEval<F>,
    pub left_instr_input: VirtualEval<F>,
    pub right_instr_input: VirtualEval<F>,
    pub lookup_output: VirtualEval<F>,
    pub left_operand: VirtualEval<F>,
    pub right_operand: VirtualEval<F>,
    pub ram_raf_eval: VirtualEval<F>,
    pub ram_val_final: VirtualEval<F>,
}

/// S3 evaluations: shift + instruction input + registers CR.
#[derive(Clone, Debug)]
pub struct S3Evals<F: Field> {
    pub eval_point: Vec<F>,
    pub rs1_value: VirtualEval<F>,
    pub rs2_value: VirtualEval<F>,
    pub rd_write_value: VirtualEval<F>,
}

/// S4 evaluations: registers RW + RAM val check.
#[derive(Clone, Debug)]
pub struct S4Evals<F: Field> {
    pub eval_point: Vec<F>,
    pub ram_inc: CommittedEval<F>,
    pub rd_inc: CommittedEval<F>,
}

/// S5 evaluations: registers val eval (ReadRaf + RamRaCR deferred).
#[derive(Clone, Debug)]
pub struct S5Evals<F: Field> {
    pub eval_point: Vec<F>,
    pub rd_inc: CommittedEval<F>,
}

/// S6 evaluations: IncCR + HammingBooleanity.
#[derive(Clone, Debug)]
pub struct S6Evals<F: Field> {
    pub r_cycle: Vec<F>,
    pub ram_inc_reduced: CommittedEval<F>,
    pub rd_inc_reduced: CommittedEval<F>,
}

/// S7 evaluations: HammingWeightCR → unified point + RA evals.
#[derive(Clone, Debug)]
pub struct S7Evals<F: Field> {
    pub unified_point: Vec<F>,
    pub instruction_ra: Vec<CommittedEval<F>>,
    pub bytecode_ra: Vec<CommittedEval<F>>,
    pub ram_ra: Vec<CommittedEval<F>>,
}

/// Packs evaluations into a flat vector for proof serialization.
pub trait PackEvals<F: Field> {
    fn pack(&self) -> Vec<F>;
}

/// Unpacks evaluations from a flat vector + eval point.
pub trait UnpackEvals<F: Field>: Sized {
    fn unpack(eval_point: &[F], evals: &[F]) -> Self;
}

impl<F: Field> PackEvals<F> for S2Evals<F> {
    fn pack(&self) -> Vec<F> {
        vec![
            self.ram_val.0, self.ram_inc.eval, self.next_is_noop.0,
            self.left_instr_input.0, self.right_instr_input.0,
            self.lookup_output.0, self.left_operand.0, self.right_operand.0,
            self.ram_raf_eval.0, self.ram_val_final.0,
        ]
    }
}

impl<F: Field> UnpackEvals<F> for S2Evals<F> {
    fn unpack(ep: &[F], e: &[F]) -> Self {
        Self {
            eval_point: ep.to_vec(),
            ram_val: VirtualEval(e[0]),
            ram_inc: CommittedEval { point: ep.to_vec(), eval: e[1] },
            next_is_noop: VirtualEval(e[2]),
            left_instr_input: VirtualEval(e[3]),
            right_instr_input: VirtualEval(e[4]),
            lookup_output: VirtualEval(e[5]),
            left_operand: VirtualEval(e[6]),
            right_operand: VirtualEval(e[7]),
            ram_raf_eval: VirtualEval(e[8]),
            ram_val_final: VirtualEval(e[9]),
        }
    }
}

impl<F: Field> PackEvals<F> for S3Evals<F> {
    fn pack(&self) -> Vec<F> {
        vec![self.rs1_value.0, self.rs2_value.0, self.rd_write_value.0]
    }
}

impl<F: Field> UnpackEvals<F> for S3Evals<F> {
    fn unpack(ep: &[F], e: &[F]) -> Self {
        Self {
            eval_point: ep.to_vec(),
            rs1_value: VirtualEval(e[0]),
            rs2_value: VirtualEval(e[1]),
            rd_write_value: VirtualEval(e[2]),
        }
    }
}

impl<F: Field> PackEvals<F> for S4Evals<F> {
    fn pack(&self) -> Vec<F> {
        vec![self.ram_inc.eval, self.rd_inc.eval]
    }
}

impl<F: Field> UnpackEvals<F> for S4Evals<F> {
    fn unpack(ep: &[F], e: &[F]) -> Self {
        Self {
            eval_point: ep.to_vec(),
            ram_inc: CommittedEval { point: ep.to_vec(), eval: e[0] },
            rd_inc: CommittedEval { point: ep.to_vec(), eval: e[1] },
        }
    }
}

impl<F: Field> PackEvals<F> for S5Evals<F> {
    fn pack(&self) -> Vec<F> {
        vec![self.rd_inc.eval]
    }
}

impl<F: Field> UnpackEvals<F> for S5Evals<F> {
    fn unpack(ep: &[F], e: &[F]) -> Self {
        Self {
            eval_point: ep.to_vec(),
            rd_inc: CommittedEval { point: ep.to_vec(), eval: e[0] },
        }
    }
}

impl<F: Field> PackEvals<F> for S6Evals<F> {
    fn pack(&self) -> Vec<F> {
        vec![self.ram_inc_reduced.eval, self.rd_inc_reduced.eval]
    }
}

impl<F: Field> UnpackEvals<F> for S6Evals<F> {
    fn unpack(ep: &[F], e: &[F]) -> Self {
        Self {
            r_cycle: ep.to_vec(),
            ram_inc_reduced: CommittedEval { point: ep.to_vec(), eval: e[0] },
            rd_inc_reduced: CommittedEval { point: ep.to_vec(), eval: e[1] },
        }
    }
}

impl<F: Field> PackEvals<F> for S7Evals<F> {
    fn pack(&self) -> Vec<F> {
        let mut v = Vec::new();
        for e in &self.instruction_ra { v.push(e.eval); }
        for e in &self.bytecode_ra { v.push(e.eval); }
        for e in &self.ram_ra { v.push(e.eval); }
        v
    }
}
