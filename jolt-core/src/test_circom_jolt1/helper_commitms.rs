use core::fmt;
use ark_bn254::{Bn254, Fq as Fp};

use crate::{jolt::vm::{bytecode::BytecodeStuff, instruction_lookups::InstructionLookupStuff, read_write_memory::ReadWriteMemoryStuff, timestamp_range_check::TimestampRangeCheckStuff, JoltStuff}, poly::commitment::hyperkzg::HyperKZGCommitment, r1cs::inputs::{AuxVariableStuff, R1CSStuff}};

use super::{convert_fp_to_3_limbs_of_scalar, struct_fq::FqCircom};

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct FpLimbs{
    pub limbs: Vec<FqCircom>
}

impl fmt::Debug for FpLimbs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                    "limbs": {:?}
            }}"#,
            self.limbs
        )
    }
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Fp2Circom{
    pub x: FpLimbs,
    pub y: FpLimbs,
}

impl fmt::Debug for Fp2Circom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                                "x": "{:?}",
                                "y": "{:?}"
                            }}"#,
            self.x,
            self.y
        )
    }
}


pub fn convert_rust_fp_to_circom(fp: &Fp) -> FpLimbs{
    let limbs_ = convert_fp_to_3_limbs_of_scalar(fp);
    let mut limbs = Vec::new();
    for i in 0..3{
        limbs.push(FqCircom(limbs_[i]));
    }
    FpLimbs{
        limbs
    }
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct G1AffineCircom{
    pub x: FpLimbs,
    pub y: FpLimbs,
}

impl fmt::Debug for G1AffineCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                            "x": {:?},
                            "y": {:?}
                            }}"#,
            self.x,
            self.y
        )
    }
}

// use ark_bn254::G1Affine;
// pub fn convert_rust_g1_to_circom(point: G1Affine) -> G1AffineCircom{
//     let x = point.x().unwrap();
//     let y = point.y().unwrap();

//     G1AffineCircom{
//         x: convert_rust_fp_to_circom(x),
//         y: convert_rust_fp_to_circom(y)
//     }
// }

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct HyperKZGCommitmentCircom{
    pub commitment: G1AffineCircom
}

impl fmt::Debug for HyperKZGCommitmentCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                "commitment": {:?}
            }}"#,
            self.commitment,
        )
    }
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct ByteCodeStuffCircom{
    pub a_read_write: HyperKZGCommitmentCircom,
    pub v_read_write: Vec<HyperKZGCommitmentCircom>,
    pub t_read: HyperKZGCommitmentCircom,
    pub t_final: HyperKZGCommitmentCircom,

}

impl fmt::Debug for ByteCodeStuffCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                "a_read_write": {:?},
                "v_read_write": {:?},
                "t_read": {:?},
                "t_final": {:?}
            }}"#,
            self.a_read_write, self.v_read_write, self.t_read, self.t_final
        )
    }
}

pub fn convert_from_byte_code_stuff_to_circom(byte_code_stuff: &BytecodeStuff<HyperKZGCommitment<Bn254>>) -> ByteCodeStuffCircom{
    let mut v_read_write = Vec::new();
    for i in 0..byte_code_stuff.v_read_write.len(){
        v_read_write.push(convert_hyperkzg_commitment_to_circom(&byte_code_stuff.v_read_write[i].clone()))
    }
    ByteCodeStuffCircom{
        a_read_write: convert_hyperkzg_commitment_to_circom(&byte_code_stuff.a_read_write),
        v_read_write: v_read_write,
        t_read: convert_hyperkzg_commitment_to_circom(&byte_code_stuff.t_read),
        t_final: convert_hyperkzg_commitment_to_circom(&byte_code_stuff.t_final)
    }
}

pub fn convert_hyperkzg_commitment_to_circom(
    commitment: &HyperKZGCommitment<Bn254>,
) -> HyperKZGCommitmentCircom {
    HyperKZGCommitmentCircom {
        commitment: G1AffineCircom{
            x: convert_rust_fp_to_circom(&commitment.0.x),
            y: convert_rust_fp_to_circom(&commitment.0.y),
        }
    }
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct ReadWriteMemoryStuffCircom{
    pub a_ram: HyperKZGCommitmentCircom,
    /// RD read_value
    pub v_read_rd: HyperKZGCommitmentCircom,
    /// RS1 read_value
    pub v_read_rs1: HyperKZGCommitmentCircom,
    /// RS2 read_value
    pub v_read_rs2: HyperKZGCommitmentCircom,
    /// RAM read_value
    pub v_read_ram: HyperKZGCommitmentCircom,
    /// RD write value
    pub v_write_rd: HyperKZGCommitmentCircom,
    /// RAM write value
    pub v_write_ram: HyperKZGCommitmentCircom,
    /// Final memory state.
    pub v_final: HyperKZGCommitmentCircom,
    /// RD read timestamp
    pub t_read_rd: HyperKZGCommitmentCircom,
    /// RS1 read timestamp
    pub t_read_rs1: HyperKZGCommitmentCircom,
    /// RS2 read timestamp
    pub t_read_rs2: HyperKZGCommitmentCircom,
    /// RAM read timestamp
    pub t_read_ram: HyperKZGCommitmentCircom,
    /// Final timestamps.
    pub t_final: HyperKZGCommitmentCircom,
}

impl fmt::Debug for ReadWriteMemoryStuffCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                "a_ram": {:?},
                "v_read_rd": {:?},
                "v_read_rs1": {:?},
                "v_read_rs2": {:?},
                "v_read_ram": {:?},
                "v_write_rd": {:?},
                "v_write_ram": {:?},
                "v_final": {:?},
                "t_read_rd": {:?},
                "t_read_rs1": {:?},
                "t_read_rs2": {:?},
                "t_read_ram": {:?},
                "t_final": {:?}
            }}"#,
            self.a_ram, self.v_read_rd, self.v_read_rs1, self.v_read_rs2, self.v_read_ram, self.v_write_rd, self.v_write_ram, self.v_final, self.t_read_rd, self.t_read_rs1, self.t_read_rs2, self.t_read_ram, self.t_final
        )
    }
}


pub fn convert_from_read_write_mem_stuff_to_circom(rw_stuff: &ReadWriteMemoryStuff<HyperKZGCommitment<Bn254>>) -> ReadWriteMemoryStuffCircom{

    ReadWriteMemoryStuffCircom{
        a_ram: convert_hyperkzg_commitment_to_circom(&rw_stuff.a_ram.clone()),
        v_read_rd: convert_hyperkzg_commitment_to_circom(&rw_stuff.v_read_rd),
        v_read_rs1: convert_hyperkzg_commitment_to_circom(&rw_stuff.v_read_rs1),
        v_read_rs2: convert_hyperkzg_commitment_to_circom(&rw_stuff.v_read_rs2),
        v_read_ram: convert_hyperkzg_commitment_to_circom(&rw_stuff.v_read_ram),
        v_write_rd: convert_hyperkzg_commitment_to_circom(&rw_stuff.v_write_rd),
        v_write_ram: convert_hyperkzg_commitment_to_circom(&rw_stuff.v_write_ram),
        v_final: convert_hyperkzg_commitment_to_circom(&rw_stuff.v_final),
        t_read_rd: convert_hyperkzg_commitment_to_circom(&rw_stuff.t_read_rd),
        t_read_rs1: convert_hyperkzg_commitment_to_circom(&rw_stuff.t_read_rs1),
        t_read_rs2: convert_hyperkzg_commitment_to_circom(&rw_stuff.t_read_rs2),
        t_read_ram: convert_hyperkzg_commitment_to_circom(&rw_stuff.t_read_ram),
        t_final: convert_hyperkzg_commitment_to_circom(&rw_stuff.t_final)
    }
}

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct InstructionLookupStuffCircom{
    pub dim: Vec<HyperKZGCommitmentCircom>,
    pub read_cts: Vec<HyperKZGCommitmentCircom>,
    pub final_cts: Vec<HyperKZGCommitmentCircom>,
    pub E_polys: Vec<HyperKZGCommitmentCircom>,
    pub instruction_flags: Vec<HyperKZGCommitmentCircom>,
    pub lookup_outputs: HyperKZGCommitmentCircom,
    // instruction_flag_bitvectors commented
}

impl fmt::Debug for InstructionLookupStuffCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                "dim": {:?},
                "read_cts": {:?},
                "final_cts": {:?},
                "E_polys": {:?},
                "instruction_flags": {:?},
                "lookup_outputs": {:?}
            }}"#,
            self.dim, self.read_cts, self.final_cts, self.E_polys, self.instruction_flags, self.lookup_outputs
        )
    }
}

pub fn convert_from_ins_lookup_stuff_to_circom(ins_lookup_stuff: &InstructionLookupStuff<HyperKZGCommitment<Bn254>>) -> InstructionLookupStuffCircom{
    let mut dim = Vec::new();
    let mut read_cts = Vec::new();
    let mut final_cts = Vec::new();
    let mut E_polys = Vec::new();
    let mut instruction_flags = Vec::new();
    for i in 0..ins_lookup_stuff.dim.len(){
        dim.push(convert_hyperkzg_commitment_to_circom(&ins_lookup_stuff.dim[i].clone()));
    }
    for i in 0..ins_lookup_stuff.read_cts.len(){
        read_cts.push(convert_hyperkzg_commitment_to_circom(&ins_lookup_stuff.read_cts[i].clone()));
    }
    for i in 0..ins_lookup_stuff.final_cts.len(){
        final_cts.push(convert_hyperkzg_commitment_to_circom(&ins_lookup_stuff.final_cts[i].clone()))
    }
    for i in 0..ins_lookup_stuff.E_polys.len(){
        E_polys.push(convert_hyperkzg_commitment_to_circom(&ins_lookup_stuff.E_polys[i].clone()));
    }
    for i in 0..ins_lookup_stuff.instruction_flags.len(){
        instruction_flags.push(convert_hyperkzg_commitment_to_circom(&ins_lookup_stuff.instruction_flags[i].clone()));
    }

    InstructionLookupStuffCircom{
        dim,
        read_cts,
        final_cts,
        E_polys,
        instruction_flags,
        lookup_outputs: convert_hyperkzg_commitment_to_circom(&ins_lookup_stuff.lookup_outputs)
    }
}


#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct TimestampRangeCheckStuffCircom{
    pub read_cts_read_timestamp: Vec<HyperKZGCommitmentCircom>,
    pub read_cts_global_minus_read: Vec<HyperKZGCommitmentCircom>,
    pub final_cts_read_timestamp: Vec<HyperKZGCommitmentCircom>,
    pub final_cts_global_minus_read: Vec<HyperKZGCommitmentCircom>,
}

impl fmt::Debug for TimestampRangeCheckStuffCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                "read_cts_read_timestamp": {:?},
                "read_cts_global_minus_read": {:?},
                "final_cts_read_timestamp": {:?},
                "final_cts_global_minus_read": {:?}
            }}"#,
            self.read_cts_read_timestamp, self.read_cts_global_minus_read, self.final_cts_read_timestamp, self.final_cts_global_minus_read
        )
    }
}

pub fn convert_from_ts_lookup_stuff_to_circom(ts_lookup_stuff: &TimestampRangeCheckStuff<HyperKZGCommitment<Bn254>>) -> TimestampRangeCheckStuffCircom{

    let mut read_cts_read_timestamp = Vec::new();
    let mut read_cts_global_minus_read = Vec::new();
    let mut final_cts_read_timestamp = Vec::new();
    let mut final_cts_global_minus_read = Vec::new();
    for i in 0..ts_lookup_stuff.read_cts_read_timestamp.len(){
        read_cts_read_timestamp.push(convert_hyperkzg_commitment_to_circom(&ts_lookup_stuff.read_cts_read_timestamp[i].clone()))
    }
    for i in 0..ts_lookup_stuff.read_cts_global_minus_read.len(){
        read_cts_global_minus_read.push(convert_hyperkzg_commitment_to_circom(&ts_lookup_stuff.read_cts_global_minus_read[i].clone()));
    }
    for i in 0..ts_lookup_stuff.final_cts_read_timestamp.len(){
        final_cts_read_timestamp.push(convert_hyperkzg_commitment_to_circom(&ts_lookup_stuff.final_cts_read_timestamp[i].clone()));
    }
    for i in 0..ts_lookup_stuff.final_cts_global_minus_read.len(){
        final_cts_global_minus_read.push(convert_hyperkzg_commitment_to_circom(&ts_lookup_stuff.final_cts_global_minus_read[i].clone()));
    }


    TimestampRangeCheckStuffCircom{
        read_cts_read_timestamp,
        read_cts_global_minus_read,
        final_cts_read_timestamp,
        final_cts_global_minus_read,
    }
}


#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct AuxVariableStuffCircom{
    pub left_lookup_operand: HyperKZGCommitmentCircom,
    pub right_lookup_operand: HyperKZGCommitmentCircom,
    pub product: HyperKZGCommitmentCircom,
    pub relevant_y_chunks: Vec<HyperKZGCommitmentCircom>,
    pub write_lookup_output_to_rd: HyperKZGCommitmentCircom,
    pub write_pc_to_rd: HyperKZGCommitmentCircom,
    pub next_pc_jump: HyperKZGCommitmentCircom,
    pub should_branch: HyperKZGCommitmentCircom,
    pub next_pc: HyperKZGCommitmentCircom

}

impl fmt::Debug for AuxVariableStuffCircom {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                f,
                r#"{{
                    "left_lookup_operand": {:?},
                    "right_lookup_operand": {:?},
                    "product": {:?},
                    "relevant_y_chunks": {:?},
                    "write_lookup_output_to_rd": {:?},
                    "write_pc_to_rd": {:?},
                    "next_pc_jump": {:?},
                    "should_branch": {:?},
                    "next_pc": {:?}
                }}"#,
                self.left_lookup_operand, self.right_lookup_operand, self.product, self.relevant_y_chunks, self.write_lookup_output_to_rd, self.write_pc_to_rd, self.next_pc_jump, self.should_branch, self.next_pc
            )
        }
    }


    pub fn convert_from_aux_stuff_to_circom(aux_stuff: &AuxVariableStuff<HyperKZGCommitment<Bn254>>) -> AuxVariableStuffCircom{
        let mut relevant_y_chunks = Vec::new();
        for i in 0..aux_stuff.relevant_y_chunks.len(){
            relevant_y_chunks.push(convert_hyperkzg_commitment_to_circom(&aux_stuff.relevant_y_chunks[i].clone()))
        }
        AuxVariableStuffCircom{
            left_lookup_operand: convert_hyperkzg_commitment_to_circom(&aux_stuff.left_lookup_operand),
            right_lookup_operand: convert_hyperkzg_commitment_to_circom(&aux_stuff.right_lookup_operand),
            product: convert_hyperkzg_commitment_to_circom(&aux_stuff.product),
            relevant_y_chunks: relevant_y_chunks,
            write_lookup_output_to_rd: convert_hyperkzg_commitment_to_circom(&aux_stuff.write_lookup_output_to_rd),
            write_pc_to_rd: convert_hyperkzg_commitment_to_circom(&aux_stuff.write_pc_to_rd),
            next_pc_jump: convert_hyperkzg_commitment_to_circom(&aux_stuff.next_pc_jump),
            should_branch: convert_hyperkzg_commitment_to_circom(&aux_stuff.should_branch),
            next_pc: convert_hyperkzg_commitment_to_circom(&aux_stuff.next_pc)
        }
    }

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct R1CSStuffCircom{
    pub chunks_x: Vec<HyperKZGCommitmentCircom>,
    pub chunks_y: Vec<HyperKZGCommitmentCircom>,
    pub circuit_flags: Vec<HyperKZGCommitmentCircom>,
    pub aux: AuxVariableStuffCircom,
}

impl fmt::Debug for R1CSStuffCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                "chunks_x": {:?},
                "chunks_y": {:?},
                "circuit_flags": {:?},
                "aux": {:?}
            }}"#,
            self.chunks_x, self.chunks_y, self.circuit_flags, self.aux
        )
    }
}

pub fn convert_from_r1cs_stuff_to_circom(r1cs_stuff: &R1CSStuff<HyperKZGCommitment<Bn254>>) -> R1CSStuffCircom{
    let mut chunks_x = Vec::new();
    let mut chunks_y = Vec::new();
    let mut circuit_flags = Vec::new();
    for i in 0..r1cs_stuff.chunks_x.len(){
        chunks_x.push(convert_hyperkzg_commitment_to_circom(&&r1cs_stuff.chunks_x[i].clone()));
    }
    for i in 0..r1cs_stuff.chunks_y.len(){
        chunks_y.push(convert_hyperkzg_commitment_to_circom(&r1cs_stuff.chunks_y[i].clone()));
    }
    for i in 0..r1cs_stuff.circuit_flags.len(){
        circuit_flags.push(convert_hyperkzg_commitment_to_circom(&r1cs_stuff.circuit_flags[i].clone()));
    }
    R1CSStuffCircom{
        chunks_x,
        chunks_y,
        circuit_flags,
        aux: convert_from_aux_stuff_to_circom(&r1cs_stuff.aux),
    }
}


#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct JoltStuffCircom{
    pub bytecode: ByteCodeStuffCircom,
    pub read_write_memory: ReadWriteMemoryStuffCircom,
    pub instruction_lookups: InstructionLookupStuffCircom,
    pub timestamp_range_check: TimestampRangeCheckStuffCircom,
    pub r1cs: R1CSStuffCircom
}

impl fmt::Debug for JoltStuffCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {


        write!(
            f,
            r#"{{
                "bytecode": {:?},
                "read_write_memory": {:?},
                "instruction_lookups": {:?},
                "timestamp_range_check": {:?},
                "r1cs": {:?}
        }}"#,
            self.bytecode, self.read_write_memory, self.instruction_lookups, self.timestamp_range_check, self.r1cs
        )
    }
}

pub fn convert_from_jolt_stuff_to_circom(jolt_stuff: &JoltStuff<HyperKZGCommitment<Bn254>>) -> JoltStuffCircom{
    JoltStuffCircom{
        bytecode: convert_from_byte_code_stuff_to_circom(&jolt_stuff.bytecode),
        read_write_memory: convert_from_read_write_mem_stuff_to_circom(&jolt_stuff.read_write_memory),
        instruction_lookups: convert_from_ins_lookup_stuff_to_circom(&jolt_stuff.instruction_lookups),
        timestamp_range_check: convert_from_ts_lookup_stuff_to_circom(&jolt_stuff.timestamp_range_check),
        r1cs: convert_from_r1cs_stuff_to_circom(&jolt_stuff.r1cs)
    }
}