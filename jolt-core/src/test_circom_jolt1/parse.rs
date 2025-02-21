use ark_bn254::Bn254;
use ark_ec::AffineRepr;
use ark_ff::{AdditiveGroup, PrimeField};
use itertools::Itertools;
use num_bigint::BigUint;
use serde_json::json;
use tracer::JoltDevice;

use crate::{
    field::JoltField,
    jolt::vm::{
        bytecode::{BytecodeProof, BytecodeStuff},
        instruction_lookups::InstructionLookupStuff,
        read_write_memory::ReadWriteMemoryStuff,
        rv32i_vm::{RV32ISubtables, C, M, RV32I},
        timestamp_range_check::TimestampRangeCheckStuff,
        JoltCommitments, JoltProof, JoltStuff,
    },
    lasso::memory_checking::{MultisetHashes, StructuredPolynomialData},
    poly::{
        commitment::{
            hyperkzg::{HyperKZG, HyperKZGCommitment, HyperKZGProof, HyperKZGVerifierKey},
            hyrax::{HyraxCommitment, HyraxOpeningProof, HyraxScheme},
            kzg::KZGVerifierKey,
            pedersen::PedersenGenerators,
        },
        unipoly::UniPoly,
    },
    r1cs::inputs::{JoltR1CSInputs, R1CSStuff},
    spartan::spartan_memory_checking::SpartanProof,
    subprotocols::{
        grand_product::{
            BatchedGrandProduct, BatchedGrandProductLayerProof, BatchedGrandProductProof,
        },
        sumcheck::SumcheckInstanceProof,
    },
    utils::poseidon_transcript::PoseidonTranscript,
};
type Fr = ark_bn254::Fr;
type Fq = ark_bn254::Fq;
pub type PCS = HyperKZG<ark_bn254::Bn254, ProofTranscript>;

// type Fr = ark_grumpkin::Fr;
// type Fq = ark_grumpkin::Fq;
pub type ProofTranscript = PoseidonTranscript<Fr, Fr>;
// pub type PCS = HyraxScheme<ark_grumpkin::Projective, ProofTranscript>;

pub trait Circomfmt {
    fn format(&self) -> serde_json::Value;
    fn format_setup(&self, _size: usize) -> serde_json::Value {
        unimplemented!("added for setup")
    }
}

pub fn convert_to_3_limbs(r: Fr) -> [Fq; 3] {
    let mut limbs = [Fq::ZERO; 3];

    let mask = BigUint::from((1u128 << 125) - 1);
    limbs[0] = Fq::from(BigUint::from(r.into_bigint()) & mask.clone());

    limbs[1] = Fq::from((BigUint::from(r.into_bigint()) >> 125) & mask.clone());

    limbs[2] = Fq::from((BigUint::from(r.into_bigint()) >> 250) & mask.clone());

    limbs
}

impl Circomfmt for Fr {
    fn format(&self) -> serde_json::Value {
        let limbs = convert_to_3_limbs(*self);
        json!({
            "limbs": [limbs[0].to_string(), limbs[1].to_string(), limbs[2].to_string()]
        })
    }
}

impl Circomfmt for HyperKZGVerifierKey<ark_bn254::Bn254> {
    fn format(&self) -> serde_json::Value {
        json!({
            "kzg_vk": self.kzg_vk.format()
        })
    }
}
impl Circomfmt for KZGVerifierKey<ark_bn254::Bn254> {
    fn format(&self) -> serde_json::Value {
        json!({
            "g1": self.g1.format(),
            "g2":{ "x": {"x": self.g2.x.c0.to_string(),
                        "y": self.g2.x.c1.to_string()
                    },

                    "y": {"x": self.g2.y.c0.to_string(),
                         "y": self.g2.y.c1.to_string()
                    },
                },
            "beta_g2": {"x": {"x": self.beta_g2.x.c0.to_string(),
                              "y": self.beta_g2.x.c1.to_string()},

                        "y": {"x": self.beta_g2.y.c0.to_string(),
                              "y": self.beta_g2.y.c1.to_string()}
                        }
        })
    }
}
impl Circomfmt for ark_bn254::G1Affine {
    fn format(&self) -> serde_json::Value {
        json!({
            "x": self.x.to_string(),
            "y": self.y.to_string()
        })
    }
}

impl Circomfmt for SpartanProof<Fr, PCS, ProofTranscript> {
    fn format(&self) -> serde_json::Value {
        json!({
            "outer_sumcheck_proof": self.outer_sumcheck_proof.format(),
            "inner_sumcheck_proof": self.inner_sumcheck_proof.format(),
            "outer_sumcheck_claims": [self.outer_sumcheck_claims.0.format(),self.outer_sumcheck_claims.1.format(),self.outer_sumcheck_claims.2.format()],
            "inner_sumcheck_claims": [self.inner_sumcheck_claims.0.format(),self.inner_sumcheck_claims.1.format(),self.inner_sumcheck_claims.2.format(),self.inner_sumcheck_claims.3.format()],
            "pi_eval": self.pi_eval.format(),
            "joint_opening_proof": self.pcs_proof.format()
        })
    }
}
impl Circomfmt for HyperKZGProof<ark_bn254::Bn254> {
    fn format(&self) -> serde_json::Value {
        let com: Vec<serde_json::Value> = self.com.iter().map(|c| c.format()).collect();
        let w: Vec<serde_json::Value> = self.w.iter().map(|c| c.format()).collect();
        let v: Vec<Vec<serde_json::Value>> = self
            .v
            .iter()
            .map(|v_inner| v_inner.iter().map(|elem| elem.format()).collect())
            .collect();
        json!({
            "com": com,
            "w": w,
            "v": v,
        })
    }
}
// impl Circomfmt for SumcheckInstanceProof<Fr, ProofTranscript> {
//     fn format(&self) -> serde_json::Value {
//         let uni_polys: Vec<serde_json::Value> =
//             self.uni_polys.iter().map(|poly| poly.format()).collect();
//         json!({
//             "uni_polys": uni_polys,
//         })
//     }
// }
// impl Circomfmt for UniPoly<Fr> {
//     fn format(&self) -> serde_json::Value {
//         let coeffs: Vec<serde_json::Value> =
//             self.coeffs.iter().map(|coeff| coeff.format()).collect();
//         json!({
//             "coeffs": coeffs,
//         })
//     }
// }
impl Circomfmt for HyperKZGCommitment<ark_bn254::Bn254> {
    fn format(&self) -> serde_json::Value {
        json!({
            "commitment": self.0.format(),
        })
    }
}
// impl Circomfmt for HyraxCommitment<ark_grumpkin::Projective> {
//     fn format(&self) -> serde_json::Value {
//         let commitments: Vec<serde_json::Value> = self
//             .row_commitments
//             .iter()
//             .map(|commit| commit.into_affine().into_group().format())
//             .collect();
//         json!({
//             "row_commitments": commitments,
//         })
//     }
// }

// impl Circomfmt for HyraxOpeningProof<ark_grumpkin::Projective, ProofTranscript> {
//     fn format(&self) -> serde_json::Value {
//         let vector_matrix_product: Vec<serde_json::Value> = self
//             .vector_matrix_product
//             .iter()
//             .map(|elem| elem.format())
//             .collect();
//         json!({
//             "tau": vector_matrix_product,
//         })
//     }
// }

impl Circomfmt for ark_grumpkin::Projective {
    fn format(&self) -> serde_json::Value {
        json!({
            "x": self.x.to_string(),
            "y": self.y.to_string(),
            "z": self.z.to_string()
        })
    }
}
impl Circomfmt for PoseidonTranscript<Fr, Fq> {
    fn format(&self) -> serde_json::Value {
        json!({
            "state": self.state.state[1].to_string(),
            "nRounds": self.n_rounds.to_string(),
        })
    }
}
impl Circomfmt for PedersenGenerators<ark_grumpkin::Projective> {
    fn format(&self) -> serde_json::Value {
        unimplemented!("Use format_setup")
    }
    fn format_setup(&self, size: usize) -> serde_json::Value {
        let generators: Vec<serde_json::Value> = self
            .generators
            .iter()
            .take(size)
            .map(|gen| gen.into_group().format())
            .collect();
        json!({
            "gens": generators
        })
    }
}
// impl Circomfmt for HyraxGenerators<ark_grumpkin::Projective> {
//     fn format(&self) -> serde_json::Value {
//         json!({
//             "gens": self.gens.format()

//         })
//     }
// }

trait ParseJolt {
    fn format(&self) -> serde_json::Value;
}
impl ParseJolt
    for JoltProof<
        { C },
        { M },
        JoltR1CSInputs,
        Fr,
        HyperKZG<Bn254, PoseidonTranscript<Fr, Fr>>,
        RV32I,
        RV32ISubtables<Fr>,
        PoseidonTranscript<Fr, Fr>,
    >
{
    fn format(&self) -> serde_json::Value {
        json!({
            "trace_length": self.trace_length.to_string(),
            "program_io": self.program_io.format(),
            "bytecode": self.bytecode.format(),
            // "read_write_memory": {:?},
            // "instruction_lookups": {:?},
            // "r1cs": {:?},
            // "opening_proof": {:?},
            // "pi_proof": {:?}

        })
    }
}
//TODO:- Verify this impl
impl ParseJolt for JoltDevice {
    fn format(&self) -> serde_json::Value {
        let inputs: Vec<String> = self
            .inputs
            .iter()
            .map(|input| (*input as usize).to_string())
            .collect();
        let outputs: Vec<String> = self
            .outputs
            .iter()
            .map(|output| (*output as usize).to_string())
            .collect();
        json!({
               "inputs": inputs,
                "outputs": outputs,
                "panic": self.panic.to_string()
        })
    }
}
impl ParseJolt
    for BytecodeProof<Fr, HyperKZG<Bn254, PoseidonTranscript<Fr, Fr>>, PoseidonTranscript<Fr, Fr>>
{
    fn format(&self) -> serde_json::Value {
        let openings: Vec<String> = self
            .openings
            .read_write_values()
            .iter()
            .chain(self.openings.init_final_values().iter())
            .collect_vec()
            .iter()
            .map(|opening| opening.to_string())
            .collect();
        json!({
            "multiset_hashes": self.multiset_hashes.format(),
            "read_write_grand_product": self.read_write_grand_product.format(),
            "init_final_grand_product": self.init_final_grand_product.format(),
            "openings":openings
        })
    }
}

impl ParseJolt for MultisetHashes<Fr> {
    fn format(&self) -> serde_json::Value {
        let read_hashes: Vec<serde_json::Value> =
            self.read_hashes.iter().map(|hash| hash.format()).collect();
        let write_hashes: Vec<serde_json::Value> =
            self.write_hashes.iter().map(|hash| hash.format()).collect();
        let init_hashes: Vec<serde_json::Value> =
            self.init_hashes.iter().map(|hash| hash.format()).collect();
        let final_hashes: Vec<serde_json::Value> =
            self.final_hashes.iter().map(|hash| hash.format()).collect();

        json!({
            "read_hashes": read_hashes,
            "write_hashes": write_hashes,
            "init_hashes": init_hashes,
            "final_hashes": final_hashes
        })
    }
}

impl ParseJolt for JoltStuff<HyperKZGCommitment<Bn254>> {
    fn format(&self) -> serde_json::Value {
        json!({
            "bytecode": self.bytecode.format(),
            "read_write_memory": self.read_write_memory.format(),
            "instruction_lookups": self.instruction_lookups.format(),
            "timestamp_range_check": self.timestamp_range_check.format(),
            "r1cs": self.r1cs.format()
        })
    }
}

impl ParseJolt for BatchedGrandProductProof<PCS, ProofTranscript> {
    fn format(&self) -> serde_json::Value {
        let num_gkr_layers = self.gkr_layers.len();

        let num_coeffs = self.gkr_layers[num_gkr_layers - 1].proof.uni_polys[0]
            .coeffs
            .len();

        let max_no_polys = self.gkr_layers[num_gkr_layers - 1].proof.uni_polys.len();

        let mut updated_gkr_layers = Vec::new();

        for idx in 0..num_gkr_layers {
            let zero_poly = UniPoly::from_coeff(vec![Fr::from(0u8); num_coeffs]);
            let len = self.gkr_layers[idx].proof.uni_polys.len();
            let updated_uni_poly: Vec<_> = self.gkr_layers[idx]
                .proof
                .uni_polys
                .clone()
                .into_iter()
                .chain(vec![zero_poly; max_no_polys - len].into_iter())
                .collect();
            let layer = BatchedGrandProductLayerProof {
                proof: SumcheckInstanceProof::new(updated_uni_poly),
                left_claim: self.gkr_layers[idx].left_claim,
                right_claim: self.gkr_layers[idx].right_claim,
            };
            updated_gkr_layers.push(layer.format());
        }

        json!({
            "gkr_layers": updated_gkr_layers
        })
    }
}
impl ParseJolt for BatchedGrandProductLayerProof<Fr, ProofTranscript> {
    fn format(&self) -> serde_json::Value {
        json!({
            "proof": self.proof.format(),
            "left_claim": self.left_claim.to_string(),
            "right_claim": self.left_claim.to_string(),
        })
    }
}
impl ParseJolt for SumcheckInstanceProof<Fr, ProofTranscript> {
    fn format(&self) -> serde_json::Value {
        let uni_polys: Vec<serde_json::Value> =
            self.uni_polys.iter().map(|poly| poly.format()).collect();
        json!({
            "uni_polys": uni_polys,
        })
    }
}
impl ParseJolt for UniPoly<Fr> {
    fn format(&self) -> serde_json::Value {
        let coeffs: Vec<String> = self.coeffs.iter().map(|coeff| coeff.to_string()).collect();
        json!({
            "coeffs": coeffs,
        })
    }
}
impl ParseJolt for BytecodeStuff<HyperKZGCommitment<Bn254>> {
    fn format(&self) -> serde_json::Value {
        let v_read_write: Vec<serde_json::Value> =
            self.v_read_write.iter().map(|v| v.format()).collect();
        json!({
            "a_read_write": self.a_read_write.format(),
            "v_read_write": v_read_write,
            "t_read": self.t_read.format(),
            "t_final": self.t_final.format()
        })
    }
}

impl ParseJolt for ReadWriteMemoryStuff<HyperKZGCommitment<Bn254>> {
    fn format(&self) -> serde_json::Value {
        json!({
            "a_ram": self.a_ram.format(),
                "v_read_rd": self.v_read_rd.format(),
                "v_read_rs1": self.v_read_rs1.format(),
                "v_read_rs2": self.v_read_rs2.format(),
                "v_read_ram": self.v_read_ram.format(),
                "v_write_rd": self.v_write_rd.format(),
                "v_write_ram": self.v_write_ram.format(),
                "v_final": self.v_final.format(),
                "t_read_rd": self.t_read_rd.format(),
                "t_read_rs1": self.t_read_rs1.format(),
                "t_read_rs2": self.t_read_rs2.format(),
                "t_read_ram": self.t_read_ram.format(),
                "t_final": self.t_final.format()
        })
    }
}
impl ParseJolt for InstructionLookupStuff<HyperKZGCommitment<Bn254>> {
    fn format(&self) -> serde_json::Value {
        let dim: Vec<serde_json::Value> = self.dim.iter().map(|com| com.format()).collect();
        let read_cts: Vec<serde_json::Value> =
            self.read_cts.iter().map(|com| com.format()).collect();
        let final_cts: Vec<serde_json::Value> =
            self.final_cts.iter().map(|com| com.format()).collect();
        let E_polys: Vec<serde_json::Value> = self.E_polys.iter().map(|com| com.format()).collect();
        let instruction_flags: Vec<serde_json::Value> = self
            .instruction_flags
            .iter()
            .map(|com| com.format())
            .collect();
        json!({
            "dim": dim,
            "read_cts": read_cts,
            "final_cts": final_cts,
            "E_polys": E_polys,
            "instruction_flags": instruction_flags,
            "lookup_outputs": self.lookup_outputs.format()
        })
    }
}
impl ParseJolt for TimestampRangeCheckStuff<HyperKZGCommitment<Bn254>> {
    fn format(&self) -> serde_json::Value {
        let read_cts_read_timestamp: Vec<serde_json::Value> = self
            .read_cts_read_timestamp
            .iter()
            .map(|com| com.format())
            .collect();
        let read_cts_global_minus_read: Vec<serde_json::Value> = self
            .read_cts_global_minus_read
            .iter()
            .map(|com| com.format())
            .collect();
        let final_cts_read_timestamp: Vec<serde_json::Value> = self
            .final_cts_read_timestamp
            .iter()
            .map(|com| com.format())
            .collect();
        let final_cts_global_minus_read: Vec<serde_json::Value> = self
            .final_cts_global_minus_read
            .iter()
            .map(|com| com.format())
            .collect();
        json!({
             "read_cts_read_timestamp": read_cts_read_timestamp,
                "read_cts_global_minus_read":read_cts_global_minus_read,
                "final_cts_read_timestamp": final_cts_read_timestamp,
                "final_cts_global_minus_read": final_cts_global_minus_read
        })
    }
}
impl ParseJolt for R1CSStuff<HyperKZGCommitment<Bn254>> {
    fn format(&self) -> serde_json::Value {
        let chunks_x: Vec<serde_json::Value> =
            self.chunks_x.iter().map(|com| com.format()).collect();
        let chunks_y: Vec<serde_json::Value> =
            self.chunks_y.iter().map(|com| com.format()).collect();
        let circuit_flags: Vec<serde_json::Value> =
            self.circuit_flags.iter().map(|com| com.format()).collect();
        json!({
            "chunks_x": chunks_x,
            "chunks_y": chunks_y,
            "circuit_flags": circuit_flags
        })
    }
}
// struct JoltPreprocessHash<F: JoltField> {
//     pub v_init_final_hash: F,
//     pub bytecode_words_hash: F,
// }
