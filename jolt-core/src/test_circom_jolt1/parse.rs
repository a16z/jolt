use std::{fs::File, io::Write};

use ark_bn254::Bn254;
use ark_ec::AffineRepr;
use ark_ff::{AdditiveGroup, PrimeField};
use itertools::Itertools;
use num_bigint::BigUint;
use serde_json::json;
use tracer::JoltDevice;

use crate::{
    jolt::vm::{
        bytecode::{BytecodeProof, BytecodeStuff},
        instruction_lookups::{
            InstructionLookupStuff, InstructionLookupsProof, PrimarySumcheck,
            PrimarySumcheckOpenings,
        },
        read_write_memory::{OutputSumcheckProof, ReadWriteMemoryProof, ReadWriteMemoryStuff},
        rv32i_vm::{RV32ISubtables, C, M, RV32I},
        timestamp_range_check::{TimestampRangeCheckStuff, TimestampValidityProof},
        JoltCommitments, JoltPreprocessing, JoltProof, JoltStuff,
    },
    lasso::memory_checking::{MultisetHashes, StructuredPolynomialData},
    poly::{
        commitment::{
            hyperkzg::{HyperKZG, HyperKZGCommitment, HyperKZGProof, HyperKZGVerifierKey},
            hyrax::{HyraxCommitment, HyraxOpeningProof, HyraxScheme},
            kzg::KZGVerifierKey,
            pedersen::PedersenGenerators,
        },
        opening_proof::ReducedOpeningProof,
        unipoly::UniPoly,
    },
    r1cs::{
        inputs::{AuxVariableStuff, JoltR1CSInputs, R1CSStuff},
        spartan::UniformSpartanProof,
    },
    spartan::spartan_memory_checking::SpartanProof,
    subprotocols::{
        grand_product::{BatchedGrandProductLayerProof, BatchedGrandProductProof},
        sumcheck::SumcheckInstanceProof,
    },
    test_circom_jolt1::fib_e2e,
    utils::{poseidon_transcript::PoseidonTranscript, transcript::Transcript},
};
type Fr = ark_bn254::Fr;
type Fq = ark_bn254::Fq;
pub type PCS = HyperKZG<ark_bn254::Bn254, ProofTranscript>;

// type Fr = ark_grumpkin::Fr;
// type Fq = ark_grumpkin::Fq;
pub type ProofTranscript = PoseidonTranscript<Fr, Fr>;
// pub type PCS = HyraxScheme<ark_grumpkin::Projective, ProofTranscript>;

// pub trait Circomfmt {
//     fn format(&self) -> serde_json::Value {
//         unimplemented!("")
//     }
//     fn format_non_native(&self) -> serde_json::Value {
//         unimplemented!("")
//     }
//     fn format_setup(&self, _size: usize) -> serde_json::Value {
//         unimplemented!("added for setup")
//     }
// }

pub fn convert_to_3_limbs(r: Fr) -> [Fq; 3] {
    let mut limbs = [Fq::ZERO; 3];

    let mask = BigUint::from((1u128 << 125) - 1);
    limbs[0] = Fq::from(BigUint::from(r.into_bigint()) & mask.clone());

    limbs[1] = Fq::from((BigUint::from(r.into_bigint()) >> 125) & mask.clone());

    limbs[2] = Fq::from((BigUint::from(r.into_bigint()) >> 250) & mask.clone());

    limbs
}

pub fn convert_fq_to_limbs(r: Fq) -> [Fr; 3] {
    let mut limbs = [Fr::ZERO; 3];

    let mask = BigUint::from((1u128 << 125) - 1);
    limbs[0] = Fr::from(BigUint::from(r.into_bigint()) & mask.clone());

    limbs[1] = Fr::from((BigUint::from(r.into_bigint()) >> 125) & mask.clone());

    limbs[2] = Fr::from((BigUint::from(r.into_bigint()) >> 250) & mask.clone());

    limbs
}

trait ParseJolt {
    fn format(&self) -> serde_json::Value {
        unimplemented!("")
    }
    fn format_non_native(&self) -> serde_json::Value {
        unimplemented!("")
    }
    fn format_setup(&self, _size: usize) -> serde_json::Value {
        unimplemented!("added for setup")
    }
    fn format_embeded(&self) -> serde_json::Value {
        unimplemented!("")
    }
}

impl ParseJolt for Fr {
    fn format_non_native(&self) -> serde_json::Value {
        let limbs = convert_to_3_limbs(*self);
        json!({
            "limbs": [limbs[0].to_string(), limbs[1].to_string(), limbs[2].to_string()]
        })
    }
}

//Parse Group
impl ParseJolt for ark_bn254::G1Affine {
    fn format(&self) -> serde_json::Value {
        json!({
            "x": self.x.to_string(),
            "y": self.y.to_string()
        })
    }

    fn format_non_native(&self) -> serde_json::Value {
        let x_limbs = convert_fq_to_limbs(self.x);
        let y_limbs = convert_fq_to_limbs(self.y);
        json!({
            "x":{"limbs": [x_limbs[0].to_string(), x_limbs[1].to_string(), x_limbs[2].to_string()]},
            "y":{"limbs": [y_limbs[0].to_string(), y_limbs[1].to_string(), y_limbs[2].to_string()]}
        })
    }
}

impl ParseJolt for ark_grumpkin::Projective {
    fn format(&self) -> serde_json::Value {
        json!({
            "x": self.x.to_string(),
            "y": self.y.to_string(),
            "z": self.z.to_string()
        })
    }
}
//Parse CommitmentSchems
//Parse Hyperkzg
impl ParseJolt for HyperKZGVerifierKey<ark_bn254::Bn254> {
    fn format(&self) -> serde_json::Value {
        json!({
            "kzg_vk": self.kzg_vk.format()
        })
    }
}

impl ParseJolt for KZGVerifierKey<ark_bn254::Bn254> {
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

impl ParseJolt for HyperKZGCommitment<ark_bn254::Bn254> {
    fn format(&self) -> serde_json::Value {
        json!({
            "commitment": self.0.format(),
        })
    }
    fn format_non_native(&self) -> serde_json::Value {
        json!({
            "commitment": self.0.format_non_native(),
        })
    }
}

impl ParseJolt for HyperKZGProof<ark_bn254::Bn254> {
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
    fn format_non_native(&self) -> serde_json::Value {
        let com: Vec<serde_json::Value> = self.com.iter().map(|c| c.format_non_native()).collect();
        let w: Vec<serde_json::Value> = self.w.iter().map(|c| c.format_non_native()).collect();
        let v: Vec<Vec<String>> = self
            .v
            .iter()
            .map(|v_inner| v_inner.iter().map(|elem| elem.to_string()).collect())
            .collect();
        json!({
            "com": com,
            "w": w,
            "v": v,
        })
    }
}
//Parse Hyrax

// impl ParseJolt for HyraxCommitment<ark_grumpkin::Projective> {
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

// impl ParseJolt for HyraxOpeningProof<ark_grumpkin::Projective, ProofTranscript> {
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

// impl ParseJolt for HyraxGenerators<ark_grumpkin::Projective> {
//     fn format(&self) -> serde_json::Value {
//         json!({
//             "gens": self.gens.format()

//         })
//     }
// }

impl ParseJolt for PedersenGenerators<ark_grumpkin::Projective> {
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

//Parse Spartan
impl ParseJolt for SpartanProof<Fr, PCS, ProofTranscript> {
    fn format(&self) -> serde_json::Value {
        json!({
            "outer_sumcheck_proof": self.outer_sumcheck_proof.format_non_native(),
            "inner_sumcheck_proof": self.inner_sumcheck_proof.format_non_native(),
            "outer_sumcheck_claims": [self.outer_sumcheck_claims.0.format_non_native(),self.outer_sumcheck_claims.1.format_non_native(),self.outer_sumcheck_claims.2.format_non_native()],
            "inner_sumcheck_claims": [self.inner_sumcheck_claims.0.format_non_native(),self.inner_sumcheck_claims.1.format_non_native(),self.inner_sumcheck_claims.2.format_non_native(),self.inner_sumcheck_claims.3.format_non_native()],
            "pi_eval": self.pi_eval.format_non_native(),
            "joint_opening_proof": self.pcs_proof.format_non_native()
        })
    }
}

pub struct BytecodeCombiners {
    pub rho: [Fr; 2],
}
impl ParseJolt for BytecodeCombiners {
    fn format_non_native(&self) -> serde_json::Value {
        json!({
            "rho": [self.rho[0].format_non_native(), self.rho[1].format_non_native()]
        })
    }
}

pub struct InstructionLookupCombiners {
    pub rho: [Fr; 3],
}

impl ParseJolt for InstructionLookupCombiners {
    fn format_non_native(&self) -> serde_json::Value {
        json!({
            "rho": [self.rho[0].format_non_native(), self.rho[1].format_non_native(), self.rho[2].format_non_native()]
        })
    }
}
pub struct ReadWriteOutputTimestampCombiners {
    pub rho: [Fr; 4],
}
impl ParseJolt for ReadWriteOutputTimestampCombiners {
    fn format_non_native(&self) -> serde_json::Value {
        json!({
            "rho": [self.rho[0].format_non_native(), self.rho[1].format_non_native(), self.rho[2].format_non_native(), self.rho[3].format_non_native()]
        })
    }
}
pub struct R1CSCombiners {
    pub rho: Fr,
}
impl ParseJolt for R1CSCombiners {
    fn format_non_native(&self) -> serde_json::Value {
        json!({
            "rho": self.rho.format_non_native()
        })
    }
}
pub struct OpeningCombiners {
    pub bytecode_combiners: BytecodeCombiners,
    pub instruction_lookup_combiners: InstructionLookupCombiners,
    pub read_write_output_timestamp_combiners: ReadWriteOutputTimestampCombiners,
    pub r1cs_combiners: R1CSCombiners,
    pub coefficient: Fr,
}

impl ParseJolt for OpeningCombiners {
    fn format_non_native(&self) -> serde_json::Value {
        json!({
            "bytecodecombiners": self.bytecode_combiners.format_non_native(),
            "instructionlookupcombiners": self.instruction_lookup_combiners.format_non_native(),
            "readwriteoutputtimestampcombiners": self.read_write_output_timestamp_combiners.format_non_native(),
            "spartancombiners": self.r1cs_combiners.format_non_native(),
            "coefficient": self.coefficient.format_non_native()
        })
    }
}
pub struct HyperKzgVerifierAdvice {
    pub r: Fr,
    pub d_0: Fr,
    pub v: Fr,
    pub q_power: Fr,
}
impl ParseJolt for HyperKzgVerifierAdvice {
    fn format_non_native(&self) -> serde_json::Value {
        json!({
            "r": self.r.format_non_native(),
            "d_0": self.d_0.format_non_native(),
            "v": self.v.format_non_native(),
            "q_power": self.q_power.format_non_native()
        })
    }
}

const NUM_MEMORIES: usize = 54;
const NUM_INSTRUCTIONS: usize = 26;
const MEMORY_OPS_PER_INSTRUCTION: usize = 4;
static chunks_x_size: usize = 4;
static chunks_y_size: usize = 4;
const NUM_CIRCUIT_FLAGS: usize = 11;
static relevant_y_chunks_len: usize = 4;

pub struct LinkingStuff1 {
    pub commitments: JoltStuff<HyperKZGCommitment<Bn254>>,
    pub opening_combiners: OpeningCombiners,
    pub hyper_kzg_verifier_advice: HyperKzgVerifierAdvice,
}

impl LinkingStuff1 {
    fn new(commitments: JoltStuff<HyperKZGCommitment<Bn254>>, witness: Vec<Fr>) -> LinkingStuff1 {
        let bytecode_stuff_size = 6 * 9;
        let read_write_memory_stuff_size = 6 * 13;
        let instruction_lookups_stuff_size = 6 * (C + 3 * NUM_MEMORIES + NUM_INSTRUCTIONS + 1);
        let timestamp_range_check_stuff_size = 6 * (4 * MEMORY_OPS_PER_INSTRUCTION);
        let aux_variable_stuff_size = 6 * (8 + relevant_y_chunks_len);
        let r1cs_stuff_size =
            6 * (chunks_x_size + chunks_y_size + NUM_CIRCUIT_FLAGS) + aux_variable_stuff_size;
        let jolt_stuff_size = bytecode_stuff_size
            + read_write_memory_stuff_size
            + instruction_lookups_stuff_size
            + timestamp_range_check_stuff_size
            + r1cs_stuff_size;

        let mut idx = 1 + jolt_stuff_size;
        let bytecode_combiners = BytecodeCombiners {
            rho: [witness[idx], witness[idx + 1]],
        };

        idx += 2;
        let instruction_lookup_combiners = InstructionLookupCombiners {
            rho: [witness[idx], witness[idx + 1], witness[idx + 2]],
        };

        idx += 3;
        let read_write_output_timestamp_combiners = ReadWriteOutputTimestampCombiners {
            rho: [
                witness[idx],
                witness[idx + 1],
                witness[idx + 2],
                witness[idx + 3],
            ],
        };

        idx += 4;
        let r1cs_combiners = R1CSCombiners { rho: witness[idx] };

        idx += 1;

        let opening_combiners = OpeningCombiners {
            bytecode_combiners,
            instruction_lookup_combiners,
            read_write_output_timestamp_combiners,
            r1cs_combiners,
            coefficient: witness[idx],
        };

        idx += 1;
        let hyper_kzg_verifier_advice = HyperKzgVerifierAdvice {
            r: witness[idx],
            d_0: witness[idx + 1],
            v: witness[idx + 2],
            q_power: witness[idx + 3],
        };

        LinkingStuff1 {
            commitments,
            opening_combiners,
            hyper_kzg_verifier_advice,
        }
    }
}
impl ParseJolt for LinkingStuff1 {
    fn format(&self) -> serde_json::Value {
        json!({
            "commitments": self.commitments.format(),
            "openingcombiners": self.opening_combiners.format_non_native(),
            "hyperkzgverifieradvice": self.hyper_kzg_verifier_advice.format_non_native()
        })
    }
    fn format_non_native(&self) -> serde_json::Value {
        json!({
            "commitments": self.commitments.format_non_native(),
            "openingcombiners": self.opening_combiners.format_non_native(),
            "hyperkzgverifieradvice": self.hyper_kzg_verifier_advice.format_non_native()
        })
    }
}

impl ParseJolt for PoseidonTranscript<Fr, Fr> {
    fn format(&self) -> serde_json::Value {
        json!({
            "state": self.state.state[1].to_string(),
            "nRounds": self.n_rounds.to_string(),
        })
    }
}

//Parse Jolt
impl ParseJolt
    for JoltProof<{ C }, { M }, JoltR1CSInputs, Fr, PCS, RV32I, RV32ISubtables<Fr>, ProofTranscript>
{
    fn format(&self) -> serde_json::Value {
        json!({
            "trace_length": self.trace_length.to_string(),
            "program_io": self.program_io.format(),
            "bytecode": self.bytecode.format(),
            "read_write_memory": self.read_write_memory.format(),
            "instruction_lookups": self.instruction_lookups.format(),
            "r1cs": self.r1cs.format(),
            "opening_proof":self.opening_proof.format()
        })
    }
}
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
                "panic": (self.panic as u8).to_string()
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
impl ParseJolt for ReadWriteMemoryProof<Fr, PCS, ProofTranscript> {
    fn format(&self) -> serde_json::Value {
        let mut openings = Vec::new();
        openings.push(self.memory_checking_proof.openings.a_ram.to_string());
        openings.push(self.memory_checking_proof.openings.v_read_rd.to_string());
        openings.push(self.memory_checking_proof.openings.v_read_rs1.to_string());
        openings.push(self.memory_checking_proof.openings.v_read_rs2.to_string());
        openings.push(self.memory_checking_proof.openings.v_read_ram.to_string());
        openings.push(self.memory_checking_proof.openings.v_write_rd.to_string());
        openings.push(self.memory_checking_proof.openings.v_write_ram.to_string());
        openings.push(self.memory_checking_proof.openings.v_final.to_string());
        openings.push(self.memory_checking_proof.openings.t_read_rd.to_string());
        openings.push(self.memory_checking_proof.openings.t_read_rs1.to_string());
        openings.push(self.memory_checking_proof.openings.t_read_rs2.to_string());
        openings.push(self.memory_checking_proof.openings.t_read_ram.to_string());
        openings.push(self.memory_checking_proof.openings.t_final.to_string());

        let mut exogenous_openings = Vec::new();
        exogenous_openings.push(
            self.memory_checking_proof
                .exogenous_openings
                .a_rd
                .to_string(),
        );
        exogenous_openings.push(
            self.memory_checking_proof
                .exogenous_openings
                .a_rs1
                .to_string(),
        );
        exogenous_openings.push(
            self.memory_checking_proof
                .exogenous_openings
                .a_rs2
                .to_string(),
        );

        json!({
            "memory_checking_proof": { "multiset_hashes": self.memory_checking_proof.multiset_hashes.format(),
                                       "read_write_grand_product" : self.memory_checking_proof.read_write_grand_product.format(),
                                       "init_final_grand_product" : self.memory_checking_proof.init_final_grand_product.format(),
                                       "openings": openings,
                                       "exogenous_openings": exogenous_openings
                            },
            "timestamp_validity_proof": self.timestamp_validity_proof.format(),
            "output_proof": self.output_proof.format()
        })
    }
}
impl ParseJolt
    for InstructionLookupsProof<{ C }, { M }, Fr, PCS, RV32I, RV32ISubtables<Fr>, ProofTranscript>
{
    fn format(&self) -> serde_json::Value {
        let openings: Vec<String> = self
            .memory_checking
            .openings
            .dim
            .iter()
            .chain(self.memory_checking.openings.read_cts.iter())
            .chain(self.memory_checking.openings.final_cts.iter())
            .chain(self.memory_checking.openings.E_polys.iter())
            .chain(self.memory_checking.openings.instruction_flags.iter())
            .chain([self.memory_checking.openings.lookup_outputs].iter())
            .map(|com| com.to_string())
            .collect();

        json!({
            "primary_sumcheck": self.primary_sumcheck.format(),
            "memory_checking_proof": { "multiset_hashes": self.memory_checking.multiset_hashes.format(),
                                       "read_write_grand_product" : self.memory_checking.read_write_grand_product.format(),
                                       "init_final_grand_product" : self.memory_checking.init_final_grand_product.format(),
                                       "openings": openings,
                            },
        })
    }
}
impl ParseJolt for UniformSpartanProof<{ C }, JoltR1CSInputs, Fr, ProofTranscript> {
    fn format(&self) -> serde_json::Value {
        json!({
            "outer_sumcheck_proof": self.outer_sumcheck_proof.format(),
            "inner_sumcheck_proof": self.inner_sumcheck_proof.format(),
            "outer_sumcheck_claims": [self.outer_sumcheck_claims.0.to_string(), self.outer_sumcheck_claims.1.to_string(), self.outer_sumcheck_claims.2.to_string()],
            "claimed_witness_evals": self.claimed_witness_evals.iter().map(|eval|eval.to_string()).collect::<Vec<String>>()
        })
    }
}
impl ParseJolt for ReducedOpeningProof<Fr, PCS, ProofTranscript> {
    fn format(&self) -> serde_json::Value {
        json!({
            "sumcheck_proof":self.sumcheck_proof.format(),
            "sumcheck_claims": self.sumcheck_claims.iter().map(|claim|claim.to_string()).collect::<Vec<String>>(),
            "joint_opening_proof": self.joint_opening_proof.format_non_native()
        })
    }
}
impl ParseJolt for PrimarySumcheck<Fr, ProofTranscript> {
    fn format(&self) -> serde_json::Value {
        json!({
            "sumcheck_proof": self.sumcheck_proof.format(),
            "openings":self.openings.format(),
        })
    }
}
impl ParseJolt for PrimarySumcheckOpenings<Fr> {
    fn format(&self) -> serde_json::Value {
        let E_poly_openings: Vec<String> = self
            .E_poly_openings
            .iter()
            .map(|opening| opening.to_string())
            .collect();
        let flag_openings: Vec<String> = self
            .flag_openings
            .iter()
            .map(|opening| opening.to_string())
            .collect();
        json!({
            "E_poly_openings": E_poly_openings,
            "flag_openings": flag_openings,
            "lookup_outputs_opening": self.lookup_outputs_opening.to_string(),
        })
    }
}
impl ParseJolt for MultisetHashes<Fr> {
    fn format(&self) -> serde_json::Value {
        let read_hashes: Vec<String> = self
            .read_hashes
            .iter()
            .map(|hash| hash.to_string())
            .collect();
        let write_hashes: Vec<String> = self
            .write_hashes
            .iter()
            .map(|hash| hash.to_string())
            .collect();
        let init_hashes: Vec<String> = self
            .init_hashes
            .iter()
            .map(|hash| hash.to_string())
            .collect();
        let final_hashes: Vec<String> = self
            .final_hashes
            .iter()
            .map(|hash| hash.to_string())
            .collect();

        json!({
            "read_hashes": read_hashes,
            "write_hashes": write_hashes,
            "init_hashes": init_hashes,
            "final_hashes": final_hashes
        })
    }
}
impl ParseJolt for TimestampValidityProof<Fr, PCS, ProofTranscript> {
    fn format(&self) -> serde_json::Value {
        json!({
            "multiset_hashes": self.multiset_hashes.format(),
            "openings": self.openings.format(),
            "exogenous_openings": self.exogenous_openings.iter().map(|opening|opening.to_string()).collect::<Vec<String>>(),
            "batched_grand_product": self.batched_grand_product.format()
        })
    }
}

impl ParseJolt for OutputSumcheckProof<Fr, PCS, ProofTranscript> {
    fn format(&self) -> serde_json::Value {
        json!({
            "sumcheck_proof": self.sumcheck_proof.format(),
            "opening": self.opening.to_string()
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
    fn format_non_native(&self) -> serde_json::Value {
        json!({
            "bytecode": self.bytecode.format_non_native(),
            "read_write_memory": self.read_write_memory.format_non_native(),
            "instruction_lookups": self.instruction_lookups.format_non_native(),
            "timestamp_range_check": self.timestamp_range_check.format_non_native(),
            "r1cs": self.r1cs.format_non_native()
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
            "right_claim": self.right_claim.to_string(),
        })
    }
    fn format_non_native(&self) -> serde_json::Value {
        json!({
            "proof": self.proof.format_non_native(),
            "left_claim": self.left_claim.format(),
            "right_claim": self.right_claim.format(),
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
    fn format_non_native(&self) -> serde_json::Value {
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
    fn format_non_native(&self) -> serde_json::Value {
        let coeffs: Vec<serde_json::Value> = self
            .coeffs
            .iter()
            .map(|coeff| coeff.format_non_native())
            .collect();
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

    fn format_non_native(&self) -> serde_json::Value {
        let v_read_write: Vec<serde_json::Value> = self
            .v_read_write
            .iter()
            .map(|v| v.format_non_native())
            .collect();
        json!({
            "a_read_write": self.a_read_write.format_non_native(),
            "v_read_write": v_read_write,
            "t_read": self.t_read.format_non_native(),
            "t_final": self.t_final.format_non_native()
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
    fn format_non_native(&self) -> serde_json::Value {
        json!({
            "a_ram": self.a_ram.format_non_native(),
                "v_read_rd": self.v_read_rd.format_non_native(),
                "v_read_rs1": self.v_read_rs1.format_non_native(),
                "v_read_rs2": self.v_read_rs2.format_non_native(),
                "v_read_ram": self.v_read_ram.format_non_native(),
                "v_write_rd": self.v_write_rd.format_non_native(),
                "v_write_ram": self.v_write_ram.format_non_native(),
                "v_final": self.v_final.format_non_native(),
                "t_read_rd": self.t_read_rd.format_non_native(),
                "t_read_rs1": self.t_read_rs1.format_non_native(),
                "t_read_rs2": self.t_read_rs2.format_non_native(),
                "t_read_ram": self.t_read_ram.format_non_native(),
                "t_final": self.t_final.format_non_native()
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

    fn format_non_native(&self) -> serde_json::Value {
        let dim: Vec<serde_json::Value> =
            self.dim.iter().map(|com| com.format_non_native()).collect();
        let read_cts: Vec<serde_json::Value> = self
            .read_cts
            .iter()
            .map(|com| com.format_non_native())
            .collect();
        let final_cts: Vec<serde_json::Value> = self
            .final_cts
            .iter()
            .map(|com| com.format_non_native())
            .collect();
        let E_polys: Vec<serde_json::Value> = self
            .E_polys
            .iter()
            .map(|com| com.format_non_native())
            .collect();
        let instruction_flags: Vec<serde_json::Value> = self
            .instruction_flags
            .iter()
            .map(|com| com.format_non_native())
            .collect();
        json!({
            "dim": dim,
            "read_cts": read_cts,
            "final_cts": final_cts,
            "E_polys": E_polys,
            "instruction_flags": instruction_flags,
            "lookup_outputs": self.lookup_outputs.format_non_native()
        })
    }
}

// impl ParseJolt for InstructionLookupStuff<Fr> {
//     fn format(&self) -> serde_json::Value {
//         let dim: Vec<String> = self.dim.iter().map(|com| com.to_string()).collect();
//         let read_cts: Vec<String> = self.read_cts.iter().map(|com| com.to_string()).collect();
//         let final_cts: Vec<String> = self.final_cts.iter().map(|com| com.to_string()).collect();
//         let E_polys: Vec<String> = self.E_polys.iter().map(|com| com.to_string()).collect();
//         let instruction_flags: Vec<String> = self
//             .instruction_flags
//             .iter()
//             .map(|com| com.to_string())
//             .collect();
//         json!({
//             "dim": dim,
//             "read_cts": read_cts,
//             "final_cts": final_cts,
//             "E_polys": E_polys,
//             "instruction_flags": instruction_flags,
//             "lookup_outputs": self.lookup_outputs.to_string()
//         })
//     }
// }
impl ParseJolt for TimestampRangeCheckStuff<Fr> {
    fn format(&self) -> serde_json::Value {
        let read_cts_read_timestamp: Vec<String> = self
            .read_cts_read_timestamp
            .iter()
            .map(|com| com.to_string())
            .collect();
        let read_cts_global_minus_read: Vec<String> = self
            .read_cts_global_minus_read
            .iter()
            .map(|com| com.to_string())
            .collect();
        let final_cts_read_timestamp: Vec<String> = self
            .final_cts_read_timestamp
            .iter()
            .map(|com| com.to_string())
            .collect();
        let final_cts_global_minus_read: Vec<String> = self
            .final_cts_global_minus_read
            .iter()
            .map(|com| com.to_string())
            .collect();
        json!({
             "read_cts_read_timestamp": read_cts_read_timestamp,
                "read_cts_global_minus_read":read_cts_global_minus_read,
                "final_cts_read_timestamp": final_cts_read_timestamp,
                "final_cts_global_minus_read": final_cts_global_minus_read
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
    fn format_non_native(&self) -> serde_json::Value {
        let read_cts_read_timestamp: Vec<serde_json::Value> = self
            .read_cts_read_timestamp
            .iter()
            .map(|com| com.format_non_native())
            .collect();
        let read_cts_global_minus_read: Vec<serde_json::Value> = self
            .read_cts_global_minus_read
            .iter()
            .map(|com| com.format_non_native())
            .collect();
        let final_cts_read_timestamp: Vec<serde_json::Value> = self
            .final_cts_read_timestamp
            .iter()
            .map(|com| com.format_non_native())
            .collect();
        let final_cts_global_minus_read: Vec<serde_json::Value> = self
            .final_cts_global_minus_read
            .iter()
            .map(|com| com.format_non_native())
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
    fn format_non_native(&self) -> serde_json::Value {
        let chunks_x: Vec<serde_json::Value> = self
            .chunks_x
            .iter()
            .map(|com| com.format_non_native())
            .collect();
        let chunks_y: Vec<serde_json::Value> = self
            .chunks_y
            .iter()
            .map(|com| com.format_non_native())
            .collect();
        let circuit_flags: Vec<serde_json::Value> = self
            .circuit_flags
            .iter()
            .map(|com| com.format_non_native())
            .collect();
        json!({
            "chunks_x": chunks_x,
            "chunks_y": chunks_y,
            "circuit_flags": circuit_flags,
            "aux": self.aux.format_non_native()
        })
    }
}
impl ParseJolt for AuxVariableStuff<HyperKZGCommitment<Bn254>> {
    fn format_non_native(&self) -> serde_json::Value {
        let relevant_y_chunks: Vec<serde_json::Value> = self
            .relevant_y_chunks
            .iter()
            .map(|com| com.format_non_native())
            .collect();
        json!({
                "left_lookup_operand": self.left_lookup_operand.format_non_native(),
                "right_lookup_operand":  self.right_lookup_operand.format_non_native(),
                "product": self.product.format_non_native(),
                "relevant_y_chunks": relevant_y_chunks,
                "write_lookup_output_to_rd": self.write_lookup_output_to_rd.format_non_native(),
                "write_pc_to_rd":  self.write_pc_to_rd.format_non_native(),
                "next_pc_jump":  self.next_pc_jump.format_non_native(),
                "should_branch":  self.should_branch.format_non_native(),
                "next_pc": self.next_pc.format_non_native(),
        })
    }
}

impl ParseJolt for JoltPreprocessing<{ C }, Fr, PCS, ProofTranscript> {
    fn format(&self) -> serde_json::Value {
        let v_init_final: Vec<Vec<String>> = self
            .bytecode
            .v_init_final
            .iter()
            .map(|poly| poly.Z.iter().map(|elem| elem.to_string()).collect())
            .collect();
        json!({
           "bytecode" : {
                        "v_init_final": v_init_final
                        },
           "read_write_memory": {"bytecode_words": self.read_write_memory.bytecode_words.iter().map(|elem|elem.to_string()).collect::<Vec<String>>() }
        })
    }
}
//Parse for Jolt2

#[test]
fn fib_e2e_hyperkzg() {
    println!("Running Fib");

    let (preprocessing, proof, commitments) = fib_e2e::<Fr, PCS, ProofTranscript>();

    let jolt1_input = json!(
    {
        "preprocessing": {
            "v_init_final_hash": preprocessing.bytecode.v_init_final_hash.to_string(),
            "bytecode_words_hash": preprocessing.read_write_memory.hash.to_string()
        },
        "proof": proof.format(),
        "commitments":commitments.format_non_native(),
        "pi_proof":preprocessing.format()
    });

    // Convert the JSON to a pretty-printed string
    let pretty_json = serde_json::to_string_pretty(&jolt1_input).expect("Failed to serialize JSON");

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(pretty_json.as_bytes())
        .expect("Failed to write to input.json");
}
