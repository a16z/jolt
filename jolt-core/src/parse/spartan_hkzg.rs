use super::{
    jolt::Fr, Parse, CHUNKS_X_SIZE, CHUNKS_Y_SIZE, MEMORY_OPS_PER_INSTRUCTION, NUM_CIRCUIT_FLAGS,
    NUM_INSTRUCTIONS, NUM_MEMORIES, RELEVANT_Y_CHUNKS_LEN,
};
use crate::{
    jolt::vm::{rv32i_vm::C, JoltStuff},
    parse::spartan_hyrax::spartan_hyrax,
    poly::commitment::{
        commitment_scheme::CommitmentScheme,
        hyperkzg::{HyperKZG, HyperKZGCommitment},
    },
    spartan::spartan_memory_checking::{SpartanPreprocessing, SpartanProof},
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::poseidon_transcript::PoseidonTranscript,
};
use ark_bn254::Bn254;
use serde_json::json;
use std::{fs::File, io::Write};

pub struct InstructionLookupCombiners {
    pub rho: [Fr; 3],
}

impl Parse for InstructionLookupCombiners {
    fn format_non_native(&self) -> serde_json::Value {
        json!({
            "rho": [self.rho[0].format_non_native(), self.rho[1].format_non_native(), self.rho[2].format_non_native()]
        })
    }
}
pub struct ReadWriteOutputTimestampCombiners {
    pub rho: [Fr; 4],
}
impl Parse for ReadWriteOutputTimestampCombiners {
    fn format_non_native(&self) -> serde_json::Value {
        json!({
            "rho": [self.rho[0].format_non_native(), self.rho[1].format_non_native(), self.rho[2].format_non_native(), self.rho[3].format_non_native()]
        })
    }
}
pub struct R1CSCombiners {
    pub rho: Fr,
}
impl Parse for R1CSCombiners {
    fn format_non_native(&self) -> serde_json::Value {
        json!({
            "rho": self.rho.format_non_native()
        })
    }
}
pub struct BytecodeCombiners {
    pub rho: [Fr; 2],
}
impl Parse for BytecodeCombiners {
    fn format_non_native(&self) -> serde_json::Value {
        json!({
            "rho": [self.rho[0].format_non_native(), self.rho[1].format_non_native()]
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

impl Parse for OpeningCombiners {
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
impl Parse for HyperKzgVerifierAdvice {
    fn format_non_native(&self) -> serde_json::Value {
        json!({
            "r": self.r.format_non_native(),
            "d_0": self.d_0.format_non_native(),
            "v": self.v.format_non_native(),
            "q_power": self.q_power.format_non_native()
        })
    }
}

pub struct LinkingStuff1 {
    pub commitments: JoltStuff<HyperKZGCommitment<Bn254>>,
    pub opening_combiners: OpeningCombiners,
    pub hyper_kzg_verifier_advice: HyperKzgVerifierAdvice,
}

impl LinkingStuff1 {
    pub fn new(
        commitments: JoltStuff<HyperKZGCommitment<Bn254>>,
        witness: Vec<Fr>,
    ) -> LinkingStuff1 {
        let bytecode_stuff_size = 6 * 9;
        let read_write_memory_stuff_size = 6 * 13;
        let instruction_lookups_stuff_size = 6 * (C + 3 * NUM_MEMORIES + NUM_INSTRUCTIONS + 1);
        let timestamp_range_check_stuff_size = 6 * (4 * MEMORY_OPS_PER_INSTRUCTION);
        let aux_variable_stuff_size = 6 * (8 + RELEVANT_Y_CHUNKS_LEN);
        let r1cs_stuff_size =
            6 * (CHUNKS_X_SIZE + CHUNKS_Y_SIZE + NUM_CIRCUIT_FLAGS) + aux_variable_stuff_size;
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

impl Parse for LinkingStuff1 {
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

pub(crate) fn spartan_hkzg(
    linking_stuff: serde_json::Value,
    jolt_pi: serde_json::Value,
    jolt2_input: serde_json::Value,
    jolt_vk: serde_json::Value,
) {
    type Fr = ark_bn254::Fr;
    type ProofTranscript = PoseidonTranscript<ark_bn254::Fr, ark_bn254::Fq>;
    type PCS = HyperKZG<ark_bn254::Bn254, ProofTranscript>;

    //Parse Spartan
    impl Parse for SpartanProof<Fr, PCS, ProofTranscript> {
        fn format(&self) -> serde_json::Value {
            json!({
                "outer_sumcheck_proof": self.outer_sumcheck_proof.format_non_native(),
                "inner_sumcheck_proof": self.inner_sumcheck_proof.format_non_native(),
                "outer_sumcheck_claims": [self.outer_sumcheck_claims.0.format_non_native(),self.outer_sumcheck_claims.1.format_non_native(),self.outer_sumcheck_claims.2.format_non_native()],
                "inner_sumcheck_claims": [self.inner_sumcheck_claims.0.format_non_native(),self.inner_sumcheck_claims.1.format_non_native(),self.inner_sumcheck_claims.2.format_non_native(),self.inner_sumcheck_claims.3.format_non_native()],
                "pi_eval": self.pi_eval.format_non_native(),
                "joint_opening_proof": self.pcs_proof.format()
            })
        }
    }
    impl Parse for SumcheckInstanceProof<Fr, ProofTranscript> {
        fn format(&self) -> serde_json::Value {
            let uni_polys: Vec<serde_json::Value> =
                self.uni_polys.iter().map(|poly| poly.format()).collect();
            json!({
                "uni_polys": uni_polys,
            })
        }
        fn format_non_native(&self) -> serde_json::Value {
            let uni_polys: Vec<serde_json::Value> = self
                .uni_polys
                .iter()
                .map(|poly| poly.format_non_native())
                .collect();
            json!({
                "uni_polys": uni_polys,
            })
        }
    }

    //TODO(Ashish):- Add code to generate jolt1_constraints

    let constraint_path = Some("src/spartan/jolt1_constraints.json");
    let witness_path = Some("src/spartan/witness.json");

    let preprocessing = SpartanPreprocessing::<Fr>::preprocess(constraint_path, witness_path, 9);
    let commitment_shapes = SpartanProof::<Fr, PCS, ProofTranscript>::commitment_shapes(
        preprocessing.inputs.len() + preprocessing.vars.len(),
    );
    let pcs_setup = PCS::setup(&commitment_shapes);
    let proof = SpartanProof::<Fr, PCS, ProofTranscript>::prove(&pcs_setup, &preprocessing);
    SpartanProof::<Fr, PCS, ProofTranscript>::verify(&pcs_setup, &preprocessing, &proof).unwrap();
    let hyperkzg_vk = pcs_setup.1.format();

    let spartan1_input = json!({
        "jolt_pi": jolt_pi,
        "linking_stuff": linking_stuff,
        "vk": hyperkzg_vk,
        "proof": proof.format(),
        "w_commitment": proof.witness_commit.format(),
    });

    let input_file_path = "spartan1_input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    let pretty_json =
        serde_json::to_string_pretty(&spartan1_input).expect("Failed to serialize JSON");
    input_file
        .write_all(pretty_json.as_bytes())
        .expect("Failed to write to input.json");

    let combine_input = json!({
        "jolt2": jolt2_input,
        "spartan1": spartan1_input
    });

    let input_file_path = "combine_input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    let pretty_json =
        serde_json::to_string_pretty(&combine_input).expect("Failed to serialize JSON");
    input_file
        .write_all(pretty_json.as_bytes())
        .expect("Failed to write to input.json");

    spartan_hyrax(linking_stuff, jolt_pi, hyperkzg_vk, jolt_vk);
}
