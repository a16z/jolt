#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
use std::path::PathBuf;

use super::{jolt::Fr, Parse};
use crate::{
    jolt::vm::JoltStuff,
    parse::{generate_r1cs, spartan2::spartan_hyrax, write_json},
    poly::commitment::{
        commitment_scheme::CommitmentScheme,
        hyperkzg::{HyperKZG, HyperKZGCommitment},
    },
    spartan::spartan_memory_checking::{SpartanPreprocessing, SpartanProof},
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{poseidon_transcript::PoseidonTranscript, thread::drop_in_background_thread},
};
use ark_bn254::Bn254;
use serde_json::json;

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
            "bytecode_combiners": self.bytecode_combiners.format_non_native(),
            "instruction_lookup_combiners": self.instruction_lookup_combiners.format_non_native(),
            "read_write_output_timestamp_combiners": self.read_write_output_timestamp_combiners.format_non_native(),
            "spartan_combiners": self.r1cs_combiners.format_non_native(),
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
        jolt_stuff_size: usize,
        witness: &Vec<Fr>,
    ) -> LinkingStuff1 {
        let mut idx = 2 + jolt_stuff_size;
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
            "opening_combiners": self.opening_combiners.format_non_native(),
            "hyperkzg_verifier_advice": self.hyper_kzg_verifier_advice.format_non_native()
        })
    }
    fn format_non_native(&self) -> serde_json::Value {
        json!({
            "commitments": self.commitments.format_non_native(),
            "opening_combiners": self.opening_combiners.format_non_native(),
            "hyperkzg_verifier_advice": self.hyper_kzg_verifier_advice.format_non_native()
        })
    }
}

pub(crate) fn spartan_hkzg(
    jolt_pi: serde_json::Value,
    linking_stuff_1: serde_json::Value,
    linking_stuff_2: serde_json::Value,
    vk_jolt_2: serde_json::Value,
    vk_jolt_2_nn: serde_json::Value,
    hyperkzg_proof: serde_json::Value,
    pub_io_len: usize,
    jolt_stuff_size: usize,
    jolt_openining_point_len: usize,
    file_paths: &Vec<PathBuf>,
    z: &Vec<Fr>,
    packages: &[&str],
    output_dir: &str,
) {
    type Fr = ark_bn254::Fr;
    type ProofTranscript = PoseidonTranscript<ark_bn254::Fr, ark_bn254::Fq>;
    type Pcs = HyperKZG<ark_bn254::Bn254, ProofTranscript>;

    //Parse Spartan
    impl Parse for SpartanProof<Fr, Pcs, ProofTranscript> {
        fn format(&self) -> serde_json::Value {
            json!({
                "outer_sumcheck_proof": self.outer_sumcheck_proof.format_non_native(),
                "inner_sumcheck_proof": self.inner_sumcheck_proof.format_non_native(),
                "outer_sumcheck_claims": [self.outer_sumcheck_claims.0.format_non_native(),self.outer_sumcheck_claims.1.format_non_native(),self.outer_sumcheck_claims.2.format_non_native()],
                "inner_sumcheck_claims": [self.inner_sumcheck_claims.0.format_non_native(),self.inner_sumcheck_claims.1.format_non_native(),self.inner_sumcheck_claims.2.format_non_native(),self.inner_sumcheck_claims.3.format_non_native()],
                "pub_io_eval": self.pi_eval.format_non_native(),
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

    let circom_template = "Combine";
    let prime = "grumpkin";

    let constraint_path = format!("{}/{}_constraints.json", output_dir, packages[0]).to_string();
    let constraint_path = Some(&constraint_path);

    let preprocessing =
        SpartanPreprocessing::<Fr>::preprocess(constraint_path, Some(z), pub_io_len - 1);

    let commitment_shapes =
        SpartanProof::<Fr, Pcs, ProofTranscript>::commitment_shapes(preprocessing.vars.len());

    let pcs_setup = Pcs::setup(&commitment_shapes);

    let proof = SpartanProof::<Fr, Pcs, ProofTranscript>::prove(&pcs_setup, &preprocessing);

    SpartanProof::<Fr, Pcs, ProofTranscript>::verify(&pcs_setup, &preprocessing, &proof).unwrap();

    let digest = preprocessing.inst.get_digest().format_non_native();
    let combine_input = json!({
        "jolt_pi": jolt_pi,
        "counter_jolt_1": 1.to_string(),
        "linking_stuff_1": linking_stuff_1,
        "digest":digest,
        "vk_spartan_1": pcs_setup.1.format(),
        "spartan_proof": proof.format(),
        "w_commitment": proof.witness_commit.format(),
        "linking_stuff_2": linking_stuff_2,
        "vk_jolt_2": vk_jolt_2,
        "hyperkzg_proof": hyperkzg_proof
    });

    write_json(&combine_input, output_dir, packages[1]);
    drop_in_background_thread(combine_input);
    drop_in_background_thread(preprocessing);

    let inner_num_rounds = proof.inner_sumcheck_proof.uni_polys.len();

    // Length of public IO of Combined R1CS including the 1 at index 0.
    // 1 + counter_combined_r1cs (1) + postponed eval size (point size = (inner num rounds - 1) * 3, eval size  = 3) +
    // counter_jolt_1 (1) + linking stuff (nn) size (jolt stuff size + 15 * 3) + jolt pi size (2 * 3)
    // + digest size (3) + 2 hyper kzg verifier keys (2 + 4 + 4).

    let pub_io_len_combine_r1cs =
        1 + 1 + (inner_num_rounds - 1) * 3 + 3 + 1 + jolt_stuff_size + 15 * 3 + 2 * 3 + 3 + 10 + 10;
    let postponed_point_len = inner_num_rounds - 1;

    let combined_r1cs_params = [
        proof.outer_sumcheck_proof.uni_polys.len(),
        proof.inner_sumcheck_proof.uni_polys.len(),
        proof.pcs_proof.com.len() + 1,
        jolt_openining_point_len,
    ]
    .to_vec();

    drop_in_background_thread(proof);

    generate_r1cs(
        &file_paths[1],
        output_dir,
        circom_template,
        combined_r1cs_params,
        prime,
    );

    spartan_hyrax(
        linking_stuff_1,
        jolt_pi,
        digest,
        pcs_setup.1.format_non_native(),
        vk_jolt_2_nn,
        pub_io_len_combine_r1cs,
        postponed_point_len,
        file_paths,
        packages,
        output_dir,
    );
}
