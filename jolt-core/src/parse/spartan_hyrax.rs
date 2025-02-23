use super::*;
use crate::field::JoltField;
use crate::parse::jolt::to_limbs;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::unipoly::UniPoly;
use crate::spartan::spartan_memory_checking::{SpartanPreprocessing, SpartanProof};
use crate::subprotocols::sumcheck::SumcheckInstanceProof;
use crate::{poly::commitment::hyrax::HyraxScheme, utils::poseidon_transcript::PoseidonTranscript};
use ark_ff::{BigInt, BigInteger, PrimeField};
use itertools::Itertools;
use num_bigint::BigUint;
use serde_json::json;
use std::fs::File;
use std::io::Write;

type Fr = ark_grumpkin::Fr;
type Fq = ark_grumpkin::Fq;
type ProofTranscript = PoseidonTranscript<Fr, ark_grumpkin::Fq>;
type PCS = HyraxScheme<ark_grumpkin::Projective, ProofTranscript>;

pub fn from_limbs(limbs: Vec<Fr>) -> Fq {
    assert_eq!(limbs.len(), 3);
    let bits = limbs[0]
        .into_bigint()
        .to_bits_le()
        .iter()
        .take(125)
        .chain(limbs[1].into_bigint().to_bits_le().iter().take(125))
        .chain(limbs[2].into_bigint().to_bits_le().iter().take(4))
        .cloned()
        .collect_vec();
    Fq::from(BigInt::from_bits_le(&bits))
}

impl Parse for Fr {
    fn format(&self) -> serde_json::Value {
        let limbs = to_limbs::<Fr, Fq>(*self);
        json!({
            "limbs": [limbs[0].to_string(), limbs[1].to_string(), limbs[2].to_string()]
        })
    }
}

struct PostponedEval {
    pub point: Vec<Fq>,
    pub eval: Fq,
}

impl PostponedEval {
    pub fn new(witness: Vec<Fr>, postponed_eval_size: usize) -> Self {
        let point = (0..postponed_eval_size)
            .into_iter()
            .map(|i| from_limbs(witness[1 + 3 * i..1 + 3 * i + 3].to_vec()))
            .collect();
        let eval = from_limbs(
            witness[1 + 3 * postponed_eval_size..1 + 3 * postponed_eval_size + 3].to_vec(),
        );

        Self { point, eval }
    }
}
impl Parse for PostponedEval {
    fn format(&self) -> serde_json::Value {
        let point: Vec<serde_json::Value> = self.point.iter().map(|elem| elem.format()).collect();
        json!({
            "point": point,
            "y": self.eval.format()
        })
    }
}

impl Parse for SpartanProof<Fr, PCS, ProofTranscript> {
    fn format(&self) -> serde_json::Value {
        json!({
            "outer_sumcheck_proof": self.outer_sumcheck_proof.format(),
            "inner_sumcheck_proof": self.inner_sumcheck_proof.format(),
            "outer_sumcheck_claims": [self.outer_sumcheck_claims.0.format(),self.outer_sumcheck_claims.1.format(),self.outer_sumcheck_claims.2.format()],
            "inner_sumcheck_claims": [self.inner_sumcheck_claims.0.format(),self.inner_sumcheck_claims.1.format(),self.inner_sumcheck_claims.2.format(),self.inner_sumcheck_claims.3.format()],
            "pub_io_eval": self.pi_eval.format(),
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
}

impl Parse for UniPoly<Fr> {
    fn format(&self) -> serde_json::Value {
        let coeffs: Vec<serde_json::Value> =
            self.coeffs.iter().map(|coeff| coeff.format()).collect();
        json!({
            "coeffs": coeffs,
        })
    }
}
pub(crate) fn spartan_hyrax(
    linking_stuff: serde_json::Value,
    jolt_pi: serde_json::Value,
    hyperkzg_vk: serde_json::Value,
    jolt_vk: serde_json::Value,
) {
    let constraint_path = Some("src/spartan/verifier_constraints.json");
    let witness_path = Some("src/spartan/witness.json");
    let file =
        File::open(witness_path.expect("Path doesn't exist")).expect("Witness file not found");
    let reader = std::io::BufReader::new(file);
    let witness: Vec<String> = serde_json::from_reader(reader).unwrap();

    let mut z = Vec::new();
    for value in witness {
        let val: BigUint = value.parse().unwrap();
        let mut bytes = val.to_bytes_le();
        bytes.resize(32, 0u8);
        let val = Fr::from_bytes(&bytes);
        z.push(val);
    }

    let preprocessing = SpartanPreprocessing::<Fr>::preprocess(constraint_path, Some(&z), 9);
    let commitment_shapes = SpartanProof::<Fr, PCS, ProofTranscript>::commitment_shapes(
        preprocessing.inputs.len() + preprocessing.vars.len(),
    );

    let pcs_setup = PCS::setup(&commitment_shapes);
    let proof: SpartanProof<Fr, PCS, ProofTranscript> =
        SpartanProof::<Fr, PCS, ProofTranscript>::prove(&pcs_setup, &preprocessing);

    SpartanProof::<Fr, PCS, ProofTranscript>::verify(&pcs_setup, &preprocessing, &proof).unwrap();

    // TODO: Read witness.json file and put the first half into witness.

    let to_eval = PostponedEval::new(z, POSTPONED_POINT_LEN);

    let input_json = json!({
        "public_io": {
                "jolt_pi": jolt_pi,
                "linking_stuff": linking_stuff,
                "vk1": jolt_vk,
                "vk2": hyperkzg_vk,
            },
        "to_eval": to_eval.format(),
        "setup": pcs_setup.format_setup(proof.pcs_proof.vector_matrix_product.len()),
        "proof": proof.format(),
        "w_commitment": proof.witness_commit.format(),
    });

    // Convert the JSON to a pretty-printed string
    let pretty_json = serde_json::to_string_pretty(&input_json).expect("Failed to serialize JSON");

    let input_file_path = "input.json";
    let mut input_file = File::create(input_file_path).expect("Failed to create input.json");
    input_file
        .write_all(pretty_json.as_bytes())
        .expect("Failed to write to input.json");
}
