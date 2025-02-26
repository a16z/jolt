use super::*;
use crate::parse::jolt::to_limbs;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::unipoly::UniPoly;
use crate::spartan::spartan_memory_checking::{SpartanPreprocessing, SpartanProof};
use crate::subprotocols::sumcheck::SumcheckInstanceProof;
use crate::utils::math::Math;
use crate::{poly::commitment::hyrax::HyraxScheme, utils::poseidon_transcript::PoseidonTranscript};
use ark_ff::{BigInt, BigInteger, PrimeField};
use itertools::Itertools;
use serde_json::json;

type Fr = ark_grumpkin::Fr;
type Fq = ark_grumpkin::Fq;
type ProofTranscript = PoseidonTranscript<Fr, ark_grumpkin::Fq>;
type Pcs = HyraxScheme<ark_grumpkin::Projective, ProofTranscript>;

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
        let point: Vec<serde_json::Value> = self
            .point
            .iter()
            .map(|elem| elem.format_non_native())
            .collect();
        json!({
            "point": point,
            "eval": self.eval.format_non_native()
        })
    }
}

impl Parse for SpartanProof<Fr, Pcs, ProofTranscript> {
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
    spartan_1_digest: serde_json::Value,
    vk_spartan_1: serde_json::Value,
    vk_jolt_2: serde_json::Value,
    pub_io_len: usize,
    postponed_point_len: usize,
    file_paths: &Vec<PathBuf>,
    packages: &[&str],
    output_dir: &str,
) {
    let circom_template = "VerifySpartan";
    let prime = "bn128";

    let witness_file_path = format!("{}/{}_witness.json", output_dir, packages[1]).to_string();

    let z = read_witness::<Fr>(&witness_file_path);

    let constraint_path = format!("{}/{}_constraints.json", output_dir, packages[1]).to_string();

    let preprocessing =
        SpartanPreprocessing::<Fr>::preprocess(Some(&constraint_path), Some(&z), pub_io_len);

    let commitment_shapes =
        SpartanProof::<Fr, Pcs, ProofTranscript>::commitment_shapes(preprocessing.vars.len());

    let pcs_setup = Pcs::setup(&commitment_shapes);
    let proof: SpartanProof<Fr, Pcs, ProofTranscript> =
        SpartanProof::<Fr, Pcs, ProofTranscript>::prove(&pcs_setup, &preprocessing);

    SpartanProof::<Fr, Pcs, ProofTranscript>::verify(&pcs_setup, &preprocessing, &proof).unwrap();

    let to_eval = PostponedEval::new(z, postponed_point_len);

    let spartan_hyrax_input = json!({
        "pub_io": {
                "jolt_pi": jolt_pi,
                "counter_jolt_1": 1.to_string(),
                "linking_stuff": linking_stuff,
                "vk_spartan_1": vk_spartan_1,
                "digest": spartan_1_digest,
                "vk_jolt_2": vk_jolt_2,
            },
        "counter_combined_r1cs": 2.to_string(),
        "to_eval": to_eval.format(),
        "setup": pcs_setup.format_setup(proof.pcs_proof.vector_matrix_product.len()),
        "digest": preprocessing.inst.get_digest().format(),
        "proof": proof.format(),
        "w_commitment": proof.witness_commit.format(),
    });

    write_json(&spartan_hyrax_input, output_dir, packages[2]);

    let spartan_hyrax_args = [
        proof.outer_sumcheck_proof.uni_polys.len(),
        proof.inner_sumcheck_proof.uni_polys.len(),
        proof.pcs_proof.vector_matrix_product.len().log_2() + 1,
        postponed_point_len,
    ]
    .to_vec();

    generate_r1cs(
        &file_paths[2],
        output_dir,
        circom_template,
        spartan_hyrax_args,
        prime,
    );

    let witness_file_path = format!("{}/{}_witness.json", output_dir, packages[2]).to_string();

    let z = read_witness::<Fr>(&witness_file_path);

    let constraint_path = format!("{}/{}_constraints.json", output_dir, packages[2]).to_string();

    let _ = SpartanPreprocessing::<Fr>::preprocess(Some(&constraint_path), Some(&z), pub_io_len);
}
