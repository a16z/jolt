use ark_bn254::Bn254;
use ark_ff::BigInteger;
use ark_ff::PrimeField;

use crate::field::JoltField;
use crate::poly::commitment::hyperkzg::{HyperKZG, HyperKZGProof, HyperKZGVerifierKey};
use crate::r1cs::inputs::JoltR1CSInputs;
use crate::r1cs::spartan::UniformSpartanProof;
use crate::subprotocols::grand_product::BatchedGrandProductLayerProof;
use crate::subprotocols::grand_product::BatchedGrandProductProof;
use crate::subprotocols::sumcheck::SumcheckInstanceProof;
use crate::utils::transcript::Transcript;
use alloy_primitives::U256;
use alloy_sol_types::sol;
use ark_bn254::FrConfig;
use ark_ff::Fp;
use ark_ff::MontBackend;

sol!(struct HyperKZGProofSol {
    uint256[] com; // G1 points represented pairwise
    uint256[] w; // G1 points represented pairwise
    uint256[] v_ypos; // Three vectors of scalars which must be ell length
    uint256[] v_yneg;
    uint256[] v_y;
});

sol!(struct VK {
    uint256 VK_g1_x;
    uint256 VK_g1_y;
    uint256[] VK_g2;
    uint256[] VK_beta_g2;
});

sol!(
    struct SumcheckProof {
        uint256[][] compressedPolys;
    }
);

sol!(
    struct SpartanProof {
        SumcheckProof outer;
        uint256 outerClaimA;
        uint256 outerClaimB;
        uint256 outerClaimC;
        SumcheckProof inner;
        uint256[] claimedEvals;
    }
);

sol!(
    struct GKRLayer {
        SumcheckProof sumcheck;
        uint256 leftClaim;
        uint256 rightClaim;
    }
);

sol!(struct GrandProductProof {
        GKRLayer[] layers;
    }
);

impl Into<HyperKZGProofSol> for &HyperKZGProof<Bn254> {
    fn into(self) -> HyperKZGProofSol {
        let mut com = vec![];
        let mut w = vec![];
        let ypos_scalar = self.v[0].clone();
        let yneg_scalar = self.v[1].clone();
        let y_scalar = self.v[2].clone();

        // Horrible type conversion here, possibly theres an easier way
        let v_ypos = ypos_scalar
            .iter()
            .map(|i| U256::from_be_slice(i.into_bigint().to_bytes_be().as_slice()))
            .collect();
        let v_yneg = yneg_scalar
            .iter()
            .map(|i| U256::from_be_slice(i.into_bigint().to_bytes_be().as_slice()))
            .collect();
        let v_y = y_scalar
            .iter()
            .map(|i| U256::from_be_slice(i.into_bigint().to_bytes_be().as_slice()))
            .collect();

        for point in self.com.iter() {
            com.push(U256::from_be_slice(&point.x.into_bigint().to_bytes_be()));
            com.push(U256::from_be_slice(&point.y.into_bigint().to_bytes_be()));
        }

        for point in self.w.iter() {
            w.push(U256::from_be_slice(&point.x.into_bigint().to_bytes_be()));
            w.push(U256::from_be_slice(&point.y.into_bigint().to_bytes_be()));
        }

        HyperKZGProofSol {
            com,
            w,
            v_ypos,
            v_yneg,
            v_y,
        }
    }
}

impl Into<VK> for &HyperKZGVerifierKey<Bn254> {
    fn into(self) -> VK {
        let g1 = self.kzg_vk.g1;
        // Must be negative b/c this is what the contracts expect
        let g2 = -self.kzg_vk.g2;
        let g2_sol = vec![
            U256::from_be_slice(&g2.x.c0.into_bigint().to_bytes_be()),
            U256::from_be_slice(&g2.x.c1.into_bigint().to_bytes_be()),
            U256::from_be_slice(&g2.y.c0.into_bigint().to_bytes_be()),
            U256::from_be_slice(&g2.y.c1.into_bigint().to_bytes_be()),
        ];
        let g2_beta = self.kzg_vk.beta_g2;
        let g2_beta_sol = vec![
            U256::from_be_slice(&g2_beta.x.c0.into_bigint().to_bytes_be()),
            U256::from_be_slice(&g2_beta.x.c1.into_bigint().to_bytes_be()),
            U256::from_be_slice(&g2_beta.y.c0.into_bigint().to_bytes_be()),
            U256::from_be_slice(&g2_beta.y.c1.into_bigint().to_bytes_be()),
        ];

        VK {
            VK_g1_x: U256::from_be_slice(&g1.x.into_bigint().to_bytes_be()),
            VK_g1_y: U256::from_be_slice(&g1.y.into_bigint().to_bytes_be()),
            VK_g2: g2_sol,
            VK_beta_g2: g2_beta_sol,
        }
    }
}

impl<F: JoltField, ProofTranscript: Transcript> Into<SumcheckProof>
    for &SumcheckInstanceProof<F, ProofTranscript>
{
    fn into(self) -> SumcheckProof {
        let mut compressed_polys = vec![];

        for poly in self.compressed_polys.iter() {
            let new_poly: Vec<U256> = poly
                .coeffs_except_linear_term
                .iter()
                .map(|i| into_uint256(*i))
                .collect();
            compressed_polys.push(new_poly);
        }

        SumcheckProof {
            compressedPolys: compressed_polys,
        }
    }
}

pub fn into_uint256<F: JoltField>(from: F) -> U256 {
    let mut buf = vec![];
    from.serialize_uncompressed(&mut buf).unwrap();
    U256::from_le_slice(&buf)
}

const C: usize = 4;
impl<ProofTranscript: Transcript> Into<SpartanProof>
    for &UniformSpartanProof<C, JoltR1CSInputs, Fp<MontBackend<FrConfig, 4>, 4>, ProofTranscript>
{
    fn into(self) -> SpartanProof {
        let claimed_evals = self
            .claimed_witness_evals
            .iter()
            .map(|i| into_uint256(*i))
            .collect();

        SpartanProof {
            outer: (&self.outer_sumcheck_proof).into(),
            outerClaimA: into_uint256(self.outer_sumcheck_claims.0),
            outerClaimB: into_uint256(self.outer_sumcheck_claims.1),
            outerClaimC: into_uint256(self.outer_sumcheck_claims.2),
            inner: (&self.inner_sumcheck_proof).into(),
            claimedEvals: claimed_evals,
        }
    }
}

impl<F: JoltField, ProofTranscript: Transcript> Into<GKRLayer>
    for BatchedGrandProductLayerProof<F, ProofTranscript>
{
    fn into(self) -> GKRLayer {
        GKRLayer {
            sumcheck: (&self.proof).into(),
            leftClaim: into_uint256(self.left_claim),
            rightClaim: into_uint256(self.left_claim),
        }
    }
}

impl<ProofTranscript: Transcript> Into<GrandProductProof>
    for BatchedGrandProductProof<HyperKZG<Bn254, ProofTranscript>, ProofTranscript>
{
    fn into(self) -> GrandProductProof {
        let layers: Vec<GKRLayer> = self.gkr_layers.into_iter().map(|i| i.into()).collect();
        assert!(self.quark_proof.is_none(), "Quarks are unsupported");
        GrandProductProof { layers }
    }
}
