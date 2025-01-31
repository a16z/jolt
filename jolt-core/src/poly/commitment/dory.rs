#![allow(dead_code)]

use crate::field::JoltField;
use crate::msm::Icicle;
use crate::poly::commitment::commitment_scheme::BatchType;
use crate::poly::commitment::commitment_scheme::CommitShape;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::utils::errors::ProofVerifyError;
use crate::utils::transcript::{AppendToTranscript, Transcript};
use ark_ec::{CurveGroup, pairnig::Pairing, AffineRepr, CurveGroup};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::marker::PhantomData;


/// Computes an Inner-Pairing-Product commitment as described in ____:
/// This leverages arkworks Pairing::multi_pairing method.
fn inner_pairing_product<P: Pairing>(g1: &[P::G1Affine], g2: &[P::G2Affine]) -> P::PairingOutput {
    // todo(pat): try to move these checks to a larger context.
    if g1.len() != g2.len() {
		panic(fmt.Sprintf("length mismatch"))
	}

	if g1.len() == 0 || g2.len() == 0 {
		panic("empty vectors")
	}

    // todo(pat): Confirm this isn't the same as performing a multi_miller_loop.
    P::multi_pairing(g1, g2)
}

#[derive(Default, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryCommitment<P: Pairing> {
    pub c: P::PairingOutput,
    pub d1: P::PairingOutput,
    pub d2: P::PairingOutput
}

impl<P: Pairing> AppendToTranscript for DoryCommitment<P> {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, _transcript: &mut ProofTranscript) {
        todo!()
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryPublicParameters<P: Pairing>(Vec<PublicParameters<P>>);

#[derive(CanonicalSerialize, CanonicalDeserialize, Default)]
pub struct PublicParameters<P: Pairings> {
    pub reducePP: ReducePublicParams<P>,
    pub Γ1: Vec<P::G1>,
	pub Γ2: Vec<P::G2>,
	pub χ:  P::PairingOutput
}

impl<P: Pairing> PublicParameters<P> {
    pub fn new(n: usize) -> Self {
        if self.Γ1.len() != 2 * n || self.Γ2.len() != 2 * n {
            panic("recursive public parameters should be twice as the public parameters it is derived from")
        }

        let χ = inner_pairing_product(self.reducePP.Γ1Prime, self.reduce.Γ2Prime);
        let reducePP = Self::reducePP(self.Γ1, self.Γ2, n);

        Self {
            reducePP,
            Γ1: self.reducePP.Γ1Prime,
            Γ2: self.reduce.Γ2Prime,
            χ
        }
    }

    pub fn reducePP(Γ1: &[P::G1], Γ2: &[P::G2], n: usize) -> ReducePublicParams<P> {
        if n == 1 {
            return ReducePP::Default()
        }
        let m = n / 2;
    
        let Γ1L = &Γ1[..m];
        let Γ1R = &Γ1[m..];
        let Γ2L = &Γ2[..m];
        let Γ2R = &Γ2[m..];
    
        // TODO(pat): make the seed phrases depend on m so they are random per reduction.
        let Γ1Prime = PedersenGenerators::<P::G1>::new(m, b"Jolt v1 Dory Public Parameters r1Prime").generators;
        let Γ2Prime = PedersenGenerators::<P::G2>::new(m, b"Jolt Dory Public Paramerets r2Prime").generators;
        let Δ1L = inner_pairing_product(Γ1L, Γ2Prime);
        let Δ1R = inner_pairing_product(Γ1R,Γ2Prime);
        let Δ2L = inner_pairing_product(Γ1Prime, Γ2L);
        let Δ2R = inner_pairing_product(Γ1Prime,Γ2R);
    
        ReducePublicParams {
            Γ1Prime,
            Γ2Prime,
            Δ1R,
            Δ1L,
            Δ2R,
            Δ2L,
        }
    }
}

// Parameters used within the reduction
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct ReducePublicParams<P: Pairing> {
    pub Γ1Prime: Vec<P::G1>,
	pub Γ2Prime: Vec<P::G2>,
	pub Δ1R: P::PairingOutput,
	pub Δ1L: P::PairingOutput,
	pub Δ2R: P::PairingOutput,
	pub Δ2L: P::PairingOutput
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryProof<P: Pairing> {

}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryBatchedProof<P: Pairing> {}

#[derive(Clone)]
pub struct DoryScheme<P: Pairing, ProofTranscript: Transcript> {
    _phantom: PhantomData<(P, ProofTranscript)>,
}

impl<P: Pairing, ProofTranscript: Transcript> CommitmentScheme<ProofTranscript>
    for DoryScheme<P, ProofTranscript>
where 
    <P as Pairing>::ScalarField: JoltField,
    <P as Pairing>::G1: Icicle,
{
    type Field = P::ScalarField;
    type Setup = DoryPublicParams<P>;
    type Commitment = DoryCommitment<P>;
    type Proof = DoryProof<P>;
    type BatchedProof = DoryBatchedProof<P>;

    fn setup(shapes: &[CommitShape]) -> Self::Setup {
        let res = Vec::new();

        // Dory's setup procedure initializes
        let mut max_len: usize = 0;
        for shape in shapes {
            let len = shape.input_length.log_2();
            if len > max_len {
                max_len = len;
            }
        }

        let Γ1 = PedersenGenerators::<P::G1>::new(max_len, b"Jolt v1 Dory G1 generators").generators;
        let Γ2 = PedersenGenerators::<P::G2>::new(max_len, b"Jolt v1 Dory G2 generators").generators;

        let χ = inner_pairing_product(g1, g2);
        let reducePP = PublicParameters::reducePP(Γ1, Γ2, max_len);

        let mut pp = DoryPublicParams {
            reducePP,
            Γ1,
            Γ2,
            χ
        };

        while max_len > 0 {
            res.append(pp);
            if n/2 == 0 {
                break;
            }
            pp = pp.new(max_len / 2);
            max_len /= 2;
        }

        return res
    }

    fn commit(poly: &DensePolynomial<Self::Field>, setup: &Self::Setup) -> Self::Commitment {
    }

    fn batch_commit(
        _evals: &[&[Self::Field]],
        _gens: &Self::Setup,
        _batch_type: BatchType,
    ) -> Vec<Self::Commitment> {
        todo!()
    }

    fn commit_slice(_evals: &[Self::Field], _setup: &Self::Setup) -> Self::Commitment {
        todo!()
    }

    fn prove(
        _none: &Self::Setup,
        _poly: &DensePolynomial<Self::Field>,
        _opening_point: &[Self::Field],
        _transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        todo!()
    }
    fn batch_prove(
        _none: &Self::Setup,
        _polynomials: &[&DensePolynomial<Self::Field>],
        _opening_point: &[Self::Field],
        _openings: &[Self::Field],
        _batch_type: BatchType,
        _transcript: &mut ProofTranscript,
    ) -> Self::BatchedProof {
        todo!()
    }

    fn verify(
        _proof: &Self::Proof,
        _setup: &Self::Setup,
        transcript: &mut ProofTranscript,
        _opening_point: &[Self::Field],
        _opening: &Self::Field,
        _commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {

        // Final Pairing Verification
        /*
        let d = transcript.challenge_scalar();
        let dInv = d.inv();

        let left = P::multi_pairing();
        */
        todo!()
    }

    fn batch_verify(
        _batch_proof: &Self::BatchedProof,
        _setup: &Self::Setup,
        _opening_point: &[Self::Field],
        _openings: &[Self::Field],
        _commitments: &[&Self::Commitment],
        _transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        todo!()
    }

    fn protocol_name() -> &'static [u8] {
        b"dory"
    }
}
