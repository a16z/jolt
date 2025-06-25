use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
    },
    subprotocols::sumcheck::{BatchableSumcheckInstance, SumcheckInstanceProof},
    utils::{math::Math, transcript::Transcript},
};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};

pub struct RAProof<F: JoltField, ProofTranscript: Transcript> {
    pub ra_i_claims: Vec<F>,
    pub sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
}

/// Virtual RA sumcheck for d-way chunked addresses
pub struct RASumcheck<F: JoltField, const D: usize> {
    /// ra(r_cycle, r_address)
    ra_claim: F,
    /// ra polynomials for each chunk
    ra_i_polys: [MultilinearPolynomial<F>; D],
    /// Eq polynomial as a multilinear polynomial
    eq_poly: EqPolynomial<F>,
    /// Random point r_cycle
    r_cycle: Vec<F>,
    /// r_address pre-chunked
    r_address: Vec<F>,
    /// Random points r_address^(i) for each chunk
    r_address_chunks: [Vec<F>; D],
    /// ra_i_ claims to be proved via evaluation proof
    ra_i_claims: Option<[F; D]>,
    /// Length of the trace
    T: usize,
}

impl<F: JoltField, const D: usize> RASumcheck<F, D> {
    pub fn new(
        ra_claim: F,
        mut ra_i_polys: [MultilinearPolynomial<F>; D],
        r_cycle: Vec<F>,
        r_address: Vec<F>,
        T: usize,
    ) -> Self {
        assert_eq!(
            r_address.len() % D,
            0,
            "r_address length must be divisible by D"
        );

        let chunk_size = r_address.len() / D;
        let mut r_address_chunks_vec = Vec::with_capacity(D);

        for i in 0..D {
            let start = i * chunk_size;
            let end = (i + 1) * chunk_size;
            r_address_chunks_vec.push(r_address[start..end].to_vec());
        }

        let r_address_chunks: [Vec<F>; D] = r_address_chunks_vec
            .try_into()
            .expect("Failed to convert Vec to array");

        let eq_poly = EqPolynomial::new(r_cycle.clone());

        // We do pre-binding. The sumcheck is in variable j hence we can pre-bind the r_address chunks:
        for (poly, chunk) in ra_i_polys.iter_mut().zip(r_address_chunks.iter()) {
            for &r_value in chunk.iter() {
                poly.bind_parallel(r_value, BindingOrder::LowToHigh);
            }
        }

        Self {
            ra_claim,
            ra_i_polys,
            eq_poly,
            r_cycle,
            r_address,
            r_address_chunks,
            ra_i_claims: None,
            T,
        }
    }

    pub fn prove<ProofTranscript: Transcript>(
        mut self,
        transcript: &mut ProofTranscript,
    ) -> (RAProof<F, ProofTranscript>, Vec<F>) {
        let (sumcheck_proof, r_cycle_bound) =
            crate::subprotocols::sumcheck::BatchedSumcheck::prove(vec![&mut self], transcript);

        let ra_i_claims = self
            .ra_i_claims
            .expect("ra_i_claims were not set after prove")
            .to_vec();

        let proof = RAProof {
            sumcheck_proof,
            ra_i_claims,
        };

        (proof, r_cycle_bound)
    }
}

impl<F: JoltField, ProofTranscript: Transcript, const D: usize>
    BatchableSumcheckInstance<F, ProofTranscript> for RASumcheck<F, D>
{
    fn degree(&self) -> usize {
        D + 1
    }

    fn num_rounds(&self) -> usize {
        self.T.log_2()
    }

    fn cache_openings(&mut self) {
        let mut openings = [F::zero(); D];

        for i in 0..D {
            openings[i] = self.ra_i_polys[i].final_sumcheck_claim();
        }

        self.ra_i_claims = Some(openings);
    }

    fn bind(&mut self, r_j: F, _: usize) {
        for ra_i in self.ra_i_polys.iter_mut() {
            ra_i.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
    }

    fn input_claim(&self) -> F {
        self.ra_claim.clone()
    }

    fn expected_output_claim(&self, _r: &[F]) -> F {
        todo!()
    }

    fn compute_prover_message(&self, _round: usize) -> Vec<F> {
        todo!()
    }
}
