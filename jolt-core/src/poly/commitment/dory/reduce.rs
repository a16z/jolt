#![allow(dead_code)]

use std::ops::{Add, Mul};

use ark_ec::pairing::Pairing;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::{
    field::JoltField,
    utils::transcript::{AppendToTranscript, Transcript},
};

use super::{
    append_gt,
    vec_operations::{mul_gt, InnerProd},
    Commitment, Error, G1Vec, G2Vec, Gt, PublicParams, ScalarProof, Witness, Zr, G1, G2,
};

/// Proof
#[derive(Clone, CanonicalDeserialize, CanonicalSerialize)]
pub struct DoryProof<Curve: Pairing> {
    pub from_prover_1: Vec<ReduceProverStep1Elements<Curve>>,
    pub from_prover_2: Vec<ReduceProverStep2Elements<Curve>>,
    pub final_proof: ScalarProof<Curve>,
}

impl<Curve: Pairing> DoryProof<Curve>
where
    Curve::ScalarField: JoltField,
{
    fn verify_recursive<ProofTranscript: Transcript>(
        transcript: &mut ProofTranscript,
        public_params: &[PublicParams<Curve>],
        commitment: Commitment<Curve>,
        from_prover_1: &[ReduceProverStep1Elements<Curve>],
        from_prover_2: &[ReduceProverStep2Elements<Curve>],
        final_proof: &ScalarProof<Curve>,
    ) -> Result<bool, Error> {
        match (public_params, from_prover_1, from_prover_2) {
            ([], _, _) => Err(Error::EmptyPublicParams),
            ([PublicParams::Single(param)], [], []) => final_proof.verify(param, &commitment),
            ([PublicParams::Single(_), ..], _, _) => Err(Error::SingleWithNonEmptySteps),
            ([PublicParams::Multi { .. }, ..], [], _)
            | ([PublicParams::Multi { .. }, ..], _, []) => Err(Error::MultiParamsWithEmptySteps),
            // take the first element of public_params, prover_step_1, prover_step_2
            (
                [PublicParams::Multi {
                    delta_1r,
                    delta_1l,
                    delta_2r,
                    delta_2l,
                    x,
                    ..
                }, public_params_rest @ ..],
                [ReduceProverStep1Elements {
                    d1l, d1r, d2l, d2r, ..
                }, from_prover_1_rest @ ..],
                [step_2_element @ ReduceProverStep2Elements {
                    c_plus, c_minus, ..
                }, from_prover_2_rest @ ..],
            ) => {
                let Commitment { c, d1, d2 } = commitment;

                let step_1_element = ReduceProverStep1Elements {
                    d1l: *d1l,
                    d1r: *d1r,
                    d2l: *d2l,
                    d2r: *d2r,
                    c,
                    d1,
                    d2,
                };

                // update transcript with step_1_elements
                step_1_element.append_to_transcript(transcript);
                // Get from Transcript
                let betha: Zr<Curve> = transcript.challenge_scalar();

                // update transcript with step_2_elements
                step_2_element.append_to_transcript(transcript);
                // Get from Transcript
                let alpha: Zr<Curve> = transcript.challenge_scalar();

                let inverse_alpha = JoltField::inverse(&alpha).ok_or(Error::ZrZero)?;
                let inverse_betha = JoltField::inverse(&betha).ok_or(Error::ZrZero)?;

                let c_prime = mul_gt(&[
                    c,
                    *x,
                    d2 * betha,
                    d1 * inverse_betha,
                    *c_plus * alpha,
                    *c_minus * inverse_alpha,
                ]);

                let d1_prime = mul_gt(&[
                    *d1l * alpha,
                    *d1r,
                    *delta_1l * alpha * betha,
                    *delta_1r * betha,
                ]);

                let d2_prime = mul_gt(&[
                    *d2l * inverse_alpha,
                    *d2r,
                    *delta_2l * inverse_alpha * inverse_betha,
                    *delta_2r * inverse_betha,
                ]);

                let next_commitment = Commitment {
                    c: c_prime,
                    d1: d1_prime,
                    d2: d2_prime,
                };

                Self::verify_recursive(
                    transcript,
                    public_params_rest,
                    next_commitment,
                    from_prover_1_rest,
                    from_prover_2_rest,
                    final_proof,
                )
            }
        }
    }

    pub fn verify<ProofTranscript: Transcript>(
        &self,
        transcript: &mut ProofTranscript,
        public_params: &[PublicParams<Curve>],
        commitment: Commitment<Curve>,
    ) -> Result<bool, Error>
    where
        Gt<Curve>: Mul<Zr<Curve>, Output = Gt<Curve>>,
        G1<Curve>: Mul<Zr<Curve>, Output = G1<Curve>>,
        G2<Curve>: Mul<Zr<Curve>, Output = G2<Curve>>,
    {
        Self::verify_recursive(
            transcript,
            public_params,
            commitment,
            &self.from_prover_1,
            &self.from_prover_2,
            &self.final_proof,
        )
    }
}

#[derive(Clone, CanonicalDeserialize, CanonicalSerialize)]
pub struct ReduceProverStep1Elements<Curve: Pairing> {
    d1l: Gt<Curve>,
    d1r: Gt<Curve>,
    d2l: Gt<Curve>,
    d2r: Gt<Curve>,
    c: Gt<Curve>,
    d1: Gt<Curve>,
    d2: Gt<Curve>,
}

impl<P: Pairing> AppendToTranscript for ReduceProverStep1Elements<P> {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript) {
        append_gt(transcript, self.d1l);
        append_gt(transcript, self.d1r);
        append_gt(transcript, self.d2l);
        append_gt(transcript, self.d2r);
        append_gt(transcript, self.c);
        append_gt(transcript, self.d1);
        append_gt(transcript, self.d2);
    }
}

#[derive(Clone, CanonicalDeserialize, CanonicalSerialize)]
pub struct ReduceProverStep2Elements<Curve: Pairing> {
    c_plus: Gt<Curve>,
    c_minus: Gt<Curve>,
}

impl<P: Pairing> AppendToTranscript for ReduceProverStep2Elements<P> {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript) {
        append_gt(transcript, self.c_plus);
        append_gt(transcript, self.c_minus);
    }
}

pub fn reduce<Curve: Pairing, ProofTranscript: Transcript>(
    transcript: &mut ProofTranscript,
    params: &[PublicParams<Curve>],
    witness: Witness<Curve>,
    Commitment { c, d1, d2 }: Commitment<Curve>,
) -> Result<DoryProof<Curve>, Error>
where
    Curve::ScalarField: JoltField,
    G1Vec<Curve>: Add<G1Vec<Curve>, Output = G1Vec<Curve>>,
    G2Vec<Curve>: Add<G2Vec<Curve>, Output = G2Vec<Curve>>,
{
    match params {
        [PublicParams::Multi {
            g1v,
            g2v,
            x,
            gamma_1_prime,
            gamma_2_prime,
            delta_1r,
            delta_1l,
            delta_2r,
            delta_2l,
        }, rest_param @ ..] => {
            let m = g1v.len() / 2;

            // P:
            let v1l: G1Vec<Curve> = (&witness.v1[..m]).into();
            let v1r: G1Vec<Curve> = (&witness.v1[m..]).into();
            let v2l: G2Vec<Curve> = (&witness.v2[..m]).into();
            let v2r: G2Vec<Curve> = (&witness.v2[m..]).into();

            // P --> V:
            let d1l = v1l.inner_prod(gamma_2_prime)?;
            let d1r = v1r.inner_prod(gamma_2_prime)?;
            let d2l = gamma_1_prime.inner_prod(&v2l)?;
            let d2r = gamma_1_prime.inner_prod(&v2r)?;

            let step_1_element = ReduceProverStep1Elements {
                d1l,
                d1r,
                d2l,
                d2r,
                c,
                d1,
                d2,
            };
            // update transcript with step 1 element
            step_1_element.append_to_transcript(transcript);

            // Get from Transcript
            let betha: Zr<Curve> = transcript.challenge_scalar();
            let inverse_betha = JoltField::inverse(&betha).ok_or(Error::ZrZero)?;

            // P:
            let v1 = witness.v1 + (g1v * betha);
            let v2 = witness.v2 + (g2v * inverse_betha);

            let v1l: G1Vec<Curve> = v1[..m].to_vec().into();
            let v1r: G1Vec<Curve> = v1[m..].to_vec().into();
            let v2l: G2Vec<Curve> = v2[..m].to_vec().into();
            let v2r: G2Vec<Curve> = v2[m..].to_vec().into();

            // P --> V:
            let c_plus = v1l.inner_prod(&v2r)?;
            let c_minus = v1r.inner_prod(&v2l)?;

            let step_2_element = ReduceProverStep2Elements { c_plus, c_minus };
            // update transcript with step 2 elements
            step_2_element.append_to_transcript(transcript);
            // Get from Transcript
            let alpha: Zr<Curve> = transcript.challenge_scalar();
            let inverse_alpha = JoltField::inverse(&alpha).ok_or(Error::ZrZero)?;

            let v1_prime = v1l * alpha + v1r;
            let v2_prime = v2l * inverse_alpha + v2r;

            let next_witness = Witness {
                v1: v1_prime,
                v2: v2_prime,
            };

            // we return earlier if == 1 since we don't need to calculate the next_commitment
            if m == 1 {
                return Ok(DoryProof {
                    from_prover_1: vec![step_1_element],
                    from_prover_2: vec![step_2_element],
                    final_proof: ScalarProof::new(next_witness),
                });
            }

            let c_prime = mul_gt(&[
                c,
                *x,
                d2 * betha,
                d1 * inverse_betha,
                c_plus * alpha,
                c_minus * inverse_alpha,
            ]);

            let d1_prime = mul_gt(&[
                d1l * alpha,
                d1r,
                *delta_1l * alpha * betha,
                *delta_1r * betha,
            ]);

            let d2_prime = mul_gt(&[
                d2l * inverse_alpha,
                d2r,
                *delta_2l * inverse_alpha * inverse_betha,
                *delta_2r * inverse_betha,
            ]);

            let next_commitment = Commitment {
                c: c_prime,
                d1: d1_prime,
                d2: d2_prime,
            };

            let DoryProof {
                from_prover_1: step_1_elements,
                from_prover_2: step_2_elements,
                final_proof: scalar_product_proof,
            } = reduce(transcript, rest_param, next_witness, next_commitment)?;

            let mut from_prover_1 = vec![step_1_element];
            from_prover_1.extend(step_1_elements);
            let mut from_prover_2 = vec![step_2_element];
            from_prover_2.extend(step_2_elements);

            Ok(DoryProof {
                from_prover_1,
                from_prover_2,
                final_proof: scalar_product_proof,
            })
        }
        [PublicParams::Single(_), ..] => Err(Error::ReduceSinglePublicParam),
        [] => Err(Error::EmptyPublicParams),
    }
}
