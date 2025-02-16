#![allow(dead_code)]

use ark_ec::pairing::Pairing;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::{
    field::JoltField,
    utils::transcript::{AppendToTranscript, Transcript},
};

use super::{
    append_gt, Commitment, Error, G1Vec, G2Vec, Gt, PublicParams, ScalarProof, Witness, Zr,
};

/// Proof
#[derive(Clone, CanonicalDeserialize, CanonicalSerialize)]
pub struct DoryProof<Curve>
where
    Curve: Pairing,
{
    pub from_prover: Vec<(
        ReduceProverRound1Elements<Curve>,
        ReduceProverRound2Elements<Curve>,
    )>,
    pub final_proof: ScalarProof<Curve>,
}

impl<Curve> DoryProof<Curve>
where
    Curve: Pairing,
    Curve::ScalarField: JoltField,
{
    fn verify_recursive<ProofTranscript>(
        transcript: &mut ProofTranscript,
        public_params: &[PublicParams<Curve>],
        commitment @ Commitment { c, d1, d2 }: Commitment<Curve>,
        from_prover: &[(
            ReduceProverRound1Elements<Curve>,
            ReduceProverRound2Elements<Curve>,
        )],
        final_proof: &ScalarProof<Curve>,
    ) -> Result<bool, Error>
    where
        ProofTranscript: Transcript,
    {
        match (public_params, from_prover) {
            ([], _) => Err(Error::EmptyPublicParams),
            ([PublicParams::Single(param)], []) => final_proof.verify(param, &commitment),
            ([PublicParams::Single(_), ..], _) => Err(Error::SingleWithNonEmptySteps),
            ([PublicParams::Multi { .. }, ..], []) => Err(Error::MultiParamsWithEmptySteps),
            // take the first element of public_params, prover_step_1, prover_step_2
            (
                [PublicParams::Multi {
                    delta_1r,
                    delta_1l,
                    delta_2r,
                    delta_2l,
                    c: c_g,
                    ..
                }, public_params_rest @ ..],
                [(
                    step_1_element @ ReduceProverRound1Elements {
                        d1l: v1l,
                        d1r: v1r,
                        d2l: v2l,
                        d2r: v2r,
                    },
                    step_2_element @ ReduceProverRound2Elements {
                        vl: c_plus,
                        vr: c_minus,
                    },
                ), from_prover_rest @ ..],
            ) => {
                // update transcript with step_1_elements
                commitment.append_to_transcript(transcript);
                step_1_element.append_to_transcript(transcript);
                // Get from Transcript
                let alpha_1: Zr<Curve> = transcript.challenge_scalar();

                // update transcript with step_2_elements
                step_2_element.append_to_transcript(transcript);
                // Get from Transcript
                let alpha_2: Zr<Curve> = transcript.challenge_scalar();

                let inverse_alpha_1 = JoltField::inverse(&alpha_1).ok_or(Error::ZrZero)?;
                let inverse_alpha_2 = JoltField::inverse(&alpha_2).ok_or(Error::ZrZero)?;

                let c_prime = [
                    c,
                    *c_g,
                    d2 * alpha_1,
                    d1 * inverse_alpha_1,
                    *c_plus * alpha_2,
                    *c_minus * inverse_alpha_2,
                ]
                .iter()
                .sum();

                let d1_prime = [
                    *v1l * alpha_2,
                    *v1r,
                    *delta_1l * alpha_2 * alpha_1,
                    *delta_1r * alpha_1,
                ]
                .iter()
                .sum();

                let d2_prime = [
                    *v2l * inverse_alpha_2,
                    *v2r,
                    *delta_2l * inverse_alpha_2 * inverse_alpha_1,
                    *delta_2r * inverse_alpha_1,
                ]
                .iter()
                .sum();

                let next_commitment = Commitment {
                    c: c_prime,
                    d1: d1_prime,
                    d2: d2_prime,
                };

                Self::verify_recursive(
                    transcript,
                    public_params_rest,
                    next_commitment,
                    from_prover_rest,
                    final_proof,
                )
            }
        }
    }

    pub fn verify<ProofTranscript>(
        &self,
        transcript: &mut ProofTranscript,
        public_params: &[PublicParams<Curve>],
        commitment: Commitment<Curve>,
    ) -> Result<bool, Error>
    where
        ProofTranscript: Transcript,
    {
        Self::verify_recursive(
            transcript,
            public_params,
            commitment,
            &self.from_prover,
            &self.final_proof,
        )
    }
}

#[derive(Clone, CanonicalDeserialize, CanonicalSerialize)]
pub struct ReduceProverRound1Elements<Curve>
where
    Curve: Pairing,
{
    d1l: Gt<Curve>,
    d1r: Gt<Curve>,
    d2l: Gt<Curve>,
    d2r: Gt<Curve>,
}

impl<P> AppendToTranscript for ReduceProverRound1Elements<P>
where
    P: Pairing,
{
    fn append_to_transcript<ProofTranscript>(&self, transcript: &mut ProofTranscript)
    where
        ProofTranscript: Transcript,
    {
        append_gt(transcript, self.d1l);
        append_gt(transcript, self.d1r);
        append_gt(transcript, self.d2l);
        append_gt(transcript, self.d2r);
    }
}

#[derive(Clone, CanonicalDeserialize, CanonicalSerialize)]
pub struct ReduceProverRound2Elements<Curve: Pairing> {
    vl: Gt<Curve>,
    vr: Gt<Curve>,
}

impl<P> AppendToTranscript for ReduceProverRound2Elements<P>
where
    P: Pairing,
{
    fn append_to_transcript<ProofTranscript>(&self, transcript: &mut ProofTranscript)
    where
        ProofTranscript: Transcript,
    {
        append_gt(transcript, self.vl);
        append_gt(transcript, self.vr);
    }
}

pub fn reduce<P, ProofTranscript>(
    transcript: &mut ProofTranscript,
    params: &[PublicParams<P>],
    witness: Witness<P>,
    commitment @ Commitment { c, d1, d2 }: Commitment<P>,
) -> Result<DoryProof<P>, Error>
where
    P: Pairing,
    P::ScalarField: JoltField,
    ProofTranscript: Transcript,
{
    match params {
        [PublicParams::Multi {
            g1v,
            g2v,
            c: c_g,
            gamma_1,
            gamma_2,
            delta_1r,
            delta_1l,
            delta_2r,
            delta_2l,
        }, rest_param @ ..] => {
            let m = g1v.len() / 2;

            // P:
            let u1l = &witness.u1[..m];
            let u1r = &witness.u1[m..];

            let u2l = &witness.u2[..m];
            let u2r = &witness.u2[m..];

            // P --> V:
            let d1l = P::multi_pairing(u1l, gamma_2.as_ref());
            let d1r = P::multi_pairing(u1r, gamma_2.as_ref());

            let d2l = P::multi_pairing(gamma_1.as_ref(), u2l);
            let d2r = P::multi_pairing(gamma_1.as_ref(), u2r);

            let step_1_element = ReduceProverRound1Elements { d1l, d1r, d2l, d2r };
            // update transcript with step 1 element
            commitment.append_to_transcript(transcript);
            step_1_element.append_to_transcript(transcript);

            // Get from Transcript
            let betha: Zr<P> = transcript.challenge_scalar();

            let inverse_betha = JoltField::inverse(&betha).ok_or(Error::ZrZero)?;

            // P:
            let w1: G1Vec<P> = witness.u1 + (g1v * betha);
            let w2: G2Vec<P> = witness.u2 + (g2v * inverse_betha);

            let w1l = &w1[..m];
            let w1r = &w1[m..];
            let w2l = &w2[..m];
            let w2r = &w2[m..];

            // P --> V:
            let vl = P::multi_pairing(w1l, w2r);
            let vr = P::multi_pairing(w1r, w2l);
            let step_2_element = ReduceProverRound2Elements { vl, vr };
            // update transcript with step 2 elements
            step_2_element.append_to_transcript(transcript);

            // Get from Transcript
            let alpha: Zr<P> = transcript.challenge_scalar();
            let inverse_alpha = JoltField::inverse(&alpha).ok_or(Error::ZrZero)?;

            let u1_prime = G1Vec::from(w1l) * alpha + G1Vec::from(w1r);
            let u2_prime = G2Vec::from(w2l) * inverse_alpha + G2Vec::from(w2r);

            let next_witness = Witness {
                u1: u1_prime,
                u2: u2_prime,
            };
            // we return earlier if == 1 since we don't need to calculate the next_commitment
            if m == 1 {
                return Ok(DoryProof {
                    from_prover: vec![(step_1_element, step_2_element)],
                    final_proof: ScalarProof::new(next_witness),
                });
            }

            let c_prime = [
                c,
                *c_g,
                d2 * betha,
                d1 * inverse_betha,
                vl * alpha,
                vr * inverse_alpha,
            ]
            .iter()
            .sum();

            let d1_prime = [
                d1l * alpha,
                d1r,
                *delta_1l * alpha * betha,
                *delta_1r * betha,
            ]
            .iter()
            .sum();

            let d2_prime = [
                d2l * inverse_alpha,
                d2r,
                *delta_2l * inverse_alpha * inverse_betha,
                *delta_2r * inverse_betha,
            ]
            .iter()
            .sum();

            let next_commitment = Commitment {
                c: c_prime,
                d1: d1_prime,
                d2: d2_prime,
            };

            let DoryProof {
                from_prover: step_elements,
                final_proof: scalar_product_proof,
            } = reduce(transcript, rest_param, next_witness, next_commitment)?;

            let mut from_prover = vec![(step_1_element, step_2_element)];
            from_prover.extend(step_elements);

            Ok(DoryProof {
                from_prover,
                final_proof: scalar_product_proof,
            })
        }
        // Send u, g and gamma
        [PublicParams::Single(_), ..] => Err(Error::ReduceSinglePublicParam),
        [] => Err(Error::EmptyPublicParams),
    }
}
