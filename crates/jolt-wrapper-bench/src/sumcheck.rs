//! Sumcheck streams: one or several instances proved in lockstep (Jolt's
//! `prove_batch` conventions — a member is active on `[offset, offset+rounds)`
//! and contributes the constant `claim/2` outside its window), with
//! compressed round messages (the constant coefficient is implied by the
//! running claim).

use jolt_field::{Field, Fr, One, Ring, Zero};
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_transcript::Transcript;

use crate::relation::Relation;

pub trait Instance {
    fn rounds(&self) -> usize;
    fn input_claim(&self) -> Fr;
    /// Coefficients (constant first) of the current round's polynomial.
    fn round_poly(&mut self) -> Vec<Fr>;
    fn bind(&mut self, r: Fr);
}

/// A member of a stream: the instance and the first round it is active in.
pub struct Member<'a> {
    pub instance: &'a mut dyn Instance,
    pub offset: usize,
}

pub struct Stream {
    /// Wire form of every round: the coefficients above the constant term.
    pub rounds: Vec<Vec<Fr>>,
    pub challenges: Vec<Fr>,
    pub input_claims: Vec<Fr>,
}

impl Stream {
    pub fn wire_bytes(&self) -> usize {
        32 * self.rounds.iter().map(Vec::len).sum::<usize>()
    }

    pub fn max_degree(&self) -> usize {
        self.rounds.iter().map(Vec::len).max().unwrap_or(0)
    }
}

pub fn horner(coefficients: &[Fr], x: Fr) -> Fr {
    coefficients
        .iter()
        .rev()
        .fold(Fr::zero(), |acc, c| acc * x + *c)
}

fn draw_betas<T: Transcript<Challenge = Fr>>(count: usize, transcript: &mut T) -> Vec<Fr> {
    if count == 1 {
        vec![Fr::one()]
    } else {
        (0..count).map(|_| transcript.challenge()).collect()
    }
}

pub fn prove_stream<T: Transcript<Challenge = Fr>>(
    members: &mut [Member<'_>],
    transcript: &mut T,
) -> Stream {
    let max = members
        .iter()
        .map(|m| m.offset + m.instance.rounds())
        .max()
        .expect("at least one member");
    let input_claims: Vec<Fr> = members.iter().map(|m| m.instance.input_claim()).collect();
    for claim in &input_claims {
        transcript.append(claim);
    }
    let betas = draw_betas(members.len(), transcript);
    let mut member_claims: Vec<Fr> = input_claims
        .iter()
        .zip(members.iter())
        .map(|(c, m)| c.mul_pow_2(max - m.instance.rounds()))
        .collect();
    // A member active before its dummy rounds have halved the padding scale
    // away emits its round polynomials at that scale.
    let scales: Vec<Fr> = members
        .iter()
        .map(|m| Fr::one().mul_pow_2(max - m.offset - m.instance.rounds()))
        .collect();
    let mut running = betas
        .iter()
        .zip(&member_claims)
        .fold(Fr::zero(), |acc, (b, c)| acc + *b * *c);
    let two_inv = Fr::from_u64(2).inverse().expect("2 is invertible");
    let mut rounds = Vec::with_capacity(max);
    let mut challenges = Vec::with_capacity(max);
    for round in 0..max {
        let mut coefficients = vec![Fr::zero(); 2];
        let mut polys: Vec<Option<Vec<Fr>>> = Vec::with_capacity(members.len());
        for (i, member) in members.iter_mut().enumerate() {
            let active = round >= member.offset && round < member.offset + member.instance.rounds();
            if !active {
                member_claims[i] *= two_inv;
                coefficients[0] += betas[i] * member_claims[i];
                polys.push(None);
                continue;
            }
            let poly: Vec<Fr> = member
                .instance
                .round_poly()
                .iter()
                .map(|c| *c * scales[i])
                .collect();
            if coefficients.len() < poly.len() {
                coefficients.resize(poly.len(), Fr::zero());
            }
            for (slot, c) in coefficients.iter_mut().zip(&poly) {
                *slot += betas[i] * *c;
            }
            polys.push(Some(poly));
        }
        while coefficients.len() > 2 && coefficients.last() == Some(&Fr::zero()) {
            let _ = coefficients.pop();
        }
        let at_one: Fr = coefficients.iter().sum();
        assert_eq!(
            coefficients[0] + at_one,
            running,
            "round {round}: s(0) + s(1) must equal the running claim"
        );
        for c in &coefficients[1..] {
            transcript.append(c);
        }
        let r: Fr = transcript.challenge();
        running = horner(&coefficients, r);
        for (i, member) in members.iter_mut().enumerate() {
            if let Some(poly) = &polys[i] {
                member_claims[i] = horner(poly, r);
                member.instance.bind(r);
            }
        }
        rounds.push(coefficients[1..].to_vec());
        challenges.push(r);
    }
    Stream {
        rounds,
        challenges,
        input_claims,
    }
}

/// Verifier replay of [`prove_stream`]: `(rounds, offset)` per member in
/// declaration order. Returns the challenges, the betas and the final claim.
pub fn verify_stream<T: Transcript<Challenge = Fr>>(
    rounds: &[Vec<Fr>],
    input_claims: &[Fr],
    members: &[(usize, usize)],
    transcript: &mut T,
) -> (Vec<Fr>, Vec<Fr>, Fr) {
    let max = members
        .iter()
        .map(|(rounds, offset)| rounds + offset)
        .max()
        .expect("at least one member");
    assert_eq!(rounds.len(), max, "round count");
    for claim in input_claims {
        transcript.append(claim);
    }
    let betas = draw_betas(members.len(), transcript);
    let mut running = betas
        .iter()
        .zip(input_claims)
        .zip(members)
        .fold(Fr::zero(), |acc, ((b, c), (n, _))| {
            acc + *b * c.mul_pow_2(max - n)
        });
    let two_inv = Fr::from_u64(2).inverse().expect("2 is invertible");
    let mut challenges = Vec::with_capacity(max);
    for wire in rounds {
        let tail: Fr = wire.iter().sum();
        let mut coefficients = Vec::with_capacity(wire.len() + 1);
        coefficients.push((running - tail) * two_inv);
        coefficients.extend_from_slice(wire);
        for c in wire {
            transcript.append(c);
        }
        let r: Fr = transcript.challenge();
        running = horner(&coefficients, r);
        challenges.push(r);
    }
    (challenges, betas, running)
}

/// Column-index sumcheck reducing a row relation's final claim
/// `Σ_j γ̃_j v_j² + γ̃'_j v_j w_j + L1_j v_j + L2_j w_j` to `v(s)`, `w(s)`.
pub struct ColumnInstance {
    gamma_sq: Vec<Fr>,
    gamma_cross: Vec<Fr>,
    l1: Vec<Fr>,
    l2: Vec<Fr>,
    v: Vec<Fr>,
    w: Vec<Fr>,
    claim: Fr,
    rounds: usize,
}

impl ColumnInstance {
    pub fn new(relation: &Relation, v: Vec<Fr>, w: Vec<Fr>) -> Self {
        let claim = relation.evaluate(&v, &w);
        Self {
            rounds: relation.log_columns,
            gamma_sq: relation.gamma_sq.clone(),
            gamma_cross: relation.gamma_cross.clone(),
            l1: relation.l1.clone(),
            l2: relation.l2.clone(),
            v,
            w,
            claim,
        }
    }

    /// `(v(s), w(s))` after every round.
    pub fn finals(&self) -> (Fr, Fr) {
        (self.v[0], self.w[0])
    }

    fn summand(gs: Fr, gc: Fr, l1: Fr, l2: Fr, v: Fr, w: Fr) -> Fr {
        v * (gs * v + l1) + w * (gc * v + l2)
    }

    /// Verifier check at `s` (big-endian) against the sent `(v(s), w(s))`.
    pub fn check(relation: &Relation, s_be: &[Fr], v: Fr, w: Fr) -> Fr {
        let eq_s = EqPolynomial::<Fr>::evals(s_be, None);
        let [gs, gc, l1, l2] = relation.coefficients_at(&eq_s);
        Self::summand(gs, gc, l1, l2, v, w)
    }
}

impl Instance for ColumnInstance {
    fn rounds(&self) -> usize {
        self.rounds
    }

    fn input_claim(&self) -> Fr {
        self.claim
    }

    fn round_poly(&mut self) -> Vec<Fr> {
        let half = self.v.len() / 2;
        let mut evals = [Fr::zero(); 4];
        for (x, eval) in evals.iter_mut().enumerate() {
            let x = Fr::from_u64(x as u64);
            let at = |c: &[Fr], i: usize| c[2 * i] + x * (c[2 * i + 1] - c[2 * i]);
            for i in 0..half {
                *eval += Self::summand(
                    at(&self.gamma_sq, i),
                    at(&self.gamma_cross, i),
                    at(&self.l1, i),
                    at(&self.l2, i),
                    at(&self.v, i),
                    at(&self.w, i),
                );
            }
        }
        UnivariatePoly::from_evals(&evals).into_coefficients()
    }

    fn bind(&mut self, r: Fr) {
        for c in [
            &mut self.gamma_sq,
            &mut self.gamma_cross,
            &mut self.l1,
            &mut self.l2,
            &mut self.v,
            &mut self.w,
        ] {
            let half = c.len() / 2;
            for i in 0..half {
                c[i] = c[2 * i] + r * (c[2 * i + 1] - c[2 * i]);
            }
            c.truncate(half);
        }
    }
}
