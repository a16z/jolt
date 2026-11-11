use super::{ProveRounds, SingleInstance, SumcheckError};
use crate::round_proof::excess_degree;
use jolt_field::Field;
use jolt_poly::UnivariatePoly;

/// Owned kernel output. Compressed transport requires `Full`, so validation
/// sees the original linear coefficient. `EqFactored` supplies the nonconstant
/// coefficients of the inner polynomial and its equality coordinate tau.
pub enum ProverRound<E: Field> {
    Full(UnivariatePoly<E>),
    EqFactored { wire: Vec<E>, tau: E },
}

/// Fused kernel API: `bind` is absent exactly at round zero; later rounds bind
/// the preceding challenge. Finish binds the last challenge once. Implementors
/// must reject invalid sequences before mutating kernel state.
pub trait ProveSingleRounds<E: Field, Err = SumcheckError<E>> {
    fn num_rounds(&self) -> usize;
    fn prove_round(
        &mut self,
        bind: Option<E>,
        round: usize,
        previous_claim: E,
    ) -> Result<ProverRound<E>, Err>;
    fn finish_rounds(&mut self, bind: E) -> Result<(), Err>;
}

/// Adapts existing full-polynomial kernels with checked call sequencing.
pub struct FullRounds<'a, K: ?Sized> {
    kernel: &'a mut K,
    next: usize,
    finished: bool,
    rounds: Option<usize>,
}

impl<'a, K: ?Sized> FullRounds<'a, K> {
    pub fn new(kernel: &'a mut K) -> Self {
        Self {
            kernel,
            next: 0,
            finished: false,
            rounds: None,
        }
    }
}

impl<E: Field, Err: From<SumcheckError<E>>, K: ProveRounds<E, Err> + ?Sized>
    ProveSingleRounds<E, Err> for FullRounds<'_, K>
{
    fn num_rounds(&self) -> usize {
        self.rounds.unwrap_or_else(|| self.kernel.num_rounds())
    }
    fn prove_round(
        &mut self,
        bind: Option<E>,
        round: usize,
        previous_claim: E,
    ) -> Result<ProverRound<E>, Err> {
        let rounds = self.num_rounds();
        if self.finished || round != self.next || round >= rounds || bind.is_some() != (round != 0)
        {
            return Err(SumcheckError::InvalidInput {
                reason: "invalid single-round sequence",
            }
            .into());
        }
        let poly = self.kernel.prove_round(bind, round, previous_claim)?;
        self.rounds = Some(rounds);
        self.next += 1;
        Ok(ProverRound::Full(poly))
    }
    fn finish_rounds(&mut self, bind: E) -> Result<(), Err> {
        if self.finished || self.next == 0 || self.next != self.num_rounds() {
            return Err(SumcheckError::InvalidInput {
                reason: "premature or repeated finish",
            }
            .into());
        }
        self.kernel.finish_rounds(bind)?;
        self.finished = true;
        Ok(())
    }
}

impl<E: Field> SingleInstance<E> {
    pub(super) fn wire_coefficients(
        &self,
        round: usize,
        claim: E,
        output: ProverRound<E>,
    ) -> Result<Vec<E>, SumcheckError<E>> {
        let tau = self.tau(round)?;
        match output {
            ProverRound::Full(poly) => {
                let coefficients = poly.coefficients();
                if let Some(got) = excess_degree(coefficients, self.degree) {
                    return Err(SumcheckError::DegreeBoundExceeded {
                        got,
                        max: self.degree,
                    });
                }
                let constant = coefficients.first().copied().unwrap_or_else(E::zero);
                let nonconstant: E = coefficients.iter().skip(1).copied().sum();
                let actual = match tau {
                    None => constant + constant + nonconstant,
                    Some(tau) => constant + tau * nonconstant,
                };
                if actual != claim {
                    return Err(SumcheckError::InvalidInput {
                        reason: "full round does not satisfy the input claim",
                    });
                }
                let mut wire = Vec::with_capacity(self.degree);
                match tau {
                    None => {
                        wire.push(constant);
                        wire.extend(coefficients.iter().skip(2).take(self.degree - 1).copied());
                    }
                    Some(_) => wire.extend(coefficients.iter().skip(1).take(self.degree).copied()),
                }
                wire.resize(self.degree, E::zero());
                Ok(wire)
            }
            ProverRound::EqFactored {
                mut wire,
                tau: supplied,
            } => {
                if tau != Some(supplied) {
                    return Err(SumcheckError::InvalidInput {
                        reason: "omitted round format or tau mismatch",
                    });
                }
                if wire.len() > self.degree {
                    return Err(SumcheckError::WrongRoundWidth {
                        round,
                        expected: self.degree,
                        got: wire.len(),
                    });
                }
                wire.resize(self.degree, E::zero());
                Ok(wire)
            }
        }
    }
}
