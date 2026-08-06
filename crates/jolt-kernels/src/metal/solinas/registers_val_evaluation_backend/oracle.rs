//! Independent dense oracle for round, handoff, and output parity.

use jolt_field::Field;

const ABSENT_REGISTER: u8 = u8::MAX;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoundSamples<F> {
    pub at_0: F,
    pub at_1: F,
    pub at_2: F,
    pub at_3: F,
}

impl<F: Field> RoundSamples<F> {
    pub fn device_samples(self) -> [F; 3] {
        [self.at_0, self.at_2, self.at_3]
    }

    pub fn claim_identity(self) -> F {
        self.at_0 + self.at_1
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersValOutputs<F> {
    pub rd_inc: F,
    pub rd_wa: F,
    pub lt_cycle: F,
    pub relation_claim: F,
}

#[derive(Clone, Debug)]
pub struct RegistersValOracle<F> {
    inc: Vec<F>,
    wa: Vec<F>,
    lt: Vec<F>,
    messages_emitted: usize,
}

impl<F: Field> RegistersValOracle<F> {
    pub fn from_indices(
        inc: Vec<F>,
        rd: &[u8],
        eq_address: &[F],
        lt: Vec<F>,
    ) -> Result<Self, RegistersValOracleError> {
        if eq_address.len() != 128 {
            return Err(RegistersValOracleError::AddressTableLength {
                got: eq_address.len(),
            });
        }
        if rd.len() != inc.len() {
            return Err(RegistersValOracleError::TableLengthMismatch {
                table: "rd",
                expected: inc.len(),
                got: rd.len(),
            });
        }
        let mut wa = Vec::with_capacity(rd.len());
        for &index in rd {
            if index == ABSENT_REGISTER {
                wa.push(F::zero());
            } else if let Some(value) = eq_address.get(index as usize) {
                wa.push(*value);
            } else {
                return Err(RegistersValOracleError::RegisterOutsideDomain { got: index });
            }
        }
        Self::from_dense(inc, wa, lt)
    }

    pub fn from_dense(
        inc: Vec<F>,
        wa: Vec<F>,
        lt: Vec<F>,
    ) -> Result<Self, RegistersValOracleError> {
        if inc.len() < 2 || !inc.len().is_power_of_two() {
            return Err(RegistersValOracleError::InvalidLength { got: inc.len() });
        }
        for (name, len) in [("wa", wa.len()), ("lt", lt.len())] {
            if len != inc.len() {
                return Err(RegistersValOracleError::TableLengthMismatch {
                    table: name,
                    expected: inc.len(),
                    got: len,
                });
            }
        }
        Ok(Self {
            inc,
            wa,
            lt,
            messages_emitted: 0,
        })
    }

    pub const fn len(&self) -> usize {
        self.inc.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.inc.is_empty()
    }

    pub const fn messages_emitted(&self) -> usize {
        self.messages_emitted
    }

    pub fn tables(&self) -> (&[F], &[F], &[F]) {
        (&self.inc, &self.wa, &self.lt)
    }

    pub fn current_claim(&self) -> F {
        self.inc
            .iter()
            .zip(&self.wa)
            .zip(&self.lt)
            .map(|((inc, wa), lt)| *inc * *wa * *lt)
            .sum()
    }

    pub fn prove_round(
        &mut self,
        bind: Option<F>,
        previous_claim: F,
    ) -> Result<RoundSamples<F>, RegistersValOracleError> {
        match (self.messages_emitted, bind) {
            (0, None) => {}
            (0, Some(_)) => return Err(RegistersValOracleError::BindBeforeFirstMessage),
            (_, None) => return Err(RegistersValOracleError::MissingPendingBind),
            (_, Some(challenge)) => self.bind(challenge)?,
        }
        let samples = self.round_samples()?;
        if samples.claim_identity() != previous_claim {
            return Err(RegistersValOracleError::ClaimMismatch);
        }
        self.messages_emitted += 1;
        Ok(samples)
    }

    pub fn finish_rounds(&mut self, challenge: F) -> Result<(), RegistersValOracleError> {
        if self.inc.len() != 2 {
            return Err(RegistersValOracleError::FinishBeforeLastMessage {
                remaining: self.inc.len(),
            });
        }
        self.bind(challenge)
    }

    pub fn outputs(&self) -> Result<RegistersValOutputs<F>, RegistersValOracleError> {
        if self.inc.len() != 1 {
            return Err(RegistersValOracleError::OutputBeforeFullyBound {
                remaining: self.inc.len(),
            });
        }
        let rd_inc = self.inc[0];
        let rd_wa = self.wa[0];
        let lt_cycle = self.lt[0];
        Ok(RegistersValOutputs {
            rd_inc,
            rd_wa,
            lt_cycle,
            relation_claim: lt_cycle * rd_inc * rd_wa,
        })
    }

    fn round_samples(&self) -> Result<RoundSamples<F>, RegistersValOracleError> {
        if self.inc.len() < 2 {
            return Err(RegistersValOracleError::MessageAfterFullyBound);
        }
        let mut sums = [F::zero(); 4];
        for pair in 0..self.inc.len() / 2 {
            let index = 2 * pair;
            let inc = samples(self.inc[index], self.inc[index + 1]);
            let wa = samples(self.wa[index], self.wa[index + 1]);
            let lt = samples(self.lt[index], self.lt[index + 1]);
            for sample in 0..4 {
                sums[sample] += inc[sample] * wa[sample] * lt[sample];
            }
        }
        Ok(RoundSamples {
            at_0: sums[0],
            at_1: sums[1],
            at_2: sums[2],
            at_3: sums[3],
        })
    }

    fn bind(&mut self, challenge: F) -> Result<(), RegistersValOracleError> {
        if self.inc.len() < 2 {
            return Err(RegistersValOracleError::BindAfterFullyBound);
        }
        bind_table(&mut self.inc, challenge);
        bind_table(&mut self.wa, challenge);
        bind_table(&mut self.lt, challenge);
        Ok(())
    }
}

fn samples<F: Field>(low: F, high: F) -> [F; 4] {
    let delta = high - low;
    let at_2 = high + delta;
    [low, high, at_2, at_2 + delta]
}

fn bind_table<F: Field>(table: &mut Vec<F>, challenge: F) {
    let half = table.len() / 2;
    for index in 0..half {
        let low = table[2 * index];
        table[index] = low + challenge * (table[2 * index + 1] - low);
    }
    table.truncate(half);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegistersValOracleError {
    InvalidLength {
        got: usize,
    },
    TableLengthMismatch {
        table: &'static str,
        expected: usize,
        got: usize,
    },
    AddressTableLength {
        got: usize,
    },
    RegisterOutsideDomain {
        got: u8,
    },
    BindBeforeFirstMessage,
    MissingPendingBind,
    ClaimMismatch,
    MessageAfterFullyBound,
    BindAfterFullyBound,
    FinishBeforeLastMessage {
        remaining: usize,
    },
    OutputBeforeFullyBound {
        remaining: usize,
    },
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test setup")]
mod tests {
    use jolt_field::AkitaField;

    use super::*;

    fn field(value: u64) -> AkitaField {
        AkitaField::from_u64(value)
    }

    fn bind(values: &[AkitaField], challenge: AkitaField) -> Vec<AkitaField> {
        values
            .chunks_exact(2)
            .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
            .collect()
    }

    fn direct_samples(
        inc: &[AkitaField],
        wa: &[AkitaField],
        lt: &[AkitaField],
    ) -> RoundSamples<AkitaField> {
        let evaluate = |point: AkitaField| {
            inc.chunks_exact(2)
                .zip(wa.chunks_exact(2))
                .zip(lt.chunks_exact(2))
                .map(|((inc, wa), lt)| {
                    let inc = inc[0] + point * (inc[1] - inc[0]);
                    let wa = wa[0] + point * (wa[1] - wa[0]);
                    let lt = lt[0] + point * (lt[1] - lt[0]);
                    inc * wa * lt
                })
                .sum()
        };
        RoundSamples {
            at_0: evaluate(AkitaField::zero()),
            at_1: evaluate(AkitaField::one()),
            at_2: evaluate(field(2)),
            at_3: evaluate(field(3)),
        }
    }

    #[test]
    fn rounds_and_outputs_match_direct_dense_evaluation() {
        let mut inc = vec![field(2), field(3), field(5), field(7)];
        let mut wa = vec![field(11), field(13), AkitaField::zero(), field(17)];
        let mut lt = vec![field(19), field(23), field(29), field(31)];
        let mut oracle =
            RegistersValOracle::from_dense(inc.clone(), wa.clone(), lt.clone()).unwrap();

        let first = oracle.prove_round(None, oracle.current_claim()).unwrap();
        assert_eq!(first, direct_samples(&inc, &wa, &lt));
        assert_eq!(first.device_samples(), [first.at_0, first.at_2, first.at_3]);

        let first_challenge = field(37);
        inc = bind(&inc, first_challenge);
        wa = bind(&wa, first_challenge);
        lt = bind(&lt, first_challenge);
        let next_claim = inc
            .iter()
            .zip(&wa)
            .zip(&lt)
            .map(|((inc, wa), lt)| *inc * *wa * *lt)
            .sum();
        let second = oracle
            .prove_round(Some(first_challenge), next_claim)
            .unwrap();
        assert_eq!(second, direct_samples(&inc, &wa, &lt));

        let final_challenge = field(41);
        oracle.finish_rounds(final_challenge).unwrap();
        let outputs = oracle.outputs().unwrap();
        let inc = bind(&inc, final_challenge)[0];
        let wa = bind(&wa, final_challenge)[0];
        let lt = bind(&lt, final_challenge)[0];
        assert_eq!(outputs.rd_inc, inc);
        assert_eq!(outputs.rd_wa, wa);
        assert_eq!(outputs.lt_cycle, lt);
        assert_eq!(outputs.relation_claim, inc * wa * lt);
    }

    #[test]
    fn native_indices_preserve_absent_and_boundary_registers() {
        let eq_address = (0..128)
            .map(|index| field(index as u64 + 1))
            .collect::<Vec<_>>();
        let oracle = RegistersValOracle::from_indices(
            vec![field(3), field(5), field(7), field(11)],
            &[0, u8::MAX, 127, 1],
            &eq_address,
            vec![field(13), field(17), field(19), field(23)],
        )
        .unwrap();
        let (_, wa, _) = oracle.tables();
        assert_eq!(wa, &[field(1), AkitaField::zero(), field(128), field(2)]);

        assert_eq!(
            RegistersValOracle::from_indices(
                vec![field(1); 2],
                &[0, 128],
                &eq_address,
                vec![field(1); 2],
            )
            .unwrap_err(),
            RegistersValOracleError::RegisterOutsideDomain { got: 128 }
        );
    }
}
