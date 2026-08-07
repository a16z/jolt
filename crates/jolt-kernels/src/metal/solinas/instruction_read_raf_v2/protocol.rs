//! Host-side round, handoff, and output ordering.

use thiserror::Error;

use super::carrier::{
    ProducerIdentity, ADDRESS_BITS, ADDRESS_PHASES, LOOKUP_TABLES, PHASE_BITS, VIRTUAL_RA_FACTORS,
};

pub const CYCLE_ROUNDS: usize = 26;
pub const MEMBER_ROUNDS: usize = ADDRESS_BITS + CYCLE_ROUNDS;
pub const OUTPUT_CLAIMS: usize = LOOKUP_TABLES + VIRTUAL_RA_FACTORS + 1;
pub const TARGET_ROWS: usize = 1 << CYCLE_ROUNDS;

/// Admission token tying the fixed 154-round schedule to a `2^26` producer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProtocolGeometry {
    producer: ProducerIdentity,
}

impl ProtocolGeometry {
    pub fn target(producer: ProducerIdentity) -> Result<Self, ProtocolError> {
        if producer.rows() != TARGET_ROWS {
            return Err(ProtocolError::WrongCycleGeometry {
                expected_rows: TARGET_ROWS,
                got_rows: producer.rows(),
            });
        }
        Ok(Self { producer })
    }

    pub const fn producer(self) -> ProducerIdentity {
        self.producer
    }

    pub const fn cycle_rounds(self) -> usize {
        CYCLE_ROUNDS
    }

    pub const fn member_rounds(self) -> usize {
        MEMBER_ROUNDS
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChallengeBinding {
    Address(usize),
    Cycle(usize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemberMessage {
    Address {
        round: usize,
        phase: usize,
        round_in_phase: usize,
    },
    Cycle(usize),
}

/// One `prove_round` call in the host transcript.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProveCall {
    geometry: ProtocolGeometry,
    local_call: usize,
    bind_before_message: Option<ChallengeBinding>,
    message: MemberMessage,
    /// The address phase that must exist before this message.
    materialize_phase: Option<usize>,
    crosses_address_handoff: bool,
}

impl ProveCall {
    pub fn at(geometry: ProtocolGeometry, local_call: usize) -> Result<Self, ProtocolError> {
        if local_call >= geometry.member_rounds() {
            return Err(ProtocolError::InvalidLocalCall(local_call));
        }
        if local_call < ADDRESS_BITS {
            let round = local_call;
            return Ok(Self {
                geometry,
                local_call,
                bind_before_message: round.checked_sub(1).map(ChallengeBinding::Address),
                message: MemberMessage::Address {
                    round,
                    phase: round / PHASE_BITS,
                    round_in_phase: round % PHASE_BITS,
                },
                materialize_phase: round
                    .is_multiple_of(PHASE_BITS)
                    .then_some(round / PHASE_BITS),
                crosses_address_handoff: false,
            });
        }

        let cycle_round = local_call - ADDRESS_BITS;
        Ok(Self {
            geometry,
            local_call,
            bind_before_message: if cycle_round == 0 {
                Some(ChallengeBinding::Address(ADDRESS_BITS - 1))
            } else {
                Some(ChallengeBinding::Cycle(cycle_round - 1))
            },
            message: MemberMessage::Cycle(cycle_round),
            materialize_phase: None,
            crosses_address_handoff: cycle_round == 0,
        })
    }

    pub const fn geometry(self) -> ProtocolGeometry {
        self.geometry
    }

    pub const fn local_call(self) -> usize {
        self.local_call
    }

    pub const fn bind_before_message(self) -> Option<ChallengeBinding> {
        self.bind_before_message
    }

    pub const fn message(self) -> MemberMessage {
        self.message
    }

    pub const fn materialize_phase(self) -> Option<usize> {
        self.materialize_phase
    }

    pub const fn crosses_address_handoff(self) -> bool {
        self.crosses_address_handoff
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FinishCall {
    geometry: ProtocolGeometry,
    bind: ChallengeBinding,
}

impl FinishCall {
    pub const fn for_geometry(geometry: ProtocolGeometry) -> Self {
        Self {
            geometry,
            bind: ChallengeBinding::Cycle(geometry.cycle_rounds() - 1),
        }
    }

    pub const fn geometry(self) -> ProtocolGeometry {
        self.geometry
    }

    pub const fn bind(self) -> ChallengeBinding {
        self.bind
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputClaimSlot {
    LookupTableFlag(usize),
    VirtualRa(usize),
    RafFlag,
}

pub fn output_claim_slot(index: usize) -> Result<OutputClaimSlot, ProtocolError> {
    if index < LOOKUP_TABLES {
        return Ok(OutputClaimSlot::LookupTableFlag(index));
    }
    if index < LOOKUP_TABLES + VIRTUAL_RA_FACTORS {
        return Ok(OutputClaimSlot::VirtualRa(index - LOOKUP_TABLES));
    }
    if index == OUTPUT_CLAIMS - 1 {
        return Ok(OutputClaimSlot::RafFlag);
    }
    Err(ProtocolError::InvalidOutputClaim(index))
}

/// Converts low-to-high cycle challenges to the canonical opening point.
pub fn normalized_cycle_point<F: Copy>(
    geometry: ProtocolGeometry,
    challenges: &[F],
) -> Result<Vec<F>, ProtocolError> {
    if challenges.len() != geometry.cycle_rounds() {
        return Err(ProtocolError::CycleChallengeCount {
            expected: geometry.cycle_rounds(),
            got: challenges.len(),
        });
    }
    Ok(challenges.iter().rev().copied().collect())
}

pub const fn address_phase_for_round(round: usize) -> Option<usize> {
    if round < ADDRESS_BITS {
        Some(round / PHASE_BITS)
    } else {
        None
    }
}

pub const fn address_suffix_len(phase: usize) -> Option<usize> {
    if phase < ADDRESS_PHASES {
        Some(ADDRESS_BITS - (phase + 1) * PHASE_BITS)
    } else {
        None
    }
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum ProtocolError {
    #[error("InstructionReadRaf target schedule needs {expected_rows} rows, got {got_rows}")]
    WrongCycleGeometry {
        expected_rows: usize,
        got_rows: usize,
    },
    #[error("InstructionReadRaf local prove call {0} is outside 0..154")]
    InvalidLocalCall(usize),
    #[error("InstructionReadRaf output claim {0} is outside the canonical 45 claims")]
    InvalidOutputClaim(usize),
    #[error("InstructionReadRaf has {got} cycle challenges, expected {expected}")]
    CycleChallengeCount { expected: usize, got: usize },
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests use fixed valid geometry")]
mod tests {
    use super::*;

    fn producer(rows: usize) -> ProducerIdentity {
        ProducerIdentity::new(3, 0x1000, 4, rows).unwrap()
    }

    #[test]
    fn target_geometry_is_required_by_every_transcript_schedule() {
        assert!(matches!(
            ProtocolGeometry::target(producer(1 << 8)),
            Err(ProtocolError::WrongCycleGeometry {
                expected_rows: TARGET_ROWS,
                got_rows: 256,
            })
        ));
        let geometry = ProtocolGeometry::target(producer(TARGET_ROWS)).unwrap();
        assert_eq!(geometry.cycle_rounds(), 26);
        assert_eq!(geometry.member_rounds(), 154);
        assert_eq!(geometry.producer().rows(), TARGET_ROWS);
        let finish = FinishCall::for_geometry(geometry);
        assert_eq!(finish.geometry(), geometry);
        assert_eq!(finish.bind(), ChallengeBinding::Cycle(25));
    }

    #[test]
    fn prove_call_schedule_preserves_all_host_boundaries() {
        let geometry = ProtocolGeometry::target(producer(TARGET_ROWS)).unwrap();
        let first = ProveCall::at(geometry, 0).unwrap();
        assert_eq!(first.geometry(), geometry);
        assert_eq!(first.bind_before_message, None);
        assert_eq!(first.materialize_phase, Some(0));
        assert_eq!(
            first.message,
            MemberMessage::Address {
                round: 0,
                phase: 0,
                round_in_phase: 0,
            }
        );

        for round in 1..ADDRESS_BITS {
            let call = ProveCall::at(geometry, round).unwrap();
            assert_eq!(
                call.bind_before_message,
                Some(ChallengeBinding::Address(round - 1))
            );
            assert_eq!(
                call.materialize_phase,
                round
                    .is_multiple_of(PHASE_BITS)
                    .then_some(round / PHASE_BITS)
            );
            assert!(!call.crosses_address_handoff);
        }

        let handoff = ProveCall::at(geometry, ADDRESS_BITS).unwrap();
        assert_eq!(
            handoff.bind_before_message,
            Some(ChallengeBinding::Address(ADDRESS_BITS - 1))
        );
        assert_eq!(handoff.message, MemberMessage::Cycle(0));
        assert!(handoff.crosses_address_handoff);
        for cycle_round in 1..CYCLE_ROUNDS {
            let call = ProveCall::at(geometry, ADDRESS_BITS + cycle_round).unwrap();
            assert_eq!(
                call.bind_before_message,
                Some(ChallengeBinding::Cycle(cycle_round - 1))
            );
            assert_eq!(call.message, MemberMessage::Cycle(cycle_round));
        }
        assert_eq!(
            ProveCall::at(geometry, MEMBER_ROUNDS),
            Err(ProtocolError::InvalidLocalCall(MEMBER_ROUNDS))
        );
    }

    #[test]
    fn opening_order_and_cycle_normalization_are_canonical() {
        let geometry = ProtocolGeometry::target(producer(TARGET_ROWS)).unwrap();
        for table in 0..LOOKUP_TABLES {
            assert_eq!(
                output_claim_slot(table).unwrap(),
                OutputClaimSlot::LookupTableFlag(table)
            );
        }
        for factor in 0..VIRTUAL_RA_FACTORS {
            assert_eq!(
                output_claim_slot(LOOKUP_TABLES + factor).unwrap(),
                OutputClaimSlot::VirtualRa(factor)
            );
        }
        assert_eq!(
            output_claim_slot(OUTPUT_CLAIMS - 1).unwrap(),
            OutputClaimSlot::RafFlag
        );

        let challenges = (0..CYCLE_ROUNDS).collect::<Vec<_>>();
        let normalized = normalized_cycle_point(geometry, &challenges).unwrap();
        assert_eq!(normalized, challenges.into_iter().rev().collect::<Vec<_>>());
        assert!(matches!(
            normalized_cycle_point(geometry, &[0usize; CYCLE_ROUNDS - 1]),
            Err(ProtocolError::CycleChallengeCount { got: 25, .. })
        ));
    }
}
