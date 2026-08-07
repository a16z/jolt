use jolt_field::{AkitaField, Field, FromPrimitiveInt};
use jolt_poly::EqPolynomial;

use super::*;

type F = AkitaField;

fn f(value: u64) -> F {
    F::from_u64(value)
}

fn source() -> SparseRamSource {
    SparseRamSource::new(0xabc, 7).unwrap()
}

fn certificates(exact: bool) -> SparseRamCertificates {
    SparseRamCertificates::new(if exact {
        HammingSupportCertificate::Exact
    } else {
        HammingSupportCertificate::Uncertified
    })
}

fn fixture_owner(exact_hamming: bool) -> SparseRamOwner {
    let accesses = vec![
        SparseRamAccess::new(0, 0, 1, 1),
        SparseRamAccess::new(1, 7, 5, 9),
        SparseRamAccess::new(3, 2, 20, 12),
        SparseRamAccess::new(4, 6, 9, 9),
        SparseRamAccess::new(7, 1, 4, 4),
        SparseRamAccess::new(8, 5, 3, 3),
        SparseRamAccess::new(14, 3, 8, 8),
        SparseRamAccess::new(15, 4, 2, 2),
    ];
    let increments = vec![
        SparseRamIncrement::new(1, 4),
        SparseRamIncrement::new(2, 7),
        SparseRamIncrement::new(3, -8),
    ];
    SparseRamOwner::new(
        4,
        8,
        source(),
        certificates(exact_hamming),
        accesses,
        increments,
    )
    .unwrap()
}

#[test]
fn owner_preserves_separate_streams_and_exact_union_merges() {
    let owner = fixture_owner(true);
    assert_eq!(owner.cycle_domain(), 16);
    assert_eq!(owner.address_bits(), 3);
    assert_eq!(owner.accesses().len(), 8);
    assert_eq!(owner.increments().len(), 3);
    assert_eq!(owner.provenance().source().identity(), 0xabc);
    assert_eq!(owner.provenance().source().generation(), 7);
    assert_eq!(owner.provenance().log_t(), 4);
    assert_eq!(owner.provenance().address_domain(), 8);
    assert_eq!(owner.provenance().access_count(), 8);
    assert_eq!(owner.provenance().increment_count(), 3);
    assert_eq!(
        owner.provenance().hamming_support(),
        HammingSupportCertificate::Exact
    );

    let topology = owner.topology();
    assert_eq!(topology.log_t(), 4);
    assert_eq!(
        topology
            .leaves()
            .iter()
            .map(|leaf| leaf.cycle())
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4, 7, 8, 14, 15]
    );
    let raw_zero_leaf = topology
        .leaves()
        .iter()
        .find(|leaf| leaf.cycle() == 2)
        .unwrap();
    assert_eq!(raw_zero_leaf.access_index(), None);
    assert_eq!(raw_zero_leaf.increment_index(), Some(1));
    let store_leaf = topology
        .leaves()
        .iter()
        .find(|leaf| leaf.cycle() == 1)
        .unwrap();
    assert_eq!(store_leaf.access_index(), Some(1));
    assert_eq!(store_leaf.increment_index(), Some(0));

    let expected_blocks = [
        vec![0, 1, 2, 3, 4, 7, 8, 14, 15],
        vec![0, 1, 2, 3, 4, 7],
        vec![0, 1, 2, 3],
        vec![0, 1],
        vec![0],
    ];
    for (level, expected) in expected_blocks.iter().enumerate() {
        assert_eq!(
            topology
                .level(level)
                .unwrap()
                .iter()
                .map(|node| node.block())
                .collect::<Vec<_>>(),
            *expected
        );
    }
    assert_eq!(topology.level_offsets().len(), 6);
    assert_eq!(
        topology.total_nodes(),
        expected_blocks.iter().map(Vec::len).sum::<usize>()
    );

    let level_one = topology.level(1).unwrap();
    assert_eq!(level_one[0].even_child(), Some(0));
    assert_eq!(level_one[0].odd_child(), Some(1));
    assert_eq!(level_one[2].even_child(), Some(4));
    assert_eq!(level_one[2].odd_child(), None);
    assert_eq!(level_one[3].even_child(), None);
    assert_eq!(level_one[3].odd_child(), Some(5));
}

#[test]
fn owner_rejects_order_domain_delta_and_zero_activity_drift() {
    let make = |accesses, increments| {
        SparseRamOwner::new(3, 8, source(), certificates(true), accesses, increments)
    };
    assert!(matches!(
        make(
            vec![
                SparseRamAccess::new(2, 0, 0, 0),
                SparseRamAccess::new(2, 1, 0, 0),
            ],
            vec![],
        ),
        Err(RamFamilyV2Error::AccessesOutOfOrder { cycle: 2 })
    ));
    assert!(matches!(
        make(vec![SparseRamAccess::new(8, 0, 0, 0)], vec![]),
        Err(RamFamilyV2Error::AccessCycleOutOfRange { cycle: 8, .. })
    ));
    assert!(matches!(
        make(vec![SparseRamAccess::new(0, 8, 0, 0)], vec![]),
        Err(RamFamilyV2Error::AddressOutOfRange { address: 8, .. })
    ));
    assert!(matches!(
        make(
            vec![SparseRamAccess::new(1, 0, 4, 9)],
            vec![SparseRamIncrement::new(1, 4)],
        ),
        Err(RamFamilyV2Error::IncrementDeltaMismatch {
            cycle: 1,
            expected: 5,
            actual: Some(4),
        })
    ));
    assert!(matches!(
        make(
            vec![SparseRamAccess::new(1, 0, 4, 4)],
            vec![SparseRamIncrement::new(1, 1)],
        ),
        Err(RamFamilyV2Error::IncrementDeltaMismatch {
            cycle: 1,
            expected: 0,
            actual: Some(1),
        })
    ));
    assert!(matches!(
        make(vec![], vec![SparseRamIncrement::new(2, 0)]),
        Err(RamFamilyV2Error::ZeroIncrement { cycle: 2 })
    ));
    assert!(SparseRamOwner::new(
        3,
        8,
        source(),
        certificates(true),
        vec![SparseRamAccess::new(0, 7, 0, 0)],
        vec![SparseRamIncrement::new(1, -3)],
    )
    .is_ok());
}

#[test]
fn empty_owner_has_all_empty_levels_and_boundary_geometry_is_checked() {
    let owner = SparseRamOwner::new(0, 1, source(), certificates(true), vec![], vec![]).unwrap();
    assert_eq!(owner.cycle_domain(), 1);
    assert_eq!(owner.topology().level(0), Some(&[] as &[SparseCycleNode]));
    assert_eq!(owner.topology().level_offsets(), &[0, 0]);
    assert_eq!(owner.topology().total_nodes(), 0);
    assert_eq!(owner.certified_hamming_accesses().unwrap(), &[]);
    assert_eq!(
        SparseRamSource::new(0, 1),
        Err(RamFamilyV2Error::ZeroSourceIdentity)
    );
    assert_eq!(
        SparseRamSource::new(1, 0),
        Err(RamFamilyV2Error::ZeroSourceGeneration)
    );
    assert!(matches!(
        SparseRamOwner::new(33, 1, source(), certificates(true), vec![], vec![]),
        Err(RamFamilyV2Error::InvalidLogT { log_t: 33 })
    ));
    assert!(matches!(
        SparseRamOwner::new(1, 3, source(), certificates(true), vec![], vec![]),
        Err(RamFamilyV2Error::InvalidAddressDomain { address_domain: 3 })
    ));
}

#[test]
fn hamming_relations_fail_closed_without_exact_support_certificate() {
    let owner = fixture_owner(false);
    assert_eq!(
        owner.certified_hamming_accesses(),
        Err(RamFamilyV2Error::HammingSupportUncertified)
    );
    let r_address = [f(2), f(3), f(5)];
    let cycle_0 = [f(7), f(11), f(13), f(17)];
    let cycle_1 = [f(19), f(23), f(29), f(31)];
    let cycle_2 = [f(37), f(41), f(43), f(47)];
    let challenges = [f(53), f(59), f(61), f(67)];
    assert_eq!(
        ram_ra_claim_reduction(
            RamRaClaimOracleInputs {
                owner: &owner,
                r_address: &r_address,
                cycle_points: [&cycle_0, &cycle_1, &cycle_2],
                gamma: f(71),
            },
            &challenges,
        ),
        Err(RamFamilyV2Error::HammingSupportUncertified)
    );
}

#[test]
fn sparse_oracle_matches_independent_dense_folding_across_carries() {
    let owner = fixture_owner(true);
    let r_address = [f(2), f(3), f(5)];
    let cycle_0 = [f(7), f(11), f(13), f(17)];
    let cycle_1 = [f(19), f(23), f(29), f(31)];
    let cycle_2 = [f(37), f(41), f(43), f(47)];
    let cycle_points = [&cycle_0[..], &cycle_1[..], &cycle_2[..]];
    let challenge_sets = [
        [f(53), f(59), f(61), f(67)],
        [F::zero(), F::one(), f(2), f(3)],
    ];
    for gamma in [F::zero(), F::one(), f(71)] {
        for challenges in challenge_sets {
            let sparse = ram_ra_claim_reduction(
                RamRaClaimOracleInputs {
                    owner: &owner,
                    r_address: &r_address,
                    cycle_points,
                    gamma,
                },
                &challenges,
            )
            .unwrap();
            let dense = dense_oracle(&owner, &r_address, cycle_points, gamma, &challenges);
            assert_eq!(sparse, dense);
            assert_eq!(
                &sparse.output_point[3..],
                challenges.iter().rev().copied().collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn empty_sparse_oracle_preserves_zero_claims_and_derived_outputs() {
    let owner = SparseRamOwner::new(2, 2, source(), certificates(true), vec![], vec![]).unwrap();
    let r_address = [f(3)];
    let cycle_0 = [f(5), f(7)];
    let cycle_1 = [f(11), f(13)];
    let cycle_2 = [f(17), f(19)];
    let challenges = [f(23), f(29)];
    let result = ram_ra_claim_reduction(
        RamRaClaimOracleInputs {
            owner: &owner,
            r_address: &r_address,
            cycle_points: [&cycle_0, &cycle_1, &cycle_2],
            gamma: f(31),
        },
        &challenges,
    )
    .unwrap();
    assert_eq!(result.input_claim, F::zero());
    assert_eq!(result.messages, vec![[F::zero(); 2]; 2]);
    assert_eq!(result.ram_ra, F::zero());
    let output_cycle = challenges.iter().rev().copied().collect::<Vec<_>>();
    assert_eq!(
        result.derived_cycle_eq,
        [
            EqPolynomial::mle(&cycle_0, &output_cycle),
            EqPolynomial::mle(&cycle_1, &output_cycle),
            EqPolynomial::mle(&cycle_2, &output_cycle),
        ]
    );
}

fn dense_oracle(
    owner: &SparseRamOwner,
    r_address: &[F],
    cycle_points: [&[F]; RAM_RA_CLAIM_TERMS],
    gamma: F,
    challenges: &[F],
) -> RamRaClaimOracleResult<F> {
    let rows = 1usize << owner.log_t();
    let eq_address = EqPolynomial::evals(r_address, None);
    let mut h = vec![F::zero(); rows];
    for access in owner.accesses() {
        h[access.cycle() as usize] = eq_address[access.address() as usize];
    }
    let mut eq_cycle = cycle_points.map(|point| EqPolynomial::evals(point, None));
    let gamma_powers = [F::one(), gamma, gamma * gamma];
    let input_claim = h.iter().enumerate().fold(F::zero(), |claim, (row, &h)| {
        let e = (0..RAM_RA_CLAIM_TERMS).fold(F::zero(), |sum, term| {
            sum + gamma_powers[term] * eq_cycle[term][row]
        });
        claim + h * e
    });
    let mut messages = Vec::with_capacity(challenges.len());
    for &challenge in challenges {
        let mut message = [F::zero(); 2];
        for pair in 0..h.len() / 2 {
            let h_0 = h[2 * pair];
            let h_1 = h[2 * pair + 1];
            let h_2 = h_1 + h_1 - h_0;
            let mut e_0 = F::zero();
            let mut e_2 = F::zero();
            for term in 0..RAM_RA_CLAIM_TERMS {
                let term_0 = eq_cycle[term][2 * pair];
                let term_1 = eq_cycle[term][2 * pair + 1];
                e_0 += gamma_powers[term] * term_0;
                e_2 += gamma_powers[term] * (term_1 + term_1 - term_0);
            }
            message[0] += h_0 * e_0;
            message[1] += h_2 * e_2;
        }
        messages.push(message);
        bind_pairs(&mut h, challenge);
        for table in &mut eq_cycle {
            bind_pairs(table, challenge);
        }
    }
    let output_cycle = challenges.iter().rev().copied().collect::<Vec<_>>();
    RamRaClaimOracleResult {
        input_claim,
        messages,
        ram_ra: h[0],
        derived_cycle_eq: cycle_points.map(|point| EqPolynomial::mle(point, &output_cycle)),
        output_point: [r_address, output_cycle.as_slice()].concat(),
    }
}

fn bind_pairs(values: &mut Vec<F>, challenge: F) {
    for pair in 0..values.len() / 2 {
        let even = values[2 * pair];
        let odd = values[2 * pair + 1];
        values[pair] = even + challenge * (odd - even);
    }
    values.truncate(values.len() / 2);
}
