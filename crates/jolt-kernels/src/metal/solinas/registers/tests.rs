use super::oracle::DenseBtreeRegisterOracle;
use super::*;

struct TestColumn<'a> {
    start_value: u64,
    rs1_positions: &'a [u8],
    rd_positions: &'a [u8],
    rd_post_values: &'a [u64],
}

fn column(csr: &RegisterCsr256, block: usize, register: usize) -> TestColumn<'_> {
    let parts = csr.parts();
    let header = block * REGISTER_CSR_COLUMNS + register;
    let rs1 = parts.rs1_offsets[header] as usize..parts.rs1_offsets[header + 1] as usize;
    let rd = parts.rd_offsets[header] as usize..parts.rd_offsets[header + 1] as usize;
    TestColumn {
        start_value: parts.start_values[header],
        rs1_positions: &parts.rs1_positions[rs1],
        rd_positions: &parts.rd_positions[rd.clone()],
        rd_post_values: &parts.rd_post_values[rd],
    }
}

fn build_owner(
    rows: &[RegisterOwnerRow],
    initial_values: &[u64; REGISTER_CSR_COLUMNS],
    cap: usize,
) -> CertifiedRegisterOwner {
    match CertifiedRegisterOwner::build(rows, initial_values, cap) {
        Ok(owner) => owner,
        Err(error) => panic!("register owner fixture failed: {error}"),
    }
}

fn build_oracle(
    rows: &[RegisterOwnerRow],
    initial_values: &[u64; REGISTER_CSR_COLUMNS],
) -> DenseBtreeRegisterOracle {
    match DenseBtreeRegisterOracle::build(rows, initial_values) {
        Ok(oracle) => oracle,
        Err(error) => panic!("register oracle fixture failed: {error:?}"),
    }
}

fn assert_matches_oracle(
    rows: &[RegisterOwnerRow],
    initial_values: &[u64; REGISTER_CSR_COLUMNS],
    cap: usize,
    owner: &CertifiedRegisterOwner,
) {
    let oracle = build_oracle(rows, initial_values);
    let oracle_parts = match oracle.to_parts() {
        Ok(parts) => parts,
        Err(error) => panic!("register oracle flatten failed: {error:?}"),
    };
    assert_eq!(owner.csr().clone().into_parts(), oracle_parts);
    assert_eq!(owner.state_flow().final_values(), oracle.final_values());
    if oracle.rd_increments().len() <= cap {
        assert_eq!(
            owner.rd_increment_activity().entries(),
            Some(oracle.rd_increments())
        );
        assert!(!owner.rd_increment_activity().overflowed());
    } else {
        assert_eq!(owner.rd_increment_activity().entries(), None);
        assert!(owner.rd_increment_activity().overflowed());
        assert_eq!(
            owner.rd_increment_activity().nonzero_count(),
            oracle.rd_increments().len()
        );
    }
}

#[test]
fn empty_fixture_has_only_terminal_offsets() {
    let initial_values = [11; REGISTER_CSR_COLUMNS];
    let owner = build_owner(&[], &initial_values, 0);
    let parts = owner.csr().parts();

    assert_eq!(owner.csr().cycles(), 0);
    assert_eq!(owner.csr().block_count(), 0);
    assert!(parts.start_values.is_empty());
    assert_eq!(parts.rs1_offsets, [0]);
    assert_eq!(parts.rs2_offsets, [0]);
    assert_eq!(parts.rd_offsets, [0]);
    assert_eq!(owner.csr().storage_bytes(), 12);
    assert_eq!(owner.state_flow().initial_values(), &initial_values);
    assert_eq!(owner.state_flow().final_values(), &initial_values);
    assert_eq!(
        owner.rd_increment_activity().entries(),
        Some(&[] as &[RdIncrement])
    );
    assert_matches_oracle(&[], &initial_values, 0, &owner);
}

#[test]
fn hot_register_events_remain_ordered_across_blocks() {
    let rows: Vec<_> = (0..257)
        .map(|cycle| {
            let value = cycle as u64;
            RegisterOwnerRow {
                rs1: Some(RegisterOwnerRead { register: 7, value }),
                rs2: None,
                rd: Some(RegisterOwnerWrite {
                    register: 7,
                    pre_value: value,
                    post_value: value + 1,
                }),
            }
        })
        .collect();
    let initial_values = [0; REGISTER_CSR_COLUMNS];
    let owner = build_owner(&rows, &initial_values, 300);
    let first = column(owner.csr(), 0, 7);
    let second = column(owner.csr(), 1, 7);

    assert_eq!(first.start_value, 0);
    assert_eq!(first.rs1_positions.len(), 256);
    assert_eq!(first.rd_positions.len(), 256);
    assert_eq!(first.rs1_positions.first(), Some(&0));
    assert_eq!(first.rs1_positions.last(), Some(&255));
    assert_eq!(second.start_value, 256);
    assert_eq!(second.rs1_positions, [0]);
    assert_eq!(second.rd_positions, [0]);
    assert_eq!(owner.state_flow().final_values()[7], 257);

    let activity = match owner.rd_increment_activity().entries() {
        Some(entries) => entries,
        None => panic!("hot-register activity unexpectedly overflowed"),
    };
    assert_eq!(activity.len(), 257);
    assert_eq!(activity.first().map(|entry| entry.cycle), Some(0));
    assert_eq!(activity.last().map(|entry| entry.cycle), Some(256));
    assert!(activity.iter().all(|entry| entry.increment == 1));
    assert_matches_oracle(&rows, &initial_values, 300, &owner);
}

#[test]
fn all_register_fixture_has_header_major_offsets() {
    let rows: Vec<_> = (0..REGISTER_CSR_COLUMNS)
        .map(|register| RegisterOwnerRow {
            rs1: Some(RegisterOwnerRead {
                register: register as u8,
                value: 0,
            }),
            rs2: None,
            rd: Some(RegisterOwnerWrite {
                register: register as u8,
                pre_value: 0,
                post_value: register as u64 + 1,
            }),
        })
        .collect();
    let initial_values = [0; REGISTER_CSR_COLUMNS];
    let owner = build_owner(&rows, &initial_values, REGISTER_CSR_COLUMNS);
    let parts = owner.csr().parts();

    for register in 0..REGISTER_CSR_COLUMNS {
        assert_eq!(parts.rs1_offsets[register], register as u32);
        assert_eq!(parts.rd_offsets[register], register as u32);
        let column = column(owner.csr(), 0, register);
        assert_eq!(column.rs1_positions, [register as u8]);
        assert_eq!(column.rd_positions, [register as u8]);
        assert_eq!(column.rd_post_values, [register as u64 + 1]);
        assert_eq!(
            owner.state_flow().final_values()[register],
            register as u64 + 1
        );
    }
    assert_eq!(parts.rs1_offsets[REGISTER_CSR_COLUMNS], 128);
    assert_eq!(parts.rd_offsets[REGISTER_CSR_COLUMNS], 128);
    assert_matches_oracle(&rows, &initial_values, REGISTER_CSR_COLUMNS, &owner);
}

#[test]
fn block_boundary_uses_previous_block_post_as_next_start() {
    let mut rows = vec![RegisterOwnerRow::default(); 257];
    rows[255].rd = Some(RegisterOwnerWrite {
        register: 9,
        pre_value: 0,
        post_value: 4,
    });
    rows[256] = RegisterOwnerRow {
        rs1: Some(RegisterOwnerRead {
            register: 9,
            value: 4,
        }),
        rs2: None,
        rd: Some(RegisterOwnerWrite {
            register: 9,
            pre_value: 4,
            post_value: 2,
        }),
    };
    let initial_values = [0; REGISTER_CSR_COLUMNS];
    let owner = build_owner(&rows, &initial_values, 2);
    let first = column(owner.csr(), 0, 9);
    let second = column(owner.csr(), 1, 9);

    assert_eq!(first.rd_positions, [255]);
    assert_eq!(first.rd_post_values, [4]);
    assert_eq!(second.start_value, 4);
    assert_eq!(second.rs1_positions, [0]);
    assert_eq!(second.rd_positions, [0]);
    assert_eq!(second.rd_post_values, [2]);
    assert_eq!(
        owner.rd_increment_activity().entries(),
        Some(
            [
                RdIncrement {
                    cycle: 255,
                    increment: 4,
                },
                RdIncrement {
                    cycle: 256,
                    increment: -2,
                },
            ]
            .as_slice()
        )
    );
    assert_matches_oracle(&rows, &initial_values, 2, &owner);
}

#[test]
fn state_flow_certificate_counts_checked_accesses() {
    let initial_values = [5; REGISTER_CSR_COLUMNS];
    let rows = [
        RegisterOwnerRow {
            rs1: Some(RegisterOwnerRead {
                register: 1,
                value: 5,
            }),
            rs2: Some(RegisterOwnerRead {
                register: 2,
                value: 5,
            }),
            rd: Some(RegisterOwnerWrite {
                register: 1,
                pre_value: 5,
                post_value: 8,
            }),
        },
        RegisterOwnerRow {
            rs1: Some(RegisterOwnerRead {
                register: 1,
                value: 8,
            }),
            rs2: None,
            rd: Some(RegisterOwnerWrite {
                register: 1,
                pre_value: 8,
                post_value: 8,
            }),
        },
    ];
    let owner = build_owner(&rows, &initial_values, 1);

    assert_eq!(owner.state_flow().cycles(), 2);
    assert_eq!(
        owner.state_flow().events(),
        RegisterEventCounts {
            rs1: 2,
            rs2: 1,
            rd: 2,
        }
    );
    assert_eq!(owner.state_flow().nonzero_rd_increments(), 1);
    assert_eq!(owner.state_flow().final_values()[1], 8);
    assert_matches_oracle(&rows, &initial_values, 1, &owner);
}

#[test]
fn rd_increment_derivation_keeps_sign_and_omits_zero() {
    let initial_values = [5; REGISTER_CSR_COLUMNS];
    let rows = [
        RegisterOwnerRow {
            rd: Some(RegisterOwnerWrite {
                register: 3,
                pre_value: 5,
                post_value: 8,
            }),
            ..RegisterOwnerRow::default()
        },
        RegisterOwnerRow {
            rd: Some(RegisterOwnerWrite {
                register: 3,
                pre_value: 8,
                post_value: 2,
            }),
            ..RegisterOwnerRow::default()
        },
        RegisterOwnerRow {
            rd: Some(RegisterOwnerWrite {
                register: 3,
                pre_value: 2,
                post_value: 2,
            }),
            ..RegisterOwnerRow::default()
        },
    ];
    let owner = build_owner(&rows, &initial_values, 2);

    assert_eq!(
        owner.rd_increment_activity().entries(),
        Some(
            [
                RdIncrement {
                    cycle: 0,
                    increment: 3,
                },
                RdIncrement {
                    cycle: 1,
                    increment: -6,
                },
            ]
            .as_slice()
        )
    );
    assert_matches_oracle(&rows, &initial_values, 2, &owner);
}

#[test]
fn malformed_offsets_and_event_order_are_rejected() {
    let rows = [RegisterOwnerRow {
        rs1: Some(RegisterOwnerRead {
            register: 0,
            value: 0,
        }),
        ..RegisterOwnerRow::default()
    }];
    let owner = build_owner(&rows, &[0; REGISTER_CSR_COLUMNS], 0);

    let mut bad_start = owner.csr().clone().into_parts();
    bad_start.rs1_offsets[0] = 1;
    assert!(matches!(
        RegisterCsr256::from_parts(bad_start),
        Err(RegisterOwnerError::OffsetStart { plane: "rs1", .. })
    ));

    let mut bad_terminal = owner.csr().clone().into_parts();
    bad_terminal.rs1_offsets[REGISTER_CSR_COLUMNS] = 0;
    assert!(matches!(
        RegisterCsr256::from_parts(bad_terminal),
        Err(RegisterOwnerError::OffsetOrder { plane: "rs1", .. }
            | RegisterOwnerError::OffsetTerminal { plane: "rs1", .. })
    ));

    let hot_rows = [
        RegisterOwnerRow {
            rs1: Some(RegisterOwnerRead {
                register: 0,
                value: 0,
            }),
            ..RegisterOwnerRow::default()
        },
        RegisterOwnerRow {
            rs1: Some(RegisterOwnerRead {
                register: 0,
                value: 0,
            }),
            ..RegisterOwnerRow::default()
        },
    ];
    let hot = build_owner(&hot_rows, &[0; REGISTER_CSR_COLUMNS], 0);
    let mut bad_order = hot.csr().clone().into_parts();
    bad_order.rs1_positions[1] = 0;
    assert!(matches!(
        RegisterCsr256::from_parts(bad_order),
        Err(RegisterOwnerError::PositionOrder { plane: "rs1", .. })
    ));
}

#[test]
fn malformed_cross_block_state_is_rejected() {
    let rows: Vec<_> = (0..257)
        .map(|cycle| RegisterOwnerRow {
            rd: Some(RegisterOwnerWrite {
                register: 0,
                pre_value: cycle as u64,
                post_value: cycle as u64 + 1,
            }),
            ..RegisterOwnerRow::default()
        })
        .collect();
    let owner = build_owner(&rows, &[0; REGISTER_CSR_COLUMNS], 0);
    let mut parts = owner.csr().clone().into_parts();
    parts.start_values[REGISTER_CSR_COLUMNS] += 1;

    assert!(matches!(
        RegisterCsr256::from_parts(parts),
        Err(RegisterOwnerError::BlockStateMismatch {
            block: 1,
            register: 0,
            ..
        })
    ));
}

#[test]
fn read_and_write_pre_mismatches_fail_certification() {
    let initial_values = [9; REGISTER_CSR_COLUMNS];
    let bad_read = [RegisterOwnerRow {
        rs1: Some(RegisterOwnerRead {
            register: 4,
            value: 8,
        }),
        ..RegisterOwnerRow::default()
    }];
    assert!(matches!(
        CertifiedRegisterOwner::build(&bad_read, &initial_values, 0),
        Err(RegisterOwnerError::ReadValueMismatch {
            cycle: 0,
            access: "rs1",
            register: 4,
            expected: 9,
            got: 8,
        })
    ));

    let bad_pre = [RegisterOwnerRow {
        rd: Some(RegisterOwnerWrite {
            register: 4,
            pre_value: 8,
            post_value: 10,
        }),
        ..RegisterOwnerRow::default()
    }];
    assert!(matches!(
        CertifiedRegisterOwner::build(&bad_pre, &initial_values, 0),
        Err(RegisterOwnerError::WritePreValueMismatch {
            cycle: 0,
            register: 4,
            expected: 9,
            got: 8,
        })
    ));
}

#[test]
fn increment_cap_overflow_discards_partial_activity() {
    let rows: Vec<_> = (0..3)
        .map(|cycle| RegisterOwnerRow {
            rd: Some(RegisterOwnerWrite {
                register: 6,
                pre_value: cycle as u64,
                post_value: cycle as u64 + 1,
            }),
            ..RegisterOwnerRow::default()
        })
        .collect();
    let initial_values = [0; REGISTER_CSR_COLUMNS];
    let owner = build_owner(&rows, &initial_values, 1);

    assert_eq!(owner.rd_increment_activity().cap(), 1);
    assert_eq!(owner.rd_increment_activity().entries(), None);
    assert_eq!(owner.rd_increment_activity().nonzero_count(), 3);
    assert!(owner.rd_increment_activity().overflowed());
    assert_eq!(owner.csr().event_counts().rd, 3);
    assert_matches_oracle(&rows, &initial_values, 1, &owner);
}

#[test]
fn analytical_census_is_only_a_checked_storage_fixture() {
    let census = match REGISTER_CSR_NON_AUTHORITATIVE_LOG_T_26_CENSUS.validate() {
        Ok(census) => census,
        Err(error) => panic!("analytical census is malformed: {error}"),
    };
    assert_eq!(census.cycles(), 1 << 26);
    assert_eq!(census.block_count(), Ok(262_144));
    assert_eq!(census.block_columns(), Ok(33_554_432));
    assert_eq!(census.storage_bytes(), Ok(1_239_649_860));

    assert!(matches!(
        RegisterCsrCensus::new(
            1,
            RegisterEventCounts {
                rs1: 2,
                rs2: 0,
                rd: 0,
            }
        ),
        Err(RegisterOwnerError::InvalidCensusEventCount {
            plane: "rs1",
            cycles: 1,
            count: 2,
        })
    ));
}
