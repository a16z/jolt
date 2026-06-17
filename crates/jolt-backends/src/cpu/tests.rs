use crate::{
    BackendError, BackendKernelMetadata, BackendRelationId, BackendValueSlot, CommitmentBackend,
    CommitmentMode, CommitmentRequest, CommitmentRequestItem, CommitmentSlot, OpeningBackend,
    OpeningRlcComponent, OpeningRlcMaterializationRequest, SumcheckBackend,
    SumcheckEvaluationRequest, SumcheckInstanceRequest, SumcheckLinearProductQuery,
    SumcheckLinearProductRequest, SumcheckMaterializationRequest, SumcheckPrefixProductSumQuery,
    SumcheckPrefixProductSumRequest, SumcheckProductUniskipRequest, SumcheckProductUniskipRow,
    SumcheckRaPushforwardRequest, SumcheckRequest, SumcheckRowProductQuery,
    SumcheckRowProductRequest, SumcheckSlot, SumcheckSpartanOuterRemainderQuery,
    SumcheckSpartanOuterRemainderRequest, SumcheckSpartanOuterRemainderStateRequest,
    SumcheckSpartanOuterRow, SumcheckSpartanOuterUniskipQuery, SumcheckSpartanOuterUniskipRequest,
    SumcheckViewEvaluationRequest, SumcheckViewMaterializationRequest, TracePolynomialEmbedding,
};

use std::sync::Arc;

use jolt_claims::protocols::jolt::formulas::dimensions::TracePolynomialOrder;
use jolt_field::{AdditiveAccumulator, Fr, FromPrimitiveInt, RingAccumulator, WithAccumulator};
use jolt_openings::{mock::MockCommitmentScheme, CommitmentScheme};
use jolt_poly::{
    eq_index_msb, BindingOrder, EqPolynomial, MultilinearPoly, OneHotIndexOrder, Polynomial,
    TensorEqTable, UnivariatePoly,
};
use jolt_witness::{
    protocols::dory_assist::{
        DoryAssistCommittedColumn, DoryAssistCommittedPolynomial, DoryAssistNamespace,
        DoryAssistOperationFamily, DoryAssistWitness,
    },
    protocols::jolt_vm::JoltVmStage6Row,
    MaterializationPolicy, NamespaceId, OracleDescriptor, OracleKind, OracleRef, OracleViewRequest,
    PolynomialChunk, PolynomialChunkKind, PolynomialEncoding, PolynomialStream, PolynomialView,
    RetentionHint, ViewRequirement, WitnessDimensions, WitnessError, WitnessNamespace,
    WitnessProvider,
};

use super::read_write_matrix::{AddressMajorBindableEntry as _, AddressMajorMessageEntry as _};
use super::{
    eq, field, lagrange, poly, ra, read_write_matrix, schedule, split_eq, univariate, CpuBackend,
    CpuBackendConfig,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum TestNamespace {}

impl WitnessNamespace for TestNamespace {
    type ChallengeId = u8;
    type CommittedId = u8;
    type OpeningId = u8;
    type PublicId = u8;
    type VirtualId = u8;

    const ID: NamespaceId = NamespaceId::new("cpu_test");
}

#[test]
fn cpu_streaming_schedule_matches_core_window_examples() {
    use schedule::StreamingSchedule;

    let schedule = schedule::HalfSplitSchedule::new(20, 2);
    assert_eq!(schedule.num_rounds(), 20);
    assert_eq!(schedule.switch_over_point(), 10);
    assert_eq!(
        schedule_window_starts(&schedule),
        vec![0, 1, 3, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    );
    assert_eq!(schedule.num_unbound_vars(3), 5);
    assert_eq!(schedule.num_unbound_vars(7), 1);

    let large = schedule::HalfSplitSchedule::new(40, 2);
    assert_eq!(large.switch_over_point(), 20);
    assert_eq!(large.num_unbound_vars(8), 12);

    let linear = schedule::LinearOnlySchedule::new(8);
    assert_eq!(linear.switch_over_point(), 0);
    assert_eq!(schedule_window_starts(&linear), (0..8).collect::<Vec<_>>());
    assert!((schedule::HalfSplitSchedule::compute_window_ratio(2) - 1.71).abs() < 0.01);
    assert_eq!(schedule::HalfSplitSchedule::optimal_window_size(3, 2), 5);
}

fn schedule_window_starts<S: schedule::StreamingSchedule>(schedule: &S) -> Vec<usize> {
    (0..schedule.num_rounds())
        .filter(|&round| schedule.is_window_start(round))
        .collect()
}

#[test]
fn cpu_read_write_one_hot_coeff_lookup_binds_like_dense_reference() {
    let initial = [0, 5, 11, 17].map(Fr::from_u64);
    let challenge = Fr::from_u64(19);
    let mut table = read_write_matrix::OneHotCoeffTable::new(initial.to_vec());
    table.bind(challenge);

    assert_eq!(table.len(), 16);
    for odd in 0..initial.len() {
        for even in 0..initial.len() {
            let index = read_write_matrix::OneHotCoeffIndex((odd * initial.len() + even) as u16);
            let expected = initial[even] + challenge * (initial[odd] - initial[even]);
            assert_eq!(table[index], expected);
        }
    }

    let even = read_write_matrix::OneHotCoeffIndex(2);
    let odd = read_write_matrix::OneHotCoeffIndex(3);
    let bound_index =
        <read_write_matrix::OneHotCoeffIndex as read_write_matrix::OneHotCoeff<Fr>>::bind(
            Some(&even),
            Some(&odd),
            challenge,
            Some(&table),
        );
    assert_eq!(bound_index, read_write_matrix::OneHotCoeffIndex(50));

    let evals = <read_write_matrix::OneHotCoeffIndex as read_write_matrix::OneHotCoeff<Fr>>::evals(
        Some(&even),
        Some(&odd),
        Some(&table),
    );
    assert_eq!(evals, [table[even], table[odd] - table[even]]);
}

#[test]
fn cpu_read_write_one_hot_coeff_lookup_saturates_at_core_limit() {
    let initial = [0, 1, 2, 3].map(Fr::from_u64);
    let challenges = [7, 11, 13].map(Fr::from_u64);
    let mut table = read_write_matrix::OneHotCoeffTable::new(initial.to_vec());

    for challenge in challenges {
        table.bind(challenge);
    }

    assert_eq!(table.len(), 1 << 16);
    assert!(table.is_saturated());
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct TestCycleMajorEntry {
    row: usize,
    column: usize,
    value: Fr,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct TestAddressMajorEntry {
    row: usize,
    column: usize,
    value: Fr,
}

impl read_write_matrix::CycleMajorMatrixEntry<Fr> for TestCycleMajorEntry {
    fn row(&self) -> usize {
        self.row
    }

    fn column(&self) -> usize {
        self.column
    }

    fn bind_entries(
        even: Option<&Self>,
        odd: Option<&Self>,
        r: Fr,
        _: Option<&read_write_matrix::OneHotCoeffTable<Fr>>,
        _: Option<&read_write_matrix::OneHotCoeffTable<Fr>>,
    ) -> Self {
        match (even, odd) {
            (Some(even), Some(odd)) => Self {
                row: even.row / 2,
                column: even.column,
                value: even.value + r * (odd.value - even.value),
            },
            (Some(even), None) => Self {
                row: even.row / 2,
                column: even.column,
                value: (Fr::from_u64(1) - r) * even.value,
            },
            (None, Some(odd)) => Self {
                row: odd.row / 2,
                column: odd.column,
                value: r * odd.value,
            },
            (None, None) => unreachable!("test bind entries requires at least one entry"),
        }
    }
}

impl read_write_matrix::AddressMajorMatrixEntry<Fr> for TestAddressMajorEntry {
    fn row(&self) -> usize {
        self.row
    }

    fn column(&self) -> usize {
        self.column
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct TestAddressMajorValEntry {
    row: usize,
    column: usize,
    prev_val: Fr,
    next_val: Fr,
    val_coeff: Fr,
    ra_coeff: Fr,
}

impl read_write_matrix::AddressMajorMatrixEntry<Fr> for TestAddressMajorValEntry {
    fn row(&self) -> usize {
        self.row
    }

    fn column(&self) -> usize {
        self.column
    }
}

impl read_write_matrix::AddressMajorBindableEntry<Fr> for TestAddressMajorValEntry {
    fn prev_val(&self) -> Fr {
        self.prev_val
    }

    fn next_val(&self) -> Fr {
        self.next_val
    }

    fn bind_entries(
        even: Option<&Self>,
        odd: Option<&Self>,
        even_checkpoint: Fr,
        odd_checkpoint: Fr,
        r: Fr,
    ) -> Self {
        match (even, odd) {
            (Some(even), Some(odd)) => Self {
                row: even.row,
                column: even.column / 2,
                prev_val: even.prev_val + r * (odd.prev_val - even.prev_val),
                next_val: even.next_val + r * (odd.next_val - even.next_val),
                val_coeff: even.val_coeff + r * (odd.val_coeff - even.val_coeff),
                ra_coeff: even.ra_coeff + r * (odd.ra_coeff - even.ra_coeff),
            },
            (Some(even), None) => Self {
                row: even.row,
                column: even.column / 2,
                prev_val: even.prev_val + r * (odd_checkpoint - even.prev_val),
                next_val: even.next_val + r * (odd_checkpoint - even.next_val),
                val_coeff: even.val_coeff + r * (odd_checkpoint - even.val_coeff),
                ra_coeff: (Fr::from_u64(1) - r) * even.ra_coeff,
            },
            (None, Some(odd)) => Self {
                row: odd.row,
                column: odd.column / 2,
                prev_val: even_checkpoint + r * (odd.prev_val - even_checkpoint),
                next_val: even_checkpoint + r * (odd.next_val - even_checkpoint),
                val_coeff: even_checkpoint + r * (odd.val_coeff - even_checkpoint),
                ra_coeff: r * odd.ra_coeff,
            },
            (None, None) => unreachable!("address-major bind requires at least one entry"),
        }
    }
}

impl read_write_matrix::AddressMajorMessageEntry<Fr> for TestAddressMajorValEntry {
    fn accumulate_evals(
        even: Option<&Self>,
        odd: Option<&Self>,
        inputs: read_write_matrix::AddressMajorMessageInputs<Fr>,
        accumulators: &mut [<Fr as WithAccumulator>::Accumulator; 2],
    ) {
        let read_write_matrix::AddressMajorMessageInputs {
            even_checkpoint,
            odd_checkpoint,
            inc_eval,
            eq_eval,
            gamma,
        } = inputs;
        let (ra_evals, val_evals) = match (even, odd) {
            (Some(even), Some(odd)) => (
                [even.ra_coeff, odd.ra_coeff + odd.ra_coeff - even.ra_coeff],
                [
                    even.val_coeff,
                    odd.val_coeff + odd.val_coeff - even.val_coeff,
                ],
            ),
            (Some(even), None) => (
                [even.ra_coeff, -even.ra_coeff],
                [
                    even.val_coeff,
                    odd_checkpoint + odd_checkpoint - even.val_coeff,
                ],
            ),
            (None, Some(odd)) => (
                [Fr::from_u64(0), odd.ra_coeff + odd.ra_coeff],
                [
                    even_checkpoint,
                    odd.val_coeff + odd.val_coeff - even_checkpoint,
                ],
            ),
            (None, None) => unreachable!("address-major message requires at least one entry"),
        };
        accumulators[0].fmadd(
            eq_eval,
            ra_evals[0] * (val_evals[0] + gamma * (inc_eval + val_evals[0])),
        );
        accumulators[1].fmadd(
            eq_eval,
            ra_evals[1] * (val_evals[1] + gamma * (inc_eval + val_evals[1])),
        );
    }
}

impl read_write_matrix::CycleMajorToAddressMajor<Fr> for TestCycleMajorEntry {
    type AddressMajor = TestAddressMajorEntry;

    fn to_address_major(
        self,
        _: Option<&read_write_matrix::OneHotCoeffTable<Fr>>,
        _: Option<&read_write_matrix::OneHotCoeffTable<Fr>>,
    ) -> Self::AddressMajor {
        TestAddressMajorEntry {
            row: self.row,
            column: self.column,
            value: self.value,
        }
    }
}

impl read_write_matrix::CycleMajorMessageEntry<Fr> for TestCycleMajorEntry {
    fn accumulate_evals(
        even: Option<&Self>,
        odd: Option<&Self>,
        inc_evals: [Fr; 2],
        gamma: Fr,
        accumulators: &mut [<Fr as WithAccumulator>::Accumulator; 2],
        _: Option<&read_write_matrix::OneHotCoeffTable<Fr>>,
        _: Option<&read_write_matrix::OneHotCoeffTable<Fr>>,
    ) {
        let [eval_at_zero, eval_slope] = test_cycle_entry_evals(even, odd);
        accumulators[0].fmadd(eval_at_zero, inc_evals[0] + gamma);
        accumulators[1].fmadd(eval_slope, inc_evals[1] + gamma);
    }
}

#[test]
fn cpu_read_write_cycle_major_bind_merges_sorted_sparse_rows() {
    let challenge = Fr::from_u64(5);
    let entries = vec![
        test_cycle_major_entry(0, 1, 10),
        test_cycle_major_entry(0, 3, 30),
        test_cycle_major_entry(0, 7, 70),
        test_cycle_major_entry(1, 3, 130),
        test_cycle_major_entry(1, 4, 140),
        test_cycle_major_entry(1, 7, 170),
        test_cycle_major_entry(2, 2, 220),
        test_cycle_major_entry(2, 5, 250),
        test_cycle_major_entry(3, 1, 310),
        test_cycle_major_entry(3, 5, 350),
        test_cycle_major_entry(3, 9, 390),
    ];
    let mut matrix = read_write_matrix::ReadWriteMatrixCycleMajor {
        entries,
        ra_lookup_table: Some(read_write_matrix::OneHotCoeffTable::new(
            [0, 1, 2, 3].map(Fr::from_u64).to_vec(),
        )),
        wa_lookup_table: Some(read_write_matrix::OneHotCoeffTable::new(
            [4, 5, 6, 7].map(Fr::from_u64).to_vec(),
        )),
    };

    matrix.bind(challenge);

    assert_eq!(
        matrix.entries,
        vec![
            test_cycle_major_value(0, 1, (Fr::from_u64(1) - challenge) * Fr::from_u64(10)),
            test_cycle_major_value(
                0,
                3,
                Fr::from_u64(30) + challenge * (Fr::from_u64(130) - Fr::from_u64(30))
            ),
            test_cycle_major_value(0, 4, challenge * Fr::from_u64(140)),
            test_cycle_major_value(
                0,
                7,
                Fr::from_u64(70) + challenge * (Fr::from_u64(170) - Fr::from_u64(70))
            ),
            test_cycle_major_value(1, 1, challenge * Fr::from_u64(310)),
            test_cycle_major_value(1, 2, (Fr::from_u64(1) - challenge) * Fr::from_u64(220)),
            test_cycle_major_value(
                1,
                5,
                Fr::from_u64(250) + challenge * (Fr::from_u64(350) - Fr::from_u64(250))
            ),
            test_cycle_major_value(1, 9, challenge * Fr::from_u64(390)),
        ]
    );
    assert_eq!(
        matrix.ra_lookup_table.as_ref().map(|table| table.len()),
        Some(16)
    );
    assert_eq!(
        matrix.wa_lookup_table.as_ref().map(|table| table.len()),
        Some(16)
    );
}

#[test]
fn cpu_read_write_cycle_major_message_contribution_matches_dense_reference() {
    let entries = vec![
        test_cycle_major_entry(0, 1, 10),
        test_cycle_major_entry(0, 3, 30),
        test_cycle_major_entry(0, 7, 70),
        test_cycle_major_entry(1, 3, 130),
        test_cycle_major_entry(1, 4, 140),
        test_cycle_major_entry(1, 7, 170),
    ];
    let matrix = read_write_matrix::ReadWriteMatrixCycleMajor::new(entries);
    let (even_row, odd_row) = matrix.entries.split_at(3);
    let inc_evals = [Fr::from_u64(11), Fr::from_u64(13)];
    let gamma = Fr::from_u64(17);

    let actual = matrix.prover_message_contribution(even_row, odd_row, inc_evals, gamma);
    let expected = [
        ((Fr::from_u64(10) * (inc_evals[0] + gamma))
            + (Fr::from_u64(30) * (inc_evals[0] + gamma))
            + (Fr::from_u64(70) * (inc_evals[0] + gamma))),
        ((-Fr::from_u64(10) * (inc_evals[1] + gamma))
            + ((Fr::from_u64(130) - Fr::from_u64(30)) * (inc_evals[1] + gamma))
            + (Fr::from_u64(140) * (inc_evals[1] + gamma))
            + ((Fr::from_u64(170) - Fr::from_u64(70)) * (inc_evals[1] + gamma))),
    ];

    assert_eq!(actual, expected);
}

#[test]
fn cpu_read_write_cycle_to_address_major_sorts_by_column_then_row() {
    let cycle_major = read_write_matrix::ReadWriteMatrixCycleMajor {
        entries: vec![
            test_cycle_major_entry(0, 4, 40),
            test_cycle_major_entry(0, 7, 70),
            test_cycle_major_entry(1, 1, 110),
            test_cycle_major_entry(1, 4, 140),
            test_cycle_major_entry(2, 0, 200),
            test_cycle_major_entry(2, 7, 270),
        ],
        ra_lookup_table: Some(read_write_matrix::OneHotCoeffTable::new(
            [0, 1, 2, 3].map(Fr::from_u64).to_vec(),
        )),
        wa_lookup_table: None,
    };

    let address_major: read_write_matrix::ReadWriteMatrixAddressMajor<Fr, TestAddressMajorEntry> =
        cycle_major.into();

    assert_eq!(
        address_major.entries,
        vec![
            test_address_major_entry(2, 0, 200),
            test_address_major_entry(1, 1, 110),
            test_address_major_entry(0, 4, 40),
            test_address_major_entry(1, 4, 140),
            test_address_major_entry(0, 7, 70),
            test_address_major_entry(2, 7, 270),
        ]
    );
}

#[test]
fn cpu_read_write_address_major_bind_uses_checkpoints() {
    let r = Fr::from_u64(7);
    let entries = vec![
        test_address_major_val_entry(0, 0, 10, 11, 12, 1),
        test_address_major_val_entry(2, 0, 20, 21, 22, 2),
        test_address_major_val_entry(0, 1, 30, 31, 32, 3),
        test_address_major_val_entry(1, 1, 40, 41, 42, 4),
        test_address_major_val_entry(3, 1, 50, 51, 52, 5),
        test_address_major_val_entry(0, 2, 60, 61, 62, 6),
        test_address_major_val_entry(1, 3, 70, 71, 72, 7),
    ];
    let val_init = [100, 200, 300, 400].map(Fr::from_u64).to_vec();
    let mut matrix =
        read_write_matrix::ReadWriteMatrixAddressMajor::new_with_val_init(entries, val_init);

    matrix.bind(r);

    let expected = vec![
        TestAddressMajorValEntry::bind_entries(
            Some(&test_address_major_val_entry(0, 0, 10, 11, 12, 1)),
            Some(&test_address_major_val_entry(0, 1, 30, 31, 32, 3)),
            Fr::from_u64(100),
            Fr::from_u64(200),
            r,
        ),
        TestAddressMajorValEntry::bind_entries(
            None,
            Some(&test_address_major_val_entry(1, 1, 40, 41, 42, 4)),
            Fr::from_u64(11),
            Fr::from_u64(31),
            r,
        ),
        TestAddressMajorValEntry::bind_entries(
            Some(&test_address_major_val_entry(2, 0, 20, 21, 22, 2)),
            None,
            Fr::from_u64(11),
            Fr::from_u64(41),
            r,
        ),
        TestAddressMajorValEntry::bind_entries(
            None,
            Some(&test_address_major_val_entry(3, 1, 50, 51, 52, 5)),
            Fr::from_u64(21),
            Fr::from_u64(41),
            r,
        ),
        TestAddressMajorValEntry::bind_entries(
            Some(&test_address_major_val_entry(0, 2, 60, 61, 62, 6)),
            None,
            Fr::from_u64(300),
            Fr::from_u64(400),
            r,
        ),
        TestAddressMajorValEntry::bind_entries(
            None,
            Some(&test_address_major_val_entry(1, 3, 70, 71, 72, 7)),
            Fr::from_u64(61),
            Fr::from_u64(400),
            r,
        ),
    ];
    assert_eq!(matrix.entries, expected);
    assert_eq!(
        matrix.val_init,
        vec![
            Fr::from_u64(100) + r * (Fr::from_u64(200) - Fr::from_u64(100)),
            Fr::from_u64(300) + r * (Fr::from_u64(400) - Fr::from_u64(300)),
        ]
    );
}

#[test]
fn cpu_read_write_address_major_message_matches_sequential_reference() {
    let even_col = vec![
        test_address_major_val_entry(0, 0, 10, 11, 12, 1),
        test_address_major_val_entry(2, 0, 20, 21, 22, 2),
    ];
    let odd_col = vec![
        test_address_major_val_entry(0, 1, 30, 31, 32, 3),
        test_address_major_val_entry(1, 1, 40, 41, 42, 4),
        test_address_major_val_entry(3, 1, 50, 51, 52, 5),
    ];
    let inc = [3, 5, 7, 11].map(Fr::from_u64);
    let eq = [13, 17, 19, 23].map(Fr::from_u64);
    let gamma = Fr::from_u64(29);

    let actual = read_write_matrix::ReadWriteMatrixAddressMajor::<
        Fr,
        TestAddressMajorValEntry,
    >::prover_message_contribution(
        &even_col,
        &odd_col,
        Fr::from_u64(100),
        Fr::from_u64(200),
        &inc,
        &eq,
        gamma,
    );
    let expected = address_major_message_reference(
        &even_col,
        &odd_col,
        Fr::from_u64(100),
        Fr::from_u64(200),
        &inc,
        &eq,
        gamma,
    );
    assert_eq!(actual, expected);
}

fn test_cycle_major_entry(row: usize, column: usize, value: u64) -> TestCycleMajorEntry {
    test_cycle_major_value(row, column, Fr::from_u64(value))
}

fn test_cycle_major_value(row: usize, column: usize, value: Fr) -> TestCycleMajorEntry {
    TestCycleMajorEntry { row, column, value }
}

fn test_address_major_entry(row: usize, column: usize, value: u64) -> TestAddressMajorEntry {
    TestAddressMajorEntry {
        row,
        column,
        value: Fr::from_u64(value),
    }
}

fn test_address_major_val_entry(
    row: usize,
    column: usize,
    prev_val: u64,
    next_val: u64,
    val_coeff: u64,
    ra_coeff: u64,
) -> TestAddressMajorValEntry {
    TestAddressMajorValEntry {
        row,
        column,
        prev_val: Fr::from_u64(prev_val),
        next_val: Fr::from_u64(next_val),
        val_coeff: Fr::from_u64(val_coeff),
        ra_coeff: Fr::from_u64(ra_coeff),
    }
}

fn address_major_message_reference(
    even: &[TestAddressMajorValEntry],
    odd: &[TestAddressMajorValEntry],
    mut even_checkpoint: Fr,
    mut odd_checkpoint: Fr,
    inc: &[Fr],
    eq: &[Fr],
    gamma: Fr,
) -> [Fr; 2] {
    let mut i = 0;
    let mut j = 0;
    let mut accumulators = [<Fr as WithAccumulator>::Accumulator::default(); 2];
    while i < even.len() && j < odd.len() {
        match even[i].row.cmp(&odd[j].row) {
            std::cmp::Ordering::Equal => {
                TestAddressMajorValEntry::accumulate_evals(
                    Some(&even[i]),
                    Some(&odd[j]),
                    read_write_matrix::AddressMajorMessageInputs {
                        even_checkpoint,
                        odd_checkpoint,
                        inc_eval: inc[even[i].row],
                        eq_eval: eq[even[i].row],
                        gamma,
                    },
                    &mut accumulators,
                );
                even_checkpoint = even[i].next_val;
                odd_checkpoint = odd[j].next_val;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => {
                TestAddressMajorValEntry::accumulate_evals(
                    Some(&even[i]),
                    None,
                    read_write_matrix::AddressMajorMessageInputs {
                        even_checkpoint,
                        odd_checkpoint,
                        inc_eval: inc[even[i].row],
                        eq_eval: eq[even[i].row],
                        gamma,
                    },
                    &mut accumulators,
                );
                even_checkpoint = even[i].next_val;
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                TestAddressMajorValEntry::accumulate_evals(
                    None,
                    Some(&odd[j]),
                    read_write_matrix::AddressMajorMessageInputs {
                        even_checkpoint,
                        odd_checkpoint,
                        inc_eval: inc[odd[j].row],
                        eq_eval: eq[odd[j].row],
                        gamma,
                    },
                    &mut accumulators,
                );
                odd_checkpoint = odd[j].next_val;
                j += 1;
            }
        }
    }
    for entry in &even[i..] {
        TestAddressMajorValEntry::accumulate_evals(
            Some(entry),
            None,
            read_write_matrix::AddressMajorMessageInputs {
                even_checkpoint,
                odd_checkpoint,
                inc_eval: inc[entry.row],
                eq_eval: eq[entry.row],
                gamma,
            },
            &mut accumulators,
        );
    }
    for entry in &odd[j..] {
        TestAddressMajorValEntry::accumulate_evals(
            None,
            Some(entry),
            read_write_matrix::AddressMajorMessageInputs {
                even_checkpoint,
                odd_checkpoint,
                inc_eval: inc[entry.row],
                eq_eval: eq[entry.row],
                gamma,
            },
            &mut accumulators,
        );
    }
    accumulators.map(AdditiveAccumulator::reduce)
}

fn test_cycle_entry_evals(
    even: Option<&TestCycleMajorEntry>,
    odd: Option<&TestCycleMajorEntry>,
) -> [Fr; 2] {
    match (even, odd) {
        (Some(even), Some(odd)) => [even.value, odd.value - even.value],
        (Some(even), None) => [even.value, -even.value],
        (None, Some(odd)) => [Fr::from_u64(0), odd.value],
        (None, None) => unreachable!("message contribution requires at least one entry"),
    }
}

#[test]
fn cpu_poly_compact_first_bind_matches_dense_reference() {
    let challenge = Fr::from_u64(37);
    let u8_coeffs = (0..64)
        .map(|index| ((index * 13 + 7) % 251) as u8)
        .collect::<Vec<_>>();
    let i64_coeffs = (0..64)
        .map(|index| index as i64 * 17 - 300)
        .collect::<Vec<_>>();
    let bool_coeffs = (0..64).map(|index| index % 3 == 0).collect::<Vec<_>>();

    assert_eq!(
        poly::bind_compact_first_high_to_low::<_, Fr>(&u8_coeffs, challenge),
        Polynomial::new(u8_coeffs.clone())
            .bind_to_field::<Fr>(challenge)
            .into_evals()
    );
    assert_eq!(
        poly::bind_compact_first_high_to_low::<_, Fr>(&i64_coeffs, challenge),
        Polynomial::new(i64_coeffs.clone())
            .bind_to_field::<Fr>(challenge)
            .into_evals()
    );
    assert_eq!(
        poly::bind_compact_first_high_to_low::<_, Fr>(&bool_coeffs, challenge),
        Polynomial::new(bool_coeffs.clone())
            .bind_to_field::<Fr>(challenge)
            .into_evals()
    );

    assert_eq!(
        poly::bind_compact_first_low_to_high::<_, Fr>(&[3u8, 5, 5, 3, 10, 10, 250, 1], challenge),
        {
            let mut reference = Polynomial::new(
                [3u8, 5, 5, 3, 10, 10, 250, 1]
                    .into_iter()
                    .map(Fr::from_u8)
                    .collect(),
            );
            reference.bind_with_order(challenge, BindingOrder::LowToHigh);
            reference.into_evals()
        }
    );
}

#[test]
fn cpu_poly_bound_field_bind_matches_dense_reference() {
    let challenge = Fr::from_u64(91);
    let values = (0..128)
        .map(|index| Fr::from_u64(10_000 + index as u64 * 19))
        .collect::<Vec<_>>();

    let mut high = values.clone();
    poly::bind_field_high_to_low(&mut high, challenge);
    let mut high_reference = Polynomial::new(values.clone());
    high_reference.bind_with_order(challenge, BindingOrder::HighToLow);
    assert_eq!(high, high_reference.into_evals());

    let low = poly::bind_field_low_to_high(&values, challenge);
    let mut low_reference = Polynomial::new(values);
    low_reference.bind_with_order(challenge, BindingOrder::LowToHigh);
    assert_eq!(low, low_reference.into_evals());
}

#[test]
fn cpu_ra_delayed_materialization_matches_dense_reference() {
    for order in [BindingOrder::LowToHigh, BindingOrder::HighToLow] {
        let indices = ra_indices(10, 64);
        let eq_evals = ra_eq_evals(64);
        let mut delayed =
            ra::RaPolynomial::<u8, Fr>::new(Arc::new(indices.clone()), eq_evals.clone());
        let mut dense = Polynomial::new(ra_dense_coeffs(&indices, &eq_evals));
        let challenges = [Fr::from_u64(17), Fr::from_u64(29), Fr::from_u64(43)];

        assert_ra_matches_dense(&delayed, &dense, order);
        for challenge in challenges {
            delayed.bind_parallel(challenge, order);
            dense.bind_with_order(challenge, order);
            assert_ra_matches_dense(&delayed, &dense, order);
        }

        let tail_challenge = Fr::from_u64(59);
        delayed.bind_parallel(tail_challenge, order);
        dense.bind_with_order(tail_challenge, order);
        assert_ra_matches_dense(&delayed, &dense, order);
    }
}

#[test]
fn cpu_ra_final_sumcheck_claim_matches_dense_tail() {
    let indices = ra_indices(6, 16);
    let eq_evals = ra_eq_evals(16);
    let mut delayed = ra::RaPolynomial::<u8, Fr>::new(Arc::new(indices.clone()), eq_evals.clone());
    let mut dense = Polynomial::new(ra_dense_coeffs(&indices, &eq_evals));
    let challenges = [
        Fr::from_u64(3),
        Fr::from_u64(5),
        Fr::from_u64(7),
        Fr::from_u64(11),
        Fr::from_u64(13),
        Fr::from_u64(17),
    ];

    for challenge in challenges {
        delayed.bind_parallel(challenge, BindingOrder::LowToHigh);
        dense.bind_with_order(challenge, BindingOrder::LowToHigh);
    }

    assert_eq!(delayed.len(), 1);
    assert_eq!(delayed.final_sumcheck_claim(), Some(dense.evaluations()[0]));
}

#[test]
fn cpu_shared_ra_delayed_materialization_matches_dense_reference() {
    for order in [BindingOrder::LowToHigh, BindingOrder::HighToLow] {
        let layout = shared_ra_layout();
        let indices = shared_ra_indices(9, layout);
        let tables = shared_ra_tables(layout);
        let mut shared = ra::SharedRaPolynomials::new(tables.clone(), indices.clone(), layout);
        let mut dense = shared_ra_dense_polys(&indices, &tables, layout);
        let challenges = [Fr::from_u64(17), Fr::from_u64(29), Fr::from_u64(43)];

        assert_shared_ra_matches_dense(&shared, &dense, "initial");
        for (round, challenge) in challenges.into_iter().enumerate() {
            shared.bind_in_place(challenge, order);
            for poly in &mut dense {
                poly.bind_with_order(challenge, order);
            }
            assert_shared_ra_matches_dense(&shared, &dense, &format!("{order:?} round {round}"));
        }

        let tail_challenge = Fr::from_u64(59);
        shared.bind_in_place(tail_challenge, order);
        for poly in &mut dense {
            poly.bind_with_order(tail_challenge, order);
        }
        assert_shared_ra_matches_dense(&shared, &dense, &format!("{order:?} tail"));
    }
}

#[test]
fn cpu_shared_ra_final_sumcheck_claim_matches_dense_tail() {
    let layout = shared_ra_layout();
    let indices = shared_ra_indices(6, layout);
    let tables = shared_ra_tables(layout);
    let mut shared = ra::SharedRaPolynomials::new(tables.clone(), indices.clone(), layout);
    let mut dense = shared_ra_dense_polys(&indices, &tables, layout);

    for challenge in [
        Fr::from_u64(3),
        Fr::from_u64(5),
        Fr::from_u64(7),
        Fr::from_u64(11),
        Fr::from_u64(13),
        Fr::from_u64(17),
    ] {
        shared.bind_in_place(challenge, BindingOrder::LowToHigh);
        for poly in &mut dense {
            poly.bind_with_order(challenge, BindingOrder::LowToHigh);
        }
    }

    assert_eq!(shared.len(), 1);
    for (poly_idx, dense_poly) in dense.iter().enumerate().take(layout.num_polys()) {
        assert_eq!(
            shared.final_sumcheck_claim(poly_idx),
            Some(dense_poly.evaluations()[0])
        );
    }
}

#[test]
fn cpu_ra_pushforward_matches_dense_reference() {
    let layout = shared_ra_layout();
    let indices = shared_ra_indices(10, layout);
    let r_cycle = (0..10)
        .map(|index| Fr::from_u64(101 + index as u64 * 13))
        .collect::<Vec<_>>();

    assert_eq!(
        ra::pushforward_indices(&indices, layout, &r_cycle),
        dense_ra_pushforward(&indices, layout, &r_cycle)
    );
}

#[derive(Clone, Debug)]
struct RaPushforwardTestWitness {
    columns: Vec<Vec<Option<usize>>>,
}

impl WitnessProvider<Fr, TestNamespace> for RaPushforwardTestWitness {
    fn describe_oracle(
        &self,
        _oracle: OracleRef<TestNamespace>,
    ) -> Result<OracleDescriptor<TestNamespace>, WitnessError> {
        Err(WitnessError::UnsupportedView {
            view: "ra pushforward test describe",
        })
    }

    fn view_requirements(
        &self,
        _oracle: OracleRef<TestNamespace>,
    ) -> Result<Vec<ViewRequirement<TestNamespace>>, WitnessError> {
        Err(WitnessError::UnsupportedView {
            view: "ra pushforward test view requirements",
        })
    }

    fn oracle_view(
        &self,
        _request: OracleViewRequest<TestNamespace>,
    ) -> Result<PolynomialView<'_, Fr, TestNamespace>, WitnessError> {
        Err(WitnessError::UnsupportedView {
            view: "ra pushforward test oracle view",
        })
    }

    fn committed_stream<'a>(
        &'a self,
        id: u8,
        _chunk_size: usize,
    ) -> Result<Box<dyn PolynomialStream<Fr> + 'a>, WitnessError>
    where
        TestNamespace: 'a,
    {
        let column = self
            .columns
            .get(usize::from(id))
            .ok_or(WitnessError::UnknownOracle {
                namespace: TestNamespace::ID.name,
            })?;
        Ok(Box::new(TestFieldStream {
            chunks: vec![PolynomialChunk::OneHot(column.clone())].into_iter(),
        }))
    }
}

#[test]
fn cpu_ra_pushforward_kernel_matches_dense_reference_via_committed_stream() -> Result<(), String> {
    let layout = ra::RaFamilyLayout::new(16, 2, 1, 1);
    let log_len = 6;
    let indices = shared_ra_indices(log_len, layout);
    let r_cycle = (0..log_len)
        .map(|index| Fr::from_u64(101 + index as u64 * 13))
        .collect::<Vec<_>>();

    let num_polys = layout.num_polys();
    let mut columns = vec![Vec::with_capacity(indices.len()); num_polys];
    for row in &indices {
        for (poly_idx, column) in columns.iter_mut().enumerate() {
            column.push(row.get_index(poly_idx, layout).map(usize::from));
        }
    }
    let witness = RaPushforwardTestWitness { columns };

    let request = SumcheckRaPushforwardRequest::<Fr, TestNamespace>::new(
        "test.ra_pushforward",
        vec![0u8, 1u8],
        vec![2u8],
        vec![3u8],
        4,
        r_cycle.clone(),
        1usize << log_len,
    );

    let mut backend = CpuBackend::default();
    let g =
        <CpuBackend as SumcheckBackend<Fr, TestNamespace>>::materialize_sumcheck_ra_pushforward(
            &mut backend,
            &request,
            &witness,
        )
        .map_err(|error| error.to_string())?;

    assert_eq!(g, dense_ra_pushforward(&indices, layout, &r_cycle));
    Ok(())
}

fn ra_indices(log_len: usize, k: usize) -> Vec<Option<u8>> {
    (0..(1usize << log_len))
        .map(|index| {
            if index % 11 == 0 {
                None
            } else {
                Some(((index * 37 + (index >> 2) * 11 + 5) % k) as u8)
            }
        })
        .collect()
}

fn ra_eq_evals(k: usize) -> Vec<Fr> {
    (0..k)
        .map(|index| Fr::from_u64(1_000 + index as u64 * 19))
        .collect()
}

fn ra_dense_coeffs(indices: &[Option<u8>], eq_evals: &[Fr]) -> Vec<Fr> {
    indices
        .iter()
        .map(|index| index.map_or(Fr::from_u64(0), |i| eq_evals[usize::from(i)]))
        .collect()
}

fn dense_ra_pushforward(
    indices: &[ra::RaCycleIndices],
    layout: ra::RaFamilyLayout,
    r_cycle: &[Fr],
) -> Vec<Vec<Fr>> {
    let eq = EqPolynomial::<Fr>::evals(r_cycle, None);
    let mut result = vec![vec![Fr::from_u64(0); layout.k_chunk]; layout.num_polys()];
    for (row, &eq_eval) in indices.iter().zip(eq.iter()) {
        for (poly_idx, result_table) in result.iter_mut().enumerate() {
            if let Some(index) = row.get_index(poly_idx, layout) {
                result_table[usize::from(index)] += eq_eval;
            }
        }
    }
    result
}

fn shared_ra_layout() -> ra::RaFamilyLayout {
    ra::RaFamilyLayout::new(16, 32, 6, 8)
}

fn shared_ra_indices(log_len: usize, layout: ra::RaFamilyLayout) -> Vec<ra::RaCycleIndices> {
    (0..(1usize << log_len))
        .map(|row| {
            let mut instruction = [0u8; ra::MAX_INSTRUCTION_CHUNKS];
            let mut bytecode = [0u8; ra::MAX_BYTECODE_CHUNKS];
            let mut ram = [None; ra::MAX_RAM_CHUNKS];
            for (chunk, value) in instruction
                .iter_mut()
                .enumerate()
                .take(layout.instruction_chunks)
            {
                *value = ((row * 13 + chunk * 5 + (row >> 2)) % layout.k_chunk) as u8;
            }
            for (chunk, value) in bytecode.iter_mut().enumerate().take(layout.bytecode_chunks) {
                *value = ((row * 7 + chunk * 11 + (row >> 3)) % layout.k_chunk) as u8;
            }
            for (chunk, value) in ram.iter_mut().enumerate().take(layout.ram_chunks) {
                if (row + chunk) % 5 != 0 {
                    *value = Some(((row * 3 + chunk * 17 + (row >> 1)) % layout.k_chunk) as u8);
                }
            }
            ra::RaCycleIndices {
                instruction,
                bytecode,
                ram,
            }
        })
        .collect()
}

fn shared_ra_tables(layout: ra::RaFamilyLayout) -> Vec<Vec<Fr>> {
    (0..layout.num_polys())
        .map(|poly_idx| {
            (0..layout.k_chunk)
                .map(|entry| Fr::from_u64(2_000 + poly_idx as u64 * 101 + entry as u64 * 19))
                .collect()
        })
        .collect()
}

fn shared_ra_dense_polys(
    indices: &[ra::RaCycleIndices],
    tables: &[Vec<Fr>],
    layout: ra::RaFamilyLayout,
) -> Vec<Polynomial<Fr>> {
    (0..layout.num_polys())
        .map(|poly_idx| {
            Polynomial::new(
                indices
                    .iter()
                    .map(|row| {
                        row.get_index(poly_idx, layout)
                            .map_or(Fr::from_u64(0), |k| tables[poly_idx][usize::from(k)])
                    })
                    .collect(),
            )
        })
        .collect()
}

fn assert_shared_ra_matches_dense(
    shared: &ra::SharedRaPolynomials<Fr>,
    dense: &[Polynomial<Fr>],
    label: &str,
) {
    assert_eq!(shared.num_polys(), dense.len());
    assert_eq!(shared.len(), dense[0].len());
    for (poly_idx, dense_poly) in dense.iter().enumerate() {
        for (row, &expected) in dense_poly.evaluations().iter().enumerate() {
            assert_eq!(
                shared.get_bound_coeff(poly_idx, row),
                expected,
                "{label}: poly={poly_idx} row={row}"
            );
        }
    }
}

fn assert_ra_matches_dense(
    delayed: &ra::RaPolynomial<u8, Fr>,
    dense: &Polynomial<Fr>,
    order: BindingOrder,
) {
    assert_eq!(delayed.len(), dense.len());
    for index in 0..dense.len() {
        assert_eq!(delayed.get_bound_coeff(index), dense.evaluations()[index]);
    }

    if dense.len() < 2 {
        return;
    }
    for index in 0..(dense.len() / 2).min(8) {
        let (lo, hi) = dense.sumcheck_eval_pair(index, order);
        let slope = hi - lo;
        assert_eq!(
            delayed.sumcheck_evals(index, 4, order),
            vec![
                lo,
                hi + slope,
                hi + slope + slope,
                hi + slope + slope + slope
            ]
        );
    }
}

#[test]
fn cpu_poly_split_eq_evaluate_matches_nested_reference() {
    for log_vars in [4usize, 16] {
        let split = log_vars / 2;
        let point = (0..log_vars)
            .map(|index| Fr::from_u64(101 + index as u64 * 17))
            .collect::<Vec<_>>();
        let eq_one = eq::evals(&point[..split], None);
        let eq_two = eq::evals(&point[split..], None);
        let values = (0..(1usize << log_vars))
            .map(|index| match index % 11 {
                0 => Fr::from_u64(0),
                1 => Fr::from_u64(1),
                _ => Fr::from_u64(1_000 + index as u64 * 13),
            })
            .collect::<Vec<_>>();
        let compact_values = (0..(1usize << log_vars))
            .map(|index| match index % 11 {
                0 => 0u8,
                1 => 1u8,
                _ => ((index * 19 + 7) % 251) as u8,
            })
            .collect::<Vec<_>>();

        let dense_expected = nested_split_eq_reference(&values, &eq_one, &eq_two);
        assert_eq!(
            poly::dense_split_eq_evaluate(&values, log_vars, &eq_one, &eq_two),
            dense_expected
        );

        let compact_as_field = compact_values
            .iter()
            .map(|&value| Fr::from_u8(value))
            .collect::<Vec<_>>();
        let compact_expected = nested_split_eq_reference(&compact_as_field, &eq_one, &eq_two);
        assert_eq!(
            poly::compact_split_eq_evaluate::<_, Fr>(&compact_values, log_vars, &eq_one, &eq_two),
            compact_expected
        );
    }
}

fn nested_split_eq_reference(values: &[Fr], eq_one: &[Fr], eq_two: &[Fr]) -> Fr {
    (0..eq_one.len()).fold(Fr::from_u64(0), |outer_acc, x1| {
        let partial_sum = (0..eq_two.len()).fold(Fr::from_u64(0), |inner_acc, x2| {
            let idx = x1 * eq_two.len() + x2;
            inner_acc + values[idx] * eq_two[x2]
        });
        outer_acc + partial_sum * eq_one[x1]
    })
}

#[test]
fn cpu_poly_inside_out_evaluate_matches_dense_reference() {
    for log_vars in [4usize, 16] {
        let point = (0..log_vars)
            .map(|index| Fr::from_u64(701 + index as u64 * 31))
            .collect::<Vec<_>>();
        let values = (0..(1usize << log_vars))
            .map(|index| match index % 13 {
                0 => Fr::from_u64(0),
                1 => Fr::from_u64(1),
                _ => Fr::from_u64(5_000 + index as u64 * 23),
            })
            .collect::<Vec<_>>();
        let compact_values = (0..(1usize << log_vars))
            .map(|index| match index % 13 {
                0 => 0u8,
                1 => 1u8,
                _ => ((index * 29 + 3) % 251) as u8,
            })
            .collect::<Vec<_>>();

        let dense_expected = Polynomial::new(values.clone()).evaluate(&point);
        assert_eq!(
            poly::dense_inside_out_evaluate(&values, &point),
            dense_expected
        );

        let compact_as_field = compact_values
            .iter()
            .map(|&value| Fr::from_u8(value))
            .collect::<Vec<_>>();
        let compact_expected = Polynomial::new(compact_as_field).evaluate(&point);
        assert_eq!(
            poly::compact_inside_out_evaluate::<_, Fr>(&compact_values, &point),
            compact_expected
        );
    }
}

#[test]
fn cpu_poly_dense_batch_evaluate_matches_individual_dense_reference() {
    const NUM_POLYS: usize = 5;
    for log_vars in [4usize, 16] {
        let split = log_vars / 2;
        let point = (0..log_vars)
            .map(|index| Fr::from_u64(901 + index as u64 * 37))
            .collect::<Vec<_>>();
        let polys = (0..NUM_POLYS)
            .map(|poly_index| {
                (0..(1usize << log_vars))
                    .map(|row| match (row + poly_index) % 17 {
                        0 => Fr::from_u64(0),
                        1 => Fr::from_u64(1),
                        _ => Fr::from_u64(8_000 + row as u64 * 13 + poly_index as u64 * 101),
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let poly_refs = polys.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let expected = polys
            .iter()
            .map(|values| Polynomial::new(values.clone()).evaluate(&point))
            .collect::<Vec<_>>();

        assert_eq!(poly::dense_batch_evaluate(&poly_refs, &point), expected);

        let eq_one = eq::evals(&point[..split], None);
        let eq_two = eq::evals(&point[split..], None);
        assert_eq!(
            poly::dense_batch_split_eq_evaluate(&poly_refs, &eq_one, &eq_two),
            expected
        );
    }
}

#[test]
fn cpu_poly_dense_dot_product_low_optimized_matches_reference() {
    let left = (0..4096)
        .map(|index| match index % 13 {
            0 => Fr::from_u64(0),
            1 => Fr::from_u64(1),
            _ => Fr::from_u64(10_000 + index as u64 * 17),
        })
        .collect::<Vec<_>>();
    let right = (0..4096)
        .map(|index| match index % 17 {
            0 => Fr::from_u64(0),
            1 => Fr::from_u64(1),
            _ => Fr::from_u64(20_000 + index as u64 * 19),
        })
        .collect::<Vec<_>>();
    let expected = left
        .iter()
        .zip(right.iter())
        .fold(Fr::from_u64(0), |acc, (&left, &right)| acc + left * right);
    assert_eq!(
        poly::dense_dot_product_low_optimized(&left, &right),
        expected
    );
}

#[test]
fn cpu_poly_linear_combination_matches_mixed_reference() {
    let dense = (0..64)
        .map(|index| match index % 11 {
            0 => Fr::from_u64(0),
            1 => Fr::from_u64(1),
            _ => Fr::from_u64(30_000 + index as u64 * 17),
        })
        .collect::<Vec<_>>();
    let u8_values = (0..64)
        .map(|index| ((index * 13 + 5) % 251) as u8)
        .collect::<Vec<_>>();
    let i64_values = (0..32)
        .map(|index| index as i64 * 19 - 300)
        .collect::<Vec<_>>();
    let coefficients = vec![Fr::from_u64(3), Fr::from_u64(5), Fr::from_u64(7)];
    let inputs = vec![
        poly::LinearCombinationInput::Dense(dense.as_slice()),
        poly::LinearCombinationInput::U8(u8_values.as_slice()),
        poly::LinearCombinationInput::I64(i64_values.as_slice()),
    ];

    let actual = poly::linear_combination(&inputs, &coefficients);
    let expected = (0..64)
        .map(|index| {
            let mut acc = dense[index] * coefficients[0];
            acc += Fr::from_u8(u8_values[index]) * coefficients[1];
            if let Some(value) = i64_values.get(index) {
                acc += Fr::from_i64(*value) * coefficients[2];
            }
            acc
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

#[test]
fn cpu_poly_one_hot_evaluate_matches_column_major_reference() {
    for (k, rows) in [(16usize, 64usize), (256, 512)] {
        let point_len = k.trailing_zeros() as usize + rows.trailing_zeros() as usize;
        let point = (0..point_len)
            .map(|index| Fr::from_u64(40_000 + index as u64 * 31))
            .collect::<Vec<_>>();
        let indices = (0..rows)
            .map(|row| {
                if row % 7 == 0 {
                    None
                } else {
                    Some(((row * 13 + row / 3 + 5) % k) as u8)
                }
            })
            .collect::<Vec<_>>();
        let reference = jolt_poly::OneHotPolynomial::new_with_index_order(
            k,
            indices.clone(),
            jolt_poly::OneHotIndexOrder::ColumnMajor,
        );

        assert_eq!(
            poly::one_hot_evaluate(k, &indices, &point),
            reference.evaluate(&point)
        );
    }
}

#[test]
fn cpu_poly_one_hot_vector_matrix_product_matches_dense_reference() {
    for (k, rows, num_columns, index_order) in [
        (
            16usize,
            256usize,
            64usize,
            jolt_poly::OneHotIndexOrder::ColumnMajor,
        ),
        (
            16usize,
            64usize,
            32usize,
            jolt_poly::OneHotIndexOrder::RowMajor,
        ),
    ] {
        let indices = (0..rows)
            .map(|row| {
                if row % 7 == 0 {
                    None
                } else {
                    Some(((row * 13 + row / 3 + 5) % k) as u8)
                }
            })
            .collect::<Vec<_>>();
        let left = (0..(k * rows / num_columns))
            .map(|index| Fr::from_u64(50_000 + index as u64 * 17))
            .collect::<Vec<_>>();
        let coeff = Fr::from_u64(19);
        let one_hot =
            jolt_poly::OneHotPolynomial::new_with_index_order(k, indices.clone(), index_order);
        let sigma = num_columns.trailing_zeros() as usize;
        let expected = one_hot
            .fold_rows(&left, sigma)
            .into_iter()
            .map(|value| coeff * value)
            .collect::<Vec<_>>();

        assert_eq!(
            poly::one_hot_vector_matrix_product(
                k,
                &indices,
                &left,
                coeff,
                num_columns,
                index_order
            ),
            expected
        );
    }
}

#[test]
fn cpu_poly_materialized_rlc_vector_matrix_product_matches_dense_reference() {
    let k = 16usize;
    let rows = 256usize;
    let num_columns = 64usize;
    let left = (0..(k * rows / num_columns))
        .map(|index| Fr::from_u64(60_000 + index as u64 * 19))
        .collect::<Vec<_>>();
    let dense_rlc = (0..rows)
        .map(|index| match index % 13 {
            0 => Fr::from_u64(0),
            1 => Fr::from_u64(1),
            _ => Fr::from_u64(70_000 + index as u64 * 23),
        })
        .collect::<Vec<_>>();
    let column_major_indices = (0..rows)
        .map(|row| {
            if row % 7 == 0 {
                None
            } else {
                Some(((row * 13 + row / 3 + 5) % k) as u8)
            }
        })
        .collect::<Vec<_>>();
    let row_major_indices = (0..rows)
        .map(|row| {
            if row % 11 == 0 {
                None
            } else {
                Some(((row * 5 + row / 7 + 3) % k) as u8)
            }
        })
        .collect::<Vec<_>>();
    let column_coeff = Fr::from_u64(29);
    let row_coeff = Fr::from_u64(31);
    let one_hot_rlc = [
        poly::OneHotVectorMatrixProductInput {
            k,
            indices: &column_major_indices,
            coefficient: column_coeff,
            index_order: jolt_poly::OneHotIndexOrder::ColumnMajor,
        },
        poly::OneHotVectorMatrixProductInput {
            k,
            indices: &row_major_indices,
            coefficient: row_coeff,
            index_order: jolt_poly::OneHotIndexOrder::RowMajor,
        },
    ];

    let actual =
        poly::materialized_rlc_vector_matrix_product(&dense_rlc, &one_hot_rlc, &left, num_columns);

    let mut expected = vec![Fr::from_u64(0); num_columns];
    for (index, &value) in dense_rlc.iter().enumerate() {
        expected[index % num_columns] += value * left[index / num_columns];
    }
    add_one_hot_vmp_reference(
        k,
        &column_major_indices,
        &left,
        column_coeff,
        num_columns,
        jolt_poly::OneHotIndexOrder::ColumnMajor,
        &mut expected,
    );
    add_one_hot_vmp_reference(
        k,
        &row_major_indices,
        &left,
        row_coeff,
        num_columns,
        jolt_poly::OneHotIndexOrder::RowMajor,
        &mut expected,
    );

    assert_eq!(actual, expected);
}

#[test]
fn cpu_poly_stage8_streaming_rlc_vmp_matches_dense_reference() {
    let log_t = 8usize;
    let rows = 1usize << log_t;
    let committed_chunk_bits = 4usize;
    let num_columns = 64usize;
    let rows_per_address = rows / num_columns;
    let address_factors = EqPolynomial::new(
        (0..committed_chunk_bits)
            .map(|index| Fr::from_u64(9 + index as u64 * 7))
            .collect(),
    )
    .evaluations();
    let row_factors = EqPolynomial::new(
        (0..rows_per_address.ilog2() as usize)
            .map(|index| Fr::from_u64(31 + index as u64 * 5))
            .collect(),
    )
    .evaluations();
    let left = address_factors
        .iter()
        .flat_map(|&address| row_factors.iter().map(move |&row| address * row))
        .collect::<Vec<_>>();
    let stage_rows = (0..rows)
        .map(|row| JoltVmStage6Row {
            instruction_lookup_index: (row * 257 + 19) as u128,
            bytecode_index: row * 17 + 3,
            remapped_ram_address: (row % 5 != 0).then_some(row * 11 + 7),
            ram_access_nonzero: row % 5 != 0,
            ram_increment: match row % 7 {
                0 => -9,
                1 => 0,
                _ => row as i128 - 50,
            },
            rd_increment: match row % 11 {
                0 => 0,
                _ => 70 - row as i128,
            },
        })
        .collect::<Vec<_>>();
    let instruction_coefficients = [Fr::from_u64(3), Fr::from_u64(5), Fr::from_u64(7)];
    let bytecode_coefficients = [Fr::from_u64(11), Fr::from_u64(13)];
    let ram_coefficients = [Fr::from_u64(17), Fr::from_u64(19), Fr::from_u64(23)];
    let ram_inc_coefficient = Fr::from_u64(29);
    let rd_inc_coefficient = Fr::from_u64(31);

    let actual = poly::stage8_streaming_rlc_vector_matrix_product(
        poly::Stage8StreamingRlcVectorMatrixProductInput {
            rows: &stage_rows,
            field_rd_inc: None,
            log_t,
            committed_chunk_bits,
            trace_polynomial_order:
                jolt_claims::protocols::jolt::formulas::dimensions::TracePolynomialOrder::CycleMajor,
            ram_inc_coefficient,
            rd_inc_coefficient,
            field_rd_inc_coefficient: None,
            instruction_coefficients: &instruction_coefficients,
            bytecode_coefficients: &bytecode_coefficients,
            ram_coefficients: &ram_coefficients,
            left_vec: &left,
            num_columns,
        },
    );

    let mut expected = vec![Fr::from_u64(0); num_columns];
    for (cycle, row) in stage_rows.iter().enumerate() {
        let dense = ram_inc_coefficient * Fr::from_i128(row.ram_increment)
            + rd_inc_coefficient * Fr::from_i128(row.rd_increment);
        add_stage8_flat_vmp_reference(cycle, dense, &left, &mut expected);
        add_stage8_ra_vmp_reference(
            cycle,
            row.instruction_lookup_index,
            &instruction_coefficients,
            committed_chunk_bits,
            rows,
            &left,
            &mut expected,
        );
        add_stage8_ra_vmp_reference(
            cycle,
            row.bytecode_index as u128,
            &bytecode_coefficients,
            committed_chunk_bits,
            rows,
            &left,
            &mut expected,
        );
        if let Some(address) = row.remapped_ram_address {
            add_stage8_ra_vmp_reference(
                cycle,
                address as u128,
                &ram_coefficients,
                committed_chunk_bits,
                rows,
                &left,
                &mut expected,
            );
        }
    }

    assert_eq!(actual, expected);
}

#[test]
fn cpu_field_linear_product_d4_matches_naive_reference() {
    let pairs = [
        (Fr::from_u64(3), Fr::from_u64(5)),
        (Fr::from_u64(7), Fr::from_u64(11)),
        (Fr::from_u64(13), Fr::from_u64(17)),
        (Fr::from_u64(19), Fr::from_u64(23)),
    ];
    let mut actual = [Fr::from_u64(0); 4];
    field::eval_linear_product_d4_assign(&pairs, &mut actual);
    assert_eq!(actual, naive_linear_product_grid(&pairs));
}

#[test]
fn cpu_field_linear_product_small_degrees_match_naive_reference() {
    let pairs2 = core::array::from_fn(|factor| {
        (
            Fr::from_u64(10_000 + factor as u64 * 3),
            Fr::from_u64(20_000 + factor as u64 * 5),
        )
    });
    let mut actual2 = [Fr::from_u64(0); 2];
    field::eval_linear_product_d2_assign(&pairs2, &mut actual2);
    assert_eq!(actual2, naive_linear_product_grid(&pairs2));

    let pairs3 = core::array::from_fn(|factor| {
        (
            Fr::from_u64(30_000 + factor as u64 * 7),
            Fr::from_u64(40_000 + factor as u64 * 11),
        )
    });
    let mut actual3 = [Fr::from_u64(0); 3];
    field::eval_linear_product_d3_assign(&pairs3, &mut actual3);
    assert_eq!(actual3, naive_linear_product_grid(&pairs3));

    let pairs5 = core::array::from_fn(|factor| {
        (
            Fr::from_u64(50_000 + factor as u64 * 13),
            Fr::from_u64(60_000 + factor as u64 * 17),
        )
    });
    let mut actual5 = [Fr::from_u64(0); 5];
    field::eval_linear_product_d5_assign(&pairs5, &mut actual5);
    assert_eq!(actual5, naive_linear_product_grid(&pairs5));

    let pairs6 = core::array::from_fn(|factor| {
        (
            Fr::from_u64(70_000 + factor as u64 * 19),
            Fr::from_u64(80_000 + factor as u64 * 23),
        )
    });
    let mut actual6 = [Fr::from_u64(0); 6];
    field::eval_linear_product_d6_assign(&pairs6, &mut actual6);
    assert_eq!(actual6, naive_linear_product_grid(&pairs6));

    let pairs7 = core::array::from_fn(|factor| {
        (
            Fr::from_u64(90_000 + factor as u64 * 29),
            Fr::from_u64(100_000 + factor as u64 * 31),
        )
    });
    let mut actual7 = [Fr::from_u64(0); 7];
    field::eval_linear_product_d7_assign(&pairs7, &mut actual7);
    assert_eq!(actual7, naive_linear_product_grid(&pairs7));
}

#[test]
fn cpu_field_linear_product_d8_matches_naive_reference() {
    let pairs = core::array::from_fn(|factor| {
        (
            Fr::from_u64(30_000 + factor as u64 * 7),
            Fr::from_u64(40_000 + factor as u64 * 11),
        )
    });
    let mut actual = [Fr::from_u64(0); 8];
    field::eval_linear_product_d8_assign(&pairs, &mut actual);
    assert_eq!(actual, naive_linear_product_grid(&pairs));
}

#[test]
fn cpu_field_linear_product_d9_matches_naive_reference() {
    let pairs = core::array::from_fn(|factor| {
        (
            Fr::from_u64(40_000 + factor as u64 * 11),
            Fr::from_u64(60_000 + factor as u64 * 13),
        )
    });
    let mut actual = [Fr::from_u64(0); 9];
    field::eval_linear_product_d9_assign(&pairs, &mut actual);
    assert_eq!(actual, naive_linear_product_grid(&pairs));
}

#[test]
fn cpu_field_linear_product_d9_accumulate_matches_assign() {
    let pairs = core::array::from_fn(|factor| {
        (
            Fr::from_u64(120_000 + factor as u64 * 17),
            Fr::from_u64(140_000 + factor as u64 * 19),
        )
    });
    let mut expected = [Fr::from_u64(0); 9];
    field::eval_linear_product_d9_assign(&pairs, &mut expected);

    let mut actual = [<Fr as WithAccumulator>::Accumulator::default(); 9];
    field::eval_linear_product_d9_accumulate(&pairs, &mut actual);
    assert_eq!(actual.map(AdditiveAccumulator::reduce), expected);
}

#[test]
fn cpu_field_linear_product_d16_matches_naive_reference() {
    let pairs = core::array::from_fn(|factor| {
        (
            Fr::from_u64(50_000 + factor as u64 * 13),
            Fr::from_u64(70_000 + factor as u64 * 17),
        )
    });
    let mut actual = [Fr::from_u64(0); 16];
    field::eval_linear_product_d16_assign(&pairs, &mut actual);
    assert_eq!(actual, naive_linear_product_grid(&pairs));
}

#[test]
fn cpu_field_linear_product_d32_matches_naive_reference() {
    let pairs = core::array::from_fn(|factor| {
        (
            Fr::from_u64(110_000 + factor as u64 * 19),
            Fr::from_u64(130_000 + factor as u64 * 23),
        )
    });
    let mut actual = [Fr::from_u64(0); 32];
    field::eval_linear_product_d32_assign(&pairs, &mut actual);
    assert_eq!(actual, naive_linear_product_grid(&pairs));
}

#[test]
fn cpu_field_accumulate_linear_product_d4_matches_individual_sum() {
    let products = (0..64)
        .map(|product| {
            core::array::from_fn(|factor| {
                (
                    Fr::from_u64(80_000 + product as u64 * 17 + factor as u64 * 3),
                    Fr::from_u64(90_000 + product as u64 * 19 + factor as u64 * 5),
                )
            })
        })
        .collect::<Vec<[(Fr, Fr); 4]>>();

    let actual = field::accumulate_linear_product_d4(&products);
    let expected = products.iter().map(naive_linear_product_grid).fold(
        [Fr::from_u64(0); 4],
        |mut acc, evals| {
            for (acc, eval) in acc.iter_mut().zip(evals) {
                *acc += eval;
            }
            acc
        },
    );
    assert_eq!(actual, expected);
}

fn naive_linear_product_grid<const D: usize>(pairs: &[(Fr, Fr); D]) -> [Fr; D] {
    let mut outputs = [Fr::from_u64(0); D];
    for (point, output) in (1..D).zip(outputs[..D - 1].iter_mut()) {
        let x = Fr::from_u64(point as u64);
        *output = pairs
            .iter()
            .fold(Fr::from_u64(1), |acc, &(p0, p1)| acc * (p0 + (p1 - p0) * x));
    }
    outputs[D - 1] = pairs
        .iter()
        .fold(Fr::from_u64(1), |acc, &(p0, p1)| acc * (p1 - p0));
    outputs
}

fn add_one_hot_vmp_reference(
    k: usize,
    indices: &[Option<u8>],
    left: &[Fr],
    coeff: Fr,
    num_columns: usize,
    index_order: jolt_poly::OneHotIndexOrder,
    result: &mut [Fr],
) {
    for (cycle, &address) in indices.iter().enumerate() {
        if let Some(address) = address {
            let flat = match index_order {
                jolt_poly::OneHotIndexOrder::RowMajor => cycle * k + usize::from(address),
                jolt_poly::OneHotIndexOrder::ColumnMajor => {
                    usize::from(address) * indices.len() + cycle
                }
            };
            result[flat % num_columns] += coeff * left[flat / num_columns];
        }
    }
}

fn add_stage8_ra_vmp_reference(
    cycle: usize,
    value: u128,
    coefficients: &[Fr],
    committed_chunk_bits: usize,
    trace_rows: usize,
    left: &[Fr],
    result: &mut [Fr],
) {
    for (index, &coefficient) in coefficients.iter().enumerate() {
        let hot = stage8_ra_chunk_reference(value, index, coefficients.len(), committed_chunk_bits);
        add_stage8_flat_vmp_reference(hot * trace_rows + cycle, coefficient, left, result);
    }
}

fn add_stage8_flat_vmp_reference(flat: usize, value: Fr, left: &[Fr], result: &mut [Fr]) {
    result[flat % result.len()] += value * left[flat / result.len()];
}

fn stage8_ra_chunk_reference(
    value: u128,
    index: usize,
    chunks: usize,
    committed_chunk_bits: usize,
) -> usize {
    let shift = (chunks - index - 1) * committed_chunk_bits;
    let mask = (1u128 << committed_chunk_bits) - 1;
    ((value >> shift) & mask) as usize
}

#[test]
fn cpu_eq_backend_materializes_core_ordered_tables() -> Result<(), String> {
    let point = (0..18)
        .map(|index| Fr::from_u64(1_001 + index * 17))
        .collect::<Vec<_>>();
    let scale = Fr::from_u64(19);

    let evals = eq::evals(&point, Some(scale));
    let reference = jolt_poly::EqPolynomial::<Fr>::evals(&point, Some(scale));
    assert_eq!(evals, reference);

    let cached = eq::evals_cached(&point, Some(scale));
    assert_eq!(cached.len(), point.len() + 1);
    assert_eq!(cached.last(), Some(&evals));
    for prefix_len in [0, 1, 4, 9, point.len()] {
        assert_eq!(
            cached[prefix_len],
            eq::evals(&point[..prefix_len], Some(scale))
        );
    }

    let cached_rev = eq::evals_cached_rev(&point, Some(scale));
    assert_eq!(cached_rev.len(), point.len() + 1);
    assert_eq!(cached_rev[0], vec![scale]);
    assert_eq!(cached_rev.last().map(Vec::len), Some(evals.len()));

    let values = (0..evals.len())
        .map(|index| Fr::from_u64(7_001 + index as u64))
        .collect::<Vec<_>>();
    let tensor_eval = eq::tensor_table(&point).evaluate_slices(&[&values]);
    let unscaled_evals = eq::evals(&point, None);
    let dense_eval = values
        .iter()
        .zip(&unscaled_evals)
        .fold(Fr::from_u64(0), |acc, (value, eq)| acc + *value * *eq);
    assert_eq!(tensor_eval, vec![dense_eval]);
    Ok(())
}

#[test]
fn cpu_eq_backend_materializes_aligned_blocks() {
    let point = (0..12)
        .map(|index| Fr::from_u64(2_003 + index * 29))
        .collect::<Vec<_>>();
    let full = eq::evals(&point, None);

    for (start, block_size) in [(0usize, 16usize), (512, 128), (1536, 512)] {
        let block = eq::evals_for_aligned_block(&point, start, block_size);
        assert_eq!(block, full[start..start + block_size]);
    }

    let mut cursor = 40usize;
    let end = 1_333usize;
    while cursor < end {
        let (block_size, block) = eq::evals_for_max_aligned_block(&point, cursor, end - cursor);
        assert_eq!(block, full[cursor..cursor + block_size]);
        cursor += block_size;
    }
    assert_eq!(cursor, end);
}

#[test]
fn cpu_split_eq_backend_materializes_streaming_window_tables() {
    let point = (0..9)
        .map(|index| Fr::from_u64(3_001 + index * 41))
        .collect::<Vec<_>>();
    let challenges = (0..point.len())
        .map(|index| Fr::from_u64(4_001 + index as u64 * 43))
        .collect::<Vec<_>>();
    let mut split = split_eq::gruen(&point, BindingOrder::LowToHigh);

    for (round, window_size) in [3usize, 4, 5].into_iter().enumerate() {
        let current_index = split.current_index();
        let (actual_out, actual_in) = split_eq::e_out_in_for_window(&split, window_size);
        let window_size = window_size.min(current_index);
        let head_len = current_index.saturating_sub(window_size);
        let split_index = point.len() / 2;
        let head_out_bits = head_len.min(split_index);
        let head_in_bits = head_len.saturating_sub(head_out_bits);
        let expected_out = eq::evals(&point[..head_out_bits], None);
        let expected_in = eq::evals(&point[split_index..split_index + head_in_bits], None);
        assert_eq!(actual_out, expected_out.as_slice(), "round {round} e_out");
        assert_eq!(actual_in, expected_in.as_slice(), "round {round} e_in");

        let expected_active = if window_size <= 1 {
            vec![Fr::from_u64(1)]
        } else {
            let remaining = &point[..current_index];
            let window_start = remaining.len() - window_size;
            let (_, window_point) = remaining.split_at(window_start);
            let (active_point, _) = window_point.split_at(window_size - 1);
            eq::evals(active_point, None)
        };
        assert_eq!(
            split_eq::e_active_for_window(&split, window_size),
            expected_active,
            "round {round} active"
        );

        split.bind(challenges[round]);
    }

    while split.current_index() > 1 {
        let challenge = challenges[point.len() - split.current_index()];
        split.bind(challenge);
    }
    let (actual_out, actual_in) = split_eq::e_out_in_for_window(&split, 8);
    assert_eq!(actual_out, &[Fr::from_u64(1)]);
    assert_eq!(actual_in, &[Fr::from_u64(1)]);
    assert_eq!(
        split_eq::e_active_for_window(&split, 8),
        vec![Fr::from_u64(1)]
    );
}

#[test]
fn cpu_univariate_backend_uses_direct_interpolation_paths() {
    let quadratic = vec![Fr::from_u64(1), Fr::from_u64(6), Fr::from_u64(15)];
    let quadratic_poly = univariate::from_evals(&quadratic);
    assert_eq!(
        quadratic_poly.coefficients(),
        &[Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(2)]
    );

    let cubic = vec![
        Fr::from_u64(1),
        Fr::from_u64(7),
        Fr::from_u64(23),
        Fr::from_u64(55),
    ];
    let cubic_poly = univariate::from_evals(&cubic);
    assert_eq!(
        cubic_poly.coefficients(),
        &[
            Fr::from_u64(1),
            Fr::from_u64(3),
            Fr::from_u64(2),
            Fr::from_u64(1),
        ]
    );

    let hint = cubic[0] + cubic[1];
    let hinted = univariate::from_evals_and_hint(hint, &[cubic[0], cubic[2], cubic[3]]);
    assert_eq!(hinted, cubic_poly);

    let toom = univariate::from_evals_toom(&[
        Fr::from_u64(1),
        Fr::from_u64(7),
        Fr::from_u64(23),
        Fr::from_u64(1),
    ]);
    assert_eq!(toom, cubic_poly);
}

#[test]
fn cpu_univariate_backend_compresses_round_polynomials() {
    let poly = UnivariatePoly::new(vec![
        Fr::from_u64(5),
        Fr::from_u64(7),
        Fr::from_u64(11),
        Fr::from_u64(13),
        Fr::from_u64(17),
    ]);
    let hint = poly.evaluate(Fr::from_u64(0)) + poly.evaluate(Fr::from_u64(1));
    let compressed = univariate::compress(&poly);
    assert_eq!(
        compressed.coeffs_except_linear_term(),
        &[
            Fr::from_u64(5),
            Fr::from_u64(11),
            Fr::from_u64(13),
            Fr::from_u64(17),
        ]
    );
    assert_eq!(univariate::decompress(&compressed, hint), poly);

    for point in [0, 1, 2, 7, 19].map(Fr::from_u64) {
        assert_eq!(
            univariate::eval_from_hint(&compressed, hint, point),
            poly.evaluate(point)
        );
    }
}

#[test]
fn cpu_lagrange_backend_evaluates_centered_domains() -> Result<(), String> {
    const N: usize = 10;
    let point = Fr::from_u64(37);
    let weights = lagrange::centered_evals::<Fr, N>(point).map_err(|error| error.to_string())?;
    let start = jolt_poly::lagrange::centered_domain_start(N).map_err(|error| error.to_string())?;
    let generic = jolt_poly::lagrange::lagrange_evals(start, N, point);
    assert_eq!(weights.as_slice(), generic.as_slice());
    assert_eq!(
        weights
            .iter()
            .copied()
            .fold(Fr::from_u64(0), |acc, value| acc + value),
        Fr::from_u64(1)
    );

    let other = Fr::from_u64(71);
    let other_weights =
        lagrange::centered_evals::<Fr, N>(other).map_err(|error| error.to_string())?;
    let expected_kernel = weights
        .iter()
        .zip(other_weights.iter())
        .fold(Fr::from_u64(0), |acc, (&left, &right)| acc + left * right);
    assert_eq!(
        lagrange::centered_kernel(N, point, other).map_err(|error| error.to_string())?,
        expected_kernel
    );

    let values = core::array::from_fn(|index| Fr::from_u64(5_001 + index as u64 * 17));
    let points = (0..(N + 3))
        .map(|index| Fr::from_u64(6_001 + index as u64 * 19))
        .collect::<Vec<_>>();
    let many = lagrange::centered_evaluate_many::<Fr, N>(&values, &points)
        .map_err(|error| error.to_string())?;
    let repeated = points
        .iter()
        .map(|&point| {
            lagrange::centered_evaluate::<Fr, N>(&values, point).map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(many, repeated);

    let coeffs = lagrange::centered_interpolate_coeffs::<Fr, N>(&values)
        .map_err(|error| error.to_string())?;
    for (offset, &expected) in values.iter().enumerate() {
        let x = Fr::from_i64(start + offset as i64);
        let mut actual = Fr::from_u64(0);
        let mut power = Fr::from_u64(1);
        for &coeff in &coeffs {
            actual += coeff * power;
            power *= x;
        }
        assert_eq!(actual, expected);
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct TestFieldWitness {
    encoding: PolynomialEncoding,
    dimensions: WitnessDimensions,
    chunks: Vec<PolynomialChunk<Fr>>,
}

struct TestFieldStream {
    chunks: std::vec::IntoIter<PolynomialChunk<Fr>>,
}

impl PolynomialStream<Fr> for TestFieldStream {
    fn next_chunk(&mut self) -> Result<Option<PolynomialChunk<Fr>>, WitnessError> {
        Ok(self.chunks.next())
    }
}

impl WitnessProvider<Fr, TestNamespace> for TestFieldWitness {
    fn describe_oracle(
        &self,
        oracle: OracleRef<TestNamespace>,
    ) -> Result<OracleDescriptor<TestNamespace>, WitnessError> {
        Ok(OracleDescriptor::new(
            oracle,
            self.dimensions,
            self.encoding,
        ))
    }

    fn view_requirements(
        &self,
        oracle: OracleRef<TestNamespace>,
    ) -> Result<Vec<ViewRequirement<TestNamespace>>, WitnessError> {
        Ok(vec![ViewRequirement::new(
            oracle,
            self.encoding,
            MaterializationPolicy::BackendChoice,
            RetentionHint::ThroughStage8,
        )])
    }

    fn oracle_view(
        &self,
        _request: OracleViewRequest<TestNamespace>,
    ) -> Result<PolynomialView<'_, Fr, TestNamespace>, WitnessError> {
        Err(WitnessError::UnsupportedView {
            view: "cpu test field oracle views",
        })
    }

    fn committed_stream<'a>(
        &'a self,
        _id: u8,
        _chunk_size: usize,
    ) -> Result<Box<dyn PolynomialStream<Fr> + 'a>, WitnessError>
    where
        TestNamespace: 'a,
    {
        Ok(Box::new(TestFieldStream {
            chunks: self.chunks.clone().into_iter(),
        }))
    }
}

#[derive(Clone, Debug)]
struct TestDenseViewWitness {
    encoding: PolynomialEncoding,
    dimensions: WitnessDimensions,
    values: Vec<Fr>,
}

#[derive(Clone, Debug)]
struct TestRlcViewWitness {
    dimensions: WitnessDimensions,
    values: Vec<Vec<Fr>>,
}

impl WitnessProvider<Fr, TestNamespace> for TestRlcViewWitness {
    fn describe_oracle(
        &self,
        oracle: OracleRef<TestNamespace>,
    ) -> Result<OracleDescriptor<TestNamespace>, WitnessError> {
        let OracleKind::Committed(id) = oracle.kind else {
            return Err(WitnessError::UnknownOracle {
                namespace: TestNamespace::ID.name,
            });
        };
        if usize::from(id) >= self.values.len() {
            return Err(WitnessError::UnknownOracle {
                namespace: TestNamespace::ID.name,
            });
        }
        Ok(OracleDescriptor::new(
            oracle,
            self.dimensions,
            PolynomialEncoding::Dense,
        ))
    }

    fn view_requirements(
        &self,
        oracle: OracleRef<TestNamespace>,
    ) -> Result<Vec<ViewRequirement<TestNamespace>>, WitnessError> {
        let _ = self.describe_oracle(oracle)?;
        Ok(vec![ViewRequirement::new(
            oracle,
            PolynomialEncoding::Dense,
            MaterializationPolicy::BackendChoice,
            RetentionHint::ThroughStage8,
        )])
    }

    fn oracle_view(
        &self,
        request: OracleViewRequest<TestNamespace>,
    ) -> Result<PolynomialView<'_, Fr, TestNamespace>, WitnessError> {
        let OracleKind::Committed(id) = request.oracle().kind else {
            return Err(WitnessError::UnknownOracle {
                namespace: TestNamespace::ID.name,
            });
        };
        let values = self
            .values
            .get(usize::from(id))
            .ok_or(WitnessError::UnknownOracle {
                namespace: TestNamespace::ID.name,
            })?;
        let descriptor = self.describe_oracle(request.oracle())?;
        Ok(PolynomialView::borrowed(descriptor, values))
    }
}

impl WitnessProvider<Fr, TestNamespace> for TestDenseViewWitness {
    fn describe_oracle(
        &self,
        oracle: OracleRef<TestNamespace>,
    ) -> Result<OracleDescriptor<TestNamespace>, WitnessError> {
        Ok(OracleDescriptor::new(
            oracle,
            self.dimensions,
            self.encoding,
        ))
    }

    fn view_requirements(
        &self,
        oracle: OracleRef<TestNamespace>,
    ) -> Result<Vec<ViewRequirement<TestNamespace>>, WitnessError> {
        Ok(vec![ViewRequirement::new(
            oracle,
            self.encoding,
            MaterializationPolicy::BackendChoice,
            RetentionHint::ThroughStage8,
        )])
    }

    fn oracle_view(
        &self,
        request: OracleViewRequest<TestNamespace>,
    ) -> Result<PolynomialView<'_, Fr, TestNamespace>, WitnessError> {
        let descriptor = self.describe_oracle(request.oracle())?;
        Ok(PolynomialView::borrowed(descriptor, &self.values))
    }
}

fn requirement(
    encoding: PolynomialEncoding,
    materialization: MaterializationPolicy,
) -> ViewRequirement<TestNamespace> {
    ViewRequirement::new(
        OracleRef::committed(7),
        encoding,
        materialization,
        RetentionHint::ThroughStage8,
    )
}

fn sumcheck_request(
    slot: SumcheckSlot,
    requirement: ViewRequirement<TestNamespace>,
) -> SumcheckRequest<TestNamespace> {
    SumcheckRequest::new(
        "cpu.test_sumcheck",
        vec![SumcheckInstanceRequest::new(
            slot,
            BackendRelationId::new(TestNamespace::ID.name, "test_relation"),
            vec![requirement],
            2,
            3,
            BackendValueSlot(1),
            BackendValueSlot(2),
        )],
    )
}

#[test]
fn cpu_sumcheck_request_builders_preserve_kernel_metadata() {
    const OPTIMIZATION_IDS: &[&str] = &["OPT-SC-007", "OPT-EQ-004"];

    let relation = BackendRelationId::new(TestNamespace::ID.name, "metadata_relation");
    let instance = SumcheckInstanceRequest::<TestNamespace>::new(
        SumcheckSlot(1),
        relation,
        Vec::new(),
        2,
        2,
        BackendValueSlot(1),
        BackendValueSlot(2),
    )
    .with_optimization_ids(OPTIMIZATION_IDS);

    assert_eq!(instance.relation, relation);
    assert_eq!(instance.optimization_ids, OPTIMIZATION_IDS);

    let witness_polynomials = Vec::<Vec<Fr>>::new();
    let input_columns = Vec::<usize>::new();
    let rows = Vec::<Vec<(usize, Fr)>>::new();
    let linear = SumcheckLinearProductRequest::new(
        "cpu.metadata_linear",
        &witness_polynomials,
        &input_columns,
        0,
        &rows,
        &rows,
        Vec::new(),
    )
    .with_relation(relation)
    .with_optimization_ids(OPTIMIZATION_IDS);

    assert_eq!(linear.kernel.relation, Some(relation));
    assert_eq!(linear.kernel.optimization_ids, OPTIMIZATION_IDS);

    let metadata = BackendKernelMetadata::empty()
        .with_relation(relation)
        .with_optimization_ids(OPTIMIZATION_IDS);
    let row = SumcheckRowProductRequest::new(
        "cpu.metadata_row",
        &witness_polynomials,
        &input_columns,
        0,
        &rows,
        &rows,
        Vec::new(),
    )
    .with_kernel_metadata(metadata);

    assert_eq!(row.kernel, metadata);
}

#[test]
fn cpu_sumcheck_backend_resolves_witness_views_by_instance_slot() -> Result<(), String> {
    let witness = TestFieldWitness {
        encoding: PolynomialEncoding::Dense,
        dimensions: WitnessDimensions::new(8, 3),
        chunks: Vec::new(),
    };
    let requirement = requirement(
        PolynomialEncoding::Dense,
        MaterializationPolicy::BackendChoice,
    );
    let request = sumcheck_request(SumcheckSlot(9), requirement);
    let mut backend = CpuBackend::default();

    let resolution = <CpuBackend as SumcheckBackend<Fr, TestNamespace>>::resolve_sumcheck_views(
        &mut backend,
        &request,
        &witness,
    )
    .map_err(|error| error.to_string())?;

    assert_eq!(resolution.resolved_witness.len(), 1);
    let resolved = &resolution.resolved_witness[0];
    assert_eq!(resolved.slot, SumcheckSlot(9));
    assert_eq!(resolved.view_index, 0);
    assert_eq!(resolved.requirement, requirement);
    assert_eq!(resolved.descriptor.reference, requirement.oracle);
    assert_eq!(resolved.descriptor.encoding, PolynomialEncoding::Dense);
    assert_eq!(resolved.descriptor.dimensions, WitnessDimensions::new(8, 3));
    Ok(())
}

#[test]
fn cpu_sumcheck_backend_rejects_mismatched_view_encoding() {
    let witness = TestFieldWitness {
        encoding: PolynomialEncoding::Compact,
        dimensions: WitnessDimensions::new(8, 3),
        chunks: Vec::new(),
    };
    let request = sumcheck_request(
        SumcheckSlot(9),
        requirement(
            PolynomialEncoding::Dense,
            MaterializationPolicy::BackendChoice,
        ),
    );
    let mut backend = CpuBackend::default();

    let result = <CpuBackend as SumcheckBackend<Fr, TestNamespace>>::resolve_sumcheck_views(
        &mut backend,
        &request,
        &witness,
    );

    assert!(matches!(
        result,
        Err(BackendError::InvalidRequest {
            task: "sumcheck view resolution",
            ..
        })
    ));
}

#[test]
fn cpu_sumcheck_backend_evaluates_materialized_dense_views() -> Result<(), String> {
    let values = vec![
        Fr::from_u64(2),
        Fr::from_u64(3),
        Fr::from_u64(5),
        Fr::from_u64(7),
    ];
    let point = vec![Fr::from_u64(11), Fr::from_u64(13)];
    let witness = TestDenseViewWitness {
        encoding: PolynomialEncoding::Dense,
        dimensions: WitnessDimensions::new(values.len(), 2),
        values: values.clone(),
    };
    let requirement = requirement(
        PolynomialEncoding::Dense,
        MaterializationPolicy::BackendChoice,
    );
    let request = SumcheckEvaluationRequest::new(
        "cpu.test_sumcheck_eval",
        vec![SumcheckViewEvaluationRequest::new(
            BackendValueSlot(9),
            requirement,
            point.clone(),
        )],
    );
    let mut backend = CpuBackend::default();

    let evaluations = <CpuBackend as SumcheckBackend<Fr, TestNamespace>>::evaluate_sumcheck_views(
        &mut backend,
        &request,
        &witness,
    )
    .map_err(|error| error.to_string())?;

    assert_eq!(evaluations.len(), 1);
    assert_eq!(evaluations[0].slot, BackendValueSlot(9));
    assert_eq!(evaluations[0].value, values.as_slice().evaluate(&point));
    Ok(())
}

#[test]
fn cpu_sumcheck_backend_evaluates_repeated_materialized_point_group() -> Result<(), String> {
    let values = vec![
        Fr::from_u64(2),
        Fr::from_u64(3),
        Fr::from_u64(5),
        Fr::from_u64(7),
        Fr::from_u64(11),
        Fr::from_u64(13),
        Fr::from_u64(17),
        Fr::from_u64(19),
    ];
    let point = vec![Fr::from_u64(23), Fr::from_u64(29), Fr::from_u64(31)];
    let witness = TestDenseViewWitness {
        encoding: PolynomialEncoding::Dense,
        dimensions: WitnessDimensions::new(values.len(), 3),
        values: values.clone(),
    };
    let requirement = requirement(
        PolynomialEncoding::Dense,
        MaterializationPolicy::BackendChoice,
    );
    let request = SumcheckEvaluationRequest::new(
        "cpu.test_grouped_sumcheck_eval",
        vec![
            SumcheckViewEvaluationRequest::new(BackendValueSlot(9), requirement, point.clone()),
            SumcheckViewEvaluationRequest::new(BackendValueSlot(10), requirement, point.clone()),
        ],
    );
    let mut backend = CpuBackend::default();

    let evaluations = <CpuBackend as SumcheckBackend<Fr, TestNamespace>>::evaluate_sumcheck_views(
        &mut backend,
        &request,
        &witness,
    )
    .map_err(|error| error.to_string())?;

    let expected = values.as_slice().evaluate(&point);
    assert_eq!(evaluations.len(), 2);
    assert_eq!(evaluations[0].slot, BackendValueSlot(9));
    assert_eq!(evaluations[0].value, expected);
    assert_eq!(evaluations[1].slot, BackendValueSlot(10));
    assert_eq!(evaluations[1].value, expected);
    Ok(())
}

#[test]
fn cpu_sumcheck_backend_materializes_dense_views_by_value_slot() -> Result<(), String> {
    let values = vec![
        Fr::from_u64(2),
        Fr::from_u64(3),
        Fr::from_u64(5),
        Fr::from_u64(7),
    ];
    let witness = TestDenseViewWitness {
        encoding: PolynomialEncoding::Dense,
        dimensions: WitnessDimensions::new(values.len(), 2),
        values: values.clone(),
    };
    let requirement = requirement(
        PolynomialEncoding::Dense,
        MaterializationPolicy::BackendChoice,
    );
    let request = SumcheckMaterializationRequest::new(
        "cpu.test_sumcheck_materialize",
        vec![SumcheckViewMaterializationRequest::new(
            BackendValueSlot(9),
            requirement,
        )],
    );
    let mut backend = CpuBackend::default();

    let materialized =
        <CpuBackend as SumcheckBackend<Fr, TestNamespace>>::materialize_sumcheck_views(
            &mut backend,
            &request,
            &witness,
        )
        .map_err(|error| error.to_string())?;

    assert_eq!(materialized.len(), 1);
    assert_eq!(materialized[0].slot, BackendValueSlot(9));
    assert_eq!(materialized[0].values, values);
    Ok(())
}

#[test]
fn cpu_opening_backend_materializes_rlc_from_dense_views() -> Result<(), String> {
    let values = vec![
        vec![
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(5),
            Fr::from_u64(7),
        ],
        vec![
            Fr::from_u64(11),
            Fr::from_u64(13),
            Fr::from_u64(17),
            Fr::from_u64(19),
        ],
    ];
    let witness = TestRlcViewWitness {
        dimensions: WitnessDimensions::new(4, 2),
        values: values.clone(),
    };
    let view0 = ViewRequirement::new(
        OracleRef::committed(0),
        PolynomialEncoding::Dense,
        MaterializationPolicy::BackendChoice,
        RetentionHint::ThroughStage8,
    );
    let view1 = ViewRequirement::new(
        OracleRef::committed(1),
        PolynomialEncoding::Dense,
        MaterializationPolicy::BackendChoice,
        RetentionHint::ThroughStage8,
    );
    let request = OpeningRlcMaterializationRequest::new(
        "cpu.test_opening_rlc_materialization",
        vec![
            OpeningRlcComponent::new(view0, Fr::from_u64(3)),
            OpeningRlcComponent::new(view1, Fr::from_u64(5)),
        ],
    );
    let mut backend = CpuBackend::default();

    let result =
        <CpuBackend as OpeningBackend<Fr, TestNamespace, MockCommitmentScheme<Fr>>>::materialize_opening_rlc(
            &mut backend,
            &request,
            &witness,
        )
        .map_err(|error| error.to_string())?;

    let expected = values[0]
        .iter()
        .zip(&values[1])
        .map(|(&left, &right)| Fr::from_u64(3) * left + Fr::from_u64(5) * right)
        .collect::<Vec<_>>();
    assert_eq!(result.values, expected);
    Ok(())
}

#[test]
fn cpu_sumcheck_backend_evaluates_linear_products_by_query_slot() -> Result<(), String> {
    let witness_polynomials = vec![
        vec![
            Fr::from_u64(1),
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(4),
        ],
        vec![
            Fr::from_u64(5),
            Fr::from_u64(6),
            Fr::from_u64(7),
            Fr::from_u64(8),
        ],
    ];
    let input_columns = vec![10, 20];
    let left_rows = vec![
        vec![(10, Fr::from_u64(2)), (0, Fr::from_u64(3))],
        vec![(20, Fr::from_u64(1))],
    ];
    let right_rows = vec![
        vec![(20, Fr::from_u64(1)), (0, Fr::from_i64(-1))],
        vec![(10, Fr::from_u64(4))],
    ];
    let point = vec![Fr::from_u64(13), Fr::from_u64(17)];
    let row_weights = vec![Fr::from_u64(7), Fr::from_u64(11)];
    let scale = Fr::from_u64(19);
    let request = SumcheckLinearProductRequest::new(
        "cpu.test_linear_product",
        &witness_polynomials,
        &input_columns,
        0,
        &left_rows,
        &right_rows,
        vec![SumcheckLinearProductQuery::new(
            BackendValueSlot(9),
            point.clone(),
            row_weights.clone(),
            scale,
        )],
    );
    let mut backend = CpuBackend::default();

    let outputs =
        <CpuBackend as SumcheckBackend<Fr, TestNamespace>>::evaluate_sumcheck_linear_products(
            &mut backend,
            &request,
        )
        .map_err(|error| error.to_string())?;

    let first = witness_polynomials[0].as_slice().evaluate(&point);
    let second = witness_polynomials[1].as_slice().evaluate(&point);
    let left =
        row_weights[0] * (Fr::from_u64(2) * first + Fr::from_u64(3)) + row_weights[1] * second;
    let right =
        row_weights[0] * (second - Fr::from_u64(1)) + row_weights[1] * Fr::from_u64(4) * first;

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].slot, BackendValueSlot(9));
    assert_eq!(outputs[0].value, scale * left * right);
    Ok(())
}

#[test]
fn cpu_sumcheck_backend_evaluates_prefix_product_sums_by_query_slot() -> Result<(), String> {
    let witness_polynomials = vec![
        vec![
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(5),
            Fr::from_u64(7),
        ],
        vec![
            Fr::from_u64(11),
            Fr::from_u64(13),
            Fr::from_u64(17),
            Fr::from_u64(19),
        ],
    ];
    let input_columns = vec![1, 2];
    let constant_column = 0;
    let left_rows = vec![vec![(1, Fr::from_u64(2)), (0, Fr::from_u64(3))]];
    let right_rows = vec![vec![(2, Fr::from_u64(5)), (0, Fr::from_u64(7))]];
    let eq_point = vec![Fr::from_u64(23), Fr::from_u64(29), Fr::from_u64(31)];
    let fixed_prefix = vec![Fr::from_u64(37), Fr::from_u64(41)];
    let row_weights_at_zero = vec![Fr::from_u64(43)];
    let row_weights_at_one = vec![Fr::from_u64(47)];
    let scale = Fr::from_u64(53);
    let request = SumcheckPrefixProductSumRequest::new(
        "cpu.test_prefix_product_sum",
        &witness_polynomials,
        &input_columns,
        constant_column,
        &left_rows,
        &right_rows,
        vec![SumcheckPrefixProductSumQuery::new(
            BackendValueSlot(12),
            eq_point.clone(),
            fixed_prefix.clone(),
            1,
            row_weights_at_zero.clone(),
            row_weights_at_one.clone(),
            scale,
        )],
    )
    .with_relation(BackendRelationId::new("jolt_vm", "spartan_outer.remainder"));
    let mut backend = CpuBackend::default();

    let outputs =
        <CpuBackend as SumcheckBackend<Fr, TestNamespace>>::evaluate_sumcheck_prefix_product_sums(
            &mut backend,
            &request,
        )
        .map_err(|error| error.to_string())?;

    let expected = (0..2)
        .map(|suffix| {
            let variables = [fixed_prefix[0], fixed_prefix[1], Fr::from_u64(suffix)];
            let eq = (0..3)
                .map(|position| {
                    let challenge = eq_point[2 - position];
                    let value = variables[position];
                    challenge * value + (Fr::from_u64(1) - challenge) * (Fr::from_u64(1) - value)
                })
                .product::<Fr>();
            let row_weight = row_weights_at_zero[0]
                + variables[0] * (row_weights_at_one[0] - row_weights_at_zero[0]);
            let cycle_point = vec![variables[2], variables[1]];
            let first = witness_polynomials[0].as_slice().evaluate(&cycle_point);
            let second = witness_polynomials[1].as_slice().evaluate(&cycle_point);
            let left = row_weight * (Fr::from_u64(2) * first + Fr::from_u64(3));
            let right = row_weight * (Fr::from_u64(5) * second + Fr::from_u64(7));
            scale * eq * left * right
        })
        .sum::<Fr>();

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].slot, BackendValueSlot(12));
    assert_eq!(outputs[0].value, expected);
    Ok(())
}

#[test]
fn cpu_spartan_outer_bound_prefix_group_matches_reference() -> Result<(), String> {
    let witness_polynomials = vec![
        vec![
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(5),
            Fr::from_u64(7),
        ],
        vec![
            Fr::from_u64(11),
            Fr::from_u64(13),
            Fr::from_u64(17),
            Fr::from_u64(19),
        ],
    ];
    let input_columns = vec![1, 2];
    let constant_column = 0;
    let left_rows = vec![
        vec![(1, Fr::from_u64(2)), (0, Fr::from_u64(3))],
        vec![(2, Fr::from_u64(5))],
    ];
    let right_rows = vec![
        vec![(2, Fr::from_u64(7)), (0, Fr::from_u64(11))],
        vec![(1, Fr::from_u64(13))],
    ];
    let eq_point = vec![Fr::from_u64(23), Fr::from_u64(29), Fr::from_u64(31)];
    let stream = Fr::from_u64(37);
    let row_weights_at_zero = vec![Fr::from_u64(41), Fr::from_u64(43)];
    let row_weights_at_one = vec![Fr::from_u64(47), Fr::from_u64(53)];
    let fixed_values = [Fr::from_u64(59), Fr::from_u64(61), Fr::from_u64(67)];
    let scales = [Fr::from_u64(71), Fr::from_u64(73), Fr::from_u64(79)];
    let queries = fixed_values
        .iter()
        .zip(scales)
        .enumerate()
        .map(|(index, (&fixed_value, scale))| {
            SumcheckPrefixProductSumQuery::new(
                BackendValueSlot(index as u32),
                eq_point.clone(),
                vec![stream, fixed_value],
                1,
                row_weights_at_zero.clone(),
                row_weights_at_one.clone(),
                scale,
            )
        })
        .collect::<Vec<_>>();
    let request = SumcheckPrefixProductSumRequest::new(
        "cpu.test_spartan_outer_bound_prefix_group",
        &witness_polynomials,
        &input_columns,
        constant_column,
        &left_rows,
        &right_rows,
        queries,
    )
    .with_relation(BackendRelationId::new("jolt_vm", "spartan_outer.remainder"));
    let mut backend = CpuBackend::default();

    let outputs =
        <CpuBackend as SumcheckBackend<Fr, TestNamespace>>::evaluate_sumcheck_prefix_product_sums(
            &mut backend,
            &request,
        )
        .map_err(|error| error.to_string())?;

    let expected = fixed_values
        .iter()
        .zip(scales)
        .map(|(&fixed_value, scale)| {
            (0..2)
                .map(|suffix| {
                    let variables = [stream, fixed_value, Fr::from_u64(suffix)];
                    let eq = (0..3)
                        .map(|position| {
                            let challenge = eq_point[2 - position];
                            let value = variables[position];
                            challenge * value
                                + (Fr::from_u64(1) - challenge) * (Fr::from_u64(1) - value)
                        })
                        .product::<Fr>();
                    let row_weights = row_weights_at_zero
                        .iter()
                        .zip(&row_weights_at_one)
                        .map(|(&at_zero, &at_one)| at_zero + stream * (at_one - at_zero))
                        .collect::<Vec<_>>();
                    let cycle_point = vec![variables[2], variables[1]];
                    let first = witness_polynomials[0].as_slice().evaluate(&cycle_point);
                    let second = witness_polynomials[1].as_slice().evaluate(&cycle_point);
                    let left = row_weights[0] * (Fr::from_u64(2) * first + Fr::from_u64(3))
                        + row_weights[1] * Fr::from_u64(5) * second;
                    let right = row_weights[0] * (Fr::from_u64(7) * second + Fr::from_u64(11))
                        + row_weights[1] * Fr::from_u64(13) * first;
                    scale * eq * left * right
                })
                .sum::<Fr>()
        })
        .collect::<Vec<_>>();

    assert_eq!(outputs.len(), expected.len());
    for (output, expected) in outputs.iter().zip(expected) {
        assert_eq!(output.value, expected);
    }
    Ok(())
}

#[test]
fn cpu_spartan_outer_raw_uniskip_rows_match_prefix_product_reference() -> Result<(), String> {
    use jolt_r1cs::constraints::{
        jolt::{
            spartan_outer_constraints, spartan_outer_opening_columns,
            SPARTAN_OUTER_FIRST_GROUP_ROWS, SPARTAN_OUTER_ROW_COUNT,
            SPARTAN_OUTER_SECOND_GROUP_ROWS, SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE,
        },
        rv64,
    };

    let rows = vec![
        noop_spartan_outer_row(),
        load_spartan_outer_row(),
        add_spartan_outer_row(),
        noop_spartan_outer_row(),
    ];
    let eq_point = vec![Fr::from_u64(17), Fr::from_u64(19), Fr::from_u64(23)];
    let targets = spartan_outer_uniskip_targets(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE);
    let raw_queries = targets
        .iter()
        .enumerate()
        .map(|(index, &target)| {
            SumcheckSpartanOuterUniskipQuery::new(
                BackendValueSlot((10 + index) as u32),
                eq_point.clone(),
                centered_lagrange_integer_coeffs(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, target),
                Fr::from_u64(1),
            )
        })
        .collect::<Vec<_>>();
    let raw_request = SumcheckSpartanOuterUniskipRequest::new(
        "cpu.test_spartan_outer.raw_uniskip",
        &rows,
        raw_queries,
    )
    .with_relation(BackendRelationId::new(
        "jolt_vm",
        "spartan_outer.uniskip_first_round",
    ));

    let witness_polynomials = spartan_outer_input_polynomials(&rows);
    let matrices = spartan_outer_constraints::<Fr>();
    let input_columns = spartan_outer_opening_columns();
    let prefix_queries = targets
        .iter()
        .enumerate()
        .map(|(index, &target)| {
            let coeffs =
                centered_lagrange_integer_coeffs(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, target);
            let mut at_zero = vec![Fr::from_u64(0); SPARTAN_OUTER_ROW_COUNT];
            let mut at_one = vec![Fr::from_u64(0); SPARTAN_OUTER_ROW_COUNT];
            for (coeff_index, &row) in SPARTAN_OUTER_FIRST_GROUP_ROWS.iter().enumerate() {
                at_zero[row] = Fr::from_i64(i64::from(coeffs[coeff_index]));
            }
            for (coeff_index, &row) in SPARTAN_OUTER_SECOND_GROUP_ROWS.iter().enumerate() {
                at_one[row] = Fr::from_i64(i64::from(coeffs[coeff_index]));
            }
            SumcheckPrefixProductSumQuery::new(
                BackendValueSlot((10 + index) as u32),
                eq_point.clone(),
                Vec::new(),
                eq_point.len(),
                at_zero,
                at_one,
                Fr::from_u64(1),
            )
        })
        .collect::<Vec<_>>();
    let prefix_request = SumcheckPrefixProductSumRequest::new(
        "cpu.test_spartan_outer.prefix_reference",
        &witness_polynomials,
        &input_columns,
        rv64::const_column(),
        &matrices.a,
        &matrices.b,
        prefix_queries,
    )
    .with_relation(BackendRelationId::new(
        "jolt_vm",
        "spartan_outer.uniskip_first_round",
    ));

    let mut backend = CpuBackend::default();
    let raw_outputs =
        <CpuBackend as SumcheckBackend<Fr, TestNamespace>>::evaluate_sumcheck_spartan_outer_uniskip_rows(
            &mut backend,
            &raw_request,
        )
        .map_err(|error| error.to_string())?;
    let reference_outputs =
        <CpuBackend as SumcheckBackend<Fr, TestNamespace>>::evaluate_sumcheck_prefix_product_sums(
            &mut backend,
            &prefix_request,
        )
        .map_err(|error| error.to_string())?;

    assert_eq!(raw_outputs, reference_outputs);
    Ok(())
}

#[test]
fn cpu_spartan_outer_remainder_rows_match_prefix_product_reference() -> Result<(), String> {
    use jolt_r1cs::constraints::{
        jolt::{
            spartan_outer_constraints, spartan_outer_opening_columns, spartan_outer_row_weights,
        },
        rv64,
    };

    let rows = vec![
        noop_spartan_outer_row(),
        load_spartan_outer_row(),
        add_spartan_outer_row(),
        noop_spartan_outer_row(),
    ];
    let witness_polynomials = spartan_outer_input_polynomials(&rows);
    let matrices = spartan_outer_constraints::<Fr>();
    let input_columns = spartan_outer_opening_columns();
    let eq_point = vec![Fr::from_u64(17), Fr::from_u64(19), Fr::from_u64(23)];
    let uniskip = Fr::from_u64(29);
    let row_weights_at_zero =
        spartan_outer_row_weights(uniskip, Fr::from_u64(0)).map_err(|error| error.to_string())?;
    let row_weights_at_one =
        spartan_outer_row_weights(uniskip, Fr::from_u64(1)).map_err(|error| error.to_string())?;

    for (fixed_prefixes, suffix_vars) in [
        (vec![vec![Fr::from_u64(31)], vec![Fr::from_u64(37)]], 2),
        (
            vec![
                vec![Fr::from_u64(41), Fr::from_u64(43)],
                vec![Fr::from_u64(41), Fr::from_u64(47)],
            ],
            1,
        ),
    ] {
        let row_queries = fixed_prefixes
            .iter()
            .enumerate()
            .map(|(index, fixed_prefix)| {
                SumcheckSpartanOuterRemainderQuery::new(
                    BackendValueSlot(index as u32),
                    eq_point.clone(),
                    fixed_prefix.clone(),
                    suffix_vars,
                    uniskip,
                    Fr::from_u64(53 + index as u64),
                )
                .with_uniskip_domain_size(
                    jolt_r1cs::constraints::jolt::SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE,
                )
            })
            .collect::<Vec<_>>();
        let row_request = SumcheckSpartanOuterRemainderRequest::new(
            "cpu.test_spartan_outer.remainder_rows",
            &rows,
            row_queries,
        )
        .with_relation(BackendRelationId::new("jolt_vm", "spartan_outer.remainder"));
        let prefix_queries = fixed_prefixes
            .iter()
            .enumerate()
            .map(|(index, fixed_prefix)| {
                SumcheckPrefixProductSumQuery::new(
                    BackendValueSlot(index as u32),
                    eq_point.clone(),
                    fixed_prefix.clone(),
                    suffix_vars,
                    row_weights_at_zero.clone(),
                    row_weights_at_one.clone(),
                    Fr::from_u64(53 + index as u64),
                )
            })
            .collect::<Vec<_>>();
        let prefix_request = SumcheckPrefixProductSumRequest::new(
            "cpu.test_spartan_outer.remainder_prefix_reference",
            &witness_polynomials,
            &input_columns,
            rv64::const_column(),
            &matrices.a,
            &matrices.b,
            prefix_queries,
        )
        .with_relation(BackendRelationId::new("jolt_vm", "spartan_outer.remainder"));

        let mut backend = CpuBackend::default();
        let row_outputs =
            <CpuBackend as SumcheckBackend<Fr, TestNamespace>>::evaluate_sumcheck_spartan_outer_remainder_rows(
                &mut backend,
                &row_request,
            )
            .map_err(|error| error.to_string())?;
        let reference_outputs =
            <CpuBackend as SumcheckBackend<Fr, TestNamespace>>::evaluate_sumcheck_prefix_product_sums(
                &mut backend,
                &prefix_request,
            )
            .map_err(|error| error.to_string())?;

        assert_eq!(row_outputs, reference_outputs);
    }
    Ok(())
}

#[test]
fn cpu_spartan_outer_remainder_state_matches_bound_prefix_reference() -> Result<(), String> {
    let witness_polynomials = vec![
        vec![
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(5),
            Fr::from_u64(7),
        ],
        vec![
            Fr::from_u64(11),
            Fr::from_u64(13),
            Fr::from_u64(17),
            Fr::from_u64(19),
        ],
    ];
    let input_columns = vec![1, 2];
    let constant_column = 0;
    let left_rows = vec![
        vec![(1, Fr::from_u64(2)), (0, Fr::from_u64(3))],
        vec![(2, Fr::from_u64(5))],
    ];
    let right_rows = vec![
        vec![(2, Fr::from_u64(7)), (0, Fr::from_u64(11))],
        vec![(1, Fr::from_u64(13))],
    ];
    let eq_point = vec![Fr::from_u64(23), Fr::from_u64(29), Fr::from_u64(31)];
    let stream = Fr::from_u64(37);
    let row_weights_at_zero = vec![Fr::from_u64(41), Fr::from_u64(43)];
    let row_weights_at_one = vec![Fr::from_u64(47), Fr::from_u64(53)];
    let scale = Fr::from_u64(59);
    let state_request = SumcheckSpartanOuterRemainderStateRequest::new(
        "cpu.test_spartan_outer.remainder_state",
        &witness_polynomials,
        &input_columns,
        constant_column,
        &left_rows,
        &right_rows,
        eq_point.clone(),
        row_weights_at_zero.clone(),
        row_weights_at_one.clone(),
        stream,
        scale,
    )
    .with_relation(BackendRelationId::new("jolt_vm", "spartan_outer.remainder"));
    let mut backend = CpuBackend::default();
    let state =
        <CpuBackend as SumcheckBackend<Fr, TestNamespace>>::materialize_sumcheck_spartan_outer_remainder_state(
            &mut backend,
            &state_request,
        )
        .map_err(|error| error.to_string())?;
    let round =
        <CpuBackend as SumcheckBackend<Fr, TestNamespace>>::evaluate_sumcheck_spartan_outer_remainder_round(
            &mut backend,
            &state,
        )
        .map_err(|error| error.to_string())?;
    let eq_tensor = TensorEqTable::<Fr>::new(&state.eq_point[..1]);
    let mut expected_q_at_zero = Fr::from_u64(0);
    let mut expected_q_at_infinity = Fr::from_u64(0);
    for (x_out, &e_out) in eq_tensor.e_out().iter().enumerate() {
        for (x_in, &e_in) in eq_tensor.e_in().iter().enumerate() {
            let row_index = eq_tensor.group_index(x_out, x_in);
            let low_index = 2 * row_index;
            let high_index = low_index + 1;
            let eq = e_out * e_in;
            expected_q_at_zero += eq * state.left[low_index] * state.right[low_index];
            expected_q_at_infinity += eq
                * (state.left[high_index] - state.left[low_index])
                * (state.right[high_index] - state.right[low_index]);
        }
    }

    assert_eq!(round.q_at_zero, expected_q_at_zero);
    assert_eq!(round.q_at_infinity, expected_q_at_infinity);
    Ok(())
}

fn noop_spartan_outer_row() -> SumcheckSpartanOuterRow {
    SumcheckSpartanOuterRow {
        left_instruction_input: 0,
        right_instruction_input: 0,
        product_magnitude: 0,
        product_is_positive: true,
        should_branch: false,
        pc: 0,
        unexpanded_pc: 0,
        imm: 0,
        ram_address: 0,
        rs1_value: 0,
        rs2_value: 0,
        rd_write_value: 0,
        ram_read_value: 0,
        ram_write_value: 0,
        left_lookup_operand: 0,
        right_lookup_operand: 0,
        next_unexpanded_pc: 0,
        next_pc: 0,
        next_is_virtual: false,
        next_is_first_in_sequence: false,
        lookup_output: 0,
        should_jump: false,
        flag_add_operands: false,
        flag_subtract_operands: false,
        flag_multiply_operands: false,
        flag_load: false,
        flag_store: false,
        flag_jump: false,
        flag_write_lookup_output_to_rd: false,
        flag_virtual_instruction: false,
        flag_assert: false,
        flag_do_not_update_unexpanded_pc: true,
        flag_advice: false,
        flag_is_compressed: false,
        flag_is_first_in_sequence: false,
        flag_is_last_in_sequence: false,
    }
}

fn load_spartan_outer_row() -> SumcheckSpartanOuterRow {
    SumcheckSpartanOuterRow {
        left_instruction_input: 9,
        right_instruction_input: 4,
        product_magnitude: 36,
        product_is_positive: true,
        should_branch: false,
        pc: 3,
        unexpanded_pc: 12,
        imm: 5,
        ram_address: 16,
        rs1_value: 11,
        rs2_value: 99,
        rd_write_value: 77,
        ram_read_value: 77,
        ram_write_value: 77,
        left_lookup_operand: 9,
        right_lookup_operand: 4,
        next_unexpanded_pc: 16,
        next_pc: 4,
        next_is_virtual: false,
        next_is_first_in_sequence: false,
        lookup_output: 0,
        should_jump: false,
        flag_add_operands: false,
        flag_subtract_operands: false,
        flag_multiply_operands: false,
        flag_load: true,
        flag_store: false,
        flag_jump: false,
        flag_write_lookup_output_to_rd: false,
        flag_virtual_instruction: false,
        flag_assert: false,
        flag_do_not_update_unexpanded_pc: false,
        flag_advice: false,
        flag_is_compressed: false,
        flag_is_first_in_sequence: false,
        flag_is_last_in_sequence: false,
    }
}

fn add_spartan_outer_row() -> SumcheckSpartanOuterRow {
    SumcheckSpartanOuterRow {
        left_instruction_input: 5,
        right_instruction_input: 7,
        product_magnitude: 35,
        product_is_positive: true,
        should_branch: false,
        pc: 4,
        unexpanded_pc: 16,
        imm: 0,
        ram_address: 0,
        rs1_value: 5,
        rs2_value: 7,
        rd_write_value: 12,
        ram_read_value: 0,
        ram_write_value: 0,
        left_lookup_operand: 0,
        right_lookup_operand: 12,
        next_unexpanded_pc: 20,
        next_pc: 5,
        next_is_virtual: false,
        next_is_first_in_sequence: false,
        lookup_output: 12,
        should_jump: false,
        flag_add_operands: true,
        flag_subtract_operands: false,
        flag_multiply_operands: false,
        flag_load: false,
        flag_store: false,
        flag_jump: false,
        flag_write_lookup_output_to_rd: true,
        flag_virtual_instruction: false,
        flag_assert: false,
        flag_do_not_update_unexpanded_pc: false,
        flag_advice: false,
        flag_is_compressed: false,
        flag_is_first_in_sequence: false,
        flag_is_last_in_sequence: false,
    }
}

fn spartan_outer_input_polynomials(rows: &[SumcheckSpartanOuterRow]) -> Vec<Vec<Fr>> {
    let mut values = (0..35)
        .map(|_| Vec::with_capacity(rows.len()))
        .collect::<Vec<_>>();
    for row in rows {
        let row_values = [
            Fr::from_u64(row.left_instruction_input),
            Fr::from_i128(row.right_instruction_input),
            signed_u128_to_fr(row.product_magnitude, row.product_is_positive),
            Fr::from_bool(row.should_branch),
            Fr::from_u64(row.pc),
            Fr::from_u64(row.unexpanded_pc),
            Fr::from_i128(row.imm),
            Fr::from_u64(row.ram_address),
            Fr::from_u64(row.rs1_value),
            Fr::from_u64(row.rs2_value),
            Fr::from_u64(row.rd_write_value),
            Fr::from_u64(row.ram_read_value),
            Fr::from_u64(row.ram_write_value),
            Fr::from_u64(row.left_lookup_operand),
            Fr::from_u128(row.right_lookup_operand),
            Fr::from_u64(row.next_unexpanded_pc),
            Fr::from_u64(row.next_pc),
            Fr::from_bool(row.next_is_virtual),
            Fr::from_bool(row.next_is_first_in_sequence),
            Fr::from_u64(row.lookup_output),
            Fr::from_bool(row.should_jump),
            Fr::from_bool(row.flag_add_operands),
            Fr::from_bool(row.flag_subtract_operands),
            Fr::from_bool(row.flag_multiply_operands),
            Fr::from_bool(row.flag_load),
            Fr::from_bool(row.flag_store),
            Fr::from_bool(row.flag_jump),
            Fr::from_bool(row.flag_write_lookup_output_to_rd),
            Fr::from_bool(row.flag_virtual_instruction),
            Fr::from_bool(row.flag_assert),
            Fr::from_bool(row.flag_do_not_update_unexpanded_pc),
            Fr::from_bool(row.flag_advice),
            Fr::from_bool(row.flag_is_compressed),
            Fr::from_bool(row.flag_is_first_in_sequence),
            Fr::from_bool(row.flag_is_last_in_sequence),
        ];
        for (column, value) in values.iter_mut().zip(row_values) {
            column.push(value);
        }
    }
    #[cfg(feature = "field-inline")]
    values.resize_with(
        jolt_r1cs::constraints::jolt::spartan_outer_opening_columns().len(),
        || vec![Fr::from_u64(0); rows.len()],
    );
    values
}

fn spartan_outer_uniskip_targets(domain_size: usize) -> Vec<i64> {
    let start = -(((domain_size - 1) / 2) as i64);
    let end = start + domain_size as i64 - 1;
    (1..domain_size)
        .map(|offset| {
            let offset = offset as i64;
            if offset % 2 == 1 {
                start - ((offset + 1) / 2)
            } else {
                end + (offset / 2)
            }
        })
        .collect()
}

fn centered_lagrange_integer_coeffs(domain_size: usize, target: i64) -> Vec<i32> {
    let start = -(((domain_size - 1) / 2) as i64);
    (0..domain_size)
        .map(|index| {
            let x_i = start + index as i64;
            let mut numerator = 1i128;
            let mut denominator = 1i128;
            for other in 0..domain_size {
                if other == index {
                    continue;
                }
                let x_j = start + other as i64;
                numerator *= i128::from(target - x_j);
                denominator *= i128::from(x_i - x_j);
            }
            assert_eq!(numerator % denominator, 0);
            let value = numerator / denominator;
            assert!(i128::from(i32::MIN) <= value && value <= i128::from(i32::MAX));
            value as i32
        })
        .collect()
}

fn signed_u128_to_fr(magnitude: u128, is_positive: bool) -> Fr {
    let value = Fr::from_u128(magnitude);
    if is_positive {
        value
    } else {
        -value
    }
}

#[test]
fn cpu_sumcheck_backend_evaluates_row_products_by_query_slot() -> Result<(), String> {
    let witness_polynomials = vec![
        vec![
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(5),
            Fr::from_u64(7),
        ],
        vec![
            Fr::from_u64(11),
            Fr::from_u64(13),
            Fr::from_u64(17),
            Fr::from_u64(19),
        ],
    ];
    let input_columns = vec![1, 2];
    let left_rows = vec![vec![(1, Fr::from_u64(2)), (0, Fr::from_u64(3))]];
    let right_rows = vec![vec![(2, Fr::from_u64(5)), (0, Fr::from_u64(7))]];
    let row_weights = vec![Fr::from_u64(23)];
    let eq_point = vec![Fr::from_u64(29), Fr::from_u64(31)];
    let scale = Fr::from_u64(37);
    let request = SumcheckRowProductRequest::new(
        "cpu.test_row_product",
        &witness_polynomials,
        &input_columns,
        0,
        &left_rows,
        &right_rows,
        vec![SumcheckRowProductQuery::new(
            BackendValueSlot(12),
            eq_point.clone(),
            row_weights.clone(),
            scale,
        )],
    );
    let mut backend = CpuBackend::default();

    let outputs =
        <CpuBackend as SumcheckBackend<Fr, TestNamespace>>::evaluate_sumcheck_row_products(
            &mut backend,
            &request,
        )
        .map_err(|error| error.to_string())?;

    let expected_inner = (0..4)
        .map(|row| {
            let left =
                row_weights[0] * (Fr::from_u64(2) * witness_polynomials[0][row] + Fr::from_u64(3));
            let right =
                row_weights[0] * (Fr::from_u64(5) * witness_polynomials[1][row] + Fr::from_u64(7));
            eq_index_msb(&eq_point, row) * left * right
        })
        .sum::<Fr>();

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].slot, BackendValueSlot(12));
    assert_eq!(outputs[0].value, scale * expected_inner);
    Ok(())
}

#[test]
fn cpu_sumcheck_backend_evaluates_spartan_product_uniskip_rows() -> Result<(), String> {
    let witness_polynomials = vec![
        vec![
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(5),
            Fr::from_u64(7),
        ],
        vec![
            Fr::from_u64(11),
            Fr::from_u64(13),
            Fr::from_u64(17),
            Fr::from_u64(19),
        ],
        vec![
            Fr::from_u64(0),
            Fr::from_u64(1),
            Fr::from_u64(0),
            Fr::from_u64(1),
        ],
        vec![
            Fr::from_u64(23),
            Fr::from_u64(29),
            Fr::from_u64(31),
            Fr::from_u64(37),
        ],
        vec![
            Fr::from_u64(1),
            Fr::from_u64(0),
            Fr::from_u64(1),
            Fr::from_u64(0),
        ],
        vec![
            Fr::from_u64(0),
            Fr::from_u64(1),
            Fr::from_u64(1),
            Fr::from_u64(0),
        ],
    ];
    let input_columns = vec![20, 21, 22, 30, 31, 32];
    let constant_column = 99;
    let left_rows = vec![
        vec![(20, Fr::from_u64(1))],
        vec![(21, Fr::from_u64(1))],
        vec![(22, Fr::from_u64(1))],
    ];
    let right_rows = vec![
        vec![(30, Fr::from_u64(1))],
        vec![(31, Fr::from_u64(1))],
        vec![(constant_column, Fr::from_u64(1)), (32, Fr::from_i64(-1))],
    ];
    let eq_point = vec![Fr::from_u64(41), Fr::from_u64(43)];
    let first_weights = vec![Fr::from_u64(47), Fr::from_u64(53), Fr::from_u64(59)];
    let second_weights = vec![Fr::from_u64(61), Fr::from_u64(67), Fr::from_u64(71)];
    let request = SumcheckRowProductRequest::new(
        "cpu.test_spartan_product_uniskip",
        &witness_polynomials,
        &input_columns,
        constant_column,
        &left_rows,
        &right_rows,
        vec![
            SumcheckRowProductQuery::new(
                BackendValueSlot(12),
                eq_point.clone(),
                first_weights.clone(),
                Fr::from_u64(73),
            ),
            SumcheckRowProductQuery::new(
                BackendValueSlot(13),
                eq_point.clone(),
                second_weights.clone(),
                Fr::from_u64(79),
            ),
        ],
    )
    .with_relation(BackendRelationId::new(
        "jolt_vm",
        "spartan_product.uniskip_first_round",
    ));
    let mut backend = CpuBackend::default();

    let outputs =
        <CpuBackend as SumcheckBackend<Fr, TestNamespace>>::evaluate_sumcheck_row_products(
            &mut backend,
            &request,
        )
        .map_err(|error| error.to_string())?;

    let expected = |weights: &[Fr], scale: Fr| {
        (0..4)
            .map(|row| {
                let left = weights[0] * witness_polynomials[0][row]
                    + weights[1] * witness_polynomials[1][row]
                    + weights[2] * witness_polynomials[2][row];
                let right = weights[0] * witness_polynomials[3][row]
                    + weights[1] * witness_polynomials[4][row]
                    + weights[2] * (Fr::from_u64(1) - witness_polynomials[5][row]);
                eq_index_msb(&eq_point, row) * left * right
            })
            .sum::<Fr>()
            * scale
    };

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].slot, BackendValueSlot(12));
    assert_eq!(outputs[0].value, expected(&first_weights, Fr::from_u64(73)));
    assert_eq!(outputs[1].slot, BackendValueSlot(13));
    assert_eq!(
        outputs[1].value,
        expected(&second_weights, Fr::from_u64(79))
    );
    Ok(())
}

#[test]
fn cpu_sumcheck_backend_evaluates_raw_spartan_product_uniskip_rows() -> Result<(), String> {
    let rows = vec![
        SumcheckProductUniskipRow::new(2, 11, false, 23, true, false),
        SumcheckProductUniskipRow::new(3, 13, true, -29, false, true),
        SumcheckProductUniskipRow::new(5, 17, false, 31, true, true),
        SumcheckProductUniskipRow::new(7, 19, true, -37, false, false),
    ];
    let witness_polynomials = vec![
        rows.iter()
            .map(|row| Fr::from_u64(row.left_instruction))
            .collect::<Vec<_>>(),
        rows.iter()
            .map(|row| Fr::from_u64(row.lookup_output))
            .collect::<Vec<_>>(),
        rows.iter()
            .map(|row| Fr::from_bool(row.jump_flag))
            .collect::<Vec<_>>(),
        rows.iter()
            .map(|row| Fr::from_i128(row.right_instruction))
            .collect::<Vec<_>>(),
        rows.iter()
            .map(|row| Fr::from_bool(row.branch_flag))
            .collect::<Vec<_>>(),
        rows.iter()
            .map(|row| Fr::from_bool(row.next_is_noop))
            .collect::<Vec<_>>(),
    ];
    let input_columns = vec![20, 21, 22, 30, 31, 32];
    let constant_column = 99;
    let left_rows = vec![
        vec![(20, Fr::from_u64(1))],
        vec![(21, Fr::from_u64(1))],
        vec![(22, Fr::from_u64(1))],
    ];
    let right_rows = vec![
        vec![(30, Fr::from_u64(1))],
        vec![(31, Fr::from_u64(1))],
        vec![(constant_column, Fr::from_u64(1)), (32, Fr::from_i64(-1))],
    ];
    let eq_point = vec![Fr::from_u64(41), Fr::from_u64(43)];
    let queries = vec![
        SumcheckRowProductQuery::new(
            BackendValueSlot(12),
            eq_point.clone(),
            vec![Fr::from_u64(47), Fr::from_u64(53), Fr::from_u64(59)],
            Fr::from_u64(73),
        ),
        SumcheckRowProductQuery::new(
            BackendValueSlot(13),
            eq_point,
            vec![Fr::from_u64(61), Fr::from_u64(67), Fr::from_u64(71)],
            Fr::from_u64(79),
        ),
    ];
    let relation = BackendRelationId::new("jolt_vm", "spartan_product.uniskip_first_round");
    let materialized_request = SumcheckRowProductRequest::new(
        "cpu.test_spartan_product_uniskip_materialized",
        &witness_polynomials,
        &input_columns,
        constant_column,
        &left_rows,
        &right_rows,
        queries.clone(),
    )
    .with_relation(relation);
    let raw_request =
        SumcheckProductUniskipRequest::new("cpu.test_spartan_product_uniskip_raw", &rows, queries)
            .with_relation(relation);
    let mut backend = CpuBackend::default();

    let materialized_outputs =
        <CpuBackend as SumcheckBackend<Fr, TestNamespace>>::evaluate_sumcheck_row_products(
            &mut backend,
            &materialized_request,
        )
        .map_err(|error| error.to_string())?;
    let raw_outputs =
        <CpuBackend as SumcheckBackend<Fr, TestNamespace>>::evaluate_sumcheck_product_uniskip_rows(
            &mut backend,
            &raw_request,
        )
        .map_err(|error| error.to_string())?;

    assert_eq!(raw_outputs, materialized_outputs);
    Ok(())
}

#[test]
fn cpu_sumcheck_backend_evaluates_raw_spartan_product_uniskip_extended_rows() -> Result<(), String>
{
    let rows = vec![
        SumcheckProductUniskipRow::new(2, 11, false, 23, true, false),
        SumcheckProductUniskipRow::new(3, 13, true, -29, false, true),
        SumcheckProductUniskipRow::new(5, 17, false, 31, true, true),
        SumcheckProductUniskipRow::new(7, 19, true, -37, false, false),
    ];
    let eq_point = vec![Fr::from_u64(41), Fr::from_u64(43)];
    let first_weights = vec![Fr::from_i64(3), Fr::from_i64(-3), Fr::from_i64(1)];
    let second_weights = vec![Fr::from_i64(1), Fr::from_i64(-3), Fr::from_i64(3)];
    let request = SumcheckProductUniskipRequest::new(
        "cpu.test_spartan_product_uniskip_raw_extended",
        &rows,
        vec![
            SumcheckRowProductQuery::new(
                BackendValueSlot(12),
                eq_point.clone(),
                first_weights.clone(),
                Fr::from_u64(1),
            ),
            SumcheckRowProductQuery::new(
                BackendValueSlot(13),
                eq_point.clone(),
                second_weights.clone(),
                Fr::from_u64(1),
            ),
        ],
    )
    .with_relation(BackendRelationId::new(
        "jolt_vm",
        "spartan_product.uniskip_first_round",
    ));
    let mut backend = CpuBackend::default();

    let outputs =
        <CpuBackend as SumcheckBackend<Fr, TestNamespace>>::evaluate_sumcheck_product_uniskip_rows(
            &mut backend,
            &request,
        )
        .map_err(|error| error.to_string())?;

    let expected = |weights: &[Fr]| {
        rows.iter()
            .enumerate()
            .map(|(index, row)| {
                let left = weights[0] * Fr::from_u64(row.left_instruction)
                    + weights[1] * Fr::from_u64(row.lookup_output)
                    + weights[2] * Fr::from_bool(row.jump_flag);
                let right = weights[0] * Fr::from_i128(row.right_instruction)
                    + weights[1] * Fr::from_bool(row.branch_flag)
                    + weights[2] * Fr::from_bool(!row.next_is_noop);
                eq_index_msb(&eq_point, index) * left * right
            })
            .sum::<Fr>()
    };

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].slot, BackendValueSlot(12));
    assert_eq!(outputs[0].value, expected(&first_weights));
    assert_eq!(outputs[1].slot, BackendValueSlot(13));
    assert_eq!(outputs[1].value, expected(&second_weights));
    Ok(())
}

#[test]
fn cpu_sumcheck_backend_rejects_wrong_evaluation_point_arity() {
    let witness = TestDenseViewWitness {
        encoding: PolynomialEncoding::Dense,
        dimensions: WitnessDimensions::new(4, 2),
        values: vec![Fr::from_u64(0); 4],
    };
    let request = SumcheckEvaluationRequest::new(
        "cpu.test_sumcheck_eval",
        vec![SumcheckViewEvaluationRequest::new(
            BackendValueSlot(9),
            requirement(
                PolynomialEncoding::Dense,
                MaterializationPolicy::BackendChoice,
            ),
            vec![Fr::from_u64(1)],
        )],
    );
    let mut backend = CpuBackend::default();

    let result = <CpuBackend as SumcheckBackend<Fr, TestNamespace>>::evaluate_sumcheck_views(
        &mut backend,
        &request,
        &witness,
    );

    assert!(matches!(
        result,
        Err(BackendError::InvalidRequest {
            task: "sumcheck view evaluation",
            ..
        })
    ));
}

#[derive(Clone, Debug)]
struct TestOracle {
    id: u8,
    encoding: PolynomialEncoding,
    dimensions: WitnessDimensions,
    chunks: Vec<PolynomialChunk<Fr>>,
}

#[derive(Clone, Debug)]
struct TestMultiOracleWitness {
    oracles: Vec<TestOracle>,
}

impl TestMultiOracleWitness {
    fn oracle(&self, id: u8) -> Result<&TestOracle, WitnessError> {
        self.oracles
            .iter()
            .find(|oracle| oracle.id == id)
            .ok_or(WitnessError::UnknownOracle {
                namespace: TestNamespace::ID.name,
            })
    }
}

impl WitnessProvider<Fr, TestNamespace> for TestMultiOracleWitness {
    fn describe_oracle(
        &self,
        oracle: OracleRef<TestNamespace>,
    ) -> Result<OracleDescriptor<TestNamespace>, WitnessError> {
        let jolt_witness::OracleKind::Committed(id) = oracle.kind else {
            return Err(WitnessError::UnknownOracle {
                namespace: TestNamespace::ID.name,
            });
        };
        let oracle = self.oracle(id)?;
        Ok(OracleDescriptor::new(
            OracleRef::committed(id),
            oracle.dimensions,
            oracle.encoding,
        ))
    }

    fn view_requirements(
        &self,
        oracle: OracleRef<TestNamespace>,
    ) -> Result<Vec<ViewRequirement<TestNamespace>>, WitnessError> {
        let descriptor = self.describe_oracle(oracle)?;
        Ok(vec![ViewRequirement::new(
            oracle,
            descriptor.encoding,
            MaterializationPolicy::BackendChoice,
            RetentionHint::ThroughStage8,
        )])
    }

    fn oracle_view(
        &self,
        _request: OracleViewRequest<TestNamespace>,
    ) -> Result<PolynomialView<'_, Fr, TestNamespace>, WitnessError> {
        Err(WitnessError::UnsupportedView {
            view: "cpu test multi oracle views",
        })
    }

    fn committed_stream<'a>(
        &'a self,
        id: u8,
        _chunk_size: usize,
    ) -> Result<Box<dyn PolynomialStream<Fr> + 'a>, WitnessError>
    where
        TestNamespace: 'a,
    {
        Ok(Box::new(TestFieldStream {
            chunks: self.oracle(id)?.chunks.clone().into_iter(),
        }))
    }
}

#[test]
fn cpu_commitment_backend_commits_compact_stream_by_slot() -> Result<(), String> {
    let values = vec![
        Fr::from_i64(1),
        Fr::from_i64(-2),
        Fr::from_i64(3),
        Fr::from_i64(4),
    ];
    let witness = TestFieldWitness {
        encoding: PolynomialEncoding::Compact,
        dimensions: WitnessDimensions::new(4, 2),
        chunks: vec![
            PolynomialChunk::I128(vec![1]),
            PolynomialChunk::I128(vec![-2, 3]),
            PolynomialChunk::I128(vec![4]),
        ],
    };
    let requirement = requirement(
        PolynomialEncoding::Compact,
        MaterializationPolicy::Streaming,
    );
    let request = CommitmentRequest::new(vec![CommitmentRequestItem::new(
        CommitmentSlot(5),
        requirement,
    )]);
    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 2,
    });

    let result =
        <CpuBackend as CommitmentBackend<Fr, TestNamespace, MockCommitmentScheme<Fr>>>::commit(
            &mut backend,
            &request,
            &witness,
            &(),
        )
        .map_err(|error| error.to_string())?;
    let expected_poly = jolt_poly::Polynomial::new(values);
    let (expected, ()) = MockCommitmentScheme::commit(&expected_poly, &());

    assert_eq!(result.resolved_witness.len(), 1);
    assert_eq!(result.resolved_witness[0].slot, CommitmentSlot(5));
    assert_eq!(result.resolved_witness[0].requirement, requirement);
    assert_eq!(result.streamed_witness[0].slot, CommitmentSlot(5));
    assert_eq!(
        result.streamed_witness[0]
            .chunks
            .iter()
            .map(|chunk| chunk.rows)
            .collect::<Vec<_>>(),
        vec![1, 2, 1]
    );
    assert_eq!(result.commitments.len(), 1);
    assert_eq!(result.commitments[0].slot, CommitmentSlot(5));
    assert_eq!(result.commitments[0].oracle, OracleRef::committed(7));
    assert_eq!(result.commitments[0].rows, 4);
    assert_eq!(result.commitments[0].commitment, expected);
    Ok(())
}

#[test]
fn cpu_commitment_backend_embeds_compact_trace_stream() -> Result<(), String> {
    let witness = TestFieldWitness {
        encoding: PolynomialEncoding::Compact,
        dimensions: WitnessDimensions::new(4, 2),
        chunks: vec![
            PolynomialChunk::I128(vec![1, -2]),
            PolynomialChunk::I128(vec![3, 4]),
        ],
    };
    let request = CommitmentRequest::new(vec![CommitmentRequestItem::new(
        CommitmentSlot(5),
        requirement(
            PolynomialEncoding::Compact,
            MaterializationPolicy::Streaming,
        ),
    )
    .with_trace_embedding(Some(TracePolynomialEmbedding::new(
        4,
        4,
        TracePolynomialOrder::CycleMajor,
    )))]);
    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 2,
    });

    let result =
        <CpuBackend as CommitmentBackend<Fr, TestNamespace, MockCommitmentScheme<Fr>>>::commit(
            &mut backend,
            &request,
            &witness,
            &(),
        )
        .map_err(|error| error.to_string())?;
    let mut expected_values = vec![
        Fr::from_i64(1),
        Fr::from_i64(-2),
        Fr::from_i64(3),
        Fr::from_i64(4),
    ];
    expected_values.resize(16, Fr::from_u64(0));
    let expected_poly = jolt_poly::Polynomial::new(expected_values);
    let (expected, ()) = MockCommitmentScheme::commit(&expected_poly, &());

    assert_eq!(result.commitments.len(), 1);
    assert_eq!(result.commitments[0].slot, CommitmentSlot(5));
    assert_eq!(result.commitments[0].rows, 4);
    assert_eq!(result.commitments[0].commitment, expected);
    Ok(())
}

#[test]
fn cpu_commitment_backend_commits_dory_assist_witness_provider() -> Result<(), String> {
    let id = DoryAssistCommittedPolynomial::OperationTrace(DoryAssistOperationFamily::Pairing);
    let values = vec![
        Fr::from_u64(1),
        Fr::from_u64(2),
        Fr::from_u64(3),
        Fr::from_u64(4),
    ];
    let witness = DoryAssistWitness::new(
        vec![DoryAssistCommittedColumn::new(id, values.clone())],
        Vec::new(),
    )
    .map_err(|error| error.to_string())?;
    let requirement = ViewRequirement::new(
        OracleRef::committed(id),
        PolynomialEncoding::Dense,
        MaterializationPolicy::Streaming,
        RetentionHint::Permanent,
    );
    let request = CommitmentRequest::new(vec![CommitmentRequestItem::new(
        CommitmentSlot(9),
        requirement,
    )]);
    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 2,
    });

    let result = <CpuBackend as CommitmentBackend<
        Fr,
        DoryAssistNamespace,
        MockCommitmentScheme<Fr>,
    >>::commit(&mut backend, &request, &witness, &())
    .map_err(|error| error.to_string())?;
    let expected_poly = jolt_poly::Polynomial::new(values);
    let (expected, ()) = MockCommitmentScheme::commit(&expected_poly, &());

    assert_eq!(result.resolved_witness[0].slot, CommitmentSlot(9));
    assert_eq!(result.resolved_witness[0].requirement, requirement);
    assert_eq!(
        result.streamed_witness[0]
            .chunks
            .iter()
            .map(|chunk| chunk.rows)
            .collect::<Vec<_>>(),
        vec![2, 2]
    );
    assert_eq!(result.commitments[0].oracle, OracleRef::committed(id));
    assert_eq!(result.commitments[0].commitment, expected);
    Ok(())
}

#[test]
fn cpu_commitment_backend_commits_one_hot_stream_without_dense_witness() -> Result<(), String> {
    let indices = [Some(1), None, Some(0), Some(3)];
    let witness = TestFieldWitness {
        encoding: PolynomialEncoding::OneHot,
        dimensions: WitnessDimensions::new(16, 4),
        chunks: vec![
            PolynomialChunk::OneHot(indices[..2].to_vec()),
            PolynomialChunk::OneHot(indices[2..].to_vec()),
        ],
    };
    let requirement = ViewRequirement::new(
        OracleRef::committed(9),
        PolynomialEncoding::OneHot,
        MaterializationPolicy::Streaming,
        RetentionHint::ThroughStage8,
    );
    let request = CommitmentRequest::new(vec![CommitmentRequestItem::new(
        CommitmentSlot(6),
        requirement,
    )]);
    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 2,
    });

    let result =
        <CpuBackend as CommitmentBackend<Fr, TestNamespace, MockCommitmentScheme<Fr>>>::commit(
            &mut backend,
            &request,
            &witness,
            &(),
        )
        .map_err(|error| error.to_string())?;
    let expected_indices = vec![Some(1), None, Some(0), Some(3)];
    let expected_poly = jolt_poly::OneHotPolynomial::new_with_index_order(
        4,
        expected_indices,
        OneHotIndexOrder::ColumnMajor,
    );
    let (expected, ()) = MockCommitmentScheme::commit(&expected_poly, &());

    assert_eq!(result.streamed_witness[0].rows, 4);
    assert_eq!(
        result.streamed_witness[0]
            .chunks
            .iter()
            .map(|chunk| (chunk.kind, chunk.rows))
            .collect::<Vec<_>>(),
        vec![
            (PolynomialChunkKind::OneHot, 2),
            (PolynomialChunkKind::OneHot, 2),
        ]
    );
    assert_eq!(result.commitments.len(), 1);
    assert_eq!(result.commitments[0].slot, CommitmentSlot(6));
    assert_eq!(result.commitments[0].oracle, OracleRef::committed(9));
    assert_eq!(result.commitments[0].rows, 16);
    assert_eq!(result.commitments[0].commitment, expected);
    Ok(())
}

#[test]
fn cpu_commitment_backend_uses_zk_commitment_mode_for_dense_and_one_hot() -> Result<(), String> {
    let dense_values = vec![
        Fr::from_u64(1),
        Fr::from_u64(2),
        Fr::from_u64(3),
        Fr::from_u64(4),
    ];
    let indices = vec![Some(1), None, Some(0), Some(3)];
    let witness = TestMultiOracleWitness {
        oracles: vec![
            TestOracle {
                id: 1,
                encoding: PolynomialEncoding::Dense,
                dimensions: WitnessDimensions::new(4, 2),
                chunks: vec![PolynomialChunk::Dense(dense_values)],
            },
            TestOracle {
                id: 2,
                encoding: PolynomialEncoding::OneHot,
                dimensions: WitnessDimensions::new(16, 4),
                chunks: vec![PolynomialChunk::OneHot(indices)],
            },
        ],
    };
    let request = CommitmentRequest::new(vec![
        CommitmentRequestItem::with_mode(
            CommitmentSlot(1),
            ViewRequirement::new(
                OracleRef::committed(1),
                PolynomialEncoding::Dense,
                MaterializationPolicy::Streaming,
                RetentionHint::ThroughBlindFold,
            ),
            CommitmentMode::Zk,
        ),
        CommitmentRequestItem::with_mode(
            CommitmentSlot(2),
            ViewRequirement::new(
                OracleRef::committed(2),
                PolynomialEncoding::OneHot,
                MaterializationPolicy::Streaming,
                RetentionHint::ThroughBlindFold,
            ),
            CommitmentMode::Zk,
        ),
    ]);
    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 2,
    });

    let result =
        <CpuBackend as CommitmentBackend<Fr, TestNamespace, MockCommitmentScheme<Fr>>>::commit(
            &mut backend,
            &request,
            &witness,
            &(),
        )
        .map_err(|error| error.to_string())?;

    assert_eq!(result.commitments.len(), 2);
    assert!(result.commitments[0].commitment.is_zk());
    assert!(result.commitments[1].commitment.is_zk());
    Ok(())
}

#[test]
fn cpu_commitment_backend_commits_zero_dense_chunks() -> Result<(), String> {
    let witness = TestMultiOracleWitness {
        oracles: vec![TestOracle {
            id: 1,
            encoding: PolynomialEncoding::Dense,
            dimensions: WitnessDimensions::new(16, 4),
            chunks: vec![PolynomialChunk::Zeros(16)],
        }],
    };
    let request = CommitmentRequest::new(vec![CommitmentRequestItem::new(
        CommitmentSlot(1),
        ViewRequirement::new(
            OracleRef::committed(1),
            PolynomialEncoding::Dense,
            MaterializationPolicy::Streaming,
            RetentionHint::ThroughStage8,
        ),
    )]);
    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 16,
    });

    let result =
        <CpuBackend as CommitmentBackend<Fr, TestNamespace, MockCommitmentScheme<Fr>>>::commit(
            &mut backend,
            &request,
            &witness,
            &(),
        )
        .map_err(|error| error.to_string())?;

    let expected_poly = jolt_poly::Polynomial::new(vec![Fr::from_u64(0); 16]);
    let (expected, ()) = MockCommitmentScheme::commit(&expected_poly, &());

    assert_eq!(
        result.streamed_witness[0]
            .chunks
            .iter()
            .map(|chunk| (chunk.kind, chunk.rows))
            .collect::<Vec<_>>(),
        vec![(PolynomialChunkKind::Zeros, 16)]
    );
    assert_eq!(result.commitments[0].commitment, expected);
    Ok(())
}

#[test]
fn cpu_commitment_backend_uses_core_one_hot_order_when_dense_trace_sets_layout(
) -> Result<(), String> {
    let indices = (0..16)
        .map(|row| if row % 5 == 0 { None } else { Some(row % 4) })
        .collect::<Vec<_>>();
    let witness = TestMultiOracleWitness {
        oracles: vec![
            TestOracle {
                id: 1,
                encoding: PolynomialEncoding::Dense,
                dimensions: WitnessDimensions::new(16, 4),
                chunks: vec![
                    PolynomialChunk::Dense(vec![Fr::from_u64(0); 8]),
                    PolynomialChunk::Dense(vec![Fr::from_u64(0); 8]),
                ],
            },
            TestOracle {
                id: 2,
                encoding: PolynomialEncoding::OneHot,
                dimensions: WitnessDimensions::new(64, 6),
                chunks: indices
                    .chunks(8)
                    .map(|chunk| PolynomialChunk::OneHot(chunk.to_vec()))
                    .collect(),
            },
        ],
    };
    let request = CommitmentRequest::new(vec![
        CommitmentRequestItem::new(
            CommitmentSlot(1),
            ViewRequirement::new(
                OracleRef::committed(1),
                PolynomialEncoding::Dense,
                MaterializationPolicy::Streaming,
                RetentionHint::ThroughStage8,
            ),
        ),
        CommitmentRequestItem::new(
            CommitmentSlot(2),
            ViewRequirement::new(
                OracleRef::committed(2),
                PolynomialEncoding::OneHot,
                MaterializationPolicy::Streaming,
                RetentionHint::ThroughStage8,
            ),
        ),
    ]);
    let mut backend = CpuBackend::new(CpuBackendConfig {
        preserve_core_fast_path: true,
        commitment_chunk_size: 16,
    });

    let result =
        <CpuBackend as CommitmentBackend<Fr, TestNamespace, MockCommitmentScheme<Fr>>>::commit(
            &mut backend,
            &request,
            &witness,
            &(),
        )
        .map_err(|error| error.to_string())?;

    let expected_poly = jolt_poly::OneHotPolynomial::new_with_index_order(
        4,
        indices
            .into_iter()
            .map(|index| index.map(|value| value as u8))
            .collect(),
        jolt_poly::OneHotIndexOrder::ColumnMajor,
    );
    let (expected, ()) = MockCommitmentScheme::commit(&expected_poly, &());

    assert_eq!(result.commitments[1].slot, CommitmentSlot(2));
    assert_eq!(result.commitments[1].commitment, expected);
    Ok(())
}

#[test]
fn cpu_commitment_backend_rejects_requirement_encoding_mismatch() {
    let witness = TestFieldWitness {
        encoding: PolynomialEncoding::Dense,
        dimensions: WitnessDimensions::new(4, 2),
        chunks: Vec::new(),
    };
    let request = CommitmentRequest::new(vec![CommitmentRequestItem::new(
        CommitmentSlot(1),
        requirement(
            PolynomialEncoding::Compact,
            MaterializationPolicy::Streaming,
        ),
    )]);
    let mut backend = CpuBackend::default();

    let result =
        <CpuBackend as CommitmentBackend<Fr, TestNamespace, MockCommitmentScheme<Fr>>>::commit(
            &mut backend,
            &request,
            &witness,
            &(),
        );

    assert!(matches!(
        result,
        Err(BackendError::Witness(WitnessError::InvalidWitnessData {
            namespace: "cpu_test",
            ..
        }))
    ));
}

#[test]
fn cpu_commitment_backend_requires_streaming_materialization() {
    let witness = TestFieldWitness {
        encoding: PolynomialEncoding::Compact,
        dimensions: WitnessDimensions::new(4, 2),
        chunks: Vec::new(),
    };
    let request = CommitmentRequest::new(vec![CommitmentRequestItem::new(
        CommitmentSlot(1),
        requirement(
            PolynomialEncoding::Compact,
            MaterializationPolicy::BackendChoice,
        ),
    )]);
    let mut backend = CpuBackend::default();

    let result =
        <CpuBackend as CommitmentBackend<Fr, TestNamespace, MockCommitmentScheme<Fr>>>::commit(
            &mut backend,
            &request,
            &witness,
            &(),
        );

    assert!(matches!(
        result,
        Err(BackendError::Witness(WitnessError::InvalidWitnessData {
            namespace: "cpu_test",
            ..
        }))
    ));
}
