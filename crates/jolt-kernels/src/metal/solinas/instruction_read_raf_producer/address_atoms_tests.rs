#![expect(clippy::unwrap_used)]

use jolt_field::AkitaField;

use super::*;

fn selector(table: Option<usize>, raf: bool) -> ProducerSelector {
    ProducerSelector::new(table, raf).unwrap()
}

fn shard(rows: usize) -> ProducerShardPlan {
    ProducerGeometry::new(rows).unwrap().shard(0).unwrap()
}

struct Fixture {
    lookup_lo: Vec<u64>,
    lookup_hi: Vec<u64>,
    claims: Vec<u8>,
}

impl Fixture {
    fn new() -> Self {
        let rows = [
            (9, selector(Some(0), false)),
            (5, selector(None, false)),
            (9, selector(Some(0), false)),
            (5, selector(None, true)),
            (u128::MAX, selector(Some(0), false)),
            (5, selector(None, false)),
            (9, selector(Some(0), true)),
            (9, selector(Some(1), false)),
        ];
        Self {
            lookup_lo: rows.iter().map(|(lookup, _)| *lookup as u64).collect(),
            lookup_hi: rows
                .iter()
                .map(|(lookup, _)| (*lookup >> 64) as u64)
                .collect(),
            claims: rows.iter().map(|(_, selector)| selector.claim()).collect(),
        }
    }

    fn source(&self) -> AddressAtomCycleSource<'_> {
        AddressAtomCycleSource::new(shard(8), &self.lookup_lo, &self.lookup_hi, &self.claims)
            .unwrap()
    }
}

#[test]
fn reference_topology_is_the_exact_v3_key_partition() {
    let fixture = Fixture::new();
    let topology = AddressAtomTopology::from_cycle_source_reference(fixture.source()).unwrap();

    assert_eq!(topology.rows(), 8);
    assert_eq!(topology.atoms(), 6);
    assert_eq!(
        topology
            .atom_lookups()
            .iter()
            .map(|lookup| lookup.value())
            .collect::<Vec<_>>(),
        [5, 5, 9, u128::MAX, 9, 9]
    );
    assert_eq!(
        topology.atom_claims(),
        &[
            selector(None, false).claim(),
            selector(None, true).claim(),
            selector(Some(0), false).claim(),
            selector(Some(0), false).claim(),
            selector(Some(0), true).claim(),
            selector(Some(1), false).claim(),
        ]
    );
    assert_eq!(topology.atom_cycle_offsets(), &[0, 2, 3, 5, 6, 7, 8]);
    assert_eq!(topology.cycle_indices(), &[1, 5, 3, 0, 2, 4, 6, 7]);
    assert_eq!(topology.cycle_to_atom(), &[2, 0, 2, 1, 3, 0, 4, 5]);
    assert_eq!(&topology.segment_atom_offsets()[..6], &[0, 1, 2, 4, 5, 6]);
    assert!(topology.segment_atom_offsets()[6..]
        .iter()
        .all(|&offset| offset == 6));
    topology.validate_against(fixture.source()).unwrap();
}

#[test]
fn atom_masses_use_the_csr_inverse_and_are_partition_linear() {
    let fixture = Fixture::new();
    let topology = AddressAtomTopology::from_cycle_source_reference(fixture.source()).unwrap();
    let weights = (1..=8).map(AkitaField::from_u64).collect::<Vec<_>>();
    let masses = topology.masses_from_cycle_weights(&weights).unwrap();
    let expected = [8, 4, 4, 5, 7, 8].map(AkitaField::from_u64);
    assert_eq!(masses, expected);

    let even = weights
        .iter()
        .enumerate()
        .map(|(cycle, &weight)| {
            if cycle % 2 == 0 {
                weight
            } else {
                AkitaField::zero()
            }
        })
        .collect::<Vec<_>>();
    let odd = weights
        .iter()
        .enumerate()
        .map(|(cycle, &weight)| {
            if cycle % 2 == 1 {
                weight
            } else {
                AkitaField::zero()
            }
        })
        .collect::<Vec<_>>();
    let even_masses = topology.masses_from_cycle_weights(&even).unwrap();
    let odd_masses = topology.masses_from_cycle_weights(&odd).unwrap();
    for atom in 0..topology.atoms() {
        assert_eq!(even_masses[atom] + odd_masses[atom], masses[atom]);
    }
}

#[test]
fn split_equality_uses_absolute_cycle_bits() {
    let fixture = Fixture::new();
    let topology = AddressAtomTopology::from_cycle_source_reference(fixture.source()).unwrap();
    let e_in = [2, 3, 5, 7].map(AkitaField::from_u64);
    let e_out = [11, 13].map(AkitaField::from_u64);
    let masses = topology.masses_from_split_equality(&e_in, &e_out).unwrap();
    assert_eq!(masses, [72, 77, 77, 26, 65, 91].map(AkitaField::from_u64));

    let global_e_out = [11, 13, 17, 19].map(AkitaField::from_u64);
    assert_eq!(
        split_equality_weight(16, 9, &e_in, &global_e_out).unwrap(),
        AkitaField::from_u64(17 * 3)
    );
}

#[test]
fn checked_topology_rejects_bad_permutations_and_inverse() {
    let fixture = Fixture::new();
    assert!(matches!(
        AddressAtomTopology::from_sorted_cycles(fixture.source(), &[0; 8]),
        Err(AddressAtomError::DuplicateCycle { .. })
    ));
    assert!(matches!(
        AddressAtomTopology::from_sorted_cycles(
            fixture.source(),
            &(0..8).rev().collect::<Vec<_>>()
        ),
        Err(AddressAtomError::NonMonotoneKey { .. })
    ));

    let topology = AddressAtomTopology::from_cycle_source_reference(fixture.source()).unwrap();
    let mut parts = topology.parts().clone();
    parts.cycle_to_atom[0] = 0;
    assert_eq!(
        AddressAtomTopology::from_checked_parts(fixture.source(), parts),
        Err(AddressAtomError::InvalidTopology(
            "cycle-to-atom is not the CSR inverse"
        ))
    );
}

#[test]
fn log26_storage_and_traffic_are_explicit() {
    let rows = 1 << 26;
    let shape = AddressAtomShape::new(shard(rows), rows).unwrap();
    let traffic = AddressAtomTraffic::for_shape(shape).unwrap();

    assert_eq!(traffic.source_read_bytes(), 1_140_850_688);
    assert_eq!(traffic.topology_write_bytes(), 1_946_157_392);
    assert_eq!(traffic.v3_handoff_bytes(), 1_677_721_936);
    assert_eq!(traffic.standalone_build_bytes().unwrap(), 3_087_008_080);
    assert_eq!(traffic.co_produced_build_bytes(), 1_946_157_392);
    assert_eq!(traffic.mass_floor_bytes(), 3_489_660_928);
    assert_eq!(traffic.topology_and_mass_bytes(), 3_019_899_216);
    assert_eq!(traffic.live_with_source_bytes().unwrap(), 4_160_749_904);

    let penalty = AddressAtomPartitionPenalty::new(120, 100).unwrap();
    assert_eq!(penalty.duplicate_atoms(), 20);
    assert_eq!(penalty.topology_bytes(), 420);
    assert_eq!(penalty.mass_bytes(), 320);
    assert_eq!(penalty.address_state_bytes(), 14_720);
}

fn source_provenance(
    shard: ProducerShardPlan,
    first_identity: usize,
) -> AddressAtomSourceProvenance {
    AddressAtomSourceProvenance::new(
        shard,
        7,
        9,
        11,
        [first_identity, first_identity + 1, first_identity + 2],
    )
    .unwrap()
}

fn plane_receipts(shape: AddressAtomShape, first_identity: usize) -> [AddressAtomPlaneReceipt; 6] {
    let mut identity = first_identity;
    shape.buffer_shapes().unwrap().map(|shape| {
        let receipt = AddressAtomPlaneReceipt::new(
            shape.role(),
            shape.elements(),
            shape.bytes() as u64,
            7,
            identity,
            9,
            12,
        )
        .unwrap();
        identity += 1;
        receipt
    })
}

fn topology_receipt(
    shard: ProducerShardPlan,
    atoms: usize,
    first_identity: usize,
) -> AddressAtomTopologyReceipt {
    let shape = AddressAtomShape::new(shard, atoms).unwrap();
    AddressAtomTopologyReceipt::new(
        shape,
        source_provenance(shard, first_identity),
        12,
        0,
        plane_receipts(shape, first_identity + 3),
    )
    .unwrap()
}

#[test]
fn receipts_fail_closed_on_status_shape_generation_and_aliases() {
    let shard = shard(8);
    let shape = AddressAtomShape::new(shard, 4).unwrap();
    let source = source_provenance(shard, 10);
    let planes = plane_receipts(shape, 20);
    let receipt = AddressAtomTopologyReceipt::new(shape, source, 12, 0, planes).unwrap();
    let masses = AddressAtomMassReceipt::new(receipt, 4, 64, 7, 30, 9, 13, 0x1234).unwrap();
    assert_eq!(masses.topology(), receipt);

    assert!(matches!(
        AddressAtomTopologyReceipt::new(shape, source, 12, 1, planes),
        Err(AddressAtomError::NonzeroStatus { .. })
    ));
    let mut wrong_shape = planes;
    wrong_shape[0] =
        AddressAtomPlaneReceipt::new(AddressAtomPlaneRole::AtomLookups, 3, 48, 7, 20, 9, 12)
            .unwrap();
    assert!(matches!(
        AddressAtomTopologyReceipt::new(shape, source, 12, 0, wrong_shape),
        Err(AddressAtomError::PlaneShape { .. })
    ));
    let mut wrong_generation = planes;
    wrong_generation[2] = AddressAtomPlaneReceipt::new(
        wrong_generation[2].role(),
        wrong_generation[2].elements(),
        wrong_generation[2].bytes(),
        7,
        wrong_generation[2].allocation_identity(),
        10,
        12,
    )
    .unwrap();
    assert!(matches!(
        AddressAtomTopologyReceipt::new(shape, source, 12, 0, wrong_generation),
        Err(AddressAtomError::GenerationMismatch { .. })
    ));
    let mut aliased = planes;
    aliased[0] = AddressAtomPlaneReceipt::new(
        aliased[0].role(),
        aliased[0].elements(),
        aliased[0].bytes(),
        7,
        source.allocation_identities()[0],
        9,
        12,
    )
    .unwrap();
    assert!(matches!(
        AddressAtomTopologyReceipt::new(shape, source, 12, 0, aliased),
        Err(AddressAtomError::AliasedAllocation { .. })
    ));
}

#[test]
fn batch_receipt_covers_every_log28_shard_once() {
    let geometry = ProducerGeometry::new(1 << 28).unwrap();
    let receipts = (0..geometry.shard_count())
        .map(|index| topology_receipt(geometry.shard(index).unwrap(), 1, 100 + index * 16))
        .collect::<Vec<_>>();
    let batch = AddressAtomTopologyBatchReceipt::new(1 << 28, receipts.clone()).unwrap();
    assert_eq!(batch.receipts().len(), 4);
    assert_eq!(batch.total_rows(), 1 << 28);

    let mut duplicate = receipts;
    duplicate[1] = duplicate[0];
    assert_eq!(
        AddressAtomTopologyBatchReceipt::new(1 << 28, duplicate),
        Err(AddressAtomError::ReceiptShard { index: 1 })
    );
}
