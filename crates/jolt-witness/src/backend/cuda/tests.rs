#![expect(
    clippy::expect_used,
    reason = "test module: device and fixture errors fail loudly"
)]

use super::*;
use crate::backend::trace::TraceBackend;
use crate::backend::ProgramSource;
use crate::witnesses::{BytecodePc, LookupIndex, RemappedRamAddress};
use crate::{collect_bundles, WitnessBundle};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use std::sync::Arc;

mod fixtures;
use fixtures::{all_kinds_backend, supported_jolt_kinds};

#[derive(Clone, Copy, Debug, PartialEq, Eq, WitnessBundle)]
struct ColumnProbe {
    lookup_index: LookupIndex,
    bytecode_pc: BytecodePc,
    remapped_ram: RemappedRamAddress,
}

const RAM_K: usize = 1 << 6;

struct Fixture {
    stream: Arc<CudaStream>,
    backend: TraceBackend,
    trace: DeviceTrace,
    cycles: usize,
}

fn fixture(seed: u64) -> Option<Fixture> {
    let stream = CudaContext::new(0).ok()?.default_stream();
    let (backend, cycles, kinds) = all_kinds_backend(seed);
    let physical = backend.trace.trace.proof_rows();
    let present: Vec<_> = physical
        .iter()
        .filter_map(|row| row.instruction_kind())
        .collect();
    let missing: Vec<_> = supported_jolt_kinds()
        .into_iter()
        .filter(|kind| !present.contains(kind))
        .collect();
    assert!(
        missing.is_empty(),
        "{} supported kinds never appear in the fixture trace, so the device dispatch would be \
         untested on them: {:?}",
        missing.len(),
        missing.iter().take(8).collect::<Vec<_>>(),
    );
    assert!(
        kinds > 60,
        "only {kinds} kinds — too narrow to be a coverage claim",
    );

    let trace = DeviceTrace::upload(
        Arc::clone(&stream),
        physical,
        cycles,
        backend.program_preprocessing(),
    )
    .expect("device residency");
    Some(Fixture {
        stream,
        backend,
        trace,
        cycles,
    })
}

impl Fixture {
    fn u32s(&self, device: &CudaSlice<u32>) -> Vec<u32> {
        self.stream.clone_dtoh(device).expect("download")
    }

    fn u64s(&self, device: &CudaSlice<u64>) -> Vec<u64> {
        self.stream.clone_dtoh(device).expect("download")
    }

    fn probes<B: WitnessBundle + Copy + Send + Sync>(&self) -> Vec<B> {
        collect_bundles(&self.backend, self.cycles).expect("reference atoms")
    }
}

#[test]
fn device_columns_match_the_reference_extractors() {
    let Some(fixture) = fixture(11) else {
        return;
    };
    let limbs = fixture.u64s(
        &fixture
            .trace
            .lookup_index_limbs()
            .expect("lookup index limbs"),
    );
    let pc = fixture.u32s(&fixture.trace.mapped_pc_words().expect("mapped pc words"));
    let (ram, _) = fixture
        .trace
        .remapped_ram_words(RAM_K)
        .expect("remapped ram words");
    let ram = fixture.u32s(&ram);
    let rows: Vec<ColumnProbe> = fixture.probes();

    assert!(
        limbs.iter().any(|&limb| limb != 0),
        "every lookup-index limb is zero, so the interleave path is untested",
    );
    assert!(
        ram.contains(&COLD),
        "no cycle is RAM-cold, so the cold sentinel is untested",
    );
    assert!(
        pc.iter().any(|&word| word != pc[0]),
        "every mapped PC is the same, so a kernel ignoring the address would pass",
    );

    for (index, row) in rows.iter().enumerate() {
        let got = u128::from(limbs[2 * index]) | (u128::from(limbs[2 * index + 1]) << 64);
        assert_eq!(got, row.lookup_index.0, "lookup index at {index}");
        assert_eq!(pc[index], row.bytecode_pc.0 as u32, "mapped pc at {index}");
        assert_eq!(
            ram[index],
            row.remapped_ram.0.map_or(COLD, |address| address as u32),
            "remapped ram at {index}",
        );
    }
}
