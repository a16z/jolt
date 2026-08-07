use super::super::BooleanityRow;
use super::{
    abi::shader_count, BytecodeReadRafConfig, BytecodeReadRafError, BytecodeReadRafIndirectGrid,
    BytecodeReadRafPushforwardParams, BytecodeReadRafRowWords, BytecodeReadRafRun,
    BytecodeReadRafShape, BytecodeReadRafStatus, BytecodeReadRafTopology, BytecodeReadRafWorkload,
    BYTECODE_ADDRESS_SIMD_WIDTH,
};

/// Direct-dispatch plan for the first executable long-worker experiment.
///
/// Each outer block contributes one run of `inner_length` rows. The caller must
/// borrow the stage-5 `BooleanityRows` allocation, verify its length and device,
/// and keep it alive through command completion. Occurrences, the run arena,
/// equality tables, deferred sums, and canonical output remain caller-owned and
/// resident between samples. This plan bypasses CSR and cannot admit a proof.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BytecodeReadRafLongWorkerSlicePlan {
    shape: BytecodeReadRafShape,
    long_runs: u32,
    threadgroups: u32,
    params: BytecodeReadRafPushforwardParams,
}

impl BytecodeReadRafLongWorkerSlicePlan {
    pub fn new(
        shape: BytecodeReadRafShape,
        config: BytecodeReadRafConfig,
    ) -> Result<Self, BytecodeReadRafError> {
        let params = config.pushforward_params(shape)?;
        if config.short_threshold >= shape.inner_length() {
            return Err(BytecodeReadRafError::InfeasibleRunPartition {
                rows: shape.rows(),
                runs: shape.outer_length(),
                long_runs: shape.outer_length(),
                short_threshold: config.short_threshold,
            });
        }
        let _ = BytecodeReadRafWorkload::new(
            shape,
            shape.outer_length(),
            shape.outer_length(),
            config.short_threshold,
        )?;
        let workers = config.long_threads / BYTECODE_ADDRESS_SIMD_WIDTH;
        let threadgroups = shape.outer_length().div_ceil(workers);
        Ok(Self {
            shape,
            long_runs: shader_count("long-worker slice runs", shape.outer_length())?,
            threadgroups: shader_count("long-worker slice threadgroups", threadgroups)?,
            params,
        })
    }

    pub const fn shape(self) -> BytecodeReadRafShape {
        self.shape
    }

    pub const fn long_runs(self) -> u32 {
        self.long_runs
    }

    pub const fn long_threads(self) -> u32 {
        self.params.long_threads
    }

    pub const fn pushforward_params(self) -> BytecodeReadRafPushforwardParams {
        self.params
    }

    pub const fn long_grid(self) -> BytecodeReadRafIndirectGrid {
        BytecodeReadRafIndirectGrid {
            threadgroups_x: self.threadgroups,
            threadgroups_y: 1,
            threadgroups_z: 1,
            reserved: 0,
        }
    }

    /// Returns the tail-growing arena slot read by the long-worker shader.
    pub const fn run_arena_index(self, run_index: usize) -> Option<usize> {
        if run_index < self.long_runs as usize {
            Some(self.shape.run_capacity() - 1 - run_index)
        } else {
            None
        }
    }

    /// Writes logical long runs into the tail-growing shader arena.
    pub fn write_run_arena(
        self,
        topology: &BytecodeReadRafTopology,
        arena: &mut [BytecodeReadRafRun],
    ) -> Result<(), BytecodeReadRafError> {
        self.validate_topology(topology)?;
        if arena.len() != self.shape.run_capacity() {
            return Err(BytecodeReadRafError::TopologyInvariant);
        }
        for (run_index, run) in topology.long_runs.iter().copied().enumerate() {
            let arena_index = self
                .run_arena_index(run_index)
                .ok_or(BytecodeReadRafError::TopologyInvariant)?;
            arena[arena_index] = run;
        }
        Ok(())
    }

    pub fn validate_topology(
        self,
        topology: &BytecodeReadRafTopology,
    ) -> Result<(), BytecodeReadRafError> {
        let identity_occurrences = topology
            .occurrences
            .iter()
            .enumerate()
            .all(|(row, &occurrence)| occurrence as usize == row);
        let valid_runs = topology.long_runs.iter().enumerate().all(|(outer, run)| {
            run.start() as usize == outer * self.shape.inner_length()
                && run.count() as usize == self.shape.inner_length()
                && run.outer() as usize == outer
                && (run.address() as usize) < self.shape.addresses()
        });
        if !topology.short_runs.is_empty()
            || topology.long_runs.len() != self.long_runs as usize
            || topology.occurrences.len() != self.shape.rows()
            || !identity_occurrences
            || !valid_runs
        {
            return Err(BytecodeReadRafError::TopologyInvariant);
        }
        Ok(())
    }

    /// Counter buffer for a direct worker dispatch. Other status words remain
    /// zero so this value cannot pass production CSR admission.
    pub const fn worker_counters(self) -> BytecodeReadRafStatus {
        BytecodeReadRafStatus {
            short_runs: 0,
            long_runs: self.long_runs,
            invalid_rows: 0,
            completed_groups: 0,
            occurrence_rows: 0,
            reserved: [0; 3],
        }
    }
}

/// Builds the deterministic topology used by the long-worker experiment.
///
/// Every row in an outer block must push to the same bytecode address. The
/// occurrence array is the identity permutation; long descriptors remain in
/// logical order and the harness writes them to `run_arena_index(i)`.
pub fn build_long_worker_slice_topology(
    rows: &[BytecodeReadRafRowWords],
    shape: BytecodeReadRafShape,
    short_threshold: usize,
) -> Result<BytecodeReadRafTopology, BytecodeReadRafError> {
    build_long_worker_slice_topology_with(rows.len(), shape, short_threshold, |index| rows[index])
}

pub fn build_long_worker_slice_topology_from_booleanity_rows(
    rows: &[BooleanityRow],
    shape: BytecodeReadRafShape,
    short_threshold: usize,
) -> Result<BytecodeReadRafTopology, BytecodeReadRafError> {
    build_long_worker_slice_topology_with(rows.len(), shape, short_threshold, |index| {
        BytecodeReadRafRowWords::from_words(rows[index].words())
    })
}

fn build_long_worker_slice_topology_with(
    row_count: usize,
    shape: BytecodeReadRafShape,
    short_threshold: usize,
    row: impl Fn(usize) -> BytecodeReadRafRowWords,
) -> Result<BytecodeReadRafTopology, BytecodeReadRafError> {
    if row_count != shape.rows() {
        return Err(BytecodeReadRafError::RowCount {
            expected: shape.rows(),
            got: row_count,
        });
    }
    let _ = BytecodeReadRafWorkload::new(
        shape,
        shape.outer_length(),
        shape.outer_length(),
        short_threshold,
    )?;

    let occurrences = (0..shape.rows())
        .map(|row| shader_count("long-worker slice occurrence", row))
        .collect::<Result<Vec<_>, _>>()?;
    let mut long_runs = Vec::with_capacity(shape.outer_length());
    for outer in 0..shape.outer_length() {
        let start = outer * shape.inner_length();
        let end = start + shape.inner_length();
        let address = checked_slice_pc(start, row(start), shape.addresses())?;
        for row_index in start + 1..end {
            let row_address = checked_slice_pc(row_index, row(row_index), shape.addresses())?;
            if row_address != address {
                return Err(BytecodeReadRafError::TopologyInvariant);
            }
        }
        long_runs.push(BytecodeReadRafRun::new(
            start,
            shape.inner_length(),
            outer,
            address,
        )?);
    }
    Ok(BytecodeReadRafTopology {
        occurrences,
        short_runs: Vec::new(),
        long_runs,
    })
}

fn checked_slice_pc(
    row_index: usize,
    row: BytecodeReadRafRowWords,
    addresses: usize,
) -> Result<usize, BytecodeReadRafError> {
    let pc = row.push_pc() as usize;
    if pc < addresses {
        Ok(pc)
    } else {
        Err(BytecodeReadRafError::MappedPcOutsideDomain {
            row: row_index,
            pc,
            addresses,
        })
    }
}
