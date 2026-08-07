use std::mem::size_of;

use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HammingWeightLaunch {
    pub threadgroups: usize,
    pub threads_per_threadgroup: usize,
    pub threadgroup_memory_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HammingWeightBufferRequirements {
    pub resident_rows: usize,
    pub e_in: usize,
    pub e_out: usize,
    pub partials: usize,
    pub output: usize,
    pub audit_rows: usize,
    pub status: usize,
}

impl HammingWeightBufferRequirements {
    pub fn owned_bytes(self) -> Result<usize, HammingWeightSuccessorError> {
        [
            checked_field_bytes(self.e_in)?,
            checked_field_bytes(self.e_out)?,
            checked_field_bytes(self.partials)?,
            checked_field_bytes(self.output)?,
            checked_bytes::<HammingWeightAuditRow>(self.audit_rows)?,
            checked_bytes::<HammingWeightStatus>(self.status)?,
        ]
        .into_iter()
        .try_fold(0usize, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or(HammingWeightSuccessorError::Overflow)
        })
    }

    pub fn validate(
        self,
        actual: HammingWeightBufferLengths,
    ) -> Result<(), HammingWeightSuccessorError> {
        check_length("resident rows", self.resident_rows, actual.resident_rows)?;
        check_length("inner equality", self.e_in, actual.e_in)?;
        check_length("outer equality", self.e_out, actual.e_out)?;
        check_length("compact partials", self.partials, actual.partials)?;
        check_length("output masses", self.output, actual.output)?;
        check_length("audit rows", self.audit_rows, actual.audit_rows)?;
        check_length("status words", self.status, actual.status)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HammingWeightBufferLengths {
    pub resident_rows: usize,
    pub e_in: usize,
    pub e_out: usize,
    pub partials: usize,
    pub output: usize,
    pub audit_rows: usize,
    pub status: usize,
}

impl From<HammingWeightBufferRequirements> for HammingWeightBufferLengths {
    fn from(requirements: HammingWeightBufferRequirements) -> Self {
        Self {
            resident_rows: requirements.resident_rows,
            e_in: requirements.e_in,
            e_out: requirements.e_out,
            partials: requirements.partials,
            output: requirements.output,
            audit_rows: requirements.audit_rows,
            status: requirements.status,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HammingWeightSlicePlan {
    shape: HammingWeightShape,
    params: HammingWeightHistogramParams,
    requirements: HammingWeightBufferRequirements,
    histogram: HammingWeightLaunch,
    finalize: HammingWeightLaunch,
}

impl HammingWeightSlicePlan {
    pub fn new(
        rows: usize,
        config: HammingWeightSuccessorConfig,
        topology: HammingWeightProtocolTopology,
    ) -> Result<Self, HammingWeightSuccessorError> {
        let selectors = topology.selectors()?;
        let shape = HammingWeightShape::new(rows, config)?;
        let partials = checked_product(&[
            shape.outer_length(),
            selectors,
            HAMMING_WEIGHT_RETAINED_BINS,
        ])?;
        let maximum_partial_index = partials
            .checked_sub(1)
            .ok_or(HammingWeightSuccessorError::Overflow)?;
        let _ = u32::try_from(maximum_partial_index).map_err(|_| {
            HammingWeightSuccessorError::ShaderCountOverflow {
                name: "maximum partial index",
                value: maximum_partial_index,
            }
        })?;
        let output = checked_product(&[selectors, topology.bins])?;
        let requirements = HammingWeightBufferRequirements {
            resident_rows: shape.rows(),
            e_in: shape.inner_length(),
            e_out: shape.outer_length(),
            partials,
            output,
            audit_rows: shape.outer_length(),
            status: 1,
        };
        let _ = requirements.owned_bytes()?;

        Ok(Self {
            shape,
            params: shape.params()?,
            requirements,
            histogram: HammingWeightLaunch {
                threadgroups: shape.outer_length(),
                threads_per_threadgroup: HAMMING_WEIGHT_THREADS,
                threadgroup_memory_bytes: HAMMING_WEIGHT_THREADGROUP_BYTES,
            },
            finalize: HammingWeightLaunch {
                threadgroups: selectors,
                threads_per_threadgroup: HAMMING_WEIGHT_BINS,
                threadgroup_memory_bytes: 0,
            },
        })
    }

    pub const fn shape(self) -> HammingWeightShape {
        self.shape
    }

    pub const fn params(self) -> HammingWeightHistogramParams {
        self.params
    }

    pub const fn requirements(self) -> HammingWeightBufferRequirements {
        self.requirements
    }

    pub const fn histogram_launch(self) -> HammingWeightLaunch {
        self.histogram
    }

    pub const fn finalize_launch(self) -> HammingWeightLaunch {
        self.finalize
    }

    pub fn partial_index(
        self,
        outer: usize,
        selector: usize,
        bin: usize,
    ) -> Result<usize, HammingWeightSuccessorError> {
        if outer >= self.shape.outer_length() {
            return Err(index_error(
                "partial outer",
                self.shape.outer_length(),
                outer,
            ));
        }
        if selector >= HAMMING_WEIGHT_SELECTORS {
            return Err(index_error(
                "partial selector",
                HAMMING_WEIGHT_SELECTORS,
                selector,
            ));
        }
        if !(1..HAMMING_WEIGHT_BINS).contains(&bin) {
            return Err(index_error(
                "partial retained bin",
                HAMMING_WEIGHT_RETAINED_BINS,
                bin,
            ));
        }
        checked_product(&[outer, HAMMING_WEIGHT_SELECTORS])?
            .checked_add(selector)
            .and_then(|value| value.checked_mul(HAMMING_WEIGHT_RETAINED_BINS))
            .and_then(|value| value.checked_add(bin - 1))
            .ok_or(HammingWeightSuccessorError::Overflow)
    }

    pub fn output_index(
        self,
        selector: usize,
        bin: usize,
    ) -> Result<usize, HammingWeightSuccessorError> {
        if selector >= HAMMING_WEIGHT_SELECTORS {
            return Err(index_error(
                "output selector",
                HAMMING_WEIGHT_SELECTORS,
                selector,
            ));
        }
        if bin >= HAMMING_WEIGHT_BINS {
            return Err(index_error("output bin", HAMMING_WEIGHT_BINS, bin));
        }
        selector
            .checked_mul(HAMMING_WEIGHT_BINS)
            .and_then(|value| value.checked_add(bin))
            .ok_or(HammingWeightSuccessorError::Overflow)
    }
}

fn checked_bytes<T>(elements: usize) -> Result<usize, HammingWeightSuccessorError> {
    elements
        .checked_mul(size_of::<T>())
        .ok_or(HammingWeightSuccessorError::Overflow)
}

fn checked_field_bytes(elements: usize) -> Result<usize, HammingWeightSuccessorError> {
    elements
        .checked_mul(HAMMING_WEIGHT_FIELD_BYTES)
        .ok_or(HammingWeightSuccessorError::Overflow)
}

fn checked_product(values: &[usize]) -> Result<usize, HammingWeightSuccessorError> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(HammingWeightSuccessorError::Overflow)
    })
}

fn check_length(
    name: &'static str,
    expected: usize,
    got: usize,
) -> Result<(), HammingWeightSuccessorError> {
    if got == expected {
        Ok(())
    } else {
        Err(HammingWeightSuccessorError::StorageLength {
            name,
            expected,
            got,
        })
    }
}

fn index_error(name: &'static str, expected: usize, got: usize) -> HammingWeightSuccessorError {
    HammingWeightSuccessorError::StorageLength {
        name,
        expected,
        got,
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests use checked production shapes")]
mod tests {
    use super::*;

    #[test]
    fn log_26_plan_matches_shader_topology() {
        let plan = HammingWeightSlicePlan::new(
            HAMMING_WEIGHT_TARGET_ROWS,
            HammingWeightSuccessorConfig::default(),
            HammingWeightProtocolTopology::PRODUCTION,
        )
        .unwrap();
        let requirements = plan.requirements();
        assert_eq!(plan.shape().inner_length(), 1 << 18);
        assert_eq!(plan.shape().outer_length(), 256);
        assert_eq!(requirements.partials, 1_893_120);
        assert_eq!(requirements.output, 7_424);
        assert_eq!(requirements.audit_rows, 256);
        assert_eq!(requirements.owned_bytes().unwrap(), 34_615_312);
        assert_eq!(plan.histogram_launch().threadgroups, 256);
        assert_eq!(plan.histogram_launch().threads_per_threadgroup, 928);
        assert_eq!(plan.histogram_launch().threadgroup_memory_bytes, 23_232);
        assert_eq!(plan.finalize_launch().threadgroups, 29);
        assert_eq!(plan.finalize_launch().threads_per_threadgroup, 256);
        requirements.validate(requirements.into()).unwrap();
    }

    #[test]
    fn compact_partial_and_output_indices_cover_exact_ranges() {
        let plan = HammingWeightSlicePlan::new(
            HAMMING_WEIGHT_TARGET_ROWS,
            HammingWeightSuccessorConfig::default(),
            HammingWeightProtocolTopology::PRODUCTION,
        )
        .unwrap();
        assert_eq!(plan.partial_index(0, 0, 1).unwrap(), 0);
        assert_eq!(
            plan.partial_index(255, 28, 255).unwrap(),
            plan.requirements().partials - 1
        );
        assert_eq!(plan.output_index(0, 0).unwrap(), 0);
        assert_eq!(
            plan.output_index(28, 255).unwrap(),
            plan.requirements().output - 1
        );
        assert!(plan.partial_index(0, 0, 0).is_err());
        assert!(plan.output_index(29, 0).is_err());
    }

    #[test]
    fn topology_and_lengths_fail_closed() {
        let bad_topology = HammingWeightProtocolTopology {
            bytecode_selectors: 1,
            ..HammingWeightProtocolTopology::PRODUCTION
        };
        assert!(matches!(
            HammingWeightSlicePlan::new(
                HAMMING_WEIGHT_TARGET_ROWS,
                HammingWeightSuccessorConfig::default(),
                bad_topology,
            ),
            Err(HammingWeightSuccessorError::UnsupportedTopology {
                name: "bytecode selectors",
                expected: 2,
                got: 1,
            })
        ));

        let plan = HammingWeightSlicePlan::new(
            HAMMING_WEIGHT_TARGET_ROWS,
            HammingWeightSuccessorConfig::default(),
            HammingWeightProtocolTopology::PRODUCTION,
        )
        .unwrap();
        let requirements = plan.requirements();
        let mut lengths = HammingWeightBufferLengths::from(requirements);
        lengths.partials -= 1;
        assert!(matches!(
            requirements.validate(lengths),
            Err(HammingWeightSuccessorError::StorageLength {
                name: "compact partials",
                ..
            })
        ));
    }
}
