//! The stage-6b claims the byte link authenticates against `Q` (spec §3).

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::lattice::byte_link::ByteLinkInputs;
use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltRelationId};
use jolt_field::JoltField;
use jolt_openings::EvaluationClaim;

use super::outputs::Stage6bClearOutput;
use crate::VerifierError;

impl<F: JoltField> Stage6bClearOutput<F> {
    /// Every one-hot opening of the batch, at its own address chunk and the
    /// shared cycle point, and the fused increment `F(r)`. The values are the
    /// inclusive evaluations the retained relations consumed: row zero is
    /// part of each one-hot column, so no zero-row term is removed.
    pub fn byte_link_inputs(&self) -> Result<ByteLinkInputs<F>, VerifierError> {
        let (values, points) = (&self.output_values, &self.output_points);
        let mut claims = BTreeMap::new();
        let mut route = |column: fn(usize) -> JoltCommittedPolynomial,
                         values: &[F],
                         points: &[Vec<F>]| {
            for (index, (value, point)) in values.iter().zip(points).enumerate() {
                let _ = claims.insert(column(index), EvaluationClaim::new(point.clone(), *value));
            }
        };
        route(
            JoltCommittedPolynomial::InstructionRa,
            &values
                .instruction_ra_virtualization
                .committed_instruction_ra,
            &points
                .instruction_ra_virtualization
                .committed_instruction_ra,
        );
        route(
            JoltCommittedPolynomial::BytecodeRa,
            &values.bytecode_read_raf.bytecode_ra,
            &points.bytecode_read_raf.bytecode_ra,
        );
        route(
            JoltCommittedPolynomial::RamRa,
            &values.ram_ra_virtualization.ram_ra,
            &points.ram_ra_virtualization.ram_ra,
        );
        let fused_inc = EvaluationClaim::new(
            points.fused_inc_opening_point().to_vec(),
            values.bytecode_read_raf.fused_inc,
        );
        ByteLinkInputs::new(&claims, &fused_inc).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::ByteLink,
                reason: error.to_string(),
            }
        })
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "the test indexes its own pack table and source map"
)]
mod tests {
    use super::*;
    use crate::stages::stage6b::bytecode_read_raf::LatticeBytecodeReadRafOutputClaims;
    use crate::stages::stage6b::outputs::{
        InstructionRaVirtualizationOutputClaims, RamRaVirtualizationOutputClaims,
        Stage6bOutputClaims, Stage6bOutputPoints,
    };
    use jolt_claims::protocols::jolt::lattice::byte_link::{BYTE_LINK_PACKS, RAM_PACK};
    use jolt_field::{Fr, Ring};

    #[test]
    fn routes_each_one_hot_claim_to_its_own_column_inclusively() {
        let r = [100, 101, 102].map(Fr::from_u64).to_vec();
        let mut seed = Fr::from_u64(0);
        let mut sources = BTreeMap::new();
        let mut family = |column: fn(usize) -> JoltCommittedPolynomial, count: usize| {
            let mut values = Vec::new();
            let mut points = Vec::new();
            for index in 0..count {
                seed += Fr::from_u64(1);
                let address = (0..8)
                    .map(|bit| seed * Fr::from_u64(1000) + Fr::from_u64(bit))
                    .collect::<Vec<_>>();
                values.push(seed);
                points.push([address.clone(), r.clone()].concat());
                let _ = sources.insert(column(index), (seed, address));
            }
            (values, points)
        };
        let (instruction_values, instruction_points) =
            family(JoltCommittedPolynomial::InstructionRa, 16);
        let (bytecode_values, bytecode_points) = family(JoltCommittedPolynomial::BytecodeRa, 2);
        let (ram_values, ram_points) = family(JoltCommittedPolynomial::RamRa, 2);
        let fused_inc = Fr::from_u64(7);
        let output = Stage6bClearOutput {
            output_values: Stage6bOutputClaims {
                bytecode_read_raf: LatticeBytecodeReadRafOutputClaims {
                    bytecode_ra: bytecode_values,
                    fused_inc,
                },
                ram_ra_virtualization: RamRaVirtualizationOutputClaims { ram_ra: ram_values },
                instruction_ra_virtualization: InstructionRaVirtualizationOutputClaims {
                    committed_instruction_ra: instruction_values,
                },
                bytecode_reduction: None,
                program_image_reduction: None,
            },
            output_points: Stage6bOutputPoints {
                bytecode_read_raf: LatticeBytecodeReadRafOutputClaims {
                    bytecode_ra: bytecode_points,
                    fused_inc: r.clone(),
                },
                ram_ra_virtualization: RamRaVirtualizationOutputClaims { ram_ra: ram_points },
                instruction_ra_virtualization: InstructionRaVirtualizationOutputClaims {
                    committed_instruction_ra: instruction_points,
                },
                bytecode_reduction: None,
                program_image_reduction: None,
            },
            bytecode_reduction_weights: None,
        };

        let inputs = output.byte_link_inputs().unwrap();
        assert_eq!(inputs.cycle_point(), r.as_slice());
        assert_eq!(inputs.fused_inc(), fused_inc);
        let columns = BYTE_LINK_PACKS
            .iter()
            .flatten()
            .filter(|column| **column != JoltCommittedPolynomial::RamActivity);
        let queries = inputs.histogram_queries();
        assert_eq!(queries.len(), sources.len());
        for (query, column) in queries.iter().zip(columns) {
            let (value, address) = &sources[column];
            let byte = BYTE_LINK_PACKS[query.pack]
                .iter()
                .position(|packed| packed == column)
                .unwrap();
            let unused_bytes_scale = Fr::from_u64(if query.pack == RAM_PACK {
                1 << 8
            } else {
                1 << 16
            });
            let window = query.point.chunks(8).nth(byte).unwrap();
            assert_eq!(window, address.as_slice(), "{column:?}");
            assert_eq!(query.value * unused_bytes_scale, *value, "{column:?}");
        }
    }
}
