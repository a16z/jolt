use cudarc::driver::{LaunchConfig, PushKernelArg};
use jolt_claims::protocols::composed::r1cs::spartan_outer_row_weights;
use jolt_claims::protocols::jolt::geometry::dimensions::OUTER_UNISKIP_DOMAIN_SIZE;
use jolt_field::JoltField;
use jolt_poly::lagrange::{centered_lagrange_evals, interpolate_to_coeffs, poly_mul};
use jolt_poly::UnivariatePoly;
use jolt_r1cs::constraint::ConstraintMatrices;

use super::columns::{DeviceR1csInputs, LinearForms};
use super::guards::{GuardForms, NODES, SHARED_BYTES};
use crate::cuda::common::context::{CudaKernelContext, BLOCK};
use crate::cuda::common::device::require_fr;
use crate::cuda::common::devices::CycleWindow;
use crate::cuda::common::error::CudaError;
use crate::cuda::common::primitives::reduce_lanes;
use crate::cuda::common::split_eq::split_eq_tables_window;

pub const EXTENDED_SIZE: usize = 2 * OUTER_UNISKIP_DOMAIN_SIZE - 1;

pub const STRIP: usize = 4;

pub fn extended_nodes() -> Vec<(usize, i64)> {
    let domain_start = -((OUTER_UNISKIP_DOMAIN_SIZE as i64 - 1) / 2);
    let extended_start = -((EXTENDED_SIZE as i64 - 1) / 2);
    let domain_end = domain_start + OUTER_UNISKIP_DOMAIN_SIZE as i64;
    (0..EXTENDED_SIZE)
        .filter_map(|position| {
            let node = extended_start + position as i64;
            (node < domain_start || node >= domain_end).then_some((position, node))
        })
        .collect()
}

pub fn node_field<F: JoltField>(node: i64) -> F {
    if node >= 0 {
        F::from_u64(node as u64)
    } else {
        -F::from_u64(node.unsigned_abs())
    }
}

pub fn push_forms<F: JoltField>(
    forms: &mut LinearForms,
    matrices: &ConstraintMatrices<F>,
    columns: &[usize],
    point: F,
    stream: F,
) -> Result<(), CudaError> {
    let weights =
        spartan_outer_row_weights(point, stream).map_err(|_| CudaError::InvariantViolation {
            reason: "the Spartan outer uni-skip domain is not a valid centered integer domain",
        })?;
    let weighted = matrices.weighted_columns(&weights, columns).map_err(|_| {
        CudaError::InvariantViolation {
            reason: "the Spartan outer constraint matrices do not span the relation's variables",
        }
    })?;
    let constants = matrices
        .public_column_contributions(&weights, 0, F::one())
        .map_err(|_| CudaError::InvariantViolation {
            reason: "the Spartan outer constraint matrices have no public column",
        })?;
    forms.push(&weighted.a, constants.a)?;
    forms.push(&weighted.b, constants.b)?;
    Ok(())
}

#[cfg(test)]
pub fn extended_evals<F: JoltField>(
    context: &CudaKernelContext,
    inputs: &DeviceR1csInputs,
    matrices: &ConstraintMatrices<F>,
    tau: &[F],
    log_t: usize,
) -> Result<Vec<F>, CudaError> {
    extended_evals_window(context, inputs, matrices, tau, log_t, 0, 1, inputs.cycles())
}

#[expect(
    clippy::too_many_arguments,
    reason = "one cycle window of the uni-skip reduce: the shard index, shard count and window               length all pick out which cycles this device sums over"
)]
pub fn extended_evals_window<F: JoltField>(
    context: &CudaKernelContext,
    inputs: &DeviceR1csInputs,
    matrices: &ConstraintMatrices<F>,
    tau: &[F],
    log_t: usize,
    shard: usize,
    shards: usize,
    cycles: usize,
) -> Result<Vec<F>, CudaError> {
    let nodes = extended_nodes();
    let device_forms = GuardForms::new(matrices)?.upload(context)?;

    let (e_in, e_out, in_bits) = split_eq_tables_window(context, &tau[..=log_t], shard, shards)?;

    if cycles == 0 || cycles * 2 > e_in.len() * e_out.len() {
        return Err(CudaError::InvariantViolation {
            reason: "the Spartan outer equality tables do not cover its cycle window",
        });
    }
    let mass = inputs.tail_weight(
        &tau[..log_t],
        &CycleWindow {
            start: shard * (e_in.len() * e_out.len() / 2),
            len: cycles,
        },
    )?;
    let stream = tau[log_t];
    let tail_weights = context.upload(&[
        require_fr(mass * (F::one() - stream))?,
        require_fr(mass * stream)?,
    ])?;
    let retained = inputs.cycles().min(cycles);
    let group_cycles = (e_in.len() / 2).min(BLOCK as usize * STRIP);
    let blocks = u32::try_from((retained - 1).div_ceil(group_cycles) + 1).map_err(|_| {
        CudaError::InvariantViolation {
            reason: "the Spartan outer uni-skip launch exceeds a u32 block count",
        }
    })?;
    let lanes = u32::try_from(2 * NODES).map_err(|_| CudaError::InvariantViolation {
        reason: "the Spartan outer uni-skip launch exceeds a u32 lane count",
    })?;
    let mut partials = context.alloc(lanes as usize * blocks as usize)?;

    let count = CudaKernelContext::count_of(retained)?;
    let tile = CudaKernelContext::count_of(group_cycles)?;
    let bits = CudaKernelContext::count_of(in_bits)?;
    let mut builder = context.stream().launch_builder(context.so_uniskip());
    let _ = builder.arg(inputs.narrow());
    let _ = builder.arg(inputs.wide());
    let _ = builder.arg(inputs.flags());
    device_forms.bind_args(&mut builder);
    let _ = builder.arg(e_in.limbs());
    let _ = builder.arg(e_out.limbs());
    let _ = builder.arg(&bits);
    let _ = builder.arg(&count);
    let _ = builder.arg(&tile);
    let _ = builder.arg(tail_weights.limbs());
    let _ = builder.arg(partials.limbs_mut());
    // SAFETY: each block stays inside one inner equality-table tile and guards
    // every input read by `t < retained`. A separate last block evaluates the
    // representative with full suffix weights, without dividing by an outer
    // factor that may be zero. GuardForms validates the native layout,
    // integer bounds, and ten-node domain. Its offsets cover all 40 forms.
    // The two stream blocks each write NODES lanes within the partials buffer.
    // SHARED_BYTES covers the 32 keys, grouped and overflow accumulators, and
    // node totals. All threads reach the barriers between those phases.
    let _ = unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (blocks, 2, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: SHARED_BYTES as u32,
        })
    }?;

    let totals = reduce_lanes(context, partials, lanes, blocks)?;
    let host = totals.to_host()?;
    let mut values = vec![F::zero(); EXTENDED_SIZE];
    for (node, &(position, _)) in nodes.iter().enumerate() {
        values[position] = crate::cuda::common::device::fr_into(host[node] + host[NODES + node])
            .ok_or(CudaError::NotImplemented {
                kernel: "CUDA kernels support only the BN254 scalar field",
            })?;
    }
    Ok(values)
}

pub fn first_round_poly<F: JoltField>(
    extended: &[F],
    tau_high: F,
) -> Result<UnivariatePoly<F>, CudaError> {
    let domain_start = -((OUTER_UNISKIP_DOMAIN_SIZE as i64 - 1) / 2);
    let extended_start = -((EXTENDED_SIZE as i64 - 1) / 2);
    let kernel_values =
        centered_lagrange_evals::<F>(OUTER_UNISKIP_DOMAIN_SIZE, tau_high).map_err(|_| {
            CudaError::InvariantViolation {
                reason: "the Spartan outer uni-skip domain is not a valid centered integer domain",
            }
        })?;
    let kernel_coefficients = interpolate_to_coeffs(domain_start, &kernel_values);
    let t1_coefficients = interpolate_to_coeffs(extended_start, extended);
    Ok(UnivariatePoly::new(poly_mul(
        &kernel_coefficients,
        &t1_coefficients,
    )))
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use std::collections::BTreeSet;

    use jolt_claims::protocols::composed::r1cs::{
        spartan_outer_constraints, spartan_outer_row_weights, SPARTAN_OUTER_FIRST_GROUP_ROWS,
        SPARTAN_OUTER_SECOND_GROUP_ROWS,
    };
    use jolt_field::{Fr, One, Ring, Zero};
    use jolt_poly::EqPolynomial;

    use super::{extended_evals_window, extended_nodes, node_field, EXTENDED_SIZE};
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::error::CudaError;
    use crate::cuda::common::testing::fr;
    use crate::cuda::spartan_outer::columns::DeviceR1csInputs;
    use crate::cuda::spartan_outer::witness::{
        Packed, KIND_FLAG, KIND_NARROW, LAYOUT, NARROW, SIGN_BIT_BASE, VARIABLES, WIDE,
    };

    #[test]
    fn uniskip_rejects_empty_and_oversized_equality_windows() {
        let Some(context) = shared_context() else {
            return;
        };
        let packed = Packed {
            narrow: vec![0; 16 * NARROW],
            wide: vec![0; 16 * WIDE * 2],
            flags: vec![0; 16],
        };
        let inputs = DeviceR1csInputs::new(context, &packed).expect("native inputs");
        let matrices = spartan_outer_constraints::<Fr>();
        let tau = vec![Fr::from_u64(7); 5];
        for cycles in [0, 9] {
            assert!(matches!(
                extended_evals_window(context, &inputs, &matrices, &tau, 3, 0, 1, cycles),
                Err(CudaError::InvariantViolation { .. })
            ));
        }
    }

    #[test]
    fn native_uniskip_matches_field_matrix_with_repeated_and_diverse_guards() {
        let Some(context) = shared_context() else {
            return;
        };
        const LOG_T: usize = 12;
        const CYCLES: usize = 1 << LOG_T;
        let matrices = spartan_outer_constraints::<Fr>();
        let columns: Vec<usize> = (0..=VARIABLES).collect();
        let nodes = extended_nodes();
        let forms: Vec<_> = nodes
            .iter()
            .flat_map(|&(_, node)| {
                let matrices = &matrices;
                let columns = &columns;
                [Fr::zero(), Fr::one()].into_iter().map(move |stream| {
                    let weights = spartan_outer_row_weights(node_field::<Fr>(node), stream)
                        .expect("canonical row weights");
                    matrices
                        .weighted_columns(&weights, columns)
                        .expect("canonical columns")
                })
            })
            .collect();
        for (diverse, exceptional) in [(false, false), (true, false), (true, true)] {
            let word = |i: usize| [0, u64::MAX, 1, 1 << 63, u64::MAX - 1][i % 5];
            let packed = Packed {
                narrow: (0..CYCLES * NARROW).map(word).collect(),
                wide: (0..CYCLES * WIDE * 2).map(|i| word(i + 1)).collect(),
                flags: (0..CYCLES)
                    .map(|t| {
                        let flags = if diverse {
                            (t as u32).wrapping_mul(0x9e37_79b9).rotate_left(5) & 0xffff
                        } else {
                            0x1044
                        };
                        flags | ((t as u32 % 8) << SIGN_BIT_BASE)
                    })
                    .collect(),
            };
            let device = DeviceR1csInputs::new(context, &packed).expect("native inputs");
            let narrow = context
                .download_u64(device.narrow())
                .expect("narrow columns");
            let wide = context.download_u64(device.wide()).expect("wide columns");
            let flags = context.download_u32(device.flags()).expect("shifted flags");
            let mut tau: Vec<Fr> = (0..LOG_T + 2).map(|i| fr(23 + i as u64)).collect();
            if exceptional {
                tau[0] = Fr::zero();
                tau[LOG_T] = Fr::one();
            }
            let cycles = CYCLES - 3;
            let weights = EqPolynomial::<Fr>::evals(&tau[..=LOG_T], None);
            let got = extended_evals_window(context, &device, &matrices, &tau, LOG_T, 0, 1, cycles)
                .expect("native extension");
            let mut expected = vec![Fr::zero(); EXTENDED_SIZE];
            let mut guard_patterns = [BTreeSet::new(), BTreeSet::new()];
            for t in 0..cycles {
                let mut z = vec![Fr::one()];
                z.extend(LAYOUT.map(|layout| {
                    let slot = (layout & 255) as usize;
                    match layout >> 8 {
                        KIND_NARROW => Fr::from_u64(narrow[t * NARROW + slot]),
                        KIND_FLAG => Fr::from_u64(u64::from((flags[t] >> slot) & 1)),
                        _ => {
                            let magnitude = u128::from(wide[t * WIDE * 2 + 2 * slot])
                                | (u128::from(wide[t * WIDE * 2 + 2 * slot + 1]) << 64);
                            let value = Fr::from_u128(magnitude);
                            if (flags[t] >> (SIGN_BIT_BASE + slot as u32)) & 1 != 0 {
                                -value
                            } else {
                                value
                            }
                        }
                    }
                }));
                if diverse && t < 64 {
                    let guards: Vec<_> = matrices
                        .a
                        .iter()
                        .map(|row| {
                            row.iter()
                                .map(|&(col, coefficient)| coefficient * z[col])
                                .sum::<Fr>()
                        })
                        .collect();
                    for (patterns, indices) in guard_patterns.iter_mut().zip([
                        SPARTAN_OUTER_FIRST_GROUP_ROWS.as_slice(),
                        SPARTAN_OUTER_SECOND_GROUP_ROWS.as_slice(),
                    ]) {
                        let pattern: Vec<_> = indices.iter().map(|&i| guards[i]).collect();
                        let _ = patterns.insert(format!("{pattern:?}"));
                    }
                }
                for (i, form) in forms.iter().enumerate() {
                    let a: Fr = form.a.iter().zip(&z).map(|(c, v)| *c * *v).sum();
                    let b: Fr = form.b.iter().zip(&z).map(|(c, v)| *c * *v).sum();
                    expected[nodes[i / 2].0] += weights[2 * t + i % 2] * a * b;
                }
            }
            if diverse {
                for patterns in guard_patterns {
                    assert!(
                        patterns.len() > 32,
                        "the fixture must exceed the group capacity"
                    );
                }
            }
            assert_eq!(
                got, expected,
                "diverse={diverse}, exceptional={exceptional}"
            );
        }
    }
}
