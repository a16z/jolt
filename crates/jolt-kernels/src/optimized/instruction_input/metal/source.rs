use std::time::Duration;

use super::*;

/// Physical layout is explicit and independent of either Rust witness layout.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct NativeOperands {
    rs1: u64,
    pc: u64,
    rs2: u64,
    rd: u64,
    imm_bits: [u64; 2],
}
const _: () = assert!(size_of::<NativeOperands>() == 48);
impl From<InstructionInputRow> for NativeOperands {
    fn from(row: InstructionInputRow) -> Self {
        let imm = row.imm.0 as u128;
        Self {
            rs1: row.rs1_value.0,
            pc: row.unexpanded_pc.0,
            rs2: row.rs2_value.0,
            rd: row.rd_write_value.0,
            imm_bits: [imm as u64, (imm >> 64) as u64],
        }
    }
}

fn fill(
    rows: &BundleStore<InstructionInputRow>,
    first_row: usize,
    operands: &mut [MaybeUninit<NativeOperands>],
    codes: &mut [MaybeUninit<u16>],
) -> Result<(), WitnessError> {
    let access = rows.access();
    let fill_pair = |(pair, (out, code)): (
        usize,
        (&mut [MaybeUninit<NativeOperands>], &mut MaybeUninit<u16>),
    )|
     -> Result<(), WitnessError> {
        let even = access.row(first_row + 2 * pair)?;
        let odd = access.row(first_row + 2 * pair + 1)?;
        let _ = out[0].write(even.into());
        let _ = out[1].write(odd.into());
        // Exactly five two-bit patterns; the canonical encoder owns ordering.
        let _ = code.write(PackedFlags::<F>::encode(even.flags(), odd.flags()) as u16);
        Ok(())
    };
    #[cfg(feature = "parallel")]
    {
        operands
            .par_chunks_mut(2)
            .zip(codes.par_iter_mut())
            .enumerate()
            .try_for_each(fill_pair)
    }
    #[cfg(not(feature = "parallel"))]
    {
        operands
            .chunks_mut(2)
            .zip(codes.iter_mut())
            .enumerate()
            .try_for_each(fill_pair)
    }
}

pub(super) fn materialize(
    gpu: Arc<Gpu>,
    rows: &BundleStore<InstructionInputRow>,
    prefix: usize,
    challenge: F,
    tails: [F; NUM_TABLES],
    eq: &GruenSplitEqPolynomial<F>,
    gamma: F,
) -> Result<InputPrepared<F>, SumcheckError<F>> {
    let span = tracing::info_span!(
        "metal_input_materialize",
        prefix,
        gpu_s = tracing::field::Empty
    );
    let _span = span.enter();
    let output_len = prefix.div_ceil(2);
    let equality = Equality::new(&gpu.device, eq).map_err(compute_error)?;
    let flag_table = FlagTable::new(
        core::array::from_fn(|i| tails[2 * i] == F::one()),
        challenge,
    );
    let lut = DeviceBuffer::from_slice(&gpu.device, &flag_table.values).map_err(compute_error)?;
    let constants = Constants {
        tails,
        tail_word: words(flag_table.tail),
        challenge,
        gamma,
    };
    // Tile boundaries coincide with message-pair and equality-block boundaries.
    let block_rows = equality
        .inner_len
        .checked_mul(4)
        .ok_or_else(|| compute_error(invalid("input tile alignment overflows")))?;
    let tile_rows = (gpu.tile_rows / block_rows).max(1) * block_rows;
    let tile_outputs = (tile_rows / 2).min(output_len);
    let allocate = tracing::info_span!("metal_input_allocate", rows = output_len).entered();
    let pending = Pending::new(&gpu, output_len, true).map_err(compute_error)?;
    let mut source =
        DeviceBuffer::<MaybeUninit<NativeOperands>>::uninit(&gpu.device, tile_outputs * 2)
            .map_err(compute_error)?;
    let mut codes = DeviceBuffer::<MaybeUninit<u16>>::uninit(&gpu.device, tile_outputs)
        .map_err(compute_error)?;
    drop(allocate);
    let mut terms = [F::zero(); 3];
    let mut gpu_time = Duration::ZERO;
    for first in (0..output_len).step_by(tile_outputs) {
        let count = tile_outputs.min(output_len - first);
        let pack = tracing::info_span!("metal_input_pack", rows = count * 2).entered();
        fill(
            rows,
            first * 2,
            &mut source.write_uninit().map_err(compute_error)?[..count * 2],
            &mut codes.write_uninit().map_err(compute_error)?[..count],
        )
        .map_err(row_extraction_error)?;
        drop(pack);
        let pairs = count.div_ceil(2);
        let groups = equality.groups(first / 2, pairs).map_err(compute_error)?;
        let partials = partials(groups, &gpu.device).map_err(compute_error)?;
        let params = Params {
            prefix: word(prefix).map_err(compute_error)?,
            output_len: word(output_len).map_err(compute_error)?,
            first_pair: word(first / 2).map_err(compute_error)?,
            pairs: word(pairs).map_err(compute_error)?,
            in_len: word(equality.inner_len).map_err(compute_error)?,
            old_bits: 0,
            new_bits: 2,
            need_one: equality.need_one,
        };
        let mut batch = Batch::new(&gpu.device).map_err(compute_error)?;
        // SAFETY: the active tile contains initialized native row pairs and
        // canonical ten-bit codes. Each invocation writes distinct output rows
        // within the global prefix and three partials per equality block. The
        // final unmatched output uses the constant tail instead of input loads.
        unsafe {
            batch
                .dispatch_unchecked(
                    gpu.library
                        .pipeline(&host_name::<F>(NATIVE))
                        .map_err(compute_error)?,
                    &[
                        Binding::buffer(&source),
                        Binding::buffer(&codes),
                        Binding::buffer(&lut),
                        Binding::buffer(&equality.inner),
                        Binding::buffer(&equality.outer),
                        Binding::buffer(&pending.values),
                        pending.words_binding(&gpu),
                        Binding::buffer(&partials),
                        Binding::value(&constants),
                        Binding::value(&params),
                    ],
                    Grid::linear(groups * THREADS, THREADS),
                )
                .map_err(compute_error)?;
        }
        gpu_time += batch.commit_and_wait().map_err(compute_error)?;
        // SAFETY: every partial was written by the completed native dispatch.
        for (sum, value) in terms
            .iter_mut()
            .zip(unsafe { sum_partials(partials) }.map_err(compute_error)?)
        {
            *sum += value;
        }
    }
    let _ = span.record("gpu_s", gpu_time.as_secs_f64());
    // SAFETY: tiles partition all output rows and all dispatches completed.
    let columns = unsafe { pending.complete() }.map_err(compute_error)?;
    Ok(InputPrepared {
        device: Box::new(ResidentInput {
            gpu,
            columns,
            flags: Some(flag_table),
            prefix: output_len,
            tails,
            gamma,
        }),
        terms,
    })
}
