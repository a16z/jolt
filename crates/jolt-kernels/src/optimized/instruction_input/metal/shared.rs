use super::super::super::native_operands::SharedCore;
use super::*;

pub(super) struct SharedInput {
    pub(super) gpu: Arc<Gpu>,
    pub(super) source: Mutex<Option<SharedCore>>,
}
struct NativeMessage {
    terms: [F; 3],
    columns: Option<ResidentInput>,
}
impl SharedInput {
    fn run(
        &self,
        source: &SharedCore,
        prefix: usize,
        challenge: Option<F>,
        tails: [F; NUM_TABLES],
        eq: &GruenSplitEqPolynomial<F>,
        gamma: F,
    ) -> Result<NativeMessage, SumcheckError<F>> {
        let core = source
            .lock()
            .map_err(|_| compute_error(invalid("native input lock poisoned")))?;
        if prefix > core.prefix {
            return Err(compute_error(invalid("native Input prefix exceeds source")));
        }
        let device = &self.gpu.device;
        let equality = Equality::new(device, eq).map_err(compute_error)?;
        let output_len = if challenge.is_some() {
            prefix.div_ceil(2)
        } else {
            prefix
        };
        let flag_table = FlagTable::new(
            core::array::from_fn(|i| tails[2 * i] == F::one()),
            challenge.unwrap_or_else(F::zero),
        );
        let lut = DeviceBuffer::from_slice(device, &flag_table.values).map_err(compute_error)?;
        let constants = Constants::new(
            tails,
            words(flag_table.tail),
            challenge.unwrap_or_else(F::zero),
            gamma,
        );
        let unused_words = DeviceBuffer::zeroed(device, 1).map_err(compute_error)?;
        let unused_lut = DeviceBuffer::zeroed(device, 1).map_err(compute_error)?;
        let dummy_output = DeviceBuffer::<F>::zeroed(device, 1).map_err(compute_error)?;
        let pending = challenge
            .map(|_| Pending::new(&self.gpu, output_len, true))
            .transpose()
            .map_err(compute_error)?;
        let pairs = output_len.div_ceil(2);
        let groups = equality.groups(0, pairs).map_err(compute_error)?;
        let partials = partials(groups, device).map_err(compute_error)?;
        let params = Params {
            prefix: word(prefix).map_err(compute_error)?,
            output_len: word(output_len).map_err(compute_error)?,
            first_pair: 0,
            pairs: word(pairs).map_err(compute_error)?,
            in_len: word(equality.inner_len).map_err(compute_error)?,
            old_bits: u32::from(challenge.is_some()),
            new_bits: 2,
            need_one: equality.need_one,
        };
        let span = tracing::info_span!(
            "metal_input_resident_native",
            bind = challenge.is_some(),
            prefix,
            gpu_s = tracing::field::Empty
        );
        let _span = span.enter();
        let mut batch = Batch::new(device).map_err(compute_error)?;
        // SAFETY: the shared source contains every explicit row plus an odd-tail
        // representative. First-message mode never writes the dummy outputs;
        // bind mode writes all operand columns and packed words exactly once.
        unsafe {
            batch
                .dispatch_unchecked(
                    self.gpu
                        .library
                        .pipeline(&host_name::<F>(RESIDENT))
                        .map_err(compute_error)?,
                    &[
                        Binding::buffer(&core.buffer),
                        Binding::buffer(&lut),
                        Binding::buffer(&equality.inner),
                        Binding::buffer(&equality.outer),
                        pending.as_ref().map_or_else(
                            || Binding::buffer(&dummy_output),
                            |pending| Binding::buffer(&pending.values),
                        ),
                        pending.as_ref().map_or_else(
                            || Binding::buffer(&unused_words),
                            |pending| pending.words_binding(&unused_words),
                        ),
                        Binding::buffer(&partials),
                        Binding::value(&constants),
                        Binding::value(&params),
                    ],
                    Grid::linear(groups * THREADS, THREADS),
                )
                .map_err(compute_error)?;
        }
        let elapsed = batch.commit_and_wait().map_err(compute_error)?;
        let _ = span.record("gpu_s", elapsed.as_secs_f64());
        // SAFETY: the completed native dispatch initialized every scalar partial.
        let terms = unsafe { sum_partials(partials) }.map_err(compute_error)?;
        let columns = if let Some(pending) = pending {
            Some(ResidentInput {
                gpu: Arc::clone(&self.gpu),
                // SAFETY: bind mode wrote every output row, column and packed word.
                columns: unsafe { pending.complete() }.map_err(compute_error)?,
                unused_words,
                unused_lut,
                flags: Some(flag_table),
                prefix: output_len,
                tails,
                gamma,
            })
        } else {
            None
        };
        Ok(NativeMessage { terms, columns })
    }
}
impl InputCompute<F> for SharedInput {
    fn admits(&self, prefix: usize, log_t: usize) -> bool {
        MetalInstructionInput(Arc::clone(&self.gpu)).admits(prefix, log_t)
    }
    fn initial_message(
        &self,
        prefix: usize,
        tail: [F; NUM_TABLES],
        eq: &GruenSplitEqPolynomial<F>,
        gamma: F,
    ) -> Result<Option<[F; 3]>, SumcheckError<F>> {
        let source = self
            .source
            .lock()
            .map_err(|_| compute_error(invalid("native input carry poisoned")))?;
        let source = source
            .as_ref()
            .ok_or_else(|| compute_error(invalid("native input source already consumed")))?;
        Ok(Some(self.run(source, prefix, None, tail, eq, gamma)?.terms))
    }
    fn materialize(
        &self,
        _rows: &BundleStore<InstructionInputRow>,
        prefix: usize,
        challenge: F,
        tail: [F; NUM_TABLES],
        eq: &GruenSplitEqPolynomial<F>,
        gamma: F,
    ) -> Result<InputPrepared<F>, SumcheckError<F>> {
        let source = self
            .source
            .lock()
            .map_err(|_| compute_error(invalid("native input carry poisoned")))?
            .take()
            .ok_or_else(|| compute_error(invalid("native input source already consumed")))?;
        let prepared = self.run(&source, prefix, Some(challenge), tail, eq, gamma)?;
        let device = prepared
            .columns
            .ok_or_else(|| compute_error(invalid("native Input bind omitted output columns")))?;
        Ok(InputPrepared {
            device: Box::new(device),
            terms: prepared.terms,
        })
    }
}
