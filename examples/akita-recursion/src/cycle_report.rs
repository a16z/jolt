use std::collections::BTreeMap;
use std::fmt::Debug;
use std::sync::{Arc, Mutex, PoisonError};

use serde::Serialize;
use tracing::{
    field::{Field, Visit},
    Event, Subscriber,
};
use tracing_subscriber::{
    layer::{Context, Layer},
    prelude::*,
};

#[derive(Clone, Default)]
pub struct Capture(Arc<Mutex<BTreeMap<String, MarkerCycles>>>);

#[derive(Default, Serialize)]
pub struct MarkerCycles {
    real_cycles: u64,
    virtual_instructions: u64,
    total_cycles: u64,
    calls: usize,
}

#[derive(Default)]
struct MarkerEvent {
    label: Option<String>,
    real: Option<u64>,
    virtuals: Option<u64>,
    total: Option<u64>,
}

impl Visit for MarkerEvent {
    fn record_debug(&mut self, _: &Field, _: &dyn Debug) {}

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "marker" {
            self.label = Some(value.to_owned());
        }
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        match field.name() {
            "real_cycles" => self.real = Some(value),
            "virtual_instructions" => self.virtuals = Some(value),
            "total_cycles" => self.total = Some(value),
            _ => {}
        }
    }
}

impl<S: Subscriber> Layer<S> for Capture {
    fn on_event(&self, event: &Event<'_>, _: Context<'_, S>) {
        let mut marker = MarkerEvent::default();
        event.record(&mut marker);
        if let (Some(label), Some(real), Some(virtuals), Some(total)) =
            (marker.label, marker.real, marker.virtuals, marker.total)
        {
            let mut map = self.0.lock().unwrap_or_else(PoisonError::into_inner);
            let entry = map.entry(label).or_default();
            entry.real_cycles += real;
            entry.virtual_instructions += virtuals;
            entry.total_cycles += total;
            entry.calls += 1;
        }
    }
}

impl Capture {
    pub fn subscriber(&self) -> impl Subscriber + Send + Sync {
        tracing_subscriber::registry()
            .with(self.clone())
            .with(tracing_subscriber::fmt::layer().with_ansi(false))
    }

    pub fn finish(self, total: usize, padded: usize) -> Result<CycleReport, String> {
        let markers = std::mem::take(&mut *self.0.lock().unwrap_or_else(PoisonError::into_inner));
        let count = |name: &str| -> Result<u64, String> {
            let marker = markers
                .get(name)
                .ok_or_else(|| format!("missing cycle marker {name}"))?;
            if marker.calls != 1 {
                return Err(format!("cycle marker {name} ran {} times", marker.calls));
            }
            Ok(marker.total_cycles)
        };
        let mut exclusive = BTreeMap::new();
        let decode = count("decode_pair")?;
        let prepare = count("prepare_akita_setup")?;
        let install = count("install_prepared_setup")?;
        exclusive.insert("decode_pair".to_owned(), decode);
        exclusive.insert("prepare_akita_setup".to_owned(), prepare);
        exclusive.insert("install_prepared_setup".to_owned(), install);
        let mut inner_sum = 0;
        for index in 0..2 {
            let jolt = count(&format!("inner{index}_jolt"))?;
            let akita = count(&format!("inner{index}_akita"))?;
            exclusive.insert(format!("inner{index}_akita"), akita);
            exclusive.insert(
                format!("inner{index}_jolt_other"),
                jolt.checked_sub(akita)
                    .ok_or("Akita marker exceeds inner verifier")?,
            );
            inner_sum += jolt;
        }
        let verify = count("verify_pair")?;
        exclusive.insert(
            "batch_overhead".to_owned(),
            verify
                .checked_sub(prepare + install + inner_sum)
                .ok_or("inner markers exceed verification")?,
        );
        let export = count("export_setup_claims")?;
        let drop_pair = count("drop_pair")?;
        exclusive.insert("export_setup_claims".to_owned(), export);
        exclusive.insert("drop_pair".to_owned(), drop_pair);
        exclusive.insert(
            "runtime_and_io".to_owned(),
            (total as u64)
                .checked_sub(decode + verify + export + drop_pair)
                .ok_or("markers exceed total execution")?,
        );
        Ok(CycleReport {
            total_cycles: total,
            padded_trace_length: padded,
            markers,
            exclusive_total_cycles: exclusive,
        })
    }
}

#[derive(Serialize)]
pub struct CycleReport {
    pub total_cycles: usize,
    pub padded_trace_length: usize,
    pub markers: BTreeMap<String, MarkerCycles>,
    pub exclusive_total_cycles: BTreeMap<String, u64>,
}
