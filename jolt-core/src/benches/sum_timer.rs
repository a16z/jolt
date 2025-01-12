use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

use tracing::Id;
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;

pub struct CumulativeTimingLayer {
    span_durations: Mutex<HashMap<String, u128>>,
    filter: Option<Vec<String>>,
}

pub struct FlushGuard {
    span_durations: Mutex<HashMap<String, u128>>,
    filter: Option<Vec<String>>,
}

impl FlushGuard {
    fn print_header(&self) {
        println!("===============================");
        println!("||                            ||");
        println!("||     Sum Timing Results     ||");
        println!("||                            ||");
        println!("===============================");

        if let Some(filter) = &self.filter {
            println!("===============================");
            println!("||     Filter Details         ||");
            println!("===============================");
            for name in filter {
                let padding = 28 - name.len().min(28);
                let left_padding = padding / 2;
                let right_padding = padding - left_padding;
                println!(
                    "||{:>left_padding$}{}{:>right_padding$}||",
                    "", name, ""
                );
            }
            println!("===============================");
        }
    }

    fn print_results(&self, durations: &HashMap<String, u128>) {
        let mut entries: Vec<(&String, &u128)> = durations.iter().collect();
        entries.sort_by_key(|&(_, duration)| duration);

        if let Some(filter) = &self.filter {
            entries.retain(|(name, _)| filter.contains(name));
        }

        for (name, duration) in entries {
            println!("'{}': {:?} ms", name, duration / 1_000_000);
        }
    }
}

impl Drop for FlushGuard {
    fn drop(&mut self) {
        self.print_header();
        if let Ok(durations) = self.span_durations.lock() {
            self.print_results(&durations);
        } else {
            eprintln!("Failed to lock span_durations");
        }
    }
}

impl CumulativeTimingLayer {
    pub fn new(filter: Option<Vec<String>>) -> Self {
        Self {
            span_durations: Mutex::new(HashMap::new()),
            filter,
        }
    }

    pub fn into_flush_guard(self) -> FlushGuard {
        FlushGuard {
            span_durations: self.span_durations,
            filter: self.filter,
        }
    }
}

impl<S> Layer<S> for CumulativeTimingLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_enter(&self, id: &tracing::span::Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(Instant::now());
        }
    }

    fn on_exit(&self, id: &tracing::span::Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            let span_name = span.name();
            if let Some(start_time) = span.extensions_mut().remove::<Instant>() {
                let duration = start_time.elapsed().as_nanos();
                let mut durations = self.span_durations.lock().unwrap();
                durations.entry(span_name.to_string()).and_modify(|v| *v += duration).or_insert(duration);
            }
        }
    }

    fn on_close(&self, _id: Id, _ctx: Context<'_, S>) {
        // Do nothing here
    }
}
