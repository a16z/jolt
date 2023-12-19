use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;
use tracing::span::Attributes;
use tracing::span::Record;
use tracing::Event;
use tracing::Id;
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;

/// SumTimingLayer sums up spans of the same name and prints.
pub struct SumTimingLayer {
    span_durations: Arc<Mutex<HashMap<String, u128>>>,
}

pub struct FlushGuard {
    span_durations: Arc<Mutex<HashMap<String, u128>>>,
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
                let mut padding = if name.len() > 28 { 0 } else { 28 - name.len() };
                let left_padding = padding / 2;
                let right_padding = padding - left_padding;
                println!("||{:>padding_left$}{}{:>padding_right$}||", "", name, "", padding_left=left_padding, padding_right=right_padding);
            }
            println!("===============================");
        }
    }
}

impl Drop for FlushGuard {
    fn drop(&mut self) {
        self.print_header();
        let durations = self.span_durations.lock().unwrap();
        match &self.filter {
            Some(filter) => {
                for name in filter {
                    if let Some(duration) = durations.get(name) {
                        println!("'{}': {:?} ms", name, duration);
                    }
                }
            }
            None => {
                for (name, duration) in durations.iter() {
                    println!("'{}': {:?} ms", name, duration);
                }
            }
        }
    }

}

impl SumTimingLayer {
    /// Creates a new SumTimingLayer with an optional `filter: Option<Vec<String>>` of named spans.
    /// Also returns a FlushGuard. Prints sum details when the FlushGuard is dropped.
    pub fn new(filter: Option<Vec<String>>) -> (Self, FlushGuard) {
        let span_durations = Arc::new(Mutex::new(HashMap::new()));
        let layer = SumTimingLayer {
            span_durations: Arc::clone(&span_durations),
        };
        let guard = FlushGuard {
            span_durations: span_durations,
            filter,
        };
        (layer, guard)
    }
}

impl<S> Layer<S> for SumTimingLayer
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
            let span_name = span.name().to_string();
            if let Some(start_time) = span.extensions_mut().remove::<Instant>() {
                let duration = start_time.elapsed().as_millis();
                let mut durations = self.span_durations.lock().unwrap();
                *durations.entry(span_name).or_insert(0) += duration;
            }
        }
    }

    fn on_close(&self, _id: Id, _ctx: Context<'_, S>) {
        // Do nothing here
    }
}