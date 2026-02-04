use serde::Serialize;
use std::sync::Mutex;
use tracing::span::{Attributes, Id};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;
use wasm_bindgen::JsCast;

static TRACE_EVENTS: Mutex<Vec<TraceEvent>> = Mutex::new(Vec::new());
static START_TIME: Mutex<Option<f64>> = Mutex::new(None);

#[derive(Serialize, Clone)]
struct TraceEvent {
    name: String,
    cat: String,
    ph: String,
    ts: f64,
    pid: u32,
    tid: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    dur: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    args: Option<serde_json::Value>,
}

fn now_micros() -> f64 {
    // Use js_sys to get performance.now() - works in both window and worker contexts
    let global = js_sys::global();
    let performance = js_sys::Reflect::get(&global, &"performance".into())
        .ok()
        .and_then(|p| p.dyn_into::<web_sys::Performance>().ok());

    match performance {
        Some(perf) => perf.now() * 1000.0,
        None => 0.0, // Fallback if performance API unavailable
    }
}

fn get_start_time() -> f64 {
    let mut start = START_TIME.lock().unwrap();
    if start.is_none() {
        *start = Some(now_micros());
    }
    start.unwrap()
}

struct ChromeTraceLayer;

impl<S> Layer<S> for ChromeTraceLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let start = get_start_time();
        let ts = now_micros() - start;

        if let Some(span) = ctx.span(id) {
            let mut extensions = span.extensions_mut();
            extensions.insert(SpanTiming { start_ts: ts });
        }

        let mut args = serde_json::Map::new();
        let mut visitor = JsonVisitor(&mut args);
        attrs.record(&mut visitor);

        let event = TraceEvent {
            name: attrs.metadata().name().to_string(),
            cat: attrs.metadata().target().to_string(),
            ph: "B".to_string(),
            ts,
            pid: 1,
            tid: 1,
            dur: None,
            args: if args.is_empty() {
                None
            } else {
                Some(serde_json::Value::Object(args))
            },
        };

        TRACE_EVENTS.lock().unwrap().push(event);
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let start = get_start_time();
        let ts = now_micros() - start;

        if let Some(span) = ctx.span(&id) {
            let name = span.name().to_string();
            let cat = span.metadata().target().to_string();

            let event = TraceEvent {
                name,
                cat,
                ph: "E".to_string(),
                ts,
                pid: 1,
                tid: 1,
                dur: None,
                args: None,
            };

            TRACE_EVENTS.lock().unwrap().push(event);
        }
    }

    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let start = get_start_time();
        let ts = now_micros() - start;

        let mut args = serde_json::Map::new();
        let mut visitor = JsonVisitor(&mut args);
        event.record(&mut visitor);

        let trace_event = TraceEvent {
            name: event.metadata().name().to_string(),
            cat: event.metadata().target().to_string(),
            ph: "i".to_string(),
            ts,
            pid: 1,
            tid: 1,
            dur: None,
            args: if args.is_empty() {
                None
            } else {
                Some(serde_json::Value::Object(args))
            },
        };

        TRACE_EVENTS.lock().unwrap().push(trace_event);
    }
}

#[allow(dead_code)]
struct SpanTiming {
    start_ts: f64,
}

struct JsonVisitor<'a>(&'a mut serde_json::Map<String, serde_json::Value>);

impl<'a> tracing::field::Visit for JsonVisitor<'a> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0.insert(
            field.name().to_string(),
            serde_json::Value::String(format!("{:?}", value)),
        );
    }

    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        self.0.insert(
            field.name().to_string(),
            serde_json::Value::Number(value.into()),
        );
    }

    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.0.insert(
            field.name().to_string(),
            serde_json::Value::Number(value.into()),
        );
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.0.insert(
            field.name().to_string(),
            serde_json::Value::String(value.to_string()),
        );
    }
}

pub fn init() {
    use tracing_subscriber::util::SubscriberInitExt;

    let subscriber = tracing_subscriber::registry().with(ChromeTraceLayer);
    let _ = subscriber.try_init();
}

pub fn get_trace_json() -> String {
    let events = TRACE_EVENTS.lock().unwrap();
    serde_json::to_string(&*events).unwrap_or_else(|_| "[]".to_string())
}

pub fn clear() {
    TRACE_EVENTS.lock().unwrap().clear();
    *START_TIME.lock().unwrap() = None;
}
