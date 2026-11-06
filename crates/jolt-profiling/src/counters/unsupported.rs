use std::io::{Error, ErrorKind, Result};

use super::{CounterSource, CounterValues};

pub(super) enum Counter {}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Reading {}

impl Reading {
    pub(crate) fn since(self, _earlier: Self) -> Option<CounterValues> {
        match self {}
    }
}

impl Counter {
    pub(super) fn open() -> Result<Self> {
        Err(Error::new(
            ErrorKind::Unsupported,
            "hardware counters need Linux perf_event_open or macOS proc_pid_rusage",
        ))
    }

    pub(super) fn read(&self) -> Result<Reading> {
        match *self {}
    }

    pub(super) fn source(&self) -> CounterSource {
        match *self {}
    }
}
