//! Per-PC cycle profile of a traced guest, enabled by `JOLT_PC_PROFILE=<path>`.
//!
//! Every executed instruction adds its trace rows (RV64IMAC cycle plus the
//! virtual instructions it expands to) to its PC. With
//! `JOLT_PC_PROFILE_RANGES=<file>` (lines `start end`, hex, from
//! `scripts/guest_pc_profile.py ranges`), rows spent inside those address
//! ranges are also attributed to the return address captured on entry, so a
//! leaf such as `memcpy` is charged to its callers, with the call count.
//! Both tables are written when the trace ends (`<path>` and `<path>.ra`),
//! for `scripts/guest_pc_profile.py report` / `callers`.
//!
//! Off by default: one relaxed load per instruction.

use std::cell::RefCell;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(false);

struct Profile {
    path: PathBuf,
    by_pc: HashMap<u64, u64>,
    ranges: Vec<(u64, u64)>,
    current_leaf: Option<(u64, u64)>,
    /// Rows and entries (calls) per (leaf start, caller return address).
    by_leaf_caller: HashMap<(u64, u64), (u64, u64)>,
}

thread_local! {
    static PROFILE: RefCell<Option<Profile>> = const { RefCell::new(None) };
}

fn load_ranges() -> Vec<(u64, u64)> {
    let Some(path) = env::var_os("JOLT_PC_PROFILE_RANGES") else {
        return Vec::new();
    };
    fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter_map(|line| {
            let mut words = line.split_whitespace();
            let start = u64::from_str_radix(words.next()?, 16).ok()?;
            let end = u64::from_str_radix(words.next()?, 16).ok()?;
            Some((start, end))
        })
        .collect()
}

/// Arms the profile for the calling (emulator) thread when the environment
/// asks for one.
pub fn init() {
    let Some(path) = env::var_os("JOLT_PC_PROFILE") else {
        return;
    };
    PROFILE.with(|profile| {
        *profile.borrow_mut() = Some(Profile {
            path: PathBuf::from(path),
            by_pc: HashMap::new(),
            ranges: load_ranges(),
            current_leaf: None,
            by_leaf_caller: HashMap::new(),
        });
    });
    ENABLED.store(true, Ordering::Relaxed);
}

/// Charges `rows` trace rows to the instruction at `pc`; `ra` is the return
/// address register at that instruction.
#[inline]
pub fn record(pc: u64, rows: u64, ra: u64) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    PROFILE.with(|profile| {
        let mut profile = profile.borrow_mut();
        let Some(profile) = profile.as_mut() else {
            return;
        };
        *profile.by_pc.entry(pc).or_insert(0) += rows;
        if profile.ranges.is_empty() {
            return;
        }
        let leaf = profile
            .ranges
            .iter()
            .find(|(start, end)| pc >= *start && pc < *end)
            .map(|(start, _)| *start);
        match leaf {
            Some(start) => {
                let entered = pc == start;
                if entered {
                    profile.current_leaf = Some((start, ra));
                }
                let key = profile.current_leaf.unwrap_or((start, 0));
                let entry = profile.by_leaf_caller.entry(key).or_insert((0, 0));
                entry.0 += rows;
                entry.1 += u64::from(entered);
            }
            None => profile.current_leaf = None,
        }
    });
}

/// Writes the tables and disarms the profile.
pub fn finish() {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    PROFILE.with(|profile| {
        let Some(profile) = profile.borrow_mut().take() else {
            return;
        };
        let mut by_pc = String::with_capacity(profile.by_pc.len() * 24);
        for (pc, rows) in &profile.by_pc {
            by_pc.push_str(&format!("{pc:x} {rows}\n"));
        }
        if let Err(error) = fs::write(&profile.path, by_pc) {
            tracing::error!("failed to write the PC profile: {error}");
        }
        let mut by_caller = String::new();
        for ((leaf, caller), (rows, entries)) in &profile.by_leaf_caller {
            by_caller.push_str(&format!("{leaf:x} {caller:x} {rows} {entries}\n"));
        }
        let mut caller_path = profile.path.clone();
        caller_path.as_mut_os_string().push(".ra");
        if let Err(error) = fs::write(&caller_path, by_caller) {
            tracing::error!("failed to write the PC profile callers: {error}");
        }
    });
    ENABLED.store(false, Ordering::Relaxed);
}
