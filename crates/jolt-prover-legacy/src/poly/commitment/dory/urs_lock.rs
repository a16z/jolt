//! Cross-process lock serializing dory-pcs's URS disk-cache critical section.
//!
//! Twin of `jolt-dory/src/urs_lock.rs`. The two crates cannot share this
//! helper: this module also compiles in `minimal` builds, where `jolt-dory`
//! is not a dependency.

use std::fs::{File, OpenOptions};
use std::path::PathBuf;

/// Resolves dory-pcs's cache directory from the same environment variables its
/// `get_storage_path` reads (`LOCALAPPDATA` first, then `HOME` with
/// macOS-vs-XDG detection), so the lock file always lands in the directory the
/// `dory_N.urs` files land in.
fn urs_cache_dir() -> Option<PathBuf> {
    let mut dir = if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
        PathBuf::from(local_app_data)
    } else {
        let home = PathBuf::from(std::env::var("HOME").ok()?);
        let macos_caches = home.join("Library").join("Caches");
        if macos_caches.exists() {
            macos_caches
        } else {
            home.join(".cache")
        }
    };
    dir.push("dory");
    Some(dir)
}

/// Takes an exclusive advisory lock on `<cache_dir>/dory.lock`, serializing
/// dory-pcs's load-or-generate-or-save URS critical section across processes.
/// The lock releases when the returned handle drops (or the process dies).
///
/// Best-effort: if the cache directory cannot be resolved or the lock file
/// cannot be created (e.g. a read-only, pre-populated cache), setup proceeds
/// unlocked — the same environments where dory itself cannot persist a URS.
pub(crate) fn lock_urs_cache() -> Option<File> {
    let dir = urs_cache_dir()?;
    std::fs::create_dir_all(&dir).ok()?;
    let lock_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(dir.join("dory.lock"))
        .ok()?;
    lock_file.lock().ok()?;
    Some(lock_file)
}
