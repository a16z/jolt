//! Cross-process lock serializing dory-pcs's URS disk-cache critical section.
//!
//! Every Dory caller shares this lock through `jolt-dory`.

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
/// Best-effort by design, but every fallback is logged: when the cache
/// directory cannot be resolved or created, dory's own persistence fails the
/// same way (its `save_setup` panics / its load skips), so no unlocked cache
/// write can race. The one genuinely racy fallback is an advisory-lock
/// failure on a writable cache (e.g. a filesystem without flock support):
/// dory will then persist unlocked and concurrent processes can overwrite
/// each other's randomized URS — a correctness/availability hazard, not a
/// soundness one (mismatched generators make verification fail). Failing
/// closed here would break read-only, pre-populated cache deployments, so we
/// warn instead.
pub(crate) fn lock_urs_cache() -> Option<File> {
    let Some(dir) = urs_cache_dir() else {
        tracing::warn!(
            "dory URS cache lock skipped: no LOCALAPPDATA/HOME to resolve the cache directory"
        );
        return None;
    };
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::warn!(
            "dory URS cache lock skipped: cannot create {}: {e}",
            dir.display()
        );
        return None;
    }
    let lock_path = dir.join("dory.lock");
    let lock_file = match OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)
    {
        Ok(file) => file,
        Err(e) => {
            tracing::warn!(
                "dory URS cache lock skipped: cannot open {}: {e}",
                lock_path.display()
            );
            return None;
        }
    };
    if let Err(e) = lock_file.lock() {
        tracing::warn!(
            "dory URS cache advisory lock failed on {}: {e}; concurrent setup may race the URS cache",
            lock_path.display()
        );
        return None;
    }
    Some(lock_file)
}
