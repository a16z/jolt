//! What the host still lets this process allocate, for callers that size a
//! memory budget from the machine instead of taking one: the smaller of
//! Linux's `MemAvailable` and, for each cgroup v2 on this process's path that
//! sets a `memory.max`, that limit less what the cgroup and its descendants
//! hold beyond evictable file cache. That is `memory.current` less the
//! `inactive_file` and `active_file` of `memory.stat`: the kernel evicts the
//! pages on those two lists before it refuses the cgroup an allocation, and
//! `MemAvailable` counts the host's as available too, down to a low
//! watermark. The rest of `memory.current` counts as used, tmpfs and shared
//! memory and the kernel's reclaimable caches included. Every reading is a
//! point-in-time observation that concurrent growth can outrun, and nothing
//! is held back for memory the caller does not price.
//!
//! The cgroup v2 hierarchy is read where it is conventionally mounted,
//! `/sys/fs/cgroup`. The process's cgroup path is relative to its cgroup
//! namespace, so under a mount made in another namespace it names another
//! cgroup's directory; the directory is read only once its `cgroup.threads`
//! lists this process. Discovery fails instead of skipping a limit when that
//! directory is missing, unreadable or another cgroup's, under a cgroup v1
//! memory controller, when a finite `memory.max` comes without a readable
//! `memory.current` or without those two counters, and on a host without
//! `/proc/meminfo`. A missing `/proc/self/cgroup` is read as no cgroup at
//! all, which is also what a `/proc` that does not show this process gives.
//! No other limit is read: not `memory.high`, swap, a resource limit such as
//! `RLIMIT_AS`, or a limit above a cgroup namespace's root, which the process
//! cannot see.

use std::fmt::{Display, Formatter, Result as FmtResult};
use std::fs;
use std::io::{Error as IoError, ErrorKind};
use std::path::{Component, Path, PathBuf};
use std::process;

use thiserror::Error;

/// The bytes this process can still allocate and the reading that sets them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Allocatable {
    pub bytes: u64,
    pub limit: MemoryLimit,
}

/// The reading that sets an [`Allocatable`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryLimit {
    /// `MemAvailable` of `/proc/meminfo`.
    MemAvailable,
    /// The `memory.max` of this cgroup v2 directory less the part of its
    /// `memory.current` that is not evictable file cache.
    Cgroup(PathBuf),
}

/// Why the host's limits could not be read.
#[derive(Debug, Error)]
pub enum MemoryDiscoveryError {
    #[error("memory discovery could not read {}: {source}", .path.display())]
    Read { path: PathBuf, source: IoError },
    #[error("memory discovery found unexpected contents in {}", .path.display())]
    Malformed { path: PathBuf },
    #[error(
        "memory discovery did not find this process in {}, so that directory is not its cgroup",
        .path.display()
    )]
    NotListed { path: PathBuf },
    #[error(
        "memory discovery does not read the cgroup v1 memory controller this process is under"
    )]
    CgroupV1Memory,
}

impl Display for Allocatable {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match &self.limit {
            MemoryLimit::MemAvailable => write!(f, "{} bytes of MemAvailable", self.bytes),
            MemoryLimit::Cgroup(dir) => write!(
                f,
                "{} bytes of memory.max less the memory.current that is not file cache in {}",
                self.bytes,
                dir.display()
            ),
        }
    }
}

impl Allocatable {
    /// Reads this host's limits now.
    pub fn read() -> Result<Self, MemoryDiscoveryError> {
        Self::read_under(Path::new("/"))
    }

    /// [`Self::read`] over the filesystem rooted at `root`.
    fn read_under(root: &Path) -> Result<Self, MemoryDiscoveryError> {
        let meminfo = root.join("proc/meminfo");
        let available = read(&meminfo)?
            .lines()
            .find_map(|line| {
                let kib = line.strip_prefix("MemAvailable:")?.trim();
                kib.strip_suffix("kB")?
                    .trim()
                    .parse::<u64>()
                    .ok()?
                    .checked_mul(1024)
            })
            .ok_or(MemoryDiscoveryError::Malformed { path: meminfo })?;
        let mut allocatable = Self {
            bytes: available,
            limit: MemoryLimit::MemAvailable,
        };
        let membership = root.join("proc/self/cgroup");
        let dirs = match fs::read_to_string(&membership) {
            Ok(text) => cgroup_dirs(&text, &membership, &root.join("sys/fs/cgroup"))?,
            Err(error) if error.kind() == ErrorKind::NotFound => Vec::new(),
            Err(source) => {
                return Err(MemoryDiscoveryError::Read {
                    path: membership,
                    source,
                })
            }
        };
        // The membership path is relative to this process's cgroup namespace,
        // the mount to the namespace that made it. Where the two differ, the
        // path names another cgroup's directory, whose limits are not this
        // process's. A cgroup v2 directory lists its threads by the ids the
        // reader's pid namespace gives them, and the main thread, whose
        // cgroup the membership file names, has the process's id. An exited
        // main thread is not listed; `cgroup.procs` lists its process, but
        // cannot be read in a threaded cgroup.
        if let Some(own) = dirs.last() {
            let threads = own.join("cgroup.threads");
            let pid = process::id().to_string();
            if !read(&threads)?.lines().any(|listed| listed == pid) {
                return Err(MemoryDiscoveryError::NotListed { path: threads });
            }
        }
        for dir in dirs {
            let limit = dir.join("memory.max");
            let max = match fs::read_to_string(&limit) {
                Ok(max) => max,
                // The hierarchy's root has no such file, and neither has a
                // cgroup whose parent does not enable the memory controller.
                Err(error) if error.kind() == ErrorKind::NotFound => continue,
                Err(source) => {
                    return Err(MemoryDiscoveryError::Read {
                        path: limit,
                        source,
                    })
                }
            };
            if max.trim() == "max" {
                continue;
            }
            let max = byte_count(&max, &limit)?;
            let usage = dir.join("memory.current");
            let mut used = byte_count(&read(&usage)?, &usage)?;
            let stat = dir.join("memory.stat");
            let counters = read(&stat)?;
            for list in ["inactive_file ", "active_file "] {
                let cache: u64 = counters
                    .lines()
                    .find_map(|line| line.strip_prefix(list)?.parse().ok())
                    .ok_or_else(|| MemoryDiscoveryError::Malformed { path: stat.clone() })?;
                used = used.saturating_sub(cache);
            }
            let left = max.saturating_sub(used);
            if left < allocatable.bytes {
                allocatable = Self {
                    bytes: left,
                    limit: MemoryLimit::Cgroup(dir),
                };
            }
        }
        Ok(allocatable)
    }
}

fn read(path: &Path) -> Result<String, MemoryDiscoveryError> {
    fs::read_to_string(path).map_err(|source| MemoryDiscoveryError::Read {
        path: path.to_path_buf(),
        source,
    })
}

/// The byte count that is the whole of `text`, read from `path`.
fn byte_count(text: &str, path: &Path) -> Result<u64, MemoryDiscoveryError> {
    text.trim()
        .parse()
        .map_err(|_| MemoryDiscoveryError::Malformed {
            path: path.to_path_buf(),
        })
}

/// The cgroup v2 directories on this process's path, from the hierarchy's
/// mount at `mount` down to the process's own cgroup, as `membership`'s
/// `text` names it; none when that names no cgroup v2 hierarchy. Each line is
/// `hierarchy:controllers:path`, and the v2 hierarchy is 0 and names no
/// controller.
fn cgroup_dirs(
    text: &str,
    membership: &Path,
    mount: &Path,
) -> Result<Vec<PathBuf>, MemoryDiscoveryError> {
    let malformed = || MemoryDiscoveryError::Malformed {
        path: membership.to_path_buf(),
    };
    let mut v2_path = None;
    for line in text.lines() {
        let mut fields = line.splitn(3, ':');
        let (Some(hierarchy), Some(controllers), Some(path)) =
            (fields.next(), fields.next(), fields.next())
        else {
            return Err(malformed());
        };
        if controllers.split(',').any(|name| name == "memory") {
            return Err(MemoryDiscoveryError::CgroupV1Memory);
        }
        if hierarchy == "0" && controllers.is_empty() {
            v2_path = Some(path);
        }
    }
    let Some(v2_path) = v2_path else {
        return Ok(Vec::new());
    };
    let mut dir = mount.to_path_buf();
    let mut dirs = vec![dir.clone()];
    for component in Path::new(v2_path).components() {
        match component {
            Component::RootDir => {}
            Component::Normal(name) => {
                dir.push(name);
                dirs.push(dir.clone());
            }
            // A path that leaves the namespace's root is not visible here.
            _ => return Err(malformed()),
        }
    }
    Ok(dirs)
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use tempfile::TempDir;

    use super::*;

    const GIB: u64 = 1 << 30;

    /// A filesystem root holding `files`, each a path under it and contents.
    fn root(files: &[(&str, String)]) -> TempDir {
        let root = tempfile::tempdir().unwrap();
        for (path, contents) in files {
            let path = root.path().join(path);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, contents).unwrap();
        }
        root
    }

    fn meminfo(available_kib: u64) -> (&'static str, String) {
        (
            "proc/meminfo",
            format!("MemTotal:       197552804 kB\nMemFree:  1024 kB\nMemAvailable:   {available_kib} kB\n"),
        )
    }

    fn membership(lines: &str) -> (&'static str, String) {
        ("proc/self/cgroup", lines.to_owned())
    }

    fn file(path: &'static str, contents: impl ToString) -> (&'static str, String) {
        (path, contents.to_string())
    }

    /// A `cgroup.threads` that lists this test's process among others.
    fn listed() -> String {
        format!("1\n{}\n", process::id())
    }

    /// A `memory.stat` with `inactive` and `active` bytes of file cache on
    /// the two reclaim lists, among counters that are not that cache.
    fn stat(inactive: u64, active: u64) -> String {
        format!(
            "anon 4096\nfile {}\nshmem 8192\ninactive_anon 12288\nactive_anon 0\n\
             inactive_file {inactive}\nactive_file {active}\nunevictable 0\n\
             slab_reclaimable 16384\n",
            inactive + active + 8192
        )
    }

    /// `MemAvailable`, in kibibytes, bounds a process with no cgroup
    /// membership file and one whose cgroups set no finite limit; the
    /// hierarchy's root has no `memory.max` at all.
    #[test]
    fn mem_available_bounds_a_process_without_cgroup_limits() {
        let expected = Allocatable {
            bytes: 8 * GIB,
            limit: MemoryLimit::MemAvailable,
        };
        let bare = root(&[meminfo(8 << 20)]);
        assert_eq!(Allocatable::read_under(bare.path()).unwrap(), expected);
        let unlimited = root(&[
            meminfo(8 << 20),
            membership("0::/job\n"),
            file("sys/fs/cgroup/job/cgroup.threads", listed()),
            file("sys/fs/cgroup/job/memory.max", "max\n"),
        ]);
        assert_eq!(Allocatable::read_under(unlimited.path()).unwrap(), expected);
    }

    /// The smallest headroom on the process's path sets the reading, whether
    /// the process's own cgroup or an ancestor holds it. File cache on either
    /// reclaim list is not usage, and no other counter of `memory.stat` is
    /// taken for it; a cgroup already over its limit leaves nothing; headroom
    /// above `MemAvailable` leaves `MemAvailable`. The cgroup of a process at
    /// its cgroup namespace's root is the mount itself.
    #[test]
    fn smallest_cgroup_headroom_on_the_path_sets_the_reading() {
        // Each level is its `memory.max`, its `memory.current`, and the bytes
        // of that usage on the inactive and on the active file list.
        let read = |outer: (u64, u64, u64, u64), leaf: (u64, u64, u64, u64)| {
            let root = root(&[
                meminfo(16 << 20),
                membership("0::/outer/unlimited/leaf\n"),
                file("sys/fs/cgroup/outer/memory.max", outer.0),
                file("sys/fs/cgroup/outer/memory.current", outer.1),
                file("sys/fs/cgroup/outer/memory.stat", stat(outer.2, outer.3)),
                file("sys/fs/cgroup/outer/unlimited/memory.max", "max"),
                file(
                    "sys/fs/cgroup/outer/unlimited/leaf/cgroup.threads",
                    listed(),
                ),
                file("sys/fs/cgroup/outer/unlimited/leaf/memory.max", leaf.0),
                file("sys/fs/cgroup/outer/unlimited/leaf/memory.current", leaf.1),
                file(
                    "sys/fs/cgroup/outer/unlimited/leaf/memory.stat",
                    stat(leaf.2, leaf.3),
                ),
            ]);
            let read = Allocatable::read_under(root.path()).unwrap();
            let limit = match read.limit {
                MemoryLimit::MemAvailable => None,
                MemoryLimit::Cgroup(dir) => Some(dir.strip_prefix(root.path()).unwrap().to_owned()),
            };
            (read.bytes, limit)
        };
        let outer = Some(PathBuf::from("sys/fs/cgroup/outer"));
        let leaf = Some(PathBuf::from("sys/fs/cgroup/outer/unlimited/leaf"));
        assert_eq!(
            read(
                (8 * GIB, 6 * GIB, GIB, 3 * GIB),
                (6 * GIB, 5 * GIB, GIB, 2 * GIB)
            ),
            (4 * GIB, leaf.clone())
        );
        assert_eq!(
            read(
                (3 * GIB, 3 * GIB, GIB / 2, 3 * GIB / 2),
                (6 * GIB, 2 * GIB, GIB / 2, GIB / 2)
            ),
            (2 * GIB, outer)
        );
        assert_eq!(
            read((8 * GIB, 2 * GIB, 0, 0), (GIB, 2 * GIB, 0, 0)),
            (0, leaf)
        );
        assert_eq!(
            read((64 * GIB, GIB, 0, 0), (32 * GIB, GIB, 0, 0)),
            (16 * GIB, None)
        );

        let namespaced = root(&[
            meminfo(16 << 20),
            membership("0::/\n"),
            file("sys/fs/cgroup/cgroup.threads", listed()),
            file("sys/fs/cgroup/memory.max", 3 * GIB),
            file("sys/fs/cgroup/memory.current", GIB),
            file("sys/fs/cgroup/memory.stat", stat(0, 0)),
        ]);
        assert_eq!(
            Allocatable::read_under(namespaced.path()).unwrap(),
            Allocatable {
                bytes: 2 * GIB,
                limit: MemoryLimit::Cgroup(namespaced.path().join("sys/fs/cgroup")),
            }
        );
    }

    /// A limit that may bind and cannot be read fails the reading: a cgroup
    /// v1 memory controller, a process cgroup that is not under the mount,
    /// whose directory there is another cgroup's or whose path leaves the
    /// namespace's root, and a limit file that cannot be read.
    #[test]
    fn limits_that_cannot_be_read_fail_the_reading() {
        let v1 = root(&[
            meminfo(1 << 20),
            membership("12:cpu,memory:/job\n0::/job\n"),
        ]);
        assert!(matches!(
            Allocatable::read_under(v1.path()),
            Err(MemoryDiscoveryError::CgroupV1Memory)
        ));
        let invisible = root(&[
            meminfo(1 << 20),
            membership("0::/elsewhere\n"),
            file("sys/fs/cgroup/job/memory.max", GIB),
        ]);
        assert!(matches!(
            Allocatable::read_under(invisible.path()),
            Err(MemoryDiscoveryError::Read { path, source })
                if path.ends_with("sys/fs/cgroup/elsewhere/cgroup.threads")
                    && source.kind() == ErrorKind::NotFound
        ));
        // The host hierarchy's root, which `0::/` names under a mount made
        // outside the process's cgroup namespace: it sets no limit and lists
        // other threads, here under ids that only contain this process's.
        let pid = process::id();
        let foreign = root(&[
            meminfo(1 << 20),
            membership("0::/\n"),
            file(
                "sys/fs/cgroup/cgroup.threads",
                format!("0\n1{pid}\n{pid}1\n"),
            ),
        ]);
        assert!(matches!(
            Allocatable::read_under(foreign.path()),
            Err(MemoryDiscoveryError::NotListed { path })
                if path.ends_with("sys/fs/cgroup/cgroup.threads")
        ));
        let outside = root(&[
            meminfo(1 << 20),
            membership("0::/../job\n"),
            file("sys/fs/cgroup/job/memory.max", GIB),
        ]);
        assert!(matches!(
            Allocatable::read_under(outside.path()),
            Err(MemoryDiscoveryError::Malformed { path }) if path.ends_with("proc/self/cgroup")
        ));
        // A directory in the file's place stands in for any read that fails
        // other than by the file's absence.
        let unreadable = root(&[
            meminfo(1 << 20),
            membership("0::/job\n"),
            file("sys/fs/cgroup/job/cgroup.threads", listed()),
            file("sys/fs/cgroup/job/memory.max/placeholder", ""),
        ]);
        assert!(matches!(
            Allocatable::read_under(unreadable.path()),
            Err(MemoryDiscoveryError::Read { path, source })
                if path.ends_with("job/memory.max") && source.kind() != ErrorKind::NotFound
        ));
    }

    /// Contents discovery does not understand are errors, never zero or an
    /// absent limit: no `MemAvailable`, one past `u64` bytes, a limit that is
    /// not a count, and a limit without its usage, without its `memory.stat`
    /// or without one of the two file cache counters.
    #[test]
    fn unexpected_contents_fail_the_reading() {
        let malformed = |files: &[(&str, String)], at: &str| {
            assert!(matches!(
                Allocatable::read_under(root(files).path()),
                Err(MemoryDiscoveryError::Malformed { path }) if path.ends_with(at)
            ));
        };
        malformed(
            &[file("proc/meminfo", "MemTotal: 1024 kB\n")],
            "proc/meminfo",
        );
        malformed(&[meminfo(u64::MAX / 1024 + 1)], "proc/meminfo");
        malformed(
            &[
                meminfo(1 << 20),
                membership("0::/job\n"),
                file("sys/fs/cgroup/job/cgroup.threads", listed()),
                file("sys/fs/cgroup/job/memory.max", "unlimited"),
            ],
            "job/memory.max",
        );
        malformed(
            &[meminfo(1 << 20), membership("0:/job\n")],
            "proc/self/cgroup",
        );
        malformed(
            &[
                meminfo(1 << 20),
                membership("0::/job\n"),
                file("sys/fs/cgroup/job/cgroup.threads", listed()),
                file("sys/fs/cgroup/job/memory.max", GIB),
                file("sys/fs/cgroup/job/memory.current", 0),
                file("sys/fs/cgroup/job/memory.stat", "file 0\ninactive_file 0\n"),
            ],
            "job/memory.stat",
        );
        let unread = |files: &[(&str, String)], at: &str| {
            assert!(matches!(
                Allocatable::read_under(root(files).path()),
                Err(MemoryDiscoveryError::Read { path, .. }) if path.ends_with(at)
            ));
        };
        unread(
            &[
                meminfo(1 << 20),
                membership("0::/job\n"),
                file("sys/fs/cgroup/job/cgroup.threads", listed()),
                file("sys/fs/cgroup/job/memory.max", GIB),
            ],
            "job/memory.current",
        );
        unread(
            &[
                meminfo(1 << 20),
                membership("0::/job\n"),
                file("sys/fs/cgroup/job/cgroup.threads", listed()),
                file("sys/fs/cgroup/job/memory.max", GIB),
                file("sys/fs/cgroup/job/memory.current", 0),
            ],
            "job/memory.stat",
        );
    }
}
