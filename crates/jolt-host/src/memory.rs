//! What the host still lets this process allocate, for callers that size a
//! memory budget from the machine instead of taking one: the smaller of
//! Linux's `MemAvailable` and, for each cgroup v2 on this process's path that
//! sets a `memory.max`, that limit less the cgroup's `memory.current`, which
//! counts its descendants and the file cache charged to it. Every reading is
//! a point-in-time observation that concurrent growth can outrun, and nothing
//! is held back for memory the caller does not price.
//!
//! The cgroup v2 hierarchy is read where it is conventionally mounted,
//! `/sys/fs/cgroup`. A cgroup limit that may bind and cannot be read fails
//! the reading instead of being skipped: a process cgroup that is missing or
//! unreadable under the mount and a cgroup v1 memory controller are errors,
//! as is a host without `/proc/meminfo`. No other limit is read: not
//! `memory.high`, swap, a resource limit such as `RLIMIT_AS`, or a limit
//! above a cgroup namespace's root, which the process cannot see.

use std::fmt::{Display, Formatter, Result as FmtResult};
use std::fs;
use std::io::{Error as IoError, ErrorKind};
use std::path::{Component, Path, PathBuf};

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
    /// The `memory.max` of this cgroup v2 directory less its
    /// `memory.current`.
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
                "{} bytes of memory.max less memory.current in {}",
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
        // Every cgroup v2 directory holds this file. Without the read, a
        // cgroup that is not under the mount would pass for one that sets no
        // limit.
        if let Some(own) = dirs.last() {
            let _ = read(&own.join("cgroup.controllers"))?;
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
            let usage = dir.join("memory.current");
            let left =
                byte_count(&max, &limit)?.saturating_sub(byte_count(&read(&usage)?, &usage)?);
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
            file("sys/fs/cgroup/job/cgroup.controllers", "memory"),
            file("sys/fs/cgroup/job/memory.max", "max\n"),
        ]);
        assert_eq!(Allocatable::read_under(unlimited.path()).unwrap(), expected);
    }

    /// The smallest headroom on the process's path sets the reading, whether
    /// the process's own cgroup or an ancestor holds it; a cgroup already
    /// over its limit leaves nothing; headroom above `MemAvailable` leaves
    /// `MemAvailable`. The cgroup of a process at its cgroup namespace's root
    /// is the mount itself.
    #[test]
    fn smallest_cgroup_headroom_on_the_path_sets_the_reading() {
        let read = |outer: (u64, u64), leaf: (u64, u64)| {
            let root = root(&[
                meminfo(16 << 20),
                membership("0::/outer/unlimited/leaf\n"),
                file("sys/fs/cgroup/outer/memory.max", outer.0),
                file("sys/fs/cgroup/outer/memory.current", outer.1),
                file("sys/fs/cgroup/outer/unlimited/memory.max", "max"),
                file(
                    "sys/fs/cgroup/outer/unlimited/leaf/cgroup.controllers",
                    "memory",
                ),
                file("sys/fs/cgroup/outer/unlimited/leaf/memory.max", leaf.0),
                file("sys/fs/cgroup/outer/unlimited/leaf/memory.current", leaf.1),
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
            read((8 * GIB, GIB), (6 * GIB, 2 * GIB)),
            (4 * GIB, leaf.clone())
        );
        assert_eq!(read((3 * GIB, GIB), (6 * GIB, 2 * GIB)), (2 * GIB, outer));
        assert_eq!(read((8 * GIB, GIB), (GIB, 2 * GIB)), (0, leaf));
        assert_eq!(read((64 * GIB, GIB), (32 * GIB, GIB)), (16 * GIB, None));

        let namespaced = root(&[
            meminfo(16 << 20),
            membership("0::/\n"),
            file("sys/fs/cgroup/cgroup.controllers", "memory"),
            file("sys/fs/cgroup/memory.max", 3 * GIB),
            file("sys/fs/cgroup/memory.current", GIB),
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
    /// v1 memory controller, a process cgroup that is not under the mount or
    /// leaves the namespace's root, and a limit file that cannot be read.
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
                if path.ends_with("sys/fs/cgroup/elsewhere/cgroup.controllers")
                    && source.kind() == ErrorKind::NotFound
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
            file("sys/fs/cgroup/job/cgroup.controllers", "memory"),
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
    /// not a count, and a limit without its usage.
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
                file("sys/fs/cgroup/job/cgroup.controllers", "memory"),
                file("sys/fs/cgroup/job/memory.max", "unlimited"),
            ],
            "job/memory.max",
        );
        malformed(
            &[meminfo(1 << 20), membership("0:/job\n")],
            "proc/self/cgroup",
        );
        let no_usage = root(&[
            meminfo(1 << 20),
            membership("0::/job\n"),
            file("sys/fs/cgroup/job/cgroup.controllers", "memory"),
            file("sys/fs/cgroup/job/memory.max", GIB),
        ]);
        assert!(matches!(
            Allocatable::read_under(no_usage.path()),
            Err(MemoryDiscoveryError::Read { path, .. }) if path.ends_with("job/memory.current")
        ));
    }
}
