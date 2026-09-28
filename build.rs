use std::path::Path;
use std::process::Command;

struct CommitInfo {
    short_hash: String,
    date: String,
}

fn commit_info_from_git() -> Option<CommitInfo> {
    if !Path::new(".git").exists() {
        return None;
    }

    let output = match Command::new("git")
        .arg("log")
        .arg("-1")
        .arg("--date=short")
        .arg("--format=%h %cd")
        .arg("--abbrev=9")
        .output()
    {
        Ok(output) if output.status.success() => output,
        _ => return None,
    };

    let stdout = String::from_utf8(output.stdout).unwrap();
    let mut parts = stdout.split_whitespace().map(|s| s.to_string());

    Some(CommitInfo {
        short_hash: parts.next()?,
        date: parts.next()?,
    })
}

fn commit_info() {
    let Some(git) = commit_info_from_git() else {
        return;
    };

    println!("cargo:rustc-env=GIT_SHORT_HASH={}", git.short_hash);
    println!("cargo:rustc-env=GIT_DATE={}", git.date);
}

fn main() {
    commit_info();
}
