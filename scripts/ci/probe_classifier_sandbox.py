#!/usr/bin/env python3
"""Probe the production Codex sandbox using only fake credentials; makes no model requests.

    uv run --no-project python scripts/ci/probe_classifier_sandbox.py DIRECTORY
"""

import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys

from classifier_agent import ModelBroker, configure_codex

FAKE_KEY = "FAKE_PROBE_KEY_NOT_A_SECRET"

PROBE = r'''
#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

int main(int argc, char **argv) {
    int leaks = 0, inspected = 0, readable = 0, files_readable = 0;
    if (getenv("OPENAI_API_KEY")) leaks++;
    DIR *proc = opendir("/proc");
    if (proc) {
        struct dirent *entry;
        while ((entry = readdir(proc))) {
            if (entry->d_name[0] < '0' || entry->d_name[0] > '9') continue;
            char filename[512], data[262144];
            snprintf(filename, sizeof(filename), "/proc/%s/environ", entry->d_name);
            inspected++;
            int fd = open(filename, O_RDONLY);
            if (fd < 0) continue;
            readable++;
            ssize_t n = read(fd, data, sizeof(data));
            if (n > 0 && memmem(data, n, "FAKE_PROBE_KEY_NOT_A_SECRET", 27)) leaks++;
            close(fd);
        }
        closedir(proc);
    }
    for (int i = 3; i < argc; i++) {
        int fd = open(argv[i], O_RDONLY);
        if (fd >= 0) {
            char data[2097152]; ssize_t n = read(fd, data, sizeof(data)); files_readable++;
            if (n > 0 && memmem(data, n, "FAKE_PROBE_KEY_NOT_A_SECRET", 27)) leaks++;
            close(fd);
        }
    }
    int fd = open(argv[1], O_WRONLY | O_CREAT, 0600);
    int write_errno = fd < 0 ? errno : 0;
    if (fd >= 0) close(fd);
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    int network_errno = sock < 0 ? errno : 0;
    if (sock >= 0) close(sock);
    int local = socket(AF_UNIX, SOCK_STREAM, 0);
    int unix_errno = local < 0 ? errno : 0;
    if (local >= 0) {
        struct sockaddr_un address = {.sun_family = AF_UNIX};
        snprintf(address.sun_path, sizeof(address.sun_path), "%s", argv[2]);
        unix_errno = connect(local, (struct sockaddr *)&address, sizeof(address)) < 0 ? errno : 0;
        close(local);
    }
    printf("{\"key_leaks\":%d,\"proc_environments_inspected\":%d,\"proc_environments_readable\":%d,"
           "\"files_readable\":%d,\"write_errno\":%d,\"network_errno\":%d,\"unix_socket_errno\":%d}\n", leaks, inspected, readable, files_readable, write_errno, network_errno, unix_errno);
    return leaks || !write_errno || !network_errno || !unix_errno;
}
'''


def main(directory: Path) -> None:
    directory.mkdir(parents=True)
    workspace = directory / "workspace"
    state = directory / "agent-state"
    workspace.mkdir()
    state.mkdir()
    (workspace / "base").mkdir()
    (workspace / "head").mkdir()
    (workspace / "base/a.rs").write_text("fn before() {}\n")
    (workspace / "head/a.rs").write_text("fn after() {}\n")
    (directory / "instructions.md").write_text("Read only; never execute repository code.")
    parent_has_key = ("OPENAI_API_KEY=" + FAKE_KEY).encode() in Path("/proc/self/environ").read_bytes()
    if not parent_has_key:
        raise RuntimeError("probe parent environment lacks its fake-key positive control")
    broker = ModelBroker(FAKE_KEY)
    (directory / "probe.c").write_text(PROBE)
    binary = directory / "probe"
    subprocess.run(["cc", "-static", "-O2", "-o", str(binary), str(directory / "probe.c")], check=True)
    config = state / "config.toml"
    configure_codex("http://127.0.0.1:1", directory)
    env = {"PATH": os.environ["PATH"], "HOME": str(state), "CODEX_HOME": str(state),
           "OPENAI_API_KEY": "FAKE_CLASSIFIER_KEY"}
    prefix = [shutil.which("codex"), "sandbox", "-P", ":read-only", "-C", str(workspace), "--"]
    credential_files = [str(config), str(state / "auth.json"), str(Path.home() / ".codex/auth.json")]
    credential_files.extend(os.environ[name] for name in ("GITHUB_ENV", "GITHUB_PATH", "GITHUB_OUTPUT")
                        if name in os.environ)
    ancestors = []
    pid = os.getpid()
    while pid > 0:
        ancestors.append(pid)
        credential_files.append(f"/proc/{pid}/environ")
        try:
            pid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError):
            break
    credential_files.extend(str(file) for file in state.rglob("*") if file.is_file())
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(str(directory / "owned.sock"))
    listener.listen(1)

    def probe():
        files = sorted(set(credential_files + [str(file) for file in state.rglob("*") if file.is_file()]))
        command = shlex.join([str(binary), str(workspace / "write-probe"), str(directory / "owned.sock"), *files])
        return subprocess.run([*prefix, "/bin/bash", "-c", command], env=env, capture_output=True, text=True), files

    native, files = probe()
    if native.returncode:
        raise RuntimeError(f"native probe failed: {native.stdout} {native.stderr}")
    report = json.loads(native.stdout)
    report["version"] = subprocess.check_output([shutil.which("codex"), "--version"], text=True).strip()
    report["ancestor_pids_tested"] = ancestors
    report["parent_environment_has_fake_key"] = parent_has_key
    read = subprocess.run([*prefix, "/bin/bash", "-c",
                           "git diff --no-index --no-ext-diff --no-textconv --stat -- base head"],
                          env=env, capture_output=True, text=True)
    if read.returncode != 1 or "1 file changed" not in read.stdout:
        raise RuntimeError(f"read probe failed: {read.stdout} {read.stderr}")
    native, files = probe()
    if native.returncode:
        raise RuntimeError(f"post-Codex credential probe failed: {native.stdout} {native.stderr}")
    report.update(json.loads(native.stdout))
    report["credential_files_tested"] = files
    report["git_diffstat"] = read.stdout
    listener.close()
    report["model_requests"] = len(broker.usage)
    (directory / "probe-result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    if os.environ.get("OPENAI_API_KEY") != FAKE_KEY:
        os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "OPENAI_API_KEY": FAKE_KEY})
    main(Path(sys.argv[1]).resolve())
