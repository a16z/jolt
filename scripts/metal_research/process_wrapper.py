from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Optional


def _write_identity(path: Path, identity: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(
            descriptor,
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(),
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity-path", required=True)
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--launch-token", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("tracked evaluator command is empty")
    _write_identity(
        Path(args.identity_path),
        {
            "schema_version": 1,
            "evaluation_id": args.evaluation_id,
            "launch_token": args.launch_token,
            "pid": os.getpid(),
            "pgid": os.getpgrp(),
        },
    )
    return subprocess.call(command, env=os.environ)


if __name__ == "__main__":
    raise SystemExit(main())
