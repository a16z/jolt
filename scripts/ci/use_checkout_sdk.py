#!/usr/bin/env python3
"""Use the checked-out SDK in a generated CI project, including on forks."""
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("project", type=Path)
project = parser.parse_args().project.resolve()
sdk = Path(__file__).resolve().parents[2] / "jolt-sdk"
source = 'git = "https://github.com/a16z/jolt"'

for manifest in [project / "Cargo.toml", project / "guest/Cargo.toml"]:
    content = manifest.read_text()
    if content.count(source) != 1:
        raise SystemExit(f"{manifest}: expected exactly one generated SDK dependency")
    relative_sdk = Path(os.path.relpath(sdk, manifest.parent)).as_posix()
    manifest.write_text(content.replace(source, f'path = "{relative_sdk}"'))
