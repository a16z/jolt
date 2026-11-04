#!/usr/bin/env python3
"""Use the checked-out SDK in a generated CI project, including on forks."""
import argparse
import os
import shutil
import tomllib
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

# The generated project is its own workspace, so dependency patches are not inherited.
checkout = sdk.parent
patch_source = "https://github.com/a16z/jolt"
patches = tomllib.loads((checkout / "Cargo.toml").read_text())["patch"][patch_source]
manifest = project / "Cargo.toml"
content = manifest.read_text() + f'\n[patch."{patch_source}"]\n'
for name, dependency in patches.items():
    relative = Path(os.path.relpath(checkout / dependency["path"], project)).as_posix()
    content += f'{name} = {{ path = "{relative}" }}\n'
manifest.write_text(content)

# Cargo still queries an unlocked Git source even when a matching patch exists.
# Seed the generated workspace with the checkout's resolved patch identities
# before its first build, so Akita's Jolt dependencies stay on the local paths.
shutil.copyfile(checkout / "Cargo.lock", project / "Cargo.lock")
