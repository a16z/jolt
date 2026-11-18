#!/usr/bin/env bash

# Builds HOL Light as an OCaml module in DIRECTORY, inside a local opam switch.
#
# CI restores the built directory from the Actions cache under a key that
# hashes this file, so every input that selects what gets built must be pinned
# here. hol.sh and the switch record DIRECTORY's absolute path, so the result
# works only at the path it was built at.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 DIRECTORY" >&2
  exit 2
fi
HOL_LIGHT_DIR=$1

# fp64-certified-builds.json records the same commit, and check-fp64.sh rejects
# a HOL Light checkout that differs from that record.
HOL_LIGHT_COMMIT=433477862bb90b328a593e012e09390e99b2439b
OPAM_REPOSITORY_COMMIT=1c93acbc4176695001cc3356717dcfe90f89ce44
OCAML_VERSION=5.4.0
CAMLP5_VERSION=8.04.00

git init --quiet "$HOL_LIGHT_DIR"
cd "$HOL_LIGHT_DIR"
git fetch --quiet --depth 1 https://github.com/jrh13/hol-light.git "$HOL_LIGHT_COMMIT"
git checkout --quiet --detach FETCH_HEAD

# --no-install keeps opam from installing HOL Light through its own opam file.
opam switch create . "ocaml-base-compiler.$OCAML_VERSION" --no-install --yes \
  --repositories="pinned=git+https://github.com/ocaml/opam-repository.git#$OPAM_REPOSITORY_COMMIT"
eval "$(opam env --switch . --set-switch)"
# Pin Camlp5 before ledit installation to avoid a dependency downgrade and rebuild.
opam pin --switch . add --yes --no-action camlp5 "$CAMLP5_VERSION"
opam install --switch . --yes zarith ledit camlp5
HOLLIGHT_USE_MODULE=1 make -j2
