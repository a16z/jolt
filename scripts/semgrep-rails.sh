#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEMGREP_HOME="${SEMGREP_HOME:-"$ROOT/target/semgrep-home"}"

if [ -n "${SSL_CERT_FILE:-}" ]; then
  CERT_FILE="$SSL_CERT_FILE"
elif [ -r /etc/ssl/cert.pem ]; then
  CERT_FILE=/etc/ssl/cert.pem
elif [ -r /etc/ssl/certs/ca-certificates.crt ]; then
  CERT_FILE=/etc/ssl/certs/ca-certificates.crt
elif [ -r /opt/homebrew/etc/ca-certificates/cert.pem ]; then
  CERT_FILE=/opt/homebrew/etc/ca-certificates/cert.pem
else
  CERT_FILE=
fi

mkdir -p "$SEMGREP_HOME"

if [ -z "$CERT_FILE" ] || [ ! -r "$CERT_FILE" ]; then
  echo "semgrep-rails: could not find a readable CA cert bundle; set SSL_CERT_FILE" >&2
  exit 1
fi

cd "$ROOT"

if [ "$#" -eq 0 ]; then
  set -- crates/jolt-prover crates/jolt-backends crates/jolt-witness
fi

HOME="$SEMGREP_HOME" \
SSL_CERT_FILE="$CERT_FILE" \
SEMGREP_SEND_METRICS="${SEMGREP_SEND_METRICS:-off}" \
exec semgrep \
  scan \
  --disable-version-check \
  --metrics off \
  --no-git-ignore \
  --config crates/jolt-prover-harness/semgrep/prover-rails.yml \
  --error \
  "$@"
